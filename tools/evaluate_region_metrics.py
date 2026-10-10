"""Evaluate metrics on a fixed completion/observation region.

The completion mask must be generated before inserting generated content.  In
the current GSCompleter implementation it is saved as
``debug/vis/<scene>/view_*_<target>_tgt_align_mask.png`` where white denotes
pixels already covered by the context and black denotes holes.  The same mask
is deliberately reused for all methods.

For PSNR, the error is averaged over the selected pixels only.  SSIM, LPIPS,
and FID use a GT-composited crop: pixels outside the selected region are copied
from GT, then the tight region bounding box is resized to a fixed size.  This
keeps those metrics focused on the selected region without introducing black
background artifacts at the mask boundary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from lpips import LPIPS
from skimage.metrics import structural_similarity
from torchmetrics.image.fid import FrechetInceptionDistance


METHOD_NAMES = ("baseline", "generated-only", "gscompleter")


def load_rgb(path: Path) -> torch.Tensor:
    """Load an RGB image as float tensor in CHW and [0, 1]."""
    with Image.open(path) as image:
        return torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)


def load_mask(path: Path, shape: Tuple[int, int]) -> torch.Tensor:
    """Load a valid-coverage PNG and return a boolean completion mask."""
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if array.shape != shape:
        raise ValueError(f"Mask shape {array.shape} does not match image shape {shape}: {path}")
    # Existing debug masks use white=covered and black=hole.
    return torch.from_numpy(array < 0.5)


@lru_cache(maxsize=None)
def find_mask(mask_root: Path, scene: str, target_id: int) -> Path:
    """Find masks from both the current and legacy scene-directory layouts."""
    target = str(target_id)
    scene_token = Path(scene).name
    scene_tokens = {scene, scene_token}

    # Legacy runs removed the final len('/nerfstudio') characters from the
    # hexadecimal scene hash when constructing debug/vis directory names.
    if len(scene_token) >= 32 and all(c in "0123456789abcdefABCDEF" for c in scene_token):
        scene_tokens.add(scene_token[:-len("/nerfstudio")])

    patterns = (
        f"view_*_{target}_tgt_align_mask.png",
        f"view_*_{target_id:06d}_tgt_align_mask.png",
    )
    search_roots = [mask_root]
    debug_vis = mask_root / "debug" / "vis"
    if debug_vis.is_dir():
        search_roots.insert(0, debug_vis)

    candidates = []
    for root in search_roots:
        for token in scene_tokens:
            scene_dir = root / token
            if scene_dir.is_dir():
                for pattern in patterns:
                    candidates.extend(scene_dir.glob(pattern))

    # Also support MASK_ROOT pointing to a parent directory or directly to
    # debug/vis, which is common for outputs copied from another run.
    if not candidates:
        for root in search_roots:
            if root.is_dir():
                for pattern in patterns:
                    candidates.extend(
                        path for path in root.rglob(pattern)
                        if path.parent.name in scene_tokens
                    )

    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError(
            f"No fixed target mask for scene={scene}, target={target_id} under {mask_root}."
        )
    return candidates[0]


def image_path(root: Path, scene: str, target_id: int, suffix: str = "") -> Path:
    return root / "images" / scene / "color" / f"{target_id:06d}{suffix}.png"


def active_bbox(mask: torch.Tensor) -> Tuple[int, int, int, int]:
    ys, xs = torch.where(mask)
    if ys.numel() == 0:
        raise ValueError("Selected region is empty")
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def composited_crop(gt: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor, size: int) -> torch.Tensor:
    """Copy GT outside mask and crop the selected region to a fixed square."""
    y0, y1, x0, x1 = active_bbox(mask)
    mask3 = mask.unsqueeze(0)
    composite = torch.where(mask3, pred, gt)
    crop = composite[:, y0:y1, x0:x1].unsqueeze(0)
    return F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False).squeeze(0)


def psnr_region(gt: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> float:
    delta = (gt.clamp(0, 1) - pred.clamp(0, 1))[:, mask]
    mse = float(delta.square().mean()) if delta.numel() else float("nan")
    return float("inf") if mse == 0 else -10.0 * math.log10(mse)


def ssim_region(gt_crop: torch.Tensor, pred_crop: torch.Tensor) -> float:
    # A fixed 11-pixel window is used by the paper's full-image evaluator.
    gt_np = gt_crop.detach().cpu().numpy()
    pred_np = pred_crop.detach().cpu().numpy()
    return float(
        structural_similarity(
            gt_np,
            pred_np,
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute fixed-mask DL3DV region metrics.")
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("assets/evaluation_index_extrapolation/dl3dv/dl3dv_ctx2_tgt1_n10.json"),
    )
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--generated-only-root", type=Path, required=True)
    parser.add_argument("--gscompleter-root", type=Path, required=True)
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=None,
        help="Root containing debug/vis masks. Defaults to --gscompleter-root.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/region_metrics_dl3dv"))
    parser.add_argument("--region", choices=("completed", "observed", "both"), default="completed")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N scenes for a dry run.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing samples. Keep disabled for the final fair comparison.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def selected_regions(name: str, completion_mask: torch.Tensor) -> Iterable[Tuple[str, torch.Tensor]]:
    if name in ("completed", "both"):
        yield "completed", completion_mask
    if name in ("observed", "both"):
        yield "observed", ~completion_mask


def main() -> None:
    args = parse_args()
    with args.index.open("r", encoding="utf-8") as handle:
        index: Dict[str, Dict[str, List[int]]] = json.load(handle)
    scenes = list(index)
    if args.limit is not None:
        scenes = scenes[: args.limit]

    roots = {
        "baseline": args.baseline_root,
        "generated-only": args.generated_only_root,
        "gscompleter": args.gscompleter_root,
    }
    mask_root = args.mask_root or args.gscompleter_root
    device = torch.device(args.device)
    lpips = LPIPS(net="vgg").to(device).eval()
    regions = ["completed", "observed"] if args.region == "both" else [args.region]
    fid_metrics = {
        region: {method: FrechetInceptionDistance(feature=2048, normalize=True).to(device)
                 for method in METHOD_NAMES}
        for region in regions
    }
    accum: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    rows: List[dict] = []
    skipped: List[dict] = []

    for scene_no, scene in enumerate(scenes, start=1):
        target_ids = index[scene]["target"]
        for target_id in target_ids:
            gt_path = image_path(args.gscompleter_root, scene, target_id, "_gt")
            if not gt_path.exists():
                # GT is identical across method output directories; use any existing root.
                for root in roots.values():
                    candidate = image_path(root, scene, target_id, "_gt")
                    if candidate.exists():
                        gt_path = candidate
                        break
            try:
                gt = load_rgb(gt_path)
                mask_path = find_mask(mask_root, scene, target_id)
                completion_mask = load_mask(mask_path, tuple(gt.shape[-2:]))
            except (FileNotFoundError, ValueError) as error:
                if not args.allow_missing:
                    raise
                skipped.append({"scene": scene, "target": target_id, "reason": str(error)})
                continue

            for region_name, region_mask in selected_regions(args.region, completion_mask):
                if not region_mask.any():
                    if not args.allow_missing:
                        raise ValueError(f"Empty {region_name} region for scene={scene}, target={target_id}")
                    skipped.append({"scene": scene, "target": target_id, "region": region_name, "reason": "empty mask"})
                    continue
                for method, root in roots.items():
                    pred_path = image_path(root, scene, target_id)
                    if not pred_path.exists():
                        if not args.allow_missing:
                            raise FileNotFoundError(f"Missing {method} prediction: {pred_path}")
                        skipped.append({"scene": scene, "target": target_id, "region": region_name, "method": method, "reason": f"missing {pred_path}"})
                        continue
                    pred = load_rgb(pred_path)
                    if pred.shape != gt.shape:
                        raise ValueError(f"Prediction/GT shape mismatch: {pred_path} {pred.shape} vs {gt.shape}")
                    gt_crop = composited_crop(gt, gt, region_mask, args.crop_size)
                    pred_crop = composited_crop(gt, pred, region_mask, args.crop_size)
                    psnr = psnr_region(gt, pred, region_mask)
                    ssim = ssim_region(gt_crop, pred_crop)
                    with torch.no_grad():
                        lpips_value = float(lpips(gt_crop.unsqueeze(0).to(device), pred_crop.unsqueeze(0).to(device), normalize=True).item())
                    real_uint8 = (gt_crop.unsqueeze(0).clamp(0, 1) * 255).to(torch.uint8).to(device)
                    fake_uint8 = (pred_crop.unsqueeze(0).clamp(0, 1) * 255).to(torch.uint8).to(device)
                    fid_metrics[region_name][method].update(real_uint8, real=True)
                    fid_metrics[region_name][method].update(fake_uint8, real=False)

                    metric_key = (region_name, method)
                    accum.setdefault(metric_key, {"psnr": [], "ssim": [], "lpips": [], "mask_ratio": []})
                    accum[metric_key]["psnr"].append(psnr)
                    accum[metric_key]["ssim"].append(ssim)
                    accum[metric_key]["lpips"].append(lpips_value)
                    accum[metric_key]["mask_ratio"].append(float(region_mask.float().mean()))
                    rows.append({
                        "scene": scene,
                        "target": target_id,
                        "region": region_name,
                        "method": method,
                        "mask_path": str(mask_path),
                        "mask_ratio": float(region_mask.float().mean()),
                        "psnr": psnr,
                        "ssim": ssim,
                        "lpips": lpips_value,
                    })
        print(f"[{scene_no}/{len(scenes)}] {scene}")

    summary: List[dict] = []
    for (region, method), values in accum.items():
        # FID is undefined for fewer than two samples. This also makes LIMIT=1
        # useful as a pipeline smoke test instead of failing at finalization.
        if len(values["psnr"]) < 2:
            fid = None
        else:
            try:
                fid = float(fid_metrics[region][method].compute().item())
            except Exception as error:
                print(f"Warning: FID failed for {region}/{method}: {error}")
                fid = None
        summary.append({
            "region": region,
            "method": method,
            "num_samples": len(values["psnr"]),
            "mean_mask_ratio": float(np.mean(values["mask_ratio"])),
            "psnr": float(np.mean(values["psnr"])),
            "ssim": float(np.mean(values["ssim"])),
            "lpips": float(np.mean(values["lpips"])),
            "fid": fid,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0]) if rows else ["scene", "target", "region", "method"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["region", "method", "num_samples", "mean_mask_ratio", "psnr", "ssim", "lpips", "fid"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"index": str(args.index), "mask_root": str(mask_root), "summary": summary, "skipped": skipped}, handle, indent=2)

    print("\nRegion metric summary")
    for item in summary:
        print(
            f"{item['region']:>9} | {item['method']:<14} | "
            f"mask {item['mean_mask_ratio']:.3f} | "
            f"PSNR {item['psnr']:.4f} | SSIM {item['ssim']:.4f} | "
            f"LPIPS {item['lpips']:.4f} | FID {item['fid'] if item['fid'] is not None else 'n/a'} | "
            f"N={item['num_samples']}"
        )
    if skipped:
        print(f"Skipped {len(skipped)} sample entries; details: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
