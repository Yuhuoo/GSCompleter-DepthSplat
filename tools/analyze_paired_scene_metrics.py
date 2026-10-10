"""Paired scene-level statistics from saved target-view PNGs.

Run this from the GSCompleter-DepthSplat repository root. PNG-based metrics
may differ slightly from metrics computed on floating-point renders.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import wilcoxon

from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim


METRICS = ("psnr", "ssim", "lpips")
TARGET_NAME = re.compile(r"\d{6}\.png$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--ours-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, help="Optional third output root containing *_gt.png")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def target_images(root: Path) -> dict[Path, Path]:
    images_dir = root / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {images_dir}")
    return {
        path.relative_to(images_dir): path
        for path in images_dir.rglob("*.png")
        if path.parent.name == "color" and TARGET_NAME.fullmatch(path.name)
    }


def load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def compute_metrics(gt: torch.Tensor, prediction: torch.Tensor) -> dict[str, float]:
    if gt.shape != prediction.shape:
        raise ValueError(f"Prediction/GT shape mismatch: {prediction.shape} vs {gt.shape}")
    with torch.no_grad():
        return {
            "psnr": float(compute_psnr(gt, prediction).item()),
            "ssim": float(compute_ssim(gt, prediction).item()),
            "lpips": float(compute_lpips(gt, prediction).item()),
        }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted = {}
    running = 0.0
    for rank, metric in enumerate(ordered):
        running = max(running, min(1.0, pvalues[metric] * (len(ordered) - rank)))
        adjusted[metric] = running
    return adjusted


def summarize(rows: list[dict], bootstrap: int, rng: np.random.Generator) -> dict:
    pvalues = {}
    summary = {}
    for metric in METRICS:
        baseline = np.asarray([row[f"baseline_{metric}"] for row in rows], dtype=np.float64)
        ours = np.asarray([row[f"ours_{metric}"] for row in rows], dtype=np.float64)
        improvement = np.asarray([row[f"improvement_{metric}"] for row in rows], dtype=np.float64)
        if not np.isfinite(improvement).all():
            raise ValueError(f"Non-finite {metric} values; inspect per_scene.csv")
        boot_means = np.empty(bootstrap, dtype=np.float64)
        for i in range(bootstrap):
            boot_means[i] = rng.choice(improvement, size=len(improvement), replace=True).mean()
        nonzero = improvement[improvement != 0]
        pvalues[metric] = float(wilcoxon(nonzero, alternative="two-sided").pvalue) if len(nonzero) else 1.0
        summary[metric] = {
            "baseline_mean": float(baseline.mean()),
            "ours_mean": float(ours.mean()),
            "baseline_sd": float(baseline.std(ddof=1)) if len(rows) > 1 else None,
            "ours_sd": float(ours.std(ddof=1)) if len(rows) > 1 else None,
            "improvement_mean": float(improvement.mean()),
            "improvement_sd": float(improvement.std(ddof=1)) if len(rows) > 1 else None,
            "improvement_ci95": [float(x) for x in np.quantile(boot_means, [0.025, 0.975])],
            "win_rate": float(np.mean(improvement > 0)),
            "loss_rate": float(np.mean(improvement < 0)),
            "tie_rate": float(np.mean(improvement == 0)),
            "wilcoxon_p_two_sided": pvalues[metric],
        }
    for metric, adjusted in holm_adjust(pvalues).items():
        summary[metric]["holm_adjusted_p"] = adjusted
    return summary


def main() -> None:
    args = parse_args()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive")
    device = torch.device(args.device)
    baseline_images = target_images(args.baseline_root)
    ours_images = target_images(args.ours_root)
    if not baseline_images or baseline_images.keys() != ours_images.keys():
        missing_ours = sorted(baseline_images.keys() - ours_images.keys())
        missing_baseline = sorted(ours_images.keys() - baseline_images.keys())
        raise ValueError(
            f"Target sets differ or are empty: baseline={len(baseline_images)}, "
            f"ours={len(ours_images)}; missing ours={missing_ours[:5]}, "
            f"missing baseline={missing_baseline[:5]}"
        )

    gt_roots = [args.gt_root] if args.gt_root else [args.ours_root, args.baseline_root]
    target_rows = []
    scene_rows = defaultdict(list)
    for index, relative_path in enumerate(sorted(baseline_images), start=1):
        scene = relative_path.parent.parent.as_posix()
        gt_name = relative_path.stem + "_gt.png"
        gt_paths = [root / "images" / relative_path.parent / gt_name for root in gt_roots]
        gt_path = next((path for path in gt_paths if path.is_file()), None)
        if gt_path is None:
            raise FileNotFoundError(f"Missing GT for {relative_path}; checked {gt_paths}")
        gt = load_rgb(gt_path, device)
        baseline = compute_metrics(gt, load_rgb(baseline_images[relative_path], device))
        ours = compute_metrics(gt, load_rgb(ours_images[relative_path], device))
        row = {"scene": scene, "target": relative_path.stem}
        for metric in METRICS:
            row[f"baseline_{metric}"] = baseline[metric]
            row[f"ours_{metric}"] = ours[metric]
            row[f"improvement_{metric}"] = (
                baseline[metric] - ours[metric] if metric == "lpips" else ours[metric] - baseline[metric]
            )
        target_rows.append(row)
        scene_rows[scene].append(row)
        if index % 100 == 0 or index == len(baseline_images):
            print(f"[{index}/{len(baseline_images)}] target views evaluated", flush=True)

    per_scene = []
    for scene, rows in sorted(scene_rows.items()):
        item = {"scene": scene, "num_targets": len(rows)}
        for metric in METRICS:
            for prefix in ("baseline", "ours", "improvement"):
                key = f"{prefix}_{metric}"
                item[key] = float(np.mean([row[key] for row in rows]))
        per_scene.append(item)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_target.csv", target_rows)
    write_csv(args.output_dir / "per_scene.csv", per_scene)
    summary = {
        "baseline_root": str(args.baseline_root),
        "ours_root": str(args.ours_root),
        "source": "saved 8-bit PNGs; results may differ slightly from float-render metrics",
        "unit": "scene; multiple targets within a scene are averaged first",
        "num_scenes": len(per_scene),
        "num_targets": len(target_rows),
        "bootstrap_resamples": args.bootstrap,
        "seed": args.seed,
        "tests": "paired two-sided Wilcoxon; Holm correction across PSNR, SSIM, LPIPS",
        "positive_improvement": "ours-baseline for PSNR/SSIM; baseline-ours for LPIPS",
        "metrics": summarize(per_scene, args.bootstrap, np.random.default_rng(args.seed)),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Scenes: {summary['num_scenes']}; targets: {summary['num_targets']}")
    for metric, result in summary["metrics"].items():
        print(
            f"{metric.upper():5} baseline {result['baseline_mean']:.4f} +/- {result['baseline_sd']:.4f} "
            f"ours {result['ours_mean']:.4f} +/- {result['ours_sd']:.4f} "
            f"improvement {result['improvement_mean']:+.4f} "
            f"CI95 [{result['improvement_ci95'][0]:+.4f}, {result['improvement_ci95'][1]:+.4f}] "
            f"win {result['win_rate']:.1%} Holm p={result['holm_adjusted_p']:.3g}"
        )
    print(f"Saved {args.output_dir / 'summary.json'} and per-scene/target CSVs")


if __name__ == "__main__":
    main()
