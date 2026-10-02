"""Export the fixed two-context/one-target DL3DV protocol.

The exporter reads the same test chunks and evaluation index used by
GSCompleter-DepthSplat and writes a small, method-neutral scene directory.
Each scene contains exactly two context images followed by one target image.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import torch
from PIL import Image


def rotation_to_xyzw(rotation: torch.Tensor) -> list[float]:
    """Convert a 3x3 rotation matrix to an [x, y, z, w] quaternion."""
    m = rotation.float().cpu().numpy()
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = (trace + 1.0) ** 0.5 * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = (1.0 + m[0, 0] - m[1, 1] - m[2, 2]) ** 0.5 * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = (1.0 + m[1, 1] - m[0, 0] - m[2, 2]) ** 0.5 * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = (1.0 + m[2, 2] - m[0, 0] - m[1, 1]) ** 0.5 * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = [float(x), float(y), float(z), float(w)]
    norm = sum(v * v for v in q) ** 0.5
    return [v / norm for v in q]


def decode_image(raw: torch.Tensor, path: Path) -> tuple[int, int]:
    image = Image.open(io.BytesIO(raw.cpu().numpy().tobytes())).convert("RGB")
    image.save(path)
    return image.height, image.width


def iter_examples(chunk_root: Path):
    chunks = sorted((chunk_root / "test").glob("*.torch"))
    if not chunks:
        chunks = sorted(chunk_root.glob("*.torch"))
    if not chunks:
        raise FileNotFoundError(f"No .torch chunks found below {chunk_root}")
    for chunk_path in chunks:
        try:
            chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
        except TypeError:
            chunk = torch.load(chunk_path, map_location="cpu")
        for example in chunk:
            yield example


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    index = json.loads(args.index_path.read_text(encoding="utf-8"))
    examples = {example["key"]: example for example in iter_examples(args.data_root)}
    args.output_root.mkdir(parents=True, exist_ok=True)

    exported = 0
    skipped = 0
    for scene, entry in index.items():
        if entry is None:
            skipped += 1
            continue
        example = examples.get(scene)
        if example is None:
            print(f"[skip] scene not found in chunks: {scene}")
            skipped += 1
            continue

        context_ids = list(entry["context"])
        target_ids = list(entry["target"])
        if len(context_ids) != 2 or len(target_ids) != 1:
            print(f"[skip] expected 2 context and 1 target: {scene}")
            skipped += 1
            continue

        safe_scene = scene.replace("/", "_").replace("\\", "_")
        scene_dir = args.output_root / safe_scene
        if scene_dir.exists() and not args.overwrite:
            print(f"[skip] exists: {scene_dir}")
            continue
        image_dir = scene_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        cameras = torch.as_tensor(example["cameras"], dtype=torch.float32)
        selected_ids = context_ids + target_ids
        image_files = []
        camera_records = []
        height = width = None
        for output_id, source_id in enumerate(selected_ids):
            image_path = image_dir / f"{output_id:06d}.png"
            height, width = decode_image(example["images"][source_id], image_path)
            image_files.append(str(image_path.relative_to(scene_dir)).replace("\\", "/"))

            pose = cameras[source_id]
            w2c = torch.eye(4, dtype=torch.float32)
            w2c[:3] = pose[6:].reshape(3, 4)
            c2w = torch.linalg.inv(w2c)
            camera_records.append(
                {
                    "cam_id": output_id,
                    "cam_quat": rotation_to_xyzw(c2w[:3, :3]),
                    "cam_trans": c2w[:3, 3].tolist(),
                    "image_name": image_files[-1],
                }
            )

        first_camera = cameras[selected_ids[0]]
        intrinsics = {
            "fx": float(first_camera[0]) / width,
            "fy": float(first_camera[1]) / height,
            "cx": float(first_camera[2]) / width,
            "cy": float(first_camera[3]) / height,
        }
        (scene_dir / "cameras.json").write_text(
            json.dumps(camera_records, indent=2), encoding="utf-8"
        )
        (scene_dir / "intrinsics.json").write_text(
            json.dumps(intrinsics, indent=2), encoding="utf-8"
        )
        metadata = {
            "scene": scene,
            "context_source_indices": context_ids,
            "target_source_indices": target_ids,
            "context_files": image_files[:2],
            "target_file": image_files[2],
            "height": height,
            "width": width,
        }
        (scene_dir / "pair.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        exported += 1

    print(f"Exported {exported} scenes; skipped {skipped} scenes.")


if __name__ == "__main__":
    main()
