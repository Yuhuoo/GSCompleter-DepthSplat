"""Report actual target-to-nearest-context camera distances for n=10/20/30.

Run from the project root, for example:
    python -m src.scripts.measure_extrapolation_distance
    python -m src.scripts.measure_extrapolation_distance --data-dir /path/to/dl3dv/test

The n=20 index is reconstructed only when its frozen JSON is absent. In that
case, the same reconstruction must reproduce every n=10 and n=30 frozen entry.
Only load trusted local PyTorch dataset chunks.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import torch


WINDOWS = (10, 20, 30)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read_index(path):
    with path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    if not isinstance(index, dict) or not index:
        raise ValueError(f"Empty or invalid evaluation index: {path}")
    return index


def load_indices(index_dir):
    indices = {}
    for n in WINDOWS:
        path = index_dir / f"dl3dv_ctx2_tgt1_n{n}.json"
        if path.is_file():
            indices[n] = read_index(path)
        elif n != 20:
            raise FileNotFoundError(f"Missing frozen evaluation index: {path}")
    if set(indices[10]) != set(indices[30]):
        raise ValueError("Frozen n=10 and n=30 indices have different scene sets")
    if 20 in indices and set(indices[20]) != set(indices[10]):
        raise ValueError("Frozen n=20 index has a different scene set")
    return indices


def camera_to_world(cameras):
    cameras = torch.as_tensor(cameras, dtype=torch.float32)
    if cameras.ndim != 2 or cameras.shape[1] != 18:
        raise ValueError(f"Expected cameras [views, 18], got {tuple(cameras.shape)}")
    w2c = torch.eye(4, dtype=torch.float32).repeat(cameras.shape[0], 1, 1)
    w2c[:, :3, :] = cameras[:, 6:].reshape(-1, 3, 4)
    c2w = torch.linalg.inv(w2c)
    if not torch.isfinite(c2w).all():
        raise ValueError("Non-finite camera pose")
    return c2w


def reconstruct_index(c2w, n):
    """Match the one-target, two-context branch of generate_extrapolation_index.py."""
    count = min(len(c2w), n)
    if count < 3:
        raise ValueError(f"Only {count} views available for n={n}")
    centers = c2w[:count, :3, 3]
    barycenter = torch.sum(centers, dim=0) / count
    target = int(torch.argmax(torch.sum((centers - barycenter) ** 2, dim=-1)))
    remaining = [i for i in range(count) if i != target]
    return {"context": [remaining[0], remaining[len(remaining) // 2]],
            "target": [target]}


def check_entry(entry, view_count, scene, n):
    context, target = entry["context"], entry["target"]
    indices = context + target
    if (len(context) != 2 or len(target) != 1
            or any(not isinstance(i, int) or i < 0 or i >= view_count for i in indices)
            or len(set(indices)) != 3):
        raise ValueError(f"Invalid two-context, one-target entry for {scene}, n={n}: {entry}")


def measure_scene(c2w, entry):
    centers = c2w[:, :3, 3]
    context, target = entry["context"], entry["target"][0]
    distances = torch.linalg.vector_norm(centers[context] - centers[target], dim=1)
    nearest_position = int(torch.argmin(distances))
    distance = float(distances[nearest_position])
    # The full-scene trajectory is fixed across all three n values.
    diameter = float(torch.cdist(centers, centers).max())
    if not math.isfinite(diameter) or diameter <= 0:
        raise ValueError("Full-scene camera trajectory has zero or invalid extent")
    return entry["context"][nearest_position], distance, distance / diameter


def quartiles(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    return [float(torch.quantile(tensor, q)) for q in (0.25, 0.5, 0.75)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=PROJECT_ROOT / "datasets/dl3dv/test",
                        help="Directory directly containing the evaluated .torch chunks")
    parser.add_argument("--index-dir", type=Path,
                        default=PROJECT_ROOT / "assets/evaluation_index_extrapolation/dl3dv",
                        help="Directory containing frozen evaluation index JSON files")
    parser.add_argument("--output-csv", type=Path,
                        default=PROJECT_ROOT / "outputs/extrapolation_pose_distance.csv")
    args = parser.parse_args()

    indices = load_indices(args.index_dir)
    reconstruct_n20 = 20 not in indices
    required = set(indices[10])
    chunk_paths = sorted(args.data_dir.glob("*.torch"))
    if not chunk_paths:
        raise FileNotFoundError(f"No .torch chunks directly in {args.data_dir}")

    rows = []
    seen = set()
    for path in chunk_paths:
        try:
            chunk = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            chunk = torch.load(path, map_location="cpu")
        for example in chunk:
            scene = example["key"]
            if scene not in required:
                continue
            if scene in seen:
                raise ValueError(f"Duplicate scene key in dataset chunks: {scene}")
            seen.add(scene)
            c2w = camera_to_world(example["cameras"])

            if reconstruct_n20:
                for n in (10, 30):
                    expected = reconstruct_index(c2w, n)
                    if indices[n][scene] != expected:
                        raise ValueError(
                            f"Index reconstruction disagrees with frozen n={n} for {scene}: "
                            f"frozen={indices[n][scene]}, reconstructed={expected}. "
                            "Do not use a reconstructed n=20 index with the old results."
                        )
                entry20 = reconstruct_index(c2w, 20)
            else:
                entry20 = indices[20][scene]

            for n in WINDOWS:
                entry = entry20 if n == 20 and reconstruct_n20 else indices[n][scene]
                check_entry(entry, len(c2w), scene, n)
                nearest, raw, normalized = measure_scene(c2w, entry)
                rows.append({"scene": scene, "n": n, "context_0": entry["context"][0],
                             "context_1": entry["context"][1], "target": entry["target"][0],
                             "nearest_context": nearest, "distance_pose_units": raw,
                             "distance_normalized": normalized,
                             "index_source": "reconstructed" if n == 20 and reconstruct_n20 else "frozen"})

    missing = required - seen
    if missing:
        raise ValueError(f"Missing {len(missing)} indexed scenes in data; first: {sorted(missing)[0]}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["scene"], row["n"])))

    print(f"Scenes: {len(required)}; n=20 index: {'reconstructed and cross-checked' if reconstruct_n20 else 'frozen'}")
    print("Target-to-nearest-context camera-center distance, median [Q1, Q3]:")
    for n in WINDOWS:
        group = [row for row in rows if row["n"] == n]
        raw_q1, raw_med, raw_q3 = quartiles([row["distance_pose_units"] for row in group])
        norm_q1, norm_med, norm_q3 = quartiles([row["distance_normalized"] for row in group])
        print(f"  n={n}: pose units {raw_med:.6f} [{raw_q1:.6f}, {raw_q3:.6f}]; "
              f"normalized {norm_med:.6f} [{norm_q1:.6f}, {norm_q3:.6f}]")

    lookup = {(row["scene"], row["n"]): row["distance_normalized"] for row in rows}
    for first, second in ((10, 20), (20, 30), (10, 30)):
        count = sum(lookup[scene, second] > lookup[scene, first] for scene in required)
        print(f"  n={first}->{second}: larger distance in {count}/{len(required)} scenes ({count/len(required):.1%})")
    print(f"Per-scene CSV: {args.output_csv}")
    if reconstruct_n20:
        print("Note: n=20 was reconstructed. Confirm it matches the index used for the published n=20 metrics.")


if __name__ == "__main__":
    main()
