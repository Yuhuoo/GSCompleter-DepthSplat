"""Pair anchor results by scene and split DL3DV n=10 by camera motion."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median


METRICS = ("psnr", "ssim", "lpips")


def read_metrics(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"scene", "context_indices", "target_indices", *METRICS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing metric columns in {path}: {required - set(reader.fieldnames or [])}")
        rows = {}
        for row in reader:
            scene = row["scene"]
            if not scene or scene in rows:
                raise ValueError(f"Empty or duplicate scene in {path}: {scene}")
            values = {metric: float(row[metric]) for metric in METRICS}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Non-finite metric for {scene} in {path}")
            rows[scene] = {"context": row["context_indices"], "target": row["target_indices"], **values}
    return rows


def read_distances(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"scene", "n", "context_0", "context_1", "target", "distance_normalized"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing distance columns in {path}: {required - set(reader.fieldnames or [])}")
        rows = {}
        for row in reader:
            if int(row["n"]) != 10:
                continue
            scene = row["scene"]
            if not scene or scene in rows:
                raise ValueError(f"Empty or duplicate n=10 scene in {path}: {scene}")
            distance = float(row["distance_normalized"])
            if not math.isfinite(distance) or distance < 0:
                raise ValueError(f"Invalid normalized distance for {scene} in {path}")
            rows[scene] = {
                "context": f"{int(row['context_0'])},{int(row['context_1'])}",
                "target": row["target"],
                "distance_normalized": distance,
            }
    return rows


def require_same_scenes(label: str, actual: dict, expected: dict) -> None:
    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        raise ValueError(f"{label} scene mismatch: missing={missing[:5]}, extra={extra[:5]}")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(group: str, rows: list[dict]) -> dict:
    return {
        "group": group,
        "scenes": len(rows),
        "distance_min": min(row["distance_normalized"] for row in rows),
        "distance_median": median(row["distance_normalized"] for row in rows),
        "distance_max": max(row["distance_normalized"] for row in rows),
        "nearest_psnr_mean": mean(row["nearest_psnr"] for row in rows),
        "default_psnr_mean": mean(row["default_psnr"] for row in rows),
        "psnr_gain_mean": mean(row["psnr_gain"] for row in rows),
        "psnr_wins": sum(row["psnr_gain"] > 0 for row in rows),
        "ssim_gain_mean": mean(row["ssim_gain"] for row in rows),
        "lpips_reduction_mean": mean(row["lpips_reduction"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default", type=Path, required=True)
    parser.add_argument("--nearest", type=Path, required=True)
    parser.add_argument("--distance-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    default = read_metrics(args.default)
    nearest = read_metrics(args.nearest)
    distances = read_distances(args.distance_csv)
    if len(distances) != 140:
        raise ValueError(f"Expected 140 n=10 distance scenes, found {len(distances)}")
    require_same_scenes("default", default, distances)
    require_same_scenes("anchor_nearest", nearest, distances)

    paired = []
    for scene, distance in distances.items():
        target = str(int(distance["target"]))
        if default[scene]["target"] != target or nearest[scene]["target"] != target:
            raise ValueError(f"Target index mismatch for {scene}: distance={target}, "
                             f"default={default[scene]['target']}, nearest={nearest[scene]['target']}")
        if default[scene]["context"] != distance["context"] or nearest[scene]["context"] != distance["context"]:
            raise ValueError(f"Context index mismatch for {scene}: distance={distance['context']}, "
                             f"default={default[scene]['context']}, nearest={nearest[scene]['context']}")
        paired.append({
            "scene": scene,
            "context_indices": distance["context"],
            "target": target,
            "distance_normalized": distance["distance_normalized"],
            "nearest_psnr": nearest[scene]["psnr"],
            "default_psnr": default[scene]["psnr"],
            "psnr_gain": default[scene]["psnr"] - nearest[scene]["psnr"],
            "nearest_ssim": nearest[scene]["ssim"],
            "default_ssim": default[scene]["ssim"],
            "ssim_gain": default[scene]["ssim"] - nearest[scene]["ssim"],
            "nearest_lpips": nearest[scene]["lpips"],
            "default_lpips": default[scene]["lpips"],
            "lpips_reduction": nearest[scene]["lpips"] - default[scene]["lpips"],
        })

    paired.sort(key=lambda row: (row["distance_normalized"], row["scene"]))
    midpoint = len(paired) // 2
    for index, row in enumerate(paired):
        row["group"] = "low_motion" if index < midpoint else "high_motion"
    summaries = [
        summarize("low_motion", paired[:midpoint]),
        summarize("high_motion", paired[midpoint:]),
        summarize("all", paired),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_scene_joined.csv", paired)
    write_csv(args.output_dir / "motion_strata_summary.csv", summaries)
    for row in summaries:
        print(f"{row['group']}: {row['scenes']} scenes, normalized distance median "
              f"{row['distance_median']:.4f}, PSNR gain {row['psnr_gain_mean']:+.4f} dB, "
              f"PSNR wins {row['psnr_wins']}/{row['scenes']}, "
              f"SSIM gain {row['ssim_gain_mean']:+.4f}, "
              f"LPIPS reduction {row['lpips_reduction_mean']:+.4f}")
    print(f"Saved paired scene data and summary to {args.output_dir}")


if __name__ == "__main__":
    main()
