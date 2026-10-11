"""Print the synchronized runtime summary produced by src.infer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STAGES = [
    ("backbone", "Backbone"),
    ("pe_field", "PE-Field"),
    ("lifting", "Lifting"),
    ("depth_alignment", "Depth alignment"),
    ("registration", "Registration"),
    ("refinement", "Refinement"),
    ("target_render", "Target render"),
    ("end_to_end", "End-to-end"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    print(f"Scenes: {data.get('num_scenes', 0)}")
    print(f"Warm-up scenes excluded: {data.get('protocol', {}).get('warmup_scenes', 0)}")
    print(f"Runtime file: {args.input}")
    print(f"{'Stage':<20} {'Mean (s)':>12} {'Std (s)':>12} {'Median (s)':>12}")
    print("-" * 62)
    mean = data.get("mean_seconds", {})
    std = data.get("std_seconds", {})
    median = data.get("median_seconds", {})
    for key, label in STAGES:
        print(
            f"{label:<20} {mean.get(key, 0.0):12.4f} "
            f"{std.get(key, 0.0):12.4f} {median.get(key, 0.0):12.4f}"
        )

    print("\nNotes:")
    print("- GPU-synchronized wall-clock measurements after model loading.")
    print("- File saving, metric computation, and one-time initialization are excluded.")
    print("- End-to-end includes the backbone, PE-Field, lifting, alignment, registration, refinement, and final target render.")


if __name__ == "__main__":
    main()
