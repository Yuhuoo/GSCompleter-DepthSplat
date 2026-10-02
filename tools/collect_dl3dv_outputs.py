"""Collect GSCompleter target renderings into the external-baseline layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--gs-output", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for pair_dir in sorted(p for p in args.pairs.iterdir() if p.is_dir()):
        metadata_path = pair_dir / "pair.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scene = metadata["scene"]
        target_id = int(metadata["target_source_indices"][0])
        candidates = [
            args.gs_output / "images" / scene / "color" / f"{target_id:06d}.png",
            args.gs_output / "images" / pair_dir.name / "color" / f"{target_id:06d}.png",
        ]
        source = next((p for p in candidates if p.exists()), None)
        if source is None:
            print(f"[skip] target rendering not found: {scene}")
            continue

        output_dir = args.baseline_root / pair_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        target_path = output_dir / "target.png"
        if target_path.exists() and not args.overwrite:
            continue
        shutil.copyfile(source, target_path)

if __name__ == "__main__":
    main()
