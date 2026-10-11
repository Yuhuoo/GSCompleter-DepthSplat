"""Select DL3DV scenes for qualitative failure-case inspection.

This script ranks scenes using saved per-scene metrics. The output is only a
candidate list: semantic labels such as dynamic, non-Lambertian, or wrong
contour must be assigned after inspecting the rendered images.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


CRITERIA = (
    ("low_ours_psnr", "ours_psnr", False),
    ("negative_psnr_delta", "improvement_psnr", False),
    ("high_ours_lpips", "ours_lpips", True),
    ("low_ours_ssim", "ours_ssim", False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-scene", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--index", type=Path, default=None)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    required = {
        "scene",
        "ours_psnr",
        "ours_ssim",
        "ours_lpips",
        "improvement_psnr",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Missing columns in {path}: {', '.join(missing)}")
    return rows


def main() -> None:
    args = parse_args()
    if args.top_k < 1 or args.max_candidates < 1:
        raise ValueError("--top-k and --max-candidates must be positive")

    rows = load_rows(args.per_scene)
    valid_scenes = None
    if args.index is not None:
        import json

        with args.index.open(encoding="utf-8") as handle:
            valid_scenes = set(json.load(handle))

    candidates: dict[str, dict] = {}
    for reason, column, reverse in CRITERIA:
        ranked = sorted(rows, key=lambda row: float(row[column]), reverse=reverse)
        for rank, row in enumerate(ranked[: args.top_k], start=1):
            scene = row["scene"]
            if valid_scenes is not None and scene not in valid_scenes:
                continue
            item = candidates.setdefault(scene, {"scene": scene, "reasons": []})
            item["reasons"].append(f"{reason}:rank{rank}")

    selected = sorted(
        candidates.values(),
        key=lambda item: (
            min(int(reason.rsplit("rank", 1)[1]) for reason in item["reasons"]),
            item["scene"],
        ),
    )[: args.max_candidates]
    row_by_scene = {row["scene"]: row for row in rows}
    for index, item in enumerate(selected, start=1):
        row = row_by_scene[item["scene"]]
        item["candidate"] = f"candidate_{index:02d}"
        item["reasons"] = ";".join(item["reasons"])
        for column in (
            "ours_psnr",
            "ours_ssim",
            "ours_lpips",
            "improvement_psnr",
            "improvement_ssim",
            "improvement_lpips",
        ):
            item[column] = row.get(column, "")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = args.output_dir / "failure_candidates.csv"
    fields = [
        "candidate",
        "scene",
        "reasons",
        "ours_psnr",
        "ours_ssim",
        "ours_lpips",
        "improvement_psnr",
        "improvement_ssim",
        "improvement_lpips",
    ]
    with candidates_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    cases = ",".join(f"{item['candidate']}:{item['scene']}" for item in selected)
    (args.output_dir / "failure_cases.txt").write_text(cases + "\n", encoding="utf-8")

    print(f"Selected {len(selected)} candidate scenes from {len(rows)} scenes.")
    print(f"Saved {candidates_path}")
    print(f"CASES='{cases}'")
    if cases:
        print(
            "Run the rendering step with:\n"
            f"CASES='{cases}' GPU=0 bash infer_failure_cases_dl3dv.sh"
        )
    print("Inspect the rendered candidates before assigning semantic failure labels.")


if __name__ == "__main__":
    main()
