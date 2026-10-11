"""Collect metric summaries from sensitivity experiment output directories."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


METRIC_RE = re.compile(r"^(psnr|ssim|lpips|fid):\s*([-+0-9.eE]+)\s*$")
FALLBACK_RE = re.compile(r"^Fallback_cnt:\s*(\d+)\s*$")


def read_summary(path: Path) -> dict[str, str | float | int]:
    values: dict[str, str | float | int] = {"case": path.parent.name}
    for line in path.read_text(encoding="utf-8").splitlines():
        metric_match = METRIC_RE.match(line.strip())
        if metric_match:
            values[metric_match.group(1)] = float(metric_match.group(2))
            continue
        fallback_match = FALLBACK_RE.match(line.strip())
        if fallback_match:
            values["fallback_cnt"] = int(fallback_match.group(1))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for summary_path in sorted(args.root.glob("*/metrics_summary.txt")):
        row = read_summary(summary_path)
        rows.append(row)

    output_path = args.output or args.root / "sensitivity_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case", "psnr", "ssim", "lpips", "fid", "fallback_cnt"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"Saved {len(rows)} sensitivity rows to {output_path}")
    print("case,PSNR,SSIM,LPIPS,FID,Fallback")
    for row in rows:
        print(
            f"{row['case']},{row.get('psnr', '')},{row.get('ssim', '')},"
            f"{row.get('lpips', '')},{row.get('fid', '')},{row.get('fallback_cnt', '')}"
        )


if __name__ == "__main__":
    main()
