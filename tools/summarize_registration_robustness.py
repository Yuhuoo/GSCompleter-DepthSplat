"""Summarize ray-constrained versus unrestricted registration runs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


VALUE_RE = re.compile(r"^(psnr|ssim|lpips|fid):\s*([-+0-9.eE]+)\s*$")
FALLBACK_RE = re.compile(r"^Fallback_cnt:\s*(\d+)\s*$")


def read_summary(path: Path) -> dict[str, str | float | int]:
    values: dict[str, str | float | int] = {"case": path.parent.name}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = VALUE_RE.match(line.strip())
        if match:
            values[match.group(1)] = float(match.group(2))
        match = FALLBACK_RE.match(line.strip())
        if match:
            values["fallback_cnt"] = int(match.group(1))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        read_summary(path)
        for path in sorted(args.root.glob("*/metrics_summary.txt"))
    ]
    if not rows:
        raise FileNotFoundError(f"No metrics_summary.txt found below {args.root}")

    output = args.root / "registration_robustness_summary.csv"
    fields = ["case", "psnr", "ssim", "lpips", "fid", "fallback_cnt"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    print(f"Saved {len(rows)} rows to {output}")
    print("case,PSNR,SSIM,LPIPS,FID,Fallback")
    for row in rows:
        print(
            f"{row['case']},{row.get('psnr', '')},{row.get('ssim', '')},"
            f"{row.get('lpips', '')},{row.get('fid', '')},"
            f"{row.get('fallback_cnt', '')}"
        )


if __name__ == "__main__":
    main()
