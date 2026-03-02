#!/usr/bin/env python3
"""
Validate GLASS predictions CSV files.

Checks schema, value ranges, and prints summary statistics.

Usage:
    python validate_glass_output.py predictions.csv
    python validate_glass_output.py /path/to/eyegaze_vad/   # validate all CSVs in dir
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = ["timestamp", "valence", "arousal", "dominance"]
VAD_COLUMNS = ["valence", "arousal", "dominance"]


def validate_file(csv_path: Path) -> dict:
    """Validate a single predictions CSV and return a report dict."""
    report = {"path": str(csv_path), "valid": True, "errors": [], "warnings": []}

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        report["valid"] = False
        report["errors"].append(f"Could not read CSV: {e}")
        return report

    if df.empty:
        report["valid"] = False
        report["errors"].append("File is empty (no rows)")
        return report

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        report["valid"] = False
        report["errors"].append(f"Missing columns: {missing}")
        return report

    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra:
        report["warnings"].append(f"Extra columns (ignored): {extra}")

    report["num_rows"] = len(df)

    for col in VAD_COLUMNS:
        vals = df[col]
        n_nan = vals.isna().sum()
        if n_nan > 0:
            report["warnings"].append(f"{col}: {n_nan} NaN values")
        below = (vals < 0).sum()
        above = (vals > 1).sum()
        if below > 0:
            report["errors"].append(f"{col}: {below} values below 0 (min={vals.min():.4f})")
            report["valid"] = False
        if above > 0:
            report["errors"].append(f"{col}: {above} values above 1 (max={vals.max():.4f})")
            report["valid"] = False

    ts = df["timestamp"]
    if not ts.is_monotonic_increasing:
        report["warnings"].append("Timestamps are not monotonically increasing")
    report["time_range"] = (float(ts.min()), float(ts.max()))

    stats = {}
    for col in VAD_COLUMNS:
        s = df[col].describe()
        stats[col] = {
            "mean": float(s["mean"]),
            "std": float(s["std"]),
            "min": float(s["min"]),
            "max": float(s["max"]),
            "median": float(df[col].median()),
        }
    report["stats"] = stats

    return report


def print_report(report: dict) -> None:
    status = "VALID" if report["valid"] else "INVALID"
    print(f"\n{'─' * 60}")
    print(f"  {report['path']}")
    print(f"  Status: {status}")

    if report.get("errors"):
        for e in report["errors"]:
            print(f"  ERROR:   {e}")

    if report.get("warnings"):
        for w in report["warnings"]:
            print(f"  WARNING: {w}")

    if "num_rows" in report:
        print(f"  Rows: {report['num_rows']}")

    if "time_range" in report:
        t0, t1 = report["time_range"]
        print(f"  Time range: {t0:.2f}s – {t1:.2f}s")

    if "stats" in report:
        print(f"\n  {'Dimension':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Median':>8} {'Max':>8}")
        print(f"  {'─'*54}")
        for col in VAD_COLUMNS:
            s = report["stats"][col]
            print(
                f"  {col:<12} {s['mean']:8.4f} {s['std']:8.4f} "
                f"{s['min']:8.4f} {s['median']:8.4f} {s['max']:8.4f}"
            )
    print(f"{'─' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Validate GLASS predictions CSV(s)")
    parser.add_argument(
        "path",
        type=str,
        help="Path to a predictions CSV file or a directory of CSVs",
    )
    args = parser.parse_args()

    target = Path(args.path)
    if target.is_dir():
        csv_files = sorted(target.glob("*.csv"))
        csv_files = [f for f in csv_files if not f.name.startswith("_")]
    elif target.is_file():
        csv_files = [target]
    else:
        print(f"ERROR: Path not found: {target}")
        sys.exit(1)

    if not csv_files:
        print(f"No CSV files found in {target}")
        sys.exit(1)

    n_valid = 0
    n_invalid = 0

    for csv_path in csv_files:
        report = validate_file(csv_path)
        print_report(report)
        if report["valid"]:
            n_valid += 1
        else:
            n_invalid += 1

    if len(csv_files) > 1:
        print(f"\n{'=' * 60}")
        print(f"  VALIDATION SUMMARY: {n_valid} valid, {n_invalid} invalid out of {len(csv_files)} files")
        print(f"{'=' * 60}")

    sys.exit(0 if n_invalid == 0 else 1)


if __name__ == "__main__":
    main()
