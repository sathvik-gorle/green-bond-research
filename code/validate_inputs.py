#!/usr/bin/env python3
"""Sanity-check raw inputs and the feature matrix before running replication."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent

REQUIRED_RAW = [
    REPO_ROOT / "data/raw/greenbond021726_cleaned_long.csv",
    REPO_ROOT / "data/raw/green_bond_data2_cleaned.csv",
    REPO_ROOT / "data/raw/fred_extended.parquet",
    REPO_ROOT / "data/raw/yahoo_prices.parquet",
    REPO_ROOT / "data/raw/cpu_index.parquet",
    REPO_ROOT / "data/raw/noaa_disasters.parquet",
    REPO_ROOT / "data/raw/acri_daily.csv",
]

FEATURE_REQUIRE_COLS = [
    "Date",
    "r_diff_BGRN_LQD",
    "r_diff_green_corp",
    "VIX_z",
    "CPU_z",
    "delta_10y",
]


def fail(msg: str) -> None:
    print(f"validate_inputs: FAIL — {msg}", file=sys.stderr)
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"validate_inputs: OK — {msg}")


def main() -> None:
    for p in REQUIRED_RAW:
        if not p.exists() or p.stat().st_size == 0:
            fail(f"missing or empty raw file {p.relative_to(REPO_ROOT)}")

    cfg_path = CODE_DIR / "configs" / "config.yaml"
    if not cfg_path.exists():
        fail("missing configs/config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ds = cfg.get("date_range", {}).get("start")
    de = cfg.get("date_range", {}).get("end")
    if not ds or not de:
        fail("config missing date_range.start / date_range.end")
    start = pd.Timestamp(ds)
    end = pd.Timestamp(de)
    if end < start:
        fail("config date_range end before start")

    feat_path = REPO_ROOT / "data/processed/features.csv"
    if not feat_path.exists():
        fail(
            "data/processed/features.csv not found — run python3 code/replication.py --stage features "
            "(or stage full)"
        )

    size = feat_path.stat().st_size
    if size < 1024:
        fail(f"features.csv is too small ({size} bytes); expected full panel")

    df = pd.read_csv(feat_path, nrows=5)
    for c in FEATURE_REQUIRE_COLS:
        if c not in df.columns:
            fail(f"features.csv missing column {c!r}")

    df_full = pd.read_csv(feat_path, parse_dates=["Date"])
    if df_full.empty:
        fail("features.csv has zero rows")
    dt = pd.to_datetime(df_full["Date"], utc=False)
    bad = (~dt.notna()).sum()
    if bad:
        fail(f"features.csv has {bad} invalid dates")

    n = len(df_full)
    if n < 1000:
        fail(f"features.csv unexpectedly short ({n} rows)")
    ok(f"features.csv rows={n}, cols={df_full.shape[1]}")

    for col in ("r_diff_BGRN_LQD", "r_LQD"):
        if col in df_full.columns:
            s = df_full[col].dropna()
            if len(s):
                extremes = (s.abs() > 500).mean()
                if extremes > 0.01:
                    fail(f"{col}: >1% observations with |return| > 500 (check units)")

    exp_tables = REPO_ROOT / "data/exports/tables"
    exp_figs = REPO_ROOT / "data/exports/figures"
    exp_tables.mkdir(parents=True, exist_ok=True)
    exp_figs.mkdir(parents=True, exist_ok=True)
    ok(f"exports dirs writable: tables, figures")

    print("validate_inputs: all checks passed")


if __name__ == "__main__":
    main()
