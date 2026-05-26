#!/usr/bin/env python3
"""
Enrich the rooftop (Western Weather, EXPLORE3b) CSV with UTCI.

The rooftop schema is different from the ground stations (Campbell-class
logger, mostly SI units, 6-min sampling, no duplicate-row ingest artifact).
We adapt to the same enriched-parquet contract as groundStations.parquet
so downstream analysis can union both datasets cleanly.

USAGE
    python enrich_rooftop.py INPUT.csv OUTPUT.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pythermalcomfort.models import utci as _pkg_utci

# Reuse MRT model and category labels from the ground pipeline so both
# datasets share the same UTCI conventions.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utci_stage1 import (
    tmrt_from_solar, utci_category, c_to_f, rh_from_temp_and_dewpoint,
)


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=[1], parse_dates=["time"], low_memory=False)
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    n_before = len(df)
    df = df.drop_duplicates(subset=["time"], keep="first")
    dropped = n_before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} duplicate timestamps")
    return df.reset_index(drop=True)


def enrich(df: pd.DataFrame, mrt_k: float = 0.025) -> pd.DataFrame:
    t_c  = df["air_temp_c"].to_numpy()
    td_c = df["dew_point_c"].to_numpy()
    sw   = df["solar_radiation_Wm2"].to_numpy()

    # Scalar mean wind, not resultant_wind_speed_ms (vector mean) or gust_speed_ms.
    # UTCI wants the energy in the flow over the averaging interval; scalar mean
    # captures that. Resultant is always ≤ scalar and diverges when direction varies.  
    v_ms = df["wind_speed_ms"].to_numpy()

    # RH from dew point — matches ground pipeline (more stable across years
    # than vendor RH on either logger).
    rh_calc = rh_from_temp_and_dewpoint(t_c, td_c)
    df["rh_from_dewpoint"] = rh_calc

    # Same k as ground pipeline for cross-dataset comparability, even though the
    # rooftop has different surface albedo and view factor (more sky, no awning,
    # water visible). Trades a bit of per-site realism for a clean ground-vs-roof
    # comparison; revisit if absolute rooftop UTCI becomes load-bearing.
    df["tmrt_c"] = tmrt_from_solar(t_c, sw, k=mrt_k)

    v_clipped = np.clip(v_ms, 0.5, 17.0)
    result = _pkg_utci(
        tdb=t_c, tr=df["tmrt_c"].to_numpy(), v=v_clipped, rh=rh_calc,
        limit_inputs=True, round_output=False,
    )
    utci_c = np.asarray(result.utci) if hasattr(result, "utci") else np.asarray(result)
    df["utci_c"] = utci_c
    df["utci_f"] = c_to_f(utci_c)
    df["utci_category"] = utci_category(utci_c)

    # Hypothetical wind-mitigation scenarios for the candidate rooftop visitor
    # space (e.g., glass screens / wind baffles). Column number = fraction of
    # measured wind that *remains*: wind50pct = v × 0.5 (50% reduction),
    # wind25pct = v × 0.25 (75% reduction).
    #
    # Caveat: UTCI's valid range floors v at 0.5 m/s, so the scenarios have no
    # effect when scaled wind falls below 0.5 (i.e., for measured wind below
    # 1.0 m/s in wind50pct, below 2.0 m/s in wind25pct). The rooftop is windy
    # enough that this only bites in a small minority of hours, but the
    # mitigation comparison will under-represent the sheltered experience in
    # exactly those calm hours visitors would most appreciate it.
    tmrt = df["tmrt_c"].to_numpy()
    for pct, factor in (("50pct", 0.5), ("25pct", 0.25)):
        v_scen = np.clip(v_ms * factor, 0.5, 17.0)
        res = _pkg_utci(tdb=t_c, tr=tmrt, v=v_scen, rh=rh_calc,
                        limit_inputs=True, round_output=False)
        scen_c = np.asarray(res.utci) if hasattr(res, "utci") else np.asarray(res)
        df[f"utci_c_wind{pct}"] = scen_c

    return df


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    args = p.parse_args(argv[1:])

    inp, outp = Path(args.input), Path(args.output)
    df = load(inp)
    print(f"Loaded {len(df):,} rows from {inp.name}")
    print(f"  time: {df['time'].min()} → {df['time'].max()}")
    print(f"  stations: {sorted(df['station_id'].unique())}")

    df = enrich(df)

    rh_diff = (df["rh_from_dewpoint"] - df["relative_humidity"]).abs()
    print(f"\nRH sanity (dewpoint-derived vs reported):")
    print(f"  mean abs diff: {rh_diff.mean():.2f} %")
    print(f"  max  abs diff: {rh_diff.max():.2f} %")

    n_nan = int(df["utci_c"].isna().sum())
    print(f"\nUTCI NaN: {n_nan:,} / {len(df):,} ({100*n_nan/len(df):.2f}%)")
    print("Category distribution:")
    print(df["utci_category"].value_counts().to_string())

    df.to_parquet(outp, index=False)
    print(f"\nWrote {outp}")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
