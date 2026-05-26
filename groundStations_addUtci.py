#!/usr/bin/env python3
"""
Stage 1: compute UTCI (Universal Thermal Climate Index) per row.

Reads a CSV with the columns produced by the Exploratorium weather station
dump (raw header line, then a units line which is skipped). Deduplicates
exact-duplicate (time, station_id) rows that arise from a known ingest-
pipeline artifact, computes UTCI via pythermalcomfort, and writes an
enriched parquet/CSV with:
  - all original columns
  - utci_c, utci_f         : UTCI in C and F
  - utci_category          : 10-category stress label
  - tmrt_c                 : approximated mean radiant temperature
  - rh_from_dewpoint       : RH recomputed from dew point (sanity check)

USAGE
    python utci_stage1.py INPUT.csv OUTPUT.parquet
    python utci_stage1.py INPUT.csv OUTPUT.csv

DEPENDENCIES
    pip install pandas numpy pyarrow pythermalcomfort
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pythermalcomfort.models import utci as _pkg_utci


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------

def f_to_c(f):
    return (f - 32.0) * 5.0 / 9.0


def c_to_f(c):
    return c * 9.0 / 5.0 + 32.0


def mph_to_ms(mph):
    return mph * 0.44704


# ---------------------------------------------------------------------------
# Humidity
# ---------------------------------------------------------------------------

def rh_from_temp_and_dewpoint(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    """
    Magnus formula. Returns RH in percent. Used as the primary RH input
    to UTCI (dew point is generally more reliable than RH sensors over
    time) and as a sanity check against the reported humidity column.

    See https://en.wikipedia.org/wiki/Dew_point
    """
    a, b = 17.625, 243.04
    es_t  = np.exp(a * t_c  / (b + t_c))
    es_td = np.exp(a * td_c / (b + td_c))
    return 100.0 * es_td / es_t


# ---------------------------------------------------------------------------
# Mean radiant temperature approximation
# ---------------------------------------------------------------------------

def tmrt_from_solar(t_air_c: np.ndarray, sw_wm2: np.ndarray,
                    k: float = 0.025) -> np.ndarray:
    """
    Cheap shortwave-only MRT approximation:
        Tmrt ≈ Tair + k * SW_down

    k = 0.025 is a reasonable mid-range value for a standing pedestrian
    on a typical urban surface. Higher k (~0.04) for very reflective
    surfaces, lower (~0.015) for shaded environments. No longwave term
    means nighttime MRT equals Tair, which slightly underestimates
    cool-discomfort on clear nights — acceptable for a first pass.

    Not entirely realistic since stations have different reflected solar
    conditions but good enough.
    """
    return t_air_c + k * sw_wm2


# ---------------------------------------------------------------------------
# UTCI category labels (Bröde 2012, Table 1)
# ---------------------------------------------------------------------------

UTCI_BINS = [
    (-np.inf, -40.0, "extreme cold stress"),
    (-40.0,   -27.0, "very strong cold stress"),
    (-27.0,   -13.0, "strong cold stress"),
    (-13.0,     0.0, "moderate cold stress"),
    (  0.0,     9.0, "slight cold stress"),
    (  9.0,    26.0, "no thermal stress"),
    ( 26.0,    32.0, "moderate heat stress"),
    ( 32.0,    38.0, "strong heat stress"),
    ( 38.0,    46.0, "very strong heat stress"),
    ( 46.0,  np.inf, "extreme heat stress"),
]


def utci_category(utci_c: np.ndarray) -> np.ndarray:
    out = np.empty(utci_c.shape, dtype=object)
    out[:] = "unknown"
    for lo, hi, label in UTCI_BINS:
        mask = (utci_c >= lo) & (utci_c < hi)
        out[mask] = label
    return out


# ---------------------------------------------------------------------------
# UTCI computation via pythermalcomfort
# ---------------------------------------------------------------------------

def compute_utci_c(ta_c, tmrt_c, va_ms, rh_pct):
    """
    UTCI in deg C via pythermalcomfort.models.utci.

    - limit_inputs=True: returns NaN for inputs outside the standard's
      applicability range (-50 < Ta < 50 °C, MRT within Ta ± [-70, +30] K,
      0.5 < v < 17 m/s). Matches the canonical Bröde 2012 envelope.
    - round_output=False: full precision, no quantization to 0.1 °C.
    
    See https://pubmed.ncbi.nlm.nih.gov/21626294/ for Bröde 2012
    """
    ta_c   = np.asarray(ta_c,   dtype=float)
    tmrt_c = np.asarray(tmrt_c, dtype=float)
    va_ms  = np.asarray(va_ms,  dtype=float)
    rh_pct = np.asarray(rh_pct, dtype=float)

    result = _pkg_utci(
        tdb=ta_c, tr=tmrt_c, v=va_ms, rh=rh_pct,
        limit_inputs=True,
        round_output=False,
    )
    if hasattr(result, "utci"):
        return np.asarray(result.utci)
    return np.asarray(result)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load(path: Path) -> pd.DataFrame:
    """
    Load the CSV. Row 1 is the header, row 2 is the units row — skip it.
    Parse the time column as UTC. Deduplicate exact-duplicate rows that
    arise from a known logging artifact in the source data: the ingest
    pipeline re-logs the same sensor reading dozens of times with only
    `pressure_relative_inHg` and `pressure_absolute_inHg` jittering by
    ±0.01 in. These duplicates would over-weight certain instants in any
    downstream resample, so we drop them here at the source.
    """
    df = pd.read_csv(path, skiprows=[1], parse_dates=["time"],
                     low_memory=False)
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")

    n_before = len(df)

    # Assuming the first sample is reasonable.
    df = df.drop_duplicates(subset=["time", "station_id"], keep="first")
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Dropped {n_dropped:,} duplicate (time, station_id) rows "
              f"({100*n_dropped/n_before:.1f}% of input)")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Enrich
# ---------------------------------------------------------------------------

# mrt_k=0.025: mid-range pedestrian/urban-surface default; see tmrt_from_solar docstring for the range
def enrich(df: pd.DataFrame, mrt_k: float = 0.025) -> pd.DataFrame:
    """Add UTCI and intermediate columns."""
    t_c     = f_to_c(df["air_temp_f"].to_numpy())
    td_c    = f_to_c(df["dew_point_f"].to_numpy())
    sw      = df["solar_watts_per_sqm"].to_numpy()
    v_ms    = mph_to_ms(df["wind_speed_mph"].to_numpy())

    # Recompute RH from dew point; keep as a column for the sanity check
    # and as the actual input to UTCI (more reliable than the RH sensor).
    rh_calc = rh_from_temp_and_dewpoint(t_c, td_c)
    df["rh_from_dewpoint"] = rh_calc

    # MRT approximation (shortwave-only)
    tmrt_c = tmrt_from_solar(t_c, sw, k=mrt_k)
    df["tmrt_c"] = tmrt_c

    # No 2.5 m → 10 m wind-profile correction is applied. The neutral log law
    # assumes an equilibrated boundary layer over uniform-roughness fetch,
    # which doesn't hold inside the ~25 m × 200 m slip channel between two
    # two-story buildings: the sensor at 2.5 m sits below the displacement
    # height (~0.7·H ≈ 5 m), in the canyon's recirculating flow rather than
    # the boundary-layer log profile. The measured wind is treated as the
    # pedestrian-felt wind for UTCI. If absolute UTCI values become load-
    # bearing, the proper fix is empirical per-station scaling from
    # concurrent rooftop (EXPLORE3b) measurements by wind direction.
    #
    # Clip wind to UTCI's valid range. The polynomial returns NaN outside
    # [0.5, 17] m/s; clipping the low end avoids NaN on calm hours, the
    # high end matters less for this microclimate but is here for symmetry.
    v_clipped = np.clip(v_ms, 0.5, 17.0)

    utci_c = compute_utci_c(t_c, tmrt_c, v_clipped, rh_calc)
    df["utci_c"]        = utci_c
    df["utci_f"]        = c_to_f(utci_c)
    df["utci_category"] = utci_category(utci_c)

    return df


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unknown output extension: {path.suffix}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Stage 1: compute UTCI per row.")
    p.add_argument("input",  help="raw CSV path")
    p.add_argument("output", help="output parquet or CSV path")
    args = p.parse_args(argv[1:])

    inp, outp = Path(args.input), Path(args.output)
    df = load(inp)
    print(f"Loaded {len(df):,} rows from {inp}")
    print(f"  time range: {df['time'].min()} → {df['time'].max()}")
    print(f"  stations:   {sorted(df['station_id'].unique())}")

    df = enrich(df)

    rh_diff = (df["rh_from_dewpoint"] - df["outdoor_humidity"]).abs()
    print(f"\nRH sanity check (computed vs reported):")
    print(f"  mean abs diff: {rh_diff.mean():.2f} %")
    print(f"  max abs diff:  {rh_diff.max():.2f} %")

    n_nan = df["utci_c"].isna().sum()
    print(f"\nUTCI: {n_nan:,} NaN out of {len(df):,} ({100*n_nan/len(df):.2f}%)")
    print("\nCategory distribution:")
    print(df["utci_category"].value_counts().sort_index().to_string())

    write(df, outp)
    print(f"\nWrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
