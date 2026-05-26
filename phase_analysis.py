#!/usr/bin/env python3
"""
Three-phase microclimate analysis of the Exploratorium pier and rooftop.

  Phase A — Station fingerprinting: confirm physical layout via data
            signatures (midday solar settles 104's awning question, wind
            roses settle who's most exposed).
  Phase B — Wind & turbulence climatology: rooftop sigma_theta, gust
            factors, resultant/scalar ratio, extreme prevalence; ground
            stations get a gust-factor + calm/gale comparison.
  Phase C — Comfort prediction: per-site UTCI distributions, monthly
            diurnal heatmaps, visitor-hours stress mix; explicit
            comparison across the canonical sites (100 shade, 102 sun,
            104 corner, EXPLORE3b roof) for layout recommendations.

All times are converted to America/Los_Angeles for diurnal analysis.
Visitor hours = 10:00–17:00 local.

USAGE
    python phase_analysis.py [--out-dir phase_analysis] \
        [--ground groundStations.parquet] [--roof rooftopStation.parquet]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

LOCAL_TZ = "America/Los_Angeles"
VISITOR_HOURS = (10, 17)  # [10:00, 17:00) local

# Roles per the saved station_layout memory
STATION_ROLE = {
    "EXPLORE8_101": "awning · land end",
    "EXPLORE8_100": "awning · mid slip",
    "EXPLORE8_102": "open deck",
    "EXPLORE8_103": "over water · wind funnel",
    "EXPLORE8_104": "sunny corner",
    "EXPLORE3b":    "rooftop · Western Weather",
}
GROUND_ORDER = ["EXPLORE8_101", "EXPLORE8_100", "EXPLORE8_102",
                "EXPLORE8_103", "EXPLORE8_104"]
ALL_ORDER = GROUND_ORDER + ["EXPLORE3b"]

# Stations called out by the user's question
KEY_SITES = ["EXPLORE8_100", "EXPLORE8_102", "EXPLORE8_104", "EXPLORE3b"]

# UTCI category banding for visitor comfort. "Slight cold stress" (UTCI 0–9 °C)
# is intentionally NOT counted as comfortable here: UTCI assumes adaptive
# clothing in that band (~1.5 clo), but visitors arriving on the rooftop from
# the museum interior are typically in indoor clothing (~0.5 clo) and would
# feel visibly cold even with no wind. The nominal UTCI 9 °C floor is still
# permissive for this use case; revisit if a use-case-tight band is needed.
COMFORT_CATS = {"no thermal stress"}

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]


# UTCI category bins (Bröde 2012, Table 1). Used only by --dry-bulb mode to
# recategorize the substituted dry-bulb temperatures; the normal pipeline
# gets categories pre-computed in the parquets by utci_stage1.py and
# enrich_rooftop.py.
_UTCI_BINS_C = [
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


def _categorize_utci_c(utci_c: np.ndarray) -> np.ndarray:
    out = np.empty(len(utci_c), dtype=object)
    out[:] = "unknown"
    for lo, hi, label in _UTCI_BINS_C:
        mask = (utci_c >= lo) & (utci_c < hi)
        out[mask] = label
    return out


def _replace_utci_with_dry_bulb(df: pd.DataFrame) -> pd.DataFrame:
    """Educational --dry-bulb mode: overwrite the UTCI columns with the air
    temperature, so phase_analysis runs entirely off dry-bulb temp. The
    rooftop scenario columns (which would normally differ from baseline by
    the wind term) collapse to the baseline value — dry-bulb is unaffected
    by wind, so the 50% / 75% reduction columns become identical to utci_c,
    which is itself the point of the experiment."""
    tf = df["air_temp_f"].to_numpy()
    tc = (tf - 32.0) * 5.0 / 9.0
    df["utci_f"] = tf
    df["utci_c"] = tc
    df["utci_category"] = _categorize_utci_c(tc)
    for col in ("utci_c_wind50pct", "utci_c_wind25pct"):
        if col in df.columns:
            df[col] = tc
    return df


# ---------------------------------------------------------------------------
# Load + harmonize
# ---------------------------------------------------------------------------

def load_data(ground_path: Path, roof_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = pd.read_parquet(ground_path)
    r = pd.read_parquet(roof_path)
    for df in (g, r):
        if df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize("UTC")
        df["time_local"] = df["time"].dt.tz_convert(LOCAL_TZ)
        df["hour"] = df["time_local"].dt.hour
        df["month"] = df["time_local"].dt.month
        df["date"] = df["time_local"].dt.date
    return g, r


# ---------------------------------------------------------------------------
# Per-station UTCI figures (carried over from the prior 4-phase analysis)
# ---------------------------------------------------------------------------

def _label_cells(ax, pivot, fmt, *, center=None, hide_zero=False,
                 white_above=None, font=6.5):
    """Annotate each cell of a heatmap with its numeric value.

    Either `center` (label color flips based on distance from centre) or
    `white_above` (label color flips above a fixed threshold) controls
    contrast. `hide_zero=True` omits zero-valued cells to reduce noise.
    """
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iat[i, j]
            if pd.isna(v):
                continue
            if hide_zero and v == 0:
                continue
            if center is not None:
                color = "white" if abs(v - center) > 14 else "black"
            elif white_above is not None:
                color = "white" if v > white_above else "black"
            else:
                color = "black"
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    color=color, fontsize=font)


def _per_station_diurnal_utci(sub: pd.DataFrame, path: Path, station_id: str) -> None:
    pivot = sub.pivot_table(values="utci_f", index="hour",
                            columns="month", aggfunc="mean")
    vmin, vmax, center = 45, 75, 68
    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(pivot, aspect="auto", origin="lower", cmap="RdBu_r", norm=norm)
    cbar = plt.colorbar(im, ax=ax, label="Mean UTCI (°F)", extend="both")
    cbar.ax.axhline(center, color="black", lw=0.8)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(24))
    ax.set_yticklabels(range(24))
    ax.set_xlabel("Month")
    ax.set_ylabel("Hour of day (local)")
    role = STATION_ROLE.get(station_id, "")
    ax.set_title(f"{station_id} ({role}) — Mean UTCI by hour × month\n"
                 "(white ≈ 68 °F thermal neutrality)")
    _label_cells(ax, pivot, "{:.0f}", center=center, font=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _per_station_diurnal_tails(sub: pd.DataFrame, path: Path, station_id: str) -> None:
    """2×2 tails grid: P(cold), P(hot), P05 UTCI, P95 UTCI."""
    sub = sub.copy()
    sub["is_hot"]  = sub["utci_f"] > 78
    sub["is_cold"] = sub["utci_f"] < 48

    p_hot  = sub.pivot_table(values="is_hot",  index="hour", columns="month", aggfunc="mean") * 100
    p_cold = sub.pivot_table(values="is_cold", index="hour", columns="month", aggfunc="mean") * 100
    p95    = sub.pivot_table(values="utci_f",  index="hour", columns="month",
                              aggfunc=lambda x: x.quantile(0.95))
    p05    = sub.pivot_table(values="utci_f",  index="hour", columns="month",
                              aggfunc=lambda x: x.quantile(0.05))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    center = 68

    # Top-left: P(cold)
    im = axes[0,0].imshow(p_cold, aspect="auto", origin="lower",
                          cmap="Blues", vmin=0, vmax=max(20, p_cold.max().max()))
    plt.colorbar(im, ax=axes[0,0], label="Cold-stress probability (%)")
    axes[0,0].set_xticks(range(len(p_cold.columns))); axes[0,0].set_xticklabels(p_cold.columns)
    axes[0,0].set_yticks(range(0, 24, 2));            axes[0,0].set_yticklabels(range(0, 24, 2))
    axes[0,0].set_xlabel("Month"); axes[0,0].set_ylabel("Hour")
    axes[0,0].set_title("Cold-stress probability  (P(UTCI < 48 °F), %)")
    _label_cells(axes[0,0], p_cold, "{:.0f}", white_above=15, hide_zero=True)

    # Top-right: P(hot)
    im = axes[0,1].imshow(p_hot, aspect="auto", origin="lower",
                          cmap="Reds", vmin=0, vmax=max(20, p_hot.max().max()))
    plt.colorbar(im, ax=axes[0,1], label="Heat-stress probability (%)")
    axes[0,1].set_xticks(range(len(p_hot.columns))); axes[0,1].set_xticklabels(p_hot.columns)
    axes[0,1].set_yticks(range(0, 24, 2));           axes[0,1].set_yticklabels(range(0, 24, 2))
    axes[0,1].set_xlabel("Month"); axes[0,1].set_ylabel("Hour")
    axes[0,1].set_title("Heat-stress probability  (P(UTCI > 78 °F), %)")
    _label_cells(axes[0,1], p_hot, "{:.0f}", white_above=15, hide_zero=True)

    # Bottom-left: P05 UTCI (cold tail)
    im = axes[1,0].imshow(p05, aspect="auto", origin="lower", cmap="RdBu_r",
                          norm=TwoSlopeNorm(vmin=35, vcenter=center, vmax=75))
    plt.colorbar(im, ax=axes[1,0], label="P05 UTCI (°F)", extend="both")
    axes[1,0].set_xticks(range(len(p05.columns))); axes[1,0].set_xticklabels(p05.columns)
    axes[1,0].set_yticks(range(0, 24, 2));         axes[1,0].set_yticklabels(range(0, 24, 2))
    axes[1,0].set_xlabel("Month"); axes[1,0].set_ylabel("Hour")
    axes[1,0].set_title("Cold tail: P05 UTCI by hour × month\n"
                        "(5% of hours are colder than this)")
    _label_cells(axes[1,0], p05, "{:.0f}", center=center)

    # Bottom-right: P95 UTCI (heat tail)
    im = axes[1,1].imshow(p95, aspect="auto", origin="lower", cmap="RdBu_r",
                          norm=TwoSlopeNorm(vmin=45, vcenter=center, vmax=95))
    plt.colorbar(im, ax=axes[1,1], label="P95 UTCI (°F)", extend="both")
    axes[1,1].set_xticks(range(len(p95.columns))); axes[1,1].set_xticklabels(p95.columns)
    axes[1,1].set_yticks(range(0, 24, 2));         axes[1,1].set_yticklabels(range(0, 24, 2))
    axes[1,1].set_xlabel("Month"); axes[1,1].set_ylabel("Hour")
    axes[1,1].set_title("Heat tail: P95 UTCI by hour × month\n"
                        "(5% of hours are warmer than this)")
    _label_cells(axes[1,1], p95, "{:.0f}", center=center)

    role = STATION_ROLE.get(station_id, "")
    fig.suptitle(f"{station_id} ({role}) — What the mean hides: tail behavior of pedestrian comfort",
                 fontsize=14, y=1.00)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _category_mix_monthly(sub: pd.DataFrame) -> pd.DataFrame:
    """Monthly % of hours in each cold/heat band used by _per_station_category_mix."""
    return sub.groupby("month").agg(
        p_slight_cold = ("utci_f", lambda x: ((x >= 39) & (x < 48)).mean()),
        p_mod_cold    = ("utci_f", lambda x: ((x >= 8.6) & (x < 39)).mean()),
        p_strong_cold = ("utci_f", lambda x: (x < 8.6).mean()),
        p_mod_heat    = ("utci_f", lambda x: ((x >= 78) & (x < 90)).mean()),
        p_strong_heat = ("utci_f", lambda x: ((x >= 90) & (x < 100)).mean()),
        p_vs_heat     = ("utci_f", lambda x: (x >= 100).mean()),
    ) * 100


def _per_station_category_mix(sub: pd.DataFrame, path: Path, station_id: str,
                              ylim: tuple[float, float] | None = None) -> None:
    """Diverging cold/heat bars by month; comfort hours dropped to surface tails.

    `ylim` (cold_min, heat_max) — when provided, applied to all stations so the
    Y axis is identical across the per-station figures and they can be flipped
    through without rescaling.
    """
    monthly = _category_mix_monthly(sub)

    fig, ax = plt.subplots(figsize=(11, 6))
    months = monthly.index.to_numpy()
    ax.bar(months, -monthly["p_slight_cold"], color="#9ecae1",
           label="slight cold (UTCI 39–48 °F)")
    ax.bar(months, -monthly["p_mod_cold"], bottom=-monthly["p_slight_cold"],
           color="#4292c6", label="moderate cold (UTCI 9–39 °F)")
    ax.bar(months, -monthly["p_strong_cold"],
           bottom=-(monthly["p_slight_cold"] + monthly["p_mod_cold"]),
           color="#08519c", label="strong+ cold")
    ax.bar(months, monthly["p_mod_heat"], color="#fcae91",
           label="moderate heat (UTCI 78–90 °F)")
    ax.bar(months, monthly["p_strong_heat"], bottom=monthly["p_mod_heat"],
           color="#de2d26", label="strong heat (90–100 °F)")
    ax.bar(months, monthly["p_vs_heat"],
           bottom=monthly["p_mod_heat"] + monthly["p_strong_heat"],
           color="#67000d", label="very strong+ heat")

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Month")
    ax.set_ylabel("% of hours  ← cold-side    heat-side →")
    role = STATION_ROLE.get(station_id, "")
    ax.set_title(f"{station_id} ({role}) — All hours: discomfort only; cold and heat tails by month\n"
                 "(comfort hours dropped to make tails visible)")
    ax.set_xticks(months)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    yt = ax.get_yticks()
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{abs(int(t))}%" for t in yt])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def phase_a_per_station(g: pd.DataFrame, r: pd.DataFrame, out: Path) -> None:
    """Diurnal-UTCI, tails, and category-mix figures for every station."""
    out.mkdir(parents=True, exist_ok=True)
    cols = ["station_id", "utci_f", "hour", "month"]
    both = pd.concat([g[cols], r[cols]], ignore_index=True)
    both = both.dropna(subset=["utci_f"])

    # Pre-pass: shared cold/heat Y limits across all stations so the per-station
    # category_mix figures use identical Y axes.
    cold_max = heat_max = 0.0
    for s in ALL_ORDER:
        sub = both[both["station_id"] == s]
        if len(sub) == 0:
            continue
        m = _category_mix_monthly(sub)
        cold_max = max(cold_max, (m["p_slight_cold"] + m["p_mod_cold"] + m["p_strong_cold"]).max())
        heat_max = max(heat_max, (m["p_mod_heat"]   + m["p_strong_heat"] + m["p_vs_heat"]).max())
    pad = 1.05
    shared_ylim = (-cold_max * pad, heat_max * pad if heat_max > 0 else cold_max * 0.1)

    for s in ALL_ORDER:
        sub = both[both["station_id"] == s]
        if len(sub) == 0:
            print(f"  [skip] {s}: no data")
            continue
        slug = s.split("_")[-1] if s.startswith("EXPLORE8_") else "Roof"
        _per_station_diurnal_utci(sub, out / f"{slug}_diurnal_utci.png", s)
        _per_station_diurnal_tails(sub, out / f"{slug}_diurnal_tails.png", s)
        _per_station_category_mix(sub, out / f"{slug}_category_mix.png", s,
                                  ylim=shared_ylim)
        print(f"  {s} ({len(sub):,} rows) → {slug}_diurnal_utci|tails|category_mix.png")


# ---------------------------------------------------------------------------
# Phase A — Station fingerprinting
# ---------------------------------------------------------------------------

def phase_a(g: pd.DataFrame, r: pd.DataFrame, out: Path) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)

    # Per-station midday (11-14 local) solar percentiles — the awning test.
    midday = g[(g["hour"] >= 11) & (g["hour"] <= 14)]
    solar_q = (
        midday.groupby("station_id")["solar_watts_per_sqm"]
        .quantile([0.5, 0.9, 0.99])
        .unstack(level=-1)
        .rename(columns={0.5: "midday_solar_p50",
                          0.9: "midday_solar_p90",
                          0.99: "midday_solar_p99"})
        .reindex(GROUND_ORDER)
    )

    # Ensemble-relative RH and temperature bias
    g["t_c"] = (g["air_temp_f"] - 32.0) * 5.0 / 9.0
    bias = (
        g.groupby("station_id")
         .agg(t_c_mean=("t_c", "mean"),
              rh_mean=("rh_from_dewpoint", "mean"),
              v_ms_mean=("wind_speed_mph", lambda s: (s * 0.44704).mean()),
              gust_ms_mean=("wind_gust_mph", lambda s: (s * 0.44704).mean()))
         .reindex(GROUND_ORDER)
    )
    ensemble = bias.mean()
    bias["t_c_bias"] = bias["t_c_mean"] - ensemble["t_c_mean"]
    bias["rh_bias"] = bias["rh_mean"] - ensemble["rh_mean"]
    bias["v_ratio"] = bias["v_ms_mean"] / ensemble["v_ms_mean"]

    sig = pd.concat([solar_q, bias], axis=1)
    sig["role"] = sig.index.map(STATION_ROLE)
    sig.to_csv(out / "station_signature.csv")

    # Normalize ground + rooftop into a common per-row view for the A1/A2/A3
    # figures. The two schemas use different units and column names; this is
    # the one place the schema difference is hidden.
    common = ["station_id", "time_local", "hour", "month"]
    ground_norm = g[common].copy()
    ground_norm["solar_wm2"]    = g["solar_watts_per_sqm"]
    ground_norm["v_ms"]         = g["wind_speed_mph"] * 0.44704
    ground_norm["wind_dir_deg"] = g["wind_direction"]
    roof_norm = r[common].copy()
    roof_norm["solar_wm2"]    = r["solar_radiation_Wm2"]
    roof_norm["v_ms"]         = r["wind_speed_ms"]
    roof_norm["wind_dir_deg"] = r["resultant_wind_direction"]
    phase_a_df = pd.concat([ground_norm, roof_norm], ignore_index=True)

    def _stn_label(s):
        return s.split("_")[-1] if s.startswith("EXPLORE8_") else "Roof"

    # Figure A1: midday solar by station, monthly p50 — all 6 stations
    midday_all = phase_a_df[(phase_a_df["hour"] >= 11) & (phase_a_df["hour"] <= 14)].copy()
    midday_all["yyyymm"] = midday_all["time_local"].dt.to_period("M")
    monthly_solar = (
        midday_all.groupby(["station_id", "yyyymm"])["solar_wm2"]
        .median().unstack(level=0).reindex(columns=ALL_ORDER)
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    for s in ALL_ORDER:
        style = dict(marker="s", linewidth=2.0, linestyle="--", color="black") \
                if s == "EXPLORE3b" else dict(marker="o")
        ax.plot(range(len(monthly_solar.index)), monthly_solar[s],
                label=f"{_stn_label(s)} · {STATION_ROLE[s]}", **style)
    ax.set_xticks(range(len(monthly_solar.index)))
    ax.set_xticklabels([str(p) for p in monthly_solar.index], rotation=45)
    ax.set_ylabel("Midday (11–14 PT) median solar  [W/m²]")
    ax.set_title("Phase A: Midday solar by station\n"
                 "Rooftop (dashed) is unobstructed reference.")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "A1_midday_solar.png", dpi=150)
    plt.close(fig)

    # Figure A2: wind roses — 2×3 grid (5 ground + rooftop). 15° bins to match
    # B3. Pre-compute per-station stats so all panels share a single color
    # scale (global vmax = max per-sector mean wind across all stations),
    # making cross-panel color comparison meaningful.
    bins = np.arange(0, 361, 15)
    theta = np.deg2rad(bins[:-1] + 7.5)
    per_station = {}
    for s in ALL_ORDER:
        sub = phase_a_df[phase_a_df["station_id"] == s]
        dirs = sub["wind_dir_deg"].to_numpy()
        spd  = sub["v_ms"].to_numpy()
        mask = np.isfinite(dirs) & np.isfinite(spd) & (spd > 0.5)
        dirs = dirs[mask]; spd = spd[mask]
        weights, _ = np.histogram(dirs, bins=bins, weights=spd)
        counts, _ = np.histogram(dirs, bins=bins)
        mean_spd = np.divide(weights, counts, out=np.zeros_like(weights),
                             where=counts > 0)
        freq = counts / counts.sum() if counts.sum() else counts
        per_station[s] = (freq, mean_spd)

    global_vmax = max((ms.max() for _, ms in per_station.values()), default=1.0) or 1.0

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5),
                             subplot_kw=dict(projection="polar"))
    for ax, s in zip(axes.flat, ALL_ORDER):
        freq, mean_spd = per_station[s]
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.bar(theta, freq, width=np.deg2rad(13),
               color=plt.cm.viridis(mean_spd / global_vmax),
               edgecolor="black", linewidth=0.3)
        ax.set_title(f"{_stn_label(s)}\n{STATION_ROLE[s]}", fontsize=9)
        ax.set_yticklabels([])
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels(["N", "E", "S", "W"], fontsize=8)
    fig.suptitle("Phase A: Per-station wind rose — frequency by direction, color = mean speed within sector",
                 fontsize=11)
    sm = plt.cm.ScalarMappable(cmap="viridis",
                                norm=mpl.colors.Normalize(vmin=0, vmax=global_vmax))
    sm.set_array([])
    fig.tight_layout(rect=[0, 0, 0.93, 1.0])
    fig.colorbar(sm, ax=axes.ravel().tolist(), label="Mean speed within sector [m/s]",
                 shrink=0.6, pad=0.04)
    fig.savefig(out / "A2_wind_roses.png", dpi=150)
    plt.close(fig)

    # Figure A3: per-station mean wind speed by hour-of-day — includes Roof
    hourly = (
        phase_a_df.groupby(["station_id", "hour"])["v_ms"].mean()
                  .unstack(level=0).reindex(columns=ALL_ORDER)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    for s in ALL_ORDER:
        style = dict(marker="s", linewidth=2.0, linestyle="--", color="black") \
                if s == "EXPLORE3b" else dict(marker="o")
        ax.plot(hourly.index, hourly[s],
                label=f"{_stn_label(s)} · {STATION_ROLE[s]}", **style)
    ax.set_xlabel("Hour of day (local)")
    ax.set_ylabel("Mean wind speed  [m/s]")
    ax.set_title("Phase A: Diurnal wind pattern by station")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(out / "A3_diurnal_wind.png", dpi=150)
    plt.close(fig)

    # Per-station diurnal-UTCI, tails, and category-mix carried over from
    # the earlier 4-phase analysis. Generated for all 6 stations.
    print("  generating per-station UTCI figures …")
    phase_a_per_station(g, r, out / "per_station")

    return sig


# ---------------------------------------------------------------------------
# Phase B — Wind & turbulence climatology
# ---------------------------------------------------------------------------

def phase_b(g: pd.DataFrame, r: pd.DataFrame, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)

    # --- Rooftop: monthly + diurnal climatology ----------------------------
    r = r.copy()
    r["gust_factor"] = r["gust_speed_ms"] / r["wind_speed_ms"].replace(0, np.nan)
    # Steadiness: resultant/scalar ratio. ~1 = perfectly steady direction.
    r["steadiness"] = r["resultant_wind_speed_ms"] / r["wind_speed_ms"].replace(0, np.nan)
    # Use Campbell sigma_theta (Yamartino — circular-aware)
    r["sigma_theta"] = r["campbell_wind_direction_std_dev"]

    monthly = (
        r.groupby("month")
         .agg(mean_wind=("wind_speed_ms", "mean"),
              p95_wind=("wind_speed_ms", lambda s: s.quantile(0.95)),
              p99_wind=("wind_speed_ms", lambda s: s.quantile(0.99)),
              mean_gust=("gust_speed_ms", "mean"),
              gust_factor=("gust_factor", "mean"),
              sigma_theta=("sigma_theta", "mean"),
              steadiness=("steadiness", "mean"))
    )
    monthly.to_csv(out / "rooftop_monthly.csv")

    diurnal = (
        r.groupby("hour")
         .agg(mean_wind=("wind_speed_ms", "mean"),
              gust_factor=("gust_factor", "mean"),
              sigma_theta=("sigma_theta", "mean"),
              steadiness=("steadiness", "mean"))
    )
    diurnal.to_csv(out / "rooftop_diurnal.csv")

    # B1: rooftop monthly 4-panel
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    months = monthly.index
    axes[0,0].plot(months, monthly["mean_wind"], "o-", label="mean")
    axes[0,0].plot(months, monthly["p95_wind"], "s--", label="p95")
    axes[0,0].plot(months, monthly["p99_wind"], "^:", label="p99")
    axes[0,0].set_ylabel("Wind speed [m/s]"); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)
    axes[0,0].set_title("Wind speed by month (rooftop)")
    axes[0,1].plot(months, monthly["gust_factor"], "o-", color="C1")
    axes[0,1].set_ylabel("Gust factor (peak / mean)"); axes[0,1].grid(alpha=0.3)
    axes[0,1].set_title("Gustiness by month")
    axes[1,0].plot(months, monthly["sigma_theta"], "o-", color="C2")
    axes[1,0].set_ylabel("σθ (wind direction std dev) [°]"); axes[1,0].grid(alpha=0.3)
    axes[1,0].set_title("Turbulence proxy by month")
    axes[1,1].plot(months, monthly["steadiness"], "o-", color="C3")
    axes[1,1].set_ylabel("Resultant / scalar speed (1 = steady)")
    axes[1,1].set_ylim(0, 1.05); axes[1,1].grid(alpha=0.3)
    axes[1,1].set_title("Directional steadiness by month")
    for ax in axes.flat:
        ax.set_xticks(range(1,13)); ax.set_xticklabels(MONTH_NAMES, fontsize=8)
    fig.suptitle("Phase B: Rooftop monthly climatology", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "B1_rooftop_monthly.png", dpi=150)
    plt.close(fig)

    # B2: rooftop diurnal 4-panel
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    h = diurnal.index
    axes[0,0].plot(h, diurnal["mean_wind"], "o-"); axes[0,0].set_ylabel("Mean wind [m/s]"); axes[0,0].grid(alpha=0.3); axes[0,0].set_title("Diurnal wind speed")
    axes[0,1].plot(h, diurnal["gust_factor"], "o-", color="C1"); axes[0,1].set_ylabel("Gust factor"); axes[0,1].grid(alpha=0.3); axes[0,1].set_title("Diurnal gustiness")
    axes[1,0].plot(h, diurnal["sigma_theta"], "o-", color="C2"); axes[1,0].set_ylabel("σθ [°]"); axes[1,0].grid(alpha=0.3); axes[1,0].set_title("Diurnal turbulence")
    axes[1,1].plot(h, diurnal["steadiness"], "o-", color="C3"); axes[1,1].set_ylabel("Steadiness"); axes[1,1].grid(alpha=0.3); axes[1,1].set_title("Diurnal directional steadiness")
    for ax in axes.flat:
        ax.set_xlabel("Hour (local)"); ax.set_xticks(range(0,24,2))
    fig.suptitle("Phase B: Rooftop diurnal climatology", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "B2_rooftop_diurnal.png", dpi=150)
    plt.close(fig)

    # B3: rooftop wind rose (annual)
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection="polar"))
    dirs = r["resultant_wind_direction"].to_numpy()
    spd  = r["wind_speed_ms"].to_numpy()
    mask = np.isfinite(dirs) & np.isfinite(spd) & (spd > 0.5)
    dirs = dirs[mask]; spd = spd[mask]
    bins = np.arange(0, 361, 15)  # finer for rooftop
    counts, _ = np.histogram(dirs, bins=bins)
    weighted, _ = np.histogram(dirs, bins=bins, weights=spd)
    mean_spd = np.divide(weighted, counts, out=np.zeros_like(weighted),
                         where=counts > 0)
    freq = counts / counts.sum()
    theta = np.deg2rad(bins[:-1] + 7.5)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    bars = ax.bar(theta, freq, width=np.deg2rad(13),
                  color=plt.cm.viridis(mean_spd / mean_spd.max()),
                  edgecolor="black", linewidth=0.3)
    ax.set_title("Phase B: Rooftop annual wind rose\n(frequency; color = mean speed)")
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"])
    sm = plt.cm.ScalarMappable(cmap="viridis",
                                norm=mpl.colors.Normalize(vmin=0, vmax=mean_spd.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Mean speed [m/s]", shrink=0.7)
    fig.tight_layout()
    fig.savefig(out / "B3_rooftop_windrose.png", dpi=150)
    plt.close(fig)

    # --- Ground stations: gust factor + calm/gale prevalence -------------
    g2 = g.copy()
    g2["v_ms"] = g2["wind_speed_mph"] * 0.44704
    g2["gust_ms"] = g2["wind_gust_mph"] * 0.44704
    g2["gf"] = g2["gust_ms"] / g2["v_ms"].replace(0, np.nan)

    # Per-station extreme prevalence — share roof in same table
    gust_thresholds = [5, 10, 15, 20]  # m/s
    rows = []
    for s in GROUND_ORDER:
        sub = g2[g2["station_id"] == s]
        v = sub["v_ms"].dropna()
        gust = sub["gust_ms"].dropna()
        rec = {
            "station_id": s,
            "role": STATION_ROLE[s],
            "n_obs": len(v),
            "calm_pct": 100 * (v < 0.5).mean(),
            "mean_wind_ms": v.mean(),
            "p95_wind_ms": v.quantile(0.95),
            "p99_wind_ms": v.quantile(0.99),
            "mean_gust_factor": sub["gf"].mean(),
        }
        for thr in gust_thresholds:
            rec[f"gust>{thr}_pct"] = 100 * (gust > thr).mean()
        rows.append(rec)

    # Rooftop comparable row
    v = r["wind_speed_ms"].dropna()
    gust = r["gust_speed_ms"].dropna()
    rec = {
        "station_id": "EXPLORE3b",
        "role": STATION_ROLE["EXPLORE3b"],
        "n_obs": len(v),
        "calm_pct": 100 * (v < 0.5).mean(),
        "mean_wind_ms": v.mean(),
        "p95_wind_ms": v.quantile(0.95),
        "p99_wind_ms": v.quantile(0.99),
        "mean_gust_factor": r["gust_factor"].mean(),
    }
    for thr in gust_thresholds:
        rec[f"gust>{thr}_pct"] = 100 * (gust > thr).mean()
    rows.append(rec)

    extremes = pd.DataFrame(rows).set_index("station_id")
    extremes.to_csv(out / "wind_extremes.csv")

    # B4: gust factor comparison across stations (annual mean by station, plus monthly grid)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # left: bar of mean gust factor
    gf_means = [g2[g2["station_id"]==s]["gf"].mean() for s in GROUND_ORDER]
    gf_means.append(r["gust_factor"].mean())
    labels = [s.split("_")[-1] if "_" in s else "Roof" for s in ALL_ORDER]
    colors = ["#4c72b0"]*5 + ["#dd8452"]
    axes[0].bar(labels, gf_means, color=colors, edgecolor="black")
    axes[0].set_ylabel("Mean gust factor (gust / mean wind)")
    axes[0].set_title("Annual mean gust factor by site")
    axes[0].axhline(1.0, color="grey", linestyle="--", linewidth=0.6)
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(gf_means):
        axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    # right: stacked calm % and p95 wind
    sites = ALL_ORDER
    calm = [extremes.loc[s, "calm_pct"] for s in sites]
    p95  = [extremes.loc[s, "p95_wind_ms"] for s in sites]
    x = np.arange(len(sites))
    width = 0.35
    ax2 = axes[1]
    ax2.bar(x - width/2, calm, width, color="#5d8aa8", label="Calm % (v<0.5 m/s)")
    ax2b = ax2.twinx()
    ax2b.bar(x + width/2, p95, width, color="#c47451", label="p95 wind [m/s]")
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel("Calm %  (blue)")
    ax2b.set_ylabel("p95 wind speed [m/s]  (orange)")
    ax2.set_title("Calm prevalence vs upper-tail wind")
    ax2.grid(axis="y", alpha=0.3)
    fig.suptitle("Phase B: Wind exposure across sites", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "B4_wind_comparison.png", dpi=150)
    plt.close(fig)

    return {"monthly": monthly, "diurnal": diurnal, "extremes": extremes}


# ---------------------------------------------------------------------------
# Phase C — Comfort prediction
# ---------------------------------------------------------------------------

def phase_c(g: pd.DataFrame, r: pd.DataFrame, out: Path) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)

    # Union ground + roof on a common schema for comparison. Use °F throughout
    # — matches the per-station diurnal heatmaps and the rest of the report.
    g2 = g[["station_id","time_local","hour","month","utci_f","utci_category"]].copy()
    r2 = r[["station_id","time_local","hour","month","utci_f","utci_category"]].copy()
    both = pd.concat([g2, r2], ignore_index=True)

    # Visitor-hours subset (10–17 local)
    vh = both[(both["hour"] >= VISITOR_HOURS[0]) & (both["hour"] < VISITOR_HOURS[1])].copy()

    # Per-site summary: % no-stress, stress fractions, UTCI percentiles in visitor hours
    rows = []
    for s in ALL_ORDER:
        sub = vh[vh["station_id"] == s]
        cat = sub["utci_category"]
        rec = {
            "station_id": s,
            "role": STATION_ROLE[s],
            "n_visitor_obs": len(sub),
            "pct_no_stress": 100 * cat.isin(COMFORT_CATS).mean(),
            "pct_slight_cold_stress": 100 * cat.isin({"slight cold stress"}).mean(),
            "pct_heat_stress": 100 * cat.isin(
                {"moderate heat stress","strong heat stress",
                 "very strong heat stress","extreme heat stress"}).mean(),
            "pct_moderate_cold_or_worse": 100 * cat.isin(
                {"moderate cold stress","strong cold stress",
                 "very strong cold stress","extreme cold stress"}).mean(),
            "utci_p05_f": sub["utci_f"].quantile(0.05),
            "utci_p50_f": sub["utci_f"].quantile(0.50),
            "utci_p95_f": sub["utci_f"].quantile(0.95),
        }
        rows.append(rec)
    site_summary = pd.DataFrame(rows).set_index("station_id")
    site_summary.to_csv(out / "site_comfort_summary.csv")

    # C1: UTCI distribution per site (visitor hours). °F bands: 9–26 °C ≈
    # 48.2–78.8 °F (no thermal stress); 0–9 °C ≈ 32–48.2 °F (slight cold).
    fig, ax = plt.subplots(figsize=(12, 5))
    data = [vh[vh["station_id"]==s]["utci_f"].dropna().to_numpy() for s in ALL_ORDER]
    labels = [f"{s.split('_')[-1] if '_' in s else 'Roof'}\n{STATION_ROLE[s]}" for s in ALL_ORDER]
    bp = ax.violinplot(data, showmedians=True, showextrema=False)
    ax.axhspan(48.2, 78.8, color="#dff0d8", alpha=0.5, label="no thermal stress (48–79 °F)")
    ax.axhspan(32.0, 48.2, color="#fcf8e3", alpha=0.5, label="slight cold (32–48 °F)")
    ax.set_xticks(range(1, len(ALL_ORDER)+1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("UTCI [°F]")
    ax.set_title("Phase C: UTCI distribution during visitor hours (10–17 PT)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "C1_utci_distribution.png", dpi=150)
    plt.close(fig)

    # C2: monthly diurnal heatmap of mean UTCI, per site (6-panel grid).
    # Uses the same color scheme as the per-station Roof_diurnal_utci figures
    # for consistency: RdBu_r with TwoSlopeNorm centered at 68 °F.
    vmin, vmax, center = 45, 75, 68
    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, s in zip(axes.flat, ALL_ORDER):
        sub = both[both["station_id"] == s]
        pivot = (sub.groupby(["month", "hour"])["utci_f"]
                    .mean().unstack(level=-1))
        im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                       extent=[-0.5, 23.5, 0.5, 12.5],
                       norm=norm, cmap="RdBu_r")
        ax.set_xticks(range(0,24,3)); ax.set_yticks(range(1,13))
        ax.set_yticklabels(MONTH_NAMES, fontsize=7)
        ax.set_title(f"{s.split('_')[-1] if '_' in s else 'Roof'} — {STATION_ROLE[s]}",
                     fontsize=9)
        # mark visitor hours
        ax.axvline(VISITOR_HOURS[0], color="k", linestyle=":", linewidth=0.5)
        ax.axvline(VISITOR_HOURS[1], color="k", linestyle=":", linewidth=0.5)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, extend="both")
    cbar.set_label("Mean UTCI [°F]")
    cbar.ax.axhline(center, color="black", lw=0.8)
    fig.suptitle("Phase C: Mean UTCI by month × hour-of-day (vertical dotted = visitor hours)",
                 fontsize=12)
    fig.savefig(out / "C2_utci_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # C3: stress-category mix by month per site (visitor hours, focusing on KEY_SITES)
    cat_order = [
        "extreme cold stress","very strong cold stress","strong cold stress",
        "moderate cold stress","slight cold stress","no thermal stress",
        "moderate heat stress","strong heat stress","very strong heat stress",
        "extreme heat stress",
    ]
    palette = {
        "extreme cold stress":"#08306b","very strong cold stress":"#2171b5",
        "strong cold stress":"#6baed6","moderate cold stress":"#c6dbef",
        "slight cold stress":"#deebf7","no thermal stress":"#a1d99b",
        "moderate heat stress":"#fdd49e","strong heat stress":"#fc8d59",
        "very strong heat stress":"#d7301f","extreme heat stress":"#7f0000",
    }
    # 2×2 grid (instead of 1×4) so each panel stays readable when the report
    # scales the figure to page width.
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), sharey=True, sharex=True)
    last_ax = None
    for ax, s in zip(axes.flat, KEY_SITES):
        sub = vh[vh["station_id"] == s]
        mix = (sub.groupby(["month","utci_category"]).size()
                  .unstack(fill_value=0))
        mix = mix.reindex(columns=[c for c in cat_order if c in mix.columns])
        mix_pct = mix.div(mix.sum(axis=1), axis=0) * 100
        bottom = np.zeros(len(mix_pct))
        for c in mix_pct.columns:
            ax.bar(mix_pct.index, mix_pct[c], bottom=bottom,
                   color=palette[c], label=c, width=0.85, edgecolor="white",
                   linewidth=0.3)
            bottom += mix_pct[c].to_numpy()
        ax.set_xticks(range(1,13)); ax.set_xticklabels(MONTH_NAMES, fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title(f"{s.split('_')[-1] if '_' in s else 'Roof'} — {STATION_ROLE[s]}",
                     fontsize=11)
        last_ax = ax
    # Y label on the left column only (sharey hides the right-column tick labels)
    for ax in axes[:, 0]:
        ax.set_ylabel("% of visitor hours")
    # Shared legend below, two rows × five cols, reversed so order matches the
    # visual stack (hot at top of bar → first in legend).
    handles, labels = last_ax.get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="lower center", fontsize=9,
               ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Phase C: UTCI category mix during visitor hours — by month, key sites",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.09, 1, 0.96])
    fig.savefig(out / "C3_category_mix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # C4: comfort gap — % of visitor hours "no thermal stress," monthly, key sites
    fig, ax = plt.subplots(figsize=(10, 5))
    for s in KEY_SITES:
        sub = vh[vh["station_id"] == s]
        m = (sub.groupby("month")["utci_category"]
                .apply(lambda c: 100 * c.isin(COMFORT_CATS).mean()))
        ax.plot(m.index, m.values, marker="o",
                label=f"{s.split('_')[-1] if '_' in s else 'Roof'} · {STATION_ROLE[s]}")
    ax.set_xticks(range(1,13)); ax.set_xticklabels(MONTH_NAMES)
    ax.set_ylabel("% of visitor hours in 'no thermal stress'")
    ax.set_title("Phase C: Comfortable visitor hours by month — key sites")
    ax.legend(fontsize=9, loc="best"); ax.grid(alpha=0.3); ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out / "C4_comfort_gap.png", dpi=150)
    plt.close(fig)

    return site_summary


# ---------------------------------------------------------------------------
# Rooftop wind-mitigation scenarios
# ---------------------------------------------------------------------------

def rooftop_wind_scenarios(r: pd.DataFrame, out: Path) -> None:
    """Two-panel heatmap: rooftop visitor-hours mean UTCI, baseline vs the 50%
    wind-reduction scenario, on a shared color scale so the comfort gain reads
    visually. Reads the wind-scenario columns produced by enrich_rooftop.py;
    converts °C → °F inline so the figure matches the rest of the report.
    """
    out.mkdir(parents=True, exist_ok=True)

    # Pythermalcomfort gives us baseline °F and °C, but only °C for the
    # scenarios — convert on the way in so all downstream code is single-unit.
    def _c_to_f(c):
        return c * 9.0 / 5.0 + 32.0

    vh = r[(r["hour"] >= VISITOR_HOURS[0]) & (r["hour"] < VISITOR_HOURS[1])].copy()
    vh["utci_f_wind50pct"] = _c_to_f(vh["utci_c_wind50pct"])
    vh["utci_f_wind25pct"] = _c_to_f(vh["utci_c_wind25pct"])

    months = list(range(1, 13))
    hours  = list(range(VISITOR_HOURS[0], VISITOR_HOURS[1]))

    pivot_base = (vh.pivot_table(values="utci_f", index="hour",
                                  columns="month", aggfunc="mean")
                    .reindex(index=hours, columns=months))
    pivot_50   = (vh.pivot_table(values="utci_f_wind50pct", index="hour",
                                  columns="month", aggfunc="mean")
                    .reindex(index=hours, columns=months))

    # Match the Roof_diurnal_utci.png color scheme exactly so the per-station
    # diurnal heatmaps and this scenario figure read on the same scale:
    # RdBu_r + TwoSlopeNorm with vmin=45 °F, vcenter=68 °F (thermal neutrality),
    # vmax=75 °F, with both-end extend arrows for out-of-range cells.
    vmin, vmax, center = 45, 75, 68
    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

    # constrained_layout (not tight_layout) — colorbar-aware, reserves the
    # right-edge strip cleanly without overlapping the data grid.
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True, sharey=True,
                             constrained_layout=True)
    panels = [
        (axes[0], pivot_base, "Baseline — measured wind"),
        (axes[1], pivot_50,   "Hypothetical: 50% wind reduction (e.g., glass screens / baffles)"),
    ]
    last_im = None
    for ax, pivot, title in panels:
        im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                       norm=norm, cmap="RdBu_r")
        ax.set_xticks(range(len(months))); ax.set_xticklabels(MONTH_NAMES, fontsize=9)
        ax.set_yticks(range(len(hours)));  ax.set_yticklabels(hours, fontsize=9)
        ax.set_ylabel("Hour of day (local)")
        ax.set_title(title, fontsize=11)
        _label_cells(ax, pivot, "{:.0f}", center=center, font=8)
        last_im = im
    axes[1].set_xlabel("Month")

    cbar = fig.colorbar(last_im, ax=axes.tolist(), label="Mean UTCI [°F]",
                        shrink=0.85, extend="both")
    cbar.ax.axhline(center, color="black", lw=0.8)
    fig.suptitle("Rooftop visitor-hours mean UTCI — baseline vs 50% wind-reduction scenario\n"
                 "(visitor hours: 10:00–17:00 local; white ≈ 68 °F thermal neutrality)",
                 fontsize=12)
    fig.savefig(out / "rooftop_wind_scenarios.png", dpi=150)
    plt.close(fig)

    # ---------- Dose-response summary across baseline / 50% / 75% ----------
    # Work in °F throughout so the printed tables match the figure and the rest
    # of the report. Convert the two scenario columns on the fly.
    vh["utci_f_baseline"] = vh["utci_f"]
    scenarios = [
        ("baseline",             "utci_f_baseline"),
        ("wind50pct (v × 0.5)",  "utci_f_wind50pct"),
        ("wind25pct (v × 0.25)", "utci_f_wind25pct"),
    ]

    # UTCI band thresholds in °F (0/9/26 °C → 32/48.2/78.8 °F)
    F_COLD     = 32.0   # below this: moderate cold or worse
    F_COMFORT  = 48.2   # 9 °C — slight-cold / no-stress boundary
    F_HEAT     = 78.8   # 26 °C — no-stress / heat boundary

    def _comfort_stats(x):
        x = x.dropna()
        no_stress         = (x >= F_COMFORT) & (x < F_HEAT)
        slight_cold       = (x >= F_COLD) & (x < F_COMFORT)
        mod_cold_or_worse = x < F_COLD
        heat              = x >= F_HEAT
        return {
            "n":                     int(len(x)),
            "mean_utci_f":           x.mean(),
            "pct_no_stress":         100 * no_stress.mean(),
            "pct_slight_cold":       100 * slight_cold.mean(),
            "pct_heat_stress":       100 * heat.mean(),
            "pct_mod_cold_or_worse": 100 * mod_cold_or_worse.mean(),
            "utci_p05_f":            x.quantile(0.05),
            "utci_p50_f":            x.quantile(0.50),
            "utci_p95_f":            x.quantile(0.95),
        }

    rows = [{"scenario": name, **_comfort_stats(vh[col])} for name, col in scenarios]
    summary = pd.DataFrame(rows).set_index("scenario")
    summary["delta_mean_utci_f"] = summary["mean_utci_f"] - summary.loc["baseline", "mean_utci_f"]
    summary.to_csv(out / "rooftop_wind_scenarios_summary.csv")

    monthly = {}
    for name, col in scenarios:
        sub = vh[["month", col]].dropna()
        comfort = (sub[col] >= F_COMFORT) & (sub[col] < F_HEAT)
        monthly[name] = (comfort.groupby(sub["month"]).mean() * 100)
    monthly_df = pd.DataFrame(monthly).round(1)
    monthly_df.index = monthly_df.index.map(lambda m: MONTH_NAMES[m-1])
    monthly_df.index.name = "month"
    monthly_df.to_csv(out / "rooftop_wind_scenarios_monthly_no_stress.csv")

    print("\nAnnual visitor-hours summary (rooftop, 10:00–17:00 local):")
    print(summary.round(2).to_string())
    print("\nMonthly % of visitor hours in 'no thermal stress' (48–79 °F):")
    print(monthly_df.to_string())


# ---------------------------------------------------------------------------
# Rooftop turbulence deep-dive
# ---------------------------------------------------------------------------

def turbulence_analysis(g: pd.DataFrame, r: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Rooftop turbulence at full data fidelity (σθ + steadiness ratio from the
    Campbell logger), plus a cross-site comparison that uses gust-factor
    proxies on the ground stations (which lack circular-statistics
    instrumentation).

    Outputs:
      - rooftop_turbulence.png — 3 subplots, visitor hours only:
          (1) mean σθ by month × hour-of-day (0–45°, YlOrRd, high = chaotic)
          (2) mean steadiness ratio (resultant / scalar) by month × hour
              (0.7–1.0, YlOrRd_r, low = swirling)
          (3) joint (wind speed, σθ) hexbin — top-right = strong AND chaotic
      - turbulence_comparison.png — bar charts: % visitor hours in "buffeting"
        conditions (all sites, gust-factor proxy) and % visitor hours in
        "swirling" conditions (rooftop only, σθ-based; ground omitted by
        design — they don't have the instrumentation).
      - turbulence_summary.csv

    The σθ panels deliberately use mean σθ per (month, hour) rather than a
    threshold count — preserves the continuous-data fidelity of the Campbell
    logger rather than collapsing it to one bit per cell.
    """
    out.mkdir(parents=True, exist_ok=True)

    # --- Rooftop turbulence columns -----------------------------------------
    r = r.copy()
    r["sigma_theta"]  = r["campbell_wind_direction_std_dev"]
    r["steadiness"]   = r["resultant_wind_speed_ms"] / r["wind_speed_ms"].replace(0, np.nan)
    r["gust_factor"]  = r["gust_speed_ms"]           / r["wind_speed_ms"].replace(0, np.nan)

    # --- Ground turbulence proxies (no σθ; use gust factor) -----------------
    g2 = g.copy()
    g2["v_ms"]       = g2["wind_speed_mph"] * 0.44704
    g2["gust_ms"]    = g2["wind_gust_mph"]  * 0.44704
    g2["gust_factor"] = g2["gust_ms"] / g2["v_ms"].replace(0, np.nan)

    # Visitor-hours subsets
    vh_r = r[(r["hour"]  >= VISITOR_HOURS[0]) & (r["hour"]  < VISITOR_HOURS[1])]
    vh_g = g2[(g2["hour"] >= VISITOR_HOURS[0]) & (g2["hour"] < VISITOR_HOURS[1])]

    months = list(range(1, 13))
    hours  = list(range(VISITOR_HOURS[0], VISITOR_HOURS[1]))

    # =====================================================================
    # Figure 1: rooftop 3-subplot
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

    # --- Panel A: σθ heatmap (fixed scale 0-45°) ---
    sigma_pivot = (vh_r.pivot_table(values="sigma_theta", index="hour",
                                     columns="month", aggfunc="mean")
                        .reindex(index=hours, columns=months))
    ax = axes[0]
    im = ax.imshow(sigma_pivot.values, aspect="auto", origin="lower",
                   vmin=0, vmax=45, cmap="YlOrRd")
    ax.set_xticks(range(len(months))); ax.set_xticklabels(MONTH_NAMES, fontsize=8)
    ax.set_yticks(range(len(hours)));  ax.set_yticklabels(hours, fontsize=9)
    ax.set_xlabel("Month"); ax.set_ylabel("Hour of day (local)")
    ax.set_title("σθ — wind-direction variability\n(Campbell Yamartino; high = chaotic)",
                 fontsize=10)
    _label_cells(ax, sigma_pivot, "{:.0f}", white_above=28, font=7)
    fig.colorbar(im, ax=ax, label="σθ [°]", extend="max")

    # --- Panel B: steadiness heatmap (fixed scale 0.7-1.0) ---
    steady_pivot = (vh_r.pivot_table(values="steadiness", index="hour",
                                      columns="month", aggfunc="mean")
                         .reindex(index=hours, columns=months))
    ax = axes[1]
    im = ax.imshow(steady_pivot.values, aspect="auto", origin="lower",
                   vmin=0.7, vmax=1.0, cmap="YlOrRd_r")
    ax.set_xticks(range(len(months))); ax.set_xticklabels(MONTH_NAMES, fontsize=8)
    ax.set_yticks(range(len(hours)));  ax.set_yticklabels(hours, fontsize=9)
    ax.set_xlabel("Month")
    ax.set_title("Steadiness — resultant / scalar wind speed\n(1.0 = steady; <1 = direction varies within sample)",
                 fontsize=10)
    _label_cells(ax, steady_pivot, "{:.2f}", font=7)
    fig.colorbar(im, ax=ax, label="resultant / scalar", extend="min")

    # --- Panel C: joint (wind, σθ) hexbin ---
    ax = axes[2]
    mask = vh_r[["wind_speed_ms", "sigma_theta"]].notna().all(axis=1)
    wind  = vh_r.loc[mask, "wind_speed_ms"].to_numpy()
    sigma = vh_r.loc[mask, "sigma_theta"].to_numpy()
    hb = ax.hexbin(wind, sigma, gridsize=28, cmap="viridis", mincnt=1,
                   bins="log", extent=(0, 15, 0, 45))
    ax.set_xlim(0, 15); ax.set_ylim(0, 45)
    ax.set_xlabel("Wind speed [m/s]"); ax.set_ylabel("σθ [°]")
    ax.set_title("Joint: wind speed × σθ\n(top-right quadrant = strong AND chaotic)",
                 fontsize=10)
    # Mark the "uncomfortable turbulent" thresholds
    ax.axvline(3, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.axhline(25, color="black", linestyle=":", linewidth=0.8, alpha=0.6)
    fig.colorbar(hb, ax=ax, label="# 6-min samples (log)")

    fig.suptitle("Rooftop turbulence — visitor hours (10:00–17:00 PT)", fontsize=13)
    fig.savefig(out / "rooftop_turbulence.png", dpi=150)
    plt.close(fig)

    # =====================================================================
    # Cross-site summary table
    # =====================================================================
    WIND_THRESH  = 3.0    # m/s mean wind — "noticeable steady wind"
    GF_THRESH    = 2.0    # gust factor — "buffeting"
    SIGMA_THRESH = 25.0   # degrees — "swirling"

    rows = []
    for s in GROUND_ORDER:
        sub = vh_g[vh_g["station_id"] == s]
        buffeting = (sub["v_ms"] > WIND_THRESH) & (sub["gust_factor"] > GF_THRESH)
        rows.append({
            "station_id":         s,
            "role":               STATION_ROLE[s],
            "mean_gust_factor":   sub["gust_factor"].mean(),
            "p95_gust_factor":    sub["gust_factor"].quantile(0.95),
            "pct_buffeting":      100 * buffeting.mean(),
            "pct_swirling_sigma": np.nan,   # no σθ instrumentation
        })

    sub = vh_r
    buffeting = (sub["wind_speed_ms"] > WIND_THRESH) & (sub["gust_factor"] > GF_THRESH)
    swirling  = (sub["wind_speed_ms"] > WIND_THRESH) & (sub["sigma_theta"] > SIGMA_THRESH)
    rows.append({
        "station_id":         "EXPLORE3b",
        "role":               STATION_ROLE["EXPLORE3b"],
        "mean_gust_factor":   sub["gust_factor"].mean(),
        "p95_gust_factor":    sub["gust_factor"].quantile(0.95),
        "pct_buffeting":      100 * buffeting.mean(),
        "pct_swirling_sigma": 100 * swirling.mean(),
    })

    summary = pd.DataFrame(rows).set_index("station_id")
    summary.to_csv(out / "turbulence_summary.csv")

    # =====================================================================
    # Figure 2: cross-site bar chart
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    labels = [s.split("_")[-1] if "_" in s else "Roof" for s in ALL_ORDER]
    colors = ["#4c72b0"] * 5 + ["#dd8452"]

    # Left: buffeting (proxy, all 6 sites)
    ax = axes[0]
    pct_buf = [summary.loc[s, "pct_buffeting"] for s in ALL_ORDER]
    bars = ax.bar(labels, pct_buf, color=colors, edgecolor="black")
    ax.set_ylabel("% of visitor hours")
    ax.set_title(f"Buffeting — gust-factor proxy, all sites\n"
                 f"(wind > {WIND_THRESH:.0f} m/s AND gust factor > {GF_THRESH:.1f})",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, pct_buf):
        ax.text(b.get_x() + b.get_width()/2, v + max(pct_buf)*0.02, f"{v:.1f}%",
                ha="center", fontsize=8)

    # Right: swirling (σθ-based, rooftop only)
    ax = axes[1]
    rooftop_swirl = summary.loc["EXPLORE3b", "pct_swirling_sigma"]
    ax.bar(["Roof"], [rooftop_swirl], color="#dd8452", edgecolor="black", width=0.5)
    ax.set_ylabel("% of visitor hours")
    ax.set_title(f"Swirling — σθ-based, rooftop only\n"
                 f"(wind > {WIND_THRESH:.0f} m/s AND σθ > {SIGMA_THRESH:.0f}°)",
                 fontsize=10)
    ax.set_ylim(0, max(rooftop_swirl * 1.4, 5))
    ax.grid(axis="y", alpha=0.3)
    ax.text(0, rooftop_swirl + max(rooftop_swirl * 0.04, 0.1),
            f"{rooftop_swirl:.1f}%", ha="center", fontsize=10)
    ax.text(0.5, -0.16,
            "(Ground stations lack circular-statistics instrumentation — cannot compute σθ)",
            transform=ax.transAxes, fontsize=8, color="#666",
            ha="center", style="italic")

    fig.suptitle("Turbulence — visitor hours (10:00–17:00 PT)", fontsize=12)
    fig.savefig(out / "turbulence_comparison.png", dpi=150)
    plt.close(fig)

    # =====================================================================
    # Printed numerical summary
    # =====================================================================
    sigma_clean  = vh_r["sigma_theta"].dropna()
    steady_clean = vh_r["steadiness"].dropna()
    print("\nRooftop σθ percentiles (visitor hours):")
    print(f"  mean: {sigma_clean.mean():.1f}°"
          f"   p50: {sigma_clean.quantile(0.50):.1f}°"
          f"   p90: {sigma_clean.quantile(0.90):.1f}°"
          f"   p99: {sigma_clean.quantile(0.99):.1f}°")
    print("Rooftop steadiness percentiles (visitor hours):")
    print(f"  mean: {steady_clean.mean():.2f}"
          f"   p10: {steady_clean.quantile(0.10):.2f}"
          f"   p50: {steady_clean.quantile(0.50):.2f}")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--ground", default="groundStations.parquet")
    p.add_argument("--roof",   default="rooftopStation.parquet")
    p.add_argument("--out-dir", default="phase_analysis")
    p.add_argument("--dry-bulb", action="store_true",
                   help="Educational: replace UTCI with dry-bulb air temperature "
                        "so the analysis runs as if humidity, wind, and MRT contributed "
                        "nothing. Useful for seeing what UTCI's correction terms actually add.")
    args = p.parse_args(argv[1:])

    out = Path(args.out_dir); out.mkdir(exist_ok=True)
    print(f"Loading {args.ground} + {args.roof} …")
    g, r = load_data(Path(args.ground), Path(args.roof))
    print(f"  ground: {len(g):,} rows, {g['station_id'].nunique()} stations")
    print(f"  roof:   {len(r):,} rows")

    if args.dry_bulb:
        print("\n[--dry-bulb] Substituting air_temp_f for UTCI everywhere. "
              "Wind-reduction scenarios will collapse to baseline.")
        g = _replace_utci_with_dry_bulb(g)
        r = _replace_utci_with_dry_bulb(r)

    print("\n=== Phase A — Station fingerprinting ===")
    sig = phase_a(g, r, out / "phase_a")
    print(sig[["midday_solar_p50","midday_solar_p90","v_ms_mean","role"]].to_string())

    print("\n=== Phase B — Wind & turbulence ===")
    b = phase_b(g, r, out / "phase_b")
    print("\nExtremes table:")
    print(b["extremes"][["calm_pct","mean_wind_ms","p95_wind_ms",
                         "mean_gust_factor","gust>10_pct","gust>15_pct"]].round(2).to_string())

    print("\n=== Phase C — Comfort prediction ===")
    cs = phase_c(g, r, out / "phase_c")
    print(cs[["pct_no_stress","pct_heat_stress",
              "pct_moderate_cold_or_worse","utci_p50_f","utci_p95_f"]].round(1).to_string())

    print("\n=== Rooftop wind-mitigation scenarios ===")
    rooftop_wind_scenarios(r, out / "rooftop_scenarios")

    print("\n=== Rooftop turbulence deep-dive ===")
    ts = turbulence_analysis(g, r, out / "turbulence")
    print("\nTurbulence summary (visitor hours):")
    print(ts[["mean_gust_factor","p95_gust_factor",
              "pct_buffeting","pct_swirling_sigma"]].round(2).to_string())

    print(f"\nAll outputs under {out}/")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
