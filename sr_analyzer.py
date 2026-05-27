"""
ES Futures Support & Resistance Analyzer
Designed for daytrading on intraday timeframes.

Levels detected:
  - Prior Day High / Low / Close (PDH, PDL, PDC)
  - Opening Range High / Low (ORH, ORL) — first 30 min
  - VWAP + 1/2 standard deviation bands
  - Pivot-based S/R (swing highs/lows with configurable lookback)
  - Volume-weighted price clusters (high-volume nodes)

Scoring:
  - Confluence  : +2 per additional method that agrees within tolerance
  - Touches     : +1 per confirmed price touch/rejection
  - Recency     : touches in last 30 bars weighted 2x
  - HVN boost   : +2 if a high-volume node is within tolerance
"""

import argparse
from datetime import datetime, timedelta, time as dtime
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Level:
    price: float
    sources: list = field(default_factory=list)   # which methods detected it
    touches: int = 0
    score: float = 0.0
    kind: str = "neutral"   # "resistance" | "support" | "neutral"


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_data(symbol: str = "ES=F", days_back: int = 3, interval: str = "5m") -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=days_back)
    df = yf.download(symbol, start=start, end=end, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}.")
    df.index = pd.to_datetime(df.index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


# ---------------------------------------------------------------------------
# Prior Day levels
# ---------------------------------------------------------------------------

def prior_day_levels(df: pd.DataFrame) -> dict:
    daily = df.resample("D").agg({"Open": "first", "High": "max", "Low": "min",
                                   "Close": "last", "Volume": "sum"}).dropna()
    if len(daily) < 2:
        return {}
    prev = daily.iloc[-2]
    return {
        "PDH": float(prev["High"]),
        "PDL": float(prev["Low"]),
        "PDC": float(prev["Close"]),
    }


# ---------------------------------------------------------------------------
# Opening Range
# ---------------------------------------------------------------------------

def opening_range(df: pd.DataFrame, rth_open: dtime = dtime(9, 30), minutes: int = 30) -> dict:
    today = df.index[-1].date()
    session_start = pd.Timestamp(datetime.combine(today, rth_open), tz=df.index.tz)
    session_or_end = session_start + timedelta(minutes=minutes)
    or_bars = df[(df.index >= session_start) & (df.index < session_or_end)]
    if or_bars.empty:
        return {}
    return {
        "ORH":    float(or_bars["High"].max()),
        "ORL":    float(or_bars["Low"].min()),
        "OR_MID": float((or_bars["High"].max() + or_bars["Low"].min()) / 2),
    }


# ---------------------------------------------------------------------------
# VWAP + bands
# ---------------------------------------------------------------------------

def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df.index.date
    df["typical"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["tp_vol"] = df["typical"] * df["Volume"]
    df["cum_tp_vol"] = df.groupby("date")["tp_vol"].cumsum()
    df["cum_vol"]    = df.groupby("date")["Volume"].cumsum()
    df["VWAP"]       = df["cum_tp_vol"] / df["cum_vol"]
    df["sq_diff"]    = (df["typical"] - df["VWAP"]) ** 2
    df["cum_sq"]     = df.groupby("date")["sq_diff"].cumsum()
    df["vwap_std"]   = np.sqrt(df["cum_sq"] / df.groupby("date").cumcount().add(1))
    df["VWAP_U1"] = df["VWAP"] + df["vwap_std"]
    df["VWAP_L1"] = df["VWAP"] - df["vwap_std"]
    df["VWAP_U2"] = df["VWAP"] + 2 * df["vwap_std"]
    df["VWAP_L2"] = df["VWAP"] - 2 * df["vwap_std"]
    return df


# ---------------------------------------------------------------------------
# Swing pivots (pure numpy)
# ---------------------------------------------------------------------------

def swing_levels(df: pd.DataFrame, order: int = 5) -> dict:
    highs = df["High"].values
    lows  = df["Low"].values
    n = len(highs)
    sh_idx = [i for i in range(order, n - order)
               if highs[i] == max(highs[i - order: i + order + 1])]
    sl_idx = [i for i in range(order, n - order)
               if lows[i]  == min(lows[i  - order: i + order + 1])]
    return {
        "swing_highs": sorted(set(highs[sh_idx]), reverse=True)[:8],
        "swing_lows":  sorted(set(lows[sl_idx]))[:8],
    }


# ---------------------------------------------------------------------------
# Volume clusters / HVN
# ---------------------------------------------------------------------------

def volume_clusters(df: pd.DataFrame, bins: int = 60, top_n: int = 5) -> list:
    price_range  = np.linspace(df["Low"].min(), df["High"].max(), bins + 1)
    vol_profile  = np.zeros(bins)
    for _, row in df.iterrows():
        bl, bh, bv = row["Low"], row["High"], row["Volume"]
        for i in range(bins):
            ol = max(bl, price_range[i])
            oh = min(bh, price_range[i + 1])
            if oh > ol:
                bw = price_range[i + 1] - price_range[i]
                vol_profile[i] += bv * (oh - ol) / max(bw, 1e-9)
    centers = (price_range[:-1] + price_range[1:]) / 2
    top_idx = np.argsort(vol_profile)[-top_n:][::-1]
    return sorted([float(centers[i]) for i in top_idx])


# ---------------------------------------------------------------------------
# Touch counter
# ---------------------------------------------------------------------------

def count_touches(price: float, df: pd.DataFrame, tolerance: float = 2.0,
                  recent_bars: int = 30) -> tuple[int, int]:
    """
    Returns (total_touches, recent_touches).
    A touch = bar where High >= level - tol AND Low <= level + tol
    AND the bar closed on the opposite side (rejection confirmation).
    """
    total, recent = 0, 0
    n = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        near = row["Low"] <= price + tolerance and row["High"] >= price - tolerance
        if near:
            # Confirm rejection: close moved away from level
            rejected = abs(row["Close"] - price) > tolerance * 0.5
            if rejected:
                total += 1
                if i >= n - recent_bars:
                    recent += 1
    return total, recent


# ---------------------------------------------------------------------------
# Level scoring
# ---------------------------------------------------------------------------

def score_levels(raw_levels: dict, df: pd.DataFrame,
                 hvn_list: list, tolerance: float = 2.0) -> list[Level]:
    """
    Build Level objects, score by confluence + touches + recency + HVN proximity.
    """
    # Collect all candidate prices with their source tags
    candidates: dict[float, list[str]] = {}

    static_map = {
        "PDH": raw_levels.get("PDH"),
        "PDL": raw_levels.get("PDL"),
        "PDC": raw_levels.get("PDC"),
        "ORH": raw_levels.get("ORH"),
        "ORL": raw_levels.get("ORL"),
        "OR_MID": raw_levels.get("OR_MID"),
    }
    for tag, price in static_map.items():
        if price is not None:
            candidates[price] = candidates.get(price, []) + [tag]

    for p in raw_levels.get("swing_highs", []):
        candidates[p] = candidates.get(p, []) + ["SwingR"]
    for p in raw_levels.get("swing_lows", []):
        candidates[p] = candidates.get(p, []) + ["SwingS"]

    # Cluster nearby candidates
    sorted_prices = sorted(candidates.keys())
    clusters: list[tuple[float, list[str]]] = []
    for price in sorted_prices:
        if clusters and abs(price - clusters[-1][0]) <= tolerance:
            # merge into existing cluster
            avg = (clusters[-1][0] + price) / 2
            merged_sources = clusters[-1][1] + candidates[price]
            clusters[-1] = (avg, merged_sources)
        else:
            clusters.append((price, candidates[price]))

    levels: list[Level] = []
    current_price = float(df["Close"].iloc[-1])

    for price, sources in clusters:
        lvl = Level(price=price, sources=sources)

        # Confluence score: base 1 + 2 per extra source
        lvl.score += 1 + (len(set(sources)) - 1) * 2

        # Touch score
        total_touches, recent_touches = count_touches(price, df, tolerance)
        lvl.touches = total_touches
        lvl.score += total_touches + recent_touches  # recent counted twice

        # HVN proximity boost
        for hvn in hvn_list:
            if abs(hvn - price) <= tolerance * 1.5:
                lvl.score += 2
                if "HVN" not in lvl.sources:
                    lvl.sources.append("HVN")
                break

        # Kind
        lvl.kind = "resistance" if price > current_price else "support"

        levels.append(lvl)

    # Normalize scores to 0-10
    max_score = max((l.score for l in levels), default=1)
    for lvl in levels:
        lvl.score = round((lvl.score / max_score) * 10, 1)

    return sorted(levels, key=lambda l: l.score, reverse=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_chart(df: pd.DataFrame, scored: list[Level], title: str):
    today = df.index[-1].date()
    plot_df = df[df.index.date == today].copy()
    if plot_df.empty:
        plot_df = df.copy()

    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax_price, ax_vol = axes

    # Candlesticks
    for i, (_, row) in enumerate(plot_df.iterrows()):
        color = "#26a69a" if row["Close"] >= row["Open"] else "#ef5350"
        ax_price.plot([i, i], [row["Low"], row["High"]], color=color, linewidth=0.8)
        body_bot = min(row["Open"], row["Close"])
        body_top = max(row["Open"], row["Close"])
        ax_price.add_patch(mpatches.Rectangle(
            (i - 0.3, body_bot), 0.6, max(body_top - body_bot, 0.25),
            color=color, zorder=2))

    x_range = range(len(plot_df))

    # VWAP
    ax_price.plot(x_range, plot_df["VWAP"], color="#ff9800", linewidth=1.5,
                  label="VWAP", zorder=3)
    ax_price.fill_between(x_range, plot_df["VWAP_L1"], plot_df["VWAP_U1"],
                          alpha=0.08, color="#ff9800")
    ax_price.fill_between(x_range, plot_df["VWAP_L2"], plot_df["VWAP_U2"],
                          alpha=0.04, color="#ff9800")

    # All levels — dim
    for lvl in scored:
        color = "#ef5350" if lvl.kind == "resistance" else "#26a69a"
        ax_price.axhline(lvl.price, color=color, linestyle="--",
                         linewidth=0.6, alpha=0.35)

    # Top 5 levels — highlighted
    top5 = scored[:5]
    for lvl in top5:
        color = "#ef5350" if lvl.kind == "resistance" else "#26a69a"
        label = f"★ {lvl.price:.2f}  [{', '.join(set(lvl.sources))}]  score:{lvl.score}"
        ax_price.axhline(lvl.price, color=color, linestyle="-",
                         linewidth=2.0, alpha=0.9, label=label, zorder=4)
        ax_price.annotate(f"{lvl.price:.2f} ★",
                          xy=(len(plot_df) - 1, lvl.price),
                          fontsize=7, color=color,
                          va="bottom" if lvl.kind == "resistance" else "top")

    # Volume bars
    vol_colors = ["#26a69a" if plot_df["Close"].iloc[i] >= plot_df["Open"].iloc[i]
                  else "#ef5350" for i in range(len(plot_df))]
    ax_vol.bar(x_range, plot_df["Volume"], color=vol_colors, alpha=0.7)
    ax_vol.set_ylabel("Volume", fontsize=8)

    # X labels
    tick_step = max(1, len(plot_df) // 12)
    ticks = list(range(0, len(plot_df), tick_step))
    ax_price.set_xticks(ticks)
    ax_price.set_xticklabels([plot_df.index[i].strftime("%H:%M") for i in ticks], fontsize=7)

    ax_price.set_title(title, fontsize=13, pad=10)
    ax_price.set_ylabel("Price", fontsize=9)
    ax_price.legend(loc="upper left", fontsize=7, ncol=1, framealpha=0.7)
    ax_price.grid(True, alpha=0.15)
    ax_vol.grid(True, alpha=0.1)

    plt.tight_layout()
    out = "es_sr_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(scored: list[Level], current_price: float):
    print("\n" + "=" * 70)
    print(f"  ES Futures — Ranked S/R Levels   |  Current: {current_price:.2f}")
    print("=" * 70)
    print(f"  {'#':<3} {'Price':>8}  {'Kind':<11} {'Score':>6}  {'Touches':>7}  Sources")
    print("-" * 70)
    for i, lvl in enumerate(scored, 1):
        star = "★" if i <= 5 else " "
        sources = ", ".join(sorted(set(lvl.sources)))
        dist = lvl.price - current_price
        arrow = "▲" if dist > 0 else "▼"
        print(f"  {star}{i:<2} {lvl.price:>8.2f}  {lvl.kind:<11} {lvl.score:>6.1f}  "
              f"{lvl.touches:>7}  {sources}  {arrow}{abs(dist):.2f}")
    print("=" * 70)
    print("  ★ = top 5 most significant levels")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ES Futures S/R Analyzer")
    parser.add_argument("--symbol",       default="ES=F")
    parser.add_argument("--interval",     default="5m", choices=["1m", "2m", "5m", "15m"])
    parser.add_argument("--days",         type=int, default=3)
    parser.add_argument("--or-minutes",   type=int, default=30)
    parser.add_argument("--pivot-order",  type=int, default=5)
    parser.add_argument("--tolerance",    type=float, default=2.0,
                        help="Points tolerance for clustering/touch detection (default: 2.0)")
    parser.add_argument("--no-chart",     action="store_true")
    args = parser.parse_args()

    print(f"Fetching {args.symbol} [{args.interval}] — {args.days} days...")
    df = fetch_data(args.symbol, days_back=args.days, interval=args.interval)
    df = compute_vwap(df)

    raw = {}
    raw.update(prior_day_levels(df))
    raw.update(opening_range(df, minutes=args.or_minutes))
    raw.update(swing_levels(df, order=args.pivot_order))

    hvn = volume_clusters(df, top_n=5)

    scored = score_levels(raw, df, hvn, tolerance=args.tolerance)

    current_price = float(df["Close"].iloc[-1])
    print_summary(scored, current_price)

    if not args.no_chart:
        plot_chart(df, scored, title=f"{args.symbol} S/R Analysis [{args.interval}]")


if __name__ == "__main__":
    main()
