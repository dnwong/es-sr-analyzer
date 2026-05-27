"""
ES Futures Support & Resistance Analyzer
Outputs JSON results and a chart image for the Node.js web app.
"""

import argparse
import json
import os
import tempfile
from datetime import datetime, timedelta, time as dtime, timezone
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Docker
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


@dataclass
class Level:
    price: float
    sources: list = field(default_factory=list)
    touches: int = 0
    score: float = 0.0
    kind: str = "neutral"


def fetch_data(symbol, days_back, interval):
    end = datetime.now()
    start = end - timedelta(days=days_back)
    df = yf.download(symbol, start=start, end=end, interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}.")
    df.index = pd.to_datetime(df.index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def prior_day_levels(df):
    daily = df.resample("D").agg({"Open": "first", "High": "max",
                                   "Low": "min", "Close": "last",
                                   "Volume": "sum"}).dropna()
    if len(daily) < 2:
        return {}
    prev = daily.iloc[-2]
    return {"PDH": float(prev["High"]), "PDL": float(prev["Low"]),
            "PDC": float(prev["Close"])}


def opening_range(df, minutes=30):
    today = df.index[-1].date()
    rth_open = dtime(9, 30)
    session_start = pd.Timestamp(datetime.combine(today, rth_open), tz=df.index.tz)
    or_end = session_start + timedelta(minutes=minutes)
    bars = df[(df.index >= session_start) & (df.index < or_end)]
    if bars.empty:
        return {}
    return {
        "ORH":    float(bars["High"].max()),
        "ORL":    float(bars["Low"].min()),
        "OR_MID": float((bars["High"].max() + bars["Low"].min()) / 2),
    }


def compute_vwap(df):
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


def swing_levels(df, order=5):
    highs, lows = df["High"].values, df["Low"].values
    n = len(highs)
    sh = [i for i in range(order, n - order)
          if highs[i] == max(highs[i - order: i + order + 1])]
    sl = [i for i in range(order, n - order)
          if lows[i]  == min(lows[i  - order: i + order + 1])]
    return {
        "swing_highs": sorted(set(highs[sh]), reverse=True)[:8],
        "swing_lows":  sorted(set(lows[sl]))[:8],
    }


def volume_clusters(df, bins=60, top_n=5):
    price_range = np.linspace(df["Low"].min(), df["High"].max(), bins + 1)
    vol_profile = np.zeros(bins)
    for _, row in df.iterrows():
        for i in range(bins):
            ol = max(row["Low"],  price_range[i])
            oh = min(row["High"], price_range[i + 1])
            if oh > ol:
                bw = price_range[i + 1] - price_range[i]
                vol_profile[i] += row["Volume"] * (oh - ol) / max(bw, 1e-9)
    centers = (price_range[:-1] + price_range[1:]) / 2
    top_idx = np.argsort(vol_profile)[-top_n:][::-1]
    return sorted([float(centers[i]) for i in top_idx])


def count_touches(price, df, tolerance=2.0, recent_bars=30):
    total, recent = 0, 0
    n = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        near = row["Low"] <= price + tolerance and row["High"] >= price - tolerance
        if near and abs(row["Close"] - price) > tolerance * 0.5:
            total += 1
            if i >= n - recent_bars:
                recent += 1
    return total, recent


def score_levels(raw, df, hvn_list, tolerance=2.0):
    candidates = {}
    for tag in ["PDH", "PDL", "PDC", "ORH", "ORL", "OR_MID"]:
        if raw.get(tag) is not None:
            p = raw[tag]
            candidates[p] = candidates.get(p, []) + [tag]
    for p in raw.get("swing_highs", []):
        candidates[p] = candidates.get(p, []) + ["SwingR"]
    for p in raw.get("swing_lows", []):
        candidates[p] = candidates.get(p, []) + ["SwingS"]

    clusters = []
    for price in sorted(candidates):
        if clusters and abs(price - clusters[-1][0]) <= tolerance:
            avg = (clusters[-1][0] + price) / 2
            clusters[-1] = (avg, clusters[-1][1] + candidates[price])
        else:
            clusters.append((price, candidates[price]))

    current = float(df["Close"].iloc[-1])
    levels = []
    for price, sources in clusters:
        lvl = Level(price=price, sources=sources)
        lvl.score += 1 + (len(set(sources)) - 1) * 2
        total, recent = count_touches(price, df, tolerance)
        lvl.touches = total
        lvl.score += total + recent
        for hvn in hvn_list:
            if abs(hvn - price) <= tolerance * 1.5:
                lvl.score += 2
                if "HVN" not in lvl.sources:
                    lvl.sources.append("HVN")
                break
        lvl.kind = "resistance" if price > current else "support"
        levels.append(lvl)

    max_score = max((l.score for l in levels), default=1)
    for lvl in levels:
        lvl.score = round((lvl.score / max_score) * 10, 1)

    return sorted(levels, key=lambda l: l.score, reverse=True)


def render_chart(df, scored, chart_out):
    today = df.index[-1].date()
    plot_df = df[df.index.date == today].copy()
    if plot_df.empty:
        plot_df = df.copy()

    fig, axes = plt.subplots(2, 1, figsize=(16, 9),
                             gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax, ax_vol = axes
    fig.patch.set_facecolor("#131722")
    for a in axes:
        a.set_facecolor("#131722")
        a.tick_params(colors="#aaaaaa")
        for spine in a.spines.values():
            spine.set_edgecolor("#333333")

    for i, (_, row) in enumerate(plot_df.iterrows()):
        color = "#26a69a" if row["Close"] >= row["Open"] else "#ef5350"
        ax.plot([i, i], [row["Low"], row["High"]], color=color, linewidth=0.8)
        bot = min(row["Open"], row["Close"])
        top = max(row["Open"], row["Close"])
        ax.add_patch(mpatches.Rectangle((i - 0.3, bot), 0.6,
                     max(top - bot, 0.25), color=color, zorder=2))

    x = range(len(plot_df))
    ax.plot(x, plot_df["VWAP"], color="#ff9800", linewidth=1.5, label="VWAP", zorder=3)
    ax.fill_between(x, plot_df["VWAP_L1"], plot_df["VWAP_U1"], alpha=0.08, color="#ff9800")
    ax.fill_between(x, plot_df["VWAP_L2"], plot_df["VWAP_U2"], alpha=0.04, color="#ff9800")

    for lvl in scored:
        color = "#ef5350" if lvl.kind == "resistance" else "#26a69a"
        ax.axhline(lvl.price, color=color, linestyle="--", linewidth=0.5, alpha=0.3)

    for lvl in scored[:5]:
        color = "#ef5350" if lvl.kind == "resistance" else "#26a69a"
        label = f"★ {lvl.price:.2f}  score:{lvl.score}  [{', '.join(set(lvl.sources))}]"
        ax.axhline(lvl.price, color=color, linestyle="-", linewidth=2.0,
                   alpha=0.95, label=label, zorder=4)
        ax.annotate(f" {lvl.price:.2f} ★", xy=(len(plot_df) - 1, lvl.price),
                    fontsize=7, color=color,
                    va="bottom" if lvl.kind == "resistance" else "top")

    vol_colors = ["#26a69a" if plot_df["Close"].iloc[i] >= plot_df["Open"].iloc[i]
                  else "#ef5350" for i in range(len(plot_df))]
    ax_vol.bar(x, plot_df["Volume"], color=vol_colors, alpha=0.7)
    ax_vol.set_ylabel("Volume", fontsize=8, color="#aaaaaa")

    tick_step = max(1, len(plot_df) // 12)
    ticks = list(range(0, len(plot_df), tick_step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([plot_df.index[i].strftime("%H:%M") for i in ticks],
                       fontsize=7, color="#aaaaaa")
    ax.set_ylabel("Price", fontsize=9, color="#aaaaaa")
    ax.yaxis.label.set_color("#aaaaaa")
    ax.tick_params(axis="y", colors="#aaaaaa")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.4,
              facecolor="#1e222d", labelcolor="white")
    ax.grid(True, alpha=0.1, color="#333333")
    ax_vol.grid(True, alpha=0.08, color="#333333")

    plt.tight_layout()
    plt.savefig(chart_out, dpi=150, bbox_inches="tight", facecolor="#131722")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",      default="ES=F")
    parser.add_argument("--interval",    default="5m")
    parser.add_argument("--days",        type=int,   default=3)
    parser.add_argument("--or-minutes",  type=int,   default=30)
    parser.add_argument("--pivot-order", type=int,   default=5)
    parser.add_argument("--tolerance",   type=float, default=2.0)
    parser.add_argument("--json-out",    default=os.path.join(tempfile.gettempdir(), "sr_results.json"))
    parser.add_argument("--chart-out",   default=os.path.join(tempfile.gettempdir(), "es_sr_analysis.png"))
    args = parser.parse_args()

    df = fetch_data(args.symbol, args.days, args.interval)
    df = compute_vwap(df)

    raw = {}
    raw.update(prior_day_levels(df))
    raw.update(opening_range(df, minutes=args.or_minutes))
    raw.update(swing_levels(df, order=args.pivot_order))
    hvn = volume_clusters(df, top_n=5)

    scored = score_levels(raw, df, hvn, tolerance=args.tolerance)
    current = float(df["Close"].iloc[-1])

    results = {
        "symbol":        args.symbol,
        "interval":      args.interval,
        "current_price": current,
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "levels": [
            {
                "rank":    i + 1,
                "price":   lvl.price,
                "kind":    lvl.kind,
                "score":   lvl.score,
                "touches": lvl.touches,
                "sources": sorted(set(lvl.sources)),
                "distance": round(lvl.price - current, 2),
                "top5":    i < 5,
            }
            for i, lvl in enumerate(scored)
        ],
    }

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f)
        print(f"Results written to {args.json_out}")
    else:
        print(json.dumps(results, indent=2))

    render_chart(df, scored, args.chart_out)
    print(f"Chart written to {args.chart_out}")


if __name__ == "__main__":
    main()
