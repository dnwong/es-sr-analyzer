"""
ES Futures Support & Resistance Analyzer
Two scoring modes:
  --mode standard   : confluence + touches + HVN (original)
  --mode floor      : floor-trader S/R approach (Shaoul-style)
Chart: Plotly JSON — interactive, drag-to-zoom, pan, hover tooltips
"""

import argparse, json, os, tempfile, socket
from datetime import datetime, timedelta, time as dtime, timezone
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests

_orig = socket.getaddrinfo
def _ipv4(host, port, family=0, *a, **kw):
    return _orig(host, port, socket.AF_INET, *a, **kw)
socket.getaddrinfo = _ipv4

SYMBOL_MAP = {
    "ES":"ES=F","MES":"MES=F","NQ":"NQ=F","MNQ":"MNQ=F","RTY":"RTY=F","YM":"YM=F",
}
INTERVAL_MAP = {"1m":"1m","2m":"2m","5m":"5m","15m":"15m","30m":"30m","60m":"60m"}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
}


@dataclass
class Level:
    price: float
    sources: list = field(default_factory=list)
    touches: int = 0
    score: float = 0.0
    kind: str = "neutral"
    zone_low: float = 0.0
    zone_high: float = 0.0
    credibility: str = ""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def fetch_data(symbol, days_back, interval):
    ticker = SYMBOL_MAP.get(symbol, symbol if symbol.endswith("=F") else symbol+"=F")
    yf_interval = INTERVAL_MAP.get(interval, "5m")
    end_ts   = int(datetime.now().timestamp())
    start_ts = int((datetime.now() - timedelta(days=days_back+3)).timestamp())
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"interval":yf_interval,"period1":start_ts,"period2":end_ts,"includePrePost":"false"}
    session = requests.Session()
    session.headers.update(HEADERS)
    try: session.get("https://finance.yahoo.com", timeout=10)
    except: pass
    resp = session.get(url, params=params, timeout=30)
    print(f"[DEBUG] status={resp.status_code} ticker={ticker}", flush=True)
    if resp.status_code != 200:
        raise ValueError(f"Yahoo returned {resp.status_code} for {ticker}.")
    try: data = resp.json()
    except: raise ValueError(f"Yahoo non-JSON: {resp.text[:200]}")
    result = data.get("chart",{}).get("result")
    error  = data.get("chart",{}).get("error")
    if error: raise ValueError(f"Yahoo error: {error.get('description',str(error))}")
    if not result or not result[0].get("timestamp"):
        raise ValueError(f"No data for {ticker}. Try during market hours Mon-Fri 9:30am-5pm ET.")
    r = result[0]
    ts = r["timestamp"]
    q  = r["indicators"]["quote"][0]
    df = pd.DataFrame({
        "Open":q.get("open",[None]*len(ts)),"High":q.get("high",[None]*len(ts)),
        "Low":q.get("low",[None]*len(ts)),"Close":q.get("close",[None]*len(ts)),
        "Volume":q.get("volume",[0]*len(ts)),
    }, index=pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York"))
    df.index.name = "Datetime"
    return df.dropna(subset=["Open","High","Low","Close"]).sort_index()


# ---------------------------------------------------------------------------
# Level detectors
# ---------------------------------------------------------------------------

def prior_day_levels(df):
    daily = df.resample("D").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    if len(daily) < 2: return {}
    p = daily.iloc[-2]
    return {"PDH":float(p["High"]),"PDL":float(p["Low"]),"PDC":float(p["Close"])}


def opening_range(df, minutes=30):
    today = df.index[-1].date()
    s = pd.Timestamp(datetime.combine(today, dtime(9,30)), tz=df.index.tz)
    bars = df[(df.index >= s) & (df.index < s+timedelta(minutes=minutes))]
    if bars.empty: return {}
    return {"ORH":float(bars["High"].max()),"ORL":float(bars["Low"].min()),
            "OR_MID":float((bars["High"].max()+bars["Low"].min())/2)}


def compute_vwap(df):
    df = df.copy()
    df["date"]    = df.index.date
    df["typical"] = (df["High"]+df["Low"]+df["Close"])/3
    df["tp_vol"]  = df["typical"]*df["Volume"]
    df["cum_tp"]  = df.groupby("date")["tp_vol"].cumsum()
    df["cum_vol"] = df.groupby("date")["Volume"].cumsum()
    df["VWAP"]    = df["cum_tp"]/df["cum_vol"]
    df["sq"]      = (df["typical"]-df["VWAP"])**2
    df["cum_sq"]  = df.groupby("date")["sq"].cumsum()
    df["std"]     = np.sqrt(df["cum_sq"]/df.groupby("date").cumcount().add(1))
    df["VWAP_U1"] = df["VWAP"]+df["std"];   df["VWAP_L1"] = df["VWAP"]-df["std"]
    df["VWAP_U2"] = df["VWAP"]+2*df["std"]; df["VWAP_L2"] = df["VWAP"]-2*df["std"]
    return df


def swing_levels(df, order=5):
    h,l,n = df["High"].values,df["Low"].values,len(df)
    sh = [i for i in range(order,n-order) if h[i]==max(h[i-order:i+order+1])]
    sl = [i for i in range(order,n-order) if l[i]==min(l[i-order:i+order+1])]
    return {"swing_highs":sorted(set(h[sh]),reverse=True)[:8],
            "swing_lows":sorted(set(l[sl]))[:8]}


def volume_profile(df, bins=60):
    pr = np.linspace(df["Low"].min(), df["High"].max(), bins+1)
    vp = np.zeros(bins)
    for _,row in df.iterrows():
        for i in range(bins):
            ol,oh = max(row["Low"],pr[i]),min(row["High"],pr[i+1])
            if oh>ol: vp[i] += row["Volume"]*(oh-ol)/max(pr[i+1]-pr[i],1e-9)
    centers = (pr[:-1]+pr[1:])/2
    poc = float(centers[np.argmax(vp)])
    threshold_hvn = np.percentile(vp[vp>0], 80)
    threshold_lvn = np.percentile(vp[vp>0], 20)
    hvn = sorted([float(centers[i]) for i in range(bins) if vp[i] >= threshold_hvn])
    lvn = sorted([float(centers[i]) for i in range(bins) if 0 < vp[i] <= threshold_lvn])
    return vp, centers, poc, hvn, lvn


def count_touches(price, df, tol=2.0, recent=30):
    tot,rec,n = 0,0,len(df)
    for i,(_,row) in enumerate(df.iterrows()):
        if row["Low"]<=price+tol and row["High"]>=price-tol:
            if abs(row["Close"]-price)>tol*0.5:
                tot+=1
                if i>=n-recent: rec+=1
    return tot,rec


def sharp_reversal_score(price, df, tol=2.0):
    score = 0
    for _,row in df.iterrows():
        if not (row["Low"]<=price+tol and row["High"]>=price-tol): continue
        bar_range = row["High"]-row["Low"]
        if bar_range < 0.25: continue
        upper_wick = row["High"]-max(row["Open"],row["Close"])
        lower_wick = min(row["Open"],row["Close"])-row["Low"]
        ratio = max(upper_wick,lower_wick)/bar_range
        if ratio > 0.6:   score += 1.5
        elif ratio > 0.4: score += 0.75
    return min(score, 3.0)


def intraday_rejection_zones(df, tol=2.0):
    zones = []
    vol_mean = df["Volume"].mean()
    for _,row in df.iterrows():
        bar_range = row["High"]-row["Low"]
        body = abs(row["Close"]-row["Open"])
        if bar_range < 0.25: continue
        if row["Volume"] > vol_mean*1.5 and body < bar_range*0.3:
            zones.append(float((row["High"]+row["Low"])/2))
    return zones


# ---------------------------------------------------------------------------
# Scoring modes
# ---------------------------------------------------------------------------

def score_standard(raw, df, hvn, tol=2.0):
    cands = {}
    for tag in ["PDH","PDL","PDC","ORH","ORL","OR_MID"]:
        if raw.get(tag): cands.setdefault(raw[tag],[]).append(tag)
    for p in raw.get("swing_highs",[]): cands.setdefault(p,[]).append("SwingR")
    for p in raw.get("swing_lows",[]): cands.setdefault(p,[]).append("SwingS")
    clusters=[]
    for price in sorted(cands):
        if clusters and abs(price-clusters[-1][0])<=tol:
            clusters[-1]=((clusters[-1][0]+price)/2, clusters[-1][1]+cands[price])
        else: clusters.append((price,cands[price]))
    cur = float(df["Close"].iloc[-1])
    levels=[]
    for price,sources in clusters:
        lvl = Level(price=price, sources=sources)
        lvl.score += 1+(len(set(sources))-1)*2
        tot,rec = count_touches(price,df,tol)
        lvl.touches=tot; lvl.score+=tot+rec
        for h in hvn:
            if abs(h-price)<=tol*1.5:
                lvl.score+=2
                if "HVN" not in lvl.sources: lvl.sources.append("HVN")
                break
        lvl.kind = "resistance" if price>cur else "support"
        lvl.zone_low  = round(price-tol, 2)
        lvl.zone_high = round(price+tol, 2)
        levels.append(lvl)
    mx = max((l.score for l in levels), default=1)
    for lvl in levels: lvl.score = round((lvl.score/mx)*10, 1)
    return sorted(levels, key=lambda l: l.score, reverse=True)


def score_floor(raw, df, poc, hvn, lvn, tol=2.0):
    cands = {}
    for tag in ["PDH","PDL","PDC","ORH","ORL","OR_MID"]:
        if raw.get(tag): cands.setdefault(raw[tag],[]).append(tag)
    for p in raw.get("swing_highs",[]): cands.setdefault(p,[]).append("SwingR")
    for p in raw.get("swing_lows",[]): cands.setdefault(p,[]).append("SwingS")
    cands.setdefault(poc,[]).append("POC")
    for h in hvn: cands.setdefault(h,[]).append("HVN")
    for l in lvn: cands.setdefault(l,[]).append("LVN")
    for z in intraday_rejection_zones(df, tol): cands.setdefault(z,[]).append("RejZone")
    clusters=[]
    for price in sorted(cands):
        if clusters and abs(price-clusters[-1][0])<=tol:
            clusters[-1]=((clusters[-1][0]+price)/2, clusters[-1][1]+cands[price])
        else: clusters.append((price,cands[price]))
    cur = float(df["Close"].iloc[-1])
    levels=[]
    for price,sources in clusters:
        src_set = set(sources)
        lvl = Level(price=price, sources=list(src_set))
        lvl.zone_low  = round(price-tol, 2)
        lvl.zone_high = round(price+tol, 2)
        if src_set & {"PDH","PDL","PDC"}: lvl.score += 4
        if "POC" in src_set:  lvl.score += 3
        if "HVN" in src_set:  lvl.score += 2
        if "LVN" in src_set:  lvl.score += 1
        if src_set & {"SwingR","SwingS"}: lvl.score += 2
        tot,rec = count_touches(price,df,tol)
        lvl.touches=tot; lvl.score+=tot*1.5+rec
        lvl.score += sharp_reversal_score(price,df,tol)
        if "RejZone" in src_set: lvl.score += 1.5
        n_m = len(src_set)
        if n_m >= 3: lvl.score += 3
        elif n_m == 2: lvl.score += 1.5
        lvl.kind = "resistance" if price>cur else "support"
        if tot >= 3:   lvl.credibility = "strong"
        elif tot == 2: lvl.credibility = "moderate"
        else:          lvl.credibility = "weak"
        levels.append(lvl)
    mx = max((l.score for l in levels), default=1)
    for lvl in levels: lvl.score = round((lvl.score/mx)*10, 1)
    return sorted(levels, key=lambda l: l.score, reverse=True)


# ---------------------------------------------------------------------------
# Chart — Plotly JSON (interactive, drag-to-zoom)
# ---------------------------------------------------------------------------

def build_chart_json(df, scored, mode, pp, symbol, interval):
    """Build Plotly figure as JSON dict — rendered interactively in browser."""
    today = df.index[-1].date()
    plot_df = df[df.index.date==today].copy()
    if plot_df.empty: plot_df = df.copy()

    times = plot_df.index.strftime("%H:%M").tolist()

    traces = []

    # Candlesticks
    traces.append({
        "type": "candlestick",
        "x": times,
        "open":  plot_df["Open"].tolist(),
        "high":  plot_df["High"].tolist(),
        "low":   plot_df["Low"].tolist(),
        "close": plot_df["Close"].tolist(),
        "name": symbol,
        "increasing": {"line": {"color": "#26a69a"}, "fillcolor": "#26a69a"},
        "decreasing": {"line": {"color": "#ef5350"}, "fillcolor": "#ef5350"},
        "xaxis": "x", "yaxis": "y",
    })

    # VWAP
    traces.append({
        "type": "scatter", "mode": "lines",
        "x": times, "y": plot_df["VWAP"].tolist(),
        "name": "VWAP", "line": {"color": "#ff9800", "width": 1.5},
        "xaxis": "x", "yaxis": "y",
    })

    # VWAP bands (±1σ as shaded area)
    traces.append({
        "type": "scatter", "mode": "lines",
        "x": times + times[::-1],
        "y": plot_df["VWAP_U1"].tolist() + plot_df["VWAP_L1"].tolist()[::-1],
        "fill": "toself", "fillcolor": "rgba(255,152,0,0.06)",
        "line": {"color": "transparent"}, "name": "VWAP ±1σ",
        "showlegend": True, "xaxis": "x", "yaxis": "y",
    })

    # S/R level lines
    for lvl in scored:
        is_top5 = scored.index(lvl) < 5
        color = "#ef5350" if lvl.kind == "resistance" else "#26a69a"
        dash  = "solid" if is_top5 else "dot"
        width = 1.5 if is_top5 else 0.7
        alpha = 0.9 if is_top5 else 0.35
        src_label = ",".join(sorted(set(lvl.sources)))
        name = f"{lvl.price:.2f} [{src_label}] score:{lvl.score}"
        traces.append({
            "type": "scatter", "mode": "lines",
            "x": [times[0], times[-1]],
            "y": [lvl.price, lvl.price],
            "name": name,
            "line": {"color": color, "width": width, "dash": dash},
            "opacity": alpha,
            "xaxis": "x", "yaxis": "y",
            "hovertemplate": f"<b>{lvl.price:.2f}</b><br>{src_label}<br>Score: {lvl.score}<br>Touches: {lvl.touches}<extra></extra>",
        })
        # Zone shading for floor mode
        if mode == "floor" and is_top5:
            traces.append({
                "type": "scatter", "mode": "lines",
                "x": [times[0], times[-1], times[-1], times[0]],
                "y": [lvl.zone_high, lvl.zone_high, lvl.zone_low, lvl.zone_low],
                "fill": "toself",
                "fillcolor": f"rgba({'239,83,80' if lvl.kind=='resistance' else '38,166,154'},0.08)",
                "line": {"color": "transparent"},
                "showlegend": False, "xaxis": "x", "yaxis": "y",
            })

    # Pivot Point line (blue, dashed, labeled)
    if pp is not None:
        traces.append({
            "type": "scatter", "mode": "lines+text",
            "x": [times[0], times[-1]],
            "y": [pp, pp],
            "name": f"PP {pp:.2f}",
            "line": {"color": "#58a6ff", "width": 2, "dash": "dashdot"},
            "text": ["", f"◆ PP {pp:.2f}"],
            "textposition": "top right",
            "textfont": {"color": "#58a6ff", "size": 11},
            "xaxis": "x", "yaxis": "y",
            "hovertemplate": f"<b>Pivot Point: {pp:.2f}</b><extra></extra>",
        })

    # Volume bars (subplot)
    vol_colors = ["#26a69a" if plot_df["Close"].iloc[i] >= plot_df["Open"].iloc[i]
                  else "#ef5350" for i in range(len(plot_df))]
    traces.append({
        "type": "bar",
        "x": times, "y": plot_df["Volume"].tolist(),
        "name": "Volume",
        "marker": {"color": vol_colors, "opacity": 0.7},
        "xaxis": "x", "yaxis": "y2",
    })

    layout = {
        "paper_bgcolor": "#131722",
        "plot_bgcolor":  "#131722",
        "font": {"color": "#aaaaaa", "size": 11},
        "xaxis": {
            "type": "category",
            "rangeslider": {"visible": False},
            "gridcolor": "#222",
            "tickfont": {"size": 9},
            "domain": [0, 1],
        },
        "yaxis": {
            "gridcolor": "#222",
            "side": "right",
            "domain": [0.25, 1],
        },
        "yaxis2": {
            "gridcolor": "#1a1a1a",
            "side": "right",
            "domain": [0, 0.22],
        },
        "legend": {
            "bgcolor": "rgba(30,34,45,0.8)",
            "bordercolor": "#333",
            "borderwidth": 1,
            "font": {"size": 9},
            "x": 0, "y": 1,
        },
        "margin": {"l": 10, "r": 60, "t": 30, "b": 30},
        "dragmode": "zoom",
        "title": {
            "text": f"{symbol} [{interval}] — {'Floor-Trader' if mode=='floor' else 'Standard'} S/R",
            "font": {"size": 12, "color": "#8b949e"},
            "x": 0.01,
        },
        "hovermode": "x unified",
    }

    return {"data": traces, "layout": layout}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",      default="ES")
    p.add_argument("--interval",    default="5m")
    p.add_argument("--days",        type=int,   default=3)
    p.add_argument("--or-minutes",  type=int,   default=30)
    p.add_argument("--pivot-order", type=int,   default=5)
    p.add_argument("--tolerance",   type=float, default=2.0)
    p.add_argument("--mode",        default="standard", choices=["standard","floor"])
    p.add_argument("--api-key",     default="")
    p.add_argument("--json-out",    default=os.path.join(tempfile.gettempdir(),"sr_results.json"))
    p.add_argument("--chart-out",   default=os.path.join(tempfile.gettempdir(),"sr_chart.json"))
    args = p.parse_args()

    df = fetch_data(args.symbol, args.days, args.interval)
    df = compute_vwap(df)

    raw = {}
    raw.update(prior_day_levels(df))
    raw.update(opening_range(df, minutes=args.or_minutes))
    raw.update(swing_levels(df, order=args.pivot_order))

    # Daily pivot point
    pp = None
    if raw.get("PDH") and raw.get("PDL") and raw.get("PDC"):
        pp = round((raw["PDH"] + raw["PDL"] + raw["PDC"]) / 3.0, 2)
        raw["PP"] = pp

    _,_,poc,hvn,lvn = volume_profile(df)

    if args.mode == "floor":
        scored = score_floor(raw, df, poc, hvn, lvn, tol=args.tolerance)
    else:
        scored = score_standard(raw, df, hvn, tol=args.tolerance)

    cur = float(df["Close"].iloc[-1])

    results = {
        "symbol":        args.symbol,
        "interval":      args.interval,
        "mode":          args.mode,
        "current_price": cur,
        "pivot_point":   pp,
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "levels": [
            {
                "rank":        i+1,
                "price":       lvl.price,
                "zone_low":    lvl.zone_low,
                "zone_high":   lvl.zone_high,
                "kind":        lvl.kind,
                "score":       lvl.score,
                "touches":     lvl.touches,
                "credibility": lvl.credibility,
                "sources":     sorted(set(lvl.sources)),
                "distance":    round(lvl.price-cur, 2),
                "top5":        i < 5,
                "is_pp":       bool(pp is not None and abs(lvl.price-pp) <= args.tolerance),
            }
            for i,lvl in enumerate(scored)
        ],
    }

    with open(args.json_out, "w") as f:
        json.dump(results, f, default=lambda x: bool(x) if hasattr(x, 'item') else str(x))

    # Build interactive Plotly chart JSON
    chart = build_chart_json(df, scored, args.mode, pp, args.symbol, args.interval)
    with open(args.chart_out, "w") as f:
        json.dump(chart, f, default=lambda x: float(x) if hasattr(x, 'item') else str(x))

    print(f"Done. Results: {args.json_out}  Chart: {args.chart_out}")


if __name__ == "__main__":
    main()
