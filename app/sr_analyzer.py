"""
ES Futures Support & Resistance Analyzer
Data source: Polygon.io (free API key at https://polygon.io)
Futures symbols: ES, MES, NQ, MNQ, RTY, YM
"""

import argparse, json, os, tempfile, socket
from datetime import datetime, timedelta, time as dtime, timezone
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_orig = socket.getaddrinfo
def _ipv4(host, port, family=0, *a, **kw):
    return _orig(host, port, socket.AF_INET, *a, **kw)
socket.getaddrinfo = _ipv4

POLY_BASE = "https://api.polygon.io"

# Polygon futures ticker format: /ES (slash prefix)
SYMBOL_MAP = {
    "ES=F": "/ES", "MES=F": "/MES", "ES": "/ES", "MES": "/MES",
    "NQ=F": "/NQ", "MNQ=F": "/MNQ", "NQ": "/NQ",  "MNQ": "/MNQ",
    "RTY=F": "/RTY", "RTY": "/RTY",
    "YM=F": "/YM",  "YM": "/YM",
}

MULTIPLIER_MAP = {"1m":"1","2m":"2","5m":"5","15m":"15","30m":"30","60m":"60"}
TIMESPAN_MAP   = {"1m":"minute","2m":"minute","5m":"minute","15m":"minute","30m":"minute","60m":"minute"}


@dataclass
class Level:
    price: float
    sources: list = field(default_factory=list)
    touches: int = 0
    score: float = 0.0
    kind: str = "neutral"


def fetch_data(symbol: str, days_back: int, interval: str, api_key: str) -> pd.DataFrame:
    ticker     = SYMBOL_MAP.get(symbol, f"/{symbol}")
    multiplier = MULTIPLIER_MAP.get(interval, "5")
    timespan   = TIMESPAN_MAP.get(interval, "minute")

    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days_back + 2)
    from_str = start_dt.strftime("%Y-%m-%d")
    to_str   = end_dt.strftime("%Y-%m-%d")

    url = f"{POLY_BASE}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_str}/{to_str}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}

    resp = requests.get(url, params=params, timeout=30)
    data = resp.json()

    if resp.status_code == 403:
        raise ValueError("Polygon API key invalid or unauthorized.")
    if resp.status_code != 200:
        raise ValueError(f"Polygon error {resp.status_code}: {data.get('error', data.get('message','unknown'))}")
    if data.get("status") == "ERROR":
        raise ValueError(f"Polygon: {data.get('error', data.get('message','unknown'))}")
    if not data.get("results"):
        raise ValueError(f"No data returned for {ticker}. Check symbol and ensure market was open in the last {days_back} days.")

    rows = [
        {
            "Datetime": pd.to_datetime(r["t"], unit="ms", utc=True).tz_convert("America/New_York"),
            "Open":   r["o"], "High":  r["h"],
            "Low":    r["l"], "Close": r["c"],
            "Volume": r.get("v", 0),
        }
        for r in data["results"]
    ]
    df = pd.DataFrame(rows).set_index("Datetime").sort_index()
    cutoff = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=days_back)).tz_convert("America/New_York")
    df = df[df.index >= cutoff]
    if df.empty:
        raise ValueError(f"No data in last {days_back} days for {ticker}.")
    return df


def prior_day_levels(df):
    daily = df.resample("D").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    if len(daily) < 2: return {}
    p = daily.iloc[-2]
    return {"PDH":float(p["High"]),"PDL":float(p["Low"]),"PDC":float(p["Close"])}


def opening_range(df, minutes=30):
    today = df.index[-1].date()
    s = pd.Timestamp(datetime.combine(today, dtime(9,30)), tz=df.index.tz)
    bars = df[(df.index >= s) & (df.index < s + timedelta(minutes=minutes))]
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


def volume_clusters(df, bins=60, top_n=5):
    pr = np.linspace(df["Low"].min(),df["High"].max(),bins+1)
    vp = np.zeros(bins)
    for _,row in df.iterrows():
        for i in range(bins):
            ol,oh = max(row["Low"],pr[i]),min(row["High"],pr[i+1])
            if oh>ol: vp[i] += row["Volume"]*(oh-ol)/max(pr[i+1]-pr[i],1e-9)
    c = (pr[:-1]+pr[1:])/2
    return sorted([float(c[i]) for i in np.argsort(vp)[-top_n:][::-1]])


def count_touches(price, df, tol=2.0, recent=30):
    tot,rec,n = 0,0,len(df)
    for i,(_,row) in enumerate(df.iterrows()):
        if row["Low"]<=price+tol and row["High"]>=price-tol:
            if abs(row["Close"]-price)>tol*0.5:
                tot+=1
                if i>=n-recent: rec+=1
    return tot,rec


def score_levels(raw, df, hvn, tol=2.0):
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
        lvl=Level(price=price,sources=sources)
        lvl.score += 1+(len(set(sources))-1)*2
        tot,rec = count_touches(price,df,tol)
        lvl.touches=tot; lvl.score+=tot+rec
        for h in hvn:
            if abs(h-price)<=tol*1.5:
                lvl.score+=2
                if "HVN" not in lvl.sources: lvl.sources.append("HVN")
                break
        lvl.kind="resistance" if price>cur else "support"
        levels.append(lvl)
    mx=max((l.score for l in levels),default=1)
    for lvl in levels: lvl.score=round((lvl.score/mx)*10,1)
    return sorted(levels,key=lambda l:l.score,reverse=True)


def render_chart(df, scored, chart_out):
    today=df.index[-1].date()
    plot_df=df[df.index.date==today].copy()
    if plot_df.empty: plot_df=df.copy()
    fig,axes=plt.subplots(2,1,figsize=(16,9),gridspec_kw={"height_ratios":[3,1]},sharex=True)
    ax,av=axes
    fig.patch.set_facecolor("#131722")
    for a in axes:
        a.set_facecolor("#131722"); a.tick_params(colors="#aaaaaa")
        for sp in a.spines.values(): sp.set_edgecolor("#333333")
    for i,(_,row) in enumerate(plot_df.iterrows()):
        c="#26a69a" if row["Close"]>=row["Open"] else "#ef5350"
        ax.plot([i,i],[row["Low"],row["High"]],color=c,linewidth=0.8)
        b,t=min(row["Open"],row["Close"]),max(row["Open"],row["Close"])
        ax.add_patch(mpatches.Rectangle((i-0.3,b),0.6,max(t-b,0.25),color=c,zorder=2))
    x=range(len(plot_df))
    ax.plot(x,plot_df["VWAP"],color="#ff9800",linewidth=1.5,label="VWAP",zorder=3)
    ax.fill_between(x,plot_df["VWAP_L1"],plot_df["VWAP_U1"],alpha=0.08,color="#ff9800")
    ax.fill_between(x,plot_df["VWAP_L2"],plot_df["VWAP_U2"],alpha=0.04,color="#ff9800")
    for lvl in scored:
        c="#ef5350" if lvl.kind=="resistance" else "#26a69a"
        ax.axhline(lvl.price,color=c,linestyle="--",linewidth=0.5,alpha=0.3)
    for lvl in scored[:5]:
        c="#ef5350" if lvl.kind=="resistance" else "#26a69a"
        ax.axhline(lvl.price,color=c,linestyle="-",linewidth=2.0,alpha=0.95,
                   label=f"* {lvl.price:.2f} score:{lvl.score} [{','.join(set(lvl.sources))}]",zorder=4)
        ax.annotate(f" {lvl.price:.2f}*",xy=(len(plot_df)-1,lvl.price),fontsize=7,color=c,
                    va="bottom" if lvl.kind=="resistance" else "top")
    vc=["#26a69a" if plot_df["Close"].iloc[i]>=plot_df["Open"].iloc[i] else "#ef5350" for i in range(len(plot_df))]
    av.bar(x,plot_df["Volume"],color=vc,alpha=0.7)
    av.set_ylabel("Volume",fontsize=8,color="#aaaaaa")
    ts=max(1,len(plot_df)//12); tks=list(range(0,len(plot_df),ts))
    ax.set_xticks(tks)
    ax.set_xticklabels([plot_df.index[i].strftime("%H:%M") for i in tks],fontsize=7,color="#aaaaaa")
    ax.set_ylabel("Price",fontsize=9,color="#aaaaaa"); ax.tick_params(axis="y",colors="#aaaaaa")
    ax.legend(loc="upper left",fontsize=7,framealpha=0.4,facecolor="#1e222d",labelcolor="white")
    ax.grid(True,alpha=0.1,color="#333333"); av.grid(True,alpha=0.08,color="#333333")
    plt.tight_layout()
    plt.savefig(chart_out,dpi=150,bbox_inches="tight",facecolor="#131722")
    plt.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--symbol",      default="ES")
    p.add_argument("--interval",    default="5m")
    p.add_argument("--days",        type=int,   default=3)
    p.add_argument("--or-minutes",  type=int,   default=30)
    p.add_argument("--pivot-order", type=int,   default=5)
    p.add_argument("--tolerance",   type=float, default=2.0)
    p.add_argument("--api-key",     default=os.environ.get("POLYGON_API_KEY",""))
    p.add_argument("--json-out",    default=os.path.join(tempfile.gettempdir(),"sr_results.json"))
    p.add_argument("--chart-out",   default=os.path.join(tempfile.gettempdir(),"es_sr_analysis.png"))
    args=p.parse_args()

    if not args.api_key:
        with open(args.json_out,"w") as f:
            json.dump({"error":"No API key. Set POLYGON_API_KEY env var."},f)
        return

    df=fetch_data(args.symbol,args.days,args.interval,args.api_key)
    df=compute_vwap(df)
    raw={}
    raw.update(prior_day_levels(df))
    raw.update(opening_range(df,minutes=args.or_minutes))
    raw.update(swing_levels(df,order=args.pivot_order))
    hvn=volume_clusters(df,top_n=5)
    scored=score_levels(raw,df,hvn,tol=args.tolerance)
    cur=float(df["Close"].iloc[-1])
    results={"symbol":args.symbol,"interval":args.interval,"current_price":cur,
             "generated_at":datetime.now(timezone.utc).isoformat(),
             "levels":[{"rank":i+1,"price":lvl.price,"kind":lvl.kind,"score":lvl.score,
                        "touches":lvl.touches,"sources":sorted(set(lvl.sources)),
                        "distance":round(lvl.price-cur,2),"top5":i<5}
                       for i,lvl in enumerate(scored)]}
    with open(args.json_out,"w") as f: json.dump(results,f)
    render_chart(df,scored,args.chart_out)

if __name__=="__main__":
    main()
