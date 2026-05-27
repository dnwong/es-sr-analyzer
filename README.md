# ES Futures S/R Analyzer

A dockerized web application for analyzing Support & Resistance levels on ES futures, designed for daytrading.

## Features

- Prior Day High / Low / Close (PDH, PDL, PDC)
- Opening Range High / Low (configurable window)
- VWAP with ±1σ / ±2σ bands
- Swing pivot detection (pure numpy)
- Volume cluster / High-Volume Nodes (HVN)
- Confluence scoring — levels ranked by touches, recency, and method agreement

## Quick Start

```bash
docker compose up --build
```

Open [http://localhost:3000](http://localhost:3000)

## Portainer Deployment

1. In Portainer → **Stacks** → **Add stack**
2. Choose **Repository** and point to this repo, or paste the `docker-compose.yml` contents directly
3. Deploy the stack

## Parameters

| Parameter | Description | Default |
|---|---|---|
| Symbol | Futures symbol | ES=F |
| Interval | Bar size (1m/2m/5m/15m) | 5m |
| History | Days of data to fetch | 3 |
| Opening Range | First N minutes of RTH | 30 |
| Pivot Order | Bars each side for swing confirmation | 5 |
| Tolerance | Points window for clustering & touch detection | 2.0 |

## Stack

- Node.js 20 + Express (web server / API)
- Python 3.11 (analysis engine)
- yfinance, pandas, numpy, matplotlib
