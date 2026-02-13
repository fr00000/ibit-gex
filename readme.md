# IBIT GEX Trading Dashboard

A web dashboard that pulls IBIT (iShares Bitcoin Trust ETF) options chain data from Yahoo Finance, calculates Gamma Exposure (GEX) using Black-Scholes, and displays actionable trading levels for **BTC perpetual futures trading with leverage**.

The core thesis: IBIT options flow creates dealer hedging dynamics that produce support/resistance levels in BTC. When the gamma regime is positive, BTC is range-bound and you can fade the extremes on perps. When negative, ranges break and you should trade momentum or sit out.

## Options Primer

If you're unfamiliar with options, here's what you need to know to use this dashboard.

**Options** are contracts that give the buyer the right to buy (call) or sell (put) an asset at a specific price (strike) by a specific date (expiration). When you buy an option, someone sells it to you — usually a market maker (dealer).

**Dealers don't want directional risk.** When a dealer sells you a call option, they're exposed if the price goes up. So they buy some of the underlying asset to offset that risk. This is called **delta hedging**. As price moves, dealers constantly adjust their hedges — buying and selling the underlying — which creates real supply and demand in the market.

**Open Interest (OI)** is the number of active option contracts at a given strike. High OI = lots of hedging activity at that price level. These strikes act like magnets or walls because of the hedging flows around them.

### The Greeks

The Black-Scholes model produces several sensitivity measures called "Greeks":

- **Delta** — How much the option price moves per $1 move in the underlying. Dealers hedge delta, so a strike with lots of OI and high delta means dealers are buying/selling a lot of stock there.
- **Gamma** — How fast delta changes as price moves. High gamma near a strike means dealers need to aggressively rebalance as price approaches it. This is why GEX levels act as support/resistance.
- **Vanna** — How delta changes when implied volatility (IV) moves. When IV drops (vol crush), dealers with positive vanna exposure must buy; when IV spikes, they must sell. This creates directional flows purely from volatility changes.
- **Charm** — How delta decays over time. Even if price and IV don't move, dealers must rebalance overnight because their delta exposure drifts as options get closer to expiration. This creates predictable overnight flows.

### Why This Matters for BTC

IBIT is a Bitcoin ETF with a large, liquid options market. When dealers hedge IBIT options, they're effectively creating supply/demand at specific BTC price levels. A strike with massive call open interest at $60 IBIT translates to a resistance zone in BTC at the equivalent price. The hedging flows are mechanical — dealers don't have a view, they just follow their risk models. This makes the levels predictable and tradeable.

## Quick Start

```bash
pip install -r requirements.txt
python3 app.py
```

Open http://localhost:5000 in your browser.

## Usage

```bash
# Default: 7 DTE, port 5000
python3 app.py

# 1 week of expirations only
python3 app.py --dte 7

# Full 45 day range
python3 app.py --dte 45

# Custom port and host
python3 app.py --port 8080 --host 0.0.0.0
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--dte`, `-d` | 7 | Max days to expiration for options chain |
| `--port`, `-p` | 5000 | Server port |
| `--host` | 127.0.0.1 | Server host (use 0.0.0.0 for WSL2) |

## Dashboard

### Header
BTC price, IBIT price, gamma regime badge, timestamp, DTE selector, candle timeframe selector (15m/1h/4h/1d), expiration date, refresh button.

### BTC Candlestick Chart
BTC candles (via Binance) with horizontal overlays at call wall, put wall, gamma flip, max pain, expected move bounds, and support/resistance levels. Timeframe is selectable from the header.

### GEX Profile
Bar chart of net gamma exposure at each strike. Green = positive gamma (dealers dampen moves), red = negative gamma (dealers amplify moves).

### Open Interest Profile
Call vs put open interest at each strike, showing where hedging activity is concentrated.

### Sidebar

- **Regime Banner** — Positive gamma (range-bound, fade extremes) or negative gamma (trending, don't fade)
- **Expected Move** — Implied straddle range for nearest expiry
- **Range Visual** — In positive gamma: tradeable range between put wall and call wall with spot indicator
- **Key Levels** — Call Wall, Put Wall, Gamma Flip, Max Pain with BTC prices and distance from spot
- **Significant Levels** — High-OI strikes with regime-adjusted behavior, OI changes (BUILDING/DECAYING), and vanna/charm notes
- **Breakout Assessment** — Upside/downside signal scoring with targets
- **Dealer Flows** — Overnight charm forecast, vanna vol scenarios, and combined overnight dealer rebalancing estimate
- **OI Changes** — Call/put open interest shifts vs prior snapshot
- **Dealer Position** — Net GEX, dealer delta, net vanna, net charm, put/call ratio
- **History** — Daily snapshots of regime and levels

## How It Works

### Data Pipeline
1. Fetches IBIT spot price and BTC-USD price from Yahoo Finance
2. Auto-calculates BTC/Share ratio (IBIT price / BTC price)
3. Pulls options chains across expirations within the DTE window
4. Calculates Black-Scholes gamma, delta, vanna, and charm using per-strike implied volatility
5. Aggregates to build a GEX profile, derive levels, and compute dealer flow forecasts

### GEX Calculation
```
GEX = Gamma x OI x 100 x Spot^2 x 0.01
```
- **Call GEX** = positive (dealers short calls -> buy dips / sell rips)
- **Put GEX** = negative (dealers short puts -> sell dips / buy rips)
- **Net GEX** at each strike = Call GEX + Put GEX

### Key Levels
- **Call Wall**: Strike with highest call GEX (resistance)
- **Put Wall**: Strike with most negative put GEX (support)
- **Gamma Flip**: Where net GEX crosses zero (regime change point)
- **Max Pain**: Strike minimizing total option payout — if the underlying settled here at expiration, option holders would lose the most. Acts as gravitational pull into expiry, strongest in positive gamma.
- **Expected Move**: ATM straddle price from nearest expiration. Represents the market's priced-in 1-standard-deviation range (~68% probability).
- **S1/S2, R1/R2**: High-OI support and resistance strikes beyond the walls. These are secondary levels where dealer hedging creates friction.

### Regime Detection
Local net GEX around spot (+/-2% of IBIT price):
- **Positive Gamma**: Dealers hedge against moves. Mean-reversion, range-bound. Fade extremes.
- **Negative Gamma**: Dealers hedge with moves. Momentum, trend extension. Don't fade.

### Regime-Adjusted Level Behavior
The same level behaves differently depending on regime:
- Positive gamma + put wall = "HARD FLOOR" (dealers buy the dip)
- Negative gamma + put wall = "REACTION" (dealers sell into the bounce, floor can break)
- Positive gamma + call wall = "HARD CEILING" (dealers sell into the rally)
- Negative gamma + call wall = "REACTION" (brief cap, watch for squeeze through)

### Dealer Flow Forecast (Vanna + Charm)

Beyond gamma, the dashboard forecasts dealer flows from two additional Greeks:

**Charm (overnight delta decay):**
Every night, dealers' delta exposure drifts as options decay. The dashboard calculates the net shares dealers must trade overnight to stay hedged. Positive charm = dealers buy tomorrow, negative = dealers sell.

**Vanna (vol-dependent flows):**
When IV changes, dealer delta shifts. The dashboard models two scenarios:
- **Vol crush** (-5pts IV): In calm/rallying markets, IV tends to fall. Dealers with positive vanna exposure must buy.
- **Vol spike** (+5pts IV): In selloffs, IV rises. Dealers with negative vanna exposure must sell, accelerating the move.

**Combined overnight:** The dashboard estimates net overnight flow by combining charm with expected vol decay (~0.5pt per calm session).

### Breakout Signals

| Signal | Bullish | Bearish |
|---|---|---|
| Wall asymmetry | Call wall GEX < 0.5x put wall | Put wall GEX < 0.5x call wall |
| Wall decay | Call wall OI declining >10% | Put wall OI declining >10% |
| Expected move vs range | Straddle width > wall range | Same |
| Negative gamma + proximity | Near call wall = squeeze | Near put wall = waterfall |
| OI beyond walls | >40% call OI above call wall | >40% put OI below put wall |
| P/C ratio extremes | P/C > 1.5 = squeeze fuel | P/C < 0.6 = call skew unwind |

### DTE Selection
Lower DTE (7) focuses on near-term expirations where gamma is strongest (gamma scales as 1/sqrt(T)). Higher DTE (14-45) includes longer-dated positioning which is structurally relevant but has less gamma impact. Default is 7 for short-term trading. The DTE can also be changed from the web UI dropdown.

## Daily OI Tracking

Snapshots are stored in SQLite (`~/.ibit_gex_history.db`). On each load the dashboard compares against the previous snapshot:
- Aggregate OI changes with positioning interpretation
- Per-level OI deltas (BUILDING / DECAYING)
- Historical regime and level trends

## API

### `GET /api/data`
Returns JSON with spot prices, levels, GEX/OI chart data, significant levels, breakout assessment, dealer flow forecast, OI changes, and history.

Accepts `?dte=N` query parameter (1-90) to override the DTE window.

## Config Constants

| Constant | Default | Purpose |
|---|---|---|
| `RISK_FREE_RATE` | 0.043 | Fed funds rate for Black-Scholes |
| `BTC_PER_SHARE` | 0.000568 | Fallback — auto-calculated from live prices |
| `STRIKE_RANGE_PCT` | 0.35 | Filter strikes to +/-35% of spot |

## Known Limitations
- Yahoo Finance OI updates once daily (after market close), not intraday
- Assumes dealers are net short all options (standard but not always true)
- BTC/Share ratio auto-calculated from spot; actual NAV drifts slightly due to fees
- Vanna/charm magnitudes are estimates — actual dealer positioning depends on their book, which isn't public
