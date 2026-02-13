# IBIT GEX Trading Dashboard

A web dashboard that pulls crypto ETF (IBIT/ETHA) options chain data from Yahoo Finance, calculates Gamma Exposure (GEX) using Black-Scholes, and displays actionable trading levels for **BTC/ETH perpetual futures trading with leverage**.

The core thesis: ETF options flow creates dealer hedging dynamics that produce support/resistance levels in the underlying crypto asset. When the gamma regime is positive, price is range-bound and you can fade the extremes on perps. When negative, ranges break and you should trade momentum or sit out.

Supports **IBIT** (Bitcoin ETF) and **ETHA** (Ethereum ETF) with ticker switching in the header.

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

IBIT and ETHA are crypto ETFs with liquid options markets. When dealers hedge these options, they're effectively creating supply/demand at specific BTC/ETH price levels. A strike with massive call open interest at $60 IBIT translates to a resistance zone in BTC at the equivalent price. The hedging flows are mechanical — dealers don't have a view, they just follow their risk models. This makes the levels predictable and tradeable.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env  # add your Anthropic API key
python3 app.py
```

Open http://localhost:5000 in your browser.

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | For AI analysis | Claude API key for automated trading analysis |

Create a `.env` file in the project root (loaded automatically via python-dotenv).

## Usage

```bash
# Default: port 5000
python3 app.py

# Custom port and host
python3 app.py --port 8080 --host 0.0.0.0
```

DTE (days to expiration) is selectable from the web UI dropdown (3d, 7d, 14d, 30d, 45d).

### Flags

| Flag | Default | Description |
|---|---|---|
| `--port`, `-p` | 5000 | Server port |
| `--host` | 127.0.0.1 | Server host (use 0.0.0.0 for WSL2) |

## Dashboard

### Header
Ticker tabs (IBIT/ETHA), crypto price, ETF price, gamma regime badge, positioning confidence indicator, ETF flow badge (daily inflow/outflow with streak), timestamp, DTE selector, candle timeframe selector (15m/1h/4h/1d), expiration date, refresh button.

### Candlestick Chart
Crypto candles (BTC/ETH via Binance, persisted in SQLite with 90-day backfill) with horizontal overlays at call wall, put wall, gamma flip, max pain, expected move bounds, and support/resistance levels. Timeframe is selectable from the header. Real-time updates via Binance WebSocket.

### GEX Profile
Stacked bar chart showing Active GEX (new OI since yesterday, solid) and Stale GEX (existing positions, faded) at each strike. Green = positive gamma, red = negative gamma. Active GEX highlights where fresh dealer exposure is concentrated vs legacy "zombie gamma."

### Open Interest Profile
Call vs put open interest at each strike, showing where hedging activity is concentrated.

### ETF Flows
Bar chart of daily ETF fund flows over the last 30 days. Green bars = inflow days (creation/accumulation), red bars = outflow days (redemption/distribution). Backfilled from Yahoo Finance historical data on first run, then updated daily from shares outstanding changes.

### Sidebar

- **AI Analysis** — Claude-powered trading analysis across all DTE timeframes (3d/7d/14d/30d/45d + combined). Auto-runs daily when fresh data arrives. Includes day-over-day level changes and prior analysis context for thesis continuity.
- **Regime Banner** — Positive gamma (range-bound, fade extremes) or negative gamma (trending, don't fade)
- **Expected Move** — Implied straddle range for nearest expiry
- **Range Visual** — In positive gamma: tradeable range between put wall and call wall with spot indicator
- **Key Levels** — Call Wall, Put Wall, Gamma Flip, Max Pain with BTC prices and distance from spot
- **Significant Levels** — High-OI strikes with regime-adjusted behavior, OI changes (BUILDING/DECAYING), and vanna/charm notes
- **Breakout Assessment** — Upside/downside signal scoring with targets, includes ETF flow streak signals
- **Dealer Flows** — Overnight charm forecast, vanna vol scenarios, and combined overnight dealer rebalancing estimate
- **OI Changes** — Call/put open interest shifts vs prior snapshot
- **ETF Fund Flows** — Daily flow amount, direction, strength, and 5-day momentum with streak dots
- **Dealer Position** — Net GEX, Active GEX, dealer delta, net vanna, net charm, put/call ratio
- **History** — Daily snapshots of regime and levels

## How It Works

### Data Pipeline
1. Fetches ETF spot price and reference crypto price (BTC-USD or ETH-USD) from Yahoo Finance
2. Auto-calculates crypto/share ratio (ETF price / crypto price)
3. Pulls options chains across expirations within the DTE window
4. Calculates Black-Scholes gamma, delta, vanna, and charm using per-strike implied volatility
5. Aggregates to build a GEX profile, derive levels, and compute dealer flow forecasts
6. Fetches ETF fund flows from shares outstanding changes (daily creation/redemption activity)

### Data Caching
Options OI updates once per day (after market close). The app caches the full computed result in SQLite per DTE. On subsequent loads it compares OI to detect when Yahoo has fresh data — if unchanged, the cache is served instantly. Yahoo is re-checked at most every 30 minutes until new data is confirmed.

A background thread pre-fetches all 5 DTE timeframes on startup and keeps the cache warm. Once all timeframes are fresh for the day, it auto-runs AI analysis (if `ANTHROPIC_API_KEY` is set).

### GEX Calculation
```
GEX = Gamma x OI x 100 x Spot^2 x 0.01
```
- **Call GEX** = positive (dealers short calls -> buy dips / sell rips)
- **Put GEX** = negative (dealers short puts -> sell dips / buy rips)
- **Net GEX** at each strike = Call GEX + Put GEX

### Active GEX
Active GEX weights net GEX by the fraction of OI that is new since yesterday (delta_OI / total_OI). This surfaces where fresh dealer exposure is concentrated vs stale positions from days ago. Active walls show the strikes with the most new positioning.

### Positioning Confidence
A 0-100% score indicating how much to trust the standard GEX sign convention (dealers long calls, short puts). Penalized by:
- Low P/C ratio (heavy speculative call buying)
- High OTM call concentration
- Elevated call volume/OI turnover
- Concentrated single-strike bets
- Sustained ETF outflow streaks

When below ~60%, call walls may act as squeeze triggers rather than resistance.

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
| ETF flow streak | 3+ day inflow streak | 3+ day outflow streak |
| ETF flow momentum | 5d avg inflow > $100M | 5d avg outflow > $100M |

### DTE Selection
Lower DTE (7) focuses on near-term expirations where gamma is strongest (gamma scales as 1/sqrt(T)). Higher DTE (14-45) includes longer-dated positioning which is structurally relevant but has less gamma impact. Default is 7 for short-term trading. DTE is selectable from the web UI dropdown.

## Daily OI Tracking

Snapshots are stored in SQLite (`~/.ibit_gex_history.db`). The dashboard compares against the previous day's snapshot:
- Aggregate OI changes with positioning interpretation
- Per-level OI deltas (BUILDING / DECAYING)
- Historical regime and level trends

The database also stores full data caches (per DTE) and AI analysis results, both keyed by date.

## API

### `GET /api/data`
Returns JSON with spot prices, levels, GEX/OI chart data, significant levels, breakout assessment, dealer flow forecast, ETF flows, OI changes, and history. Served from cache when available.

Accepts `?ticker=IBIT` (or `ETHA`) and `?dte=N` (1-90) query parameters.

### `GET /api/candles`
Returns candlestick data from SQLite (90-day backfill from Binance). Accepts `?ticker=IBIT` and `?tf=15m` (15m/1h/4h/1d).

### `GET /api/flows`
Returns last 30 days of ETF fund flow data (date, flow in dollars, shares outstanding, AUM, NAV). Accepts `?ticker=IBIT`.

### `GET /api/analysis`
Returns the cached AI analysis for today, or `{"status": "pending"}` if not yet generated. Accepts `?ticker=IBIT`.

### `POST /api/analyze`
Force re-run AI analysis across all DTE timeframes. Returns the analysis JSON directly. Requires `ANTHROPIC_API_KEY`. Accepts `?ticker=IBIT`.

## Config Constants

| Constant | Default | Purpose |
|---|---|---|
| `RISK_FREE_RATE` | 13-week T-bill (^IRX) | Fetched daily from Yahoo Finance, falls back to 4.3% |
| `BTC_PER_SHARE` | 0.000568 (IBIT), 0.0091 (ETHA) | Fallback — auto-calculated from live prices |
| `STRIKE_RANGE_PCT` | 0.35 | Filter strikes to +/-35% of spot |

## Known Limitations
- Yahoo Finance OI updates once daily (after market close), not intraday
- Assumes dealers are net short all options (standard but not always true — see positioning confidence)
- Crypto/share ratio auto-calculated from spot; actual NAV drifts slightly due to fees
- Vanna/charm magnitudes are estimates — actual dealer positioning depends on their book, which isn't public
- ETF flow backfill uses a volume-based heuristic (~15% of daily volume as creation/redemption proxy); real flow calculations begin after 2+ days of actual shares outstanding data
- AI analysis requires an Anthropic API key and uses Claude Opus (~$0.15/day)
