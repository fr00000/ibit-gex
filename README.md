# IBIT GEX Trading Dashboard

A web dashboard that pulls crypto ETF (IBIT/ETHA) options chain data from Yahoo Finance and BTC options from Deribit, calculates Gamma Exposure (GEX) using Black-Scholes, and displays actionable trading levels for **BTC/ETH perpetual futures trading with leverage**.

The core thesis: ETF options flow creates dealer hedging dynamics that produce support/resistance levels in the underlying crypto asset. When the gamma regime is positive, price is range-bound and you can fade the extremes on perps. When negative, ranges break and you should trade momentum or sit out.

Supports **IBIT** (Bitcoin ETF) and **ETHA** (Ethereum ETF) with ticker switching in the header. For IBIT, Deribit BTC options are automatically integrated, bringing total BTC options market coverage from ~52% (IBIT only) to ~91% (IBIT + Deribit).

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

DTE windows are selectable from the web UI dropdown. Windows are non-overlapping so each shows distinct positioning: 0-3 (immediate), 4-7 (next week), 8-14 (two weeks), 15-30 (monthly), 31-45 (structural).

### Flags

| Flag | Default | Description |
|---|---|---|
| `--port`, `-p` | 5000 | Server port |
| `--host` | 127.0.0.1 | Server host (use 0.0.0.0 for WSL2) |

## Dashboard

### Header
Ticker tabs (IBIT/ETHA), crypto price, ETF price, gamma regime badge, positioning confidence indicator, ETF flow badge (daily inflow/outflow with streak), timestamp, DTE window selector (non-overlapping ranges), source selector (ALL/IBIT/DERIBIT — shown when Deribit data is available), Deribit OI badge, candle timeframe selector (15m/1h/4h/1d), expiration date, refresh button.

### Candlestick Chart
Crypto candles (BTC/ETH via Binance, persisted in SQLite with 90-day backfill) with horizontal overlays at call wall, put wall, gamma flip, max pain, expected move bounds, support/resistance levels, and dealer delta flip points (purple dotted). Overlay lines update when the source selector changes. Timeframe is selectable from the header. Real-time updates via Binance WebSocket.

### GEX Profile
Stacked bar chart showing GEX at each strike. Green = positive gamma, red = negative gamma. In ALL source mode, bars are stacked by venue (IBIT solid, Deribit translucent). In IBIT mode, bars stack by expiry with distinct colors. In DERIBIT mode, Deribit-only bars are shown. An expiry filter dropdown (ALL EXP, NEAREST, NEXT 2, NEXT 3) controls which expirations are displayed. Tooltip shows net GEX, per-venue breakdown, and volume.

### Open Interest Profile
Call vs put open interest at each strike, showing where hedging activity is concentrated.

### ETF Flows
Dual-bar chart of daily ETF fund flows over the last 30 days sourced from Farside Investors. Wide faded bars show total BTC ETF flow (all spot Bitcoin ETFs combined), narrow solid bars show IBIT-specific flow. Green = inflow, red = outflow. When IBIT outflows coincide with positive total flow, it signals fund rotation rather than genuine institutional exit.

### Dealer Delta Profile
Bar chart of pre-computed dealer delta (hedging pressure) at hypothetical prices across the key level grid. Green bars = dealers must BUY (supportive), red bars = dealers must SELL (resistive). Shows where dealer hedging creates natural support/resistance independent of the GEX profile.

### Sidebar

- **AI Analysis** — Claude-powered trading analysis across all non-overlapping DTE windows (0-3d/4-7d/8-14d/15-30d/31-45d + cross-timeframe). Uses combined IBIT+Deribit levels as primary data with per-venue breakdown for divergence analysis. Auto-runs daily when fresh data arrives (saved automatically). Manual refresh via the refresh button generates a new analysis without saving; use the Save button to persist it when satisfied. Includes day-over-day level changes, historical trend context, and prior analysis for thesis continuity.
- **Regime Banner** — Positive gamma (range-bound, fade extremes) or negative gamma (trending, don't fade)
- **Expected Move** — Implied straddle range for nearest expiry
- **Range Visual** — In positive gamma: tradeable range between put wall and call wall with spot indicator
- **Key Levels** — Call Wall, Put Wall, Gamma Flip, Max Pain with BTC prices and distance from spot
- **Significant Levels** — High-OI strikes with regime-adjusted behavior, OI changes (BUILDING/DECAYING), and vanna/charm notes
- **Breakout Assessment** — Upside/downside signal scoring with targets, includes ETF flow streak signals
- **Dealer Flows** — Overnight charm forecast, vanna vol scenarios, and combined overnight dealer rebalancing estimate
- **OI Changes** — Call/put open interest shifts vs prior snapshot
- **ETF Fund Flows** — IBIT and total BTC ETF daily flow amount, direction, strength, and 5-day momentum with streak dots
- **Dealer Hedging Pressure** — Scenario analysis showing current dealer delta position, delta flip points (where hedging direction reverses), dealer delta at key levels, and a morning briefing summary
- **Dealer Position** — Net GEX, Active GEX, dealer delta, net vanna, net charm, put/call ratio
- **History** — Daily snapshots of regime and levels

## How It Works

### Data Pipeline
1. Fetches ETF spot price and reference crypto price (BTC-USD or ETH-USD) from Yahoo Finance
2. Auto-calculates crypto/share ratio (ETF price / crypto price)
3. Pulls options chains across expirations within the DTE window
4. Calculates Black-Scholes gamma, delta, vanna, and charm using per-strike implied volatility
5. Aggregates to build a GEX profile, derive levels, and compute dealer flow forecasts
6. For IBIT: fetches Deribit BTC options via public API (`get_book_summary_by_currency`), computes Greeks with 1 BTC/contract multiplier, and merges into a combined BTC-price-indexed profile
7. Fetches ETF fund flows from Farside Investors (IBIT-specific and total BTC ETF daily flows)

### Data Caching
Options OI updates once per day (after market close). The app caches the full computed result in SQLite per DTE. On subsequent loads it compares OI to detect when Yahoo has fresh data — if unchanged, the cache is served instantly. Yahoo is re-checked at most every 30 minutes until new data is confirmed. Deribit data is cached in-memory for 1 hour (single API call returns all BTC options).

A background thread pre-fetches all 5 non-overlapping DTE windows on startup and keeps the cache warm. Once all windows are fresh for the day, it auto-runs AI analysis (if `ANTHROPIC_API_KEY` is set).

### GEX Calculation
```
IBIT:    GEX = Gamma x OI x 100 x Spot^2 x 0.01    (100 shares/contract)
Deribit: GEX = Gamma x OI x 1 x Spot^2 x 0.01      (1 BTC/contract)
```
- **Call GEX** = positive (dealers short calls -> buy dips / sell rips)
- **Put GEX** = negative (dealers short puts -> sell dips / buy rips)
- **Net GEX** at each strike = Call GEX + Put GEX

IBIT strikes are converted to BTC via `strike / btc_per_share`. Deribit strikes are already in BTC/USD. Both merge into a single BTC-price-indexed profile.

### Active GEX
Active GEX weights net GEX by the fraction of OI that is new since yesterday (delta_OI / total_OI). This surfaces where fresh dealer exposure is concentrated vs stale positions from days ago. Active walls show the strikes with the most new positioning.

### Dealer Delta Scenarios
Pre-computes dealer delta (net hedging pressure) at hypothetical prices across the key level grid. At each price point, Black-Scholes delta is recalculated for every option in the chain. Key outputs:
- **Delta flip points** — Prices where net dealer delta crosses zero (hedging direction reverses). Distinct from gamma flip.
- **Hedging acceleration** — Rate of change of dealer delta. High acceleration zones are inflection points where small price moves trigger large hedging flows.
- **Key level deltas** — Dealer delta values at call wall, put wall, gamma flip, and max pain.

Sign convention: negative dealer delta = dealers must BUY = supportive (green). Positive = dealers must SELL = resistive (red).

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

### DTE Windows
DTE windows are **non-overlapping** so each shows distinct option positioning rather than cumulative views dominated by near-term expirations:

| Window | Range | What it shows |
|---|---|---|
| 0-3 DTE | Today through 3 days | Immediate expirations — highest gamma, strongest hedging pressure. Most actionable. |
| 4-7 DTE | 4 through 7 days | Next week's setup — where the next wave of gamma is forming. |
| 8-14 DTE | 8 through 14 days | Two-week positioning — emerging walls that become dominant after near-term expiry. |
| 15-30 DTE | 15 through 30 days | Monthly cycle — institutional positioning around monthly options expiration. |
| 31-45 DTE | 31 through 45 days | Structural — quarterly and longer-dated positioning that forms the backdrop. |

Default is 0-3 for morning briefings. When comparing across windows, matching levels (e.g. same call wall in 0-3d and 4-7d) indicate high-conviction multi-week positioning. Divergent levels flag potential level migration after near-term expiry.

## Daily OI Tracking

Snapshots are stored in SQLite (`~/.ibit_gex_history.db`). The dashboard compares against the previous day's snapshot:
- Aggregate OI changes with positioning interpretation
- Per-level OI deltas (BUILDING / DECAYING)
- Historical regime and level trends

The database also stores full data caches (per DTE) and AI analysis results, both keyed by date.

## API

### `GET /api/data`
Returns JSON with spot prices, levels, GEX/OI chart data, significant levels, breakout assessment, dealer flow forecast, ETF flows, OI changes, and history. Served from cache when available.

Query parameters:
- `?ticker=IBIT` (or `ETHA`)
- `?dte=N` — max DTE (1-90, default 3)
- `?min_dte=N` — min DTE (default 0). Use with `dte` to define non-overlapping windows (e.g. `?min_dte=4&dte=7`).

### `GET /api/candles`
Returns candlestick data from SQLite (90-day backfill from Binance). Accepts `?ticker=IBIT` and `?tf=15m` (15m/1h/4h/1d).

### `GET /api/flows`
Returns last 30 days of ETF fund flow data from Farside Investors (date, flow in dollars, total BTC ETF flow). Accepts `?ticker=IBIT`.

### `GET /api/analysis`
Returns the cached AI analysis for today, or `{"status": "pending"}` if not yet generated. Accepts `?ticker=IBIT`.

### `POST /api/analyze`
Re-run AI analysis across all DTE timeframes. Returns the analysis JSON without saving to the database — use `POST /api/analysis/save` to persist. Requires `ANTHROPIC_API_KEY`. Accepts `?ticker=IBIT`.

### `POST /api/analysis/save`
Save an AI analysis result to the database for today. Accepts the analysis JSON as the request body. Overwrites any previously saved analysis for today. Accepts `?ticker=IBIT`.

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
- ETF flow data from Farside Investors is only available for IBIT (and total BTC ETFs); ETHA flow data is not yet supported
- Deribit integration is BTC-only (IBIT); ETHA/ETH Deribit options are not fetched
- Deribit data is cached for 1 hour; IBIT OI is daily — refresh cadences differ
- Expected move is derived from IBIT ATM straddle only, not Deribit
- Significant levels, breakout signals, and flow forecast are computed from IBIT data only; dealer delta scenarios include both venues
- If Deribit API is unreachable, the dashboard falls back to IBIT-only seamlessly
- AI analysis requires an Anthropic API key and uses Claude Opus (~$0.15/day)
