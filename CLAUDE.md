# CLAUDE.md — GEX Dashboard Codebase Guide

## What This Is

Options gamma exposure (GEX) dashboard for BTC/ETH ETFs (IBIT, ETHA). Flask backend + vanilla JS frontend in a single `app.py` (~6300 lines) and `templates/index.html` (~2500 lines). Combines IBIT options data (via Yahoo Finance), Deribit crypto options, Coinglass derivatives data, and ETF flows (Farside Investors) into a real-time positioning analysis tool.

The AI analysis feature sends structured per-window data to Claude Opus 4.6 for cross-timeframe positioning synthesis.

## File Structure

```
app.py                  # Everything: Flask app, data fetching, analysis, all logic
templates/index.html    # Dashboard UI: Chart.js + custom canvas charts, all frontend JS
templates/macro.html    # Standalone macro regime page with signal charts (~1100 lines)
.env                    # ANTHROPIC_API_KEY, COINGLASS_API_KEY (not committed)
gex_data.db             # SQLite database (auto-created)
```

## app.py Section Map (approximate line ranges)

| Lines | Section | Key Functions |
|-------|---------|--------------|
| 1-130 | Config, logging, BS math | `DTE_WINDOWS`, `TICKER_CONFIG`, `bs_gamma/delta/vanna/charm` |
| 170-330 | Database | `init_db()` — 9 tables: snapshots, strike_history, data_cache, analysis_cache, btc_candles, etf_flows, predictions, coinglass_data, expiry_cache |
| 330-780 | History & structure trends | `get_prev_strikes`, `summarize_history_trends`, `summarize_structure_trends` |
| 780-1200 | Phase 2+3 signal functions | `_compute_funding_signal`, `_compute_oi_signal`, `_compute_liquidation_signal`, `_compute_pcr_direction`, `_compute_wall_breach`, `_compute_iv_term_signal`, `_compute_score_history` |
| 1200-2030 | Macro regime | `compute_macro_regime` — 11-signal scoring system (-100 to +100) |
| 2030-2450 | External data | ETF flows (Farside HTML parsing), Coinglass API (funding, OI, liquidations), Deribit freshness |
| 2450-2625 | Deribit + candles | `fetch_deribit_options`, BTC candle backfill/update |
| 2625-2755 | Level computation | `_compute_levels_from_df` (GEX computation from dataframe) |
| 2755-3465 | Core data pipeline | `fetch_and_analyze` — main data fetch: Yahoo + Deribit → combined levels, IV term structure aggregation, per-expiry caching |
| 3465-3820 | Flow & dealer delta | `compute_flow_forecast` (charm/vanna), `compute_dealer_delta_scenarios`, `generate_dealer_delta_briefing` |
| 3820-4030 | Significant levels & breakout | `compute_significant_levels`, `compute_breakout` |
| 4030-4115 | Cache layer | `get_latest_cache`, `set_cached_data`, `fetch_with_cache` |
| 4115-4540 | Background refresh | `_refresh_deribit_only`, `_bg_deribit_overlay` (per-DTE Deribit merge + IV storage), `_bg_refresh` (5min loop) |
| 4540-4935 | API routes (data) | `/api/data`, `/api/outlook`, `/api/range-cone`, `/api/structure`, `/api/structure/heatmap`, `/api/candles`, `/api/flows` |
| 4935-5090 | Per-expiry data system | `/api/expiry-data` — list expiries, single expiry, date range. Reads from `expiry_cache` table |
| 5090-5400 | Main data API | `/api/data` — the primary endpoint, calls `fetch_with_cache` |
| 5400-5750 | Analysis data builder | `build_analysis_data()` — assembles JSON blob for AI including IV term structure per DTE, macro regime, structure trends |
| 5750-6080 | AI system prompt + runner | `run_analysis()` — system prompt (~330 lines), Claude Opus 4.6 API call |
| 6080-6290 | Analysis API + accuracy | `/api/analysis`, `/api/analyze`, `/api/accuracy`, `/api/macro-regime` |
| 6290-6355 | Macro page + main | `/macro` route, `app.run()` |

## Key Architecture Concepts

### DTE Windows (non-overlapping)
```python
DTE_WINDOWS = [(3, 0, 3), (7, 4, 7), (14, 8, 14), (30, 15, 30), (45, 31, 45)]
```
Each window shows distinct option positioning. When comparing across windows, same-strike walls = high conviction. Different-strike walls = level migration after expiry.

### Data Flow
1. Yahoo Finance → IBIT/ETHA options chains (updates at market close ~4:15 PM ET)
2. Deribit API → BTC/ETH crypto options (near real-time, cached 60min)
3. Combined into `combined_levels_btc` with per-venue breakdown
4. Cached in SQLite `data_cache` table per ticker+DTE window
5. `build_analysis_data()` assembles all windows + history + macro into one JSON blob
6. `run_analysis()` sends blob to Claude Opus for synthesis

### Macro Regime (11-signal scoring, -100 to +100)

**Phase 1 — GEX-derived (±12 each):**
- Regime persistence & transition (consecutive days in regime, transition bonus at 7+ days)
- Structural wall migration (31-45d walls, needs 7+ days history)
- Range compression + spot position (uses same DTE for range history and spot %)
- ETF flow momentum (reversal detection)
- Venue wall convergence (IBIT vs Deribit, DTE-scaled threshold: 2% at dte=3, 4% at dte=45)

**Phase 2 — Coinglass (±13 each, requires `COINGLASS_API_KEY`):**
- Funding rate
- Aggregate OI
- Liquidation intensity

**Phase 3 — Options-derived (±8 to ±13):**
- PCR direction (±8, contrarian: surging PCR = bullish)
- Wall breach detection (±13, spot through GEX wall + regime)

**Phase 4 — Volatility (±8):**
- IV term structure shape (backwardation = fear = contrarian bullish)

Total clamped to ±100.

### Background Refresh (`_bg_refresh`)
- **Post-close (primary)**: 4:20 PM ET Mon-Fri, force-refresh all windows → run AI analysis → save predictions
- **Weekend**: Once per day if no analysis cached (Deribit-primary)
- **>2% move**: Re-runs if ref asset moved >2% since last analysis
- Tracked by `post_close_done` dict to prevent re-triggering

### AI Analysis Design
- Output is a **positioning map**, not trade signals. No trade plans, entry/exit, stop losses.
- Negative GEX = amplifies moves (acceleration). NEVER call it "support" or "floor."
- Positive GEX = dampens moves (stabilization). This IS mechanical support/resistance.
- Uses `changes_vs_prev` and `_history_trends` for level changes — no prior analysis text is injected.
- Behavior labels in `compute_significant_levels` use structural descriptions: "acceleration zone — dealers sell + gamma amplifies"

### Frontend Charts (in index.html)
- **Price chart**: TradingView lightweight-charts (top-left)
- **Positioning Outlook**: Chart.js line chart — walls, flips, expected move across DTE windows (top-right)
- **GEX Profile / Open Interest / OI Skew**: Chart.js charts in GEX cell (middle-left, tabbed)
- **Dealer Delta Profile**: Chart.js bar chart (middle-right)
- **Wall Migration with Forward Projection**: Chart.js line chart — 30d historical 31-45d walls + forward-projected walls from DTE windows with NOW divider (bottom-left, DEFAULT tab)
- **Range Cone**: Custom canvas — continuous GEX heatmap strips per window + expected move funnel (bottom-left, tab 2)
- **OI/GEX Heatmap**: Custom canvas (bottom-left, tabs 3-4)
- **ETF Flows**: Chart.js bar chart (bottom-right)
- **Macro Regime Bar**: Score badge + signal pills in header
- **Per-Expiry Selector**: Dropdown with DTE windows, date ranges, individual expiries

### Database Tables
| Table | Purpose |
|-------|---------|
| `snapshots` | Daily level snapshots (spot, walls, flip, regime) |
| `strike_history` | Per-strike OI for day-over-day comparison |
| `data_cache` | Cached full chain data per ticker+DTE (JSON blob) |
| `analysis_cache` | AI analysis output per day per ticker |
| `btc_candles` | OHLCV candles from Binance |
| `etf_flows` | Daily ETF fund flows from Farside |
| `predictions` | Saved level predictions for accuracy scoring |
| `coinglass_data` | Funding rates, aggregate OI, liquidations |
| `expiry_cache` | Per-expiry strike data (GEX, OI, greeks per strike per expiry date) |

### IV Term Structure
ATM IV is computed per expiry during both IBIT processing (from `impliedVolatility`) and Deribit overlay (from `mark_iv`). Stored as `iv_term_structure` array in the `data_cache` blob. Each entry has `{expiry, dte, atm_iv, call_iv, put_iv, source}`. ATM = OI-weighted average of options within 2% of spot. IBIT IV is decimal (0.65), Deribit is percentage (65.0) — converted during computation.

### Per-Expiry Data System
Separate from the DTE window cache. `expiry_cache` stores per-strike data for each individual expiry date, including merged IBIT + Deribit data. Frontend expiry selector dynamically populates from `/api/expiry-data` with expiry count, OI, and DTE. Client-side filters expired entries and recomputes DTE.

### Wall Migration Forward Projection
The wall migration chart merges historical data from `/api/structure` with forward data from `/api/outlook`. Each DTE window's current walls are plotted at their forward date (4-7d→+5d, 8-14d→+11d, 15-30d→+22d, 31-45d→+38d). These are the market's implied forward wall positions — directly observable, no model needed.

## Common Tasks

### Adding a new field to AI analysis data
1. Add computation in `build_analysis_data()` (~line 5400)
2. Add to per-window summary dict
3. Add interpretation instructions to system prompt in `run_analysis()` (~line 5750)
4. Token budget: currently ~16K max_tokens, data blob should stay under ~8K tokens

### Adding a new chart
1. Backend: Add `/api/your-endpoint` route
2. Frontend: Add canvas element in HTML, render function in JS
3. Call load function from `loadData()` success path (~line 1194)
4. Use custom canvas for non-standard visualizations; Chart.js for standard line/bar charts

### Adding a new macro signal
1. Write the signal function (e.g. `_compute_new_signal(c, ticker, btc_per_share)`) returning `(score, detail, history)`
2. Call it in `compute_macro_regime()` after the appropriate phase
3. Add to `total_score` sum
4. Add to `signals` dict in return with `{'score':..., 'max':..., 'detail':...}`
5. Add to `history` dict if it produces historical data
6. Update `_compute_score_history()` to include in daily recompute
7. Update the AI system prompt's macro section to explain the signal
8. Frontend macro page renders dynamically — no hardcoded signal names needed
9. Main dashboard macro pills render dynamically via the `abbr` map — add abbreviation there

### Modifying the system prompt
The AI system prompt is a single long f-string starting at ~line 5750 in `run_analysis()`. It contains:
- Data field explanations
- Output format instructions (POSITIONING summary, not trade signals)
- Quality rules (negative GEX language, fabricated numbers, OI vs expiry)
- Cross-timeframe synthesis instructions

### Adding a new DTE window
Add to `DTE_WINDOWS` list. Everything else (caching, analysis, charts) automatically picks it up.

## Environment Variables
- `ANTHROPIC_API_KEY` — Required for AI analysis
- `COINGLASS_API_KEY` — Optional, enables funding rate, aggregate OI, liquidation signals in macro regime

## Style Conventions
- Dark theme: background `#06080c`, text `#e2e8f0`
- Font: IBM Plex Mono throughout
- Colors: green `#00dc82` (positive/support), red `#ff4060` (negative/resistance), yellow `#ffb020` (gamma flip), purple `#c084fc` (delta flip)
- Chart patterns: see existing heatmap and range-cone for custom canvas; see outlook and GEX profile for Chart.js

## Important Gotchas
- `app.py` is one large file (~6300 lines). When editing, use precise line numbers and verify context.
- `gex_chart` in the cache vs `gex_distribution` in analysis data are related but different: `gex_chart` is the raw per-strike data, `gex_distribution` is the top-20 extracted for the AI prompt.
- IBIT data is daily (stale overnight/weekends). Deribit is near real-time. The `data_freshness` field tracks this.
- `btc_per_share` converts between IBIT share prices and BTC prices. All analysis uses BTC prices.
- The `combined_levels_btc` field in cache has the merged IBIT+Deribit levels. `levels` has IBIT-only. `deribit_levels_btc` has Deribit-only.