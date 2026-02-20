# CLAUDE.md — GEX Dashboard Codebase Guide

## What This Is

Options gamma exposure (GEX) dashboard for BTC/ETH ETFs (IBIT, ETHA). Flask backend + vanilla JS frontend in a single `app.py` (~5100 lines) and `templates/index.html` (~1800 lines). Combines IBIT options data (via Yahoo Finance), Deribit crypto options, Coinglass derivatives data, and ETF flows (Farside Investors) into a real-time positioning analysis tool.

The AI analysis feature sends structured per-window data to Claude Opus 4.6 for cross-timeframe positioning synthesis.

## File Structure

```
app.py                  # Everything: Flask app, data fetching, analysis, all logic
templates/index.html    # Dashboard UI: Chart.js + custom canvas charts, all frontend JS
templates/macro.html    # Standalone macro regime page
.env                    # ANTHROPIC_API_KEY, COINGLASS_API_KEY (not committed)
gex_data.db             # SQLite database (auto-created)
```

## app.py Section Map (approximate line ranges)

| Lines | Section | Key Functions |
|-------|---------|--------------|
| 1-125 | Config, logging | `DTE_WINDOWS`, `TICKER_CONFIG`, env vars |
| 127-165 | Black-Scholes | `bs_gamma`, `bs_delta`, `bs_vanna`, `bs_charm` |
| 170-315 | Database | `init_db()` — 8 tables: snapshots, strike_history, data_cache, analysis_cache, btc_candles, etf_flows, predictions, coinglass_data |
| 316-675 | History & trends | `get_prev_strikes`, `summarize_history_trends` (30d regime, level migration, range evolution) |
| 675-770 | Structure trends | `summarize_structure_trends` (per-window wall migration over 7d) |
| 770-1570 | Macro regime | `compute_macro_regime` — scoring system (-100 to +100) combining funding, OI, ETF flows, venue convergence, liquidation |
| 1570-1985 | External data | ETF flows (Farside HTML parsing), Coinglass API (funding, OI, liquidations), data freshness |
| 1985-2160 | Deribit + candles | `fetch_deribit_options`, BTC candle backfill/update |
| 2160-2770 | Core data pipeline | `_compute_levels_from_df` (GEX computation), `fetch_and_analyze` (main data fetch: Yahoo + Deribit → combined levels) |
| 2770-3100 | Flow & dealer delta | `compute_flow_forecast` (charm/vanna), `compute_dealer_delta_scenarios`, `generate_dealer_delta_briefing` |
| 3100-3325 | Significant levels | `compute_significant_levels` (behavioral labels), `compute_breakout` |
| 3325-3410 | Cache layer | `get_latest_cache`, `set_cached_data`, `fetch_with_cache` |
| 3410-3650 | Background refresh | `_bg_refresh()` — 5min loop: post-close trigger (4:20 PM ET), weekend runs, >2% move re-analysis |
| 3650-3990 | API routes | `/api/data`, `/api/outlook`, `/api/range-cone`, `/api/structure`, `/api/candles`, `/api/flows` |
| 3990-4380 | Analysis cache | `get_cached_analysis`, `save_predictions`, `score_expired_predictions`, `detect_structural_patterns` |
| 4380-4600 | Analysis data builder | `build_analysis_data()` — assembles the JSON blob sent to AI. GEX distribution extraction at ~4504 |
| 4600-4940 | AI system prompt + runner | `run_analysis()` — system prompt (~300 lines), Claude API call. **No prior analysis injection** (removed intentionally) |
| 4940-5150 | Analysis API + accuracy | `/api/analysis`, `/api/analyze`, `/api/accuracy`, `/api/macro-regime` |

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
- **GEX Profile**: Chart.js bar chart, IBIT vs Deribit stacked (bottom-left, tab 1)
- **OI Profile**: Chart.js bar chart (bottom-left, tab 2)
- **Positioning Outlook**: Chart.js line chart — walls, flips, expected move across DTE windows (top-right)
- **Range Cone**: Custom canvas — continuous GEX heatmap strips per window + expected move funnel (bottom-right, default tab)
- **Walls / Structure**: Chart.js line chart — 30d wall evolution (bottom-right, tab 2)
- **OI/GEX Heatmap**: Custom canvas (bottom-right, tabs 3-4)
- **Dealer Delta Profile**: Chart.js bar chart (middle-right)
- **Flow Forecast**: Chart.js bar chart (sidebar)

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

## Common Tasks

### Adding a new field to AI analysis data
1. Add computation in `build_analysis_data()` (~line 4381)
2. Add to per-window summary dict
3. Add interpretation instructions to system prompt in `run_analysis()` (~line 4600)
4. Token budget: currently ~16K max_tokens, data blob should stay under ~8K tokens

### Adding a new chart
1. Backend: Add `/api/your-endpoint` route
2. Frontend: Add canvas element in HTML, render function in JS
3. Call load function from `loadData()` success path (~line 1194)
4. Use custom canvas for non-standard visualizations; Chart.js for standard line/bar charts

### Modifying the system prompt
The AI system prompt is a single long f-string starting at ~line 4600 in `run_analysis()`. It contains:
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
- `app.py` is one large file (~5100 lines). When editing, use precise line numbers and verify context.
- `gex_chart` in the cache vs `gex_distribution` in analysis data are related but different: `gex_chart` is the raw per-strike data, `gex_distribution` is the top-20 extracted for the AI prompt.
- IBIT data is daily (stale overnight/weekends). Deribit is near real-time. The `data_freshness` field tracks this.
- `btc_per_share` converts between IBIT share prices and BTC prices. All analysis uses BTC prices.
- The `combined_levels_btc` field in cache has the merged IBIT+Deribit levels. `levels` has IBIT-only. `deribit_levels_btc` has Deribit-only.