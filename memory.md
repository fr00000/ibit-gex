# memory.md — Architectural Decisions & Context

This file captures the "why" behind design choices, debugging outcomes, and signal tuning rationale. Updated as decisions are made.

---

## Macro Regime Signal Design

### Venue Convergence DTE Order (Feb 25, 2026)
**Decision:** Fallback order is `[3, 7, 14, 30, 45]` (shortest first), NOT longest first.
**Why:** DB analysis showed Deribit coverage is 100% at dte=3 but only 70% at dte=45. At dte=30, Deribit walls often land on sparse round strikes (80K/60K) that aren't real positioning. Short DTEs have dense strikes where market makers actively hedge — convergence there is a stronger signal.
**Data:** Feb 25 snapshot: dte=3 has deribit 11/11 days, dte=45 has deribit 7/10 days.

### Venue Convergence Threshold Scaling (Feb 25, 2026)
**Decision:** Threshold scales by DTE: 2% at dte=3, 2.5% at dte=7, 3% at dte=14, 3.5% at dte=30, 4% at dte=45.
**Why:** At short DTEs, strike grids are dense and 4% is too generous — gives false convergences. At long DTEs, strikes are sparse and wider tolerance is appropriate.

### Range Compression Timeframe Fix (Feb 25, 2026)
**Decision:** Range history percentile and spot position now use the SAME DTE window (shortest available with 7+ days).
**Why:** Previously, range history used dte=45 structural walls while spot_pct used dte=3 tactical walls. The 45-DTE call wall could be $80K while the 0-3 DTE wall is $70K — so "compressed range" and "spot position within range" referred to different ranges. Now both use the same walls for consistency.

### Range Compression Spot Position Logic (Feb 25, 2026)
**Decision:** Score direction depends on WHERE spot sits within the compressed range + gamma regime.
**Why:** Previous logic was a blanket directional assumption: positive gamma = breaks DOWN, negative gamma = breaks UP. This ignored spot position entirely. Example: spot at 94% of range (near call wall) in positive gamma should be rejection (-12), but spot at 6% (near put wall) in positive gamma should be bounce (+12). Same regime, opposite direction — position matters.

### Wall Migration Threshold (Feb 25, 2026)
**Decision:** Lowered from 14 days to 7 days required, with dynamic half-window comparison.
**Why:** DB only had ~10 days of data when implemented. 14-day requirement made the signal permanently stuck at 0. With 7-day minimum and `half = len(wall_history) // 2` for comparison windows, it activates earlier and adapts to available data.

### Regime Transition Threshold (Feb 25, 2026)
**Decision:** Lowered from 20 consecutive days in old regime to 7.
**Why:** With only 11 days of snapshot data, the 20-day requirement was literally unreachable. 7 days is still meaningful (a full trading week in one regime then flipping).

## Signal Additions

### PCR Direction (Feb 25, 2026)
**Why added:** Put/call ratio is a classic contrarian sentiment indicator. The data was already stored in `levels.pcr` per DTE per day but wasn't being used in the macro score. Surging PCR = fear building = contrarian bullish. Scored ±8 (lower weight than structural signals).

### Wall Breach Detection (Feb 25, 2026)
**Why added:** The most important missing signal. When spot trades through a GEX wall, dealer hedging shifts from stabilizing to amplifying. This is the most mechanically significant event in the GEX framework. Scored ±13 (high weight). Direction depends on gamma regime: negative gamma + above call wall = breakout acceleration, positive gamma + above call wall = likely pinning.

### IV Term Structure (Feb 25, 2026)
**Why added:** Backwardation (short-dated IV > long-dated IV) is a reliable marker of stress that often precedes bottoms. The raw IV existed per option during computation but was discarded after Greeks calculation. Now aggregated as OI-weighted ATM IV per expiry and stored in `data_cache`. Scored ±8. Signal needs ~5 days to accumulate history before it starts scoring.
**Implementation note:** IBIT IV is decimal (0.65 = 65%), Deribit IV is percentage (65.0 = 65%). Must divide Deribit by 100 during ATM IV computation. ATM defined as options within 2% of spot.

## Data Architecture

### Per-Expiry Data System (Feb 24, 2026)
**Why:** Users wanted to view positioning for individual expiry dates, not just DTE windows. Each DTE window aggregates multiple expiries which can hide important structure.
**Design:** Separate `expiry_cache` table stores per-strike data for each expiry date. Built during `fetch_and_analyze()` alongside the existing DTE window cache. Frontend expiry selector populates dynamically from `/api/expiry-data` list endpoint.
**Gotcha:** Stored `dte` field is computed at write time and becomes stale. Client-side filter removes expired entries and recomputes DTE from actual date difference.

### IV Term Structure Storage (Feb 25, 2026)
**Design:** `iv_term_structure` array stored inside the `data_cache` JSON blob. Each entry: `{expiry, dte, atm_iv, call_iv, put_iv, source}`. IBIT entries created during `fetch_and_analyze()`, Deribit entries added during `_bg_deribit_overlay()`.
**Why in data_cache:** Keeps it co-located with all other per-DTE data. No new table needed. Macro signal reads it from the cached blob.

## Frontend Design

### Wall Migration Chart with Forward Projection (Feb 25, 2026)
**Why:** The original "Walls" chart plotted all 5 DTE windows' walls, which was visually busy and hard to read. Replaced with focused 31-45d structural walls + forward projection.
**Insight:** Each DTE window's current walls ARE the market's implied forward wall positions. Today's 4-7d walls are where the 0-3d walls will migrate after near-term expiries clear. Plotting them at their forward date (midpoint of DTE range) creates a natural projection without any model.
**Data:** Historical from `/api/structure`, forward from `/api/outlook`. Both fetched in parallel.

### OI Skew Moved to GEX Cell (Feb 25, 2026)
**Why:** OI Skew is a per-strike chart (strike space, same as GEX Profile and Open Interest). It belongs with the other strike-space charts. Wall Migration is a time-series chart and belongs with Range Cone and heatmaps. Making Wall Migration the default gives immediate structural context on page load.

## Debugging Notes

### Stale DTE in Expiry Selector (Feb 25, 2026)
**Problem:** DTE annotations were inflated by expired entries. "0-3 DTE" showed 6 expiries with 751K OI, but "This Week + Next Week" only showed 148K.
**Root cause:** `dte` field stored at write time, never updated. A Feb 22 expiry with `dte=0` stored on Feb 22 still matched `e.dte >= 0 && e.dte <= 3` on Feb 25.
**Fix:** Client-side filter: `expiryList.filter(e => e.expiry_date >= today)` then recompute DTE from actual date diff.

### Weekend Deribit Data (prior)
**Problem:** IBIT data is stale over weekends (no Yahoo updates), but Deribit keeps trading. Dashboard showed stale combined data.
**Fix:** `_refresh_deribit_only()` runs Deribit overlay on cached IBIT data during weekends. Deribit levels update while IBIT levels stay frozen. `data_freshness` shows age for each venue so the UI can flag staleness.
