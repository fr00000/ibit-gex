# Macro Regime Scoring System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a swing-trade macro regime scoring system (-100 to +100) with 8 signals, a dedicated `/macro` page, and a compact status bar on the main dashboard.

**Architecture:** New `compute_macro_regime()` function queries 30 days of existing DB tables (snapshots, data_cache, etf_flows) plus a new `coinglass_data` table for Phase 2 signals. Returns a rich dict with scores, breakdowns, and 30-day history for charting. Exposed via `/api/macro-regime` endpoint. Two frontend components: a compact status bar on the main dashboard and a full `/macro` page with gauge, signal table, and 6 charts.

**Tech Stack:** Python/Flask (app.py), SQLite, Chart.js, vanilla HTML/CSS/JS

**Key File Locations:**
- `app.py`: init_db() at L169, summarize_history_trends() at L357, fetch_farside_flows() at L870, _bg_refresh() at L2415, routes at L2551+, run_analysis() at L3267
- `templates/index.html`: CSS vars L11-36, header L264-302, grid L304-392, JS L395+, auto-refresh L1644

---

### Task 1: Branch + Database Schema

**Files:**
- Modify: `app.py:169-230` (init_db function)

**Step 1: Create branch**

```bash
git checkout -b macro-regime main
```

**Step 2: Add coinglass_data table to init_db()**

Inside `init_db()` (after the last CREATE TABLE, before the ALTER TABLE migrations), add:

```python
c.execute('''CREATE TABLE IF NOT EXISTS coinglass_data (
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL,
    extra_json TEXT,
    UNIQUE(date, symbol, metric)
)''')
```

**Step 3: Verify app starts**

```bash
cd /home/faisal/work/ibit && python app.py --port 5099 &
sleep 3 && kill %1
```
Expected: No errors, table created.

**Step 4: Commit**

```bash
git add app.py
git commit -m "Add coinglass_data table schema for macro regime signals"
```

---

### Task 2: Coinglass Data Fetching

**Files:**
- Modify: `app.py` (new function near fetch_farside_flows at L870)

**Step 1: Add COINGLASS_API_KEY loading**

Near the top of app.py where ANTHROPIC_API_KEY is loaded from env (around the config section), add:

```python
COINGLASS_API_KEY = os.environ.get('COINGLASS_API_KEY')
```

Also add to `.env.example`:
```
COINGLASS_API_KEY=your_key_here
```

**Step 2: Implement fetch_coinglass_data()**

Add after `fetch_farside_flows()` (~L988). Follow the same pattern: check API key, skip if already fetched today, fetch from endpoints, store in DB.

```python
_coinglass_lock = threading.Lock()

def fetch_coinglass_data():
    """Fetch aggregate funding rate and OI from Coinglass API.
    Stores daily snapshots in coinglass_data table. Graceful degradation if no API key."""
    api_key = os.environ.get('COINGLASS_API_KEY')
    if not api_key:
        return

    today = datetime.now().strftime('%Y-%m-%d')
    conn = get_db()
    c = conn.cursor()

    # Skip if already fetched today
    existing = c.execute(
        'SELECT 1 FROM coinglass_data WHERE date=? AND symbol=? AND metric=? LIMIT 1',
        (today, 'BTC', 'avg_funding_rate')
    ).fetchone()
    if existing:
        conn.close()
        return

    headers = {'accept': 'application/json', 'CG-API-KEY': api_key}

    # --- Endpoint 1: OI-Weighted Funding Rate ---
    try:
        url = 'https://open-api-v3.coinglass.com/api/futures/funding-rate/oi-weight-ohlc-history'
        req = urllib.request.Request(
            f'{url}?symbol=BTC&interval=h8&limit=90',
            headers=headers
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))

        if data.get('code') == '0' and data.get('data'):
            rows = data['data']
            for row in rows:
                ts = int(row['t']) // 1000 if row['t'] > 1e12 else int(row['t'])
                dt = datetime.utcfromtimestamp(ts)
                date_str = dt.strftime('%Y-%m-%d')
                rate = float(row['c'])  # close value
                # Store each 8h period with hour suffix for granularity
                hour_key = dt.strftime('%H')
                c.execute(
                    '''INSERT OR REPLACE INTO coinglass_data
                       (date, symbol, metric, value, extra_json)
                       VALUES (?, ?, ?, ?, ?)''',
                    (date_str, 'BTC', f'funding_rate_{hour_key}', rate,
                     json.dumps({'hour': hour_key, 'timestamp': ts}))
                )
            # Also store daily average (avg of last 3 periods = 24h)
            recent = rows[-3:] if len(rows) >= 3 else rows
            avg_rate = sum(float(r['c']) for r in recent) / len(recent)
            c.execute(
                '''INSERT OR REPLACE INTO coinglass_data
                   (date, symbol, metric, value, extra_json) VALUES (?, ?, ?, ?, ?)''',
                (today, 'BTC', 'avg_funding_rate', avg_rate, None)
            )
            # Backfill daily averages for historical dates
            from collections import defaultdict
            daily_rates = defaultdict(list)
            for row in rows:
                ts = int(row['t']) // 1000 if row['t'] > 1e12 else int(row['t'])
                dt = datetime.utcfromtimestamp(ts)
                daily_rates[dt.strftime('%Y-%m-%d')].append(float(row['c']))
            for d, rates in daily_rates.items():
                c.execute(
                    '''INSERT OR IGNORE INTO coinglass_data
                       (date, symbol, metric, value, extra_json) VALUES (?, ?, ?, ?, ?)''',
                    (d, 'BTC', 'avg_funding_rate', sum(rates) / len(rates), None)
                )
            log.info(f"[coinglass] Stored {len(rows)} funding rate records")
    except Exception as e:
        log.warning(f"[coinglass] Funding rate fetch failed: {e}")

    # --- Endpoint 2: Aggregated OI History ---
    try:
        url = 'https://open-api-v3.coinglass.com/api/futures/open-interest/ohlc-aggregated-history'
        req = urllib.request.Request(
            f'{url}?symbol=BTC&interval=1d&limit=90',
            headers=headers
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))

        if data.get('code') == '0' and data.get('data'):
            rows = data['data']
            for row in rows:
                ts = int(row['t']) // 1000 if row['t'] > 1e12 else int(row['t'])
                date_str = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')
                oi_usd = float(row['c'])  # close value
                c.execute(
                    '''INSERT OR REPLACE INTO coinglass_data
                       (date, symbol, metric, value, extra_json) VALUES (?, ?, ?, ?, ?)''',
                    (date_str, 'BTC', 'total_oi_usd', oi_usd, None)
                )
            log.info(f"[coinglass] Stored {len(rows)} OI records")
    except Exception as e:
        log.warning(f"[coinglass] OI fetch failed: {e}")

    conn.commit()
    conn.close()
```

**Step 3: Verify function works without API key (graceful)**

```bash
cd /home/faisal/work/ibit && python -c "
import app
app.init_db()
app.fetch_coinglass_data()  # Should return silently (no key)
print('OK')
"
```

**Step 4: Commit**

```bash
git add app.py .env.example
git commit -m "Add Coinglass data fetching (funding rate + aggregate OI)"
```

---

### Task 3: compute_macro_regime() — Phase 1 Signals

**Files:**
- Modify: `app.py` (new function near summarize_history_trends at L357)

**Step 1: Implement compute_macro_regime() with all 5 Phase 1 signals**

Add after `summarize_history_trends()`. This is a large function (~250 lines). Each signal is a self-contained block.

```python
def compute_macro_regime(conn, ticker, days=30):
    """Compute macro regime score from -100 (top) to +100 (bottom).
    Returns dict with overall score, per-signal breakdown, and swing recommendation."""

    c = conn.cursor()
    cfg = TICKER_CONFIG.get(ticker, TICKER_CONFIG['IBIT'])
    btc_per_share = cfg['per_share']
    today = datetime.now().strftime('%Y-%m-%d')

    # ── Signal 1: Regime Persistence & Transition ────────────────────────
    regime_score = 0
    regime_detail = ''
    regime_history = []

    rows = c.execute(
        'SELECT date, regime, net_gex FROM snapshots WHERE ticker=? ORDER BY date DESC LIMIT ?',
        (ticker, days)
    ).fetchall()

    if rows:
        regime_history = [{'date': r[0], 'regime': r[1], 'net_gex': r[2]} for r in rows]
        total = len(rows)
        neg_count = sum(1 for r in rows if r[1] == 'negative_gamma')
        pos_count = total - neg_count

        # Count sign flips
        flips = 0
        for i in range(1, len(rows)):
            if rows[i][1] != rows[i-1][1]:
                flips += 1

        neg_pct = neg_count / total if total > 0 else 0
        pos_pct = pos_count / total if total > 0 else 0

        # Base score
        if neg_pct > 0.70:
            regime_score = 8  # Washout forming
            regime_detail = f'{neg_count}/{total} days negative gamma ({neg_pct:.0%})'
        elif pos_pct > 0.70:
            regime_score = -8  # Complacency forming
            regime_detail = f'{pos_count}/{total} days positive gamma ({pos_pct:.0%})'
        else:
            regime_score = 0
            regime_detail = f'Oscillating: {pos_count} pos / {neg_count} neg days, {flips} flips'

        # Transition bonus: check if persistent regime just flipped
        if len(rows) >= 4:
            recent_regime = rows[0][1]  # Most recent
            recent_streak = 1
            for i in range(1, min(4, len(rows))):
                if rows[i][1] == recent_regime:
                    recent_streak += 1
                else:
                    break

            if recent_streak <= 3:
                # Check what was before the flip
                old_regime_count = 0
                for i in range(recent_streak, len(rows)):
                    if rows[i][1] != recent_regime:
                        old_regime_count += 1
                    else:
                        break

                if old_regime_count >= 20:
                    if recent_regime == 'positive_gamma':
                        # Flipped FROM persistent negative TO positive = bottom confirmed
                        regime_score = 12
                        regime_detail += f' | TRANSITION: {old_regime_count}d neg → pos (streak {recent_streak}d)'
                    else:
                        # Flipped FROM persistent positive TO negative = top confirmed
                        regime_score = -12
                        regime_detail += f' | TRANSITION: {old_regime_count}d pos → neg (streak {recent_streak}d)'

    # ── Signal 2: Structural Wall Migration ──────────────────────────────
    wall_migration_score = 0
    wall_detail = ''
    wall_history = []

    cache_rows = c.execute(
        'SELECT date, data_json FROM data_cache WHERE ticker=? AND dte=45 ORDER BY date DESC LIMIT ?',
        (ticker, days)
    ).fetchall()

    if len(cache_rows) >= 14:  # Need at least 2 weeks
        # Extract put_wall and call_wall in BTC for each day
        for row in cache_rows:
            try:
                d = json.loads(row[1])
                comb = d.get('combined_levels_btc') or {}
                levels = d.get('levels', {})
                cw = comb.get('call_wall') or (levels.get('call_wall', 0) / btc_per_share if btc_per_share else 0)
                pw = comb.get('put_wall') or (levels.get('put_wall', 0) / btc_per_share if btc_per_share else 0)
                if cw and pw:
                    wall_history.append({'date': row[0], 'call_wall': cw, 'put_wall': pw})
            except (json.JSONDecodeError, TypeError):
                continue

        wall_history.reverse()  # Oldest first

        if len(wall_history) >= 14:
            first_7 = wall_history[:7]
            last_7 = wall_history[-7:]

            avg_pw_first = sum(w['put_wall'] for w in first_7) / 7
            avg_pw_last = sum(w['put_wall'] for w in last_7) / 7
            avg_cw_first = sum(w['call_wall'] for w in first_7) / 7
            avg_cw_last = sum(w['call_wall'] for w in last_7) / 7

            pw_change = (avg_pw_last - avg_pw_first) / avg_pw_first if avg_pw_first else 0
            cw_change = (avg_cw_last - avg_cw_first) / avg_cw_first if avg_cw_first else 0

            pw_part = 0
            cw_part = 0
            pw_dir = 'stable'
            cw_dir = 'stable'

            if pw_change > 0.02:
                pw_part = 6
                pw_dir = 'rising'
            elif pw_change < -0.02:
                pw_part = -6
                pw_dir = 'falling'

            if cw_change > 0.02:
                cw_part = 6
                cw_dir = 'rising'
            elif cw_change < -0.02:
                cw_part = -6
                cw_dir = 'falling'

            wall_migration_score = pw_part + cw_part
            wall_detail = f'Put wall {pw_dir} ({pw_change:+.1%}), Call wall {cw_dir} ({cw_change:+.1%})'

    # ── Signal 3: Range Compression + Regime Context ─────────────────────
    compression_score = 0
    compression_detail = ''

    if len(cache_rows) >= 10:
        ranges = []
        for row in cache_rows:
            try:
                d = json.loads(row[1])
                comb = d.get('combined_levels_btc') or {}
                levels = d.get('levels', {})
                cw = comb.get('call_wall') or (levels.get('call_wall', 0) / btc_per_share if btc_per_share else 0)
                pw = comb.get('put_wall') or (levels.get('put_wall', 0) / btc_per_share if btc_per_share else 0)
                if cw and pw and cw > pw:
                    ranges.append(cw - pw)
            except (json.JSONDecodeError, TypeError):
                continue

        if ranges:
            current_range = ranges[0]  # Most recent (desc order)
            sorted_ranges = sorted(ranges)
            percentile_idx = sorted_ranges.index(current_range) if current_range in sorted_ranges else 0
            percentile = (percentile_idx / len(sorted_ranges)) * 100

            if percentile <= 20:
                current_regime = rows[0][1] if rows else None
                if current_regime == 'positive_gamma':
                    compression_score = -12
                    compression_detail = f'{percentile:.0f}th percentile range + positive gamma → breaks DOWN'
                elif current_regime == 'negative_gamma':
                    compression_score = 12
                    compression_detail = f'{percentile:.0f}th percentile range + negative gamma → breaks UP'
                else:
                    compression_detail = f'{percentile:.0f}th percentile range, oscillating regime'
            else:
                compression_detail = f'{percentile:.0f}th percentile range (not compressed)'

    # ── Signal 4: ETF Flow Momentum ──────────────────────────────────────
    etf_flow_score = 0
    flow_detail = ''
    flow_history = []

    flow_rows = c.execute(
        '''SELECT date, daily_flow_dollars, total_btc_etf_flow
           FROM etf_flows WHERE ticker=? ORDER BY date DESC LIMIT ?''',
        (ticker, days)
    ).fetchall()

    if flow_rows:
        flow_history = [
            {'date': r[0], 'ibit_flow': r[1], 'total_flow': r[2],
             'cumulative_10d': None}
            for r in flow_rows
        ]
        flow_history.reverse()  # Oldest first

        # Use total_btc_etf_flow, fall back to daily_flow_dollars
        flows = []
        for r in flow_rows:
            val = r[2] if r[2] is not None else r[1]
            flows.append(val or 0)

        flow_10d = sum(flows[:10]) if len(flows) >= 10 else sum(flows)
        flow_30d = sum(flows[:30]) if len(flows) >= 30 else sum(flows)

        # Compute cumulative 10d for history
        for i, fh in enumerate(flow_history):
            start = max(0, i - 9)
            cum = sum(
                (flow_history[j].get('total_flow') or flow_history[j].get('ibit_flow') or 0)
                for j in range(start, i + 1)
            )
            fh['cumulative_10d'] = cum

        # Check for flow reversal in last 5 days
        if len(flows) >= 15:
            flow_5d_recent = sum(flows[:5])
            flow_5d_prior = sum(flows[5:10])

            if flow_5d_prior < 0 and flow_5d_recent > 0:
                etf_flow_score = 12
                flow_detail = f'Flow reversal: 5d prior ${flow_5d_prior/1e6:.0f}M → recent ${flow_5d_recent/1e6:.0f}M'
            elif flow_5d_prior > 0 and flow_5d_recent < 0:
                etf_flow_score = -12
                flow_detail = f'Flow reversal: 5d prior ${flow_5d_prior/1e6:.0f}M → recent ${flow_5d_recent/1e6:.0f}M'
            elif flow_30d != 0:
                pace_10d = flow_10d / 10 if len(flows) >= 10 else flow_10d / len(flows)
                pace_30d = flow_30d / 30 if len(flows) >= 30 else flow_30d / len(flows)

                if flow_10d > 0 and pace_10d > pace_30d:
                    etf_flow_score = 8
                    flow_detail = f'Accelerating inflows: 10d ${flow_10d/1e6:.0f}M, pace above 30d'
                elif flow_10d < 0 and abs(pace_10d) > abs(pace_30d):
                    etf_flow_score = -8
                    flow_detail = f'Accelerating outflows: 10d ${flow_10d/1e6:.0f}M, pace above 30d'
                else:
                    # Linear scale between -8 and +8
                    ratio = pace_10d / pace_30d if pace_30d != 0 else 0
                    etf_flow_score = max(-8, min(8, int(ratio * 8)))
                    flow_detail = f'10d flow ${flow_10d/1e6:.0f}M, 10d/30d ratio {ratio:.2f}'
            else:
                flow_detail = 'Insufficient flow data for scoring'
        elif flows:
            flow_detail = f'Only {len(flows)} days of flow data'

    # ── Signal 5: Venue Wall Convergence ─────────────────────────────────
    venue_score = 0
    venue_detail = ''

    latest_cache = c.execute(
        'SELECT data_json FROM data_cache WHERE ticker=? AND dte=45 ORDER BY date DESC LIMIT 1',
        (ticker,)
    ).fetchone()

    if latest_cache:
        try:
            d = json.loads(latest_cache[0])
            comb = d.get('combined_levels_btc') or {}
            deribit = d.get('deribit_levels_btc') or {}
            levels = d.get('levels', {})

            if comb and deribit:
                # Get IBIT levels in BTC
                ibit_cw = levels.get('call_wall', 0) / btc_per_share if btc_per_share else 0
                ibit_pw = levels.get('put_wall', 0) / btc_per_share if btc_per_share else 0
                deribit_cw = deribit.get('call_wall', 0)
                deribit_pw = deribit.get('put_wall', 0)

                if ibit_pw and deribit_pw and ibit_pw > 0:
                    pw_diff_pct = abs(ibit_pw - deribit_pw) / ibit_pw
                    pw_converging = pw_diff_pct <= 0.02

                    if pw_converging:
                        # Check wall direction from wall_history
                        if wall_history and len(wall_history) >= 7:
                            pw_rising = wall_history[-1]['put_wall'] > wall_history[0]['put_wall'] * 1.02
                            if pw_rising:
                                venue_score += 12
                                venue_detail = f'Put walls converging ({pw_diff_pct:.1%} apart) + rising'
                            else:
                                venue_score += 6
                                venue_detail = f'Put walls converging ({pw_diff_pct:.1%} apart), stable/falling'

                if ibit_cw and deribit_cw and ibit_cw > 0:
                    cw_diff_pct = abs(ibit_cw - deribit_cw) / ibit_cw
                    cw_converging = cw_diff_pct <= 0.02

                    if cw_converging:
                        if wall_history and len(wall_history) >= 7:
                            cw_falling = wall_history[-1]['call_wall'] < wall_history[0]['call_wall'] * 0.98
                            if cw_falling:
                                venue_score -= 12
                                venue_detail += f'{" | " if venue_detail else ""}Call walls converging ({cw_diff_pct:.1%} apart) + falling'
                            else:
                                venue_score -= 6
                                venue_detail += f'{" | " if venue_detail else ""}Call walls converging ({cw_diff_pct:.1%} apart), stable/rising'

                # Clamp to ±12
                venue_score = max(-12, min(12, venue_score))

                if not venue_detail:
                    venue_detail = 'Venue walls not converging'
            else:
                venue_detail = 'Deribit data not available'
        except (json.JSONDecodeError, TypeError) as e:
            venue_detail = f'Parse error: {e}'

    # ── Phase 2 signals (computed in next section) ───────────────────────
    funding_score, funding_detail, funding_history = _compute_funding_signal(c)
    oi_score, oi_detail, oi_hist = _compute_oi_signal(c)
    liquidation_score = 0  # Stub

    # ── Final Score ──────────────────────────────────────────────────────
    total_score = (regime_score + wall_migration_score + compression_score
                   + etf_flow_score + venue_score
                   + funding_score + oi_score + liquidation_score)
    total_score = max(-100, min(100, total_score))

    # ── Score History (recompute for last 30 days) ───────────────────────
    score_history = _compute_score_history(conn, ticker, days)

    # ── Structural entry / invalidation ──────────────────────────────────
    structural_entry = None
    invalidation = None
    if abs(total_score) > 50:
        if total_score > 50 and wall_history:
            pw = wall_history[-1]['put_wall']
            structural_entry = f'${pw:,.0f} zone (31-45d put wall convergence)'
            invalidation = f'Below ${pw * 0.97:,.0f} (new low below structural floor)'
        elif total_score < -50 and wall_history:
            cw = wall_history[-1]['call_wall']
            structural_entry = f'${cw:,.0f} zone (31-45d call wall convergence)'
            invalidation = f'Above ${cw * 1.03:,.0f} (new high above structural ceiling)'

    return {
        'score': total_score,
        'bias': 'BOTTOMING' if total_score > 30 else 'TOPPING' if total_score < -30 else 'NEUTRAL',
        'swing_signal': total_score > 50 or total_score < -50,
        'high_conviction': abs(total_score) > 75,
        'signals': {
            'regime_persistence': {'score': regime_score, 'max': 12, 'detail': regime_detail},
            'wall_migration': {'score': wall_migration_score, 'max': 12, 'detail': wall_detail},
            'range_compression': {'score': compression_score, 'max': 12, 'detail': compression_detail},
            'etf_flow_momentum': {'score': etf_flow_score, 'max': 12, 'detail': flow_detail},
            'venue_convergence': {'score': venue_score, 'max': 12, 'detail': venue_detail},
            'funding_rate': {'score': funding_score, 'max': 13, 'detail': funding_detail, 'source': 'coinglass'},
            'aggregate_oi': {'score': oi_score, 'max': 13, 'detail': oi_detail, 'source': 'coinglass'},
            'liquidation': {'score': liquidation_score, 'max': 1, 'detail': 'stub', 'source': 'coinglass'},
        },
        'history': {
            'wall_migration': wall_history,
            'regime_history': regime_history,
            'etf_flows': flow_history,
            'funding_rates': funding_history,
            'oi_history': oi_hist,
            'score_components_daily': score_history,
        },
        'coinglass_available': bool(os.environ.get('COINGLASS_API_KEY')),
        'structural_entry': structural_entry,
        'invalidation': invalidation,
        'timestamp': datetime.now().isoformat(),
    }
```

**Step 2: Verify function runs against existing DB**

```bash
cd /home/faisal/work/ibit && python -c "
import app
app.init_db()
conn = app.get_db()
result = app.compute_macro_regime(conn, 'IBIT')
conn.close()
print(f'Score: {result[\"score\"]}, Bias: {result[\"bias\"]}')
for name, sig in result['signals'].items():
    print(f'  {name}: {sig[\"score\"]}/{sig[\"max\"]} - {sig[\"detail\"][:60]}')
"
```

**Step 3: Commit**

```bash
git add app.py
git commit -m "Add compute_macro_regime() with Phase 1 signals (5 signals, ±60 max)"
```

---

### Task 4: Phase 2 Signal Helpers

**Files:**
- Modify: `app.py` (helper functions before compute_macro_regime)

**Step 1: Implement _compute_funding_signal()**

```python
def _compute_funding_signal(c):
    """Signal 6: Aggregate Funding Rate (-13 to +13). Returns (score, detail, history)."""
    rows = c.execute(
        '''SELECT date, value FROM coinglass_data
           WHERE symbol='BTC' AND metric='avg_funding_rate'
           ORDER BY date DESC LIMIT 30''',
    ).fetchall()

    if not rows or len(rows) < 7:
        return 0, 'Insufficient funding data', []

    history = [{'date': r[0], 'rate': r[1]} for r in rows]
    history.reverse()

    # Compute 7d and 30d averages
    rates_7d = [r[1] for r in rows[:7]]
    rates_30d = [r[1] for r in rows[:30]]
    avg_7d = sum(rates_7d) / len(rates_7d)
    avg_30d = sum(rates_30d) / len(rates_30d)

    # Add 7d avg to history
    for i, h in enumerate(history):
        start = max(0, i - 6)
        vals = [history[j]['rate'] for j in range(start, i + 1)]
        h['avg_7d'] = sum(vals) / len(vals)

    score = 0
    detail = f'7d avg: {avg_7d:.4%}, 30d avg: {avg_30d:.4%}'

    if avg_7d < -0.0001:  # < -0.01%
        if avg_7d > avg_30d:  # Reverting toward 0
            score = 13
            detail += ' | Deeply negative, reverting → BOTTOM'
        else:
            score = 8
            detail += ' | Deeply negative, still falling → washout'
    elif avg_7d > 0.0003:  # > 0.03%
        if avg_7d < avg_30d:  # Reverting toward 0
            score = -13
            detail += ' | Deeply positive, reverting → TOP'
        else:
            score = -8
            detail += ' | Deeply positive, still rising → euphoria'
    else:
        detail += ' | Normal range, no signal'

    return score, detail, history
```

**Step 2: Implement _compute_oi_signal()**

```python
def _compute_oi_signal(c):
    """Signal 7: Aggregate Futures OI (-13 to +13). Returns (score, detail, history)."""
    rows = c.execute(
        '''SELECT date, value FROM coinglass_data
           WHERE symbol='BTC' AND metric='total_oi_usd'
           ORDER BY date DESC LIMIT 90''',
    ).fetchall()

    if not rows or len(rows) < 7:
        return 0, 'Insufficient OI data', []

    history = [{'date': r[0], 'oi_usd': r[1]} for r in rows]
    history.reverse()

    oi_current = rows[0][1]
    oi_90d_peak = max(r[1] for r in rows)
    change_from_peak = (oi_current - oi_90d_peak) / oi_90d_peak * 100

    # Add peak_pct to history
    for h in history:
        h['peak_pct'] = (h['oi_usd'] - oi_90d_peak) / oi_90d_peak * 100

    # 7d slope
    if len(rows) >= 7:
        oi_7d_ago = rows[6][1]
        oi_7d_slope = 'rising' if oi_current > oi_7d_ago * 1.02 else (
            'falling' if oi_current < oi_7d_ago * 0.98 else 'flat')
    else:
        oi_7d_slope = 'unknown'

    # Cross-reference funding for topping condition
    funding_rows = c.execute(
        '''SELECT value FROM coinglass_data
           WHERE symbol='BTC' AND metric='avg_funding_rate'
           ORDER BY date DESC LIMIT 7''',
    ).fetchall()
    funding_7d_avg = sum(r[0] for r in funding_rows) / len(funding_rows) if funding_rows else 0

    score = 0
    detail = f'OI ${oi_current/1e9:.1f}B, {change_from_peak:+.1f}% from 90d peak, 7d {oi_7d_slope}'

    if change_from_peak < -20:
        if oi_7d_slope in ('flat', 'rising'):
            score = 13
            detail += ' | Flushed + stabilizing → BOTTOM'
        else:
            score = 6
            detail += ' | Flush in progress'
    elif change_from_peak > -5:
        if funding_7d_avg > 0:
            score = -13
            detail += f' | Near peak + positive funding ({funding_7d_avg:.4%}) → TOP'
        else:
            score = -6
            detail += f' | Near peak but shorts crowded ({funding_7d_avg:.4%})'

    return score, detail, history
```

**Step 3: Implement _compute_score_history()**

```python
def _compute_score_history(conn, ticker, days=30):
    """Recompute macro score for each of the last 30 days for charting."""
    c = conn.cursor()
    cfg = TICKER_CONFIG.get(ticker, TICKER_CONFIG['IBIT'])
    btc_per_share = cfg['per_share']

    dates = c.execute(
        'SELECT DISTINCT date FROM snapshots WHERE ticker=? ORDER BY date DESC LIMIT ?',
        (ticker, days)
    ).fetchall()

    score_history = []
    for (date_str,) in reversed(dates):
        # Simplified daily score: just use snapshot regime + whatever cache exists
        snap = c.execute(
            'SELECT regime, net_gex FROM snapshots WHERE ticker=? AND date=?',
            (ticker, date_str)
        ).fetchone()

        if not snap:
            continue

        # Count regime persistence up to this date
        prior = c.execute(
            'SELECT regime FROM snapshots WHERE ticker=? AND date<=? ORDER BY date DESC LIMIT ?',
            (ticker, date_str, days)
        ).fetchall()

        neg_count = sum(1 for r in prior if r[0] == 'negative_gamma')
        total = len(prior)
        neg_pct = neg_count / total if total else 0
        regime_s = 8 if neg_pct > 0.7 else (-8 if neg_pct < 0.3 else 0)

        # ETF flow for this date
        flow = c.execute(
            '''SELECT COALESCE(total_btc_etf_flow, daily_flow_dollars)
               FROM etf_flows WHERE ticker=? AND date<=? ORDER BY date DESC LIMIT 10''',
            (ticker, date_str)
        ).fetchall()
        flow_10d = sum(r[0] or 0 for r in flow)
        flow_s = 8 if flow_10d > 100e6 else (-8 if flow_10d < -100e6 else int(flow_10d / 100e6 * 8))
        flow_s = max(-12, min(12, flow_s))

        # Funding + OI (if available)
        fr = c.execute(
            'SELECT value FROM coinglass_data WHERE symbol=? AND metric=? AND date<=? ORDER BY date DESC LIMIT 7',
            ('BTC', 'avg_funding_rate', date_str)
        ).fetchall()
        funding_s = 0
        if len(fr) >= 7:
            avg = sum(r[0] for r in fr) / len(fr)
            if avg < -0.0001: funding_s = 8
            elif avg > 0.0003: funding_s = -8

        day_total = regime_s + flow_s + funding_s
        day_total = max(-100, min(100, day_total))

        score_history.append({
            'date': date_str,
            'total_score': day_total,
            'regime': regime_s,
            'flow': flow_s,
            'funding': funding_s,
        })

    return score_history
```

**Step 2: Verify Phase 2 helpers return 0 without Coinglass data**

```bash
cd /home/faisal/work/ibit && python -c "
import app
app.init_db()
conn = app.get_db()
s, d, h = app._compute_funding_signal(conn.cursor())
print(f'Funding: {s}, {d}')
s, d, h = app._compute_oi_signal(conn.cursor())
print(f'OI: {s}, {d}')
conn.close()
"
```
Expected: Both return 0 with "Insufficient data" messages.

**Step 3: Commit**

```bash
git add app.py
git commit -m "Add Phase 2 signal helpers and score history computation"
```

---

### Task 5: API Endpoint + Background Refresh Integration

**Files:**
- Modify: `app.py` (routes section ~L2551, _bg_refresh ~L2415, run_analysis ~L3267)

**Step 1: Add /api/macro-regime endpoint**

Add near other API routes (after `/api/accuracy`):

```python
@app.route('/api/macro-regime')
def api_macro_regime():
    ticker = request.args.get('ticker', 'IBIT').upper()
    conn = get_db()
    result = compute_macro_regime(conn, ticker)
    conn.close()
    return Response(json.dumps(result, cls=NumpyEncoder), mimetype='application/json')
```

**Step 2: Add /macro route**

```python
@app.route('/macro')
def macro_page():
    return render_template('macro.html')
```

**Step 3: Add Coinglass fetch to _bg_refresh()**

In `_bg_refresh()`, right after the `fetch_farside_flows()` call (around L2420), add:

```python
# Phase 0b: Fetch Coinglass data (funding rates, aggregate OI)
try:
    fetch_coinglass_data()
except Exception as e:
    log.error(f"[bg-refresh] Coinglass data fetch error: {e}")
```

**Step 4: Integrate macro regime into run_analysis()**

In `run_analysis()`, after the history_trends computation (around where summaries are built), add:

```python
# Macro regime score
try:
    macro_conn = get_db()
    macro = compute_macro_regime(macro_conn, ticker)
    macro_conn.close()
    summaries['_macro_regime'] = {
        'score': macro['score'],
        'bias': macro['bias'],
        'swing_signal': macro['swing_signal'],
        'high_conviction': macro['high_conviction'],
        'signals': macro['signals'],
        'structural_entry': macro.get('structural_entry'),
        'invalidation': macro.get('invalidation'),
        'coinglass_available': macro['coinglass_available'],
    }
except Exception as e:
    log.warning(f"[analysis] Macro regime computation failed: {e}")
```

Also add to the system prompt (in the cross-timeframe section):

```
MACRO REGIME SCORE (when present):
_macro_regime contains the swing-trade regime score from -100 (topping) to +100 (bottoming).

Score interpretation:
- Score > 50 (SWING LONG): Bottoming conditions. Bias toward long setups at structural support.
- Score < -50 (SWING SHORT): Topping conditions. Bias toward short setups at structural resistance.
- Score -50 to +50 (NEUTRAL): No macro edge. Stick to tactical range-trading.
- |Score| > 75 (HIGH CONVICTION): Strong directional bias. Mention structural_entry and invalidation.

Key combinations:
- regime_persistence + funding_rate agree → strongest signal
- wall_migration disagrees with regime_persistence → flag conflict
- aggregate_oi flush + negative funding → textbook capitulation
- aggregate_oi at peak + positive funding → textbook crowded top

Do NOT override tactical analysis with macro score. A macro bottom doesn't mean "buy today" — it means "the structural floor is forming, look for tactical entry near that floor."
When swing_signal is true, include structural_entry and invalidation in TRADE PLAN as a SWING SETUP.
```

**Step 5: Verify endpoint works**

```bash
cd /home/faisal/work/ibit && python app.py --port 5099 &
sleep 3
curl -s http://localhost:5099/api/macro-regime?ticker=IBIT | python -m json.tool | head -30
kill %1
```

**Step 6: Commit**

```bash
git add app.py
git commit -m "Add /api/macro-regime endpoint, /macro route, bg-refresh + AI integration"
```

---

### Task 6: Macro Status Bar on Main Dashboard

**Files:**
- Modify: `templates/index.html` (between header L302 and grid L304)

**Step 1: Add CSS for macro bar**

In the `<style>` section (around L250), add:

```css
.macro-bar {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 12px;
  height: 28px;
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  font-family: var(--mono);
  font-size: 10px;
  color: var(--t2);
}
.macro-bar .macro-score {
  font-size: 16px;
  font-weight: 700;
  font-family: var(--mono);
}
.macro-bar .macro-badge {
  padding: 1px 6px;
  border-radius: 3px;
  font-weight: 600;
  font-size: 10px;
  text-transform: uppercase;
}
.macro-bar .macro-badge.swing-long { background: var(--green-bg); color: var(--green); border: 1px solid var(--green-bd); }
.macro-bar .macro-badge.swing-short { background: var(--red-bg); color: var(--red); border: 1px solid var(--red-bd); }
.macro-bar .macro-badge.neutral { background: var(--bg3); color: var(--t3); border: 1px solid var(--border); }
.macro-bar .macro-badge.high-conv { background: var(--green); color: var(--bg); }
.macro-bar .macro-badge.high-conv-short { background: var(--red); color: #fff; }
.macro-bar .pill {
  padding: 1px 4px;
  border-radius: 2px;
  font-size: 9px;
  font-family: var(--mono);
}
.macro-bar .pill.pos { color: var(--green); background: var(--green-bg); }
.macro-bar .pill.neg { color: var(--red); background: var(--red-bg); }
.macro-bar .pill.zero { color: var(--t3); }
.macro-bar .pill.dim { color: var(--t3); opacity: 0.5; }
.macro-bar .macro-link {
  margin-left: auto;
  color: var(--blue);
  text-decoration: none;
  font-size: 10px;
  font-weight: 600;
}
.macro-bar .macro-link:hover { text-decoration: underline; }
```

**Step 2: Add HTML between header and grid**

After `</div>` closing the header (L302), before `<div class="grid">` (L304):

```html
<div class="macro-bar" id="macro-bar" style="display:none">
  <span style="color:var(--t3)">MACRO</span>
  <span class="macro-score" id="macro-score">0</span>
  <span class="macro-badge neutral" id="macro-badge">NEUTRAL</span>
  <span class="hdr-sep"></span>
  <span id="macro-pills"></span>
  <a href="/macro" class="macro-link">OPEN MACRO →</a>
</div>
```

**Step 3: Update grid height**

Change the grid height from `calc(100vh - 36px)` to `calc(100vh - 64px)` to account for the macro bar (36px header + 28px macro bar = 64px).

**Step 4: Add JavaScript to fetch and render macro bar**

In the JS section, add a `loadMacro()` function and call it from `loadData()`:

```javascript
async function loadMacro() {
  try {
    const r = await fetch(`/api/macro-regime?ticker=${currentTicker}`);
    const m = await r.json();
    const bar = document.getElementById('macro-bar');
    bar.style.display = 'flex';

    const scoreEl = document.getElementById('macro-score');
    scoreEl.textContent = m.score;
    scoreEl.style.color = m.score > 30 ? 'var(--green)' : m.score < -30 ? 'var(--red)' : 'var(--t2)';

    const badge = document.getElementById('macro-badge');
    badge.className = 'macro-badge';
    if (m.high_conviction && m.score > 0) {
      badge.classList.add('high-conv');
      badge.textContent = 'HIGH CONVICTION SWING LONG';
    } else if (m.high_conviction && m.score < 0) {
      badge.classList.add('high-conv-short');
      badge.textContent = 'HIGH CONVICTION SWING SHORT';
    } else if (m.swing_signal && m.score > 0) {
      badge.classList.add('swing-long');
      badge.textContent = 'SWING LONG';
    } else if (m.swing_signal && m.score < 0) {
      badge.classList.add('swing-short');
      badge.textContent = 'SWING SHORT';
    } else {
      badge.classList.add('neutral');
      badge.textContent = m.bias;
    }

    // Render pills
    const abbr = {
      regime_persistence: 'REG', wall_migration: 'WALL', range_compression: 'RNG',
      etf_flow_momentum: 'FLOW', venue_convergence: 'VEN',
      funding_rate: 'FR', aggregate_oi: 'OI', liquidation: 'LIQ'
    };
    const pills = document.getElementById('macro-pills');
    pills.innerHTML = Object.entries(m.signals).map(([k, v]) => {
      const cls = !m.coinglass_available && v.source === 'coinglass' ? 'dim'
        : v.score > 0 ? 'pos' : v.score < 0 ? 'neg' : 'zero';
      const display = !m.coinglass_available && v.source === 'coinglass'
        ? `${abbr[k]} —` : `${abbr[k]} ${v.score > 0 ? '+' : ''}${v.score}`;
      const title = !m.coinglass_available && v.source === 'coinglass'
        ? 'Add COINGLASS_API_KEY to .env' : v.detail;
      return `<span class="pill ${cls}" title="${title}">${display}</span>`;
    }).join(' ');
  } catch (e) {
    console.warn('Macro load failed:', e);
  }
}
```

Add `loadMacro()` call at end of `loadData()` and in the auto-refresh interval.

**Step 5: Verify in browser**

```bash
cd /home/faisal/work/ibit && python app.py --port 5099 &
```
Open http://localhost:5099 — macro bar should appear between header and grid.

**Step 6: Commit**

```bash
git add templates/index.html
git commit -m "Add macro regime status bar to main dashboard"
```

---

### Task 7: Dedicated Macro Page

**Files:**
- Create: `templates/macro.html`

**Step 1: Create the full macro.html template**

This is a complete new page (~500-600 lines) with:
- Same CSS variables as index.html
- Header with back link, title, ticker selector, large score display
- Score gauge (horizontal bar -100 to +100)
- Signal breakdown table (8 rows)
- 6 charts (2×3 grid): Score Over Time, Regime History, Wall Migration, ETF Flows, Funding Rate, Aggregate OI
- Swing Setup Box (conditionally visible)
- Auto-refresh every 5 minutes
- Chart.js loaded from CDN
- All data from single `/api/macro-regime` fetch

Key design decisions:
- Use same `:root` CSS variables from index.html for consistent dark theme
- Load Chart.js 4.4.1 from cdnjs (same as main dashboard)
- Responsive grid: 2 columns on wide, 1 column on narrow
- Phase 2 charts show "Add COINGLASS_API_KEY to enable" placeholder when unavailable
- Score gauge uses CSS gradient background with positioned marker

**Step 2: Verify page loads**

```bash
curl -s http://localhost:5099/macro | head -20
```

**Step 3: Commit**

```bash
git add templates/macro.html
git commit -m "Add dedicated /macro page with gauge, signals table, and 6 charts"
```

---

### Task 8: Final Verification + Squash Commit

**Step 1: Run full verification checklist**

1. `python app.py` starts without errors (no COINGLASS_API_KEY)
2. `/api/macro-regime` returns valid JSON
3. Phase 2 signals return 0, `coinglass_available` is False
4. Main dashboard shows macro status bar with pills
5. `/macro` page loads with all components
6. Phase 2 chart placeholders show when Coinglass unavailable
7. Cold start (< 30 days data) → score 0, bias "NEUTRAL"
8. Auto-refresh works on both pages

**Step 2: Final commit with full message**

```bash
git add -A
git commit -m "Add macro regime scoring system (-100 to +100)

Swing-trade regime detector using 8 signals:

Phase 1 (existing data, ±12 each):
- Regime persistence & transition
- Structural wall migration (31-45d slopes)
- Range compression + regime context
- ETF flow momentum (10d vs 30d)
- Venue wall convergence

Phase 2 (Coinglass API, ±13 each):
- OI-weighted aggregate funding rate (all exchanges)
- Aggregate futures open interest
- Liquidation intensity (stub ±1, needs calibration)

Includes:
- /api/macro-regime endpoint
- Macro status bar on main dashboard
- Dedicated /macro page with gauge, signal table, 6 charts
- AI analysis integration
- Graceful degradation without Coinglass key"
```

**Step 3: Push branch**

```bash
git push -u origin macro-regime
```
