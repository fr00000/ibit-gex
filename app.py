#!/usr/bin/env python3
"""
IBIT GEX Web Dashboard
======================
Flask app that calculates Gamma Exposure (GEX) from IBIT options chains
and serves an interactive trading dashboard with BTC candlestick chart,
GEX/OI profiles, and regime-adjusted level overlays.

Usage:
  python3 app.py                    # default: 7 DTE, port 5000
  python3 app.py --dte 14           # 14-day expiration window
  python3 app.py --host 0.0.0.0     # accessible from WSL2 host
"""

import json
import os
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from flask import Flask, render_template, Response, request

app = Flask(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ── CONFIG ──────────────────────────────────────────────────────────────────
RISK_FREE_RATE = 0.043
BTC_PER_SHARE = 0.000568
STRIKE_RANGE_PCT = 0.35
DB_PATH = os.path.join(str(Path.home()), ".ibit_gex_history.db")


# ── BLACK-SCHOLES ───────────────────────────────────────────────────────────
def bs_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def bs_delta(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


# ── DATABASE ────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL, ticker TEXT NOT NULL,
        spot REAL, btc_price REAL, gamma_flip REAL,
        call_wall REAL, put_wall REAL, max_pain REAL,
        regime TEXT, net_gex REAL,
        total_call_oi INTEGER, total_put_oi INTEGER,
        UNIQUE(date, ticker)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS strike_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL, ticker TEXT NOT NULL,
        strike REAL NOT NULL, call_oi INTEGER,
        put_oi INTEGER, total_oi INTEGER, net_gex REAL,
        UNIQUE(date, ticker, strike)
    )''')
    conn.commit()
    return conn


def get_prev_strikes(conn, ticker):
    c = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    c.execute('SELECT date FROM snapshots WHERE ticker=? AND date<? ORDER BY date DESC LIMIT 1',
              (ticker, today))
    row = c.fetchone()
    if not row:
        return None, {}
    prev_date = row[0]
    c.execute('SELECT strike, call_oi, put_oi, total_oi, net_gex FROM strike_history WHERE date=? AND ticker=?',
              (prev_date, ticker))
    strikes = {}
    for r in c.fetchall():
        strikes[r[0]] = {'call_oi': r[1], 'put_oi': r[2], 'total_oi': r[3], 'net_gex': r[4]}
    return prev_date, strikes


def save_snapshot(conn, ticker, spot, btc_price, levels, df):
    date_str = datetime.now().strftime('%Y-%m-%d')
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO snapshots
        (date,ticker,spot,btc_price,gamma_flip,call_wall,put_wall,max_pain,regime,net_gex,total_call_oi,total_put_oi)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
        (date_str, ticker, spot, btc_price, levels.get('gamma_flip'), levels.get('call_wall'),
         levels.get('put_wall'), levels.get('max_pain'), levels.get('regime'),
         levels.get('net_gex_total'), levels.get('total_call_oi'), levels.get('total_put_oi')))
    for _, row in df.iterrows():
        c.execute('''INSERT OR REPLACE INTO strike_history (date,ticker,strike,call_oi,put_oi,total_oi,net_gex)
            VALUES (?,?,?,?,?,?,?)''',
            (date_str, ticker, row['strike'], int(row['call_oi']), int(row['put_oi']),
             int(row['total_oi']), row['net_gex']))
    conn.commit()


def get_history(conn, ticker, days=10):
    c = conn.cursor()
    c.execute('''SELECT date, spot, btc_price, gamma_flip, call_wall, put_wall,
                        max_pain, regime, net_gex, total_call_oi, total_put_oi
                 FROM snapshots WHERE ticker=? ORDER BY date DESC LIMIT ?''', (ticker, days))
    return c.fetchall()


# ── DATA ────────────────────────────────────────────────────────────────────
def fetch_and_analyze(ticker_symbol='IBIT', max_dte=7):
    global BTC_PER_SHARE

    ticker = yf.Ticker(ticker_symbol)
    spot = ticker.info.get('regularMarketPrice')
    if spot is None:
        hist = ticker.history(period="1d")
        spot = float(hist['Close'].iloc[-1])

    is_btc = ticker_symbol in ('IBIT', 'BITO', 'GBTC', 'FBTC')

    # Auto BTC/Share
    if is_btc:
        try:
            btc_price = yf.Ticker("BTC-USD").info.get('regularMarketPrice')
            if btc_price and btc_price > 0:
                BTC_PER_SHARE = spot / btc_price
        except:
            pass

    btc_spot = spot / BTC_PER_SHARE if is_btc else None

    all_exps = list(ticker.options)
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=max_dte)
    selected_exps = [e for e in all_exps if datetime.strptime(e, "%Y-%m-%d").replace(tzinfo=timezone.utc) <= cutoff]
    if not selected_exps:
        selected_exps = all_exps[:3]

    # Collect options data
    strike_data = {}
    for exp_str in selected_exps:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        T = max((exp_date - now).days / 365.0, 0.5 / 365)
        try:
            chain = ticker.option_chain(exp_str)
        except:
            continue

        for opt_type, df_chain, sign in [('call', chain.calls, 1), ('put', chain.puts, -1)]:
            for _, row in df_chain.iterrows():
                strike = row['strike']
                if strike < spot * (1 - STRIKE_RANGE_PCT) or strike > spot * (1 + STRIKE_RANGE_PCT):
                    continue
                oi = row.get('openInterest', 0)
                if pd.isna(oi) or oi == 0:
                    continue
                oi = int(oi)
                iv = row.get('impliedVolatility', 0)
                if pd.isna(iv) or iv <= 0:
                    continue

                gamma = bs_gamma(spot, strike, T, RISK_FREE_RATE, iv)
                delta = bs_delta(spot, strike, T, RISK_FREE_RATE, iv, opt_type)
                gex = sign * gamma * oi * 100 * spot ** 2 * 0.01
                dealer_delta = -delta * oi * 100

                if strike not in strike_data:
                    strike_data[strike] = {'call_oi': 0, 'put_oi': 0, 'call_gex': 0, 'put_gex': 0,
                                           'call_delta': 0, 'put_delta': 0}
                strike_data[strike][f'{opt_type}_oi'] += oi
                strike_data[strike][f'{opt_type}_gex'] += gex
                strike_data[strike][f'{opt_type}_delta'] += dealer_delta

    # Build dataframe
    rows = []
    for strike, d in sorted(strike_data.items()):
        rows.append({
            'strike': strike,
            'btc_price': strike / BTC_PER_SHARE if is_btc else strike,
            'call_oi': d['call_oi'], 'put_oi': d['put_oi'],
            'total_oi': d['call_oi'] + d['put_oi'],
            'call_gex': d['call_gex'], 'put_gex': d['put_gex'],
            'net_gex': d['call_gex'] + d['put_gex'],
            'net_dealer_delta': d['call_delta'] + d['put_delta'],
        })
    df = pd.DataFrame(rows)

    # Derive levels
    levels = {}
    levels['call_wall'] = float(df.loc[df['call_gex'].idxmax(), 'strike'])
    levels['put_wall'] = float(df.loc[df['put_gex'].idxmin(), 'strike'])

    # Gamma flip nearest to spot
    df_s = df.sort_values('strike')
    all_flips = []
    for i in range(len(df_s) - 1):
        g1, g2 = df_s.iloc[i]['net_gex'], df_s.iloc[i + 1]['net_gex']
        s1, s2 = df_s.iloc[i]['strike'], df_s.iloc[i + 1]['strike']
        if (g1 < 0 and g2 > 0) or (g1 > 0 and g2 < 0):
            if (g2 - g1) != 0:
                flip = s1 + (s2 - s1) * (-g1) / (g2 - g1)
                all_flips.append(flip)
    levels['gamma_flip'] = float(min(all_flips, key=lambda x: abs(x - spot))) if all_flips else float(spot)

    # Max pain
    def calc_payout(settle):
        return sum(row['call_oi'] * max(0, settle - row['strike']) * 100 +
                   row['put_oi'] * max(0, row['strike'] - settle) * 100
                   for _, row in df.iterrows())
    pain = {s: calc_payout(s) for s in df['strike'].values}
    levels['max_pain'] = float(min(pain, key=pain.get))

    # Regime
    near_spot = df[(df['strike'] >= spot * 0.98) & (df['strike'] <= spot * 1.02)]
    local_gex = near_spot['net_gex'].sum() if not near_spot.empty else 0
    levels['regime'] = 'positive_gamma' if local_gex > 0 else 'negative_gamma'

    # Totals
    levels['net_gex_total'] = float(df['net_gex'].sum())
    levels['net_dealer_delta'] = float(df['net_dealer_delta'].sum())
    levels['total_call_oi'] = int(df['call_oi'].sum())
    levels['total_put_oi'] = int(df['put_oi'].sum())
    levels['pcr'] = levels['total_put_oi'] / max(levels['total_call_oi'], 1)

    # Resistance / support
    levels['resistance'] = df[df['net_gex'] > 0].nlargest(3, 'net_gex')['strike'].tolist()
    levels['support'] = df[df['net_gex'] < 0].nsmallest(3, 'net_gex')['strike'].tolist()

    # OI magnets
    levels['oi_magnets'] = df.nlargest(5, 'total_oi')[['strike', 'total_oi', 'call_oi', 'put_oi']].to_dict('records')

    # Expected move
    expected_move = None
    try:
        nearest_exp = selected_exps[0]
        ch = ticker.option_chain(nearest_exp)
        atm_c = ch.calls.iloc[(ch.calls['strike'] - spot).abs().argsort()[:1]]
        atm_p = ch.puts.iloc[(ch.puts['strike'] - spot).abs().argsort()[:1]]
        straddle = (atm_c['bid'].values[0] + atm_c['ask'].values[0]) / 2 + \
                   (atm_p['bid'].values[0] + atm_p['ask'].values[0]) / 2
        exp_date = datetime.strptime(nearest_exp, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        dte = max((exp_date - now).days, 1)
        expected_move = {
            'straddle': float(straddle), 'pct': float((straddle / spot) * 100),
            'upper': float(spot + straddle), 'lower': float(spot - straddle),
            'upper_btc': float((spot + straddle) / BTC_PER_SHARE) if is_btc else None,
            'lower_btc': float((spot - straddle) / BTC_PER_SHARE) if is_btc else None,
            'expiration': nearest_exp, 'dte': dte,
        }
    except:
        pass

    # Breakout signals
    breakout = compute_breakout(df, spot, levels, expected_move, None)

    # Significant levels with regime behavior
    sig_levels = compute_significant_levels(df, spot, levels, None, is_btc)

    # Save to DB
    conn = init_db()
    prev_date, prev_strikes = get_prev_strikes(conn, ticker_symbol)
    save_snapshot(conn, ticker_symbol, spot, btc_spot, levels, df)

    # Recompute with prev data
    if prev_strikes:
        breakout = compute_breakout(df, spot, levels, expected_move, prev_strikes)
        sig_levels = compute_significant_levels(df, spot, levels, prev_strikes, is_btc)

    # OI aggregate changes
    oi_changes = None
    if prev_strikes:
        total_oi_prev = sum(s['total_oi'] for s in prev_strikes.values())
        total_oi_now = int(df['total_oi'].sum())
        call_oi_prev = sum(s['call_oi'] for s in prev_strikes.values())
        put_oi_prev = sum(s['put_oi'] for s in prev_strikes.values())
        oi_changes = {
            'prev_date': prev_date,
            'total_delta': total_oi_now - total_oi_prev,
            'total_pct': ((total_oi_now - total_oi_prev) / max(total_oi_prev, 1)) * 100,
            'call_delta': int(df['call_oi'].sum()) - call_oi_prev,
            'put_delta': int(df['put_oi'].sum()) - put_oi_prev,
        }

    history = get_history(conn, ticker_symbol, 10)
    conn.close()

    # Build response
    gex_chart_data = []
    for _, row in df[(df['strike'] >= spot * 0.82) & (df['strike'] <= spot * 1.22)].iterrows():
        gex_chart_data.append({
            'strike': float(row['strike']),
            'btc': float(row['strike'] / BTC_PER_SHARE) if is_btc else float(row['strike']),
            'net_gex': float(row['net_gex']),
            'call_oi': int(row['call_oi']),
            'put_oi': int(row['put_oi']),
            'total_oi': int(row['total_oi']),
        })

    oi_chart_data = gex_chart_data  # same data, different rendering

    history_data = []
    for h in history:
        history_data.append({
            'date': h[0], 'spot': h[1], 'btc_price': h[2],
            'gamma_flip': h[3], 'call_wall': h[4], 'put_wall': h[5],
            'max_pain': h[6], 'regime': h[7], 'net_gex': h[8],
            'total_call_oi': h[9], 'total_put_oi': h[10],
        })

    return {
        'spot': float(spot),
        'btc_spot': float(btc_spot) if btc_spot else None,
        'btc_per_share': float(BTC_PER_SHARE),
        'is_btc': bool(is_btc),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'levels': {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                   for k, v in levels.items()},
        'expected_move': expected_move,
        'gex_chart': gex_chart_data,
        'oi_chart': oi_chart_data,
        'significant_levels': sig_levels,
        'breakout': breakout,
        'oi_changes': oi_changes,
        'history': history_data,
        'expirations': selected_exps,
    }


def compute_significant_levels(df, spot, levels, prev_strikes, is_btc):
    """Build list of significant levels with regime-adjusted behavior and OI deltas."""
    oi_90 = df['total_oi'].quantile(0.90)
    oi_75 = df['total_oi'].quantile(0.75)
    regime = levels['regime']
    result = []

    for _, row in df.iterrows():
        strike = row['strike']
        if row['total_oi'] < oi_75:
            continue
        dist_pct = abs((strike - spot) / spot) * 100
        if dist_pct > 25:
            continue

        net_gex = row['net_gex']
        call_oi, put_oi, total_oi = int(row['call_oi']), int(row['put_oi']), int(row['total_oi'])
        is_major = total_oi > oi_90

        if put_oi > call_oi * 1.5 and net_gex < 0:
            ltype = 'put_wall'
        elif call_oi > put_oi * 1.5 and net_gex > 0:
            ltype = 'call_wall'
        elif total_oi > oi_90:
            ltype = 'oi_magnet'
        else:
            continue

        # Regime-adjusted behavior
        if regime == 'negative_gamma':
            if ltype == 'put_wall':
                behavior = "REACTION — scalp long, not conviction" if is_major else "minor support — dealers sell into bounce"
            elif ltype == 'call_wall':
                behavior = "REACTION — brief cap, watch for breakout" if is_major else "minor resistance"
            else:
                behavior = "OI magnet — gravitational pull near expiry"
        else:
            if ltype == 'put_wall':
                behavior = "HARD FLOOR — lever long with confidence" if is_major else "support — dealers buy into weakness"
            elif ltype == 'call_wall':
                behavior = "HARD CEILING — lever short with confidence" if is_major else "resistance — dealers sell into strength"
            else:
                behavior = "OI magnet — gravitational pull near expiry"

        # OI delta
        oi_delta = None
        if prev_strikes and strike in prev_strikes:
            prev_oi = prev_strikes[strike]['total_oi']
            delta = total_oi - prev_oi
            if delta != 0:
                pct_chg = (delta / max(prev_oi, 1)) * 100
                oi_delta = {'delta': delta, 'pct': pct_chg,
                            'status': 'BUILDING' if delta > 0 and abs(pct_chg) > 10 else
                                      'DECAYING' if delta < 0 and abs(pct_chg) > 10 else 'stable'}

        result.append({
            'strike': float(strike),
            'btc': float(strike / BTC_PER_SHARE) if is_btc else float(strike),
            'type': ltype,
            'call_oi': call_oi, 'put_oi': put_oi, 'total_oi': total_oi,
            'net_gex': float(net_gex),
            'dist_pct': float(((strike - spot) / spot) * 100),
            'is_major': is_major,
            'behavior': behavior,
            'oi_delta': oi_delta,
        })

    result.sort(key=lambda x: x['strike'])
    return result


def compute_breakout(df, spot, levels, expected_move, prev_strikes):
    """Compute breakout signals for both directions."""
    cw = levels['call_wall']
    pw = levels['put_wall']
    regime = levels['regime']

    up_signals, down_signals = [], []

    # Wall asymmetry
    cw_gex = float(df[df['strike'] == cw]['call_gex'].sum()) if cw in df['strike'].values else 0
    pw_gex = float(abs(df[df['strike'] == pw]['put_gex'].sum())) if pw in df['strike'].values else 0
    if pw_gex > 0 and cw_gex > 0:
        ratio = cw_gex / pw_gex
        if ratio < 0.5:
            up_signals.append(f"Weak ceiling: call wall GEX is {ratio:.1f}x put wall")
        elif ratio > 2.0:
            down_signals.append(f"Weak floor: put wall GEX is {1/ratio:.1f}x call wall")

    # Wall decay
    if prev_strikes:
        if cw in prev_strikes:
            prev = prev_strikes[cw]['total_oi']
            curr = int(df[df['strike'] == cw]['total_oi'].sum()) if cw in df['strike'].values else 0
            if prev > 0:
                chg = ((curr - prev) / prev) * 100
                if chg < -10:
                    up_signals.append(f"Call wall DECAYING: OI down {chg:.0f}%")
                elif chg > 15:
                    down_signals.append(f"Call wall BUILDING: OI up +{chg:.0f}%")
        if pw in prev_strikes:
            prev = prev_strikes[pw]['total_oi']
            curr = int(df[df['strike'] == pw]['total_oi'].sum()) if pw in df['strike'].values else 0
            if prev > 0:
                chg = ((curr - prev) / prev) * 100
                if chg < -10:
                    down_signals.append(f"Put wall DECAYING: OI down {chg:.0f}%")
                elif chg > 15:
                    up_signals.append(f"Put wall BUILDING: OI up +{chg:.0f}%")

    # Expected move > range
    if expected_move:
        em_width = expected_move['pct'] * 2
        range_width = ((cw - pw) / spot) * 100
        if em_width > range_width:
            up_signals.append(f"Expected move ({em_width:.1f}%) > range ({range_width:.1f}%)")
            down_signals.append(f"Expected move ({em_width:.1f}%) > range ({range_width:.1f}%)")

    # Neg gamma near wall
    if regime == 'negative_gamma':
        if ((cw - spot) / spot) * 100 < 5:
            up_signals.append("Negative gamma near call wall — gamma squeeze potential")
        if ((spot - pw) / spot) * 100 < 5:
            down_signals.append("Negative gamma near put wall — waterfall risk")

    # OI beyond walls
    total_call = df['call_oi'].sum()
    total_put = df['put_oi'].sum()
    above = df[df['strike'] > cw]['call_oi'].sum()
    below = df[df['strike'] < pw]['put_oi'].sum()
    if total_call > 0 and (above / total_call) * 100 > 40:
        up_signals.append(f"{(above/total_call)*100:.0f}% of call OI above call wall")
    if total_put > 0 and (below / total_put) * 100 > 40:
        down_signals.append(f"{(below/total_put)*100:.0f}% of put OI below put wall")

    # P/C ratio
    pcr = levels.get('pcr', 1.0)
    if pcr > 1.5:
        up_signals.append(f"P/C ratio {pcr:.2f} — short squeeze fuel")
        down_signals.append(f"P/C ratio {pcr:.2f} — heavy put skew")
    elif pcr < 0.6:
        up_signals.append(f"P/C ratio {pcr:.2f} — aggressive call skew")

    # Targets
    up_targets = df[(df['strike'] > cw) & (df['net_gex'] > 0)].nlargest(2, 'net_gex')[
        ['strike', 'total_oi', 'net_gex']].to_dict('records')
    down_targets = df[(df['strike'] < pw) & (df['net_gex'] < 0)].nsmallest(2, 'net_gex')[
        ['strike', 'total_oi', 'net_gex']].to_dict('records')

    up_score = len(up_signals)
    down_score = len(down_signals)
    if up_score > down_score + 1:
        bias = 'upside'
    elif down_score > up_score + 1:
        bias = 'downside'
    else:
        bias = 'balanced'

    return {
        'up_signals': up_signals, 'down_signals': down_signals,
        'up_score': up_score, 'down_score': down_score,
        'up_targets': [{'strike': float(t['strike']),
                        'btc': float(t['strike'] / BTC_PER_SHARE),
                        'total_oi': int(t['total_oi']),
                        'net_gex': float(t['net_gex'])} for t in up_targets],
        'down_targets': [{'strike': float(t['strike']),
                          'btc': float(t['strike'] / BTC_PER_SHARE),
                          'total_oi': int(t['total_oi']),
                          'net_gex': float(t['net_gex'])} for t in down_targets],
        'bias': bias,
    }


# ── ROUTES ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/data')
def api_data():
    try:
        dte = request.args.get('dte', app.config.get('MAX_DTE', 7), type=int)
        dte = max(1, min(dte, 90))
        data = fetch_and_analyze('IBIT', dte)
        return Response(json.dumps(data, cls=NumpyEncoder), mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({'error': str(e)}), mimetype='application/json'), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', '-p', type=int, default=5000)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--dte', '-d', type=int, default=7, help='Max days to expiration (default: 7)')
    args = parser.parse_args()
    app.config['MAX_DTE'] = args.dte
    print(f"\n  IBIT GEX Dashboard → http://{args.host}:{args.port}  (DTE: {args.dte})\n")
    app.run(host=args.host, port=args.port, debug=True)