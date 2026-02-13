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
import threading
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from flask import Flask, render_template, Response, request
import anthropic
from dotenv import load_dotenv

load_dotenv()

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
BTC_PER_SHARE_DEFAULT = 0.000568
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

def bs_vanna(S, K, T, r, sigma):
    """Vanna = dDelta/dVol — dealer rebalancing when IV moves."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return -norm.pdf(d1) * d2 / sigma

def bs_charm(S, K, T, r, sigma, option_type='call'):
    """Charm = dDelta/dT — dealer rebalancing from time decay of delta."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    charm = -norm.pdf(d1) * (2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
    if option_type == 'put':
        charm += r * np.exp(-r * T) * norm.cdf(-d2)
    return charm


# ── DATABASE ────────────────────────────────────────────────────────────────
def init_db():
    """Create tables once at startup."""
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
    c.execute('''CREATE TABLE IF NOT EXISTS data_cache (
        date TEXT NOT NULL, ticker TEXT NOT NULL,
        dte INTEGER NOT NULL, data_json TEXT NOT NULL,
        UNIQUE(date, ticker, dte)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS analysis_cache (
        date TEXT NOT NULL, ticker TEXT NOT NULL,
        analysis_json TEXT NOT NULL,
        UNIQUE(date, ticker)
    )''')
    conn.commit()
    conn.close()


def get_db():
    """Get a SQLite connection (one per call, thread-safe)."""
    return sqlite3.connect(DB_PATH)


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
    ticker = yf.Ticker(ticker_symbol)
    spot = ticker.info.get('regularMarketPrice')
    if spot is None:
        hist = ticker.history(period="1d")
        spot = float(hist['Close'].iloc[-1])

    is_btc = ticker_symbol in ('IBIT', 'BITO', 'GBTC', 'FBTC')

    # Auto BTC/Share — compute locally to avoid race conditions
    btc_per_share = BTC_PER_SHARE_DEFAULT
    if is_btc:
        try:
            btc_price = yf.Ticker("BTC-USD").info.get('regularMarketPrice')
            if btc_price and btc_price > 0:
                btc_per_share = spot / btc_price
        except Exception:
            pass

    btc_spot = spot / btc_per_share if is_btc else None

    all_exps = list(ticker.options)
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=max_dte)
    selected_exps = [e for e in all_exps if datetime.strptime(e, "%Y-%m-%d").replace(tzinfo=timezone.utc) <= cutoff]
    if not selected_exps:
        selected_exps = all_exps[:3]

    # Collect options data
    strike_data = {}
    cached_chains = {}  # exp_str -> chain, reused for expected move
    for exp_str in selected_exps:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        T = max((exp_date - now).days / 365.0, 0.5 / 365)
        try:
            chain = ticker.option_chain(exp_str)
        except Exception:
            continue
        cached_chains[exp_str] = chain

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
                vanna = bs_vanna(spot, strike, T, RISK_FREE_RATE, iv)
                charm = bs_charm(spot, strike, T, RISK_FREE_RATE, iv, opt_type)
                gex = sign * gamma * oi * 100 * spot ** 2 * 0.01
                dealer_delta = -delta * oi * 100
                # Dealer vanna exposure: how dealer delta changes with IV
                dealer_vanna = -sign * vanna * oi * 100 * spot * 0.01
                # Dealer charm exposure: how dealer delta changes overnight
                # Charm is dDelta/dT per year, convert to per-day: divide by 365
                dealer_charm = -charm * oi * 100 / 365.0

                if strike not in strike_data:
                    strike_data[strike] = {'call_oi': 0, 'put_oi': 0, 'call_gex': 0, 'put_gex': 0,
                                           'call_delta': 0, 'put_delta': 0,
                                           'call_vanna': 0, 'put_vanna': 0,
                                           'call_charm': 0, 'put_charm': 0}
                strike_data[strike][f'{opt_type}_oi'] += oi
                strike_data[strike][f'{opt_type}_gex'] += gex
                strike_data[strike][f'{opt_type}_delta'] += dealer_delta
                strike_data[strike][f'{opt_type}_vanna'] += dealer_vanna
                strike_data[strike][f'{opt_type}_charm'] += dealer_charm

    # Build dataframe
    rows = []
    for strike, d in sorted(strike_data.items()):
        rows.append({
            'strike': strike,
            'btc_price': strike / btc_per_share if is_btc else strike,
            'call_oi': d['call_oi'], 'put_oi': d['put_oi'],
            'total_oi': d['call_oi'] + d['put_oi'],
            'call_gex': d['call_gex'], 'put_gex': d['put_gex'],
            'net_gex': d['call_gex'] + d['put_gex'],
            'net_dealer_delta': d['call_delta'] + d['put_delta'],
            'net_vanna': d['call_vanna'] + d['put_vanna'],
            'net_charm': d['call_charm'] + d['put_charm'],
            'call_vanna': d['call_vanna'], 'put_vanna': d['put_vanna'],
            'call_charm': d['call_charm'], 'put_charm': d['put_charm'],
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
    levels['net_vanna'] = float(df['net_vanna'].sum())
    levels['net_charm'] = float(df['net_charm'].sum())
    levels['total_call_oi'] = int(df['call_oi'].sum())
    levels['total_put_oi'] = int(df['put_oi'].sum())
    levels['pcr'] = levels['total_put_oi'] / max(levels['total_call_oi'], 1)

    # Resistance / support
    levels['resistance'] = df[df['net_gex'] > 0].nlargest(3, 'net_gex')['strike'].tolist()
    levels['support'] = df[df['net_gex'] < 0].nsmallest(3, 'net_gex')['strike'].tolist()

    # OI magnets
    levels['oi_magnets'] = df.nlargest(5, 'total_oi')[['strike', 'total_oi', 'call_oi', 'put_oi']].to_dict('records')

    # Expected move (reuse cached chain instead of re-fetching)
    expected_move = None
    try:
        nearest_exp = selected_exps[0]
        ch = cached_chains[nearest_exp]
        atm_c = ch.calls.iloc[(ch.calls['strike'] - spot).abs().argsort()[:1]]
        atm_p = ch.puts.iloc[(ch.puts['strike'] - spot).abs().argsort()[:1]]
        straddle = (atm_c['bid'].values[0] + atm_c['ask'].values[0]) / 2 + \
                   (atm_p['bid'].values[0] + atm_p['ask'].values[0]) / 2
        exp_date = datetime.strptime(nearest_exp, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        dte = max((exp_date - now).days, 1)
        expected_move = {
            'straddle': float(straddle), 'pct': float((straddle / spot) * 100),
            'upper': float(spot + straddle), 'lower': float(spot - straddle),
            'upper_btc': float((spot + straddle) / btc_per_share) if is_btc else None,
            'lower_btc': float((spot - straddle) / btc_per_share) if is_btc else None,
            'expiration': nearest_exp, 'dte': dte,
        }
    except Exception:
        pass

    # Breakout signals
    breakout = compute_breakout(df, spot, levels, expected_move, None, btc_per_share)

    # Significant levels with regime behavior
    sig_levels = compute_significant_levels(df, spot, levels, None, is_btc, btc_per_share)

    # Dealer flow forecast (vanna + charm)
    flow_forecast = compute_flow_forecast(df, spot, levels, is_btc)

    # Save to DB
    conn = get_db()
    prev_date, prev_strikes = get_prev_strikes(conn, ticker_symbol)
    save_snapshot(conn, ticker_symbol, spot, btc_spot, levels, df)

    # Recompute with prev data
    if prev_strikes:
        breakout = compute_breakout(df, spot, levels, expected_move, prev_strikes, btc_per_share)
        sig_levels = compute_significant_levels(df, spot, levels, prev_strikes, is_btc, btc_per_share)

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
            'btc': float(row['strike'] / btc_per_share) if is_btc else float(row['strike']),
            'net_gex': float(row['net_gex']),
            'net_vanna': float(row['net_vanna']),
            'net_charm': float(row['net_charm']),
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
        'btc_per_share': float(btc_per_share),
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
        'flow_forecast': flow_forecast,
        'history': history_data,
        'expirations': selected_exps,
    }


def compute_flow_forecast(df, spot, levels, is_btc):
    """Compute dealer flow forecasts from vanna and charm exposures."""
    net_vanna = float(df['net_vanna'].sum())
    net_charm = float(df['net_charm'].sum())
    regime = levels['regime']

    # Charm: overnight dealer rebalancing
    # Positive net charm = dealers need to BUY delta tomorrow (bullish flow)
    # Negative net charm = dealers need to SELL delta tomorrow (bearish flow)
    charm_delta_shares = net_charm  # shares dealers need to trade
    charm_delta_notional = charm_delta_shares * spot  # dollar notional (shares × IBIT price)
    charm_direction = 'buy' if net_charm > 0 else 'sell'
    charm_magnitude = abs(net_charm)

    # Categorize charm impact
    if charm_magnitude < 1000:
        charm_strength = 'negligible'
    elif charm_magnitude < 10000:
        charm_strength = 'minor'
    elif charm_magnitude < 50000:
        charm_strength = 'moderate'
    else:
        charm_strength = 'significant'

    # Vanna: IV-dependent dealer rebalancing
    # Positive net vanna = vol CRUSH forces dealers to BUY (bullish)
    # Negative net vanna = vol CRUSH forces dealers to SELL (bearish)
    # (reverse for vol spike)
    vanna_magnitude = abs(net_vanna)
    if vanna_magnitude < 1000:
        vanna_strength = 'negligible'
    elif vanna_magnitude < 10000:
        vanna_strength = 'minor'
    elif vanna_magnitude < 50000:
        vanna_strength = 'moderate'
    else:
        vanna_strength = 'significant'

    # Scenarios: 5-point IV move
    iv_move = 5  # vol points
    vanna_crush_delta = net_vanna * iv_move  # shares from -5pt IV
    vanna_spike_delta = -net_vanna * iv_move  # shares from +5pt IV

    # Combined overnight forecast (charm + expected vanna from typical overnight vol decay)
    # Overnight vol typically drifts down ~0.5-1pt in calm markets
    overnight_vanna_adj = net_vanna * 0.5  # conservative overnight vol decay
    overnight_total = net_charm + overnight_vanna_adj
    overnight_direction = 'buy' if overnight_total > 0 else 'sell'

    # Narrative
    charm_narrative = f"Dealers {charm_direction} ~{abs(charm_delta_shares):,.0f} shares overnight from delta decay"
    if is_btc:
        charm_narrative += f" (~${abs(charm_delta_notional):,.0f} notional)"

    if net_vanna > 0:
        vanna_crush_narrative = "Vol crush → dealers BUY (supportive)"
        vanna_spike_narrative = "Vol spike → dealers SELL (pressuring)"
    else:
        vanna_crush_narrative = "Vol crush → dealers SELL (pressuring)"
        vanna_spike_narrative = "Vol spike → dealers BUY (supportive)"

    # Regime interaction
    if regime == 'positive_gamma':
        regime_note = "Positive gamma reinforces mean-reversion. Charm and vanna flows add directional bias within the range."
    else:
        regime_note = "Negative gamma amplifies moves. Vanna and charm flows can accelerate breakouts."

    return {
        'charm': {
            'net_shares': float(net_charm),
            'direction': charm_direction,
            'strength': charm_strength,
            'notional': float(charm_delta_notional),
            'narrative': charm_narrative,
        },
        'vanna': {
            'net_exposure': float(net_vanna),
            'strength': vanna_strength,
            'crush_scenario': {
                'delta_shares': float(vanna_crush_delta),
                'notional': float(vanna_crush_delta * spot),
                'direction': 'buy' if vanna_crush_delta > 0 else 'sell',
                'narrative': vanna_crush_narrative,
            },
            'spike_scenario': {
                'delta_shares': float(vanna_spike_delta),
                'notional': float(vanna_spike_delta * spot),
                'direction': 'buy' if vanna_spike_delta > 0 else 'sell',
                'narrative': vanna_spike_narrative,
            },
        },
        'overnight': {
            'net_shares': float(overnight_total),
            'direction': overnight_direction,
            'notional': float(overnight_total * spot),
        },
        'regime_note': regime_note,
    }


def _level_greeks_note(ltype, regime, row, is_major):
    """Generate a short note about vanna/charm dynamics at this level."""
    notes = []
    net_vanna = row.get('net_vanna', 0) if isinstance(row, dict) else (row['net_vanna'] if 'net_vanna' in row.index else 0)
    net_charm = row.get('net_charm', 0) if isinstance(row, dict) else (row['net_charm'] if 'net_charm' in row.index else 0)

    # Vanna context
    if abs(net_vanna) > 500:
        if ltype == 'put_wall' and net_vanna > 0:
            notes.append("Vol spike weakens support (vanna sells)")
        elif ltype == 'put_wall' and net_vanna < 0:
            notes.append("Vol spike strengthens support (vanna buys)")
        elif ltype == 'call_wall' and net_vanna > 0:
            notes.append("Vol crush strengthens ceiling (vanna buys into resistance)")
        elif ltype == 'call_wall' and net_vanna < 0:
            notes.append("Vol crush weakens ceiling (vanna sells)")

    # Charm context
    if abs(net_charm) > 500:
        if net_charm > 0:
            notes.append("Charm: dealers buy overnight")
        else:
            notes.append("Charm: dealers sell overnight")

    return ' | '.join(notes) if notes else None


def compute_significant_levels(df, spot, levels, prev_strikes, is_btc, btc_per_share):
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
            'btc': float(strike / btc_per_share) if is_btc else float(strike),
            'type': ltype,
            'call_oi': call_oi, 'put_oi': put_oi, 'total_oi': total_oi,
            'net_gex': float(net_gex),
            'net_vanna': float(row['net_vanna']) if 'net_vanna' in row else 0.0,
            'net_charm': float(row['net_charm']) if 'net_charm' in row else 0.0,
            'dist_pct': float(((strike - spot) / spot) * 100),
            'is_major': is_major,
            'behavior': behavior,
            'greeks_note': _level_greeks_note(ltype, regime, row, is_major),
            'oi_delta': oi_delta,
        })

    result.sort(key=lambda x: x['strike'])
    return result


def compute_breakout(df, spot, levels, expected_move, prev_strikes, btc_per_share):
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
                        'btc': float(t['strike'] / btc_per_share),
                        'total_oi': int(t['total_oi']),
                        'net_gex': float(t['net_gex'])} for t in up_targets],
        'down_targets': [{'strike': float(t['strike']),
                          'btc': float(t['strike'] / btc_per_share),
                          'total_oi': int(t['total_oi']),
                          'net_gex': float(t['net_gex'])} for t in down_targets],
        'bias': bias,
    }


# ── CACHE ──────────────────────────────────────────────────────────────────
# OI updates once per day (after market close, available next morning).
# Strategy: compare OI to detect when Yahoo has fresh data rather than
# relying on a fixed clock cutoff. Throttle Yahoo checks to every 30 min.
YAHOO_CHECK_INTERVAL = 1800  # seconds between re-checks
_last_yahoo_check = {}  # (ticker, dte) -> datetime
_yahoo_check_lock = threading.Lock()


def get_latest_cache(ticker, dte):
    """Return the most recent cached data and its date, or (None, None)."""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT date, data_json FROM data_cache WHERE ticker=? AND dte=? ORDER BY date DESC LIMIT 1',
              (ticker, dte))
    row = c.fetchone()
    conn.close()
    if row:
        return row[0], json.loads(row[1])
    return None, None


def get_prev_cache(ticker, dte):
    """Return the second most recent cached data (yesterday's), or None."""
    conn = get_db()
    c = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    c.execute('SELECT date, data_json FROM data_cache WHERE ticker=? AND dte=? AND date<? ORDER BY date DESC LIMIT 1',
              (ticker, dte, today))
    row = c.fetchone()
    conn.close()
    if row:
        return row[0], json.loads(row[1])
    return None, None


def set_cached_data(ticker, dte, data):
    """Cache the full response JSON keyed to today's date."""
    conn = get_db()
    c = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    c.execute('INSERT OR REPLACE INTO data_cache (date, ticker, dte, data_json) VALUES (?,?,?,?)',
              (today, ticker, dte, json.dumps(data, cls=NumpyEncoder)))
    conn.commit()
    conn.close()


def fetch_with_cache(ticker, dte):
    """Return cached data if fresh, otherwise check Yahoo for new OI."""
    today = datetime.now().strftime('%Y-%m-%d')
    cache_date, cached = get_latest_cache(ticker, dte)

    # Already confirmed today's data
    if cached and cache_date == today:
        return cached

    # Throttle: don't re-check Yahoo more than every 30 min
    check_key = (ticker, dte)
    with _yahoo_check_lock:
        last_check = _last_yahoo_check.get(check_key)
        if cached and last_check and (datetime.now() - last_check).total_seconds() < YAHOO_CHECK_INTERVAL:
            return cached

    # Fetch from Yahoo
    data = fetch_and_analyze(ticker, dte)
    with _yahoo_check_lock:
        _last_yahoo_check[check_key] = datetime.now()

    # Compare OI to detect if data actually changed
    if cached:
        old_oi = (cached['levels']['total_call_oi'], cached['levels']['total_put_oi'])
        new_oi = (data['levels']['total_call_oi'], data['levels']['total_put_oi'])
        if old_oi == new_oi:
            return cached  # Still stale, serve previous cache

    # New data (or first run) — save as today
    set_cached_data(ticker, dte, data)
    return data


# ── BACKGROUND REFRESH ─────────────────────────────────────────────────────

REFRESH_DTES = [3, 7, 14, 30, 45]

def _bg_refresh():
    """Background thread: pre-fetch all DTEs on startup, then re-check
    every 30 min until today's fresh data is confirmed for all DTEs.
    Once all fresh, auto-run AI analysis if not already cached."""
    while True:
        today = datetime.now().strftime('%Y-%m-%d')
        all_fresh = True
        for dte in REFRESH_DTES:
            try:
                cache_date, _ = get_latest_cache('IBIT', dte)
                if cache_date == today:
                    continue  # already fresh
                all_fresh = False
                fetch_with_cache('IBIT', dte)
            except Exception as e:
                print(f"  [bg-refresh] DTE {dte} error: {e}")
                all_fresh = False

        if all_fresh:
            # Auto-run AI analysis if not already cached for today
            if not get_cached_analysis('IBIT'):
                try:
                    print("  [bg-refresh] All DTEs fresh — running AI analysis...")
                    run_analysis('IBIT')
                    print("  [bg-refresh] AI analysis complete and cached")
                except Exception as e:
                    print(f"  [bg-refresh] AI analysis error: {e}")
            time.sleep(3600)
        else:
            time.sleep(YAHOO_CHECK_INTERVAL)


def start_bg_refresh():
    t = threading.Thread(target=_bg_refresh, daemon=True)
    t.start()
    print("  [bg-refresh] Background data refresh started")


# ── ROUTES ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/data')
def api_data():
    try:
        dte = request.args.get('dte', app.config.get('MAX_DTE', 7), type=int)
        dte = max(1, min(dte, 90))
        data = fetch_with_cache('IBIT', dte)
        return Response(json.dumps(data, cls=NumpyEncoder), mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({'error': str(e)}), mimetype='application/json'), 500


# ── AI ANALYSIS ────────────────────────────────────────────────────────────
def get_cached_analysis(ticker):
    """Return cached analysis for today if it exists."""
    conn = get_db()
    c = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    c.execute('SELECT analysis_json FROM analysis_cache WHERE date=? AND ticker=?',
              (today, ticker))
    row = c.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None


def get_prev_analysis(ticker):
    """Return the most recent analysis before today, or None."""
    conn = get_db()
    c = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    c.execute('SELECT date, analysis_json FROM analysis_cache WHERE ticker=? AND date<? ORDER BY date DESC LIMIT 1',
              (ticker, today))
    row = c.fetchone()
    conn.close()
    if row:
        return row[0], json.loads(row[1])
    return None, None


def set_cached_analysis(ticker, analysis):
    """Cache AI analysis for today."""
    conn = get_db()
    c = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    c.execute('INSERT OR REPLACE INTO analysis_cache (date, ticker, analysis_json) VALUES (?,?,?)',
              (today, ticker, json.dumps(analysis)))
    conn.commit()
    conn.close()


def run_analysis(ticker='IBIT'):
    """Run AI analysis across all DTEs. Returns analysis dict or raises."""
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise RuntimeError('ANTHROPIC_API_KEY not set')

    dtes = [3, 7, 14, 30, 45]
    results = {}

    def fetch_dte(dte):
        return dte, fetch_with_cache(ticker, dte)

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fetch_dte, d): d for d in dtes}
        for fut in as_completed(futures):
            dte, data = fut.result()
            results[dte] = data

    # Build concise summaries (strip chart data to keep prompt small)
    summaries = {}
    for dte in dtes:
        d = results[dte]
        bps = d['btc_per_share']
        lvl = d['levels']
        summaries[f"{dte}d"] = {
            'spot_btc': round(d['btc_spot']),
            'spot_ibit': d['spot'],
            'btc_per_share': bps,
            'levels_btc': {
                'call_wall': round(lvl['call_wall'] / bps),
                'put_wall': round(lvl['put_wall'] / bps),
                'gamma_flip': round(lvl['gamma_flip'] / bps),
                'max_pain': round(lvl['max_pain'] / bps),
                'resistance': [round(s / bps) for s in lvl.get('resistance', [])],
                'support': [round(s / bps) for s in lvl.get('support', [])],
            },
            'levels': lvl,
            'expected_move': d['expected_move'],
            'breakout': {
                'up_signals': d['breakout']['up_signals'],
                'down_signals': d['breakout']['down_signals'],
                'up_score': d['breakout']['up_score'],
                'down_score': d['breakout']['down_score'],
                'bias': d['breakout']['bias'],
            },
            'flow_forecast': {
                'charm': d['flow_forecast']['charm'],
                'vanna': {
                    'net_exposure': d['flow_forecast']['vanna']['net_exposure'],
                    'strength': d['flow_forecast']['vanna']['strength'],
                    'crush_scenario': d['flow_forecast']['vanna']['crush_scenario'],
                    'spike_scenario': d['flow_forecast']['vanna']['spike_scenario'],
                },
                'overnight': d['flow_forecast']['overnight'],
                'regime_note': d['flow_forecast']['regime_note'],
            },
            'oi_changes': d['oi_changes'],
            'significant_levels': [
                {k: v for k, v in sl.items() if k != 'greeks_note'}
                for sl in d['significant_levels'][:8]
            ],
        }

        # Add day-over-day level changes if previous data exists
        prev_date, prev_data = get_prev_cache(ticker, dte)
        if prev_data:
            prev_lvl = prev_data['levels']
            prev_bps = prev_data.get('btc_per_share', bps)
            summaries[f"{dte}d"]['changes_vs_prev'] = {
                'prev_date': prev_date,
                'spot_btc_prev': round(prev_data.get('btc_spot', 0)),
                'spot_btc_change': round(d['btc_spot'] - prev_data.get('btc_spot', d['btc_spot'])),
                'regime_prev': prev_lvl.get('regime', 'unknown'),
                'regime_changed': prev_lvl.get('regime') != lvl.get('regime'),
                'call_wall_btc_prev': round(prev_lvl['call_wall'] / prev_bps),
                'call_wall_btc_change': round(lvl['call_wall'] / bps - prev_lvl['call_wall'] / prev_bps),
                'put_wall_btc_prev': round(prev_lvl['put_wall'] / prev_bps),
                'put_wall_btc_change': round(lvl['put_wall'] / bps - prev_lvl['put_wall'] / prev_bps),
                'gamma_flip_btc_prev': round(prev_lvl['gamma_flip'] / prev_bps),
                'gamma_flip_btc_change': round(lvl['gamma_flip'] / bps - prev_lvl['gamma_flip'] / prev_bps),
                'net_gex_prev': prev_lvl.get('net_gex_total', 0),
                'net_gex_change': lvl.get('net_gex_total', 0) - prev_lvl.get('net_gex_total', 0),
                'pcr_prev': prev_lvl.get('pcr', 0),
                'pcr_change': round(lvl.get('pcr', 0) - prev_lvl.get('pcr', 0), 3),
            }

    prompt_data = json.dumps(summaries, cls=NumpyEncoder, indent=1)

    system_prompt = """You are a GEX (Gamma Exposure) trading analyst for IBIT (Bitcoin ETF). You analyze options flow data across multiple DTE timeframes to provide actionable trading insights.

IMPORTANT: IBIT is a Bitcoin ETF proxy. The data contains both IBIT share prices and BTC-equivalent prices (in levels_btc). ALWAYS use BTC prices in your analysis (e.g. "$65,200" not "$37.0"). Use the levels_btc fields for all price references. You can convert any IBIT price to BTC by dividing by btc_per_share.

If a "changes_vs_prev" field is present for a timeframe, it contains day-over-day changes — use this to highlight what shifted overnight:
- Level migrations (walls, gamma flip moving up/down)
- Regime flips (positive ↔ negative gamma)
- GEX and P/C ratio shifts
- Spot movement relative to level changes
This context is critical — a static snapshot is less useful than understanding the direction of positioning changes.

For each timeframe, provide:
- What changed overnight and why it matters (if changes_vs_prev available)
- Regime summary and implication (1-2 sentences)
- Key levels in BTC price and their significance relative to spot
- Dealer flow direction (charm/vanna implications)
- Risk assessment
- Actionable setup (if any clear one exists)

If a previous analysis is provided, use it to:
- Note whether your prior calls played out or not (e.g. "yesterday's $65K support held as expected" or "prior upside bias was invalidated")
- Update your thesis based on how positioning evolved
- Maintain continuity — don't repeat the same analysis if nothing changed, focus on what's new

For the "all" key: provide cross-timeframe alignment analysis — whether short-term and long-term signals agree, overall directional bias, and the highest-conviction trade setup. Highlight any divergences between short-term and long-term positioning changes.

Keep each analysis to 4-6 bullet points. Be concise and direct — no walls of text. Use trader shorthand where appropriate. Reference specific BTC price levels.

IMPORTANT: Return ONLY valid JSON with keys "3d", "7d", "14d", "30d", "45d", "all". Each value should be a string containing your analysis with newlines for formatting. Do not wrap in markdown code blocks."""

    # Build user message with optional previous analysis
    prev_analysis_date, prev_analysis = get_prev_analysis(ticker)
    user_content = f"Analyze the following IBIT GEX data across all timeframes:\n\n{prompt_data}"
    if prev_analysis:
        prev_json = json.dumps(prev_analysis, indent=1)
        user_content += f"\n\n--- PREVIOUS ANALYSIS ({prev_analysis_date}) ---\n{prev_json}"

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=5120,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": user_content
        }],
    )

    if msg.stop_reason == 'max_tokens':
        raise RuntimeError('LLM response truncated — analysis too long')

    raw = msg.content[0].text.strip()
    if raw.startswith('```'):
        raw = raw.split('\n', 1)[1]
        if raw.endswith('```'):
            raw = raw[:-3].strip()

    try:
        analysis = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f'Failed to parse LLM response: {e}. Raw: {raw[:500]}')
    set_cached_analysis(ticker, analysis)
    return analysis


@app.route('/api/analysis')
def api_analysis():
    """GET cached analysis. Returns today's if available, otherwise most recent."""
    cached = get_cached_analysis('IBIT')
    if cached:
        return Response(json.dumps(cached), mimetype='application/json')
    # Fall back to most recent analysis (e.g. overnight before new data arrives)
    prev_date, prev = get_prev_analysis('IBIT')
    if prev:
        return Response(json.dumps(prev), mimetype='application/json')
    return Response(json.dumps({'status': 'pending'}), mimetype='application/json')


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Force re-run analysis (manual refresh)."""
    try:
        analysis = run_analysis('IBIT')
        return Response(json.dumps(analysis), mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({'error': str(e)}),
                        mimetype='application/json'), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', '-p', type=int, default=5000)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--dte', '-d', type=int, default=7, help='Max days to expiration (default: 7)')
    args = parser.parse_args()
    init_db()
    app.config['MAX_DTE'] = args.dte
    print(f"\n  IBIT GEX Dashboard → http://{args.host}:{args.port}  (DTE: {args.dte})\n")
    # Start background refresh (skip reloader parent to avoid double threads)
    if os.environ.get('WERKZEUG_RUN_MAIN') or not app.debug:
        start_bg_refresh()
    app.run(host=args.host, port=args.port, debug=True)