#!/usr/bin/env python3
"""
GEX Terminal
============
Flask app that calculates Gamma Exposure (GEX) from crypto ETF options chains
(IBIT/ETHA) and serves an interactive trading dashboard with candlestick chart,
GEX/OI profiles, and regime-adjusted level overlays.

Usage:
  python3 app.py                    # default: 7 DTE, port 5000
  python3 app.py --dte 14           # 14-day expiration window
  python3 app.py --host 0.0.0.0     # accessible from WSL2 host
"""

import json
import math
import os
import argparse
import sqlite3
import threading
import time
import urllib.request
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
RISK_FREE_RATE_DEFAULT = 0.043
STRIKE_RANGE_PCT = 0.35
DB_PATH = os.path.join(str(Path.home()), ".ibit_gex_history.db")

TICKER_CONFIG = {
    'IBIT': {
        'name': 'IBIT', 'asset_label': 'BTC', 'ref_ticker': 'BTC-USD',
        'binance_symbol': 'BTCUSDT', 'per_share_default': 0.000568,
    },
    'ETHA': {
        'name': 'ETHA', 'asset_label': 'ETH', 'ref_ticker': 'ETH-USD',
        'binance_symbol': 'ETHUSDT', 'per_share_default': 0.0091,
    },
}


_rfr_cache = {'rate': None, 'date': None}

def get_risk_free_rate():
    """Fetch 13-week T-bill rate (^IRX) from Yahoo Finance, cached daily."""
    today = datetime.now().strftime('%Y-%m-%d')
    if _rfr_cache['rate'] is not None and _rfr_cache['date'] == today:
        return _rfr_cache['rate']
    try:
        irx = yf.Ticker('^IRX').info.get('regularMarketPrice')
        if irx and irx > 0:
            rate = irx / 100.0
            _rfr_cache['rate'] = rate
            _rfr_cache['date'] = today
            print(f"  [rfr] 13-week T-bill rate: {irx:.2f}% ({rate:.4f})")
            return rate
    except Exception as e:
        print(f"  [rfr] Failed to fetch ^IRX, using default: {e}")
    return RISK_FREE_RATE_DEFAULT


# ── BLACK-SCHOLES ───────────────────────────────────────────────────────────
def _bs_d1d2(S, K, T, r, sigma):
    """Shared d1/d2 computation for all Black-Scholes Greeks."""
    sqrt_T = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sqrt_T
    d2 = d1 - sqrt_T
    return d1, d2

def bs_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1, _ = _bs_d1d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def bs_delta(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1, _ = _bs_d1d2(S, K, T, r, sigma)
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

def bs_vanna(S, K, T, r, sigma):
    """Vanna = dDelta/dVol — dealer rebalancing when IV moves."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1, d2 = _bs_d1d2(S, K, T, r, sigma)
    return -norm.pdf(d1) * d2 / sigma

def bs_charm(S, K, T, r, sigma, option_type='call'):
    """Charm = dDelta/dT — dealer rebalancing from time decay of delta."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1, d2 = _bs_d1d2(S, K, T, r, sigma)
    charm = -norm.pdf(d1) * (2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
    if option_type == 'put':
        charm += r * np.exp(-r * T) * norm.cdf(-d2)
    return charm


def dte_weight(dte_days, max_dte):
    """Weight by gamma's natural time scaling: 1/sqrt(T), normalized to max DTE window."""
    dte_clamped = max(dte_days, 0.25)  # expiration day floor (~6 hours)
    return 1.0 / math.sqrt(dte_clamped / max(max_dte, 1))


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
    c.execute('''CREATE TABLE IF NOT EXISTS btc_candles (
        symbol TEXT NOT NULL DEFAULT 'BTCUSDT',
        tf TEXT NOT NULL,
        time INTEGER NOT NULL,
        open REAL NOT NULL, high REAL NOT NULL,
        low REAL NOT NULL, close REAL NOT NULL,
        UNIQUE(symbol, tf, time)
    )''')
    # Migrate old btc_candles table (no symbol column) to new schema
    cols = [row[1] for row in c.execute('PRAGMA table_info(btc_candles)').fetchall()]
    if 'symbol' not in cols:
        c.execute('''CREATE TABLE btc_candles_new (
            symbol TEXT NOT NULL DEFAULT 'BTCUSDT',
            tf TEXT NOT NULL,
            time INTEGER NOT NULL,
            open REAL NOT NULL, high REAL NOT NULL,
            low REAL NOT NULL, close REAL NOT NULL,
            UNIQUE(symbol, tf, time)
        )''')
        c.execute("INSERT INTO btc_candles_new (symbol, tf, time, open, high, low, close) SELECT 'BTCUSDT', tf, time, open, high, low, close FROM btc_candles")
        c.execute('DROP TABLE btc_candles')
        c.execute('ALTER TABLE btc_candles_new RENAME TO btc_candles')
    c.execute('CREATE INDEX IF NOT EXISTS idx_candles_sym_tf_time ON btc_candles(symbol, tf, time)')
    c.execute('''CREATE TABLE IF NOT EXISTS etf_flows (
        date TEXT NOT NULL, ticker TEXT NOT NULL,
        shares_outstanding REAL, aum REAL, nav REAL,
        daily_flow_shares REAL, daily_flow_dollars REAL,
        UNIQUE(date, ticker)
    )''')
    # Migrate: add weighted_net_gex column to existing tables
    for table in ('snapshots', 'strike_history'):
        cols = [row[1] for row in c.execute(f'PRAGMA table_info({table})').fetchall()]
        if 'weighted_net_gex' not in cols:
            c.execute(f'ALTER TABLE {table} ADD COLUMN weighted_net_gex REAL')
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
    c.execute('SELECT strike, call_oi, put_oi, total_oi, net_gex, weighted_net_gex FROM strike_history WHERE date=? AND ticker=?',
              (prev_date, ticker))
    strikes = {}
    for r in c.fetchall():
        strikes[r[0]] = {'call_oi': r[1], 'put_oi': r[2], 'total_oi': r[3], 'net_gex': r[4], 'weighted_net_gex': r[5]}
    return prev_date, strikes


def save_snapshot(conn, ticker, spot, btc_price, levels, df):
    date_str = datetime.now().strftime('%Y-%m-%d')
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO snapshots
        (date,ticker,spot,btc_price,gamma_flip,call_wall,put_wall,max_pain,regime,net_gex,total_call_oi,total_put_oi,weighted_net_gex)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''',
        (date_str, ticker, spot, btc_price, levels.get('gamma_flip'), levels.get('call_wall'),
         levels.get('put_wall'), levels.get('max_pain'), levels.get('regime'),
         levels.get('net_gex_total'), levels.get('total_call_oi'), levels.get('total_put_oi'),
         levels.get('net_gex_w_total')))
    for _, row in df.iterrows():
        c.execute('''INSERT OR REPLACE INTO strike_history (date,ticker,strike,call_oi,put_oi,total_oi,net_gex,weighted_net_gex)
            VALUES (?,?,?,?,?,?,?,?)''',
            (date_str, ticker, row['strike'], int(row['call_oi']), int(row['put_oi']),
             int(row['total_oi']), row['net_gex'], row['net_gex_w']))
    conn.commit()


def get_history(conn, ticker, days=10):
    c = conn.cursor()
    c.execute('''SELECT date, spot, btc_price, gamma_flip, call_wall, put_wall,
                        max_pain, regime, net_gex, total_call_oi, total_put_oi, weighted_net_gex
                 FROM snapshots WHERE ticker=? ORDER BY date DESC LIMIT ?''', (ticker, days))
    return c.fetchall()


def fetch_etf_flows(ticker_symbol):
    """Calculate daily ETF fund flows from shares outstanding changes."""
    try:
        info = yf.Ticker(ticker_symbol).info
        shares = info.get('sharesOutstanding')
        aum = info.get('totalAssets')
        nav = info.get('navPrice') or info.get('regularMarketPrice')
        # Derive shares from AUM/NAV if sharesOutstanding not available
        if not shares and aum and nav:
            shares = aum / nav
        if not shares or not nav:
            return None

        conn = get_db()
        try:
            c = conn.cursor()
            today = datetime.now().strftime('%Y-%m-%d')

            # Get previous day's data
            c.execute('SELECT date, shares_outstanding, aum, nav FROM etf_flows WHERE ticker=? AND date<? ORDER BY date DESC LIMIT 1',
                      (ticker_symbol, today))
            prev = c.fetchone()

            # Calculate daily flow
            daily_flow_shares = 0.0
            daily_flow_dollars = 0.0
            if prev and prev[1]:
                daily_flow_shares = shares - prev[1]
                daily_flow_dollars = daily_flow_shares * nav

            # Store today's snapshot
            c.execute('INSERT OR REPLACE INTO etf_flows (date, ticker, shares_outstanding, aum, nav, daily_flow_shares, daily_flow_dollars) VALUES (?,?,?,?,?,?,?)',
                      (today, ticker_symbol, shares, aum, nav, daily_flow_shares, daily_flow_dollars))
            conn.commit()

            # Compute 5-day flow streak and momentum (real data only, not backfill estimates)
            c.execute('SELECT daily_flow_shares, daily_flow_dollars FROM etf_flows WHERE ticker=? AND shares_outstanding IS NOT NULL ORDER BY date DESC LIMIT 5',
                      (ticker_symbol,))
            history = c.fetchall()
        finally:
            conn.close()

        streak = 0
        if history:
            direction = 1 if history[0][0] >= 0 else -1
            for h in history:
                if (h[0] >= 0) == (direction > 0):
                    streak += 1
                else:
                    break
            streak *= direction

        avg_flow_5d = sum(h[1] for h in history) / len(history) if history else 0

        # Categorize strength by dollar amount
        abs_flow = abs(daily_flow_dollars)
        if abs_flow < 10_000_000:
            strength = 'negligible'
        elif abs_flow < 50_000_000:
            strength = 'minor'
        elif abs_flow < 200_000_000:
            strength = 'moderate'
        else:
            strength = 'strong'

        return {
            'daily_flow_shares': float(daily_flow_shares),
            'daily_flow_dollars': float(daily_flow_dollars),
            'direction': 'inflow' if daily_flow_shares >= 0 else 'outflow',
            'strength': strength,
            'streak': int(streak),          # positive = N consecutive inflow days, negative = outflow
            'avg_flow_5d': float(avg_flow_5d),
            'shares_outstanding': float(shares),
        }
    except Exception as e:
        print(f"  [etf-flows] Failed for {ticker_symbol}: {e}")
        return None



def backfill_etf_flows(ticker_symbol, days=90):
    """Backfill ETF flow history from Yahoo Finance historical data.
    Uses a volume-based heuristic — estimates only, for chart visualization.
    Real flow calculations use actual shares outstanding deltas (fetch_etf_flows)."""
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM etf_flows WHERE ticker=?', (ticker_symbol,))
        existing = c.fetchone()[0]
        conn.close()
        if existing >= 5:
            return  # already have history

        tk = yf.Ticker(ticker_symbol)
        hist = tk.history(period=f'{days}d')
        if hist.empty:
            return

        info = tk.info
        current_nav = info.get('navPrice') or info.get('regularMarketPrice')
        if not current_nav:
            return

        conn = get_db()
        try:
            c = conn.cursor()
            for date_idx, row in hist.iterrows():
                date_str = date_idx.strftime('%Y-%m-%d')
                nav = float(row['Close'])
                volume = float(row.get('Volume', 0))
                open_p = float(row['Open'])
                close_p = float(row['Close'])
                direction = 1 if close_p >= open_p else -1
                # Rough proxy: ~15% of volume is creation/redemption
                estimated_flow = direction * volume * nav * 0.15

                # shares_outstanding=NULL marks this as an estimate (not real data)
                c.execute('INSERT OR IGNORE INTO etf_flows (date, ticker, shares_outstanding, aum, nav, daily_flow_shares, daily_flow_dollars) VALUES (?,?,?,?,?,?,?)',
                          (date_str, ticker_symbol, None, None, nav, 0, estimated_flow))
            conn.commit()
        finally:
            conn.close()
        print(f"  [etf-flows] Backfilled {len(hist)} days for {ticker_symbol}")
    except Exception as e:
        print(f"  [etf-flows] Backfill failed for {ticker_symbol}: {e}")


# ── CANDLES ─────────────────────────────────────────────────────────────────
_candle_backfill_done = {}  # symbol -> threading.Event

TF_DURATIONS_MS = {
    '15m': 15 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '4h': 4 * 60 * 60 * 1000,
    '1d': 24 * 60 * 60 * 1000,
}


def _fetch_binance_klines(symbol, tf, start_ms, end_ms, limit=1000):
    """Fetch klines from Binance REST API with .com/.us fallback."""
    urls = [
        f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={tf}&startTime={start_ms}&endTime={end_ms}&limit={limit}',
        f'https://api.binance.us/api/v3/klines?symbol={symbol}&interval={tf}&startTime={start_ms}&endTime={end_ms}&limit={limit}',
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'ibit-gex/1.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                if isinstance(data, list):
                    return data
        except Exception as e:
            print(f"  [candles] Binance fetch failed ({url[:50]}...): {e}")
    return []


def backfill_btc_candles(symbol, tf, days=90):
    """Paginated backfill of candles for a given symbol and timeframe."""
    tf_ms = TF_DURATIONS_MS.get(tf)
    if not tf_ms:
        return
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - days * 24 * 60 * 60 * 1000
    conn = get_db()
    c = conn.cursor()
    cursor = start_ms
    total = 0
    while cursor < now_ms:
        klines = _fetch_binance_klines(symbol, tf, cursor, now_ms, limit=1000)
        if not klines:
            break
        rows = [(symbol, tf, int(k[0] // 1000), float(k[1]), float(k[2]), float(k[3]), float(k[4]))
                for k in klines]
        c.executemany('INSERT OR REPLACE INTO btc_candles (symbol, tf, time, open, high, low, close) VALUES (?,?,?,?,?,?,?)', rows)
        conn.commit()
        total += len(rows)
        last_open_ms = int(klines[-1][0])
        cursor = last_open_ms + tf_ms
        if len(klines) < 1000:
            break
        time.sleep(0.2)
    conn.close()
    print(f"  [candles] Backfilled {total} candles for {symbol}/{tf}")


def get_btc_candles(symbol, tf):
    """Return all stored candles for a symbol/timeframe as TradingView-format dicts."""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT time, open, high, low, close FROM btc_candles WHERE symbol=? AND tf=? ORDER BY time ASC', (symbol, tf))
    rows = c.fetchall()
    conn.close()
    return [{'time': r[0], 'open': r[1], 'high': r[2], 'low': r[3], 'close': r[4]} for r in rows]


def update_btc_candles(symbol, tf):
    """Fetch candles since the latest stored one and upsert."""
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT MAX(time) FROM btc_candles WHERE symbol=? AND tf=?', (symbol, tf))
    row = c.fetchone()
    if row and row[0]:
        start_ms = row[0] * 1000
    else:
        start_ms = int(time.time() * 1000) - 90 * 24 * 60 * 60 * 1000
    now_ms = int(time.time() * 1000)
    klines = _fetch_binance_klines(symbol, tf, start_ms, now_ms, limit=1000)
    if klines:
        rows = [(symbol, tf, int(k[0] // 1000), float(k[1]), float(k[2]), float(k[3]), float(k[4]))
                for k in klines]
        c.executemany('INSERT OR REPLACE INTO btc_candles (symbol, tf, time, open, high, low, close) VALUES (?,?,?,?,?,?,?)', rows)
        conn.commit()
    conn.close()


# ── DATA ────────────────────────────────────────────────────────────────────
def fetch_and_analyze(ticker_symbol='IBIT', max_dte=7):
    cfg = TICKER_CONFIG.get(ticker_symbol)
    is_crypto = cfg is not None

    ticker = yf.Ticker(ticker_symbol)
    spot = ticker.info.get('regularMarketPrice')
    if spot is None:
        hist = ticker.history(period="1d")
        spot = float(hist['Close'].iloc[-1])

    # Auto ref_per_share — compute locally to avoid race conditions
    ref_per_share = cfg['per_share_default'] if cfg else 1.0
    if is_crypto:
        try:
            ref_price = yf.Ticker(cfg['ref_ticker']).info.get('regularMarketPrice')
            if ref_price and ref_price > 0:
                ref_per_share = spot / ref_price
        except Exception:
            pass

    btc_spot = spot / ref_per_share if is_crypto else None

    rfr = get_risk_free_rate()

    all_exps = list(ticker.options)
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=max_dte)
    selected_exps = [e for e in all_exps if datetime.strptime(e, "%Y-%m-%d").replace(tzinfo=timezone.utc) <= cutoff]
    if not selected_exps:
        selected_exps = all_exps[:3]

    # Fetch previous day's strike data (needed for Active GEX + breakout + sig levels)
    conn = get_db()
    prev_date, prev_strikes = get_prev_strikes(conn, ticker_symbol)
    conn.close()

    etf_flows = fetch_etf_flows(ticker_symbol) if is_crypto else None

    # Collect options data
    strike_data = {}
    cached_chains = {}  # exp_str -> chain, reused for expected move
    for exp_str in selected_exps:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        T = max((exp_date - now).days / 365.0, 0.5 / 365)
        dte_days = max((exp_date - now).days, 0)
        w = dte_weight(dte_days, max_dte)
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

                gamma = bs_gamma(spot, strike, T, rfr, iv)
                delta = bs_delta(spot, strike, T, rfr, iv, opt_type)
                vanna = bs_vanna(spot, strike, T, rfr, iv)
                charm = bs_charm(spot, strike, T, rfr, iv, opt_type)
                gex = sign * gamma * oi * 100 * spot ** 2 * 0.01
                gex_w = gex * w
                dealer_delta = -delta * oi * 100
                # Dealer vanna exposure: how dealer delta changes with IV
                dealer_vanna = -sign * vanna * oi * 100 * spot * 0.01
                # Dealer charm exposure: how dealer hedge changes overnight
                # bs_charm returns -dΔ/dτ; dealer hedge change = bs_charm * OI * 100 / 365
                dealer_charm = charm * oi * 100 / 365.0

                vol = row.get('volume', 0)
                if pd.isna(vol):
                    vol = 0
                vol = int(vol)

                if strike not in strike_data:
                    strike_data[strike] = {'call_oi': 0, 'put_oi': 0, 'call_gex': 0, 'put_gex': 0,
                                           'call_gex_w': 0, 'put_gex_w': 0,
                                           'call_delta': 0, 'put_delta': 0,
                                           'call_vanna': 0, 'put_vanna': 0,
                                           'call_charm': 0, 'put_charm': 0,
                                           'call_volume': 0, 'put_volume': 0,
                                           'expiry_gex': {}}
                strike_data[strike][f'{opt_type}_oi'] += oi
                strike_data[strike][f'{opt_type}_gex'] += gex
                strike_data[strike][f'{opt_type}_gex_w'] += gex_w
                strike_data[strike][f'{opt_type}_volume'] += vol
                strike_data[strike][f'{opt_type}_delta'] += dealer_delta
                strike_data[strike][f'{opt_type}_vanna'] += dealer_vanna
                strike_data[strike][f'{opt_type}_charm'] += dealer_charm
                # Per-expiry breakdown
                if exp_str not in strike_data[strike]['expiry_gex']:
                    strike_data[strike]['expiry_gex'][exp_str] = {
                        'call_gex_w': 0, 'put_gex_w': 0, 'dte': dte_days, 'weight': round(w, 2)
                    }
                strike_data[strike]['expiry_gex'][exp_str][f'{opt_type}_gex_w'] += gex_w

    # Build dataframe
    rows = []
    for strike, d in sorted(strike_data.items()):
        # Compute net_gex_w for each expiry entry
        expiry_gex = {}
        for exp_str, eg in d.get('expiry_gex', {}).items():
            eg['net_gex_w'] = eg['call_gex_w'] + eg['put_gex_w']
            expiry_gex[exp_str] = eg
        rows.append({
            'strike': strike,
            'btc_price': strike / ref_per_share if is_crypto else strike,
            'call_oi': d['call_oi'], 'put_oi': d['put_oi'],
            'total_oi': d['call_oi'] + d['put_oi'],
            'call_gex': d['call_gex'], 'put_gex': d['put_gex'],
            'net_gex': d['call_gex'] + d['put_gex'],
            'call_gex_w': d['call_gex_w'], 'put_gex_w': d['put_gex_w'],
            'net_gex_w': d['call_gex_w'] + d['put_gex_w'],
            'net_dealer_delta': d['call_delta'] + d['put_delta'],
            'net_vanna': d['call_vanna'] + d['put_vanna'],
            'net_charm': d['call_charm'] + d['put_charm'],
            'call_vanna': d['call_vanna'], 'put_vanna': d['put_vanna'],
            'call_charm': d['call_charm'], 'put_charm': d['put_charm'],
            'call_volume': d['call_volume'], 'put_volume': d['put_volume'],
            'total_volume': d['call_volume'] + d['put_volume'],
            'expiry_gex': expiry_gex,
        })
    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No options data found for {ticker_symbol} within {max_dte} DTE")

    # Compute Active GEX: weight net_gex by fraction of OI that is new since yesterday
    active_gex_values = []
    for _, row in df.iterrows():
        strike = row['strike']
        total_oi = row['total_oi']
        if prev_strikes and strike in prev_strikes and total_oi > 0:
            prev_total_oi = prev_strikes[strike]['total_oi']
            delta_oi = max(total_oi - prev_total_oi, 0)
            ratio = min(delta_oi / total_oi, 1.0)
        else:
            ratio = 1.0  # no history = all new
        active_gex_values.append(row['net_gex_w'] * ratio)
    df['active_gex'] = active_gex_values

    # Derive levels
    levels = {}
    # Call wall: highest call GEX (weighted) at or above spot (resistance)
    calls_above = df[df['strike'] >= spot]
    if not calls_above.empty:
        levels['call_wall'] = float(calls_above.loc[calls_above['call_gex_w'].idxmax(), 'strike'])
    else:
        levels['call_wall'] = float(df.loc[df['call_gex_w'].idxmax(), 'strike'])
    # Put wall: most negative put GEX (weighted) at or below spot (support)
    puts_below = df[df['strike'] <= spot]
    if not puts_below.empty:
        levels['put_wall'] = float(puts_below.loc[puts_below['put_gex_w'].idxmin(), 'strike'])
    else:
        levels['put_wall'] = float(df.loc[df['put_gex_w'].idxmin(), 'strike'])

    # Gamma flip nearest to spot (using weighted GEX)
    df_s = df.sort_values('strike')
    all_flips = []
    for i in range(len(df_s) - 1):
        g1, g2 = df_s.iloc[i]['net_gex_w'], df_s.iloc[i + 1]['net_gex_w']
        s1, s2 = df_s.iloc[i]['strike'], df_s.iloc[i + 1]['strike']
        if (g1 < 0 and g2 > 0) or (g1 > 0 and g2 < 0):
            if (g2 - g1) != 0:
                flip = s1 + (s2 - s1) * (-g1) / (g2 - g1)
                all_flips.append(flip)
    levels['gamma_flip'] = float(min(all_flips, key=lambda x: abs(x - spot))) if all_flips else float(spot)

    # Max pain (vectorized)
    strikes_arr = df['strike'].values
    call_oi_arr = df['call_oi'].values
    put_oi_arr = df['put_oi'].values
    # For each potential settle price, compute total payout across all strikes
    # Shape: (num_settles, num_strikes) via broadcasting
    settle_grid = strikes_arr[:, np.newaxis]  # column of settle prices
    call_pain = call_oi_arr * np.maximum(0, settle_grid - strikes_arr) * 100
    put_pain = put_oi_arr * np.maximum(0, strikes_arr - settle_grid) * 100
    total_pain = (call_pain + put_pain).sum(axis=1)
    levels['max_pain'] = float(strikes_arr[np.argmin(total_pain)])

    # Regime (using weighted GEX)
    near_spot = df[(df['strike'] >= spot * 0.98) & (df['strike'] <= spot * 1.02)]
    local_gex = near_spot['net_gex_w'].sum() if not near_spot.empty else 0
    levels['regime'] = 'positive_gamma' if local_gex > 0 else 'negative_gamma'

    # Totals
    levels['net_gex_total'] = float(df['net_gex'].sum())
    levels['net_gex_w_total'] = float(df['net_gex_w'].sum())
    levels['net_dealer_delta'] = float(df['net_dealer_delta'].sum())
    levels['net_vanna'] = float(df['net_vanna'].sum())
    levels['net_charm'] = float(df['net_charm'].sum())
    levels['total_call_oi'] = int(df['call_oi'].sum())
    levels['total_put_oi'] = int(df['put_oi'].sum())
    levels['pcr'] = levels['total_put_oi'] / max(levels['total_call_oi'], 1)

    # Active GEX totals and walls
    levels['active_gex_total'] = float(df['active_gex'].sum())
    active_pos = df[df['active_gex'] > 0]
    active_neg = df[df['active_gex'] < 0]
    levels['active_call_wall'] = float(active_pos.loc[active_pos['active_gex'].idxmax(), 'strike']) if not active_pos.empty else levels['call_wall']
    levels['active_put_wall'] = float(active_neg.loc[active_neg['active_gex'].idxmin(), 'strike']) if not active_neg.empty else levels['put_wall']

    # Positioning confidence: how much to trust the naive GEX sign convention
    # (dealers long calls, short puts — the Perfiliev/SqueezeMetrics assumption)
    pos_conf = 100
    pos_warnings = []
    pcr = levels['pcr']
    if pcr < 0.5:
        pos_conf -= 40
        pos_warnings.append(f'P/C ratio {pcr:.2f} — heavy speculative call buying, dealers likely short calls')
    elif pcr < 0.7:
        pos_conf -= 25
        pos_warnings.append(f'P/C ratio {pcr:.2f} — call-heavy flow suggests dealers short calls')
    elif pcr < 0.85:
        pos_conf -= 10
        pos_warnings.append(f'P/C ratio {pcr:.2f} — mild call skew')

    otm_call_oi = int(df[df['strike'] > spot]['call_oi'].sum())
    total_call_oi = levels['total_call_oi']
    if total_call_oi > 0 and (otm_call_oi / total_call_oi) > 0.6:
        pos_conf -= 15
        pos_warnings.append(f'{otm_call_oi/total_call_oi*100:.0f}% of call OI is OTM — speculative accumulation')

    total_call_vol = int(df['call_volume'].sum())
    total_put_vol = int(df['put_volume'].sum())
    call_voi = total_call_vol / max(total_call_oi, 1)
    put_voi = total_put_vol / max(levels['total_put_oi'], 1)
    if put_voi > 0 and call_voi > 2 * put_voi:
        pos_conf -= 10
        pos_warnings.append(f'Call V/OI {call_voi:.2f} vs put V/OI {put_voi:.2f} — active speculative call turnover')

    if total_call_oi > 0:
        max_call_oi_strike = int(df['call_oi'].max())
        if (max_call_oi_strike / total_call_oi) > 0.2:
            pos_conf -= 10
            pos_warnings.append(f'Single strike holds {max_call_oi_strike/total_call_oi*100:.0f}% of call OI — concentrated speculative bet')

    # ETF fund flow context
    if etf_flows and etf_flows['streak'] <= -3 and etf_flows['strength'] in ('moderate', 'strong'):
        pos_conf -= 15
        pos_warnings.append(f"ETF outflow streak ({etf_flows['streak']}d, avg ${abs(etf_flows['avg_flow_5d'])/1e6:.0f}M/d) — institutional exit weakens support")

    levels['positioning_confidence'] = max(0, min(100, pos_conf))
    levels['positioning_warnings'] = pos_warnings

    # Resistance / support (weighted GEX)
    levels['resistance'] = df[df['net_gex_w'] > 0].nlargest(3, 'net_gex_w')['strike'].tolist()
    levels['support'] = df[df['net_gex_w'] < 0].nsmallest(3, 'net_gex_w')['strike'].tolist()

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
            'upper_btc': float((spot + straddle) / ref_per_share) if is_crypto else None,
            'lower_btc': float((spot - straddle) / ref_per_share) if is_crypto else None,
            'expiration': nearest_exp, 'dte': dte,
        }
    except Exception:
        pass

    # Level strength trajectory
    level_trajectory = {}
    for lname, lstrike in [('call_wall', levels['call_wall']), ('put_wall', levels['put_wall']), ('gamma_flip', levels['gamma_flip'])]:
        traj = {'status': 'STABLE', 'change_pct': 0.0, 'dominant_expiry': None, 'dominant_expiry_dte': None}
        # Current weighted GEX at this strike
        strike_rows = df[df['strike'] == lstrike]
        current_w = float(strike_rows['net_gex_w'].sum()) if not strike_rows.empty else 0
        # Previous weighted GEX (fallback to raw for pre-migration)
        if prev_strikes and lstrike in prev_strikes:
            prev_w = prev_strikes[lstrike].get('weighted_net_gex')
            if prev_w is None:
                prev_w = prev_strikes[lstrike].get('net_gex', 0)
            if prev_w and prev_w != 0:
                change_pct = ((current_w - prev_w) / abs(prev_w)) * 100
                traj['change_pct'] = round(change_pct, 1)
                if change_pct > 10:
                    traj['status'] = 'STRENGTHENING'
                elif change_pct < -10:
                    traj['status'] = 'WEAKENING'
        # Dominant expiry: which expiration contributes most weighted GEX
        if not strike_rows.empty:
            expiry_gex = strike_rows.iloc[0].get('expiry_gex', {})
            if isinstance(expiry_gex, dict) and expiry_gex:
                best_exp = max(expiry_gex.items(), key=lambda x: abs(x[1].get('net_gex_w', 0)))
                traj['dominant_expiry'] = best_exp[0]
                traj['dominant_expiry_dte'] = best_exp[1].get('dte')
        level_trajectory[lname] = traj
    levels['level_trajectory'] = level_trajectory

    # Breakout signals (prev_strikes already available from early fetch)
    breakout = compute_breakout(df, spot, levels, expected_move, prev_strikes, ref_per_share, etf_flows)

    # Significant levels with regime behavior
    sig_levels = compute_significant_levels(df, spot, levels, prev_strikes, is_crypto, ref_per_share, etf_flows)

    # Dealer flow forecast (vanna + charm)
    flow_forecast = compute_flow_forecast(df, spot, levels, is_crypto)

    # Dealer delta scenario analysis
    dealer_delta_profile = None
    dealer_delta_briefing = None
    delta_flip_points = []
    try:
        scenario_result = compute_dealer_delta_scenarios(
            cached_chains, spot, levels, expected_move,
            rfr, ref_per_share, is_crypto
        )
        dealer_delta_profile = scenario_result['profile']
        delta_flip_points = scenario_result['flip_points']
        dealer_delta_briefing = generate_dealer_delta_briefing(
            scenario_result, spot, levels, ref_per_share, is_crypto
        )
    except Exception as e:
        print(f"  [delta-scenario] Failed: {e}")

    # Save to DB
    conn = get_db()
    save_snapshot(conn, ticker_symbol, spot, btc_spot, levels, df)

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
        # Per-expiry breakdown for stacked chart
        expiry_gex = row.get('expiry_gex', {})
        expiry_breakdown = {}
        if isinstance(expiry_gex, dict):
            for exp_str, eg in expiry_gex.items():
                expiry_breakdown[exp_str] = {
                    'net_gex_w': eg.get('net_gex_w', 0),
                    'dte': eg.get('dte'),
                    'weight': eg.get('weight'),
                }
        gex_chart_data.append({
            'strike': float(row['strike']),
            'btc': float(row['strike'] / ref_per_share) if is_crypto else float(row['strike']),
            'net_gex': float(row['net_gex']),
            'net_gex_w': float(row['net_gex_w']),
            'active_gex': float(row['active_gex']),
            'net_vanna': float(row['net_vanna']),
            'net_charm': float(row['net_charm']),
            'call_oi': int(row['call_oi']),
            'put_oi': int(row['put_oi']),
            'total_oi': int(row['total_oi']),
            'call_volume': int(row['call_volume']),
            'put_volume': int(row['put_volume']),
            'total_volume': int(row['total_volume']),
            'expiry_breakdown': expiry_breakdown,
        })

    history_data = []
    for h in history:
        history_data.append({
            'date': h[0], 'spot': h[1], 'btc_price': h[2],
            'gamma_flip': h[3], 'call_wall': h[4], 'put_wall': h[5],
            'max_pain': h[6], 'regime': h[7], 'net_gex': h[8],
            'total_call_oi': h[9], 'total_put_oi': h[10],
            'weighted_net_gex': h[11] if len(h) > 11 else None,
        })

    # Expiry metadata for frontend legend/palette
    expiry_meta = []
    for exp_str in selected_exps:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        dte_days = max((exp_date - now).days, 0)
        expiry_meta.append({'exp': exp_str, 'dte': dte_days, 'weight': round(dte_weight(dte_days, max_dte), 2)})
    expiry_meta.sort(key=lambda x: x['dte'])

    return {
        'ticker': ticker_symbol,
        'asset_label': cfg['asset_label'] if cfg else ticker_symbol,
        'spot': float(spot),
        'btc_spot': float(btc_spot) if btc_spot else None,
        'btc_per_share': float(ref_per_share),
        'is_btc': bool(is_crypto),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'levels': {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                   for k, v in levels.items()},
        'expected_move': expected_move,
        'gex_chart': gex_chart_data,
        'significant_levels': sig_levels,
        'breakout': breakout,
        'etf_flows': etf_flows,
        'oi_changes': oi_changes,
        'flow_forecast': flow_forecast,
        'dealer_delta_profile': dealer_delta_profile,
        'dealer_delta_briefing': dealer_delta_briefing,
        'delta_flip_points': delta_flip_points,
        'history': history_data,
        'expirations': selected_exps,
        'expiry_meta': expiry_meta,
    }


def compute_flow_forecast(df, spot, levels, is_crypto):
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
    if is_crypto:
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


def compute_dealer_delta_scenarios(cached_chains, spot, levels, expected_move, rfr, ref_per_share, is_crypto):
    """Pre-compute dealer delta at hypothetical prices across the key level grid."""
    cw = levels.get('call_wall', spot * 1.05)
    pw = levels.get('put_wall', spot * 0.95)
    gf = levels.get('gamma_flip', spot)
    mp = levels.get('max_pain', spot)

    # Build price grid
    grid_prices = set()
    grid_prices.update([cw, pw, gf, mp, spot])

    if expected_move:
        if expected_move.get('upper'):
            grid_prices.add(expected_move['upper'])
        if expected_move.get('lower'):
            grid_prices.add(expected_move['lower'])

    for s in levels.get('resistance', []):
        grid_prices.add(s)
    for s in levels.get('support', []):
        grid_prices.add(s)

    # 0.5% increments between put_wall and call_wall
    step = spot * 0.005
    if step > 0:
        p = pw
        while p <= cw:
            grid_prices.add(p)
            p += step

    # Extend ±2% beyond walls
    lo_ext = pw * 0.98
    hi_ext = cw * 1.02
    p = lo_ext
    while p <= hi_ext:
        grid_prices.add(p)
        p += step

    # Filter to reasonable range, deduplicate, sort
    lo_bound = spot * 0.85
    hi_bound = spot * 1.15
    grid_prices = sorted([p for p in grid_prices if lo_bound <= p <= hi_bound])

    # Cap grid to max 80 points
    if len(grid_prices) > 80:
        indices = np.linspace(0, len(grid_prices) - 1, 80, dtype=int)
        grid_prices = [grid_prices[i] for i in indices]
        # Ensure spot is in grid
        if spot not in grid_prices:
            grid_prices.append(spot)
            grid_prices.sort()

    now = datetime.now(timezone.utc)

    # Compute dealer delta at each grid price
    profile = []
    for S_hyp in grid_prices:
        net_dd = 0.0
        call_dd = 0.0
        put_dd = 0.0
        for exp_str, chain in cached_chains.items():
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            T = max((exp_date - now).days / 365.0, 0.5 / 365)
            for opt_type, df_chain, in [('call', chain.calls), ('put', chain.puts)]:
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
                    delta = bs_delta(S_hyp, strike, T, rfr, iv, opt_type)
                    dd = -delta * oi * 100
                    net_dd += dd
                    if opt_type == 'call':
                        call_dd += dd
                    else:
                        put_dd += dd

        profile.append({
            'price_ibit': float(S_hyp),
            'price_btc': float(S_hyp / ref_per_share) if is_crypto else float(S_hyp),
            'net_dealer_delta': float(net_dd),
            'call_dealer_delta': float(call_dd),
            'put_dealer_delta': float(put_dd),
        })

    # Delta flip detection
    flip_points = []
    for i in range(len(profile) - 1):
        d1 = profile[i]['net_dealer_delta']
        d2 = profile[i + 1]['net_dealer_delta']
        p1 = profile[i]['price_ibit']
        p2 = profile[i + 1]['price_ibit']
        if (d1 < 0 and d2 > 0) or (d1 > 0 and d2 < 0):
            if (d2 - d1) != 0:
                flip = p1 + (p2 - p1) * (-d1) / (d2 - d1)
                from_dir = 'BUY' if d1 < 0 else 'SELL'
                to_dir = 'SELL' if d1 < 0 else 'BUY'
                flip_points.append({
                    'price_ibit': float(flip),
                    'price_btc': float(flip / ref_per_share) if is_crypto else float(flip),
                    'from_direction': from_dir,
                    'to_direction': to_dir,
                })

    # Hedging acceleration (rate of change of delta)
    for i in range(len(profile)):
        if i == 0 or i == len(profile) - 1:
            profile[i]['acceleration'] = 0.0
            profile[i]['accel_class'] = 'LOW'
            continue
        dp = profile[i + 1]['price_ibit'] - profile[i - 1]['price_ibit']
        if dp > 0:
            dd_change = abs(profile[i + 1]['net_dealer_delta'] - profile[i - 1]['net_dealer_delta'])
            accel = dd_change / dp
        else:
            accel = 0.0
        profile[i]['acceleration'] = float(accel)
        if accel > 50000:
            profile[i]['accel_class'] = 'HIGH'
        elif accel > 10000:
            profile[i]['accel_class'] = 'MODERATE'
        else:
            profile[i]['accel_class'] = 'LOW'

    # Max acceleration price
    max_accel_idx = max(range(len(profile)), key=lambda i: profile[i]['acceleration']) if profile else 0
    max_accel_price = profile[max_accel_idx]['price_ibit'] if profile else spot

    # Key level deltas
    key_level_deltas = {}
    for name, strike in [('call_wall', cw), ('put_wall', pw), ('gamma_flip', gf), ('max_pain', mp)]:
        closest = min(profile, key=lambda p: abs(p['price_ibit'] - strike)) if profile else None
        key_level_deltas[name] = closest['net_dealer_delta'] if closest else None

    return {
        'profile': profile,
        'flip_points': flip_points,
        'max_acceleration_price': float(max_accel_price),
        'key_level_deltas': key_level_deltas,
    }


def generate_dealer_delta_briefing(scenarios, spot, levels, ref_per_share, is_crypto):
    """Generate plain English briefing from dealer delta scenario analysis."""
    profile = scenarios.get('profile', [])
    flip_points = scenarios.get('flip_points', [])
    key_level_deltas = scenarios.get('key_level_deltas', {})

    if not profile:
        return None

    # Current delta (closest to spot)
    spot_entry = min(profile, key=lambda p: abs(p['price_ibit'] - spot))
    cur_delta = spot_entry['net_dealer_delta']
    if cur_delta < 0:
        current_delta = f"Dealers are net SHORT {abs(cur_delta/1e6):.1f}M shares — must BUY on dips (supportive)"
    else:
        current_delta = f"Dealers are net LONG {abs(cur_delta/1e6):.1f}M shares — must SELL on rallies (resistive)"

    # Flip summary
    if flip_points:
        fp = flip_points[0]
        price_str = f"${fp['price_btc']:,.0f}" if is_crypto else f"${fp['price_ibit']:.2f}"
        flip_summary = f"Dealer pressure flips from {fp['from_direction'].lower()}ing to {fp['to_direction'].lower()}ing at {price_str}"
    else:
        direction = "buyers" if cur_delta < 0 else "sellers"
        flip_summary = f"No flip — dealers are net {direction} across entire range"

    # Acceleration zone
    max_accel_price = scenarios.get('max_acceleration_price', spot)
    accel_btc = max_accel_price / ref_per_share if is_crypto else max_accel_price
    price_str = f"${accel_btc:,.0f}" if is_crypto else f"${max_accel_price:.2f}"
    acceleration_zone = f"Heaviest rebalancing pressure near {price_str} — dealers most reactive here"

    # Range bias
    lo_entry = profile[0]
    hi_entry = profile[-1]
    lo_price = f"${lo_entry['price_btc']:,.0f}" if is_crypto else f"${lo_entry['price_ibit']:.2f}"
    hi_price = f"${hi_entry['price_btc']:,.0f}" if is_crypto else f"${hi_entry['price_ibit']:.2f}"
    lo_dir = "buying" if lo_entry['net_dealer_delta'] < 0 else "selling"
    hi_dir = "selling" if hi_entry['net_dealer_delta'] > 0 else "buying"
    range_bias = f"Below {lo_price}: dealer {lo_dir} pressure. Above {hi_price}: dealer {hi_dir} pressure."

    # Morning take
    parts = []
    if cur_delta < 0:
        parts.append(f"Dealers are net short {abs(cur_delta/1e6):.1f}M shares at spot, creating a bid under the market")
    else:
        parts.append(f"Dealers are net long {abs(cur_delta/1e6):.1f}M shares at spot, creating selling pressure")
    if flip_points:
        fp = flip_points[0]
        dist_pct = ((fp['price_ibit'] - spot) / spot) * 100
        parts.append(f"pressure flips {dist_pct:+.1f}% from here")
    morning_take = "; ".join(parts) + "."

    return {
        'current_delta': current_delta,
        'flip_summary': flip_summary,
        'acceleration_zone': acceleration_zone,
        'range_bias': range_bias,
        'morning_take': morning_take,
        'key_level_deltas': key_level_deltas,
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


def compute_significant_levels(df, spot, levels, prev_strikes, is_crypto, ref_per_share, etf_flows=None):
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

        net_gex_w = row['net_gex_w']
        call_oi, put_oi, total_oi = int(row['call_oi']), int(row['put_oi']), int(row['total_oi'])
        is_major = total_oi > oi_90

        if put_oi > call_oi * 1.5 and net_gex_w < 0:
            ltype = 'put_wall'
        elif call_oi > put_oi * 1.5 and net_gex_w > 0:
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

        # Flow modifier
        if etf_flows and etf_flows['strength'] in ('moderate', 'strong'):
            if ltype == 'put_wall' and etf_flows['direction'] == 'inflow':
                behavior += " + inflows reinforce"
            elif ltype == 'put_wall' and etf_flows['direction'] == 'outflow':
                behavior += " — outflows weaken"
            elif ltype == 'call_wall' and etf_flows['direction'] == 'outflow':
                behavior += " + outflows reinforce"
            elif ltype == 'call_wall' and etf_flows['direction'] == 'inflow':
                behavior += " — inflows pressure"

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
            'btc': float(strike / ref_per_share) if is_crypto else float(strike),
            'type': ltype,
            'call_oi': call_oi, 'put_oi': put_oi, 'total_oi': total_oi,
            'net_gex': float(row['net_gex']),
            'net_gex_w': float(net_gex_w),
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


def compute_breakout(df, spot, levels, expected_move, prev_strikes, ref_per_share, etf_flows=None):
    """Compute breakout signals for both directions."""
    cw = levels['call_wall']
    pw = levels['put_wall']
    regime = levels['regime']

    up_signals, down_signals = [], []

    # Wall asymmetry (weighted)
    cw_gex = float(df[df['strike'] == cw]['call_gex_w'].sum()) if cw in df['strike'].values else 0
    pw_gex = float(abs(df[df['strike'] == pw]['put_gex_w'].sum())) if pw in df['strike'].values else 0
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

    # Expected move vs range (non-directional — tracked separately)
    em_note = None
    if expected_move and cw > pw:
        em_width = expected_move['pct'] * 2
        range_width = ((cw - pw) / spot) * 100
        if range_width > 0.5 and em_width > range_width:
            em_note = f"Expected move ({em_width:.1f}%) > range ({range_width:.1f}%) — breakout likely"

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

    # ETF fund flow signals
    if etf_flows:
        if etf_flows['streak'] >= 3 and etf_flows['strength'] in ('moderate', 'strong'):
            up_signals.append(f"ETF inflow streak ({etf_flows['streak']}d) — institutional accumulation")
        elif etf_flows['streak'] <= -3 and etf_flows['strength'] in ('moderate', 'strong'):
            down_signals.append(f"ETF outflow streak ({abs(etf_flows['streak'])}d) — institutional distribution")
        if etf_flows['avg_flow_5d'] > 100_000_000:
            up_signals.append(f"5d avg inflow ${etf_flows['avg_flow_5d']/1e6:.0f}M — sustained buying pressure")
        elif etf_flows['avg_flow_5d'] < -100_000_000:
            down_signals.append(f"5d avg outflow ${abs(etf_flows['avg_flow_5d'])/1e6:.0f}M — sustained selling pressure")

    # Targets (weighted)
    up_targets = df[(df['strike'] > cw) & (df['net_gex_w'] > 0)].nlargest(2, 'net_gex_w')[
        ['strike', 'total_oi', 'net_gex_w']].to_dict('records')
    down_targets = df[(df['strike'] < pw) & (df['net_gex_w'] < 0)].nsmallest(2, 'net_gex_w')[
        ['strike', 'total_oi', 'net_gex_w']].to_dict('records')

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
                        'btc': float(t['strike'] / ref_per_share),
                        'total_oi': int(t['total_oi']),
                        'net_gex': float(t['net_gex_w'])} for t in up_targets],
        'down_targets': [{'strike': float(t['strike']),
                          'btc': float(t['strike'] / ref_per_share),
                          'total_oi': int(t['total_oi']),
                          'net_gex': float(t['net_gex_w'])} for t in down_targets],
        'bias': bias,
        'em_note': em_note,
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
    """Background thread: backfill candles on startup for all tickers,
    pre-fetch all DTEs, then periodically update candles (every 5 min) and re-check GEX data."""
    # Phase 0: Backfill ETF flow history (estimates for chart, real data accumulates daily)
    for tk in TICKER_CONFIG:
        try:
            backfill_etf_flows(tk, days=90)
        except Exception as e:
            print(f"  [bg-refresh] ETF flow backfill {tk} error: {e}")

    # Phase 1: Backfill candles for all tickers
    for tk, cfg in TICKER_CONFIG.items():
        symbol = cfg['binance_symbol']
        _candle_backfill_done.setdefault(symbol, threading.Event())
        print(f"  [bg-refresh] Starting {symbol} candle backfill...")
        for tf in ('15m', '1h', '4h', '1d'):
            try:
                backfill_btc_candles(symbol, tf, days=90)
            except Exception as e:
                print(f"  [bg-refresh] Backfill {symbol}/{tf} error: {e}")
        _candle_backfill_done[symbol].set()
        print(f"  [bg-refresh] {symbol} candle backfill complete")

    # Phase 2: Main loop — GEX refresh + candle updates
    last_candle_update = 0
    while True:
        # Update candles every 5 minutes for all tickers
        now = time.time()
        if now - last_candle_update >= 300:
            for tk, cfg in TICKER_CONFIG.items():
                symbol = cfg['binance_symbol']
                for tf in ('15m', '1h', '4h', '1d'):
                    try:
                        update_btc_candles(symbol, tf)
                    except Exception as e:
                        print(f"  [bg-refresh] Candle update {symbol}/{tf} error: {e}")
            last_candle_update = now

        # ETF flow snapshots for all tickers
        for tk in TICKER_CONFIG:
            try:
                fetch_etf_flows(tk)
            except Exception as e:
                print(f"  [bg-refresh] ETF flow {tk} error: {e}")

        # GEX refresh logic for all tickers
        for tk, cfg in TICKER_CONFIG.items():
            today = datetime.now().strftime('%Y-%m-%d')
            all_fresh = True
            stale_dtes = []
            for dte in REFRESH_DTES:
                cache_date, _ = get_latest_cache(tk, dte)
                if cache_date != today:
                    stale_dtes.append(dte)
            if stale_dtes:
                all_fresh = False
                with ThreadPoolExecutor(max_workers=len(stale_dtes)) as pool:
                    futures = {pool.submit(fetch_with_cache, tk, d): d for d in stale_dtes}
                    for fut in as_completed(futures):
                        try:
                            fut.result()
                        except Exception as e:
                            print(f"  [bg-refresh] {tk} DTE {futures[fut]} error: {e}")

            if all_fresh:
                # Auto-run AI analysis if not cached, or re-run if ref asset moved >2%
                cached_analysis = get_cached_analysis(tk)
                should_run = False
                if not cached_analysis:
                    should_run = True
                elif cached_analysis.get('_btc_price'):
                    try:
                        current_price = yf.Ticker(cfg['ref_ticker']).info.get('regularMarketPrice')
                        if current_price:
                            old_price = cached_analysis['_btc_price']
                            pct_move = abs(current_price - old_price) / old_price * 100
                            if pct_move > 2:
                                print(f"  [bg-refresh] {cfg['asset_label']} moved {pct_move:.1f}% since last {tk} analysis (${old_price:,.0f} -> ${current_price:,.0f})")
                                should_run = True
                    except Exception:
                        pass
                if should_run:
                    try:
                        print(f"  [bg-refresh] Running {tk} AI analysis...")
                        run_analysis(tk)
                        print(f"  [bg-refresh] {tk} AI analysis complete and cached")
                    except Exception as e:
                        print(f"  [bg-refresh] {tk} AI analysis error: {e}")

        # Sleep 5 min (candle update cadence), but also covers GEX re-check
        time.sleep(300)


def start_bg_refresh():
    t = threading.Thread(target=_bg_refresh, daemon=True)
    t.start()
    print("  [bg-refresh] Background data refresh started")


# ── ROUTES ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/candles')
@app.route('/api/btc-candles')
def api_candles():
    ticker = request.args.get('ticker', 'IBIT').upper()
    cfg = TICKER_CONFIG.get(ticker)
    if not cfg:
        return Response(json.dumps({'error': f'Unknown ticker: {ticker}'}), mimetype='application/json'), 400
    symbol = cfg['binance_symbol']
    tf = request.args.get('tf', '15m')
    if tf not in ('15m', '1h', '4h', '1d'):
        return Response(json.dumps({'error': f'Invalid tf: {tf}'}), mimetype='application/json'), 400
    evt = _candle_backfill_done.get(symbol)
    if not evt or not evt.is_set():
        return Response(json.dumps({'error': 'Backfill in progress'}), mimetype='application/json'), 503
    candles = get_btc_candles(symbol, tf)
    return Response(json.dumps(candles), mimetype='application/json')


@app.route('/api/flows')
def api_flows():
    ticker = request.args.get('ticker', 'IBIT').upper()
    if ticker not in TICKER_CONFIG:
        return Response(json.dumps({'error': f'Unknown ticker: {ticker}'}), mimetype='application/json'), 400
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT date, daily_flow_dollars, shares_outstanding, aum, nav FROM etf_flows WHERE ticker=? ORDER BY date DESC LIMIT 30', (ticker,))
    rows = c.fetchall()
    conn.close()
    rows.reverse()
    data = [{'date': r[0], 'flow': r[1], 'shares': r[2], 'aum': r[3], 'nav': r[4]} for r in rows]
    return Response(json.dumps(data), mimetype='application/json')


@app.route('/api/data')
def api_data():
    try:
        ticker = request.args.get('ticker', 'IBIT').upper()
        if ticker not in TICKER_CONFIG:
            return Response(json.dumps({'error': f'Unknown ticker: {ticker}'}), mimetype='application/json'), 400
        dte = request.args.get('dte', 7, type=int)
        dte = max(1, min(dte, 90))
        try:
            data = fetch_with_cache(ticker, dte)
        except (ValueError, KeyError):
            # Serve most recent cached data for this DTE if available
            _, cached = get_latest_cache(ticker, dte)
            if cached:
                data = cached
                data['stale'] = True
            elif dte != 7:
                data = fetch_with_cache(ticker, 7)
                data['fallback_from'] = dte
            else:
                raise
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


def set_cached_analysis(ticker, analysis, btc_price=None):
    """Cache AI analysis for today, with the BTC price at analysis time."""
    if btc_price is not None:
        analysis['_btc_price'] = btc_price
        analysis['_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    conn = get_db()
    c = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    c.execute('INSERT OR REPLACE INTO analysis_cache (date, ticker, analysis_json) VALUES (?,?,?)',
              (today, ticker, json.dumps(analysis)))
    conn.commit()
    conn.close()


def run_analysis(ticker='IBIT'):
    """Run AI analysis across all DTEs. Returns analysis dict or raises."""
    cfg = TICKER_CONFIG.get(ticker)
    if not cfg:
        raise ValueError(f'Unknown ticker: {ticker}')
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise RuntimeError('ANTHROPIC_API_KEY not set')

    dtes = [3, 7, 14, 30, 45]
    results = {}

    # Fetch live ref asset price so the AI sees current price, not stale cache
    live_ref_price = None
    try:
        live_ref_price = yf.Ticker(cfg['ref_ticker']).info.get('regularMarketPrice')
    except Exception:
        pass

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
        current_btc = live_ref_price if live_ref_price else d['btc_spot']
        summaries[f"{dte}d"] = {
            'spot_btc': round(current_btc),
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
            'active_gex_total': lvl.get('active_gex_total', 0),
            'net_gex_w_total': lvl.get('net_gex_w_total', 0),
            'level_trajectory': lvl.get('level_trajectory', {}),
            'active_call_wall_btc': round(lvl.get('active_call_wall', lvl['call_wall']) / bps),
            'active_put_wall_btc': round(lvl.get('active_put_wall', lvl['put_wall']) / bps),
            'positioning_confidence': lvl.get('positioning_confidence', 100),
            'positioning_warnings': lvl.get('positioning_warnings', []),
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
            'etf_flows': {
                'daily_flow_millions': round(d['etf_flows']['daily_flow_dollars'] / 1e6, 1),
                'direction': d['etf_flows']['direction'],
                'strength': d['etf_flows']['strength'],
                'streak': d['etf_flows']['streak'],
                'avg_5d_millions': round(d['etf_flows']['avg_flow_5d'] / 1e6, 1),
            } if d.get('etf_flows') else None,
            'significant_levels': [
                {k: v for k, v in sl.items() if k != 'greeks_note'}
                for sl in d['significant_levels'][:8]
            ],
            'dealer_delta_briefing': d.get('dealer_delta_briefing'),
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

    asset = cfg['asset_label']
    system_prompt = f"""You are a GEX (Gamma Exposure) trading analyst for {cfg['name']} ({asset} ETF). You analyze options flow data across multiple DTE timeframes to provide actionable trading insights.

IMPORTANT: {cfg['name']} is a {asset} ETF proxy. The data contains both {cfg['name']} share prices and {asset}-equivalent prices (in levels_btc). ALWAYS use {asset} prices in your analysis (e.g. "$65,200" not "$37.0"). Use the levels_btc fields for all price references. You can convert any {cfg['name']} price to {asset} by dividing by btc_per_share.

If a "changes_vs_prev" field is present for a timeframe, it contains day-over-day changes — use this to highlight what shifted overnight:
- Level migrations (walls, gamma flip moving up/down)
- Regime flips (positive ↔ negative gamma)
- GEX and P/C ratio shifts
- Spot movement relative to level changes
This context is critical — a static snapshot is less useful than understanding the direction of positioning changes.

Active GEX (active_gex_total) weights net GEX by the fraction of OI that is new since yesterday. It surfaces where FRESH dealer exposure is concentrated vs stale "zombie gamma" from old positions. Active call/put walls show where the freshest positioning is strongest.

Positioning confidence (positioning_confidence, 0-100%) indicates how much to trust the GEX sign convention (dealers long calls, short puts). When below 60%, the call wall may act as a squeeze trigger rather than resistance, and regime labels may be inverted. Always note the confidence level and any warnings in your analysis.

ETF fund flows (etf_flows) show daily creation/redemption activity. Positive = institutional inflows, negative = outflows. A streak of 3+ days in one direction is significant. Cross-reference with GEX regime: inflows + positive gamma = strong range, outflows + negative gamma = breakdown risk.

Weighted GEX (net_gex_w_total) applies DTE-based time weighting: 1/sqrt(DTE/max_DTE). Near-term expirations contribute disproportionately more gamma exposure. This makes levels more responsive to imminent expiry dynamics. Use net_gex_w_total as the primary GEX measure. The level_trajectory field shows whether key levels (call_wall, put_wall, gamma_flip) are STRENGTHENING (>10% increase), WEAKENING (>10% decrease), or STABLE vs the previous session. It also identifies the dominant_expiry driving each level — if a wall is dominated by a near-term expiry, it may evaporate quickly after that expiration.

Dealer Delta Scenario Analysis (dealer_delta_briefing): Pre-computed dealer hedging pressure at hypothetical price levels across the key level grid. 'current_delta' shows dealer positioning at spot. 'flip_summary' identifies where dealer pressure reverses direction. 'acceleration_zone' shows where dealers are most reactive to price moves. Negative dealer delta = dealers must BUY to hedge = supportive (acts as a bid). Positive dealer delta = dealers must SELL = resistive (acts as an offer). Use this to identify price levels where dealer hedging creates natural support or resistance, independent of the GEX profile.

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
    user_content = f"Analyze the following {cfg['name']} GEX data across all timeframes:\n\n{prompt_data}"
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
    # Store BTC price at analysis time for staleness checks
    btc_price = live_ref_price
    if not btc_price:
        for dte in dtes:
            if dte in results and results[dte].get('btc_spot'):
                btc_price = results[dte]['btc_spot']
                break
    set_cached_analysis(ticker, analysis, btc_price)
    return analysis


@app.route('/api/analysis')
def api_analysis():
    """GET cached analysis. Returns today's if available, otherwise most recent."""
    ticker = request.args.get('ticker', 'IBIT').upper()
    if ticker not in TICKER_CONFIG:
        return Response(json.dumps({'error': f'Unknown ticker: {ticker}'}), mimetype='application/json'), 400
    cached = get_cached_analysis(ticker)
    if cached:
        return Response(json.dumps(cached), mimetype='application/json')
    # Fall back to most recent analysis (e.g. overnight before new data arrives)
    prev_date, prev = get_prev_analysis(ticker)
    if prev:
        return Response(json.dumps(prev), mimetype='application/json')
    return Response(json.dumps({'status': 'pending'}), mimetype='application/json')


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Force re-run analysis (manual refresh)."""
    ticker = request.args.get('ticker', 'IBIT').upper()
    if ticker not in TICKER_CONFIG:
        return Response(json.dumps({'error': f'Unknown ticker: {ticker}'}), mimetype='application/json'), 400
    try:
        analysis = run_analysis(ticker)
        return Response(json.dumps(analysis), mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({'error': str(e)}),
                        mimetype='application/json'), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', '-p', type=int, default=5000)
    parser.add_argument('--host', default='127.0.0.1')
    args = parser.parse_args()
    init_db()
    print(f"\n  GEX Terminal → http://{args.host}:{args.port}\n")
    # Start background refresh (skip reloader parent to avoid double threads)
    if os.environ.get('WERKZEUG_RUN_MAIN') or not app.debug:
        start_bg_refresh()
    app.run(host=args.host, port=args.port, debug=True)