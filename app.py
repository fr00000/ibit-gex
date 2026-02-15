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
import re
import argparse
import sqlite3
import threading
import time
import urllib.request
from html.parser import HTMLParser
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

# Non-overlapping DTE windows: each shows distinct positioning
# (label, min_dte, max_dte)
DTE_WINDOWS = [
    (3,  0,  3),   # This week's expirations — immediate hedging
    (7,  4,  7),   # Next week's setup
    (14, 8,  14),  # Two weeks out
    (30, 15, 30),  # Monthly cycle positioning
    (45, 31, 45),  # Structural / quarterly
]

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
    # Migrate: add total_btc_etf_flow column to etf_flows
    cols = [row[1] for row in c.execute('PRAGMA table_info(etf_flows)').fetchall()]
    if 'total_btc_etf_flow' not in cols:
        c.execute('ALTER TABLE etf_flows ADD COLUMN total_btc_etf_flow REAL')
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
         levels.get('net_gex_total')))
    for _, row in df.iterrows():
        c.execute('''INSERT OR REPLACE INTO strike_history (date,ticker,strike,call_oi,put_oi,total_oi,net_gex,weighted_net_gex)
            VALUES (?,?,?,?,?,?,?,?)''',
            (date_str, ticker, row['strike'], int(row['call_oi']), int(row['put_oi']),
             int(row['total_oi']), row['net_gex'], row['net_gex']))
    conn.commit()


def get_history(conn, ticker, days=10):
    c = conn.cursor()
    c.execute('''SELECT date, spot, btc_price, gamma_flip, call_wall, put_wall,
                        max_pain, regime, net_gex, total_call_oi, total_put_oi, weighted_net_gex
                 FROM snapshots WHERE ticker=? ORDER BY date DESC LIMIT ?''', (ticker, days))
    return c.fetchall()


def summarize_history_trends(conn, ticker, btc_per_share, current_levels):
    """Compress 10 days of snapshots into trend signals for AI analysis."""
    history = get_history(conn, ticker, 10)
    if not history or len(history) < 2:
        return None

    # history rows: (date, spot, btc_price, gamma_flip, call_wall, put_wall,
    #                max_pain, regime, net_gex, total_call_oi, total_put_oi, weighted_net_gex)
    # Rows are DESC order (newest first)
    n = len(history)

    # ── Regime streak ──
    current_regime = history[0][7]
    streak = 1
    for i in range(1, n):
        if history[i][7] == current_regime:
            streak += 1
        else:
            break
    prior_regime = history[streak][7] if streak < n else None
    regime_abbrev = ['pos' if h[7] == 'positive_gamma' else 'neg' for h in history]

    regime_streak = {
        'current_regime': current_regime,
        'streak_days': streak,
        'prior_regime': prior_regime,
        'regime_history': regime_abbrev,
    }

    # ── Helper: per-share ratio for each row ──
    def row_bps(h):
        """btc_per_share for a historical row: spot_ibit / btc_price."""
        if h[2] and h[2] > 0 and h[1] and h[1] > 0:
            return h[1] / h[2]
        return btc_per_share

    # ── Helper: direction classification ──
    def classify_direction(values_newest_first, current_price_btc):
        """Classify direction from a list of values (newest first)."""
        if len(values_newest_first) < 2:
            return 'stable', 0, None
        recent = values_newest_first[:min(3, len(values_newest_first))]
        net_change = recent[0] - recent[-1]
        threshold = current_price_btc * 0.01 if current_price_btc > 0 else 0
        if abs(net_change) <= threshold:
            avg_val = sum(recent) / len(recent)
            return 'stable', 0, round(avg_val)
        elif net_change > 0:
            return 'rising', 0, None
        else:
            return 'falling', 0, None

    def consecutive_dir(values_newest_first):
        """Count consecutive days moving in the same direction (newest first)."""
        if len(values_newest_first) < 2:
            return 0
        first_dir = values_newest_first[0] - values_newest_first[1]
        if first_dir == 0:
            return 0
        count = 1
        for i in range(1, len(values_newest_first) - 1):
            diff = values_newest_first[i] - values_newest_first[i + 1]
            if (diff > 0 and first_dir > 0) or (diff < 0 and first_dir < 0):
                count += 1
            else:
                break
        return count

    # ── Level migration ──
    current_bps = btc_per_share
    level_migration = {}
    # (field_name, history_index, current_value)
    level_defs = [
        ('call_wall', 4, current_levels.get('call_wall', 0)),
        ('put_wall', 5, current_levels.get('put_wall', 0)),
        ('gamma_flip', 3, current_levels.get('gamma_flip', 0)),
    ]
    for lname, hidx, cur_ibit in level_defs:
        # Convert each day's value to BTC
        btc_vals = []
        for h in history:
            bps_h = row_bps(h)
            val = h[hidx]
            if val and bps_h > 0:
                btc_vals.append(round(val / bps_h))
            else:
                btc_vals.append(None)

        # Filter Nones
        valid = [v for v in btc_vals if v is not None]
        cur_btc = round(cur_ibit / current_bps) if current_bps > 0 else 0

        if len(valid) < 2:
            level_migration[lname] = {
                'current_btc': cur_btc,
                'direction': 'stable',
                'change_3d_btc': None,
                'change_5d_btc': None,
                'held_at_btc': cur_btc,
                'consecutive_direction': 0,
            }
            continue

        direction, _, held_at = classify_direction(valid, cur_btc)
        consec = consecutive_dir(valid)

        change_3d = valid[0] - valid[min(2, len(valid) - 1)] if len(valid) >= 2 else None
        change_5d = valid[0] - valid[min(4, len(valid) - 1)] if len(valid) >= 5 else None

        level_migration[lname] = {
            'current_btc': cur_btc,
            'direction': direction,
            'change_3d_btc': change_3d,
            'change_5d_btc': change_5d,
            'held_at_btc': held_at if direction == 'stable' else None,
            'consecutive_direction': consec,
        }

    # ── GEX trend ──
    def pct_change_nd(values, nd):
        """Percentage change over nd days (values newest first)."""
        if len(values) <= nd:
            return None
        old = values[nd]
        if old and old != 0:
            return round(((values[0] - old) / abs(old)) * 100, 1)
        return None

    def gex_direction(pct):
        if pct is None:
            return 'flat'
        if pct > 10:
            return 'building'
        elif pct < -10:
            return 'decaying'
        return 'flat'

    net_gex_vals = [h[8] for h in history if h[8] is not None]
    net_gex_3d_pct = pct_change_nd(net_gex_vals, min(3, len(net_gex_vals) - 1)) if len(net_gex_vals) >= 2 else None

    gex_trend = {
        'net_gex_direction': gex_direction(net_gex_3d_pct),
        'net_gex_3d_change_pct': net_gex_3d_pct,
    }

    # ── OI trend ──
    call_oi_vals = [h[9] for h in history if h[9] is not None]
    put_oi_vals = [h[10] for h in history if h[10] is not None]

    call_3d_pct = pct_change_nd(call_oi_vals, min(3, len(call_oi_vals) - 1)) if len(call_oi_vals) >= 2 else None
    put_3d_pct = pct_change_nd(put_oi_vals, min(3, len(put_oi_vals) - 1)) if len(put_oi_vals) >= 2 else None

    # PCR trend
    pcr_vals = []
    for h in history:
        if h[9] and h[10] and h[9] > 0:
            pcr_vals.append(h[10] / h[9])
    pcr_trend = 'stable'
    if len(pcr_vals) >= 2:
        pcr_diff = pcr_vals[0] - pcr_vals[-1]
        if pcr_diff > 0.05:
            pcr_trend = 'rising'
        elif pcr_diff < -0.05:
            pcr_trend = 'falling'

    oi_trend = {
        'total_call_oi_direction': gex_direction(call_3d_pct),
        'total_put_oi_direction': gex_direction(put_3d_pct),
        'call_oi_3d_change_pct': call_3d_pct,
        'put_oi_3d_change_pct': put_3d_pct,
        'pcr_trend': pcr_trend,
    }

    # ── Range evolution ──
    cw_btc_vals = []
    pw_btc_vals = []
    for h in history:
        bps_h = row_bps(h)
        if h[4] and h[5] and bps_h > 0:
            cw_btc_vals.append(round(h[4] / bps_h))
            pw_btc_vals.append(round(h[5] / bps_h))

    range_current = cw_btc_vals[0] - pw_btc_vals[0] if cw_btc_vals and pw_btc_vals else None
    range_3d_ago = (cw_btc_vals[min(2, len(cw_btc_vals) - 1)] - pw_btc_vals[min(2, len(pw_btc_vals) - 1)]) if len(cw_btc_vals) >= 2 else None

    # Range trend
    range_trend = 'stable'
    if range_current is not None and range_3d_ago is not None:
        range_diff = range_current - range_3d_ago
        threshold = range_current * 0.1 if range_current > 0 else 0
        if range_diff > threshold:
            range_trend = 'expanding'
        elif range_diff < -threshold:
            range_trend = 'contracting'

    # Range bias
    range_bias = 'stable'
    if len(cw_btc_vals) >= 2 and len(pw_btc_vals) >= 2:
        cw_chg = cw_btc_vals[0] - cw_btc_vals[-1]
        pw_chg = pw_btc_vals[0] - pw_btc_vals[-1]
        cw_thresh = cw_btc_vals[0] * 0.01 if cw_btc_vals[0] > 0 else 0
        pw_thresh = pw_btc_vals[0] * 0.01 if pw_btc_vals[0] > 0 else 0
        cw_rising = cw_chg > cw_thresh
        cw_falling = cw_chg < -cw_thresh
        pw_rising = pw_chg > pw_thresh
        pw_falling = pw_chg < -pw_thresh

        if cw_rising and pw_rising:
            range_bias = 'upward_shift'
        elif cw_falling and pw_falling:
            range_bias = 'downward_shift'
        elif cw_rising and pw_falling:
            range_bias = 'expanding_symmetric'
        elif cw_falling and pw_rising:
            range_bias = 'contracting'
        elif cw_rising and not pw_rising and not pw_falling:
            range_bias = 'expanding_upward'
        elif cw_falling and not pw_rising and not pw_falling:
            range_bias = 'contracting_from_above'
        elif pw_falling and not cw_rising and not cw_falling:
            range_bias = 'expanding_downward'
        elif pw_rising and not cw_rising and not cw_falling:
            range_bias = 'contracting_from_below'

    range_evolution = {
        'range_width_current_btc': range_current,
        'range_width_3d_ago_btc': range_3d_ago,
        'range_trend': range_trend,
        'range_bias': range_bias,
    }

    # ── Narrative ──
    regime_str = 'Positive gamma' if current_regime == 'positive_gamma' else 'Negative gamma'
    parts = [f"{regime_str} regime for {streak} day{'s' if streak != 1 else ''}"]

    # Dominant level movement
    cw_mig = level_migration.get('call_wall', {})
    pw_mig = level_migration.get('put_wall', {})
    if cw_mig.get('direction') == 'rising' and pw_mig.get('direction') == 'rising':
        chg = cw_mig.get('change_5d_btc') or cw_mig.get('change_3d_btc')
        suffix = f" +${chg:,}" if chg else ""
        nd = '5d' if cw_mig.get('change_5d_btc') else '3d'
        parts.append(f"walls migrating upward{suffix} over {nd}")
    elif cw_mig.get('direction') == 'falling' and pw_mig.get('direction') == 'falling':
        chg = cw_mig.get('change_5d_btc') or cw_mig.get('change_3d_btc')
        suffix = f" {chg:,}" if chg else ""
        nd = '5d' if cw_mig.get('change_5d_btc') else '3d'
        parts.append(f"walls migrating downward{suffix} over {nd}")
    elif range_trend == 'contracting':
        parts.append("range contracting")
    elif range_trend == 'expanding':
        parts.append("range expanding")
    elif cw_mig.get('direction') != 'stable' or pw_mig.get('direction') != 'stable':
        moves = []
        if cw_mig.get('direction') == 'rising':
            moves.append("call wall rising")
        elif cw_mig.get('direction') == 'falling':
            moves.append("call wall decaying")
        if pw_mig.get('direction') == 'rising':
            moves.append("put wall rising")
        elif pw_mig.get('direction') == 'falling':
            moves.append("put wall falling")
        if moves:
            parts.append(", ".join(moves))

    # OI context
    if oi_trend['total_call_oi_direction'] == 'building' and oi_trend['total_put_oi_direction'] == 'building':
        parts.append("OI building on both sides")
    elif oi_trend['total_call_oi_direction'] == 'decaying' and oi_trend['total_put_oi_direction'] == 'decaying':
        parts.append("OI unwinding")
    elif oi_trend['total_call_oi_direction'] == 'building':
        parts.append("call OI building")
    elif oi_trend['total_put_oi_direction'] == 'building':
        parts.append("put OI building")

    narrative = " — ".join(parts[:3])

    return {
        'regime_streak': regime_streak,
        'level_migration': level_migration,
        'gex_trend': gex_trend,
        'oi_trend': oi_trend,
        'range_evolution': range_evolution,
        'narrative': narrative,
    }


class _FarsideTableParser(HTMLParser):
    """Parse the Farside Investors BTC ETF flow HTML table."""
    def __init__(self):
        super().__init__()
        self.headers = []
        self.rows = []
        self._current_row = []
        self._current_cell = ''
        self._in_cell = False
        self._in_thead = False

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get('class', '')
        if tag == 'table' and 'etf' in cls:
            self._in_thead = False
        if tag == 'thead':
            self._in_thead = True
        if tag in ('td', 'th'):
            self._in_cell = True
            self._current_cell = ''

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self._in_cell:
            self._in_cell = False
            text = self._current_cell.strip()
            self._current_row.append(text)
        if tag == 'thead':
            self._in_thead = False
        if tag == 'tr' and self._current_row:
            if self._in_thead or (self._current_row and not self.headers):
                self.headers = self._current_row
            else:
                self.rows.append(self._current_row)
            self._current_row = []

    def handle_data(self, data):
        if self._in_cell:
            self._current_cell += data


def _parse_farside_value(text):
    """Parse a Farside flow cell: '1,113.7' -> 1113.7, '(157.6)' -> -157.6, '-' -> 0."""
    text = text.strip()
    if not text or text == '-':
        return 0.0
    text = text.replace(',', '')
    m = re.match(r'^\((.+)\)$', text)
    if m:
        return -float(m.group(1))
    try:
        return float(text)
    except ValueError:
        return 0.0


# Farside page cache: at most 1 fetch per hour
_farside_cache = {'html': None, 'time': 0}
_farside_lock = threading.Lock()
FARSIDE_URL = 'https://farside.co.uk/bitcoin-etf-flow-all-data/'
FARSIDE_CACHE_SECONDS = 3600

# Deribit BTC options cache: at most 1 fetch per hour
_deribit_cache = {'data': None, 'time': 0}
_deribit_lock = threading.Lock()
DERIBIT_URL = 'https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option'
DERIBIT_CACHE_SECONDS = 3600


def fetch_farside_flows():
    """Fetch BTC spot ETF daily flows from Farside Investors.
    Parses the full history page, stores all rows in the DB, and returns
    today's IBIT flow summary dict (same shape as old fetch_etf_flows)."""
    with _farside_lock:
        now = time.time()
        if _farside_cache['html'] and (now - _farside_cache['time']) < FARSIDE_CACHE_SECONDS:
            html = _farside_cache['html']
        else:
            try:
                req = urllib.request.Request(FARSIDE_URL, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                })
                with urllib.request.urlopen(req, timeout=30) as resp:
                    html = resp.read().decode('utf-8')
                _farside_cache['html'] = html
                _farside_cache['time'] = now
                print(f"  [farside] Fetched {len(html)//1024}KB from Farside")
            except Exception as e:
                print(f"  [farside] Fetch failed: {e}")
                html = _farside_cache.get('html')
                if not html:
                    return None

    # Parse the table
    parser = _FarsideTableParser()
    parser.feed(html)

    if not parser.headers or len(parser.headers) < 13:
        print(f"  [farside] Bad table structure: {len(parser.headers)} headers")
        return None

    # Find column indices
    header_names = [h.strip() for h in parser.headers]
    try:
        ibit_col = header_names.index('IBIT')
        total_col = header_names.index('Total')
    except ValueError:
        print(f"  [farside] Missing IBIT or Total column in headers: {header_names}")
        return None

    # Parse all valid date rows and store in DB
    conn = get_db()
    c = conn.cursor()
    # Clear old Yahoo-era rows (total_btc_etf_flow IS NULL) on first Farside load
    c.execute('DELETE FROM etf_flows WHERE ticker=? AND total_btc_etf_flow IS NULL', ('IBIT',))
    inserted = 0
    for row in parser.rows:
        if len(row) < max(ibit_col, total_col) + 1:
            continue
        date_str_raw = row[0].strip()
        # Skip summary rows (Total, Average, Maximum, Minimum)
        try:
            dt = datetime.strptime(date_str_raw, "%d %b %Y")
            date_str = dt.strftime('%Y-%m-%d')
        except ValueError:
            continue

        ibit_flow = _parse_farside_value(row[ibit_col]) * 1_000_000
        total_flow = _parse_farside_value(row[total_col]) * 1_000_000

        c.execute('''INSERT OR REPLACE INTO etf_flows
            (date, ticker, shares_outstanding, aum, nav, daily_flow_shares, daily_flow_dollars, total_btc_etf_flow)
            VALUES (?,?,?,?,?,?,?,?)''',
            (date_str, 'IBIT', None, None, None, 0, ibit_flow, total_flow))
        inserted += 1
    conn.commit()

    # Read back recent data for streak/momentum calculation
    c.execute('SELECT date, daily_flow_dollars, total_btc_etf_flow FROM etf_flows WHERE ticker=? ORDER BY date DESC LIMIT 5',
              ('IBIT',))
    history = c.fetchall()
    conn.close()

    if inserted > 0:
        print(f"  [farside] Stored {inserted} days of IBIT flow data")

    if not history:
        return None

    # Today's flow
    daily_flow_dollars = history[0][1] or 0.0
    total_btc_etf_flow = history[0][2] or 0.0

    # Streak: consecutive same-direction days
    streak = 0
    direction = 1 if daily_flow_dollars >= 0 else -1
    for h in history:
        flow = h[1] or 0.0
        if (flow >= 0) == (direction > 0):
            streak += 1
        else:
            break
    streak *= direction

    # 5-day average
    avg_flow_5d = sum((h[1] or 0.0) for h in history) / len(history)

    # Strength
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
        'daily_flow_shares': 0,
        'daily_flow_dollars': float(daily_flow_dollars),
        'direction': 'inflow' if daily_flow_dollars >= 0 else 'outflow',
        'strength': strength,
        'streak': int(streak),
        'avg_flow_5d': float(avg_flow_5d),
        'shares_outstanding': None,
        'total_btc_etf_flow': float(total_btc_etf_flow),
    }


def fetch_deribit_options(min_dte=0, max_dte=7):
    """Fetch BTC options from Deribit public API. Returns list of dicts with
    strike, expiry, dte, option_type, oi (in BTC), iv, underlying_price.
    Cached for 1 hour. Returns empty list on any failure."""
    with _deribit_lock:
        now = time.time()
        if _deribit_cache['data'] is not None and (now - _deribit_cache['time']) < DERIBIT_CACHE_SECONDS:
            raw = _deribit_cache['data']
        else:
            try:
                req = urllib.request.Request(DERIBIT_URL, headers={
                    'User-Agent': 'ibit-gex/1.0'
                })
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw = json.loads(resp.read().decode())
                _deribit_cache['data'] = raw
                _deribit_cache['time'] = now
                result_count = len(raw.get('result', []))
                print(f"  [deribit] Fetched {result_count} instruments from Deribit")
            except Exception as e:
                print(f"  [deribit] Fetch failed: {e}")
                raw = _deribit_cache.get('data')
                if not raw:
                    return []

    instruments = raw.get('result', [])
    if not instruments:
        return []

    now_dt = datetime.now(timezone.utc)
    options = []
    for inst in instruments:
        name = inst.get('instrument_name', '')
        # Parse: BTC-27MAR26-100000-C
        parts = name.split('-')
        if len(parts) != 4:
            continue
        try:
            expiry_date = datetime.strptime(parts[1], "%d%b%y").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        strike = float(parts[2])
        opt_type = 'call' if parts[3] == 'C' else 'put'

        dte = (expiry_date - now_dt).days
        if dte < min_dte or dte > max_dte:
            continue

        oi = inst.get('open_interest', 0)
        if oi is None or oi <= 10:
            continue

        iv = inst.get('mark_iv', 0)
        if iv is None or iv <= 0:
            continue

        underlying_price = inst.get('underlying_price', 0)
        if underlying_price is None or underlying_price <= 0:
            continue

        options.append({
            'strike': strike,
            'expiry_str': parts[1],
            'expiry_date': expiry_date,
            'dte': dte,
            'option_type': opt_type,
            'oi': float(oi),
            'iv': float(iv),
            'underlying_price': float(underlying_price),
        })

    print(f"  [deribit] {len(options)} instruments in {min_dte}-{max_dte} DTE window")
    return options


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
def _compute_levels_from_df(df, spot, etf_flows=None):
    """Compute levels (call wall, put wall, gamma flip, max pain, regime, etc.)
    from a GEX DataFrame. 'spot' is in the same units as df['strike'].
    Returns levels dict."""
    if df.empty:
        return {}
    levels = {}

    # Call wall: highest call GEX at or above spot (resistance)
    calls_above = df[df['strike'] >= spot]
    if not calls_above.empty:
        levels['call_wall'] = float(calls_above.loc[calls_above['call_gex'].idxmax(), 'strike'])
    else:
        levels['call_wall'] = float(df.loc[df['call_gex'].idxmax(), 'strike'])

    # Put wall: most negative put GEX at or below spot (support)
    puts_below = df[df['strike'] <= spot]
    if not puts_below.empty:
        levels['put_wall'] = float(puts_below.loc[puts_below['put_gex'].idxmin(), 'strike'])
    else:
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

    # Max pain (vectorized)
    strikes_arr = df['strike'].values
    call_oi_arr = df['call_oi'].values
    put_oi_arr = df['put_oi'].values
    settle_grid = strikes_arr[:, np.newaxis]
    call_pain = call_oi_arr * np.maximum(0, settle_grid - strikes_arr) * 100
    put_pain = put_oi_arr * np.maximum(0, strikes_arr - settle_grid) * 100
    total_pain = (call_pain + put_pain).sum(axis=1)
    levels['max_pain'] = float(strikes_arr[np.argmin(total_pain)])

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

    # Active GEX totals and walls
    active_col = 'active_gex' if 'active_gex' in df.columns else 'net_gex'
    levels['active_gex_total'] = float(df[active_col].sum())
    active_pos = df[df[active_col] > 0]
    active_neg = df[df[active_col] < 0]
    levels['active_call_wall'] = float(active_pos.loc[active_pos[active_col].idxmax(), 'strike']) if not active_pos.empty else levels['call_wall']
    levels['active_put_wall'] = float(active_neg.loc[active_neg[active_col].idxmin(), 'strike']) if not active_neg.empty else levels['put_wall']

    # Positioning confidence
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

    if 'call_volume' in df.columns:
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

    # Resistance / support
    levels['resistance'] = df[df['net_gex'] > 0].nlargest(3, 'net_gex')['strike'].tolist()
    levels['support'] = df[df['net_gex'] < 0].nsmallest(3, 'net_gex')['strike'].tolist()

    # OI magnets
    levels['oi_magnets'] = df.nlargest(5, 'total_oi')[['strike', 'total_oi', 'call_oi', 'put_oi']].to_dict('records')

    return levels


def fetch_and_analyze(ticker_symbol='IBIT', max_dte=7, min_dte=0):
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
    cutoff_max = now + timedelta(days=max_dte)
    cutoff_min = now + timedelta(days=min_dte)
    selected_exps = [
        e for e in all_exps
        if cutoff_min <= datetime.strptime(e, "%Y-%m-%d").replace(tzinfo=timezone.utc) <= cutoff_max
    ]
    if not selected_exps:
        # If no expirations in this window, find the nearest expiration after cutoff_min
        future_exps = [e for e in all_exps if datetime.strptime(e, "%Y-%m-%d").replace(tzinfo=timezone.utc) > cutoff_min]
        if future_exps:
            selected_exps = future_exps[:1]  # take just the nearest one
        else:
            selected_exps = all_exps[-1:]  # last resort: furthest available

    # Fetch previous day's strike data (needed for Active GEX + breakout + sig levels)
    conn = get_db()
    prev_date, prev_strikes = get_prev_strikes(conn, ticker_symbol)
    conn.close()

    etf_flows = fetch_farside_flows() if is_crypto else None

    # Collect options data
    strike_data = {}
    cached_chains = {}  # exp_str -> chain, reused for expected move
    for exp_str in selected_exps:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        T = max((exp_date - now).days / 365.0, 0.5 / 365)
        dte_days = max((exp_date - now).days, 0)
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
                dealer_delta = -delta * oi * 100
                # Dealer vanna exposure: how dealer delta changes with IV
                # Vanna is identical for calls/puts (put-call parity), and dealers
                # are short both, so no sign flip per option type.
                dealer_vanna = -vanna * oi * 100 * spot * 0.01
                # Dealer charm exposure: how dealer hedge changes overnight
                # bs_charm returns holder's dΔ/dT; dealer is short, so negate.
                # Positive dealer_charm = dealers BUY delta (bullish overnight).
                dealer_charm = -charm * oi * 100 / 365.0

                vol = row.get('volume', 0)
                if pd.isna(vol):
                    vol = 0
                vol = int(vol)

                if strike not in strike_data:
                    strike_data[strike] = {'call_oi': 0, 'put_oi': 0, 'call_gex': 0, 'put_gex': 0,
                                           'call_delta': 0, 'put_delta': 0,
                                           'call_vanna': 0, 'put_vanna': 0,
                                           'call_charm': 0, 'put_charm': 0,
                                           'call_volume': 0, 'put_volume': 0,
                                           'expiry_gex': {}}
                strike_data[strike][f'{opt_type}_oi'] += oi
                strike_data[strike][f'{opt_type}_gex'] += gex
                strike_data[strike][f'{opt_type}_volume'] += vol
                strike_data[strike][f'{opt_type}_delta'] += dealer_delta
                strike_data[strike][f'{opt_type}_vanna'] += dealer_vanna
                strike_data[strike][f'{opt_type}_charm'] += dealer_charm
                # Per-expiry breakdown
                if exp_str not in strike_data[strike]['expiry_gex']:
                    strike_data[strike]['expiry_gex'][exp_str] = {
                        'call_gex': 0, 'put_gex': 0, 'dte': dte_days
                    }
                strike_data[strike]['expiry_gex'][exp_str][f'{opt_type}_gex'] += gex

    # Build dataframe
    rows = []
    for strike, d in sorted(strike_data.items()):
        # Compute net_gex for each expiry entry
        expiry_gex = {}
        for exp_str, eg in d.get('expiry_gex', {}).items():
            eg['net_gex'] = eg['call_gex'] + eg['put_gex']
            expiry_gex[exp_str] = eg
        rows.append({
            'strike': strike,
            'btc_price': strike / ref_per_share if is_crypto else strike,
            'call_oi': d['call_oi'], 'put_oi': d['put_oi'],
            'total_oi': d['call_oi'] + d['put_oi'],
            'call_gex': d['call_gex'], 'put_gex': d['put_gex'],
            'net_gex': d['call_gex'] + d['put_gex'],
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
        active_gex_values.append(row['net_gex'] * ratio)
    df['active_gex'] = active_gex_values

    # Derive levels (IBIT-only, using extracted helper)
    levels = _compute_levels_from_df(df, spot, etf_flows)

    # Deribit options integration
    deribit_available = False
    deribit_strike_data = {}
    deribit_oi_btc = 0
    deribit_options = []
    if is_crypto and ticker_symbol == 'IBIT':
        try:
            deribit_options = fetch_deribit_options(min_dte, max_dte)
            if deribit_options:
                deribit_available = True
                for opt in deribit_options:
                    strike_btc = opt['strike']
                    oi = opt['oi']
                    iv = opt['iv'] / 100.0  # Deribit gives IV as percentage
                    btc_s = opt['underlying_price']
                    T = max(opt['dte'] / 365.0, 0.5 / 365)
                    opt_type = opt['option_type']
                    sign = 1 if opt_type == 'call' else -1

                    gamma = bs_gamma(btc_s, strike_btc, T, rfr, iv)
                    delta = bs_delta(btc_s, strike_btc, T, rfr, iv, opt_type)
                    vanna = bs_vanna(btc_s, strike_btc, T, rfr, iv)
                    charm = bs_charm(btc_s, strike_btc, T, rfr, iv, opt_type)

                    # Contract multiplier = 1 BTC (not 100 shares)
                    gex = sign * gamma * oi * 1 * btc_s ** 2 * 0.01
                    dealer_delta = -delta * oi * 1
                    dealer_vanna = -vanna * oi * 1 * btc_s * 0.01
                    dealer_charm = -charm * oi * 1 / 365.0

                    deribit_oi_btc += oi

                    if strike_btc not in deribit_strike_data:
                        deribit_strike_data[strike_btc] = {
                            'call_oi': 0, 'put_oi': 0, 'call_gex': 0, 'put_gex': 0,
                            'call_delta': 0, 'put_delta': 0,
                            'call_vanna': 0, 'put_vanna': 0,
                            'call_charm': 0, 'put_charm': 0,
                        }
                    d_entry = deribit_strike_data[strike_btc]
                    d_entry[f'{opt_type}_oi'] += oi
                    d_entry[f'{opt_type}_gex'] += gex
                    d_entry[f'{opt_type}_delta'] += dealer_delta
                    d_entry[f'{opt_type}_vanna'] += dealer_vanna
                    d_entry[f'{opt_type}_charm'] += dealer_charm
        except Exception as e:
            print(f"  [deribit] Failed: {e}")

    # Build Deribit DataFrame
    deribit_rows = []
    for strike_btc, d_entry in sorted(deribit_strike_data.items()):
        deribit_rows.append({
            'strike': strike_btc,
            'btc_price': strike_btc,
            'call_oi': d_entry['call_oi'], 'put_oi': d_entry['put_oi'],
            'total_oi': d_entry['call_oi'] + d_entry['put_oi'],
            'call_gex': d_entry['call_gex'], 'put_gex': d_entry['put_gex'],
            'net_gex': d_entry['call_gex'] + d_entry['put_gex'],
            'net_dealer_delta': d_entry['call_delta'] + d_entry['put_delta'],
            'net_vanna': d_entry['call_vanna'] + d_entry['put_vanna'],
            'net_charm': d_entry['call_charm'] + d_entry['put_charm'],
            'call_vanna': d_entry['call_vanna'], 'put_vanna': d_entry['put_vanna'],
            'call_charm': d_entry['call_charm'], 'put_charm': d_entry['put_charm'],
            'call_volume': 0, 'put_volume': 0, 'total_volume': 0,
            'active_gex': d_entry['call_gex'] + d_entry['put_gex'],
            'expiry_gex': {},
        })
    deribit_df = pd.DataFrame(deribit_rows) if deribit_rows else pd.DataFrame()

    # Build combined DataFrame (BTC-price-keyed)
    ibit_btc_df = df.copy()
    # Use btc_price as the strike key for combined
    if is_crypto and 'btc_price' in ibit_btc_df.columns:
        ibit_btc_df = ibit_btc_df.copy()
        ibit_btc_df['strike'] = ibit_btc_df['btc_price']

    if not deribit_df.empty:
        combined_df = pd.concat([ibit_btc_df, deribit_df], ignore_index=True)
        combined_df = combined_df.sort_values('strike').reset_index(drop=True)
    else:
        combined_df = ibit_btc_df

    # Compute combined levels (BTC-space) and Deribit-only levels
    combined_levels_btc = _compute_levels_from_df(combined_df, btc_spot, etf_flows) if btc_spot else None
    deribit_levels_btc = _compute_levels_from_df(deribit_df, btc_spot) if (not deribit_df.empty and btc_spot) else None

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
        # Current GEX at this strike
        strike_rows = df[df['strike'] == lstrike]
        current_gex = float(strike_rows['net_gex'].sum()) if not strike_rows.empty else 0
        # Previous GEX (check weighted_net_gex for pre-migration snapshots, then net_gex)
        if prev_strikes and lstrike in prev_strikes:
            prev_gex = prev_strikes[lstrike].get('weighted_net_gex')
            if prev_gex is None:
                prev_gex = prev_strikes[lstrike].get('net_gex', 0)
            if prev_gex and prev_gex != 0:
                change_pct = ((current_gex - prev_gex) / abs(prev_gex)) * 100
                traj['change_pct'] = round(change_pct, 1)
                if change_pct > 10:
                    traj['status'] = 'STRENGTHENING'
                elif change_pct < -10:
                    traj['status'] = 'WEAKENING'
        # Dominant expiry: which expiration contributes most GEX
        if not strike_rows.empty:
            expiry_gex = strike_rows.iloc[0].get('expiry_gex', {})
            if isinstance(expiry_gex, dict) and expiry_gex:
                best_exp = max(expiry_gex.items(), key=lambda x: abs(x[1].get('net_gex', 0)))
                traj['dominant_expiry'] = best_exp[0]
                traj['dominant_expiry_dte'] = best_exp[1].get('dte')
        level_trajectory[lname] = traj
    levels['level_trajectory'] = level_trajectory

    # Breakout signals (prev_strikes already available from early fetch)
    breakout = compute_breakout(df, spot, levels, expected_move, prev_strikes, ref_per_share, etf_flows)

    # Significant levels with regime behavior
    sig_levels = compute_significant_levels(df, spot, levels, prev_strikes, is_crypto, ref_per_share, etf_flows)

    # Dealer flow forecast (vanna + charm) — use combined data when available
    if deribit_available and combined_levels_btc and not combined_df.empty:
        flow_forecast = compute_flow_forecast(combined_df, btc_spot, combined_levels_btc, is_crypto)
    else:
        flow_forecast = compute_flow_forecast(df, spot, levels, is_crypto)

    # Dealer delta scenario analysis
    dealer_delta_profile = None
    dealer_delta_briefing = None
    delta_flip_points = []
    try:
        scenario_result = compute_dealer_delta_scenarios(
            cached_chains, spot, levels, expected_move,
            rfr, ref_per_share, is_crypto,
            deribit_options=deribit_options if deribit_available else None
        )
        dealer_delta_profile = scenario_result['profile']
        delta_flip_points = scenario_result['flip_points']
        dealer_delta_briefing = generate_dealer_delta_briefing(
            scenario_result, spot, levels, ref_per_share, is_crypto
        )
    except Exception as e:
        print(f"  [delta-scenario] Failed: {e}")

    # Save to DB (only for 0-min_dte windows — most relevant for day-over-day comparisons)
    conn = get_db()
    if min_dte == 0:
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

    # Build response — BTC-price-indexed chart with per-venue breakdown
    gex_chart_data = []
    # Build IBIT lookup by BTC price
    ibit_by_btc = {}
    for _, row in df[(df['strike'] >= spot * 0.82) & (df['strike'] <= spot * 1.22)].iterrows():
        btc_p = round(float(row['strike'] / ref_per_share)) if is_crypto else round(float(row['strike']))
        ibit_by_btc[btc_p] = row

    # Build Deribit lookup by BTC price
    deribit_by_btc = {}
    if not deribit_df.empty and btc_spot:
        btc_lo, btc_hi = btc_spot * 0.82, btc_spot * 1.22
        for _, row in deribit_df[(deribit_df['strike'] >= btc_lo) & (deribit_df['strike'] <= btc_hi)].iterrows():
            deribit_by_btc[round(float(row['strike']))] = row

    all_btc_prices = sorted(set(list(ibit_by_btc.keys()) + list(deribit_by_btc.keys())))

    for btc_p in all_btc_prices:
        ibit_row = ibit_by_btc.get(btc_p)
        deribit_row = deribit_by_btc.get(btc_p)

        i_gex = float(ibit_row['net_gex']) if ibit_row is not None else 0
        d_gex = float(deribit_row['net_gex']) if deribit_row is not None else 0

        # Per-expiry breakdown from IBIT
        expiry_breakdown = {}
        if ibit_row is not None:
            expiry_gex = ibit_row.get('expiry_gex', {})
            if isinstance(expiry_gex, dict):
                for exp_str, eg in expiry_gex.items():
                    expiry_breakdown[exp_str] = {
                        'net_gex': eg.get('net_gex', 0),
                        'dte': eg.get('dte'),
                    }

        entry = {
            'strike': float(ibit_row['strike']) if ibit_row is not None else 0,
            'btc': float(btc_p),
            'net_gex': i_gex + d_gex,
            'ibit_gex': i_gex,
            'deribit_gex': d_gex,
        }

        if ibit_row is not None:
            entry.update({
                'active_gex': float(ibit_row['active_gex']) + d_gex,
                'net_vanna': float(ibit_row['net_vanna']) + (float(deribit_row['net_vanna']) if deribit_row is not None else 0),
                'net_charm': float(ibit_row['net_charm']) + (float(deribit_row['net_charm']) if deribit_row is not None else 0),
                'call_oi': int(ibit_row['call_oi']) + (int(deribit_row['call_oi']) if deribit_row is not None else 0),
                'put_oi': int(ibit_row['put_oi']) + (int(deribit_row['put_oi']) if deribit_row is not None else 0),
                'total_oi': int(ibit_row['total_oi']) + (int(deribit_row['total_oi']) if deribit_row is not None else 0),
                'call_volume': int(ibit_row['call_volume']),
                'put_volume': int(ibit_row['put_volume']),
                'total_volume': int(ibit_row['total_volume']),
                'expiry_breakdown': expiry_breakdown,
            })
        else:
            entry.update({
                'active_gex': d_gex,
                'net_vanna': float(deribit_row['net_vanna']),
                'net_charm': float(deribit_row['net_charm']),
                'call_oi': int(deribit_row['call_oi']),
                'put_oi': int(deribit_row['put_oi']),
                'total_oi': int(deribit_row['total_oi']),
                'call_volume': 0, 'put_volume': 0, 'total_volume': 0,
                'expiry_breakdown': {},
            })
        gex_chart_data.append(entry)

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
        expiry_meta.append({'exp': exp_str, 'dte': dte_days})
    expiry_meta.sort(key=lambda x: x['dte'])

    return {
        'ticker': ticker_symbol,
        'asset_label': cfg['asset_label'] if cfg else ticker_symbol,
        'spot': float(spot),
        'btc_spot': float(btc_spot) if btc_spot else None,
        'btc_per_share': float(ref_per_share),
        'is_btc': bool(is_crypto),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'dte_window': {'min': min_dte, 'max': max_dte},
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
        'deribit_available': deribit_available,
        'deribit_oi_btc': float(deribit_oi_btc) if deribit_available else 0,
        'deribit_net_gex': float(deribit_df['net_gex'].sum()) if not deribit_df.empty else 0,
        'combined_levels_btc': {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                for k, v in combined_levels_btc.items()} if combined_levels_btc else None,
        'deribit_levels_btc': {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                               for k, v in deribit_levels_btc.items()} if deribit_levels_btc else None,
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


def compute_dealer_delta_scenarios(cached_chains, spot, levels, expected_move, rfr, ref_per_share, is_crypto, deribit_options=None):
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

        # Deribit options contribution (1 BTC per contract)
        if deribit_options:
            S_hyp_btc = S_hyp / ref_per_share if is_crypto else S_hyp
            for opt in deribit_options:
                T_d = max(opt['dte'] / 365.0, 0.5 / 365)
                iv_d = opt['iv'] / 100.0
                delta_d = bs_delta(S_hyp_btc, opt['strike'], T_d, rfr, iv_d, opt['option_type'])
                dd_d = -delta_d * opt['oi'] * 1
                net_dd += dd_d
                if opt['option_type'] == 'call':
                    call_dd += dd_d
                else:
                    put_dd += dd_d

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

        net_gex_val = row['net_gex']
        call_oi, put_oi, total_oi = int(row['call_oi']), int(row['put_oi']), int(row['total_oi'])
        is_major = total_oi > oi_90

        if put_oi > call_oi * 1.5 and net_gex_val < 0:
            ltype = 'put_wall'
        elif call_oi > put_oi * 1.5 and net_gex_val > 0:
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
                        'btc': float(t['strike'] / ref_per_share),
                        'total_oi': int(t['total_oi']),
                        'net_gex': float(t['net_gex'])} for t in up_targets],
        'down_targets': [{'strike': float(t['strike']),
                          'btc': float(t['strike'] / ref_per_share),
                          'total_oi': int(t['total_oi']),
                          'net_gex': float(t['net_gex'])} for t in down_targets],
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


def fetch_with_cache(ticker, dte, min_dte=0):
    """Return cached data if fresh, otherwise check Yahoo for new OI."""
    today = datetime.now().strftime('%Y-%m-%d')
    cache_date, cached = get_latest_cache(ticker, dte)

    # Already confirmed today's data (and min_dte matches)
    if cached and cache_date == today:
        cached_min = cached.get('dte_window', {}).get('min', 0)
        if cached_min == min_dte:
            return cached

    # Throttle: don't re-check Yahoo more than every 30 min
    check_key = (ticker, dte, min_dte)
    with _yahoo_check_lock:
        last_check = _last_yahoo_check.get(check_key)
        if cached and last_check and (datetime.now() - last_check).total_seconds() < YAHOO_CHECK_INTERVAL:
            return cached

    # Fetch from Yahoo
    data = fetch_and_analyze(ticker, dte, min_dte)
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

REFRESH_DTES = DTE_WINDOWS  # refresh all non-overlapping windows

def _bg_refresh():
    """Background thread: backfill candles on startup for all tickers,
    pre-fetch all DTEs, then periodically update candles (every 5 min) and re-check GEX data."""
    # Phase 0: Fetch Farside ETF flow data (full history on first run)
    try:
        fetch_farside_flows()
    except Exception as e:
        print(f"  [bg-refresh] Farside flow fetch error: {e}")

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

        # Refresh Farside ETF flow data
        try:
            fetch_farside_flows()
        except Exception as e:
            print(f"  [bg-refresh] Farside flow error: {e}")

        # GEX refresh logic for all tickers
        for tk, cfg in TICKER_CONFIG.items():
            today = datetime.now().strftime('%Y-%m-%d')
            all_fresh = True
            stale_windows = []
            for label, min_d, max_d in REFRESH_DTES:
                cache_date, cached = get_latest_cache(tk, max_d)
                # Check both date and that min_dte matches
                if cache_date != today or (cached and cached.get('dte_window', {}).get('min', 0) != min_d):
                    stale_windows.append((label, min_d, max_d))
            if stale_windows:
                all_fresh = False
                with ThreadPoolExecutor(max_workers=len(stale_windows)) as pool:
                    futures = {pool.submit(fetch_with_cache, tk, max_d, min_d): (label, min_d, max_d) for label, min_d, max_d in stale_windows}
                    for fut in as_completed(futures):
                        try:
                            fut.result()
                        except Exception as e:
                            w = futures[fut]
                            print(f"  [bg-refresh] {tk} DTE {w[1]}-{w[2]} error: {e}")

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
    c.execute('SELECT date, daily_flow_dollars, shares_outstanding, aum, nav, total_btc_etf_flow FROM etf_flows WHERE ticker=? ORDER BY date DESC LIMIT 30', (ticker,))
    rows = c.fetchall()
    conn.close()
    rows.reverse()
    data = [{'date': r[0], 'flow': r[1], 'shares': r[2], 'aum': r[3], 'nav': r[4], 'total_flow': r[5]} for r in rows]
    return Response(json.dumps(data), mimetype='application/json')


@app.route('/api/data')
def api_data():
    try:
        ticker = request.args.get('ticker', 'IBIT').upper()
        if ticker not in TICKER_CONFIG:
            return Response(json.dumps({'error': f'Unknown ticker: {ticker}'}), mimetype='application/json'), 400
        dte = request.args.get('dte', 3, type=int)
        min_dte = request.args.get('min_dte', 0, type=int)
        dte = max(1, min(dte, 90))
        min_dte = max(0, min(min_dte, dte - 1))
        try:
            data = fetch_with_cache(ticker, dte, min_dte)
        except (ValueError, KeyError):
            # Serve most recent cached data for this DTE if available
            _, cached = get_latest_cache(ticker, dte)
            if cached:
                data = cached
                data['stale'] = True
            elif dte != 3:
                data = fetch_with_cache(ticker, 3, 0)
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


def run_analysis(ticker='IBIT', save=True):
    """Run AI analysis across all DTEs. Returns analysis dict or raises."""
    cfg = TICKER_CONFIG.get(ticker)
    if not cfg:
        raise ValueError(f'Unknown ticker: {ticker}')
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise RuntimeError('ANTHROPIC_API_KEY not set')

    dtes = DTE_WINDOWS  # list of (label, min_dte, max_dte)
    results = {}

    # Fetch live ref asset price so the AI sees current price, not stale cache
    live_ref_price = None
    try:
        live_ref_price = yf.Ticker(cfg['ref_ticker']).info.get('regularMarketPrice')
    except Exception:
        pass

    def fetch_dte(window):
        label, min_d, max_d = window
        return label, fetch_with_cache(ticker, max_d, min_d)

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fetch_dte, w): w for w in dtes}
        for fut in as_completed(futures):
            label, data = fut.result()
            results[label] = data

    # Compute history trends (ticker-level, shared across all DTEs)
    first_label = dtes[0][0]
    conn = get_db()
    history_trends = summarize_history_trends(conn, ticker, results[first_label]['btc_per_share'], results[first_label]['levels'])
    conn.close()

    # Build concise summaries (strip chart data to keep prompt small)
    summaries = {}
    summaries['_history_trends'] = history_trends
    for label, min_d, max_d in dtes:
        d = results[label]
        bps = d['btc_per_share']
        ibit_lvl = d['levels']  # IBIT-only levels (in IBIT strike space)
        deribit_avail = d.get('deribit_available', False)
        combined_lvl = d.get('combined_levels_btc') or {}  # combined levels (in BTC space)
        deribit_lvl = d.get('deribit_levels_btc') or {}  # Deribit-only levels (in BTC space)
        current_btc = live_ref_price if live_ref_price else d['btc_spot']
        key = f"{min_d}-{max_d}d"

        # Use combined levels as primary when Deribit is available
        if deribit_avail and combined_lvl:
            primary_lvl_btc = {
                'call_wall': round(combined_lvl.get('call_wall', 0)),
                'put_wall': round(combined_lvl.get('put_wall', 0)),
                'gamma_flip': round(combined_lvl.get('gamma_flip', 0)),
                'max_pain': round(combined_lvl.get('max_pain', 0)),
                'resistance': [round(s) for s in combined_lvl.get('resistance', [])],
                'support': [round(s) for s in combined_lvl.get('support', [])],
            }
            primary_net_gex = combined_lvl.get('net_gex_total', 0)
            primary_active_gex = combined_lvl.get('active_gex_total', 0)
            primary_regime = combined_lvl.get('regime', ibit_lvl.get('regime'))
            primary_active_cw = round(combined_lvl.get('active_call_wall', combined_lvl.get('call_wall', 0)))
            primary_active_pw = round(combined_lvl.get('active_put_wall', combined_lvl.get('put_wall', 0)))
            primary_confidence = combined_lvl.get('positioning_confidence', 100)
            primary_warnings = combined_lvl.get('positioning_warnings', [])
            source_label = 'IBIT + Deribit (~91% of BTC options OI)'
        else:
            primary_lvl_btc = {
                'call_wall': round(ibit_lvl['call_wall'] / bps),
                'put_wall': round(ibit_lvl['put_wall'] / bps),
                'gamma_flip': round(ibit_lvl['gamma_flip'] / bps),
                'max_pain': round(ibit_lvl['max_pain'] / bps),
                'resistance': [round(s / bps) for s in ibit_lvl.get('resistance', [])],
                'support': [round(s / bps) for s in ibit_lvl.get('support', [])],
            }
            primary_net_gex = ibit_lvl.get('net_gex_total', 0)
            primary_active_gex = ibit_lvl.get('active_gex_total', 0)
            primary_regime = ibit_lvl.get('regime')
            primary_active_cw = round(ibit_lvl.get('active_call_wall', ibit_lvl['call_wall']) / bps)
            primary_active_pw = round(ibit_lvl.get('active_put_wall', ibit_lvl['put_wall']) / bps)
            primary_confidence = ibit_lvl.get('positioning_confidence', 100)
            primary_warnings = ibit_lvl.get('positioning_warnings', [])
            source_label = 'IBIT only (~52% of BTC options OI)'

        summaries[key] = {
            'spot_btc': round(current_btc),
            'spot_ibit': d['spot'],
            'btc_per_share': bps,
            'source': source_label,
            'levels_btc': primary_lvl_btc,
            'regime': primary_regime,
            'active_gex_total': primary_active_gex,
            'net_gex_total': primary_net_gex,
            'level_trajectory': ibit_lvl.get('level_trajectory', {}),
            'active_call_wall_btc': primary_active_cw,
            'active_put_wall_btc': primary_active_pw,
            'positioning_confidence': primary_confidence,
            'positioning_warnings': primary_warnings,
            'levels': ibit_lvl,
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
                'total_btc_etf_flow_millions': round(d['etf_flows'].get('total_btc_etf_flow', 0) / 1e6, 1),
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
            'delta_flip_points': [
                {
                    'price_btc': fp['price_btc'],
                    'distance_pct': round(((fp['price_ibit'] - d['spot']) / d['spot']) * 100, 1),
                    'from_direction': fp['from_direction'],
                    'to_direction': fp['to_direction'],
                }
                for fp in (d.get('delta_flip_points') or [])
            ],
            'key_level_dealer_delta_btc': {
                name: round(delta * d['btc_per_share']) if delta is not None else None
                for name, delta in (d.get('dealer_delta_briefing', {}).get('key_level_deltas', {}) or {}).items()
            } if d.get('dealer_delta_briefing') else None,
        }

        # Per-venue breakdown for divergence analysis
        if deribit_avail:
            summaries[key]['venue_breakdown'] = {
                'ibit': {
                    'oi_contracts': ibit_lvl.get('total_call_oi', 0) + ibit_lvl.get('total_put_oi', 0),
                    'net_gex': ibit_lvl.get('net_gex_total', 0),
                    'call_wall_btc': round(ibit_lvl['call_wall'] / bps),
                    'put_wall_btc': round(ibit_lvl['put_wall'] / bps),
                    'gamma_flip_btc': round(ibit_lvl['gamma_flip'] / bps),
                    'regime': ibit_lvl.get('regime'),
                    'net_vanna': ibit_lvl.get('net_vanna', 0),
                    'net_charm': ibit_lvl.get('net_charm', 0),
                },
                'deribit': {
                    'oi_btc': d.get('deribit_oi_btc', 0),
                    'net_gex': d.get('deribit_net_gex', 0),
                    'call_wall_btc': round(deribit_lvl.get('call_wall', 0)) if deribit_lvl else None,
                    'put_wall_btc': round(deribit_lvl.get('put_wall', 0)) if deribit_lvl else None,
                    'gamma_flip_btc': round(deribit_lvl.get('gamma_flip', 0)) if deribit_lvl else None,
                    'regime': deribit_lvl.get('regime') if deribit_lvl else None,
                    'net_vanna': deribit_lvl.get('net_vanna', 0) if deribit_lvl else 0,
                    'net_charm': deribit_lvl.get('net_charm', 0) if deribit_lvl else 0,
                },
            }

        # Add day-over-day level changes if previous data exists
        prev_date, prev_data = get_prev_cache(ticker, max_d)
        if prev_data:
            prev_lvl = prev_data['levels']
            prev_bps = prev_data.get('btc_per_share', bps)
            summaries[key]['changes_vs_prev'] = {
                'prev_date': prev_date,
                'spot_btc_prev': round(prev_data.get('btc_spot', 0)),
                'spot_btc_change': round(d['btc_spot'] - prev_data.get('btc_spot', d['btc_spot'])),
                'regime_prev': prev_lvl.get('regime', 'unknown'),
                'regime_changed': prev_lvl.get('regime') != ibit_lvl.get('regime'),
                'call_wall_btc_prev': round(prev_lvl['call_wall'] / prev_bps),
                'call_wall_btc_change': round(ibit_lvl['call_wall'] / bps - prev_lvl['call_wall'] / prev_bps),
                'put_wall_btc_prev': round(prev_lvl['put_wall'] / prev_bps),
                'put_wall_btc_change': round(ibit_lvl['put_wall'] / bps - prev_lvl['put_wall'] / prev_bps),
                'gamma_flip_btc_prev': round(prev_lvl['gamma_flip'] / prev_bps),
                'gamma_flip_btc_change': round(ibit_lvl['gamma_flip'] / bps - prev_lvl['gamma_flip'] / prev_bps),
                'net_gex_prev': prev_lvl.get('net_gex_total', 0),
                'net_gex_change': ibit_lvl.get('net_gex_total', 0) - prev_lvl.get('net_gex_total', 0),
                'pcr_prev': prev_lvl.get('pcr', 0),
                'pcr_change': round(ibit_lvl.get('pcr', 0) - prev_lvl.get('pcr', 0), 3),
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

ETF fund flows (etf_flows) show daily creation/redemption activity from Farside Investors. 'daily_flow_millions' is IBIT-specific flow; 'total_btc_etf_flow_millions' is the total across ALL BTC spot ETFs (IBIT, FBTC, ARKB, GBTC, etc.). Positive = inflows, negative = outflows. A streak of 3+ days in one direction is significant. IMPORTANT: Distinguish between IBIT-specific outflows and total BTC ETF outflows. If IBIT has outflows but total BTC ETFs have inflows, that's fund rotation (not bearish). If ALL BTC ETFs have outflows, that's genuine institutional exit (bearish). Cross-reference with GEX regime: inflows + positive gamma = strong range, outflows + negative gamma = breakdown risk.

VENUE DATA:
levels_btc contains the COMBINED IBIT + Deribit gamma exposure profile (~91% of all BTC options OI when Deribit is available). These are your primary levels for analysis — call walls, put walls, gamma flip, max pain all reflect the full options market. The 'source' field indicates what data is included.

venue_breakdown (when present) shows per-venue positioning. Use it to identify divergences:
- If both IBIT and Deribit have a call wall at the same BTC level → HIGH CONVICTION resistance (TradFi + crypto-native agree)
- If both have a put wall at the same level → HIGH CONVICTION support
- If IBIT has a big wall that Deribit doesn't → TradFi-driven level, may not hold if crypto-native flow pushes through
- If Deribit has a big wall that IBIT doesn't → crypto-native positioning that TradFi hasn't hedged around; watch for convergence
- If gamma flips differ between venues → note which venue is in positive vs negative gamma territory
Only call out divergences when they're material (different wall locations or different gamma regimes). Don't list venue breakdowns mechanically — synthesize them into a trading-relevant insight.
venue_breakdown also includes per-venue vanna and charm totals. If overnight charm flow is dominated by one venue, note it — e.g., "Deribit charm is 3x IBIT charm, suggesting crypto-native MMs will drive overnight rebalancing" or "IBIT charm dominates, overnight flow will come through ETF share market."

Note: significant_levels and breakout are derived from IBIT options data only. flow_forecast uses combined IBIT + Deribit data. Dealer delta scenarios include both IBIT and Deribit contributions.

GEX (net_gex_total) is raw Black-Scholes gamma exposure — no additional time weighting is applied because BS gamma already incorporates natural time sensitivity via 1/(S*sigma*sqrt(T)). Near-term options naturally have higher gamma, so their GEX contribution is already appropriately scaled. All levels (call_wall, put_wall, gamma_flip, regime) derive from this raw GEX. The per-expiry breakdown shows each expiration's GEX contribution separately, and can be filtered via the expiry_filter parameter. The level_trajectory field shows whether key levels are STRENGTHENING (>10% increase), WEAKENING (>10% decrease), or STABLE vs the previous session. It also identifies the dominant_expiry driving each level — if a wall is dominated by a near-term expiry, it may evaporate quickly after that expiration.

Historical Trends (_history_trends): A compressed summary of the last 10 daily snapshots, shared across all timeframes. This is NOT per-DTE — it reflects the ticker's overall positioning evolution.
- regime_streak: How many days the current gamma regime has held. A streak of 3+ days means the regime is established, not transitional. A streak of 1 means it just flipped — watch for reversion.
- level_migration: Direction and magnitude of key level movements over 3d and 5d windows. Rising walls with a long streak = strong directional positioning. Stable walls held at a specific price = anchored range. Use consecutive_direction to gauge conviction.
- gex_trend: Whether overall gamma exposure is building (market adding options, levels strengthening) or decaying (positions unwinding, levels weakening).
- oi_trend: Call and put OI direction separately. Divergences matter: call OI building + put OI decaying = bullish positioning shift.
- range_evolution: Whether the tradeable range (call wall to put wall) is expanding, contracting, or shifting directionally. 'Contracting' often precedes a breakout. 'Expanding_symmetric' means both sides being defended.
- narrative: One-line summary of the dominant trend pattern.

Use history_trends to contextualize today's snapshot. A call wall at $107K means different things if it's been there for 7 days (strong, tested resistance) vs if it jumped there overnight (new, untested). Always mention regime streak length and level migration direction in your analysis.

Dealer Delta Scenario Analysis (dealer_delta_briefing): Pre-computed dealer hedging pressure at hypothetical price levels across the key level grid. 'current_delta' shows dealer positioning at spot. 'flip_summary' identifies where dealer pressure reverses direction. 'acceleration_zone' shows where dealers are most reactive to price moves. Negative dealer delta = dealers must BUY to hedge = supportive (acts as a bid). Positive dealer delta = dealers must SELL = resistive (acts as an offer). Use this to identify price levels where dealer hedging creates natural support or resistance, independent of the GEX profile.

When analyzing dealer delta alongside GEX levels, look for CONVERGENCE and DIVERGENCE:
- CONVERGENCE (high conviction): Put wall at $98K + dealers net buyers at $98K + level STRENGTHENING = strong support. Trade it.
- DIVERGENCE (caution): Call wall at $105K but dealers still net buyers at $105K = wall may not hold because dealer hedging flow supports upside through it.
- ACCELERATION ZONES: Where dealer delta changes fastest (high acceleration), small price moves trigger large hedging flows. These are inflection points — price tends to move quickly through these zones rather than consolidate.
- DELTA FLIP vs GAMMA FLIP: These are different concepts. Gamma flip = where net GEX crosses zero (regime change). Delta flip = where dealer hedging direction reverses. When they're at different prices, the zone between them is a transition zone with mixed signals.

For each timeframe, provide:
- What changed overnight and why it matters (if changes_vs_prev available)
- Regime summary and implication (1-2 sentences)
- Historical context: Reference regime streak, level migration trends, and range evolution from _history_trends. Is today's positioning a continuation or a change?
- Key levels in BTC price with level trajectory status (STRENGTHENING/WEAKENING/STABLE) and dominant expiry if near-term. Example: "Call wall $108,200 (STRENGTHENING, driven by Feb 21 exp — 3 DTE)"
- Dealer delta context: Are dealers net buyers or sellers at spot? Where does pressure flip? Which key levels have the strongest dealer hedging behind them? Reference specific BTC prices and delta magnitudes from dealer_delta_briefing and key_level_dealer_delta_btc.
- Dealer flow direction (charm/vanna implications)
- Risk assessment — specifically note where dealer delta ACCELERATES (acceleration_zone), as these are prices where moves become self-reinforcing
- Actionable setup (if any clear one exists) — incorporate both GEX levels AND dealer delta direction. A level is strongest when BOTH GEX and dealer delta agree (e.g., put wall + dealers buying = high-conviction support). Flag levels where they diverge.

If a previous analysis is provided, use it to:
- Note whether your prior calls played out or not (e.g. "yesterday's $65K support held as expected" or "prior upside bias was invalidated")
- Update your thesis based on how positioning evolved
- Maintain continuity — don't repeat the same analysis if nothing changed, focus on what's new

For the "all" key: provide cross-timeframe alignment analysis — whether short-term and long-term signals agree, overall directional bias, and the highest-conviction trade setup. Highlight any divergences between short-term and long-term positioning changes. Specifically:
- Do dealer delta flip points align across timeframes? If the 7d delta flips at $104K but 14d flips at $107K, that gap is meaningful.
- Are level trajectories consistent? If the call wall is STRENGTHENING on short-term but WEAKENING on longer-term, the wall has a shelf life.
- What is the highest-conviction zone where GEX levels, dealer delta direction, level trajectory, and ETF flows ALL agree?

ALWAYS lead each timeframe analysis with a single bold top-line: **BOTTOM LINE:** followed by the single most actionable takeaway in one sentence. This should answer "what do I do today?" — include direction (long/short/flat), the key price levels to watch, and the setup trigger. Examples:
- "**BOTTOM LINE:** Fade rallies into $71,728 call wall, buy dips at $68,141 put wall — tight $3,500 range with positive gamma and dealer selling confirms both walls."
- "**BOTTOM LINE:** Stay flat — gamma just flipped negative, dealer delta flipping at $104K means neither wall is reliable until positioning stabilizes."
- "**BOTTOM LINE:** Long above $70,200 gamma flip targeting $72,750 delta flip — dealers are net short 5M shares creating a bid, but charm selling into $72.8M headwind caps upside speed."

After the bottom line, provide 3-5 supporting bullet points. Be concise and direct — no walls of text. Use trader shorthand where appropriate. Reference specific BTC price levels.

DTE windows are NON-OVERLAPPING. Each timeframe shows distinct option positioning:
- 0-3d: Immediate expirations. Highest gamma, strongest near-term hedging pressure. These are today's actionable levels.
- 4-7d: Next week's expirations. Where the next wave of gamma concentration is forming.
- 8-14d: Two-week positioning. Emerging walls that will become dominant after near-term expiries clear.
- 15-30d: Monthly cycle. Institutional positioning around monthly options expiration.
- 31-45d: Structural. Quarterly and longer-dated positioning that forms the backdrop.

When comparing across windows: if 0-3d and 4-7d call walls are at the same strike, that level has multi-week support and is high conviction. If they differ, the near-term wall will expire and levels will shift — flag this as a potential level migration.

IMPORTANT: Return ONLY valid JSON with keys "0-3d", "4-7d", "8-14d", "15-30d", "31-45d", "all". Each value should be a string containing your analysis with newlines for formatting. Do not wrap in markdown code blocks."""

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
    # Attach BTC price and timestamp metadata
    btc_price = live_ref_price
    if not btc_price:
        for dte in dtes:
            if dte in results and results[dte].get('btc_spot'):
                btc_price = results[dte]['btc_spot']
                break
    if btc_price is not None:
        analysis['_btc_price'] = btc_price
        analysis['_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    if save:
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
    """Re-run analysis without saving. Use /api/analysis/save to persist."""
    ticker = request.args.get('ticker', 'IBIT').upper()
    if ticker not in TICKER_CONFIG:
        return Response(json.dumps({'error': f'Unknown ticker: {ticker}'}), mimetype='application/json'), 400
    try:
        analysis = run_analysis(ticker, save=False)
        return Response(json.dumps(analysis), mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({'error': str(e)}),
                        mimetype='application/json'), 500


@app.route('/api/analysis/save', methods=['POST'])
def api_analysis_save():
    """Save the provided analysis as today's cached analysis."""
    ticker = request.args.get('ticker', 'IBIT').upper()
    if ticker not in TICKER_CONFIG:
        return Response(json.dumps({'error': f'Unknown ticker: {ticker}'}), mimetype='application/json'), 400
    try:
        analysis = request.get_json()
        if not analysis:
            return Response(json.dumps({'error': 'No analysis data'}), mimetype='application/json'), 400
        btc_price = analysis.get('_btc_price')
        set_cached_analysis(ticker, analysis, btc_price)
        return Response(json.dumps({'status': 'saved'}), mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({'error': str(e)}), mimetype='application/json'), 500


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