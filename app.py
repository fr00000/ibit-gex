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
import logging
import logging.handlers
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
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, Response, request
import anthropic
from dotenv import load_dotenv
from gex import (
    bs_gamma, bs_delta, bs_vanna, bs_charm,
    compute_atm_iv, compute_deribit_iv_term, process_deribit_options, build_deribit_df,
    _compute_levels_from_df, compute_flow_forecast, compute_dealer_delta_scenarios,
    generate_dealer_delta_briefing, compute_positioning_depth,
    compute_significant_levels, compute_breakout,
    STRIKE_RANGE_PCT,
)

load_dotenv()

ET = ZoneInfo('America/New_York')

def now_et():
    """Current time in Eastern, used for all timestamps and date keys."""
    return datetime.now(ET)

def today_str_et():
    """Today's date as YYYY-MM-DD string in Eastern Time."""
    return now_et().strftime('%Y-%m-%d')

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Logging — rotating file + console
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)

log = logging.getLogger('gex')
log.setLevel(logging.DEBUG)

_fmt = logging.Formatter('%(asctime)s %(levelname)-5s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

_fh = logging.handlers.RotatingFileHandler(
    LOG_DIR / 'gex.log', maxBytes=10 * 1024 * 1024, backupCount=5)
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_fmt)
log.addHandler(_fh)

_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(_fmt)
log.addHandler(_ch)


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
WEEKLY_OUTLOOK_DTE = 999  # sentinel DTE key for weekly outlook in data_cache

TICKER_CONFIG = {
    'IBIT': {
        'name': 'IBIT', 'asset_label': 'BTC', 'ref_ticker': 'BTC-USD',
        'binance_symbol': 'BTCUSDT', 'per_share_default': 0.000568,
        'deribit_currency': 'BTC',
    },
    'ETHA': {
        'name': 'ETHA', 'asset_label': 'ETH', 'ref_ticker': 'ETH-USD',
        'binance_symbol': 'ETHUSDT', 'per_share_default': 0.0091,
        'deribit_currency': 'ETH',
    },
}

COINGLASS_API_KEY = os.environ.get('COINGLASS_API_KEY')
ADMIN_KEY = os.environ.get('ADMIN_KEY', '')

_rfr_cache = {'rate': None, 'date': None}

def get_risk_free_rate():
    """Fetch 13-week T-bill rate (^IRX) from Yahoo Finance, cached daily."""
    today = today_str_et()
    if _rfr_cache['rate'] is not None and _rfr_cache['date'] == today:
        return _rfr_cache['rate']
    try:
        irx = yf.Ticker('^IRX').info.get('regularMarketPrice')
        if irx and irx > 0:
            rate = irx / 100.0
            _rfr_cache['rate'] = rate
            _rfr_cache['date'] = today
            log.info(f"[rfr] 13-week T-bill rate: {irx:.2f}% ({rate:.4f})")
            return rate
    except Exception as e:
        log.warning(f"[rfr] Failed to fetch ^IRX, using default: {e}")
    return RISK_FREE_RATE_DEFAULT


# ── DATABASE ────────────────────────────────────────────────────────────────
def init_db():
    """Create tables once at startup."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA busy_timeout=30000')
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
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_date TEXT NOT NULL,
        ticker TEXT NOT NULL,
        expiry_date TEXT NOT NULL,
        dte INTEGER NOT NULL,
        dte_window TEXT NOT NULL,
        spot_btc REAL,
        call_wall_btc REAL,
        put_wall_btc REAL,
        gamma_flip_btc REAL,
        max_pain_btc REAL,
        regime TEXT,
        net_gex REAL,
        net_dealer_delta REAL,
        dealer_delta_direction TEXT,
        charm_direction TEXT,
        charm_notional REAL,
        charm_strength TEXT,
        vanna_strength TEXT,
        overnight_direction TEXT,
        overnight_notional REAL,
        deribit_available INTEGER DEFAULT 0,
        ibit_call_wall_btc REAL,
        ibit_put_wall_btc REAL,
        deribit_call_wall_btc REAL,
        deribit_put_wall_btc REAL,
        venue_walls_agree INTEGER,
        em_upper_btc REAL,
        em_lower_btc REAL,
        ai_bottom_line TEXT,
        scored INTEGER DEFAULT 0,
        scored_date TEXT,
        btc_high_on_expiry REAL,
        btc_low_on_expiry REAL,
        btc_close_on_expiry REAL,
        btc_high_in_window REAL,
        btc_low_in_window REAL,
        call_wall_held INTEGER,
        put_wall_held INTEGER,
        range_held INTEGER,
        em_held INTEGER,
        regime_correct INTEGER,
        charm_correct INTEGER,
        venue_agree_held INTEGER,
        max_breach_call_pct REAL,
        max_breach_put_pct REAL,
        realized_range_pct REAL,
        call_wall_error_pct REAL,
        put_wall_error_pct REAL,
        UNIQUE(analysis_date, ticker, expiry_date)
    )''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_pred_expiry ON predictions(expiry_date, scored)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_pred_dte ON predictions(dte, scored)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_pred_ticker ON predictions(ticker, scored)')
    c.execute('''CREATE TABLE IF NOT EXISTS coinglass_data (
        date TEXT NOT NULL,
        symbol TEXT NOT NULL,
        metric TEXT NOT NULL,
        value REAL,
        extra_json TEXT,
        UNIQUE(date, symbol, metric)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS expiry_cache (
        date TEXT NOT NULL,
        ticker TEXT NOT NULL,
        expiry_date TEXT NOT NULL,
        data_json TEXT NOT NULL,
        UNIQUE(date, ticker, expiry_date)
    )''')
    # Migrate: add T+2 scoring columns to predictions
    pred_cols = {r[1] for r in c.execute('PRAGMA table_info(predictions)').fetchall()}
    for col, typ in [
        ('btc_high_t2', 'REAL'), ('btc_low_t2', 'REAL'), ('btc_close_t2', 'REAL'),
        ('call_wall_held_t2', 'INTEGER'), ('put_wall_held_t2', 'INTEGER'),
        ('range_held_t2', 'INTEGER'), ('regime_correct_t2', 'INTEGER'),
    ]:
        if col not in pred_cols:
            c.execute(f'ALTER TABLE predictions ADD COLUMN {col} {typ}')
    conn.commit()
    conn.close()


def get_db():
    """Get a SQLite connection (one per call, thread-safe)."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute('PRAGMA busy_timeout=30000')
    return conn


def get_prev_strikes(conn, ticker):
    c = conn.cursor()
    today = today_str_et()
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
    date_str = today_str_et()
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
    """Compress 30 days of snapshots into trend signals for AI analysis."""
    history = get_history(conn, ticker, 30)
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

    # ── Regime persistence (30-day window) ──
    regime_signs = [1 if h[7] == 'positive_gamma' else -1 for h in history]
    n_positive = sum(1 for s in regime_signs if s > 0)
    n_negative = len(regime_signs) - n_positive
    dominant_count = max(n_positive, n_negative)
    persistence_pct = round(dominant_count / len(regime_signs) * 100) if regime_signs else 0
    sign_flips = sum(1 for i in range(1, len(regime_signs)) if regime_signs[i] != regime_signs[i-1])

    # Net GEX magnitudes for the 30-day window
    net_gex_vals_30d = [h[8] for h in history if h[8] is not None]
    avg_gex_magnitude = abs(sum(net_gex_vals_30d) / len(net_gex_vals_30d)) if net_gex_vals_30d else 0

    regime_streak['persistence_30d'] = {
        'positive_days': n_positive,
        'negative_days': n_negative,
        'persistence_pct': persistence_pct,
        'sign_flips': sign_flips,
        'avg_gex_magnitude': round(avg_gex_magnitude),
        'is_persistent': persistence_pct >= 70 and sign_flips <= 5,
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


def summarize_structure_trends(conn, ticker, days=7):
    """Summarize per-DTE-window level migration for AI analysis."""
    c = conn.cursor()
    cutoff = (now_et() - timedelta(days=days)).strftime('%Y-%m-%d')

    dte_to_window = {3: '0-3', 7: '4-7', 14: '8-14', 30: '15-30', 45: '31-45'}

    window_history = {}

    cache_rows = c.execute('''SELECT date, dte, data_json FROM data_cache
                              WHERE ticker=? AND date >= ? ORDER BY date, dte''',
                           (ticker, cutoff)).fetchall()

    for date, dte, data_json_str in cache_rows:
        d = json.loads(data_json_str)
        window = dte_to_window.get(dte)
        if not window:
            continue
        bps = d.get('btc_per_share', 1)
        lvl = d.get('levels', {})
        combined = d.get('combined_levels_btc') or {}

        if combined:
            cw = combined.get('call_wall')
            pw = combined.get('put_wall')
            gf = combined.get('gamma_flip')
            regime = combined.get('regime', lvl.get('regime'))
        else:
            cw = lvl.get('call_wall', 0) / bps if bps else None
            pw = lvl.get('put_wall', 0) / bps if bps else None
            gf = lvl.get('gamma_flip', 0) / bps if bps else None
            regime = lvl.get('regime')

        if window not in window_history:
            window_history[window] = []
        window_history[window].append({
            'date': date, 'call_wall': cw, 'put_wall': pw,
            'gamma_flip': gf, 'regime': regime,
        })

    if not window_history:
        return None

    def direction(old, new, threshold_pct=3):
        if old is None or new is None or old == 0:
            return 'unknown'
        pct = (new - old) / old * 100
        if pct > threshold_pct:
            return 'rising'
        elif pct < -threshold_pct:
            return 'falling'
        return 'stable'

    trends = {}
    for window, entries in window_history.items():
        if len(entries) < 2:
            trends[window] = {'days': len(entries), 'trend': 'insufficient_data'}
            continue

        first = entries[0]
        last = entries[-1]

        trends[window] = {
            'days': len(entries),
            'call_wall': {'first': first['call_wall'], 'last': last['call_wall'],
                          'direction': direction(first['call_wall'], last['call_wall'])},
            'put_wall': {'first': first['put_wall'], 'last': last['put_wall'],
                         'direction': direction(first['put_wall'], last['put_wall'])},
            'gamma_flip': {'first': first['gamma_flip'], 'last': last['gamma_flip'],
                           'direction': direction(first['gamma_flip'], last['gamma_flip'])},
            'regime_changes': sum(1 for i in range(1, len(entries))
                                 if entries[i]['regime'] != entries[i-1]['regime']),
        }

    convergence = {}
    for metric in ['call_wall', 'put_wall']:
        values_first = [window_history[w][0].get(metric) for w in ['0-3', '4-7', '8-14', '15-30', '31-45']
                        if w in window_history and len(window_history[w]) > 0 and window_history[w][0].get(metric)]
        values_last = [window_history[w][-1].get(metric) for w in ['0-3', '4-7', '8-14', '15-30', '31-45']
                       if w in window_history and len(window_history[w]) > 1 and window_history[w][-1].get(metric)]
        if len(values_first) >= 2 and len(values_last) >= 2:
            spread_first = max(values_first) - min(values_first)
            spread_last = max(values_last) - min(values_last)
            if spread_first > 0:
                change = (spread_last - spread_first) / spread_first * 100
                convergence[metric] = 'converging' if change < -10 else 'diverging' if change > 10 else 'stable'

    return {
        'window_trends': trends,
        'convergence': convergence,
    }


# ── Macro Regime Scoring ─────────────────────────────────────────────────

def _compute_funding_signal(c):
    """Signal 6: Aggregate Funding Rate (-13 to +13). Returns (score, detail, history)."""
    rows = c.execute(
        '''SELECT date, value FROM coinglass_data
           WHERE symbol='BTC' AND metric='avg_funding_rate'
           ORDER BY date DESC LIMIT 30'''
    ).fetchall()

    if not rows or len(rows) < 7:
        return 0, 'Insufficient funding data', []

    history = [{'date': r[0], 'rate': r[1]} for r in rows]
    history.reverse()  # Oldest first for charts

    rates_7d = [r[1] for r in rows[:7]]
    rates_30d = [r[1] for r in rows[:30]]
    avg_7d = sum(rates_7d) / len(rates_7d)
    avg_30d = sum(rates_30d) / len(rates_30d)

    # Add 7d moving avg to history
    for i, h in enumerate(history):
        start = max(0, i - 6)
        vals = [history[j]['rate'] for j in range(start, i + 1)]
        h['avg_7d'] = sum(vals) / len(vals)

    score = 0
    detail = f'7d avg: {avg_7d:.4%}, 30d avg: {avg_30d:.4%}'

    # Normal BTC funding is slightly positive (0.005-0.015%) - thresholds account for asymmetry
    if avg_7d < -0.0001:  # < -0.01%
        if avg_7d > avg_30d:  # Reverting toward 0
            score = 13
            detail += ' | Deeply negative, reverting up'
        else:
            score = 8
            detail += ' | Deeply negative, still falling'
    elif avg_7d > 0.0003:  # > 0.03%
        if avg_7d < avg_30d:  # Reverting toward 0
            score = -13
            detail += ' | Deeply positive, reverting down'
        else:
            score = -8
            detail += ' | Deeply positive, still rising'

    return score, detail, history


def _compute_oi_signal(c):
    """Signal 7: Aggregate Futures OI (-13 to +13). Returns (score, detail, history)."""
    rows = c.execute(
        '''SELECT date, value FROM coinglass_data
           WHERE symbol='BTC' AND metric='total_oi_usd'
           ORDER BY date DESC LIMIT 90'''
    ).fetchall()

    if not rows or len(rows) < 7:
        return 0, 'Insufficient OI data', []

    history = [{'date': r[0], 'oi_usd': r[1]} for r in rows]
    history.reverse()  # Oldest first

    oi_current = rows[0][1]
    oi_90d_peak = max(r[1] for r in rows)
    change_from_peak = (oi_current - oi_90d_peak) / oi_90d_peak * 100

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
           ORDER BY date DESC LIMIT 7'''
    ).fetchall()
    funding_7d_avg = sum(r[0] for r in funding_rows) / len(funding_rows) if funding_rows else 0

    score = 0
    detail = f'OI ${oi_current/1e9:.1f}B, {change_from_peak:+.1f}% from 90d peak, 7d {oi_7d_slope}'

    if change_from_peak < -20:
        if oi_7d_slope in ('flat', 'rising'):
            score = 13
            detail += ' | Flushed + stabilizing'
        else:
            score = 6
            detail += ' | Flush in progress'
    elif change_from_peak > -5:  # Within 5% of peak
        if funding_7d_avg > 0:
            score = -13
            detail += f' | Near peak + positive funding ({funding_7d_avg:.4%})'
        else:
            score = -6
            detail += f' | Near peak, shorts crowded ({funding_7d_avg:.4%})'

    return score, detail, history


def _compute_liquidation_signal(c):
    """Signal 8: Liquidation Intensity (-13 to +13). Returns (score, detail, history)."""
    # Historical daily totals for chart
    hist_rows = c.execute(
        '''SELECT date, value FROM coinglass_data
           WHERE symbol='BTC' AND metric='total_liquidation_usd'
           ORDER BY date DESC LIMIT 90'''
    ).fetchall()

    history = [{'date': r[0], 'liq_usd': r[1]} for r in hist_rows] if hist_rows else []
    history.reverse()  # Oldest first for charts

    # Today's long/short snapshot
    snap = c.execute(
        '''SELECT value, extra_json FROM coinglass_data
           WHERE symbol='BTC' AND metric='liquidation_snapshot'
           ORDER BY date DESC LIMIT 1'''
    ).fetchone()

    if not snap or not snap[1]:
        return 0, 'No liquidation data', history

    total_24h = snap[0]
    try:
        extra = json.loads(snap[1])
    except (json.JSONDecodeError, TypeError):
        return 0, 'Bad liquidation snapshot', history

    long_24h = extra.get('long_24h', 0)
    short_24h = extra.get('short_24h', 0)

    # 30-day average daily liquidation for intensity baseline
    if len(hist_rows) >= 7:
        avg_30d = sum(r[1] for r in hist_rows[:30]) / min(30, len(hist_rows))
    else:
        return 0, 'Insufficient liquidation history', history

    # Intensity multiplier: how extreme is today vs rolling average
    # 1x = normal, 2x = elevated, 3x+ = massive
    intensity = total_24h / avg_30d if avg_30d > 0 else 0

    # Dominant side
    long_ratio = long_24h / total_24h if total_24h > 0 else 0.5
    dominant_side = 'long' if long_ratio > 0.65 else ('short' if long_ratio < 0.35 else 'balanced')

    # Cross-reference BTC 7d price slope from candles
    candle_rows = c.execute(
        '''SELECT close FROM btc_candles
           WHERE symbol='BTCUSDT' AND tf='1d'
           ORDER BY time DESC LIMIT 7'''
    ).fetchall()

    price_slope = 'unknown'
    if candle_rows and len(candle_rows) >= 2:
        latest = candle_rows[0][0]
        oldest = candle_rows[-1][0]
        pct_chg = (latest - oldest) / oldest * 100 if oldest else 0
        if pct_chg > 2:
            price_slope = 'rising'
        elif pct_chg < -2:
            price_slope = 'falling'
        else:
            price_slope = 'flat'

    score = 0
    detail = (f'24h: ${total_24h/1e6:.0f}M (L:${long_24h/1e6:.0f}M / S:${short_24h/1e6:.0f}M), '
              f'{intensity:.1f}x avg, {dominant_side}, price 7d {price_slope}')

    # Scoring: intensity determines magnitude, side + price direction determines sign
    # Need both elevated intensity AND a dominant side to generate a signal
    if intensity >= 2.0 and dominant_side != 'balanced':
        # Scale: 2x avg = base score, 3x+ = max score
        magnitude = 6 if intensity < 3.0 else 13

        if dominant_side == 'long':
            # Heavy long liquidation = longs getting flushed = bullish (capitulation)
            if price_slope in ('flat', 'rising'):
                score = magnitude  # Capitulation complete, bottom signal
                detail += ' | Long capitulation + price stabilizing'
            else:
                score = max(6, magnitude - 4)  # Flush in progress, not done yet
                detail += ' | Long flush in progress'
        elif dominant_side == 'short':
            # Heavy short liquidation = shorts getting squeezed = bearish (exhaustion)
            if price_slope in ('flat', 'falling'):
                score = -magnitude  # Squeeze exhaustion, top signal
                detail += ' | Short squeeze exhaustion'
            else:
                score = -max(6, magnitude - 4)  # Squeeze in progress
                detail += ' | Short squeeze in progress'

    return score, detail, history


def _compute_pcr_direction(c, ticker, btc_per_share, days=14):
    """Signal 9: Put/Call Ratio Direction (-8 to +8). Returns (score, detail, history)."""
    rows = []
    pcr_dte = 3
    for try_dte in [3, 7, 14]:
        rows = c.execute(
            'SELECT date, data_json FROM data_cache WHERE ticker=? AND dte=? ORDER BY date DESC LIMIT ?',
            (ticker, try_dte, days)
        ).fetchall()
        if len(rows) >= 5:
            pcr_dte = try_dte
            break

    if not rows or len(rows) < 5:
        return 0, 'Insufficient PCR data', []

    pcr_history = []
    for r in rows:
        try:
            d = json.loads(r[1])
            pcr_val = d.get('levels', {}).get('pcr', None)
            if pcr_val is not None:
                pcr_history.append({'date': r[0], 'pcr': pcr_val})
        except (json.JSONDecodeError, TypeError):
            continue

    pcr_history.reverse()  # oldest first for charts

    if len(pcr_history) < 5:
        return 0, 'Insufficient PCR data', pcr_history

    recent_3 = [p['pcr'] for p in pcr_history[-3:]]
    prior = [p['pcr'] for p in pcr_history[:-3]]
    avg_recent = sum(recent_3) / len(recent_3)
    avg_prior = sum(prior) / len(prior)
    current_pcr = pcr_history[-1]['pcr']

    score = 0
    dte_to_label = {3: '0-3d', 7: '4-7d', 14: '8-14d'}
    dte_label = dte_to_label.get(pcr_dte, f'{pcr_dte}d')
    detail = f'PCR {current_pcr:.2f}, 3d avg {avg_recent:.2f} vs prior {avg_prior:.2f} [{dte_label}]'

    pct_change = (avg_recent - avg_prior) / avg_prior if avg_prior else 0

    # Rising PCR = more puts = fear building = contrarian bullish
    # Falling PCR = more calls = greed building = contrarian bearish
    if pct_change > 0.15:
        score = 8
        detail += ' | PCR surging (fear, contrarian bullish)'
    elif pct_change > 0.08:
        score = 4
        detail += ' | PCR rising (put buying increasing)'
    elif pct_change < -0.15:
        score = -8
        detail += ' | PCR plunging (greed, contrarian bearish)'
    elif pct_change < -0.08:
        score = -4
        detail += ' | PCR falling (call buying increasing)'

    return score, detail, pcr_history


def _compute_wall_breach(c, ticker, btc_per_share):
    """Signal 10: Wall Breach Detection (-13 to +13). Returns (score, detail, [])."""
    for try_dte in [3, 7, 14, 30]:
        row = c.execute(
            'SELECT data_json FROM data_cache WHERE ticker=? AND dte=? ORDER BY date DESC LIMIT 1',
            (ticker, try_dte)
        ).fetchone()
        if row:
            try:
                d = json.loads(row[0])
            except (json.JSONDecodeError, TypeError):
                continue
            spot = d.get('btc_spot', 0)
            comb = d.get('combined_levels_btc') or {}
            lvl = d.get('levels', {})
            cw = comb.get('call_wall') or (lvl.get('call_wall', 0) / btc_per_share if btc_per_share else 0)
            pw = comb.get('put_wall') or (lvl.get('put_wall', 0) / btc_per_share if btc_per_share else 0)
            regime = lvl.get('regime', '')

            if spot and cw and pw:
                break
    else:
        return 0, 'No wall data for breach check', []

    dte_to_label = {3: '0-3d', 7: '4-7d', 14: '8-14d', 30: '15-30d'}
    dte_label = dte_to_label.get(try_dte, f'{try_dte}d')
    detail = f'Spot ${spot:,.0f}, CW ${cw:,.0f}, PW ${pw:,.0f} [{dte_label}]'

    # Breach thresholds: within 0.5% = testing, beyond = breached
    cw_dist_pct = (spot - cw) / cw if cw else 0
    pw_dist_pct = (pw - spot) / pw if pw else 0

    score = 0
    if cw_dist_pct > 0.005:
        # Spot ABOVE call wall
        if regime == 'negative_gamma':
            score = 13
            detail += f' | ABOVE call wall ({cw_dist_pct:.1%}) + neg \u03b3 \u2192 breakout'
        else:
            score = 6
            detail += f' | ABOVE call wall ({cw_dist_pct:.1%}) + pos \u03b3 \u2192 likely pin'
    elif cw_dist_pct > -0.005:
        # Testing call wall
        detail += f' | Testing call wall ({abs(cw_dist_pct):.1%} away)'
    elif pw_dist_pct > 0.005:
        # Spot BELOW put wall
        if regime == 'negative_gamma':
            score = -13
            detail += f' | BELOW put wall ({pw_dist_pct:.1%}) + neg \u03b3 \u2192 breakdown'
        else:
            score = -6
            detail += f' | BELOW put wall ({pw_dist_pct:.1%}) + pos \u03b3 \u2192 dealer support'
    elif pw_dist_pct > -0.005:
        # Testing put wall
        detail += f' | Testing put wall ({abs(pw_dist_pct):.1%} away)'
    else:
        detail += ' | Between walls, no breach'

    return score, detail, []


def _compute_iv_term_signal(c, ticker, btc_per_share, days=14):
    """Signal 11: IV Term Structure Shape (-8 to +8).
    Backwardation (short > long IV) = fear = contrarian bullish.
    Steep contango = complacency = contrarian bearish."""
    iv_ts = []
    used_dte = 3
    for try_dte in [3, 7, 14, 30, 45]:
        row = c.execute(
            'SELECT data_json FROM data_cache WHERE ticker=? AND dte=? ORDER BY date DESC LIMIT 1',
            (ticker, try_dte)
        ).fetchone()
        if row:
            try:
                d = json.loads(row[0])
            except (json.JSONDecodeError, TypeError):
                continue
            iv_ts = d.get('iv_term_structure', [])
            if len(iv_ts) >= 2:
                used_dte = try_dte
                break

    if not iv_ts or len(iv_ts) < 2:
        return 0, 'No IV term structure data', []

    # Group by source, prefer deribit (more expiries), fall back to ibit
    by_source = {}
    for entry in iv_ts:
        src = entry.get('source', 'ibit')
        by_source.setdefault(src, []).append(entry)

    source = 'deribit' if len(by_source.get('deribit', [])) >= 2 else 'ibit'
    entries = sorted(by_source.get(source, []), key=lambda e: e['dte'])

    if len(entries) < 2:
        return 0, 'Need at least 2 expiries for term structure', []

    # Build history for charts
    history = []
    hist_rows = c.execute(
        'SELECT date, data_json FROM data_cache WHERE ticker=? AND dte=? ORDER BY date DESC LIMIT ?',
        (ticker, used_dte, days)
    ).fetchall()

    for hr in hist_rows:
        try:
            hd = json.loads(hr[1])
            h_iv_ts = hd.get('iv_term_structure', [])
            h_entries = sorted(
                [e for e in h_iv_ts if e.get('source') == source],
                key=lambda e: e['dte']
            )
            if len(h_entries) >= 2:
                short_iv = h_entries[0]['atm_iv']
                long_iv = h_entries[-1]['atm_iv']
                slope = (long_iv - short_iv) / short_iv if short_iv > 0 else 0
                history.append({
                    'date': hr[0],
                    'short_iv': short_iv, 'long_iv': long_iv,
                    'short_dte': h_entries[0]['dte'],
                    'long_dte': h_entries[-1]['dte'],
                    'slope': round(slope, 4),
                })
        except (json.JSONDecodeError, TypeError, KeyError):
            continue

    history.reverse()  # oldest first for charts

    # Current term structure slope
    short_entry = entries[0]
    long_entry = entries[-1]
    short_iv = short_entry['atm_iv']
    long_iv = long_entry['atm_iv']

    if short_iv <= 0:
        return 0, 'Invalid short-dated IV', history

    slope = (long_iv - short_iv) / short_iv

    score = 0
    detail = (f'Short IV {short_iv:.1%} ({short_entry["dte"]}d) vs '
              f'Long IV {long_iv:.1%} ({long_entry["dte"]}d), '
              f'slope {slope:+.1%} [{source}]')

    if slope < -0.10:
        score = 8
        detail += ' | Strong backwardation (fear, contrarian bullish)'
    elif slope < -0.03:
        score = 4
        detail += ' | Mild backwardation (hedging demand)'
    elif slope > 0.25:
        score = -8
        detail += ' | Steep contango (complacency, contrarian bearish)'
    elif slope > 0.15:
        score = -4
        detail += ' | Elevated contango (low near-term fear)'

    return score, detail, history


def _compute_score_history(conn, ticker, days=30):
    """Recompute full macro score (all 11 signals) for each of the last N days."""
    c = conn.cursor()
    cfg = TICKER_CONFIG.get(ticker, TICKER_CONFIG['IBIT'])
    btc_per_share = cfg.get('per_share') or cfg.get('per_share_default', 0.000568)

    # ── Pre-load all data once ────────────────────────────────────────────
    lookback = days + 30  # Extra rows for signal lookback windows

    all_snapshots = c.execute(
        'SELECT date, regime, net_gex FROM snapshots WHERE ticker=? ORDER BY date DESC LIMIT ?',
        (ticker, lookback)
    ).fetchall()

    all_cache = c.execute(
        'SELECT date, data_json FROM data_cache WHERE ticker=? AND dte=45 ORDER BY date DESC LIMIT ?',
        (ticker, lookback)
    ).fetchall()

    all_flows = c.execute(
        '''SELECT date, daily_flow_dollars, total_btc_etf_flow
           FROM etf_flows WHERE ticker=? ORDER BY date DESC LIMIT ?''',
        (ticker, lookback)
    ).fetchall()

    all_funding = c.execute(
        'SELECT date, value FROM coinglass_data WHERE symbol=? AND metric=? ORDER BY date DESC LIMIT 90',
        ('BTC', 'avg_funding_rate')
    ).fetchall()

    all_oi = c.execute(
        'SELECT date, value FROM coinglass_data WHERE symbol=? AND metric=? ORDER BY date DESC LIMIT 90',
        ('BTC', 'total_oi_usd')
    ).fetchall()

    all_liquidation = c.execute(
        'SELECT date, value, extra_json FROM coinglass_data WHERE symbol=? AND metric=? ORDER BY date DESC LIMIT 90',
        ('BTC', 'total_liquidation_usd')
    ).fetchall()

    all_liq_snapshots = c.execute(
        'SELECT date, value, extra_json FROM coinglass_data WHERE symbol=? AND metric=? ORDER BY date DESC LIMIT 90',
        ('BTC', 'liquidation_snapshot')
    ).fetchall()

    # Get dates we want scores for (oldest first)
    dates = list(dict.fromkeys(r[0] for r in all_snapshots[:days]))
    dates.reverse()

    # ── Pre-parse cache JSON once, keyed by date ──────────────────────────
    parsed_cache = {}  # date -> {call_wall, put_wall, deribit_cw, deribit_pw, ibit_cw, ibit_pw}
    for row in all_cache:
        if row[0] in parsed_cache:
            continue
        try:
            d = json.loads(row[1])
            comb = d.get('combined_levels_btc') or {}
            levels = d.get('levels', {})
            deribit = d.get('deribit_levels_btc') or {}
            cw = comb.get('call_wall') or (levels.get('call_wall', 0) / btc_per_share if btc_per_share else 0)
            pw = comb.get('put_wall') or (levels.get('put_wall', 0) / btc_per_share if btc_per_share else 0)
            parsed_cache[row[0]] = {
                'call_wall': cw, 'put_wall': pw,
                'ibit_cw': levels.get('call_wall', 0) / btc_per_share if btc_per_share else 0,
                'ibit_pw': levels.get('put_wall', 0) / btc_per_share if btc_per_share else 0,
                'deribit_cw': deribit.get('call_wall', 0),
                'deribit_pw': deribit.get('put_wall', 0),
                'has_deribit': bool(comb and deribit),
                'pcr': levels.get('pcr'),
                'btc_spot': d.get('btc_spot', 0),
                'regime': levels.get('regime', ''),
                'iv_term_structure': d.get('iv_term_structure', []),
            }
        except (json.JSONDecodeError, TypeError):
            continue

    score_history = []
    for date_str in dates:
        # ── Signal 1: Regime Persistence & Transition ─────────────────
        snaps = [r for r in all_snapshots if r[0] <= date_str][:days]
        regime_s = 0
        if snaps:
            total = len(snaps)
            neg_count = sum(1 for r in snaps if r[1] == 'negative_gamma')
            pos_count = total - neg_count
            neg_pct = neg_count / total
            pos_pct = pos_count / total

            if neg_pct > 0.70:
                regime_s = 8
            elif pos_pct > 0.70:
                regime_s = -8

            # Transition bonus
            if len(snaps) >= 4:
                recent_regime = snaps[0][1]
                recent_streak = 1
                for i in range(1, min(4, len(snaps))):
                    if snaps[i][1] == recent_regime:
                        recent_streak += 1
                    else:
                        break
                if recent_streak <= 3:
                    old_count = 0
                    for i in range(recent_streak, len(snaps)):
                        if snaps[i][1] != recent_regime:
                            old_count += 1
                        else:
                            break
                    if old_count >= 7:
                        regime_s = 12 if recent_regime == 'positive_gamma' else -12

        # ── Signal 2: Wall Migration ──────────────────────────────────
        wall_s = 0
        cache_dates = sorted(dt for dt in parsed_cache if dt <= date_str)
        wall_entries = [parsed_cache[dt] for dt in cache_dates if parsed_cache[dt]['call_wall'] and parsed_cache[dt]['put_wall']]

        if len(wall_entries) >= 7:
            half = len(wall_entries) // 2
            first_half = wall_entries[:half]
            last_half = wall_entries[half:]
            avg_pw_f = sum(w['put_wall'] for w in first_half) / len(first_half)
            avg_pw_l = sum(w['put_wall'] for w in last_half) / len(last_half)
            avg_cw_f = sum(w['call_wall'] for w in first_half) / len(first_half)
            avg_cw_l = sum(w['call_wall'] for w in last_half) / len(last_half)
            pw_chg = (avg_pw_l - avg_pw_f) / avg_pw_f if avg_pw_f else 0
            cw_chg = (avg_cw_l - avg_cw_f) / avg_cw_f if avg_cw_f else 0
            wall_s = (6 if pw_chg > 0.02 else (-6 if pw_chg < -0.02 else 0)) + \
                     (6 if cw_chg > 0.02 else (-6 if cw_chg < -0.02 else 0))

        # ── Signal 3: Range Compression ───────────────────────────────
        comp_s = 0
        if len(wall_entries) >= 10:
            ranges = [w['call_wall'] - w['put_wall'] for w in wall_entries if w['call_wall'] > w['put_wall']]
            if ranges:
                current_range = ranges[-1]  # Most recent
                sorted_ranges = sorted(ranges)
                below = sum(1 for r in sorted_ranges if r < current_range)
                percentile = (below / len(sorted_ranges)) * 100
                if percentile <= 20:
                    cur_regime = snaps[0][1] if snaps else None
                    if cur_regime == 'positive_gamma':
                        comp_s = -12
                    elif cur_regime == 'negative_gamma':
                        comp_s = 12

        # ── Signal 4: ETF Flow Momentum ───────────────────────────────
        flow_s = 0
        flows_up_to = [r for r in all_flows if r[0] <= date_str]
        if flows_up_to:
            flow_vals = [(r[2] if r[2] is not None else r[1]) or 0 for r in flows_up_to]
            flow_10d = sum(flow_vals[:10]) if len(flow_vals) >= 10 else sum(flow_vals)

            if len(flow_vals) >= 15:
                flow_30d = sum(flow_vals[:30]) if len(flow_vals) >= 30 else sum(flow_vals)
                f5_recent = sum(flow_vals[:5])
                f5_prior = sum(flow_vals[5:10])
                if f5_prior < 0 and f5_recent > 0:
                    flow_s = 12
                elif f5_prior > 0 and f5_recent < 0:
                    flow_s = -12
                elif flow_30d != 0:
                    pace_10d = flow_10d / min(10, len(flow_vals))
                    pace_30d = flow_30d / min(30, len(flow_vals))
                    if flow_10d > 0 and pace_10d > pace_30d:
                        flow_s = 8
                    elif flow_10d < 0 and abs(pace_10d) > abs(pace_30d):
                        flow_s = -8
                    elif pace_30d != 0:
                        flow_s = max(-8, min(8, int((pace_10d / pace_30d) * 8)))

        # ── Signal 5: Venue Wall Convergence ──────────────────────────
        venue_s = 0
        if cache_dates:
            latest = parsed_cache[cache_dates[-1]]
            if latest['has_deribit']:
                combined_pw = latest['put_wall']
                combined_cw = latest['call_wall']
                d_pw = latest['deribit_pw']
                d_cw = latest['deribit_cw']
                # Use dte=45 threshold for score history (structural data)
                conv_thresh = 0.04

                if combined_pw and d_pw and combined_pw > 0:
                    pw_diff = abs(combined_pw - d_pw) / combined_pw
                    if pw_diff <= conv_thresh and len(wall_entries) >= 7:
                        pw_rising = wall_entries[-1]['put_wall'] > wall_entries[0]['put_wall'] * 1.02
                        venue_s += 12 if pw_rising else 6

                if combined_cw and d_cw and combined_cw > 0:
                    cw_diff = abs(combined_cw - d_cw) / combined_cw
                    if cw_diff <= conv_thresh and len(wall_entries) >= 7:
                        cw_falling = wall_entries[-1]['call_wall'] < wall_entries[0]['call_wall'] * 0.98
                        venue_s -= 12 if cw_falling else 6

                venue_s = max(-12, min(12, venue_s))

        # ── Signal 6: Funding Rate ────────────────────────────────────
        funding_s = 0
        fr = [r for r in all_funding if r[0] <= date_str]
        if len(fr) >= 7:
            avg_7d = sum(r[1] for r in fr[:7]) / 7
            avg_30d = sum(r[1] for r in fr[:30]) / min(30, len(fr))
            if avg_7d < -0.0001:
                funding_s = 13 if avg_7d > avg_30d else 8
            elif avg_7d > 0.0003:
                funding_s = -13 if avg_7d < avg_30d else -8

        # ── Signal 7: Open Interest ───────────────────────────────────
        oi_s = 0
        oi_slope = 'unknown'
        oi = [r for r in all_oi if r[0] <= date_str]
        if len(oi) >= 7:
            oi_current = oi[0][1]
            oi_peak = max(r[1] for r in oi)
            chg_from_peak = (oi_current - oi_peak) / oi_peak * 100 if oi_peak else 0

            oi_7d_ago = oi[6][1]
            oi_slope = 'rising' if oi_current > oi_7d_ago * 1.02 else (
                'falling' if oi_current < oi_7d_ago * 0.98 else 'flat')

            fr_for_oi = [r for r in all_funding if r[0] <= date_str][:7]
            funding_avg = sum(r[1] for r in fr_for_oi) / len(fr_for_oi) if fr_for_oi else 0

            if chg_from_peak < -20:
                oi_s = 13 if oi_slope in ('flat', 'rising') else 6
            elif chg_from_peak > -5:
                oi_s = -13 if funding_avg > 0 else -6

        # ── Signal 8: Liquidation Intensity ───────────────────────────
        liq_s = 0
        liq_snap = next((r for r in all_liq_snapshots if r[0] <= date_str), None)
        liq_hist = [r for r in all_liquidation if r[0] <= date_str]
        if liq_snap and liq_snap[2] and len(liq_hist) >= 7:
            try:
                liq_extra = json.loads(liq_snap[2])
                long_24h = liq_extra.get('long_24h', 0)
                short_24h = liq_extra.get('short_24h', 0)
                total_24h = long_24h + short_24h
                avg_30d = sum(r[1] for r in liq_hist[:30]) / min(30, len(liq_hist))
                intensity = total_24h / avg_30d if avg_30d > 0 else 0
                long_ratio = long_24h / total_24h if total_24h > 0 else 0.5

                if intensity >= 2.0 and (long_ratio > 0.65 or long_ratio < 0.35):
                    magnitude = 6 if intensity < 3.0 else 13
                    if long_ratio > 0.65:  # Long dominant
                        liq_s = magnitude if oi_slope != 'falling' else max(6, magnitude - 4)
                    else:  # Short dominant
                        liq_s = -magnitude if oi_slope != 'rising' else -max(6, magnitude - 4)
            except (json.JSONDecodeError, TypeError):
                pass

        # ── Signal 9: PCR Direction ───────────────────────────────────
        pcr_s = 0
        # Use parsed_cache for PCR — it stores levels.pcr from dte=45 data
        # For history, just check if PCR data exists in the latest entry
        if cache_dates:
            latest_pc = parsed_cache[cache_dates[-1]]
            pcr_val = latest_pc.get('pcr')
            if pcr_val is not None:
                # Build PCR series from available dates
                pcr_vals = []
                for dt in cache_dates:
                    pv = parsed_cache[dt].get('pcr')
                    if pv is not None:
                        pcr_vals.append(pv)
                if len(pcr_vals) >= 5:
                    recent_3 = pcr_vals[-3:]
                    prior_pcr = pcr_vals[:-3]
                    if prior_pcr:
                        avg_r = sum(recent_3) / len(recent_3)
                        avg_p = sum(prior_pcr) / len(prior_pcr)
                        pct_chg = (avg_r - avg_p) / avg_p if avg_p else 0
                        if pct_chg > 0.15:
                            pcr_s = 8
                        elif pct_chg > 0.08:
                            pcr_s = 4
                        elif pct_chg < -0.15:
                            pcr_s = -8
                        elif pct_chg < -0.08:
                            pcr_s = -4

        # ── Signal 10: Wall Breach Detection ─────────────────────────
        breach_s = 0
        if cache_dates:
            latest_wb = parsed_cache[cache_dates[-1]]
            wb_spot = latest_wb.get('btc_spot', 0)
            wb_cw = latest_wb.get('call_wall', 0)
            wb_pw = latest_wb.get('put_wall', 0)
            wb_regime = latest_wb.get('regime', '')
            if wb_spot and wb_cw and wb_pw:
                cw_dist = (wb_spot - wb_cw) / wb_cw if wb_cw else 0
                pw_dist = (wb_pw - wb_spot) / wb_pw if wb_pw else 0
                if cw_dist > 0.005:
                    breach_s = 13 if wb_regime == 'negative_gamma' else 6
                elif pw_dist > 0.005:
                    breach_s = -13 if wb_regime == 'negative_gamma' else -6

        # ── Signal 11: IV Term Structure Shape ────────────────────────
        iv_term_s = 0
        if cache_dates:
            latest_iv = parsed_cache[cache_dates[-1]]
            iv_ts_data = latest_iv.get('iv_term_structure', [])
            # Group by source, prefer deribit
            iv_by_src = {}
            for iv_e in iv_ts_data:
                iv_by_src.setdefault(iv_e.get('source', 'ibit'), []).append(iv_e)
            iv_src = 'deribit' if len(iv_by_src.get('deribit', [])) >= 2 else 'ibit'
            iv_entries = sorted(iv_by_src.get(iv_src, []), key=lambda e: e.get('dte', 0))
            if len(iv_entries) >= 2:
                s_iv = iv_entries[0].get('atm_iv', 0)
                l_iv = iv_entries[-1].get('atm_iv', 0)
                if s_iv > 0:
                    iv_slope = (l_iv - s_iv) / s_iv
                    if iv_slope < -0.10:
                        iv_term_s = 8
                    elif iv_slope < -0.03:
                        iv_term_s = 4
                    elif iv_slope > 0.25:
                        iv_term_s = -8
                    elif iv_slope > 0.15:
                        iv_term_s = -4

        day_total = max(-100, min(100,
            regime_s + wall_s + comp_s + flow_s + venue_s + funding_s + oi_s + liq_s + pcr_s + breach_s + iv_term_s))

        score_history.append({
            'date': date_str,
            'total_score': day_total,
            'regime': regime_s,
            'walls': wall_s,
            'compression': comp_s,
            'flow': flow_s,
            'venue': venue_s,
            'funding': funding_s,
            'oi': oi_s,
            'liquidation': liq_s,
            'pcr': pcr_s,
            'breach': breach_s,
            'iv_term': iv_term_s,
        })

    return score_history


def compute_macro_regime(conn, ticker, days=30):
    """Compute macro regime score from -100 (top) to +100 (bottom).
    Returns dict with overall score, per-signal breakdown, and swing recommendation."""
    c = conn.cursor()
    cfg = TICKER_CONFIG.get(ticker, TICKER_CONFIG['IBIT'])
    btc_per_share = cfg.get('per_share') or cfg.get('per_share_default', 0.000568)
    today = today_str_et()

    # ── Signal 1: Regime Persistence & Transition (-12 to +12) ───────────
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

        flips = sum(1 for i in range(1, len(rows)) if rows[i][1] != rows[i-1][1])

        neg_pct = neg_count / total if total > 0 else 0
        pos_pct = pos_count / total if total > 0 else 0

        if neg_pct > 0.70:
            regime_score = 8
            regime_detail = f'{neg_count}/{total} days negative gamma ({neg_pct:.0%})'
        elif pos_pct > 0.70:
            regime_score = -8
            regime_detail = f'{pos_count}/{total} days positive gamma ({pos_pct:.0%})'
        else:
            regime_detail = f'Oscillating: {pos_count} pos / {neg_count} neg, {flips} flips'

        # Transition bonus: persistent regime just flipped
        if len(rows) >= 4:
            recent_regime = rows[0][1]
            recent_streak = 1
            for i in range(1, min(4, len(rows))):
                if rows[i][1] == recent_regime:
                    recent_streak += 1
                else:
                    break

            if recent_streak <= 3:
                old_regime_count = 0
                for i in range(recent_streak, len(rows)):
                    if rows[i][1] != recent_regime:
                        old_regime_count += 1
                    else:
                        break

                if old_regime_count >= 7:
                    if recent_regime == 'positive_gamma':
                        regime_score = 12
                        regime_detail += f' | TRANSITION: {old_regime_count}d neg->pos (streak {recent_streak}d)'
                    else:
                        regime_score = -12
                        regime_detail += f' | TRANSITION: {old_regime_count}d pos->neg (streak {recent_streak}d)'

    # ── Signal 2: Structural Wall Migration (-12 to +12) ─────────────────
    wall_migration_score = 0
    wall_detail = ''
    wall_history = []

    cache_rows = c.execute(
        'SELECT date, data_json FROM data_cache WHERE ticker=? AND dte=45 ORDER BY date DESC LIMIT ?',
        (ticker, days)
    ).fetchall()

    if len(cache_rows) >= 7:
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

        if len(wall_history) >= 7:
            half = len(wall_history) // 2
            first_half = wall_history[:half]
            last_half = wall_history[half:]

            avg_pw_first = sum(w['put_wall'] for w in first_half) / len(first_half)
            avg_pw_last = sum(w['put_wall'] for w in last_half) / len(last_half)
            avg_cw_first = sum(w['call_wall'] for w in first_half) / len(first_half)
            avg_cw_last = sum(w['call_wall'] for w in last_half) / len(last_half)

            pw_change = (avg_pw_last - avg_pw_first) / avg_pw_first if avg_pw_first else 0
            cw_change = (avg_cw_last - avg_cw_first) / avg_cw_first if avg_cw_first else 0

            pw_part = 6 if pw_change > 0.02 else (-6 if pw_change < -0.02 else 0)
            cw_part = 6 if cw_change > 0.02 else (-6 if cw_change < -0.02 else 0)
            pw_dir = 'rising' if pw_change > 0.02 else ('falling' if pw_change < -0.02 else 'stable')
            cw_dir = 'rising' if cw_change > 0.02 else ('falling' if cw_change < -0.02 else 'stable')

            wall_migration_score = pw_part + cw_part
            wall_detail = f'Put wall {pw_dir} ({pw_change:+.1%}), Call wall {cw_dir} ({cw_change:+.1%})'

    if not wall_detail:
        wall_detail = f'Insufficient data ({len(cache_rows)} days, need 7+ of 31-45d cache)'

    # ── Signal 3: Range Compression + Spot Position (-12 to +12) ─────────
    # Compressed range = coiled spring. Direction depends on WHERE spot sits
    # within the range and whether gamma regime amplifies or dampens.
    compression_score = 0
    compression_detail = ''

    # Get spot BTC and current walls from freshest available data
    spot_btc = 0
    current_cw = 0
    current_pw = 0
    for try_dte in [3, 7, 14, 30, 45]:
        fresh = c.execute(
            'SELECT data_json FROM data_cache WHERE ticker=? AND dte=? ORDER BY date DESC LIMIT 1',
            (ticker, try_dte)
        ).fetchone()
        if fresh:
            fd = json.loads(fresh[0])
            sb = fd.get('btc_spot', 0)
            comb = fd.get('combined_levels_btc') or {}
            lvl = fd.get('levels', {})
            cw = comb.get('call_wall') or (lvl.get('call_wall', 0) / btc_per_share if btc_per_share else 0)
            pw = comb.get('put_wall') or (lvl.get('put_wall', 0) / btc_per_share if btc_per_share else 0)
            if sb and cw and pw:
                spot_btc = sb
                current_cw = cw
                current_pw = pw
                break

    # Build range history from same DTE window as spot data
    compression_dte = 3
    range_rows = []
    for try_dte in [3, 7, 14, 30, 45]:
        range_rows = c.execute(
            'SELECT date, data_json FROM data_cache WHERE ticker=? AND dte=? ORDER BY date DESC LIMIT ?',
            (ticker, try_dte, days)
        ).fetchall()
        if len(range_rows) >= 7:
            compression_dte = try_dte
            break

    ranges = []
    for row in range_rows:
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
            current_range = ranges[0]
            sorted_ranges = sorted(ranges)
            below = sum(1 for r in sorted_ranges if r < current_range)
            percentile = (below / len(sorted_ranges)) * 100

            if percentile <= 20 and spot_btc and current_cw and current_pw and current_cw > current_pw:
                # Compressed — score by spot position + regime
                spot_pct = (spot_btc - current_pw) / (current_cw - current_pw)
                spot_pct = max(0, min(1, spot_pct))  # clamp
                current_regime = rows[0][1] if rows else None

                if current_regime == 'positive_gamma':
                    # Positive gamma: walls hold. Near call wall = rejection, near put wall = bounce
                    if spot_pct > 0.75:
                        compression_score = -12
                        compression_detail = f'{percentile:.0f}th pctl range, spot at {spot_pct:.0%} (call wall) + pos \u03b3 \u2192 rejection'
                    elif spot_pct > 0.60:
                        compression_score = -6
                        compression_detail = f'{percentile:.0f}th pctl range, spot at {spot_pct:.0%} (lean resistance) + pos \u03b3'
                    elif spot_pct < 0.25:
                        compression_score = 12
                        compression_detail = f'{percentile:.0f}th pctl range, spot at {spot_pct:.0%} (put wall) + pos \u03b3 \u2192 bounce'
                    elif spot_pct < 0.40:
                        compression_score = 6
                        compression_detail = f'{percentile:.0f}th pctl range, spot at {spot_pct:.0%} (lean support) + pos \u03b3'
                    else:
                        compression_score = 0
                        compression_detail = f'{percentile:.0f}th pctl range, spot at {spot_pct:.0%} (mid-range) + pos \u03b3 \u2192 range-bound'

                elif current_regime == 'negative_gamma':
                    # Negative gamma: walls break. Near call wall = breakout up, near put wall = breakdown
                    if spot_pct > 0.75:
                        compression_score = 12
                        compression_detail = f'{percentile:.0f}th pctl range, spot at {spot_pct:.0%} (call wall) + neg \u03b3 \u2192 breakout'
                    elif spot_pct > 0.60:
                        compression_score = 6
                        compression_detail = f'{percentile:.0f}th pctl range, spot at {spot_pct:.0%} (lean resistance) + neg \u03b3'
                    elif spot_pct < 0.25:
                        compression_score = -12
                        compression_detail = f'{percentile:.0f}th pctl range, spot at {spot_pct:.0%} (put wall) + neg \u03b3 \u2192 breakdown'
                    elif spot_pct < 0.40:
                        compression_score = -6
                        compression_detail = f'{percentile:.0f}th pctl range, spot at {spot_pct:.0%} (lean support) + neg \u03b3'
                    else:
                        compression_score = 0
                        compression_detail = f'{percentile:.0f}th pctl range, spot at {spot_pct:.0%} (mid-range) + neg \u03b3 \u2192 volatile'
                else:
                    compression_detail = f'{percentile:.0f}th pctl range, spot at {spot_pct:.0%}, oscillating regime'

            elif percentile <= 20:
                compression_detail = f'{percentile:.0f}th pctl range (compressed, no wall data for position)'
            else:
                compression_detail = f'{percentile:.0f}th pctl range (not compressed)'

    if not compression_detail:
        compression_detail = 'Insufficient data for range analysis'

    # ── Signal 4: ETF Flow Momentum (-12 to +12) ────────────────────────
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
            {'date': r[0], 'ibit_flow': r[1], 'total_flow': r[2], 'cumulative_10d': None}
            for r in flow_rows
        ]
        flow_history.reverse()  # Oldest first

        # Compute cumulative 10d for history
        for i, fh in enumerate(flow_history):
            start = max(0, i - 9)
            cum = sum(
                (flow_history[j].get('total_flow') or flow_history[j].get('ibit_flow') or 0)
                for j in range(start, i + 1)
            )
            fh['cumulative_10d'] = cum

        # Use total_btc_etf_flow, fall back to daily_flow_dollars
        flows = []
        for r in flow_rows:
            val = r[2] if r[2] is not None else r[1]
            flows.append(val or 0)

        flow_10d = sum(flows[:10]) if len(flows) >= 10 else sum(flows)
        flow_30d = sum(flows[:30]) if len(flows) >= 30 else sum(flows)

        if len(flows) >= 15:
            flow_5d_recent = sum(flows[:5])
            flow_5d_prior = sum(flows[5:10])

            if flow_5d_prior < 0 and flow_5d_recent > 0:
                etf_flow_score = 12
                flow_detail = f'Flow reversal: prior ${flow_5d_prior/1e6:.0f}M -> recent ${flow_5d_recent/1e6:.0f}M'
            elif flow_5d_prior > 0 and flow_5d_recent < 0:
                etf_flow_score = -12
                flow_detail = f'Flow reversal: prior ${flow_5d_prior/1e6:.0f}M -> recent ${flow_5d_recent/1e6:.0f}M'
            elif flow_30d != 0:
                pace_10d = flow_10d / min(10, len(flows))
                pace_30d = flow_30d / min(30, len(flows))

                if flow_10d > 0 and pace_10d > pace_30d:
                    etf_flow_score = 8
                    flow_detail = f'Accelerating inflows: 10d ${flow_10d/1e6:.0f}M'
                elif flow_10d < 0 and abs(pace_10d) > abs(pace_30d):
                    etf_flow_score = -8
                    flow_detail = f'Accelerating outflows: 10d ${flow_10d/1e6:.0f}M'
                else:
                    ratio = pace_10d / pace_30d if pace_30d != 0 else 0
                    etf_flow_score = max(-8, min(8, int(ratio * 8)))
                    flow_detail = f'10d flow ${flow_10d/1e6:.0f}M, 10d/30d ratio {ratio:.2f}'
            else:
                flow_detail = 'Flat flows'
        elif flows:
            flow_detail = f'Only {len(flows)} days of flow data'

    if not flow_detail:
        flow_detail = 'No ETF flow data'

    # ── Signal 5: Venue Wall Convergence (-12 to +12) ────────────────────
    # Compare IBIT and Deribit wall positions. When both venues agree on
    # where support/resistance sits, that level is high conviction.
    #
    # Try DTE windows from widest to narrowest to find one with Deribit data.
    # Wider windows capture more structural positioning but Deribit coverage
    # is spotty at longer tenors.
    venue_score = 0
    venue_detail = ''

    CONVERGENCE_THRESHOLDS = {3: 0.02, 7: 0.025, 14: 0.03, 30: 0.035, 45: 0.04}

    venue_data = None
    venue_dte_label = ''
    venue_dte_used = 3
    for try_dte in [3, 7, 14, 30, 45]:
        row = c.execute(
            'SELECT data_json FROM data_cache WHERE ticker=? AND dte=? ORDER BY date DESC LIMIT 1',
            (ticker, try_dte)
        ).fetchone()
        if row:
            d = json.loads(row[0])
            if d.get('combined_levels_btc') and d.get('deribit_levels_btc'):
                venue_data = d
                venue_dte_used = try_dte
                dte_to_label = {3: '0-3d', 7: '4-7d', 14: '8-14d', 30: '15-30d', 45: '31-45d'}
                venue_dte_label = dte_to_label.get(try_dte, f'{try_dte}d')
                break

    if venue_data:
        try:
            comb = venue_data.get('combined_levels_btc') or {}
            deribit = venue_data.get('deribit_levels_btc') or {}

            combined_cw = comb.get('call_wall', 0)
            combined_pw = comb.get('put_wall', 0)
            deribit_cw = deribit.get('call_wall', 0)
            deribit_pw = deribit.get('put_wall', 0)

            convergence_pct = CONVERGENCE_THRESHOLDS.get(venue_dte_used, 0.04)

            pw_converging = False
            cw_converging = False
            pw_diff_pct = 0
            cw_diff_pct = 0

            if combined_pw and deribit_pw and combined_pw > 0:
                pw_diff_pct = abs(combined_pw - deribit_pw) / combined_pw
                if pw_diff_pct <= convergence_pct:
                    pw_converging = True

            if combined_cw and deribit_cw and combined_cw > 0:
                cw_diff_pct = abs(combined_cw - deribit_cw) / combined_cw
                if cw_diff_pct <= convergence_pct:
                    cw_converging = True

            thresh_label = f'{convergence_pct:.0%} thresh'

            if pw_converging and cw_converging:
                if wall_history and len(wall_history) >= 7:
                    pw_rising = wall_history[-1]['put_wall'] > wall_history[0]['put_wall'] * 1.02
                    cw_falling = wall_history[-1]['call_wall'] < wall_history[0]['call_wall'] * 0.98
                    pw_falling = wall_history[-1]['put_wall'] < wall_history[0]['put_wall'] * 0.98
                    cw_rising = wall_history[-1]['call_wall'] > wall_history[0]['call_wall'] * 1.02

                    if pw_rising and cw_falling:
                        venue_score = 12
                        venue_detail = f'Venue range pinch upward — PW conv {pw_diff_pct:.1%}, CW conv {cw_diff_pct:.1%} [{venue_dte_label}] ({thresh_label}, migration from 31-45d)'
                    elif pw_falling and cw_rising:
                        venue_score = -12
                        venue_detail = f'Venue range pinch downward — PW conv {pw_diff_pct:.1%}, CW conv {cw_diff_pct:.1%} [{venue_dte_label}] ({thresh_label}, migration from 31-45d)'
                    elif pw_rising:
                        venue_score = 8
                        venue_detail = f'Venue range confirmed, support rising — PW {pw_diff_pct:.1%}, CW {cw_diff_pct:.1%} [{venue_dte_label}] ({thresh_label}, migration from 31-45d)'
                    elif cw_falling:
                        venue_score = -8
                        venue_detail = f'Venue range confirmed, resistance falling — PW {pw_diff_pct:.1%}, CW {cw_diff_pct:.1%} [{venue_dte_label}] ({thresh_label}, migration from 31-45d)'
                    else:
                        venue_score = 0
                        venue_detail = f'Venue range confirmed, stable — PW {pw_diff_pct:.1%}, CW {cw_diff_pct:.1%} [{venue_dte_label}] ({thresh_label}, migration from 31-45d)'
                else:
                    venue_score = 0
                    venue_detail = f'Venue range confirmed (no migration data) — PW {pw_diff_pct:.1%}, CW {cw_diff_pct:.1%} [{venue_dte_label}] ({thresh_label})'

            elif pw_converging:
                if wall_history and len(wall_history) >= 7:
                    pw_rising = wall_history[-1]['put_wall'] > wall_history[0]['put_wall'] * 1.02
                    if pw_rising:
                        venue_score = 12
                        venue_detail = f'Put walls converging ({pw_diff_pct:.1%}) + rising [{venue_dte_label}] ({thresh_label}, migration from 31-45d)'
                    else:
                        venue_score = 6
                        venue_detail = f'Put walls converging ({pw_diff_pct:.1%}), stable/falling [{venue_dte_label}] ({thresh_label}, migration from 31-45d)'
                else:
                    venue_score = 4
                    venue_detail = f'Put walls converging ({pw_diff_pct:.1%}) [{venue_dte_label}] ({thresh_label})'

            elif cw_converging:
                if wall_history and len(wall_history) >= 7:
                    cw_falling = wall_history[-1]['call_wall'] < wall_history[0]['call_wall'] * 0.98
                    if cw_falling:
                        venue_score = -12
                        venue_detail = f'Call walls converging ({cw_diff_pct:.1%}) + falling [{venue_dte_label}] ({thresh_label}, migration from 31-45d)'
                    else:
                        venue_score = -6
                        venue_detail = f'Call walls converging ({cw_diff_pct:.1%}), stable/rising [{venue_dte_label}] ({thresh_label}, migration from 31-45d)'
                else:
                    venue_score = -4
                    venue_detail = f'Call walls converging ({cw_diff_pct:.1%}) [{venue_dte_label}] ({thresh_label})'

            else:
                venue_detail = f'Venue walls not converging — PW diff {pw_diff_pct:.1%}, CW diff {cw_diff_pct:.1%} [{venue_dte_label}] ({thresh_label})'

        except (json.JSONDecodeError, TypeError) as e:
            venue_detail = f'Parse error: {e}'
    else:
        venue_detail = 'Deribit data not available in any DTE window'

    # ── Phase 2 signals ──────────────────────────────────────────────────
    funding_score, funding_detail, funding_history = _compute_funding_signal(c)
    oi_score, oi_detail, oi_hist = _compute_oi_signal(c)
    liquidation_score, liquidation_detail, liquidation_history = _compute_liquidation_signal(c)

    # ── Phase 3 signals ──────────────────────────────────────────────────
    pcr_score, pcr_detail, pcr_history = _compute_pcr_direction(c, ticker, btc_per_share)
    breach_score, breach_detail, _ = _compute_wall_breach(c, ticker, btc_per_share)

    # ── Phase 4 signals ──────────────────────────────────────────────────
    iv_term_score, iv_term_detail, iv_term_history = _compute_iv_term_signal(c, ticker, btc_per_share)

    # ── Final Score ──────────────────────────────────────────────────────
    total_score = (regime_score + wall_migration_score + compression_score
                   + etf_flow_score + venue_score
                   + funding_score + oi_score + liquidation_score
                   + pcr_score + breach_score + iv_term_score)
    total_score = max(-100, min(100, total_score))

    # ── Score History ────────────────────────────────────────────────────
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
            'liquidation': {'score': liquidation_score, 'max': 13, 'detail': liquidation_detail, 'source': 'coinglass'},
            'pcr_direction': {'score': pcr_score, 'max': 8, 'detail': pcr_detail},
            'wall_breach': {'score': breach_score, 'max': 13, 'detail': breach_detail},
            'iv_term_structure': {'score': iv_term_score, 'max': 8, 'detail': iv_term_detail},
        },
        'history': {
            'wall_migration': wall_history,
            'regime_history': regime_history,
            'etf_flows': flow_history,
            'funding_rates': funding_history,
            'oi_history': oi_hist,
            'liquidation_history': liquidation_history,
            'pcr_history': pcr_history,
            'iv_term_history': iv_term_history,
            'score_components_daily': score_history,
        },
        'coinglass_available': bool(os.environ.get('COINGLASS_API_KEY')),
        'structural_entry': structural_entry,
        'invalidation': invalidation,
        'timestamp': now_et().isoformat(),
        'data_freshness': _get_data_freshness(c, ticker),
    }


def _get_data_freshness(c, ticker):
    """Compute live IBIT/Deribit data freshness."""
    return {
        'ibit': _compute_ibit_freshness(),
        'deribit': _compute_deribit_freshness(TICKER_CONFIG.get(ticker, {}).get('deribit_currency', 'BTC')),
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


def _compute_ibit_freshness():
    """Compute hours since last US options market close (Mon-Fri 4:15 PM ET)."""
    t = now_et()

    candidate = t.replace(hour=16, minute=15, second=0, microsecond=0)

    if t.weekday() < 5:  # Mon-Fri
        if t >= candidate:
            last_close = candidate
        else:
            days_back = 1 if t.weekday() > 0 else 3
            last_close = candidate - timedelta(days=days_back)
    else:
        days_back = t.weekday() - 4
        last_close = candidate - timedelta(days=days_back)

    age_hours = (t - last_close).total_seconds() / 3600
    as_of_str = last_close.strftime('%Y-%m-%d %H:%M ET')
    in_market = (t.weekday() < 5 and
                 t.replace(hour=9, minute=30) <= t <= candidate)

    return {
        'age_hours': round(age_hours, 1),
        'as_of': as_of_str,
        'in_market_hours': in_market,
    }


def _compute_deribit_freshness(currency='BTC'):
    """Compute minutes since last Deribit data fetch for given currency."""
    with _deribit_lock:
        cache = _deribit_caches.get(currency, _deribit_caches['BTC'])
        cache_time = cache.get('time', 0)
    if cache_time == 0:
        return {'age_minutes': None, 'as_of': None}
    age_min = (time.time() - cache_time) / 60
    as_of_str = datetime.fromtimestamp(cache_time, tz=ET).strftime('%Y-%m-%d %H:%M ET')
    return {
        'age_minutes': round(age_min, 0),
        'as_of': as_of_str,
    }


# Farside page cache: at most 1 fetch per hour
_farside_cache = {'html': None, 'time': 0}
_farside_lock = threading.Lock()
FARSIDE_URL = 'https://farside.co.uk/bitcoin-etf-flow-all-data/'
FARSIDE_CACHE_SECONDS = 3600

# Deribit options cache: per-currency, at most 1 fetch per hour
_deribit_caches = {
    'BTC': {'data': None, 'time': 0},
    'ETH': {'data': None, 'time': 0},
}
_deribit_lock = threading.Lock()
DERIBIT_BASE_URL = 'https://www.deribit.com/api/v2/public/get_book_summary_by_currency?kind=option&currency='
DERIBIT_CACHE_SECONDS = 900


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
                log.info(f"[farside] Fetched {len(html)//1024}KB from Farside")
            except Exception as e:
                log.warning(f"[farside] Fetch failed: {e}")
                html = _farside_cache.get('html')
                if not html:
                    return None

    # Parse the table
    parser = _FarsideTableParser()
    parser.feed(html)

    if not parser.headers or len(parser.headers) < 13:
        log.warning(f"[farside] Bad table structure: {len(parser.headers)} headers")
        return None

    # Find column indices
    header_names = [h.strip() for h in parser.headers]
    try:
        ibit_col = header_names.index('IBIT')
        total_col = header_names.index('Total')
    except ValueError:
        log.warning(f"[farside] Missing IBIT or Total column in headers: {header_names}")
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
        log.info(f"[farside] Stored {inserted} days of IBIT flow data")

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


# ── COINGLASS: Aggregate funding rates & OI ─────────────────────────────────
_coinglass_lock = threading.Lock()
COINGLASS_BASE_URL = 'https://open-api-v4.coinglass.com/api'


def fetch_coinglass_data():
    """Fetch aggregate funding rate and OI from Coinglass API.
    Stores daily snapshots in coinglass_data table. Graceful degradation if no API key."""
    api_key = os.environ.get('COINGLASS_API_KEY')
    if not api_key:
        return  # Graceful - Phase 2 signals just return 0

    today = today_str_et()

    with _coinglass_lock:
        conn = get_db()
        c = conn.cursor()

        try:
            # Skip if already fetched today
            existing = c.execute(
                'SELECT 1 FROM coinglass_data WHERE date=? AND symbol=? AND metric=? LIMIT 1',
                (today, 'BTC', 'avg_funding_rate')
            ).fetchone()
            if existing:
                conn.close()
                return

            headers = {'accept': 'application/json', 'CG-API-KEY': api_key}

            # ── Endpoint 1: OI-Weighted Average Funding Rate ────────────
            try:
                url_fr = (
                    f'{COINGLASS_BASE_URL}/futures/funding-rate/'
                    f'oi-weight-history?symbol=BTC&interval=h8&limit=90'
                )
                req = urllib.request.Request(url_fr, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read().decode('utf-8'))

                if body.get('code') == '0' and body.get('data'):
                    # Group 8-hour data points by date and compute daily averages
                    daily_rates = {}
                    for pt in body['data']:
                        ts = pt.get('time', 0)
                        rate = pt.get('close')
                        if rate is None:
                            continue
                        # Timestamp is in milliseconds
                        day = datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d')
                        daily_rates.setdefault(day, []).append(float(rate))

                    for day, rates in daily_rates.items():
                        avg_rate = sum(rates) / len(rates)
                        c.execute(
                            'INSERT OR REPLACE INTO coinglass_data '
                            '(date, symbol, metric, value, extra_json) VALUES (?, ?, ?, ?, ?)',
                            (day, 'BTC', 'avg_funding_rate', avg_rate,
                             json.dumps({'samples': len(rates)}))
                        )
                    log.info(f"[coinglass] Stored funding rates for {len(daily_rates)} days")
                else:
                    log.warning(f"[coinglass] Funding rate response code: {body.get('code')}, msg: {body.get('msg')}")

            except Exception as e:
                log.warning(f"[coinglass] Funding rate fetch failed: {e}")

            # ── Endpoint 2: Aggregated Open Interest History ────────────
            try:
                url_oi = (
                    f'{COINGLASS_BASE_URL}/futures/open-interest/'
                    f'aggregated-history?symbol=BTC&interval=1d&limit=90'
                )
                req = urllib.request.Request(url_oi, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read().decode('utf-8'))

                if body.get('code') == '0' and body.get('data'):
                    count = 0
                    for pt in body['data']:
                        ts = pt.get('time', 0)
                        oi_val = pt.get('close')
                        if oi_val is None:
                            continue
                        day = datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d')
                        c.execute(
                            'INSERT OR REPLACE INTO coinglass_data '
                            '(date, symbol, metric, value, extra_json) VALUES (?, ?, ?, ?, ?)',
                            (day, 'BTC', 'total_oi_usd', float(oi_val), None)
                        )
                        count += 1
                    log.info(f"[coinglass] Stored OI data for {count} days")
                else:
                    log.warning(f"[coinglass] OI response code: {body.get('code')}, msg: {body.get('msg')}")

            except Exception as e:
                log.warning(f"[coinglass] OI fetch failed: {e}")

            # ── Endpoint 3: Aggregated Liquidation History ──────────────
            try:
                url_liq = (
                    f'{COINGLASS_BASE_URL}/futures/liquidation/'
                    f'aggregated-history?symbol=BTC&interval=1d&limit=90&exchange_list=Binance,OKX,Bybit,Bitget,BingX,dYdX,CoinEx,Kraken,Bitmex'
                )
                req = urllib.request.Request(url_liq, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read().decode('utf-8'))

                if body.get('code') == '0' and body.get('data'):
                    count = 0
                    for pt in body['data']:
                        ts = pt.get('time', 0)
                        long_liq = float(pt.get('aggregated_long_liquidation_usd', 0) or 0)
                        short_liq = float(pt.get('aggregated_short_liquidation_usd', 0) or 0)
                        liq_val = long_liq + short_liq
                        if liq_val <= 0:
                            continue
                        day = datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d')
                        c.execute(
                            'INSERT OR REPLACE INTO coinglass_data '
                            '(date, symbol, metric, value, extra_json) VALUES (?, ?, ?, ?, ?)',
                            (day, 'BTC', 'total_liquidation_usd', float(liq_val), None)
                        )
                        count += 1
                    log.info(f"[coinglass] Stored liquidation history for {count} days")
                else:
                    log.warning(f"[coinglass] Liquidation history response code: {body.get('code')}, msg: {body.get('msg')}")

            except Exception as e:
                log.warning(f"[coinglass] Liquidation history fetch failed: {e}")

            # ── Endpoint 4: Liquidation Coin List (current snapshot) ───
            try:
                url_snap = f'{COINGLASS_BASE_URL}/futures/liquidation/coin-list'
                req = urllib.request.Request(url_snap, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read().decode('utf-8'))

                if body.get('code') == '0' and body.get('data'):
                    for coin in body['data']:
                        if coin.get('symbol') != 'BTC':
                            continue
                        long_24h = coin.get('long_liquidation_usd_24h', 0) or 0
                        short_24h = coin.get('short_liquidation_usd_24h', 0) or 0
                        total_24h = long_24h + short_24h
                        extra = {
                            'long_24h': long_24h, 'short_24h': short_24h,
                            'long_12h': coin.get('long_liquidation_usd_12h', 0) or 0,
                            'short_12h': coin.get('short_liquidation_usd_12h', 0) or 0,
                            'long_4h': coin.get('long_liquidation_usd_4h', 0) or 0,
                            'short_4h': coin.get('short_liquidation_usd_4h', 0) or 0,
                            'long_1h': coin.get('long_liquidation_usd_1h', 0) or 0,
                            'short_1h': coin.get('short_liquidation_usd_1h', 0) or 0,
                        }
                        c.execute(
                            'INSERT OR REPLACE INTO coinglass_data '
                            '(date, symbol, metric, value, extra_json) VALUES (?, ?, ?, ?, ?)',
                            (today, 'BTC', 'liquidation_snapshot', float(total_24h), json.dumps(extra))
                        )
                        log.info(f"[coinglass] Stored BTC liquidation snapshot: ${total_24h/1e6:.0f}M (L:${long_24h/1e6:.0f}M / S:${short_24h/1e6:.0f}M)")
                        break
                else:
                    log.warning(f"[coinglass] Liquidation coin-list response code: {body.get('code')}, msg: {body.get('msg')}")

            except Exception as e:
                log.warning(f"[coinglass] Liquidation snapshot fetch failed: {e}")

            conn.commit()

        except Exception as e:
            log.warning(f"[coinglass] Unexpected error: {e}")
        finally:
            conn.close()


def fetch_deribit_options(min_dte=0, max_dte=7, currency='BTC'):
    """Fetch options from Deribit public API for given currency (BTC or ETH).
    Returns list of dicts with strike, expiry, dte, option_type, oi, iv,
    underlying_price. Cached for 1 hour per currency. Returns empty list on failure."""
    with _deribit_lock:
        now = time.time()
        cache = _deribit_caches.get(currency, _deribit_caches['BTC'])
        if cache['data'] is not None and (now - cache['time']) < DERIBIT_CACHE_SECONDS:
            raw = cache['data']
        else:
            try:
                url = f'{DERIBIT_BASE_URL}{currency}'
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'ibit-gex/1.0'
                })
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw = json.loads(resp.read().decode())
                cache['data'] = raw
                cache['time'] = now
                result_count = len(raw.get('result', []))
                log.info(f"[deribit] Fetched {result_count} {currency} instruments from Deribit")
            except Exception as e:
                log.warning(f"[deribit] {currency} fetch failed: {e}")
                raw = cache.get('data')
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

        dte = max((expiry_date - now_dt).total_seconds() / 86400.0, 0.0)
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

    log.info(f"[deribit] {len(options)} instruments in {min_dte}-{max_dte} DTE window")
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
    """Fetch klines from Binance .us, .com, or OKX fallback."""
    # Map Binance intervals to OKX bar notation
    okx_tf = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1H', '4h': '4H', '1d': '1D'}.get(tf, tf)
    # OKX uses BTC-USDT not BTCUSDT
    okx_inst = symbol[:3] + '-' + symbol[3:] if len(symbol) > 3 else symbol
    urls = [
        ('binance.us', f'https://api.binance.us/api/v3/klines?symbol={symbol}&interval={tf}&startTime={start_ms}&endTime={end_ms}&limit={limit}'),
        ('binance.com', f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={tf}&startTime={start_ms}&endTime={end_ms}&limit={limit}'),
        ('okx', f'https://www.okx.com/api/v5/market/history-candles?instId={okx_inst}&bar={okx_tf}&after={end_ms}&before={start_ms}&limit={min(limit, 300)}'),
    ]
    for source, url in urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'ibit-gex/1.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = json.loads(resp.read().decode())
                if source == 'okx':
                    # OKX returns {"code":"0","data":[[ts,o,h,l,c,vol,...],...]}
                    if isinstance(raw, dict) and raw.get('code') == '0' and raw.get('data'):
                        # Convert to Binance kline format: [ts,o,h,l,c,vol,close_ts,...]
                        return [[int(r[0]),r[1],r[2],r[3],r[4],r[5],int(r[0]),r[5],'0','0','0','0']
                                for r in reversed(raw['data'])]
                elif isinstance(raw, list):
                    return raw
        except Exception as e:
            log.warning(f"[candles] {source} fetch failed: {e}")
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
    log.info(f"[candles] Backfilled {total} candles for {symbol}/{tf}")


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
    cutoff_max = (now + timedelta(days=max_dte)).replace(hour=23, minute=59, second=59, microsecond=0)
    cutoff_min = (now + timedelta(days=min_dte)).replace(hour=0, minute=0, second=0, microsecond=0)
    selected_exps = [
        e for e in all_exps
        if cutoff_min <= datetime.strptime(e, "%Y-%m-%d").replace(tzinfo=timezone.utc) <= cutoff_max
    ]
    # Drop today's expiry after 4 PM ET — Yahoo still reports OI but options have settled
    now_eastern = now_et()
    if now_eastern.hour >= 16:
        today_str = now_eastern.strftime('%Y-%m-%d')
        selected_exps = [e for e in selected_exps if e != today_str]
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

    etf_flows = fetch_farside_flows() if (is_crypto and ticker_symbol == 'IBIT') else None

    # Collect options data
    strike_data = {}
    cached_chains = {}  # exp_str -> chain, reused for expected move
    expiry_strike_data = {}  # exp_str -> {strike -> {call_oi, put_oi, call_gex, ...}}
    expiry_iv_data = {}     # exp_str -> list of {strike, iv, oi, opt_type, dte}
    for exp_str in selected_exps:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        dte_float = max((exp_date - now).total_seconds() / 86400.0, 0.0)
        T = max(dte_float / 365.0, 0.5 / 365)
        dte_days = int(dte_float)  # integer for display/labels only
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

                # Collect IV for term structure
                if exp_str not in expiry_iv_data:
                    expiry_iv_data[exp_str] = []
                expiry_iv_data[exp_str].append({
                    'strike': strike, 'iv': iv, 'oi': oi, 'opt_type': opt_type, 'dte': dte_days
                })

                gamma = bs_gamma(spot, strike, T, rfr, iv)
                delta = bs_delta(spot, strike, T, rfr, iv, opt_type)
                vanna = bs_vanna(spot, strike, T, rfr, iv)
                charm = bs_charm(spot, strike, T, rfr, iv, opt_type)
                gex = sign * gamma * oi * 100 * spot ** 2 * 0.01
                vol = row.get('volume', 0)
                if pd.isna(vol):
                    vol = 0
                vol = int(vol)
                vol_gex = sign * gamma * vol * 100 * spot ** 2 * 0.01
                dealer_delta = -delta * oi * 100 * spot  # dollar notional
                # Dealer vanna exposure: how dealer delta changes with IV
                # Vanna is identical for calls/puts (put-call parity), and dealers
                # are short both, so no sign flip per option type.
                dealer_vanna = -vanna * oi * 100 * spot * 0.01  # dollar notional (vanna * n * S * 0.01)
                # Dealer charm exposure: how dealer hedge changes overnight
                # bs_charm returns holder's dΔ/dT; dealer is short, so negate.
                # Positive dealer_charm = dealers BUY delta (bullish overnight).
                dealer_charm = -charm * oi * 100 / 365.0 * spot  # dollar notional

                if strike not in strike_data:
                    strike_data[strike] = {'call_oi': 0, 'put_oi': 0, 'call_gex': 0, 'put_gex': 0,
                                           'call_delta': 0, 'put_delta': 0,
                                           'call_vanna': 0, 'put_vanna': 0,
                                           'call_charm': 0, 'put_charm': 0,
                                           'call_volume': 0, 'put_volume': 0,
                                           'call_vol_gex': 0, 'put_vol_gex': 0,
                                           'call_iv_sum': 0, 'call_iv_oi': 0,
                                           'put_iv_sum': 0, 'put_iv_oi': 0,
                                           'expiry_gex': {}}
                strike_data[strike][f'{opt_type}_oi'] += oi
                strike_data[strike][f'{opt_type}_gex'] += gex
                strike_data[strike][f'{opt_type}_vol_gex'] += vol_gex
                strike_data[strike][f'{opt_type}_volume'] += vol
                strike_data[strike][f'{opt_type}_delta'] += dealer_delta
                strike_data[strike][f'{opt_type}_vanna'] += dealer_vanna
                strike_data[strike][f'{opt_type}_charm'] += dealer_charm
                strike_data[strike][f'{opt_type}_iv_sum'] += iv * oi
                strike_data[strike][f'{opt_type}_iv_oi'] += oi
                # Per-expiry breakdown
                if exp_str not in strike_data[strike]['expiry_gex']:
                    strike_data[strike]['expiry_gex'][exp_str] = {
                        'call_gex': 0, 'put_gex': 0, 'dte': dte_days
                    }
                strike_data[strike]['expiry_gex'][exp_str][f'{opt_type}_gex'] += gex

                # Per-expiry accumulator
                if exp_str not in expiry_strike_data:
                    expiry_strike_data[exp_str] = {}
                esd = expiry_strike_data[exp_str]
                if strike not in esd:
                    esd[strike] = {
                        'call_oi': 0, 'put_oi': 0,
                        'call_gex': 0, 'put_gex': 0,
                        'call_delta': 0, 'put_delta': 0,
                        'call_vanna': 0, 'put_vanna': 0,
                        'call_charm': 0, 'put_charm': 0,
                        'call_volume': 0, 'put_volume': 0,
                        'call_iv_sum': 0, 'call_iv_oi': 0,
                        'put_iv_sum': 0, 'put_iv_oi': 0,
                        'dte': dte_days,
                    }
                esd[strike][f'{opt_type}_oi'] += oi
                esd[strike][f'{opt_type}_gex'] += gex
                esd[strike][f'{opt_type}_delta'] += dealer_delta
                esd[strike][f'{opt_type}_vanna'] += dealer_vanna
                esd[strike][f'{opt_type}_charm'] += dealer_charm
                esd[strike][f'{opt_type}_volume'] += vol
                esd[strike][f'{opt_type}_iv_sum'] += iv * oi
                esd[strike][f'{opt_type}_iv_oi'] += oi

    # Compute ATM IV per expiry for term structure
    iv_term_structure = []
    for exp_str, iv_entries in sorted(expiry_iv_data.items()):
        if not iv_entries:
            continue
        dte_val = iv_entries[0]['dte']

        result = compute_atm_iv(iv_entries, spot)
        if not result:
            continue

        iv_term_structure.append({
            'expiry': exp_str,
            'dte': dte_val,
            'atm_iv': result['atm_iv'],
            'call_iv': result['call_iv'],
            'put_iv': result['put_iv'],
            'num_atm_options': result['num_entries'],
            'source': 'ibit',
        })

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
            'call_vol_gex': d.get('call_vol_gex', 0), 'put_vol_gex': d.get('put_vol_gex', 0),
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
    deribit_expiry_strike_data = {}
    deribit_iv_data = {}
    deribit_oi_btc = 0
    deribit_options = []
    deribit_currency = TICKER_CONFIG.get(ticker_symbol, {}).get('deribit_currency')
    if is_crypto and deribit_currency:
        try:
            deribit_options = fetch_deribit_options(min_dte, max_dte, currency=deribit_currency)
            if deribit_options:
                deribit_available = True
                deribit_strike_data, deribit_iv_data, deribit_oi_btc, deribit_expiry_strike_data = process_deribit_options(
                    deribit_options, rfr, full_greeks=True
                )
        except Exception as e:
            log.warning(f"[deribit] Failed: {e}")

    # Compute Deribit ATM IV per expiry and merge into IBIT term structure
    iv_term_structure.extend(compute_deribit_iv_term(deribit_iv_data))

    # Build Deribit DataFrame
    deribit_df = build_deribit_df(deribit_strike_data, full_greeks=True)

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
    # Try each expiration in order; skip any with a zero straddle (dead options)
    expected_move = None
    for em_exp in selected_exps:
        try:
            ch = cached_chains[em_exp]
            atm_c = ch.calls.iloc[(ch.calls['strike'] - spot).abs().argsort()[:1]]
            atm_p = ch.puts.iloc[(ch.puts['strike'] - spot).abs().argsort()[:1]]
            straddle = (atm_c['bid'].values[0] + atm_c['ask'].values[0]) / 2 + \
                       (atm_p['bid'].values[0] + atm_p['ask'].values[0]) / 2
            if straddle <= 0:
                continue
            exp_date = datetime.strptime(em_exp, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            dte = max((exp_date - now).days, 1)
            expected_move = {
                'straddle': float(straddle), 'pct': float((straddle / spot) * 100),
                'upper': float(spot + straddle), 'lower': float(spot - straddle),
                'upper_btc': float((spot + straddle) / ref_per_share) if is_crypto else None,
                'lower_btc': float((spot - straddle) / ref_per_share) if is_crypto else None,
                'expiration': em_exp, 'dte': dte,
            }
            break
        except Exception as e:
            continue

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

    # Dealer flow forecast (vanna + charm) — use combined data when available
    if deribit_available and combined_levels_btc and not combined_df.empty:
        flow_forecast = compute_flow_forecast(combined_df, btc_spot, combined_levels_btc, is_crypto)
    else:
        flow_forecast = compute_flow_forecast(df, spot, levels, is_crypto)

    # Breakout signals — now with flow_forecast + venue data
    breakout = compute_breakout(df, spot, levels, expected_move, prev_strikes, ref_per_share,
                                etf_flows=etf_flows, flow_forecast=flow_forecast,
                                deribit_levels_btc=deribit_levels_btc,
                                btc_per_share=ref_per_share, btc_spot=btc_spot)

    # Significant levels with regime behavior
    sig_levels = compute_significant_levels(df, spot, levels, prev_strikes, is_crypto, ref_per_share, etf_flows)

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
        log.warning(f"[delta-scenario] Failed: {e}")

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

    # Store per-expiry chain snapshots (merged IBIT + Deribit)
    try:
        conn_exp = get_db()
        c_exp = conn_exp.cursor()
        today_str = today_str_et()

        all_exp_dates = sorted(set(
            list(expiry_strike_data.keys()) + list(deribit_expiry_strike_data.keys())
        ))

        for exp_str in all_exp_dates:
            exp_date_obj = datetime.strptime(exp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            dte_val = max(int((exp_date_obj - now).total_seconds() / 86400.0), 0)
            spot_btc_val = spot / ref_per_share if is_crypto else spot

            # Build IBIT lookup by BTC price (same pattern as gex_chart_data)
            ibit_by_btc = {}
            for strike_val, d in expiry_strike_data.get(exp_str, {}).items():
                btc_p = round(strike_val / ref_per_share) if is_crypto else round(strike_val)
                ibit_by_btc[btc_p] = (strike_val, d)

            # Deribit lookup already in BTC
            deribit_by_btc = {}
            for strike_btc, d in deribit_expiry_strike_data.get(exp_str, {}).items():
                deribit_by_btc[round(strike_btc)] = d

            all_btc = sorted(set(list(ibit_by_btc.keys()) + list(deribit_by_btc.keys())))

            strike_rows = []
            for btc_p in all_btc:
                ib_entry = ibit_by_btc.get(btc_p)
                db_entry = deribit_by_btc.get(btc_p)
                ib = ib_entry[1] if ib_entry else None
                ibit_strike = ib_entry[0] if ib_entry else 0

                i_call_gex = ib['call_gex'] if ib else 0
                i_put_gex = ib['put_gex'] if ib else 0
                d_call_gex = db_entry['call_gex'] if db_entry else 0
                d_put_gex = db_entry['put_gex'] if db_entry else 0

                # Blended OI-weighted IV from both venues
                c_iv_sum = (ib.get('call_iv_sum', 0) if ib else 0) + (db_entry.get('call_iv_sum', 0) if db_entry else 0)
                c_iv_oi = (ib.get('call_iv_oi', 0) if ib else 0) + (db_entry.get('call_iv_oi', 0) if db_entry else 0)
                p_iv_sum = (ib.get('put_iv_sum', 0) if ib else 0) + (db_entry.get('put_iv_sum', 0) if db_entry else 0)
                p_iv_oi = (ib.get('put_iv_oi', 0) if ib else 0) + (db_entry.get('put_iv_oi', 0) if db_entry else 0)

                strike_rows.append({
                    'strike': ibit_strike,
                    'btc': float(btc_p),
                    'call_oi': (ib['call_oi'] if ib else 0) + (db_entry['call_oi'] if db_entry else 0),
                    'put_oi': (ib['put_oi'] if ib else 0) + (db_entry['put_oi'] if db_entry else 0),
                    'ibit_call_oi': ib['call_oi'] if ib else 0,
                    'ibit_put_oi': ib['put_oi'] if ib else 0,
                    'deribit_call_oi': db_entry['call_oi'] if db_entry else 0,
                    'deribit_put_oi': db_entry['put_oi'] if db_entry else 0,
                    'call_gex': round(i_call_gex + d_call_gex, 2),
                    'put_gex': round(i_put_gex + d_put_gex, 2),
                    'net_gex': round(i_call_gex + i_put_gex + d_call_gex + d_put_gex, 2),
                    'ibit_gex': round(i_call_gex + i_put_gex, 2),
                    'deribit_gex': round(d_call_gex + d_put_gex, 2),
                    'net_dealer_delta': round(
                        (ib['call_delta'] + ib['put_delta'] if ib else 0) +
                        (db_entry['call_delta'] + db_entry['put_delta'] if db_entry else 0), 2),
                    'net_vanna': round(
                        (ib['call_vanna'] + ib['put_vanna'] if ib else 0) +
                        (db_entry['call_vanna'] + db_entry['put_vanna'] if db_entry else 0), 2),
                    'net_charm': round(
                        (ib['call_charm'] + ib['put_charm'] if ib else 0) +
                        (db_entry['call_charm'] + db_entry['put_charm'] if db_entry else 0), 2),
                    'call_volume': ib['call_volume'] if ib else 0,
                    'put_volume': ib['put_volume'] if ib else 0,
                    'call_iv': round(c_iv_sum / c_iv_oi, 4) if c_iv_oi > 0 else 0,
                    'put_iv': round(p_iv_sum / p_iv_oi, 4) if p_iv_oi > 0 else 0,
                })

            # Compute per-expiry levels
            level_rows = [{
                'strike': sr['btc'], 'net_gex': sr['net_gex'],
                'call_gex': sr['call_gex'], 'put_gex': sr['put_gex'],
                'call_oi': sr['call_oi'], 'put_oi': sr['put_oi'],
                'total_oi': sr['call_oi'] + sr['put_oi'],
                'net_dealer_delta': sr['net_dealer_delta'],
                'net_vanna': sr['net_vanna'], 'net_charm': sr['net_charm'],
                'call_volume': sr['call_volume'], 'put_volume': sr['put_volume'],
                'call_vol_gex': 0, 'put_vol_gex': 0,
                'active_gex': sr['net_gex'],
            } for sr in strike_rows]
            level_df = pd.DataFrame(level_rows)
            levels_exp = _compute_levels_from_df(level_df, spot_btc_val) if not level_df.empty else {}

            net_gex = sum(sr['net_gex'] for sr in strike_rows)
            total_call = sum(sr['call_oi'] for sr in strike_rows)
            total_put = sum(sr['put_oi'] for sr in strike_rows)
            has_deribit = any(sr['deribit_gex'] != 0 for sr in strike_rows)

            expiry_blob = {
                'expiry_date': exp_str, 'dte': dte_val,
                'spot': spot, 'spot_btc': round(spot_btc_val, 2),
                'btc_per_share': ref_per_share,
                'deribit_available': has_deribit,
                'strikes': strike_rows,
                'summary': {
                    'net_gex_total': round(net_gex, 2),
                    'total_call_oi': total_call, 'total_put_oi': total_put,
                    'pcr': round(total_put / total_call, 3) if total_call > 0 else 0,
                    'regime': 'negative_gamma' if net_gex < 0 else 'positive_gamma',
                    'call_wall_btc': levels_exp.get('call_wall'),
                    'put_wall_btc': levels_exp.get('put_wall'),
                    'gamma_flip_btc': levels_exp.get('gamma_flip'),
                }
            }

            c_exp.execute(
                'INSERT OR REPLACE INTO expiry_cache (date, ticker, expiry_date, data_json) VALUES (?, ?, ?, ?)',
                (today_str, ticker_symbol, exp_str, json.dumps(expiry_blob))
            )
        conn_exp.commit()
        conn_exp.close()
    except Exception as e:
        log.warning(f"[expiry-cache] Failed to store per-expiry data: {e}")

    # Positioning depth metrics (dealer cushion, put density, call overwrite zone)
    positioning_depth = {}
    try:
        levels_btc_for_depth = combined_levels_btc if combined_levels_btc else {
            k: (v / ref_per_share if isinstance(v, (int, float)) and k not in ('regime', 'pcr') else v)
            for k, v in levels.items()
        }
        positioning_depth = compute_positioning_depth(
            gex_chart_data, dealer_delta_profile,
            btc_spot or spot, levels_btc_for_depth
        )
    except Exception as e:
        log.warning(f"[positioning-depth] Failed: {e}")

    return {
        'ticker': ticker_symbol,
        'asset_label': cfg['asset_label'] if cfg else ticker_symbol,
        'spot': float(spot),
        'btc_spot': float(btc_spot) if btc_spot else None,
        'btc_per_share': float(ref_per_share),
        'is_btc': bool(is_crypto),
        'timestamp': now_et().strftime('%Y-%m-%d %H:%M'),
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
        'positioning_depth': positioning_depth,
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
        'iv_term_structure': iv_term_structure,
        'data_freshness': {
            'ibit': _compute_ibit_freshness(),
            'deribit': _compute_deribit_freshness(deribit_currency) if deribit_available else {'age_minutes': None, 'as_of': None},
        },
    }


# ── CACHE ──────────────────────────────────────────────────────────────────
# OI updates once per day (after market close, available next morning).
# Strategy: compare OI to detect when Yahoo has fresh data rather than
# relying on a fixed clock cutoff. Throttle Yahoo checks to every 30 min.
YAHOO_CHECK_INTERVAL = 1800  # seconds between re-checks
_last_yahoo_check = {}  # (ticker, dte) -> datetime
_yahoo_check_lock = threading.Lock()


def get_latest_cache(ticker, dte, target_date=None):
    """Return cached data and its date, or (None, None).
    If target_date given, return that specific date's data."""
    conn = get_db()
    c = conn.cursor()
    if target_date:
        c.execute('SELECT date, data_json FROM data_cache WHERE ticker=? AND dte=? AND date=? LIMIT 1',
                  (ticker, dte, target_date))
    else:
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
    today = today_str_et()
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
    today = today_str_et()
    c.execute('INSERT OR REPLACE INTO data_cache (date, ticker, dte, data_json) VALUES (?,?,?,?)',
              (today, ticker, dte, json.dumps(data, cls=NumpyEncoder)))
    conn.commit()
    conn.close()


def fetch_with_cache(ticker, dte, min_dte=0, force_refresh=False):
    """Return cached data if fresh, otherwise check Yahoo for new OI.
    force_refresh=True bypasses date check and OI comparison (used for Deribit refresh)."""
    today = today_str_et()
    cache_date, cached = get_latest_cache(ticker, dte)

    # Already confirmed today's data (and min_dte matches) — unless force_refresh
    if cached and cache_date == today and not force_refresh:
        cached_min = cached.get('dte_window', {}).get('min', 0)
        if cached_min == min_dte:
            return cached

    # Throttle: don't re-check Yahoo more than every 30 min
    check_key = (ticker, dte, min_dte)
    with _yahoo_check_lock:
        last_check = _last_yahoo_check.get(check_key)
        if cached and last_check and not force_refresh and \
           (now_et() - last_check).total_seconds() < YAHOO_CHECK_INTERVAL:
            return cached

    # Fetch from Yahoo (+ Deribit if applicable)
    data = fetch_and_analyze(ticker, dte, min_dte)
    with _yahoo_check_lock:
        _last_yahoo_check[check_key] = now_et()

    # Compare OI to detect if data actually changed
    # When force_refresh, always save (Deribit data changed even if IBIT didn't)
    if cached and not force_refresh:
        old_oi = (cached['levels']['total_call_oi'], cached['levels']['total_put_oi'])
        new_oi = (data['levels']['total_call_oi'], data['levels']['total_put_oi'])
        if old_oi == new_oi:
            return cached  # Still stale, serve previous cache

    # New data (or first run or force_refresh) — save as today
    set_cached_data(ticker, dte, data)
    return data


# ── BACKGROUND REFRESH ─────────────────────────────────────────────────────

REFRESH_DTES = DTE_WINDOWS  # refresh all non-overlapping windows


def _refresh_deribit_only(ticker):
    """Patch cached blobs with fresh Deribit data. Does NOT touch IBIT data,
    dealer delta, flow forecast, or any other computed field.
    Used when Yahoo/IBIT is unavailable (weekends)."""
    rfr = get_risk_free_rate()
    deribit_currency = TICKER_CONFIG.get(ticker, {}).get('deribit_currency', 'BTC')

    for label, min_d, max_d in REFRESH_DTES:
        try:
            cache_date, cached = get_latest_cache(ticker, max_d)
            if not cached:
                continue

            btc_spot = cached.get('btc_spot')
            if not btc_spot:
                continue

            # Fetch fresh Deribit for this DTE window
            deribit_options = fetch_deribit_options(min_d, max_d, currency=deribit_currency)
            if not deribit_options:
                continue

            deribit_strike_data, deribit_iv_data, deribit_oi_btc, _ = process_deribit_options(
                deribit_options, rfr, full_greeks=False
            )

            deribit_iv_term = compute_deribit_iv_term(deribit_iv_data)

            # Build lookup: BTC strike -> net_gex
            deribit_by_btc = {}
            deribit_net_gex_total = 0
            btc_lo, btc_hi = btc_spot * 0.82, btc_spot * 1.22
            for strike_btc, d in deribit_strike_data.items():
                net = d['call_gex'] + d['put_gex']
                deribit_net_gex_total += net
                if btc_lo <= strike_btc <= btc_hi:
                    deribit_by_btc[round(strike_btc)] = net

            # Patch gex_chart: update deribit_gex per bar, recalc net_gex
            gex_chart = cached.get('gex_chart', [])
            seen = set()
            for entry in gex_chart:
                btc_p = round(entry.get('btc', 0))
                seen.add(btc_p)
                new_d_gex = deribit_by_btc.get(btc_p, 0)
                entry['deribit_gex'] = new_d_gex
                entry['net_gex'] = entry.get('ibit_gex', 0) + new_d_gex

            # Add Deribit-only strikes not already in chart
            for btc_p, net_gex in deribit_by_btc.items():
                if btc_p not in seen:
                    d_data = deribit_strike_data.get(btc_p, deribit_strike_data.get(float(btc_p), {}))
                    gex_chart.append({
                        'strike': 0, 'btc': float(btc_p),
                        'net_gex': net_gex, 'ibit_gex': 0, 'deribit_gex': net_gex,
                        'call_gex': d_data.get('call_gex', 0), 'put_gex': d_data.get('put_gex', 0),
                        'active_gex': net_gex,
                        'net_vanna': 0, 'net_charm': 0,
                        'call_oi': d_data.get('call_oi', 0), 'put_oi': d_data.get('put_oi', 0),
                        'total_oi': d_data.get('call_oi', 0) + d_data.get('put_oi', 0),
                        'call_volume': 0, 'put_volume': 0, 'total_volume': 0,
                        'expiry_breakdown': {},
                    })
            gex_chart.sort(key=lambda x: x.get('btc', 0))

            # Build simple Deribit-only df for level computation
            deribit_df = build_deribit_df(deribit_strike_data, full_greeks=False)
            deribit_levels_btc = _compute_levels_from_df(deribit_df, btc_spot) if not deribit_df.empty else None

            # Rebuild combined levels from patched gex_chart
            combined_rows = []
            for entry in gex_chart:
                if entry.get('net_gex', 0) != 0:
                    combined_rows.append({
                        'strike': entry['btc'],
                        'call_gex': entry.get('call_gex', 0),
                        'put_gex': entry.get('put_gex', 0),
                        'net_gex': entry['net_gex'],
                        'call_oi': entry.get('call_oi', 0),
                        'put_oi': entry.get('put_oi', 0),
                        'total_oi': entry.get('total_oi', 0),
                        'net_dealer_delta': 0, 'net_vanna': 0, 'net_charm': 0,
                        'active_gex': entry.get('active_gex', entry['net_gex']),
                    })
            combined_df = pd.DataFrame(combined_rows)
            combined_levels_btc = _compute_levels_from_df(combined_df, btc_spot) if not combined_df.empty else None

            # Patch blob — ONLY Deribit fields
            cached['gex_chart'] = gex_chart
            cached['deribit_available'] = True
            cached['deribit_oi_btc'] = float(deribit_oi_btc)
            cached['deribit_net_gex'] = float(deribit_net_gex_total)
            if deribit_levels_btc:
                cached['deribit_levels_btc'] = {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                                 for k, v in deribit_levels_btc.items()}
            if combined_levels_btc:
                cached['combined_levels_btc'] = {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                                  for k, v in combined_levels_btc.items()}
            cached['data_freshness']['deribit'] = _compute_deribit_freshness()
            cached['timestamp'] = now_et().strftime('%Y-%m-%d %H:%M')

            # Merge Deribit IV term structure into cached blob
            existing_iv = cached.get('iv_term_structure', [])
            # Remove old deribit entries, add fresh ones
            existing_iv = [e for e in existing_iv if e.get('source') != 'deribit']
            existing_iv.extend(deribit_iv_term)
            cached['iv_term_structure'] = existing_iv

            set_cached_data(ticker, max_d, cached)
            log.info(f"[bg-refresh] {ticker} DTE {min_d}-{max_d} Deribit overlay done")

        except Exception as e:
            log.error(f"[bg-refresh] {ticker} DTE {min_d}-{max_d} Deribit overlay error: {e}")


def _bg_refresh():
    """Background thread: backfill candles on startup for all tickers,
    pre-fetch all DTEs, then periodically update candles (every 5 min) and re-check GEX data."""
    # Phase 0: Fetch Farside ETF flow data (full history on first run)
    try:
        fetch_farside_flows()
    except Exception as e:
        log.error(f"[bg-refresh] Farside flow fetch error: {e}")

    # Phase 0b: Fetch Coinglass data (funding rates, aggregate OI)
    try:
        fetch_coinglass_data()
    except Exception as e:
        log.error(f"[bg-refresh] Coinglass data fetch error: {e}")

    # Phase 1: Backfill candles for all tickers
    for tk, cfg in TICKER_CONFIG.items():
        symbol = cfg['binance_symbol']
        _candle_backfill_done.setdefault(symbol, threading.Event())
        log.info(f"[bg-refresh] Starting {symbol} candle backfill...")
        for tf in ('15m', '1h', '4h', '1d'):
            try:
                backfill_btc_candles(symbol, tf, days=90)
            except Exception as e:
                log.error(f"[bg-refresh] Backfill {symbol}/{tf} error: {e}")
        _candle_backfill_done[symbol].set()
        log.info(f"[bg-refresh] {symbol} candle backfill complete")

    # Phase 2: Main loop — GEX refresh + candle updates
    last_candle_update = 0
    # Track whether post-close refresh has been done today for each ticker
    post_close_done = {}  # {ticker: date_string}
    while True:
        # Cleanup old expiry_cache rows
        try:
            conn = get_db()
            conn.execute("DELETE FROM expiry_cache WHERE date < date('now', '-90 days')")
            conn.execute("DELETE FROM expiry_cache WHERE expiry_date < date('now', '-7 days')")
            conn.commit()
            conn.close()
        except Exception:
            pass

        # Update candles every 5 minutes for all tickers
        now = time.time()
        if now - last_candle_update >= 300:
            for tk, cfg in TICKER_CONFIG.items():
                symbol = cfg['binance_symbol']
                for tf in ('15m', '1h', '4h', '1d'):
                    try:
                        update_btc_candles(symbol, tf)
                    except Exception as e:
                        log.error(f"[bg-refresh] Candle update {symbol}/{tf} error: {e}")
            last_candle_update = now

        # Refresh Farside ETF flow data
        try:
            fetch_farside_flows()
        except Exception as e:
            log.error(f"[bg-refresh] Farside flow error: {e}")

        # GEX refresh logic for all tickers
        for tk, cfg in TICKER_CONFIG.items():
            today = today_str_et()
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
                            log.error(f"[bg-refresh] {tk} DTE {w[1]}-{w[2]} error: {e}")

                # Pre-compute weekly outlook with fresh expiry data
                try:
                    _compute_weekly_outlook(tk)
                except Exception as e:
                    log.warning(f"[bg-refresh] {tk} weekly outlook compute error: {e}")

            if all_fresh:
                # Re-run AI analysis only if ref asset moved >2% since last analysis
                # (Primary analysis is triggered post-close at 4:20 PM ET)
                cached_analysis = get_cached_analysis(tk)
                should_run = False
                if cached_analysis and cached_analysis.get('_btc_price'):
                    try:
                        current_price = yf.Ticker(cfg['ref_ticker']).info.get('regularMarketPrice')
                        if current_price:
                            old_price = cached_analysis['_btc_price']
                            pct_move = abs(current_price - old_price) / old_price * 100
                            if pct_move > 2:
                                log.info(f"[bg-refresh] {cfg['asset_label']} moved {pct_move:.1f}% since last {tk} analysis (${old_price:,.0f} -> ${current_price:,.0f})")
                                should_run = True
                    except Exception:
                        pass
                if should_run:
                    try:
                        log.info(f"[bg-refresh] Running {tk} AI analysis (>2% move)...")
                        run_analysis(tk)
                        log.info(f"[bg-refresh] {tk} AI analysis complete and cached")
                    except Exception as e:
                        log.error(f"[bg-refresh] {tk} AI analysis error: {e}")

            # Deribit intraday refresh: re-fetch hourly to keep Deribit data fresh
            # IBIT OI is EOD-only so it won't change, but Deribit OI/IV updates 24/7
            deribit_currency = TICKER_CONFIG.get(tk, {}).get('deribit_currency')
            if deribit_currency:
                try:
                    freshness = _compute_deribit_freshness(deribit_currency)
                    age_min = freshness.get('age_minutes')
                    deribit_stale = age_min is None or age_min >= 16

                    if deribit_stale:
                        with _deribit_lock:
                            _deribit_caches[deribit_currency]['time'] = 0

                        if all_fresh:
                            # Weekday: IBIT is current, safe to do full refresh
                            log.info(f"[bg-refresh] {tk} Deribit stale, full refresh...")
                            with ThreadPoolExecutor(max_workers=len(REFRESH_DTES)) as pool:
                                futures = {
                                    pool.submit(fetch_with_cache, tk, max_d, min_d, True): (label, min_d, max_d)
                                    for label, min_d, max_d in REFRESH_DTES
                                }
                                for fut in as_completed(futures):
                                    try:
                                        fut.result()
                                    except Exception as e:
                                        w = futures[fut]
                                        log.error(f"[bg-refresh] {tk} Deribit refresh DTE {w[1]}-{w[2]} error: {e}")
                            log.info(f"[bg-refresh] {tk} Deribit full refresh complete")
                            # Re-compute weekly outlook with fresh Deribit data
                            try:
                                _compute_weekly_outlook(tk)
                            except Exception as e:
                                log.warning(f"[bg-refresh] {tk} weekly outlook recompute error: {e}")
                        else:
                            # Weekend: IBIT stale, only update Deribit fields in cached blobs
                            log.info(f"[bg-refresh] {tk} Deribit stale, weekend overlay...")
                            _refresh_deribit_only(tk)
                            log.info(f"[bg-refresh] {tk} Deribit weekend overlay complete")

                except Exception as e:
                    log.error(f"[bg-refresh] {tk} Deribit freshness check error: {e}")

            # Post-close analysis: force-refresh IBIT windows after market close
            # Yahoo options data updates ~4:15 PM ET. We trigger at 4:20 PM ET.
            try:
                t = now_et()
                is_trading_day = t.weekday() < 5
                post_close_window = (t.hour == 16 and t.minute >= 20) or t.hour == 17
                already_done_today = post_close_done.get(tk) == today

                if is_trading_day and post_close_window and not already_done_today:
                    log.info(f"[bg-refresh] {tk} post-close refresh: re-fetching all windows with fresh IBIT data...")

                    # Force-refresh all DTE windows to pull new IBIT close data
                    with ThreadPoolExecutor(max_workers=len(REFRESH_DTES)) as pool:
                        futures = {
                            pool.submit(fetch_with_cache, tk, max_d, min_d, True): (label, min_d, max_d)
                            for label, min_d, max_d in REFRESH_DTES
                        }
                        for fut in as_completed(futures):
                            try:
                                fut.result()
                            except Exception as e:
                                w = futures[fut]
                                log.error(f"[bg-refresh] {tk} post-close DTE {w[1]}-{w[2]} error: {e}")

                    # Pre-compute weekly outlook with fresh post-close data
                    try:
                        _compute_weekly_outlook(tk)
                    except Exception as e:
                        log.warning(f"[bg-refresh] {tk} weekly outlook post-close error: {e}")

                    # Run analysis with fresh data
                    try:
                        log.info(f"[bg-refresh] {tk} post-close analysis running...")
                        run_analysis(tk)
                        log.info(f"[bg-refresh] {tk} post-close analysis complete")

                        # Save predictions with fresh post-close data
                        try:
                            conn = get_db()
                            c = conn.cursor()
                            already_saved = c.execute(
                                'SELECT 1 FROM predictions WHERE analysis_date=? AND ticker=? LIMIT 1',
                                (today, tk)).fetchone()
                            if not already_saved:
                                pred_results = {}
                                for label, min_d, max_d in DTE_WINDOWS:
                                    _, cached = get_latest_cache(tk, max_d)
                                    if cached:
                                        pred_results[label] = cached
                                cached_analysis = get_cached_analysis(tk)
                                if pred_results:
                                    save_predictions(tk, DTE_WINDOWS, pred_results, cached_analysis)
                                    log.info(f"[predictions] Saved {tk} predictions (post-close snapshot)")
                            conn.close()
                        except Exception as e:
                            log.error(f"[predictions] Post-close save failed for {tk}: {e}")

                    except Exception as e:
                        log.error(f"[bg-refresh] {tk} post-close analysis error: {e}")

                    post_close_done[tk] = today

                # Weekend analysis: run once per day using Deribit-refreshed data
                elif not is_trading_day and not already_done_today:
                    cached_analysis = get_cached_analysis(tk)
                    if not cached_analysis:
                        log.info(f"[bg-refresh] {tk} weekend analysis (Deribit-primary)...")
                        try:
                            run_analysis(tk)
                            log.info(f"[bg-refresh] {tk} weekend analysis complete")

                            # Save predictions with weekend data
                            try:
                                conn = get_db()
                                c = conn.cursor()
                                already_saved = c.execute(
                                    'SELECT 1 FROM predictions WHERE analysis_date=? AND ticker=? LIMIT 1',
                                    (today, tk)).fetchone()
                                if not already_saved:
                                    pred_results = {}
                                    for label, min_d, max_d in DTE_WINDOWS:
                                        _, cached = get_latest_cache(tk, max_d)
                                        if cached:
                                            pred_results[label] = cached
                                    cached_analysis = get_cached_analysis(tk)
                                    if pred_results:
                                        save_predictions(tk, DTE_WINDOWS, pred_results, cached_analysis)
                                        log.info(f"[predictions] Saved {tk} predictions (weekend snapshot)")
                                conn.close()
                            except Exception as e:
                                log.error(f"[predictions] Weekend save failed for {tk}: {e}")

                        except Exception as e:
                            log.error(f"[bg-refresh] {tk} weekend analysis error: {e}")
                        post_close_done[tk] = today

            except Exception as e:
                log.error(f"[bg-refresh] {tk} post-close check error: {e}")

        # Score any expired predictions
        try:
            conn = get_db()
            scored = score_expired_predictions(conn)
            if scored:
                log.info(f"[predictions] Scored {scored} expired predictions")
            conn.close()
        except Exception as e:
            log.error(f"[predictions] Scoring failed: {e}")

        # Sleep 5 min (candle update cadence), but also covers GEX re-check
        time.sleep(300)


def start_bg_refresh():
    t = threading.Thread(target=_bg_refresh, daemon=True)
    t.start()
    log.info("[bg-refresh] Background data refresh started")


# ── ROUTES ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/snapshot/dates')
def api_snapshot_dates():
    """Return available snapshot dates for a ticker (only dates with at least 4 DTE windows)."""
    ticker = request.args.get('ticker', 'IBIT').upper()
    if ticker not in TICKER_CONFIG:
        return Response(json.dumps({'error': f'Unknown ticker: {ticker}'}), mimetype='application/json'), 400
    conn = get_db()
    c = conn.cursor()
    rows = c.execute('''
        SELECT date, COUNT(DISTINCT dte) as window_count
        FROM data_cache WHERE ticker=?
        GROUP BY date HAVING window_count >= 4
        ORDER BY date DESC
    ''', (ticker,)).fetchall()
    conn.close()
    return Response(json.dumps({'dates': [r[0] for r in rows]}), mimetype='application/json')


def _compute_weekly_outlook(ticker):
    """Pre-compute weekly outlook and store in data_cache via set_cached_data."""
    from datetime import datetime as dt_cls
    conn = get_db()
    c = conn.cursor()

    # Get the most recent snapshot date per expiry (same pattern as DTE's get_latest_cache)
    latest_per_expiry = c.execute('''
        SELECT expiry_date, MAX(date) as latest_date
        FROM expiry_cache WHERE ticker=?
        GROUP BY expiry_date
        ORDER BY expiry_date
    ''', (ticker,)).fetchall()

    if not latest_per_expiry:
        conn.close()
        return

    expiry_dates = [r[0] for r in latest_per_expiry]
    expiry_snapshot = {r[0]: r[1] for r in latest_per_expiry}

    now_eastern = now_et()
    today_str = now_eastern.strftime('%Y-%m-%d')
    if now_eastern.hour >= 16:
        expiry_dates = [d for d in expiry_dates if d > today_str]
    else:
        expiry_dates = [d for d in expiry_dates if d >= today_str]

    weekly_buckets = {}
    for exp in expiry_dates:
        exp_dt = dt_cls.strptime(exp, '%Y-%m-%d')
        monday = exp_dt - timedelta(days=exp_dt.weekday())
        key = monday.strftime('%Y-%m-%d')
        if key not in weekly_buckets:
            weekly_buckets[key] = {'monday': monday, 'expiries': []}
        weekly_buckets[key]['expiries'].append(exp)

    windows = []
    spot_btc = None

    for week_key in sorted(weekly_buckets.keys()):
        bucket = weekly_buckets[week_key]
        mon = bucket['monday']
        week_expiries = bucket['expiries']

        # Query the most recent row for each expiry in this bucket
        rows = []
        for exp in week_expiries:
            snap = expiry_snapshot[exp]
            row = c.execute(
                'SELECT data_json FROM expiry_cache '
                'WHERE date=? AND ticker=? AND expiry_date=?',
                (snap, ticker, exp)
            ).fetchone()
            if row:
                rows.append(row)

        if not rows:
            continue

        agg = _aggregate_expiry_blobs(rows)
        if not agg:
            continue

        if spot_btc is None:
            spot_btc = agg['spot_btc']

        lvl = agg['levels_btc'] or {}
        cw = lvl.get('call_wall')
        pw = lvl.get('put_wall')
        gf = lvl.get('gamma_flip')

        today_dt = now_eastern.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
        min_dte = (mon - today_dt).days
        last_exp = dt_cls.strptime(week_expiries[-1], '%Y-%m-%d')
        max_dte = (last_exp - today_dt).days

        delta_flip_btc = None
        if agg['delta_flip_points']:
            delta_flip_btc = agg['delta_flip_points'][0].get('price_btc')

        ff = agg.get('flow_forecast') or {}
        charm = ff.get('charm', {})

        dd_dir = None
        if agg['dealer_delta_profile'] and spot_btc:
            closest = min(agg['dealer_delta_profile'],
                          key=lambda x: abs(x['price_btc'] - spot_btc))
            dd_val = closest.get('net_dealer_delta', 0)
            dd_dir = 'short' if dd_val < 0 else 'long' if dd_val > 0 else None

        last_exp_dt = dt_cls.strptime(week_expiries[-1], '%Y-%m-%d')
        label = f"{mon.strftime('%m/%d')}-{last_exp_dt.strftime('%d')}"

        windows.append({
            'label': label,
            'min_dte': max(round(min_dte), 0),
            'max_dte': max(round(max_dte), 0),
            'call_wall': cw,
            'put_wall': pw,
            'gamma_flip': gf,
            'regime': agg['regime'],
            'net_gex': agg['net_gex_total'],
            'total_oi': (agg.get('total_call_oi', 0) or 0) + (agg.get('total_put_oi', 0) or 0),
            'call_oi': agg.get('total_call_oi', 0) or 0,
            'put_oi': agg.get('total_put_oi', 0) or 0,
            'em_upper': None,
            'em_lower': None,
            'venue_agree': False,
            'ibit_cw': None,
            'ibit_pw': None,
            'deribit_cw': None,
            'deribit_pw': None,
            'dealer_delta_dir': dd_dir,
            'charm_dir': charm.get('direction'),
            'delta_flip_btc': round(delta_flip_btc) if delta_flip_btc else None,
        })

    conn.close()

    result = {
        'spot': spot_btc,
        'mode': 'weekly',
        'windows': windows,
    }
    set_cached_data(ticker, WEEKLY_OUTLOOK_DTE, result)
    log.info(f"[weekly-outlook] Computed {ticker}: {len(windows)} weekly windows")


@app.route('/api/outlook')
def api_outlook():
    """Return key levels from all DTE windows for the outlook funnel chart."""
    ticker = request.args.get('ticker', 'IBIT').upper()
    snapshot_date = request.args.get('date')
    mode = request.args.get('mode', 'dte')  # 'dte' (default) or 'weekly'
    if ticker not in TICKER_CONFIG:
        return Response(json.dumps({'error': f'Unknown ticker: {ticker}'}), mimetype='application/json'), 400

    windows = []
    spot_btc = None

    if mode == 'weekly':
        _, cached = get_latest_cache(ticker, WEEKLY_OUTLOOK_DTE, target_date=snapshot_date)
        if cached:
            return Response(json.dumps(cached, cls=NumpyEncoder),
                            mimetype='application/json')
        return Response(json.dumps({'spot': None, 'mode': 'weekly', 'windows': []}),
                        mimetype='application/json')

    for dte_key, min_d, max_d in DTE_WINDOWS:
        _, cached = get_latest_cache(ticker, dte_key, target_date=snapshot_date)
        if not cached:
            continue

        bps = cached.get('btc_per_share', 1)
        lvl = cached.get('levels', {})
        combined = cached.get('combined_levels_btc') or {}
        deribit_lvl = cached.get('deribit_levels_btc') or {}
        em = cached.get('expected_move') or {}
        flow = cached.get('flow_forecast', {})
        dd = cached.get('dealer_delta_briefing', {})

        if spot_btc is None:
            spot_btc = cached.get('btc_spot')

        if combined:
            cw = combined.get('call_wall')
            pw = combined.get('put_wall')
            gf = combined.get('gamma_flip')
            mp = combined.get('max_pain')
            regime = combined.get('regime', lvl.get('regime'))
            net_gex = combined.get('net_gex_total', 0)
        else:
            cw = lvl.get('call_wall', 0) / bps if bps else None
            pw = lvl.get('put_wall', 0) / bps if bps else None
            gf = lvl.get('gamma_flip', 0) / bps if bps else None
            mp = lvl.get('max_pain', 0) / bps if bps else None
            regime = lvl.get('regime')
            net_gex = lvl.get('net_gex_total', 0)

        ibit_cw = lvl.get('call_wall', 0) / bps if bps else None
        ibit_pw = lvl.get('put_wall', 0) / bps if bps else None
        deribit_cw = deribit_lvl.get('call_wall') if deribit_lvl else None
        deribit_pw = deribit_lvl.get('put_wall') if deribit_lvl else None

        venue_agree = False
        if deribit_cw and ibit_cw and ibit_cw > 0:
            venue_agree = abs(deribit_cw - ibit_cw) / ibit_cw < 0.03

        net_dd = dd.get('current_delta')
        dd_dir = 'short' if (isinstance(net_dd, (int, float)) and net_dd < 0) else 'long' if isinstance(net_dd, (int, float)) else None

        charm = flow.get('charm', {})

        flip_points = cached.get('delta_flip_points', [])
        delta_flip_btc = None
        if flip_points:
            delta_flip_btc = flip_points[0].get('price_btc')
            if delta_flip_btc is None:
                fp_ibit = flip_points[0].get('price_ibit', 0)
                delta_flip_btc = fp_ibit / bps if bps and fp_ibit else None

        windows.append({
            'label': f'{min_d}-{max_d}d',
            'min_dte': min_d,
            'max_dte': max_d,
            'call_wall': cw,
            'put_wall': pw,
            'gamma_flip': gf,
            'regime': regime,
            'net_gex': net_gex,
            'total_oi': (lvl.get('total_call_oi', 0) or 0) + (lvl.get('total_put_oi', 0) or 0),
            'call_oi': lvl.get('total_call_oi', 0) or 0,
            'put_oi': lvl.get('total_put_oi', 0) or 0,
            'em_upper': em.get('upper_btc'),
            'em_lower': em.get('lower_btc'),
            'venue_agree': venue_agree,
            'ibit_cw': ibit_cw,
            'ibit_pw': ibit_pw,
            'deribit_cw': deribit_cw,
            'deribit_pw': deribit_pw,
            'dealer_delta_dir': dd_dir,
            'charm_dir': charm.get('direction'),
            'delta_flip_btc': round(delta_flip_btc) if delta_flip_btc else None,
        })

    return Response(json.dumps({
        'spot': spot_btc,
        'mode': 'dte',
        'windows': windows,
    }, cls=NumpyEncoder), mimetype='application/json')


@app.route('/api/range-cone')
def api_range_cone():
    """Return per-window expected move, GEX zones, and flip levels for range cone chart."""
    ticker = request.args.get('ticker', 'IBIT').upper()
    snapshot_date = request.args.get('date')
    if ticker not in TICKER_CONFIG:
        return Response(json.dumps({'error': f'Unknown ticker: {ticker}'}), mimetype='application/json'), 400

    windows = []
    spot_btc = None

    for dte_key, min_d, max_d in DTE_WINDOWS:
        _, cached = get_latest_cache(ticker, dte_key, target_date=snapshot_date)
        if not cached:
            continue

        if spot_btc is None:
            spot_btc = cached.get('btc_spot')

        bps = cached.get('btc_per_share', 1)
        em = cached.get('expected_move') or {}
        combined = cached.get('combined_levels_btc') or {}

        # GEX distribution from gex_chart — fields: btc, net_gex
        gex_chart = cached.get('gex_chart', [])
        gex_zones = []
        for row in gex_chart:
            btc_price = row.get('btc', 0)
            net_gex = row.get('net_gex', 0)
            if btc_price and net_gex:
                gex_zones.append({'btc': round(btc_price), 'gex': round(net_gex)})
        gex_zones.sort(key=lambda x: abs(x['gex']), reverse=True)
        gex_zones = gex_zones[:25]
        gex_zones.sort(key=lambda x: x['btc'])

        # Delta flip
        flip_points = cached.get('delta_flip_points', [])
        delta_flip = None
        if flip_points:
            delta_flip = flip_points[0].get('price_btc')
            if delta_flip is None:
                fp_ibit = flip_points[0].get('price_ibit', 0)
                delta_flip = fp_ibit / bps if bps and fp_ibit else None

        # Gamma flip
        gamma_flip = None
        if combined:
            gamma_flip = combined.get('gamma_flip')
        else:
            gf_ibit = cached.get('levels', {}).get('gamma_flip', 0)
            gamma_flip = gf_ibit / bps if bps and gf_ibit else None

        windows.append({
            'label': f'{min_d}-{max_d}d',
            'min_dte': min_d,
            'max_dte': max_d,
            'em_upper': em.get('upper_btc'),
            'em_lower': em.get('lower_btc'),
            'gex_zones': gex_zones,
            'delta_flip': round(delta_flip) if delta_flip else None,
            'gamma_flip': round(gamma_flip) if gamma_flip else None,
            'regime': (combined.get('regime') or cached.get('levels', {}).get('regime')),
        })

    return Response(json.dumps({
        'spot': spot_btc,
        'windows': windows,
    }, cls=NumpyEncoder), mimetype='application/json')


@app.route('/api/structure')
def api_structure():
    conn = get_db()
    c = conn.cursor()
    ticker = request.args.get('ticker', 'IBIT').upper()
    days = int(request.args.get('days', 30))

    cutoff = (now_et() - timedelta(days=days)).strftime('%Y-%m-%d')

    rows = c.execute('''
        SELECT p.analysis_date, p.dte_window, p.spot_btc, p.call_wall_btc, p.put_wall_btc,
               p.gamma_flip_btc, p.regime, p.venue_walls_agree,
               p.deribit_call_wall_btc, p.deribit_put_wall_btc,
               p.ibit_call_wall_btc, p.ibit_put_wall_btc
        FROM predictions p
        INNER JOIN (
            SELECT analysis_date, dte_window, MIN(dte) as min_dte
            FROM predictions WHERE ticker=? AND analysis_date >= ?
            GROUP BY analysis_date, dte_window
        ) g ON p.analysis_date = g.analysis_date AND p.dte_window = g.dte_window AND p.dte = g.min_dte
        WHERE p.ticker=? AND p.analysis_date >= ?
        ORDER BY p.analysis_date, p.dte_window
    ''', (ticker, cutoff, ticker, cutoff)).fetchall()

    # Layer 0: snapshots (oldest data, single-window only)
    data = {}
    snap_rows = c.execute('''SELECT date, spot, btc_price, call_wall, put_wall, gamma_flip, regime
                             FROM snapshots WHERE ticker=? AND date >= ? ORDER BY date''',
                          (ticker, cutoff)).fetchall()
    for date, spot_ibit, btc_p, cw_ibit, pw_ibit, gf_ibit, regime in snap_rows:
        bps = btc_p / spot_ibit if spot_ibit else 1
        data[date] = {'spot': btc_p, 'windows': {
            '0-3': {
                'call_wall': cw_ibit / spot_ibit * btc_p if spot_ibit else None,
                'put_wall': pw_ibit / spot_ibit * btc_p if spot_ibit else None,
                'gamma_flip': gf_ibit / spot_ibit * btc_p if spot_ibit else None,
                'regime': regime, 'venue_agree': False,
                'deribit_cw': None, 'deribit_pw': None,
                'ibit_cw': cw_ibit / spot_ibit * btc_p if spot_ibit else None,
                'ibit_pw': pw_ibit / spot_ibit * btc_p if spot_ibit else None,
            }
        }}

    # Layer 1: data_cache (per-DTE-window, overwrites snapshots)
    dte_to_window = {3: '0-3', 7: '4-7', 14: '8-14', 30: '15-30', 45: '31-45'}
    cache_rows = c.execute('''SELECT date, dte, data_json FROM data_cache
                              WHERE ticker=? AND date >= ? ORDER BY date, dte''',
                           (ticker, cutoff)).fetchall()
    for date, dte, data_json_str in cache_rows:
        d = json.loads(data_json_str)
        window = dte_to_window.get(dte)
        if not window:
            continue
        bps = d.get('btc_per_share', 1)
        lvl = d.get('levels', {})
        combined = d.get('combined_levels_btc') or {}
        deribit = d.get('deribit_levels_btc') or {}

        if combined:
            cw = combined.get('call_wall')
            pw = combined.get('put_wall')
            gf = combined.get('gamma_flip')
            regime = combined.get('regime', lvl.get('regime'))
        else:
            cw = lvl.get('call_wall', 0) / bps if bps else None
            pw = lvl.get('put_wall', 0) / bps if bps else None
            gf = lvl.get('gamma_flip', 0) / bps if bps else None
            regime = lvl.get('regime')

        ibit_cw = lvl.get('call_wall', 0) / bps if bps else None
        ibit_pw = lvl.get('put_wall', 0) / bps if bps else None
        deribit_cw = deribit.get('call_wall')
        deribit_pw = deribit.get('put_wall')
        venue_agree = False
        if deribit_cw and ibit_cw and ibit_cw > 0:
            venue_agree = abs(deribit_cw - ibit_cw) / ibit_cw < 0.03

        if date not in data:
            data[date] = {'spot': d.get('btc_spot'), 'windows': {}}
        data[date]['windows'][window] = {
            'call_wall': cw, 'put_wall': pw,
            'gamma_flip': gf, 'regime': regime,
            'venue_agree': venue_agree,
            'deribit_cw': deribit_cw, 'deribit_pw': deribit_pw,
            'ibit_cw': ibit_cw, 'ibit_pw': ibit_pw,
        }

    # Override layer: predictions (higher fidelity, takes priority)
    for r in rows:
        date = r[0]
        if date not in data:
            data[date] = {'spot': r[2], 'windows': {}}
        data[date]['windows'][r[1]] = {
            'call_wall': r[3], 'put_wall': r[4],
            'gamma_flip': r[5], 'regime': r[6],
            'venue_agree': bool(r[7]),
            'deribit_cw': r[8], 'deribit_pw': r[9],
            'ibit_cw': r[10], 'ibit_pw': r[11],
        }

    conn.close()
    return Response(json.dumps(data), mimetype='application/json')


@app.route('/api/structure/heatmap')
def api_structure_heatmap():
    """Return bucketed OI/GEX heatmap data for the structure chart."""
    conn = get_db()
    c = conn.cursor()
    ticker = request.args.get('ticker', 'IBIT').upper()
    days = int(request.args.get('days', 30))
    bucket_size = int(request.args.get('bucket', 1000))

    cutoff = (now_et() - timedelta(days=days)).strftime('%Y-%m-%d')

    rows = c.execute('''
        SELECT sh.date, sh.strike, sh.total_oi, sh.net_gex, sh.call_oi, sh.put_oi,
               s.spot, s.btc_price
        FROM strike_history sh
        JOIN snapshots s ON sh.date = s.date AND sh.ticker = s.ticker
        WHERE sh.ticker=? AND sh.date >= ?
        ORDER BY sh.date, sh.strike
    ''', (ticker, cutoff)).fetchall()

    conn.close()

    if not rows:
        return Response(json.dumps({'dates': [], 'buckets': [], 'data': {}}),
                        mimetype='application/json')

    heatmap = {}
    all_buckets = set()

    for date, strike, total_oi, net_gex, call_oi, put_oi, spot_ibit, btc_price in rows:
        if not spot_ibit or not btc_price:
            continue
        bps = btc_price / spot_ibit
        strike_btc = strike * bps
        bucket = int(round(strike_btc / bucket_size) * bucket_size)
        all_buckets.add(bucket)

        if date not in heatmap:
            heatmap[date] = {}
        if bucket not in heatmap[date]:
            heatmap[date][bucket] = {'total_oi': 0, 'net_gex': 0, 'call_oi': 0, 'put_oi': 0}

        heatmap[date][bucket]['total_oi'] += total_oi or 0
        heatmap[date][bucket]['net_gex'] += net_gex or 0
        heatmap[date][bucket]['call_oi'] += call_oi or 0
        heatmap[date][bucket]['put_oi'] += put_oi or 0

    dates = sorted(heatmap.keys())
    buckets = sorted(all_buckets)

    data = {}
    for date in dates:
        cells = []
        for bucket in buckets:
            vals = heatmap[date].get(bucket)
            if vals and (vals['total_oi'] > 0 or vals['net_gex'] != 0):
                cells.append({
                    'price': bucket,
                    'total_oi': vals['total_oi'],
                    'net_gex': round(vals['net_gex'], 2),
                    'call_oi': vals['call_oi'],
                    'put_oi': vals['put_oi'],
                })
        data[date] = cells

    return Response(json.dumps({
        'dates': dates,
        'buckets': buckets,
        'bucket_size': bucket_size,
        'data': data,
    }), mimetype='application/json')


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


def _aggregate_expiry_blobs(rows, *, ticker=None, min_dte=None, max_dte=None,
                            etf_flows=None, prev_strikes=None, history=None,
                            iv_term_structure=None):
    """Aggregate multiple expiry_cache rows into full response matching /api/data shape.

    Args:
        rows: list of (data_json_str,) tuples from expiry_cache query
        ticker: ticker symbol (enables data_freshness)
        min_dte/max_dte: DTE range (informational)
        etf_flows: ETF flow data (enables sig_levels etf context)
        prev_strikes: previous day strikes (enables breakout, oi_changes)
        history: snapshot history rows
        iv_term_structure: IV term structure array

    Returns:
        Full response dict matching /api/data shape. Returns None if no valid data.
    """
    if not rows:
        return None

    strike_agg = {}
    included_expiries = []
    first_blob = json.loads(rows[0][0])
    repricing_rows = []

    for row in rows:
        d = json.loads(row[0])
        included_expiries.append(d['expiry_date'])
        dte_val = d.get('dte', 1)
        T_exp = max(dte_val / 365.0, 0.5 / 365)
        for s in d['strikes']:
            btc = round(s['btc'])
            if btc not in strike_agg:
                strike_agg[btc] = {
                    'btc': s['btc'], 'strike': s['strike'],
                    'call_oi': 0, 'put_oi': 0,
                    'call_gex': 0, 'put_gex': 0,
                    'net_gex': 0, 'ibit_gex': 0, 'deribit_gex': 0,
                    'net_dealer_delta': 0, 'net_vanna': 0, 'net_charm': 0,
                    'call_volume': 0, 'put_volume': 0, 'total_volume': 0,
                }
            a = strike_agg[btc]
            a['call_oi'] += s['call_oi']
            a['put_oi'] += s['put_oi']
            a['net_gex'] += s['net_gex']
            a['call_gex'] += s.get('call_gex', 0)
            a['put_gex'] += s.get('put_gex', 0)
            a['ibit_gex'] += s.get('ibit_gex', s['net_gex'])
            a['deribit_gex'] += s.get('deribit_gex', 0)
            a['net_dealer_delta'] += s['net_dealer_delta']
            a['net_vanna'] += s['net_vanna']
            a['net_charm'] += s['net_charm']
            a['call_volume'] += s['call_volume']
            a['put_volume'] += s['put_volume']
            a['total_volume'] += s['call_volume'] + s['put_volume']

            c_iv = s.get('call_iv', 0)
            p_iv = s.get('put_iv', 0)
            ibit_c_oi = s.get('ibit_call_oi', 0)
            ibit_p_oi = s.get('ibit_put_oi', 0)
            deri_c_oi = s.get('deribit_call_oi', 0)
            deri_p_oi = s.get('deribit_put_oi', 0)
            if ibit_c_oi == 0 and deri_c_oi == 0 and s.get('call_oi', 0) > 0:
                ibit_c_oi = s['call_oi']
            if ibit_p_oi == 0 and deri_p_oi == 0 and s.get('put_oi', 0) > 0:
                ibit_p_oi = s['put_oi']
            if c_iv > 0 or p_iv > 0:
                repricing_rows.append((s['btc'], T_exp, c_iv, p_iv, ibit_c_oi, ibit_p_oi, deri_c_oi, deri_p_oi))

    gex_chart = sorted(strike_agg.values(), key=lambda x: x['btc'])
    spot_btc = first_blob.get('spot_btc', 0)

    level_rows = [{
        'strike': s['btc'], 'net_gex': s['net_gex'],
        'call_gex': s.get('call_gex', 0), 'put_gex': s.get('put_gex', 0),
        'call_oi': s['call_oi'], 'put_oi': s['put_oi'],
        'total_oi': s['call_oi'] + s['put_oi'],
        'net_dealer_delta': s['net_dealer_delta'],
        'net_vanna': s['net_vanna'], 'net_charm': s['net_charm'],
        'call_volume': s.get('call_volume', 0), 'put_volume': s.get('put_volume', 0),
        'call_vol_gex': 0, 'put_vol_gex': 0,
        'active_gex': s['net_gex'],
    } for s in gex_chart]
    level_df = pd.DataFrame(level_rows)
    levels_btc = _compute_levels_from_df(level_df, spot_btc) if not level_df.empty else {}

    net_gex = sum(s['net_gex'] for s in gex_chart)
    total_call = sum(s['call_oi'] for s in gex_chart)
    total_put = sum(s['put_oi'] for s in gex_chart)

    flow_forecast = None
    if not level_df.empty and levels_btc:
        try:
            flow_forecast = compute_flow_forecast(level_df, spot_btc, levels_btc, True)
        except Exception:
            pass

    dealer_delta_profile = []
    delta_flip_points = []
    if repricing_rows and spot_btc > 0 and levels_btc:
        try:
            btc_per_share = first_blob.get('btc_per_share', 0.000568)
            rfr = 0.05
            cw = levels_btc.get('call_wall', spot_btc * 1.05)
            pw = levels_btc.get('put_wall', spot_btc * 0.95)
            grid_prices = set()
            grid_prices.update([cw, pw, spot_btc])
            gf = levels_btc.get('gamma_flip')
            if gf:
                grid_prices.add(gf)
            step = spot_btc * 0.005
            if step > 0:
                p = pw
                while p <= cw:
                    grid_prices.add(p)
                    p += step
                p = pw * 0.98
                while p <= cw * 1.02:
                    grid_prices.add(p)
                    p += step
            lo_bound = spot_btc * 0.85
            hi_bound = spot_btc * 1.15
            grid_prices = sorted([p for p in grid_prices if lo_bound <= p <= hi_bound])
            if len(grid_prices) > 80:
                indices = np.linspace(0, len(grid_prices) - 1, 80, dtype=int)
                grid_prices = [grid_prices[i] for i in indices]
                if spot_btc not in grid_prices:
                    grid_prices.append(spot_btc)
                    grid_prices.sort()

            for S_hyp in grid_prices:
                net_dd = 0.0
                call_dd = 0.0
                put_dd = 0.0
                for (strike_btc, T, c_iv, p_iv, ib_c_oi, ib_p_oi, dr_c_oi, dr_p_oi) in repricing_rows:
                    if strike_btc < lo_bound or strike_btc > hi_bound:
                        continue
                    if c_iv > 0 and (ib_c_oi > 0 or dr_c_oi > 0):
                        delta = bs_delta(S_hyp, strike_btc, T, rfr, c_iv, 'call')
                        if ib_c_oi > 0:
                            dd = -delta * ib_c_oi * 100 * btc_per_share * S_hyp
                            net_dd += dd; call_dd += dd
                        if dr_c_oi > 0:
                            dd = -delta * dr_c_oi * 1 * S_hyp
                            net_dd += dd; call_dd += dd
                    if p_iv > 0 and (ib_p_oi > 0 or dr_p_oi > 0):
                        delta = bs_delta(S_hyp, strike_btc, T, rfr, p_iv, 'put')
                        if ib_p_oi > 0:
                            dd = -delta * ib_p_oi * 100 * btc_per_share * S_hyp
                            net_dd += dd; put_dd += dd
                        if dr_p_oi > 0:
                            dd = -delta * dr_p_oi * 1 * S_hyp
                            net_dd += dd; put_dd += dd
                dealer_delta_profile.append({
                    'price_btc': float(S_hyp),
                    'net_dealer_delta': float(net_dd),
                })

            for i in range(len(dealer_delta_profile) - 1):
                d1 = dealer_delta_profile[i]['net_dealer_delta']
                d2 = dealer_delta_profile[i + 1]['net_dealer_delta']
                p1 = dealer_delta_profile[i]['price_btc']
                p2 = dealer_delta_profile[i + 1]['price_btc']
                if (d1 < 0 and d2 > 0) or (d1 > 0 and d2 < 0):
                    if (d2 - d1) != 0:
                        flip = p1 + (p2 - p1) * (-d1) / (d2 - d1)
                        delta_flip_points.append({'price_btc': round(float(flip))})
            if delta_flip_points:
                delta_flip_points.sort(key=lambda x: abs(x['price_btc'] - spot_btc))
                delta_flip_points = delta_flip_points[:2]
        except Exception:
            pass

    # ── Enrichment: build full response matching /api/data shape ──

    regime = 'negative_gamma' if net_gex < 0 else 'positive_gamma'
    btc_per_share = first_blob.get('btc_per_share', 0.000568)
    spot_share = first_blob.get('spot', spot_btc * btc_per_share if spot_btc else 0)
    is_crypto = True

    # combined_levels_btc — same as levels_btc with float conversion
    combined_levels_btc = {
        k: float(v) if isinstance(v, (int, float, np.floating)) else v
        for k, v in levels_btc.items()
    } if levels_btc else None

    # deribit_levels_btc — filter gex_chart for deribit-only strikes
    deribit_strikes = [s for s in gex_chart if s.get('deribit_gex', 0) != 0]
    deribit_levels_btc = None
    deribit_net_gex = 0.0
    deribit_oi_btc = 0.0
    if deribit_strikes:
        deri_rows = [{
            'strike': s['btc'], 'net_gex': s.get('deribit_gex', 0),
            'call_gex': 0, 'put_gex': 0,
            'call_oi': s.get('call_oi', 0), 'put_oi': s.get('put_oi', 0),
            'total_oi': s.get('call_oi', 0) + s.get('put_oi', 0),
            'net_dealer_delta': 0, 'net_vanna': 0, 'net_charm': 0,
            'call_volume': 0, 'put_volume': 0,
            'call_vol_gex': 0, 'put_vol_gex': 0, 'active_gex': s.get('deribit_gex', 0),
        } for s in deribit_strikes]
        deri_df = pd.DataFrame(deri_rows)
        if not deri_df.empty:
            deribit_levels_btc = _compute_levels_from_df(deri_df, spot_btc)
            if deribit_levels_btc:
                deribit_levels_btc = {
                    k: float(v) if isinstance(v, (int, float, np.floating)) else v
                    for k, v in deribit_levels_btc.items()
                }
        deribit_net_gex = sum(s.get('deribit_gex', 0) for s in gex_chart)
        deribit_oi_btc = sum(s.get('call_oi', 0) + s.get('put_oi', 0) for s in deribit_strikes)

    deribit_available = len(deribit_strikes) > 0

    # levels_share — levels_btc converted to share prices
    levels_share = {
        k: float(v * btc_per_share) if isinstance(v, (int, float, np.floating)) else v
        for k, v in levels_btc.items()
    } if levels_btc else {}

    # Expected move from repricing_rows (ATM straddle approximation)
    expected_move = None
    if repricing_rows and spot_btc > 0:
        # Find near-ATM IV for expected move estimate
        atm_rows = [(btc, T, c_iv, p_iv) for (btc, T, c_iv, p_iv, *_) in repricing_rows
                     if abs(btc - spot_btc) / spot_btc < 0.02 and (c_iv > 0 or p_iv > 0)]
        if atm_rows:
            # Use shortest-DTE ATM for expected move
            atm_rows.sort(key=lambda x: x[1])
            _, T_em, c_iv_em, p_iv_em = atm_rows[0]
            avg_iv = (c_iv_em + p_iv_em) / 2 if (c_iv_em > 0 and p_iv_em > 0) else max(c_iv_em, p_iv_em)
            # Normalize Deribit-style IV (percentage) to decimal
            if avg_iv > 5:
                avg_iv = avg_iv / 100.0
            straddle_btc = spot_btc * avg_iv * (T_em ** 0.5)
            straddle_share = straddle_btc * btc_per_share
            dte_em = max(int(T_em * 365), 1)
            expected_move = {
                'straddle': float(straddle_share),
                'pct': float((straddle_btc / spot_btc) * 100),
                'upper': float(spot_share + straddle_share),
                'lower': float(spot_share - straddle_share),
                'upper_btc': float(spot_btc + straddle_btc),
                'lower_btc': float(spot_btc - straddle_btc),
                'expiration': included_expiries[0] if included_expiries else None,
                'dte': dte_em,
            }

    # Significant levels
    sig_levels = []
    if not level_df.empty and levels_btc:
        try:
            sig_levels = compute_significant_levels(
                level_df, spot_btc, levels_btc,
                prev_strikes or {}, is_crypto, 1.0, etf_flows
            )
        except Exception as e:
            log.warning(f"[expiry-agg] significant_levels failed: {e}")

    # Positioning depth
    positioning_depth = {}
    if gex_chart and dealer_delta_profile and spot_btc > 0 and levels_btc:
        try:
            positioning_depth = compute_positioning_depth(
                gex_chart, dealer_delta_profile, spot_btc, levels_btc
            )
        except Exception as e:
            log.warning(f"[expiry-agg] positioning_depth failed: {e}")

    # Dealer delta briefing
    dealer_delta_briefing = None
    if dealer_delta_profile and levels_btc and spot_btc > 0:
        try:
            scenario_result = {
                'profile': dealer_delta_profile,
                'flip_points': delta_flip_points,
            }
            dealer_delta_briefing = generate_dealer_delta_briefing(
                scenario_result, spot_btc, levels_btc, 1.0, is_crypto
            )
        except Exception as e:
            log.warning(f"[expiry-agg] dealer_delta_briefing failed: {e}")

    # Breakout (gated on prev_strikes)
    breakout = None
    if prev_strikes and not level_df.empty and levels_btc:
        try:
            breakout = compute_breakout(
                level_df, spot_btc, levels_btc, expected_move,
                prev_strikes, 1.0,
                etf_flows=etf_flows, flow_forecast=flow_forecast,
                deribit_levels_btc=deribit_levels_btc,
                btc_per_share=1.0, btc_spot=spot_btc
            )
        except Exception as e:
            log.warning(f"[expiry-agg] breakout failed: {e}")

    # OI changes (gated on prev_strikes)
    oi_changes = None
    if prev_strikes:
        total_oi_prev = sum(s['total_oi'] for s in prev_strikes.values())
        total_oi_now = int(sum(s['call_oi'] + s['put_oi'] for s in gex_chart))
        call_oi_prev = sum(s['call_oi'] for s in prev_strikes.values())
        put_oi_prev = sum(s['put_oi'] for s in prev_strikes.values())
        oi_changes = {
            'total_delta': total_oi_now - total_oi_prev,
            'total_pct': ((total_oi_now - total_oi_prev) / max(total_oi_prev, 1)) * 100,
            'call_delta': int(total_call) - call_oi_prev,
            'put_delta': int(total_put) - put_oi_prev,
        }

    # Data freshness (gated on ticker)
    data_freshness = None
    if ticker:
        cfg = TICKER_CONFIG.get(ticker)
        deribit_currency = cfg['deribit_currency'] if cfg else 'BTC'
        data_freshness = {
            'ibit': _compute_ibit_freshness(),
            'deribit': _compute_deribit_freshness(deribit_currency) if deribit_available else {'age_minutes': None, 'as_of': None},
        }

    # Expiry metadata
    expiry_meta = []
    for exp_str in sorted(included_expiries):
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            dte_days = max((exp_date - datetime.now(timezone.utc)).days, 0)
            expiry_meta.append({'exp': exp_str, 'dte': dte_days})
        except Exception:
            pass

    pcr = round(total_put / total_call, 3) if total_call > 0 else 0

    return {
        'ticker': ticker or first_blob.get('ticker', 'IBIT'),
        'spot': float(spot_share),
        'btc_spot': float(spot_btc),
        'btc_per_share': float(btc_per_share),
        'is_btc': is_crypto,
        'timestamp': now_et().strftime('%Y-%m-%d %H:%M'),
        'dte_window': {'min': min_dte or 0, 'max': max_dte or 90},
        'levels': levels_share,
        'levels_btc': levels_btc,
        'expected_move': expected_move,
        'gex_chart': gex_chart,
        'significant_levels': sig_levels,
        'breakout': breakout,
        'etf_flows': etf_flows,
        'oi_changes': oi_changes,
        'flow_forecast': flow_forecast,
        'dealer_delta_profile': dealer_delta_profile,
        'dealer_delta_briefing': dealer_delta_briefing,
        'delta_flip_points': delta_flip_points,
        'positioning_depth': positioning_depth,
        'history': history,
        'expirations': sorted(included_expiries),
        'expiry_meta': expiry_meta,
        'deribit_available': deribit_available,
        'deribit_oi_btc': float(deribit_oi_btc),
        'deribit_net_gex': float(deribit_net_gex),
        'combined_levels_btc': combined_levels_btc,
        'deribit_levels_btc': deribit_levels_btc,
        'iv_term_structure': iv_term_structure,
        'data_freshness': data_freshness,
        'regime': regime,
        'net_gex_total': net_gex,
        'total_call_oi': total_call,
        'total_put_oi': total_put,
        'pcr': pcr,
        'included_expiries': sorted(included_expiries),
    }


@app.route('/api/expiry-data')
def api_expiry_data():
    """Query per-expiry options data.

    Params:
      ticker: IBIT or ETHA (default IBIT)
      expiry: single expiry date YYYY-MM-DD
      from_date: range start (inclusive)
      to_date: range end (inclusive)
      date: snapshot date (default: latest available)

    No expiry/from_date/to_date = list available expiries.
    Single expiry = full strike data for that expiry.
    from_date + to_date = aggregates all expiries in range.

    Response gex_chart shape matches existing charts.
    """
    ticker = request.args.get('ticker', 'IBIT').upper()
    expiry = request.args.get('expiry')
    from_date = request.args.get('from_date')
    to_date = request.args.get('to_date')
    snapshot_date = request.args.get('date')

    conn = get_db()
    c = conn.cursor()

    if not snapshot_date:
        row = c.execute(
            'SELECT MAX(date) FROM expiry_cache WHERE ticker=?', (ticker,)
        ).fetchone()
        snapshot_date = row[0] if row and row[0] else None
        if not snapshot_date:
            conn.close()
            return Response(json.dumps({'error': 'No expiry data available'}),
                            mimetype='application/json'), 404

    # --- List mode: return available expiries ---
    if not expiry and not from_date:
        now_eastern = now_et()
        today_str = now_eastern.strftime('%Y-%m-%d')
        skip_today = now_eastern.hour >= 16

        if request.args.get('date'):
            # Explicit date requested — use that snapshot's data as-is
            expiries = c.execute(
                'SELECT expiry_date, data_json FROM expiry_cache '
                'WHERE date=? AND ticker=? ORDER BY expiry_date',
                (snapshot_date, ticker)
            ).fetchall()
        else:
            # No explicit date — use latest data per expiry (handles partial-day writes)
            expiries = c.execute('''
                SELECT e.expiry_date, e.data_json
                FROM expiry_cache e
                INNER JOIN (
                    SELECT expiry_date, MAX(date) as latest_date
                    FROM expiry_cache WHERE ticker=? AND expiry_date>=?
                    GROUP BY expiry_date
                ) latest ON e.expiry_date = latest.expiry_date
                       AND e.date = latest.latest_date
                WHERE e.ticker=?
                ORDER BY e.expiry_date
            ''', (ticker, today_str, ticker)).fetchall()

        conn.close()
        result = []
        for exp_date, dj in expiries:
            if skip_today and exp_date == today_str:
                continue
            d = json.loads(dj)
            result.append({
                'expiry_date': exp_date,
                'dte': d.get('dte'),
                'total_call_oi': d['summary']['total_call_oi'],
                'total_put_oi': d['summary']['total_put_oi'],
                'total_oi': d['summary']['total_call_oi'] + d['summary']['total_put_oi'],
                'net_gex_total': d['summary']['net_gex_total'],
                'regime': d['summary']['regime'],
                'deribit_available': d.get('deribit_available', False),
            })
        return Response(json.dumps({'date': snapshot_date, 'expiries': result}),
                        mimetype='application/json')

    # --- Single expiry or range mode ---
    if request.args.get('date'):
        # Explicit date — use that snapshot
        if expiry:
            rows = c.execute(
                'SELECT data_json FROM expiry_cache '
                'WHERE date=? AND ticker=? AND expiry_date=?',
                (snapshot_date, ticker, expiry)
            ).fetchall()
        else:
            rows = c.execute(
                'SELECT data_json FROM expiry_cache '
                'WHERE date=? AND ticker=? AND expiry_date>=? AND expiry_date<=?',
                (snapshot_date, ticker, from_date, to_date)
            ).fetchall()
    else:
        # No explicit date — use latest per expiry
        if expiry:
            row = c.execute(
                'SELECT data_json FROM expiry_cache '
                'WHERE ticker=? AND expiry_date=? ORDER BY date DESC LIMIT 1',
                (ticker, expiry)
            ).fetchone()
            rows = [row] if row else []
        else:
            # Range: get latest row for each expiry in range
            target_expiries = c.execute('''
                SELECT expiry_date, MAX(date) as latest_date
                FROM expiry_cache
                WHERE ticker=? AND expiry_date>=? AND expiry_date<=?
                GROUP BY expiry_date
            ''', (ticker, from_date, to_date)).fetchall()
            rows = []
            for exp_date, latest_date in target_expiries:
                row = c.execute(
                    'SELECT data_json FROM expiry_cache '
                    'WHERE date=? AND ticker=? AND expiry_date=?',
                    (latest_date, ticker, exp_date)
                ).fetchone()
                if row:
                    rows.append(row)

    conn.close()

    # Filter out today's expired expiry after 4 PM ET
    now_eastern = now_et()
    if now_eastern.hour >= 16:
        today_str = now_eastern.strftime('%Y-%m-%d')
        rows = [row for row in rows if json.loads(row[0]).get('expiry_date') != today_str]

    if not rows:
        return Response(json.dumps({'error': 'All expiries in range have settled'}),
                        mimetype='application/json'), 404

    # Fetch enrichment data for full response
    conn2 = get_db()
    _, prev_strikes = get_prev_strikes(conn2, ticker)
    history_raw = get_history(conn2, ticker, 10)
    conn2.close()

    history_data = [{'date': h[0], 'spot': h[1], 'btc_price': h[2],
                     'gamma_flip': h[3], 'call_wall': h[4], 'put_wall': h[5],
                     'max_pain': h[6], 'regime': h[7], 'net_gex': h[8],
                     'total_call_oi': h[9], 'total_put_oi': h[10],
                     'weighted_net_gex': h[11] if len(h) > 11 else None}
                    for h in history_raw]

    etf_flows_data = fetch_farside_flows() if ticker == 'IBIT' else None

    agg = _aggregate_expiry_blobs(
        rows, ticker=ticker, min_dte=0, max_dte=90,
        etf_flows=etf_flows_data, prev_strikes=prev_strikes,
        history=history_data,
    )
    if not agg:
        return Response(json.dumps({'error': 'No data to aggregate'}),
                        mimetype='application/json'), 404

    agg['date'] = snapshot_date
    return Response(json.dumps(agg, cls=NumpyEncoder), mimetype='application/json')


@app.route('/api/data')
def api_data():
    try:
        ticker = request.args.get('ticker', 'IBIT').upper()
        if ticker not in TICKER_CONFIG:
            return Response(json.dumps({'error': f'Unknown ticker: {ticker}'}), mimetype='application/json'), 400
        dte = request.args.get('dte', 3, type=int)
        min_dte = request.args.get('min_dte', 0, type=int)
        snapshot_date = request.args.get('date')
        dte = max(1, min(dte, 90))
        min_dte = max(0, min(min_dte, dte - 1))

        if snapshot_date:
            # Historical mode: serve from cache directly
            _, cached = get_latest_cache(ticker, dte, target_date=snapshot_date)
            if cached:
                cached['snapshot_date'] = snapshot_date
                return Response(json.dumps(cached, cls=NumpyEncoder), mimetype='application/json')
            return Response(json.dumps({'error': f'No data for {snapshot_date}'}), mimetype='application/json'), 404

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
    today = today_str_et()
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
    today = today_str_et()
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
        analysis['_timestamp'] = now_et().strftime('%Y-%m-%d %H:%M')
    conn = get_db()
    c = conn.cursor()
    today = today_str_et()
    c.execute('INSERT OR REPLACE INTO analysis_cache (date, ticker, analysis_json) VALUES (?,?,?)',
              (today, ticker, json.dumps(analysis)))
    conn.commit()
    conn.close()


def save_predictions(ticker, dtes, results, analysis_text=None):
    """Save structured predictions for each expiry date in each DTE window."""
    today = today_str_et()
    conn = get_db()
    c = conn.cursor()

    for label, min_d, max_d in dtes:
        d = results.get(label)
        if not d:
            continue

        window_key = f"{min_d}-{max_d}"
        bps = d['btc_per_share']
        lvl = d.get('levels', {})
        combined_lvl = d.get('combined_levels_btc') or {}
        deribit_lvl = d.get('deribit_levels_btc') or {}
        flow = d.get('flow_forecast', {})
        em = d.get('expected_move') or {}
        deribit_avail = d.get('deribit_available', False)

        # Use combined levels when available, else IBIT-only converted to BTC
        if deribit_avail and combined_lvl:
            cw_btc = combined_lvl.get('call_wall')
            pw_btc = combined_lvl.get('put_wall')
            gf_btc = combined_lvl.get('gamma_flip')
            mp_btc = combined_lvl.get('max_pain')
            regime = combined_lvl.get('regime', lvl.get('regime'))
            net_gex = combined_lvl.get('net_gex_total', 0)
        else:
            cw_btc = lvl.get('call_wall', 0) / bps if bps else None
            pw_btc = lvl.get('put_wall', 0) / bps if bps else None
            gf_btc = lvl.get('gamma_flip', 0) / bps if bps else None
            mp_btc = lvl.get('max_pain', 0) / bps if bps else None
            regime = lvl.get('regime')
            net_gex = lvl.get('net_gex_total', 0)

        # Per-venue walls in BTC
        ibit_cw = lvl.get('call_wall', 0) / bps if bps else None
        ibit_pw = lvl.get('put_wall', 0) / bps if bps else None
        deribit_cw = deribit_lvl.get('call_wall') if deribit_lvl else None
        deribit_pw = deribit_lvl.get('put_wall') if deribit_lvl else None

        # Venue wall agreement: call walls within 3%
        walls_agree = 0
        if deribit_cw and ibit_cw and deribit_cw > 0 and ibit_cw > 0:
            cw_diff = abs(deribit_cw - ibit_cw) / ibit_cw
            pw_diff = abs(deribit_pw - ibit_pw) / ibit_pw if (ibit_pw and deribit_pw) else 1
            walls_agree = 1 if (cw_diff < 0.03 and pw_diff < 0.03) else 0

        # Dealer delta (numeric value from levels, not the narrative string)
        net_dd = lvl.get('net_dealer_delta')
        if deribit_avail and combined_lvl:
            net_dd = combined_lvl.get('net_dealer_delta', net_dd)
        dd_dir = 'short' if (net_dd is not None and net_dd < 0) else 'long' if net_dd is not None else None

        # Flow forecast fields
        charm = flow.get('charm', {})
        overnight = flow.get('overnight', {})
        vanna = flow.get('vanna', {})

        # AI bottom line (extract from analysis text if available)
        ai_bl = None
        if analysis_text and isinstance(analysis_text, dict):
            text_key = f"{min_d}-{max_d}d"
            text = analysis_text.get(text_key, '')
            if 'BOTTOM LINE:' in text:
                bl_start = text.index('BOTTOM LINE:')
                bl_end = text.find('\n', bl_start)
                ai_bl = text[bl_start:bl_end].strip() if bl_end > 0 else text[bl_start:bl_start + 300]

        # Save one row per expiry date in this window
        expirations = d.get('expirations', [])
        if not expirations:
            continue

        spot_btc = d.get('btc_spot')

        for exp_date_str in expirations:
            try:
                exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d')
                analysis_dt = datetime.strptime(today, '%Y-%m-%d')
                dte = (exp_date - analysis_dt).days
            except (ValueError, TypeError):
                continue

            if dte < 0:
                continue

            c.execute('''INSERT OR REPLACE INTO predictions (
                analysis_date, ticker, expiry_date, dte, dte_window,
                spot_btc, call_wall_btc, put_wall_btc, gamma_flip_btc, max_pain_btc,
                regime, net_gex, net_dealer_delta, dealer_delta_direction,
                charm_direction, charm_notional, charm_strength,
                vanna_strength, overnight_direction, overnight_notional,
                deribit_available, ibit_call_wall_btc, ibit_put_wall_btc,
                deribit_call_wall_btc, deribit_put_wall_btc, venue_walls_agree,
                em_upper_btc, em_lower_btc,
                ai_bottom_line
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
                today, ticker, exp_date_str, dte, window_key,
                spot_btc, cw_btc, pw_btc, gf_btc, mp_btc,
                regime, net_gex, net_dd, dd_dir,
                charm.get('direction'), charm.get('net_notional'), charm.get('strength'),
                vanna.get('strength'), overnight.get('direction'), overnight.get('net_notional'),
                1 if deribit_avail else 0, ibit_cw, ibit_pw,
                deribit_cw, deribit_pw, walls_agree,
                em.get('upper_btc'), em.get('lower_btc'),
                ai_bl,
            ))
    conn.commit()
    conn.close()


def score_expired_predictions(conn):
    """Score predictions whose expiry date has passed."""
    c = conn.cursor()
    today = today_str_et()

    c.execute('''SELECT DISTINCT expiry_date, ticker FROM predictions
                 WHERE scored=0 AND expiry_date < ?''', (today,))
    expired = c.fetchall()

    if not expired:
        return 0

    scored_count = 0
    for exp_date, pred_ticker in expired:
        # Derive candle symbol from ticker config
        cfg = TICKER_CONFIG.get(pred_ticker)
        candle_symbol = cfg['binance_symbol'] if cfg else 'BTCUSDT'
        exp_ts = int(datetime.strptime(exp_date, '%Y-%m-%d').timestamp())
        exp_candle = c.execute('''SELECT high, low, close FROM btc_candles
                                  WHERE symbol=? AND tf='1d'
                                  AND time >= ? AND time < ?
                                  ORDER BY time LIMIT 1''',
                               (candle_symbol, exp_ts, exp_ts + 86400)).fetchone()

        if not exp_candle:
            continue  # no candle data yet

        exp_high, exp_low, exp_close = exp_candle

        preds = c.execute('''SELECT id, analysis_date, spot_btc, call_wall_btc, put_wall_btc,
                                    gamma_flip_btc, regime, em_upper_btc, em_lower_btc,
                                    charm_direction, venue_walls_agree, deribit_available
                             FROM predictions
                             WHERE expiry_date=? AND ticker=? AND scored=0''', (exp_date, pred_ticker)).fetchall()

        for row in preds:
            (pred_id, analysis_date, spot_btc, cw, pw, gf, regime, em_up, em_lo,
             charm_dir, venue_agree, deribit_avail) = row

            if not spot_btc:
                continue

            analysis_ts = int(datetime.strptime(analysis_date, '%Y-%m-%d').timestamp())
            window_candles = c.execute('''SELECT high, low FROM btc_candles
                                         WHERE symbol=? AND tf='1d'
                                         AND time >= ? AND time <= ?''',
                                      (candle_symbol, analysis_ts, exp_ts + 86400)).fetchall()

            if not window_candles:
                continue

            win_high = max(r[0] for r in window_candles)
            win_low = min(r[1] for r in window_candles)

            # Binary scores
            call_wall_held = 1 if (cw and win_high <= cw) else 0 if cw else None
            put_wall_held = 1 if (pw and win_low >= pw) else 0 if pw else None
            range_held = 1 if (call_wall_held == 1 and put_wall_held == 1) else (
                0 if (call_wall_held is not None and put_wall_held is not None) else None)

            em_held = None
            if em_up and em_lo:
                em_held = 1 if (exp_high <= em_up and exp_low >= em_lo) else 0

            # Regime: positive gamma -> tight range, negative -> wide range
            realized_range_pct = (win_high - win_low) / spot_btc * 100
            dte_days = max((datetime.strptime(exp_date, '%Y-%m-%d') -
                           datetime.strptime(analysis_date, '%Y-%m-%d')).days, 1)
            daily_range = realized_range_pct / dte_days
            # NOTE: thresholds are initial guesses — tune with real data
            if regime == 'positive_gamma':
                regime_correct = 1 if daily_range < 2.0 else 0
            elif regime == 'negative_gamma':
                regime_correct = 1 if daily_range > 1.0 else 0
            else:
                regime_correct = None

            # Charm: did next session open in predicted direction?
            next_day = c.execute('''SELECT open FROM btc_candles
                                    WHERE symbol=? AND tf='1d'
                                    AND time > ? ORDER BY time LIMIT 1''',
                                 (candle_symbol, analysis_ts,)).fetchone()
            charm_correct = None
            if next_day and charm_dir and spot_btc:
                next_open = next_day[0]
                if charm_dir == 'buy':
                    charm_correct = 1 if next_open > spot_btc else 0
                else:
                    charm_correct = 1 if next_open < spot_btc else 0

            # Breach and error metrics
            max_breach_call = ((win_high - cw) / cw * 100) if (cw and win_high > cw) else 0
            max_breach_put = ((pw - win_low) / pw * 100) if (pw and win_low < pw) else 0
            call_wall_error = ((win_high - cw) / spot_btc * 100) if cw else None
            put_wall_error = ((pw - win_low) / spot_btc * 100) if pw else None

            venue_agree_held = range_held if venue_agree else None

            # T+2 scoring: look up the day AFTER expiry
            t2_ts = exp_ts + 86400
            t2_candle = c.execute('''SELECT high, low, close FROM btc_candles
                                     WHERE symbol=? AND tf='1d'
                                     AND time >= ? AND time < ?
                                     ORDER BY time LIMIT 1''',
                                  (candle_symbol, t2_ts, t2_ts + 86400)).fetchone()

            t2_high, t2_low, t2_close = (None, None, None)
            cw_held_t2, pw_held_t2, range_held_t2, regime_correct_t2 = (None, None, None, None)
            if t2_candle:
                t2_high, t2_low, t2_close = t2_candle
                # T+2 window: analysis date through T+2
                t2_window_candles = c.execute('''SELECT high, low FROM btc_candles
                                                 WHERE symbol=? AND tf='1d'
                                                 AND time >= ? AND time <= ?''',
                                              (candle_symbol, analysis_ts, t2_ts + 86400)).fetchall()
                if t2_window_candles:
                    t2_win_high = max(r[0] for r in t2_window_candles)
                    t2_win_low = min(r[1] for r in t2_window_candles)
                    cw_held_t2 = 1 if (cw and t2_win_high <= cw * 1.01) else 0 if cw else None
                    pw_held_t2 = 1 if (pw and t2_win_low >= pw * 0.99) else 0 if pw else None
                    range_held_t2 = 1 if (cw_held_t2 == 1 and pw_held_t2 == 1) else (
                        0 if (cw_held_t2 is not None and pw_held_t2 is not None) else None)
                    t2_range_pct = (t2_win_high - t2_win_low) / spot_btc * 100
                    t2_daily = t2_range_pct / max(dte_days + 1, 1)
                    if regime == 'positive_gamma':
                        regime_correct_t2 = 1 if t2_daily < 2.0 else 0
                    elif regime == 'negative_gamma':
                        regime_correct_t2 = 1 if t2_daily > 1.0 else 0

            c.execute('''UPDATE predictions SET
                scored=1, scored_date=?,
                btc_high_on_expiry=?, btc_low_on_expiry=?, btc_close_on_expiry=?,
                btc_high_in_window=?, btc_low_in_window=?,
                call_wall_held=?, put_wall_held=?, range_held=?, em_held=?,
                regime_correct=?, charm_correct=?, venue_agree_held=?,
                max_breach_call_pct=?, max_breach_put_pct=?, realized_range_pct=?,
                call_wall_error_pct=?, put_wall_error_pct=?,
                btc_high_t2=?, btc_low_t2=?, btc_close_t2=?,
                call_wall_held_t2=?, put_wall_held_t2=?, range_held_t2=?, regime_correct_t2=?
                WHERE id=?''', (
                today, exp_high, exp_low, exp_close,
                win_high, win_low,
                call_wall_held, put_wall_held, range_held, em_held,
                regime_correct, charm_correct, venue_agree_held,
                max_breach_call, max_breach_put, realized_range_pct,
                call_wall_error, put_wall_error,
                t2_high, t2_low, t2_close,
                cw_held_t2, pw_held_t2, range_held_t2, regime_correct_t2,
                pred_id,
            ))
            scored_count += 1

    conn.commit()
    return scored_count


def detect_structural_patterns(levels, spot_btc, venue_breakdown=None, changes_vs_prev=None):
    """Rule-based pattern pre-screen with concrete thresholds.
    Returns list of detected pattern dicts."""
    patterns = []
    net_gex = levels.get('net_gex_total', 0)
    gamma_flip = levels.get('gamma_flip', 0)
    call_wall = levels.get('call_wall', 0)
    put_wall = levels.get('put_wall', 0)
    regime = levels.get('regime', '')
    pcr = levels.get('pcr', 0)
    activity = levels.get('gex_activity_ratio', 0)

    # Pattern 1: Gamma squeeze setup
    if gamma_flip and spot_btc:
        flip_dist_pct = abs(spot_btc - gamma_flip) / spot_btc * 100
        if regime == 'negative_gamma' and flip_dist_pct < 2.0 and pcr < 0.7:
            patterns.append({
                'pattern': 'gamma_squeeze_setup',
                'signal': 'Spot within 2% of gamma flip in negative gamma with call-heavy flow',
                'flip_distance_pct': round(flip_dist_pct, 2),
            })

    # Pattern 2: Wall pinning (near expiry)
    if call_wall and put_wall and spot_btc:
        cw_dist = abs(spot_btc - call_wall) / spot_btc * 100
        pw_dist = abs(spot_btc - put_wall) / spot_btc * 100
        nearest_wall = min(cw_dist, pw_dist)
        if nearest_wall < 0.5:
            pinned_to = 'call_wall' if cw_dist < pw_dist else 'put_wall'
            patterns.append({
                'pattern': 'wall_pinning',
                'signal': f'Spot within 0.5% of {pinned_to} — gravitational pull likely',
                'distance_pct': round(nearest_wall, 2),
                'pinned_to': pinned_to,
            })

    # Pattern 3: Venue convergence (high conviction)
    if venue_breakdown:
        ibit = venue_breakdown.get('ibit', {})
        deribit = venue_breakdown.get('deribit', {})
        ibit_cw = ibit.get('call_wall_btc')
        deribit_cw = deribit.get('call_wall_btc')
        ibit_pw = ibit.get('put_wall_btc')
        deribit_pw = deribit.get('put_wall_btc')

        if ibit_cw and deribit_cw and spot_btc:
            cw_diff = abs(ibit_cw - deribit_cw) / spot_btc * 100
            if cw_diff < 1.0:
                patterns.append({
                    'pattern': 'venue_convergence_resistance',
                    'signal': f'IBIT + Deribit call walls within {cw_diff:.1f}% — high conviction resistance',
                    'ibit_cw': ibit_cw, 'deribit_cw': deribit_cw,
                })
        if ibit_pw and deribit_pw and spot_btc:
            pw_diff = abs(ibit_pw - deribit_pw) / spot_btc * 100
            if pw_diff < 1.0:
                patterns.append({
                    'pattern': 'venue_convergence_support',
                    'signal': f'IBIT + Deribit put walls within {pw_diff:.1f}% — high conviction support',
                    'ibit_pw': ibit_pw, 'deribit_pw': deribit_pw,
                })

    # Pattern 4: Regime transition
    if changes_vs_prev and changes_vs_prev.get('regime_changed'):
        patterns.append({
            'pattern': 'regime_transition',
            'signal': f"Regime flipped from {changes_vs_prev.get('regime_prev')} to {regime} — mixed signals, reduce sizing",
            'prev_regime': changes_vs_prev.get('regime_prev'),
            'new_regime': regime,
        })

    # Pattern 5: High activity ratio (actively defended levels)
    if activity > 1.5:
        patterns.append({
            'pattern': 'high_hedging_activity',
            'signal': f'Activity ratio {activity:.1f}x — dealers actively hedging, levels are being tested today',
            'activity_ratio': activity,
        })

    return patterns


def build_analysis_data(ticker='IBIT'):
    """Build the analysis data summaries dict for a ticker.
    Returns the same data blob that gets sent to the AI as prompt context."""
    cfg = TICKER_CONFIG.get(ticker)
    if not cfg:
        raise ValueError(f'Unknown ticker: {ticker}')

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
    structure_trends = summarize_structure_trends(conn, ticker, days=7)
    conn.close()

    # Build concise summaries (strip chart data to keep prompt small)
    summaries = {}
    summaries['_history_trends'] = history_trends
    if structure_trends:
        summaries['_structure_trends'] = structure_trends

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
            'volume_gex_total': ibit_lvl.get('volume_gex_total', 0),
            'gex_activity_ratio': ibit_lvl.get('gex_activity_ratio', 0),
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
                    'net_notional': d['flow_forecast']['vanna']['net_notional'],
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
            'key_level_dealer_delta': {
                name: round(delta) if delta is not None else None
                for name, delta in (d.get('dealer_delta_briefing', {}).get('key_level_deltas', {}) or {}).items()
            } if d.get('dealer_delta_briefing') else None,
            'positioning_depth': d.get('positioning_depth', {}),
            'data_freshness': d.get('data_freshness'),
        }

        # IV term structure for this DTE window
        iv_ts = d.get('iv_term_structure', [])
        if iv_ts:
            summaries[key]['iv_term_structure'] = [
                {
                    'expiry': e['expiry'],
                    'dte': e['dte'],
                    'atm_iv': e['atm_iv'],
                    'call_iv': e.get('call_iv'),
                    'put_iv': e.get('put_iv'),
                    'source': e.get('source', 'ibit'),
                }
                for e in sorted(iv_ts, key=lambda x: x.get('dte', 0))
            ]

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

        # GEX distribution: condensed per-strike profile for AI analysis
        # Gives the AI the shape of the curve, not just the peaks
        gex_chart = d.get('gex_chart', [])
        if gex_chart:
            # Sort by absolute GEX to get the most impactful strikes
            sorted_by_gex = sorted(gex_chart, key=lambda x: abs(x.get('net_gex', 0)), reverse=True)
            # Take top 20 strikes, then re-sort by BTC price for readable output
            top_strikes = sorted(sorted_by_gex[:20], key=lambda x: x.get('btc', 0))
            summaries[key]['gex_distribution'] = [
                {
                    'btc': round(s['btc']),
                    'net_gex': round(s['net_gex']),
                    'ibit_gex': round(s.get('ibit_gex', 0)),
                    'deribit_gex': round(s.get('deribit_gex', 0)),
                    'call_oi': s.get('call_oi', 0),
                    'put_oi': s.get('put_oi', 0),
                    'total_volume': s.get('total_volume', 0),
                }
                for s in top_strikes
            ]
        else:
            summaries[key]['gex_distribution'] = []

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

        # Pattern pre-screen (use BTC-space levels, not IBIT share-price levels)
        pattern_levels = {
            'call_wall': primary_lvl_btc.get('call_wall', 0),
            'put_wall': primary_lvl_btc.get('put_wall', 0),
            'gamma_flip': primary_lvl_btc.get('gamma_flip', 0),
            'regime': primary_regime,
            'net_gex_total': primary_net_gex,
            'pcr': ibit_lvl.get('pcr', 0),
            'gex_activity_ratio': ibit_lvl.get('gex_activity_ratio', 0),
        }
        detected = detect_structural_patterns(
            pattern_levels,
            current_btc,
            venue_breakdown=summaries[key].get('venue_breakdown'),
            changes_vs_prev=summaries[key].get('changes_vs_prev'),
        )
        if detected:
            summaries[key]['detected_patterns'] = detected

    # Prediction accuracy feedback (if enough data exists)
    try:
        conn = get_db()
        c = conn.cursor()
        accuracy = {}
        dte_acc_buckets = [('0-1', 0, 1), ('2-3', 2, 3), ('4-7', 4, 7),
                           ('8-14', 8, 14), ('15-30', 15, 30), ('31-45', 31, 45)]
        for bucket_name, min_dte_b, max_dte_b in dte_acc_buckets:
            row = c.execute('''SELECT COUNT(*), SUM(call_wall_held), SUM(put_wall_held),
                                      SUM(range_held), SUM(regime_correct),
                                      SUM(CASE WHEN venue_walls_agree=1 THEN range_held END),
                                      SUM(venue_walls_agree)
                               FROM predictions WHERE ticker=? AND dte >= ? AND dte <= ? AND scored=1''',
                            (ticker, min_dte_b, max_dte_b)).fetchone()
            total = row[0] or 0
            if total >= 5:
                accuracy[bucket_name] = {
                    'n': total,
                    'call_wall_held_pct': round((row[1] or 0) / total * 100),
                    'put_wall_held_pct': round((row[2] or 0) / total * 100),
                    'range_held_pct': round((row[3] or 0) / total * 100),
                    'regime_correct_pct': round((row[4] or 0) / total * 100),
                    'venue_agree_held_pct': round((row[5] or 0) / max(row[6] or 1, 1) * 100) if row[6] else None,
                }
                # T+2 accuracy
                row_t2 = c.execute('''SELECT COUNT(*), SUM(call_wall_held_t2), SUM(put_wall_held_t2),
                                             SUM(range_held_t2)
                                      FROM predictions WHERE ticker=? AND dte >= ? AND dte <= ?
                                      AND call_wall_held_t2 IS NOT NULL''',
                                   (ticker, min_dte_b, max_dte_b)).fetchone()
                total_t2 = row_t2[0] or 0
                if total_t2 >= 5:
                    accuracy[bucket_name]['t2_n'] = total_t2
                    accuracy[bucket_name]['t2_range_held_pct'] = round((row_t2[3] or 0) / total_t2 * 100)
        if accuracy:
            summaries['_prediction_accuracy'] = accuracy
        conn.close()
    except Exception as e:
        log.error(f"[predictions] Accuracy query failed: {e}")

    summaries['_live_ref_price'] = live_ref_price
    return summaries


def run_analysis(ticker='IBIT', save=True):
    """Run AI analysis across all DTEs. Returns analysis dict or raises."""
    cfg = TICKER_CONFIG.get(ticker)
    if not cfg:
        raise ValueError(f'Unknown ticker: {ticker}')
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise RuntimeError('ANTHROPIC_API_KEY not set')

    summaries = build_analysis_data(ticker)
    live_ref_price = summaries.pop('_live_ref_price', None)
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

Volume GEX (volume_gex_total) weights gamma exposure by TODAY's trading volume instead of total open interest. It measures what dealers are ACTIVELY hedging right now, vs the structural positioning from OI-weighted GEX.

The activity_ratio (gex_activity_ratio) = |Volume GEX / OI GEX|. Interpretation:
- Ratio > 1.5: Dealers actively hedging well above their structural positioning — high conviction day, levels are being tested
- Ratio 0.5-1.5: Normal hedging activity proportional to positioning
- Ratio < 0.5: Stale positioning, low hedging activity — levels are less "defended" today, may not hold
- Ratio near 0: Weekend/after-hours, no volume data

Use activity_ratio to calibrate confidence in levels. A call wall with high activity_ratio is being actively defended. A call wall with low activity_ratio is a ghost level from old OI.

Positioning confidence (positioning_confidence, 0-100%) indicates how much to trust the standard GEX sign convention (dealers short both calls and puts — the market-maker assumption). When confidence is high, positive GEX = stabilizing (dealers hedge by buying dips / selling rallies) and call/put walls act as resistance/support. When below 60%, speculative flow likely dominates — dealers may be NET LONG calls (not short), which inverts the mechanics: the call wall becomes a squeeze trigger rather than resistance, and regime labels may be unreliable. Always note the confidence level and any warnings in your analysis.

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

GEX DISTRIBUTION:
gex_distribution shows the top 20 strikes by GEX magnitude for each window, sorted by BTC price. Use this to assess whether support/resistance is concentrated at a single strike (sharp wall) or spread across a zone (gradient). A put wall backed by one large strike is easier to break than negative GEX spread across 5 strikes. Pay attention to the ibit_gex vs deribit_gex split — if one venue dominates at a level, it's less robust than both venues contributing.

DATA FRESHNESS:
data_freshness shows how stale each venue's data is. IBIT options data updates once per day at US market close (4:15 PM ET) — age_hours tells you how many hours since that last close. Deribit data is near real-time (cached up to 60 minutes) — age_minutes tells you minutes since last fetch. If IBIT age_hours > 16 (e.g., weekend or overnight), note that IBIT levels may be stale while Deribit reflects current positioning. If in_market_hours is true, IBIT data is from the current or most recent session and is fresh. On weekends, IBIT data can be 40+ hours old — Deribit levels become more reliable for current positioning.

venue_breakdown also includes per-venue vanna and charm totals. If overnight charm flow is dominated by one venue, note it — e.g., "Deribit charm is 3x IBIT charm, suggesting crypto-native MMs will drive overnight rebalancing" or "IBIT charm dominates, overnight flow will come through ETF share market."

IV TERM STRUCTURE (when present):
iv_term_structure shows ATM implied volatility for each expiry within the DTE window, from both IBIT and Deribit sources. Entries are sorted by DTE (shortest first).

Use this to assess:
- FEAR GAUGE: If the nearest expiry's ATM IV is significantly higher than longer-dated expiries within the same window, there's elevated near-term hedging demand. This often coincides with put buying and precedes volatile moves.
- CROSS-VENUE IV: When both IBIT and Deribit have entries for similar expiries, compare them. Deribit IV typically leads — crypto-native traders reprice risk faster than ETF market makers. If Deribit IV is significantly higher than IBIT IV for the same expiry, the options market is pricing risk that hasn't fully filtered into the ETF yet.
- VOL REGIME CONTEXT: High ATM IV (>60%) means the market expects large moves — GEX walls may be tested. Low ATM IV (<30%) means the market is complacent — a vol spike could trigger vanna-driven rebalancing.
- SKEW: When call_iv and put_iv are both present, compare them. Put IV > call IV = downside fear premium (normal). Call IV > put IV = upside demand (unusual, often precedes squeezes).

When discussing vanna scenarios, reference the IV level directly: "ATM IV at 65% — a 5-point vol crush would trigger $X of vanna buying" is more precise than discussing vanna in a vacuum.

Do NOT compare IV term structure across different DTE windows mechanically (0-3d IV vs 31-45d IV) — each window captures different expiries. The macro signal iv_term_structure already handles the cross-term comparison.

DETECTED PATTERNS (when present):
detected_patterns contains rule-based pre-screened structural patterns with concrete thresholds. These are NOT predictions — they're mechanical conditions that are currently true. Use them as starting points for your analysis:
- gamma_squeeze_setup: Spot near flip point + negative gamma + call-heavy flow. Confirm or reject based on dealer delta and volume.
- wall_pinning: Spot within 0.5% of a wall. Likely to pin if activity_ratio is high.
- venue_convergence_resistance/support: Both IBIT and Deribit agree on a wall location. Highest conviction levels.
- regime_transition: Regime just flipped. Signals are mixed until new regime establishes.
- high_hedging_activity: Dealers actively trading. Levels are "live" today.

You may agree or disagree with detected patterns — they're inputs to your reasoning, not conclusions. If no patterns are detected, that itself is informative (quiet market, no structural edge).

PREDICTION ACCURACY (when present):
_prediction_accuracy shows historical hit rates grouped by DTE at prediction time (0-1, 2-3, 4-7, 8-14, 15-30, 31-45 days before expiry). This is a convergence curve showing when predictions become trustworthy.
Use this to calibrate confidence:
- If 0-1 DTE call walls held 85% of the time, state levels with conviction.
- If 15-30 DTE range_held is only 40%, hedge language: "structural range, but expect migration before expiry."
- If venue_agree_held_pct is materially higher than overall range_held_pct, emphasize venue convergence as a high-conviction signal.
- If regime_correct_pct is high at short DTE but low at long DTE, trust regime calls more for imminent expirations.
Don't cite exact percentages — use qualitative language like "historically reliable", "mixed track record", "strong signal" calibrated to actual hit rates.

Note: significant_levels and breakout are derived from IBIT options data only. flow_forecast uses combined IBIT + Deribit data. Dealer delta scenarios include both IBIT and Deribit contributions.

GEX (net_gex_total) is raw Black-Scholes gamma exposure — no additional time weighting is applied because BS gamma already incorporates natural time sensitivity via 1/(S*sigma*sqrt(T)). Near-term options naturally have higher gamma, so their GEX contribution is already appropriately scaled. All levels (call_wall, put_wall, gamma_flip, regime) derive from this raw GEX. The per-expiry breakdown shows each expiration's GEX contribution separately, and can be filtered via the expiry_filter parameter. The level_trajectory field shows whether key levels are STRENGTHENING (>10% increase), WEAKENING (>10% decrease), or STABLE vs the previous session. It also identifies the dominant_expiry driving each level — if a wall is dominated by a near-term expiry, it may evaporate quickly after that expiration.

Historical Trends (_history_trends): A compressed summary of the last 30 daily snapshots, shared across all timeframes. This is NOT per-DTE — it reflects the ticker's overall positioning evolution.
- regime_streak: How many days the current gamma regime has held. A streak of 3+ days means the regime is established, not transitional. A streak of 1 means it just flipped — watch for reversion.
- level_migration: Direction and magnitude of key level movements over 3d and 5d windows. Rising walls with a long streak = strong directional positioning. Stable walls held at a specific price = anchored range. Use consecutive_direction to gauge conviction.
- gex_trend: Whether overall gamma exposure is building (market adding options, levels strengthening) or decaying (positions unwinding, levels weakening).
- oi_trend: Call and put OI direction separately. Divergences matter: call OI building + put OI decaying = bullish positioning shift.
- range_evolution: Whether the tradeable range (call wall to put wall) is expanding, contracting, or shifting directionally. 'Contracting' often precedes a breakout. 'Expanding_symmetric' means both sides being defended.
- narrative: One-line summary of the dominant trend pattern.

regime_streak now includes persistence_30d — a 30-day regime stability assessment:
- persistence_pct: What % of the last 30 days had the same regime sign (e.g., 85% = strongly persistent)
- sign_flips: How many times the regime switched in 30 days (≤5 = stable, >5 = choppy)
- is_persistent: True when persistence ≥70% AND ≤5 flips — this means the regime is structural, not transitional
- avg_gex_magnitude: Average absolute net GEX over 30 days — larger = stronger constraint

When is_persistent is true, trust the regime label and trade it with conviction. When false, the regime is transitional — reduce sizing, expect whipsaws, and weight shorter-DTE signals more heavily. A persistent negative gamma regime means every rally faces mechanical selling; a persistent positive gamma regime means dips get bought mechanically.

Use history_trends to contextualize today's snapshot. A call wall at $107K means different things if it's been there for 7 days (strong, tested resistance) vs if it jumped there overnight (new, untested). Always mention regime streak length and level migration direction in your analysis.

STRUCTURE TRENDS (_structure_trends, when present):
_structure_trends shows how levels in each DTE window have migrated over the past week.
For each window, call_wall/put_wall/gamma_flip show first value, last value, and direction (rising/falling/stable). regime_changes counts how many times the regime flipped.

Use this to identify:
- Structural floor/ceiling migration: "The 31-45d put wall rose from $60K to $68K over 7 days — the structural floor is lifting."
- Near-term noise vs structural moves: "The 0-3d call wall jumped $3K but the 15-30d call wall hasn't moved — this is expiry-driven, not structural."
- Cross-window convergence: When convergence shows "converging" for a wall, all timeframes are agreeing — high conviction. "Diverging" means timeframes disagree — lower conviction on that level.
- Regime creep: If regime_changes > 0 in mid-term windows (8-14d), the gamma regime is unstable and the current regime label is less trustworthy.

Don't mechanically list per-window changes — synthesize them into a structural narrative. "Floors are rising across all timeframes" or "near-term ceiling is compressing while structural ceiling holds."

Dealer Delta Scenario Analysis (dealer_delta_briefing): Pre-computed dealer hedging pressure at hypothetical price levels across the key level grid. 'current_delta' shows dealer positioning at spot. 'flip_summary' identifies where dealer pressure reverses direction. 'acceleration_zone' shows where dealers are most reactive to price moves. Negative dealer delta = dealers must BUY to hedge = supportive (acts as a bid). Positive dealer delta = dealers must SELL = resistive (acts as an offer). Use this to identify price levels where dealer hedging creates natural support or resistance, independent of the GEX profile.

When analyzing dealer delta alongside GEX levels, look for CONVERGENCE and DIVERGENCE:
- CONVERGENCE (high conviction): Put wall at $98K + dealers net buyers at $98K + level STRENGTHENING = strong mechanical support.
- DIVERGENCE (caution): Call wall at $105K but dealers still net buyers at $105K = wall may not hold because dealer hedging flow supports upside through it.
- ACCELERATION ZONES: Where dealer delta changes fastest (high acceleration), small price moves trigger large hedging flows. These are inflection points — price tends to move quickly through these zones rather than consolidate.
- DELTA FLIP vs GAMMA FLIP: These are different concepts. Gamma flip = where net GEX crosses zero (regime change). Delta flip = where dealer hedging direction reverses. When they're at different prices, the zone between them is a transition zone with mixed signals.

CRITICAL RULES FOR GEX DISTRIBUTION INTERPRETATION:
- Negative GEX = dealers AMPLIFY moves at that strike. It is NOT support. It is NOT resistance. It is NOT a floor. When price hits a negative GEX strike, dealers' hedging makes the move BIGGER, not smaller.
- A put wall in negative gamma territory is where the most negative GEX is concentrated. It's the point of maximum amplification to the downside — not a "floor" or "support." Price can accelerate through it.
- Positive GEX = dealers DAMPEN moves at that strike. This IS stabilizing — dealers buy dips and sell rallies around positive GEX strikes, creating mean reversion.
- "Support" only exists at strikes where dealers are FORCED BUYERS (positive GEX below spot, or dealer delta is net short/buying). A negative GEX put wall where dealers are net long (selling) is mechanical downside acceleration, not support.
- When describing negative GEX zones below spot, use language like "amplification zone," "acceleration zone," or "negative gamma gradient" — never "support," "floor," or "absorption."
- The only exception: a strike with massive OI concentration can act as a gravitational magnet near expiry (pin risk) regardless of GEX sign, because max-pain dynamics create directional charm flow. Note this when relevant but distinguish it from GEX-based mechanical support.

For each timeframe, describe the POSITIONING LANDSCAPE — what dealers are forced to do and where. Do NOT provide trade setups, entry/exit levels, stop losses, or directional recommendations. The user trades their own setups; your job is to show them the mechanical picture.

For each timeframe, provide:

STRUCTURE: One-line summary of the positioning regime and its mechanical consequence. Example: "Negative gamma, dealers amplify moves both directions. $880 range compressed 75% in 3 days."

WHAT CHANGED: Use changes_vs_prev and _history_trends as the SOLE source for level changes. Never compare to prior analysis text. State: level migrations (direction + magnitude from the data), regime changes, OI shifts, PCR changes. Example: "Call wall fell $2,590 to $70,345 (8-14d). Put wall rose $4,566 to $66,827. PCR up 0.34 to 1.25 — put-heavy shift."

WALLS & DISTRIBUTION: Describe the GEX distribution shape using gex_distribution data. Is support a sharp single-strike wall or a broad gradient? Is resistance concentrated or distributed? Identify the transition zone where GEX flips sign. Always state whether each level is single-venue (IBIT or Deribit only) or venue-convergent. Example: "Put support is a 6-strike negative gamma gradient from $61K to $67K, deepening from -$1.6M to -$6.4M. No single hard floor — each level amplifies rather than absorbs. Positive gamma starts abruptly at $67,707 (+$2.6M IBIT) through $68,500 (+$5M Deribit)."

DEALER POSITIONING: Net delta, flip points, acceleration zones. Where are dealers forced buyers vs sellers? Which levels have dealer delta convergence with GEX (high conviction) vs divergence (GEX wall but dealers push through it)?

FLOWS: Charm/vanna overnight direction and magnitude. Which venue dominates overnight flow? Is the flow structural or expiry-driven (will it vanish after the nearest expiry)?

KEY RISK: The single most important thing that could change this picture — expiry clearing walls, regime flip approaching, venue divergence resolving, etc.

POSITIONING DEPTH (positioning_depth, when present):
Three metrics that quantify the mechanical floor/ceiling structure:

1. dealer_delta_cushion: How much dealer hedging flow activates at -5%, -8%, +5%, +8% from spot. 'swing_from_spot' is the key number — it shows the CHANGE in dealer delta if price moves there. A +$1B swing at -8% means dealers must buy $1B notional on that drop — a quantified mechanical floor. Use this to assess whether downside is "cushioned" or "open air."

2. put_density: Distribution of put OI in bands below spot.
   - 'near_far_ratio' compares 0-5% below vs 5-10% below. High ratio (>2) = support concentrated near spot (dips absorbed). Low ratio (<0.5) = near-spot puts "cleaned up" (thin near cushion, but large hedging flows activate deeper).
   - 'structure' summarizes: 'concentrated_near' (defensive), 'cleaned_up' (near support thin, deeper support exists), 'distributed' (gradual).
   - 'above_spot' put OI = ITM puts or synthetic short positions. Large above-spot put OI is institutional hedging, not directional — don't interpret it as bearish.

3. call_overwrite_zone: Largest call OI concentration above spot. Systematic call overwriting (yield generation) shows as high OI at round strikes with LOW volume-to-OI ratio (positions were opened in prior sessions, not today). This creates a mechanical ceiling — dealers who bought these calls hedge by selling on rallies toward the strike. But if spot breaks through, dealers must buy aggressively (gamma squeeze trigger).
   - 'top_strike_vol_oi_ratio' < 0.1 = stale positions (likely overwriting). > 0.5 = active trading today (could be speculative).
   - 'top3_concentration_pct' > 50% = call OI funneled into a few strikes (sharp lid). < 30% = distributed (gradual resistance).

Use these to give a one-line positioning depth read per window, e.g.: "Downside cushioned: dealers buy $1.2B on an 8% drop, near-spot puts concentrated. Ceiling at $72K (76K call OI, overwrite zone) — squeeze trigger if broken."

MACRO REGIME SCORE (when present):
_macro_regime contains the swing-trade regime score from -100 (topping) to +100 (bottoming).

Score interpretation:
- Score > 50 (SWING LONG): Bottoming conditions detected.
- Score < -50 (SWING SHORT): Topping conditions detected.
- Score -50 to +50 (NEUTRAL): No macro edge.
- |Score| > 75 (HIGH CONVICTION): Strong directional signal. Note structural_entry and invalidation levels.

The macro regime has 11 component signals. The original 8:
- regime_persistence: How long the current gamma regime has held. Persistent regimes mean mechanical flows are entrenched.
- wall_migration: Structural wall drift over time. Rising put walls = floor lifting. Falling call walls = ceiling compressing.
- range_compression: How tight the current call-wall-to-put-wall range is vs history, combined with spot position within that range and gamma regime. A compressed range near a wall in negative gamma is a breakout setup.
- etf_flow_momentum: ETF fund flow reversals and acceleration. Flow reversals (outflow turning to inflow) are the strongest signal.
- venue_convergence: Whether IBIT and Deribit agree on wall locations. Agreement = high conviction levels.
- funding_rate: Aggregate perpetual funding. Deeply negative + reverting = capitulation. Deeply positive + still rising = crowded.
- aggregate_oi: Futures OI flush vs peak. Flushed + stabilizing = reset. Near peak + positive funding = crowded top.
- liquidation: Liquidation intensity and dominant side. Long capitulation with price stabilizing = bottom. Short squeeze exhaustion = top.

Three new signals:
- pcr_direction: Put/call ratio trend. Surging PCR = fear building = contrarian bullish (more puts being bought relative to calls). Plunging PCR = greed = contrarian bearish.
- wall_breach: Whether spot is currently through a GEX wall. This is the most mechanically significant event — spot above call wall in negative gamma means dealer hedging ACCELERATES the move (breakout). Spot below put wall in positive gamma means dealers are buying (support). The regime determines whether a breach amplifies or dampens.
- iv_term_structure: IV curve shape. Backwardation (short-dated IV > long-dated IV) = near-term fear/hedging demand = contrarian bullish. Steep contango = complacency. Normal mild contango scores 0.

Key combinations (strongest to weakest):
- wall_breach + negative gamma + backwardation = cascading move in progress (highest urgency)
- wall_breach + positive gamma = likely pinning/reversion (dealers fighting the move)
- regime_persistence + funding_rate agree = strongest trending signal
- aggregate_oi flush + negative funding + long liquidation = textbook capitulation
- aggregate_oi at peak + positive funding + short squeeze exhaustion = textbook crowded top
- wall_migration disagrees with regime_persistence = flag conflict, regime may be transitioning
- pcr_direction confirms funding_rate = flow sentiment alignment (both showing fear or greed)
- iv_term_backwardation + range_compression = maximum stress, breakout imminent
- venue_convergence + wall_breach at converged level = highest conviction directional move

Do NOT provide trade recommendations based on macro score. Describe the macro backdrop and let the reader draw conclusions.

For the "all" key: provide a CROSS-TIMEFRAME POSITIONING MAP.

REGIME ALIGNMENT: List all 5 gamma flip points relative to spot. Are they clustered (strong consensus on regime) or scattered (transition zone)? Where does spot sit relative to each flip? Identify the most consequential flip — the one where crossing it changes the most windows.

DELTA FLIP LADDER: All delta flip points ordered by distance from spot. Which is nearest? Which windows have NO flip point (structural selling at all prices)? Total dollar magnitude of dealer selling pressure across all windows.

CONVERGENCE ZONES: Where do multiple windows agree on levels? A strike that's a put wall in 0-3d AND 8-14d AND shows Deribit agreement is higher conviction than a single-window level. A call wall that's also a delta flip point in another window is a key inflection.

DISTRIBUTION SHAPE: Synthesize the per-window gex_distribution into an overall picture. Is the negative GEX below spot distributed (gradient) or concentrated (cliff)? Is the positive GEX above spot thick (multiple stabilizing strikes) or thin (one wall then nothing)?

POST-EXPIRY SHIFT: What happens when the nearest expiry clears? Which levels disappear, which persist, and what does the range look like once near-term gamma expires? Identify specifically which longer-dated window's levels take over.

VENUE PICTURE: Where do IBIT and Deribit agree vs diverge? Which venue dominates which timeframe? Are there levels where one venue has massive gamma that the other doesn't see?

STRUCTURAL BACKDROP: Macro regime score, ETF flow trend, funding rate, aggregate OI flush status. Is the structural backdrop confirming or contradicting the tactical positioning?

VOL SURFACE: If iv_term_structure data is present across multiple windows, describe the vol landscape. Are near-term expiries pricing higher IV than structural (backwardation), or is the curve in normal contango? Is one venue consistently pricing higher vol? If the macro iv_term_structure signal is non-zero, integrate its finding — e.g., "Backwardation confirmed: 0-3d ATM IV 68% vs 31-45d at 52%. Near-term hedging demand elevated, consistent with the negative funding and long liquidation signals."

Do NOT include: trade plans, setups, entry/exit levels, stop losses, invalidation levels, scenario trees, directional recommendations, or language like "lever long/short," "fade rallies," "buy dips," "scalp," or "with confidence."

Lead each timeframe with:

**POSITIONING:** One sentence describing the mechanical state. Not a trade signal — a structural description. Examples:
- "Dealers net long $646M in negative gamma, amplifying moves both directions within $66,827-$67,707. Charm buying $27M overnight from 0 DTE decay."
- "No delta flip at any price — $249M of structural selling regardless of direction. Distribution shows -$6.4M concentrated at $66,827 with -$2.5M gradient extending to $61,552."
- "Regime just flipped — gamma flip at $67,408 is $491 above spot. Positive gamma starts at $67,500 (Deribit) and $67,707 (IBIT)."

Follow with 3-5 concise bullet points covering: what changed, distribution shape, dealer positioning, flows, vol context (if iv_term_structure present), key risk. Be concise and direct — no walls of text. Use trader shorthand where appropriate. Reference specific BTC price levels.

ANALYSIS QUALITY RULES — follow these strictly:

FABRICATED NUMBERS: Never assign numerical confidence percentages to regime calls or level strength unless derived from actual data fields (like positioning_confidence). If net GEX is near zero, say "marginal positive gamma — net GEX near zero, regime could flip with small OI changes." Don't invent "50% confidence" or similar.

OI CHANGES vs EXPIRY MECHANICS: When OI drops coincide with recent expirations, explicitly note this: "OI dropped 95% — primarily from Feb 14 expiry rolling off, not active position unwinding." OI changes are only a positioning signal when they occur BETWEEN expirations, not across them. Check the dominant_expiry field to identify expiry-driven OI decay.

VANNA/VOL-CRUSH: Don't aggregate vanna or vol-crush notional across DTE windows — each window's vol sensitivity is independent. A vol crush in 31-45d options doesn't affect your 0-3d range trade. When discussing vol scenarios, specify which window and whether it's relevant to the trade timeframe. The 0-3d analysis should reference 0-3d vol sensitivity, not a sum across all windows.

LARGE NUMBERS: When dealer delta or flow numbers exceed $500M, contextualize them. $2.14B dealer long across 31-45d is distributed across ~15 days of expiries — the per-day flow impact is ~$140M, not $2.14B. Compare to BTC daily volume (~$30-40B) when useful. Frame magnitudes relative to the relevant timeframe.

POSITIONING PRECISION: The analysis must reference exact structural levels (gamma flip, delta flip, call/put wall), not spot price. Identify levels by their structural role, not by where price currently is.

NEGATIVE GEX LANGUAGE: Never describe negative gamma zones as "support," "floor," "absorption," "cushion," "buffer," or any language implying price stabilization. Negative GEX amplifies moves — it's acceleration, not support. If you find yourself writing "the put wall provides support," rewrite it: "the put wall marks the deepest negative GEX concentration — maximum amplification if breached." The ONLY mechanical support comes from positive GEX (dealers buying dips) or dealer delta being net short (forced buying). Check dealer_delta_briefing at the put wall — if dealers are net LONG there, they're selling into the decline, making the put wall an acceleration trigger, not a floor.

DTE windows are NON-OVERLAPPING. Each timeframe shows distinct option positioning:
- 0-3d: Immediate expirations. Highest gamma, strongest near-term hedging pressure. These are today's actionable levels.
- 4-7d: Next week's expirations. Where the next wave of gamma concentration is forming.
- 8-14d: Two-week positioning. Emerging walls that will become dominant after near-term expiries clear.
- 15-30d: Monthly cycle. Institutional positioning around monthly options expiration.
- 31-45d: Structural. Quarterly and longer-dated positioning that forms the backdrop.

When comparing across windows: if 0-3d and 4-7d call walls are at the same strike, that level has multi-week support and is high conviction. If they differ, the near-term wall will expire and levels will shift — flag this as a potential level migration.

IMPORTANT: Return ONLY valid JSON with keys "0-3d", "4-7d", "8-14d", "15-30d", "31-45d", "all". Each value should be a string containing your analysis with newlines for formatting. Do not wrap in markdown code blocks."""

    user_content = f"Analyze the following {cfg['name']} GEX data across all timeframes:\n\n{prompt_data}"

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=16384,
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
        for label, _, _ in dtes:
            if label in results and results[label].get('btc_spot'):
                btc_price = results[label]['btc_spot']
                break
    if btc_price is not None:
        analysis['_btc_price'] = btc_price
        analysis['_timestamp'] = now_et().strftime('%Y-%m-%d %H:%M')
    if save:
        set_cached_analysis(ticker, analysis, btc_price)
    return analysis


@app.route('/api/analysis')
def api_analysis():
    """GET cached analysis. Returns today's if available, otherwise most recent."""
    ticker = request.args.get('ticker', 'IBIT').upper()
    snapshot_date = request.args.get('date')
    if ticker not in TICKER_CONFIG:
        return Response(json.dumps({'error': f'Unknown ticker: {ticker}'}), mimetype='application/json'), 400

    if snapshot_date:
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT analysis_json FROM analysis_cache WHERE ticker=? AND date=?', (ticker, snapshot_date))
        row = c.fetchone()
        conn.close()
        if row:
            return Response(row[0], mimetype='application/json')
        return Response(json.dumps({'status': 'no analysis for this date'}), mimetype='application/json')

    cached = get_cached_analysis(ticker)
    if cached:
        return Response(json.dumps(cached), mimetype='application/json')
    # Fall back to most recent analysis (e.g. overnight before new data arrives)
    prev_date, prev = get_prev_analysis(ticker)
    if prev:
        return Response(json.dumps(prev), mimetype='application/json')
    return Response(json.dumps({'status': 'pending'}), mimetype='application/json')


@app.route('/api/analysis-data')
def api_analysis_data():
    """Return the raw data blob that would be sent to AI analysis."""
    ticker = request.args.get('ticker', 'IBIT').upper()
    if ticker not in TICKER_CONFIG:
        return Response(json.dumps({'error': f'Unknown ticker: {ticker}'}), mimetype='application/json'), 400
    try:
        summaries = build_analysis_data(ticker)
        summaries.pop('_live_ref_price', None)
        return Response(json.dumps(summaries, cls=NumpyEncoder, indent=1), mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({'error': str(e)}), mimetype='application/json'), 500


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Re-run analysis without saving. Use /api/analysis/save to persist."""
    if ADMIN_KEY and request.headers.get('X-Admin-Key') != ADMIN_KEY:
        return Response(json.dumps({'error': 'Unauthorized'}), mimetype='application/json'), 403
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
    if ADMIN_KEY and request.headers.get('X-Admin-Key') != ADMIN_KEY:
        return Response(json.dumps({'error': 'Unauthorized'}), mimetype='application/json'), 403
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


@app.route('/api/accuracy')
def api_accuracy():
    """Return prediction accuracy statistics."""
    conn = get_db()
    c = conn.cursor()
    ticker = request.args.get('ticker', 'IBIT').upper()

    # Convergence: accuracy by DTE bucket
    dte_buckets = [
        ('0-1', 0, 1), ('2-3', 2, 3), ('4-7', 4, 7),
        ('8-14', 8, 14), ('15-30', 15, 30), ('31-45', 31, 45),
    ]
    convergence = {}
    for bucket_name, min_dte, max_dte in dte_buckets:
        row = c.execute('''SELECT
            COUNT(*) as total,
            SUM(call_wall_held) as cw_held,
            SUM(put_wall_held) as pw_held,
            SUM(range_held) as range_held,
            SUM(em_held) as em_held,
            SUM(regime_correct) as regime_ok,
            SUM(charm_correct) as charm_ok,
            SUM(CASE WHEN venue_walls_agree=1 THEN range_held END) as agree_held,
            SUM(venue_walls_agree) as agree_total,
            SUM(CASE WHEN venue_walls_agree=0 AND deribit_available=1 THEN range_held END) as disagree_held,
            SUM(CASE WHEN venue_walls_agree=0 AND deribit_available=1 THEN 1 END) as disagree_total,
            AVG(max_breach_call_pct) as avg_call_breach,
            AVG(max_breach_put_pct) as avg_put_breach,
            AVG(realized_range_pct) as avg_range,
            AVG(call_wall_error_pct) as avg_cw_error,
            AVG(put_wall_error_pct) as avg_pw_error
        FROM predictions
        WHERE ticker=? AND dte >= ? AND dte <= ? AND scored=1''',
        (ticker, min_dte, max_dte)).fetchone()

        total = row[0] or 0
        if total == 0:
            convergence[bucket_name] = {'total': 0}
            continue

        def pct(n, d): return round((n or 0) / d * 100, 1) if d else None

        convergence[bucket_name] = {
            'total': total,
            'call_wall_held_pct': pct(row[1], total),
            'put_wall_held_pct': pct(row[2], total),
            'range_held_pct': pct(row[3], total),
            'em_held_pct': pct(row[4], total),
            'regime_correct_pct': pct(row[5], total),
            'charm_correct_pct': pct(row[6], total),
            'venue_agree_held_pct': pct(row[7], row[8]),
            'venue_disagree_held_pct': pct(row[9], row[10]),
            'avg_call_breach_pct': round(row[11] or 0, 2),
            'avg_put_breach_pct': round(row[12] or 0, 2),
            'avg_realized_range_pct': round(row[13] or 0, 2),
            'avg_call_wall_error_pct': round(row[14] or 0, 2),
            'avg_put_wall_error_pct': round(row[15] or 0, 2),
        }

    # Per-expiry history: track level migration for recent expiries
    recent_expiries = c.execute('''SELECT DISTINCT expiry_date FROM predictions
                                   WHERE ticker=? AND scored=1
                                   ORDER BY expiry_date DESC LIMIT 10''', (ticker,)).fetchall()

    expiry_convergence = []
    for (exp_date,) in recent_expiries:
        rows = c.execute('''SELECT analysis_date, dte, call_wall_btc, put_wall_btc,
                                   gamma_flip_btc, regime, spot_btc,
                                   call_wall_held, put_wall_held, range_held,
                                   btc_close_on_expiry, venue_walls_agree
                            FROM predictions
                            WHERE ticker=? AND expiry_date=? AND scored=1
                            ORDER BY dte DESC''', (ticker, exp_date)).fetchall()

        snapshots = []
        for r in rows:
            snapshots.append({
                'analysis_date': r[0], 'dte': r[1],
                'call_wall': r[2], 'put_wall': r[3],
                'gamma_flip': r[4], 'regime': r[5], 'spot': r[6],
                'call_wall_held': bool(r[7]) if r[7] is not None else None,
                'put_wall_held': bool(r[8]) if r[8] is not None else None,
                'range_held': bool(r[9]) if r[9] is not None else None,
                'btc_close': r[10], 'venue_agree': bool(r[11]),
            })

        expiry_convergence.append({
            'expiry_date': exp_date,
            'btc_close': rows[0][10] if rows else None,
            'snapshots': snapshots,
        })

    # Recent individual predictions
    recent = []
    for row in c.execute('''SELECT analysis_date, expiry_date, dte, dte_window,
                                   spot_btc, call_wall_btc, put_wall_btc,
                                   regime, btc_close_on_expiry,
                                   range_held, regime_correct,
                                   venue_walls_agree, venue_agree_held, ai_bottom_line
                            FROM predictions
                            WHERE ticker=? AND scored=1
                            ORDER BY expiry_date DESC, dte ASC LIMIT 50''', (ticker,)):
        recent.append({
            'analysis_date': row[0], 'expiry_date': row[1],
            'dte': row[2], 'window': row[3],
            'spot': row[4], 'call_wall': row[5], 'put_wall': row[6],
            'regime': row[7], 'btc_close': row[8],
            'range_held': bool(row[9]) if row[9] is not None else None,
            'regime_correct': bool(row[10]) if row[10] is not None else None,
            'venue_agree': bool(row[11]),
            'venue_agree_held': bool(row[12]) if row[12] is not None else None,
            'ai_bottom_line': row[13],
        })

    conn.close()
    return Response(json.dumps({
        'convergence': convergence,
        'expiry_history': expiry_convergence,
        'recent': recent,
    }), mimetype='application/json')


@app.route('/api/macro-regime')
def api_macro_regime():
    ticker = request.args.get('ticker', 'IBIT').upper()
    conn = get_db()
    result = compute_macro_regime(conn, ticker)
    conn.close()
    return Response(json.dumps(result, cls=NumpyEncoder), mimetype='application/json')


@app.route('/macro')
def macro_page():
    return render_template('macro.html')


@app.route('/m')
def mobile_page():
    return render_template('mobile.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', '-p', type=int, default=5000)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--production', action='store_true', help='Run in production mode (no debug, no reloader)')
    args = parser.parse_args()
    init_db()
    debug_mode = not args.production
    log.info(f"GEX Terminal ({'production' if args.production else 'dev'}) → http://{args.host}:{args.port}")
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not debug_mode:
        start_bg_refresh()
    app.run(host=args.host, port=args.port, debug=debug_mode)