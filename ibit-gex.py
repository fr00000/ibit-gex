#!/usr/bin/env python3
"""
IBIT GEX Trading Dashboard
===========================
Pre-market analysis tool for trading IBIT using options-derived levels.

Outputs:
  1. Terminal summary of key trading levels
  2. GEX profile chart (PNG)
  3. OI profile chart (PNG)
  4. CSV export for charting platform overlay

Usage:
  python3 ibit_gex_trading.py              # default: all expirations within 45 DTE
  python3 ibit_gex_trading.py --dte 30     # custom DTE cutoff
  python3 ibit_gex_trading.py --ticker SPY # works for any optionable ticker
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timezone, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import argparse
import warnings
import sys
import sqlite3
import os
from pathlib import Path
warnings.filterwarnings('ignore')


# ── CONFIG ──────────────────────────────────────────────────────────────────
RISK_FREE_RATE = 0.043
BTC_PER_SHARE = 0.000568   # IBIT-specific, ignored for non-BTC tickers
STRIKE_RANGE_PCT = 0.35    # ±35% from spot
OUTPUT_DIR = "."
DB_PATH = os.path.join(str(Path.home()), ".ibit_gex_history.db")


# ── HISTORY DATABASE ────────────────────────────────────────────────────────
def init_db(db_path=DB_PATH):
    """Initialize SQLite database for OI history tracking."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        ticker TEXT NOT NULL,
        spot REAL,
        btc_price REAL,
        gamma_flip REAL,
        call_wall REAL,
        put_wall REAL,
        max_pain REAL,
        regime TEXT,
        net_gex REAL,
        total_call_oi INTEGER,
        total_put_oi INTEGER,
        UNIQUE(date, ticker)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS strike_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        ticker TEXT NOT NULL,
        strike REAL NOT NULL,
        call_oi INTEGER,
        put_oi INTEGER,
        total_oi INTEGER,
        net_gex REAL,
        UNIQUE(date, ticker, strike)
    )''')
    conn.commit()
    return conn


def save_snapshot(conn, ticker, spot, btc_price, levels, df, date_str=None):
    """Save today's data to the database."""
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    c = conn.cursor()
    
    # Save summary snapshot
    c.execute('''INSERT OR REPLACE INTO snapshots 
                 (date, ticker, spot, btc_price, gamma_flip, call_wall, put_wall, 
                  max_pain, regime, net_gex, total_call_oi, total_put_oi)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (date_str, ticker, spot, btc_price,
               levels.get('gamma_flip'), levels.get('call_wall'), levels.get('put_wall'),
               levels.get('max_pain'), levels.get('regime'),
               levels.get('net_gex_total'),
               levels.get('total_call_oi'), levels.get('total_put_oi')))
    
    # Save per-strike data
    for _, row in df.iterrows():
        c.execute('''INSERT OR REPLACE INTO strike_history
                     (date, ticker, strike, call_oi, put_oi, total_oi, net_gex)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (date_str, ticker, row['strike'],
                   int(row['call_oi']), int(row['put_oi']),
                   int(row['total_oi']), row['net_gex']))
    
    conn.commit()


def get_previous_snapshot(conn, ticker, before_date=None):
    """Get the most recent previous snapshot for comparison."""
    c = conn.cursor()
    if before_date is None:
        before_date = datetime.now().strftime('%Y-%m-%d')
    
    c.execute('''SELECT date FROM snapshots 
                 WHERE ticker = ? AND date < ? 
                 ORDER BY date DESC LIMIT 1''', (ticker, before_date))
    row = c.fetchone()
    if not row:
        return None, None, None
    
    prev_date = row[0]
    
    # Get summary
    c.execute('SELECT * FROM snapshots WHERE date = ? AND ticker = ?', (prev_date, ticker))
    prev_summary = c.fetchone()
    
    # Get strike data as dict: strike → {call_oi, put_oi, total_oi, net_gex}
    c.execute('''SELECT strike, call_oi, put_oi, total_oi, net_gex 
                 FROM strike_history WHERE date = ? AND ticker = ?''', (prev_date, ticker))
    prev_strikes = {}
    for s_row in c.fetchall():
        prev_strikes[s_row[0]] = {
            'call_oi': s_row[1], 'put_oi': s_row[2],
            'total_oi': s_row[3], 'net_gex': s_row[4]
        }
    
    return prev_date, prev_summary, prev_strikes


def get_history(conn, ticker, days=7):
    """Get historical snapshots for trend display."""
    c = conn.cursor()
    c.execute('''SELECT date, spot, btc_price, gamma_flip, call_wall, put_wall, 
                        max_pain, regime, net_gex, total_call_oi, total_put_oi
                 FROM snapshots WHERE ticker = ?
                 ORDER BY date DESC LIMIT ?''', (ticker, days))
    return c.fetchall()


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
    """Vanna = dDelta/dVol — shows how delta changes with IV shifts."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return -norm.pdf(d1) * d2 / sigma


# ── DATA COLLECTION ─────────────────────────────────────────────────────────
def fetch_options_data(ticker_symbol, max_dte=45):
    """Fetch and aggregate options data across expirations."""
    ticker = yf.Ticker(ticker_symbol)

    # Spot price
    spot = ticker.info.get('regularMarketPrice')
    if spot is None:
        hist = ticker.history(period="1d")
        spot = float(hist['Close'].iloc[-1])

    all_exps = list(ticker.options)
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=max_dte)

    # Filter expirations by DTE
    selected_exps = []
    for exp_str in all_exps:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if exp_date <= cutoff:
            selected_exps.append(exp_str)

    if not selected_exps:
        selected_exps = all_exps[:3]  # fallback

    strike_data = {}
    exp_details = []

    for exp_str in selected_exps:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        dte = max((exp_date - now).days, 0)
        T = max(dte / 365.0, 0.5/365)

        # Time-weighting: near-term expirations have more gamma impact
        # Weight by 1/sqrt(DTE) — gamma concentrates as expiry approaches
        time_weight = 1.0  # raw aggregation, weighting is implicit in gamma calc

        try:
            chain = ticker.option_chain(exp_str)
        except Exception as e:
            print(f"  ⚠ Skipping {exp_str}: {e}")
            continue

        exp_call_oi = int(chain.calls['openInterest'].fillna(0).sum())
        exp_put_oi = int(chain.puts['openInterest'].fillna(0).sum())
        exp_details.append({
            'expiration': exp_str, 'dte': dte,
            'call_oi': exp_call_oi, 'put_oi': exp_put_oi
        })

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

                bid = row.get('bid', 0) or 0
                ask = row.get('ask', 0) or 0
                mid = (bid + ask) / 2 if (bid + ask) > 0 else row.get('lastPrice', 0)

                gamma = bs_gamma(spot, strike, T, RISK_FREE_RATE, iv)
                delta = bs_delta(spot, strike, T, RISK_FREE_RATE, iv, opt_type)
                vanna = bs_vanna(spot, strike, T, RISK_FREE_RATE, iv)

                # GEX from dealer perspective (dealers are short options)
                gex = sign * gamma * oi * 100 * spot**2 * 0.01

                # Dealer delta: dealers are short options, so flip sign
                dealer_delta = -delta * oi * 100

                # Dealer vanna exposure
                dealer_vanna = -sign * vanna * oi * 100 * spot * 0.01

                if strike not in strike_data:
                    strike_data[strike] = {
                        'call_oi': 0, 'put_oi': 0,
                        'call_gex': 0, 'put_gex': 0,
                        'call_delta': 0, 'put_delta': 0,
                        'call_vanna': 0, 'put_vanna': 0,
                        'call_premium': 0, 'put_premium': 0,
                    }

                prefix = opt_type
                strike_data[strike][f'{prefix}_oi'] += oi
                strike_data[strike][f'{prefix}_gex'] += gex
                strike_data[strike][f'{prefix}_delta'] += dealer_delta
                strike_data[strike][f'{prefix}_vanna'] += dealer_vanna
                strike_data[strike][f'{prefix}_premium'] += mid * oi * 100

    return spot, strike_data, selected_exps, exp_details


# ── ANALYSIS ────────────────────────────────────────────────────────────────
def analyze(spot, strike_data):
    """Derive all trading levels from the raw data."""
    rows = []
    for strike, d in sorted(strike_data.items()):
        net_gex = d['call_gex'] + d['put_gex']
        net_delta = d['call_delta'] + d['put_delta']
        net_vanna = d['call_vanna'] + d['put_vanna']
        total_oi = d['call_oi'] + d['put_oi']

        rows.append({
            'strike': strike,
            'call_oi': d['call_oi'],
            'put_oi': d['put_oi'],
            'total_oi': total_oi,
            'call_gex': d['call_gex'],
            'put_gex': d['put_gex'],
            'net_gex': net_gex,
            'net_dealer_delta': net_delta,
            'net_vanna': net_vanna,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df, {}

    levels = {}

    # ── Call Wall: strike with highest call GEX
    levels['call_wall'] = df.loc[df['call_gex'].idxmax(), 'strike']

    # ── Put Wall: strike with most negative put GEX (largest magnitude)
    levels['put_wall'] = df.loc[df['put_gex'].idxmin(), 'strike']

    # ── Gamma Flip: where net GEX crosses zero — find the one nearest to spot
    df_s = df.sort_values('strike')
    all_flips = []
    for i in range(len(df_s) - 1):
        g1, g2 = df_s.iloc[i]['net_gex'], df_s.iloc[i+1]['net_gex']
        s1, s2 = df_s.iloc[i]['strike'], df_s.iloc[i+1]['strike']
        # Detect any zero crossing (neg→pos or pos→neg)
        if (g1 < 0 and g2 > 0) or (g1 > 0 and g2 < 0):
            if (g2 - g1) != 0:
                flip = s1 + (s2 - s1) * (-g1) / (g2 - g1)
            else:
                flip = (s1 + s2) / 2
            direction = 'neg_to_pos' if g1 < 0 else 'pos_to_neg'
            all_flips.append({'strike': flip, 'direction': direction,
                              'dist_from_spot': abs(flip - spot)})
    
    if all_flips:
        # Prefer the flip nearest to spot
        nearest_flip = min(all_flips, key=lambda x: x['dist_from_spot'])
        levels['gamma_flip'] = nearest_flip['strike']
        levels['gamma_flip_direction'] = nearest_flip['direction']
        levels['all_gamma_flips'] = [f['strike'] for f in all_flips]
    else:
        levels['gamma_flip'] = spot
        levels['gamma_flip_direction'] = 'none'
        levels['all_gamma_flips'] = []

    # ── Max Pain: strike that minimizes total option value (pain for holders)
    def calc_pain(strike_price):
        pain = 0
        for _, row in df.iterrows():
            k = row['strike']
            pain += row['call_oi'] * max(0, k - strike_price) * 100  # calls ITM loss
            pain += row['put_oi'] * max(0, strike_price - k) * 100   # puts ITM loss
        return pain

    # More efficient: evaluate at each strike
    pain_values = {row['strike']: calc_pain(row['strike']) for _, row in df.iterrows()}
    # Actually we want to minimize total exercised value (max pain = min total payout)
    # Recalculate properly:
    def calc_total_payout(settle_price):
        total = 0
        for _, row in df.iterrows():
            k = row['strike']
            total += row['call_oi'] * max(0, settle_price - k) * 100
            total += row['put_oi'] * max(0, k - settle_price) * 100
        return total

    pain_df = pd.DataFrame([
        {'strike': s, 'payout': calc_total_payout(s)}
        for s in df['strike'].values
    ])
    levels['max_pain'] = pain_df.loc[pain_df['payout'].idxmin(), 'strike']

    # ── Expected Move: from ATM straddle (nearest expiration)
    atm_idx = (df['strike'] - spot).abs().idxmin()
    atm_strike = df.loc[atm_idx, 'strike']
    levels['atm_strike'] = atm_strike

    # ── Net GEX per 1% move
    total_net_gex = df['net_gex'].sum()
    levels['net_gex_total'] = total_net_gex
    levels['gex_per_1pct'] = total_net_gex / 100  # approximate

    # ── Total dealer delta (directional pressure)
    levels['net_dealer_delta'] = df['net_dealer_delta'].sum()
    levels['net_dealer_delta_$'] = levels['net_dealer_delta'] * spot

    # ── Regime: based on local net GEX around spot
    near_spot = df[(df['strike'] >= spot - 1.0) & (df['strike'] <= spot + 1.0)]
    local_net_gex = near_spot['net_gex'].sum() if not near_spot.empty else 0
    levels['regime'] = 'positive_gamma' if local_net_gex > 0 else 'negative_gamma'
    levels['local_net_gex'] = local_net_gex

    # ── Key resistance/support zones (top 3 each)
    resistance = df[df['net_gex'] > 0].nlargest(3, 'net_gex')['strike'].tolist()
    support = df[df['net_gex'] < 0].nsmallest(3, 'net_gex')['strike'].tolist()
    levels['resistance_levels'] = resistance
    levels['support_levels'] = support

    # ── High OI magnets (top 5 by total OI)
    magnets = df.nlargest(5, 'total_oi')[['strike', 'total_oi', 'call_oi', 'put_oi']].to_dict('records')
    levels['oi_magnets'] = magnets

    # ── Total OI
    levels['total_call_oi'] = df['call_oi'].sum()
    levels['total_put_oi'] = df['put_oi'].sum()
    levels['pcr'] = levels['total_put_oi'] / max(levels['total_call_oi'], 1)

    return df, levels


# ── CHARTS ──────────────────────────────────────────────────────────────────
def plot_gex_profile(df, spot, levels, ticker_symbol, output_path, expected_move=None, is_btc=False):
    """GEX profile bar chart — the money chart."""
    fig, ax = plt.subplots(figsize=(16, 9))

    # Filter to a tighter range for readability
    plot_df = df[(df['strike'] >= spot * 0.85) & (df['strike'] <= spot * 1.20)].copy()

    # Use BTC prices for x-axis if applicable
    if is_btc:
        x_values = plot_df['strike'] / BTC_PER_SHARE
        spot_x = spot / BTC_PER_SHARE
        gf_x = levels.get('gamma_flip', spot) / BTC_PER_SHARE
        cw_x = levels.get('call_wall', 0) / BTC_PER_SHARE
        pw_x = levels.get('put_wall', 0) / BTC_PER_SHARE
        mp_x = levels.get('max_pain', 0) / BTC_PER_SHARE
        x_label = 'BTC Price'
        bar_width = 500
    else:
        x_values = plot_df['strike']
        spot_x = spot
        gf_x = levels.get('gamma_flip', spot)
        cw_x = levels.get('call_wall', 0)
        pw_x = levels.get('put_wall', 0)
        mp_x = levels.get('max_pain', 0)
        x_label = 'Strike'
        bar_width = 0.35

    colors = ['#22c55e' if g > 0 else '#ef4444' for g in plot_df['net_gex']]

    bars = ax.bar(x_values, plot_df['net_gex'], width=bar_width,
                  color=colors, alpha=0.85, edgecolor='none')

    # Expected move range
    if expected_move:
        if is_btc:
            em_lo = expected_move['lower'] / BTC_PER_SHARE
            em_hi = expected_move['upper'] / BTC_PER_SHARE
        else:
            em_lo = expected_move['lower']
            em_hi = expected_move['upper']
        ax.axvspan(em_lo, em_hi,
                   alpha=0.08, color='#60a5fa', label=f'Expected Move ±{expected_move["pct"]:.1f}%')

    # Spot line
    price_fmt = lambda v: f'${v:,.0f}' if is_btc else f'${v:.2f}'
    ax.axvline(spot_x, color='#3b82f6', linewidth=2.5, linestyle='-', alpha=0.9,
               label=f'Spot {price_fmt(spot_x)}')

    # Gamma flip
    ax.axvline(gf_x, color='#f59e0b', linewidth=2, linestyle='--', alpha=0.8,
               label=f'Gamma Flip {price_fmt(gf_x)}')

    # Call wall
    cw_strike = levels.get('call_wall')
    if cw_strike and cw_strike in plot_df['strike'].values:
        ax.axvline(cw_x, color='#10b981', linewidth=1.5, linestyle=':', alpha=0.7,
                   label=f'Call Wall {price_fmt(cw_x)}')

    # Put wall
    pw_strike = levels.get('put_wall')
    if pw_strike and pw_strike in plot_df['strike'].values:
        ax.axvline(pw_x, color='#f43f5e', linewidth=1.5, linestyle=':', alpha=0.7,
                   label=f'Put Wall {price_fmt(pw_x)}')

    # Max pain
    mp_strike = levels.get('max_pain')
    if mp_strike and mp_strike in plot_df['strike'].values:
        ax.axvline(mp_x, color='#a855f7', linewidth=1.5, linestyle='-.', alpha=0.7,
                   label=f'Max Pain {price_fmt(mp_x)}')

    # Zero line
    ax.axhline(0, color='white', linewidth=0.5, alpha=0.3)

    # Regime shading
    ax.axvspan(x_values.min(), gf_x, alpha=0.04, color='#ef4444')
    ax.axvspan(gf_x, x_values.max(), alpha=0.04, color='#22c55e')

    # Regime labels
    y_pos = ax.get_ylim()[1] * 0.92
    ax.text(gf_x - (gf_x - x_values.min()) * 0.3, y_pos,
            '− GAMMA\n(volatile)', fontsize=9, color='#f87171',
            ha='center', va='top', fontweight='bold', alpha=0.6)
    ax.text(gf_x + (x_values.max() - gf_x) * 0.3, y_pos,
            '+ GAMMA\n(stable)', fontsize=9, color='#4ade80',
            ha='center', va='top', fontweight='bold', alpha=0.6)

    # Styling
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#0f0f1a')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.tick_params(colors='#ccc', labelsize=10)

    if is_btc:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        # Add secondary x-axis for IBIT prices
        ax2 = ax.twiny()
        ax2.set_xlim([l * BTC_PER_SHARE for l in ax.get_xlim()])
        ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter('IBIT $%.1f'))
        ax2.tick_params(colors='#666', labelsize=8)
        ax2.spines['top'].set_color('#333')
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    else:
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('$%.1f'))

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    ax.set_xlabel(x_label, fontsize=12, color='#ccc', labelpad=10)
    ax.set_ylabel('Net GEX ($)', fontsize=12, color='#ccc', labelpad=10)

    regime_str = "▼ NEGATIVE GAMMA" if levels['regime'] == 'negative_gamma' else "▲ POSITIVE GAMMA"
    spot_label = f'BTC ${spot_x:,.0f}' if is_btc else f'${spot:.2f}'

    ax.set_title(
        f'{ticker_symbol} Gamma Exposure Profile  ·  {spot_label}  ·  {regime_str}',
        fontsize=15, color='white', fontweight='bold', pad=20 if is_btc else 15
    )

    legend = ax.legend(loc='upper left', fontsize=10, facecolor='#1a1a2e',
                       edgecolor='#444', labelcolor='#ccc')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


def plot_oi_profile(df, spot, levels, ticker_symbol, output_path, is_btc=False):
    """Call vs Put OI profile — shows positioning."""
    fig, ax = plt.subplots(figsize=(16, 6))

    plot_df = df[(df['strike'] >= spot * 0.85) & (df['strike'] <= spot * 1.20)].copy()

    if is_btc:
        x_values = plot_df['strike'] / BTC_PER_SHARE
        spot_x = spot / BTC_PER_SHARE
        width = 400
    else:
        x_values = plot_df['strike']
        spot_x = spot
        width = 0.35

    ax.bar(x_values - width/2, plot_df['call_oi'], width,
           color='#22c55e', alpha=0.7, label='Call OI')
    ax.bar(x_values + width/2, plot_df['put_oi'], width,
           color='#ef4444', alpha=0.7, label='Put OI')

    price_fmt = lambda v: f'${v:,.0f}' if is_btc else f'${v:.2f}'
    ax.axvline(spot_x, color='#3b82f6', linewidth=2, linestyle='-', alpha=0.9,
               label=f'Spot {price_fmt(spot_x)}')

    # Highlight OI magnets
    for mag in levels.get('oi_magnets', [])[:3]:
        if mag['strike'] in plot_df['strike'].values:
            mag_x = mag['strike'] / BTC_PER_SHARE if is_btc else mag['strike']
            ax.annotate(f"{mag['total_oi']:,}",
                       (mag_x, max(mag['call_oi'], mag['put_oi'])),
                       textcoords="offset points", xytext=(0, 10),
                       ha='center', fontsize=8, color='#fbbf24', fontweight='bold')

    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#0f0f1a')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.tick_params(colors='#ccc', labelsize=10)

    if is_btc:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax2 = ax.twiny()
        ax2.set_xlim([l * BTC_PER_SHARE for l in ax.get_xlim()])
        ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter('IBIT $%.1f'))
        ax2.tick_params(colors='#666', labelsize=8)
        ax2.spines['top'].set_color('#333')
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    else:
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('$%.1f'))

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    x_label = 'BTC Price' if is_btc else 'Strike'
    ax.set_xlabel(x_label, fontsize=12, color='#ccc', labelpad=10)
    ax.set_ylabel('Open Interest', fontsize=12, color='#ccc', labelpad=10)
    ax.set_title(f'{ticker_symbol} Open Interest Profile  ·  P/C Ratio: {levels["pcr"]:.2f}',
                 fontsize=15, color='white', fontweight='bold', pad=20 if is_btc else 15)

    legend = ax.legend(loc='upper right', fontsize=10, facecolor='#1a1a2e',
                       edgecolor='#444', labelcolor='#ccc')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()


# ── TERMINAL OUTPUT ─────────────────────────────────────────────────────────
def print_summary(spot, levels, selected_exps, exp_details, ticker_symbol, is_btc=False, expected_move=None, df=None, prev_date=None, prev_strikes=None):
    """Clean, glanceable pre-market summary."""
    W = 70
    btc_spot = spot / BTC_PER_SHARE if is_btc else None

    print(f"\n{'━'*W}")
    print(f"  {ticker_symbol} GEX TRADING LEVELS → BTC Reference")
    print(f"  {datetime.now().strftime('%A %B %d, %Y %H:%M')}")
    print(f"{'━'*W}")

    if is_btc:
        print(f"\n  BTC SPOT: ${btc_spot:,.0f}  (IBIT ${spot:.2f})")
        print(f"  BTC/Share: {BTC_PER_SHARE:.6f}")
    else:
        print(f"\n  SPOT:  ${spot:.2f}")

    regime = levels['regime']
    if regime == 'negative_gamma':
        print(f"  REGIME: ▼ NEGATIVE GAMMA — expect amplified moves, trend-following")
        print(f"          Dealers hedge WITH the move → momentum, wider ranges")
    else:
        print(f"  REGIME: ▲ POSITIVE GAMMA — expect dampened moves, mean-reversion")
        print(f"          Dealers hedge AGAINST the move → pinning, tighter ranges")

    if expected_move:
        em = expected_move
        if is_btc:
            btc_lower = em['lower'] / BTC_PER_SHARE
            btc_upper = em['upper'] / BTC_PER_SHARE
            print(f"\n  EXPECTED MOVE ({em['expiration']}, {em['dte']}d):")
            print(f"    Straddle: ±{em['pct']:.1f}%")
            print(f"    BTC Range: ${btc_lower:,.0f} — ${btc_upper:,.0f}")
        else:
            print(f"\n  EXPECTED MOVE ({em['expiration']}, {em['dte']}d):")
            print(f"    Straddle: ${em['straddle']:.2f} ({em['pct']:.1f}%)")
            print(f"    Range:    ${em['lower']:.2f} — ${em['upper']:.2f}")

    print(f"\n  {'─'*60}")
    print(f"  KEY LEVELS")
    print(f"  {'─'*60}")

    def level_str(name, val, extra=""):
        dist = ((val - spot) / spot) * 100
        direction = "▲" if val > spot else "▼"
        if is_btc:
            btc_val = val / BTC_PER_SHARE
            return f"  {name:<22} ${btc_val:<10,.0f} {direction} {abs(dist):>5.1f}%  (IBIT ${val:.2f}){extra}"
        else:
            return f"  {name:<22} ${val:<8.2f}  {direction} {abs(dist):>5.1f}%{extra}"

    print(level_str("Call Wall (resist)", levels['call_wall']))
    for r in levels.get('resistance_levels', [])[1:3]:
        if r != levels['call_wall']:
            print(level_str("  Resistance", r))

    print(level_str("Gamma Flip", levels['gamma_flip'], "  ← KEY PIVOT"))
    shown_flips = {round(levels['gamma_flip'], 0)}
    for gf in sorted(levels.get('all_gamma_flips', []), key=lambda x: abs(x - spot)):
        rounded = round(gf, 0)
        if all(abs(gf - s) > 2.0 for s in shown_flips) and abs(gf - levels['gamma_flip']) > 2.0:
            print(level_str("  Alt Gamma Flip", gf))
            shown_flips.add(rounded)
    print(level_str("Max Pain", levels['max_pain']))

    for s in levels.get('support_levels', [])[1:3]:
        if s != levels['put_wall']:
            print(level_str("  Support", s))
    print(level_str("Put Wall (support)", levels['put_wall']))

    print(f"\n  {'─'*60}")
    print(f"  DEALER POSITIONING")
    print(f"  {'─'*60}")
    nd = levels['net_dealer_delta']
    nd_dir = "LONG" if nd > 0 else "SHORT"
    print(f"  Net Dealer Delta:   {nd:>+12,.0f} shares ({nd_dir})")
    print(f"  Net Dealer Delta $: ${levels['net_dealer_delta_$']:>+12,.0f}")
    print(f"  Net GEX:            ${levels['net_gex_total']:>+12,.0f}")
    print(f"  Put/Call OI Ratio:  {levels['pcr']:>12.2f}")
    print(f"  Total Call OI:      {levels['total_call_oi']:>12,}")
    print(f"  Total Put OI:       {levels['total_put_oi']:>12,}")

    print(f"\n  {'─'*60}")
    print(f"  OI MAGNETS (price gravitates here near expiry)")
    print(f"  {'─'*60}")
    for mag in levels.get('oi_magnets', []):
        pct = ((mag['strike'] - spot) / spot) * 100
        if is_btc:
            btc_val = mag['strike'] / BTC_PER_SHARE
            print(f"  ${btc_val:<10,.0f} OI: {mag['total_oi']:>8,}  "
                  f"(C:{mag['call_oi']:>6,} / P:{mag['put_oi']:>6,})  "
                  f"{'▲' if pct > 0 else '▼'} {abs(pct):.1f}%")
        else:
            print(f"  ${mag['strike']:<8.2f}  OI: {mag['total_oi']:>8,}  "
                  f"(C:{mag['call_oi']:>6,} / P:{mag['put_oi']:>6,})  "
                  f"{'▲' if pct > 0 else '▼'} {abs(pct):.1f}%")

    print(f"\n  {'─'*60}")
    print(f"  EXPIRATIONS INCLUDED ({len(exp_details)})")
    print(f"  {'─'*60}")
    for ed in exp_details[:8]:
        print(f"  {ed['expiration']}  ({ed['dte']:>2}d)  C:{ed['call_oi']:>8,}  P:{ed['put_oi']:>8,}")
    if len(exp_details) > 8:
        print(f"  ... and {len(exp_details) - 8} more")

    print(f"\n{'━'*W}")

    # Trading implications
    print(f"\n  PERPS TRADE SETUP")
    print(f"  {'─'*60}")

    if prev_date and prev_strikes:
        # Calculate aggregate OI changes
        total_oi_now = int(df['total_oi'].sum())
        total_oi_prev = sum(s['total_oi'] for s in prev_strikes.values())
        oi_change = total_oi_now - total_oi_prev
        oi_pct = (oi_change / max(total_oi_prev, 1)) * 100
        
        call_oi_now = int(df['call_oi'].sum())
        put_oi_now = int(df['put_oi'].sum())
        call_oi_prev = sum(s['call_oi'] for s in prev_strikes.values())
        put_oi_prev = sum(s['put_oi'] for s in prev_strikes.values())
        
        call_delta = call_oi_now - call_oi_prev
        put_delta = put_oi_now - put_oi_prev
        
        print(f"  OI CHANGES vs {prev_date}:")
        print(f"    Total: {oi_change:>+10,} ({oi_pct:+.1f}%)  |  "
              f"Calls: {call_delta:>+8,}  |  Puts: {put_delta:>+8,}")
        
        if put_delta > 0 and call_delta < 0:
            print(f"    → Puts building, calls closing = BEARISH positioning")
        elif call_delta > 0 and put_delta < 0:
            print(f"    → Calls building, puts closing = BULLISH positioning")
        elif call_delta > 0 and put_delta > 0:
            print(f"    → Both building = hedging activity increasing")
        elif call_delta < 0 and put_delta < 0:
            print(f"    → Both closing = derisking / expiry runoff")
        print()

    def price_str(ibit_val):
        if is_btc:
            return f"${ibit_val / BTC_PER_SHARE:,.0f}"
        return f"${ibit_val:.2f}"

    gf = levels['gamma_flip']
    cw = levels['call_wall']
    pw = levels['put_wall']

    # Build a ranked list of significant levels with regime-adjusted behavior
    significant_levels = []
    oi_90th = df['total_oi'].quantile(0.90)
    oi_75th = df['total_oi'].quantile(0.75)
    
    for _, row in df.iterrows():
        strike = row['strike']
        if row['total_oi'] < oi_75th:
            continue
        
        net_gex = row['net_gex']
        call_oi = row['call_oi']
        put_oi = row['put_oi']
        total_oi = int(row['total_oi'])
        
        # Only keep levels that are actionable (within ±25% of spot)
        dist_pct = abs((strike - spot) / spot) * 100
        if dist_pct > 25:
            continue
        
        # Determine level type
        if put_oi > call_oi * 1.5 and net_gex < 0:
            level_type = 'put_wall'
        elif call_oi > put_oi * 1.5 and net_gex > 0:
            level_type = 'call_wall'
        elif total_oi > oi_90th:
            level_type = 'oi_magnet'
        else:
            continue
        
        significant_levels.append({
            'strike': strike,
            'btc': strike / BTC_PER_SHARE if is_btc else strike,
            'type': level_type,
            'call_oi': int(call_oi),
            'put_oi': int(put_oi),
            'total_oi': total_oi,
            'net_gex': net_gex,
            'abs_gex': abs(net_gex),
            'dist_pct': ((strike - spot) / spot) * 100,
            'is_major': total_oi > oi_90th,
        })
    
    significant_levels.sort(key=lambda x: x['strike'])

    if regime == 'negative_gamma':
        print(f"  ⚠ NEGATIVE GAMMA — NOT RANGE-BOUND")
        print(f"    Dealers amplify moves → trends extend, ranges break")
        print(f"    Fading extremes is dangerous here.")
        print(f"")
        print(f"    Gamma flip at {price_str(gf)} — reclaiming this = regime shift")
        if spot < gf:
            print(f"    Below flip: bias SHORT, momentum sells accelerate")
        print(f"")
        print(f"  LEVEL BEHAVIOR IN NEGATIVE GAMMA:")
        print(f"  {'─'*60}")
        print(f"  In this regime, dealers hedge WITH the move.")
        print(f"  Put walls produce bounces, NOT reversals.")
        print(f"  Call walls cap briefly, then break if momentum continues.")
        print(f"")

        # Show levels with regime-adjusted notes
        for lv in significant_levels:
            strike = lv['strike']
            dist = lv['dist_pct']
            side = "▲" if dist > 0 else "▼"
            
            if lv['type'] == 'put_wall':
                if lv['is_major']:
                    behavior = "REACTION — scalp long, not conviction"
                    emoji = "⚡"
                else:
                    behavior = "minor support — dealers sell into bounce"
                    emoji = "  "
            elif lv['type'] == 'call_wall':
                if lv['is_major']:
                    behavior = "REACTION — brief cap, watch for breakout"
                    emoji = "⚡"
                else:
                    behavior = "minor resistance — may pause briefly"
                    emoji = "  "
            elif lv['type'] == 'oi_magnet':
                behavior = "OI magnet — gravitational pull near expiry"
                emoji = "🧲"
            
            # OI delta from previous day
            oi_delta_str = ""
            if prev_strikes and strike in prev_strikes:
                prev_oi = prev_strikes[strike]['total_oi']
                delta = lv['total_oi'] - prev_oi
                if delta != 0:
                    pct_chg = (delta / max(prev_oi, 1)) * 100
                    arrow = "↑" if delta > 0 else "↓"
                    strength = "BUILDING" if delta > 0 and abs(pct_chg) > 10 else \
                               "DECAYING" if delta < 0 and abs(pct_chg) > 10 else ""
                    oi_delta_str = f"  {arrow}{abs(delta):,} ({pct_chg:+.0f}%)"
                    if strength:
                        oi_delta_str += f" {strength}"
            
            oi_str = f"OI:{lv['total_oi']:>7,}"
            print(f"  {emoji} {price_str(strike):<12} {side} {abs(dist):>5.1f}%  "
                  f"{oi_str}{oi_delta_str}")
            print(f"     {'':>12} {behavior}")

        print(f"")
        print(f"  WATCH FOR REGIME CHANGE:")
        print(f"    Price reclaims {price_str(gf)} → positive gamma")
        print(f"    Then the range trade activates between walls")

    else:
        # Positive gamma — this is the range-bound setup
        range_low = pw
        range_high = cw
        range_width_pct = ((cw - pw) / spot) * 100

        # GEX strength at walls
        cw_gex = df[df['strike'] == cw]['call_gex'].sum() if cw in df['strike'].values else 0
        pw_gex = abs(df[df['strike'] == pw]['put_gex'].sum()) if pw in df['strike'].values else 0

        dist_to_ceiling = ((cw - spot) / spot) * 100
        dist_to_floor = ((spot - pw) / spot) * 100

        print(f"  ✅ POSITIVE GAMMA — RANGE-BOUND")
        print(f"    Dealers dampen moves → mean-reversion, price pins")
        print(f"    Fade the extremes with leverage.")
        print(f"")
        print(f"  ┌─────────────────────────────────────────────────┐")
        print(f"  │  TRADEABLE RANGE ({range_width_pct:.1f}% wide)                   │")
        print(f"  │                                                 │")
        print(f"  │  CEILING (short zone):  {price_str(cw):<12}           │")
        print(f"  │  {'▔' * 40}         │")

        total_range = cw - pw
        if total_range > 0:
            pos_in_range = (spot - pw) / total_range
            bar_width = 40
            pos_marker = int(pos_in_range * bar_width)
            pos_marker = max(0, min(bar_width - 1, pos_marker))
            bar = '·' * pos_marker + '█' + '·' * (bar_width - pos_marker - 1)
            print(f"  │  {bar}  │")
            print(f"  │  {'':>{pos_marker}}▲ SPOT {price_str(spot):<20}       │")
        
        print(f"  │  {'▁' * 40}         │")
        print(f"  │  FLOOR (long zone):    {price_str(pw):<12}           │")
        print(f"  │                                                 │")
        print(f"  └─────────────────────────────────────────────────┘")
        print(f"")
        print(f"  ENTRIES:")
        print(f"    LONG  near {price_str(pw)} (floor)    ▼ {dist_to_floor:.1f}% below spot")
        print(f"    SHORT near {price_str(cw)} (ceiling)  ▲ {dist_to_ceiling:.1f}% above spot")
        print(f"")
        print(f"  TARGETS:")
        print(f"    Long target  → {price_str(gf)} (gamma flip / midrange)")
        print(f"    Short target → {price_str(gf)} (gamma flip / midrange)")
        print(f"")

        max_gex = max(cw_gex, pw_gex, 1)
        cw_strength = min(cw_gex / max_gex * 100, 100) if max_gex > 0 else 0
        pw_strength = min(pw_gex / max_gex * 100, 100) if max_gex > 0 else 0
        
        def strength_bar(pct):
            filled = int(pct / 10)
            return '█' * filled + '░' * (10 - filled)

        print(f"  WALL STRENGTH (more GEX = harder to break):")
        print(f"    Call wall: {strength_bar(cw_strength)} GEX ${cw_gex:,.0f}")
        print(f"    Put wall:  {strength_bar(pw_strength)} GEX ${pw_gex:,.0f}")
        print(f"")
        print(f"  LEVEL BEHAVIOR IN POSITIVE GAMMA:")
        print(f"  {'─'*60}")
        print(f"  In this regime, dealers hedge AGAINST the move.")
        print(f"  Put walls = hard floors (dealers buy the dip).")
        print(f"  Call walls = hard ceilings (dealers sell the rip).")
        print(f"")

        for lv in significant_levels:
            strike = lv['strike']
            dist = lv['dist_pct']
            side = "▲" if dist > 0 else "▼"
            
            if lv['type'] == 'put_wall':
                if lv['is_major']:
                    behavior = "HARD FLOOR — lever long with confidence"
                    emoji = "🟢"
                else:
                    behavior = "support — dealers buy into weakness"
                    emoji = "🟢"
            elif lv['type'] == 'call_wall':
                if lv['is_major']:
                    behavior = "HARD CEILING — lever short with confidence"
                    emoji = "🔴"
                else:
                    behavior = "resistance — dealers sell into strength"
                    emoji = "🔴"
            elif lv['type'] == 'oi_magnet':
                behavior = "OI magnet — gravitational pull near expiry"
                emoji = "🧲"
            
            # OI delta from previous day
            oi_delta_str = ""
            if prev_strikes and strike in prev_strikes:
                prev_oi = prev_strikes[strike]['total_oi']
                delta = lv['total_oi'] - prev_oi
                if delta != 0:
                    pct_chg = (delta / max(prev_oi, 1)) * 100
                    arrow = "↑" if delta > 0 else "↓"
                    strength = "BUILDING" if delta > 0 and abs(pct_chg) > 10 else \
                               "DECAYING" if delta < 0 and abs(pct_chg) > 10 else ""
                    oi_delta_str = f"  {arrow}{abs(delta):,} ({pct_chg:+.0f}%)"
                    if strength:
                        oi_delta_str += f" {strength}"
            
            oi_str = f"OI:{lv['total_oi']:>7,}"
            print(f"  {emoji} {price_str(strike):<12} {side} {abs(dist):>5.1f}%  "
                  f"{oi_str}{oi_delta_str}")
            print(f"     {'':>12} {behavior}")
        
        print(f"")
        print(f"  INVALIDATION (stop zone):")
        print(f"    Long invalidation:  below {price_str(pw)} → negative gamma, trend down")
        print(f"    Short invalidation: above {price_str(cw)} → breakout, gamma squeeze")
        
        # Inner levels for tighter range / higher leverage
        resistance_above_spot = [r for r in levels.get('resistance_levels', []) 
                                  if r > spot and r != cw and r < cw]
        support_below_spot = [s for s in levels.get('support_levels', [])
                              if s < spot and s != pw and s > pw]
        
        if resistance_above_spot or support_below_spot:
            print(f"")
            print(f"  INNER LEVELS (tighter range for higher leverage):")
            for r in resistance_above_spot[:2]:
                print(f"    Resistance: {price_str(r)}")
            for s in support_below_spot[:2]:
                print(f"    Support:    {price_str(s)}")

    # ── BREAKOUT ASSESSMENT ───────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  BREAKOUT ASSESSMENT")
    print(f"  {'─'*60}")
    
    breakout_signals_up = []
    breakout_signals_down = []
    
    # 1. Wall strength asymmetry
    cw_gex_total = df[df['strike'] == cw]['call_gex'].sum() if cw in df['strike'].values else 0
    pw_gex_total = abs(df[df['strike'] == pw]['put_gex'].sum()) if pw in df['strike'].values else 0
    cw_oi_total = df[df['strike'] == cw]['call_oi'].sum() if cw in df['strike'].values else 0
    pw_oi_total = df[df['strike'] == pw]['put_oi'].sum() if pw in df['strike'].values else 0
    
    if pw_gex_total > 0 and cw_gex_total > 0:
        wall_ratio = cw_gex_total / pw_gex_total
        if wall_ratio < 0.5:
            breakout_signals_up.append(
                f"Weak ceiling: call wall GEX is {wall_ratio:.1f}x put wall "
                f"→ less resistance overhead")
        elif wall_ratio > 2.0:
            breakout_signals_down.append(
                f"Weak floor: put wall GEX is {1/wall_ratio:.1f}x call wall "
                f"→ less support below")
    
    # 2. Wall decay (OI tracking)
    if prev_strikes:
        if cw in prev_strikes:
            cw_prev_oi = prev_strikes[cw].get('total_oi', 0)
            cw_curr_oi = int(df[df['strike'] == cw]['total_oi'].sum()) if cw in df['strike'].values else 0
            if cw_prev_oi > 0:
                cw_oi_chg = ((cw_curr_oi - cw_prev_oi) / cw_prev_oi) * 100
                if cw_oi_chg < -10:
                    breakout_signals_up.append(
                        f"Call wall DECAYING: OI down {cw_oi_chg:.0f}% "
                        f"→ ceiling weakening")
                elif cw_oi_chg > 15:
                    breakout_signals_down.append(
                        f"Call wall BUILDING: OI up +{cw_oi_chg:.0f}% "
                        f"→ ceiling hardening, harder to break up")
        
        if pw in prev_strikes:
            pw_prev_oi = prev_strikes[pw].get('total_oi', 0)
            pw_curr_oi = int(df[df['strike'] == pw]['total_oi'].sum()) if pw in df['strike'].values else 0
            if pw_prev_oi > 0:
                pw_oi_chg = ((pw_curr_oi - pw_prev_oi) / pw_prev_oi) * 100
                if pw_oi_chg < -10:
                    breakout_signals_down.append(
                        f"Put wall DECAYING: OI down {pw_oi_chg:.0f}% "
                        f"→ floor weakening")
                elif pw_oi_chg > 15:
                    breakout_signals_up.append(
                        f"Put wall BUILDING: OI up +{pw_oi_chg:.0f}% "
                        f"→ floor hardening, harder to break down")
    
    # 3. Expected move vs range width
    if expected_move:
        em_width_pct = expected_move['pct'] * 2  # full straddle width (±)
        range_width = ((cw - pw) / spot) * 100
        if em_width_pct > range_width:
            overshot = em_width_pct - range_width
            breakout_signals_up.append(
                f"Expected move ({em_width_pct:.1f}%) > range ({range_width:.1f}%) "
                f"→ market pricing a breakout")
            breakout_signals_down.append(
                f"Expected move ({em_width_pct:.1f}%) > range ({range_width:.1f}%) "
                f"→ market pricing a breakout")
    
    # 4. Negative gamma near wall = gamma squeeze potential
    if regime == 'negative_gamma':
        dist_to_cw = ((cw - spot) / spot) * 100
        dist_to_pw = ((spot - pw) / spot) * 100
        
        if dist_to_cw < 5:
            breakout_signals_up.append(
                f"Negative gamma + only {dist_to_cw:.1f}% from call wall "
                f"→ dealers forced to chase, gamma squeeze potential")
        if dist_to_pw < 5:
            breakout_signals_down.append(
                f"Negative gamma + only {dist_to_pw:.1f}% from put wall "
                f"→ dealers forced to sell, waterfall risk")
    
    # 5. OI buildup beyond walls
    oi_above_cw = df[df['strike'] > cw]['call_oi'].sum()
    oi_below_pw = df[df['strike'] < pw]['put_oi'].sum()
    total_call_oi = df['call_oi'].sum()
    total_put_oi = df['put_oi'].sum()
    
    if total_call_oi > 0:
        pct_above = (oi_above_cw / total_call_oi) * 100
        if pct_above > 40:
            breakout_signals_up.append(
                f"{pct_above:.0f}% of call OI is above call wall "
                f"→ positioning for upside breakout")
    
    if total_put_oi > 0:
        pct_below = (oi_below_pw / total_put_oi) * 100
        if pct_below > 40:
            breakout_signals_down.append(
                f"{pct_below:.0f}% of put OI is below put wall "
                f"→ positioning for downside breakdown")
    
    # 6. Put/call ratio extremes
    pcr = levels.get('pcr', 1.0)
    if pcr > 1.5:
        breakout_signals_down.append(
            f"P/C ratio {pcr:.2f} — heavy put skew, protective positioning "
            f"(can fuel short squeeze if wrong)")
        breakout_signals_up.append(
            f"P/C ratio {pcr:.2f} — extreme put skew → short squeeze fuel "
            f"if sentiment flips")
    elif pcr < 0.6:
        breakout_signals_up.append(
            f"P/C ratio {pcr:.2f} — heavy call skew, aggressive upside bets")
    
    # Score and display
    up_score = len(breakout_signals_up)
    down_score = len(breakout_signals_down)
    
    if up_score == 0 and down_score == 0:
        print(f"  No strong breakout signals — range trade preferred")
    else:
        if up_score > 0:
            bar = '█' * min(up_score, 5) + '░' * max(0, 5 - up_score)
            print(f"  UPSIDE BREAKOUT   [{bar}] {up_score} signal{'s' if up_score != 1 else ''}")
            for sig in breakout_signals_up:
                print(f"    ▲ {sig}")
            
            # Find breakout targets above call wall
            targets_above = df[(df['strike'] > cw) & (df['net_gex'] > 0)].nlargest(2, 'net_gex')
            if not targets_above.empty:
                print(f"    BREAKOUT TARGETS (next resistance above call wall):")
                for _, t in targets_above.iterrows():
                    print(f"      → {price_str(t['strike'])} "
                          f"(OI: {int(t['total_oi']):,}, GEX: ${t['net_gex']:,.0f})")
            print()
        
        if down_score > 0:
            bar = '█' * min(down_score, 5) + '░' * max(0, 5 - down_score)
            print(f"  DOWNSIDE BREAKDOWN [{bar}] {down_score} signal{'s' if down_score != 1 else ''}")
            for sig in breakout_signals_down:
                print(f"    ▼ {sig}")
            
            # Find breakdown targets below put wall
            targets_below = df[(df['strike'] < pw) & (df['net_gex'] < 0)].nsmallest(2, 'net_gex')
            if not targets_below.empty:
                print(f"    BREAKDOWN TARGETS (next support below put wall):")
                for _, t in targets_below.iterrows():
                    print(f"      → {price_str(t['strike'])} "
                          f"(OI: {int(t['total_oi']):,}, GEX: ${t['net_gex']:,.0f})")
            print()
        
        # Net bias
        if up_score > down_score + 1:
            print(f"  → NET BIAS: UPSIDE BREAKOUT more likely ({up_score} vs {down_score} signals)")
        elif down_score > up_score + 1:
            print(f"  → NET BIAS: DOWNSIDE BREAKDOWN more likely ({down_score} vs {up_score} signals)")
        else:
            print(f"  → NET BIAS: BALANCED — watch which wall gets tested first")

    # Max pain note
    mp = levels['max_pain']
    mp_dist = ((mp - spot) / spot) * 100
    if abs(mp_dist) > 2:
        direction = "upward" if mp > spot else "downward"
        print(f"")
        print(f"  💀 Max pain {price_str(mp)} ({mp_dist:+.1f}%) — expiry gravity pulls {direction}")

    # Expected move context
    if expected_move:
        em = expected_move
        if is_btc:
            em_lo = em['lower'] / BTC_PER_SHARE
            em_hi = em['upper'] / BTC_PER_SHARE
        else:
            em_lo = em['lower']
            em_hi = em['upper']
        print(f"  📐 Expected move ({em['dte']}d): ±{em['pct']:.1f}%", end="")
        if is_btc:
            print(f" (${em_lo:,.0f} — ${em_hi:,.0f})")
        else:
            print(f" (${em_lo:.2f} — ${em_hi:.2f})")

    print()


# ── EXPORT ──────────────────────────────────────────────────────────────────
def export_csv(df, levels, spot, ticker_symbol, output_path, is_btc=False):
    """Export key levels for charting platform overlay (TradingView, etc.)."""
    export = df[['strike', 'call_oi', 'put_oi', 'total_oi', 'net_gex',
                 'net_dealer_delta']].copy()
    export = export.sort_values('strike')
    if is_btc:
        export.insert(1, 'btc_price', export['strike'] / BTC_PER_SHARE)
    export.to_csv(output_path, index=False)

    # Also export a simple key-levels file
    levels_path = output_path.replace('.csv', '_levels.csv')
    levels_rows = [
        {'level': 'spot', 'ibit_strike': spot, 'btc_price': spot / BTC_PER_SHARE if is_btc else ''},
        {'level': 'call_wall', 'ibit_strike': levels['call_wall'], 'btc_price': levels['call_wall'] / BTC_PER_SHARE if is_btc else ''},
        {'level': 'put_wall', 'ibit_strike': levels['put_wall'], 'btc_price': levels['put_wall'] / BTC_PER_SHARE if is_btc else ''},
        {'level': 'gamma_flip', 'ibit_strike': levels['gamma_flip'], 'btc_price': levels['gamma_flip'] / BTC_PER_SHARE if is_btc else ''},
        {'level': 'max_pain', 'ibit_strike': levels['max_pain'], 'btc_price': levels['max_pain'] / BTC_PER_SHARE if is_btc else ''},
    ]
    for i, r in enumerate(levels.get('resistance_levels', [])):
        levels_rows.append({'level': f'resistance_{i+1}', 'ibit_strike': r, 'btc_price': r / BTC_PER_SHARE if is_btc else ''})
    for i, s in enumerate(levels.get('support_levels', [])):
        levels_rows.append({'level': f'support_{i+1}', 'ibit_strike': s, 'btc_price': s / BTC_PER_SHARE if is_btc else ''})

    pd.DataFrame(levels_rows).to_csv(levels_path, index=False)
    return levels_path


# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='GEX Trading Dashboard')
    parser.add_argument('--ticker', '-t', default='IBIT', help='Ticker symbol (default: IBIT)')
    parser.add_argument('--dte', '-d', type=int, default=45, help='Max DTE to include (default: 45)')
    parser.add_argument('--output', '-o', default='.', help='Output directory')
    parser.add_argument('--history', action='store_true', help='Show historical trend of key levels')
    parser.add_argument('--no-save', action='store_true', help='Do not save snapshot to history database')
    parser.add_argument('--db', default=DB_PATH, help=f'Database path (default: {DB_PATH})')
    args = parser.parse_args()

    ticker_symbol = args.ticker.upper()
    is_btc = ticker_symbol in ('IBIT', 'BITO', 'GBTC', 'FBTC')

    # Dynamically calculate BTC/Share ratio if possible
    global BTC_PER_SHARE
    if is_btc and ticker_symbol == 'IBIT':
        try:
            btc_ticker = yf.Ticker("BTC-USD")
            btc_price = btc_ticker.info.get('regularMarketPrice')
            ibit_ticker = yf.Ticker("IBIT")
            ibit_price = ibit_ticker.info.get('regularMarketPrice')
            if btc_price and ibit_price and btc_price > 0:
                BTC_PER_SHARE = ibit_price / btc_price
                print(f"  BTC/Share ratio: {BTC_PER_SHARE:.6f} (live: IBIT ${ibit_price:.2f} / BTC ${btc_price:,.0f})")
        except Exception:
            print(f"  BTC/Share ratio: {BTC_PER_SHARE:.6f} (default)")

    print(f"\n  Fetching {ticker_symbol} options data (≤{args.dte} DTE)...")

    spot, strike_data, selected_exps, exp_details = fetch_options_data(ticker_symbol, args.dte)

    if not strike_data:
        print("  ERROR: No options data retrieved. Check ticker or market hours.")
        sys.exit(1)

    # Calculate expected move from nearest expiry ATM straddle
    expected_move = None
    try:
        t = yf.Ticker(ticker_symbol)
        nearest_exp = selected_exps[0]
        chain = t.option_chain(nearest_exp)
        atm_calls = chain.calls.iloc[(chain.calls['strike'] - spot).abs().argsort()[:1]]
        atm_puts = chain.puts.iloc[(chain.puts['strike'] - spot).abs().argsort()[:1]]
        call_mid = (atm_calls['bid'].values[0] + atm_calls['ask'].values[0]) / 2
        put_mid = (atm_puts['bid'].values[0] + atm_puts['ask'].values[0]) / 2
        straddle = call_mid + put_mid
        exp_date = datetime.strptime(nearest_exp, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        dte_nearest = max((exp_date - datetime.now(timezone.utc)).days, 1)
        expected_move = {
            'straddle': straddle,
            'pct': (straddle / spot) * 100,
            'upper': spot + straddle,
            'lower': spot - straddle,
            'expiration': nearest_exp,
            'dte': dte_nearest,
        }
    except Exception:
        pass

    df, levels = analyze(spot, strike_data)

    if df.empty:
        print("  ERROR: Analysis failed — no data after filtering.")
        sys.exit(1)

    # ── HISTORY DATABASE ────────────────────────────────────────────────
    conn = init_db(args.db)
    btc_price = spot / BTC_PER_SHARE if is_btc else None
    
    # Get previous snapshot for OI deltas
    prev_date, prev_summary, prev_strikes = get_previous_snapshot(conn, ticker_symbol)
    
    if prev_date:
        print(f"  📅 Comparing against previous snapshot: {prev_date}")
    else:
        print(f"  📅 No previous snapshot found — run daily to track OI changes")

    # Show history if requested
    if args.history:
        history = get_history(conn, ticker_symbol, days=10)
        if history:
            print(f"\n  {'─'*70}")
            print(f"  HISTORICAL TREND (last {len(history)} snapshots)")
            print(f"  {'─'*70}")
            print(f"  {'Date':<12} {'BTC':>10} {'Regime':<10} {'Gamma Flip':>12} "
                  f"{'Call Wall':>12} {'Put Wall':>12} {'Net GEX':>14}")
            print(f"  {'─'*70}")
            for h in history:
                h_date, h_spot, h_btc, h_gf, h_cw, h_pw, h_mp, h_regime, h_gex, h_coi, h_poi = h
                regime_icon = "▲+" if h_regime == 'positive_gamma' else "▼−"
                if is_btc and h_btc:
                    print(f"  {h_date:<12} ${h_btc:>9,.0f} {regime_icon:<10} "
                          f"${h_gf/BTC_PER_SHARE if h_gf else 0:>11,.0f} "
                          f"${h_cw/BTC_PER_SHARE if h_cw else 0:>11,.0f} "
                          f"${h_pw/BTC_PER_SHARE if h_pw else 0:>11,.0f} "
                          f"${h_gex:>13,.0f}" if h_gex else "")
                else:
                    print(f"  {h_date:<12} ${h_spot:>9.2f} {regime_icon:<10} "
                          f"${h_gf:>11.2f} ${h_cw:>11.2f} ${h_pw:>11.2f} "
                          f"${h_gex:>13,.0f}" if h_gex else "")
            
            # Show key level drift
            if len(history) >= 2:
                latest = history[0]
                oldest = history[-1]
                _, _, _, gf_now, cw_now, pw_now, _, regime_now, _, _, _ = latest
                _, _, _, gf_old, cw_old, pw_old, _, regime_old, _, _, _ = oldest
                
                print(f"\n  {len(history)}-DAY DRIFT:")
                if gf_now and gf_old and is_btc:
                    gf_drift = (gf_now - gf_old) / BTC_PER_SHARE
                    cw_drift = (cw_now - cw_old) / BTC_PER_SHARE if cw_now and cw_old else 0
                    pw_drift = (pw_now - pw_old) / BTC_PER_SHARE if pw_now and pw_old else 0
                    print(f"    Gamma flip: {'↑' if gf_drift > 0 else '↓'} ${abs(gf_drift):,.0f}")
                    print(f"    Call wall:  {'↑' if cw_drift > 0 else '↓'} ${abs(cw_drift):,.0f}")
                    print(f"    Put wall:   {'↑' if pw_drift > 0 else '↓'} ${abs(pw_drift):,.0f}")
                
                if regime_now != regime_old:
                    print(f"    ⚠ REGIME CHANGED: {regime_old} → {regime_now}")
        else:
            print(f"\n  No history yet. Run daily to build trend data.")

    # Save today's snapshot
    if not args.no_save:
        save_snapshot(conn, ticker_symbol, spot, btc_price, levels, df)
        print(f"  💾 Snapshot saved to {args.db}")
    
    conn.close()

    # Terminal output
    print_summary(spot, levels, selected_exps, exp_details, ticker_symbol, is_btc, 
                  expected_move, df, prev_date, prev_strikes)

    # Charts
    gex_chart_path = f"{args.output}/{ticker_symbol}_gex_profile.png"
    oi_chart_path = f"{args.output}/{ticker_symbol}_oi_profile.png"
    csv_path = f"{args.output}/{ticker_symbol}_gex_data.csv"

    plot_gex_profile(df, spot, levels, ticker_symbol, gex_chart_path, expected_move, is_btc)
    plot_oi_profile(df, spot, levels, ticker_symbol, oi_chart_path, is_btc)
    levels_path = export_csv(df, levels, spot, ticker_symbol, csv_path, is_btc)

    print(f"  📊 GEX Chart:   {gex_chart_path}")
    print(f"  📊 OI Chart:    {oi_chart_path}")
    print(f"  📄 Full Data:   {csv_path}")
    print(f"  📄 Key Levels:  {levels_path}")
    print()


if __name__ == "__main__":
    main()