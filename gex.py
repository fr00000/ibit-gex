"""GEX computation engine — pure functions for gamma exposure analysis.

Black-Scholes Greeks, Deribit option processing, level computation,
flow forecasting, dealer delta scenarios, breakout detection, and
significant level identification. No shared state or database access.
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import norm

log = logging.getLogger(__name__)

STRIKE_RANGE_PCT = 0.35


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


# ── DERIBIT UTILITIES ───────────────────────────────────────────────────────
def compute_atm_iv(iv_entries, reference_price):
    """Compute OI-weighted ATM IV from a list of IV entries.

    Each entry: {'strike': float, 'iv': float, 'oi': int, 'opt_type': str, ...}
    reference_price: spot or underlying price to define 'ATM' (within 2%).

    Returns dict with atm_iv, call_iv, put_iv, num_entries, or None if insufficient data.
    """
    if not iv_entries:
        return None

    atm_entries = [e for e in iv_entries if abs(e['strike'] - reference_price) / reference_price < 0.02]
    if not atm_entries:
        sorted_by_dist = sorted(iv_entries, key=lambda e: abs(e['strike'] - reference_price))
        atm_entries = sorted_by_dist[:2]
    if not atm_entries:
        return None

    total_oi = sum(e['oi'] for e in atm_entries)
    if total_oi <= 0:
        return None

    atm_iv = sum(e['iv'] * e['oi'] for e in atm_entries) / total_oi

    atm_calls = [e for e in atm_entries if e['opt_type'] == 'call']
    atm_puts = [e for e in atm_entries if e['opt_type'] == 'put']
    call_oi_sum = sum(e['oi'] for e in atm_calls)
    put_oi_sum = sum(e['oi'] for e in atm_puts)
    call_iv = sum(e['iv'] * e['oi'] for e in atm_calls) / call_oi_sum if atm_calls and call_oi_sum > 0 else None
    put_iv = sum(e['iv'] * e['oi'] for e in atm_puts) / put_oi_sum if atm_puts and put_oi_sum > 0 else None

    return {
        'atm_iv': round(atm_iv, 4),
        'call_iv': round(call_iv, 4) if call_iv else None,
        'put_iv': round(put_iv, 4) if put_iv else None,
        'num_entries': len(atm_entries),
    }


def compute_deribit_iv_term(iv_data):
    """Compute Deribit ATM IV term structure from per-expiry IV data.

    Args:
        iv_data: {exp_str: [{strike, iv, oi, opt_type, dte, underlying}, ...]}

    Returns list of {'expiry', 'dte', 'atm_iv', 'source': 'deribit'} dicts.
    """
    term = []
    for exp_str, iv_entries in sorted(iv_data.items()):
        if not iv_entries:
            continue
        dte_val = iv_entries[0]['dte']
        underlying = iv_entries[0]['underlying']
        result = compute_atm_iv(iv_entries, underlying)
        if not result:
            continue
        term.append({
            'expiry': exp_str,
            'dte': round(dte_val, 1),
            'atm_iv': result['atm_iv'],
            'source': 'deribit',
        })
    return term


def process_deribit_options(options, rfr, full_greeks=False):
    """Process raw Deribit options into strike-level aggregations.

    Args:
        options: list of dicts from fetch_deribit_options()
        rfr: risk-free rate
        full_greeks: if True, compute delta/vanna/charm and per-expiry strike data
                     (weekday path). If False, compute only GEX + OI (weekend overlay).

    Returns:
        strike_data: {strike_btc: {call_gex, put_gex, call_oi, put_oi, ...}}
        iv_data: {exp_str: [{strike, iv, oi, opt_type, dte, underlying}, ...]}
        total_oi_btc: sum of all OI in BTC contracts
        expiry_strike_data: {exp_str: {strike_btc: {...}}} (empty dict when full_greeks=False)
    """
    strike_data = {}
    iv_data = {}
    expiry_strike_data = {}
    total_oi_btc = 0

    for opt in options:
        strike_btc = opt['strike']
        oi = opt['oi']
        iv = opt['iv'] / 100.0
        btc_s = opt['underlying_price']
        T = max(opt['dte'] / 365.0, 0.5 / 365)
        opt_type = opt['option_type']
        sign = 1 if opt_type == 'call' else -1

        gamma = bs_gamma(btc_s, strike_btc, T, rfr, iv)
        gex = sign * gamma * oi * btc_s ** 2 * 0.01
        total_oi_btc += oi

        # Initialize strike entry
        if strike_btc not in strike_data:
            fields = {'call_gex': 0, 'put_gex': 0, 'call_oi': 0, 'put_oi': 0}
            if full_greeks:
                fields.update({
                    'call_delta': 0, 'put_delta': 0,
                    'call_vanna': 0, 'put_vanna': 0,
                    'call_charm': 0, 'put_charm': 0,
                })
            strike_data[strike_btc] = fields

        d_entry = strike_data[strike_btc]
        d_entry[f'{opt_type}_oi'] += oi
        d_entry[f'{opt_type}_gex'] += gex

        if full_greeks:
            delta = bs_delta(btc_s, strike_btc, T, rfr, iv, opt_type)
            vanna = bs_vanna(btc_s, strike_btc, T, rfr, iv)
            charm = bs_charm(btc_s, strike_btc, T, rfr, iv, opt_type)
            dealer_delta = -delta * oi * btc_s
            dealer_vanna = -vanna * oi * btc_s * 0.01
            dealer_charm = -charm * oi / 365.0 * btc_s
            d_entry[f'{opt_type}_delta'] += dealer_delta
            d_entry[f'{opt_type}_vanna'] += dealer_vanna
            d_entry[f'{opt_type}_charm'] += dealer_charm

            # Per-expiry strike data (for expiry cache)
            exp_str = opt['expiry_date'].strftime('%Y-%m-%d')
            if exp_str not in expiry_strike_data:
                expiry_strike_data[exp_str] = {}
            desd = expiry_strike_data[exp_str]
            if strike_btc not in desd:
                desd[strike_btc] = {
                    'call_oi': 0, 'put_oi': 0,
                    'call_gex': 0, 'put_gex': 0,
                    'call_delta': 0, 'put_delta': 0,
                    'call_vanna': 0, 'put_vanna': 0,
                    'call_charm': 0, 'put_charm': 0,
                    'call_iv_sum': 0, 'call_iv_oi': 0,
                    'put_iv_sum': 0, 'put_iv_oi': 0,
                    'dte': int(opt['dte']),
                }
            desd[strike_btc][f'{opt_type}_oi'] += oi
            desd[strike_btc][f'{opt_type}_gex'] += gex
            desd[strike_btc][f'{opt_type}_delta'] += dealer_delta
            desd[strike_btc][f'{opt_type}_vanna'] += dealer_vanna
            desd[strike_btc][f'{opt_type}_charm'] += dealer_charm
            desd[strike_btc][f'{opt_type}_iv_sum'] += iv * oi
            desd[strike_btc][f'{opt_type}_iv_oi'] += oi

        # Collect IV for term structure
        exp_str = opt['expiry_date'].strftime('%Y-%m-%d')
        if exp_str not in iv_data:
            iv_data[exp_str] = []
        iv_data[exp_str].append({
            'strike': strike_btc, 'iv': iv,
            'oi': oi, 'opt_type': opt_type,
            'dte': opt['dte'], 'underlying': btc_s,
        })

    return strike_data, iv_data, total_oi_btc, expiry_strike_data


def build_deribit_df(strike_data, full_greeks=False):
    """Build a pandas DataFrame from Deribit strike_data dict.

    When full_greeks=True, includes delta/vanna/charm and extra columns.
    When False, fills required columns with zeros for _compute_levels_from_df."""
    rows = []
    for strike_btc, d in sorted(strike_data.items()):
        row = {
            'strike': strike_btc,
            'call_gex': d['call_gex'], 'put_gex': d['put_gex'],
            'net_gex': d['call_gex'] + d['put_gex'],
            'call_oi': d['call_oi'], 'put_oi': d['put_oi'],
            'total_oi': d['call_oi'] + d['put_oi'],
            'active_gex': d['call_gex'] + d['put_gex'],
        }
        if full_greeks:
            row.update({
                'btc_price': strike_btc,
                'net_dealer_delta': d['call_delta'] + d['put_delta'],
                'net_vanna': d['call_vanna'] + d['put_vanna'],
                'net_charm': d['call_charm'] + d['put_charm'],
                'call_vanna': d['call_vanna'], 'put_vanna': d['put_vanna'],
                'call_charm': d['call_charm'], 'put_charm': d['put_charm'],
                'call_volume': 0, 'put_volume': 0, 'total_volume': 0,
                'expiry_gex': {},
            })
        else:
            row.update({
                'net_dealer_delta': 0, 'net_vanna': 0, 'net_charm': 0,
            })
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── LEVEL COMPUTATION ───────────────────────────────────────────────────────
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

    # Volume-weighted GEX: what dealers are DOING today vs what they HAVE
    if 'call_vol_gex' in df.columns:
        levels['volume_gex_total'] = float(df['call_vol_gex'].sum() + df['put_vol_gex'].sum())
        oi_gex = levels['net_gex_total']
        vol_gex = levels['volume_gex_total']
        levels['gex_activity_ratio'] = round(abs(vol_gex / oi_gex), 2) if oi_gex != 0 else 0.0
    else:
        levels['volume_gex_total'] = 0.0
        levels['gex_activity_ratio'] = 0.0

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


# ── FLOW & DEALER ANALYSIS ─────────────────────────────────────────────────
def compute_flow_forecast(df, spot, levels, is_crypto):
    """Compute dealer flow forecasts from vanna and charm exposures.
    net_vanna and net_charm in df are already in dollar notional."""
    net_vanna = float(df['net_vanna'].sum())
    net_charm = float(df['net_charm'].sum())
    regime = levels['regime']

    # Charm: overnight dealer rebalancing (dollar notional)
    # Positive net charm = dealers need to BUY delta tomorrow (bullish flow)
    # Negative net charm = dealers need to SELL delta tomorrow (bearish flow)
    charm_direction = 'buy' if net_charm > 0 else 'sell'
    charm_magnitude = abs(net_charm)

    # Categorize charm impact (dollar notional thresholds)
    if charm_magnitude < 50_000:
        charm_strength = 'negligible'
    elif charm_magnitude < 500_000:
        charm_strength = 'minor'
    elif charm_magnitude < 2_500_000:
        charm_strength = 'moderate'
    else:
        charm_strength = 'significant'

    # Vanna: IV-dependent dealer rebalancing (dollar notional)
    # Positive net vanna = vol CRUSH forces dealers to BUY (bullish)
    # Negative net vanna = vol CRUSH forces dealers to SELL (bearish)
    # (reverse for vol spike)
    vanna_magnitude = abs(net_vanna)
    if vanna_magnitude < 50_000:
        vanna_strength = 'negligible'
    elif vanna_magnitude < 500_000:
        vanna_strength = 'minor'
    elif vanna_magnitude < 2_500_000:
        vanna_strength = 'moderate'
    else:
        vanna_strength = 'significant'

    # Scenarios: 5-point IV move (result is dollar notional)
    iv_move = 5  # vol points
    vanna_crush_notional = net_vanna * iv_move  # from -5pt IV
    vanna_spike_notional = -net_vanna * iv_move  # from +5pt IV

    # Combined overnight forecast (charm + expected vanna from typical overnight vol decay)
    # Overnight vol typically drifts down ~0.5-1pt in calm markets
    overnight_vanna_adj = net_vanna * 0.5  # conservative overnight vol decay
    overnight_total = net_charm + overnight_vanna_adj
    overnight_direction = 'buy' if overnight_total > 0 else 'sell'

    # Narrative
    charm_narrative = f"Dealers {charm_direction} ~${abs(net_charm):,.0f} notional overnight from delta decay"

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
            'net_notional': float(net_charm),
            'direction': charm_direction,
            'strength': charm_strength,
            'narrative': charm_narrative,
        },
        'vanna': {
            'net_notional': float(net_vanna),
            'strength': vanna_strength,
            'crush_scenario': {
                'notional': float(vanna_crush_notional),
                'direction': 'buy' if vanna_crush_notional > 0 else 'sell',
                'narrative': vanna_crush_narrative,
            },
            'spike_scenario': {
                'notional': float(vanna_spike_notional),
                'direction': 'buy' if vanna_spike_notional > 0 else 'sell',
                'narrative': vanna_spike_narrative,
            },
        },
        'overnight': {
            'net_notional': float(overnight_total),
            'direction': overnight_direction,
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
            T = max((exp_date - now).total_seconds() / 86400.0 / 365.0, 0.5 / 365)
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
                    dd = -delta * oi * 100 * S_hyp  # dollar notional
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
                dd_d = -delta_d * opt['oi'] * 1 * S_hyp_btc  # dollar notional
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
    cur_abs = abs(cur_delta)
    if cur_abs >= 1e6:
        cur_str = f"~${cur_abs/1e6:.0f}M notional"
    else:
        cur_str = f"~${cur_abs/1e3:.0f}K notional"
    if cur_delta < 0:
        current_delta = f"Dealers are net SHORT {cur_str} — must BUY on dips (supportive)"
    else:
        current_delta = f"Dealers are net LONG {cur_str} — must SELL on rallies (resistive)"

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
    notional = abs(cur_delta)  # already dollar notional
    if notional >= 1e6:
        notional_str = f"~${notional/1e6:.0f}M notional"
    else:
        notional_str = f"~${notional/1e3:.0f}K notional"
    if cur_delta < 0:
        parts.append(f"Dealers are net short {notional_str} at spot, creating a bid under the market")
    else:
        parts.append(f"Dealers are net long {notional_str} at spot, creating selling pressure")
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


def compute_positioning_depth(gex_chart_data, dealer_delta_profile, spot_btc, levels_btc):
    """Compute positioning depth metrics:
    1. Dealer delta cushion: how much dealers must buy/sell at -5%, -8%, +5%, +8% from spot
    2. Put density profile: near-spot (0-5% below) vs far (5-10% below) put OI distribution
    3. Call overwrite zone: largest call OI concentration above spot

    These quantify the mechanical floor/ceiling structure for the AI analysis."""

    result = {}

    # --- 1. Dealer delta cushion ---
    if dealer_delta_profile and len(dealer_delta_profile) > 1:
        # Find dealer delta at spot
        spot_entry = min(dealer_delta_profile, key=lambda p: abs(p['price_btc'] - spot_btc))
        dd_at_spot = spot_entry['net_dealer_delta']

        cushion = {'at_spot': round(dd_at_spot)}
        for pct in [-8, -5, 5, 8]:
            target = spot_btc * (1 + pct / 100)
            nearest = min(dealer_delta_profile, key=lambda p: abs(p['price_btc'] - target))
            dd_at_target = nearest['net_dealer_delta']
            swing = dd_at_target - dd_at_spot
            cushion[f"{'down' if pct < 0 else 'up'}_{abs(pct)}pct"] = {
                'price_btc': round(nearest['price_btc']),
                'dealer_delta': round(dd_at_target),
                'swing_from_spot': round(swing),
            }
        result['dealer_delta_cushion'] = cushion

    # --- 2. Put density profile ---
    if gex_chart_data and spot_btc:
        near_put_oi = 0   # 0-5% below spot
        far_put_oi = 0    # 5-10% below spot
        deep_put_oi = 0   # 10%+ below spot
        above_put_oi = 0  # above spot (ITM puts / synthetic shorts)

        for entry in gex_chart_data:
            btc = entry.get('btc', 0)
            put_oi = entry.get('put_oi', 0)
            if not btc or not put_oi:
                continue
            pct_below = (spot_btc - btc) / spot_btc * 100
            if pct_below < 0:
                above_put_oi += put_oi
            elif pct_below <= 5:
                near_put_oi += put_oi
            elif pct_below <= 10:
                far_put_oi += put_oi
            else:
                deep_put_oi += put_oi

        total_put = near_put_oi + far_put_oi + deep_put_oi + above_put_oi
        near_far_ratio = round(near_put_oi / far_put_oi, 2) if far_put_oi > 0 else None

        # Interpretation: high ratio = support clustered near spot (absorbed on dips)
        # low ratio = OI spread deep (acceleration risk, "cleaned up" near spot)
        if near_far_ratio is not None:
            if near_far_ratio > 2.0:
                structure = 'concentrated_near'
                description = 'Put OI concentrated near spot — dips likely absorbed by dealer hedging'
            elif near_far_ratio < 0.5:
                structure = 'cleaned_up'
                description = 'Near-spot puts cleaned up — thin cushion, but a slide picks up large hedging flows deeper'
            else:
                structure = 'distributed'
                description = 'Put OI spread across range — gradual dealer hedging on moves down'
        else:
            structure = 'minimal'
            description = 'Minimal put OI below spot'

        result['put_density'] = {
            'near_0_5pct': near_put_oi,
            'far_5_10pct': far_put_oi,
            'deep_10pct_plus': deep_put_oi,
            'above_spot': above_put_oi,
            'total': total_put,
            'near_far_ratio': near_far_ratio,
            'structure': structure,
            'description': description,
        }

    # --- 3. Call overwrite zone ---
    if gex_chart_data and spot_btc:
        # Find the strike with highest call OI above spot (likely overwrite zone)
        above_spot = [e for e in gex_chart_data if e.get('btc', 0) > spot_btc and e.get('call_oi', 0) > 0]

        if above_spot:
            # Sort by call OI descending, take top 3
            top_strikes = sorted(above_spot, key=lambda e: e.get('call_oi', 0), reverse=True)[:3]
            total_call_above = sum(e.get('call_oi', 0) for e in above_spot)

            # Check volume-to-OI ratio for top strike (low ratio = likely overwriting, not speculative)
            top = top_strikes[0]
            top_volume = top.get('call_volume', 0)
            top_oi = top.get('call_oi', 0)
            vol_oi_ratio = round(top_volume / top_oi, 2) if top_oi > 0 else None

            # Concentration: what % of above-spot call OI is in top 3 strikes
            top3_oi = sum(s.get('call_oi', 0) for s in top_strikes)
            concentration = round(top3_oi / total_call_above * 100, 1) if total_call_above > 0 else 0

            result['call_overwrite_zone'] = {
                'top_strike_btc': round(top.get('btc', 0)),
                'top_strike_oi': top_oi,
                'top_strike_pct_above': round((top.get('btc', 0) - spot_btc) / spot_btc * 100, 1),
                'top_strike_vol_oi_ratio': vol_oi_ratio,
                'top3': [
                    {
                        'strike_btc': round(s.get('btc', 0)),
                        'call_oi': s.get('call_oi', 0),
                        'pct_above': round((s.get('btc', 0) - spot_btc) / spot_btc * 100, 1),
                    }
                    for s in top_strikes
                ],
                'top3_concentration_pct': concentration,
                'total_call_oi_above': total_call_above,
            }

    return result


# ── SIGNIFICANT LEVELS & BREAKOUT ───────────────────────────────────────────
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
                behavior = "acceleration zone — dealers sell + gamma amplifies" if is_major else "minor acceleration zone"
            elif ltype == 'call_wall':
                behavior = "mechanical resistance — dealers sell + gamma stabilizes" if is_major else "minor resistance"
            else:
                behavior = "OI magnet — gravitational pull near expiry"
        else:
            if ltype == 'put_wall':
                behavior = "mechanical support — dealers buy + gamma stabilizes" if is_major else "minor support"
            elif ltype == 'call_wall':
                behavior = "mechanical resistance — dealers sell + gamma stabilizes" if is_major else "minor resistance"
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


def compute_breakout(df, spot, levels, expected_move, prev_strikes, ref_per_share,
                     etf_flows=None, flow_forecast=None, deribit_levels_btc=None,
                     btc_per_share=None, btc_spot=None):
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

    # Wall decay: only meaningful if strike was significant yesterday (not migration)
    if prev_strikes:
        oi_80 = df['total_oi'].quantile(0.80) if not df.empty else 0
        if cw in prev_strikes:
            prev_total = prev_strikes[cw]['total_oi']
            curr_total = int(df[df['strike'] == cw]['total_oi'].sum()) if cw in df['strike'].values else 0
            if prev_total > 0 and prev_total > oi_80:
                chg = ((curr_total - prev_total) / prev_total) * 100
                if chg < -20:
                    up_signals.append(f"Call wall DECAYING: OI down {chg:.0f}%")
                elif chg > 25:
                    down_signals.append(f"Call wall BUILDING: OI up +{chg:.0f}%")
        if pw in prev_strikes:
            prev_total = prev_strikes[pw]['total_oi']
            curr_total = int(df[df['strike'] == pw]['total_oi'].sum()) if pw in df['strike'].values else 0
            if prev_total > 0 and prev_total > oi_80:
                chg = ((curr_total - prev_total) / prev_total) * 100
                if chg < -20:
                    down_signals.append(f"Put wall DECAYING: OI down {chg:.0f}%")
                elif chg > 25:
                    up_signals.append(f"Put wall BUILDING: OI up +{chg:.0f}%")

    # Expected move vs range (non-directional — tracked separately)
    em_note = None
    if expected_move and cw > pw:
        em_width = expected_move['pct'] * 2
        range_width = ((cw - pw) / spot) * 100
        if range_width > 0.5 and em_width > range_width:
            em_note = f"Expected move ({em_width:.1f}%) > range ({range_width:.1f}%) — breakout likely"

    # Neg gamma near wall — threshold scaled to range width, not fixed 5%
    if regime == 'negative_gamma' and cw > pw:
        range_pct = (cw - pw) / spot
        near_thresh = max(range_pct * 0.3, 0.005)  # at least 0.5%, scaled to range
        if ((cw - spot) / spot) < near_thresh:
            up_signals.append("Negative gamma near call wall — gamma squeeze potential")
        if ((spot - pw) / spot) < near_thresh:
            down_signals.append("Negative gamma near put wall — waterfall risk")

    # OI beyond walls — raised threshold from 40% to 65%
    total_call = df['call_oi'].sum()
    total_put = df['put_oi'].sum()
    above = df[df['strike'] > cw]['call_oi'].sum()
    below = df[df['strike'] < pw]['put_oi'].sum()
    if total_call > 0 and (above / total_call) * 100 > 65:
        up_signals.append(f"{(above/total_call)*100:.0f}% of call OI above call wall")
    if total_put > 0 and (below / total_put) * 100 > 65:
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

    # Venue wall convergence: IBIT + Deribit agreement strengthens wall, divergence shows room
    if deribit_levels_btc and btc_per_share and btc_spot:
        ibit_cw_btc = cw / btc_per_share if btc_per_share else 0
        ibit_pw_btc = pw / btc_per_share if btc_per_share else 0
        deribit_cw = deribit_levels_btc.get('call_wall', 0)
        deribit_pw = deribit_levels_btc.get('put_wall', 0)

        if deribit_cw and ibit_cw_btc:
            cw_diff_pct = abs(deribit_cw - ibit_cw_btc) / ibit_cw_btc * 100
            if cw_diff_pct < 2:
                down_signals.append(f"Venue-reinforced call wall (IBIT+Deribit within {cw_diff_pct:.1f}%)")
            elif cw_diff_pct > 5 and deribit_cw > ibit_cw_btc:
                up_signals.append(f"Deribit call wall ${deribit_cw:,.0f} above IBIT — room to run")

        if deribit_pw and ibit_pw_btc:
            pw_diff_pct = abs(deribit_pw - ibit_pw_btc) / ibit_pw_btc * 100
            if pw_diff_pct < 2:
                up_signals.append(f"Venue-reinforced put wall (IBIT+Deribit within {pw_diff_pct:.1f}%)")
            elif pw_diff_pct > 5 and deribit_pw < ibit_pw_btc:
                down_signals.append(f"Deribit put wall ${deribit_pw:,.0f} below IBIT — deeper downside")

    # Overnight charm flow: significant dealer rebalancing gives directional pressure
    if flow_forecast:
        charm = flow_forecast.get('charm', {})
        charm_str = charm.get('strength', 'negligible')
        charm_dir = charm.get('direction')
        if charm_str in ('moderate', 'significant'):
            if charm_dir == 'buy':
                up_signals.append(f"Overnight charm flow: dealers BUY ${abs(charm.get('net_notional',0))/1e6:.0f}M")
            elif charm_dir == 'sell':
                down_signals.append(f"Overnight charm flow: dealers SELL ${abs(charm.get('net_notional',0))/1e6:.0f}M")

    # Targets
    up_targets = df[(df['strike'] > cw) & (df['net_gex'] > 0)].nlargest(2, 'net_gex')[
        ['strike', 'total_oi', 'net_gex']].to_dict('records')
    down_targets = df[(df['strike'] < pw) & (df['net_gex'] < 0)].nsmallest(2, 'net_gex')[
        ['strike', 'total_oi', 'net_gex']].to_dict('records')

    up_score = len(up_signals)
    down_score = len(down_signals)
    if up_score >= down_score + 2:
        bias = 'upside'
    elif down_score >= up_score + 2:
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
