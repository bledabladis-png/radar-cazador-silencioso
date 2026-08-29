import pandas as pd
import numpy as np
from config.tickers import SECTOR_NAMES, CYCLICAL_SECTORS, DEFENSIVE_SECTORS, MARKET_TICKERS
from config.weights import SECTOR_SCORE_WEIGHTS, SECTOR_DISPERSION_PENALTY
from indicators.momentum import compute_returns, compute_flow_proxy, compute_price_momentum
from indicators.trend import trend_position
from indicators.volatility import atr
from indicators.breadth import compute_breadth
from indicators.wyckoff import wyckoff_structure_core
from src.utils import tanh_normalize, get_col

def compute_sector_scores(df, benchmark='^GSPC'):
    sectors = MARKET_TICKERS['sectors']
    returns = compute_returns(df, sectors + [benchmark])
    if returns.empty:
        return None

    scores = pd.DataFrame(index=returns.index)
    wyckoff_phases = {}

    try:
        bench_close = get_col(df, benchmark, 'Close')
    except KeyError:
        return None

    for sector in sectors:
        try:
            close_sector = get_col(df, sector, 'Close')
        except KeyError:
            continue
        rs = close_sector / bench_close
        rs_ret = rs.pct_change(fill_method=None)

        mom20 = rs_ret.rolling(20).mean() / (rs_ret.rolling(20).std() + 1e-9)
        mom50 = rs_ret.rolling(50).mean() / (rs_ret.rolling(50).std() + 1e-9)
        mom126 = rs_ret.rolling(126).mean() / (rs_ret.rolling(126).std() + 1e-9)
        trend = trend_position(close_sector)
        atr_val = atr(df, sector)
        vol_inv = -tanh_normalize(atr_val)
        _, breadth_50, _, _, _ = compute_breadth(df)

        comp_rs20 = tanh_normalize(mom20).fillna(0)
        comp_rs50 = tanh_normalize(mom50).fillna(0)
        comp_rs126 = tanh_normalize(mom126).fillna(0)
        comp_trend = trend.fillna(0)
        comp_vol = vol_inv.fillna(0)
        comp_breadth = breadth_50.fillna(0) if not breadth_50.empty else 0

        scores[sector] = (
            SECTOR_SCORE_WEIGHTS['rs_mom_20'] * comp_rs20 +
            SECTOR_SCORE_WEIGHTS['rs_mom_50'] * comp_rs50 +
            SECTOR_SCORE_WEIGHTS['rs_mom_126'] * comp_rs126 +
            SECTOR_SCORE_WEIGHTS['trend'] * comp_trend +
            SECTOR_SCORE_WEIGHTS['volatility_inv'] * comp_vol +
            SECTOR_SCORE_WEIGHTS['breadth'] * comp_breadth
        )

        # Penalización por desacuerdo entre sub-componentes
        sub_components = [comp_rs20.iloc[-1], comp_rs50.iloc[-1], comp_rs126.iloc[-1], comp_trend.iloc[-1], comp_vol.iloc[-1]]
        if not isinstance(comp_breadth, (int, float)):
            comp_breadth_val = comp_breadth.iloc[-1] if not comp_breadth.empty else 0
        else:
            comp_breadth_val = comp_breadth
        sub_components = [comp_rs20.iloc[-1], comp_rs50.iloc[-1], comp_rs126.iloc[-1], comp_trend.iloc[-1], comp_vol.iloc[-1], comp_breadth_val]
        dispersion = np.std(sub_components) / (np.abs(np.mean(sub_components)) + 1e-9)
        penalty = max(0, 1 - SECTOR_DISPERSION_PENALTY * dispersion)
        scores[sector] *= penalty

        try:
            wyckoff_phases[sector] = wyckoff_structure_core(df, sector)
        except Exception as e:
            print(f"Wyckoff error en {sector}: {e}")
            wyckoff_phases[sector] = "N/A"

    if scores.empty:
        return None

    last_scores = scores.iloc[-1].sort_values(ascending=False)
    ranking = [(sector, SECTOR_NAMES[sector], last_scores[sector], wyckoff_phases.get(sector, ""))
               for sector in last_scores.index]

    top3 = last_scores.head(3).index
    cyclical_leadership = any(s in CYCLICAL_SECTORS for s in top3)
    defensive_leadership = any(s in DEFENSIVE_SECTORS for s in top3)
    positive_count = (last_scores > 0).sum()

    if positive_count >= 8:
        regime = 'BROAD PARTICIPATION'
    elif positive_count <= 3:
        regime = 'NARROW RALLY'
    elif 4 <= positive_count <= 7:
        regime = 'ROTATIONAL'
    elif cyclical_leadership and not defensive_leadership:
        regime = 'CYCLICAL LEADERSHIP'
    elif defensive_leadership and not cyclical_leadership:
        regime = 'DEFENSIVE LEADERSHIP'
    else:
        regime = 'MIXED'

    
    # Sub-componentes del último día para validación
    components = {}
    for sector in sectors:
        try:
            components[sector] = {
                'rs_mom_20': float(comp_rs20.iloc[-1]) if hasattr(comp_rs20, 'iloc') else float(comp_rs20),
                'rs_mom_50': float(comp_rs50.iloc[-1]) if hasattr(comp_rs50, 'iloc') else float(comp_rs50),
                'rs_mom_126': float(comp_rs126.iloc[-1]) if hasattr(comp_rs126, 'iloc') else float(comp_rs126),
                'trend': float(comp_trend.iloc[-1]) if hasattr(comp_trend, 'iloc') else float(comp_trend),
                'volatility_inv': float(comp_vol.iloc[-1]) if hasattr(comp_vol, 'iloc') else float(comp_vol),
                'breadth': float(comp_breadth.iloc[-1]) if hasattr(comp_breadth, 'iloc') else float(comp_breadth)
            }
        except Exception:
            components[sector] = {}

    return {'scores': scores, 'ranking': ranking, 'regime': regime, 'last_scores': last_scores, 'components': components}


def compute_price_flow_rankings(df):
    # Sectores (11 ETFs)
    sectors = ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
    # Otros activos
    otros = {
        'Indices': ['^GSPC', '^NDX', '^RUT', '^STOXX50E', 'EEM', 'EWJ'],
        'Bonos': ['BIL', 'IEF', 'TLT'],
        'Credito': ['HYG', 'LQD'],
        'Factores': ['VLUE', 'MTUM', 'QUAL'],
        'Small Caps Intl': ['SCHC', 'EWX'],
        'Bonos Emergentes': ['EMB', 'ELD'],
        'Materias Primas': ['^SPGSCI', 'GC=F', 'HG=F', 'CL=F', 'BZ=F', 'NG=F'],
        'Divisas': ['DX-Y.NYB', 'EURUSD=X', 'USDJPY=X', 'USDCNY=X'],
    }

    # Calcular para sectores
    sector_price = {}
    sector_flow = {}
    for t in sectors:
        try:
            mom = compute_price_momentum(df, t, window=20).iloc[-1]
            if pd.notna(mom):
                sector_price[t] = mom
        except Exception:
            pass
        try:
            flow = compute_flow_proxy(df, t).iloc[-1]
            if pd.notna(flow):
                sector_flow[t] = flow
        except Exception:
            pass

    # Calcular para otros activos
    otros_price = {}
    otros_flow = {}
    for cat, tickers in otros.items():
        for t in tickers:
            try:
                mom = compute_price_momentum(df, t, window=20).iloc[-1]
                if pd.notna(mom):
                    otros_price[t] = mom
            except Exception:
                pass
            try:
                flow = compute_flow_proxy(df, t).iloc[-1]
                if pd.notna(flow):
                    otros_flow[t] = flow
            except Exception:
                pass

    # Retornar 4 listas ordenadas
    return (
        sorted(sector_price.items(), key=lambda x: x[1], reverse=True),
        sorted(sector_flow.items(), key=lambda x: x[1], reverse=True),
        sorted(otros_price.items(), key=lambda x: x[1], reverse=True),
        sorted(otros_flow.items(), key=lambda x: x[1], reverse=True)
    )


