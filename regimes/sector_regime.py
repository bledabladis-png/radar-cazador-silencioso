import pandas as pd

from config.tickers import SECTOR_NAMES, CYCLICAL_SECTORS, DEFENSIVE_SECTORS, MARKET_TICKERS
from config.weights import SECTOR_SCORE_WEIGHTS, SECTOR_DISPERSION_PENALTY
from indicators.momentum import compute_returns, compute_flow_proxy, compute_price_momentum
from indicators.trend import trend_position
from indicators.volatility import atr
from indicators.breadth import compute_breadth
from indicators.wyckoff import wyckoff_structure_core
from src.utils import safe_mean, safe_std, tanh_normalize, get_col

def compute_sector_scores(df, benchmark='^GSPC', df_stocks=None, holdings_df=None):
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
        # Amplitud interna sectorial si hay datos de stocks y holdings
        # Amplitud sectorial interna basada en EMAs del propio sector
        ema20_sector = close_sector.ewm(span=20, adjust=False, min_periods=20).mean()
        ema50_sector = close_sector.ewm(span=50, adjust=False, min_periods=50).mean()
        ema200_sector = close_sector.ewm(span=200, adjust=False, min_periods=200).mean()
        breadth_sector = ((close_sector > ema20_sector).astype(float) +
                          (close_sector > ema50_sector).astype(float) +
                          (close_sector > ema200_sector).astype(float)) / 3.0
        comp_breadth = tanh_normalize(breadth_sector)



        comp_rs20 = tanh_normalize(mom20)
        comp_rs50 = tanh_normalize(mom50)
        comp_rs126 = tanh_normalize(mom126)
        comp_trend = trend
        comp_vol = vol_inv


        idx = comp_rs20.index
        comp_dict = {
            'rs_mom_20': comp_rs20,
            'rs_mom_50': comp_rs50,
            'rs_mom_126': comp_rs126,
            'trend': comp_trend,
            'volatility_inv': comp_vol,
            'breadth': comp_breadth if isinstance(comp_breadth, pd.Series) else pd.Series(comp_breadth, index=idx),
        }
        comp_df = pd.DataFrame({k: (v if isinstance(v, pd.Series) else pd.Series(v, index=idx)).reindex(idx) for k, v in comp_dict.items()})

        weights = pd.Series(SECTOR_SCORE_WEIGHTS)
        mask = comp_df.notna()
        weighted_sum = comp_df.mul(weights, axis=1).sum(axis=1)
        valid_weight_sum = mask.mul(weights, axis=1).sum(axis=1)
        score_series = weighted_sum / valid_weight_sum.replace(0, float('nan'))

        def _dispersion(row):
            vals = row.dropna().tolist()
            if len(vals) < 2:
                return 0.0
            return safe_std(vals) / (abs(safe_mean(vals)) + 1e-9)

        dispersion_series = comp_df.apply(_dispersion, axis=1)
        penalty_series = (1 - SECTOR_DISPERSION_PENALTY * dispersion_series).clip(lower=0)
        scores[sector] = score_series * penalty_series


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


