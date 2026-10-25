# -*- coding: utf-8 -*-
"""Metricas per-ticker y WLS para el universo de holdings.

Extraido de stock_leader.py (C4.1). Produce:
- compute_stock_metrics: metricas por ticker de un sector/indice.
- compute_wls: agrega las metricas por sector en un WLS score.
- compute_all_stock_metrics: aplica lo anterior a TODOS los
  holdings y devuelve el DataFrame agregado (antes
  full_metrics_df).

No depende de la seleccion de lideres. La seleccion vive en
stock_leader.py::generate_leader_section y se retira en C4.3.
"""
import pandas as pd
import numpy as np

from indicators.wyckoff_v1 import wyckoff_score, wyckoff_stability, detect_spring, detect_sos
from indicators.wyckoff_orchestrator import classify_phase
from src.utils import robust_zscore, get_col
from config.settings import TOP_N_CANDIDATES
from config.weights import WLS_WEIGHTS, WLS_PERSISTENCE_FACTOR


# ============================================================

WLS_COMPONENTS = ('rs_z', 'flow_proxy_z_norm', 'rws_z', 'stab_z')

def _validate_wls_inputs(df):
    """Valida el contrato de WLS.

    Returns:
        (persistence_clipped, status_series, warnings)
        - persistence_clipped: persistencia recortada a [0, 1].
        - status_series: 'OK' | 'PERSISTENCE_OUT_OF_RANGE' |
          'INSUFFICIENT_COMPONENTS'.
        - warnings: lista de strings con avisos (vacia si todo OK).
    """
    warnings = []
    pers = pd.to_numeric(df['persistence_10d'], errors='coerce')
    out_of_range = pers.notna() & ((pers < 0.0) | (pers > 1.0))
    if bool(out_of_range.any()):
        bad = pers[out_of_range]
        warnings.append(
            f"persistence_10d fuera de [0,1] en {int(out_of_range.sum())} "
            f"filas (min={bad.min()}, max={bad.max()}). Clipeado."
        )
    pers_clipped = pers.clip(0.0, 1.0)

    components_missing = df[list(WLS_COMPONENTS)].isna().any(axis=1)
    status = pd.Series('OK', index=df.index)
    status[out_of_range] = 'PERSISTENCE_OUT_OF_RANGE'
    status[components_missing] = 'INSUFFICIENT_COMPONENTS'
    return pers_clipped, status, warnings


def compute_stock_metrics(df_market, df_stocks, etf_ticker, stock_list, temporal_meta=None):
    results = []
    # FU-021-5 Fase 5.3a (P0): fecha contractual para columnas temporales.
    from src.utils import get_effective_meta
    _eff_meta = get_effective_meta(temporal_meta or {}, ['EQUITY_EOD'])
    price_etf = get_col(df_market, etf_ticker, 'Close')

    for ticker in stock_list:
        try:
            close = get_col(df_stocks, ticker, 'Close')
            volume = get_col(df_stocks, ticker, 'Volume')
        except KeyError:
            continue

        if len(close.dropna()) < 60:
            continue

        # Alinear índices para evitar NaN por diferencias de calendario
        common_idx = close.index.intersection(price_etf.index)
        # Fix 2026-10-06: rs puede tener NaN residuales por festivos
        # USA (Labor Day, Thanksgiving, MLK, Presidents, etc.).
        # diff(20) los propaga al ultimo valor -> rs_mom = NaN.
        # dropna() + guard len >= 21 para que sea robusto.
        rs = (close.loc[common_idx] / price_etf.loc[common_idx]).dropna()
        # H2 (2026-10-15, auditoria externa): se verifico empiricamente
        # que rs_mom es INVARIANTE a la escala del precio (splits,
        # precio nominal por accion). Razon matematica:
        #   rs_mom = log(P_t / P_{t-20}) - log(E_t / E_{t-20})
        # Cualquier factor k sobre P se cancela: log(k*P_t) - log(k*P_{t-20})
        # = log(P_t) - log(P_{t-20}). Por tanto, el rs_z que alimenta el
        # WLS (via robust_zscore intra-sector de rs_mom) tambien es
        # invariante a la escala. El problema del rs nominal documentado
        # en N2 afecta solo a LIS (slpm_v12.py::compute_leader_integrity),
        # no al WLS. Verificado: NVDA con split 10:1 simulado produce
        # rs_mom identico (diferencia < 1e-12).
        rs_mom = np.log(rs).diff(20).iloc[-1] if len(rs) >= 21 else np.nan

        # P1.1 (2026-10-13): retorno relativo 20d (correcto para
        # criterio binario de liderazgo). rs (precio accion / precio
        # ETF) depende de precio nominal y splits: no es comparable
        # entre acciones. rs_ratio_20d SI lo es.
        # Fórmula: (1+ret20_stock)/(1+ret20_etf) - 1, mismo criterio
        # que rs_internal.rs_abs_20d.
        try:
            _close = close.loc[common_idx].dropna()
            _etf = price_etf.loc[common_idx].dropna()
            _idx = _close.index.intersection(_etf.index)
            if len(_idx) >= 21:
                _ret_stk = _close.loc[_idx].iloc[-1] / _close.loc[_idx].iloc[-21] - 1
                _ret_etf = _etf.loc[_idx].iloc[-1] / _etf.loc[_idx].iloc[-21] - 1
                rs_ratio_20d = ((1 + _ret_stk) / (1 + _ret_etf) - 1) if pd.notna(_ret_stk) and pd.notna(_ret_etf) else np.nan
            else:
                rs_ratio_20d = np.nan
        except (KeyError, IndexError, ValueError, ZeroDivisionError):
            rs_ratio_20d = np.nan

        ret = close.pct_change(fill_method=None)
        dollar_vol = close * volume
        flow_raw = ret * dollar_vol
        flow_proxy_z = robust_zscore(flow_raw, window=60)
        flow_signal = flow_proxy_z.ewm(span=5).mean().iloc[-1]

        # Fix 2026-10-15 (H7): rolling sobre serie dropna.
        # ret_positive tiene NaN donde ret es NaN (huecos USA/EU).
        # rolling(N, mp=N) sobre el indice completo devuelve todo NaN.
        ret_positive = (ret > 0).astype(int).where(ret.notna())
        _ret_pos_clean = ret_positive.dropna()
        if len(_ret_pos_clean) >= 20:
            persistence_10d = _ret_pos_clean.rolling(10, min_periods=10).mean().iloc[-1]
            persistence_5d = _ret_pos_clean.rolling(5, min_periods=5).mean().iloc[-1]
            persistence_20d = _ret_pos_clean.rolling(20, min_periods=20).mean().iloc[-1]
        else:
            persistence_10d = np.nan
            persistence_5d = np.nan
            persistence_20d = np.nan
        if pd.isna(persistence_10d):
            persistence_10d = 0.5

        try:
            ticker_df = pd.DataFrame({
                'Open': get_col(df_stocks, ticker, 'Open'),
                'High': get_col(df_stocks, ticker, 'High'),
                'Low': get_col(df_stocks, ticker, 'Low'),
                'Close': close,
                'Volume': volume
            }).dropna()
        except KeyError:
            continue

        if len(ticker_df) >= 60:
            wyckoff_series, struct_score, tact_score, t_n, c_n, v_n, e_n = wyckoff_score(ticker_df, ticker)
            wyckoff_sc = wyckoff_series.iloc[-1]
            wyckoff_ph = classify_phase(ticker_df, ticker).phase
            # H-06 (auditoria externa 2026-10-25): usar wyckoff_stability
            # (definicion correcta: 1 - 2*tanh(MAD(combined)/(K+eps))) en
            # lugar de tanh(median/MAD). La version previa:
            #   (a) incluia el signo de la mediana (convicion direccional,
            #       no estabilidad). corr(stab_sm, t_norm) = +0.49 vs
            #       corr(stab_v1, t_norm) = +0.16.
            #   (b) saturaba al 36% de tickers en +-1.0 (mad+1e-9 explota
            #       cuando med=0).
            #   (c) 43% de signos opuestos vs wyckoff_stability sobre
            #       muestra de 108 tickers.
            # El peso se llama WLS_WEIGHTS['stability']. Debe medir
            # estabilidad, no conviccion. Doc 68.
            _stab_series = wyckoff_stability(wyckoff_series).dropna()
            stability = float(_stab_series.iloc[-1]) if len(_stab_series) > 0 else 0.0
            spring = detect_spring(ticker_df, ticker).iloc[-1]
            sos = detect_sos(ticker_df, ticker).iloc[-1]
        else:
            wyckoff_sc = np.nan
            wyckoff_ph = 'INSUFFICIENT_DATA'
            stability = 0.0
            spring = 0
            sos = 0

        results.append({
            'ticker': ticker,
            'effective_date': _eff_meta.get('effective_date'),
            'expected_date': _eff_meta.get('expected_date'),
            'coverage': _eff_meta.get('coverage'),
            'rs': rs.iloc[-1],
            'rs_mom': rs_mom,
            'rs_ratio_20d': rs_ratio_20d,
            'flow_proxy_z': flow_signal,
            'wyckoff_score': wyckoff_sc,
            'wyckoff_phase': wyckoff_ph,
            'persistence_5d': persistence_5d,
            'persistence_10d': persistence_10d,
            'persistence_20d': persistence_20d,
            'stability': stability,
            'spring': spring,
            'sos': sos,
        })

    return pd.DataFrame(results)

def compute_wls(df_metrics, weights=None):
    df = df_metrics.copy()
    if df.empty:
        return df

    def robust_intra(s):
        if s.empty or s.isna().all():
            return pd.Series(0.0, index=s.index)
        median = s.median()
        mad = (s - median).abs().median()
        if pd.isna(mad) or mad == 0:
            return pd.Series(0.0, index=s.index)
        return (s - median) / (1.4826 * mad + 1e-9)

    df['rs_z'] = df.groupby('sector')['rs_mom'].transform(robust_intra).clip(-3, 3)
    df['flow_proxy_z_norm'] = df.groupby('sector')['flow_proxy_z'].transform(robust_intra).clip(-3, 3)

    # M8: robust_zscore intra-sectorial en lugar de percentil 70
    df['rws_z'] = df.groupby('sector')['wyckoff_score'].transform(robust_intra).clip(-3, 3)

    df['stab_z'] = df.groupby('sector')['stability'].transform(robust_intra).clip(-3, 3)

    for sector in df['sector'].unique():
        mask = df['sector'] == sector
        if mask.sum() < 3:
            continue
        rho = df.loc[mask, 'rs_mom'].corr(df.loc[mask, 'flow_proxy_z'], method='spearman')
        if pd.notna(rho) and rho > 0.7:
            factor = 1 - min(1.0, (rho - 0.7) / 0.3)
            df.loc[mask, 'flow_proxy_z_norm'] *= factor

    # P1.3 (2026-10-13, auditor): los pesos vienen de config.weights
    # (WLS_WEIGHTS). El parametro `weights` permite override explicito
    # (usado por tests), pero los defaults son la fuente unica.
    if weights is None:
        _w = WLS_WEIGHTS
    else:
        _w = weights
    w_rs = _w.get('rs', WLS_WEIGHTS['rs'])
    w_flow = _w.get('flow', WLS_WEIGHTS['flow'])
    w_struct = _w.get('structure', _w.get('wyckoff', WLS_WEIGHTS['wyckoff']))
    w_stab = _w.get('stability', WLS_WEIGHTS['stability'])

    # N3 (2026-10-13, auditor): validacion extraida a funcion pura.
    _pers_clipped, _status, _warns = _validate_wls_inputs(df)
    for _w in _warns:
        print(f"  [WARN] WLS: {_w}")

    df['wls'] = (w_rs*df['rs_z'] + w_flow*df['flow_proxy_z_norm']
                 + w_struct*df['rws_z'] + w_stab*df['stab_z'])
    df['wls'] *= (1 + WLS_PERSISTENCE_FACTOR * _pers_clipped)
    df['wls_status'] = _status

    df['sector_rank_pct'] = df.groupby('sector')['wls'].rank(pct=True)

    # leader_confidence eliminado (no se usaba)
    for sector in df['sector'].unique():
        mask = df['sector'] == sector
        sector_df = df[mask]
        if len(sector_df) >= 20:
            current_rank = sector_df['wls'].rank()
            historical_rank = sector_df['wls'].iloc[:-20].rank()
            common_idx = current_rank.index.intersection(historical_rank.index)
            if len(common_idx) > 5:
                rho = current_rank.loc[common_idx].corr(historical_rank.loc[common_idx], method='spearman')
                # leader_confidence eliminado

    return df.sort_values('wls', ascending=False)


def compute_all_stock_metrics(df_market, df_stocks, holdings_df, temporal_meta=None):
    """Aplica compute_stock_metrics + compute_wls a TODOS los holdings.

    Sustituye la produccion de `full_metrics_df` que hacia
    stock_leader.py::generate_leader_section sin filtrar por fase.

    Returns:
        DataFrame concatenado de WLS por sector. Vacio si no hay datos.
    """
    if holdings_df is None or holdings_df.empty:
        return pd.DataFrame()
    if df_stocks is None or df_stocks.empty:
        return pd.DataFrame()

    full_metrics_data = []
    for sector_etf, group in holdings_df.groupby("etf"):
        sector_holdings = group.copy()
        if "weight" in sector_holdings.columns:
            sector_holdings = sector_holdings.sort_values("weight", ascending=False)
        stocks = sector_holdings.head(TOP_N_CANDIDATES)["ticker"].tolist()
        if not stocks:
            continue
        try:
            metrics_df = compute_stock_metrics(
                df_market, df_stocks, sector_etf, stocks, temporal_meta=temporal_meta,
            )
        except (KeyError, ValueError, TypeError, IndexError, AttributeError, OSError, RuntimeError):
            continue
        if metrics_df is None or metrics_df.empty:
            continue
        metrics_df["sector"] = sector_etf
        try:
            wls_df = compute_wls(metrics_df)
        except (KeyError, ValueError, TypeError, IndexError, AttributeError):
            continue
        if wls_df is None or wls_df.empty:
            continue
        full_metrics_data.append(wls_df)

    if not full_metrics_data:
        return pd.DataFrame()
    return pd.concat(full_metrics_data, ignore_index=True)
