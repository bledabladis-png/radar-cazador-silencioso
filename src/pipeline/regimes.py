# -*- coding: utf-8 -*-
"""Fase 2 del pipeline: regimenes (Financial + Liquidity + Volatility + Macro).

Extraido de run.py (refactor C2, fase C2-4).
"""

import pandas as pd

from regimes.financial_conditions import compute_financial_conditions
# v1 mantenido para rollback (no se invoca). Para revertir: cambiar
# compute_real_liquidity a compute_liquidity_score (v1).
# v2 activo (dictamen N8, sustituye al score v1 que daba senales erroneas).
from regimes.liquidity_v2 import compute_liquidity_score_v2
from regimes.volatility_regime import compute_volatility_regime
from regimes.macro_regime import compute_macro_regime
from src.utils import get_col


def compute_real_liquidity():
    """Descarga FRED + NY Fed y evalua v2. Contrato: 4-tuple v1."""
    _sofr, _effr, _c1 = _fetch_liquidity_v2_data()
    return compute_liquidity_score_v2(_sofr, _effr, _c1)


def _fetch_liquidity_v2_data():
    """Descarga SOFR, EFFR y p99 NY Fed. CSV-first con fallback a API.

    2026-10-22 (REP-AUDIT-20261022-LIQ-V2-DATA-SOURCE): antes solo
    usaba pandas_datareader (FRED API). La API va 1-2 dias por detras
    de los CSVs data/macro_manual/ (pusheados por bot diario). El
    state v2 quedaba con date=2026-10-19 mientras sofr.csv tenia
    2026-10-20. Ahora:
      - Si sofr.csv y effr.csv existen y tienen datos -> CSV-first.
      - Interseccion de indices SOFR/EFFR para no mezclar fechas.
      - p99 NY Fed (opcional, diagnostico C1) sigue via API.
      - Si CSVs faltan o fallan -> fallback a pandas_datareader.
    Devuelve (sofr, effr, c1_bp) o (None, None, None).
    """
    from pathlib import Path as _P

    start = "2018-04-02"
    sofr = None
    effr = None

    # 1) CSV-first
    sofr_csv = _P("data/macro_manual/sofr.csv")
    effr_csv = _P("data/macro_manual/effr.csv")
    if sofr_csv.exists() and effr_csv.exists():
        try:
            sofr_df = pd.read_csv(sofr_csv, index_col=0, parse_dates=True)
            effr_df = pd.read_csv(effr_csv, index_col=0, parse_dates=True)
            if not sofr_df.empty and not effr_df.empty:
                s = sofr_df.iloc[:, 0].dropna().sort_index()
                e = effr_df.iloc[:, 0].dropna().sort_index()
                # Interseccion: solo fechas donde ambas series tienen dato
                common = s.index.intersection(e.index)
                if len(common) > 0:
                    sofr = s.loc[common]
                    effr = e.loc[common]
                    print(f"  [liq_v2] CSVs: sofr+effr hasta {common.max().date()} ({len(common)} fechas comunes)")
        except Exception as e:
            print(f"  [WARN] liq_v2 CSV load: {e}")

    # 2) Fallback: pandas_datareader (API FRED)
    if sofr is None or effr is None:
        try:
            import pandas_datareader.data as web
        except ImportError:
            return None, None, None
        try:
            sofr_df = web.DataReader("SOFR", "fred", start=start)
            sofr = sofr_df.iloc[:, 0].sort_index().dropna()
        except Exception as e:
            print(f"  [WARN] SOFR API: {e}")
            return None, None, None
        try:
            effr_df = web.DataReader("EFFR", "fred", start=start)
            effr = effr_df.iloc[:, 0].sort_index().dropna()
        except Exception as e:
            print(f"  [WARN] EFFR API: {e}")
            return None, None, None
        print(f"  [liq_v2] API FRED: hasta {effr.index.max().date()}")

    # 3) p99 NY Fed (opcional, solo diagnostico C1)
    c1_bp = None
    try:
        import requests
        url = "https://markets.newyorkfed.org/api/rates/secured/sofr/search.json"
        params = {"startDate": start, "endDate": str(pd.Timestamp.now().date()), "type": "rate,volume"}
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 200:
            refs = r.json().get("refRates", [])
            def _num(x):
                if x is None or x == "NA" or x == "":
                    return float("nan")
                try:
                    return float(x)
                except (ValueError, TypeError):
                    return float("nan")
            p99_rows = []
            for ref in refs:
                p99_rows.append({
                    "date": pd.Timestamp(ref.get("effectiveDate")),
                    "p99": _num(ref.get("percentPercentile99")),
                })
            if p99_rows:
                p99_df = pd.DataFrame(p99_rows).set_index("date").sort_index()
                idx = p99_df.index.intersection(effr.index)
                c1_bp = (p99_df.loc[idx, "p99"] - effr.loc[idx]) * 100.0
    except Exception as e:
        print(f"  [WARN] NY Fed p99: {e}")

    return sofr, effr, c1_bp
def compute_all_regimes(df_market, df_macro_manual, temporal_meta=None, reference_date=None):
    """Calcula los 4 regimenes del sistema.

    Returns:
        dict con keys:
            financial_score, financial_regime, liq_conf,
            real_liq_score, real_liq_regime, real_liq_conf, real_liq_prev,
            vol_score, vol_regime, vol_conf,
            macro_score, macro_regime, macro_conf, all_signals
    """
    print("Calculando regimen de Cond. Financieras...")
    financial_score, financial_regime, liq_conf = compute_financial_conditions(df_market)
    print(f"  Cond. Financieras: {financial_regime} (conf: {liq_conf:.0%})")

    print("Calculando liquidez real (FRED, v2)...")
    # N8 (2026-10-14): sustituye a liquidity.py::compute_liquidity_score (v1).
    # v1 daba senales erroneas por: MAD nula sobre series planas, doble ffill
    # sobre FRED, ausencia de guard de frescura. v2 implementa la especificacion
    # B3 congelada (dictamen N8 Fase J).
    # Rollback: cambiar a _compute_real_liquidity_v1().
    real_liq_score, real_liq_regime, real_liq_conf, real_liq_prev = compute_real_liquidity()
    if real_liq_score is None:
        real_liq_regime, real_liq_conf = 'N/A', 0.0
        print("  Liquidez real: no disponible (sin datos FRED)")
    else:
        print(f"  Liquidez real: {real_liq_regime} (conf: {real_liq_conf:.0%})")

    print("Calculando regimen de volatilidad...")
    try:
        vix_close = get_col(df_market, '^VIX', 'Close')
        vix_returns = vix_close.pct_change(fill_method=None)
    except KeyError:
        print("  ^VIX no disponible, usando volatilidad plana.")
        vix_returns = pd.Series(dtype=float)

    vol_score, vol_regime, vol_conf = compute_volatility_regime(vix_returns)
    print(f"  Volatilidad: {vol_regime} (conf: {vol_conf:.0%})")

    print("Calculando regimen macro...")
    macro_score, macro_regime, macro_conf, all_signals = compute_macro_regime(
        df_market, df_macro_manual, financial_score, vol_score,
        temporal_meta=temporal_meta,
        reference_date=reference_date,
    )
    print(f"  Macro: {macro_regime} (conf: {macro_conf:.0%})")

    return {
        'financial_score': financial_score,
        'financial_regime': financial_regime,
        'liq_conf': liq_conf,
        'real_liq_score': real_liq_score,
        'real_liq_regime': real_liq_regime,
        'real_liq_conf': real_liq_conf,
        'real_liq_prev': real_liq_prev,
        'vol_score': vol_score,
        'vol_regime': vol_regime,
        'vol_conf': vol_conf,
        'macro_score': macro_score,
        'macro_regime': macro_regime,
        'macro_conf': macro_conf,
        'all_signals': all_signals,
    }
