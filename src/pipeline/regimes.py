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
from src.market_calendar import last_expected_market_date


def compute_real_liquidity(reference_date=None):
    """Descarga FRED + NY Fed y evalua v2. Contrato: 4-tuple v1.

    reference_date (opcional): se propaga a _fetch_liquidity_v2_data
    para que la query a NY Fed sea determinista respecto al run.
    """
    _sofr, _effr, _c1 = _fetch_liquidity_v2_data(reference_date=reference_date)
    return compute_liquidity_score_v2(_sofr, _effr, _c1)


def _fetch_liquidity_v2_data(reference_date=None):
    """Descarga SOFR, EFFR y p99 NY Fed. CSV-first con fallback a API.

    reference_date (opcional): fecha de ejecucion del run. Si se pasa,
    se usa para calcular `endDate` de la query a NY Fed via
    last_expected_market_date. Antes se usaba pd.Timestamp.now().date()
    directo: re-ejecutar otro dia con los mismos CSVs daba otro
    `endDate`, rompiendo el determinismo dado un snapshot. Con
    reference_date, dos runs con el mismo `reference_date` producen
    el mismo `endDate`.

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
    start = "2018-04-02"
    sofr = None
    effr = None

    # Fase 6 (2026-10-26): unificacion via router. La politica CSV-first
    # sobre data/macro_manual vive ahora en
    # data.providers.router.DataRouter.get_fed_data(columns=...), que
    # gestiona frescura (bot diario) y fallback a la API FRED. Elimina
    # la doble implementacion que existia aqui.
    try:
        from data.providers.router import DataRouter
        _df = DataRouter().get_fed_data(columns=["sofr", "effr"])
        if _df is not None and not _df.empty and "sofr" in _df.columns and "effr" in _df.columns:
            _df = _df[["sofr", "effr"]].dropna()
            _df = _df[_df.index >= pd.Timestamp(start)]
            if len(_df) > 0:
                sofr = _df["sofr"].sort_index()
                effr = _df["effr"].sort_index()
                print(f"  [liq_v2] router: sofr+effr hasta {sofr.index.max().date()} ({len(sofr)} fechas comunes)")
    except Exception as e:
        print(f"  [WARN] liq_v2 router: {e}")

    if sofr is None or effr is None:
        return None, None, None

    # 3) p99 NY Fed (opcional, solo diagnostico C1)
    c1_bp = None
    try:
        import requests
        url = "https://markets.newyorkfed.org/api/rates/secured/sofr/search.json"
        if reference_date is not None:
            try:
                _end = last_expected_market_date(reference_date).strftime("%Y-%m-%d")
            except (TypeError, ValueError):
                _end = str(pd.Timestamp.now().date())
        else:
            _end = str(pd.Timestamp.now().date())
        params = {"startDate": start, "endDate": _end, "type": "rate,volume"}
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
    real_liq_score, real_liq_regime, real_liq_conf, real_liq_prev = compute_real_liquidity(reference_date=reference_date)
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
