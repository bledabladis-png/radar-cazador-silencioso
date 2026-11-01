"""Scoring de MTE (DT2 Fase 4).

Extraccion literal de mte_legacy.py. Sin cambios funcionales.

Contiene:
- tanh, _get_last: helpers compartidos (usados por compute_mte en legacy)
- sector_rotation_score, safe_haven_score, credit_stress_score, inflation_pressure_score
- compute_msi, compute_ipi
- SCENARIO_WEIGHTS + score_scenarios
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import robust_zscore, get_col
from config.tickers import CYCLICAL_SECTORS, DEFENSIVE_SECTORS


# F2.4-05 (2026-10-13): WONT FIX razonado. Wrapper trivial
# sobre np.tanh. Se mantiene por legibilidad del modulo MTE
# (evita import directo de numpy en el cuerpo de cada funcion)
# y por permitir sustitucion futura (clip, escalado) sin tocar
# los callers. No es dead code.
def tanh(x):
    return np.tanh(x)


def _get_last(x):


    """Extrae el último valor de una Series o float."""


    if isinstance(x, pd.Series):


        return x.iloc[-1]


    return x


# ============================================================


# 1. SECTOR ROTATION SCORE (SRS)


# ============================================================


def sector_rotation_score(df_market):


    # MTE-SECTOR-LISTS-DIVERGENTES (2026-10-31): antes hardcodeado.
    # v3.15 (2026-10-20) reclasifico XLC como ciclico en
    # config/tickers.py, pero el MTE mantenia la lista vieja
    # (XLC defensivo). Divergencia real desde v3.15. Alineado a
    # config: fuente unica de verdad. Test centinela en
    # tests/test_mte_scoring.py verifica consistencia.
    cyclical = list(CYCLICAL_SECTORS)
    defensive = list(DEFENSIVE_SECTORS)


    sectors = cyclical + defensive


    rs = {}


    for s in sectors:


        try:


            close_s = get_col(df_market, s, 'Close')


            close_spy = get_col(df_market, '^GSPC', 'Close')


            rs[s] = close_s / close_spy


        except KeyError:


            continue


    if not rs:


        return 0.0


    rs_cyclical = pd.concat([rs[s] for s in cyclical if s in rs], axis=1).mean(axis=1)


    rs_defensive = pd.concat([rs[s] for s in defensive if s in rs], axis=1).mean(axis=1)


    spread = rs_defensive - rs_cyclical


    mom_spread = spread.pct_change(20, fill_method=None)


    speed_spread = (spread - spread.shift(20)).abs()


    # Dispersión cross-sectional


    rs_all = pd.concat([rs[s] for s in sectors if s in rs], axis=1)


    dispersion = rs_all.std(axis=1)


    # Amplitud interna: % de sectores defensivos que baten al SPY


    try:


        spy_close = get_col(df_market, '^GSPC', 'Close')


        spy_mom = spy_close.pct_change(20, fill_method=None)


        defensive_mom = pd.concat([rs[s].pct_change(20, fill_method=None) for s in defensive if s in rs], axis=1)


        defensive_beating_spy = (defensive_mom.gt(spy_mom, axis=0)).mean(axis=1)


    except (KeyError, ValueError, TypeError, AttributeError):


        defensive_beating_spy = pd.Series(0.5, index=df_market.index)


    z_spread   = robust_zscore(spread, 60)


    z_mom      = robust_zscore(mom_spread, 60)


    z_speed    = robust_zscore(speed_spread, 60)


    z_disp     = robust_zscore(dispersion, 60)


    z_breadth_def = robust_zscore(defensive_beating_spy, 60)


    srs = (0.35 * tanh(z_spread) +


           0.20 * tanh(z_mom) +


           0.15 * tanh(z_speed) +


           0.15 * tanh(z_disp) +


           0.15 * tanh(z_breadth_def))


    return _get_last(srs)


# ============================================================


# 2. SAFE HAVEN SCORE (SHS)


# ============================================================


# MTE-SECTOR-LISTS-DIVERGENTES (2026-10-31): constantes nombradas
# para listas propias del MTE (no en config). Evita literales sueltos.
HARD_HAVENS = ['GLD', 'SLV', 'TLT']
DEFENSIVE_HAVENS = ['XLP', 'XLV', 'XLU', 'QUAL', 'IEF', 'BIL']


def safe_haven_score(df_market):


    hard_havens = HARD_HAVENS
    defensive_havens = DEFENSIVE_HAVENS


    def haven_subscore(tickers):


        signals = []


        for t in tickers:


            try:


                close = get_col(df_market, t, 'Close')


                mom = close.pct_change(20, fill_method=None)


                signals.append(tanh(robust_zscore(mom, 60)))


            except KeyError:


                pass


        if not signals:


            return 0.0


        return _get_last(pd.concat(signals, axis=1).mean(axis=1))


    hard_score = haven_subscore(hard_havens)


    defensive_score = haven_subscore(defensive_havens)


    return 0.6 * hard_score + 0.4 * defensive_score


# ============================================================


# 3. CREDIT STRESS SCORE (CLS)


# ============================================================


def credit_stress_score(financial_conditions, credit_signal,


                        volatility_signal, vix_term, darkpool_z, pcr_z):


    """



    CLS v1.1 - Arquitectura por familias.



    Bloquea CRISIS si alguna familia es NaN (retorna NaN).



    """


    def stress(val):

        # Fix 2026-09-29: isfinite en lugar de solo pd.notna. Con val=inf,
        # pd.notna(inf) es True -> np.tanh(inf/2)=1.0 -> clip 1.0. El
        # resultado 'estres extremo' se colaba silenciosamente. Familia de
        # invariantes de finitud.
        if val is None or not np.isfinite(val):

            return 0.5

        return float(np.clip(np.tanh(val / 2), 0, 1))


    # Familia 1: Liquidez (financial_conditions)

    nfci_stress = stress(-financial_conditions) if financial_conditions is not None else 0.5

    # Familia 2: Crédito (HYG/LQD)

    credit_stress_val = stress(-credit_signal) if credit_signal is not None else 0.5

    # Familia 3: Volatilidad (VIX)


    vix_stress = stress(volatility_signal) if volatility_signal is not None else 0.5


    # Familia 4: Complementarios


    pcr_stress = stress(pcr_z) if pcr_z is not None else 0.5


    dp_stress = stress(darkpool_z) if (darkpool_z is not None and not np.isnan(darkpool_z)) else 0.5


    complementary_stress = 0.50 * pcr_stress + 0.50 * dp_stress


    cls = (0.25 * nfci_stress +


           0.35 * credit_stress_val +


           0.25 * vix_stress +


           0.15 * complementary_stress)


    # Bloquear si algún componente es NaN


    if np.isnan(cls):


        return np.nan


    return float(np.clip(cls, 0.0, 1.0))


# ============================================================


# 4. INFLATION PRESSURE SCORE (IPS)


# ============================================================


def inflation_pressure_score(df_market):


    assets = ['XLE', '^SPGSCI', 'TIP']


    signals = []


    for t in assets:


        try:


            close = get_col(df_market, t, 'Close')


            mom = close.pct_change(20, fill_method=None)


            signals.append(tanh(robust_zscore(mom, 60)))


        except KeyError:


            pass


    try:


        tip_close = get_col(df_market, 'TIP', 'Close')


        ief_close = get_col(df_market, 'IEF', 'Close')


        tip_ief_ratio = tip_close / ief_close


        tip_ief_mom = tip_ief_ratio.pct_change(20, fill_method=None)


        signals.append(tanh(robust_zscore(tip_ief_mom, 60)))


    except KeyError:


        pass


    if not signals:


        return 0.0


    ips_raw = pd.concat(signals, axis=1).mean(axis=1)


    ips = ips_raw.ewm(span=20).mean()


    return _get_last(ips)


# ============================================================


# 5. ÍNDICES AGREGADOS


# ============================================================


def compute_msi(srs, shs, cls):


    srs_mapped = (srs + 1) / 2


    shs_mapped = (shs + 1) / 2


    raw = 0.40 * srs_mapped + 0.35 * cls + 0.25 * shs_mapped


    return max(0, min(100, raw * 100))


def compute_ipi(ips):


    ips_mapped = (ips + 1) / 2


    return max(0, min(100, ips_mapped * 100))


# ============================================================


# 6. SISTEMA DE PUNTUACIÓN DE ESCENARIOS


# ============================================================


SCENARIO_WEIGHTS = {


    "CLS": {"weight": 3, "reason": "El deterioro financiero es condición necesaria en una crisis."},


    "SHS": {"weight": 2, "reason": "Los activos refugio suelen liderar durante las fases defensivas."},


    "SRS": {"weight": 2, "reason": "La rotación sectorial suele preceder al deterioro macro."},


    "IPS": {"weight": 3, "reason": "La inflación diferencia recesión de estanflación."}


}


# H7 B6 (2026-10-31): SCENARIO_RULES data-driven (v2 con finite y pre_invalid).
# score_scenarios y la nota del reporte (render_mte) derivan de aqui.
# finite: la regla solo se evalua si esas variables son finitas (bug
# latente de inf>0.85 se preserva hasta item separado).
# pre_invalid: si la variable no es finita, base=value; luego se siguen
# aplicando las reglas sin finite (comportamiento exacto del codigo previo).
SCENARIO_RULES = {
    "CRISIS": {
        "pre_invalid": {"var": "cls", "value": -999},
        "rules": [
            {"cond": [("cls", ">", 0.5)], "bonus": 6, "finite": ["cls"]},
            {"cond": [("cls", ">", 0.7)], "bonus": 3, "finite": ["cls"]},
            {"cond": [("shs", ">", 0.3)], "bonus_ref": "SHS", "finite": ["cls"]},
            {"cond": [("srs", ">", 0.3)], "bonus_ref": "SRS", "finite": ["cls"]},
            {"cond": [("cls", ">", 0.85)], "bonus": 3, "finite": ["cls"]},
        ],
    },
    "RECESSION": {
        "pre_invalid": {"var": "cls", "value": 0},
        "rules": [
            {"cond": [("cls", ">", 0.25), ("srs", ">", 0.1), ("cls", "<=", 0.5)], "bonus": 1, "finite": ["cls"]},
            {"cond": [("cls", ">", 0.2)], "bonus_ref": "CLS", "finite": ["cls"]},
            {"cond": [("cls", ">", 0.4), ("cls", "<=", 0.5)], "bonus": 1, "finite": ["cls"]},
            {"cond": [("shs", ">", 0.2)], "bonus_ref": "SHS", "finite": ["shs"]},
            {"cond": [("srs", ">", 0.2)], "bonus_ref": "SRS", "finite": ["srs"]},
        ],
    },
    "STAGFLATION": {
        "pre_invalid": None,
        "rules": [
            {"cond": [("ips", ">", 0.15), ("srs", ">", 0)], "bonus": 1, "finite": ["ips", "srs"]},
            {"cond": [("ips", ">", 0.15)], "bonus_ref": "IPS", "finite": ["ips"]},
            {"cond": [("srs", ">", 0)], "bonus_ref": "SRS", "finite": ["srs"]},
            {"cond": [("cls", "<", 0.3)], "bonus_ref": "CLS", "finite": ["cls"]},
            {"cond": [("ips", ">", 0.5)], "bonus": 2, "finite": ["ips"]},
        ],
    },
    "SOFT LANDING": {
        "pre_invalid": None,
        "rules": [
            {"cond": [("srs", ">", 0.1), ("shs", ">", 0.1)], "bonus": 1, "finite": ["srs", "shs"]},
            {"cond": [("srs", ">", 0.1)], "bonus_ref": "SRS", "finite": ["srs"]},
            {"cond": [("shs", ">", 0.1)], "bonus_ref": "SHS", "finite": ["shs"]},
            {"cond": [("cls", "<", 0)], "bonus_ref": "CLS", "finite": ["cls"]},
            {"cond": [("cls", "<", -0.2)], "bonus": 1, "finite": ["cls"]},
        ],
    },
    "EXPANSION": {
        "pre_invalid": None,
        "rules": [
            {"cond": [("srs", "<", -0.1)], "bonus_ref": "SRS", "finite": ["srs"]},
            {"cond": [("cls", "<", -0.1)], "bonus_ref": "CLS", "finite": ["cls"]},
            {"cond": [("shs", "<", 0)], "bonus_ref": "SHS", "finite": ["shs"]},
            {"cond": [("srs", "<", -0.3)], "bonus": 1, "finite": ["srs"]},
        ],
    },
}

MIXED_BASE = 2


def _eval_cond(var_value, op, thr):
    """Evaluador de condicion (var, op, thr). Opcionalmente NaN-safe."""
    if op == ">":
        return var_value > thr
    if op == "<":
        return var_value < thr
    if op == ">=":
        return var_value >= thr
    if op == "<=":
        return var_value <= thr
    raise ValueError("operador desconocido: " + op)


def score_scenarios(srs, shs, cls, ips):
    """Scores por escenario derivados de SCENARIO_RULES."""
    values = {"srs": srs, "shs": shs, "cls": cls, "ips": ips}
    scores = {}
    for scenario, spec in SCENARIO_RULES.items():
        base = 0
        pre = spec.get("pre_invalid")
        if pre is not None:
            if not np.isfinite(values[pre["var"]]):
                base = pre["value"]
        for rule in spec["rules"]:
            finite_vars = rule.get("finite", [])
            if any(not np.isfinite(values[v]) for v in finite_vars):
                continue
            ok = all(_eval_cond(values[v], op, thr) for v, op, thr in rule["cond"])
            if not ok:
                continue
            if "bonus" in rule:
                base += rule["bonus"]
            else:
                base += SCENARIO_WEIGHTS[rule["bonus_ref"]]["weight"]
        scores[scenario] = base
    scores["MIXED"] = MIXED_BASE
    return scores

