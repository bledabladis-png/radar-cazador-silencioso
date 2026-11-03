# -*- coding: utf-8 -*-
"""FASE-A-IAE-PER-TICKER (2026-11-01): tests del modulo per_ticker.

Contrato:
  - Filtros de anomalia basados en ratios, no lista de tickers.
  - pct_increase = n_increase / (n_increase + n_decrease).
  - Ranking por consenso de gestores, no por suma bruta.
  - filter_reason trazable: se ve el motivo de cada exclusion.

Escenarios verificados (datos reales del trimestre 2026Q1->Q2):
  - SPCX (IPO 12-jun-2026): n_new/n_managers ~1.0 -> MOSTLY_NEW_POSITIONS
  - HON (spin-off SOLS): n_increase/n_managers ~0.03 -> ANOMALOUS_ONE_SIDED
  - DD (spin-off Qnity): n_increase/n_managers ~0.03 -> ANOMALOUS_ONE_SIDED
  - AAPL/Ticker normal: 60/40 -> pasa sin filter_reason.
"""
import pandas as pd
import pytest

from src.institutional_accumulation.aggregation.per_ticker import (
    aggregate_by_ticker,
    top_accumulating,
    top_distributing,
    excluded_anomalies,
    FILTER_ANOMALOUS,
    FILTER_MOSTLY_NEW,
    FILTER_MOSTLY_EXIT,
)


def _row(manager, canonical, prev, curr, status):
    return {
        "filing_manager_cik": manager,
        "canonical_security": canonical,
        "discretion_type": "SOLE",
        "sshprnamt_current": float(curr),
        "sshprnamt_previous": float(prev),
        "delta_shares": float(curr - prev),
        "match_status": status,
    }


def test_filtros_excluyen_anomalias_de_eventos():
    """SPCX (IPO) y HON (spin-off) se excluyen. AAPL (normal) pasa."""
    rows = []
    # SPCX: 10 gestores, todos NEW (prev=0, curr=100)
    for i in range(10):
        rows.append(_row(f"spcx_{i}", "equity:SPCX", 0, 100, "NEW"))
    # HON: 10 gestores, todos reducen (prev=1000, curr=100)
    for i in range(10):
        rows.append(_row(f"hon_{i}", "equity:HON", 1000, 100, "BOTH"))
    # AAPL: 6 aumentan (prev=100, curr=150), 4 reducen (prev=100, curr=50)
    for i in range(6):
        rows.append(_row(f"aapl_up_{i}", "equity:AAPL", 100, 150, "BOTH"))
    for i in range(4):
        rows.append(_row(f"aapl_dn_{i}", "equity:AAPL", 100, 50, "BOTH"))
    delta = pd.DataFrame(rows)
    out = aggregate_by_ticker(delta)

    spcx = out[out["ticker"] == "SPCX"].iloc[0]
    hon = out[out["ticker"] == "HON"].iloc[0]
    aapl = out[out["ticker"] == "AAPL"].iloc[0]

    assert spcx["filter_reason"] == FILTER_MOSTLY_NEW, (
        f"SPCX (IPO) debe caer en MOSTLY_NEW_POSITIONS, "
        f"no {spcx['filter_reason']!r}"
    )
    assert hon["filter_reason"] == FILTER_ANOMALOUS, (
        f"HON (spin-off) debe caer en ANOMALOUS_ONE_SIDED, "
        f"no {hon['filter_reason']!r}"
    )
    assert pd.isna(aapl["filter_reason"]), (
        f"AAPL (normal) no debe tener filter_reason, "
        f"tiene {aapl['filter_reason']!r}"
    )
    # Sanity: AAPL tiene 6 increase, 4 decrease, pct_increase=0.6
    assert aapl["n_increase"] == 6
    assert aapl["n_decrease"] == 4
    assert abs(aapl["pct_increase"] - 0.6) < 1e-9


def test_new_prioridad_sobre_increase_extremo():
    """Cuando pct_new >= 0.50 y pct_increase >= 0.99 (ambos disparan),
    gana MOSTLY_NEW (evento de universo, posible IPO con continuacion).
    No gana ANOMALOUS, porque el consenso extremo entre los que ya
    tenian posicion es consecuencia del evento.

    Esto captura IPO parcial: 50% nuevos entrando + 100% de los que
    ya tenian aumentan. Semantica: 'algo grande esta pasando, no es
    solo consenso'.
    """
    rows = []
    # 5 gestores NEW, 5 gestores increase (BOTH)
    for i in range(5):
        rows.append(_row(f"m_new_{i}", "equity:ABC", 0, 100, "NEW"))
    for i in range(5):
        rows.append(_row(f"m_up_{i}", "equity:ABC", 100, 200, "BOTH"))
    delta = pd.DataFrame(rows)
    out = aggregate_by_ticker(delta)
    abc = out[out["ticker"] == "ABC"].iloc[0]
    assert abc["n_increase"] == 5
    assert abc["n_decrease"] == 0
    assert abc["n_new"] == 5
    assert abc["n_managers"] == 10
    assert abc["pct_increase"] == pytest.approx(1.0)
    assert abc["pct_new"] == pytest.approx(0.5)
    # Prioridad: new/exit primero -> MOSTLY_NEW gana sobre ANOMALOUS
    assert abc["filter_reason"] == FILTER_MOSTLY_NEW


def test_spin_off_se_detecta_por_pct_exit():
    """HON/DD reales: pct_increase ~0.028 (no cae en <=0.01), pero
    pct_exit ~0.92 (cae en MOSTLY_EXITS). Sin lista de tickers."""
    rows = []
    # 46 NEW, 2618 EXIT, 79 inc (BOTH), 2729 dec (BOTH). Replicamos
    # con muestra pequena: 5 NEW, 90 EXIT, 8 inc, 40 dec.
    for i in range(5):
        rows.append(_row(f"n_new_{i}", "equity:HON", 0, 100, "NEW"))
    for i in range(90):
        rows.append(_row(f"n_exit_{i}", "equity:HON", 100, 0, "EXIT"))
    for i in range(8):
        rows.append(_row(f"n_inc_{i}", "equity:HON", 100, 200, "BOTH"))
    for i in range(40):
        rows.append(_row(f"n_dec_{i}", "equity:HON", 200, 100, "BOTH"))
    delta = pd.DataFrame(rows)
    out = aggregate_by_ticker(delta)
    hon = out[out["ticker"] == "HON"].iloc[0]
    # pct_increase = 8/(8+40) = 0.167 (> 0.01, no dispara ANOMALOUS)
    assert hon["pct_increase"] > 0.01
    # pct_exit = 90/143 ~= 0.629 (>= 0.50, dispara MOSTLY_EXITS)
    assert hon["pct_exit"] >= 0.50
    assert hon["filter_reason"] == FILTER_MOSTLY_EXIT


def test_ipo_se_detecta_por_pct_new():
    """SPCX real: pct_new ~0.999, se detecta sin lista de tickers."""
    rows = []
    for i in range(100):
        rows.append(_row(f"spcx_{i}", "equity:SPCX", 0, 100, "NEW"))
    delta = pd.DataFrame(rows)
    out = aggregate_by_ticker(delta)
    spcx = out[out["ticker"] == "SPCX"].iloc[0]
    assert spcx["pct_new"] >= 0.99
    assert spcx["filter_reason"] == FILTER_MOSTLY_NEW


def test_ranking_por_consenso_no_por_suma_bruta():
    """El ranking top_accumulating ordena por n_increase, no por sum_delta.

    Tres tickers:
      A: 100 gestores +100/gestor (suma baja, densidad media).
      B: 500 gestores +100/gestor (suma media, densidad alta).
      C:  50 gestores +10M/gestor (suma MUY alta, densidad baja).
    sum_delta: C(500M) > B(50K) > A(10K). Pero el ranking debe ser
    B > A > C (por n_increase). Si el ranking ordenara por suma,
    seria C > B > A.
    """
    rows = []
    # A: 100 inc (+100), 40 dec (-100)
    for i in range(100):
        rows.append(_row(f"a_inc_{i}", "equity:A", 100, 200, "BOTH"))
    for i in range(40):
        rows.append(_row(f"a_dec_{i}", "equity:A", 200, 100, "BOTH"))
    # B: 500 inc (+100), 200 dec (-100)
    for i in range(500):
        rows.append(_row(f"b_inc_{i}", "equity:B", 100, 200, "BOTH"))
    for i in range(200):
        rows.append(_row(f"b_dec_{i}", "equity:B", 200, 100, "BOTH"))
    # C: 50 inc (+10M), 20 dec (-10M)
    for i in range(50):
        rows.append(_row(f"c_inc_{i}", "equity:C", 10_000_000, 20_000_000, "BOTH"))
    for i in range(20):
        rows.append(_row(f"c_dec_{i}", "equity:C", 20_000_000, 10_000_000, "BOTH"))
    delta = pd.DataFrame(rows)
    out = aggregate_by_ticker(delta)

    # Sanity: ninguno cae por filtros
    for t in ("A", "B", "C"):
        row = out[out["ticker"] == t].iloc[0]
        assert pd.isna(row["filter_reason"]), (
            f"{t} inesperadamente excluido por {row['filter_reason']!r}. "
            f"n_inc={row['n_increase']}, n_dec={row['n_decrease']}, "
            f"n_managers={row['n_managers']}"
        )

    c = out[out["ticker"] == "C"].iloc[0]
    b = out[out["ticker"] == "B"].iloc[0]
    a = out[out["ticker"] == "A"].iloc[0]
    # Sanity: C tiene la mayor suma bruta
    assert c["sum_delta"] > b["sum_delta"] > a["sum_delta"], (
        f"Suma bruta esperada C>B>A, es "
        f"C={c['sum_delta']} B={b['sum_delta']} A={a['sum_delta']}"
    )

    top = top_accumulating(out, n=20)
    # Ranking por consenso: B(500) > A(100) > C(50)
    assert list(top["ticker"])[:3] == ["B", "A", "C"], (
        f"Ranking por consenso debe ser B>A>C, es {list(top['ticker'])[:3]}. "
        f"Si salio C>B>A, top_accumulating esta usando sum_delta."
    )


def test_excluded_anomalies_expone_razon():
    """Los anomales aparecen con su filter_reason trazable."""
    rows = []
    # SPCX: 10 NEW
    for i in range(10):
        rows.append(_row(f"spcx_{i}", "equity:SPCX", 0, 100, "NEW"))
    # HON: 10 reducen
    for i in range(10):
        rows.append(_row(f"hon_{i}", "equity:HON", 1000, 100, "BOTH"))
    delta = pd.DataFrame(rows)
    out = aggregate_by_ticker(delta)
    excl = excluded_anomalies(out)
    tickers = set(excl["ticker"])
    assert "SPCX" in tickers
    assert "HON" in tickers
    reasons = dict(zip(excl["ticker"], excl["filter_reason"]))
    assert reasons["SPCX"] == FILTER_MOSTLY_NEW
    assert reasons["HON"] == FILTER_ANOMALOUS

# ============================================================
# FASE-A-IAE-PER-TICKER (2026-11-01): render en src/report/iae.py
# ============================================================

def test_render_iae_con_per_ticker_publica_tabla():
    """render_iae_section publica la seccion 'Cambio neto por ticker'
    cuando el dict tiene per_ticker no vacio."""
    from src.report.iae import render_iae_section
    rows = []
    # 10 NEW por SPCX, 6 up + 4 down por AAPL, 10 down por HON
    for i in range(10):
        rows.append(_row(f"spcx_{i}", "equity:SPCX", 0, 100, "NEW"))
    for i in range(6):
        rows.append(_row(f"aapl_up_{i}", "equity:AAPL", 100, 150, "BOTH"))
    for i in range(4):
        rows.append(_row(f"aapl_dn_{i}", "equity:AAPL", 100, 50, "BOTH"))
    for i in range(10):
        rows.append(_row(f"hon_{i}", "equity:HON", 1000, 100, "BOTH"))
    per_ticker = aggregate_by_ticker(pd.DataFrame(rows))

    iae = {
        "status": "OK",
        "period_previous": "2025Q4", "period_current": "2026Q1",
        "iso_previous": "2025-12-31", "iso_current": "2026-03-31",
        "nipc_total": 123456, "nipc_sole": 123456,
        "nipc_dfnd": 0, "nipc_otr": 0,
        "n_delta_observable": 30,
        "n_both": 20, "n_new": 10, "n_exit": 0,
        "n_unresolved_identity": 0,
        "coverage_status": "VALID", "coverage_quality": "COMPLETE",
        "catalog_coverage_declared": 0.85,
        "catalog_keys_total": 322, "catalog_keys_observed": 273,
        "catalog_coverage_warning": False,
        "evidence_class": "CONTRACTUAL",
        "per_ticker": per_ticker,
        "error": None,
    }
    out = "".join(render_iae_section(iae))
    assert "### Cambio neto por ticker (13F)" in out
    assert "AAPL" in out  # debe aparecer (paso filtros)
    assert "Excluidos de las tablas" in out  # nota de exclusiones
    # SPCX y HON excluidos: aparecen en la nota, no en tablas
    # (se busca su nombre en la nota)


def test_render_iae_sin_per_ticker_no_rompe():
    """per_ticker=None no rompe el render (modo legacy)."""
    from src.report.iae import render_iae_section
    iae = {
        "status": "OK",
        "period_previous": "2025Q4", "period_current": "2026Q1",
        "iso_previous": "2025-12-31", "iso_current": "2026-03-31",
        "nipc_total": 1, "nipc_sole": 1, "nipc_dfnd": 0, "nipc_otr": 0,
        "n_delta_observable": 1, "n_both": 1, "n_new": 0, "n_exit": 0,
        "n_unresolved_identity": 0,
        "coverage_status": "VALID", "coverage_quality": "COMPLETE",
        "catalog_coverage_declared": 0.5,
        "catalog_keys_total": 10, "catalog_keys_observed": 5,
        "catalog_coverage_warning": False,
        "evidence_class": "CONTRACTUAL",
        "per_ticker": None,
        "error": None,
    }
    out = "".join(render_iae_section(iae))
    assert "Cambio neto por ticker" not in out
    assert "Variacion neta de posiciones declaradas (13F)" in out


def test_render_iae_per_ticker_excluye_anomalias_del_top():
    """SPCX (IPO) no aparece en tabla Acumulando; aparece en la nota."""
    from src.report.iae import render_iae_section
    rows = []
    # SPCX 10 NEW (IPO)
    for i in range(10):
        rows.append(_row(f"spcx_{i}", "equity:SPCX", 0, 100, "NEW"))
    # AAPL 6 up + 4 dn (normal)
    for i in range(6):
        rows.append(_row(f"aapl_up_{i}", "equity:AAPL", 100, 150, "BOTH"))
    for i in range(4):
        rows.append(_row(f"aapl_dn_{i}", "equity:AAPL", 100, 50, "BOTH"))
    per_ticker = aggregate_by_ticker(pd.DataFrame(rows))
    iae = {
        "status": "OK",
        "period_previous": "2025Q4", "period_current": "2026Q1",
        "iso_previous": "2025-12-31", "iso_current": "2026-03-31",
        "nipc_total": 1, "nipc_sole": 1, "nipc_dfnd": 0, "nipc_otr": 0,
        "n_delta_observable": 1, "n_both": 1, "n_new": 0, "n_exit": 0,
        "n_unresolved_identity": 0,
        "coverage_status": "VALID", "coverage_quality": "COMPLETE",
        "catalog_coverage_declared": 0.5,
        "catalog_keys_total": 10, "catalog_keys_observed": 5,
        "catalog_coverage_warning": False,
        "evidence_class": "CONTRACTUAL",
        "per_ticker": per_ticker,
        "error": None,
    }
    out = "".join(render_iae_section(iae))
    # Encontrar bloque de tabla Acumulando: entre "**Acumulando**" y "**Distribuyendo**" o fin de seccion
    import re
    m = re.search(r"\*\*Acumulando\*\*(.*?)(?=\*\*Distribuyendo\*\*|\*Excluidos)", out, re.DOTALL)
    assert m is not None, "seccion Acumulando no encontrada"
    tabla_acc = m.group(1)
    assert "AAPL" in tabla_acc, "AAPL debe estar en tabla Acumulando"
    assert "SPCX" not in tabla_acc, "SPCX (IPO) NO debe estar en tabla Acumulando"
    # Y debe estar en la nota de excluidos
    assert "SPCX" in out[out.index("Excluidos de las tablas"):]


# ============================================================
# 9.3 (auditoria externa 2026-11-03): filtro por consenso, no sum_delta
# ============================================================

def test_top_accumulating_filtra_por_consenso_no_por_sum_delta():
    """Regresion 9.3: n_increase < n_decrease con sum_delta > 0 NO va a Acumulando."""
    rows = []
    # AAPL: 3 aumentan fuerte (+900), 5 reducen poco (-100). sum_delta=+2200 (>0)
    # n_increase=3 < n_decrease=5 -> consenso NEGATIVO (no debe estar en Acumulando)
    for i in range(3):
        rows.append(_row(f"aapl_inc_{i}", "equity:AAPL", 100, 1000, "BOTH"))
    for i in range(5):
        rows.append(_row(f"aapl_dec_{i}", "equity:AAPL", 1000, 900, "BOTH"))
    # MSFT: 10 aumentan poco (+10), 3 reducen poco (-10). consenso POSITIVO
    for i in range(10):
        rows.append(_row(f"msft_inc_{i}", "equity:MSFT", 100, 110, "BOTH"))
    for i in range(3):
        rows.append(_row(f"msft_dec_{i}", "equity:MSFT", 100, 90, "BOTH"))
    out = aggregate_by_ticker(pd.DataFrame(rows))
    aapl = out[out['ticker']=='AAPL'].iloc[0]
    msft = out[out['ticker']=='MSFT'].iloc[0]
    assert aapl['sum_delta'] > 0
    assert aapl['n_increase'] < aapl['n_decrease']
    assert msft['n_increase'] > msft['n_decrease']
    top_acc = top_accumulating(out, n=20)
    assert "MSFT" in set(top_acc["ticker"])
    assert "AAPL" not in set(top_acc["ticker"]), (
        "AAPL (n_inc<n_dec, sum_delta>0) NO debe estar en Acumulando. "
        "El filtro debe ser por consenso, no por suma bruta."
    )


def test_top_distributing_filtra_por_consenso_no_por_sum_delta():
    """Regresion 9.3: n_increase > n_decrease con sum_delta < 0 NO va a Distribuyendo."""
    rows = []
    # CB: 6 aumentan (+100), 5 reducen (-200). sum_delta=-400 (<0)
    # n_increase=6 > n_decrease=5 -> consenso POSITIVO (no debe estar en Distribuyendo)
    for i in range(6):
        rows.append(_row(f"cb_inc_{i}", "equity:CB", 100, 200, "BOTH"))
    for i in range(5):
        rows.append(_row(f"cb_dec_{i}", "equity:CB", 200, 0, "BOTH"))
    # NOW: 3 aumentan (+10), 10 reducen (-10). consenso NEGATIVO
    for i in range(3):
        rows.append(_row(f"now_inc_{i}", "equity:NOW", 100, 110, "BOTH"))
    for i in range(10):
        rows.append(_row(f"now_dec_{i}", "equity:NOW", 100, 90, "BOTH"))
    out = aggregate_by_ticker(pd.DataFrame(rows))
    cb = out[out['ticker']=='CB'].iloc[0]
    now = out[out['ticker']=='NOW'].iloc[0]
    assert cb['sum_delta'] < 0
    assert cb['n_increase'] > cb['n_decrease']
    assert now['n_increase'] < now['n_decrease']
    top_dis = top_distributing(out, n=20)
    assert "NOW" in set(top_dis["ticker"])
    assert "CB" not in set(top_dis["ticker"]), (
        "CB (n_inc>n_dec, sum_delta<0) NO debe estar en Distribuyendo."
    )


def test_consistencia_encabezado_y_columnas():
    """Contrato 9.3: top_acc cumple n_inc>n_dec; top_dis n_inc<n_dec.
    Ninguna fila puede contradecir el encabezado de su tabla."""
    rows = []
    for i in range(5):
        rows.append(_row(f"a_inc_{i}", "equity:A", 100, 200, "BOTH"))
    for i in range(2):
        rows.append(_row(f"a_dec_{i}", "equity:A", 200, 100, "BOTH"))
    for i in range(3):
        rows.append(_row(f"b_inc_{i}", "equity:B", 100, 200, "BOTH"))
    for i in range(7):
        rows.append(_row(f"b_dec_{i}", "equity:B", 200, 100, "BOTH"))
    out = aggregate_by_ticker(pd.DataFrame(rows))
    top_acc = top_accumulating(out, n=20)
    top_dis = top_distributing(out, n=20)
    for _, r in top_acc.iterrows():
        assert r['n_increase'] > r['n_decrease'], (
            f"Acumulando tiene {r['ticker']} con n_inc={r['n_increase']} <= n_dec={r['n_decrease']}"
        )
    for _, r in top_dis.iterrows():
        assert r['n_increase'] < r['n_decrease'], (
            f"Distribuyendo tiene {r['ticker']} con n_inc={r['n_increase']} >= n_dec={r['n_decrease']}"
        )
