# -*- coding: utf-8 -*-
"""2026-10-30: exponer en el reporte WARNs que solo vivian en el state.

N1 - mte_state.json declara futures_status STALE/UNAVAILABLE, pero el
     reporte MTE solo publicaba scenario/confidence. El lector no sabia
     que los inputs de futuros estaban stale.

N3 - liquidity_state.json declara severidad_cruda=TENSION_ELEVADA y
     diff_bp=6.0, pero el reporte solo publicaba el regimen agregado
     ESTRECHA. El lector no sabia que la severidad cruda era alta.

Los fixes NO degradan confidence ni scores. Solo visibilizan.
"""

from src.report.volatility_mte import render_mte
from src.report.header import render_regimenes


# ============================================================
# N1: render_mte debe publicar futures_status si no es OK
# ============================================================
def _mte_base(**over):
    base = {
        "scenario": "SOFT LANDING",
        "confidence": 0.56,
        "msi": 51,
        "ipi": 48,
        "srs": 0.33,
        "shs": 0.32,
        "cls": 0.22,
        "ips": -0.05,
    }
    base.update(over)
    return base


def test_mte_sin_futures_status_no_pinta_linea():
    """Sin clave futures_status (state antiguo), no se pinta nada."""
    out = "".join(render_mte(_mte_base()))
    assert "Futures Status" not in out


def test_mte_futures_status_ok_no_pinta_linea():
    """Si futures_status == 'OK', no se pinta nada. Es el caso normal."""
    out = "".join(render_mte(_mte_base(futures_status="OK")))
    assert "Futures Status" not in out


def test_mte_futures_status_stale_pinta_linea():
    """STALE es un WARN real: el reporte debe publicarlo."""
    out = "".join(render_mte(_mte_base(futures_status="STALE")))
    assert "Futures Status:** STALE" in out
    assert "inputs de futuros no frescos" in out


def test_mte_futures_status_unavailable_pinta_linea():
    out = "".join(render_mte(_mte_base(futures_status="UNAVAILABLE")))
    assert "Futures Status:** UNAVAILABLE" in out


def test_mte_futures_status_no_altera_confidence():
    """El fix es solo visual: la confidence publicada no cambia."""
    out_ok = "".join(render_mte(_mte_base(futures_status="OK")))
    out_stale = "".join(render_mte(_mte_base(futures_status="STALE")))
    # Misma linea de escenario (confidence 0.56)
    assert "Confidence Score no calibrado: 0.56" in out_ok
    assert "Confidence Score no calibrado: 0.56" in out_stale


# ============================================================
# N3: render_regimenes debe publicar severidad_cruda si no es OK
# ============================================================
def _regimenes(tmp_path, liq_state=None, monkeypatch=None):
    """Helper: renderiza regimenes con liquidity_state inyectado como kwarg."""
    # H4 B6 (2026-10-31): render_regimenes ya no lee el fichero.
    return render_regimenes(
        macro_score=[-0.22], macro_regime="SLOWDOWN", macro_conf=0.39,
        liquidity_score=[-0.44], financial_regime="HIGH_STRESS", liq_conf=0.31,
        volatility_score=[0.27], vol_regime="NORMAL", vol_conf=0.14,
        real_liquidity_regime="ESTRECHA", real_liquidity_conf=0.90,
        real_liq_score=[-0.20], real_liq_prev=None,
        sector_regime="BROAD PARTICIPATION",
        liquidity_state=liq_state,
    )


def test_liquidity_sin_state_no_pinta_severidad(tmp_path, monkeypatch):
    out = "".join(_regimenes(tmp_path, liq_state=None, monkeypatch=monkeypatch))
    assert "Severidad:" not in out


def test_liquidity_severidad_ok_no_pinta(tmp_path, monkeypatch):
    state = {"date": "2026-10-28", "regime": "ESTRECHA",
             "v2": {"severidad_cruda": "OK", "diff_bp": 0.0}}
    out = "".join(_regimenes(tmp_path, liq_state=state, monkeypatch=monkeypatch))
    assert "Severidad:" not in out


def test_liquidity_severidad_tension_elevada_pinta(tmp_path, monkeypatch):
    state = {"date": "2026-10-28", "regime": "ESTRECHA",
             "v2": {"severidad_cruda": "TENSION_ELEVADA", "diff_bp": 6.0}}
    out = "".join(_regimenes(tmp_path, liq_state=state, monkeypatch=monkeypatch))
    assert "Severidad: TENSION_ELEVADA" in out
    assert "diff_bp=+6.0" in out


def test_liquidity_severidad_sin_diff_bp_pinta_nd(tmp_path, monkeypatch):
    state = {"date": "2026-10-28", "regime": "ESTRECHA",
             "v2": {"severidad_cruda": "TENSION_ELEVADA"}}
    out = "".join(_regimenes(tmp_path, liq_state=state, monkeypatch=monkeypatch))
    assert "Severidad: TENSION_ELEVADA" in out
    assert "diff_bp=N/D" in out


def test_liquidity_severidad_no_altera_regimen(tmp_path, monkeypatch):
    """El fix es solo visual: el regimen agregado no cambia."""
    state = {"date": "2026-10-28", "regime": "ESTRECHA",
             "v2": {"severidad_cruda": "TENSION_ELEVADA", "diff_bp": 6.0}}
    out = "".join(_regimenes(tmp_path, liq_state=state, monkeypatch=monkeypatch))
    assert "**Liquidez Real (FRED):** ESTRECHA" in out
    assert "Signal Consistency: 90%" in out


# ============================================================
# Pending transition (2026-10-30, clase 1fb791ab)
# ============================================================
def test_mte_pending_pinta_linea():
    """pending != None -> linea "Pending transition: X" visible.

    Caso real: run 41037457375 tenia scenario=SOFT LANDING con
    pending=RECESSION. El run siguiente (local 20261030_201420)
    confirmo RECESSION. Sin la linea, el lector ve dos dias con
    mismos indices (msi, ipi, srs, shs, cls, ips) y distinto
    scenario sin explicacion.
    """
    out = "".join(render_mte(_mte_base(pending="RECESSION")))
    assert "**Pending transition:** RECESSION" in out
    assert "se confirmara" in out


def test_mte_pending_none_no_pinta_linea():
    """pending=None -> no se pinta linea. Caso normal."""
    out = "".join(render_mte(_mte_base(pending=None)))
    assert "Pending transition" not in out


def test_mte_pending_sin_clave_no_pinta_linea():
    """Sin clave pending (state antiguo) -> no rompe, no pinta."""
    out = "".join(render_mte(_mte_base()))  # sin pending
    assert "Pending transition" not in out


def test_mte_pending_no_altera_scenario():
    """El fix es solo visual: el scenario publicado no cambia."""
    out_sin = "".join(render_mte(_mte_base(pending=None)))
    out_con = "".join(render_mte(_mte_base(pending="RECESSION")))
    assert "Escenario:**" in out_sin
    assert "Escenario:**" in out_con
    # El scenario (SOFT LANDING) es el mismo en ambos casos
    assert "SOFT LANDING" in out_sin
    assert "SOFT LANDING" in out_con
