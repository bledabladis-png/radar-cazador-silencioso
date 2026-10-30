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
import json

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
    """Helper: renderiza regimenes con un liquidity_state.json controlado."""
    state_dir = tmp_path / "outputs" / "state"
    state_dir.mkdir(parents=True)
    if liq_state is not None:
        (state_dir / "liquidity_state.json").write_text(
            json.dumps(liq_state), encoding="utf-8"
        )
    # Cambiar CWD al tmp_path para que la ruta relativa funcione
    monkeypatch.chdir(tmp_path)
    return render_regimenes(
        macro_score=[-0.22], macro_regime="SLOWDOWN", macro_conf=0.39,
        liquidity_score=[-0.44], financial_regime="HIGH_STRESS", liq_conf=0.31,
        volatility_score=[0.27], vol_regime="NORMAL", vol_conf=0.14,
        real_liquidity_regime="ESTRECHA", real_liquidity_conf=0.90,
        real_liq_score=[-0.20], real_liq_prev=None,
        sector_regime="BROAD PARTICIPATION",
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
