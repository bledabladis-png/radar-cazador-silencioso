# -*- coding: utf-8 -*-
# Centinela H7: volatility_mte.py con NaN en confidence/msi/ipi/cls.
# Cubre V2 (conf NaN -> UNCONFIRMED) y V3 (campos con pd.notna).

from src.report.volatility_mte import render_mte


def _base(**kwargs):
    d = {
        "confidence": 0.7, "scenario": "EXPANSION",
        "msi": 30.0, "ipi": 40.0, "srs": 0.1, "shs": 0.2, "cls": 0.3, "ips": 0.4,
    }
    d.update(kwargs)
    return d


def _out(**kwargs):
    return "".join(render_mte(_base(**kwargs)))


def test_conf_nan_pinta_unconfirmed_no_confirmado():
    out = _out(confidence=float("nan"))
    assert "Escenario (UNCONFIRMED)" in out
    assert "Escenario:**" not in out or "Escenario (UNCONFIRMED)" in out
    assert "nan%" not in out


def test_conf_baja_sigue_unconfirmed():
    out = _out(confidence=0.3)
    assert "Escenario (UNCONFIRMED)" in out


def test_conf_alta_confirma_escenario():
    out = _out(confidence=0.7)
    assert "Escenario:**" in out
    assert "Escenario (UNCONFIRMED)" not in out


def test_msi_nan_pinta_nd():
    out = _out(msi=float("nan"))
    assert "Market Stress Index (MSI):** N/D" in out


def test_ipi_nan_pinta_nd():
    out = _out(ipi=float("nan"))
    assert "Inflation Pressure Index (IPI):** N/D" in out


def test_cls_nan_pinta_nd():
    out = _out(cls=float("nan"))
    assert "Credit Stress Score:** N/D" in out


def test_sin_nan_literal_con_todo_nan():
    out = _out(confidence=float("nan"), msi=float("nan"), ipi=float("nan"),
               srs=float("nan"), shs=float("nan"), cls=float("nan"), ips=float("nan"))
    assert "nan" not in out.replace("tanh", "")


def test_valores_normales_no_se_rompen():
    out = _out()
    assert "Market Stress Index (MSI):** 30" in out
    assert "Credit Stress Score:** 0.30" in out
    assert "Inflation Pressure Score:** 0.40" in out
