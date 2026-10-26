# -*- coding: utf-8 -*-
"""Tests del selector por cobertura de peso (P1-H01 dictamen 2026-10-26).

Cubren los criterios del dictamen:
- Universo invalido (None/vacio, todos NaN, total<=0, NaN ratio > umbral)
  NUNCA se considera conforme al objetivo.
- Objetivo inalcanzable devuelve UNREACHABLE, no conformidad fingida.
- Pesos NaN descartados solo cuando se autoriza explicitamente.
- Denominador = peso total del grupo, no de la seleccion.
- Validacion de min/max/target/max_missing_weight_ratio.
- Determinismo.
"""
import numpy as np
import pandas as pd
import pytest

from indicators.wyckoff_holdings_report import _select_target_holdings


def _grupo(weights, tickers=None):
    n = len(weights)
    if tickers is None:
        tickers = ["T{}".format(i) for i in range(n)]
    return pd.DataFrame({"ticker": tickers, "weight": weights})


# ---------------- Universos invalidos (nunca conformes) ----------------

def test_grupo_vacio_devuelve_INVALID_UNIVERSE():
    out, cob, estado = _select_target_holdings(
        pd.DataFrame(columns=["ticker", "weight"]))
    assert len(out) == 0
    assert cob == 0.0
    assert estado == "INVALID_UNIVERSE"


def test_grupo_none_devuelve_INVALID_UNIVERSE():
    out, cob, estado = _select_target_holdings(None)
    assert len(out) == 0
    assert cob == 0.0
    assert estado == "INVALID_UNIVERSE"


def test_grupo_peso_total_cero_devuelve_INVALID_UNIVERSE():
    g = _grupo([0.0, 0.0, 0.0])
    out, cob, estado = _select_target_holdings(g)
    assert len(out) == 0
    assert cob == 0.0
    assert estado == "INVALID_UNIVERSE"


def test_todos_pesos_ausentes_devuelve_INVALID_UNIVERSE():
    g = _grupo([np.nan, np.nan, np.nan])
    out, cob, estado = _select_target_holdings(g)
    assert len(out) == 0
    assert cob == 0.0
    assert estado == "INVALID_UNIVERSE"


def test_nan_sin_permiso_devuelve_INVALID_UNIVERSE():
    # 1 NaN de 4 = 25% > umbral default 0%. Universo invalido.
    g = pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "weight": [50.0, np.nan, 30.0, 20.0],
    })
    out, cob, estado = _select_target_holdings(g, target_weight=0.75, min_tickers=1)
    assert len(out) == 0
    assert cob == 0.0
    assert estado == "INVALID_UNIVERSE"


def test_nan_descartado_con_max_missing_explicito():
    # Con permiso explicito, el NaN se descarta, no se imputa.
    g = pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "weight": [50.0, np.nan, 30.0, 20.0],
    })
    out, cob, estado = _select_target_holdings(
        g, target_weight=0.75, min_tickers=1, max_missing_weight_ratio=0.5)
    assert set(out["ticker"]) == {"A", "C"}
    assert estado == "REACHED"
    assert cob == pytest.approx(0.80, abs=1e-9)


# ---------------- Indice no numerico (bug idxmax original) ----------------

def test_indice_string_no_confunde_posicion_con_etiqueta():
    g = _grupo([50.0, 30.0, 10.0, 6.0, 4.0],
               tickers=["AAPL", "MSFT", "GOOG", "AMZN", "META"])
    out, cob, estado = _select_target_holdings(g, target_weight=0.85, min_tickers=1)
    assert len(out) == 3
    assert list(out["ticker"]) == ["AAPL", "MSFT", "GOOG"]
    assert estado == "REACHED"
    assert cob == pytest.approx(0.90, abs=1e-9)


# ---------------- Denominador correcto ----------------

def test_denominador_peso_total_grupo_no_de_seleccion():
    g = _grupo([5.0] * 20)
    out, cob, estado = _select_target_holdings(g, target_weight=0.85, min_tickers=1)
    assert len(out) == 17
    assert cob == pytest.approx(0.85, abs=1e-9)
    assert estado == "REACHED"


# ---------------- Objetivo inalcanzable ----------------

def test_objetivo_inalcanzable_por_max_tickers():
    # Target 85%, cumsum alcanza en n=85 pero max=50 trunca -> UNREACHABLE.
    g = _grupo([1.0] * 100)
    out, cob, estado = _select_target_holdings(g, target_weight=0.85, max_tickers=50)
    assert len(out) == 50
    assert estado == "UNREACHABLE"
    assert cob == pytest.approx(0.50, abs=1e-9)


def test_objetivo_inalcanzable_por_peso_insuficiente():
    g = _grupo([40.0, 35.0, 25.0])
    out, cob, estado = _select_target_holdings(
        g, target_weight=0.99, max_tickers=2, min_tickers=1)
    assert len(out) == 2
    assert estado == "UNREACHABLE"
    assert cob == pytest.approx(0.75, abs=1e-9)


# ---------------- min_tickers / max_tickers ----------------

def test_min_tickers_se_respeta_cuando_target_cabe_en_menos():
    g = _grupo([60.0, 20.0, 10.0, 5.0, 5.0])
    out, cob, estado = _select_target_holdings(g, target_weight=0.50, min_tickers=3)
    assert len(out) == 3
    assert estado == "REACHED"
    assert cob == pytest.approx(0.90, abs=1e-9)


def test_grupo_concentrado_selecciona_pocos():
    g = _grupo([30.0, 25.0, 15.0, 10.0, 10.0, 3.0, 3.0, 2.0, 1.0, 1.0])
    out, cob, estado = _select_target_holdings(g, target_weight=0.85, min_tickers=1)
    assert len(out) == 5
    assert estado == "REACHED"
    assert cob == pytest.approx(0.90, abs=1e-9)


def test_grupo_plano_selecciona_muchos():
    g = _grupo([0.5] * 200)
    out, cob, estado = _select_target_holdings(g, target_weight=0.85)
    assert len(out) == 170
    assert estado == "REACHED"
    assert cob == pytest.approx(0.85, abs=1e-9)


# ---------------- Validacion de parametros ----------------

def test_target_fuera_de_rango_lanza_valueerror():
    g = _grupo([50.0, 50.0])
    with pytest.raises(ValueError):
        _select_target_holdings(g, target_weight=0.0)
    with pytest.raises(ValueError):
        _select_target_holdings(g, target_weight=-0.1)
    with pytest.raises(ValueError):
        _select_target_holdings(g, target_weight=1.01)


def test_min_tickers_invalido():
    g = _grupo([50.0, 50.0])
    with pytest.raises(ValueError):
        _select_target_holdings(g, min_tickers=0)
    with pytest.raises(ValueError):
        _select_target_holdings(g, min_tickers=-1)


def test_max_menor_que_min():
    g = _grupo([50.0, 50.0])
    with pytest.raises(ValueError):
        _select_target_holdings(g, min_tickers=5, max_tickers=3)


def test_max_missing_weight_ratio_invalido():
    g = _grupo([50.0, 50.0])
    with pytest.raises(ValueError):
        _select_target_holdings(g, max_missing_weight_ratio=-0.1)
    with pytest.raises(ValueError):
        _select_target_holdings(g, max_missing_weight_ratio=1.0)
    with pytest.raises(ValueError):
        _select_target_holdings(g, max_missing_weight_ratio=1.5)


# ---------------- Determinismo ----------------

def test_dos_ejecuciones_misma_seleccion():
    g = _grupo([30.0, 25.0, 20.0, 15.0, 10.0])
    out1, cob1, estado1 = _select_target_holdings(g, target_weight=0.85)
    out2, cob2, estado2 = _select_target_holdings(g, target_weight=0.85)
    assert list(out1["ticker"]) == list(out2["ticker"])
    assert cob1 == cob2
    assert estado1 == estado2
