# -*- coding: utf-8 -*-
"""Tests del selector por cobertura de peso (P0-H01 dictamen 2026-10-26).

Cubren los puntos que el auditor exigio:
- No idxmax() (busqueda explicita de posicion).
- Objetivo inalcanzable devuelve alcanzado=False, no conformidad fingida.
- Pesos NaN descartados, no imputados.
- Grupos vacios y suma <= 0.
- Validacion de min/max/target.
- Denominador = peso total del grupo, no de la seleccion.
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


# ---------------- Casos degenerados ----------------

def test_grupo_vacio_devuelve_vacio_alcanzado_true():
    out, cob, ok = _select_target_holdings(pd.DataFrame(columns=["ticker", "weight"]))
    assert len(out) == 0
    assert cob == 0.0
    assert ok is True


def test_grupo_none_devuelve_vacio():
    out, cob, ok = _select_target_holdings(None)
    assert len(out) == 0
    assert cob == 0.0
    assert ok is True


def test_grupo_peso_total_cero_devuelve_vacio():
    g = _grupo([0.0, 0.0, 0.0])
    out, cob, ok = _select_target_holdings(g)
    assert len(out) == 0
    assert cob == 0.0
    assert ok is True


def test_nan_en_weight_se_descarta_no_se_imputa():
    g = pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "weight": [50.0, np.nan, 30.0, 20.0],
    })
    out, cob, ok = _select_target_holdings(g, target_weight=0.75, min_tickers=1)
    # 75% de 100 = 75 abs. Con A=50, C=30 -> cumsum=80 >= 75, n=2.
    assert set(out["ticker"]) == {"A", "C"}
    assert ok is True
    assert cob == pytest.approx(0.80, abs=1e-9)


# ---------------- Indice no numerico (bug idxmax original) ----------------

def test_indice_string_no_confunde_posicion_con_etiqueta():
    # Tickers como strings: el bug original intentaba idxmax() + 1,
    # que fallaria o daria resultado incorrecto con indice no numerico.
    g = _grupo([50.0, 30.0, 10.0, 6.0, 4.0], tickers=["AAPL", "MSFT", "GOOG", "AMZN", "META"])
    out, cob, ok = _select_target_holdings(g, target_weight=0.85, min_tickers=1)
    # 85% de 100 = 85. cumsum: 50, 80, 90 -> n=3.
    assert len(out) == 3
    assert list(out["ticker"]) == ["AAPL", "MSFT", "GOOG"]
    assert ok is True
    assert cob == pytest.approx(0.90, abs=1e-9)


# ---------------- Denominador correcto ----------------

def test_denominador_peso_total_grupo_no_de_seleccion():
    # 20 tickers, cada uno 5%. Target 85% = 85 abs. Requiere 17 tickers.
    # Si se usara peso de la seleccion como denominador, 17 tickers al 85%
    # real serian (17*5)/(17*5)=100% y no 85%. El correcto es 17*5/100=0.85.
    g = _grupo([5.0] * 20)
    out, cob, ok = _select_target_holdings(g, target_weight=0.85, min_tickers=1)
    assert len(out) == 17
    assert cob == pytest.approx(0.85, abs=1e-9)


# ---------------- Objetivo inalcanzable ----------------

def test_objetivo_inalcanzable_por_max_tickers():
    # 100 tickers al 1%. Target 85% = 85 abs, requiere 85 tickers.
    # max_tickers=50 -> imposible. alcanzado=False, devuelve 50.
    g = _grupo([1.0] * 100)
    out, cob, ok = _select_target_holdings(g, target_weight=0.85, max_tickers=50)
    assert len(out) == 50
    assert ok is False
    assert cob == pytest.approx(0.50, abs=1e-9)


def test_objetivo_inalcanzable_por_peso_insuficiente():
    # 3 tickers, target 0.95 (95%), suma total 100. Alcanzable.
    # Pero con target 1.01 (invalido, se valida). Caso real inalcanzable:
    # target 0.95 con 3 tickers de 30, 30, 40 -> cumsum 30, 60, 100.
    # cumsum>=95 en i=2 -> n=3. Alcanzable.
    # Inalcanzable real: target 0.99 con 3 de 30, 30, 40 -> cumsum 30,60,100
    # >=99 en i=2 -> n=3. Alcanzable.
    # El caso "peso insuficiente" solo ocurre si target > 1 (invalido) o
    # si hay NaN que reducen la suma.
    # Probamos target 0.99 con max_tickers=2 -> imposible.
    g = _grupo([40.0, 35.0, 25.0])
    out, cob, ok = _select_target_holdings(g, target_weight=0.99, max_tickers=2, min_tickers=1)
    assert len(out) == 2
    assert ok is False
    assert cob == pytest.approx(0.75, abs=1e-9)


# ---------------- min_tickers / max_tickers ----------------

def test_min_tickers_se_respeta_cuando_target_cabe_en_menos():
    # 5 tickers, target 0.50 = 50 abs. Con 1 ticker de 60 ya se cumple.
    # min_tickers=3 fuerza 3.
    g = _grupo([60.0, 20.0, 10.0, 5.0, 5.0])
    out, cob, ok = _select_target_holdings(g, target_weight=0.50, min_tickers=3)
    assert len(out) == 3
    assert ok is True
    # Cobertura real con 3 (60+20+10=90).
    assert cob == pytest.approx(0.90, abs=1e-9)


def test_grupo_concentrado_selecciona_pocos():
    # 10 tickers, top-5 cubren 90%. Target 85% requiere 5.
    g = _grupo([30.0, 25.0, 15.0, 10.0, 10.0, 3.0, 3.0, 2.0, 1.0, 1.0])
    out, cob, ok = _select_target_holdings(g, target_weight=0.85, min_tickers=1)
    assert len(out) == 5
    assert ok is True
    assert cob == pytest.approx(0.90, abs=1e-9)


def test_grupo_plano_selecciona_muchos():
    # 200 tickers al 0.5%. Target 85% requiere 170.
    g = _grupo([0.5] * 200)
    out, cob, ok = _select_target_holdings(g, target_weight=0.85)
    assert len(out) == 170
    assert ok is True
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


# ---------------- Determinismo ----------------

def test_dos_ejecuciones_misma_seleccion():
    g = _grupo([30.0, 25.0, 20.0, 15.0, 10.0])
    out1, cob1, ok1 = _select_target_holdings(g, target_weight=0.85)
    out2, cob2, ok2 = _select_target_holdings(g, target_weight=0.85)
    assert list(out1["ticker"]) == list(out2["ticker"])
    assert cob1 == cob2
    assert ok1 == ok2