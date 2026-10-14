"""Tests del modulo experimental regimes/liquidity_v2.py.

Verifican que la implementacion del modulo congelado (dictamen Fase J)
reproduce los numeros reportados al auditor.
"""
import numpy as np
import pandas as pd
import pytest

from regimes.liquidity_v2 import (
    UMBRAL_BAJA, UMBRAL_ELEVADA, UMBRAL_EXTREMA, UMBRAL_ALERTA,
    PERSIST_ENTRADA, PERSIST_SALIDA,
    clasificar_severidad,
    calcular_regimen_persistente,
    calcular_alerta_diferencial_extremo,
    calcular_c1_candidate,
    calendar_context,
    period_context,
    evaluar_serie,
)


# ============================================================
# Capa 1: severidad cruda
# ============================================================

@pytest.mark.parametrize("diff,esperado", [
    (-10.0, "TENSION_BAJA"),
    (-0.1,  "TENSION_BAJA"),
    (0.0,   "NORMAL"),
    (4.99,  "NORMAL"),
    (5.0,   "TENSION_ELEVADA"),
    (19.99, "TENSION_ELEVADA"),
    (20.0,  "TENSION_EXTREMA"),
    (295.0, "TENSION_EXTREMA"),
    (np.nan, "N/D"),
])
def test_severidad_cruda_umbrales(diff, esperado):
    assert clasificar_severidad(diff) == esperado


# ============================================================
# Capa 2: regimen persistente 3/3
# ============================================================

def test_persistencia_entrada_3_sesiones():
    """Confirmacion tras 3 sesiones consecutivas con condicion.
    diff = [0, 10, 10, 10, 10]. Racha: 0(no), 1, 2, 3(confirma), 4.
    """
    diff = pd.Series([0.0, 10.0, 10.0, 10.0, 10.0],
                     index=pd.date_range("2024-01-01", periods=5))
    r = calcular_regimen_persistente(diff)
    assert not r.iloc[0]  # racha=0
    assert not r.iloc[1]  # racha=1
    assert not r.iloc[2]  # racha=2
    assert r.iloc[3]      # racha=3 -> confirma
    assert r.iloc[4]      # sigue


def test_persistencia_salida_3_sesiones():
    """Desconfirmacion tras 3 sesiones consecutivas sin condicion."""
    cond = pd.Series([True]*5 + [False]*5,
                     index=pd.date_range("2024-01-01", periods=10))
    r = calcular_regimen_persistente(cond.astype(float) * 10)
    # Dias 5-6-7: primera salida tras 3
    # Día 5 (idx 5): racha_out=1, sigue confirmado
    # Día 6 (idx 6): racha_out=2, sigue confirmado
    # Día 7 (idx 7): racha_out=3, desconfirmado
    assert r.iloc[5]  # sigue
    assert r.iloc[6]  # sigue
    assert not r.iloc[7]  # desconfirmado
    assert not r.iloc[8]
    assert not r.iloc[9]


def test_persistencia_racha_se_rompe_por_un_dia():
    """Un dia que no cumple rompe la racha de entrada."""
    cond = pd.Series([True, True, False, True, True, True],
                     index=pd.date_range("2024-01-01", periods=6))
    r = calcular_regimen_persistente(cond.astype(float) * 10)
    # Racha: 1,2,rompe,1,2,3 -> confirma en idx 5
    assert not r.iloc[0]
    assert not r.iloc[1]
    assert not r.iloc[2]
    assert not r.iloc[3]
    assert not r.iloc[4]
    assert r.iloc[5]


def test_persistencia_nan_no_cuenta():
    """Datos ausentes (NaN) no cuentan como sesion valida.
    diff = [10, NaN, 10, 10, 10]. Racha: 1, mantenida(1), 2, 3, sigue.
    Confirma en idx 3.
    """
    diff = pd.Series([10.0, np.nan, 10.0, 10.0, 10.0],
                     index=pd.date_range("2024-01-01", periods=5))
    r = calcular_regimen_persistente(diff)
    assert not r.iloc[0]  # racha=1
    assert not r.iloc[1]  # NaN: no cambia racha (sigue=1)
    assert not r.iloc[2]  # racha=2
    assert r.iloc[3]      # racha=3 -> confirma
    assert r.iloc[4]      # sigue


# ============================================================
# Capa 3: alerta diferencial extremo
# ============================================================

def test_alerta_extremo_100pb():
    diff = pd.Series([10.0, 50.0, 99.9, 100.0, 295.0, np.nan],
                     index=pd.date_range("2024-01-01", periods=6))
    a = calcular_alerta_diferencial_extremo(diff)
    assert not a.iloc[0]
    assert not a.iloc[1]
    assert not a.iloc[2]
    assert a.iloc[3]   # >=100
    assert a.iloc[4]   # 295
    assert not a.iloc[5]  # NaN -> False


# ============================================================
# C1 diagnostico
# ============================================================

def test_c1_candidate_condiciones():
    c1 = pd.Series([30.0, 60.0, 60.0, 60.0, np.nan],
                   index=pd.date_range("2024-01-01", periods=5))
    a  = pd.Series([10.0, 10.0, 25.0, 5.0, 10.0],
                   index=pd.date_range("2024-01-01", periods=5))
    r = calcular_c1_candidate(c1, a)
    assert r.iloc[0] == False  # C1 < 48.46
    assert r.iloc[1] == True   # C1 > 48.46 y A < 20
    assert r.iloc[2] == False  # A = 25 (>= 20)
    assert r.iloc[3] == True   # C1 > 48.46 y A < 20
    assert r.iloc[4] is pd.NA  # C1 NaN -> NA, no False

def test_c1_candidate_distingue_na_de_false():
    """NA != False. Datos ausentes no se convierten en negacion."""
    c1 = pd.Series([30.0, np.nan, 60.0, 60.0],
                   index=pd.date_range("2024-01-01", periods=4))
    a  = pd.Series([10.0, 10.0, np.nan, 5.0],
                   index=pd.date_range("2024-01-01", periods=4))
    r = calcular_c1_candidate(c1, a)
    assert r.iloc[0] == False     # C1 < 48.46
    assert r.iloc[1] is pd.NA     # C1 NaN
    assert r.iloc[2] is pd.NA     # A NaN
    assert r.iloc[3] == True      # ambos validos, condicion OK


# ============================================================
# Calendario
# ============================================================

def test_calendar_context():
    assert calendar_context(pd.Timestamp("2024-03-01")) == "inicio_natural"
    assert calendar_context(pd.Timestamp("2024-03-15")) == "mid_natural"
    assert calendar_context(pd.Timestamp("2024-03-29")) == "fin_natural"


def test_period_context_prioridad_year_sobre_month():
    """31-dic: fin año, no fin mes. Prioridad year > month."""
    assert period_context(pd.Timestamp("2024-12-31")) == "turn_of_year_end"
    assert period_context(pd.Timestamp("2024-12-30")) == "turn_of_year_end"


def test_period_context_quarter_end():
    """29-mar-2019 (viernes cierre Q1). Prioridad quarter > month."""
    assert period_context(pd.Timestamp("2019-03-29")) == "turn_of_quarter_end"


def test_period_context_mid():
    """Dia normal."""
    assert period_context(pd.Timestamp("2024-06-12")) == "mid_period"


# ============================================================
# Integracion: evaluar_serie reproduce el evento 2019
# ============================================================

def test_evaluar_serie_repo_2019():
    """Reproduce la deteccion del repo crisis 2019 (dato del sandbox)."""
    idx = pd.date_range("2019-09-03", "2019-10-10", freq="B")
    sofr = pd.Series([2.40, 2.42, 2.42, 2.40, 2.40, 2.40, 2.40, 2.40,
                      2.40, 2.42, 2.43, 2.43, 2.42, 2.43, 2.43, 2.40,
                      2.40, 2.40, 2.40, 2.40, 2.40, 2.40, 2.40, 2.40,
                      2.40, 2.40, 2.40], index=idx[:27])
    effr = pd.Series([2.40, 2.40, 2.40, 2.40, 2.40, 2.40, 2.40, 2.40,
                      2.40, 2.40, 2.40, 2.40, 2.40, 2.40, 2.40, 2.40,
                      2.40, 2.40, 2.40, 2.40, 2.40, 2.40, 2.40, 2.40,
                      2.40, 2.40, 2.40], index=idx[:27])
    # Sobrescribir 16, 17, 18 sep con pico
    sofr.loc["2019-09-16"] = 2.43
    sofr.loc["2019-09-17"] = 5.25
    sofr.loc["2019-09-18"] = 2.70
    effr.loc["2019-09-17"] = 2.30

    df = evaluar_serie(sofr, effr)
    # El 17-sep debe ser alerta_extremo
    assert df.loc["2019-09-17", "alerta_diferencial_extremo"]
    # El 17-sep debe ser severidad TENSION_EXTREMA
    assert df.loc["2019-09-17", "severidad_cruda"] == "TENSION_EXTREMA"
    # 17-sep debe ser turn_of_month (no, es mid_period - dia 17)
    assert df.loc["2019-09-17", "period_context"] == "mid_period"


# ============================================================
# Constantes congeladas (no deben cambiar sin dictamen)
# ============================================================

def test_umbrales_congelados():
    assert UMBRAL_BAJA == 0.0
    assert UMBRAL_ELEVADA == 5.0
    assert UMBRAL_EXTREMA == 20.0
    assert UMBRAL_ALERTA == 100.0
    assert PERSIST_ENTRADA == 3
    assert PERSIST_SALIDA == 3
