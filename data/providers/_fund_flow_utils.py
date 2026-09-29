"""Utilidades compartidas de fund-flow providers.

Este modulo contiene la logica comun de normalizacion robusta para los
4 providers de ETF flows (DAXEX, ISF, IWM, LYXI).

CONTRATO ESTADISTICO (aprobado por auditor 2026-09-27):

  - Localizacion: mediana de la ventana.
  - Escala: MAD (Median Absolute Deviation) de esa misma ventana.
  - Factor de escala: 1.4826 (asintotico a sigma en distribucion normal).
  - Sin ffill: festivo no es lo mismo que valor repetido en fund-flow.
  - Ventana: 120 sesiones, min_periods=20.
  - Clip contractual: [-5, +5].

TRATAMIENTO MAD == 0:

  Si la escala robusta es cero, el z-score no esta definido por la
  division habitual. Se distinguen dos subcasos:

    - MAD == 0 & s == median  ->  0.0 (sin desviacion real).
    - MAD == 0 & s != median  ->  NaN (desviacion sin escala).

  PROHIBIDO que un consumidor convierta el NaN en 0.0 via fillna().
  Hacerlo enmascara exactamente la senal que este contrato pretende
  preservar.

CONTRATO DE NOMBRE:

  Esta funcion tiene un contrato DISTINTO al canonico
  src/utils.py::robust_zscore. Ese ultimo opera sobre precios/RS
  donde el ffill(3) representa continuidad del precio entre sesiones.
  Aqui el dominio es fund-flow, donde un festivo no implica un flujo
  repetido. Son dos contratos estadisticos distintos que comparten
  nombre por herencia historica. No fusionar.
"""
from __future__ import annotations

import pandas as pd


WINDOW_DEFAULT = 120
MIN_PERIODS_DEFAULT = 20
MAD_SCALE = 1.4826
MAD_EPSILON = 1e-12
CLIP_ABS = 5.0


def fund_flow_robust_zscore(
    series: pd.Series,
    window: int = WINDOW_DEFAULT,
    min_periods: int = MIN_PERIODS_DEFAULT,
) -> pd.Series:
    """Z-score robusto para series de fund-flow (rolling window).

    Calcula, para cada ventana, la mediana y el MAD de esa misma ventana,
    y aplica el z-score al ultimo valor de la ventana.

    Args:
        series: pd.Series de flow_pct (o equivalente).
        window: ventana rolling (default 120).
        min_periods: minimo de observaciones validas por ventana (default 20).

    Returns:
        pd.Series con el z-score, saturado a [-5, +5].
        NaN donde la ventana no tiene suficientes valores validos,
        donde el ultimo valor es NaN, o donde MAD==0 y el ultimo
        valor difiere de la mediana.
    """
    # Fix 2026-09-29 (patron 8.5 #6): delegar en _compute_z_value.
    # Antes esta funcion tenia una copia interna 'calculate' identica
    # a _compute_z_value (19 lineas duplicadas). Ahora ambos callers
    # comparten la misma implementacion. El test de coherencia
    # test_regime_z_coincide_con_funcion_sin_regime garantiza que z y
    # z_regime siguen produciendo los mismos valores.
    return series.rolling(
        window,
        min_periods=min_periods,
    ).apply(lambda w: _compute_z_value(w, min_periods), raw=False)


# ============================================================
# REGIMEN DE LA SALIDA (auditor 2026-09-27, condicion 2)
# ============================================================

REGIME_NORMAL = "NORMAL"
REGIME_SATURATED = "SATURATED"
REGIME_MAD0_SAME = "MAD0_SAME"
REGIME_MAD0_NAN = "MAD0_NAN"
REGIME_INSUFFICIENT = "INSUFFICIENT"

_CODE_NORMAL = 0.0
_CODE_SATURATED = 1.0
_CODE_MAD0_SAME = 2.0
_CODE_MAD0_NAN = 3.0
_CODE_INSUFFICIENT = 4.0

_CODE_TO_REGIME = {
    _CODE_NORMAL: REGIME_NORMAL,
    _CODE_SATURATED: REGIME_SATURATED,
    _CODE_MAD0_SAME: REGIME_MAD0_SAME,
    _CODE_MAD0_NAN: REGIME_MAD0_NAN,
    _CODE_INSUFFICIENT: REGIME_INSUFFICIENT,
}


def _compute_z_value(window_values: pd.Series, min_periods: int) -> float:
    values = window_values.dropna()
    if len(values) < min_periods:
        return float("nan")

    latest = window_values.iloc[-1]
    if pd.isna(latest):
        return float("nan")

    median = float(values.median())
    mad = float((values - median).abs().median())

    if pd.isna(mad) or mad <= MAD_EPSILON:
        if latest == median:
            return 0.0
        return float("nan")

    z = (float(latest) - median) / (MAD_SCALE * mad)
    return float(max(-CLIP_ABS, min(CLIP_ABS, z)))


def _compute_regime_code(window_values: pd.Series, min_periods: int) -> float:
    values = window_values.dropna()
    if len(values) < min_periods:
        return _CODE_INSUFFICIENT

    latest = window_values.iloc[-1]
    if pd.isna(latest):
        return _CODE_INSUFFICIENT

    median = float(values.median())
    mad = float((values - median).abs().median())

    if pd.isna(mad) or mad <= MAD_EPSILON:
        if latest == median:
            return _CODE_MAD0_SAME
        return _CODE_MAD0_NAN

    z = (float(latest) - median) / (MAD_SCALE * mad)
    if z > CLIP_ABS or z < -CLIP_ABS:
        return _CODE_SATURATED
    return _CODE_NORMAL


def fund_flow_robust_zscore_with_regime(
    series: pd.Series,
    window: int = WINDOW_DEFAULT,
    min_periods: int = MIN_PERIODS_DEFAULT,
) -> tuple:
    """Version extendida que devuelve (z, regime).

    Comparte el mismo contrato estadistico que fund_flow_robust_zscore,
    pero expone el diagnostico para trazabilidad.

    Regimenes:
      NORMAL       -> MAD > 0 y z dentro del clip.
      SATURATED    -> MAD > 0 y z bruto fuera del clip [-5, +5].
      MAD0_SAME    -> escala MAD nula y observacion actual coincidente con
                      la mediana. z = 0.
      MAD0_NAN     -> escala MAD nula y observacion actual distinta de la
                      mediana. z no definido: NaN.
      INSUFFICIENT -> la ventana tiene menos de min_periods valores validos,
                      o el valor actual es NaN.

    Notas de contrato:
      - SATURATED no significa "absurdo": significa que el z-score bruto
        supero el limite contractual del sistema.
      - MAD0_SAME/MAD0_NAN: MAD=0 no implica que todos los valores de la
        ventana sean identicos; solo que la escala robusta estimada por MAD
        es cero.
    """
    min_p = min_periods

    z_values = series.rolling(window, min_periods=min_p).apply(
        lambda w: _compute_z_value(w, min_p), raw=False,
    )
    regime_codes = series.rolling(window, min_periods=min_p).apply(
        lambda w: _compute_regime_code(w, min_p), raw=False,
    )

    regimes = regime_codes.map(
        lambda c: _CODE_TO_REGIME.get(c, REGIME_INSUFFICIENT)
        if pd.notna(c)
        else REGIME_INSUFFICIENT
    )
    return z_values, regimes