"""regimes/liquidity_v2.py

Especificacion experimental B3 CONGELADA — dictamen auditor Fase J (14-oct-2026).

NO PRODUCCION. Modulo sandbox para validacion walk-forward / prospectiva.

Arquitectura (3 capas independientes):
  Capa 1: severidad cruda (4 categorias por dia)
  Capa 2: regimen persistente binario (entrada 3/3, salida 3/3)
  Capa 3: alerta diferencial extremo (>=100 pb, sin persistencia)
  Diagnostico: C1 (p99 SOFR - EFFR), sin efecto decisorio

Umbrales congelados (no modificar tras resultados):
  diff <  0 pb  -> TENSION_BAJA
  0 <= diff < 5 -> NORMAL
  5 <= diff < 20 -> TENSION_ELEVADA
  diff >= 20 pb -> TENSION_EXTREMA
  diff >= 100 pb -> alerta_diferencial_extremo

period_context prioridad (una etiqueta exclusiva por dia):
  1. turn_of_year_end       (diciembre, ventana +-3d al ultimo dia habil)
  2. turn_of_year_start     (enero, ventana +-3d al primer dia habil)
  3. turn_of_quarter_end    (mar/jun/sep/dic, ultimo dia habil)
  4. turn_of_quarter_start  (ene/abr/jul/oct, primer dia habil)
  5. turn_of_month_end      (cualquier mes, ultimo dia habil)
  6. turn_of_month_start    (cualquier mes, primer dia habil)
  7. mid_period

Referencias:
  - Dictamen N8 Fase J (2026-10-14).
  - Sandbox de pruebas: outputs/audit/liq_backtest/.
  - Modulo productivo (intacto): regimes/liquidity.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


# ===================================================================
# Umbrales congelados (NO MODIFICAR tras congelacion experimental)
# ===================================================================

UMBRAL_BAJA = 0.0           # pb
UMBRAL_ELEVADA = 5.0        # pb
UMBRAL_EXTREMA = 20.0       # pb
UMBRAL_ALERTA = 100.0       # pb

PERSIST_ENTRADA = 3         # sesiones consecutivas
PERSIST_SALIDA = 3          # sesiones consecutivas

C1_UMBRAL_DIAGNOSTICO = 48.46  # pb (p99 de C1 historico)


# ===================================================================
# Festivos USA (subset minimo para 2018-2026)
# ===================================================================
_FESTIVOS_FIJOS = []
for _y in range(2018, 2030):
    _FESTIVOS_FIJOS.extend([
        f"{_y}-01-01",  # New Year
        f"{_y}-07-04",  # Independence
        f"{_y}-12-24",  # Christmas Eve (observado)
        f"{_y}-12-25",  # Christmas
        f"{_y}-11-11",  # Veterans
        f"{_y}-06-19",  # Juneteenth (2022+)
    ])
_FESTIVOS = set(pd.Timestamp(f) for f in _FESTIVOS_FIJOS)


def _is_habil(d: pd.Timestamp) -> bool:
    """Dia habil USA (weekday + festivos fijos). Aproximacion conservadora."""
    if d.weekday() >= 5:
        return False
    return d.normalize() not in _FESTIVOS


def _ultimo_dia_habil_mes(year: int, month: int) -> Optional[pd.Timestamp]:
    for day in range(31, 0, -1):
        try:
            d = pd.Timestamp(year=year, month=month, day=day)
        except ValueError:
            continue
        if _is_habil(d):
            return d
    return None


def _primer_dia_habil_mes(year: int, month: int) -> Optional[pd.Timestamp]:
    for day in range(1, 32):
        try:
            d = pd.Timestamp(year=year, month=month, day=day)
        except ValueError:
            continue
        if _is_habil(d):
            return d
    return None


# ===================================================================
# Capa 1: severidad cruda
# ===================================================================

def clasificar_severidad(diff_bp: float) -> str:
    """Clasificacion cruda por dia. Umbrales congelados."""
    if pd.isna(diff_bp):
        return "N/D"
    if diff_bp < UMBRAL_BAJA:
        return "TENSION_BAJA"
    if diff_bp < UMBRAL_ELEVADA:
        return "NORMAL"
    if diff_bp < UMBRAL_EXTREMA:
        return "TENSION_ELEVADA"
    return "TENSION_EXTREMA"


# ===================================================================
# Capa 2: regimen persistente (state machine binaria)
# ===================================================================

def _persistir_binario(cond: pd.Series, n_in: int = PERSIST_ENTRADA,
                       n_out: int = PERSIST_SALIDA) -> pd.Series:
    """State machine: entrada tras n_in sesiones True; salida tras n_out
    sesiones False. Datos ausentes no cuentan como sesion valida.

    Devuelve serie bool: True = tension confirmada.
    """
    out = []
    estado = False
    racha_in = 0
    racha_out = 0
    for c in cond:
        if pd.isna(c):
            # Dato ausente: no cuenta como sesion valida
            out.append(estado)
            continue
        if c:
            racha_out = 0
            if estado:
                out.append(True)
            else:
                racha_in += 1
                if racha_in >= n_in:
                    estado = True
                    out.append(True)
                else:
                    out.append(False)
        else:
            racha_in = 0
            if estado:
                racha_out += 1
                if racha_out >= n_out:
                    estado = False
                    out.append(False)
                else:
                    out.append(True)
            else:
                out.append(False)
    return pd.Series(out, index=cond.index)


def calcular_regimen_persistente(diff_bp: pd.Series,
                                 n_in: int = PERSIST_ENTRADA,
                                 n_out: int = PERSIST_SALIDA) -> pd.Series:
    """Regimen persistente binario. La condicion es TENSION_ELEVADA
    o superior (>= 5 pb)."""
    cond = diff_bp >= UMBRAL_ELEVADA
    cond = cond.where(diff_bp.notna())  # NaN si diff es NaN
    return _persistir_binario(cond, n_in, n_out)


# ===================================================================
# Capa 3: alerta diferencial extremo
# ===================================================================

def calcular_alerta_diferencial_extremo(diff_bp: pd.Series) -> pd.Series:
    """Alerta inmediata. Independiente de la persistencia."""
    return (diff_bp >= UMBRAL_ALERTA).fillna(False)


# ===================================================================
# Diagnostico C1
# ===================================================================

def calcular_c1_candidate(c1_bp: pd.Series, diff_bp: pd.Series) -> pd.Series:
    """Campo diagnostico: divergencia de cola C1 > umbral y A < extrema.
    NO es una alerta operativa. Solo un campo de analisis."""
    return (
        (c1_bp > C1_UMBRAL_DIAGNOSTICO)
        & (diff_bp < UMBRAL_EXTREMA)
    ).fillna(False)


# ===================================================================
# Contexto de calendario
# ===================================================================

def calendar_context(d: pd.Timestamp) -> str:
    """Contexto respecto al calendario NATURAL."""
    day = d.day
    if day <= 3:
        return "inicio_natural"
    if day >= 28:
        return "fin_natural"
    return "mid_natural"


def period_context(d: pd.Timestamp) -> str:
    """Contexto FINANCIERO. Prioridad exclusiva (una sola etiqueta).

    Regla de prioridad (mas especifica primero):
    year > quarter > month.
    """
    y, m = d.year, d.month
    uh = _ultimo_dia_habil_mes(y, m)
    ph = _primer_dia_habil_mes(y, m)

    # Cierre (mayor prioridad)
    if uh is not None and 0 <= (uh - d.normalize()).days <= 3:
        if m == 12:
            return "turn_of_year_end"
        if m in (3, 6, 9, 12):
            return "turn_of_quarter_end"
        return "turn_of_month_end"

    # Apertura
    if ph is not None and 0 <= (d.normalize() - ph).days <= 3:
        if m == 1:
            return "turn_of_year_start"
        if m in (1, 4, 7, 10):
            return "turn_of_quarter_start"
        return "turn_of_month_start"

    return "mid_period"


# ===================================================================
# API publica
# ===================================================================

@dataclass
class LiquidityStateV2:
    """Estado compuesto de las 3 capas + diagnostico."""
    fecha: pd.Timestamp
    diff_bp: float
    severidad_cruda: str              # Capa 1
    regimen_persistente: bool         # Capa 2
    alerta_diferencial_extremo: bool  # Capa 3
    c1_tail_divergence_candidate: bool  # Diagnostico
    calendar_context: str
    period_context: str


def evaluar_dia(
    fecha: pd.Timestamp,
    diff_bp: float,
    regimen_persistente: bool,
    alerta_extremo: bool,
    c1_bp: float,
) -> LiquidityStateV2:
    """Evalua un dia completo. Las capas 2 y 3 se calculan por serie
    completa antes de llamar aqui, para no depender del orden."""
    return LiquidityStateV2(
        fecha=fecha,
        diff_bp=diff_bp,
        severidad_cruda=clasificar_severidad(diff_bp),
        regimen_persistente=bool(regimen_persistente),
        alerta_diferencial_extremo=bool(alerta_extremo),
        c1_tail_divergence_candidate=bool(
            pd.notna(c1_bp)
            and c1_bp > C1_UMBRAL_DIAGNOSTICO
            and pd.notna(diff_bp)
            and diff_bp < UMBRAL_EXTREMA
        ),
        calendar_context=calendar_context(fecha),
        period_context=period_context(fecha),
    )


def evaluar_serie(sofr: pd.Series, effr: pd.Series,
                  c1_bp: pd.Series | None = None) -> pd.DataFrame:
    """Evalua la serie completa. Devuelve DataFrame con todas las capas.

    Args:
        sofr: serie SOFR en %.
        effr: serie EFFR en %.
        c1_bp: p99 SOFR - EFFR en pb (opcional). Si None, C1 queda False.

    Returns:
        DataFrame indexado por fecha con columnas:
        diff_bp, severidad_cruda, regimen_persistente,
        alerta_diferencial_extremo, c1_tail_divergence_candidate,
        calendar_context, period_context.
    """
    idx = sofr.index.intersection(effr.index)
    diff_pct = sofr.loc[idx] - effr.loc[idx]
    diff_bp = diff_pct * 100.0

    severidad = diff_bp.apply(clasificar_severidad)
    persist = calcular_regimen_persistente(diff_bp)
    alerta = calcular_alerta_diferencial_extremo(diff_bp)

    if c1_bp is not None:
        c1_aligned = c1_bp.reindex(idx)
        c1_cand = calcular_c1_candidate(c1_aligned, diff_bp)
    else:
        c1_cand = pd.Series(False, index=idx)

    cal_ctx = pd.Series(
        [calendar_context(d) for d in idx], index=idx, name="calendar_context"
    )
    per_ctx = pd.Series(
        [period_context(d) for d in idx], index=idx, name="period_context"
    )

    return pd.DataFrame({
        "diff_bp": diff_bp,
        "severidad_cruda": severidad,
        "regimen_persistente": persist,
        "alerta_diferencial_extremo": alerta,
        "c1_tail_divergence_candidate": c1_cand,
        "calendar_context": cal_ctx,
        "period_context": per_ctx,
    })
