# -*- coding: utf-8 -*-
"""Contrato AST: compute_darkpool_signals <-> render_darkpool.

Productor: indicators/darkpool.py::compute_darkpool_signals. Dict
literal con 10 claves top-level: status, week, fecha, media_dark_pool,
z_score, percentile, state, n_tickers, datos, z_windows. Retorna None
en 5 ramas de fallo (sin semana FINRA, ats_data vacio, columnas
faltantes, sin volumenes). El render hace `if darkpool_data:` y no
pinta. Patron identico a _compute_mte en ERROR.

Consumidor: src/report/darkpool.py::render_darkpool. Lee 8 de las 10
claves:
- media_dark_pool, n_tickers, z_score, percentile, state, week:
  valores escalares, pinta lineas y tabla de z-scores por ventana.
- datos: DataFrame con per-ticker (% ATS, volumen), se itera.
- z_windows: dict {13w: {z, state}, 26w: ..., 52w: ..., 104w: ...}.
No lee: status (siempre "OK" por construccion), fecha (alias
historico de week).

Top-level: contrato AST puro.
Nested (datos: df, z_windows: dict): cubierto por tests de
integracion existentes de render_darkpool (test_report_alerts_header
y otros).
"""
from pathlib import Path

from tests._dict_contract import (
    producer_return_keys,
    consumer_read_keys,
    assert_render_subset_return,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PRODUCER = REPO_ROOT / "indicators" / "darkpool.py"
RENDER = REPO_ROOT / "src" / "report" / "darkpool.py"

EXPECTED_TOP_LEVEL = frozenset({
    "status", "week", "fecha", "media_dark_pool", "z_score",
    "percentile", "state", "n_tickers", "datos", "z_windows",
})


# --- Downstream: render lee X => X en producer ---

def test_render_darkpool_no_lee_campos_ausentes_en_producer():
    """render_darkpool lee X => X en compute_darkpool_signals."""
    assert_render_subset_return(
        producer_module=PRODUCER,
        producer_fn="compute_darkpool_signals",
        consumer_module=RENDER,
        consumer_fn="render_darkpool",
        consumer_var="darkpool_data",
    )


# --- Top-level exacto: cierra whitelist creep ---

def test_top_level_darkpool_coincide_con_esperado():
    """El dict top-level publica exactamente EXPECTED_TOP_LEVEL.

    Si crece, revisar si el nuevo campo es funcional (debe estar en el
    render o justificarse). Si decrece, los consumidores pierden el
    campo silenciosamente.
    """
    keys = producer_return_keys(PRODUCER, "compute_darkpool_signals")
    assert keys == EXPECTED_TOP_LEVEL, (
        f"El dict top-level de compute_darkpool_signals cambio. "
        f"Esperado: {sorted(EXPECTED_TOP_LEVEL)}. "
        f"Actual: {sorted(keys)}. "
        f"Si es una decision consciente, actualizar este test y "
        f"revisar render_darkpool."
    )


# --- Regresion: campos criticos siguen leidos ---

def test_render_darkpool_lee_campos_criticos():
    """Regresion: campos que el reporte no puede perder.

    - media_dark_pool: % ATS medio (linea principal).
    - n_tickers: cobertura (sin el, no se puede juzgar la
      representatividad del %).
    - z_score + percentile: nivel de actividad ATS.
    - state: categoria (normal / baja / alta).
    - datos: tabla per-ticker (top ATS %).
    - z_windows: tabla multi-ventana 13w/26w/52w/104w.
    """
    reads = consumer_read_keys(
        RENDER, "render_darkpool", "darkpool_data")
    for campo in ("media_dark_pool", "n_tickers", "z_score",
                  "percentile", "state", "datos", "z_windows"):
        assert campo in reads, (
            f"render_darkpool dejo de leer '{campo}'. El reporte "
            f"perderia informacion de dark pools. Si es intencional, "
            f"borrar este test con justificacion."
        )
