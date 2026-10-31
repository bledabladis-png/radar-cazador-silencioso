# -*- coding: utf-8 -*-
"""Contrato AST: compute_pcr_signals <-> render_sentimiento_opciones.

Productor: indicators/options.py::compute_pcr_signals. Dict literal
con 23 claves top-level. Retorna None en 2 ramas de fallo (CBOE no
disponible, data vacio o sin 'total_pcr'). El caller en
src/pipeline/market_data.py::_compute_pcr propaga None tal cual.

Consumidor: src/report/sentiment.py::render_sentimiento_opciones.
Lee 20 de las 23 claves via pcr_data.get() y pcr_data[...] (z_score
via []). No lee: status, score, percentile_20d (status se usa en
pipeline, score y percentile_20d en volatility_structure).

Top-level: contrato AST puro.
Nested: no aplica (todas las claves son escalares).
"""
from pathlib import Path

from tests._dict_contract import (
    producer_return_keys,
    consumer_read_keys,
    assert_render_subset_return,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PRODUCER = REPO_ROOT / "indicators" / "options.py"
RENDER = REPO_ROOT / "src" / "report" / "sentiment.py"

EXPECTED_TOP_LEVEL = frozenset({
    "status", "total_pcr", "pcr_ewm", "z_score", "momentum",
    "percentile", "state", "score", "percentile_20d",
    "index_pcr", "equity_pcr", "etp_pcr", "spx_pcr", "vix_pcr",
    "ihr", "ihr_state", "index_volume_share", "put_share",
    "call_share", "volume_pcr", "oi_pcr", "last_date", "timestamp",
})


def test_render_pcr_no_lee_campos_ausentes_en_producer():
    """render_sentimiento_opciones lee X => X en compute_pcr_signals."""
    assert_render_subset_return(
        producer_module=PRODUCER,
        producer_fn="compute_pcr_signals",
        consumer_module=RENDER,
        consumer_fn="render_sentimiento_opciones",
        consumer_var="pcr_data",
    )


def test_top_level_pcr_coincide_con_esperado():
    """El dict top-level publica exactamente EXPECTED_TOP_LEVEL.

    Si crece, revisar si el nuevo campo es funcional (debe estar en el
    render o justificarse). Si decrece, los consumidores pierden el
    campo silenciosamente.
    """
    keys = producer_return_keys(PRODUCER, "compute_pcr_signals")
    assert keys == EXPECTED_TOP_LEVEL, (
        f"El dict top-level de compute_pcr_signals cambio. "
        f"Esperado: {sorted(EXPECTED_TOP_LEVEL)}. "
        f"Actual: {sorted(keys)}. "
        f"Si es una decision consciente, actualizar este test y "
        f"revisar render_sentimiento_opciones."
    )


def test_render_pcr_lee_campos_criticos():
    """Regresion: campos que el reporte no puede perder.

    - total_pcr: linea principal.
    - z_score: nivel de actividad.
    - state: categoria.
    - ihr: proxy institucional.
    - volume_pcr + oi_pcr: metricas calculadas derivadas.
    """
    reads = consumer_read_keys(
        RENDER, "render_sentimiento_opciones", "pcr_data")
    for campo in ("total_pcr", "z_score", "state", "ihr",
                  "volume_pcr", "oi_pcr"):
        assert campo in reads, (
            f"render_sentimiento_opciones dejo de leer '{campo}'. El "
            f"reporte perderia informacion de PCR. Si es intencional, "
            f"borrar este test con justificacion."
        )
