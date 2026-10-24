# -*- coding: utf-8 -*-
"""REP-AUDIT-20261018-STRUCT: struct_score en Rankings Sectoriales e
Indices (Fases Wyckoff).

Motivacion: la etiqueta binaria MARKUP/RANGE oculta la magnitud. Un
sector con struct=+0.45 (MARKUP con margen) es distinto de struct=+0.31
(margen minimo). Un sector RANGE con struct=+0.23 (XLF) esta cerca de
MARKUP; otro con struct=-0.01 (XLB) esta lejos. El reporte publica ahora
la columna Struct para que el lector vea el margen sin consultar el CSV.

Struct umbral MARKUP = +0.30 (STRUCT_STRONG en indicators/wyckoff_v1.py).
"""
import pandas as pd

from src.report.synthesis import render_indices_internacionales
from src.report.rankings import render_ranking_sectorial_unificado


# ---------- Indices ----------

def test_indices_con_struct_publica_columna():
    """Con index_struct_scores, la tabla Indices tiene columna Struct
    con valores numericos."""
    phases = {"S&P 500": "RANGE", "Nasdaq-100": "RANGE"}
    struct = {"S&P 500": 0.3116, "Nasdaq-100": 0.2979}
    out = "".join(render_indices_internacionales(phases, None, struct))
    assert "| Indice | Ticker | Fase Wyckoff | Struct |" in out
    assert "+0.31" in out
    assert "+0.30" in out  # Nasdaq cerca del umbral (0.2979 → +0.30)


def test_indices_sin_struct_publica_nd():
    """Sin index_struct_scores, columna Struct con N/D. No rompe."""
    phases = {"S&P 500": "RANGE"}
    out = "".join(render_indices_internacionales(phases, None))
    assert "| Indice | Ticker | Fase Wyckoff | Struct |" in out
    assert "N/D" in out


def test_indices_estructura_equivalente():
    """Llamada con y sin struct: misma estructura, solo difiere la
    columna final."""
    phases = {"S&P 500": "RANGE"}
    out_con = "".join(render_indices_internacionales(phases, None, {"S&P 500": 0.31}))
    out_sin = "".join(render_indices_internacionales(phases, None))
    # Ambos tienen el header con Struct
    assert "| Fase Wyckoff | Struct |" in out_con
    assert "| Fase Wyckoff | Struct |" in out_sin


# ---------- Rankings sectoriales ----------

def _make_sector_results(struct_scores=None):
    ranking = [
        ("XLP", "Consumer Staples", 0.59, "RANGE"),
        ("XLF", "Financials", 0.21, "RANGE"),
        ("XLK", "Technology", -0.38, "MARKUP"),
    ]
    d = {
        "scores": pd.DataFrame(),
        "ranking": ranking,
        "regime": "ROTATIONAL",
        "last_scores": pd.Series(),
    }
    if struct_scores is not None:
        d["struct_scores"] = struct_scores
    return d





def test_compute_indices_intl_devuelve_index_struct_scores():
    """Contrato: compute_indices_intl devuelve index_struct_scores (dict)."""
    from unittest.mock import patch
    from src.pipeline.indices_intl import compute_indices_intl

    with patch("src.pipeline.indices_intl.compute_index_phases",
               return_value=({}, {})),          patch("src.pipeline.indices_intl.download_stock_prices",
               return_value=None):
        out = compute_indices_intl(pd.DataFrame())
    assert "index_struct_scores" in out
    assert isinstance(out["index_struct_scores"], dict)
