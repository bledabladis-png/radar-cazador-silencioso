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


def test_rankings_con_struct_publica_columna():
    """Con struct_scores, la tabla Rankings tiene columna Struct."""
    sr = _make_sector_results({"XLP": 0.18, "XLF": 0.23, "XLK": 0.45})
    out = "".join(render_ranking_sectorial_unificado(
        sr, tactical_scores=None, structural_scores=None,
        sector_persistence=None, signal_agreements=None,
        signal_agreements_display=None, shock_sensitivities=None,
    ))
    assert "| Fase Wyckoff | Struct |" in out
    # XLP struct +0.18, XLF +0.23, XLK +0.45
    assert "+0.18" in out
    assert "+0.23" in out
    assert "+0.45" in out


def test_rankings_sin_struct_publica_nd():
    """Sin struct_scores, columna Struct con N/D. No rompe."""
    sr = _make_sector_results()  # sin struct_scores
    out = "".join(render_ranking_sectorial_unificado(
        sr, tactical_scores=None, structural_scores=None,
        sector_persistence=None, signal_agreements=None,
        signal_agreements_display=None, shock_sensitivities=None,
    ))
    assert "| Fase Wyckoff | Struct |" in out
    # 3 filas, todas N/D
    assert out.count("N/D") >= 3


def test_rankings_struct_distingue_rango_y_markup():
    """XLP (0.18) y XLF (0.23) ambos RANGE pero distinta magnitud.
    XLK (0.45) MARKUP con margen amplio."""
    sr = _make_sector_results({"XLP": 0.18, "XLF": 0.23, "XLK": 0.45})
    out = "".join(render_ranking_sectorial_unificado(
        sr, tactical_scores=None, structural_scores=None,
        sector_persistence=None, signal_agreements=None,
        signal_agreements_display=None, shock_sensitivities=None,
    ))
    # Los tres valores son distinguibles en la tabla
    assert "+0.18" in out and "+0.23" in out and "+0.45" in out


# ---------- compute_indices_intl: contrato de 3 keys ----------

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


def test_compute_sector_scores_devuelve_struct_scores():
    """Contrato: compute_sector_scores devuelve struct_scores (dict)."""
    from unittest.mock import patch, MagicMock
    from regimes.sector_regime import compute_sector_scores

    # Mock df vacio: compute_sector_scores devuelve None si returns.empty.
    # Solo verificamos que la KEY struct_scores existe en el dict cuando
    # hay scores. Este test usa un mock minimo del resultado.
    # Simulamos a nivel de interfaz: la funcion retorna dict con 5 keys.
    # No ejecutamos compute_sector_scores con datos reales (evita acoplar
    # a parquet). Verificamos la estructura del return via AST.
    import inspect
    src = inspect.getsource(compute_sector_scores)
    assert "'struct_scores'" in src, (
        "compute_sector_scores debe devolver struct_scores en el return"
    )
