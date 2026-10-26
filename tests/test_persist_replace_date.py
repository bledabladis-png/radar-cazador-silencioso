# -*- coding: utf-8 -*-
"""Test regresion: replace_date en persist (bug GE/PWR/VRT 2026-10-26).

Los 6 CSVs de wyckoff_holdings son SUBCONJUNTOS filtrados por fase.
Sin replace_date, un ticker que sale de la fase deja su fila vieja
huerfana. Con replace_date, la fecha se reescribe completa.
"""
import pandas as pd

from src.pipeline.output_writer import persist


def test_replace_date_elimina_filas_viejas_del_mismo_dia(tmp_path):
    """Un ticker que cambia de fase no debe quedar con la fila vieja."""
    p = tmp_path / "test.csv"
    
    # Dia 1: GE es DISTRIBUTION_FORMING
    d1 = pd.DataFrame({
        "date": ["2026-10-23", "2026-10-23"],
        "group": ["XLI", "XLI"],
        "group_type": ["sector_usa", "sector_usa"],
        "ticker": ["GE", "CAT"],
        "phase": ["DISTRIBUTION_FORMING", "MARKDOWN"],
        "event_phase": ["DISTRIBUTION_FORMING", "MARKDOWN"],
        "phase_since": ["", "2026-03-23"],
        "close": [100.0, 200.0],
        "basis": ["", ""],
        "coverage_weight_pct": [85.0, 85.0],
        "n_bars": [500, 500],
        "evaluable": [True, True],
    })
    persist("wyckoff_usa_sector_distribution", d1,
            date_value="2026-10-23", path_override=p)
    df1 = pd.read_csv(p)
    assert len(df1) == 2
    assert set(df1["ticker"]) == {"GE", "CAT"}
    assert df1[df1["ticker"] == "GE"]["phase"].iloc[0] == "DISTRIBUTION_FORMING"
    
    # Dia 2: GE cambia a RANGE, sale del subconjunto DIST_PHASES.
    # Solo CAT sigue en DISTRIBUTION_FORMING.
    d2 = pd.DataFrame({
        "date": ["2026-10-23", "2026-10-23"],
        "group": ["XLI", "XLI"],
        "group_type": ["sector_usa", "sector_usa"],
        "ticker": ["CAT", "RTX"],
        "phase": ["MARKDOWN", "DISTRIBUTION"],
        "event_phase": ["MARKDOWN", "DISTRIBUTION"],
        "phase_since": ["2026-03-23", "2026-03-23"],
        "close": [200.0, 300.0],
        "basis": ["", ""],
        "coverage_weight_pct": [85.0, 85.0],
        "n_bars": [500, 500],
        "evaluable": [True, True],
    })
    persist("wyckoff_usa_sector_distribution", d2,
            date_value="2026-10-23", path_override=p, replace_date=True)
    df2 = pd.read_csv(p)
    
    # Sin replace_date, GE seguia con DISTRIBUTION_FORMING (bug).
    # Con replace_date, GE desaparece.
    assert "GE" not in df2["ticker"].values, (
        f"GE sigue en CSV tras replace_date. Filas: {df2.to_dict('records')}"
    )
    # CAT sigue
    assert "CAT" in df2["ticker"].values
    # RTX añadido
    assert "RTX" in df2["ticker"].values
    # Solo 2 filas (CAT + RTX), no 3 (GE huerfana + CAT + RTX)
    assert len(df2) == 2


def test_replace_date_sin_flag_mantiene_comportamiento_append(tmp_path):
    """Sin replace_date, el comportamiento es append_dedup clasico (GE persiste)."""
    p = tmp_path / "test.csv"
    
    d1 = pd.DataFrame({
        "date": ["2026-10-23"],
        "group": ["XLI"], "group_type": ["sector_usa"], "ticker": ["GE"],
        "phase": ["DISTRIBUTION_FORMING"], "event_phase": ["DISTRIBUTION_FORMING"],
        "phase_since": [""], "close": [100.0], "basis": [""],
        "coverage_weight_pct": [85.0], "n_bars": [500], "evaluable": [True],
    })
    persist("wyckoff_usa_sector_distribution", d1,
            date_value="2026-10-23", path_override=p)
    
    d2 = pd.DataFrame({
        "date": ["2026-10-23"],
        "group": ["XLI"], "group_type": ["sector_usa"], "ticker": ["CAT"],
        "phase": ["MARKDOWN"], "event_phase": ["MARKDOWN"],
        "phase_since": ["2026-03-23"], "close": [200.0], "basis": [""],
        "coverage_weight_pct": [85.0], "n_bars": [500], "evaluable": [True],
    })
    persist("wyckoff_usa_sector_distribution", d2,
            date_value="2026-10-23", path_override=p)  # sin replace_date
    
    df2 = pd.read_csv(p)
    # Append clasico: GE y CAT ambos presentes (GE es huerfano, bug conocido)
    assert "GE" in df2["ticker"].values
    assert "CAT" in df2["ticker"].values
    assert len(df2) == 2