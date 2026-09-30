# -*- coding: utf-8 -*-
"""Tests de src/macro_manual_loader.load_macro_manual.

Cobertura objetivo: los 5 caminos de la funcion (D18, 2026-09-30):
1. directorio inexistente -> None.
2. directorio vacio -> None.
3. un CSV valido -> DataFrame con prefijo por fichero.
4. dos CSV -> concat por fecha, prefijos distintos.
5. CSV sin columna 'date' -> ignorado.

La rama except (CSV no parseable) es defensiva: pandas es lenient,
dificil de disparar sin corromper encoding. No se testea.
"""
import pandas as pd

from src.macro_manual_loader import load_macro_manual


def test_load_directorio_inexistente(tmp_path):
    result = load_macro_manual(data_dir=str(tmp_path / "no_existe"))
    assert result is None


def test_load_directorio_vacio(tmp_path):
    result = load_macro_manual(data_dir=str(tmp_path))
    assert result is None


def test_load_un_csv_valido(tmp_path):
    df = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "value": [1.0, 2.0, 3.0],
    })
    df.to_csv(tmp_path / "indicator.csv", index=False)

    result = load_macro_manual(data_dir=str(tmp_path))

    assert result is not None
    assert "date" in result.columns
    assert "indicator_value" in result.columns
    assert len(result) == 3


def test_load_dos_csv_concat_por_fecha(tmp_path):
    df1 = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02"],
        "a": [1.0, 2.0],
    })
    df2 = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-03"],
        "b": [10.0, 30.0],
    })
    df1.to_csv(tmp_path / "i1.csv", index=False)
    df2.to_csv(tmp_path / "i2.csv", index=False)

    result = load_macro_manual(data_dir=str(tmp_path))

    assert result is not None
    assert "i1_a" in result.columns
    assert "i2_b" in result.columns
    # union de fechas: 3 fechas distintas (01, 02, 03)
    assert len(result) == 3


def test_load_prefijos_evitan_colision(tmp_path):
    """Dos CSV con la misma columna 'value' -> prefijos distintos."""
    df1 = pd.DataFrame({"date": ["2026-01-01"], "value": [1.0]})
    df2 = pd.DataFrame({"date": ["2026-01-01"], "value": [10.0]})
    df1.to_csv(tmp_path / "alpha.csv", index=False)
    df2.to_csv(tmp_path / "beta.csv", index=False)

    result = load_macro_manual(data_dir=str(tmp_path))

    assert result is not None
    assert "alpha_value" in result.columns
    assert "beta_value" in result.columns


def test_load_csv_sin_date_ignorado(tmp_path):
    df = pd.DataFrame({"value": [1.0, 2.0]})
    df.to_csv(tmp_path / "sin_date.csv", index=False)

    result = load_macro_manual(data_dir=str(tmp_path))

    assert result is None


def test_load_solo_ficheros_csv(tmp_path):
    """Ficheros no .csv se ignoran."""
    df = pd.DataFrame({"date": ["2026-01-01"], "v": [1.0]})
    df.to_csv(tmp_path / "bien.csv", index=False)
    (tmp_path / "notas.txt").write_text("esto no es un csv")
    (tmp_path / "config.json").write_text("{}")

    result = load_macro_manual(data_dir=str(tmp_path))

    assert result is not None
    assert "bien_v" in result.columns
