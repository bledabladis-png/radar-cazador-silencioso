# -*- coding: utf-8 -*-
"""D-04 (2026-10-12): divergencia entre render_data_freshness (runtime)
y render_calidad_datos (CSV persistido).

Ambos renders documentan frescura por caminos distintos:
- render_data_freshness: lee liquidity_state.json, pcr_data, parquets.
- render_calidad_datos: lee data_quality.csv (persistido).

Si el CSV esta desactualizado, ambas tablas pueden declarar fechas
distintas para la misma fuente. Estos tests documentan el
comportamiento y la nota cruzada que las conecta.
"""
import json
from datetime import datetime

import pandas as pd

from src.report.freshness import render_data_freshness, render_calidad_datos


def _joined(out):
    return "".join(out)


def test_ambas_mencionan_la_otra_tabla(tmp_path, monkeypatch):
    """Las dos tablas se referencian cruzadamente en sus notas."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "state").mkdir(parents=True)
    (tmp_path / "outputs" / "state" / "liquidity_state.json").write_text(
        json.dumps({"date": "2026-09-29"}), encoding="utf-8")

    # Side runtime
    ref = datetime(2026, 9, 30, 12, 0)
    out_runtime = render_data_freshness(None, None, None, reference_date=ref)
    joined_runtime = _joined(out_runtime)

    # Side CSV
    dq = pd.DataFrame([{
        "date": "2026-09-30", "source": "FRED",
        "last_date": "2026-09-29", "age_calendar_days": 1.0,
        "frequency": "daily", "freshness": "CURRENT",
        "coverage": 1.0, "notes": "test",
    }])
    out_csv = render_calidad_datos(dq)
    joined_csv = _joined(out_csv)

    # La tabla runtime menciona "Calidad, frescura y cobertura" (la otra)
    assert "Calidad, frescura y cobertura" in joined_runtime
    # La tabla CSV menciona "Data Freshness" (la otra)
    assert "Data Freshness" in joined_csv


def test_divergencia_documentada_cuando_csv_stale(tmp_path, monkeypatch):
    """Escenario: runtime ve 2026-09-29, CSV congelado en 2026-09-10.

    Ambas fechas aparecen en la salida. La divergencia es observable,
    no se oculta."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "state").mkdir(parents=True)
    (tmp_path / "outputs" / "state" / "liquidity_state.json").write_text(
        json.dumps({"date": "2026-09-29"}), encoding="utf-8")

    ref = datetime(2026, 9, 30, 12, 0)
    out_runtime = render_data_freshness(None, None, None, reference_date=ref)
    joined_runtime = _joined(out_runtime)

    # CSV con FRED congelado en 2026-09-10 (20 dias de antiguedad)
    dq = pd.DataFrame([{
        "date": "2026-09-30", "source": "FRED",
        "last_date": "2026-09-10", "age_calendar_days": 20.0,
        "frequency": "daily", "freshness": "STALE",
        "coverage": 1.0, "notes": "test",
    }])
    out_csv = render_calidad_datos(dq)
    joined_csv = _joined(out_csv)

    # Runtime ve 2026-09-29 (fresh)
    assert "2026-09-29" in joined_runtime
    # CSV ve 2026-09-10 (stale)
    assert "2026-09-10" in joined_csv
    # Documentado: las notas cruzadas avisan de la posibilidad
    assert "Pueden discrepar" in joined_runtime
    assert "Pueden discrepar" in joined_csv


def test_csv_vacio_no_emite_tabla(tmp_path, monkeypatch):
    """render_calidad_datos con DataFrame vacio no emite la seccion."""
    monkeypatch.chdir(tmp_path)
    out = render_calidad_datos(pd.DataFrame())
    assert out == []

    out2 = render_calidad_datos(None)
    assert out2 == []


def test_runtime_sin_datos_emite_nd(tmp_path, monkeypatch):
    """render_data_freshness sin datos emite filas N/D para las
    fuentes que el render siempre emite. FINRA no se emite cuando
    darkpool_data es None (comportamiento intencional del bloque
    `if darkpool_data:` sin else)."""
    monkeypatch.chdir(tmp_path)
    out = render_data_freshness(None, None, None)
    joined = _joined(out)
    assert "CBOE (Opciones) | N/D" in joined
    assert "FRED (Macro) | N/D" in joined
    assert "Yahoo Finance (Precios) | N/D" in joined
    # FINRA no aparece cuando darkpool_data es None
    assert "FINRA (Dark Pools) | N/D" not in joined
