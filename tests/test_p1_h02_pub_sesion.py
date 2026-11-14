# -*- coding: utf-8 -*-
"""FIX-P1H02-PUB-SESION (2026-11-14): pub_n por sesion, no acumulado.

Antes: _published_by_group sumaba todas las filas historicas del CSV,
dando pub_n > sel_n (QQQ: 302 vs 98) y pub_peso > 100% del grupo.
Despues: filtra por la fecha maxima del CSV.
"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = (Path(__file__).resolve().parent.parent
          / "scripts" / "audit" / "p1_h02_cobertura_efectiva.py")


def _load_module(tmp_path, monkeypatch):
    """Carga el script con HIST apuntando a tmp_path/outputs/history."""
    hist = tmp_path / "outputs" / "history"
    hist.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pd.DataFrame())
    spec = importlib.util.spec_from_file_location("p1_h02_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "HIST", hist)
    return mod, hist


def _csv(tmp_path, hist, name, rows):
    p = hist / (name + ".csv")
    pd.DataFrame(rows).to_csv(p, index=False)


def test_una_sola_fecha_acumula_normal(tmp_path, monkeypatch):
    mod, hist = _load_module(tmp_path, monkeypatch)
    _csv(tmp_path, hist, "wyckoff_usa_index_distribution", [
        {"date": "2026-11-13", "group": "QQQ", "weight": 1.0},
        {"date": "2026-11-13", "group": "QQQ", "weight": 2.0},
        {"date": "2026-11-13", "group": "SPY", "weight": 5.0},
    ])
    pub = mod._published_by_group()
    assert pub[("index_usa", "QQQ")] == {"n": 2, "peso": pytest.approx(3.0)}
    assert pub[("index_usa", "SPY")] == {"n": 1, "peso": pytest.approx(5.0)}


def test_dos_fechas_solo_cuenta_la_mas_reciente(tmp_path, monkeypatch):
    mod, hist = _load_module(tmp_path, monkeypatch)
    _csv(tmp_path, hist, "wyckoff_usa_index_distribution", [
        # historico: no debe contar
        {"date": "2026-11-12", "group": "QQQ", "weight": 1.0},
        {"date": "2026-11-12", "group": "QQQ", "weight": 2.0},
        {"date": "2026-11-12", "group": "QQQ", "weight": 3.0},
        # sesion actual: 2 filas
        {"date": "2026-11-13", "group": "QQQ", "weight": 10.0},
        {"date": "2026-11-13", "group": "QQQ", "weight": 20.0},
    ])
    pub = mod._published_by_group()
    assert pub[("index_usa", "QQQ")] == {"n": 2, "peso": pytest.approx(30.0)}


def test_sin_columna_date_acumula_compatibilidad(tmp_path, monkeypatch):
    """Fixture antigua sin 'date': se acumula todo (backward-compat)."""
    mod, hist = _load_module(tmp_path, monkeypatch)
    _csv(tmp_path, hist, "wyckoff_usa_index_distribution", [
        {"group": "QQQ", "weight": 1.0},
        {"group": "QQQ", "weight": 2.0},
    ])
    pub = mod._published_by_group()
    assert pub[("index_usa", "QQQ")] == {"n": 2, "peso": pytest.approx(3.0)}
