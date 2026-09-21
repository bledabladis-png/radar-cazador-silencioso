"""Tests de caracterizacion para iae_pipeline.report.

Solo la funcion `report` es testeable sin fixtures externos
(load_canonical y build_identities requieren parquets 13F).
Se verifica el comportamiento del reporte por etapa sin tocar
el script ni sus dependencias.

Nota: importar scripts.iae_pipeline inyecta sys.path.insert ROOT
en sys.modules. Es el unico efecto colateral; no altera el state
global de tests (mismo ROOT que ya usa el proyecto).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\Macro_Sectorial")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util


def _load_pipeline_module():
    """Carga scripts/iae_pipeline.py como modulo (no paquete)."""
    spec = importlib.util.spec_from_file_location(
        "iae_pipeline", ROOT / "scripts" / "iae_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pipe = _load_pipeline_module()


def _empty_snap():
    return {
        "INFOTABLE": pd.DataFrame({
            "CUSIP": [],
            "SSHPRNAMTTYPE": [],
            "PUTCALL": [],
            "TITLEOFCLASS": [],
        })
    }


def _one_row_snap():
    return {
        "INFOTABLE": pd.DataFrame({
            "CUSIP": ["123456789"],
            "SSHPRNAMTTYPE": ["SH"],
            "PUTCALL": [None],
            "TITLEOFCLASS": ["COM"],
        })
    }


def test_report_snapshot_vacio(capsys):
    """Caso borde: snapshot sin filas. No debe romper."""
    snap = _empty_snap()
    pipe.report("2026-03-31", "2026Q1", snap, {}, pd.DataFrame())
    out = capsys.readouterr().out
    assert "IAE PIPELINE - 2026Q1" in out
    assert "2026-03-31" in out
    assert "filas INFOTABLE:      0" in out
    assert "Etapa 1" in out
    assert "Etapa 2" in out
    assert "Etapa 3" in out


def test_report_con_datos_muestra_pct_operational(capsys):
    snap = _one_row_snap()
    ids = {"123456789": {"security_resolution_status": "CANONICAL"}}
    oper = pd.DataFrame({"TITLEOFCLASS": ["COM"]})
    pipe.report("2026-03-31", "2026Q1", snap, ids, oper)
    out = capsys.readouterr().out
    assert "filas INFOTABLE:      1" in out
    assert "CUSIPs unicos:        1" in out
    assert "filas:                1" in out
    assert "CANONICAL" in out
    assert "pct operational:      100.0%" in out
    assert "Top 10 TITLEOFCLASS" in out


def test_report_u51_vacio_no_muestra_pct(capsys):
    """Si u51 vacio (ningun SH + PUTCALL NULL), no imprime pct."""
    snap = {
        "INFOTABLE": pd.DataFrame({
            "CUSIP": ["123"],
            "SSHPRNAMTTYPE": ["PRN"],  # no SH
            "PUTCALL": [None],
            "TITLEOFCLASS": ["COM"],
        })
    }
    pipe.report("2026-03-31", "2026Q1", snap, {}, pd.DataFrame())
    out = capsys.readouterr().out
    assert "pct operational" not in out
    assert "Top 10 TITLEOFCLASS" not in out


def test_report_identities_status_ordenados_descendente(capsys):
    """Etapa 2 ordena por frecuencia descendente."""
    snap = _empty_snap()
    ids = {
        "A": {"security_resolution_status": "CANONICAL"},
        "B": {"security_resolution_status": "CANONICAL"},
        "C": {"security_resolution_status": "OBSERVED_ONLY"},
        "D": {"security_resolution_status": "UNRESOLVED"},
    }
    pipe.report("2026-03-31", "2026Q1", snap, ids, pd.DataFrame())
    out = capsys.readouterr().out
    # CANONICAL (2) debe aparecer antes que OBSERVED_ONLY (1) y UNRESOLVED (1)
    i_canon = out.index("CANONICAL")
    i_obs = out.index("OBSERVED_ONLY")
    i_unres = out.index("UNRESOLVED")
    assert i_canon < i_obs
    assert i_canon < i_unres
