"""Tests del fix fail-close en write_artifact_with_manifest.

Bug (except-audit 2026-10-28): write_artifact_with_manifest atrapaba
cualquier excepcion y devolvia {} silencioso. Un parquet nuevo sin
manifest rompia el contrato sin que ningun step fallara. El receipt
del run y el guard de write_completion_attempt no se enteraban hasta
el siguiente slot.

Fix: `fail_on_invalid` (ya existente como parametro) se consulta en
el except. Si True, raise (fail-close). Si False, return {}
(best-effort, scripts de mantenimiento).

Tambien fix de 1 char en data_loader.py:343: `str(e)` referenciaba
el except externo (proveedor principal) en vez del except del backup
(`str(be)`). Reportaba el error equivocado.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import write_artifact_with_manifest


def _df_minimo():
    idx = pd.DatetimeIndex(["2026-10-23", "2026-10-26"])
    return pd.DataFrame({"Close": [1.0, 2.0]}, index=idx)


def test_fail_on_invalid_false_devuelve_vacio(tmp_path, monkeypatch):
    """Con fail_on_invalid=False (default), best-effort: {}."""
    # Forzar fallo dentro del try del writer via monkeypatch de to_parquet.
    # El bug del audit: except atrapaba y devolvia {} silencioso.
    def _boom(*a, **k):
        raise RuntimeError("simulated parquet write failure")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", _boom)

    out = write_artifact_with_manifest(
        _df_minimo(), str(tmp_path / "x.parquet"),
        source="test", reference_date=None, run_id="20261028_120000",
        fail_on_invalid=False,
    )
    assert out == {}


def test_fail_on_invalid_true_propaga_bien(tmp_path, monkeypatch):
    """Mismo fallo simulado, pero con fail_on_invalid=True: raise."""
    def _boom(*a, **k):
        raise RuntimeError("simulated parquet write failure")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", _boom)

    with pytest.raises(RuntimeError, match="simulated parquet write failure"):
        write_artifact_with_manifest(
            _df_minimo(), str(tmp_path / "x.parquet"),
            source="test", reference_date=None, run_id="20261028_120000",
            fail_on_invalid=True,
        )


def test_callers_pipeline_pasan_fail_on_invalid_true():
    """AST: los 2 callers del pipeline pasan fail_on_invalid=True.

    Si alguien anade un nuevo write_artifact_with_manifest en el
    pipeline sin fail_on_invalid=True, el parquet+manifest puede
    quedar inconsistente de forma silenciosa. Este test lo detecta.
    """
    callers_pipeline = [
        "src/data_loader.py",
        "src/stock_data_loader.py",
    ]
    for rel in callers_pipeline:
        src = (ROOT / rel).read_text(encoding="utf-8-sig", errors="replace")
        tree = ast.parse(src)
        encontrado = False
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "write_artifact_with_manifest"):
                tiene_kw = any(
                    kw.arg == "fail_on_invalid"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                    for kw in node.keywords
                )
                assert tiene_kw, (
                    f"{rel}:{node.lineno} llama a "
                    f"write_artifact_with_manifest sin fail_on_invalid=True. "
                    f"El pipeline debe ser fail-close."
                )
                encontrado = True
        assert encontrado, f"{rel}: no se encontro la llamada (regresion?)"


def test_no_except_pass_silencioso_en_run_py_gate_y_receipts():
    """Regresion: run.py no debe tener `except Exception: pass` en la
    escritura del gate ni en la escritura del receipt post-mortem.
    """
    src = (ROOT / "run.py").read_text(encoding="utf-8-sig", errors="replace")
    # Buscar el patron exacto `except Exception:\n        pass`
    malo = "except Exception:" + chr(10) + "        pass"
    malo_crlf = "except Exception:" + chr(13) + chr(10) + "        pass"
    assert malo not in src, (
        "run.py tiene `except Exception: pass` (silencioso). "
        "Anadir print del error."
    )
    assert malo_crlf not in src, (
        "run.py tiene `except Exception: pass` (silencioso). "
        "Anadir print del error."
    )


def test_data_loader_backup_usa_str_be(tmp_path):
    """Regresion del fix de 1 char: `str(be)` en el except anidado del
    backup, no `str(e)`. Verifica leyendo el AST del fichero.
    """
    src = (ROOT / "src" / "data_loader.py").read_text(encoding="utf-8-sig", errors="replace")
    # Debe contener str(be) al menos una vez (el fix)
    assert "batch_errors.append(str(be))" in src, (
        "data_loader.py: el except del backup no reporta str(be). "
        "Bug regresion: reportaba str(e) (error del proveedor principal)."
    )
    # Y el bug NO debe estar: en la linea del except be, no debe haber str(e)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.name == "be":
            for stmt in ast.walk(node):
                if (isinstance(stmt, ast.Call)
                        and isinstance(stmt.func, ast.Attribute)
                        and stmt.func.attr == "append"
                        and stmt.args
                        and isinstance(stmt.args[0], ast.Call)
                        and isinstance(stmt.args[0].func, ast.Name)
                        and stmt.args[0].func.id == "str"):
                    arg = stmt.args[0].args[0] if stmt.args[0].args else None
                    if isinstance(arg, ast.Name):
                        assert arg.id == "be", (
                            f"data_loader.py:{stmt.lineno}: dentro del "
                            f"except 'be' se usa str({arg.id}), deberia ser str(be)."
                        )
