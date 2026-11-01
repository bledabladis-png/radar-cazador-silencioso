# -*- coding: utf-8 -*-
"""FASE2-BARRIDO-CONSUMIDORES-ROTOS (B1-RUN-IMPORT-REGRESION).

Cierra la clase 1fb791ab aplicada a entry points: la suite no importaba
run.py, ni los scripts de produccion, ni los scripts de audit invocados
desde CI. Cualquier refactor de firma o movimiento de constante puede
romper un consumidor no verificado.

Este test importa cada uno de los 27 entry points invocados por
.github/workflows/*.yml, verificando que el modulo carga sin
AttributeError/ImportError/NameError y sin disparar SystemExit a nivel
de import (lo que indicaria falta de if __name__ guard o un
parse_args()/sys.exit() mal ubicado).

Cualquier script nuevo anadido a CI debe registrarse en
CI_ENTRYPOINTS. El heuristico
test_ci_entrypoints_detecta_scripts_workflow_no_registrados verifica
que .github/workflows/*.yml no invoca ningun script ausente del
registro.
"""
import importlib.util
import pathlib
import re
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CI_ENTRYPOINTS = [
    "run.py",
    "scripts/verifica_gobernanza.py",
    "scripts/audit/p1_h02_cobertura_efectiva.py",
    "scripts/build_catalog_csvs.py",
    "scripts/guard_coverage.py",
    "scripts/issue_manager.py",
    "scripts/pipeline_gate.py",
    "scripts/post_receipt_audit.py",
    "scripts/qqq_returns_yahoo.py",
    "scripts/regenerate_radar_catalog.py",
    "scripts/update_cboe.py",
    "scripts/write_completion_attempt.py",
    "scripts/write_completion_receipt.py",
    "scripts/health_check.py",
    "scripts/update_european_holdings.py",
    "scripts/update_index_holdings.py",
    "scripts/update_macro_manual.py",
    "scripts/update_qqq_sec_flow.py",
    "scripts/regenerate_cusip_crosswalk.py",
    "scripts/update_sec_13f.py",
    "scripts/update_sec_nport_data.py",
    "scripts/update_sector_holdings.py",
    "validation/monitor_double_counting.py",
    "validation/qqq_sec_nport_cross_validation.py",
    "validation/validate_history_quality.py",
    "data/providers/qqq_nport_flow.py",
    "data/providers/sec_nport_quarters_position_change.py",
]


def _import_from_path(relpath):
    path = ROOT / relpath
    assert path.exists(), f"Entry point no existe: {relpath}"
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", relpath)
    spec = importlib.util.spec_from_file_location(f"_ci_{safe}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit as e:
        pytest.fail(
            f"El modulo {relpath} llamo SystemExit({e.code}) a nivel "
            f"de import. Falta if __name__ guard, o un parse_args() "
            f"/ sys.exit() mal ubicado."
        )
    return mod


@pytest.mark.parametrize("relpath", CI_ENTRYPOINTS)
def test_ci_entrypoint_importa_limpio(relpath):
    """Cada entry point importa sin lanzar excepcion."""
    _import_from_path(relpath)


def test_ci_entrypoints_detecta_scripts_workflow_no_registrados():
    """Si .github/workflows invoca un py no registrado, fallar."""
    wf_dir = ROOT / ".github" / "workflows"
    assert wf_dir.exists(), "No hay .github/workflows"
    referenced = set()
    pat = re.compile(r"(?:py|python3?)\s+(?:-m\s+)?([\w/\.]+\.py)")
    for wf in wf_dir.glob("*.yml"):
        text = wf.read_text(encoding="utf-8", errors="replace")
        for m in pat.finditer(text):
            referenced.add(m.group(1))
    missing = referenced - set(CI_ENTRYPOINTS)
    assert not missing, (
        "Scripts invocados por workflows no registrados en "
        "CI_ENTRYPOINTS: " + str(sorted(missing)) + ". "
        "Anadelos al registro y a su test de import."
    )