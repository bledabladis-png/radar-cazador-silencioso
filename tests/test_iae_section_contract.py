# -*- coding: utf-8 -*-
"""Contrato AST: compute_iae_section <-> render_iae_section.

Clase 1fb791ab aplicada a IAE. Productor: compute_iae_section
(src/pipeline/iae_section.py). Tiene 4 ramas: OK, STALE
(insufficient_quarters), STALE (official_list_pending), ERROR. Las 4
publican el mismo conjunto de 23 claves (verificado 2026-10-30).

Consumidor: render_iae_section (src/report/iae.py). Lee 20 de las 23
claves via iae_section.get(...). Las 3 restantes (iso_previous,
iso_current, evidence_class) no se publican en el reporte (para
write_completion_receipt, validators, etc.).

Test cubre:
1. Downstream: render lee X => X en return (union de 4 ramas).
2. Consistencia entre ramas: las 4 ramas tienen el mismo set.
3. Regresion: campos criticos siguen leidos.
"""
from pathlib import Path

from tests._dict_contract import (
    producer_all_return_keys,
    consumer_read_keys,
    assert_render_subset_return,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PRODUCER = REPO_ROOT / "src" / "pipeline" / "iae_section.py"
RENDER = REPO_ROOT / "src" / "report" / "iae.py"


def test_render_iae_no_lee_campos_ausentes_en_return():
    """render_iae_section lee X => X en alguna rama del return."""
    assert_render_subset_return(
        producer_module=PRODUCER, producer_fn="compute_iae_section",
        consumer_module=RENDER, consumer_fn="render_iae_section",
        consumer_var="iae_section",
    )


def test_iae_section_returns_consistentes_entre_ramas():
    """Las 4 ramas de compute_iae_section tienen el mismo set de claves.

    Si una rama nueva (p.ej. PARTIAL) se anade sin actualizar las
    demas, el render puede romperse en runs que caigan en la rama
    nueva. Este test lo detecta antes de que llegue a produccion.
    """
    all_keys = producer_all_return_keys(PRODUCER, "compute_iae_section")
    assert len(all_keys) >= 4, (
        f"esperadas 4+ ramas (OK, STALE x2, ERROR), "
        f"{len(all_keys)} encontradas"
    )
    first = all_keys[0]
    for i, keys in enumerate(all_keys[1:], start=2):
        assert keys == first, (
            f"Rama {i} de compute_iae_section tiene claves distintas:\n"
            f"  rama 1: {sorted(first)}\n"
            f"  rama {i}: {sorted(keys)}\n"
            f"  solo en rama 1: {sorted(first - keys)}\n"
            f"  solo en rama {i}: {sorted(keys - first)}\n"
            f"El render debe funcionar en todas las ramas. Anadir las "
            f"claves faltantes a la rama."
        )


def test_render_iae_lee_campos_criticos():
    """Regresion: campos que el reporte no puede perder.

    - status: distingue OK / STALE / ERROR.
    - catalog_coverage_warning: dispara [WARN] o [INFO].
    - catalog_coverage_declared: valor observable/total.
    - coverage_quality: COMPLETE / PARTIAL / UNAVAILABLE.
    """
    reads = consumer_read_keys(
        RENDER, "render_iae_section", "iae_section")
    for campo in ("status", "catalog_coverage_warning",
                  "catalog_coverage_declared", "coverage_quality"):
        assert campo in reads, (
            f"render_iae_section dejo de leer '{campo}'. El reporte "
            f"perderia informacion critica del bloque IAE. Si es "
            f"intencional, borrar este test con justificacion."
        )
