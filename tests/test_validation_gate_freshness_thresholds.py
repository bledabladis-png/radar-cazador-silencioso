# -*- coding: utf-8 -*-
"""Centinela: validation_gate usa FRESHNESS_* de config.settings, no hardcoded.

AUDIT-20261104-DECISION-FRESHNESS-POR-BLOQUE. Bug fixeado:
validation_gate hardcodeaba age > 14 (FINRA) y age > 5 (CBOE) en vez
de usar FRESHNESS_FINRA[1] y FRESHNESS_CBOE[1]. El umbral STALE
de FINRA es 28, no 14: la advertencia saltaba cuando el dato aun
era CURRENT segun el render. Divergencia silenciosa entre render
y gate.

Este test verifica por AST que:
  1. validation_gate importa FRESHNESS_CBOE y FRESHNESS_FINRA.
  2. Usa FRESHNESS_FINRA[1] y FRESHNESS_CBOE[1] en los checks.
  3. No hay literales 14 ni 5 comparados con age en el source.

Si falla, alguien reintrodujo un hardcoded. Ver DECISION-FRESHNESS-POR-BLOQUE.
"""
from pathlib import Path

VG = Path(__file__).resolve().parent.parent / "src" / "pipeline" / "validation_gate.py"


def _src():
    return VG.read_text(encoding="utf-8")


def test_importa_freshness_de_config():
    src = _src()
    assert "from config.settings import" in src
    linea_import = [ln for ln in src.splitlines() if "from config.settings import" in ln]
    assert linea_import, "falta import de config.settings"
    assert "FRESHNESS_FINRA" in linea_import[0]
    assert "FRESHNESS_CBOE" in linea_import[0]


def test_usa_finra_1_no_14():
    src = _src()
    assert "FRESHNESS_FINRA[1]" in src, "no usa FRESHNESS_FINRA[1]"
    assert "if age > 14:" not in src, "hardcoded 14 reintroducido (FINRA)"


def test_usa_cboe_1_no_5():
    src = _src()
    assert "FRESHNESS_CBOE[1]" in src, "no usa FRESHNESS_CBOE[1]"
    assert "if age > 5:" not in src, "hardcoded 5 reintroducido (CBOE)"


def test_import_fuera_de_try():
    """Regresion 2026-11-04: import dentro de try rompio py_compile."""
    src = _src()
    assert "    try:" + chr(10) + "        from config.settings" not in src, (
        "import de config.settings dentro de try: puede romper el balance except/finally"
    )


def test_umbrales_coinciden_con_config():
    """Los umbrales referenciados existen y son (CURRENT, RECENT, STALE)."""
    from config.settings import FRESHNESS_FINRA, FRESHNESS_CBOE
    assert isinstance(FRESHNESS_FINRA, tuple) and len(FRESHNESS_FINRA) == 3
    assert isinstance(FRESHNESS_CBOE, tuple) and len(FRESHNESS_CBOE) == 3
    assert FRESHNESS_FINRA[1] > FRESHNESS_FINRA[0], "RECENT debe ser > CURRENT"
    assert FRESHNESS_FINRA[2] > FRESHNESS_FINRA[1], "STALE debe ser > RECENT"
    assert FRESHNESS_CBOE[1] > FRESHNESS_CBOE[0]
    assert FRESHNESS_CBOE[2] > FRESHNESS_CBOE[1]

