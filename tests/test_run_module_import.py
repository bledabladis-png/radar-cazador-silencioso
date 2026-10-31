# -*- coding: utf-8 -*-
"""Centinela: run.py resuelve todos sus imports de modulo.

Regresion 2026-10-31: el commit 3aa51d51 movio USA_INDEX_ETFS y
EU_INDEX_ETFS de indicators/wyckoff_holdings_report.py a
config/tickers.py, pero run.py:21 seguia importandolas del modulo
antiguo. CI verde (ningun test importa run.py), E2E lo destapo:
ImportError al arrancar el pipeline.

Este test cierra la clase 1fb791ab aplicada a run.py: consumidor
(run.py) no verificado por la suite, bug solo visible en produccion.

Coste ~1-2s (run.py importa casi todo el sistema).
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_run_import_resuelve():
    """import run no lanza. Fuerza la resolucion de todos los imports."""
    import run
    assert hasattr(run, "main")
    assert callable(run.main)
    # Constantes que deben estar accesibles en el namespace de run
    assert hasattr(run, "USA_INDEX_ETFS")
    assert hasattr(run, "EU_INDEX_ETFS")
    assert hasattr(run, "compute_wyckoff_snapshot")


def test_constantes_index_etfs_vienen_de_config():
    """USA_INDEX_ETFS y EU_INDEX_ETFS se importan de config.tickers.

    AST check: en run.py y src/pipeline/wyckoff_holdings.py, ningun
    import de estas constantes puede venir de
    indicators/wyckoff_holdings_report (fuente unica es config.tickers).
    """
    modulos = [
        "run.py",
        "src/pipeline/wyckoff_holdings.py",
    ]
    prohibidas = {"USA_INDEX_ETFS", "EU_INDEX_ETFS"}
    origenes_prohibidos = {
        "indicators.wyckoff_holdings_report",
        "indicators/wyckoff_holdings_report",
    }
    violaciones = []
    for rel in modulos:
        src = (ROOT / rel).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = node.module or ""
            if module not in origenes_prohibidos:
                continue
            for alias in node.names:
                if alias.name in prohibidas:
                    violaciones.append(
                        f"{rel}:{node.lineno} importa {alias.name} de "
                        f"{module} (debe venir de config.tickers)"
                    )
    assert not violaciones, "\n".join(violaciones)
