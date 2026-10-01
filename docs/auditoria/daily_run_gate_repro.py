# -*- coding: utf-8 -*-
"""Demuestra el bug del gate del daily_run sin GitHub ni yfinance.

Contexto: ver docs/auditoria/daily_run_gate_discrepancia.md.

Ejecutar:

    py docs\\auditoria\\daily_run_gate_repro.py

Salida esperada:

    [_manifest_satisfies] parquet ausente, manifest valido
      -> resultado: False  <-- CI: gate nunca alcanza CURRENT
      -> en local: False (correcto, evita falso-CURRENT)

No depende de red. No toca el repo. Trabaja en un directorio temporal.
"""
import json
import sys
import tempfile
from pathlib import Path

# Importar el modulo del gate. No ejecuta main() por el guard.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import pipeline_gate as pg


def _manifest_sintetico(target_session):
    return {
        "quality": {
            "expected_session": target_session,
            "coverage_pct_last": 0.99,
        },
        "artifact": {
            "sha256": "0" * 64,
        },
    }


def main():
    target_session = "2026-09-30"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Manifest presente, parquet ausente (simula runner CI limpio).
        manifest = _manifest_sintetico(target_session)
        manifest_path = tmp / "data" / "stock_prices.parquet.manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        # Monkeypatch: PROJECT_ROOT -> tmp, MANIFEST_PATH relativo.
        pg.PROJECT_ROOT = tmp
        pg.MANIFEST_PATH = "data/stock_prices.parquet.manifest.json"

        result = pg._manifest_satisfies(manifest, target_session)

        print("[_manifest_satisfies] parquet ausente, manifest valido")
        print("  -> resultado: {0}".format(result))
        if result is False:
            print("  -> CI: gate nunca alcanza CURRENT (BUG)")
            print("  -> local: False es correcto (evita falso-CURRENT)")
            return 0
        else:
            print("  -> INESPERADO: el bug no se reproduce")
            return 1


if __name__ == "__main__":
    sys.exit(main())
