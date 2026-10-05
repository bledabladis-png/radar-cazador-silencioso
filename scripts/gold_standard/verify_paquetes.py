# -*- coding: utf-8 -*-
"""Verificador post-run_capa1 --go.

Comprueba que los paquetes generados cumplen los requisitos del
auditor antes de entregar a anotadores.

Uso:
    py scripts/gold_standard/verify_paquetes.py
    py scripts/gold_standard/verify_paquetes.py --out-dir <ruta>

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

PNG_HEADER = b"\x89PNG\r\n\x1a\n"
PNG_IEND = b"IEND\xaeB`\x82"
EXPECTED_WIDTH = 1600
EXPECTED_HEIGHT = 900
N_MUESTRA_A = 400
N_MUESTRA_B = 200
N_TOTAL = N_MUESTRA_A + N_MUESTRA_B


def _png_dimensions(path: Path) -> tuple[int, int] | None:
    """Lee ancho/alto del IHDR del PNG sin PIL."""
    with path.open("rb") as f:
        header = f.read(33)
    if len(header) < 33:
        return None
    if header[:8] != PNG_HEADER:
        return None
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    return width, height


def _png_integrity(path: Path) -> bool:
    """Verifica header PNG + presencia de IEND en los ultimos bytes."""
    try:
        with path.open("rb") as f:
            header = f.read(8)
            if header != PNG_HEADER:
                return False
            # IEND en los ultimos 16 bytes
            f.seek(0, 2)
            size = f.tell()
            if size < 16:
                return False
            f.seek(size - 12)
            tail = f.read(12)
        return PNG_IEND in tail
    except OSError:
        return False


def verificar(out_dir: Path, verbose: bool = True) -> dict:
    t0 = time.time()
    rep = {"checks": {}, "errores": [], "out_dir": str(out_dir)}

    # 1. Estructura
    admin_dir = out_dir / "ADMIN"
    cases_master = out_dir / "_cases_master"
    rep["checks"]["admin_existe"] = admin_dir.exists()
    rep["checks"]["cases_master_existe"] = cases_master.exists()
    for i in (1, 2, 3):
        an_dir = out_dir / f"ANNOTATOR_{i}"
        rep["checks"][f"annotator_{i}_existe"] = an_dir.exists()

    # 2. Mapping
    mapping_path = admin_dir / "mapping.csv"
    if not mapping_path.exists():
        rep["errores"].append("mapping.csv no existe")
        rep["checks"]["ok"] = False
        return rep

    mapping = pd.read_csv(mapping_path)
    rep["checks"]["mapping_n"] = len(mapping)
    rep["checks"]["mapping_n_ok"] = len(mapping) == N_TOTAL
    rep["checks"]["mapping_blind_ids_unicos"] = bool(mapping["blind_id"].is_unique)
    rep["checks"]["mapping_sin_nan_esenciales"] = bool(
        mapping[["blind_id", "ticker", "t"]].notna().all().all()
    )

    # Muestras A y B
    if "muestra" in mapping.columns:
        n_a = int((mapping["muestra"] == "A").sum())
        n_b = int((mapping["muestra"] == "B").sum())
        rep["checks"]["mapping_n_A"] = n_a
        rep["checks"]["mapping_n_B"] = n_b
        rep["checks"]["mapping_n_A_ok"] = n_a == N_MUESTRA_A
        rep["checks"]["mapping_n_B_ok"] = n_b == N_MUESTRA_B

    # 3. PNGs en _cases_master
    if cases_master.exists():
        pngs = sorted(cases_master.glob("case_*.png"))
        rep["checks"]["pngs_master_n"] = len(pngs)
        rep["checks"]["pngs_master_n_ok"] = len(pngs) == N_TOTAL

        # Integridad de una muestra (todos seria lento)
        sample = pngs[::max(1, len(pngs) // 30)]  # ~30 PNGs
        integridad_ok = 0
        dims_ok = 0
        for p in sample:
            if _png_integrity(p):
                integridad_ok += 1
            d = _png_dimensions(p)
            if d == (EXPECTED_WIDTH, EXPECTED_HEIGHT):
                dims_ok += 1
        rep["checks"]["png_integridad_muestra"] = f"{integridad_ok}/{len(sample)}"
        rep["checks"]["png_dims_muestra"] = f"{dims_ok}/{len(sample)}"
        rep["checks"]["png_integridad_ok"] = integridad_ok == len(sample)
        rep["checks"]["png_dims_ok"] = dims_ok == len(sample)

    # 4. Por anotador
    for i, nombre in enumerate(("anotador_1", "anotador_2", "anotador_3"), 1):
        an_dir = out_dir / f"ANNOTATOR_{i}"
        csv_path = an_dir / f"annotation_{nombre}.csv"
        cases_dir = an_dir / "cases"
        rep["checks"][f"annotator_{i}_csv"] = csv_path.exists()
        rep["checks"][f"annotator_{i}_cases"] = cases_dir.exists()
        if cases_dir.exists():
            pngs = list(cases_dir.glob("case_*.png"))
            rep["checks"][f"annotator_{i}_n_pngs"] = len(pngs)
            rep["checks"][f"annotator_{i}_n_pngs_ok"] = len(pngs) == N_TOTAL
        # No debe haber mapping en ANNOTATOR
        rep["checks"][f"annotator_{i}_sin_mapping"] = not (
            an_dir / "mapping.csv"
        ).exists()
        rep["checks"][f"annotator_{i}_sin_manifest_admin"] = not (
            an_dir / "manifest_admin.json"
        ).exists()

    # 5. Manifest global
    manifest_path = admin_dir / "manifest.json"
    if manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            rep["checks"]["manifest_ok"] = True
            rep["checks"]["manifest_pngs_ok"] = m.get("n_pngs_ok", 0)
            rep["checks"]["manifest_pngs_fail"] = m.get("n_pngs_fail", 0)
        except Exception as e:
            rep["checks"]["manifest_ok"] = False
            rep["errores"].append(f"manifest.json no parseable: {e}")
    else:
        rep["checks"]["manifest_ok"] = False

    # 6. Resumen
    checks_bool = {k: v for k, v in rep["checks"].items()
                   if isinstance(v, bool)}
    rep["checks"]["todos_ok"] = all(checks_bool.values())
    rep["duracion_seg"] = round(time.time() - t0, 2)
    return rep


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path,
                   default=_ROOT / "outputs" / "gold_standard" / "capa1")
    args = p.parse_args(argv)

    rep = verificar(args.out_dir, verbose=True)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0 if rep["checks"].get("todos_ok") else 1


if __name__ == "__main__":
    sys.exit(main())