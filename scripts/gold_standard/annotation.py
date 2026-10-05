# -*- coding: utf-8 -*-
"""Generacion de CSVs vacios y mapping para el panel de anotadores.

Especificacion (protocolo v7, secciones 2.3, 2.6, 3.6):
  - blind_id aleatorio no secuencial en rango [10^8, 10^8+n).
  - Un fichero mapping.csv con (blind_id, ticker, t, muestra, sector,
    periodo). NO se entrega a anotadores.
  - Un CSV por anotador con (blind_id, label, confidence,
    technical_ineligible). Vacio salvo blind_id.
  - Orden aleatorio distinto por anotador, determinista por semilla.

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.gold_standard.constants import SEED_GLOBAL

BLIND_ID_MIN = 10**8
BLIND_ID_MAX = 10**9 - 1

CSV_MAPPING_COLS = [
    "blind_id", "ticker", "t", "muestra",
    "sector", "periodo", "estrato_sector", "estrato_periodo",
]

CSV_ANOTADOR_COLS = [
    "blind_id", "label", "confidence", "technical_ineligible",
]

ANOTADORES = ("anotador_1", "anotador_2", "anotador_3")


@dataclass
class BlindMuestra:
    mapping: pd.DataFrame
    orden_por_anotador: dict[str, np.ndarray]


def _generar_blind_ids(n: int, seed: int) -> np.ndarray:
    """IDs unicos en [10^8, 10^9), muestreados sin reemplazo.

    Dictamen P0.5: IDs aleatorios, no permutacion de rango.
    No materializar el rango (900M enteros = 7.2 GB).
    Muestreo directo con resolucion de colisiones.
    """
    if n <= 0:
        return np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    vistos: set[int] = set()
    while len(vistos) < n:
        falta = n - len(vistos)
        # batch con margen: si quedan pocos por sacar, tirar mas
        tam = max(falta * 2, 100)
        batch = rng.integers(BLIND_ID_MIN, BLIND_ID_MAX, size=tam)
        vistos.update(int(x) for x in batch)
    ids = np.fromiter(vistos, dtype=np.int64, count=len(vistos))[:n]
    return ids


def build_blind_muestra(
    muestra: pd.DataFrame,
    seed: int = SEED_GLOBAL,
) -> BlindMuestra:
    """Asigna blind_id a la muestra y prepara orden por anotador.

    muestra: DataFrame con columnas ticker, t, muestra (A/B).
             Puede incluir sector, periodo, estrato_sector, etc.
    """
    if muestra.empty:
        raise ValueError("Muestra vacia")
    if muestra["ticker"].isna().any() or muestra["t"].isna().any():
        raise ValueError("Muestra contiene ticker/t nulos")
    dup = muestra.duplicated(subset=["ticker", "t"]).any()
    if dup:
        raise ValueError("Muestra contiene (ticker, t) duplicados")

    n = len(muestra)
    blind_ids = _generar_blind_ids(n, seed)

    df = muestra.copy().reset_index(drop=True)
    df.insert(0, "blind_id", blind_ids)

    # Mapping con las columnas disponibles
    mapping_cols = [c for c in CSV_MAPPING_COLS if c in df.columns]
    mapping = df[mapping_cols].sort_values("blind_id").reset_index(drop=True)

    # Orden por anotador: permutacion distinta, determinista
    orden = {}
    for i, nombre in enumerate(ANOTADORES):
        rng = np.random.default_rng(seed + 1000 + i)
        orden[nombre] = rng.permutation(n)

    return BlindMuestra(mapping=mapping, orden_por_anotador=orden)


def write_mapping_csv(
    blind: BlindMuestra,
    out_path: Path,
) -> None:
    """Escribe mapping.csv (no se entrega a anotadores)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blind.mapping.to_csv(
        out_path, index=False, encoding="utf-8", lineterminator="\n",
    )


def write_anotador_csv(
    blind: BlindMuestra,
    anotador: str,
    out_path: Path,
) -> None:
    """Escribe CSV vacio para un anotador, en su orden aleatorio."""
    if anotador not in blind.orden_por_anotador:
        raise ValueError(
            f"Anotador desconocido: {anotador}. "
            f"Validos: {list(ANOTADORES)}"
        )
    orden = blind.orden_por_anotador[anotador]
    ids_ordenados = blind.mapping["blind_id"].to_numpy()[orden]
    df = pd.DataFrame({
        "blind_id": ids_ordenados,
        "label": ["" for _ in range(len(ids_ordenados))],
        "confidence": ["" for _ in range(len(ids_ordenados))],
        "technical_ineligible": ["" for _ in range(len(ids_ordenados))],
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        out_path, index=False, encoding="utf-8", lineterminator="\n",
    )


def write_all_csvs(
    blind: BlindMuestra,
    out_dir: Path,
) -> dict[str, Path]:
    """Escribe mapping.csv y los 3 CSV de anotador. Devuelve rutas."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    p_map = out_dir / "mapping.csv"
    write_mapping_csv(blind, p_map)
    paths["mapping"] = p_map
    for nombre in ANOTADORES:
        p = out_dir / f"annotation_{nombre}.csv"
        write_anotador_csv(blind, nombre, p)
        paths[nombre] = p
    return paths



def write_paquetes_anotadores(
    blind: "BlindMuestra",
    out_dir: Path,
    pngs_por_anotador: dict[str, dict[int, Path]] | None = None,
) -> dict:
    """Escribe estructura separada: ADMIN/ + ANNOTATOR_N/.

    Dictamen P1.4: el paquete de cada anotador NO debe contener
    ninguna ruta administrativa (mapping).

    pngs_por_anotador: dict {anotador: {blind_id: path_png}}.
        Si None, no copia PNGs; solo CSVs.

    Devuelve dict con rutas escritas.
    """
    import json
    import shutil

    out_dir.mkdir(parents=True, exist_ok=True)
    admin_dir = out_dir / "ADMIN"
    admin_dir.mkdir(exist_ok=True)

    # 1. Admin: mapping + manifest
    mapping_path = admin_dir / "mapping.csv"
    write_mapping_csv(blind, mapping_path)

    manifest_admin = {
        "n_casos": int(len(blind.mapping)),
        "anotadores": list(ANOTADORES),
        "orden_por_anotador_distinto": True,
    }
    manifest_path = admin_dir / "manifest_admin.json"
    manifest_path.write_text(
        json.dumps(manifest_admin, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 2. Por anotador
    rutas = {"admin": {"mapping": mapping_path, "manifest": manifest_path}}
    for i, nombre in enumerate(ANOTADORES, 1):
        an_dir = out_dir / f"ANNOTATOR_{i}"
        an_dir.mkdir(exist_ok=True)

        # CSV del anotador
        csv_path = an_dir / f"annotation_{nombre}.csv"
        write_anotador_csv(blind, nombre, csv_path)

        # PNGs en su orden aleatorio, con nombre blind_id (sin mapa)
        png_dir = an_dir / "cases"
        if pngs_por_anotador is not None and nombre in pngs_por_anotador:
            png_dir.mkdir(exist_ok=True)
            for bid, src in pngs_por_anotador[nombre].items():
                dst = png_dir / f"case_{bid}.png"
                if src != dst:
                    shutil.copyfile(src, dst)

        rutas[nombre] = {
            "csv": csv_path,
            "casos_dir": png_dir if pngs_por_anotador else None,
        }

    return rutas
