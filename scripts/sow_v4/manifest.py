"""Manifiesto de congelacion del experimento SOW v4.1.

Seccion 2.2 del protocolo. Registra:
- SHA-256 del dataset
- Commit base del codigo
- Estado del arbol de trabajo (limpio o no)
- Version del protocolo
- Versiones de dependencias
- Timestamp de ejecucion

Regla bloqueante: si CODE_TREE_CLEAN != True o DATA_SHA256 no coincide,
la ejecucion aborta.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.sow_v4 import config


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hex de un fichero, leido en chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def git_rev_parse_head() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=config.ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def git_status_porcelain() -> str:
    out = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=config.ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def package_version(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return "not_installed"


def universe_fingerprint(tickers) -> tuple[int, str]:
    """Devuelve (count, sha256) de la lista ordenada de tickers."""
    sorted_tk = sorted(str(t) for t in tickers)
    payload = "\n".join(sorted_tk).encode("utf-8")
    return len(sorted_tk), hashlib.sha256(payload).hexdigest()


def build_manifest(
    experiment_tickers=None,
    regime_tickers=None,
) -> dict:
    if not config.DATA_PARQUET.exists():
        raise FileNotFoundError(f"No existe {config.DATA_PARQUET}")

    data_sha = sha256_file(config.DATA_PARQUET)
    porcelain = git_status_porcelain()
    tree_clean = (porcelain.strip() == "")
    code_base_commit = git_rev_parse_head()

    manifest = {
        "data_path": str(config.DATA_PARQUET.relative_to(config.ROOT)),
        "data_sha256": data_sha,
        "code_base_commit": code_base_commit,
        "code_tree_clean": tree_clean,
        "code_tree_porcelain": porcelain,
        "protocol_version": "v4.1",
        "python_version": sys.version,
        "pandas_version": package_version("pandas"),
        "numpy_version": package_version("numpy"),
        "scipy_version": package_version("scipy"),
        "statsmodels_version": package_version("statsmodels"),
        "fecha_ejecucion_utc": datetime.now(timezone.utc).isoformat(),
        "sector_map_csv": str(config.SECTOR_MAP_CSV.relative_to(config.ROOT)),

    }
    if experiment_tickers is not None:
        n_exp, sha_exp = universe_fingerprint(experiment_tickers)
        manifest["experiment_universe_count"] = n_exp
        manifest["experiment_universe_sha256"] = sha_exp

    if regime_tickers is not None:
        n_ref, sha_ref = universe_fingerprint(regime_tickers)
        manifest["regime_reference_count"] = n_ref
        manifest["regime_reference_sha256"] = sha_ref

    return manifest


def assert_manifest_valid(manifest: dict) -> None:
    if not manifest["code_tree_clean"]:
        raise RuntimeError(
            "ABORT: working tree no limpio.\n" + manifest["code_tree_porcelain"]
        )
    if not config.DATA_PARQUET.exists():
        raise FileNotFoundError(f"ABORT: no existe {config.DATA_PARQUET}")
    recomputed = sha256_file(config.DATA_PARQUET)
    if recomputed != manifest["data_sha256"]:
        raise RuntimeError(
            f"ABORT: DATA_SHA256 no coincide.\n"
            f"  registrado: {manifest['data_sha256']}\n"
            f"  recomputado: {recomputed}"
        )


def write_manifest(path: Path) -> None:
    manifest = build_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    out = config.OUT_DIR / "manifest.json"
    write_manifest(out)
    m = json.loads(out.read_text(encoding="utf-8"))
    print(json.dumps(m, indent=2))