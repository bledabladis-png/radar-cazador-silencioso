#!/usr/bin/env python3
"""Guard de cobertura para manifests del pipeline.

Bloquea el step Commit si algun manifest protegido muestra:
  - coverage_pct_last < threshold
  - status == INVALID
  - last_date > expected_session
  - manifest ausente o JSON invalido
  - validacion contractual (enmienda v1.1): contrato critico incumplido

FU-002-bymarket (dictamen auditor 2026-09-24):
  Exencion condicional cuando la cobertura global es baja pero el
  universo es heterogeneo por mercado y todos los mercados activos
  tienen su propia cobertura >= threshold en su ultima sesion cerrada.
  Condiciones (C-2 y C-3 del dictamen):
    - quality.status debe ser VALID_WITH_MISSING (no INVALID).
    - quality.by_market debe existir.
    - Cada mercado con n>0 debe tener status VALID y
      coverage_at_session >= threshold (mismo threshold del guard).

Enmienda v1.1 (2026-10-16): se anade validacion contractual compartida
con write_completion_receipt.py. Ambas consumen
src.temporal_contracts.validation::validate_artifact_contracts.

Uso: python scripts/guard_coverage.py [--threshold 0.95]
Exit 0 si todos OK, exit 1 si alguno falla.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

# Enmienda v1.1: fuente unica de validacion contractual.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.temporal_contracts.validation import (
    validate_artifact_contracts,
    STATUS_OK as V_OK,
    STATUS_WARN as V_WARN,
    STATUS_BLOCKED as V_BLOCKED,
    STATUS_UNAVAILABLE as V_UNAVAILABLE,
)


GUARDED_MANIFESTS = (
    "data/stock_prices.parquet.manifest.json",
    "data/market_data.parquet.manifest.json",
)

DEFAULT_THRESHOLD = 0.95


def _load_manifest(path):
    """Devuelve (manifest_dict, error_str)."""
    p = Path(path)
    if not p.exists():
        return None, "manifest missing"
    try:
        raw = p.read_text(encoding="utf-8")
    except Exception as e:
        return None, "manifest unreadable: {0}".format(e)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, "manifest invalid JSON: {0}".format(e)
    if not isinstance(data, dict):
        return None, "manifest root is not a dict"
    return data, None


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _all_markets_valid(by_market, threshold):
    """True si cada mercado con n>0 cumple coverage_at_session >= threshold.

    FU-002-bymarket (dictamen auditor 2026-09-24):
      - C-3: usa el MISMO threshold que el guard, no un valor fijo.
      - Punto 5: mercados con status != VALID (incluye INVALID por
                 UNKNOWN con n>0) no eximen.
      - SKIP (n==0) se ignora.
      - Devuelve False si by_market ausente, vacio, o todos n==0.
    """
    if not isinstance(by_market, dict) or not by_market:
        return False

    n_active = 0
    n_valid = 0
    for market, info in by_market.items():
        if not isinstance(info, dict):
            return False
        n = info.get("n", 0)
        if n == 0:
            continue
        n_active += 1
        st = info.get("status")
        if st == "PENDING":
            continue
        if st != "VALID":
            return False
        n_valid += 1
        cov = info.get("coverage_at_session")
        if not isinstance(cov, (int, float)):
            return False
        if cov < threshold:
            return False
    return n_active > 0 and n_valid > 0


def _check_manifest(path, threshold):
    """Devuelve lista de motivos de fallo (vacia si OK)."""
    data, err = _load_manifest(path)
    if err is not None:
        return ["{0}: {1}".format(path, err)]

    quality = data.get("quality")
    if not isinstance(quality, dict):
        return ["{0}: quality block missing or not a dict".format(path)]

    reasons = []

    declared_sha = (data.get("artifact") or {}).get("sha256")
    if not isinstance(declared_sha, str) or not declared_sha:
        reasons.append("{0}: artifact.sha256 missing".format(path))
    else:
        parquet_path = Path(str(path).replace(".manifest.json", ""))
        if not parquet_path.exists():
            reasons.append("{0}: parquet missing ({1})".format(path, parquet_path))
        else:
            real_sha = _sha256_file(parquet_path)
            if real_sha.lower() != declared_sha.lower():
                reasons.append(
                    "{0}: sha256 mismatch (real={1}, declared={2})".format(
                        path, real_sha[:12], declared_sha[:12])
                )

    status = quality.get("status")
    if status == "INVALID":
        reasons.append("{0}: status=INVALID".format(path))
    elif not isinstance(status, str):
        reasons.append("{0}: status missing or non-string".format(path))

    cov = quality.get("coverage_pct_last")
    if not isinstance(cov, (int, float)):
        reasons.append("{0}: coverage_pct_last missing or non-numeric".format(path))
    elif cov < threshold:
        by_market = quality.get("by_market")
        if status == "VALID_WITH_MISSING" and _all_markets_valid(by_market, threshold):
            print("[EXEMPT] {0}: coverage_pct_last={1} < {2} "
                  "pero todos los mercados activos cumplen el threshold "
                  "en su ultima sesion cerrada".format(path, cov, threshold))
        else:
            reasons.append("{0}: coverage_pct_last={1} < {2}".format(path, cov, threshold))

    last = quality.get("last_date")
    expected = quality.get("expected_session")
    if not isinstance(last, str) or not isinstance(expected, str):
        reasons.append("{0}: last_date or expected_session missing".format(path))
    elif last > expected:
        reasons.append("{0}: last_date={1} > expected_session={2}".format(path, last, expected))

    # Enmienda v1.1: validacion contractual compartida.
    v = validate_artifact_contracts(data)
    if v.status == V_BLOCKED:
        for r in v.reasons:
            reasons.append("{0}: [CONTRACT] {1}".format(path, r))
    elif v.status == V_UNAVAILABLE:
        for r in v.reasons:
            reasons.append("{0}: [CONTRACT-UNAVAILABLE] {1}".format(path, r))
    elif v.status == V_WARN:
        # WARN no bloquea; se imprime para observabilidad.
        for r in v.reasons:
            print("[WARN] {0}: [CONTRACT] {1}".format(path, r))

    return reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    all_failures = []
    for manifest in GUARDED_MANIFESTS:
        reasons = _check_manifest(manifest, args.threshold)
        if reasons:
            for r in reasons:
                print("[FAIL] " + r)
            all_failures.extend(reasons)
        else:
            print("[OK] " + manifest)

    if all_failures:
        print("")
        print("Guard coverage: {0} fallo(s). Abortando.".format(len(all_failures)))
        return 1

    print("Guard coverage: todos OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())