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
import os
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Enmienda v1.1: fuente unica de validacion contractual.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.temporal_contracts.validation import (
    validate_artifact_contracts,
    STATUS_WARN as V_WARN,
    STATUS_BLOCKED as V_BLOCKED,
    STATUS_UNAVAILABLE as V_UNAVAILABLE,
)


GUARDED_MANIFESTS = (
    "data/stock_prices.parquet.manifest.json",
    "data/market_data.parquet.manifest.json",
)

DEFAULT_THRESHOLD = 0.95

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "outputs" / "state" / "guard_coverage_result.json"

RESULT_SCHEMA_VERSION = 1


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


def _read_target_session():
    """target_session declarado por el primer manifest legible (H19).

    None si ningun manifest declara expected_session.
    """
    for mpath in GUARDED_MANIFESTS:
        m, _ = _load_manifest(mpath)
        if not isinstance(m, dict):
            continue
        quality = m.get("quality")
        if isinstance(quality, dict):
            ts = quality.get("expected_session")
            if isinstance(ts, str) and ts:
                return ts
    return None


def _split_failure(failure):
    """Descompone '<manifest>: <motivo>' en dict estructurado.

    Si no hay separador, devuelve el string completo como reason.
    """
    s = str(failure)
    if ": " in s:
        manifest, _, reason = s.partition(": ")
        return {"manifest": manifest, "reason": reason}
    return {"manifest": "", "reason": s}


def _write_result(ok, threshold, failures, status):
    """Persiste el resultado del guard como artifact para el attempt.

    Contrato H19 (dictamen externo 2026-10-17):
    - schema_version y generated_at: detectar resultados obsoletos.
    - run_id / run_attempt: identidad de ejecucion.
    - target_session: extraido del manifest (no del entorno).
    - threshold: valor efectivo (respeta --threshold).
    - failures: strings (compat con tests existentes).
    - failures_detail: lista estructurada {manifest, reason}.
    - escritura atomica via tmp + os.replace: el lector nunca ve
      un JSON truncado.
    status: running (placeholder), passed o failed.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = os.environ.get("GITHUB_RUN_ID", "0")
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": status,
        "run_id": int(run_id) if run_id.isdigit() else 0,
        "run_attempt": int(run_attempt) if run_attempt.isdigit() else 1,
        "target_session": _read_target_session(),
        "generated_at": now,
        "threshold": float(threshold),
        "ok": bool(ok) if ok is not None else None,
        "failures": list(failures),
        "failures_detail": [_split_failure(f) for f in failures],
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESULT_PATH.with_name(RESULT_PATH.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, RESULT_PATH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    # H19: placeholder al inicio. Si el guard muere entre medias
    # (excepcion no capturada, timeout, kill), el attempt leera
    # status=running y podra registrarlo como causa de observabilidad,
    # en lugar de encontrar un resultado obsoleto o ninguno.
    try:
        _write_result(None, args.threshold, [], "running")
    except (OSError, ValueError) as _e:
        print("[WARN] no se pudo escribir placeholder inicial: {0}".format(_e))

    all_failures = []
    for manifest in GUARDED_MANIFESTS:
        try:
            reasons = _check_manifest(manifest, args.threshold)
        except Exception as exc:  # noqa: BLE001 - captura deliberada
            reasons = [
                "{0}: EXCEPTION {1}: {2}".format(
                    manifest, type(exc).__name__, exc
                )
            ]
        if reasons:
            for r in reasons:
                print("[FAIL] " + r)
            all_failures.extend(reasons)
        else:
            print("[OK] " + manifest)

    if all_failures:
        print("")
        print("Guard coverage: {0} fallo(s). Abortando.".format(len(all_failures)))
        _write_result(False, args.threshold, all_failures, "failed")
        return 1

    print("Guard coverage: todos OK.")
    _write_result(True, args.threshold, [], "passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())