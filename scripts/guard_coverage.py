#!/usr/bin/env python3
"""Guard de cobertura para manifests del pipeline.

Bloquea el step Commit si algun manifest protegido muestra:
  - coverage_pct_last < threshold
  - status == INVALID
  - last_date > expected_session
  - manifest ausente o JSON invalido

FU-002-bymarket (dictamen auditor 2026-09-24):
  Exencion condicional cuando la cobertura global es baja pero el
  universo es heterogeneo por mercado y todos los mercados activos
  tienen su propia cobertura >= threshold en su ultima sesion cerrada.
  Condiciones (C-2 y C-3 del dictamen):
    - quality.status debe ser VALID_WITH_MISSING (no INVALID).
    - quality.by_market debe existir.
    - Cada mercado con n>0 debe tener status VALID y
      coverage_at_session >= threshold (mismo threshold del guard).

Uso: python scripts/guard_coverage.py [--threshold 0.95]
Exit 0 si todos OK, exit 1 si alguno falla.
"""
import argparse
import json
import sys
from pathlib import Path

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
    for market, info in by_market.items():
        if not isinstance(info, dict):
            return False
        n = info.get("n", 0)
        if n == 0:
            continue
        n_active += 1
        if info.get("status") != "VALID":
            return False
        cov = info.get("coverage_at_session")
        if not isinstance(cov, (int, float)):
            return False
        if cov < threshold:
            return False
    return n_active > 0


def _check_manifest(path, threshold):
    """Devuelve lista de motivos de fallo (vacia si OK)."""
    data, err = _load_manifest(path)
    if err is not None:
        return ["{0}: {1}".format(path, err)]

    quality = data.get("quality")
    if not isinstance(quality, dict):
        return ["{0}: quality block missing or not a dict".format(path)]

    reasons = []

    status = quality.get("status")
    if status == "INVALID":
        reasons.append("{0}: status=INVALID".format(path))
    elif not isinstance(status, str):
        reasons.append("{0}: status missing or non-string".format(path))

    cov = quality.get("coverage_pct_last")
    if not isinstance(cov, (int, float)):
        reasons.append("{0}: coverage_pct_last missing or non-numeric".format(path))
    elif cov < threshold:
        # FU-002-bymarket (dictamen auditor 2026-09-24, C-2 + C-3):
        # exencion condicional. C-2: solo si status == VALID_WITH_MISSING
        # (nunca si INVALID). C-3: _all_markets_valid usa este threshold.
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