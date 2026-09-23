"""E.1 - Validacion E2E compute_nipc_contractual contra §12.5.

Encadena el pipeline contractual via
src.institutional_accumulation.pipeline_contractual.run_contractual_nipc
y reconcilia el resultado con la evidencia auditada §12.5.

Salida esperada (identica a §12.5):
  delta_radar_rows = 553321
  nipc_total       = -4316734936.0
  nipc_sole        = -32484713330.0
  nipc_dfnd        = +28317652408.0
  nipc_otr         = -149674014.0
"""
from __future__ import annotations
import json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"D:\Macro_Sectorial")
sys.path.insert(0, str(ROOT))

from src.institutional_accumulation.pipeline_contractual import (
    run_contractual_nipc)

OUT = ROOT / "outputs" / "audit" / "iae_e1_contractual_nipc"

EXPECTED = {
    "delta_radar_rows": 553321,
    "nipc_total": -4316734936.0,
    "nipc_sole":  -32484713330.0,
    "nipc_dfnd":   28317652408.0,
    "nipc_otr":    -149674014.0,
}


def git_head():
    try:
        r = subprocess.run(["git","rev-parse","HEAD"], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def reconcile(result, expected):
    checks = []
    def check(name, actual, exp):
        ok = abs(actual - exp) <= 0 if isinstance(exp, (int, float)) \
             else actual == exp
        checks.append({"name": name, "actual": actual, "expected": exp, "pass": ok})

    check("delta_radar_rows", result["delta_radar_rows"], expected["delta_radar_rows"])
    check("nipc_total", result["nipc_total"], expected["nipc_total"])
    check("nipc_sole",  result["nipc_sole"],  expected["nipc_sole"])
    check("nipc_dfnd",  result["nipc_dfnd"],  expected["nipc_dfnd"])
    check("nipc_otr",   result["nipc_otr"],   expected["nipc_otr"])
    check("evidence_class", result.get("evidence_class"), "CONTRACTUAL")
    return checks


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("E.1 - VALIDACION E2E compute_nipc_contractual vs §12.5")
    print("=" * 72)

    print("\n--- Ejecucion 1 ---")
    r1 = run_contractual_nipc()
    checks1 = reconcile(r1, EXPECTED)

    print("\n--- Ejecucion 2 (determinismo) ---")
    r2 = run_contractual_nipc()
    det_total = (r1["nipc_total"] == r2["nipc_total"])
    det_rows  = (r1["delta_radar_rows"] == r2["delta_radar_rows"])

    print("\n" + "=" * 72)
    print("RECONCILIACION §12.5")
    print("=" * 72)
    for c in checks1:
        m = "PASS" if c["pass"] else "FAIL"
        print(f"  [{m}] {c['name']:20s} actual={c['actual']}  esperado={c['expected']}")
    print(f"  [{'PASS' if det_total else 'FAIL'}] determinismo_nipc_total")
    print(f"  [{'PASS' if det_rows else 'FAIL'}] determinismo_delta_radar_rows")

    all_pass = all(c["pass"] for c in checks1) and det_total and det_rows

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = OUT / (ts + "_iae_e1.json")
    payload = {
        "run_id": ts,
        "git_head": git_head(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "expected_12_5": EXPECTED,
        "result_summary": {k: r1.get(k) for k in (
            "periods","delta_full_rows","delta_radar_rows",
            "target_q4_size","target_q1_size","feasibility",
            "nipc_total","nipc_sole","nipc_dfnd","nipc_otr",
            "n_delta_observable","coverage_status","coverage_quality",
            "status","evidence_class")},
        "determinism": {"nipc_total": det_total, "delta_radar_rows": det_rows},
        "checks": checks1,
        "passed": all_pass,
    }
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("\nJSON: " + str(out))
    print("RESULTADO: " + ("PASS" if all_pass else "FAIL"))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())