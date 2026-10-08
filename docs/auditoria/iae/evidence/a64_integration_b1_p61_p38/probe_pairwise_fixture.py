"""P38 pairwise - FIXTURE SINTETICO (dictamen #77, ciclo B-04).

*** NON-PRODUCTION / CONTRACT_TEST_FIXTURE ***

Probe deterministico. NO toca el universo productivo, NO modifica
mappings, NO lee parquets 13F. Su unico proposito es ejercitar la
rama pairwise de P38 (TARGET_PAIRWISE > 0) y publicar la evidencia
en result_pairwise_fixture.json.

Sin red. Sin datetime.now(). Determinista.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\Macro_Sectorial")
sys.path.insert(0, str(ROOT))

from src.institutional_accumulation.aggregation import coverage as cov


HERE = Path(__file__).parent
OUT = HERE / "result_pairwise_fixture.json"


def _rec(period, key, figi, weight, op_status="VERIFIED"):
    return cov.PositionRecord(
        period=period,
        observed_security_key=key,
        share_class_figi=figi,
        canonical_security=None,
        resolution_status="CANONICAL",
        operational_mapping_status=op_status,
        weight=weight,
    )


def main():
    rq4 = [
        _rec("Q4", "key_A", "FIGI_A", 1000.0),
        _rec("Q4", "key_B", "FIGI_B",  500.0),
        _rec("Q4", "key_D", "FIGI_D",  200.0),
    ]
    rq1 = [
        _rec("Q1", "key_A", "FIGI_A", 1200.0),
        _rec("Q1", "key_C", "FIGI_C",  400.0),
        _rec("Q1", "key_D", "FIGI_D",  200.0,
             op_status="TEMPORAL_UNVERIFIED"),
    ]
    target_q4 = {"FIGI_A", "FIGI_B", "FIGI_D"}
    target_q1 = {"FIGI_A", "FIGI_C", "FIGI_D"}

    result = cov.compute_contractual_coverage(
        target_q4=target_q4, target_q1=target_q1,
        records_q4=rq4, records_q1=rq1,
    )

    agg_q4 = cov.aggregate_positions_by_shareclass_figi(rq4, "Q4")
    agg_q1 = cov.aggregate_positions_by_shareclass_figi(rq1, "Q1")

    out = {
        "marker": "NON-PRODUCTION / CONTRACT_TEST_FIXTURE",
        "dictamen": "#77",
        "ciclo": "B-04",
        "fixture": {
            "target_q4": sorted(target_q4),
            "target_q1": sorted(target_q1),
            "records_q4": [
                {"key": r.observed_security_key,
                 "figi": r.share_class_figi,
                 "weight": r.weight,
                 "op_status": r.operational_mapping_status}
                for r in rq4
            ],
            "records_q1": [
                {"key": r.observed_security_key,
                 "figi": r.share_class_figi,
                 "weight": r.weight,
                 "op_status": r.operational_mapping_status}
                for r in rq1
            ],
        },
        "aggregate_by_figi": {
            "Q4": agg_q4,
            "Q1": agg_q1,
        },
        "p38_result": result,
    }

    OUT.write_text(
        json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )

    print("=" * 60)
    print("P38 pairwise fixture (NON-PRODUCTION)")
    print("=" * 60)
    print("TARGET_Q4     :", sorted(target_q4))
    print("TARGET_Q1     :", sorted(target_q1))
    print("agg Q4 por FIGI:", agg_q4)
    print("agg Q1 por FIGI:", agg_q1)
    print("---")
    for k in ("coverage_previous", "coverage_current",
              "paired_security_coverage",
              "paired_weighted_share_coverage",
              "coverage_status"):
        print(k.ljust(35), result.get(k))
    print("OK -> result_pairwise_fixture.json")


if __name__ == "__main__":
    main()