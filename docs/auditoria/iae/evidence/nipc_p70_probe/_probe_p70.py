"""Probe P70 - diagnostico del contrato de amendments.

Preguntas:
  1. Cuantos grupos llegan a HR_PLUS_RESTATEMENT?
  2. Cuantos tienen len(group) > 2?
  3. En cuantos iloc[1] != iloc[-1] en AMENDMENTTYPE?
  4. Existen casos iloc[1] == iloc[-1] con filas intermedias?

NO modifica codigo productivo. Solo lectura + diagnostico.
Reutiliza filter_by_period + order_filings + classify_strategy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\Macro_Sectorial")
sys.path.insert(0, str(ROOT))

from src.institutional_accumulation.sec_13f.storage import load_parquets
from src.institutional_accumulation.sec_13f.identity import (
    filter_by_period,
    order_filings,
    classify_strategy,
    STRATEGY_HR_PLUS_RESTATEMENT,
    STRATEGY_HR_CHAIN_RESTATEMENT,
    STRATEGY_HR_COMPOSITE,
    STRATEGY_HR_PLUS_NEW_HOLDINGS,
)

PERIODS = [
    ("2026Q1", "2026-03-31"),
    ("2025Q4", "2025-12-31"),
]
OUT_DIR = Path("docs/auditoria/iae/evidence/nipc_p70_probe")
OUT_DIR.mkdir(parents=True, exist_ok=True)
def _norm_atype(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s if s else None


def analyse_quarter(quarter, period_str):
    """Devuelve dict con metricas del trimestre."""
    print("=== " + quarter + " (period=" + period_str + ") ===")
    dfs_full = load_parquets(quarter)
    dfs = filter_by_period(dfs_full, period=period_str)

    sub = dfs["SUBMISSION"]
    cov = dfs.get("COVERPAGE")
    ordered = order_filings(sub, cov)

    n_groups_total = 0
    n_groups_hr_pr = 0
    n_groups_hr_cr = 0
    n_groups_hr_comp = 0
    n_groups_hr_newh = 0
    n_len_gt_2 = 0
    n_second_vs_last_diff = 0
    n_second_eq_last_with_intermediate = 0
    max_len = 0
    examples = []
    for (cik, per), grp in ordered.groupby(["CIK", "PERIODOFREPORT"], sort=False):
        grp = grp.sort_values("_order").reset_index(drop=True)
        n_groups_total += 1
        strategy = classify_strategy(grp)

        if strategy == STRATEGY_HR_PLUS_RESTATEMENT:
            n_groups_hr_pr += 1
            L = len(grp)
            if L > max_len:
                max_len = L
            atype_1 = _norm_atype(grp.iloc[1].get("AMENDMENTTYPE"))
            atype_last = _norm_atype(grp.iloc[-1].get("AMENDMENTTYPE"))

            if L > 2:
                n_len_gt_2 += 1
            if atype_1 != atype_last:
                n_second_vs_last_diff += 1
            if L > 2 and atype_1 == atype_last:
                n_second_eq_last_with_intermediate += 1
            if L > 2 or atype_1 != atype_last:
                filings_list = []
                for _, r in grp.iterrows():
                    ano = r.get("AMENDMENTNO")
                    filings_list.append({
                        "order": int(r["_order"]),
                        "ACCESSION_NUMBER": str(r["ACCESSION_NUMBER"]),
                        "SUBMISSIONTYPE": str(r["SUBMISSIONTYPE"]),
                        "AMENDMENTNO": None if pd.isna(ano) else str(ano),
                        "AMENDMENTTYPE": _norm_atype(r.get("AMENDMENTTYPE")),
                    })
                examples.append({
                    "quarter": quarter,
                    "period": str(per),
                    "CIK": str(cik),
                    "n_filings": int(L),
                    "AMENDMENTTYPE_iloc1": atype_1,
                    "AMENDMENTTYPE_iloc_last": atype_last,
                    "filings": filings_list,
                })
        elif strategy == STRATEGY_HR_CHAIN_RESTATEMENT:
            n_groups_hr_cr += 1
            if len(grp) > max_len:
                max_len = len(grp)
        elif strategy == STRATEGY_HR_COMPOSITE:
            n_groups_hr_comp += 1
            if len(grp) > max_len:
                max_len = len(grp)
        elif strategy == STRATEGY_HR_PLUS_NEW_HOLDINGS:
            n_groups_hr_newh += 1
    return {
        "quarter": quarter,
        "period": period_str,
        "n_groups_total": n_groups_total,
        "n_groups_hr_plus_restatement": n_groups_hr_pr,
        "n_groups_hr_chain_restatement": n_groups_hr_cr,
        "n_groups_hr_composite": n_groups_hr_comp,
        "n_groups_hr_plus_new_holdings": n_groups_hr_newh,
        "n_groups_len_gt_2": n_len_gt_2,
        "n_groups_second_vs_last_diff": n_second_vs_last_diff,
        "n_groups_second_eq_last_with_intermediate": n_second_eq_last_with_intermediate,
        "max_group_len": max_len,
        "examples": examples,
    }


def main():
    all_results = []
    for quarter, period_str in PERIODS:
        try:
            r = analyse_quarter(quarter, period_str)
            all_results.append(r)
        except FileNotFoundError as e:
            print("[SKIP] " + quarter + ": " + str(e))
            all_results.append({"quarter": quarter, "error": str(e)})
    out_json = OUT_DIR / "probe_p70_result.json"
    with out_json.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    out_txt = OUT_DIR / "probe_p70_summary.txt"
    with out_txt.open("w", encoding="utf-8", newline="\n") as f:
        for r in all_results:
            if "error" in r:
                f.write(r["quarter"] + ": ERROR " + r["error"] + "\n\n")
                continue
            f.write("=== " + r["quarter"] + " (period=" + r["period"] + ") ===\n")
            f.write("n_groups_total:                            " + str(r["n_groups_total"]) + "\n")
            f.write("n_groups_hr_plus_restatement:              " + str(r["n_groups_hr_plus_restatement"]) + "\n")
            f.write("n_groups_hr_chain_restatement:             " + str(r["n_groups_hr_chain_restatement"]) + "\n")
            f.write("n_groups_hr_composite:                     " + str(r["n_groups_hr_composite"]) + "\n")
            f.write("n_groups_hr_plus_new_holdings:             " + str(r["n_groups_hr_plus_new_holdings"]) + "\n")
            f.write("n_groups_len_gt_2:                         " + str(r["n_groups_len_gt_2"]) + "\n")
            f.write("n_groups_second_vs_last_diff:              " + str(r["n_groups_second_vs_last_diff"]) + "\n")
            f.write("n_groups_second_eq_last_with_intermediate: " + str(r["n_groups_second_eq_last_with_intermediate"]) + "\n")
            f.write("max_group_len:                             " + str(r["max_group_len"]) + "\n")
            f.write("n_examples_saved:                          " + str(len(r["examples"])) + "\n")
            f.write("\n")

    print()
    print("[OK] Resultado: " + str(out_json))
    print("[OK] Resumen:   " + str(out_txt))


if __name__ == "__main__":
    main()