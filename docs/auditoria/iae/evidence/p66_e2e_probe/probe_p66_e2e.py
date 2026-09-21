"""P66 e2e probe - validacion de resolve_r3 / resolve_r4 sobre datos reales.

Determinista. Sin datetime.now(). Sin IO de red. Cero escritura en parquets.

Objetivo: verificar que las funciones puras implementadas en
reporting_dedup.py (14.3.1 R3 tri-state + 14.3.4 R4) reproducen las
cifras documentadas en los Gates P66 0.6, 0.8 y 0.9.

Refs:
    docs/auditoria/iae/evidence/p66_gate06_combination_probe/
    docs/auditoria/iae/evidence/p66_gate08_full_r4_coverage/
    docs/auditoria/iae/evidence/p66_gate09_multiple_base/
"""
import sys
from pathlib import Path

import pandas as pd

# Raiz del repo al sys.path (el probe vive en docs/auditoria/...).
_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.institutional_accumulation.aggregation.reporting_dedup import (
    R3_ND,
    build_formnum_cik_mapping,
    classify_filing_family,
    classify_filing_role,
    resolve_r3,
    resolve_r4,
)


ROOT = Path(r"D:\Macro_Sectorial")
DATA = ROOT / "data" / "sec_13f" / "processed"


def _load(quarter):
    base = DATA / quarter
    return {
        "SUBMISSION": pd.read_parquet(base / "SUBMISSION.parquet"),
        "COVERPAGE": pd.read_parquet(base / "COVERPAGE.parquet"),
        "OTHERMANAGER": pd.read_parquet(base / "OTHERMANAGER.parquet"),
    }


def _to_period_str(v):
    """Normaliza cualquier representacion de fecha a ISO YYYY-MM-DD."""
    if v is None:
        return None
    ts = pd.to_datetime(v, errors="coerce")
    if pd.isna(ts):
        return str(v).strip()
    return ts.date().isoformat()


def _dominant_period(sub):
    vals = sub["PERIODOFREPORT"].map(_to_period_str).dropna()
    counts = vals.value_counts()
    return counts.index[0], int(counts.iloc[0])


def _normalize_amendmenttype(v):
    if v is None:
        return None
    s = str(v).strip().upper()
    if not s or s == "<NA>":
        return None
    return s.replace(" ", "_")


def _build_filings_index(sub, cov):
    df = sub.merge(cov, on="ACCESSION_NUMBER", how="inner")
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "ACCESSION_NUMBER": r["ACCESSION_NUMBER"],
            "CIK": str(r["CIK"]).strip(),
            "PERIODOFREPORT": _to_period_str(r["PERIODOFREPORT"]),
            "SUBMISSIONTYPE": str(r["SUBMISSIONTYPE"]).strip(),
            "REPORTTYPE": str(r.get("REPORTTYPE") or "").strip(),
            "AMENDMENTNO": r.get("AMENDMENTNO"),
            "AMENDMENTTYPE": _normalize_amendmenttype(r.get("AMENDMENTTYPE")),
            "ISAMENDMENT": r.get("ISAMENDMENT"),
        })
    return rows


def _group_by_cik_period(filings, period):
    groups = {}
    for f in filings:
        if f["PERIODOFREPORT"] != period:
            continue
        key = f["CIK"]
        groups.setdefault(key, []).append(f)
    return groups


def _count_scope_completeness(filings, period):
    """Cuenta filings sin familia clasificable. PASO 0 del contrato."""
    total = 0
    unclassifiable = 0
    for f in filings:
        if f["PERIODOFREPORT"] != period:
            continue
        total += 1
        fam = classify_filing_family(f["SUBMISSIONTYPE"], f["REPORTTYPE"])
        rol = classify_filing_role(f["SUBMISSIONTYPE"])
        if fam is None and rol is None:
            unclassifiable += 1
    return total, unclassifiable


def main():
    for quarter, expected in (("2025Q4", {"true": 2286, "nd": 1}),
                              ("2026Q1", {"true": 2309, "nd": 0})):
        print("=" * 72)
        print("QUARTER:", quarter)
        print("=" * 72)
        data = _load(quarter)
        sub = data["SUBMISSION"]
        cov = data["COVERPAGE"]
        om = data["OTHERMANAGER"]

        period, n_dom = _dominant_period(sub)
        print("[S1] Periodo dominante:", period, "| filings:", n_dom)

        filings = _build_filings_index(sub, cov)
        total, unclassifiable = _count_scope_completeness(filings, period)
        scope_ok = (unclassifiable == 0)
        print("[S2] Scope: total=%d, no clasificables=%d, verificado=%s"
              % (total, unclassifiable, scope_ok))

        groups = _group_by_cik_period(filings, period)
        print("[S3] Grupos (CIK, PERIOD):", len(groups))

        counter = {"TRUE": 0, "FALSE": 0, "N/D": 0}
        nd_cases = []
        for cik, grp in groups.items():
            state = resolve_r3(
                grp, period=period, scope_completeness_verified=scope_ok,
            )
            counter[state] = counter.get(state, 0) + 1
            if state == R3_ND:
                nd_cases.append((cik, [g["SUBMISSIONTYPE"] for g in grp]))

        print("     R3 = TRUE: ", counter["TRUE"])
        print("     R3 = FALSE:", counter["FALSE"])
        print("     R3 = N/D:  ", counter["N/D"])
        print("     esperado:  TRUE=%d, N/D=%d" % (expected["true"], expected["nd"]))
        print("     N/D cases detalle:")
        for cik, types in nd_cases:
            print("       CIK=%s tipos=%s" % (cik, types))
        print("     N/D drill-down (todos los filings del CIK):")
        for cik, _ in nd_cases:
            if cik == "0002016827":
                print("       CIK=%s (MIXED documentado Gate 0.9)" % cik)
            for f in filings:
                if f["CIK"] != cik or f["PERIODOFREPORT"] != period:
                    continue
                print("         ACC=%s ST=%s RT=%s ISAMD=%s AMDNO=%s AMDTYPE=%s"
                      % (f["ACCESSION_NUMBER"], f["SUBMISSIONTYPE"],
                         f["REPORTTYPE"], f["ISAMENDMENT"],
                         f["AMENDMENTNO"], f["AMENDMENTTYPE"]))

        if quarter == "2025Q4":
            print()
            print("[S4] Caso CIK 0002016827 (Gate 0.9):")
            grp = groups.get("0002016827")
            if grp:
                st = resolve_r3(grp, period=period, scope_completeness_verified=scope_ok)
                print("     tipos:", sorted({g["SUBMISSIONTYPE"] for g in grp}))
                print("     R3 =", st, "(esperado N/D)")
            else:
                print("     NO ENCONTRADO")

        print()
        print("[S5] R4 - casos testigo")
        cov_full = cov.merge(
            sub[["ACCESSION_NUMBER", "CIK", "PERIODOFREPORT"]],
            on="ACCESSION_NUMBER", how="left",
        )
        cov_full["PERIODOFREPORT"] = cov_full["PERIODOFREPORT"].map(_to_period_str)
        mapping = build_formnum_cik_mapping(cov_full, period)
        print("     mapping entradas:", len(mapping))

        # Caso MATCH: primer filing con OTHERMANAGER CIK resuelto
        om_q = om[om["ACCESSION_NUMBER"].isin(cov["ACCESSION_NUMBER"])]
        om_q = om_q.merge(cov[["ACCESSION_NUMBER"]], on="ACCESSION_NUMBER", how="inner")
        tested = {"MATCH": False, "NO_MATCH": False, "N/D": False}
        for acc, grp_om in om_q.groupby("ACCESSION_NUMBER"):
            rows = grp_om.to_dict("records")
            cik_candidates = [r.get("CIK") for r in rows if r.get("CIK") is not None]
            if not tested["MATCH"] and cik_candidates:
                a_cik = str(cik_candidates[0]).strip()
                st = resolve_r4(a_cik, rows, mapping)
                print("     MATCH probe: A=%s acc=%s -> R4=%s" % (a_cik, acc, st))
                tested["MATCH"] = True
            if not tested["NO_MATCH"] and cik_candidates:
                a_cik = "9999999999"
                st = resolve_r4(a_cik, rows, mapping)
                print("     NO_MATCH probe: A=%s acc=%s -> R4=%s" % (a_cik, acc, st))
                tested["NO_MATCH"] = True
            if not tested["N/D"]:
                rows_nd = [r for r in rows
                           if r.get("CIK") is None and r.get("FORM13FFILENUMBER") is None]
                if rows_nd:
                    st = resolve_r4("0000000001", rows_nd, mapping)
                    print("     N/D probe: acc=%s filas_null=%d -> R4=%s"
                          % (acc, len(rows_nd), st))
                    tested["N/D"] = True
            if all(tested.values()):
                break

        print()
        print("[S5] estados observados: MATCH=%s NO_MATCH=%s N/D=%s"
              % (tested["MATCH"], tested["NO_MATCH"], tested["N/D"]))
        print()


if __name__ == "__main__":
    main()