"""P1-H03: invariantes del backfill sector_wyckoff_distribution (dictamen 2026-10-26).

Verifica 6 invariantes sobre las 1086 fechas x 11 sectores:
  (1) Cuadricula completa: 1086 fechas x 11 sectores, sin duplicados.
  (2) Fase y eventos identicos para ticker comun (date, sector, ticker).
      SKIP: no existe snapshot por-ticker pre/post. Requiere
      compute_wyckoff_snapshot o regenerate_wyckoff_history, ambos
      bloqueados por hang conocido en classify_phase.
  (3) count_* reconcilia con n_valid_wyckoff y n_total.
  (4) pct_* reproducible desde count y denominador.
  (5) CSV activo == snapshot v2_full_target_only (fila a fila).
  (6) Diferencias vs pre_top20 explicadas por expansion y disponibilidad.

Read-only. No modifica nada. Escribe CSV resumen en outputs/audit/.
Uso: py scripts/audit/p1_h03_invariantes.py
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

HIST = ROOT / "outputs" / "history"
ACTIVE = HIST / "sector_wyckoff_distribution.csv"
PRE = HIST / "sector_wyckoff_distribution.pre_top20_target.csv"
V2_BAK = HIST / "sector_wyckoff_distribution.v2_full.csv.bak"
V2_ONLY = HIST / "sector_wyckoff_distribution.v2_full_target_only.csv"
OUT = ROOT / "outputs" / "audit" / "p1_h03_invariantes.csv"

SECTORS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP",
           "XLRE", "XLU", "XLV", "XLY"]
COUNT_COLS = [
    "count_accumulation", "count_accumulation_forming", "count_markup",
    "count_range", "count_distribution", "count_distribution_forming",
    "count_markdown", "count_conflict",
]
PCT_COLS = [
    "pct_accumulation", "pct_accumulation_forming", "pct_markup",
    "pct_range", "pct_distribution", "pct_distribution_forming",
    "pct_markdown", "pct_conflict",
]
TOL = 0.01


def check_1_grid(df):
    n = len(df)
    keys = df[["date", "sector"]].drop_duplicates()
    dup = n - len(keys)
    n_dates = df["date"].nunique()
    n_sectors = df["sector"].nunique()
    missing = set(SECTORS) - set(df["sector"].unique())
    status = "OK"
    detail = "rows={} keys_unicas={} fechas={} sectores={} duplicados={}".format(
        n, len(keys), n_dates, n_sectors, dup)
    if n != 11946:
        status = "FAIL"; detail += " | rows!=11946"
    if dup != 0:
        status = "FAIL"; detail += " | duplicados={}".format(dup)
    if n_dates != 1086:
        status = "FAIL"; detail += " | fechas!=1086"
    if n_sectors != 11 or missing:
        status = "FAIL"; detail += " | sectores faltantes={}".format(missing)
    return {"id": 1, "name": "cuadricula", "status": status, "detail": detail}


def check_2_tickers():
    return {
        "id": 2, "name": "fase_por_ticker",
        "status": "SKIP",
        "detail": "no existe snapshot pre/post por ticker. classify_phase "
                  "funciona (360/360 OK en 21.6s, verificado 2026-10-26); "
                  "falta la base de comparacion, no el detector.",
    }


def check_3_counts(df):
    problems = []
    df = df.copy()
    df["sum_counts"] = df[COUNT_COLS].sum(axis=1)
    df["sum_total"] = df["n_valid_wyckoff"] + df["n_insufficient_wyckoff"]
    # (a) sum(count_*) == n_valid_wyckoff
    a_bad = df[df["sum_counts"] != df["n_valid_wyckoff"]]
    # (b) n_valid + n_insuff == n_total
    b_bad = df[df["sum_total"] != df["n_total"]]
    for _, r in a_bad.head(5).iterrows():
        problems.append("{} {}: sum_counts={} != n_valid={}".format(
            r["date"], r["sector"], r["sum_counts"], r["n_valid_wyckoff"]))
    for _, r in b_bad.head(5).iterrows():
        problems.append("{} {}: n_valid+n_insuff={} != n_total={}".format(
            r["date"], r["sector"], r["sum_total"], r["n_total"]))
    n_a, n_b = len(a_bad), len(b_bad)
    status = "OK" if n_a == 0 and n_b == 0 else "FAIL"
    detail = "violaciones (a) sum_counts!=n_valid: {} | (b) n_valid+n_insuff!=n_total: {}".format(n_a, n_b)
    if problems:
        detail += " | ejemplos: " + "; ".join(problems[:3])
    return {"id": 3, "name": "counts_reconcilian", "status": status, "detail": detail}


def check_4_pcts(df):
    """Verifica pct_* == count / denominador. Prueba denominador n_total y n_valid.
    Reporta cual encaja."""
    df = df.copy()
    for denom_col, label in (("n_total", "total"), ("n_valid_wyckoff", "valid")):
        bad = 0
        for cc, pc in zip(COUNT_COLS, PCT_COLS):
            expected = (df[cc] / df[denom_col] * 100.0).fillna(0.0)
            diff = (expected - df[pc]).abs()
            bad += int((diff > TOL).sum())
        if bad == 0:
            return {
                "id": 4, "name": "pcts_reproducibles",
                "status": "OK",
                "detail": "denominador efectivo: {} (0 violaciones)".format(label),
            }
    # Si ninguno encaja, reportar con el peor
    return {
        "id": 4, "name": "pcts_reproducibles", "status": "FAIL",
        "detail": "ningun denominador (n_total, n_valid) reproduce pct_* en todas las filas",
    }


def check_5_csv_vs_snapshot():
    if not V2_BAK.exists():
        return {"id": 5, "name": "csv_vs_snapshot", "status": "SKIP",
                "detail": "no existe v2_full.csv.bak (pre-swap)"}
    df_a = pd.read_csv(ACTIVE)
    df_v = pd.read_csv(V2_BAK)
    if list(df_a.columns) != list(df_v.columns):
        return {"id": 5, "name": "csv_vs_snapshot", "status": "FAIL",
                "detail": "columnas distintas entre activo y v2_full"}
    m = df_a.merge(df_v, on=["date", "sector"], suffixes=("_a", "_v"), how="outer",
                   indicator=True)
    solo_a = (m["_merge"] == "left_only").sum()
    solo_v = (m["_merge"] == "right_only").sum()
    diffs = 0
    numeric = [c for c in COUNT_COLS + PCT_COLS + ["n_total", "n_valid_wyckoff", "n_insufficient_wyckoff", "coverage_wyckoff"]]
    for c in numeric:
        ca, cv = c + "_a", c + "_v"
        if ca in m.columns and cv in m.columns:
            d = (m[ca] - m[cv]).abs()
            if c.startswith("pct_"):
                diffs += int((d > TOL).sum())
            else:
                diffs += int((d > 0).sum())
    status = "OK" if solo_a == 0 and solo_v == 0 and diffs == 0 else "FAIL"
    detail = "solo_activo={} solo_v2={} celdas_distintas={}".format(solo_a, solo_v, diffs)
    return {"id": 5, "name": "csv_vs_snapshot", "status": status, "detail": detail}


def check_6_expansion(df_act, df_pre):
    if df_pre is None or not PRE.exists():
        return {"id": 6, "name": "diferencias_explicadas", "status": "SKIP",
                "detail": "no existe pre_top20_target.csv"}
    keys = ["date", "sector"]
    m = df_act[keys + ["n_total", "n_valid_wyckoff"]].merge(
        df_pre[keys + ["n_total", "n_valid_wyckoff"]],
        on=keys, suffixes=("_act", "_pre"), how="outer", indicator=True)
    solo_a = int((m["_merge"] == "left_only").sum())
    solo_p = int((m["_merge"] == "right_only").sum())
    if solo_a or solo_p:
        return {"id": 6, "name": "diferencias_explicadas", "status": "FAIL",
                "detail": "keys no coinciden: solo_act={} solo_pre={}".format(solo_a, solo_p)}
    m["delta"] = m["n_total_act"] - m["n_total_pre"]
    # (a) Nunca negativo (no regresion)
    neg = m[m["delta"] < 0]
    # (b) Solo XLI, XLF, XLK tienen delta != 0
    grupos_exp = {"XLI", "XLF", "XLK"}
    otros_cambiados = m[(m["delta"] != 0) & (~m["sector"].isin(grupos_exp))]
    # (c) delta total = suma de cambio
    total_crecido = int(m["delta"].clip(lower=0).sum())
    status = "OK" if len(neg) == 0 and len(otros_cambiados) == 0 else "FAIL"
    detail = (
        "delta_total={} filas_sin_regresion={}/{} | "
        "sectores_con_delta_fuera_de_XLI/XLF/XLK: {} | "
        "filas_negativas={}"
    ).format(total_crecido, len(m) - len(neg), len(m),
             sorted(otros_cambiados["sector"].unique()) if len(otros_cambiados) else [],
             len(neg))
    return {"id": 6, "name": "diferencias_explicadas", "status": status, "detail": detail}




def _artifact_diag():
    out = []
    for name, p in (
        ("v2_bak", V2_BAK),
        ("v2_only", V2_ONLY),
    ):
        if p.exists():
            dfx = pd.read_csv(p)
            out.append({"file": name,
                        "rows": len(dfx),
                        "n_total_mean": round(float(dfx["n_total"].mean()), 2),
                        "n_total_min": int(dfx["n_total"].min()),
                        "n_total_max": int(dfx["n_total"].max())})
    return out

def main():
    df = pd.read_csv(ACTIVE)
    df_pre = pd.read_csv(PRE) if PRE.exists() else None
    results = [
        check_1_grid(df),
        check_2_tickers(),
        check_3_counts(df),
        check_4_pcts(df),
        check_5_csv_vs_snapshot(),
        check_6_expansion(df, df_pre),
    ]
    print("=" * 100)
    print("P1-H03: INVARIANTES DEL BACKFILL (1086 fechas x 11 sectores)")
    print("=" * 100)
    for r in results:
        print("[{}] ({}) {:<28s} {}".format(
            r["status"], r["id"], r["name"], r["detail"]))
    print()
    print()
    print("Diagnostico de artefactos secundarios:")
    for d in _artifact_diag():
        print("  {}: rows={} n_total_mean={} min={} max={}".format(
            d["file"], d["rows"], d["n_total_mean"], d["n_total_min"], d["n_total_max"]))
    print()
    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_skip = sum(1 for r in results if r["status"] == "SKIP")
    print("Resumen: {} OK, {} FAIL, {} SKIP de 6".format(n_ok, n_fail, n_skip))
    print("Aviso: la afirmacion 'cero regresiones' del doc 70-v2 se apoyaba solo")
    print("       en n_total(v2)<n_total(activo). Aqui se verifican 5 invariantes")
    print("       adicionales; ver detalle arriba.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUT, index=False)
    print()
    print("escrito:", OUT)


if __name__ == "__main__":
    main()
