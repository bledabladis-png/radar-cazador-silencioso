# -*- coding: utf-8 -*-
"""H-04: intervalos de confianza para structure_ratio.

Pre-registro congelado en docs/auditoria/wyckoff/69_h04_intervals_2026-10-26.md.
NO publica en el reporte. Solo produce CSVs de auditoria.

Comando: py scripts/h04_intervals.py
Salidas (no commiteadas, regenerables):
  - outputs/audit/h04_intervals.csv
  - outputs/audit/h04_cross_sector_corr.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

IN_CSV = Path("outputs/history/sector_wyckoff_distribution.csv")
OUT_CI = Path("outputs/audit/h04_intervals.csv")
OUT_CORR = Path("outputs/audit/h04_cross_sector_corr.csv")

B_BOOT = 2000
SEED = 44
STRATA = [(1, 4), (5, 10), (11, 20), (21, int(1e9))]


def compute_structure_ratio_row(row):
    """Formula §2 del pre-registro. None si estructura_clara == 0."""
    acc = row["count_accumulation"]
    accf = row["count_accumulation_forming"]
    markup = row["count_markup"]
    dist = row["count_distribution"]
    distf = row["count_distribution_forming"]
    markdown = row["count_markdown"]
    clara = acc + accf + markup + dist + distf + markdown
    if clara == 0:
        return None, 0
    return (acc + accf + markup) / clara, int(clara)


def enrich(df):
    """Anade structure_ratio y n_estructuras recalculados desde counts."""
    enriched = df.copy()
    pares = enriched.apply(compute_structure_ratio_row, axis=1)
    enriched["structure_ratio"] = [p[0] for p in pares]
    enriched["n_estructuras"] = [p[1] for p in pares]
    enriched["date"] = pd.to_datetime(enriched["date"])
    return enriched


def daily_aggregate(df):
    """Metrica principal: media cross-sector por dia de structure_ratio."""
    valid = df[df["structure_ratio"].notna()]
    daily = valid.groupby("date")["structure_ratio"].mean().rename("daily_ratio")
    return daily


def cell_means(df, cohort_key):
    """Medias de celda (cohorte, sector)."""
    valid = df[df["structure_ratio"].notna()].copy()
    if cohort_key == "annual":
        valid["cohort"] = valid["date"].dt.year.astype(str)
    else:
        valid["cohort"] = (valid["date"].dt.year.astype(str)
                           + "-H" + ((valid["date"].dt.month - 1) // 6 + 1).astype(str))
    grouped = valid.groupby(["cohort", "sector"])["structure_ratio"].mean()
    return grouped.to_dict()


def multiway_bootstrap(cells, B=B_BOOT, seed=SEED):
    """Bootstrap two-way (cohorte x sector), pre-registro §6.

    cells: dict[(cohort, sector)] -> float (media de celda)
    """
    if not cells:
        return None, None
    cohorts = sorted(set(c for c, _ in cells))
    sectors_by_cohort = {c: sorted(set(s for cc, s in cells if cc == c)) for c in cohorts}
    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(B):
        cohorts_b = rng.choice(cohorts, size=len(cohorts), replace=True)
        vals = []
        for c in cohorts_b:
            sects = sectors_by_cohort[c]
            sects_b = rng.choice(sects, size=len(sects), replace=True)
            for s in sects_b:
                k = (c, s)
                if k in cells:
                    vals.append(cells[k])
        if vals:
            boot_means.append(float(np.mean(vals)))
    if not boot_means:
        return None, None
    return (float(np.percentile(boot_means, 2.5)),
            float(np.percentile(boot_means, 97.5)))


def bootstrap_naive(vals, B=B_BOOT, seed=SEED):
    """Bootstrap iid para comparar. No es el metodo oficial."""
    if not vals:
        return None, None
    arr = np.array(vals, dtype=float)
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(B)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _enrich_cohort(cells, n_days_by_cohort, cohort):
    """Devuelve (n_years_effective, n_sectors_effective, n_days)."""
    if cohort == "ALL":
        n_years = len(set(c for c, _ in cells))
        sectors = set(s for _, s in cells)
    else:
        n_years = 1
        sectors = set(s for cc, s in cells if cc == cohort)
    return n_years, len(sectors), n_days_by_cohort.get(cohort)


def summarize_cohorts(cells, cohort_label, scope, n_days_by_cohort):
    """Resumen por cohorte: mean, n_cells, n_days, IC multiway, IC naive."""
    rows = []
    cohorts = sorted(set(c for c, _ in cells))
    for c in cohorts:
        vals = [v for (cc, _), v in cells.items() if cc == c]
        if not vals:
            continue
        n_years, n_sectors, n_days = _enrich_cohort(cells, n_days_by_cohort, c)
        mean = float(np.mean(vals))
        ci_lo, ci_hi = multiway_bootstrap({k: v for k, v in cells.items() if k[0] == c})
        nv_lo, nv_hi = bootstrap_naive(vals)
        rows.append({
            "scope": scope,
            "cohort": c,
            "n_cells": len(vals),
            "n_days": n_days,
            "n_years_effective": n_years,
            "n_sectors_effective": n_sectors,
            "mean": mean,
            "ci_lo_multiway": ci_lo,
            "ci_hi_multiway": ci_hi,
            "ci_lo_naive": nv_lo,
            "ci_hi_naive": nv_hi,
        })
    all_vals = list(cells.values())
    if all_vals:
        n_years, n_sectors, n_days = _enrich_cohort(cells, n_days_by_cohort, "ALL")
        ci_lo, ci_hi = multiway_bootstrap(cells)
        nv_lo, nv_hi = bootstrap_naive(all_vals)
        rows.append({
            "scope": scope,
            "cohort": "ALL",
            "n_cells": len(all_vals),
            "n_days": n_days,
            "n_years_effective": n_years,
            "n_sectors_effective": n_sectors,
            "mean": float(np.mean(all_vals)),
            "ci_lo_multiway": ci_lo,
            "ci_hi_multiway": ci_hi,
            "ci_lo_naive": nv_lo,
            "ci_hi_naive": nv_hi,
        })
    return rows


def strata_summary(df):
    """Analisis por estrato de n_estructuras, pre-registro §7b."""
    valid = df[df["structure_ratio"].notna()]
    rows = []
    for lo, hi in STRATA:
        label = "n_estructuras_{}-{}".format(lo, "inf" if hi >= 999 else hi)
        sub = valid[(valid["n_estructuras"] >= lo) & (valid["n_estructuras"] <= hi)]
        if len(sub) == 0:
            rows.append({
                "scope": "stratum", "cohort": label,
                "n_cells": 0, "n_days": 0,
                "n_years_effective": 0, "n_sectors_effective": 0,
                "mean": None, "ci_lo_multiway": None, "ci_hi_multiway": None,
                "ci_lo_naive": None, "ci_hi_naive": None,
            })
            continue
        cells = sub.groupby(["date", "sector"])["structure_ratio"].mean().to_dict()
        annual_cells = {}
        for (date, sector), val in cells.items():
            y = int(pd.Timestamp(date).year)
            annual_cells[(y, sector)] = val
        n_years, n_sectors, _ = _enrich_cohort(annual_cells, {}, "ALL")
        ci_lo, ci_hi = multiway_bootstrap(annual_cells)
        rows.append({
            "scope": "stratum", "cohort": label,
            "n_cells": len(cells),
            "n_days": int(sub["date"].nunique()),
            "n_years_effective": n_years,
            "n_sectors_effective": n_sectors,
            "mean": float(sub["structure_ratio"].mean()),
            "ci_lo_multiway": ci_lo, "ci_hi_multiway": ci_hi,
            "ci_lo_naive": None, "ci_hi_naive": None,
        })
    return rows


def cross_sector_corr(df):
    """Matriz 11x11 y rho_bar sobre dias comunes."""
    valid = df[df["structure_ratio"].notna()]
    pivot = valid.pivot_table(index="date", columns="sector",
                              values="structure_ratio", aggfunc="mean")
    corr = pivot.corr()
    n = len(corr)
    off_diag = [corr.iloc[i, j] for i in range(n) for j in range(i + 1, n)
                if pd.notna(corr.iloc[i, j])]
    rho_bar = float(np.mean(off_diag)) if off_diag else None
    return corr, rho_bar


def run():
    if not IN_CSV.exists():
        print("[ERROR] no existe {}".format(IN_CSV))
        return 1

    raw = pd.read_csv(IN_CSV)
    df = enrich(raw)
    n_total = len(df)
    n_valid = int(df["structure_ratio"].notna().sum())
    print("Filas totales: {}".format(n_total))
    print("Con structure_ratio valido: {} ({:.1f}%)".format(
        n_valid, 100.0 * n_valid / max(1, n_total)))

    # n_days por cohorte
    valid_df = df[df["structure_ratio"].notna()].copy()
    valid_df["year"] = valid_df["date"].dt.year.astype(str)
    valid_df["half"] = (valid_df["date"].dt.year.astype(str)
                        + "-H" + ((valid_df["date"].dt.month - 1) // 6 + 1).astype(str))
    n_days_annual = valid_df.groupby("year")["date"].nunique().to_dict()
    n_days_annual["ALL"] = int(valid_df["date"].nunique())
    n_days_sem = valid_df.groupby("half")["date"].nunique().to_dict()
    n_days_sem["ALL"] = int(valid_df["date"].nunique())

    annual_cells = cell_means(df, "annual")
    print("Celdas (year, sector): {}".format(len(annual_cells)))
    rows = summarize_cohorts(annual_cells, "annual", "annual", n_days_annual)

    sem_cells = cell_means(df, "semestral")
    print("Celdas (semestre, sector): {}".format(len(sem_cells)))
    rows += summarize_cohorts(sem_cells, "semestral", "semestral", n_days_sem)

    rows += strata_summary(df)

    OUT_CI.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CI, index=False, encoding="utf-8")
    print("[OK] {}".format(OUT_CI))
    print(out_df.to_string(index=False))

    corr, rho_bar = cross_sector_corr(df)
    corr.to_csv(OUT_CORR, encoding="utf-8")
    print("[OK] {}".format(OUT_CORR))
    if rho_bar is not None:
        print("rho_bar cross-sector: {:.4f}".format(rho_bar))

    return 0


def main():
    ap = argparse.ArgumentParser(description="H-04 pre-registro 69")
    ap.parse_args()
    return run()


if __name__ == "__main__":
    raise SystemExit(main())