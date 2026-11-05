"""Deteccion de split/reverse-split CUSIP entre dos trimestres 13F.

Motivo (2026-11-05): BKNG sufrio un split 24:1 entre 2026Q1 y 2026Q2.
Los filings 13F pre-split reportan shares pre-split y precio pre-split;
los post-split, post-split. compute_delta_shares compara SSHPRNAMT sin
ajustar, produciendo un delta artificial de +688M acciones (24x el float
real, fisicamente imposible).

Deteccion robusta SIN lista de splits, SIN fuente externa:
  - Para cada CUSIP presente en ambos trimestres con >= MIN_FILERS,
    calcular dos ratios independientes:
      ratio_precio = mediana(implied_price_prev) / mediana(implied_price_curr)
      ratio_shares = sum_shares_curr / sum_shares_prev
      donde implied_price = VALUE / SSHPRNAMT (mismo calculo del 13F).
  - Si AMBOS ratios coinciden con un entero K >= 2 (o 1/K para reverse)
    dentro de TOLERANCIA_RELATIVA, marcar como split.

Exclusiones (evitan falsos positivos):
  - TITLEOFCLASS contiene NOTE/BOND/DEB/PFD: bonos y preferentes no
    hacen split de acciones aunque SSHPRNAMTTYPE="SH".
  - TITLEOFCLASS contiene 2X/3X/DAILY/LEVERAGED/INVERSE/SHORT/BULL/
    BEAR: ETFs apalancados con rebalanceo diario. Sus unidades fluctuan
    por rebalanceo mecanico, no por split real. NOTA: no se excluye "ETF"
    a secas porque ETFs no apalancados (VUG, SPY, etc.) si pueden hacer
    split real.
  - Reverse splits exigen n_filers >= MIN_FILERS_REVERSE (20). ETFs
    apalancados con <15 filers fluctuan por rebalanceo diario.
  - ratio_precio y ratio_shares deben coincidir con K. Si discrepan
    (crecimiento organico o movimiento de precio aislado), no se marca.

Ajuste: sshprnamt_total en units_prev se multiplica por K (o divide si
K < 1). El delta entre pre y post split es ~0 si no hubo cambio economico.
"""
from __future__ import annotations

import re

import pandas as pd


MIN_FILERS = 5
MIN_FILERS_REVERSE = 20
TOLERANCIA_RELATIVA = 0.10
TOLERANCIA_REVERSE = 0.05
MIN_RATIO = 1.5

# Renta fija: NOTE/BOND/DEB/PFD -> no split de acciones.
_BOND_HINTS_RE = re.compile(
    r"\b(?:NOTE|BOND|DEB|DEBENTURE|PFD|PREFERRED|PREF)\b",
    re.IGNORECASE,
)
# ETFs apalancados: rebalanceo diario, no split real.
# NO incluir "ETF" a secas: VUG, SPY, QQQ pueden hacer splits reales.
_LEVERAGED_ETF_HINTS_RE = re.compile(
    r"\b(?:2X|3X|DAILY|LEVERAGED|INVERSE|SHORT|BULL|BEAR)\b",
    re.IGNORECASE,
)


def _implied_price_per_cusip(infotable_df):
    """Agrupa por CUSIP: mediana implied_price, sum_shares, n_filers.

    Filtra SH + PUTCALL null (igual que _filter_canonical) y excluye
    bonos/preferentes + ETFs apalancados por TITLEOFCLASS. Excluye
    implied_price absurdos (defensa; mediana robusta).
    """
    empty = pd.DataFrame(columns=["CUSIP_norm", "med_implied", "sum_shares", "n_filers"])
    if infotable_df is None or infotable_df.empty:
        return empty
    df = infotable_df.copy()
    mask_sh = df["SSHPRNAMTTYPE"].astype(str).str.strip() == "SH"
    mask_null = df["PUTCALL"].isna()
    df = df[mask_sh & mask_null].copy()
    if df.empty:
        return empty
    # _toc extraido DESPUES del filtro para que su index coincida con df.index.
    if "TITLEOFCLASS" in df.columns:
        df = df[
            ~df["TITLEOFCLASS"].fillna("").astype(str)
               .str.contains(_BOND_HINTS_RE, regex=True)
        ].copy()
        if df.empty:
            return empty
        df = df[
            ~df["TITLEOFCLASS"].fillna("").astype(str)
               .str.contains(_LEVERAGED_ETF_HINTS_RE, regex=True)
        ].copy()
        if df.empty:
            return empty
    df["CUSIP_norm"] = df["CUSIP"].astype(str).str.strip().str.upper()
    df["SSHPRNAMT_f"] = pd.to_numeric(df["SSHPRNAMT"], errors="coerce")
    df["VALUE_f"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df = df[(df["SSHPRNAMT_f"] > 0) & (df["VALUE_f"] > 0)].copy()
    df["implied"] = df["VALUE_f"] / df["SSHPRNAMT_f"]
    df = df[(df["implied"] > 0.001) & (df["implied"] < 1e7)].copy()
    if df.empty:
        return empty
    return df.groupby("CUSIP_norm").agg(
        med_implied=("implied", "median"),
        sum_shares=("SSHPRNAMT_f", "sum"),
        n_filers=("implied", "size"),
    ).reset_index()


def _is_split(ratio_price, ratio_shares, n_filers_prev, n_filers_curr):
    """Devuelve K (split) o 1/K (reverse split), o None.

    Split directo: K >= 2, tol 10%.
    Reverse: 1/K, tol 5%, y n_filers >= MIN_FILERS_REVERSE ambos lados.
    """
    if ratio_price >= MIN_RATIO:
        k = round(ratio_price)
        if k >= 2:
            tol = TOLERANCIA_RELATIVA * k
            if abs(ratio_price - k) <= tol and abs(ratio_shares - k) <= tol:
                return float(k)

    if 0 < ratio_price <= (1.0 / MIN_RATIO):
        if n_filers_prev < MIN_FILERS_REVERSE or n_filers_curr < MIN_FILERS_REVERSE:
            return None
        k_den = round(1.0 / ratio_price)
        if k_den >= 2:
            k = 1.0 / k_den
            tol = TOLERANCIA_REVERSE * abs(k)
            if (ratio_shares > 0
                    and abs(ratio_price - k) <= tol
                    and abs(ratio_shares - k) <= tol):
                return float(k)

    return None


def detect_splits(infotable_prev, infotable_curr) -> dict:
    """Devuelve {CUSIP: K}. K>1 split, K<1 reverse. Sin entrada = no split."""
    prev = _implied_price_per_cusip(infotable_prev)
    curr = _implied_price_per_cusip(infotable_curr)
    if prev.empty or curr.empty:
        return {}

    merged = prev.merge(curr, on="CUSIP_norm", how="inner",
                        suffixes=("_prev", "_curr"))
    merged = merged[
        (merged["n_filers_prev"] >= MIN_FILERS)
        & (merged["n_filers_curr"] >= MIN_FILERS)
        & (merged["med_implied_prev"] > 0)
        & (merged["med_implied_curr"] > 0)
        & (merged["sum_shares_prev"] > 0)
    ]
    if merged.empty:
        return {}

    splits = {}
    for _, row in merged.iterrows():
        ratio_price = float(row["med_implied_prev"]) / float(row["med_implied_curr"])
        ratio_shares = float(row["sum_shares_curr"]) / float(row["sum_shares_prev"])
        k = _is_split(ratio_price, ratio_shares,
                      int(row["n_filers_prev"]), int(row["n_filers_curr"]))
        if k is not None:
            splits[row["CUSIP_norm"]] = k
    return splits


def adjust_units_for_splits(units_prev, splits: dict) -> pd.DataFrame:
    """Multiplica sshprnamt_total en units_prev por K para cada CUSIP split."""
    if not splits or units_prev is None or units_prev.empty:
        return units_prev
    out = units_prev.copy()
    if "observed_security_key" not in out.columns:
        return out
    for cusip, k in splits.items():
        key = "cusip:" + cusip
        mask = out["observed_security_key"].astype(str).str.strip() == key
        if mask.any():
            out.loc[mask, "sshprnamt_total"] = (
                out.loc[mask, "sshprnamt_total"].astype(float) * float(k)
            )
    return out


__all__ = ["detect_splits", "adjust_units_for_splits",
           "MIN_FILERS", "MIN_FILERS_REVERSE",
           "TOLERANCIA_RELATIVA", "TOLERANCIA_REVERSE"]
