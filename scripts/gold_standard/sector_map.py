# -*- coding: utf-8 -*-
"""Mapping canonico ticker -> sector para el estudio Gold Standard.

Fuente: data/etf_holdings.csv (11 ETFs sectoriales, top-20 por weight).
Regla: cada ticker pertenece a exactamente 1 ETF (verificado 2026-10-05:
504 tickers unicos, 0 en >1 ETF).
Normalizacion: BRK.B -> BRK-B via src.instrument_registry.

El mapping se materializa a outputs/gold_standard/sector_map.csv con
SHA-256 en el manifest del estudio.

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.instrument_registry import normalize_yahoo_ticker

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_HOLDINGS = ROOT / "data" / "etf_holdings.csv"


def load_sector_map(
    holdings_csv: Path | None = None,
    top_n_per_etf: int = 20,
    dataset_tickers: set[str] | None = None,
) -> pd.DataFrame:
    """Devuelve DataFrame (ticker, sector, weight, etf) normalizado.

    Holdings: top-N por weight de cada ETF.
    Filtra a tickers del dataset si dataset_tickers se pasa.
    """
    path = holdings_csv if holdings_csv is not None else DEFAULT_HOLDINGS
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")

    h = pd.read_csv(path)
    for c in ("etf", "ticker", "weight"):
        if c not in h.columns:
            raise ValueError(f"Falta columna {c} en {path}")

    # Top-N por weight de cada ETF (coherente con sector_breadth)
    top = (
        h.sort_values("weight", ascending=False)
        .groupby("etf", group_keys=False)
        .head(top_n_per_etf)
        .copy()
    )

    # Normalizar ticker (BRK.B -> BRK-B)
    top["ticker_norm"] = top["ticker"].apply(normalize_yahoo_ticker)

    # Verificar unicidad ticker_norm
    dup = top["ticker_norm"].duplicated()
    if dup.any():
        raise ValueError(
            "Tickers normalizados duplicados: "
            + str(sorted(top.loc[dup, "ticker_norm"].unique()))
        )

    # Filtrar a dataset si se pasa
    if dataset_tickers is not None:
        top = top[top["ticker_norm"].isin(dataset_tickers)].copy()

    out = pd.DataFrame({
        "ticker": top["ticker_norm"].to_numpy(),
        "sector": top["etf"].to_numpy(),
        "weight": top["weight"].astype(float).to_numpy(),
        "etf": top["etf"].to_numpy(),
    }).sort_values(["sector", "ticker"]).reset_index(drop=True)
    return out


def sector_map_dict(df: pd.DataFrame) -> dict[str, str]:
    """dict {ticker: sector}."""
    return dict(zip(df["ticker"], df["sector"]))


SECTOR_UNKNOWN = "UNKNOWN"


def assign_sector(
    ticker: str,
    smap: dict[str, str],
) -> str:
    """Asigna sector. Devuelve UNKNOWN si no hay mapping.

    Politica (protocolo v7 corregido, dictamen DOC 53):
    la ausencia de sector NO excluye la observacion del universo.
    """
    return smap.get(ticker, SECTOR_UNKNOWN)


def coverage_report(
    sector_df: pd.DataFrame,
    dataset_tickers: set[str],
) -> dict:
    """Diagnostico de cobertura."""
    con_sector = set(sector_df["ticker"]) & dataset_tickers
    sin_sector = dataset_tickers - set(sector_df["ticker"])
    por_sector = (
        sector_df[sector_df["ticker"].isin(dataset_tickers)]
        .groupby("sector").size().to_dict()
    )
    return {
        "n_dataset": len(dataset_tickers),
        "n_con_sector": len(con_sector),
        "n_sin_sector": len(sin_sector),
        "por_sector": por_sector,
        "sin_sector_sample": sorted(list(sin_sector))[:20],
    }