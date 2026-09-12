# -*- coding: utf-8 -*-
"""Fase 6c del pipeline: Sector Breadth & Health + Momentum de amplitud.

Extraido de run.py (refactor C2, fase C2-7c).
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.utils import append_dedup
from src.market_calendar import is_market_day, last_expected_market_date
from config.settings import EXPECTED_SECTOR_COUNT
from indicators.sector_breadth import compute_sector_breadth
from indicators.sector_breadth_momentum import compute_sector_breadth_momentum


def _compute_momentum_amplitud(df_stocks):
    try:
        if df_stocks is not None and not df_stocks.empty:
            sector_breadth_momentum_df = compute_sector_breadth_momentum(
                'outputs/history/sector_breadth.csv'
            )
            sbm_path = Path('outputs/history/sector_breadth_momentum.csv')
            sbm_path.parent.mkdir(parents=True, exist_ok=True)
            if not sector_breadth_momentum_df.empty:
                if sbm_path.exists():
                    hist_sbm = pd.read_csv(sbm_path)
                    sector_breadth_momentum_df = append_dedup(hist_sbm, sector_breadth_momentum_df, ["date","sector"])
                sector_breadth_momentum_df.to_csv(sbm_path, index=False)
                print("  Momentum de amplitud sectorial calculado.")
        else:
            sector_breadth_momentum_df = None
    except Exception as e:
        print(f"  Momentum de amplitud sectorial omitido: {e}")
        sector_breadth_momentum_df = None
    return sector_breadth_momentum_df


def _load_latest_valid_breadth_snapshot(csv_path):
    """C2-followup: carga el ultimo snapshot historico VALIDO de breadth.

    Filtra fechas no-bursatiles (heredadas de runs previos a C2) y exige
    un snapshot completo (al menos EXPECTED_SECTOR_COUNT sectores en la
    misma fecha). Devuelve DataFrame o None.

    No modifica el fichero. Solo lectura.
    """
    try:
        p = Path(csv_path)
        if not p.exists():
            print(f"  [WARN] C2F: snapshot no disponible: {csv_path} no existe.")
            return None
        df = pd.read_csv(p)
        if df.empty or 'date' not in df.columns:
            print("  [WARN] C2F: snapshot no disponible: CSV vacio o sin date.")
            return None
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        if df.empty:
            print("  [WARN] C2F: snapshot no disponible: sin fechas validas.")
            return None
        market_mask = df['date'].apply(lambda d: is_market_day(d.date()))
        df = df[market_mask]
        if df.empty:
            print("  [WARN] C2F: snapshot no disponible: sin fechas de mercado.")
            return None
        for d in sorted(df['date'].unique(), reverse=True):
            snap = df[df['date'] == d]
            if len(snap) >= EXPECTED_SECTOR_COUNT:
                return snap.copy()
        print(f"  [WARN] C2F: snapshot no disponible: ningun dia con {EXPECTED_SECTOR_COUNT} sectores.")
        return None
    except Exception as e:
        print(f"  [WARN] C2F: snapshot no disponible: {e}")
        return None


def _compute_sector_breadth_health(df_stocks, df_market, holdings_df,
                                    reference_date=None, output_path=None):
    """Calcula y persiste Sector Breadth & Health.

    B2 (2026-09-12): la observacion solo se genera si reference_date es
    una sesion NYSE y la sesion esperada esta presente en df_stocks.

    C2-followup (2026-09-12): cuando no hay nueva observacion, devuelve
    el ultimo snapshot historico VALIDO como fallback con is_stale=True.
    El CSV real NO se modifica en ese caso.

    Returns:
        (df_or_None, is_stale_bool).

    Args:
        reference_date: fecha del run. Si None, se resuelve a now() UNA vez.
        output_path: ruta alternativa para tests. Por defecto
                     'outputs/history/sector_breadth.csv'.
    """
    try:
        if df_stocks is None or df_stocks.empty:
            return None, False

        if reference_date is None:
            reference_date = datetime.now()

        sb_path = (Path(output_path) if output_path is not None
                   else Path('outputs/history/sector_breadth.csv'))

        # B2: control de dia bursatil en el caller.
        if not is_market_day(reference_date.date()):
            print(f"  B2: {reference_date.date()} no es sesion NYSE. Omitiendo nueva observacion.")
            fallback = _load_latest_valid_breadth_snapshot(sb_path)
            if fallback is not None:
                print(f"  C2F: usando snapshot {fallback['date'].max().date()} como stale.")
            return fallback, True

        expected_session = last_expected_market_date(reference_date)

        # B2: la sesion esperada debe estar presente en df_stocks.
        observed_last = pd.Timestamp(df_stocks.index[-1]).normalize().date()
        expected_norm = pd.Timestamp(expected_session).normalize().date()
        if expected_norm > observed_last:
            print(f"  B2: EXPECTED_SESSION_ABSENT ({expected_norm} > {observed_last}). Omitiendo nueva observacion.")
            fallback = _load_latest_valid_breadth_snapshot(sb_path)
            if fallback is not None:
                print(f"  C2F: usando snapshot {fallback['date'].max().date()} como stale.")
            return fallback, True

        sector_breadth_df = compute_sector_breadth(
            df_market, df_stocks, holdings_df, as_of_date=expected_session)

        sb_path.parent.mkdir(parents=True, exist_ok=True)
        if not sector_breadth_df.empty:
            if sb_path.exists():
                hist_sb = pd.read_csv(sb_path)
                sector_breadth_df = append_dedup(hist_sb, sector_breadth_df, ["date","sector"])
            sector_breadth_df.to_csv(sb_path, index=False)
            print("  Sector Breadth & Health calculado.")
        return sector_breadth_df, False
    except Exception as e:
        print(f"  Sector Breadth & Health omitido: {e}")
        return None, False


def compute_breadth_metrics(df_stocks, df_market, holdings_df, reference_date=None):
    """Calcula Sector Breadth & Health + Momentum de amplitud.

    C2-followup: sector_breadth_is_stale indica si la observacion
    entregada es la ultima valida del historico (mercado cerrado) en
    lugar de una actualizacion del dia.

    Returns:
        dict con keys:
            sector_breadth_momentum_df, sector_breadth_df,
            sector_breadth_is_stale (bool)
    """
    sb_df, sb_is_stale = _compute_sector_breadth_health(
        df_stocks, df_market, holdings_df, reference_date=reference_date)
    return {
        'sector_breadth_momentum_df': _compute_momentum_amplitud(df_stocks),
        'sector_breadth_df': sb_df,
        'sector_breadth_is_stale': sb_is_stale,
    }
