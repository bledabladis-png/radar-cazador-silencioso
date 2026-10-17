# -*- coding: utf-8 -*-
"""Fase 6c del pipeline: Sector Breadth & Health + Momentum de amplitud.

Extraido de run.py (refactor C2, fase C2-7c).
"""

from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import pandas as pd

from src.market_calendar import is_market_day, last_expected_market_date
from src.pipeline.output_writer import persist, skip
from config.settings import EXPECTED_SECTOR_COUNT
from indicators.sector_breadth import compute_sector_breadth
from indicators.sector_breadth_momentum import compute_sector_breadth_momentum


def _compute_momentum_amplitud(df_stocks):
    try:
        if df_stocks is not None and not df_stocks.empty:
            sector_breadth_momentum_df = compute_sector_breadth_momentum(
                'outputs/history/sector_breadth.csv'
            )
            if not sector_breadth_momentum_df.empty:
                date_val = sector_breadth_momentum_df['date'].max()
                persist('sector_breadth_momentum', sector_breadth_momentum_df,
                        date_value=date_val)
                print("  Momentum de amplitud sectorial calculado.")
            else:
                skip('sector_breadth_momentum', 'EMPTY_DF')
        else:
            sector_breadth_momentum_df = None
    except (KeyError, ValueError, TypeError, IndexError, AttributeError, OSError, RuntimeError, pd.errors.ParserError) as e:
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
    except (KeyError, ValueError, TypeError, IndexError, AttributeError, OSError, RuntimeError, pd.errors.ParserError) as e:
        print(f"  [WARN] C2F: snapshot no disponible: {e}")
        return None


def _resolve_breadth_effective_date(df_stocks, df_market, holdings_df,
                                     min_coverage=0.90):
    """N-C6: resuelve la fecha efectiva para Sector Breadth.

    Aplica resolve_effective_date sobre el universo combinado:
    tickers equity de df_stocks + ETFs sectoriales de df_market
    (los ETFs sectoriales participan en el calculo de RS).

    Devuelve (effective_date_ts, meta_dict) o (None, meta) si
    INSUFFICIENT_COVERAGE. El consumidor decide si ir al fallback.

    R2: la fecha no se resuelve por posicion fisica.
    R3: misma resolucion para sector_breadth y su derivado
        sector_breadth_momentum (que lee el CSV escrito con esta fecha).
    """
    from src.effective_date import resolve_effective_date

    if df_stocks is None or df_stocks.empty:
        return None, {'status': 'INSUFFICIENT_COVERAGE', 'reason': 'df_stocks empty'}

    # Universo elegible: SOLO tickers equity de df_stocks.
    #
    # H7 (2026-10-17, REP-AUDIT-20261017-H7): bug detectado en el run
    # del 21:11. Antes se anadian los ETFs sectoriales de df_market
    # a este universo 'como input de RS'. Pero resolve_effective_date
    # opera SOBRE df_stocks: los ETFs no estan en df_stocks.columns,
    # asi que se contaban como 'no observados' en el denominador sin
    # poder sumar en el numerador. Resultado: cobertura artificialmente
    # baja (305/340=0.897 en el 16 en lugar de 305/329=0.927), que hacia
    # retroceder effective_date al 15 y disparaba DATA_PENDING falso.
    #
    # Los ETFs sectoriales siguen disponibles para el calculo de RS en
    # compute_sector_breadth(df_market, ...), que los recibe por
    # separado. No forman parte del universo de cobertura temporal de
    # esta funcion.
    eligible = [c[1] for c in df_stocks.columns if c[0] == 'Close']

    if not eligible:
        return None, {'status': 'INSUFFICIENT_COVERAGE', 'reason': 'sin elegibles'}

    meta = resolve_effective_date(df_stocks, eligible, min_coverage=min_coverage)
    if meta['status'] != 'OK' or meta['date'] is None:
        return None, meta
    return meta['date'], meta


def _compute_sector_breadth_health(df_stocks, df_market, holdings_df,
                                    reference_date=None, output_path=None,
                                    temporal_meta=None, return_reason=False):
    """Calcula y persiste Sector Breadth & Health.

    B2 (2026-09-12, revisado 2026-10-10): la observacion se genera si la
    sesion esperada (last_expected_market_date(reference_date)) esta
    presente en df_stocks. Ya NO se exige que reference_date sea dia
    bursatil: reference_date es la fecha del run (reloj), no la sesion
    objetivo. El pipeline corre en slots de fin de semana que producen
    la sesion del viernes.

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
            return (None, False, None) if return_reason else (None, False)

        if reference_date is None:
            # S-02c (2026-10-03): tz-aware por coherencia con run.py:43.
            # El caller productivo (run.py:140) siempre pasa reference_date;
            # este fallback solo se activa en tests o uso aislado.
            reference_date = datetime.now(ZoneInfo("Europe/Madrid"))

        sb_path = (Path(output_path) if output_path is not None
                   else Path('outputs/history/sector_breadth.csv'))

        # 2026-10-10: eliminado el candado is_market_day(reference_date).
        # Motivo: reference_date es la fecha del run (reloj), no la sesion
        # objetivo. El pipeline corre en slots de fin de semana que
        # producen la sesion del viernes. El candado correcto es el de
        # mas abajo (expected_session vs observed_last). Coherente con
        # pipeline_gate.py (target_session != day of slot) y con el
        # resto de modulos del pipeline.

        expected_session = last_expected_market_date(reference_date)
        expected_norm = pd.Timestamp(expected_session).normalize().date()

        # N-C6 (dictamen auditor externo 2026-10-16): la fecha efectiva
        # NO se resuelve por posicion fisica (df_stocks.index[-1]).
        # Se resuelve por resolve_effective_date sobre el universo
        # combinado (equity df_stocks + ETFs sectoriales de df_market)
        # con la misma cobertura exigida por el resto del pipeline.
        effective_date, eff_meta = _resolve_breadth_effective_date(
            df_stocks, df_market, holdings_df, min_coverage=0.90
        )
        if effective_date is None:
            print("  B2/N-C6: sin fecha efectiva (INSUFFICIENT_COVERAGE). "
                  "Omitiendo nueva observacion.")
            fallback = _load_latest_valid_breadth_snapshot(sb_path)
            if fallback is not None:
                print(f"  C2F: usando snapshot {fallback['date'].max().date()} como stale.")
            skip('sector_breadth', 'STALE',
                 path_override=sb_path if output_path is not None else None)
            if return_reason:
                return fallback, True, 'DATA_PENDING'
            return fallback, True

        eff_norm = pd.Timestamp(effective_date).normalize().date()

        # R4 (dictamen auditor externo 2026-10-16): el filtro de sesion
        # y el control de cobertura son controles independientes. Si la
        # fecha efectiva (por dato) supera la sesion esperada (por
        # reloj), se capa a expected_session. Ejemplo: run pre-publish
        # en que el df_stocks ya tiene la sesion del dia por un cierre
        # parcial. Sin cap, la sesion publicada seria posterior a la
        # que el reloj considera disponible.
        if eff_norm > expected_norm:
            print(f"  R4/N-C6: cap effective_date {eff_norm} -> "
                  f"expected_session {expected_norm} (reloj por detras del dato).")
            eff_norm = expected_norm
            effective_date = pd.Timestamp(expected_norm)

        if expected_norm > eff_norm:
            print(f"  B2/N-C6: EXPECTED_SESSION_ABSENT "
                  f"(expected={expected_norm}, effective={eff_norm}, "
                  f"lag={(expected_norm - eff_norm).days}d). "
                  f"Omitiendo nueva observacion.")
            fallback = _load_latest_valid_breadth_snapshot(sb_path)
            if fallback is not None:
                print(f"  C2F: usando snapshot {fallback['date'].max().date()} como stale.")
            skip('sector_breadth', 'STALE',
                 path_override=sb_path if output_path is not None else None)
            if return_reason:
                return fallback, True, 'DATA_PENDING'
            return fallback, True

        # as_of_date = fecha efectiva (capped por R4). Garantiza que
        # df_market y df_stocks comparten sesion.
        sector_breadth_df = compute_sector_breadth(
            df_market, df_stocks, holdings_df, as_of_date=effective_date,
            temporal_meta=temporal_meta)

        if not sector_breadth_df.empty:
            date_val = sector_breadth_df['date'].max()
            persist('sector_breadth', sector_breadth_df, date_value=date_val,
                    path_override=sb_path if output_path is not None else None)
            print("  Sector Breadth & Health calculado.")
        else:
            skip('sector_breadth', 'EMPTY_DF',
                 path_override=sb_path if output_path is not None else None)
        if return_reason:
            return sector_breadth_df, False, None
        return sector_breadth_df, False
    except (KeyError, ValueError, TypeError, IndexError, AttributeError, OSError, RuntimeError, pd.errors.ParserError) as e:
        print(f"  Sector Breadth & Health omitido: {e}")
        if return_reason:
            return None, False, 'ERROR'
        return None, False


def compute_breadth_metrics(df_stocks, df_market, holdings_df, reference_date=None, temporal_meta=None):
    """Calcula Sector Breadth & Health + Momentum de amplitud.

    C2-followup: sector_breadth_is_stale indica si la observacion
    entregada es la ultima valida del historico (mercado cerrado) en
    lugar de una actualizacion del dia.

    Returns:
        dict con keys:
            sector_breadth_momentum_df, sector_breadth_df,
            sector_breadth_is_stale (bool),
            sector_breadth_stale_reason (str | None)
    """
    sb_df, sb_is_stale, sb_stale_reason = _compute_sector_breadth_health(
        df_stocks, df_market, holdings_df, reference_date=reference_date,
        temporal_meta=temporal_meta, return_reason=True)
    return {
        'sector_breadth_momentum_df': _compute_momentum_amplitud(df_stocks),
        'sector_breadth_df': sb_df,
        'sector_breadth_is_stale': sb_is_stale,
        'sector_breadth_stale_reason': sb_stale_reason,
    }
