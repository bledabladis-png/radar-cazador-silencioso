"""DT3 Fase 4: backfill de historico darkpool (FINRA + Yahoo).

Extraida de indicators/darkpool.py. Re-exportada para preservar
la API interna del modulo original.
"""
from datetime import timedelta

import pandas as pd
import yfinance as yf

from src.utils import safe_mean
from indicators.darkpool_io import _get_all_tickers


def _backfill_history(hist, finra):
    print("  Historial insuficiente. Descargando semanas historicas...")
    needed = 104 - len(hist)
    if needed <= 0:
        return hist
    MAX_PER_RUN = 1
    to_download = min(needed, MAX_PER_RUN)
    latest_week = finra.get_latest_week()
    if not latest_week:
        print("    No se pudo determinar la semana actual.")
        return hist
    current = pd.to_datetime(latest_week) - timedelta(weeks=1)
    tickers = _get_all_tickers()
    new_rows = []
    attempts = 0
    max_attempts = to_download * 5
    while len(new_rows) < to_download and attempts < max_attempts:
        week_str = current.strftime('%Y-%m-%d')
        attempts += 1
        if current in pd.to_datetime(hist['week']).values:
            current -= timedelta(weeks=1)
            continue
        try:
            ats_data = finra.get_all_tiers(week_str)
            if ats_data.empty:
                current -= timedelta(weeks=1)
                continue
            if 'issueSymbolIdentifier' in ats_data.columns and 'totalWeeklyShareQuantity' in ats_data.columns:
                ats_volume = ats_data.groupby('issueSymbolIdentifier')['totalWeeklyShareQuantity'].sum()
                ats_volume_dict = ats_volume.to_dict()
            else:
                current -= timedelta(weeks=1)
                continue
            end_date = current + timedelta(days=4)
            end_date_str = end_date.strftime('%Y-%m-%d')
            volumes = {}
            for t in tickers:
                try:
                    data = yf.download(t, start=week_str, end=end_date_str, progress=False, auto_adjust=True)
                    if not data.empty:
                        vol_col = ('Volume', t) if isinstance(data.columns, pd.MultiIndex) else 'Volume'
                        if vol_col in data.columns:
                            total = float(data[vol_col].sum())
                            if total > 0:
                                volumes[t] = total
                except Exception as e:
                    print(f'  [WARN] darkpool backfill {t}: {e}')
            if not volumes:
                current -= timedelta(weeks=1)
                continue
            resultados = []
            for t, vol_total in volumes.items():
                vol_ats = ats_volume_dict.get(t, 0)
                if vol_ats > 0:
                    dark_pool_pct = (vol_ats / vol_total) * 100
                    if dark_pool_pct <= 100:
                        resultados.append(dark_pool_pct)
            if resultados:
                media_dp = safe_mean(resultados)
                new_rows.append({'week': current, 'ratio': media_dp / 100})
                print(f"      OK: {week_str} - Ratio={media_dp/100:.4f} ({len(resultados)} tickers)")
            else:
                print(f"      Sin resultados para {week_str}")
        except Exception as e:
            print(f"      Error en {week_str}: {e}")
        current -= timedelta(weeks=1)
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        hist = pd.concat([hist, new_df], ignore_index=True)
        hist.sort_values('week', inplace=True)
        hist.reset_index(drop=True, inplace=True)
        print(f"    Historial: {len(hist)} semanas (faltan {104-len(hist)})")
    else:
        print(f"    No se descargaron nuevas semanas. El historial contiene {len(hist)} semanas.")
    return hist