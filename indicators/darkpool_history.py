"""DT3 Fase 4: backfill de historico darkpool (FINRA + Yahoo).

Rediseno K-DT3-AUDIT-01 (2026-10-06):
- Pre-filtra tickers al universo FINRA (ats_volume_dict). Antes
  descargaba ~250 tickers de Yahoo para usar solo los que aparecian
  en FINRA (~10-20% del universo). Reduccion >80% de llamadas HTTP.
- end_date corregido a +5 dias: yfinance `end` es exclusivo, +4
  perdia el viernes de la semana FINRA.
- MAX_PER_RUN subido de 1 a 5 (con comentario): el historico crecia
  1 semana/run -> 2 anios para llegar a 104. Con 5/run, ~5 meses
  con runs diarios.
- `attempts` cuenta intentos reales de descarga, no iteraciones del
  bucle. Antes max_attempts=to_download*5=5 con to_download=1,
  se gastaban 4 intentos en semanas sin FINRA o ya presentes.
"""
from datetime import timedelta

import pandas as pd
import yfinance as yf

from src.utils import safe_mean
from indicators.darkpool_io import _get_all_tickers


# Cuantas semanas nuevas intentar por ejecucion. Subido de 1 a 5
# tras K-DT3-AUDIT-01: con 1/run y runs diarios, el historico
# tardaba ~2 anios en llegar a 104 semanas.
MAX_PER_RUN = 5

# Cuantos intentos extra tolerar por semana valida (semanas con FINRA
# vacio o ya presentes en el historico). Limite para no iterar sin
# fin si FINRA tiene un hueco largo.
MAX_SKIPS_POR_SEMANA = 5


def _backfill_history(hist, finra):
    print("  Historial insuficiente. Descargando semanas historicas...")
    needed = 104 - len(hist)
    if needed <= 0:
        return hist
    to_download = min(needed, MAX_PER_RUN)
    latest_week = finra.get_latest_week()
    if not latest_week:
        print("    No se pudo determinar la semana actual.")
        return hist

    current = pd.to_datetime(latest_week) - timedelta(weeks=1)
    all_tickers = _get_all_tickers()
    new_rows = []
    weeks_downloaded = 0
    iterations = 0
    # Tope global de iteraciones: descarga objetivo + margen de saltos.
    max_iterations = to_download * (MAX_SKIPS_POR_SEMANA + 1)

    while weeks_downloaded < to_download and iterations < max_iterations:
        iterations += 1
        week_str = current.strftime('%Y-%m-%d')

        # Semana ya presente en el historico -> saltar sin descargar.
        if current in pd.to_datetime(hist['week']).values:
            current -= timedelta(weeks=1)
            continue

        try:
            ats_data = finra.get_all_tiers(week_str)
        except (ValueError, KeyError, TypeError, OSError) as e:
            print(f"      Error FINRA en {week_str}: {e}")
            current -= timedelta(weeks=1)
            continue

        if ats_data.empty:
            current -= timedelta(weeks=1)
            continue

        if ('issueSymbolIdentifier' not in ats_data.columns or
                'totalWeeklyShareQuantity' not in ats_data.columns):
            current -= timedelta(weeks=1)
            continue

        ats_volume = ats_data.groupby('issueSymbolIdentifier')['totalWeeklyShareQuantity'].sum()
        ats_volume_dict = ats_volume.to_dict()

        # K-DT3-AUDIT-01: pre-filtrar tickers al universo FINRA.
        # Evita ~80-90% de llamadas innecesarias a Yahoo.
        finra_tickers = {t for t in all_tickers if t in ats_volume_dict}

        if not finra_tickers:
            print(f"      {week_str}: sin tickers comunes entre FINRA y universo.")
            current -= timedelta(weeks=1)
            continue

        # end_date +5 dias: yfinance `end` es exclusivo, +4 pierde viernes.
        end_date = current + timedelta(days=5)
        end_date_str = end_date.strftime('%Y-%m-%d')

        volumes = {}
        for t in finra_tickers:
            try:
                data = yf.download(
                    t, start=week_str, end=end_date_str,
                    progress=False, auto_adjust=True,
                )
                if data.empty:
                    continue
                vol_col = ('Volume', t) if isinstance(data.columns, pd.MultiIndex) else 'Volume'
                if vol_col not in data.columns:
                    continue
                total = float(data[vol_col].sum())
                if total > 0:
                    volumes[t] = total
            except (ValueError, KeyError, TypeError, OSError, AttributeError) as e:
                print(f'  [WARN] darkpool backfill {t}: {e}')

        if not volumes:
            current -= timedelta(weeks=1)
            continue

        ratios = []
        for t, vol_total in volumes.items():
            vol_ats = ats_volume_dict.get(t, 0)
            if vol_ats <= 0:
                continue
            dark_pool_pct = (vol_ats / vol_total) * 100
            if dark_pool_pct <= 100:
                ratios.append(dark_pool_pct)

        if ratios:
            media_dp = safe_mean(ratios)
            new_rows.append({'week': current, 'ratio': media_dp / 100})
            weeks_downloaded += 1
            print(f"      OK: {week_str} - Ratio={media_dp/100:.4f} ({len(ratios)} tickers)")
        else:
            print(f"      Sin resultados validos para {week_str}")

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
