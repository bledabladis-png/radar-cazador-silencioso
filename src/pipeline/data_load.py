# -*- coding: utf-8 -*-
"""Fase 1 del pipeline: descarga y validacion de datos.

Extraido de run.py (refactor C2, fase C2-3).
"""

from src.data_loader import download_market_data
from src.macro_manual_loader import load_macro_manual
from src.utils import trim_to_last_valid_date
from data.validator import validate_market_data


def load_all_data():
    """Descarga mercado, valida, y carga datos macro manuales.

    Returns:
        dict | None: Diccionario con keys:
            - 'df_market': DataFrame trimmeado y validado (o None si fallo critico)
            - 'df_macro_manual': DataFrame macro o None
            - 'valid_tickers': lista de tickers validos
            - 'issues': dict de issues por ticker
        Devuelve None si hay fallo critico que debe abortar main().
    """
    print("Descargando datos de mercado...")
    df_market = download_market_data()
    if df_market is None or df_market.empty:
        print("Error: no se pudieron descargar datos.")
        return None

    df_market = trim_to_last_valid_date(df_market)
    if df_market is None or df_market.empty:
        print("Error: no hay datos validos de mercado.")
        return None

    print("Validando datos...")
    valid, issues = validate_market_data(df_market)
    if issues:
        for t, msg in issues.items():
            print(f"  {t}: {msg}")

    if len(valid) < 5:
        print("Pocos tickers validos. Abortando.")
        return None

    print("Cargando datos macro manuales (si existen)...")
    df_macro_manual = load_macro_manual()
    if df_macro_manual is not None:
        print(f"  Datos manuales cargados: {len(df_macro_manual)} filas.")

    return {
        'df_market': df_market,
        'df_macro_manual': df_macro_manual,
        'valid_tickers': valid,
        'issues': issues,
    }
