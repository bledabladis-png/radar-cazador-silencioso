import pandas as pd
import numpy as np

def filter_relevant_markets(df):
    keywords = [
        "E-MINI S&P 500", "NASDAQ-100", "RUSSELL",
        "UST", "FED FUNDS", "SOFR",
        "CRUDE OIL", "BBG COMMODITY",
        "USD INDEX", "EURO FX", "JAPANESE YEN",
        "VIX"
    ]
    # Asegurar que la columna 'market' existe
    if 'market' not in df.columns:
        return pd.DataFrame()
    mask = df["market"].str.contains("|".join(keywords), case=False, na=False)
    return df[mask].copy()

def map_cftc_to_sector(market_name, z_score):
    mapping = {
        "E-MINI S&P 500": "SPY",
        "NASDAQ-100": "XLK",
        "RUSSELL": "XLI",
        "UST 10Y": "XLF",
        "FED FUNDS": "XLF",
        "SOFR": "XLF",
        "CRUDE OIL": "XLE",
        "BBG COMMODITY": "XLE",
        "USD INDEX": "MACRO",
        "EURO FX": "MACRO",
        "JAPANESE YEN": "MACRO",
        "VIX": "RISK"
    }
    for key, sector in mapping.items():
        if key in market_name:
            return sector, z_score
    return None, None

def aggregate_sector_signals(df):
    signals = {}
    for _, row in df.iterrows():
        market = row.get('market', '')
        z = row.get('cftc_z', np.nan)
        if pd.isna(z):
            continue
        sector, z_val = map_cftc_to_sector(market, z)
        if sector is None:
            continue
        if sector not in signals:
            signals[sector] = []
        signals[sector].append(z_val)
    result = {}
    for sector, z_list in signals.items():
        median_z = np.median(z_list)
        n = len(z_list)
        if median_z > 1.5:
            interp = "🔥 Convicción FUERTEMENTE alcista"
        elif median_z > 0.8:
            interp = "📈 Convicción alcista moderada"
        elif median_z < -1.5:
            interp = "⚠️ Convicción FUERTEMENTE bajista"
        elif median_z < -0.8:
            interp = "📉 Convicción bajista moderada"
        else:
            interp = "➖ Neutral"
        result[sector] = {
            'z_median': median_z,
            'n_instruments': n,
            'interpretacion': interp
        }
    return result
