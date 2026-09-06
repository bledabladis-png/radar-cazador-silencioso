# -*- coding: utf-8 -*-
"""
Sector Breadth & Health v1.0
Describe la distribucion interna de la fortaleza por sector.
No alimenta motores, scores, pesos ni State Machine.
"""
import pandas as pd
import numpy as np
from src.utils import get_col
from indicators.wyckoff import wyckoff_score, classify_wyckoff_phase

def compute_sector_breadth(df_market, df_stocks, holdings_df):
    rows = []
    for sector_etf, group in holdings_df.groupby('etf'):
        tickers = group['ticker'].tolist()
        # Métricas por ticker
        ema20_above = []
        ema50_above = []
        ema200_above = []
        rs_positive = []
        mom_positive = []
        wyckoff_phases = []
        nh_count = 0
        nl_count = 0
        advances = 0
        declines = 0
        unchanged = 0

        n_valid_ema20 = 0
        n_valid_ema50 = 0
        n_valid_ema200 = 0
        n_valid_nhnl = 0
        n_valid_momentum = 0
        n_valid_ad = 0

        # Precio del sector
        try:
            sector_price = get_col(df_market, sector_etf, 'Close')
        except KeyError:
            sector_price = None

        for ticker in tickers:
            try:
                close = get_col(df_stocks, ticker, 'Close')
                high = get_col(df_stocks, ticker, 'High')
                low = get_col(df_stocks, ticker, 'Low')
                volume = get_col(df_stocks, ticker, 'Volume')
            except KeyError:
                continue

            close = close.dropna()
            if len(close) == 0:
                continue

            # EMA20
            if len(close) >= 20:
                n_valid_ema20 += 1
                ema20 = close.ewm(span=20, min_periods=20, adjust=False).mean()
                if close.iloc[-1] > ema20.iloc[-1]:
                    ema20_above.append(1)
                else:
                    ema20_above.append(0)

            # EMA50
            if len(close) >= 50:
                n_valid_ema50 += 1
                ema50 = close.ewm(span=50, min_periods=50, adjust=False).mean()
                if close.iloc[-1] > ema50.iloc[-1]:
                    ema50_above.append(1)
                else:
                    ema50_above.append(0)

            # EMA200
            if len(close) >= 200:
                n_valid_ema200 += 1
                ema200 = close.ewm(span=200, min_periods=200, adjust=False).mean()
                if close.iloc[-1] > ema200.iloc[-1]:
                    ema200_above.append(1)
                else:
                    ema200_above.append(0)

            # RS positivo (definición oficial: rs_mom > 0)
            if sector_price is not None and len(close) >= 21:
                n_valid_momentum += 1
                common = close.index.intersection(sector_price.index)
                rs = close.loc[common] / sector_price.loc[common]
                rs_mom = np.log(rs).diff(20).iloc[-1]
                if pd.notna(rs_mom) and rs_mom > 0:
                    rs_positive.append(1)
                else:
                    rs_positive.append(0)

            # Momentum 20d
            if len(close) >= 21:
                mom = close.pct_change(20).iloc[-1]
                if pd.notna(mom) and mom > 0:
                    mom_positive.append(1)
                else:
                    mom_positive.append(0)

            # NH/NL 52 semanas (High/Low, 252 sesiones previas)
            if len(high) >= 253 and len(low) >= 253:
                n_valid_nhnl += 1
                prev_high = high.shift(1).rolling(252, min_periods=252).max().iloc[-1]
                prev_low = low.shift(1).rolling(252, min_periods=252).min().iloc[-1]
                if high.iloc[-1] >= prev_high:
                    nh_count += 1
                if low.iloc[-1] <= prev_low:
                    nl_count += 1

            # Advance/Decline diario (último día)
            if len(close) >= 2:
                n_valid_ad += 1
                daily_ret = close.iloc[-1] - close.iloc[-2]
                if daily_ret > 0:
                    advances += 1
                elif daily_ret < 0:
                    declines += 1
                else:
                    unchanged += 1

            # Wyckoff (consumir indicador oficial)
            if len(close) >= 60:
                try:
                    ticker_df = pd.DataFrame({
                        'Open': get_col(df_stocks, ticker, 'Open'),
                        'High': high,
                        'Low': low,
                        'Close': close,
                        'Volume': get_col(df_stocks, ticker, 'Volume')
                    }).dropna()
                    phase = classify_wyckoff_phase(ticker_df, ticker)
                    wyckoff_phases.append(phase)
                except Exception:
                    wyckoff_phases.append('INSUFICIENTE')

        # Agregar sector
        row = {
            'date': pd.Timestamp.now().normalize(),
            'sector': sector_etf,
            'n_total': len(tickers),
            'n_valid_ema20': n_valid_ema20,
            'n_valid_ema50': n_valid_ema50,
            'n_valid_ema200': n_valid_ema200,
            'n_valid_nhnl': n_valid_nhnl,
            'n_valid_momentum': n_valid_momentum,
            'n_valid_ad': n_valid_ad,
            'pct_above_ema20': (sum(ema20_above) / n_valid_ema20 * 100) if n_valid_ema20 else np.nan,
            'pct_above_ema50': (sum(ema50_above) / n_valid_ema50 * 100) if n_valid_ema50 else np.nan,
            'pct_above_ema200': (sum(ema200_above) / n_valid_ema200 * 100) if n_valid_ema200 else np.nan,
            'pct_rs_positive': (sum(rs_positive) / n_valid_momentum * 100) if n_valid_momentum else np.nan,
            'pct_momentum_positive': (sum(mom_positive) / n_valid_momentum * 100) if n_valid_momentum else np.nan,
            'count_accumulation': wyckoff_phases.count('ACCUMULATION'),
            'count_markup': wyckoff_phases.count('MARKUP'),
            'count_distribution': wyckoff_phases.count('DISTRIBUTION'),
            'count_markdown': wyckoff_phases.count('MARKDOWN'),
            'new_highs': nh_count,
            'new_lows': nl_count,
            'advances': advances,
            'declines': declines,
            'unchanged': unchanged,
            'ad_net': advances - declines,
            'advance_pct': (advances / (advances + declines) * 100) if (advances + declines) > 0 else np.nan,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df
