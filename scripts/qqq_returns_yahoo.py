# -*- coding: utf-8 -*-
"""
Calcula rendimientos de QQQ usando precios ajustados de Yahoo Finance.

Fuente: Yahoo Finance (auto_adjust=True)
Salida: outputs/history/qqq_returns_yahoo.csv

No usa Invesco.
No calcula flujo primario.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT = PROJECT_ROOT / "outputs" / "history" / "qqq_returns_yahoo.csv"
TICKER = "QQQ"

def get_adjusted_prices(ticker: str) -> pd.Series:
    """Descarga precios ajustados de cierre desde Yahoo Finance."""
    df = yf.download(ticker, period="max", auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No se pudieron descargar precios para {ticker}")
    prices = df["Close"].squeeze()
    prices = prices.dropna()
    if len(prices) < 252:
        raise RuntimeError(f"Historial insuficiente para {ticker}: {len(prices)} filas")
    return prices

def calculate_returns(prices: pd.Series) -> dict:
    """Calcula rendimientos porcentuales para varios periodos."""
    latest = prices.iloc[-1]
    latest_date = prices.index[-1]

    year = latest_date.year
    last_day_prev_year = pd.Timestamp(year - 1, 12, 31)
    prev_year_prices = prices[prices.index <= last_day_prev_year]
    if prev_year_prices.empty:
        ytd = float("nan")
    else:
        ytd = (latest / prev_year_prices.iloc[-1] - 1) * 100

    def period_return(days: int) -> float:
        if len(prices) <= days:
            return float("nan")
        past_price = prices.iloc[-days - 1] if days > 0 else prices.iloc[0]
        return (latest / past_price - 1) * 100

    y1 = period_return(252)
    y3 = period_return(756)
    y5 = period_return(1260)
    y10 = period_return(2520)
    inception = (latest / prices.iloc[0] - 1) * 100

    return {
        "ytd": ytd,
        "y1": y1,
        "y3": y3,
        "y5": y5,
        "y10": y10,
        "inception": inception,
        "label": "marketPrice",
        "displayLabel": "QQQ (Yahoo Finance)",
        "effectiveDate": latest_date.strftime("%Y-%m-%d"),
        "performancePeriod": "daily",
    }

def save_csv(data: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([data])
    # Reordenar para que coincida con el formato esperado
    df = df[["ytd", "y1", "y3", "y5", "y10", "inception", "label", "displayLabel", "effectiveDate", "performancePeriod"]]
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

def main() -> None:
    print("Descargando precios de QQQ desde Yahoo Finance...")
    prices = get_adjusted_prices(TICKER)
    returns = calculate_returns(prices)
    save_csv(returns)
    print(f"Rendimientos QQQ calculados y guardados en {OUTPUT}")
    print(f"  YTD: {returns['ytd']:.2f}% | 1Y: {returns['y1']:.2f}% | 3Y: {returns['y3']:.2f}% | "
          f"5Y: {returns['y5']:.2f}% | 10Y: {returns['y10']:.2f}% | Inicio: {returns['inception']:.2f}%")

if __name__ == "__main__":
    main()
