import numpy as np
from indicators.rs_internal import classify_rs

def test_classify_rs():
    assert classify_rs(0.05, 0.03) == 'Liderazgo relativo doble'
    assert classify_rs(0.05, -0.02) == 'Fortaleza sectorial'
    assert classify_rs(-0.05, 0.02) == 'Liderazgo interno en sector débil'
    assert classify_rs(-0.05, -0.03) == 'Debilidad relativa doble'
    assert classify_rs(np.nan, 0.02) == 'N/D'
    assert classify_rs(0.05, np.nan) == 'N/D'


def test_compute_rs_internal_normaliza_ticker_de_holdings():
    """Holdings con BRK.B, parquet con BRK-B (2026-10-30).

    Antes: get_col(df_stocks, 'BRK.B', 'Close') -> KeyError -> continue
    silencioso. BRK.B nunca aparecia en rs_internal aunque estuviera
    descargado. Bug medido en XLF: 33 tickers vs 34 de sector_concentration.
    """
    import pandas as pd
    import numpy as np

    # Parquet: BRK-B (yahoo) + AAPL (control)
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    close_data = {
        ("Close", "BRK-B"): np.linspace(100, 120, 100),
        ("Close", "AAPL"): np.linspace(150, 180, 100),
        ("Close", "XLF"): np.linspace(35, 40, 100),  # ETF del sector
        ("Close", "SPY"): np.linspace(400, 450, 100),  # benchmark
    }
    df_stocks = pd.DataFrame(close_data, index=idx)
    # df_market: ETF XLF y SPY
    df_market = pd.DataFrame({
        ("Close", "XLF"): np.linspace(35, 40, 100),
        ("Close", "SPY"): np.linspace(400, 450, 100),
    }, index=idx)
    # holdings con BRK.B (dot)
    holdings = pd.DataFrame({
        "etf": ["XLF", "XLF"],
        "ticker": ["BRK.B", "AAPL"],
        "weight": [50.0, 50.0],
    })

    from indicators.rs_internal import compute_rs_internal
    out = compute_rs_internal(
        df_stocks, holdings, df_market, benchmark="SPY"
    )
    assert not out.empty
    tickers = set(out["ticker"].tolist())
    assert "BRK.B" in tickers, (
        f"BRK.B de holdings no aparece en rs_internal. "
        f"Tickers: {sorted(tickers)}. Bug: get_col sin normalizar "
        f"descarta BRK.B (parquet tiene BRK-B)."
    )
