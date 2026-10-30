import pandas as pd
from indicators.sector_concentration import safe_quantile, compute_sector_concentration

def test_safe_quantile_min_obs():
    s = pd.Series([1,2,3,4])
    assert pd.isna(safe_quantile(s, 0.5))
    s = pd.Series([1,2,3,4,5])
    assert safe_quantile(s, 0.5) == 3.0

def test_compute_concentration_synthetic():
    # full_metrics_df con 5 tickers, sector XLK
    metrics = pd.DataFrame({
        'ticker': ['A','B','C','D','E'],
        'sector': ['XLK']*5,
        'rs': [1.1, 0.9, 1.2, 0.8, 1.0],
        'rs_mom': [0.01, -0.02, 0.03, 0.00, 0.02],
        'flow_proxy_z': [0.5, 0.2, -0.1, 0.3, 0.0],
        'wyckoff_score': [0.8, 0.6, 0.7, 0.5, 0.9],
        'wls': [1.0, 0.7, 1.2, 0.6, 0.9],
    })
    # df_stocks con Close para generar ret20
    idx = pd.date_range('2026-01-01', periods=30, freq='D')
    df_stocks = pd.DataFrame({
        ('A','Close'): [10 + i*0.1 for i in range(30)],
        ('B','Close'): [10 + i*0.2 for i in range(30)],
        ('C','Close'): [10 + i*0.3 for i in range(30)],
        ('D','Close'): [10 + i*0.05 for i in range(30)],
        ('E','Close'): [10 + i*0.15 for i in range(30)],
    }, index=idx)
    df_stocks.columns = pd.MultiIndex.from_tuples(df_stocks.columns)
    holdings = pd.DataFrame({'etf': ['XLK']*5, 'ticker': ['A','B','C','D','E']})
    result = compute_sector_concentration(df_stocks, holdings, metrics, reference_date=df_stocks.index[-1])
    assert not result.empty
    assert len(result) == 1
    assert result.iloc[0]['sector'] == 'XLK'
    assert pd.notna(result.iloc[0]['top1_positive_return_concentration'])
    assert pd.notna(result.iloc[0]['rs_median'])


def test_compute_concentration_sin_reference_date_valueerror():
    """FU-008 (2026-09-13): reference_date obligatorio."""
    metrics = pd.DataFrame({
        'ticker': ['A'], 'sector': ['XLK'],
        'rs': [1.0], 'rs_mom': [0.01], 'flow_proxy_z': [0.1],
        'wyckoff_score': [0.5], 'wls': [1.0],
    })
    idx = pd.date_range('2026-01-01', periods=30, freq='D')
    df_stocks = pd.DataFrame({('A','Close'): [10 + i*0.1 for i in range(30)]}, index=idx)
    df_stocks.columns = pd.MultiIndex.from_tuples(df_stocks.columns)
    holdings = pd.DataFrame({'etf': ['XLK'], 'ticker': ['A']})
    import pytest
    with pytest.raises(ValueError, match='reference_date'):
        compute_sector_concentration(df_stocks, holdings, metrics)


def test_compute_concentration_corre_sin_lideres(tmp_path):
    """FU-008-b (2026-09-13): el writer se ejecuta aunque leader_df este vacio."""
    from src.pipeline.sector_metrics import _compute_concentration
    idx = pd.date_range('2026-01-01', periods=30, freq='D')
    df_stocks = pd.DataFrame({('A','Close'): [10 + i*0.1 for i in range(30)]}, index=idx)
    df_stocks.columns = pd.MultiIndex.from_tuples(df_stocks.columns)
    holdings = pd.DataFrame({'etf': ['XLK'], 'ticker': ['A']})
    metrics = pd.DataFrame({
        'ticker': ['A'], 'sector': ['XLK'],
        'rs': [1.0], 'rs_mom': [0.01], 'flow_proxy_z': [0.1],
        'wyckoff_score': [0.5], 'wls': [1.0],
    })
    target = tmp_path / 'sector_concentration.csv'
    result = _compute_concentration(
        df_stocks, holdings, metrics,
        reference_date=df_stocks.index[-1],
        sc_path=target,
    )
    assert result is not None and not result.empty
    assert target.exists()
    saved = pd.read_csv(target)
    assert pd.notna(saved.iloc[0]['date'])
    assert str(saved.iloc[0]['date']).startswith('2026-')


def test_sector_concentration_normaliza_ticker_de_holdings():
    """Holdings con BRK.B, parquet con BRK-B (2026-10-30).

    Antes: _get_series(df_stocks, 'BRK.B', 'Close') -> Series vacia
    -> continue. BRK.B no aparecia en ret_df, por lo que
    n_valid_return20 podia ser menor que n_valid_rs. Bug medido en
    XLF: 33 vs 34.
    """
    import pandas as pd
    import numpy as np

    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    close_data = {
        ("Close", "BRK-B"): np.linspace(100, 120, 100),
        ("Close", "AAPL"): np.linspace(150, 180, 100),
    }
    df_stocks = pd.DataFrame(close_data, index=idx)
    holdings = pd.DataFrame({
        "etf": ["XLF", "XLF"],
        "ticker": ["BRK.B", "AAPL"],
        "weight": [50.0, 50.0],
    })
    # full_metrics_df simula el output de stock_metrics (ya normalizado)
    full_metrics = pd.DataFrame({
        "ticker": ["BRK-B", "AAPL"],
        "sector": ["XLF", "XLF"],
        "rs": [1.5, 1.2],
        "rs_mom": [0.01, 0.02],
        "flow_proxy_z": [0.3, 0.4],
        "wls": [0.1, 0.2],
        "wyckoff_score": [0.5, 0.6],
    })

    from indicators.sector_concentration import compute_sector_concentration
    out = compute_sector_concentration(
        df_stocks, holdings, full_metrics,
        reference_date=idx[-1],
    )
    assert not out.empty
    xlf = out[out["sector"] == "XLF"]
    assert not xlf.empty
    # n_valid_return20 debe ser 2 (BRK.B + AAPL), no 1
    assert int(xlf["n_valid_return20"].iloc[0]) == 2, (
        f"n_valid_return20 = {xlf['n_valid_return20'].iloc[0]}, "
        f"esperado 2. BRK.B de holdings no se normalizo a BRK-B antes "
        f"de _get_series."
    )
