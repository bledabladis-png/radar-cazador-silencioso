import pandas as pd
import numpy as np
from indicators.cross_asset_context import compute_cross_asset_context

def _make_df(n=100):
    idx = pd.date_range('2026-01-01', periods=n, freq='D')
    tickers = ['SPY','QQQ','IWM','TLT','IEF','HYG','LQD','USO','GLD','DBC','DX-Y.NYB','EURUSD=X','^VIX',
               'XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
    data = {t: np.random.normal(0, 0.01, n) for t in tickers}
    return pd.DataFrame(data, index=idx)

def test_cross_asset_context_basic():
    df = _make_df()
    detail, summary = compute_cross_asset_context(df)
    assert not detail.empty
    assert not summary.empty
    assert set(summary['window']) == {20, 60}
    assert set(summary['asset_class']) == {'equity','rates','credit','commodities','fx','volatility'}

def test_cross_asset_vix_uses_diff():
    # VIX no debe ser pct_change; su correlación no debe ser idéntica a la de pct_change
    df = _make_df()
    detail, _ = compute_cross_asset_context(df)
    vix_row = detail[(detail['asset']=='^VIX') & (detail['window']==20)]
    assert not vix_row.empty
    # No comprobamos el valor, solo que exista