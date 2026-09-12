from indicators.data_quality import classify_freshness, compute_data_quality

def test_freshness_daily():
    assert classify_freshness(2, 'daily') == 'CURRENT'
    assert classify_freshness(5, 'daily') == 'RECENT'
    assert classify_freshness(10, 'daily') == 'STALE'
    assert classify_freshness(20, 'daily') == 'ARCHIVAL'

def test_freshness_finra():
    assert classify_freshness(28, 'finra') == 'CURRENT'
    assert classify_freshness(50, 'finra') == 'STALE'

def test_data_quality_basic():
    df = compute_data_quality()
    assert not df.empty
    assert {'source','last_date','freshness','coverage'}.issubset(df.columns)


def test_data_quality_aplica_walk_back_fechas_no_bursatiles():
    """FU-007-b (2026-09-13): last_date siempre es dia bursatil para sources daily/fred.

    Los sources con frecuencia finra/sec/cftc pueden tener fechas no bursatiles
    por diseno (semanal, trimestral). Solo validamos daily y fred.
    """
    import pandas as pd
    from src.market_calendar import is_market_day
    df = compute_data_quality()
    assert not df.empty
    for _, row in df.iterrows():
        if row['frequency'] not in ('daily', 'fred'):
            continue
        ld = row['last_date']
        if pd.isna(ld):
            continue
        d = pd.to_datetime(ld).date()
        assert is_market_day(d), f"{row['source']}: last_date={d} no es bursatil"
