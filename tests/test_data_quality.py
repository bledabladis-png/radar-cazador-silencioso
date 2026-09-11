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