import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# Asegurar que el root del proyecto esté en sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.providers.blackrock_iwm_fund_data import (
    parse_historical_sheet,
    compute_primary_flow,
)
from data.providers.amundi_fund_data import (
    parse_historical_series,
    compute_primary_flow as compute_amundi_flow,
)

# ---------------------------------------------------------------------------
# IWM
# ---------------------------------------------------------------------------

def test_iwm_parse_historical_sheet():
    xml = r'''<ss:Worksheet ss:Name="Historical">
      <ss:Table>
        <ss:Row><ss:Cell><ss:Data ss:Type="String">As Of</ss:Data></ss:Cell><ss:Cell><ss:Data ss:Type="String">NAV per Share</ss:Data></ss:Cell><ss:Cell><ss:Data ss:Type="String">Ex-Dividends</ss:Data></ss:Cell><ss:Cell><ss:Data ss:Type="String">Shares Outstanding</ss:Data></ss:Cell></ss:Row>
        <ss:Row><ss:Cell><ss:Data ss:Type="String">Aug 14, 2026</ss:Data></ss:Cell><ss:Cell><ss:Data ss:Type="Number">304.981716</ss:Data></ss:Cell><ss:Cell><ss:Data ss:Type="Number">0.0</ss:Data></ss:Cell><ss:Cell><ss:Data ss:Type="Number">273250000</ss:Data></ss:Cell></ss:Row>
        <ss:Row><ss:Cell><ss:Data ss:Type="String">Aug 13, 2026</ss:Data></ss:Cell><ss:Cell><ss:Data ss:Type="Number">303.123456</ss:Data></ss:Cell><ss:Cell><ss:Data ss:Type="Number">0.0</ss:Data></ss:Cell><ss:Cell><ss:Data ss:Type="Number">273000000</ss:Data></ss:Cell></ss:Row>
      </ss:Table>
    </ss:Worksheet>'''

    df = parse_historical_sheet(xml)
    assert len(df) == 2
    assert list(df.columns) == ['date', 'nav', 'shares_outstanding']
    # El DataFrame está ordenado ascendentemente por fecha
    assert df.iloc[0]['shares_outstanding'] == 273000000.0
    assert df.iloc[1]['shares_outstanding'] == 273250000.0

def test_iwm_compute_primary_flow():
    df = pd.DataFrame({
        'date': pd.to_datetime(['2026-08-13', '2026-08-14']),
        'nav': [303.123456, 304.981716],
        'shares_outstanding': [273000000.0, 273250000.0],
    })
    result = compute_primary_flow(df)
    row = result.iloc[-1]
    assert row['shares_change'] == 250000.0
    expected_flow = 250000.0 * 304.981716
    assert abs(row['primary_flow_usd'] - expected_flow) < 1.0
    assert 'primary_flow_pct' in result.columns
    assert 'primary_flow_z' in result.columns


def test_amundi_parse_historical_series():
    product = {
        'historics': [
            {
                'indicator': 'sharesOut',
                'historicalData': [
                    {'date': 1735776000000, 'data': 1857685.0},
                    {'date': 1735862400000, 'data': 1857685.0},
                ]
            },
            {
                'indicator': 'officialNav',
                'historicalData': [
                    {'date': 1735776000000, 'data': 118.1195},
                    {'date': 1735862400000, 'data': 117.886},
                ]
            },
            {
                'indicator': 'fundAumInMCcy',
                'historicalData': [
                    {'date': 1735776000000, 'data': 295848924.79},
                    {'date': 1735862400000, 'data': 295264082.93},
                ]
            },
        ]
    }
    df = parse_historical_series(product)
    assert set(['sharesOut', 'officialNav']).issubset(df.columns)
    assert len(df) == 2
    assert df.iloc[0]['sharesOut'] == 1857685.0

def test_amundi_compute_primary_flow():
    df = pd.DataFrame({
        'date': pd.to_datetime(['2026-08-11', '2026-08-12']),
        'sharesOut': [2200205.0, 2200205.0],
        'officialNav': [213.6423, 213.5403],
        'fundAumInMCcy': [954296507.03, 953841148.45],
    })
    result = compute_amundi_flow(df)
    row = result.iloc[-1]
    assert row['shares_outstanding'] == 2200205.0
    assert 'estimated_flow_eur' in result.columns
    assert 'flow_pct_assets' in result.columns
    assert not np.isnan(row['estimated_flow_eur'])

# ---------------------------------------------------------------------------
# Ejecutar si se llama directamente
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
