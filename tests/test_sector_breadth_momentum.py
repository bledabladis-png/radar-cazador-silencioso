import pandas as pd
from indicators.sector_breadth_momentum import compute_sector_breadth_momentum

def _crear_csv(tmp_path):
    sectors = ['XLK', 'XLF']
    fechas = pd.date_range('2026-01-01', periods=30, freq='D')
    data = []
    for s in sectors:
        for i, f in enumerate(fechas):
            data.append({
                'date': f,
                'sector': s,
                'pct_above_ema20': 50 + i,   # creciente
                'pct_above_ema50': 50 + i*0.5,
                'pct_above_ema200': 60 + i*0.2,
            })
    df = pd.DataFrame(data)
    csv_path = tmp_path / 'sector_breadth.csv'
    df.to_csv(csv_path, index=False)
    return csv_path

def test_deltas_basicos(tmp_path):
    csv_path = _crear_csv(tmp_path)
    out = compute_sector_breadth_momentum(str(csv_path))
    assert not out.empty
    assert len(out) == 2
    # Para XLK, el último día es índice 29, hace 5 días índice 24.
    # delta5d = (50+29) - (50+24) = 5
    xlk = out[out['sector']=='XLK'].iloc[0]
    assert xlk['delta_5d_ema20'] == 5.0
    # delta20d = (50+29)-(50+9)=20
    assert xlk['delta_20d_ema20'] == 20.0

def test_fechas_no_consecutivas(tmp_path):
    fechas = [pd.Timestamp('2026-01-01'), pd.Timestamp('2026-01-02'), pd.Timestamp('2026-01-03'),
              pd.Timestamp('2026-01-04'), pd.Timestamp('2026-01-05'), pd.Timestamp('2026-01-20')]
    data = []
    for i, f in enumerate(fechas):
        data.append({'date': f, 'sector': 'XLK', 'pct_above_ema20': 50+i, 'pct_above_ema50': 50, 'pct_above_ema200': 50})
    df = pd.DataFrame(data)
    csv_path = tmp_path / 'breadth_salto.csv'
    df.to_csv(csv_path, index=False)
    out = compute_sector_breadth_momentum(str(csv_path))
    xlk = out[out['sector']=='XLK'].iloc[0]
    # Delta1d entre 01-20 y 01-05 es > 3 días => NaN
    assert pd.isna(xlk['delta_1d_ema20'])

# ============================================================
# P3 (2026-10-21, auditor externo): umbral aplicado al redondeado
# ============================================================
from indicators.sector_breadth_momentum import _is_deterioration, _is_expansion


def test_deterioration_caso_xlf_borde():
    """Caso real 2026-10-20: delta=-4.9999 (display -5.0) -> True."""
    assert _is_deterioration(-4.9999) is True


def test_deterioration_exacto_menos_5():
    assert _is_deterioration(-5.0) is True


def test_deterioration_por_encima():
    assert _is_deterioration(-4.9) is False


def test_deterioration_por_debajo():
    assert _is_deterioration(-5.2) is True


def test_deterioration_nan():
    assert _is_deterioration(float("nan")) is False


def test_expansion_por_encima():
    assert _is_expansion(5.0, 65.0) is True


def test_expansion_por_debajo():
    assert _is_expansion(4.9, 65.0) is False


def test_expansion_nivel_bajo():
    assert _is_expansion(5.0, 55.0) is False


def test_expansion_nan():
    assert _is_expansion(float("nan"), 65.0) is False
