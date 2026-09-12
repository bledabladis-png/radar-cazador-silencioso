# tests/test_c4_code.py
"""Tests C4-code (2026-09-12): writers historicos no usan fecha de ejecucion.

Cubre:
- Helper _observation_date_from_df: contrato completo.
- save_regime_history: fecha derivada de df_macro_manual.
- sector_dispersion: reference_date inyectado.
- sector_rank_history: date obligatorio (ValueError sin el).
- leader_representativeness: reference_date obligatorio.
- Regresion: run sabado + datos viernes -> writer escribe viernes.
- Inventario: los 15 writers B2 no usan now() como fecha de observacion.
"""
from pathlib import Path
import re

import pandas as pd
import pytest

from src.utils import _observation_date_from_df
from src.pipeline.finalize import save_regime_history
from indicators.sector_dispersion import compute_sector_dispersion
from indicators.sector_rank_history import update_rank_history
from indicators.leader_representativeness import compute_leader_representativeness


# ============================================================
# HELPER - contrato
# ============================================================

def test_helper_df_none():
    assert _observation_date_from_df(None) is None


def test_helper_df_vacio():
    assert _observation_date_from_df(pd.DataFrame()) is None


def test_helper_index_datetime():
    df = pd.DataFrame({'x': [1, 2, 3]},
                      index=pd.date_range('2026-09-09', periods=3, freq='D'))
    assert _observation_date_from_df(df) == pd.Timestamp('2026-09-11')


def test_helper_index_con_nat():
    idx = pd.DatetimeIndex(['2026-09-09', '2026-09-10', None, '2026-09-11'])
    df = pd.DataFrame({'x': [1, 2, 3, 4]}, index=idx)
    assert _observation_date_from_df(df) == pd.Timestamp('2026-09-11')


def test_helper_col_valida():
    df = pd.DataFrame({
        'date': ['2026-09-09', '2026-09-11', '2026-09-10'],
        'x': [1, 2, 3],
    })
    assert _observation_date_from_df(df, col='date') == pd.Timestamp('2026-09-11')


def test_helper_col_inexistente_lanza_keyerror():
    df = pd.DataFrame({'date': ['2026-09-11'], 'x': [1]})
    with pytest.raises(KeyError, match="columna 'data' no encontrada"):
        _observation_date_from_df(df, col='data')


def test_helper_col_todo_nat():
    df = pd.DataFrame({'date': [None, None, None], 'x': [1, 2, 3]})
    assert _observation_date_from_df(df, col='date') is None


def test_helper_no_fallback_a_now(monkeypatch):
    # Con df sin fecha valida, no debe devolver una fecha de hoy.
    df = pd.DataFrame({'x': [1, 2, 3]})  # RangeIndex
    result = _observation_date_from_df(df)
    assert result is None
    # Verificar que no coincide con today
    hoy = pd.Timestamp.now().normalize()
    assert result != hoy


# ============================================================
# save_regime_history
# ============================================================

def test_save_regime_history_usa_fecha_macro(monkeypatch, tmp_path):
    """run sabado + df_macro_manual con ultima fecha viernes 11 -> escribe 11."""
    escrituras = []

    def fake_to_csv(self, path, **kw):
        escrituras.append((str(path), self.copy()))
        return None

    monkeypatch.setattr(pd.DataFrame, 'to_csv', fake_to_csv)
    monkeypatch.setattr('os.path.exists', lambda p: False)

    df_macro = pd.DataFrame({
        'date': ['2026-09-09', '2026-09-10', '2026-09-11'],
        'cpi': [1.0, 1.1, 1.2],
    })
    macro_score = pd.Series([0.3, 0.35])
    sector_results = {'regime': 'NARROW RALLY'}

    save_regime_history(macro_score, 'INFLATION SHOCK', 0.5,
                        'ESTRECHA', 'LOW', sector_results,
                        df_macro_manual=df_macro)

    assert len(escrituras) == 1
    _, df = escrituras[0]
    assert pd.Timestamp(df.iloc[0]['date']) == pd.Timestamp('2026-09-11')


def test_save_regime_history_sin_df_macro_no_escribe(monkeypatch, capsys):
    """Sin df_macro_manual -> no escribe, emite WARN."""
    escrituras = []

    def fake_to_csv(self, path, **kw):
        escrituras.append((str(path), self.copy()))
        return None

    monkeypatch.setattr(pd.DataFrame, 'to_csv', fake_to_csv)

    macro_score = pd.Series([0.3, 0.35])
    sector_results = {'regime': 'NARROW RALLY'}

    save_regime_history(macro_score, 'INFLATION SHOCK', 0.5,
                        'ESTRECHA', 'LOW', sector_results,
                        df_macro_manual=None)

    assert len(escrituras) == 0
    captured = capsys.readouterr()
    assert 'save_regime_history' in captured.out or 'sin fecha macro' in captured.out


# ============================================================
# sector_dispersion
# ============================================================

def test_sector_dispersion_usa_reference_date():
    rank = [('XLE', 0.05), ('XLK', -0.02), ('XLF', 0.01),
            ('XLV', -0.01), ('XLI', -0.03), ('XLP', 0.005),
            ('XLY', -0.015), ('XLU', -0.02), ('XLB', 0.0)]
    df = compute_sector_dispersion(rank, reference_date='2026-09-11')
    assert pd.Timestamp(df.iloc[0]['date']) == pd.Timestamp('2026-09-11')


def test_sector_dispersion_sin_reference_date_valueerror():
    rank = [('XLE', 0.05), ('XLK', -0.02)]
    with pytest.raises(ValueError, match='reference_date'):
        compute_sector_dispersion(rank)


# ============================================================
# sector_rank_history
# ============================================================

def test_sector_rank_history_sin_date_valueerror(tmp_path):
    ranking = [('XLK', 'Tech', 0.5, 'RANGE')] * 3
    sector_results = {'ranking': ranking}
    with pytest.raises(ValueError, match='date'):
        update_rank_history(sector_results, str(tmp_path / 'hist.csv'))


# ============================================================
# leader_representativeness
# ============================================================

def test_leader_representativeness_sin_reference_date_valueerror(tmp_path):
    with pytest.raises(ValueError, match='reference_date'):
        compute_leader_representativeness(pd.DataFrame(), str(tmp_path / 'c.csv'))


def test_leader_representativeness_con_reference_date(tmp_path):
    # Construir un sector_concentration.csv minimo
    conc = tmp_path / 'sector_concentration.csv'
    pd.DataFrame([{
        'date': '2026-09-11', 'sector': 'XLK',
        'rs_median': 1.0, 'momentum_median': 0.01,
        'flow_median': 0.5, 'wls_median': 0.3,
        'wyckoff_median': 0.5,
        'n_valid_rs': 20, 'n_valid_momentum': 20,
    }]).to_csv(conc, index=False)

    leader_df = pd.DataFrame([{
        'sector': 'XLK', 'ticker': 'AAPL',
        'rs_mom': 1.1, 'flow_proxy_z': 0.6, 'wls': 0.4,
        'sector_rank_pct': 0.9,
    }])

    df = compute_leader_representativeness(
        leader_df, str(conc), reference_date='2026-09-11')

    if not df.empty:
        assert pd.Timestamp(df.iloc[0]['date']) == pd.Timestamp('2026-09-11')


# ============================================================
# T-C4-code-10: writer nunca usa execution date como observation date
# ============================================================

def test_t_c4_code_10_sabado_datos_viernes(monkeypatch):
    """Run sabado, datos del viernes -> la fecha escrita es el viernes."""
    escrituras = []

    def fake_to_csv(self, path, **kw):
        escrituras.append(self.copy())
        return None

    monkeypatch.setattr(pd.DataFrame, 'to_csv', fake_to_csv)
    monkeypatch.setattr('os.path.exists', lambda p: False)

    df_macro = pd.DataFrame({'date': ['2026-09-11'], 'x': [1.0]})
    save_regime_history(pd.Series([0.5]), 'X', 0.5, 'Y', 'Z',
                        {'regime': 'R'}, df_macro_manual=df_macro)

    assert len(escrituras) == 1
    fecha_escrita = pd.Timestamp(escrituras[0].iloc[0]['date'])
    assert fecha_escrita == pd.Timestamp('2026-09-11')
    assert fecha_escrita != pd.Timestamp('2026-09-12')


# ============================================================
# T-C4-code-9: inventario post-patch
# ============================================================

def test_t_c4_code_9_writers_b2_sin_now():
    """Los 15 writers B2 no deben usar now()/today() como fecha de observacion."""
    root = Path(__file__).resolve().parent.parent
    targets = [
        'indicators/cross_asset_context.py',
        'indicators/sector_correlation.py',
        'indicators/sector_dispersion.py',
        'indicators/sector_regime_matrix.py',
        'indicators/leader_representativeness.py',
        'indicators/rs_internal.py',
        'indicators/sector_concentration.py',
        'indicators/sector_leader_divergence.py',
        'indicators/sector_wyckoff_distribution.py',
        'indicators/volatility_structure.py',
        'indicators/evidence_matrix.py',
        'indicators/sector_rank_history.py',
        'src/pipeline/finalize.py',
        'src/pipeline/sectors_base.py',
        'src/pipeline/engines.py',
    ]
    patrones = [
        r'Timestamp\.now\(\)',
        r'datetime\.now\(\)',
        r'datetime\.today\(\)',
        r'pd\.Timestamp\.today\(\)',
    ]
    residuales = []
    for rel in targets:
        f = root / rel
        if not f.exists():
            continue
        for i, line in enumerate(f.read_text(encoding='utf-8-sig').splitlines(), 1):
            for p in patrones:
                if re.search(p, line):
                    residuales.append(f'{rel}:{i}: {line.strip()}')
                    break
    assert residuales == [], f'Residuales: {residuales}'
