"""Tests de slice_last_n_sessions (P7-ROLLING-WINDOW-GAPS).

Helper creado 2026-10-28 para evitar `df.iloc[-n:]` sobre series con
huecos. Extraido del fix previo de sector_correlation.py. Cierra el
mismo patron en cross_asset_context y options.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.market_calendar import slice_last_n_sessions, get_trading_days


def _serie_sin_huecos(n=100):
    idx = pd.bdate_range("2026-01-01", periods=n, freq="B")
    return pd.Series(np.arange(n, dtype=float), index=idx)


def test_sin_huecos_devuelve_ultimas_n():
    s = _serie_sin_huecos(100)
    out = slice_last_n_sessions(s, 20)
    assert len(out) == 20
    assert out.iloc[-1] == 99.0


def test_con_un_hueco_devuelve_n_sesiones_no_n_filas():
    """Serie con 1 hueco DENTRO de la ventana pedida: iloc[-20:]
    cogeria 19 sesiones + el hueco como fila. El helper reindexa y
    devuelve 20 SESIONES, con NaN en la posicion del hueco.
    """
    s = _serie_sin_huecos(100)
    # Quitar la posicion -5 (dentro de las ultimas 20 sesiones).
    hueco_pos = s.index[-5]
    s = s.drop(hueco_pos)
    assert len(s) == 99
    out = slice_last_n_sessions(s, 20)
    assert len(out) == 20, f"esperado 20 sesiones, obtenido {len(out)}"
    assert hueco_pos in out.index, "la sesion del hueco debe aparecer con NaN"
    assert pd.isna(out.loc[hueco_pos])


def test_con_muchos_huecos_sigue_devolviendo_n_sesiones():
    s = _serie_sin_huecos(100)
    # Quitar 10 de las ultimas 50 -> 40 sesiones reales en las ultimas 50
    drop_idx = s.index[-50::5][:10]
    s = s.drop(drop_idx)
    out = slice_last_n_sessions(s, 20)
    assert len(out) == 20


def test_dataframe_serie_equivalente():
    s = _serie_sin_huecos(60)
    df = s.to_frame("col")
    o_s = slice_last_n_sessions(s, 10)
    o_df = slice_last_n_sessions(df, 10)
    assert len(o_s) == len(o_df) == 10
    assert (o_s.values == o_df["col"].values).all()


def test_vacio_devuelve_tal_cual():
    s_vacia = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    out = slice_last_n_sessions(s_vacia, 10)
    assert len(out) == 0


def test_n_mayor_que_len_no_rompe():
    """n mayor que la serie: devuelve todas las sesiones que haya.
    bdate_range no garantiza que todas sean sesiones NYSE (2026-01-01
    es festivo), asi que el limite superior es <= len(s).
    """
    s = _serie_sin_huecos(10)
    out = slice_last_n_sessions(s, 100)
    assert 1 <= len(out) <= len(s)
    # El valor mas reciente debe estar siempre
    assert out.iloc[-1] == s.iloc[-1]


def test_indice_no_datetime_fallback_iloc():
    """Indice entero (no Datetime): fallback directo a iloc[-n:],
    sin pasar por el calendario NYSE."""
    s = pd.Series([1, 2, 3, 4, 5], index=[0, 1, 2, 3, 4])
    out = slice_last_n_sessions(s, 3)
    assert len(out) == 3
    assert list(out.values) == [3, 4, 5]


def test_n_cero_devuelve_slice_vacio():
    s = _serie_sin_huecos(10)
    out = slice_last_n_sessions(s, 0)
    assert len(out) == 0


def test_idempotente_con_calendario_lleno():
    """Sin huecos, el helper devuelve lo mismo que iloc[-n:]."""
    idx = pd.DatetimeIndex(get_trading_days("2026-01-05", "2026-04-01"))
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    assert (
        slice_last_n_sessions(s, 20).values.tolist()
        == s.iloc[-20:].values.tolist()
    )
