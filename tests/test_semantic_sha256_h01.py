"""Tests del hash semantico (H-01 dictamen doc 76)."""
import hashlib
import pandas as pd
import numpy as np

from scripts.audit.holdout_manifest import _semantic_content_sha256


def _make_df(val=1.0):
    cols = pd.MultiIndex.from_tuples([
        ('Close', 'AAPL'), ('Close', 'MSFT'), ('Volume', 'AAPL'), ('Volume', 'MSFT'),
    ])
    idx = pd.DatetimeIndex(['2024-01-01', '2024-01-02', '2024-01-03'])
    return pd.DataFrame(val, index=idx, columns=cols)


def test_hash_deterministico():
    df = _make_df()
    assert _semantic_content_sha256(df) == _semantic_content_sha256(df.copy())


def test_hash_cambia_con_un_valor():
    a = _make_df(val=1.0)
    b = _make_df(val=1.0)
    b.iloc[0, 0] = 1.0001
    assert _semantic_content_sha256(a) != _semantic_content_sha256(b)


def test_hash_inmune_a_reordenacion_filas():
    a = _make_df()
    b = a.sort_index(ascending=False)
    assert _semantic_content_sha256(a) == _semantic_content_sha256(b)


def test_hash_inmune_a_reordenacion_columnas():
    a = _make_df()
    b = a.sort_index(axis=1, ascending=False)
    assert _semantic_content_sha256(a) == _semantic_content_sha256(b)


def test_hash_inmune_a_reserializacion_parquet(tmp_path):
    df = _make_df()
    p1 = tmp_path / 'a.parquet'
    p2 = tmp_path / 'b.parquet'
    df.to_parquet(p1, compression='snappy')
    df.to_parquet(p2, compression='gzip')
    assert hashlib.sha256(p1.read_bytes()).hexdigest() != hashlib.sha256(p2.read_bytes()).hexdigest()
    df1 = pd.read_parquet(p1)
    df2 = pd.read_parquet(p2)
    assert _semantic_content_sha256(df1) == _semantic_content_sha256(df2)


def test_hash_maneja_nan():
    df = _make_df()
    df.iloc[0, 0] = np.nan
    h = _semantic_content_sha256(df)
    assert h is not None
    assert len(h) == 64


def test_hash_vacio_devuelve_none():
    df = pd.DataFrame()
    assert _semantic_content_sha256(df) is None
    assert _semantic_content_sha256(None) is None
