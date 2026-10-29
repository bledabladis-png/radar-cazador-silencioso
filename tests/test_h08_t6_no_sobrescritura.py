"""Test T6 (H-08 dictamen doc 76): generadores no sobrescriben sellado.

Verifica el contrato del helper safe_write_parquet que usan
fetch_reserved_cohort.py y fetch_reserved_cohort_r3000.py.

Regla: una ejecucion normal (sin --promote) nunca reemplaza el sellado.
Solo con --promote (y --allow-hash-change si el contenido difiere) se
sustituye.
"""
import hashlib
import pandas as pd

from scripts._holdout_write import safe_write_parquet


def _make_df(val=1.0):
    cols = pd.MultiIndex.from_tuples([
        ('Close', 'AAPL'), ('Close', 'MSFT'), ('Volume', 'AAPL'),
    ])
    idx = pd.DatetimeIndex(['2024-01-01', '2024-01-02'])
    return pd.DataFrame(val, index=idx, columns=cols)


def _file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_primer_sellado_se_escribe(tmp_path):
    out = tmp_path / 'a.parquet'
    r = safe_write_parquet(_make_df(), out)
    assert r['action'] == 'written'
    assert out.exists()


def test_sin_promote_no_sobrescribe(tmp_path):
    """T6: ejecucion normal (sin --promote) NO reemplaza el sellado."""
    out = tmp_path / 'a.parquet'
    safe_write_parquet(_make_df(1.0), out)
    h_antes = _file_hash(out)
    mtime_antes = out.stat().st_mtime

    # Contenido DIFERENTE, sin --promote
    r = safe_write_parquet(_make_df(2.0), out)
    assert r['action'] == 'staged'
    assert r['semantic_match'] is False
    assert _file_hash(out) == h_antes, 'sellado fue sobrescrito sin --promote'
    assert out.stat().st_mtime == mtime_antes
    # Staging existe
    staging = tmp_path / 'a.parquet.staged'
    assert staging.exists()


def test_sin_promote_contenido_identico_no_sobrescribe(tmp_path):
    out = tmp_path / 'a.parquet'
    safe_write_parquet(_make_df(1.0), out)
    h_antes = _file_hash(out)

    # Mismo contenido, sin --promote
    r = safe_write_parquet(_make_df(1.0), out)
    assert r['action'] == 'staged'
    assert r['semantic_match'] is True
    assert _file_hash(out) == h_antes


def test_promote_contenido_distinto_requiere_allow_hash_change(tmp_path):
    out = tmp_path / 'a.parquet'
    safe_write_parquet(_make_df(1.0), out)
    h_antes = _file_hash(out)

    # --promote sin --allow-hash-change y contenido distinto -> refused
    r = safe_write_parquet(_make_df(2.0), out, promote=True)
    assert r['action'] == 'refused'
    assert r['semantic_match'] is False
    assert _file_hash(out) == h_antes


def test_promote_con_allow_hash_change_sustituye(tmp_path):
    out = tmp_path / 'a.parquet'
    safe_write_parquet(_make_df(1.0), out)
    h_antes = _file_hash(out)

    r = safe_write_parquet(_make_df(2.0), out, promote=True,
                           allow_hash_change=True)
    assert r['action'] == 'written'
    assert r['semantic_match'] is False
    assert _file_hash(out) != h_antes
    # Contenido es el nuevo
    df_leido = pd.read_parquet(out)
    assert df_leido.iloc[0, 0] == 2.0


def test_promote_contenido_identico_no_requiere_allow(tmp_path):
    out = tmp_path / 'a.parquet'
    safe_write_parquet(_make_df(1.0), out)
    r = safe_write_parquet(_make_df(1.0), out, promote=True)
    assert r['action'] == 'written'
    assert r['semantic_match'] is True


def test_sellado_ilegible_se_refusa(tmp_path):
    out = tmp_path / 'a.parquet'
    out.write_bytes(b'not a valid parquet')
    r = safe_write_parquet(_make_df(), out)
    assert r['action'] == 'refused'
    assert 'no se puede leer' in r['message']
