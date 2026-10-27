# -*- coding: utf-8 -*-
"""Tests del snapshot append-only de holdings (PIT-fix)."""
import pandas as pd

from src.pipeline.holdings_snapshot import (
    _ticker_hash, _build_snapshot_df,
    compute_and_persist_holdings_snapshot, SNAPSHOT_COLUMNS,
)


def _etf_df():
    return pd.DataFrame({
        "etf": ["XLK", "XLK", "XLF"],
        "ticker": ["AAPL", "MSFT", "JPM"],
        "identifier": ["037833100", "594918104", "46647P209"],
        "weight": [15.0, 13.0, 10.0],
    })


def _idx_df():
    return pd.DataFrame({
        "etf": ["SPY", "SPY"],
        "ticker": ["AAPL", "MSFT"],
        "name": ["APPLE INC", "MICROSOFT CORP"],
        "weight": [7.0, 6.0],
    })


def test_ticker_hash_determinista():
    h1 = _ticker_hash(["AAPL", "MSFT"])
    h2 = _ticker_hash(["MSFT", "AAPL"])
    assert h1 == h2
    assert len(h1) == 16


def test_ticker_hash_cambia_con_composicion():
    h1 = _ticker_hash(["AAPL", "MSFT"])
    h2 = _ticker_hash(["AAPL", "MSFT", "NVDA"])
    assert h1 != h2


def test_build_snapshot_schema():
    df = _build_snapshot_df("2026-10-23", _etf_df(), _idx_df())
    assert list(df.columns) == SNAPSHOT_COLUMNS
    assert len(df) == 5  # 3 etf + 2 index
    assert (df["date"] == "2026-10-23").all()
    # ETF y INDEX separados
    assert set(df["source_file"]) == {"etf_holdings", "index_holdings"}
    # Hash distinto por (source_file, etf)
    assert df[df["etf"] == "XLK"]["ticker_hash"].nunique() == 1
    assert df[df["etf"] == "SPY"]["ticker_hash"].nunique() == 1


def test_build_snapshot_sin_identifier_en_indices():
    df = _build_snapshot_df("2026-10-23", _etf_df(), _idx_df())
    spy_rows = df[df["etf"] == "SPY"]
    # identifier vacio para indices (no existe en el CSV de origen)
    assert (spy_rows["identifier"] == "").all()
    # name rellenado
    assert "APPLE INC" in spy_rows["name"].values


def test_persist_primera_vez(tmp_path):
    path = tmp_path / "holdings_snapshots.csv"
    out = compute_and_persist_holdings_snapshot(
        reference_date=pd.Timestamp("2026-10-23"),
        etf_holdings_df=_etf_df(),
        index_holdings_df=_idx_df(),
        path_override=str(path),
    )
    assert out is not None
    assert path.exists()
    df = pd.read_csv(path)
    assert len(df) == 5
    assert list(df.columns) == SNAPSHOT_COLUMNS


def test_persist_segunda_vez_misma_composicion_skip(tmp_path):
    path = tmp_path / "holdings_snapshots.csv"
    compute_and_persist_holdings_snapshot(
        reference_date=pd.Timestamp("2026-10-23"),
        etf_holdings_df=_etf_df(),
        index_holdings_df=_idx_df(),
        path_override=str(path),
    )
    size1 = path.stat().st_size
    # Misma composicion al dia siguiente -> skip
    out = compute_and_persist_holdings_snapshot(
        reference_date=pd.Timestamp("2026-10-24"),
        etf_holdings_df=_etf_df(),
        index_holdings_df=_idx_df(),
        path_override=str(path),
    )
    assert out is None
    size2 = path.stat().st_size
    assert size1 == size2


def test_persist_cambio_composicion_anade(tmp_path):
    """Si XLK cambia composicion, se anade con la nueva fecha.

    Estado inicial (2026-10-23): XLK=2 tickers, XLF=1, SPY=2. Total 5 filas.
    Cambio en XLK (MSFT -> NVDA) con fecha 2026-10-24:
    - XLK hash cambia -> 2 filas nuevas (2 tickers del nuevo XLK).
    - XLF, SPY hash igual -> skip.
    Total: 5 + 2 = 7 filas. XLK tiene 4 filas (2 tickers x 2 fechas).
    """
    path = tmp_path / "holdings_snapshots.csv"
    out1 = compute_and_persist_holdings_snapshot(
        reference_date=pd.Timestamp("2026-10-23"),
        etf_holdings_df=_etf_df(),
        index_holdings_df=_idx_df(),
        path_override=str(path),
    )
    assert out1 is not None
    assert len(out1) == 5

    etf_mod = _etf_df().copy()
    etf_mod.loc[1, "ticker"] = "NVDA"
    out2 = compute_and_persist_holdings_snapshot(
        reference_date=pd.Timestamp("2026-10-24"),
        etf_holdings_df=etf_mod,
        index_holdings_df=_idx_df(),
        path_override=str(path),
    )
    assert out2 is not None
    assert len(out2) == 2  # solo XLK, 2 tickers

    df = pd.read_csv(path)
    assert len(df) == 7
    xlk = df[df["etf"] == "XLK"]
    assert len(xlk) == 4
    assert xlk["date"].nunique() == 2


def test_inputs_vacios_skip(tmp_path):
    """DataFrames vacios (no None) -> skip. None = leer de disco."""
    path = tmp_path / "holdings_snapshots.csv"
    empty = pd.DataFrame()
    out = compute_and_persist_holdings_snapshot(
        reference_date=pd.Timestamp("2026-10-23"),
        etf_holdings_df=empty,
        index_holdings_df=empty,
        path_override=str(path),
    )
    assert out is None
    assert not path.exists()


def test_snapshot_no_reordena_historico(tmp_path):
    """Contrato PIT append-only: una segunda escritura anade filas al
    final sin reordenar el historico previo (fix 2026-10-27).

    Antes del fix, sort_by=['source_file','etf','ticker'] reescribia
    todo el CSV en orden alfabetico en cada llamada, generando un
    diff git de 3000+3000 lineas cosmeticas aunque el contenido
    fuera identico."""
    from src.pipeline.holdings_snapshot import compute_and_persist_holdings_snapshot
    import pandas as pd

    path = tmp_path / "holdings_snapshots.csv"

    etf1 = pd.DataFrame({
        "etf": ["XLK", "XLK", "XLK"],
        "ticker": ["NVDA", "AAPL", "MSFT"],
        "name": ["NVIDIA", "APPLE", "MICROSOFT"],
        "weight": [15.0, 13.0, 10.0],
    })
    idx1 = pd.DataFrame({"etf": [], "ticker": [], "name": [], "weight": []})

    compute_and_persist_holdings_snapshot(
        reference_date="2026-10-23",
        etf_holdings_df=etf1,
        index_holdings_df=idx1,
        path_override=str(path),
    )
    df1 = pd.read_csv(path)
    orden_original = df1["ticker"].tolist()
    fecha_resuelta_1 = df1["date"].iloc[0]  # pd.Timestamp naive -> sesion anterior por lag

    # Segunda escritura con composicion distinta (nuevo ticker).
    etf2 = pd.DataFrame({
        "etf": ["XLK", "XLK", "XLK", "XLK"],
        "ticker": ["NVDA", "AAPL", "MSFT", "AMD"],
        "name": ["NVIDIA", "APPLE", "MICROSOFT", "ADVANCED MICRO"],
        "weight": [15.0, 13.0, 10.0, 5.0],
    })
    # datetime(2026,10,27,14:00) resuelve a sesion 2026-10-26.
    # reference_date="2026-10-26" a las 00:00 Madrid resuelve a
    # 2026-10-23 por el lag de publicacion (PUBLISH_HOUR).
    from datetime import datetime
    compute_and_persist_holdings_snapshot(
        reference_date=datetime(2026, 10, 27, 14, 0),
        etf_holdings_df=etf2,
        index_holdings_df=idx1,
        path_override=str(path),
    )
    df2 = pd.read_csv(path)

    # El bloque de la primera sesion debe conservar el orden original.
    bloque_previo = df2[df2["date"] == fecha_resuelta_1]["ticker"].tolist()
    assert bloque_previo == orden_original, (
        f"historico reordenado: {bloque_previo} != {orden_original}"
    )
    # Las filas nuevas van al final.
    assert df2.iloc[-1]["date"] == "2026-10-26"