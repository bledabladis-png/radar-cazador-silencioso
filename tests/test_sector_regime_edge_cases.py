"""DT1 Fase 3 (2026-09-17): edge cases de regimes.sector_regime.

Complementa tests/test_sector_regime_characterization.py (contrato observable).
Aqui: comportamiento defensivo y robustez.

NO depende de data/*.parquet.
"""
import numpy as np
import pandas as pd

from config.tickers import MARKET_TICKERS
from regimes.sector_regime import compute_sector_scores, compute_price_flow_rankings


def _make_df(tickers, n=300, seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    data = {}
    for k, t in enumerate(tickers):
        drift = 0.0001 * (k + 1)
        close = 100 * np.cumprod(1 + rng.normal(drift, 0.015, n))
        data[("Close", t)] = close
        data[("Open", t)] = close * 0.998
        data[("High", t)] = close * 1.01
        data[("Low", t)] = close * 0.99
        data[("Volume", t)] = rng.randint(1000000, 5000000, n)
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _all_tickers():
    return ["^GSPC"] + list(MARKET_TICKERS["sectors"])


# --- compute_sector_scores ---

def test_df_vacio_devuelve_none():
    df = pd.DataFrame(
        columns=pd.MultiIndex.from_tuples([("Close", "^GSPC")], names=["f", "t"])
    )
    assert compute_sector_scores(df) is None


def test_sin_benchmark_devuelve_none():
    sectors = MARKET_TICKERS["sectors"]
    df = _make_df(sectors)  # sin ^GSPC
    assert compute_sector_scores(df) is None


def test_sin_sectores_suficientes_devuelve_none():
    df = _make_df(["^GSPC"])  # solo benchmark
    res = compute_sector_scores(df)
    # Sin sectores no puede haber scores -> None o scores vacio con ranking vacio
    # Documentamos el comportamiento real observado.
    assert res is None or not res.get("ranking")


def test_sector_faltante_no_rompe():
    tickers = _all_tickers()
    df = _make_df(tickers)
    # Borrar 3 sectores
    drop = MARKET_TICKERS["sectors"][:3]
    df = df.drop(columns=[("Close", t) for t in drop])
    df = df.drop(columns=[("Open", t) for t in drop], errors="ignore")
    df = df.drop(columns=[("High", t) for t in drop], errors="ignore")
    df = df.drop(columns=[("Low", t) for t in drop], errors="ignore")
    df = df.drop(columns=[("Volume", t) for t in drop], errors="ignore")

    res = compute_sector_scores(df)
    assert res is not None
    got = {t for t, _, _, _ in res["ranking"]}
    assert not (got & set(drop)), f"sectores borrados aparecen: {got & set(drop)}"


def test_regime_es_uno_de_los_conocidos():
    df = _make_df(_all_tickers())
    res = compute_sector_scores(df)
    valid = {
        "BROAD PARTICIPATION",
        "NARROW RALLY",
        "ROTATIONAL",
        "CYCLICAL LEADERSHIP",
        "DEFENSIVE LEADERSHIP",
        "MIXED",
    }
    assert res["regime"] in valid


def test_ranking_ordenado_descendente():
    df = _make_df(_all_tickers())
    res = compute_sector_scores(df)
    scores = [s for _, _, s, _ in res["ranking"]]
    assert scores == sorted(scores, reverse=True)


def test_scores_series_no_vacia():
    df = _make_df(_all_tickers())
    res = compute_sector_scores(df)
    assert isinstance(res["scores"], pd.DataFrame)
    assert not res["scores"].empty


def test_top3_es_prefijo_de_ranking():
    df = _make_df(_all_tickers())
    res = compute_sector_scores(df)
    ranking_top3 = [t for t, _, _, _ in res["ranking"][:3]]
    top3_last_scores = [str(s) for s in res["last_scores"].head(3).index]
    assert ranking_top3 == top3_last_scores


# --- compute_price_flow_rankings ---

def test_price_flow_rankings_devuelve_4_listas():
    df = _make_df(_all_tickers())
    sp, sf, op, of = compute_price_flow_rankings(df)
    assert isinstance(sp, list)
    assert isinstance(sf, list)
    assert isinstance(op, list)
    assert isinstance(of, list)


def test_price_flow_rankings_sin_datos_no_rompe():
    df = pd.DataFrame(
        columns=pd.MultiIndex.from_tuples([("Close", "^GSPC")], names=["f", "t"])
    )
    sp, sf, op, of = compute_price_flow_rankings(df)
    assert sp == []
    assert sf == []
    assert op == []
    assert of == []


def test_price_flow_rankings_ordenados_descendente():
    df = _make_df(_all_tickers())
    sp, sf, op, of = compute_price_flow_rankings(df)
    for name, lst in (("sector_price", sp), ("sector_flow", sf),
                      ("otros_price", op), ("otros_flow", of)):
        vals = [v for _, v in lst]
        assert vals == sorted(vals, reverse=True), f"{name} no ordenado"