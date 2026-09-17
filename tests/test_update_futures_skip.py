"""Tests K-FUTURES-REFRESH-01: skip logic por cobertura en update_futures.

El skip logic antiguo solo verificaba last_date, proxy de completitud.
Un parquet con last_date == expected puede tener tickers con Close=NaN
(fila parcial). FUTURE_SETTLEMENT exige cobertura completa. Estos tests
cubren _inspect_parquet, el nuevo helper que decide skip/fetch.

Sin red. Sin API. Parquets temporales sinteticos.
"""
from __future__ import annotations

import pandas as pd

from scripts.update_futures import _inspect_parquet


def _mk_parquet(path, dates, values_by_ticker):
    """Construye parquet wide con MultiIndex (Close, ticker).

    values_by_ticker: {ticker: [v1, v2, ...]} alineado con dates.
    None se escribe como NaN.
    """
    idx = pd.to_datetime(dates)
    cols = pd.MultiIndex.from_tuples(
        [("Close", t) for t in values_by_ticker.keys()],
        names=["field", "ticker"],
    )
    df = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for t, vals in values_by_ticker.items():
        df[("Close", t)] = vals
    df.to_parquet(path)
    return df


class TestInspectParquet:

    def test_last_distinto_expected_devuelve_fetch(self, tmp_path):
        p = tmp_path / "fut.parquet"
        _mk_parquet(
            p, ["2026-09-15"],
            {"BZ=F": [100.0], "CL=F": [90.0]},
        )
        expected = pd.Timestamp("2026-09-16").date()
        complete, missing = _inspect_parquet(
            str(p), expected, ["BZ=F", "CL=F"])
        assert complete is False
        assert set(missing) == {"BZ=F", "CL=F"}

    def test_last_ok_cobertura_parcial_devuelve_fetch(self, tmp_path):
        p = tmp_path / "fut.parquet"
        _mk_parquet(
            p, ["2026-09-16"],
            {"BZ=F": [None], "CL=F": [97.21]},
        )
        expected = pd.Timestamp("2026-09-16").date()
        complete, missing = _inspect_parquet(
            str(p), expected, ["BZ=F", "CL=F"])
        assert complete is False
        assert missing == ["BZ=F"]

    def test_last_ok_cobertura_completa_skip(self, tmp_path):
        p = tmp_path / "fut.parquet"
        _mk_parquet(
            p, ["2026-09-16"],
            {"BZ=F": [105.52], "CL=F": [97.21]},
        )
        expected = pd.Timestamp("2026-09-16").date()
        complete, missing = _inspect_parquet(
            str(p), expected, ["BZ=F", "CL=F"])
        assert complete is True
        assert missing == []

    def test_columna_ausente_devuelve_missing(self, tmp_path):
        p = tmp_path / "fut.parquet"
        # Solo BZ=F en el parquet; CL=F no existe como columna.
        _mk_parquet(
            p, ["2026-09-16"],
            {"BZ=F": [105.52]},
        )
        expected = pd.Timestamp("2026-09-16").date()
        complete, missing = _inspect_parquet(
            str(p), expected, ["BZ=F", "CL=F"])
        assert complete is False
        assert missing == ["CL=F"]

    def test_fichero_no_existe_devuelve_fetch(self, tmp_path):
        p = tmp_path / "no_existe.parquet"
        expected = pd.Timestamp("2026-09-16").date()
        complete, missing = _inspect_parquet(
            str(p), expected, ["BZ=F", "CL=F"])
        assert complete is False
        assert set(missing) == {"BZ=F", "CL=F"}

    def test_expected_none_devuelve_fetch(self, tmp_path):
        p = tmp_path / "fut.parquet"
        _mk_parquet(
            p, ["2026-09-16"],
            {"BZ=F": [105.52], "CL=F": [97.21]},
        )
        complete, missing = _inspect_parquet(
            str(p), None, ["BZ=F", "CL=F"])
        assert complete is False
        assert set(missing) == {"BZ=F", "CL=F"}


class TestFetchCommoditiesSelective:
    """K-FUTURES-REFRESH-01: fetch selectivo con skip_spot.

    Verifica que fetch_commodities ejecuta solo los fetches necesarios:
      - only_futures filtro de tickers.
      - skip_spot evita _fetch_spot cuando la cobertura spot esta OK.
    Sin red. Sin API. Se mockean _fetch_futures_front_month y _fetch_spot.
    """

    def _mk_provider(self, monkeypatch):
        from data.providers.futures import FuturesProvider
        p = FuturesProvider()
        calls = {"futures": [], "spot": []}

        def fake_futures(ticker):
            calls["futures"].append(ticker)
            return {
                "date": "2026-09-16", "ticker": ticker,
                "Open": 1.0, "High": 1.0, "Low": 1.0,
                "Close": 1.0, "Volume": 1.0,
            }

        def fake_spot():
            calls["spot"].append(True)
            return []

        monkeypatch.setattr(p, "_fetch_futures_front_month", fake_futures)
        monkeypatch.setattr(p, "_fetch_spot", fake_spot)
        return p, calls

    def test_skip_spot_true_no_llama_fetch_spot(self, monkeypatch):
        p, calls = self._mk_provider(monkeypatch)
        p.fetch_commodities(only_futures=["BZ=F"], skip_spot=True)
        assert calls["futures"] == ["BZ=F"]
        assert calls["spot"] == []

    def test_skip_spot_false_llama_fetch_spot(self, monkeypatch):
        p, calls = self._mk_provider(monkeypatch)
        p.fetch_commodities(only_futures=["BZ=F"], skip_spot=False)
        assert calls["futures"] == ["BZ=F"]
        assert calls["spot"] == [True]

    def test_only_futures_vacio_no_llama_futures(self, monkeypatch):
        p, calls = self._mk_provider(monkeypatch)
        p.fetch_commodities(only_futures=[], skip_spot=False)
        assert calls["futures"] == []
        assert calls["spot"] == [True]
