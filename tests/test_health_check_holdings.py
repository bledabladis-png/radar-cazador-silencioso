"""check_holdings_csvs: contrato minimo de etf_holdings e index_holdings.

Fix D (2026-10-08): deuda reconocida en 58936778. Estos CSVs alimentan
el universo del radar y el IAE. Sin vigilancia, un fallo parcial de los
workflows trimestrales deja datos mixtos sin senal.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd

from scripts.health_check import check_holdings_csvs


SECTORIALES = ("XLB", "XLC", "XLE", "XLF", "XLI", "XLK",
               "XLP", "XLRE", "XLU", "XLV", "XLY")
INDICES = ("DAXEX", "DIA", "FEZ", "ISF.L", "IWM", "LYXI", "QQQ", "SPY")


def _rows_for(etfs, n):
    return [{"etf": e, "ticker": f"T{i}", "name": "X", "weight": 1.0}
            for e in etfs for i in range(n)]


def _make_csv(tmp_path, monkeypatch, fname, rows, age_days=0):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / fname
    pd.DataFrame(rows).to_csv(path, index=False)
    if age_days:
        old = (datetime.now() - timedelta(days=age_days)).timestamp()
        os.utime(path, (old, old))
    monkeypatch.setattr("scripts.health_check.PROJECT_ROOT", tmp_path)
    return path


def _setup_ok(tmp_path, monkeypatch):
    _make_csv(tmp_path, monkeypatch, "etf_holdings.csv",
              _rows_for(SECTORIALES, 20))
    _make_csv(tmp_path, monkeypatch, "index_holdings.csv",
              _rows_for(INDICES, 20))


def test_ficheros_ok(tmp_path, monkeypatch):
    _setup_ok(tmp_path, monkeypatch)
    res = check_holdings_csvs()
    assert len(res) == 2
    assert all(r.status == "OK" for r in res), res


def test_fichero_ausente_es_fail(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.health_check.PROJECT_ROOT", tmp_path)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    res = check_holdings_csvs()
    assert all(r.status == "FAIL" for r in res)
    assert all("no existe" in r.message for r in res)


def test_columna_etf_ausente_es_fail(tmp_path, monkeypatch):
    _make_csv(tmp_path, monkeypatch, "etf_holdings.csv",
              [{"ticker": "AAPL", "weight": 1.0}])
    _make_csv(tmp_path, monkeypatch, "index_holdings.csv",
              _rows_for(INDICES, 20))
    res = check_holdings_csvs()
    etf_res = [r for r in res if "etf_holdings" in r.name][0]
    assert etf_res.status == "FAIL"
    assert "etf" in etf_res.message


def test_etf_esperado_ausente_es_fail(tmp_path, monkeypatch):
    rows = [r for r in _rows_for(SECTORIALES, 20) if r["etf"] != "XLY"]
    _make_csv(tmp_path, monkeypatch, "etf_holdings.csv", rows)
    _make_csv(tmp_path, monkeypatch, "index_holdings.csv",
              _rows_for(INDICES, 20))
    res = check_holdings_csvs()
    etf_res = [r for r in res if "etf_holdings" in r.name][0]
    assert etf_res.status == "FAIL"
    assert "ausentes" in etf_res.message
    assert "XLY" in etf_res.message


def test_etf_con_pocos_tickers_es_fail(tmp_path, monkeypatch):
    # XLB con solo 5 tickers (min_tickers=15)
    rows = [r for r in _rows_for(SECTORIALES, 20) if r["etf"] != "XLB"]
    rows += _rows_for(("XLB",), 5)
    _make_csv(tmp_path, monkeypatch, "etf_holdings.csv", rows)
    _make_csv(tmp_path, monkeypatch, "index_holdings.csv",
              _rows_for(INDICES, 20))
    res = check_holdings_csvs()
    etf_res = [r for r in res if "etf_holdings" in r.name][0]
    assert etf_res.status == "FAIL"
    assert "tickers" in etf_res.message
    assert "XLB" in etf_res.message


def test_edad_excesiva_es_fail(tmp_path, monkeypatch):
    _make_csv(tmp_path, monkeypatch, "etf_holdings.csv",
              _rows_for(SECTORIALES, 20), age_days=200)
    _make_csv(tmp_path, monkeypatch, "index_holdings.csv",
              _rows_for(INDICES, 20))
    res = check_holdings_csvs()
    etf_res = [r for r in res if "etf_holdings" in r.name][0]
    assert etf_res.status == "FAIL"
    assert "edad" in etf_res.message


# ---------------------------------------------------------------------------
# F2-BR-HE-EURONEXT + sufijos (S36-bis fase 3): todo ticker debe tener
# mercado conocido en get_market. Un sufijo no mapeado cae a UNKNOWN ->
# _compute_by_market lo marca INVALID -> guard C-2 bloquea la exencion
# condicional. Tests sinteticos (tmp_path): cero violacion de 01_METODO 3.
# ---------------------------------------------------------------------------


def _rows_with_tickers(etfs, tickers):
    rows = []
    for e in etfs:
        for t in tickers:
            rows.append({"etf": e, "ticker": t, "name": "X", "weight": 1.0})
    return rows


def test_ticker_con_sufijo_desconocido_es_fail(tmp_path, monkeypatch):
    # 19 tickers validos + 1 con sufijo no mapeado (.TO Toronto).
    base = [f"T{i}" for i in range(19)] + ["RY.TO"]
    _make_csv(tmp_path, monkeypatch, "etf_holdings.csv",
              _rows_with_tickers(SECTORIALES, base))
    _make_csv(tmp_path, monkeypatch, "index_holdings.csv",
              _rows_for(INDICES, 20))
    res = check_holdings_csvs()
    etf_res = [r for r in res if "etf_holdings" in r.name][0]
    assert etf_res.status == "FAIL"
    assert "TO" in etf_res.message
    assert "sufijos no mapeados" in etf_res.message


def test_ticker_sufijos_europeos_validos_no_disparan(tmp_path, monkeypatch):
    # .BR/.HE anadidos en F2 (7a2b016a); .PA/.AS/.MI/.DE/.MC/.L ya existian.
    valid = ["ABI.BR", "NOKIA.HE", "AIR.PA", "ASML.AS", "UCG.MI",
             "SAP.DE", "SAN.MC", "HSBA.L"] + [f"T{i}" for i in range(12)]
    _make_csv(tmp_path, monkeypatch, "etf_holdings.csv",
              _rows_with_tickers(SECTORIALES, valid))
    _make_csv(tmp_path, monkeypatch, "index_holdings.csv",
              _rows_for(INDICES, 20))
    res = check_holdings_csvs()
    etf_res = [r for r in res if "etf_holdings" in r.name][0]
    assert etf_res.status == "OK", etf_res.message


def test_ticker_con_punto_clase_accionaria_no_dispara(tmp_path, monkeypatch):
    # BRK.B y BF.B se normalizan a BRK-B/BF-B en get_market (US_EQUITY).
    base = ["BRK.B", "BF.B"] + [f"T{i}" for i in range(18)]
    _make_csv(tmp_path, monkeypatch, "etf_holdings.csv",
              _rows_with_tickers(SECTORIALES, base))
    _make_csv(tmp_path, monkeypatch, "index_holdings.csv",
              _rows_for(INDICES, 20))
    res = check_holdings_csvs()
    etf_res = [r for r in res if "etf_holdings" in r.name][0]
    assert etf_res.status == "OK", etf_res.message
