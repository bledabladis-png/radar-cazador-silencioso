# -*- coding: utf-8 -*-
"""Centinela: load_crosswalk_internal normaliza tickers de etf_holdings."""
import pandas as pd

from src.institutional_accumulation.sec_13f.identity.security_identity import (
    load_crosswalk_internal,
)


def _mk_csvs(tmp_path, exceptions_rows, etf_rows):
    exc = tmp_path / "exceptions.csv"
    etf = tmp_path / "etf.csv"
    pd.DataFrame(exceptions_rows, columns=["CUSIP", "ticker", "valid_from", "valid_to"]).to_csv(exc, index=False)
    pd.DataFrame(etf_rows, columns=["etf", "ticker", "identifier", "weight"]).to_csv(etf, index=False)
    return exc, etf


def test_etf_holdings_brk_b_normalizado(tmp_path):
    exc, etf = _mk_csvs(tmp_path, [], [["XLF", "BRK.B", "084670702", "12.3"]])
    df = load_crosswalk_internal(exceptions_path=exc, etf_holdings_path=etf)
    fila = df[df["CUSIP"] == "084670702"].iloc[0]
    got = fila["ticker"]
    assert got == "BRK-B", "esperado BRK-B, obtenido " + repr(got)
    assert fila["source"] == "etf_holdings"


def test_etf_holdings_bf_b_normalizado(tmp_path):
    exc, etf = _mk_csvs(tmp_path, [], [["XLP", "BF.B", "115637209", "0.3"]])
    df = load_crosswalk_internal(exceptions_path=exc, etf_holdings_path=etf)
    fila = df[df["CUSIP"] == "115637209"].iloc[0]
    assert fila["ticker"] == "BF-B"


def test_etf_holdings_aapl_idempotente(tmp_path):
    exc, etf = _mk_csvs(tmp_path, [], [["QQQ", "AAPL", "037833100", "8.0"]])
    df = load_crosswalk_internal(exceptions_path=exc, etf_holdings_path=etf)
    fila = df[df["CUSIP"] == "037833100"].iloc[0]
    assert fila["ticker"] == "AAPL"


def test_exceptions_tambien_normaliza(tmp_path):
    exc, etf = _mk_csvs(tmp_path, [["084670702", "BRK.B", None, None]], [])
    df = load_crosswalk_internal(exceptions_path=exc, etf_holdings_path=etf)
    fila = df[df["CUSIP"] == "084670702"].iloc[0]
    assert fila["ticker"] == "BRK-B"
    assert fila["source"] == "cusip_ticker_exceptions"
