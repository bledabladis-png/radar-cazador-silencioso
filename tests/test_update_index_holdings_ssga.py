# -*- coding: utf-8 -*-
"""get_state_street_holdings: tickers invalidos no deben desalinear listas.

Bug (2026-10-01): el append de `names` y `weights` estaba fuera del
`if is_valid_holding_ticker`. Cuando SSGA incluye una fila con ticker
invalido (cash placeholder '-', codigo numerico, etc.), `tickers` se
queda corto y `names`/`weights` crecen. `pd.DataFrame({...})` revienta
con "All arrays must be of the same length".

Evidencia en produccion: run 36885052188 (workflow_dispatch del
2026-10-01) fallo con "2 ETFs fallidos (SPY, DIA)" por este motivo.
"""
from io import BytesIO

import openpyxl

from scripts import update_index_holdings as m


def _make_ssga_excel_bytes(rows):
    """Genera bytes de un .xlsx con cabecera Name/Ticker/Weight (%)/Sector/Asset Class."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Name", "Ticker", "Weight (%)", "Sector", "Asset Class"])
    for r in rows:
        ws.append(r)
    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


class _FakeResp:
    def __init__(self, content):
        self.status_code = 200
        self.content = content


def test_ssga_holdings_alinea_listas_con_tickers_invalidos(monkeypatch):
    """Con 2 tickers validos + 2 invalidos: el df debe tener 2 filas."""
    excel = _make_ssga_excel_bytes([
        ["Apple Inc",    "AAPL", 6.5, "Tech", "Equity"],
        ["Microsoft",    "MSFT", 5.1, "Tech", "Equity"],
        ["Cash USD",     "-",    0.1, "",     "Cash"],
        ["Placeholder",  "123",  0.0, "",     "Equity"],
    ])

    def _fake_get(url, **kwargs):
        return _FakeResp(excel)

    monkeypatch.setattr(m.requests, "get", _fake_get)

    df = m.get_state_street_holdings("SPY", "http://fake")
    assert len(df) == 2
    assert list(df["ticker"]) == ["AAPL", "MSFT"]
    assert list(df["etf"]) == ["SPY", "SPY"]


def test_ssga_holdings_sin_invalidos_longitudes_coinciden(monkeypatch):
    """Control: sin invalidos, todas las listas tienen la misma longitud."""
    excel = _make_ssga_excel_bytes([
        ["Apple Inc", "AAPL", 6.5, "Tech", "Equity"],
        ["Microsoft", "MSFT", 5.1, "Tech", "Equity"],
        ["Nvidia",    "NVDA", 4.2, "Tech", "Equity"],
    ])

    def _fake_get(url, **kwargs):
        return _FakeResp(excel)

    monkeypatch.setattr(m.requests, "get", _fake_get)

    df = m.get_state_street_holdings("SPY", "http://fake")
    assert len(df) == 3
    assert list(df["ticker"]) == ["AAPL", "MSFT", "NVDA"]
