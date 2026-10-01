# -*- coding: utf-8 -*-
"""Fix N: fund flow debe intentar descarga siempre, cache como fallback.

Bug: los 5 writers (ssga, amundi, _blackrock_base, blackrock_iwm,
cftc) usan `mtime < 23h -> cache-hit`. Si la fuente publica despues
del ultimo run (SSGA publica el 29-sep despues del run del 30-sep
5:40), el siguiente run acepta el cache de 22h como "fresco" y no
refresca. Resultado: cache SSGA con 28-sep, fuente con 29-sep.

Coste real medido: 13s descargar todos los providers sin cache. El
run dura ~12min. El ahorro del cache (13s) no justifica perder dias
de dato.

Fix: invertir la logica. Intentar descarga primero. Cache solo si
la descarga falla.

Verificado 2026-10-01: SSGA FEZ 29-sep en fuente, cache 28-sep.
"""
import pandas as pd


# --- SSGA ---

def test_ssga_descarga_siempre_aunque_cache_fresca(tmp_path, monkeypatch):
    """Con cache fresca (< 23h), debe descargar igual."""
    from data.providers import ssga_fund_data as ssga

    # Cache fresca: escribir ahora un CSV
    cache_dir = tmp_path / "ssga_cache"
    cache_dir.mkdir()
    cache_file = cache_dir / "XLK.csv"
    pd.DataFrame({
        "Date": pd.date_range(end="2026-09-27", periods=3, freq="B"),
        "nav": [1.0, 1.0, 1.0],
        "shares_outstanding": [100.0, 100.0, 100.0],
        "total_net_assets": [100.0, 100.0, 100.0],
    }).to_csv(cache_file, index=False)

    monkeypatch.setattr(ssga, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(ssga, "SECTOR_TICKERS", ["XLK"])

    # Simular que download devuelve datos NUEVOS (28-sep en cache, 29 en fuente)
    download_calls = []
    def fake_download(ticker):
        download_calls.append(ticker)
        return pd.DataFrame({
            "Date": pd.date_range(end="2026-09-29", periods=3, freq="B"),
            "nav": [1.0, 1.0, 1.0],
            "shares_outstanding": [100.0, 100.0, 100.0],
            "total_net_assets": [100.0, 100.0, 100.0],
        })
    monkeypatch.setattr(ssga, "_download_single", fake_download)
    monkeypatch.setattr(ssga, "HISTORY_PATH", tmp_path / "out.csv")

    ssga.get_etf_primary_flow_data(force_download=False)

    # La descarga DEBE haber ocurrido
    assert download_calls == ["XLK"], (
        f"Esperado 1 descarga, ocurrido: {download_calls}. "
        f"El cache fresco bloqueó la descarga."
    )


def test_ssga_fallback_a_cache_si_descarga_falla(tmp_path, monkeypatch):
    """Si descarga falla, usar cache como fallback."""
    from data.providers import ssga_fund_data as ssga

    cache_dir = tmp_path / "ssga_cache"
    cache_dir.mkdir()
    cache_file = cache_dir / "XLK.csv"
    pd.DataFrame({
        "Date": pd.date_range(end="2026-09-27", periods=3, freq="B"),
        "nav": [1.0, 1.0, 1.0],
        "shares_outstanding": [100.0, 100.0, 100.0],
        "total_net_assets": [100.0, 100.0, 100.0],
    }).to_csv(cache_file, index=False)

    monkeypatch.setattr(ssga, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(ssga, "SECTOR_TICKERS", ["XLK"])
    monkeypatch.setattr(ssga, "HISTORY_PATH", tmp_path / "out.csv")

    def fake_download(ticker):
        raise ConnectionError("network down")
    monkeypatch.setattr(ssga, "_download_single", fake_download)

    # No debe lanzar; debe caer a cache
    result = ssga.get_etf_primary_flow_data(force_download=False)
    assert result is not None
    assert not result.empty



# --- Amundi ---

def test_amundi_descarga_siempre_aunque_cache_fresca(tmp_path, monkeypatch):
    """Amundi debe intentar descarga aunque el cache sea reciente."""
    from data.providers import amundi_fund_data as amundi

    cache_dir = tmp_path / "amundi_cache"
    cache_dir.mkdir()
    cache_file = cache_dir / "FR0010251744_hist_2018-01-01_2026-10-01.json"
    cache_file.write_text('{"historics": []}')

    monkeypatch.setattr(amundi, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(amundi, "ISIN_LYXI", "FR0010251744")

    calls = []
    class FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            calls.append("called")
            return {"products": [{"historics": []}]}

    monkeypatch.setattr(amundi.requests, "post", lambda *a, **k: FakeResp())

    amundi.download_historical_data("FR0010251744", "2018-01-01", "2026-10-01")
    assert calls == ["called"], (
        f"Esperado 1 descarga. Cache fresco bloqueo. calls={calls}"
    )


def test_amundi_fallback_a_cache_si_descarga_falla(tmp_path, monkeypatch):
    from data.providers import amundi_fund_data as amundi

    cache_dir = tmp_path / "amundi_cache"
    cache_dir.mkdir()
    cache_file = cache_dir / "FR0010251744_hist_2018-01-01_2026-10-01.json"
    cache_file.write_text('{"cached": true}')

    monkeypatch.setattr(amundi, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(amundi, "ISIN_LYXI", "FR0010251744")

    import requests as _r
    def fake_post(*a, **k):
        raise _r.RequestException("network down")
    monkeypatch.setattr(amundi.requests, "post", fake_post)

    result = amundi.download_historical_data("FR0010251744", "2018-01-01", "2026-10-01")
    assert result == {"cached": True}, f"Esperado fallback a cache. Got: {result}"


# --- CFTC ---

def test_cftc_descarga_siempre_aunque_cache_fresca(tmp_path, monkeypatch):
    from data.providers import cftc_data as cftc

    cache_path = tmp_path / "cftc_tff.csv"
    cache_path.write_text("col1,col2\n1,2\n")
    monkeypatch.setattr(cftc, "CACHE_PATH", cache_path)

    calls = []
    class FakeResp:
        status_code = 200
        text = ("Market_and_Exchange_Names,Report_Date_as_YYYY_MM_DD,"
                "CFTC_Contract_Market_Code,"
                "Dealer_Positions_Long_All,Dealer_Positions_Short_All\n"
                "TEST,2026-09-22,001,100,50\n")
        def raise_for_status(self): pass
    def fake_post(*a, **k):
        calls.append("called")
        return FakeResp()
    monkeypatch.setattr(cftc.requests, "post", fake_post)

    cftc._download_and_cache()
    assert calls == ["called"], f"Esperado descarga. calls={calls}"


# --- BlackRock base (DAX, ISF) ---

def test_blackrock_base_descarga_siempre(tmp_path, monkeypatch):
    from data.providers import _blackrock_base as bb

    cache_file = tmp_path / "test.xml"
    cache_file.write_bytes(b"<old>old cache</old>")

    calls = []
    class FakeResp:
        status_code = 200
        content = b"<ss:Workbook>" + b"x" * 200000
        headers = {"Content-Type": "application/vnd.ms-excel"}
        def raise_for_status(self): pass
    def fake_get(*a, **k):
        calls.append("called")
        return FakeResp()
    monkeypatch.setattr(bb.requests, "get", fake_get)

    bb.download_fund_file(
        url="http://example.com/fund.xls",
        cache_file=cache_file,
        referer="http://example.com",
        label="TEST",
    )
    assert calls == ["called"], f"Esperado descarga. calls={calls}"


def test_blackrock_base_fallback_a_cache_si_falla(tmp_path, monkeypatch):
    from data.providers import _blackrock_base as bb

    cache_file = tmp_path / "test.xml"
    cache_file.write_bytes(b"<old>valid cache</old>")

    import requests as _r
    def fake_get(*a, **k):
        raise _r.RequestException("network down")
    monkeypatch.setattr(bb.requests, "get", fake_get)

    outcome = bb.download_fund_file(
        url="http://example.com/fund.xls",
        cache_file=cache_file,
        referer="http://example.com",
        label="TEST",
    )
    assert outcome == bb.FundFileOutcome.STALE_CACHE


# --- IWM ---

def test_iwm_descarga_siempre(tmp_path, monkeypatch):
    from data.providers import blackrock_iwm_fund_data as iwm

    cache_file = tmp_path / "iwm.bin"
    cache_file.write_bytes(b"<old>cache</old>")
    monkeypatch.setattr(iwm, "RAW_CACHE_FILE", cache_file)
    monkeypatch.setattr(iwm, "CACHE_DIR", tmp_path)

    calls = []
    class FakeResp:
        status_code = 200
        content = b"<ss:Workbook>" + b"x" * 5000
        headers = {"Content-Type": "application/vnd.ms-excel"}
        def raise_for_status(self): pass
    def fake_get(*a, **k):
        calls.append("called")
        return FakeResp()
    monkeypatch.setattr(iwm.requests, "get", fake_get)

    iwm.download_fund_file(force_download=False)
    assert calls == ["called"], f"Esperado descarga. calls={calls}"


def test_iwm_fallback_a_cache_si_falla(tmp_path, monkeypatch):
    from data.providers import blackrock_iwm_fund_data as iwm

    cache_file = tmp_path / "iwm.bin"
    cache_file.write_bytes(b"<old>valid cache</old>")
    monkeypatch.setattr(iwm, "RAW_CACHE_FILE", cache_file)
    monkeypatch.setattr(iwm, "CACHE_DIR", tmp_path)

    import requests as _r
    def fake_get(*a, **k):
        raise _r.RequestException("network down")
    monkeypatch.setattr(iwm.requests, "get", fake_get)

    result = iwm.download_fund_file(force_download=False)
    assert result == b"<old>valid cache</old>"
