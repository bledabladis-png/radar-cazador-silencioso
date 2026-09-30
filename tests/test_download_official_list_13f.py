"""Tests de scripts/download_official_list_13f.py (sin red)."""
from __future__ import annotations

import sys
import urllib.error
import urllib.request as _ur

import pandas as pd
import pytest

from scripts import download_official_list_13f as dol


# --- _parse_quarter ---

@pytest.mark.parametrize("q,year,qn", [
    ("2025Q1", "2025", "1"),
    ("2025Q2", "2025", "2"),
    ("2025Q3", "2025", "3"),
    ("2025Q4", "2025", "4"),
    ("2026Q1", "2026", "1"),
    ("2026Q4", "2026", "4"),
])
def test_parse_quarter_valid(q, year, qn):
    assert dol._parse_quarter(q) == (year, qn)


@pytest.mark.parametrize("q", [
    None, 123, "2026", "2026Q", "2026Q5", "2026Q0",
    "26Q1", "2026-Q1", "2026q1", "XXXXQ1",
])
def test_parse_quarter_invalid(q):
    with pytest.raises(ValueError):
        dol._parse_quarter(q)


# --- _build_url ---

@pytest.mark.parametrize("q,suffix", [
    ("2025Q1", "13flist2025q1.txt"),
    ("2026Q1", "13flist2026q1.txt"),
    ("2026Q3", "13flist2026q3.txt"),
])
def test_build_url_canonica(q, suffix):
    assert dol._build_url(q) == dol.SEC_BASE + "/" + suffix


@pytest.mark.parametrize("q", ["2025Q3", "2026Q2"])
def test_build_url_excepcion(q):
    assert dol._build_url(q) == dol.SEC_BASE + "/" + dol.URL_EXCEPTIONS[q]

# --- _dest_path ---

def test_dest_path():
    p = dol._dest_path("2026Q1")
    assert p.name == "13flist_2026Q1.txt"
    assert p.parent == dol.DEST_DIR


# --- _validate_content ---

def test_validate_content_ok(monkeypatch):
    fake = pd.DataFrame({"x": range(dol.MIN_ROWS + 100)})
    monkeypatch.setattr(dol, "parse_official_list_text", lambda t: fake)
    assert dol._validate_content("irrelevante") == len(fake)


def test_validate_content_sospechosamente_corto(monkeypatch):
    fake = pd.DataFrame({"x": range(dol.MIN_ROWS - 1)})
    monkeypatch.setattr(dol, "parse_official_list_text", lambda t: fake)
    with pytest.raises(ValueError, match="sospechosamente corto"):
        dol._validate_content("irrelevante")


# --- download_official_list ---

def _mock_parser(monkeypatch, n=None):
    """Mockea el parser para devolver n filas (default: MIN_ROWS+1)."""
    n = n if n is not None else dol.MIN_ROWS + 1
    fake = pd.DataFrame({"x": range(n)})
    monkeypatch.setattr(dol, "parse_official_list_text", lambda t: fake)


def test_download_cache_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(dol, "DEST_DIR", tmp_path)
    dest = tmp_path / "13flist_2026Q1.txt"
    dest.write_text("cached", encoding="utf-8")
    # Sin mock de _http_get: si se llamara, fallaria por red.
    result = dol.download_official_list("2026Q1")
    assert result == dest
    assert dest.read_text(encoding="utf-8") == "cached"

def test_download_force_ignora_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(dol, "DEST_DIR", tmp_path)
    dest = tmp_path / "13flist_2026Q1.txt"
    dest.write_text("viejo", encoding="utf-8")
    cuerpo = b"cuerpo nuevo"
    monkeypatch.setattr(dol, "_http_get", lambda url: cuerpo)
    _mock_parser(monkeypatch)
    result = dol.download_official_list("2026Q1", force=True)
    assert result == dest
    assert dest.read_text(encoding="utf-8") == cuerpo.decode("utf-8")


def test_download_404_cae_a_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(dol, "DEST_DIR", tmp_path)
    calls = []
    def fake_get(url):
        calls.append(url)
        if url.endswith("13flist2026q1.txt"):
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        return b"data"
    monkeypatch.setattr(dol, "_http_get", fake_get)
    _mock_parser(monkeypatch)
    dol.download_official_list("2026Q1")
    assert len(calls) == 2
    assert calls[0].endswith("13flist2026q1.txt")
    assert calls[1].endswith("13flist2026q1-txt.txt")

def test_download_404_en_excepcion_propaga(tmp_path, monkeypatch):
    """2025Q3 ya esta en URL_EXCEPTIONS: sin segundo intento."""
    monkeypatch.setattr(dol, "DEST_DIR", tmp_path)
    calls = []
    def fake_get(url):
        calls.append(url)
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
    monkeypatch.setattr(dol, "_http_get", fake_get)
    with pytest.raises(urllib.error.HTTPError):
        dol.download_official_list("2025Q3")
    assert len(calls) == 1


def test_download_500_propaga(tmp_path, monkeypatch):
    monkeypatch.setattr(dol, "DEST_DIR", tmp_path)
    calls = []
    def fake_get(url):
        calls.append(url)
        raise urllib.error.HTTPError(url, 500, "Server Error", {}, None)
    monkeypatch.setattr(dol, "_http_get", fake_get)
    with pytest.raises(urllib.error.HTTPError):
        dol.download_official_list("2026Q1")
    assert len(calls) == 1


# --- _http_get (mock urlopen) ---

def test_http_get_user_agent_y_timeout(monkeypatch):
    captured = {}
    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"body"
    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["ua"] = req.get_header("User-agent")
        captured["timeout"] = timeout
        return FakeResp()
    monkeypatch.setattr(_ur, "urlopen", fake_urlopen)
    body = dol._http_get("http://example.com/x")
    assert body == b"body"
    assert captured["ua"] == dol.USER_AGENT
    assert captured["timeout"] == 60

# --- main ---

def test_main_exito(tmp_path, monkeypatch):
    monkeypatch.setattr(dol, "DEST_DIR", tmp_path)
    monkeypatch.setattr(dol, "_http_get", lambda url: b"data")
    _mock_parser(monkeypatch)
    monkeypatch.setattr(sys, "argv",
                        ["download_official_list_13f.py", "--quarter", "2026Q1"])
    assert dol.main() == 0


def test_main_http_error_devuelve_1(tmp_path, monkeypatch):
    monkeypatch.setattr(dol, "DEST_DIR", tmp_path)
    def fake_get(url):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
    monkeypatch.setattr(dol, "_http_get", fake_get)
    monkeypatch.setattr(sys, "argv",
                        ["download_official_list_13f.py", "--quarter", "2025Q3"])
    assert dol.main() == 1


def test_main_url_error_devuelve_1(tmp_path, monkeypatch):
    monkeypatch.setattr(dol, "DEST_DIR", tmp_path)
    def fake_get(url):
        raise urllib.error.URLError("dns failure")
    monkeypatch.setattr(dol, "_http_get", fake_get)
    monkeypatch.setattr(sys, "argv",
                        ["download_official_list_13f.py", "--quarter", "2026Q1"])
    assert dol.main() == 1


def test_main_excepcion_generica_devuelve_1(tmp_path, monkeypatch):
    monkeypatch.setattr(dol, "DEST_DIR", tmp_path)
    def fake_get(url):
        raise RuntimeError("boom")
    monkeypatch.setattr(dol, "_http_get", fake_get)
    monkeypatch.setattr(sys, "argv",
                        ["download_official_list_13f.py", "--quarter", "2026Q1"])
    assert dol.main() == 1