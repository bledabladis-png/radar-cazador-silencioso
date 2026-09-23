# -*- coding: utf-8 -*-
"""Tests de sec_13f.downloader (FA-1.1). Sin red."""
import zipfile
from unittest.mock import patch, MagicMock

import pytest
import requests

from src.institutional_accumulation.sec_13f import downloader
from src.institutional_accumulation.sec_13f.schema import EXPECTED_FILES


def _make_zip(path, files_dict):
    """Crea un ZIP con contenido dict {nombre: bytes}."""
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files_dict.items():
            zf.writestr(name, content)

def test_build_url_valido():
    url = downloader._build_url("01mar2026-31may2026")
    assert url == (
        "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/"
        "01mar2026-31may2026_form13f.zip"
    )


def test_build_url_vacio_lanza():
    with pytest.raises(ValueError):
        downloader._build_url("")


def test_validate_user_agent_valido():
    downloader._validate_user_agent("Test <test@test.com>")  # no raise


def test_validate_user_agent_vacio_lanza():
    with pytest.raises(ValueError):
        downloader._validate_user_agent("")


def test_validate_user_agent_whitespace_lanza():
    with pytest.raises(ValueError):
        downloader._validate_user_agent("   ")


def test_sha256_conocido(tmp_path):
    p = tmp_path / "test.bin"
    p.write_bytes(b"hello")
    expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert downloader._sha256(p) == expected

def test_http_401_no_retry():
    # 401 Unauthorized: 4xx real, no mejora con retry.
    with patch("requests.get") as mock_get:
        resp = MagicMock()
        resp.status_code = 401
        resp.raise_for_status.side_effect = requests.HTTPError("401")
        mock_get.return_value = resp
        with pytest.raises(requests.HTTPError):
            downloader._http_get_with_retry("http://x", {})
        assert mock_get.call_count == 1


def test_http_404_reintenta_por_rate_limit():
    # 404: Akamai/SEC lo usa como senal de rate limiting transitorio.
    # Retry con backoff largo (5s, 15s, 45s) y falla tras agotar intentos.
    with patch("requests.get") as mock_get, patch("time.sleep"):
        resp = MagicMock()
        resp.status_code = 404
        resp.raise_for_status.side_effect = requests.HTTPError("404")
        mock_get.return_value = resp
        with pytest.raises(requests.HTTPError):
            downloader._http_get_with_retry("http://x", {}, max_retries=3)
        assert mock_get.call_count == 3


def test_http_408_reintenta():
    # 408 Request Timeout: transitorio, retry.
    with patch("requests.get") as mock_get, patch("time.sleep"):
        resp = MagicMock()
        resp.status_code = 408
        resp.raise_for_status.side_effect = requests.HTTPError("408")
        mock_get.return_value = resp
        with pytest.raises(requests.HTTPError):
            downloader._http_get_with_retry("http://x", {}, max_retries=3)
        assert mock_get.call_count == 3


def test_http_429_reintenta():
    # 429 Too Many Requests: rate limit explicito, retry.
    with patch("requests.get") as mock_get, patch("time.sleep"):
        resp = MagicMock()
        resp.status_code = 429
        resp.raise_for_status.side_effect = requests.HTTPError("429")
        mock_get.return_value = resp
        with pytest.raises(requests.HTTPError):
            downloader._http_get_with_retry("http://x", {}, max_retries=3)
        assert mock_get.call_count == 3


def test_http_5xx_retry_agotado():
    with patch("requests.get") as mock_get, patch("time.sleep"):
        resp = MagicMock()
        resp.status_code = 500
        resp.raise_for_status.side_effect = requests.HTTPError("500")
        mock_get.return_value = resp
        with pytest.raises(requests.HTTPError):
            downloader._http_get_with_retry("http://x", {}, max_retries=3)
        assert mock_get.call_count == 3


def test_http_5xx_recupera():
    with patch("requests.get") as mock_get, patch("time.sleep"):
        resp_fail = MagicMock()
        resp_fail.status_code = 500
        resp_ok = MagicMock()
        resp_ok.status_code = 200
        mock_get.side_effect = [resp_fail, resp_ok]
        result = downloader._http_get_with_retry("http://x", {}, max_retries=3)
        assert result is resp_ok
        assert mock_get.call_count == 2

def test_download_cache_hit(tmp_path):
    zip_path = tmp_path / "01mar2026-31may2026_form13f.zip"
    zip_path.write_bytes(b"cached")
    with patch("requests.get") as mock_get:
        result = downloader.download_13f_zip("01mar2026-31may2026", tmp_path)
        assert result == zip_path
        mock_get.assert_not_called()


def test_download_cache_hit_force_redescarga(tmp_path):
    zip_path = tmp_path / "01mar2026-31may2026_form13f.zip"
    zip_path.write_bytes(b"cached")
    with patch("requests.get") as mock_get:
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"Content-Length": "6"}
        resp.iter_content.return_value = [b"fresh!"]
        mock_get.return_value = resp
        result = downloader.download_13f_zip("01mar2026-31may2026", tmp_path, force=True)
        assert result == zip_path
        assert zip_path.read_bytes() == b"fresh!"
        assert mock_get.call_count == 1


def test_extract_ok(tmp_path):
    zip_path = tmp_path / "test.zip"
    content = {f: b"col1\tcol2\nv1\tv2" for f in EXPECTED_FILES}
    _make_zip(zip_path, content)
    result = downloader.extract_13f_zip(zip_path, tmp_path / "out")
    assert len(result) == 7
    for key in ["SUBMISSION", "COVERPAGE", "INFOTABLE"]:
        assert key in result
        assert result[key].exists()


def test_extract_falta_tsv(tmp_path):
    zip_path = tmp_path / "test.zip"
    content = {f: b"x" for f in EXPECTED_FILES if f != "INFOTABLE.tsv"}
    _make_zip(zip_path, content)
    with pytest.raises(ValueError, match="Faltan TSVs"):
        downloader.extract_13f_zip(zip_path, tmp_path / "out")


def test_extract_extras_ignorados(tmp_path):
    zip_path = tmp_path / "test.zip"
    content = {f: b"x" for f in EXPECTED_FILES}
    content["FORM13F_readme.htm"] = b"extra"
    content["FORM13F_metadata.json"] = b"{}"
    _make_zip(zip_path, content)
    result = downloader.extract_13f_zip(zip_path, tmp_path / "out")
    assert len(result) == 7
    assert "FORM13F_readme" not in result


def test_extract_zip_no_existe(tmp_path):
    with pytest.raises(FileNotFoundError):
        downloader.extract_13f_zip(tmp_path / "no_existe.zip", tmp_path / "out")