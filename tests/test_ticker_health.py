# -*- coding: utf-8 -*-
"""Tests del registro canonico de estado por ticker.

Cubre:
- Carga del yaml, validacion de estructura.
- Estados validos e invalidos.
- API publica: get_estado, is_eligible, filter_eligible, exclusion_reason.
- Degradacion limpia: yaml ausente -> todos ACTIVE.
"""
import pytest
import yaml

from src import ticker_health as th


@pytest.fixture(autouse=True)
def _limpiar_cache():
    th._reset_cache()
    yield
    th._reset_cache()


def test_yaml_existe_y_valido():
    p = th.DEFAULT_PATH
    assert p.exists(), f"no existe {p}"
    data = yaml.safe_load(p.read_text(encoding="utf-8-sig"))
    assert "tickers" in data
    assert isinstance(data["tickers"], dict)


def test_wbd_es_delisted():
    assert th.get_estado("WBD") == "DELISTED"


def test_wbd_tiene_evidencia():
    entry = th._load_registry()["WBD"]
    assert "evidencia" in entry and len(entry["evidencia"]) > 20
    assert "reabrir_si" in entry
    assert "2026-10-06" in entry["evento"]


def test_ticker_no_registrado_es_active():
    assert th.get_estado("AAPL") == "ACTIVE"
    assert th.get_estado("MSFT") == "ACTIVE"


def test_ticker_case_insensitive():
    assert th.get_estado("wbd") == "DELISTED"
    assert th.get_estado("Wbd") == "DELISTED"


def test_ticker_vacio_o_none_es_active():
    assert th.get_estado("") == "ACTIVE"
    assert th.get_estado(None) == "ACTIVE"


def test_is_eligible_wbd_false():
    assert not th.is_eligible("WBD")
    assert th.is_eligible("AAPL")


def test_filter_eligible_quita_delisted():
    result = th.filter_eligible(["AAPL", "WBD", "MSFT"])
    assert result == ["AAPL", "MSFT"]


def test_filter_eligible_preserva_orden():
    result = th.filter_eligible(["MSFT", "AAPL", "GOOG"])
    assert result == ["MSFT", "AAPL", "GOOG"]


def test_filter_eligible_sin_duplicados():
    result = th.filter_eligible(["AAPL", "AAPL", "MSFT", "WBD", "WBD"])
    assert result == ["AAPL", "MSFT"]


def test_filter_eligible_vacio():
    assert th.filter_eligible([]) == []


def test_exclusion_reason_wbd_no_none():
    r = th.exclusion_reason("WBD")
    assert r is not None
    assert "Skydance" in r or "2026-10-06" in r


def test_exclusion_reason_aapl_none():
    assert th.exclusion_reason("AAPL") is None


def test_yaml_ausente_todos_active(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "DEFAULT_PATH", tmp_path / "no_existe.yaml")
    th._reset_cache()
    assert th.get_estado("WBD") == "ACTIVE"
    assert th.is_eligible("WBD")


def test_yaml_estado_invalido_falla(tmp_path, monkeypatch):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "schema_version: 1\ntickers:\n  XXX:\n    estado: NO_EXISTE\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(th, "DEFAULT_PATH", p)
    th._reset_cache()
    with pytest.raises(ValueError, match="estado 'NO_EXISTE' invalido"):
        th.get_estado("XXX")


def test_yaml_tickers_no_dict_falla(tmp_path, monkeypatch):
    p = tmp_path / "bad.yaml"
    p.write_text("schema_version: 1\ntickers: [a, b, c]\n", encoding="utf-8")
    monkeypatch.setattr(th, "DEFAULT_PATH", p)
    th._reset_cache()
    with pytest.raises(ValueError, match="debe ser dict"):
        th.get_estado("XXX")