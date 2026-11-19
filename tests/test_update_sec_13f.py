"""Tests unitarios de scripts/update_sec_13f.py (F.3.c).

Solo funciones puras: sin red, sin I/O, sin SEC.

Cubre:
  - _is_leap, _last_day_of_month
  - _parse_quarter (validaciones)
  - quarter_to_iso_end (4 trimestres)
  - quarter_to_source_period (formato SEC, cruce de año, bisiesto)
  - latest_published_quarter (ancla ventana de envio + margen)
  - _pick_base_url (OLD vs NEW)
"""
from __future__ import annotations

from datetime import date

import pytest

from config import settings
from scripts.update_sec_13f import (
    _is_leap,
    _last_day_of_month,
    _parse_quarter,
    quarter_to_iso_end,
    quarter_to_source_period,
    quarter_to_window_end,
    latest_published_quarter,
)
from src.institutional_accumulation.sec_13f.downloader import (
    _pick_base_url,
    SEC_BASE_URL_OLD,
    SEC_BASE_URL_NEW,
)


# --- _is_leap -----------------------------------------------------------

@pytest.mark.parametrize("year,expected", [
    (2024, True), (2025, False), (2000, True), (1900, False), (2100, False),
])
def test_is_leap(year, expected):
    assert _is_leap(year) == expected


# --- _last_day_of_month -------------------------------------------------

@pytest.mark.parametrize("year,month,expected", [
    (2026, 1, 31), (2026, 2, 28), (2024, 2, 29), (2026, 3, 31),
    (2026, 4, 30), (2026, 6, 30), (2026, 9, 30), (2026, 12, 31),
])
def test_last_day_of_month(year, month, expected):
    assert _last_day_of_month(year, month) == expected


# --- _parse_quarter -----------------------------------------------------

@pytest.mark.parametrize("q,expected", [
    ("2026Q1", (2026, "Q1")), ("2025Q4", (2025, "Q4")), ("2030Q3", (2030, "Q3")),
])
def test_parse_quarter_valido(q, expected):
    assert _parse_quarter(q) == expected


@pytest.mark.parametrize("q", [
    "2026Q5", "2026-Q1", "26Q1", "", None, "2026X1", "2026Q", "QWERTY",
])
def test_parse_quarter_invalido(q):
    with pytest.raises((ValueError, TypeError)):
        _parse_quarter(q)

# --- quarter_to_iso_end -------------------------------------------------

@pytest.mark.parametrize("q,expected", [
    ("2025Q4", "2025-12-31"),
    ("2026Q1", "2026-03-31"),
    ("2026Q2", "2026-06-30"),
    ("2026Q3", "2026-09-30"),
    ("2026Q4", "2026-12-31"),
    ("2024Q2", "2024-06-30"),
])
def test_quarter_to_iso_end(q, expected):
    assert quarter_to_iso_end(q) == expected


# --- quarter_to_source_period -------------------------------------------

@pytest.mark.parametrize("q,expected", [
    ("2025Q4", "01dec2025-28feb2026"),
    ("2026Q1", "01mar2026-31may2026"),
    ("2026Q2", "01jun2026-31aug2026"),
    ("2026Q3", "01sep2026-30nov2026"),
    ("2026Q4", "01dec2026-28feb2027"),
    ("2027Q1", "01mar2027-31may2027"),
    # Bisiesto: 2024Q4 -> end_year 2025 (no leap) -> 28feb2025
    ("2024Q4", "01dec2024-28feb2025"),
    # Bisiesto: 2023Q4 -> end_year 2024 (leap) -> 29feb2024
    ("2023Q4", "01dec2023-29feb2024"),
])
def test_quarter_to_source_period(q, expected):
    assert quarter_to_source_period(q) == expected


# --- latest_published_quarter -------------------------------------------

@pytest.mark.parametrize("today,expected", [
    # Casos genericos: lejos de fronteras, mismo expected con el ancla
    # nueva (ventana + margen 7) que con la vieja (cierre + 50).
    (date(2026, 5, 1), "2025Q4"),   # Q1 ventana cierra 31-may; no elegible
    (date(2026, 6, 30), "2026Q1"),  # Q1 elegible 7-jun; Q2 ventana 31-ago
    (date(2026, 9, 23), "2026Q2"),  # Q2 elegible 7-sep; Q3 ventana 30-nov
    (date(2027, 2, 15), "2026Q3"),  # Q3 elegible 7-dic; Q4 ventana 28-feb
    (date(2024, 1, 15), "2023Q3"),  # Q3-23 elegible 7-dic-23; Q4-23 ventana 29-feb-24
    # Regresion run 44585740091 (19-nov) y cron viejo (20-nov): Q3
    # ventana cierra 30-nov -> dataset inexistente -> latest = Q2.
    (date(2026, 11, 19), "2026Q2"),
    (date(2026, 11, 20), "2026Q2"),
    # Fronteras viejas (cierre+50): caen ANTES del cierre de ventana.
    # Con el ancla correcta devuelven el quarter anterior (flip
    # semantico deliberado; con lag 50 devolvian el no publicado).
    (date(2026, 5, 25), "2025Q4"),  # ventana Q1 cierra 31-may
    (date(2026, 8, 25), "2026Q1"),  # ventana Q2 cierra 31-ago
    (date(2026, 11, 25), "2026Q2"), # ventana Q3 cierra 30-nov
    (date(2027, 2, 20), "2026Q3"),  # ventana Q4 cierra 28-feb-27
    # Fronteras nuevas: window_end + margen - 1 vs + margen (7d).
    (date(2026, 6, 6), "2025Q4"),   # Q1: 31-may + 6 = 6-jun (no elegible)
    (date(2026, 6, 7), "2026Q1"),   # Q1: 31-may + 7 = 7-jun (elegible)
    (date(2026, 9, 6), "2026Q1"),   # Q2: 31-ago + 6
    (date(2026, 9, 7), "2026Q2"),   # Q2: 31-ago + 7
    (date(2026, 12, 6), "2026Q2"),  # Q3: 30-nov + 6
    (date(2026, 12, 7), "2026Q3"),  # Q3: 30-nov + 7
    (date(2027, 3, 6), "2026Q3"),   # Q4: 28-feb + 6
    (date(2027, 3, 7), "2026Q4"),   # Q4: 28-feb + 7
    # Bisiesto: Q4-2023 ventana cierra 29-feb-2024.
    (date(2024, 3, 6), "2023Q3"),   # 29-feb + 6
    (date(2024, 3, 7), "2023Q4"),   # 29-feb + 7
])
def test_latest_published_quarter(today, expected):
    assert latest_published_quarter(today) == expected


# --- _pick_base_url -----------------------------------------------------

@pytest.mark.parametrize("period,expected", [
    ("01dec2025-28feb2026", SEC_BASE_URL_OLD),
    ("01mar2026-31may2026", SEC_BASE_URL_OLD),
    ("01jun2026-31aug2026", SEC_BASE_URL_NEW),
    ("01sep2026-30nov2026", SEC_BASE_URL_NEW),
    ("01dec2026-28feb2027", SEC_BASE_URL_NEW),
    ("01mar2027-31may2027", SEC_BASE_URL_NEW),
])
def test_pick_base_url_por_period(period, expected):
    assert _pick_base_url(period) == expected


def test_pick_base_url_override():
    assert _pick_base_url("01mar2026-31may2026",
                          override="http://custom") == "http://custom"


def test_pick_base_url_vacio_lanza():
    with pytest.raises(ValueError):
        _pick_base_url("")


# --- _prev_quarter / _quarters_range (backfill) ------------------------

from scripts.update_sec_13f import _prev_quarter, _quarters_range
from scripts.update_sec_13f import _write_latest_quarter_if_newer
from scripts.update_sec_13f import _is_quarter_fully_ingested


@pytest.mark.parametrize("q,expected", [
    ("2026Q2", "2026Q1"),
    ("2026Q1", "2025Q4"),
    ("2025Q1", "2024Q4"),
    ("2024Q4", "2024Q3"),
])
def test_prev_quarter(q, expected):
    assert _prev_quarter(q) == expected


def test_prev_quarter_invalido_lanza():
    with pytest.raises((ValueError, TypeError)):
        _prev_quarter("INVALID")


def test_quarters_range_cero():
    assert _quarters_range("2026Q2", 0) == ["2026Q2"]


def test_quarters_range_dos():
    assert _quarters_range("2026Q2", 2) == ["2025Q4", "2026Q1", "2026Q2"]


def test_quarters_range_cruce_ano():
    assert _quarters_range("2026Q1", 2) == ["2025Q3", "2025Q4", "2026Q1"]


def test_quarters_range_tres():
    assert _quarters_range("2026Q3", 3) == [
        "2025Q4", "2026Q1", "2026Q2", "2026Q3"
    ]


# --- _write_latest_quarter_if_newer (monotonia, fix 2026-09-29) --------

def test_latest_quarter_primer_valor(tmp_path):
    f = tmp_path / "latest.txt"
    written, prev = _write_latest_quarter_if_newer("2026Q2", f)
    assert written is True
    assert prev is None
    assert f.read_text(encoding="utf-8") == "2026Q2\n"


def test_latest_quarter_avanza(tmp_path):
    f = tmp_path / "latest.txt"
    f.write_text("2026Q1\n", encoding="utf-8")
    written, prev = _write_latest_quarter_if_newer("2026Q2", f)
    assert written is True
    assert prev == "2026Q1"
    assert f.read_text(encoding="utf-8") == "2026Q2\n"


def test_latest_quarter_no_retrocede(tmp_path):
    """Bug fix 2026-09-29: --backfill no debe retroceder latest_quarter."""
    f = tmp_path / "latest.txt"
    f.write_text("2026Q2\n", encoding="utf-8")
    written, prev = _write_latest_quarter_if_newer("2026Q1", f)
    assert written is False
    assert prev == "2026Q2"
    assert f.read_text(encoding="utf-8") == "2026Q2\n"


def test_latest_quarter_mismo_valor_no_reescribe(tmp_path):
    f = tmp_path / "latest.txt"
    f.write_text("2026Q2\n", encoding="utf-8")
    written, prev = _write_latest_quarter_if_newer("2026Q2", f)
    assert written is False
    assert prev == "2026Q2"


def test_latest_quarter_fichero_ilegible_reescribe(tmp_path):
    f = tmp_path / "latest.txt"
    f.write_bytes(b"\xff\xfe\x00")
    written, prev = _write_latest_quarter_if_newer("2026Q2", f)
    assert written is True
    assert prev is None
    assert f.read_text(encoding="utf-8") == "2026Q2\n"


def test_latest_quarter_crea_directorio(tmp_path):
    f = tmp_path / "sub" / "latest.txt"
    assert not f.parent.exists()
    written, prev = _write_latest_quarter_if_newer("2026Q2", f)
    assert written is True
    assert f.read_text(encoding="utf-8") == "2026Q2\n"

# --- _is_quarter_fully_ingested (fix 2026-09-29) ---------------------

def _mk_quarter(tmp_path, quarter, *, n_parquets=7, manifest=True,
                schema_valid=True, crc_valid=True, files_present=True):
    """Construye un data_dir minimo con processed/<q> y manifest."""
    import json
    data_dir = tmp_path / "sec_13f"
    proc = data_dir / "processed" / quarter
    proc.mkdir(parents=True)
    expected = ["COVERPAGE", "INFOTABLE", "OTHERMANAGER",
                "OTHERMANAGER2", "SIGNATURE", "SUBMISSION", "SUMMARYPAGE"]
    for name in expected[:n_parquets]:
        (proc / (name + ".parquet")).write_bytes(b"x")
    if manifest:
        mp = data_dir / "manifests"
        mp.mkdir(parents=True)
        m = {"quarter": quarter, "validation": {
            "expected_files_present": files_present,
            "zip_crc_valid": crc_valid,
            "schema_valid": schema_valid,
        }}
        (mp / ("sec_13f_" + quarter + ".json")).write_text(
            json.dumps(m), encoding="utf-8")
    return data_dir


def test_quarter_fully_ingested_ok(tmp_path):
    dd = _mk_quarter(tmp_path, "2026Q2")
    assert _is_quarter_fully_ingested("2026Q2", dd) is True


def test_quarter_no_ingested_sin_dir(tmp_path):
    dd = tmp_path / "sec_13f"
    dd.mkdir()
    assert _is_quarter_fully_ingested("2026Q2", dd) is False


def test_quarter_parcial_5_de_7(tmp_path):
    dd = _mk_quarter(tmp_path, "2026Q2", n_parquets=5)
    assert _is_quarter_fully_ingested("2026Q2", dd) is False


def test_quarter_sin_manifest(tmp_path):
    dd = _mk_quarter(tmp_path, "2026Q2", manifest=False)
    assert _is_quarter_fully_ingested("2026Q2", dd) is False


def test_quarter_manifest_invalido_schema(tmp_path):
    dd = _mk_quarter(tmp_path, "2026Q2", schema_valid=False)
    assert _is_quarter_fully_ingested("2026Q2", dd) is False


def test_quarter_manifest_invalido_crc(tmp_path):
    dd = _mk_quarter(tmp_path, "2026Q2", crc_valid=False)
    assert _is_quarter_fully_ingested("2026Q2", dd) is False


def test_quarter_manifest_invalido_files_present(tmp_path):
    dd = _mk_quarter(tmp_path, "2026Q2", files_present=False)
    assert _is_quarter_fully_ingested("2026Q2", dd) is False


def test_quarter_manifest_corrupto(tmp_path):
    dd = _mk_quarter(tmp_path, "2026Q2")
    mp = dd / "manifests" / "sec_13f_2026Q2.json"
    mp.write_text("{{not valid", encoding="utf-8")
    assert _is_quarter_fully_ingested("2026Q2", dd) is False


# --- quarter_to_window_end (ancla de publicacion SEC) ------------------

@pytest.mark.parametrize("q,expected", [
    ("2026Q1", date(2026, 5, 31)),
    ("2026Q2", date(2026, 8, 31)),
    ("2026Q3", date(2026, 11, 30)),
    ("2026Q4", date(2027, 2, 28)),
    ("2023Q4", date(2024, 2, 29)),  # bisiesto
])
def test_quarter_to_window_end(q, expected):
    assert quarter_to_window_end(q) == expected


def test_latest_published_quarter_none_si_margen_enorme(monkeypatch):
    """Margen absurdo -> ningun quarter elegible -> None (rama defensiva)."""
    monkeypatch.setattr(settings, "SEC_13F_PUBLICATION_MARGIN_DAYS", 10000)
    assert latest_published_quarter(date(2026, 11, 19)) is None


def test_constante_vieja_ausente():
    """Anti-zombie: el ancla vieja (cierre+lag) no debe reintroducirse."""
    assert not hasattr(settings, "SEC_13F_QUARTER_LAG_DAYS")


def test_cron_y_guarda_coherentes():
    """Centinela de hardcode con garantia: el cron mensual debe caer
    despues de window_end + margen para las 4 ventanas (incl. bisiesto),
    y el primer intento util debe quedar dentro del contrato SEC_13F
    (max_lag 90d desde cierre de trimestre)."""
    import re
    from pathlib import Path
    wf = (Path(__file__).resolve().parent.parent /
          ".github" / "workflows" / "update_sec_13f.yml")
    text = wf.read_text(encoding="utf-8-sig")
    crons = re.findall(r"cron:\s*'([^']+)'", text)
    assert crons == ["17 6 10 * *"], crons
    day = int(crons[0].split()[2])
    margin = settings.SEC_13F_PUBLICATION_MARGIN_DAYS
    for q in ("2026Q1", "2026Q2", "2026Q3", "2026Q4", "2023Q4"):
        we = quarter_to_window_end(q)
        ny, nm = ((we.year + 1, 1) if we.month == 12
                  else (we.year, we.month + 1))
        trigger = date(ny, nm, day)
        assert (trigger - we).days >= margin, (q, we, trigger)
        close = date.fromisoformat(quarter_to_iso_end(q))
        assert (trigger - close).days < 90, (q, trigger)
    assert latest_published_quarter(date(2026, 12, 10)) == "2026Q3"


# --- trace de ingesta fallida (addon H5.3) ------------------------------

def test_ingest_fallo_deja_trace_fail(tmp_path, monkeypatch):
    """Un intento fallido registra outcome=FAIL y propaga la excepcion."""
    import scripts.update_sec_13f as mod
    calls = []
    monkeypatch.setattr(
        mod, "_write_ingest_trace",
        lambda q, s, a, o, trace_path=None: calls.append((q, s, a, o)))
    def _boom(**kwargs):
        raise RuntimeError("404 simulado")
    monkeypatch.setattr(mod, "ingest_13f", _boom)
    monkeypatch.setattr(mod, "download_official_list",
                        lambda q, force=False: tmp_path / "ol.txt")
    monkeypatch.setattr(mod, "LATEST_FILE", tmp_path / "latest.txt")
    with pytest.raises(RuntimeError):
        mod.ensure_quarter("2026Q3", data_dir=tmp_path / "sec13f",
                           official_dir=tmp_path / "ol",
                           source="cron", actor="tester")
    assert calls == [("2026Q3", "cron", "tester", "FAIL")]


def test_ingest_exito_mantiene_trace_ingest(tmp_path, monkeypatch):
    """El camino de exito sigue registrando outcome=INGEST (sin regresion)."""
    import scripts.update_sec_13f as mod
    calls = []
    monkeypatch.setattr(
        mod, "_write_ingest_trace",
        lambda q, s, a, o, trace_path=None: calls.append((q, s, a, o)))
    monkeypatch.setattr(mod, "ingest_13f", lambda **kw: {})
    monkeypatch.setattr(mod, "download_official_list",
                        lambda q, force=False: tmp_path / "ol.txt")
    monkeypatch.setattr(mod, "LATEST_FILE", tmp_path / "latest.txt")
    mod.ensure_quarter("2026Q3", data_dir=tmp_path / "sec13f",
                       official_dir=tmp_path / "ol",
                       source="cron", actor="tester")
    assert calls == [("2026Q3", "cron", "tester", "INGEST")]
