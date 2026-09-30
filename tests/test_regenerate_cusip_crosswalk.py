"""Tests de scripts/regenerate_cusip_crosswalk.py (sin red)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts import regenerate_cusip_crosswalk as rcc


# --- helpers ---

def _mk_infotable(tmp_path: Path, quarter: str, rows: list) -> Path:
    """Crea processed/<q>/INFOTABLE.parquet con las 5 columnas necesarias.

    rows: lista de tuplas (CUSIP, FIGI, TITLEOFCLASS, SSHPRNAMTTYPE, PUTCALL).
          SSHPRNAMTTYPE default 'SH', PUTCALL default None.
    """
    d = tmp_path / quarter
    d.mkdir(parents=True, exist_ok=True)
    p = d / "INFOTABLE.parquet"
    norm = []
    for r in rows:
        if len(r) == 3:
            norm.append((r[0], r[1], r[2], "SH", None))
        elif len(r) == 4:
            norm.append((r[0], r[1], r[2], r[3], None))
        else:
            norm.append(r)
    pd.DataFrame(norm, columns=["CUSIP", "FIGI", "TITLEOFCLASS",
                                "SSHPRNAMTTYPE", "PUTCALL"]).to_parquet(p)
    return p


def _mk_catalog(path: Path, mapping: dict) -> None:
    """Escribe radar_target_catalog.csv con {scf: ticker}."""
    rows = [{"radar_ticker": tk, "share_class_figi": scf}
            for scf, tk in mapping.items()]
    pd.DataFrame(rows).to_csv(path, index=False)

# --- _period_end ---

@pytest.mark.parametrize("q,expected", [
    ("2025Q4", "2025-12-31"),
    ("2026Q1", "2026-03-31"),
    ("2026Q2", "2026-06-30"),
    ("2026Q3", "2026-09-30"),
])
def test_period_end(q, expected):
    assert rcc._period_end(q) == expected


# --- _discover_quarters ---

def test_discover_quarters(tmp_path):
    _mk_infotable(tmp_path, "2025Q4", [])
    _mk_infotable(tmp_path, "2026Q1", [])
    (tmp_path / "2026Q2").mkdir()
    (tmp_path / "2026Q2" / "INFOTABLE.parquet").write_bytes(b"")
    (tmp_path / "garbage").mkdir()
    out = rcc._discover_quarters(tmp_path)
    assert out == ["2025Q4", "2026Q1", "2026Q2"]


def test_discover_quarters_vacio(tmp_path):
    assert rcc._discover_quarters(tmp_path) == []
    assert rcc._discover_quarters(tmp_path / "nope") == []


# --- _load_radar_index ---

def test_load_radar_index(tmp_path):
    cat = tmp_path / "catalog.csv"
    _mk_catalog(cat, {"SC_AAPL": "AAPL", "SC_MSFT": "MSFT"})
    idx = rcc._load_radar_index(cat)
    assert idx == {"SC_AAPL": "AAPL", "SC_MSFT": "MSFT"}


# --- _extract_observations ---

def test_extract_filtra_por_radar(tmp_path):
    p = _mk_infotable(tmp_path, "2026Q1", [
        ("037833100", "SC_AAPL", "COM"),
        ("999999999", "SC_OTHER", "COM"),  # no en radar
        ("00287Y109", "SC_ABBV", "COM"),
    ])
    idx = {"SC_AAPL": "AAPL", "SC_ABBV": "ABBV"}
    obs = rcc._extract_observations("2026Q1", p, idx)
    assert set(obs.keys()) == {("037833100", "AAPL"), ("00287Y109", "ABBV")}
    assert obs[("037833100", "AAPL")]["title"] == "COM"


def test_extract_ignora_figi_nan(tmp_path):
    p = _mk_infotable(tmp_path, "2026Q1", [
        ("037833100", None, "COM"),
        ("00287Y109", "SC_ABBV", "COM"),
    ])
    idx = {"SC_AAPL": "AAPL", "SC_ABBV": "ABBV"}
    obs = rcc._extract_observations("2026Q1", p, idx)
    assert len(obs) == 1


# --- _build_auto_rows ---

def test_build_auto_rows_valid_to_null_si_ultimo():
    obs = {("037833100", "AAPL"): {"title": "COM", "quarters": ["2026Q2"]}}
    rows = rcc._build_auto_rows(obs, latest_quarter="2026Q2")
    assert rows[0]["valid_from"] == "2026-06-30"
    assert rows[0]["valid_to"] == ""
    assert rows[0]["verified_by"] == "auto"


def test_build_auto_rows_valid_to_si_no_ultimo():
    obs = {("037833100", "AAPL"): {"title": "COM", "quarters": ["2025Q4"]}}
    rows = rcc._build_auto_rows(obs, latest_quarter="2026Q2")
    assert rows[0]["valid_from"] == "2025-12-31"
    assert rows[0]["valid_to"] == "2025-12-31"


# --- _merge ---

def test_merge_manual_gana():
    manual = pd.DataFrame([{
        "CUSIP": "037833100", "ticker": "AAPL_MANUAL", "valid_from": "2020-01-01",
        "valid_to": "", "source": "SEC-EDGAR", "reason": "manual",
        "title_of_class": "COM", "verified_by": "manual",
    }])
    auto = [{
        "CUSIP": "037833100", "ticker": "AAPL", "valid_from": "2025-12-31",
        "valid_to": "", "source": "SEC-EDGAR", "reason": "auto",
        "title_of_class": "COM", "verified_by": "auto",
    }]
    out = rcc._merge(manual, auto)
    assert len(out) == 1
    assert out.iloc[0]["ticker"] == "AAPL_MANUAL"


# --- _validate ---

def test_validate_ok():
    df = pd.DataFrame([{
        "CUSIP": "037833100", "ticker": "AAPL", "valid_from": "2025-12-31",
        "valid_to": "2026-03-31", "source": "SEC-EDGAR", "reason": "x",
        "title_of_class": "COM", "verified_by": "auto",
    }])
    # F4-05a (2026-09-28): WONT FIX. El contrato del test es "no lanza".
    # La ausencia de assert es intencional: la propia ejecucion es la
    # verificacion. Documentado para que no parezca un placeholder.
    rcc._validate(df)  # no lanza


def test_validate_duplicado_falla():
    df = pd.DataFrame([
        {"CUSIP": "037833100", "ticker": "AAPL", "valid_from": "2025-12-31",
         "valid_to": "", "source": "SEC-EDGAR", "reason": "x",
         "title_of_class": "COM", "verified_by": "auto"},
        {"CUSIP": "037833100", "ticker": "AAPL2", "valid_from": "2025-12-31",
         "valid_to": "", "source": "SEC-EDGAR", "reason": "x",
         "title_of_class": "COM", "verified_by": "auto"},
    ])
    with pytest.raises(ValueError):
        rcc._validate(df)


def test_validate_valid_from_mayor_falla():
    df = pd.DataFrame([{
        "CUSIP": "037833100", "ticker": "AAPL", "valid_from": "2026-06-30",
        "valid_to": "2025-12-31", "source": "SEC-EDGAR", "reason": "x",
        "title_of_class": "COM", "verified_by": "auto",
    }])
    with pytest.raises(ValueError):
        rcc._validate(df)


# --- main sin trimestres ---

def test_main_aborta_sin_trimestres(tmp_path, monkeypatch):
    monkeypatch.setattr(rcc, "DATA_DIR", tmp_path / "nope")
    monkeypatch.setattr("sys.argv", ["regenerate_cusip_crosswalk.py"])
    assert rcc.main() == 1


def test_extract_excluye_call_put(tmp_path):
    """Filtro 5.1: SSHPRNAMTTYPE != 'SH' o PUTCALL no-null -> excluido."""
    p = _mk_infotable(tmp_path, "2026Q1", [
        ("111", "SC_AAPL", "COM", "SH", None),        # OK
        ("222", "SC_AAPL", "COM", "SH", "CALL"),      # PUTCALL -> fuera
        ("333", "SC_AAPL", "COM", "PRN", None),       # no SH -> fuera
        ("444", "SC_ABBV", "COM", "SH", "PUT"),       # PUTCALL -> fuera
    ])
    idx = {"SC_AAPL": "AAPL", "SC_ABBV": "ABBV"}
    obs = rcc._extract_observations("2026Q1", p, idx)
    assert set(obs.keys()) == {("111", "AAPL")}


# --- _load_historical_auto + merge acumulativo (fix CI) -----------------

def _mk_crosswalk_row(cusip, ticker, valid_from, valid_to, verified_by="auto"):
    return {
        "CUSIP": cusip, "ticker": ticker, "valid_from": valid_from,
        "valid_to": valid_to, "source": "SEC-EDGAR", "reason": "x",
        "title_of_class": "COM", "verified_by": verified_by,
    }


def test_load_historical_auto_filtra_por_cutoff():
    current = pd.DataFrame([
        _mk_crosswalk_row("111", "AAA", "2025-12-31", ""),   # historica
        _mk_crosswalk_row("222", "BBB", "2026-06-30", ""),   # mismo cutoff
        _mk_crosswalk_row("333", "CCC", "2025-12-31", "", verified_by="manual"),
    ])
    hist = rcc._load_historical_auto(current, "2026-06-30")
    # Solo la fila 111 es auto con valid_from < 2026-06-30
    assert len(hist) == 1
    assert hist.iloc[0]["CUSIP"] == "111"


def test_load_historical_auto_vacio_si_df_vacio():
    hist = rcc._load_historical_auto(pd.DataFrame(columns=list(rcc.COLUMNS)), "2026-06-30")
    assert hist.empty


def test_merge_con_historical_preserva_cobertura_ci():
    """Escenario CI: solo hay Q2, pero la historica no se pierde."""
    manual = pd.DataFrame([_mk_crosswalk_row("999", "MAN", "2020-01-01", "2025-12-31", verified_by="manual")])
    auto_nuevas = [_mk_crosswalk_row("222", "BBB", "2026-06-30", "")]
    historicas = pd.DataFrame([_mk_crosswalk_row("111", "AAA", "2025-12-31", "")])
    out = rcc._merge(manual, auto_nuevas, historical_df=historicas)
    assert len(out) == 3
    cusips = set(out["CUSIP"])
    assert cusips == {"111", "222", "999"}


def test_merge_precedencia_manual_sobre_historical():
    """Si el mismo CUSIP esta en manual y en historica, gana manual."""
    manual = pd.DataFrame([_mk_crosswalk_row("111", "MAN", "2020-01-01", "", verified_by="manual")])
    historicas = pd.DataFrame([_mk_crosswalk_row("111", "HIST", "2025-12-31", "")])
    out = rcc._merge(manual, [], historical_df=historicas)
    assert len(out) == 1
    assert out.iloc[0]["ticker"] == "MAN"


def test_merge_precedencia_auto_nueva_sobre_historical():
    """Si el mismo CUSIP esta en auto nueva y en historica, gana la nueva."""
    auto_nuevas = [_mk_crosswalk_row("111", "NEW", "2026-06-30", "")]
    historicas = pd.DataFrame([_mk_crosswalk_row("111", "HIST", "2025-12-31", "")])
    out = rcc._merge(pd.DataFrame(columns=list(rcc.COLUMNS)), auto_nuevas, historical_df=historicas)
    assert len(out) == 1
    assert out.iloc[0]["ticker"] == "NEW"


# --- _load_radar_index: descartes ---

def test_load_radar_index_descarta_vacios_y_nan(tmp_path):
    cat = tmp_path / "catalog.csv"
    pd.DataFrame([
        {"radar_ticker": "AAPL", "share_class_figi": "SC_AAPL"},
        {"radar_ticker": "", "share_class_figi": "SC_EMPTY_TK"},
        {"radar_ticker": "MSFT", "share_class_figi": ""},
        {"radar_ticker": "NANTK", "share_class_figi": "nan"},
        {"radar_ticker": "  GOOG  ", "share_class_figi": "  SC_GOOG  "},
    ]).to_csv(cat, index=False)
    idx = rcc._load_radar_index(cat)
    assert idx == {"SC_AAPL": "AAPL", "SC_GOOG": "GOOG"}


# --- _extract_observations: title rellenado en segunda observacion ---

def test_extract_rellena_title_en_segunda_observacion(tmp_path):
    d = tmp_path / "2026Q1"
    d.mkdir()
    pd.DataFrame([
        ("037833100", "SC_AAPL", "", "SH", None),
        ("037833100", "SC_AAPL", "COM", "SH", None),
    ], columns=["CUSIP", "FIGI", "TITLEOFCLASS",
                "SSHPRNAMTTYPE", "PUTCALL"]).to_parquet(d / "INFOTABLE.parquet")
    idx = {"SC_AAPL": "AAPL"}
    obs = rcc._extract_observations("2026Q1", d / "INFOTABLE.parquet", idx)
    assert obs[("037833100", "AAPL")]["title"] == "COM"


# --- _build_auto_rows: title vacio -> COM ---

def test_build_auto_rows_title_vacio_usa_com():
    obs = {("111", "AAPL"): {"title": "", "quarters": ["2026Q1"]}}
    rows = rcc._build_auto_rows(obs, latest_quarter="2026Q1")
    assert rows[0]["title_of_class"] == "COM"


# --- main happy path ---

def _mk_workspace(tmp_path):
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    q = data_dir / "2026Q1"
    q.mkdir()
    pd.DataFrame([
        ("037833100", "SC_AAPL", "COM", "SH", None),
    ], columns=["CUSIP", "FIGI", "TITLEOFCLASS",
                "SSHPRNAMTTYPE", "PUTCALL"]).to_parquet(q / "INFOTABLE.parquet")
    catalog = tmp_path / "catalog.csv"
    pd.DataFrame([
        {"radar_ticker": "AAPL", "share_class_figi": "SC_AAPL"},
    ]).to_csv(catalog, index=False)
    return data_dir, catalog, tmp_path / "crosswalk.csv"


def test_main_happy_path(tmp_path, monkeypatch):
    data_dir, catalog, crosswalk = _mk_workspace(tmp_path)
    monkeypatch.setattr(rcc, "DATA_DIR", data_dir)
    monkeypatch.setattr(rcc, "CATALOG", catalog)
    monkeypatch.setattr(rcc, "CROSSWALK", crosswalk)
    monkeypatch.setattr("sys.argv", ["regenerate_cusip_crosswalk.py"])
    assert rcc.main() == 0
    assert crosswalk.exists()
    df = pd.read_csv(crosswalk, dtype=str)
    assert len(df) == 1
    assert df.iloc[0]["CUSIP"] == "037833100"
    assert df.iloc[0]["ticker"] == "AAPL"
    assert df.iloc[0]["verified_by"] == "auto"


def test_main_dry_run_no_escribe(tmp_path, monkeypatch):
    data_dir, catalog, crosswalk = _mk_workspace(tmp_path)
    monkeypatch.setattr(rcc, "DATA_DIR", data_dir)
    monkeypatch.setattr(rcc, "CATALOG", catalog)
    monkeypatch.setattr(rcc, "CROSSWALK", crosswalk)
    monkeypatch.setattr("sys.argv",
                        ["regenerate_cusip_crosswalk.py", "--dry-run"])
    assert rcc.main() == 0
    assert not crosswalk.exists()


# --- _clean_str ---

def test_clean_str_none_devuelve_vacio():
    assert rcc._clean_str(None) == ""


def test_clean_str_nan_float_devuelve_vacio():
    assert rcc._clean_str(float("nan")) == ""


def test_clean_str_string_nan_devuelve_vacio():
    assert rcc._clean_str("nan") == ""
    assert rcc._clean_str("  nan  ") == ""


def test_clean_str_vacio_y_espacios():
    assert rcc._clean_str("") == ""
    assert rcc._clean_str("   ") == ""


def test_clean_str_normaliza_strip():
    assert rcc._clean_str("  AAPL  ") == "AAPL"
    assert rcc._clean_str("COM") == "COM"


def test_clean_str_excepcion_pd_isna(monkeypatch):
    """Si pd.isna lanza (objeto raro), cae al try/except y sigue."""
    def fake_isna(v):
        raise TypeError("no soportado")
    monkeypatch.setattr(rcc.pd, "isna", fake_isna)
    assert rcc._clean_str("AAPL") == "AAPL"

# --- _load_historical_auto: df no vacio sin autos (hueco 139) ---

def test_load_historical_auto_sin_autos_devuelve_vacio():
    current = pd.DataFrame([
        _mk_crosswalk_row("111", "AAA", "2025-12-31", "", verified_by="manual"),
    ])
    hist = rcc._load_historical_auto(current, "2026-06-30")
    assert hist.empty


# --- _validate: columna faltante (hueco 162) ---

def test_validate_columna_faltante_falla():
    df = pd.DataFrame([{"CUSIP": "111"}])  # faltan 7 columnas
    with pytest.raises(ValueError, match="columna faltante"):
        rcc._validate(df)


# --- main: _validate lanza (huecos 223-225) ---

def test_main_validate_falla_devuelve_1(tmp_path, monkeypatch):
    data_dir, catalog, crosswalk = _mk_workspace(tmp_path)
    monkeypatch.setattr(rcc, "DATA_DIR", data_dir)
    monkeypatch.setattr(rcc, "CATALOG", catalog)
    monkeypatch.setattr(rcc, "CROSSWALK", crosswalk)
    def fake_validate(df):
        raise ValueError("simulado")
    monkeypatch.setattr(rcc, "_validate", fake_validate)
    monkeypatch.setattr("sys.argv", ["regenerate_cusip_crosswalk.py"])
    assert rcc.main() == 1
    assert not crosswalk.exists()

# --- main: mismo cusip en 2 quarters (huecos 201-203) ---

def _mk_workspace_2q(tmp_path):
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    for q, title in (("2026Q1", ""), ("2026Q2", "COM")):
        d = data_dir / q
        d.mkdir()
        pd.DataFrame([
            ("037833100", "SC_AAPL", title, "SH", None),
        ], columns=["CUSIP", "FIGI", "TITLEOFCLASS",
                    "SSHPRNAMTTYPE", "PUTCALL"]).to_parquet(d / "INFOTABLE.parquet")
    catalog = tmp_path / "catalog.csv"
    pd.DataFrame([
        {"radar_ticker": "AAPL", "share_class_figi": "SC_AAPL"},
    ]).to_csv(catalog, index=False)
    return data_dir, catalog, tmp_path / "crosswalk.csv"


def test_main_mismo_cusip_dos_quarters(tmp_path, monkeypatch):
    data_dir, catalog, crosswalk = _mk_workspace_2q(tmp_path)
    monkeypatch.setattr(rcc, "DATA_DIR", data_dir)
    monkeypatch.setattr(rcc, "CATALOG", catalog)
    monkeypatch.setattr(rcc, "CROSSWALK", crosswalk)
    monkeypatch.setattr("sys.argv", ["regenerate_cusip_crosswalk.py"])
    assert rcc.main() == 0
    df = pd.read_csv(crosswalk, dtype=str, keep_default_na=False)
    assert len(df) == 1
    assert df.iloc[0]["valid_from"] == "2026-03-31"
    assert df.iloc[0]["valid_to"] == ""
    assert df.iloc[0]["title_of_class"] == "COM"