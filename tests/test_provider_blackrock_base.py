"""Tests directos de data/providers/_blackrock_base.py.

G12 A5-70: tests del codigo comun extraido. No sustituyen a los
tests de wrappers (test_provider_blackrock_parse_fecha.py), los
complementan cubriendo:
  - download_fund_file: cache reciente, cache obsoleta, error de red.
  - parse_hist_sheet: XML SpreadsheetML minimo, hoja ausente.
  - get_blackrock_primary_flow: pipeline completo end-to-end con cache.

Sin acceso de red (todos los requests van mockeados o no se invocan).
"""
import time

import pandas as pd
import pytest

from data.providers import _blackrock_base as bb


# ---------------------------------------------------------------------
# Fixture: XML SpreadsheetML minimo con hoja Histórico
# ---------------------------------------------------------------------

def _make_xml(rows):
    """Genera un SpreadsheetML con hoja Histórico. rows = filas de datos (sin header)."""
    def cell(v, t="String"):
        return f'<Cell><Data ss:Type="{t}">{v}</Data></Cell>'
    header = (
        '<Row>'
        + cell("Date") + cell("NAV") + cell("Shares") + cell("Total Net Assets")
        + '</Row>'
    )
    body = ""
    for r in rows:
        body += (
            '<Row>'
            + cell(r["date"], "String")
            + cell(r["nav"], "Number")
            + cell(r["shares"], "Number")
            + cell(r["tna"], "Number")
            + '</Row>'
        )
    return (
        '<?xml version="1.0"?>'
        '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"'
        ' xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">'
        '<Worksheet ss:Name="Histórico">'
        '<Table>' + header + body + '</Table>'
        '</Worksheet>'
        '</Workbook>'
    )


MIN_ROWS = [
    {"date": "1 ene 2026", "nav": "100.0", "shares": "1000", "tna": "100000"},
    {"date": "2 ene 2026", "nav": "101.0", "shares": "1010", "tna": "102010"},
    {"date": "3 ene 2026", "nav": "102.0", "shares": "1005", "tna": "102510"},
]


# ---------------------------------------------------------------------
# parse_fecha_es (re-export)
# ---------------------------------------------------------------------

class TestParseFechaEs:
    def test_dic_acento_correcto(self):
        assert bb.parse_fecha_es("27 dic 2000") == pd.Timestamp(2000, 12, 27)

    def test_mes_en_mayusculas(self):
        assert bb.parse_fecha_es("27 DIC 2000") == pd.Timestamp(2000, 12, 27)

    def test_invalido_devuelve_none(self):
        assert bb.parse_fecha_es("32 dic 2000") is None
        assert bb.parse_fecha_es("27 foo 2000") is None
        assert bb.parse_fecha_es("") is None
        assert bb.parse_fecha_es(None) is None


# ---------------------------------------------------------------------
# download_fund_file
# ---------------------------------------------------------------------

class TestDownloadFundFile:

    def test_cache_reciente_no_descarga(self, tmp_path, monkeypatch):
        cache = tmp_path / "fund.xml"
        cache.write_bytes(b"x" * 200000)

        # Si se llama requests.get, fallar el test
        def _forbidden(*a, **kw):
            pytest.fail("requests.get no deberia invocarse con cache reciente")
        monkeypatch.setattr(bb.requests, "get", _forbidden)

        ok = bb.download_fund_file(
            url="http://example.com/x",
            cache_file=cache,
            referer="http://example.com/",
            label="TEST",
        )
        assert ok is True

    def test_cache_obsoleta_descarga(self, tmp_path, monkeypatch):
        cache = tmp_path / "fund.xml"
        cache.write_bytes(b"x" * 200000)
        # mtime = 2 dias atras -> expira TTL 23h
        old = time.time() - 2 * 86400
        import os
        os.utime(cache, (old, old))

        called = {"n": 0}

        class _Resp:
            content = b"y" * 500000
            def raise_for_status(self):
                pass

        def _get(url, headers=None, timeout=None):
            called["n"] += 1
            return _Resp()

        monkeypatch.setattr(bb.requests, "get", _get)

        ok = bb.download_fund_file(
            url="http://example.com/x",
            cache_file=cache,
            referer="http://example.com/",
            label="TEST",
        )
        assert ok is True
        assert called["n"] == 1
        assert cache.read_bytes() == b"y" * 500000

    def test_error_red_con_cache_previa_usa_cache(self, tmp_path, monkeypatch):
        cache = tmp_path / "fund.xml"
        cache.write_bytes(b"x" * 200000)
        old = time.time() - 2 * 86400
        import os
        os.utime(cache, (old, old))

        def _boom(*a, **kw):
            raise ConnectionError("red caida")
        monkeypatch.setattr(bb.requests, "get", _boom)

        ok = bb.download_fund_file(
            url="http://example.com/x",
            cache_file=cache,
            referer="http://example.com/",
            label="TEST",
        )
        assert ok is True

    def test_error_red_sin_cache_devuelve_false(self, tmp_path, monkeypatch):
        cache = tmp_path / "no_existe.xml"

        def _boom(*a, **kw):
            raise ConnectionError("red caida")
        monkeypatch.setattr(bb.requests, "get", _boom)

        ok = bb.download_fund_file(
            url="http://example.com/x",
            cache_file=cache,
            referer="http://example.com/",
            label="TEST",
        )
        assert ok is False


# ---------------------------------------------------------------------
# parse_hist_sheet
# ---------------------------------------------------------------------

class TestParseHistSheet:

    def test_xml_minimo(self, tmp_path):
        xml_path = tmp_path / "sample.xml"
        xml_path.write_text(_make_xml(MIN_ROWS), encoding="utf-8")

        df = bb.parse_hist_sheet(xml_path)
        assert list(df.columns) == ["date", "nav", "shares_outstanding", "total_net_assets"]
        assert len(df) == 3
        assert df.iloc[0]["nav"] == 100.0
        assert df.iloc[0]["shares_outstanding"] == 1000.0
        assert df.iloc[-1]["nav"] == 102.0
        assert df["date"].dtype.kind == "M"  # datetime64

    def test_hoja_ausente_raises(self, tmp_path):
        xml_path = tmp_path / "sin_hist.xml"
        xml_path.write_text(
            '<?xml version="1.0"?>'
            '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"'
            ' xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">'
            '<Worksheet ss:Name="Otra"><Table><Row>'
            '<Cell><Data ss:Type="String">x</Data></Cell>'
            '</Row></Table></Worksheet>'
            '</Workbook>',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Hoja"):
            bb.parse_hist_sheet(xml_path)


# ---------------------------------------------------------------------
# get_blackrock_primary_flow (pipeline completo, sin red)
# ---------------------------------------------------------------------

class TestGetBlackrockPrimaryFlow:

    def test_pipeline_completo(self, tmp_path):
        cache = tmp_path / "fund.xml"
        cache.write_text(_make_xml(MIN_ROWS), encoding="utf-8")
        out_csv = tmp_path / "out.csv"

        df = bb.get_blackrock_primary_flow(
            url="http://example.com/x",
            cache_file=cache,
            output_csv=out_csv,
            referer="http://example.com/",
            label="TEST",
            force_download=False,
        )

        # DataFrame devuelto: 1 fila (la ultima)
        assert len(df) == 1
        expected_cols = [
            "date", "nav", "shares_outstanding", "shares_change",
            "total_net_assets", "estimated_flow_eur", "flow_pct_assets",
            "flow_zscore", "flow_zscore_regime", "flow_5d", "flow_20d",
        ]
        assert list(df.columns) == expected_cols

        # CSV escrito con las 3 filas completas
        assert out_csv.exists()
        df_csv = pd.read_csv(out_csv)
        assert len(df_csv) == 3
        assert list(df_csv.columns) == expected_cols

        # Calculo verificado: shares_change entre fila 2 y fila 1 = +10
        assert df_csv.iloc[1]["shares_change"] == 10.0
        # estimated_flow_eur = 10 * 101.0
        assert df_csv.iloc[1]["estimated_flow_eur"] == pytest.approx(1010.0)

    def test_force_download_falla_sin_red_devuelve_vacio(self, tmp_path, monkeypatch):
        cache = tmp_path / "no_existe.xml"

        def _boom(*a, **kw):
            raise ConnectionError("red caida")
        monkeypatch.setattr(bb.requests, "get", _boom)

        df = bb.get_blackrock_primary_flow(
            url="http://example.com/x",
            cache_file=cache,
            output_csv=tmp_path / "out.csv",
            referer="http://example.com/",
            label="TEST",
            force_download=False,
        )
        assert isinstance(df, pd.DataFrame)
        assert df.empty
