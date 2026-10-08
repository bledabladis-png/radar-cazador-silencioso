"""Contrato: check_iae_section no confunde el STALE del N-PORT
con el STALE de la seccion IAE.

Regresion 2026-10-08: el check buscaba "STALE" en todo el reporte.
El N-PORT STALE (tabla Data Freshness) disparaba un falso WARN.
"""
from __future__ import annotations

from scripts.health_check import check_iae_section


def _write_report(tmp_path, monkeypatch, body):
    report = tmp_path / "outputs" / "report" / "reporte_diario.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(body, encoding="utf-8")
    monkeypatch.setattr("scripts.health_check.PROJECT_ROOT", tmp_path)
    return report


def test_nport_stale_no_contamina_iae(tmp_path, monkeypatch):
    # El reporte contiene un STALE en Data Freshness (N-PORT) pero
    # la seccion IAE no menciona la palabra.
    body = (
        "## Data Freshness\n"
        "| SEC N-PORT | 2026-03-31 | 191 | sec | STALE |\n"
        "\n"
        "## Acumulacion Institucional (13F)\n"
        "Contenido limpio, sin marcas de estado.\n"
    )
    _write_report(tmp_path, monkeypatch, body)
    res = check_iae_section(is_ci=False)
    assert len(res) == 1
    assert res[0].status == "OK", res[0]
    assert "presente" in res[0].message


def test_iae_stale_con_official_list_es_ok(tmp_path, monkeypatch):
    body = (
        "## Acumulacion Institucional (13F)\n"
        "Estado: STALE (Official List pendiente).\n"
    )
    _write_report(tmp_path, monkeypatch, body)
    res = check_iae_section(is_ci=False)
    assert res[0].status == "OK"
    assert "official_list_pending" in res[0].message


def test_iae_stale_sin_official_list_es_warn(tmp_path, monkeypatch):
    body = (
        "## Acumulacion Institucional (13F)\n"
        "Estado: STALE por motivo X.\n"
    )
    _write_report(tmp_path, monkeypatch, body)
    res = check_iae_section(is_ci=False)
    assert res[0].status == "WARN"
    assert "razon desconocida" in res[0].message


def test_seccion_iae_ausente(tmp_path, monkeypatch):
    body = "## Otra seccion\nContenido.\n"
    _write_report(tmp_path, monkeypatch, body)
    res = check_iae_section(is_ci=False)
    assert res[0].status == "WARN"
    assert "no encontrada" in res[0].message
