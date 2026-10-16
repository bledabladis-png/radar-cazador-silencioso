# -*- coding: utf-8 -*-
"""Tests H5: consolidador N-PORT tolera 1 solo trimestre.

Contexto (auditoria 2026-10-16): la SEC publica el data set N-PORT
trimestral con latencia irregular. Cuando el ZIP del quarter objetivo
no esta disponible todavia, el workflow solo obtiene 1 ZIP (el anterior
con --backfill 1). Antes, el consolidador fallaba con exit 1 y tumbaba
el workflow entero. Ahora genera el consolidado con ese unico
trimestre (filas con POSITION_CHANGE=NaN).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_consolidador_tolera_1_trimestre(tmp_path, monkeypatch):
    """Con 1 solo trimestre disponible, generar consolidado (no fallar)."""
    # Simular data/nport/<q>/ con los ficheros minimos que espera el script
    qdir = tmp_path / "data" / "nport" / "2026q2"
    qdir.mkdir(parents=True)
    # SUBMISSION: ACCESSION_NUMBER + REPORT_DATE
    (qdir / "SUBMISSION.tsv").write_text(
        "ACCESSION_NUMBER\tREPORT_DATE\nACC001\t31-MAR-2026\n",
        encoding="utf-8",
    )
    # REGISTRANT: solo CIK objetivo (1064641 es SELECT SECTOR SPDR TRUST)
    (qdir / "REGISTRANT.tsv").write_text(
        "ACCESSION_NUMBER\tCIK\tREGISTRANT_NAME\n"
        "ACC001\t1064641\tSELECT SECTOR SPDR TRUST\n",
        encoding="utf-8",
    )
    # FUND_REPORTED_INFO
    (qdir / "FUND_REPORTED_INFO.tsv").write_text(
        "ACCESSION_NUMBER\tSERIES_NAME\tSERIES_ID\n"
        "ACC001\tTest Fund\tS001\n",
        encoding="utf-8",
    )
    # FUND_REPORTED_HOLDING
    (qdir / "FUND_REPORTED_HOLDING.tsv").write_text(
        "ACCESSION_NUMBER\tHOLDING_ID\tISSUER_NAME\tISSUER_CUSIP\tBALANCE\tCURRENCY_VALUE\tPERCENTAGE\tASSET_CAT\tISSUER_TYPE\n"
        "ACC001\tH1\tTest Issuer\t123456789\t1000\t50000\t5.0\tEC\tCORP\n",
        encoding="utf-8",
    )
    # IDENTIFIERS
    (qdir / "IDENTIFIERS.tsv").write_text(
        "HOLDING_ID\tIDENTIFIER_ISIN\tIDENTIFIER_TICKER\nH1\tUS1234567890\tTST\n",
        encoding="utf-8",
    )

    # Ejecutar script con cwd en tmp_path
    out_dir = tmp_path / "outputs" / "history"
    out_dir.mkdir(parents=True)

    # Monkeypatch script para usar tmp_path
    # Ejecutamos como subprocess desde tmp_path
    r = subprocess.run(
        [sys.executable, str(ROOT / "data" / "providers" / "sec_nport_quarters_position_change.py")],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    # Antes: exit 1. Ahora: exit 0 + warning.
    assert r.returncode == 0, (
        f"consolidador debe tolerar 1 trimestre. "
        f"stdout: {r.stdout}\nstderr: {r.stderr}"
    )
    assert "WARN" in r.stdout or "1 trimestre" in r.stdout
    # El CSV debe existir aunque solo tenga PREV_BALANCE=NaN
    csv = out_dir / "sec_nport_position_change_quarterly.csv"
    assert csv.exists(), f"consolidado no generado. stdout: {r.stdout}"


def test_consolidador_falla_sin_trimestres(tmp_path):
    """Con 0 trimestres, el consolidador SI debe fallar (exit 1)."""
    (tmp_path / "data" / "nport").mkdir(parents=True)
    r = subprocess.run(
        [sys.executable, str(ROOT / "data" / "providers" / "sec_nport_quarters_position_change.py")],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 1
    assert "no contiene ningun trimestre" in r.stdout.lower() or "ERROR" in r.stdout