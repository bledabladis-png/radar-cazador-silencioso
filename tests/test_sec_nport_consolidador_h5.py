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
    import pandas as pd
    df_out = pd.read_csv(csv)
    assert len(df_out) == 1, f"esperaba 1 fila con 1 trimestre, got {len(df_out)}"
    assert df_out["POSITION_CHANGE"].isna().all()


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

def test_consolidador_agrega_y_aisla_series_id(tmp_path):
    """Mismo CIK con dos SERIES_ID y multiples lotes no mezcla ETFs."""
    import pandas as pd
    for q, dt, bal_s1, bal_s2 in [("2026q1", "31-DEC-2025", [600, 400], [9999]), ("2026q2", "31-MAR-2026", [1500], [8888])]:
        qd = tmp_path / "data" / "nport" / q
        qd.mkdir(parents=True)
        (qd / "SUBMISSION.tsv").write_text(f"ACCESSION_NUMBER\tREPORT_DATE\nA1\t{dt}\nA2\t{dt}\n", encoding="utf-8")
        (qd / "REGISTRANT.tsv").write_text("ACCESSION_NUMBER\tCIK\tREGISTRANT_NAME\nA1\t1064641\tTRUST\nA2\t1064641\tTRUST\n", encoding="utf-8")
        (qd / "FUND_REPORTED_INFO.tsv").write_text("ACCESSION_NUMBER\tSERIES_NAME\tSERIES_ID\nA1\tFund1\tS001\nA2\tFund2\tS002\n", encoding="utf-8")
        hrows = ["ACCESSION_NUMBER\tHOLDING_ID\tISSUER_NAME\tISSUER_CUSIP\tBALANCE\tCURRENCY_VALUE\tPERCENTAGE\tASSET_CAT\tISSUER_TYPE"]
        irows = ["HOLDING_ID\tIDENTIFIER_ISIN\tIDENTIFIER_TICKER"]
        for idx, b in enumerate(bal_s1, 1):
            hrows.append(f"A1\tH1_{idx}\tIssuer\t123456789\t{b}\t100\t1.0\tEC\tCORP")
            irows.append(f"H1_{idx}\tUS1234567890\tTST")
        for idx, b in enumerate(bal_s2, 1):
            hrows.append(f"A2\tH2_{idx}\tIssuer\t123456789\t{b}\t100\t1.0\tEC\tCORP")
            irows.append(f"H2_{idx}\tUS1234567890\tTST")
        (qd / "FUND_REPORTED_HOLDING.tsv").write_text("\n".join(hrows) + "\n", encoding="utf-8")
        (qd / "IDENTIFIERS.tsv").write_text("\n".join(irows) + "\n", encoding="utf-8")
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    r = subprocess.run([sys.executable, str(ROOT / "data" / "providers" / "sec_nport_quarters_position_change.py")], cwd=str(tmp_path), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(tmp_path / "outputs" / "history" / "sec_nport_position_change_quarterly.csv")
    s1 = df[df["SERIES_ID"] == "S001"]
    assert len(s1) == 1 and float(s1.iloc[0]["PREV_BALANCE"]) == 1000.0 and float(s1.iloc[0]["POSITION_CHANGE"]) == 500.0
    s2 = df[df["SERIES_ID"] == "S002"]
    assert len(s2) == 1 and float(s2.iloc[0]["PREV_BALANCE"]) == 9999.0 and float(s2.iloc[0]["POSITION_CHANGE"]) == -1111.0
