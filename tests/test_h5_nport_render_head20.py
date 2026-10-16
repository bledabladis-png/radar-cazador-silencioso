# -*- coding: utf-8 -*-
"""Test H5 dirigido (auditor 2026-10-16): confirmar que la tabla de
posiciones N-PORT usa el trimestre mas reciente.

La tabla se construye con:
    df_nport = pd.read_csv(quarterly_csv)
    head(20) = sort_values('REPORT_DATE', ascending=False).head(20)

Con el consolidado actual (26304 filas Q2-2026 + 100 filas Q1-2026),
el head(20) debe devolver 20 filas Q2.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUARTERLY = ROOT / "outputs" / "history" / "sec_nport_position_change_quarterly.csv"


def _head20(df):
    """Replica de compute_flows_secondary (linea 213)."""
    return df.sort_values("REPORT_DATE", ascending=False).head(20)


def test_head20_sobre_consolidado_real(tmp_path):
    """Si el consolidado real existe, head(20) debe ser todo Q2-2026."""
    if not QUARTERLY.exists():
        import pytest
        pytest.skip("consolidado no presente en local")
    df = pd.read_csv(QUARTERLY, parse_dates=["REPORT_DATE"])
    top = _head20(df)
    # Todos los REPORT_DATE del top20 deben ser el maximo del consolidado
    fecha_max = df["REPORT_DATE"].max()
    fechas_top = set(top["REPORT_DATE"].unique())
    assert fechas_top == {fecha_max}, (
        "head(20) no prioriza la fecha mas reciente. "
        "Fechas encontradas: {0}. Maximo: {1}".format(
            fechas_top, fecha_max
        )
    )


def test_head20_prioriza_q2_sintetico():
    """Con datos sinteticos (100 filas Q1 + 500 filas Q2), head(20) = 20 Q2."""
    # Nota: cada fila es una posicion declarada. Muchas filas comparten
    # la misma REPORT_DATE (fecha de cierre trimestral).
    q1_date = pd.Timestamp("2026-03-31")
    q2_date = pd.Timestamp("2026-06-30")
    df = pd.DataFrame({
        "REPORT_DATE": [q1_date] * 100 + [q2_date] * 500,
        "CIK": [1064641] * 600,
    })
    top = _head20(df)
    assert len(top) == 20
    assert (top["REPORT_DATE"] == q2_date).all(), (
        "head(20) no priorizo Q2 sobre Q1"
    )


def test_head20_con_un_solo_trimestre():
    """Con solo Q1 (100 filas misma fecha), head(20) = 20 filas Q1."""
    q1_date = pd.Timestamp("2026-03-31")
    df = pd.DataFrame({
        "REPORT_DATE": [q1_date] * 100,
        "CIK": [1064641] * 100,
    })
    top = _head20(df)
    assert len(top) == 20
    assert (top["REPORT_DATE"] == q1_date).all()