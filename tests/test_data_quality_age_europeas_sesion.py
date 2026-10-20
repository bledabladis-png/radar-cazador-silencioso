"""REP-AUDIT-20261020-DQ-EU-AGE: age de fuentes europeas contra sesion.

Bug detectado en auditoria 2026-10-20: la rama europea de
compute_data_quality media age contra `now` (fecha del run) en vez
de `_session_date` (sesion bursatil). 3 filas de data_quality.csv
con age=4 cuando calc_age=3.

Fix: usa _session_date, igual que las demas fuentes.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators import data_quality as dq


def _fake_eu_csv(tmp_path, date_iso, last_date_iso):
    eu = tmp_path / "outputs" / "history" / "european_coverage.csv"
    eu.parent.mkdir(parents=True, exist_ok=True)
    eu.write_text(
        f"date,ticker,source,last_date,days_gap,status\n"
        f"{date_iso},AIR.PA,Euronext,{last_date_iso},0,OK\n",
        encoding="utf-8",
    )


def test_age_europeas_contra_sesion_no_contra_run(tmp_path, monkeypatch):
    """reference_date martes 02:50 Madrid -> sesion = lunes. last_date = viernes.
    age esperado = 3 (lunes - viernes), NO 4 (martes - viernes)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True, exist_ok=True)
    (tmp_path / "outputs" / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data").mkdir(exist_ok=True)
    _fake_eu_csv(tmp_path, "2026-10-16", "2026-10-16")

    ref = pd.Timestamp("2026-10-20 02:50:00")
    df = dq.compute_data_quality(reference_date=ref)
    eu = df[df["source"] == "Euronext"]
    assert not eu.empty, "sin fila Euronext"
    row = eu.iloc[0]
    assert row["date"] == "2026-10-19", f"date={row['date']}, esperado 2026-10-19"
    assert row["age_calendar_days"] == 3, (
        f"age={row['age_calendar_days']}, esperado 3 (lunes - viernes). "
        "Si es 4, el bug del 2026-10-20 vuelve."
    )