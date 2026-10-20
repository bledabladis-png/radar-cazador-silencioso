"""REP-AUDIT-20261020-MONITOR-DATE: monitor_double_counting escribe date=sesion.

Antes: columna `date` = datetime.now().isoformat() (timestamp del run).
Violaba el contrato del 00_ARRANQUE: ningun writer publica `date` no
bursatil. 41 filas del CSV con sabados/domingos/futuras.

Ahora: date = last_expected_market_date() (sesion NYSE), run_timestamp
preserva el ISO del run como columna separada.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.market_calendar import is_market_day


def test_monitor_escribe_date_bursatil(tmp_path, monkeypatch):
    """Simula un run sobre un CSV sintetico y verifica la columna date."""
    import importlib
    mod = importlib.import_module("validation.monitor_double_counting")

    # Redirigir OUTPUT_CSV a tmp
    target = tmp_path / "monitor.csv"
    monkeypatch.setattr(mod, "OUTPUT_CSV", target)

    # Crear CSV origen: analisis_lideres.csv con columnas esperadas
    report_dir = tmp_path / "outputs" / "report"
    report_dir.mkdir(parents=True)
    # PROJECT_ROOT del modulo apunta al repo real; parcheamos la ruta
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    leaders = report_dir / "analisis_lideres.csv"
    leaders.write_text(
        "ticker,rs_mom,flow_proxy_z,wls,stability,wyckoff_score\n"
        + "\n".join(f"T{i},{i*0.1:.2f},{-i*0.05:.2f},{i*0.2:.2f},{i*0.05:.2f},{i*0.1-0.5:.2f}"
                    for i in range(15)),
        encoding="utf-8",
    )

    try:
        mod.main()
    except SystemExit:
        pass

    assert target.exists(), "CSV no escrito"
    df = pd.read_csv(target)
    assert "date" in df.columns
    assert "run_timestamp" in df.columns
    for d in df["date"].unique():
        assert is_market_day(pd.Timestamp(d).date()), f"date={d} no bursatil"
    # run_timestamp debe tener formato ISO (con T)
    assert df["run_timestamp"].str.contains("T").any()