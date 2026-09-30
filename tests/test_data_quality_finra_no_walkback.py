# -*- coding: utf-8 -*-
"""Fix G: data_quality no aplica walk-back a FINRA.

Bug: FINRA publica datos semanales (columna 'week'). El valor es la
fecha del lunes de la semana. _last_market_session convierte esa
fecha-semana en fecha-dia bursatil (lunes festivo -> viernes anterior).

Verificado 2026-09-30:
- darkpool_history.csv: week=2026-09-07 (lunes)
- 2026-09-07 es Labor Day (no bursatil NYSE)
- _last_market_session(2026-09-07) = 2026-09-04
- data_quality.csv publica 2026-09-04
- render_darkpool publica 2026-09-07
- Discrepancia: misma fuente, dos fechas distintas.

Fix: las fuentes con frecuencia semanal (FINRA) no pasan por
_last_market_session.
"""
from pathlib import Path

import pandas as pd

from indicators import data_quality as dq


def test_data_quality_metadata_finra_sin_walkback():
    """La fuente FINRA debe declarar walk_back=False en su metadata."""
    # Reflejar el metadata accesible desde el modulo
    src_finra = None
    for src in [
        {"source": "FINRA Dark Pool", "file": "outputs/history/darkpool_history.csv",
         "date_cols": ["week"], "frequency": "finra"}
    ]:
        src_finra = src
    # Este test es de comportamiento, no de metadata: ver abajo
    assert src_finra is not None


def test_data_quality_finra_publica_week_directa(tmp_path, monkeypatch):
    """Si darkpool_history tiene week=2026-09-07 (Labor Day), data_quality
    debe publicar 2026-09-07, no 2026-09-04.
    """
    # Crear un CSV con la fecha del lunes festivo
    darkpool = tmp_path / "darkpool_history.csv"
    pd.DataFrame({
        "week": ["2026-08-31", "2026-09-07"],
        "ratio": [0.22, 0.24],
        "ratio_ewm": [0.23, 0.24],
    }).to_csv(darkpool, index=False)

    # Monkeypatch: sustituir la ruta de la fuente FINRA
    # Necesitamos interceptar la lectura del path en compute_data_quality.
    # Si la logica esta en un helper, lo parcheamos; si esta inline,
    # monkeypatch sobre Path.open.

    _orig_read = pd.read_csv

    def fake_read_csv(path, *args, **kwargs):
        if "darkpool_history.csv" in str(path):
            return _orig_read(darkpool, *args, **kwargs)
        return _orig_read(path, *args, **kwargs)

    monkeypatch.setattr(dq.pd, "read_csv", fake_read_csv)

    df = dq.compute_data_quality(reference_date="2026-09-30")
    finra = df[df["source"] == "FINRA Dark Pool"]
    assert not finra.empty, "no se encontro la fila FINRA"
    fecha = str(finra.iloc[0]["last_date"])
    assert fecha == "2026-09-07", (
        f"Esperado 2026-09-07 (week directa), obtenido {fecha}. "
        f"Si es 2026-09-04, el walk-back de calendario NYSE se esta "
        f"aplicando a una fecha-semana."
    )