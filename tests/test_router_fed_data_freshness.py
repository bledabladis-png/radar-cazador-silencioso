# -*- coding: utf-8 -*-
"""Test de regresion: router.get_fed_data debe preferir CSVs frescos.

Item: REP-AUDIT-20261022-LIQUIDITY-ROUTER-PRIORITY
Causa raiz: router priorizaba FRED API (con lag) sobre CSVs
data/macro_manual/ (pusheados por bot diario). liquidity.py veia
datos stale y devolvia N/A mientras fls.py (lectura directa de
CSVs) veia datos frescos. Contradiccion visible en el reporte.
"""

import os
import pandas as pd
import pytest
from data.providers.router import DataRouter


def test_get_fed_data_prioriza_csvs_sobre_api():
    """Los CSVs manuales existen y son mas frescos -> deben usarse."""
    macro_dir = "data/macro_manual"
    if not os.path.exists(macro_dir):
        pytest.skip("data/macro_manual no existe (entorno sin datos)")

    sofr_path = os.path.join(macro_dir, "sofr.csv")
    if not os.path.exists(sofr_path):
        pytest.skip("sofr.csv no existe")

    sofr = pd.read_csv(sofr_path, index_col=0, parse_dates=True)
    last_csv = sofr.index.max()

    router = DataRouter()
    df = router.get_fed_data()
    assert df is not None and not df.empty, "get_fed_data devolvio vacio"

    last_router = df.dropna(how="all").index.max()
    assert last_router >= last_csv, (
        f"Router devuelve hasta {last_router}, CSV tiene {last_csv}. "
        f"El router NO esta usando los CSVs frescos."
    )


def test_get_fed_data_columnas_esperadas():
    """El DataFrame devuelto incluye las columnas que consume liquidity.py."""
    macro_dir = "data/macro_manual"
    if not os.path.exists(macro_dir):
        pytest.skip("data/macro_manual no existe")

    router = DataRouter()
    df = router.get_fed_data()
    assert df is not None, "get_fed_data devolvio None"

    for col in ["fed_balance", "reverse_repo", "sofr"]:
        assert col in df.columns, f"falta columna {col}: {list(df.columns)}"

def test_router_no_asigna_dprime_a_fed_funds(tmp_path, monkeypatch):
    """discount_rate.csv (DPRIME) no debe mapearse a fed_funds."""
    import pandas as pd
    import data.providers.fred as fred_mod
    import requests
    (tmp_path / "data" / "macro_manual").mkdir(parents=True)
    pd.DataFrame({"date": ["2026-11-18"], "DPRIME": [7.0]}).to_csv(tmp_path / "data" / "macro_manual" / "discount_rate.csv", index=False)
    pd.DataFrame({"date": ["2026-11-18"], "EFFR": [3.88]}).to_csv(tmp_path / "data" / "macro_manual" / "effr.csv", index=False)
    pd.DataFrame({"date": ["2026-11-18"], "SOFR": [3.91]}).to_csv(tmp_path / "data" / "macro_manual" / "sofr.csv", index=False)
    monkeypatch.chdir(tmp_path)
    df = DataRouter().get_fed_data()
    assert "prime_rate" in df.columns and float(df["prime_rate"].iloc[-1]) == 7.0
    assert float(df["fed_funds"].iloc[-1]) == 3.88
    log_p = tmp_path / "fred.jsonl"
    monkeypatch.setattr(fred_mod, "_FRED_LOG_PATH", log_p)
    def _boom(*a, **kw): raise requests.ConnectionError("net down")
    monkeypatch.setattr(fred_mod.web, "DataReader", _boom)
    out = fred_mod.FredProvider()._download_series("SOFR")
    assert out.empty and log_p.exists() and "error" in log_p.read_text(encoding="utf-8")
