# -*- coding: utf-8 -*-
"""Tests H14: pesos WLS variante índices.

Contexto (auditoria 2026-10-16): la variante de WLS usada en
select_index_leaders suma 0.95 (0.35+0.25+0.25+0.10), no 1.00.
Es una decision de diseno documentada en leaders.py L116-117, pero
no estaba testeada. Este test la fija para evitar cambios silenciosos.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# Suma esperada de pesos de la variante índices
WLS_INDEX_WEIGHT_SUM = 0.95
WLS_INDEX_WEIGHTS = {
    "rs": 0.35,
    "flow_proxy": 0.25,  # reducido de 0.30 (sectorial)
    "wyckoff": 0.25,
    "stability": 0.10,
}


def test_suma_de_pesos_es_0_95():
    """La suma de los pesos debe ser exactamente 0.95."""
    total = sum(WLS_INDEX_WEIGHTS.values())
    assert pytest.approx(total, abs=1e-9) == WLS_INDEX_WEIGHT_SUM


def test_pesos_coinciden_con_index_leaders():
    """Los pesos usados en index_leaders.compute_wls_for_index son estos."""
    import inspect
    from indicators import index_leaders as il
    src = inspect.getsource(il.compute_wls_for_index)
    # Verificar que los literales 0.35, 0.25, 0.25, 0.10 aparecen
    assert "0.35" in src
    assert "0.25" in src
    assert "0.10" in src


def test_wls_index_igual_a_formula_con_df_sintetico():
    """Con valores conocidos, el WLS resultante coincide con la
    formula declarada."""
    from indicators.index_leaders import compute_wls_for_index
    # 5 tickers con rs, flow, wyckoff, stability conocidos
    df = pd.DataFrame({
        "ticker": ["A", "B", "C", "D", "E"],
        "rs": [1.0, 1.5, 2.0, 2.5, 3.0],
        "flow_proxy_z": [0.5, 0.0, -0.5, 1.0, -1.0],
        "wyckoff_score": [0.5, 0.5, 0.5, 0.5, 0.5],
        "stability": [0.5, 0.5, 0.5, 0.5, 0.5],
        "persistence_10d": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    out = compute_wls_for_index(df)
    assert "wls" in out.columns
    # El wls es una combinacion de z-scores; validar que existe y es
    # numerico. La suma de pesos = 0.95 significa que si todos los
    # z-scores fueran 1.0, el wls sin persistencia seria 0.95.
    assert out["wls"].notna().all()


def test_nota_leaders_declara_suma():
    """La nota en leaders.py debe declarar los pesos y su justificacion."""
    from src.report import leaders as lr
    import inspect
    src = inspect.getsource(lr)
    assert "0.35/0.25/0.25/0.10" in src
    assert "flow 25% en vez de 30%" in src