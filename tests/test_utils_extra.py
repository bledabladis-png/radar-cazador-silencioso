# -*- coding: utf-8 -*-
"""D20 (2026-09-30): tests de las funciones puras de src/utils sin
cobertura previa.

Complementa test_utils.py (que cubre robust_zscore, tanh_normalize,
get_col, clean_oil_prices, append_dedup) y test_fu002_bymarket.py
(que cubre _latest_closed_session, _compute_by_market).

Cubre:
- detect_cross_module_conflict: 5 niveles + mte + None states.
- _confidence_range_row: version escalar de confidence_from_range.
- _try_cleanup: borrado tolerante a fallos.
"""
import numpy as np
import pandas as pd

from src.utils import (
    detect_cross_module_conflict,
    confidence_from_range,
    _confidence_range_row,
    _try_cleanup,
)


# ---------- detect_cross_module_conflict ----------
def test_detect_consensus_estres_financiero():
    """Todos los modulos con sesgo estres financiero -> CONSENSUS.

    Nota: RECESSION NO esta en financial_stress_states. La lista del
    sistema es ['CRISIS', 'HIGH_STRESS', 'ESTRECHA', 'STRESS']. RECESSION
    es regimen macro, no estrés financiero (decision documentada en el
    codigo).
    """
    out = detect_cross_module_conflict(
        macro_regime="ESTRECHA",
        financial_regime="CRISIS",
        volatility_regime="STRESS",
        liquidity_regime="HIGH_STRESS",
    )
    assert out["conflict_level"] == "CONSENSUS"
    assert "estres financiero" in out["message"]
    assert "Financial Stress: 4/4" in out["blocks"]


def test_detect_consensus_expansion():
    """Todos expansion -> CONSENSUS."""
    out = detect_cross_module_conflict(
        macro_regime="EXPANSION",
        financial_regime="ABUNDANTE",
        volatility_regime="LOW",
        liquidity_regime="EXPANSION",
    )
    assert out["conflict_level"] == "CONSENSUS"
    assert "expansivo" in out["message"]


def test_detect_conflict_estres_vs_expansion():
    """2 estres + 2 expansion -> CONFLICT."""
    out = detect_cross_module_conflict(
        macro_regime="ESTRECHA",
        financial_regime="CRISIS",
        volatility_regime="LOW",
        liquidity_regime="ABUNDANTE",
    )
    assert out["conflict_level"] == "CONFLICT"
    assert "Contradiccion significativa" in out["message"]


def test_detect_divergence_estres_mas_inflacion():
    """2+ estres + inflacion -> DIVERGENCE."""
    out = detect_cross_module_conflict(
        macro_regime="STAGFLATION",
        financial_regime="ESTRECHA",
        volatility_regime="STRESS",
        liquidity_regime="NEUTRAL",
    )
    assert out["conflict_level"] == "DIVERGENCE"


def test_detect_mixed_sin_direccion():
    """Ningun sesgo claro -> MIXED."""
    out = detect_cross_module_conflict(
        macro_regime="MIXED",
        financial_regime="NEUTRAL",
        volatility_regime="NORMAL",
        liquidity_regime="NEUTRAL",
    )
    assert out["conflict_level"] == "MIXED"
    assert "Sin direccion clara" in out["message"]


def test_detect_mixed_con_un_estres_y_una_inflacion():
    """1 estres + 1 inflacion -> MIXED (rama diferente)."""
    out = detect_cross_module_conflict(
        macro_regime="INFLATION SHOCK",
        financial_regime="ESTRECHA",
        volatility_regime="NORMAL",
        liquidity_regime="NEUTRAL",
    )
    assert out["conflict_level"] == "MIXED"
    assert "Senhales mixtas" in out["message"]


def test_detect_incluye_mte_si_presente():
    """mte_scenario anade modulo al total."""
    out = detect_cross_module_conflict(
        macro_regime="ESTRECHA",
        financial_regime="CRISIS",
        volatility_regime="STRESS",
        liquidity_regime="HIGH_STRESS",
        mte_scenario="CRISIS",
    )
    assert "Financial Stress: 5/5" in out["blocks"]
    assert "mte" in out["details"]


def test_detect_sin_mte_no_incluye_modulo():
    out = detect_cross_module_conflict(
        macro_regime="MIXED",
        financial_regime="NEUTRAL",
        volatility_regime="NORMAL",
        liquidity_regime="NEUTRAL",
    )
    assert "mte" not in out["details"]
    assert len(out["details"]) == 4


def test_detect_none_states_sesgo_cero():
    """None en los 4 -> todos sesgo 0 -> MIXED."""
    out = detect_cross_module_conflict(None, None, None, None)
    assert out["conflict_level"] == "MIXED"
    for name, d in out["details"].items():
        assert d["bias_financial"] == 0
        assert d["bias_inflation"] == 0


def test_detect_details_estructura():
    """Cada modulo tiene state + bias_financial + bias_inflation."""
    out = detect_cross_module_conflict(
        "RECESSION", "CRISIS", "STRESS", "CRISIS",
    )
    for name in ("macro", "financial", "volatility", "liquidity"):
        assert name in out["details"]
        d = out["details"][name]
        assert "state" in d
        assert "bias_financial" in d
        assert "bias_inflation" in d


# ---------- _confidence_range_row ----------
def test_confidence_range_row_valores_acordes():
    """Todos iguales -> rango 0 -> conf 1.0."""
    row = pd.Series([0.5, 0.5, 0.5, 0.5])
    assert _confidence_range_row(row) == 1.0


def test_confidence_range_row_valores_opuestos():
    """-1 y +1 -> rango 2.0 -> conf 0.0."""
    row = pd.Series([-1.0, 1.0])
    assert _confidence_range_row(row) == 0.0


def test_confidence_range_row_un_solo_valor():
    """<2 valores no NaN -> 0.5 (evidencia insuficiente)."""
    row = pd.Series([0.5])
    assert _confidence_range_row(row) == 0.5


def test_confidence_range_row_nan_ignorados():
    """NaN no cuentan como valores validos."""
    row = pd.Series([0.5, float("nan"), float("nan")])
    assert _confidence_range_row(row) == 0.5


def test_confidence_range_row_divisor_custom():
    """divisor=4.0 relaja la penalizacion."""
    row = pd.Series([-1.0, 1.0])
    # rango 2.0 / divisor 4.0 -> 0.5
    assert _confidence_range_row(row, divisor=4.0) == 0.5


def test_confidence_range_row_clip_negativo():
    """Rango > divisor -> clip a 0."""
    row = pd.Series([-5.0, 5.0])
    # rango 10 / divisor 2 = 5 -> 1-5 = -4 -> clip 0
    assert _confidence_range_row(row) == 0.0


# ---------- _try_cleanup ----------
def test_try_cleanup_borra_fichero(tmp_path):
    f = tmp_path / "temporal.txt"
    f.write_text("x")
    assert f.exists()
    _try_cleanup(str(f))
    assert not f.exists()


def test_try_cleanup_ignora_inexistente():
    """Path que no existe: no lanza."""
    _try_cleanup("/ruta/que/no/existe/xyz123.txt")


def test_try_cleanup_multiple_mixto(tmp_path):
    """Mezcla de existentes e inexistentes: no falla."""
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("a")
    b.write_text("b")
    _try_cleanup(str(a), "/no/existe/xyz.txt", str(b))
    assert not a.exists()
    assert not b.exists()


def test_try_cleanup_vacio():
    """Sin argumentos: no lanza."""
    _try_cleanup()


# ---------- confidence_from_range (DataFrame path) ----------
# D-07 (2026-10-03): bug de divergencia con _confidence_range_row.
# El chequeo previo era a nivel de columnas; por fila, <2 validos
# producia rng=0 -> conf=1.0 en lugar de 0.5.

def test_confidence_from_range_un_solo_valido_devuelve_05():
    """Fila con 1 componente valido -> 0.5 (no 1.0)."""
    df = pd.DataFrame({
        "a": [1.0, 0.8],
        "b": [np.nan, 0.6],
        "c": [np.nan, 0.4],
    })
    res = confidence_from_range(df)
    assert float(res.iloc[0]) == 0.5, f"esperado 0.5, obtenido {res.iloc[0]}"


def test_confidence_from_range_dos_validos_comportamiento_intacto():
    """Fila con 2+ validos: formula 1 - (max-min)/divisor, sin cambios."""
    df = pd.DataFrame({
        "a": [1.0, 0.5],
        "b": [-1.0, 0.5],
    })
    res = confidence_from_range(df, divisor=2.0)
    assert float(res.iloc[0]) == 0.0
    assert float(res.iloc[1]) == 1.0


def test_confidence_from_range_cero_validos_devuelve_05():
    """Fila con 0 validos -> 0.5."""
    df = pd.DataFrame({
        "a": [np.nan, 1.0],
        "b": [np.nan, 0.5],
    })
    res = confidence_from_range(df)
    assert float(res.iloc[0]) == 0.5


def test_confidence_from_range_y_scalar_coinciden():
    """Coherencia DataFrame path == scalar path para el mismo input.

    Evita que vuelva a existir divergencia semantica entre ambas
    implementaciones (D-07).
    """
    filas = [
        pd.Series({"a": 1.0, "b": 0.5, "c": 0.0}),
        pd.Series({"a": 1.0, "b": np.nan, "c": np.nan}),
        pd.Series({"a": np.nan, "b": np.nan, "c": np.nan}),
        pd.Series({"a": 1.0, "b": -1.0, "c": np.nan}),
    ]
    for i, row in enumerate(filas):
        df = row.to_frame().T
        res_df = float(confidence_from_range(df).iloc[0])
        res_scalar = float(_confidence_range_row(row))
        assert res_df == res_scalar, (
            f"fila {i}: DataFrame={res_df} != scalar={res_scalar}"
        )
