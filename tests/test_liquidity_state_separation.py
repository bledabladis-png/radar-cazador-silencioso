# -*- coding: utf-8 -*-
"""Test: v1 y v2 de liquidez escriben a ficheros de state distintos.

Item: REP-AUDIT-20261022-LIQUIDITY-STATE-SEPARATION
Causa: v1 (regimes/liquidity.py) y v2 (regimes/liquidity_v2.py)
escribian ambos a outputs/state/liquidity_state.json. Los tests de
v1 (test_regimes.py) contaminaban el state de produccion con esquema
v1 (sin regime, sin v2). Fix: v1 -> liquidity_v1_state.json.
"""


def test_v1_escribe_a_fichero_separado():
    """v1 debe escribir a liquidity_v1_state.json, no a liquidity_state.json."""
    from regimes import liquidity as v1_mod
    src = open(v1_mod.__file__, encoding="utf-8").read()
    assert "liquidity_v1_state.json" in src, "v1 no referencia su propio state"
    assert "delta_file = 'outputs/state/liquidity_state.json'" not in src, (
        "v1 sigue escribiendo al state de v2"
    )


def test_v2_escribe_a_fichero_original():
    """v2 debe seguir escribiendo a liquidity_state.json."""
    from regimes import liquidity_v2 as v2_mod
    src = open(v2_mod.__file__, encoding="utf-8").read()
    assert 'STATE_PATH = Path("outputs/state/liquidity_state.json")' in src


def test_no_colision_de_paths():
    """v2 no debe ESCRIBIR al state de v1 (mencionarlo en docstring es OK)."""
    from regimes import liquidity as v1_mod
    from regimes import liquidity_v2 as v2_mod
    src_v1 = open(v1_mod.__file__, encoding="utf-8").read()
    src_v2 = open(v2_mod.__file__, encoding="utf-8").read()
    assert "liquidity_v1_state.json" in src_v1, "v1 no asigna su state"
    forbidden_patterns = [
        'Path("outputs/state/liquidity_v1_state.json")',
        "Path('outputs/state/liquidity_v1_state.json')",
        'open("outputs/state/liquidity_v1_state.json"',
        "open('outputs/state/liquidity_v1_state.json'",
        '= "outputs/state/liquidity_v1_state.json"',
        "= 'outputs/state/liquidity_v1_state.json'",
    ]
    for pat in forbidden_patterns:
        assert pat not in src_v2, f"v2 usa el state de v1: {pat}"