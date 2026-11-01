# -*- coding: utf-8 -*-
"""C.3 (2026-11-01): registro declarativo de variables compartidas.

Algunas variables se usan en varios modulos del sistema. Cuando varios
modulos generan senales a partir de la misma variable, esas senales NO
son confirmaciones independientes del mismo riesgo. Este registro
declara que variable se usa en que modulos, para que el reporte pueda
avisar al lector.

Contrato: la asociacion es informacion de diseno. Si un refactor
elimina el uso de una variable en un modulo, el test AST
tests/test_shared_inputs.py falla y fuerza actualizar este registro.

Verificado 2026-11-01 (Gate 0): solo se listan variables con >=2
modulos y uso comprobado por grep en el codigo productivo. WALCL y RRP
se excluyen: solo aparecen en fls.py (no compartidas en el sistema
actual). VIX, HYG/LQD, credit_signal, SOFR, rs_mom, flow_proxy_z
estan todos en >=2 modulos productivos.

Sin hardcode en el render: _shared_vars_note(module) lee este dict y
genera el texto en tiempo de ejecucion.
"""

SHARED_INPUTS = {
    "VIX": [
        "financial_conditions",  # regimes/financial_conditions.py
        "volatility_regime",     # regimes/volatility_regime.py
        "macro_regime",          # regimes/macro_regime.py
        "engine",                # indicators/mte/engine.py
    ],
    "HYG/LQD": [
        "financial_conditions",
        "cross_asset",           # indicators/cross_asset.py
        "macro_regime",
    ],
    "credit_signal": [
        "financial_conditions",
        "evidence_matrix",       # indicators/evidence_matrix.py
        "macro_regime",
    ],
    "SOFR": [
        "liquidity_v2",          # regimes/liquidity_v2.py
        "fls",                   # indicators/fls.py
    ],
    "rs_mom": [
        "stock_metrics",         # indicators/stock_metrics.py
        "sector_concentration",  # indicators/sector_concentration.py
    ],
    "flow_proxy_z": [
        "stock_metrics",
        "sector_concentration",
    ],
}


def shared_with(variable):
    """Modulos que usan `variable`, o [] si no esta en el registro."""
    return list(SHARED_INPUTS.get(variable, []))


def variables_in(module):
    """Variables que usa `module`, o [] si no aparece."""
    return [var for var, mods in SHARED_INPUTS.items() if module in mods]
