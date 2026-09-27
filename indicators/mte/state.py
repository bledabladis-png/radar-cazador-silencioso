"""Persistencia de estado MTE (DT2 Fase 3).

Extraccion literal de mte_legacy.py. Sin cambios funcionales:
- misma ruta (MTE_STATE_FILE)
- mismo esquema JSON
- mismo manejo de errores (except silencioso)
- mismo encoding (utf-8 en escritura)
- mismo formato (indent=2)

Unica diferencia: expuesto como modulo publico `indicators.mte.state`
para permitir monkeypatch del path en tests sin tocar produccion.
"""
from __future__ import annotations

import json
import os

from config.settings import MTE_STATE_FILE, CURRENT_TEMPORAL_CONTRACT_VERSION


def load_previous_scenario():
    """Lee scenario, pending y flag temporal_reset del state previo.

    Retorna (scenario, pending, temporal_reset):
      - (MIXED, None, True)   si la version del contrato no coincide
      - (scenario, pending, False) si el fichero es valido y compatible
      - (MIXED, None, False)  si no existe o hay error de lectura

    F2.4-12 (ii): temporal_reset=True invalida el estado previo
    (prev=MIXED, pending=None) pero NO invalida el calculo MTE del
    run actual. El writer final (engine.py) persiste el scenario
    calculado normalmente y actualiza temporal_contract_version.
    """
    try:
        with open(MTE_STATE_FILE, 'r') as f:
            data = json.load(f)
            _stored_version = data.get('temporal_contract_version')
            if _stored_version != CURRENT_TEMPORAL_CONTRACT_VERSION:
                print(f'  MTE state: contrato cambio ({_stored_version} -> {CURRENT_TEMPORAL_CONTRACT_VERSION}). Reset.')
                return 'MIXED', None, True
            return data.get('scenario', 'MIXED'), data.get('pending', None), False
    except Exception:
        return 'MIXED', None, False


def save_scenario(scenario, pending=None):
    """Persiste scenario + pending en MTE_STATE_FILE con schema minimo."""
    os.makedirs(os.path.dirname(MTE_STATE_FILE), exist_ok=True)
    with open(MTE_STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            'schema_version': 1,
            'temporal_contract_version': CURRENT_TEMPORAL_CONTRACT_VERSION,
            'scenario': scenario,
            'pending': pending,
        }, f, indent=2)