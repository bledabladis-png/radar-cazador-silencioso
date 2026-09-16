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
    """Lee scenario y pending del state previo. Reset a MIXED si version no cuadra."""
    try:
        with open(MTE_STATE_FILE, 'r') as f:
            data = json.load(f)
            _stored_version = data.get('temporal_contract_version')
            if _stored_version != CURRENT_TEMPORAL_CONTRACT_VERSION:
                print(f'  MTE state: contrato cambio ({_stored_version} -> {CURRENT_TEMPORAL_CONTRACT_VERSION}). Reset.')
                return 'MIXED', None
            return data.get('scenario', 'MIXED'), data.get('pending', None)
    except Exception:
        return 'MIXED', None


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