# -*- coding: utf-8 -*-
"""state_transition.py -- Histeresis temporal para SLPM v1.2.

Aplica histeresis sobre las transiciones de estado del SLPM v1.2. Persiste
el estado en disco para que sobreviva entre runs del pipeline.

CONTEXTO DE AUDITORIA (2026-09-27)
----------------------------------
Reemplaza la version anterior. Hallazgos cubiertos:

  F2-19: el state file NO estaba particionado por sector. Si el sector
         lider rotaba, el nuevo sector heredaba el consecutive_count
         del anterior. Ahora el state file incluye `sector_etf`; si el
         sector cambia, se resetea.

  F2-21: consecutive_count media "runs consecutivos del estado previo",
         no del candidato. Un candidato aislado se aceptaba si el estado
         previo llevaba >=2 runs. Ahora hay un candidato explicito
         (candidate_state + candidate_count) que debe aparecer N veces
         consecutivas antes de confirmar la transicion.

  F2-13: los valores de HYSTERESIS eran dead code. Ahora HYSTERESIS es
         un set de pares.

SCHEMA v2
---------
{
  "schema_version": 2,
  "sector_etf": "XLK",
  "state": "CONFIRMED",
  "consecutive_count": 5,
  "candidate_state": null | "EMERGING",
  "candidate_count": 0,
  "last_updated": "2026-09-27T..."
}

Compatibilidad: un state file v1 (sin schema_version / sin sector_etf)
se trata como reset. Un fichero corrupto tambien.

CONFIRMACION DE TRANSICION
--------------------------
Una transicion (previous -> instant) se confirma cuando:

  - (previous, instant) NO esta en HYSTERESIS -> inmediata.
  - (previous, instant) SI esta en HYSTERESIS -> requiere que instant
    aparezca CANDIDATE_THRESHOLD (2) runs consecutivos.

Si instant == previous: incrementa consecutive_count y limpia candidato.
"""
from __future__ import annotations

import json
import os
from datetime import datetime


SLPM_STATE_FILE = "outputs/state/slpm_state.json"

SCHEMA_VERSION = 2
CANDIDATE_THRESHOLD = 2

# Pares (previous, candidate) que requieren histeresis.
# Sets en lugar de dict: los valores del dict anterior eran dead code (F2-13).
#
# A3.3-05 (2026-09-28): ("CONFIRMED","TACTICAL_CORRECTION") es
# inalcanzable en produccion: classify_leadership_state (state_machine.py)
# solo produce CONFIRMED, EMERGING, STRUCTURAL_DECAY, LOST, UNRESOLVED.
# TACTICAL_CORRECTION no se genera en ningun caso.
# WONT FIX razonado: (a) se conserva como defensa en profundidad para
# extension futura de la state machine; (b) test_slpm_state_transition.py
# lo ejercita como parte del contrato de confirm_transition; (c) borrarlo
# obligaria a modificar el test sin ganancia funcional.
HYSTERESIS = {
    ("EMERGING", "CONFIRMED"),
    ("CONFIRMED", "EMERGING"),
    ("CONFIRMED", "TACTICAL_CORRECTION"),
}


def _empty_state() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "sector_etf": None,
        "state": None,
        "consecutive_count": 0,
        "candidate_state": None,
        "candidate_count": 0,
        "last_updated": None,
    }


def _load_state() -> dict:
    """Carga el state file. Devuelve _empty_state() si no existe o es invalido."""
    if not os.path.exists(SLPM_STATE_FILE):
        return _empty_state()
    try:
        with open(SLPM_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError, ValueError):
        # A3.3-11: fichero corrupto / ilegible / JSON invalido -> reset.
        return _empty_state()
    if not isinstance(data, dict):
        return _empty_state()
    # Schema v1 (sin schema_version) se trata como reset.
    if data.get("schema_version") != SCHEMA_VERSION:
        return _empty_state()
    return data


def _save_state(payload: dict) -> None:
    """Persiste el state file. Rellena last_updated."""
    os.makedirs(os.path.dirname(SLPM_STATE_FILE), exist_ok=True)
    payload = dict(payload)
    payload["schema_version"] = SCHEMA_VERSION
    payload["last_updated"] = datetime.now().isoformat()
    # Fix 2026-09-29: escritura atomica (.tmp + os.replace). Sin esto,
    # si el proceso muere a mitad de json.dump, el state file queda
    # truncado. _load_state lo detecta como corrupto (except
    # JSONDecodeError, A3.3-11) y RESETEA el state a vacio. La maquina
    # de estados SLPM pierde consecutive_count y confirmed_state
    # silenciosamente. Mismo patron que downloader.py y storage.py.
    _tmp = SLPM_STATE_FILE + ".tmp"
    with open(_tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)
    os.replace(_tmp, SLPM_STATE_FILE)


def confirm_transition(sector_etf: str, instant_state: str) -> dict:
    """Aplica histeresis a la transicion (state_prev -> instant_state).

    Args:
        sector_etf: ticker del sector lider actual (p.ej. "XLK"). Si
                    difiere del guardado en disco, resetea el historial.
        instant_state: estado instantaneo clasificado por state_machine.

    Returns:
        dict con confirmed_state, previous_state, transition,
        consecutive_count.
    """
    if not sector_etf:
        raise ValueError("confirm_transition requiere sector_etf no vacio.")

    prev = _load_state()

    # Reset si: no hay estado, cambia el sector, o el schema era v1/corrupto.
    prev_state = prev.get("state")
    prev_sector = prev.get("sector_etf")
    if prev_state is None or prev_sector != sector_etf:
        _save_state({
            "sector_etf": sector_etf,
            "state": instant_state,
            "consecutive_count": 1,
            "candidate_state": None,
            "candidate_count": 0,
        })
        return {
            "confirmed_state": instant_state,
            "previous_state": None,
            "transition": f"INITIAL_{instant_state}",
            "consecutive_count": 1,
        }

    prev_count = int(prev.get("consecutive_count", 0))
    cand_state = prev.get("candidate_state")
    cand_count = int(prev.get("candidate_count", 0))

    # Caso 1: mismo estado que el confirmado -> estable, limpia candidato.
    if instant_state == prev_state:
        new_count = prev_count + 1
        _save_state({
            "sector_etf": sector_etf,
            "state": prev_state,
            "consecutive_count": new_count,
            "candidate_state": None,
            "candidate_count": 0,
        })
        return {
            "confirmed_state": prev_state,
            "previous_state": prev_state,
            "transition": f"STABLE_{prev_state}",
            "consecutive_count": new_count,
        }

    # Caso 2: transicion sin histeresis -> inmediata.
    if (prev_state, instant_state) not in HYSTERESIS:
        _save_state({
            "sector_etf": sector_etf,
            "state": instant_state,
            "consecutive_count": 1,
            "candidate_state": None,
            "candidate_count": 0,
        })
        return {
            "confirmed_state": instant_state,
            "previous_state": prev_state,
            "transition": f"{prev_state}_TO_{instant_state}",
            "consecutive_count": 1,
        }

    # Caso 3: transicion con histeresis -> requiere CANDIDATE_THRESHOLD runs.
    if cand_state == instant_state:
        new_cand_count = cand_count + 1
    else:
        # Nuevo candidato (o cambio de candidato) -> arranca contador en 1.
        new_cand_count = 1

    if new_cand_count >= CANDIDATE_THRESHOLD:
        # Confirmar transicion.
        _save_state({
            "sector_etf": sector_etf,
            "state": instant_state,
            "consecutive_count": 1,
            "candidate_state": None,
            "candidate_count": 0,
        })
        return {
            "confirmed_state": instant_state,
            "previous_state": prev_state,
            "transition": f"{prev_state}_TO_{instant_state}",
            "consecutive_count": 1,
        }

    # Retener estado previo, registrar candidato.
    _save_state({
        "sector_etf": sector_etf,
        "state": prev_state,
        "consecutive_count": prev_count,
        "candidate_state": instant_state,
        "candidate_count": new_cand_count,
    })
    return {
        "confirmed_state": prev_state,
        "previous_state": prev_state,
        "transition": f"HOLDING_{prev_state}_CANDIDATE_{instant_state}",
        "consecutive_count": prev_count,
    }
