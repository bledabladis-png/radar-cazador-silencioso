# -*- coding: utf-8 -*-
"""
state_transition.py -- Histéresis temporal para SLPM v1.2
Añade persistencia temporal a los estados sin modificar state_machine.py.
"""
import json
import os
from datetime import datetime

SLPM_STATE_FILE = "outputs/slpm_state.json"

HYSTERESIS = {
    ("EMERGING", "CONFIRMED"): ("EMERGING", "EMERGING"),
    ("CONFIRMED", "EMERGING"): ("CONFIRMED", "CONFIRMED"),
    ("CONFIRMED", "TACTICAL_CORRECTION"): ("CONFIRMED", "CONFIRMED"),
}

def load_previous_state():
    if not os.path.exists(SLPM_STATE_FILE):
        return None, 0
    try:
        with open(SLPM_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("state"), data.get("consecutive_count", 0)
    except:
        return None, 0

def save_current_state(state, count):
    os.makedirs(os.path.dirname(SLPM_STATE_FILE), exist_ok=True)
    with open(SLPM_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "state": state,
            "consecutive_count": count,
            "last_updated": datetime.now().isoformat()
        }, f, indent=2)

def confirm_transition(instant_state):
    previous_state, count = load_previous_state()

    if previous_state is None:
        save_current_state(instant_state, 1)
        return {
            "confirmed_state": instant_state,
            "previous_state": None,
            "transition": f"INITIAL_{instant_state}",
            "consecutive_count": 1
        }

    rule = HYSTERESIS.get((previous_state, instant_state))
    if rule:
        if count >= 2:
            save_current_state(instant_state, 1)
            return {
                "confirmed_state": instant_state,
                "previous_state": previous_state,
                "transition": f"{previous_state}_TO_{instant_state}",
                "consecutive_count": 1
            }
        else:
            save_current_state(previous_state, count + 1)
            return {
                "confirmed_state": previous_state,
                "previous_state": previous_state,
                "transition": f"HOLDING_{previous_state}_CANDIDATE_{instant_state}",
                "consecutive_count": count + 1
            }
    else:
        if instant_state != previous_state:
            save_current_state(instant_state, 1)
            return {
                "confirmed_state": instant_state,
                "previous_state": previous_state,
                "transition": f"{previous_state}_TO_{instant_state}",
                "consecutive_count": 1
            }
        else:
            save_current_state(instant_state, count + 1)
            return {
                "confirmed_state": instant_state,
                "previous_state": previous_state,
                "transition": f"STABLE_{instant_state}",
                "consecutive_count": count + 1
            }
