"""
watchdog.py - Vigilante del Radar Macro Rotación Global
Se ejecuta periódicamente (cada hora) para comprobar que el radar se ha ejecutado correctamente.
Si la última ejecución falla o es muy antigua, activa un freeze.
"""

import os
import json
from datetime import datetime, timedelta, timezone

def leer_ultimo_log():
    """Lee el archivo ultimo_log.json de la carpeta logs."""
    log_path = 'logs/json/ultimo_log.json'
    if not os.path.exists(log_path):
        return None
    with open(log_path, 'r') as f:
        return json.load(f)

def verificar_ejecucion(log):
    """
    Verifica que el log no sea None, que el timestamp sea reciente (<24h)
    y que el estado sea 'success'.
    Devuelve (ok, mensaje).
    """
    if log is None:
        return False, "No hay ningún log"

    # Obtener timestamp del log
    try:
        # El timestamp viene con formato '2025-03-05T01:01:01Z'
        # Lo convertimos a objeto datetime (asumiendo UTC)
        ts_str = log['timestamp'].replace('Z', '+00:00')
        ts = datetime.fromisoformat(ts_str)
    except Exception as e:
        return False, f"Formato de timestamp inválido: {e}"

    # Calcular diferencia con la hora actual en UTC
    ahora = datetime.now(timezone.utc)
    diferencia = ahora - ts

    if diferencia > timedelta(hours=24):
        return False, f"Última ejecución hace más de 24h: {ts}"

    if log.get('status') != 'success':
        return False, f"Estado no exitoso: {log.get('status')}"

    return True, "OK"

def activar_freeze(razon):
    """
    Crea el archivo freeze_watchdog.txt y escribe una alerta en alertas.txt.
    """
    with open('freeze_watchdog.txt', 'w') as f:
        f.write(f"FREEZE por watchdog: {razon}\n")

    with open('alertas.txt', 'a') as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} - ALERTA: {razon}\n")

    print(f"ALERTA: {razon}")

def main():
    log = leer_ultimo_log()
    ok, msg = verificar_ejecucion(log)
    if not ok:
        activar_freeze(msg)
    else:
        # Si todo está bien, eliminamos el freeze anterior si existe
        if os.path.exists('freeze_watchdog.txt'):
            os.remove('freeze_watchdog.txt')
        print("Watchdog: todo correcto")

if __name__ == "__main__":
    main()