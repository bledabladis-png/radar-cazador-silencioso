"""
run_radar.py - Script principal del Radar Macro Rotación Global
Ejecuta el pipeline completo:
- Carga datos más recientes (o descarga si es necesario)
- Calcula los cinco motores (régimen, liderazgo, geográfico, bonos, estrés)
- Obtiene score global, dispersión y régimen mediante scoring.py
- Aplica gestión de riesgo con risk.py
- Guarda resultados (logs JSON, CSV, Parquet)
- Detecta drift (opcional)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import os
import json
import yaml
import logging
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_layer import DataLayer
from src.regime_engine import RegimeEngine
from src.leadership_engine import LeadershipEngine
from src.geographic_engine import GeographicEngine
from src.bond_engine import BondEngine
from src.stress_engine import StressEngine
from src.scoring import ScoringEngine
from src.risk import RiskManager
from src.drift_handler import handle_drift

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("=" * 60)
    print("RADAR MACRO ROTACIÓN GLOBAL v2.3.1")
    print("=" * 60)

    # --- 1. Cargar datos ---
    print("\n[1] Cargando datos más recientes...")
    dl = DataLayer()
    try:
        df = dl.load_latest()
        logger.info("Datos cargados desde caché.")
    except FileNotFoundError:
        logger.warning("No hay datos previos. Descargando últimos 5 años...")
        end = datetime.now()
        start = end - pd.Timedelta(days=1825)
        df = dl.download_all(start_date=start, end_date=end, force=False)

    ultima_fecha = df.index[-1]
    print(f"    Última fecha disponible: {ultima_fecha.date()}")

    # --- 2. Ejecutar motores ---
    print("\n[2] Calculando motores...")
    regime = RegimeEngine()
    regime_df = regime.calcular_todo(df)
    print("    - Régimen: OK")

    leadership = LeadershipEngine()
    leadership_df = leadership.calcular_todo(df)
    print("    - Liderazgo: OK")

    geo = GeographicEngine()
    geo_df = geo.calcular_todo(df)
    print("    - Geográfico: OK")

    bond = BondEngine()
    bond_df = bond.calcular_todo(df)
    print("    - Bonos: OK")

    stress = StressEngine()
    stress_df = stress.calcular_todo(df)
    print("    - Estrés: OK")

    # --- 3. Scoring global ---
    print("\n[3] Calculando score global...")
    scoring = ScoringEngine()
    vix_series = df['^VIX'] if '^VIX' in df.columns else None
    resultados_df = scoring.calcular_todo(
        regime_df, leadership_df, geo_df, bond_df, stress_df, vix_series=vix_series
    )

    # Obtener último valor
    ultimo = resultados_df.iloc[-1]
    score_global = ultimo['score_global']
    score_smoothed = ultimo['score_smoothed']
    dispersion = ultimo['dispersion']
    regime_state = ultimo['regime_state']

    print(f"    Score global: {score_global:.4f}")
    print(f"    Score suavizado: {score_smoothed:.4f}")
    print(f"    Dispersión: {dispersion:.4f}")
    print(f"    Régimen: {regime_state}")

    # --- 4. Detección de drift (opcional) ---
    # Cargar historial de scores para detectar drift
    historial_path = 'historial_radar.csv'
    if os.path.exists(historial_path):
        df_hist = pd.read_csv(historial_path)
    else:
        df_hist = pd.DataFrame(columns=['score_global'])

    with open('config/config.yaml', 'r') as f:
        config_full = yaml.safe_load(f)

    contexto = {}  # solo para drift_handler
    contexto = handle_drift(df_hist, score_global, config_full, contexto)
    if contexto.get('operations_freeze', False):
        print("    ⚠️  Freeze activado por drift.")

    # Comprobar watchdog
    if os.path.exists('freeze_watchdog.txt'):
        contexto['operations_freeze'] = True
        print("    ⚠️  Freeze activado por watchdog.")

    # --- 5. Preparar contexto para RiskManager ---
    print("\n[4] Aplicando gestión de riesgo...")
    vix = df['^VIX'].iloc[-1] if '^VIX' in df.columns else 20.0
    if pd.isna(vix):
        vix = 20.0

    # Spreads simulados (o reales si se obtienen)
    spreads = {
        'SPY': 0.001,
        'EEM': 0.004,
        'JNK': 0.006,
        'LQD': 0.002,
        'IWM': 0.002,
        'XLY': 0.002,
        'XLP': 0.002,
        'EFA': 0.003,
        'ACWI': 0.002,
        'TLT': 0.001,
        'IBGL': 0.002,
        'QQQ': 0.001
    }

    # Exposición anterior
    exp_file = 'ultima_exposicion.txt'
    if os.path.exists(exp_file):
        try:
            with open(exp_file, 'r') as f:
                exp_prev = float(f.read().strip())
        except:
            exp_prev = 0.5
    else:
        exp_prev = 0.5

    capital = 100000  # capital fijo

    contexto_riesgo = {
        'vix': vix,
        'dispersion': dispersion,
        'spreads': spreads,
        'exp_prev': exp_prev,
        'capital': capital
    }

    # Si hay freeze, forzar exposición 0
    if contexto.get('operations_freeze', False):
        exp_final = 0.0
        penalizaciones = {'freeze': True}
        print("    Freeze activado. Exposición forzada a 0.")
    else:
        rm = RiskManager()
        exp_final, penalizaciones, log_riesgo = rm.aplicar_reglas_riesgo(score_global, contexto_riesgo)

    print(f"    Exposición final: {exp_final:.4f}")
    if not contexto.get('operations_freeze', False):
        print(f"    Factores aplicados:")
        print(f"      - VIX: {penalizaciones.get('vix_factor', 0):.3f}")
        print(f"      - Dispersión: {penalizaciones.get('dispersion_factor', 0):.3f}")
        print(f"      - Turnover: {penalizaciones.get('turnover', 0):.3f}")
        print(f"      - Comisión: {penalizaciones.get('comision', 0):.6f}")

    # Guardar exposición para el próximo día
    with open(exp_file, 'w') as f:
        f.write(str(exp_final))
    print(f"\n[5] Exposición guardada en {exp_file}")

    # --- 6. Guardar resultados ---
    print("\n[6] Guardando resultados...")
    # Guardar score global y métricas en CSV histórico
    nuevo_registro = pd.DataFrame({
        'fecha': [ultima_fecha.date()],
        'score_global': [score_global],
        'score_smoothed': [score_smoothed],
        'dispersion': [dispersion],
        'regime_state': [regime_state],
        'exposicion': [exp_final]
    })
    if os.path.exists(historial_path):
        historial = pd.read_csv(historial_path)
        historial = pd.concat([historial, nuevo_registro], ignore_index=True)
    else:
        historial = nuevo_registro
    historial.to_csv(historial_path, index=False)
    print(f"    Historial actualizado en {historial_path}")

    # Guardar log JSON
    os.makedirs('logs', exist_ok=True)
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "status": "success",
        "score_global": score_global,
        "score_smoothed": score_smoothed,
        "dispersion": dispersion,
        "regime_state": regime_state,
        "exposicion": exp_final,
        "freeze_activated": contexto.get('operations_freeze', False),
        "vix": vix,
        "modules": {
            "regime": "ok",
            "leadership": "ok",
            "geo": "ok",
            "bonds": "ok",
            "stress": "ok"
        },
        "errors": []
    }
    with open('logs/ultimo_log.json', 'w') as f:
        json.dump(log_entry, f, indent=2)
    log_filename = f"logs/radar_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_filename, 'w') as f:
        json.dump(log_entry, f, indent=2)
    print(f"    Log guardado en {log_filename}")

    # Guardar log de riesgo en Parquet (opcional)
    if not contexto.get('operations_freeze', False):
        try:
            os.makedirs('logs/exposicion', exist_ok=True)
            log_df = pd.DataFrame([log_riesgo])
            log_df.to_parquet(f'logs/exposicion/exposicion_{ultima_fecha.date()}.parquet', index=False)
            print(f"    Log de riesgo guardado en logs/exposicion/")
        except Exception as e:
            logger.warning(f"No se pudo guardar log de riesgo: {e}")

    # --- 7. Resumen final ---
    print("\n" + "=" * 60)
    print("RESUMEN EJECUTIVO")
    print("=" * 60)
    print(f"Fecha: {ultima_fecha.date()}")
    print(f"Score global: {score_global:.4f}")
    print(f"Score suavizado: {score_smoothed:.4f}")
    print(f"Dispersión: {dispersion:.4f}")
    print(f"Régimen: {regime_state}")
    print(f"Exposición recomendada: {exp_final:.2%}")
    print("=" * 60)

if __name__ == "__main__":
    main()