"""
run_radar.py - Script principal del Radar Macro Rotación Global (versión simplificada)
Ejecuta el pipeline completo:
- Carga datos más recientes
- Calcula los seis motores (régimen, liderazgo, geográfico, bonos, estrés, liquidez)
- Obtiene score global, exposure_factor y dispersión mediante scoring.py
- Aplica gestión de riesgo con risk.py
- Guarda resultados (logs JSON, CSV, Parquet)
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
from src.liquidity_engine import LiquidityEngine
from src.scoring import ScoringEngine
from src.risk import RiskManager

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("=" * 60)
    print("RADAR MACRO ROTACIÓN GLOBAL v2.3.1 (simplificado)")
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

    # --- 2. Ejecutar motores con manejo de errores ---
    print("\n[2] Calculando motores...")
    errores = []  # Lista para acumular errores

    # Régimen
    try:
        regime = RegimeEngine()
        regime_df = regime.calcular_todo(df)
        print("    - Régimen: OK")
    except Exception as e:
        logger.error(f"Error en Régimen: {e}")
        errores.append(f"regime: {str(e)}")
        regime_df = pd.DataFrame({'score_regime': 0}, index=df.index)
        print("    - Régimen: ERROR (usando 0)")

    # Liderazgo
    try:
        leadership = LeadershipEngine()
        leadership_df = leadership.calcular_todo(df)
        print("    - Liderazgo: OK")
    except Exception as e:
        logger.error(f"Error en Liderazgo: {e}")
        errores.append(f"leadership: {str(e)}")
        leadership_df = pd.DataFrame({'score_leadership': 0}, index=df.index)
        print("    - Liderazgo: ERROR (usando 0)")

    # Geográfico
    try:
        geo = GeographicEngine()
        geo_df = geo.calcular_todo(df)
        print("    - Geográfico: OK")
    except Exception as e:
        logger.error(f"Error en Geográfico: {e}")
        errores.append(f"geo: {str(e)}")
        geo_df = pd.DataFrame({'score_geographic': 0}, index=df.index)
        print("    - Geográfico: ERROR (usando 0)")

    # Bonos
    try:
        bond = BondEngine()
        bond_df = bond.calcular_todo(df)
        print("    - Bonos: OK")
    except Exception as e:
        logger.error(f"Error en Bonos: {e}")
        errores.append(f"bonds: {str(e)}")
        bond_df = pd.DataFrame({'score_bonds': 0}, index=df.index)
        print("    - Bonos: ERROR (usando 0)")

    # Estrés
    try:
        stress = StressEngine()
        stress_df = stress.calcular_todo(df)
        print("    - Estrés: OK")
    except Exception as e:
        logger.error(f"Error en Estrés: {e}")
        errores.append(f"stress: {str(e)}")
        stress_df = pd.DataFrame({'score_stress': 0}, index=df.index)
        print("    - Estrés: ERROR (usando 0)")

    # Liquidez
    try:
        liquidity = LiquidityEngine()
        liquidity_df = liquidity.calcular_todo(df)
        print("    - Liquidez: OK")
    except Exception as e:
        logger.error(f"Error en Liquidez: {e}")
        errores.append(f"liquidity: {str(e)}")
        liquidity_df = pd.DataFrame({'score_liquidity': 0}, index=df.index)
        print("    - Liquidez: ERROR (usando 0)")

    # --- 3. Scoring global ---
    print("\n[3] Calculando score global...")
    scoring = ScoringEngine()
    # Nota: ya no se pasa vix_series, el scoring simplificado no lo usa
    resultados_df = scoring.calcular_todo(regime_df, leadership_df, geo_df, bond_df, stress_df, liquidity_df)

    # Obtener último valor
    ultimo = resultados_df.iloc[-1]
    score_global = ultimo['score_global']
    score_smoothed = ultimo['score_smoothed']
    exposure_factor = ultimo['exposure_factor']
    dispersion = ultimo['dispersion']
    # penalty_disp y penalty_stress están disponibles si se quieren mostrar

    print(f"    Score global: {score_global:.4f}")
    print(f"    Score suavizado: {score_smoothed:.4f}")
    print(f"    Factor de exposición (de scoring): {exposure_factor:.4f}")
    print(f"    Dispersión: {dispersion:.4f}")

    # --- Comprobar watchdog (freeze) ---
    freeze_activado = False
    if os.path.exists('freeze_watchdog.txt'):
        freeze_activado = True
        print("    ⚠️  Freeze activado por watchdog.")

    # Definir el archivo de exposición
    exp_file = 'ultima_exposicion.txt' 

    # --- 4. Aplicar gestión de riesgo ---
    print("\n[4] Aplicando gestión de riesgo...")
    
    if freeze_activado:
        exp_final = 0.0
        penalizaciones = {'freeze': True}
        # Creamos un log_riesgo simple para que no falle al guardar Parquet
        log_riesgo = {
            'fecha': ultima_fecha.date(),
            'score': score_global,
            'exp_final': 0.0,
            'motivo': 'watchdog_freeze'
        }
        print("    Freeze activado. Exposición forzada a 0.")
        # También podemos asignar valores por defecto para vix, etc., aunque no se usen
        vix = 0.0
        vix_atr_ratio = 0.0
    else:
        # Calcular VIX y VIX/ATR para cash forzado
        vix = df['^VIX'].iloc[-1] if '^VIX' in df.columns else 20.0
        if pd.isna(vix):
            vix = 20.0
        
        # Calcular ATR20 de SPY si está disponible (para ratio VIX/ATR)
        if 'SPY_ATR20' in df.columns:
            atr20 = df['SPY_ATR20'].iloc[-1]
            if pd.notna(atr20) and atr20 > 0:
                vix_atr_ratio = vix / atr20
            else:
                vix_atr_ratio = vix / 0.5  # valor por defecto si ATR no es válido
        else:
            # Si no hay ATR, usamos solo VIX (o un ratio simulado)
            vix_atr_ratio = vix / 20.0  # aproximación burda
            logger.warning("SPY_ATR20 no encontrado, usando ratio aproximado VIX/20")
        
        # Obtener spreads reales del último día desde el DataFrame
        spreads = {}
        tickers_spread = ['SPY', 'EEM', 'JNK', 'LQD', 'IWM', 'XLY', 'XLP', 'EFA', 'ACWI', 'TLT', 'IBGL', 'QQQ']
        for ticker in tickers_spread:
            col = f"spread_{ticker}"
            if col in df.columns:
                # Tomamos el último valor no nulo
                ultimo_spread = df[col].iloc[-1]
                if pd.notna(ultimo_spread):
                    spreads[ticker] = ultimo_spread
                else:
                    # Si es NaN, usamos valor por defecto conservador
                    spreads[ticker] = 0.005
                    logger.warning(f"Spread de {ticker} no disponible, usando 0.005")
            else:
                # Si no existe la columna, usamos default
                spreads[ticker] = 0.005
                logger.warning(f"Columna {col} no encontrada, usando 0.005")

        # Exposición anterior        
        if os.path.exists(exp_file):
            try:
                with open(exp_file, 'r') as f:
                    exp_prev = float(f.read().strip())
            except:
                exp_prev = 0.5
        else:
            exp_prev = 0.5

        capital = 100000  # capital fijo (ajústalo según tu caso)

        contexto_riesgo = {
            'exposure_factor': exposure_factor,  # de scoring
            'vix_atr_ratio': vix_atr_ratio,
            'spreads': spreads,
            'exp_prev': exp_prev,
            'capital': capital
        }

        # Ejecutar risk manager
        rm = RiskManager()
        exp_final, penalizaciones, log_riesgo = rm.aplicar_reglas_riesgo(score_global, contexto_riesgo)

    print(f"    Exposición final: {exp_final:.4f}")
    print(f"    Factores aplicados:")
    # Mostrar factores solo si no hay freeze
    if not freeze_activado:
        print(f"      - Factor exposure (scoring): {penalizaciones.get('exposure_factor', 0):.3f}")
        print(f"      - Cash factor (VIX/ATR): {penalizaciones.get('cash_factor', 0):.3f}")
        print(f"      - Spreads: {penalizaciones.get('spread_factors', {})}")
        print(f"      - Turnover: {penalizaciones.get('turnover', 0):.3f}")
        print(f"      - Comisión: {penalizaciones.get('comision', 0):.6f}")
    else:
        print(f"      - Freeze activado")

    # Guardar exposición para el próximo día
    with open(exp_file, 'w') as f:
        f.write(str(exp_final))
    print(f"\n[5] Exposición guardada en {exp_file}")

    # --- 5. Guardar resultados con estructura organizada ---
    print("\n[6] Guardando resultados...")
    
    # Crear carpetas si no existen
    os.makedirs('logs/json', exist_ok=True)
    os.makedirs('logs/parquet', exist_ok=True)
    os.makedirs('logs/exposicion', exist_ok=True)

    # Obtener últimos valores de cada motor para guardarlos en el CSV
    ultimo_regime = regime_df['score_regime'].iloc[-1] if not regime_df.empty else 0.0
    ultimo_leadership = leadership_df['score_leadership'].iloc[-1] if not leadership_df.empty else 0.0
    ultimo_geo = geo_df['score_geographic'].iloc[-1] if not geo_df.empty else 0.0
    ultimo_bonds = bond_df['score_bonds'].iloc[-1] if not bond_df.empty else 0.0
    ultimo_stress = stress_df['score_stress'].iloc[-1] if not stress_df.empty else 0.0
    ultimo_liquidity = liquidity_df['score_liquidity'].iloc[-1] if not liquidity_df.empty else 0.0
    
    # Guardar score global y métricas en CSV histórico (se mantiene en raíz)
    historial_path = 'historial_radar.csv'
    nuevo_registro = pd.DataFrame({
        'fecha': [ultima_fecha.date()],
        'score_global': [score_global],
        'score_smoothed': [score_smoothed],
        'dispersion': [dispersion],
        'exposure_factor': [exposure_factor],
        'exposicion_final': [exp_final],
        'vix': [vix],
        'score_regime': [ultimo_regime],
        'score_leadership': [ultimo_leadership],
        'score_geographic': [ultimo_geo],
        'score_bonds': [ultimo_bonds],
        'score_stress': [ultimo_stress],
        'score_liquidity': [ultimo_liquidity]
    })
    if os.path.exists(historial_path):
        historial = pd.read_csv(historial_path)
        historial = pd.concat([historial, nuevo_registro], ignore_index=True)
    else:
        historial = nuevo_registro
    historial.to_csv(historial_path, index=False)
    print(f"    Historial actualizado en {historial_path}")

    # Preparar entrada JSON
    fecha_str = ultima_fecha.strftime('%Y%m%d')
    timestamp_utc = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    log_entry = {
        "timestamp": timestamp_utc,
        "status": "success" if not errores else "partial",
        "fecha": str(ultima_fecha.date()),
        "score_global": score_global,
        "score_smoothed": score_smoothed,
        "dispersion": dispersion,
        "exposure_factor": exposure_factor,
        "exposicion_final": exp_final,
        "vix": vix,
        "vix_atr_ratio": vix_atr_ratio,
        "penalizaciones": {
            "exposure_factor": penalizaciones.get('exposure_factor'),
            "cash_factor": penalizaciones.get('cash_factor'),
            "spread_factors": penalizaciones.get('spread_factors'),
            "turnover": penalizaciones.get('turnover'),
            "coste_turnover": penalizaciones.get('coste_turnover'),
            "comision": penalizaciones.get('comision')
        },
        "modules": {
            "regime": "ok" if 'regime' not in str(errores) else "error",
            "leadership": "ok" if 'leadership' not in str(errores) else "error",
            "geo": "ok" if 'geo' not in str(errores) else "error",
            "bonds": "ok" if 'bonds' not in str(errores) else "error",
            "stress": "ok" if 'stress' not in str(errores) else "error",
            "liquidity": "ok" if 'liquidity' not in str(errores) else "error"
        },
        "errors": errores
    }
    
    # Guardar último log (sobrescribe)
    with open('logs/json/ultimo_log.json', 'w') as f:
        json.dump(log_entry, f, indent=2)
    
    # Guardar log histórico con fecha
    log_filename = f"logs/json/radar_{fecha_str}.json"
    with open(log_filename, 'w') as f:
        json.dump(log_entry, f, indent=2)
    print(f"    Log JSON guardado en {log_filename}")

    # Guardar log de riesgo en Parquet (si existe)
    try:
        log_df = pd.DataFrame([log_riesgo])
        parquet_path = f"logs/parquet/exposicion_{fecha_str}.parquet"
        log_df.to_parquet(parquet_path, index=False)
        print(f"    Log de riesgo guardado en {parquet_path}")
    except Exception as e:
        logger.warning(f"No se pudo guardar log de riesgo: {e}")

    # Actualizar el archivo de última exposición (ya se hizo antes, pero lo dejamos)
    # (opcional, ya se guardó en ultima_exposicion.txt)

    # --- 6. Resumen final ---
    print("\n" + "=" * 60)
    print("RESUMEN EJECUTIVO")
    print("=" * 60)
    print(f"Fecha: {ultima_fecha.date()}")
    print(f"Score global: {score_global:.4f}")
    print(f"Score suavizado: {score_smoothed:.4f}")
    print(f"Dispersión: {dispersion:.4f}")
    print(f"Factor de exposición (scoring): {exposure_factor:.2%}")
    print(f"Exposición final (con riesgo): {exp_final:.2%}")
    print("=" * 60)

if __name__ == "__main__":
    main()