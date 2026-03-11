"""
run_radar.py - Script principal del Radar Macro Rotación Global (versión v3.0)
Ejecuta el pipeline completo:
- Carga datos más recientes
- Calcula los siete motores (régimen, liderazgo, geográfico, bonos, estrés, liquidez, breadth)
- Obtiene score global, exposure_factor y dispersión mediante scoring.py
- Aplica gestión de riesgo con risk.py
- Guarda resultados (logs JSON, CSV, Parquet)
- Incluye nuevos motores V7: riesgo sistémico y carry trade
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import os
import json
import yaml
import logging
import sys

# Configurar logger
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.financial_conditions_engine import FinancialConditionsEngine
from src.etf_flow_engine_twelve import EtfFlowEngineTwelve as EtfFlowEngine
from src.breadth_advanced_engine import BreadthAdvancedEngine
from src.cftc_engine import CftcEngine
from src.global_liquidity_engine import GlobalLiquidityEngine
from src.macro_engines_v7 import riesgo_sistemico, carry_trade
from src.breadth_engine import BreadthEngine
from src.data_layer import DataLayer
from src.regime_engine import RegimeEngine
from src.leadership_engine import LeadershipEngine
from src.geographic_engine import GeographicEngine
from src.bond_engine import BondEngine
from src.stress_engine import StressEngine
from src.liquidity_engine import LiquidityEngine
from src.scoring import ScoringEngine
from src.risk import RiskManager

# Configurar logging a archivo (nada en consola)
logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('radar.log', mode='a')])
# Silenciar logs de librerías externas
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('requests').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

def main():
    print("=" * 60)
    print("RADAR MACRO ROTACIÓN GLOBAL v3.0")
    print("=" * 60)

    # --- 1. Cargar datos ---
    print("\n[1] Cargando datos más recientes...")
    dl = DataLayer()
    # Asegurar datos BIS
    try:
        bis_dl = DataLayer()
        bis_dl.download_bis_data(force=False)
    except Exception as e:
        logger.warning(f"No se pudieron descargar datos BIS: {e}")
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

    
    # Asegurar que existan las columnas de spread y ATR20
    tickers_spread = ['SPY', 'EEM', 'JNK', 'LQD', 'IWM', 'XLY', 'XLP', 'EFA', 'ACWI', 'TLT', 'IBGL', 'QQQ']
    for ticker in tickers_spread:
        col = f"spread_{ticker}"
        if col not in df.columns:
            df[col] = np.nan
            logger.debug(f"Columna {col} añadida con NaN")
    
    if 'SPY_ATR20' not in df.columns:
        df['SPY_ATR20'] = np.nan
        logger.debug("Columna SPY_ATR20 añadida con NaN")# --- 2. Ejecutar motores con manejo de errores ---
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

    # Breadth
    try:
        breadth = BreadthEngine()
        breadth_df = breadth.calcular_todo(df)
        print("    - Breadth: OK")
    except Exception as e:
        logger.error(f"Error en Breadth: {e}")
        errores.append(f"breadth: {str(e)}")
        breadth_df = pd.DataFrame({'score_breadth': 0}, index=df.index)
        print("    - Breadth: ERROR (usando 0)")

    # --- Nuevos motores V7 ---
    # Inicializar variables para motores avanzados
    ultimo_gli = 0.0
    ultimo_cftc = 0.0
    ultimo_breadth_adv = 0.0
    ultimo_fc = 0.0
    ultimo_etf = 0.0

    try:
        riesgo_series = riesgo_sistemico(df)
        riesgo_df = pd.DataFrame({'score_riesgo_sistemico': riesgo_series}, index=df.index)
        print("    - Riesgo Sistémico: OK")
    except Exception as e:
        logger.error(f"Error en Riesgo Sistémico: {e}")
        errores.append(f"riesgo_sistemico: {str(e)}")
        riesgo_df = pd.DataFrame({'score_riesgo_sistemico': 0}, index=df.index)
        print("    - Riesgo Sistémico: ERROR (usando 0)")

    try:
        carry_series = carry_trade(df)
        carry_df = pd.DataFrame({'score_carry': carry_series}, index=df.index)
        print("    - Carry Trade: OK")
    except Exception as e:
        logger.error(f"Error en Carry Trade: {e}")
        errores.append(f"carry: {str(e)}")
        carry_df = pd.DataFrame({'score_carry': 0}, index=df.index)
        print("    - Carry Trade: ERROR (usando 0)")

    # --- 3. Scoring global ---
    print("\n[3] Calculando score global...")
    scoring = ScoringEngine()
    resultados_df = scoring.calcular_todo(regime_df, leadership_df, geo_df, bond_df, stress_df, liquidity_df, breadth_df)

    # Añadir nuevos motores V7 al DataFrame de resultados (alineando índices)
    resultados_df['score_riesgo_sistemico'] = riesgo_series.reindex(resultados_df.index, method='ffill')
    resultados_df['score_carry'] = carry_series.reindex(resultados_df.index, method='ffill')

    # --- Motor de liquidez global (BIS) ---
    try:
        gli_engine = GlobalLiquidityEngine()
        gli_df = gli_engine.calcular_todo()
        gli_series = gli_df['score_global_liquidity'].reindex(resultados_df.index, method='ffill')
        resultados_df['score_global_liquidity'] = gli_series
        ultimo_gli = gli_series.iloc[-1] if not gli_series.empty else 0.0
        print("    - Liquidez Global (BIS): OK")
    except Exception as e:
        logger.error(f"Error en Liquidez Global: {e}")
        errores.append(f"global_liquidity: {str(e)}")
        resultados_df['score_global_liquidity'] = 0.0
        print("    - Liquidez Global: ERROR (usando 0)")

    # --- Motor de posicionamiento CFTC ---
    try:
        cftc_engine = CftcEngine()
        cftc_df = cftc_engine.calcular_todo()
        cftc_series = cftc_df['score_cftc'].reindex(resultados_df.index, method='ffill')
        resultados_df['score_cftc'] = cftc_series
        ultimo_cftc = cftc_series.iloc[-1] if not cftc_series.empty else 0.0
        print("    - CFTC Positioning: OK")
    except Exception as e:
        logger.error(f"Error en CFTC: {e}")
        errores.append(f"cftc: {str(e)}")
        resultados_df['score_cftc'] = 0.0
        print("    - CFTC Positioning: ERROR (usando 0)")

    # --- Motor de Breadth Avanzado ---
    try:
        breadth_adv_engine = BreadthAdvancedEngine()
        breadth_adv_df = breadth_adv_engine.calcular_todo(df_principal=df)
        breadth_adv_series = breadth_adv_df['score_breadth_advanced'].reindex(resultados_df.index, method='ffill')
        resultados_df['score_breadth_advanced'] = breadth_adv_series
        ultimo_breadth_adv = breadth_adv_series.iloc[-1] if not breadth_adv_series.empty else 0.0
        print("    - Breadth Avanzado: OK")
    except Exception as e:
        logger.error(f"Error en Breadth Avanzado: {e}")
        errores.append(f"breadth_adv: {str(e)}")
        resultados_df['score_breadth_advanced'] = 0.0
        print("    - Breadth Avanzado: ERROR (usando 0)")

    # --- Motor de Flujos de ETFs ---
    try:
        etf_engine = EtfFlowEngine()
        etf_df = etf_engine.calcular_todo()
        etf_series = etf_df['score_etf_flow'].reindex(resultados_df.index, method='ffill')
        resultados_df['score_etf_flow'] = etf_series
        ultimo_etf = etf_series.iloc[-1] if not etf_series.empty else 0.0
        print("    - ETF Flows: OK")
    except Exception as e:
        logger.error(f"Error en ETF Flows: {e}")
        errores.append(f"etf_flow: {str(e)}")
        resultados_df['score_etf_flow'] = 0.0
        print("    - ETF Flows: ERROR (usando 0)")

    # --- Motor de Condiciones Financieras ---
    try:
        fc_engine = FinancialConditionsEngine()
        # Usar liquidity_df directamente (ya calculado antes)
        liquidity_series = liquidity_df['score_liquidity'] if not liquidity_df.empty else pd.Series(0, index=df.index)
        fc_df = fc_engine.calcular_todo(df, liquidity_series)
        fc_series = fc_df['score_financial_conditions'].reindex(resultados_df.index, method='ffill')
        resultados_df['score_financial_conditions'] = fc_series
        ultimo_fc = fc_series.iloc[-1] if not fc_series.empty else 0.0
        print("    - Condiciones Financieras: OK")
    except Exception as e:
        logger.error(f"Error en Condiciones Financieras: {e}")
        errores.append(f"financial_conditions: {str(e)}")
        resultados_df['score_financial_conditions'] = 0.0
        print("    - Condiciones Financieras: ERROR (usando 0)")

    # Obtener último valor
    ultimo = resultados_df.iloc[-1]
    # Fallback para score_smoothed por si acaso
    if 'score_smoothed' in ultimo:
        score_smoothed = ultimo['score_smoothed']
    else:
        score_smoothed = ultimo['score_global']
    pend_3d = ultimo.get('pend_3d', 0.0)
    pend_5d = ultimo.get('pend_5d', 0.0)
    pend_10d = ultimo.get('pend_10d', 0.0)
    aceleracion = ultimo.get('aceleracion', 0.0)
    motores_mejorando = ultimo.get('motores_mejorando', 0)
    fase_ciclo = ultimo.get('fase_ciclo', 'NEUTRAL')
    flujo_puro = ultimo.get('flujo_puro', 0.0)
    score_global = ultimo['score_global']
    if 'score_smoothed' in ultimo:
        score_smoothed = ultimo['score_smoothed']
    else:
        score_smoothed = ultimo['score_global']
    exposure_factor = ultimo['exposure_factor']
    dispersion = ultimo['dispersion']
    ultimo_breadth = ultimo.get('score_breadth', 0.0)
    # Nuevos motores V7 (último valor)
    ultimo_riesgo = resultados_df['score_riesgo_sistemico'].iloc[-1] if 'score_riesgo_sistemico' in resultados_df.columns else 0.0
    ultimo_carry = resultados_df['score_carry'].iloc[-1] if 'score_carry' in resultados_df.columns else 0.0
    ciclo_inst = ultimo.get('ciclo_institucional', 'NEUTRAL')

    print(f"    Score global: {score_global:.4f}")
    print(f"    Score suavizado: {score_smoothed:.4f}")
    print(f"    Factor de exposición (de scoring): {exposure_factor:.4f}")
    print(f"    Dispersión: {dispersion:.4f}")
    print(f"    Fase del ciclo: {fase_ciclo}")
    print(f"    Pendientes (3d,5d,10d): {pend_3d:.4f}, {pend_5d:.4f}, {pend_10d:.4f}")
    print(f"    Aceleración: {aceleracion:.4f}")
    print(f"    Motores mejorando: {motores_mejorando}")
    print(f"    Riesgo Sistémico: {ultimo_riesgo:.4f}")
    print(f"    Carry Trade: {ultimo_carry:.4f}")
    print(f"    Ciclo Institucional: {ciclo_inst}")

    # --- Comprobar watchdog (freeze) ---
    freeze_activado = False
    if os.path.exists('freeze_watchdog.txt'):
        freeze_activado = True
        print("    ⚠️  Freeze activado por watchdog.")

    exp_file = 'ultima_exposicion.txt'

    # --- 4. Aplicar gestión de riesgo ---
    print("\n[4] Aplicando gestión de riesgo...")

    if freeze_activado:
        exp_final = 0.0
        penalizaciones = {'freeze': True}
        log_riesgo = {
            'fecha': ultima_fecha.date(),
            'score': score_global,
            'exp_final': 0.0,
            'motivo': 'watchdog_freeze'
        }
        print("    Freeze activado. Exposición forzada a 0.")
        vix = 0.0
        vix_atr_ratio = 0.0
    else:
        vix = df['^VIX'].iloc[-1] if '^VIX' in df.columns else 20.0
        if pd.isna(vix):
            vix = 20.0

        if 'SPY_ATR20' in df.columns:
            atr20 = df['SPY_ATR20'].iloc[-1]
            if pd.notna(atr20) and atr20 > 0:
                vix_atr_ratio = vix / atr20
            else:
                vix_atr_ratio = vix / 0.5
        else:
            vix_atr_ratio = vix / 20.0
            logger.warning("SPY_ATR20 no encontrado, usando ratio aproximado VIX/20")

        spreads = {}
        tickers_spread = ['SPY', 'EEM', 'JNK', 'LQD', 'IWM', 'XLY', 'XLP', 'EFA', 'ACWI', 'TLT', 'IBGL', 'QQQ']
        for ticker in tickers_spread:
            col = f"spread_{ticker}"
            if col in df.columns:
                ultimo_spread = df[col].iloc[-1]
                if pd.notna(ultimo_spread):
                    spreads[ticker] = ultimo_spread
                else:
                    spreads[ticker] = 0.005
                    logger.warning(f"Spread de {ticker} no disponible, usando 0.005")
            else:
                spreads[ticker] = 0.005
                logger.warning(f"Columna {col} no encontrada, usando 0.005")

        if os.path.exists(exp_file):
            try:
                with open(exp_file, 'r') as f:
                    exp_prev = float(f.read().strip())
            except:
                exp_prev = 0.5
        else:
            exp_prev = 0.5

        capital = 100000

        volatilidad_spy = None
        if 'SPY' in df.columns:
            ret_spy = df['SPY'].pct_change(fill_method=None).dropna()
            if len(ret_spy) >= 20:
                vol_diaria = ret_spy.tail(20).std()
                volatilidad_spy = vol_diaria * np.sqrt(252)
                logger.info(f"Volatilidad SPY (anualizada): {volatilidad_spy:.2%}")
            else:
                logger.warning("No hay suficientes datos para calcular volatilidad de SPY")
        else:
            logger.warning("Columna SPY no encontrada, no se aplicará volatility targeting")

        contexto_riesgo = {
            'exposure_factor': exposure_factor,
            'vix_atr_ratio': vix_atr_ratio,
            'spreads': spreads,
            'exp_prev': exp_prev,
            'capital': capital,
            'breadth_score': ultimo_breadth,
            'volatilidad_spy': volatilidad_spy
        }

        rm = RiskManager()
        exp_final, penalizaciones, log_riesgo = rm.aplicar_reglas_riesgo(score_global, contexto_riesgo)

    print(f"    Exposición final: {exp_final:.4f}")
    print(f"    Factores aplicados:")
    if not freeze_activado:
        print(f"      - Factor exposure (scoring): {penalizaciones.get('exposure_factor', 0):.3f}")
        print(f"      - Cash factor (VIX/ATR): {penalizaciones.get('cash_factor', 0):.3f}")
        print(f"      - Factor volatilidad: {penalizaciones.get('vol_factor', 0):.3f}")
        print(f"      - Spreads: {penalizaciones.get('spread_factors', {})}")
        print(f"      - Turnover: {penalizaciones.get('turnover', 0):.3f}")
        print(f"      - Comisión: {penalizaciones.get('comision', 0):.6f}")
    else:
        print(f"      - Freeze activado")

    with open(exp_file, 'w') as f:
        f.write(str(exp_final))
    print(f"\n[5] Exposición guardada en {exp_file}")

    # --- 5. Guardar resultados con estructura organizada ---
    print("\n[6] Guardando resultados...")

    os.makedirs('logs/json', exist_ok=True)
    os.makedirs('logs/parquet', exist_ok=True)

    ultimo_regime = regime_df['score_regime'].iloc[-1] if not regime_df.empty else 0.0
    ultimo_leadership = leadership_df['score_leadership'].iloc[-1] if not leadership_df.empty else 0.0
    ultimo_geo = geo_df['score_geographic'].iloc[-1] if not geo_df.empty else 0.0
    ultimo_bonds = bond_df['score_bonds'].iloc[-1] if not bond_df.empty else 0.0
    ultimo_stress = stress_df['score_stress'].iloc[-1] if not stress_df.empty else 0.0
    ultimo_liquidity = liquidity_df['score_liquidity'].iloc[-1] if not liquidity_df.empty else 0.0
    ultimo_breadth = breadth_df['score_breadth'].iloc[-1] if not breadth_df.empty else 0.0

    historial_path = 'historial_radar.csv'
    nuevo_registro = pd.DataFrame({
        'flujo_puro': [flujo_puro],
        'fase_ciclo': [fase_ciclo],
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
        'score_liquidity': [ultimo_liquidity],
        'pend_3d': [pend_3d],
        'pend_5d': [pend_5d],
        'pend_10d': [pend_10d],
        'aceleracion': [aceleracion],
        'score_breadth': [ultimo_breadth],
        'motores_mejorando': [motores_mejorando],
        'score_riesgo_sistemico': [ultimo_riesgo],
        'score_carry': [ultimo_carry],
        'ciclo_institucional': [ciclo_inst],
        'score_global_liquidity': [ultimo_gli],
        'score_cftc': [ultimo_cftc],
        'score_breadth_advanced': [ultimo_breadth_adv],
        'score_financial_conditions': [ultimo_fc],
        'score_etf_flow': [ultimo_etf]
    })

    if os.path.exists(historial_path):
        historial = pd.read_csv(historial_path)
        for col in nuevo_registro.columns:
            if col not in historial.columns:
                if col in ['fase_ciclo', 'ciclo_institucional']:
                    historial[col] = 'NEUTRAL'
                else:
                    historial[col] = np.nan
        historial = pd.concat([historial, nuevo_registro], ignore_index=True)
    else:
        historial = nuevo_registro
    historial.to_csv(historial_path, index=False)
    print(f"    Historial actualizado en {historial_path}")

    fecha_str = ultima_fecha.strftime('%Y%m%d')
    timestamp_utc = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

    log_entry = {
        "fase_ciclo": fase_ciclo,
        "timestamp": timestamp_utc,
        "status": "success" if not errores else "partial",
        "fecha": str(ultima_fecha.date()),
        "score_global": float(score_global),
        "score_smoothed": float(score_smoothed),
        "dispersion": float(dispersion),
        "exposure_factor": float(exposure_factor),
        "exposicion_final": float(exp_final),
        "vix": float(vix),
        "vix_atr_ratio": float(vix_atr_ratio) if not freeze_activado else 0.0,
        "score_riesgo_sistemico": float(ultimo_riesgo),
        "score_carry": float(ultimo_carry),
        "ciclo_institucional": ciclo_inst,
        "penalizaciones": {
            "exposure_factor": float(penalizaciones.get('exposure_factor')) if penalizaciones.get('exposure_factor') is not None else None,
            "cash_factor": float(penalizaciones.get('cash_factor')) if penalizaciones.get('cash_factor') is not None else None,
            "vol_factor": float(penalizaciones.get('vol_factor')) if penalizaciones.get('vol_factor') is not None else None,
            "spread_factors": {k: float(v) for k, v in penalizaciones.get('spread_factors', {}).items()},
            "turnover": float(penalizaciones.get('turnover', 0)),
            "coste_turnover": float(penalizaciones.get('coste_turnover', 0)),
            "comision": float(penalizaciones.get('comision', 0))
        },
        "dinamica": {
            "pend_3d": float(pend_3d),
            "pend_5d": float(pend_5d),
            "pend_10d": float(pend_10d),
            "aceleracion": float(aceleracion),
            "motores_mejorando": int(motores_mejorando)
        },
        "modules": {
            "regime": "ok" if 'regime' not in str(errores) else "error",
            "leadership": "ok" if 'leadership' not in str(errores) else "error",
            "geo": "ok" if 'geo' not in str(errores) else "error",
            "bonds": "ok" if 'bonds' not in str(errores) else "error",
            "stress": "ok" if 'stress' not in str(errores) else "error",
            "liquidity": "ok" if 'liquidity' not in str(errores) else "error",
            "breadth": "ok" if 'breadth' not in str(errores) else "error",
            "etf_flow": "ok" if 'etf_flow' not in str(errores) else "error"
        },
        "errors": errores
    }

    with open('logs/json/ultimo_log.json', 'w') as f:
        json.dump(log_entry, f, indent=2)

    log_filename = f"logs/json/radar_{fecha_str}.json"
    with open(log_filename, 'w') as f:
        json.dump(log_entry, f, indent=2)
    print(f"    Log JSON guardado en {log_filename}")

    try:
        log_df = pd.DataFrame([log_riesgo])
        parquet_path = f"logs/parquet/exposicion_{fecha_str}.parquet"
        log_df.to_parquet(parquet_path, index=False)
        print(f"    Log de riesgo guardado en {parquet_path}")
    except Exception as e:
        logger.warning(f"No se pudo guardar log de riesgo: {e}")

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





