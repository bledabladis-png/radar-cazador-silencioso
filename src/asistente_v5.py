#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
asistente_v5.py - Asistente de Interpretación Macro V5.0 (con mejoras V6 y V7)
Incluye: percentiles históricos, motor de alertas profesional, contribución de factores,
modo rápido, mapa de riesgo, probabilidad de fase y escenarios macro.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import yaml
from datetime import datetime
import logging
import argparse

# Añadir ruta del proyecto al path para que encuentre los módulos src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.narrative.scenario_engine import ScenarioEngine
from src.alerts.alert_engine import AlertEngine
from src.interpretation.factor_contributor import FactorContributor
from src.visualization.risk_map import plot_risk_map, get_risk_quadrant
from src.interpretation.phase_probability import PhaseProbability
from src.context.percentile_engine import PercentileEngine

# Configuración de logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Cargar configuración desde YAML
# ------------------------------------------------------------
CONFIG_PATH = 'config/asistente_config.yaml'
if not os.path.exists(CONFIG_PATH):
    logger.error(f"No se encuentra el archivo de configuración {CONFIG_PATH}")
    exit(1)

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

UMBRALES = config['umbrales']
INTERP_MOTORES = config['interpretaciones_motores']

# ------------------------------------------------------------
# FUNCIONES DE INTERPRETACIÓN NARRATIVA
# ------------------------------------------------------------
def interpretar_motor(nombre, valor):
    """Devuelve una frase descriptiva para un motor dado su valor."""
    if valor > UMBRALES['score_muy_alto']:
        return INTERP_MOTORES[nombre]['muy_positivo']
    elif valor > UMBRALES['score_alto']:
        return INTERP_MOTORES[nombre]['positivo']
    elif valor > UMBRALES['score_bajo']:
        return INTERP_MOTORES[nombre]['neutral']
    elif valor > -UMBRALES['score_alto']:
        return INTERP_MOTORES[nombre]['negativo']
    else:
        return INTERP_MOTORES[nombre]['muy_negativo']

def interpretar_entradas_salidas(motores):
    """Devuelve frases sobre dónde entra y sale el dinero."""
    entradas = []
    salidas = []
    for nombre, valor in motores:
        if valor > UMBRALES['score_medio']:
            entradas.append(f"{nombre} ({valor:+.2f})")
        elif valor < -UMBRALES['score_medio']:
            salidas.append(f"{nombre} ({valor:+.2f})")
    
    texto_entradas = "El dinero está entrando en: " + (", ".join(entradas) if entradas else "ningún motor claramente.")
    texto_salidas = "El dinero está saliendo de: " + (", ".join(salidas) if salidas else "ningún motor claramente.")
    return texto_entradas, texto_salidas

def interpretar_fuerza(flujo_puro, pend_5d):
    """Evalúa la fuerza del flujo global."""
    if flujo_puro > UMBRALES['score_muy_alto']:
        return f"MUY FUERTE (flujo puro {flujo_puro:+.3f}). Entrada masiva de capital."
    elif flujo_puro > UMBRALES['score_alto']:
        return f"FUERTE (flujo puro {flujo_puro:+.3f}). Entrada significativa."
    elif flujo_puro > 0:
        if pend_5d > UMBRALES['pend_positiva']:
            return f"DÉBIL PERO MEJORANDO (flujo puro {flujo_puro:+.3f}, pendiente +{pend_5d:.3f})."
        else:
            return f"DÉBIL (flujo puro {flujo_puro:+.3f}). Entrada leve sin aceleración."
    elif flujo_puro > -UMBRALES['score_bajo']:
        if pend_5d < -UMBRALES['pend_positiva']:
            return f"DÉBIL PERO EMPEORANDO (flujo puro {flujo_puro:+.3f}, pendiente {pend_5d:.3f})."
        else:
            return f"DÉBIL (flujo puro {flujo_puro:+.3f}). Salida leve."
    elif flujo_puro > -UMBRALES['score_alto']:
        return f"FUERTE (flujo puro {flujo_puro:+.3f}). Salida significativa de capital."
    else:
        return f"MUY FUERTE (flujo puro {flujo_puro:+.3f}). Salida masiva, estrés elevado."

def interpretar_fiabilidad(dispersion):
    """Evalúa la fiabilidad de la señal basada en consenso."""
    if dispersion < 0.3:
        return f"ALTA (dispersión {dispersion:.2f}). Consenso entre motores, señal fiable."
    elif dispersion < 0.6:
        return f"MEDIA (dispersión {dispersion:.2f}). Cierta dispersión, señal con dudas."
    else:
        return f"BAJA (dispersión {dispersion:.2f}). Motores desalineados, señal poco fiable."

def interpretar_fase(fase):
    """Devuelve una descripción detallada de la fase del ciclo (7 fases)."""
    descripciones = {
        "CONTRACCION": "Contracción: mercado débil, sin señales de giro. Exposición defensiva.",
        "CAPITULACION": "Capitulación: venta masiva, posible suelo cercano. Prepararse para compras.",
        "ACUMULACION": "Acumulación institucional: capital entrando temprano. Aumentar exposición gradual.",
        "EXPANSION": "Expansión: crecimiento económico, mercado alcista. Exposición alta.",
        "EUFORIA": "Euforia: sobrecalentamiento, vigilar señales de agotamiento.",
        "LATE_CYCLE": "Ciclo tardío: señales de deterioro. Reducir exposición.",
        "NEUTRAL": "Mercado sin dirección clara. Esperar confirmación."
    }
    return descripciones.get(fase, "Fase no reconocida.")

def interpretar_ciclo_institucional(ciclo):
    """Devuelve una descripción para el ciclo institucional de 4 fases."""
    desc = {
        "EXPANSION": "Fase de expansión: crecimiento generalizado, confianza alta.",
        "ACUMULACION": "Fase de acumulación: smart money entrando, mercado lateral o bajista.",
        "DISTRIBUCION": "Fase de distribución: capital saliendo, posible techo.",
        "CAPITULACION": "Capitulación: venta masiva, cerca del suelo.",
        "NEUTRAL": "Fase neutral: sin dirección clara."
    }
    return desc.get(ciclo, "No disponible")

def interpretar_riesgo_sistemico(score):
    """Interpreta el score de riesgo sistémico."""
    if score < -0.5:
        return "MUY BAJO (entorno estable)"
    elif score < -0.2:
        return "BAJO"
    elif score < 0.2:
        return "MODERADO"
    elif score < 0.5:
        return "ELEVADO"
    else:
        return "MUY ALTO (alerta)"

def interpretar_carry_trade(score):
    """Interpreta el score de carry trade."""
    if score > 0.3:
        return "POSITIVO (apetito por carry)"
    elif score > 0:
        return "LIGERAMENTE POSITIVO"
    elif score > -0.3:
        return "NEUTRAL"
    else:
        return "NEGATIVO (desapalancamiento)"

def interpretar_mejora(score, pend_5d, aceleracion):
    """Evalúa si el mercado está mejorando."""
    if score < 0 and pend_5d > UMBRALES['pend_positiva'] and aceleracion > UMBRALES['acel_positiva']:
        return "SÍ, el mercado está mostrando signos de mejora (pendiente y aceleración positivas a pesar de score negativo)."
    elif score > 0 and pend_5d > UMBRALES['pend_positiva']:
        return "SÍ, el impulso alcista se mantiene."
    elif score > 0 and pend_5d < -UMBRALES['pend_positiva']:
        return "NO, el mercado está perdiendo impulso a pesar de estar en positivo."
    elif score < 0 and pend_5d < -UMBRALES['pend_positiva']:
        return "NO, el deterioro continúa."
    else:
        return "ESTABLE, sin cambios significativos."

def interpretar_acumulacion(motores_mejorando, flujo_puro, pend_5d):
    """Detecta posible acumulación institucional."""
    if motores_mejorando >= 3 and flujo_puro < 0 and pend_5d > UMBRALES['pend_positiva']:
        return "POSIBLE ACUMULACIÓN: varios motores mejorando a pesar de flujo negativo."
    elif motores_mejorando >= 4 and flujo_puro > 0:
        return "ACUMULACIÓN CONFIRMADA: amplia mejora con flujo positivo."
    else:
        return "Sin señales claras de acumulación."

# ------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Asistente V5.0 - Radar Macro')
    parser.add_argument('--quick', action='store_true', help='Modo rápido: solo resumen en consola, sin gráficos ni PDF')
    args = parser.parse_args()
    quick = args.quick

    # Verificar existencia del historial
    if not os.path.exists('historial_radar.csv'):
        logger.error("No se encuentra historial_radar.csv")
        return

    # Cargar datos
    df = pd.read_csv('historial_radar.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values('fecha')

    if len(df) == 0:
        logger.error("El historial está vacío")
        return

    # Validar columnas esenciales y opcionales
    columnas_esenciales = ['fecha', 'score_global', 'flujo_puro', 'fase_ciclo']
    missing_essential = [col for col in columnas_esenciales if col not in df.columns]
    if missing_essential:
        logger.error(f"Faltan columnas esenciales en el CSV: {missing_essential}")
        return

    columnas_opcionales = [
        'dispersion', 'pend_3d', 'pend_5d', 'pend_10d', 'aceleracion', 'motores_mejorando',
        'score_regime', 'score_leadership', 'score_geographic', 'score_bonds',
        'score_stress', 'score_liquidity', 'score_breadth',
        'score_riesgo_sistemico', 'score_carry', 'ciclo_institucional',
        'score_global_liquidity', 'score_cftc', 'score_breadth_advanced',
        'exposicion_final', 'exposure_factor'
    ]
    for col in columnas_opcionales:
        if col not in df.columns:
            logger.warning(f"Columna opcional '{col}' no encontrada. Se usará 0/None.")
            if col != 'ciclo_institucional':
                df[col] = 0.0
            else:
                df[col] = 'NEUTRAL'

    # Configuración de percentiles
    pct_config = config.get('percentiles', {})
    pct_thresholds = pct_config.get('umbrales', None)
    pct_metrics_config = pct_config.get('metricas', [])

    # Crear motor de percentiles con umbrales personalizados
    pct_engine = PercentileEngine(df, thresholds=pct_thresholds)

    # Cargar métricas definidas en la configuración (si existen)
    if pct_metrics_config:
        for m in pct_metrics_config:
            nombre = m['nombre']
            if nombre in df.columns:
                pct_engine.add_metric(nombre, df[nombre].dropna())
            else:
                logger.warning(f"Métrica para percentil '{nombre}' no encontrada en CSV")
    else:
        # Fallback a métricas por defecto (por si no hay config)
        pct_engine.add_metric('score_global', df['score_global'].dropna())
        pct_engine.add_metric('score_breadth', df['score_breadth'].dropna())
        pct_engine.add_metric('score_stress', df['score_stress'].dropna())
        pct_engine.add_metric('dispersion', df['dispersion'].dropna())
        pct_engine.add_metric('score_riesgo_sistemico', df['score_riesgo_sistemico'].dropna())
        pct_engine.add_metric('score_carry', df['score_carry'].dropna())

    # Inicializar calculador de probabilidades de fase
    feature_cols = ['score_global', 'score_breadth', 'score_stress', 'score_liquidity', 'score_leadership']
    available_cols = [c for c in feature_cols if c in df.columns]
    if len(available_cols) >= 3:
        phase_prob = PhaseProbability(df, available_cols)
    else:
        phase_prob = None
        logger.warning("No hay suficientes columnas para calcular probabilidad de fase.")

    # Última fila
    fila = df.iloc[-1]
    fila_anterior = df.iloc[-2] if len(df) >= 2 else None

    # Extraer valores
    flujo_puro = fila['flujo_puro']
    dispersion = fila['dispersion']
    fase = fila['fase_ciclo']
    pend_3d = fila['pend_3d']
    pend_5d = fila['pend_5d']
    pend_10d = fila['pend_10d']
    aceleracion = fila['aceleracion']
    motores_mejorando = fila['motores_mejorando']
    riesgo_score = fila['score_riesgo_sistemico']
    carry_score = fila['score_carry']
    ciclo_inst = fila['ciclo_institucional']
    liquidez_global_score = fila['score_global_liquidity']
    cftc_score = fila['score_cftc']
    breadth_adv_score = fila['score_breadth_advanced']
    exp_final = fila.get('exposicion_final', 0.0)

    # Calcular pendientes de breadth y bonds (para alerta de inflexión)
    if len(df) >= 6:
        pend_breadth = (df['score_breadth'].iloc[-1] - df['score_breadth'].iloc[-6]) / 5
        pend_bonds = (df['score_bonds'].iloc[-1] - df['score_bonds'].iloc[-6]) / 5
    else:
        pend_breadth = 0.0
        pend_bonds = 0.0

    # Scores de motores con nombres para el informe
    motores_nombres = [
        ('Régimen', fila['score_regime']),
        ('Liderazgo', fila['score_leadership']),
        ('Geográfico', fila['score_geographic']),
        ('Bonos', fila['score_bonds']),
        ('Estrés', fila['score_stress']),
        ('Liquidez', fila['score_liquidity']),
        ('Breadth', fila['score_breadth']),
        ('Liquidez Global', fila['score_global_liquidity']),
        ('CFTC Positioning', fila['score_cftc']),
        ('Breadth Avanzado', fila['score_breadth_advanced'])
    ]

    # Ordenar de mayor a menor
    motores_nombres.sort(key=lambda x: x[1], reverse=True)

    # Generar interpretaciones para cada motor
    interpretaciones = []
    for nombre, valor in motores_nombres:
        interp = interpretar_motor(nombre, valor)
        interpretaciones.append(interp)

    df_motores = pd.DataFrame({
        'Motor': [m[0] for m in motores_nombres],
        'Score': [m[1] for m in motores_nombres],
        'Interpretacion': interpretaciones
    })

    # Interpretaciones narrativas globales
    entradas, salidas = interpretar_entradas_salidas(motores_nombres)
    fuerza = interpretar_fuerza(flujo_puro, pend_5d)
    fiabilidad = interpretar_fiabilidad(dispersion)
    fase_desc = interpretar_fase(fase)
    riesgo_interp = interpretar_riesgo_sistemico(riesgo_score)
    carry_interp = interpretar_carry_trade(carry_score)
    ciclo_desc = interpretar_ciclo_institucional(ciclo_inst)

    # Interpretaciones adicionales
    mejora = interpretar_mejora(flujo_puro, pend_5d, aceleracion)
    acumulacion = interpretar_acumulacion(motores_mejorando, flujo_puro, pend_5d)

    # Calcular contribuciones
    snapshot_motores = {
        'regime': fila['score_regime'],
        'leadership': fila['score_leadership'],
        'geo': fila['score_geographic'],
        'bonds': fila['score_bonds'],
        'stress': fila['score_stress'],
        'liquidity': fila['score_liquidity'],
        'breadth': fila['score_breadth']
    }
    fc = FactorContributor()
    contrib_result = fc.compute_contributions(snapshot_motores, fase)
    contrib_text = fc.format_contributions(contrib_result)

    # Determinar cuadrante de riesgo
    stress = fila['score_stress']
    risk_quadrant = get_risk_quadrant(fila['score_global'], stress)

    # Calcular probabilidades de fase
    if phase_prob is not None:
        sample = {col: fila[col] for col in available_cols}
        phase_probs = phase_prob.get_probabilities(sample, temperature=0.5)
        top_phases = phase_prob.get_top_phases(sample, n=3, temperature=0.5)
    else:
        phase_probs = None
        top_phases = []

    # Inicializar motor de alertas y generar alertas
    alert_engine = AlertEngine()
    snapshot_alertas = {
        'score_global': fila['score_global'],
        'score_breadth': fila['score_breadth'],
        'score_stress': stress,
        'score_leadership': fila['score_leadership'],
        'score_liquidity': fila['score_liquidity'],
        'score_riesgo_sistemico': riesgo_score,
        'score_carry': carry_score,
        'dispersion': dispersion,
        'pend_3d': pend_3d,
        'aceleracion': aceleracion,
        'motores_mejorando': motores_mejorando,
        'pend_breadth': pend_breadth,
        'pend_bonds': pend_bonds,
    }
    # Añadir percentiles al snapshot de alertas
    for metric in ['score_global', 'score_breadth', 'score_stress', 'score_liquidity', 
                   'score_riesgo_sistemico', 'score_carry', 'dispersion', 'pend_5d']:
        if metric in fila and not pd.isna(fila[metric]):
            pct_val = pct_engine.percentile(metric, fila[metric])
            if not np.isnan(pct_val):
                snapshot_alertas[f'{metric}_percentile'] = pct_val
    alertas_obj = alert_engine.generate(snapshot_alertas)
    alertas = [f"{a.level}: {a.message}" for a in alertas_obj]

    # Generar escenarios macro

    # Crear motor de escenarios (con datos históricos)
    scenario_engine = ScenarioEngine(history_df=df, feature_cols=['score_global', 'score_stress', 'score_breadth', 'score_liquidity', 'score_riesgo_sistemico', 'dispersion'])
    snapshot_scenarios = {
        'score_global': fila['score_global'],
        'score_stress': stress,
        'score_breadth': fila['score_breadth'],
        'score_liquidity': fila['score_liquidity'],
        'score_riesgo_sistemico': riesgo_score
    }
    # Añadir percentiles al snapshot de escenarios
    for metric in ['score_global', 'score_stress', 'score_breadth', 'score_liquidity', 'score_riesgo_sistemico', 'dispersion']:
        pct_key = f"{metric}_percentile"
        if metric in fila and not pd.isna(fila[metric]):
            pct_val = pct_engine.percentile(metric, fila[metric])
            if not np.isnan(pct_val):
                snapshot_scenarios[pct_key] = pct_val
    scenarios, similar_periods = scenario_engine.generate_scenarios(snapshot_scenarios, return_similar=True)
    scenarios_text = scenario_engine.format_scenarios(scenarios)


    # Generar informe Markdown (función definida al final)
    generar_informe_markdown(df_motores, fase, flujo_puro, dispersion, pend_3d, pend_5d, pend_10d,
                             aceleracion, motores_mejorando, alertas, entradas, salidas, fuerza,
                             fiabilidad, fase_desc, riesgo_score, riesgo_interp, carry_score,
                             carry_interp, ciclo_inst, ciclo_desc, liquidez_global_score, cftc_score,
                             breadth_adv_score, mejora, acumulacion, pct_engine, risk_quadrant,
                             phase_probs, phase_prob, contrib_text, scenarios_text,
                             similar_periods=similar_periods, latest_dict=fila.to_dict())
    if quick:
        print("\n" + "="*50)
        print("📡 RESUMEN RÁPIDO DEL RADAR")
        print("="*50)
        print(f"Fecha: {fila['fecha']}")
        print(f"Fase del ciclo: {fase}")
        print(f"Score global: {fila['score_global']:.3f}")
        print(f"Exposición final: {exp_final:.2%}")
        print(f"Dispersión: {dispersion:.3f}")
        print(f"Riesgo sistémico: {riesgo_score:.3f} ({riesgo_interp})")
        print(f"Cuadrante: {risk_quadrant}")
        print("\nAlertas principales:")
        for a in alertas[:3]:
            print(f"  - {a}")
        print("="*50)
    else:
        # Preparar variables para PDF
        score_global = fila['score_global']
        breadth_score = fila['score_breadth']
        liquidity_score = fila['score_liquidity']
        fases_historicas = df['fase_ciclo'].tolist()[-min(30, len(df)):]

        # Generar informe PDF
        generar_informe_pdf(df_motores, score_global, stress, breadth_score, liquidity_score, fase,
                            flujo_puro, dispersion, pend_3d, pend_5d, pend_10d,
                            aceleracion, motores_mejorando, alertas, entradas, salidas, fuerza,
                            fiabilidad, fase_desc, riesgo_score, riesgo_interp, carry_score,
                            carry_interp, ciclo_inst, ciclo_desc, fases_historicas, mejora, acumulacion,
                            liquidez_global_score, cftc_score, breadth_adv_score, pct_engine,
                            risk_quadrant, phase_probs, phase_prob, contrib_text, quick,
                            scenarios_text=scenarios_text, similar_periods=similar_periods, latest_dict=fila.to_dict(), pct_metrics=pct_metrics_config)

    # Mensajes de confirmación
    print("\n✅ Informe Markdown generado: informes/informe_radar.md")
    if not quick:
        print("✅ Informe PDF generado: informes/informe_radar.pdf")
    print("✅ Proceso completado.")
    logger.info("Proceso completado.")

# ------------------------------------------------------------
# FUNCIÓN DE GENERACIÓN DE INFORME MARKDOWN
# ------------------------------------------------------------
def generar_informe_markdown(df_motores, fase, flujo_puro, dispersion, pend_3d, pend_5d, pend_10d,
                             aceleracion, motores_mejorando, alertas, entradas, salidas, fuerza,
                             fiabilidad, fase_desc, riesgo_score, riesgo_interp, carry_score,
                             carry_interp, ciclo_inst, ciclo_desc, liquidez_global_score, cftc_score,
                             breadth_adv_score, mejora, acumulacion, pct_engine, risk_quadrant,
                             phase_probs, phase_prob, contrib_text, scenarios_text,
                             similar_periods=None, latest_dict=None, filename="informes/informe_radar.md"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# 📡 INFORME RADAR CAZADOR - {datetime.now().strftime('%d/%m/%Y')}\n\n")
        
        f.write("## 🧠 RESUMEN EJECUTIVO\n")
        f.write(f"{entradas} {salidas} {fuerza} {fiabilidad} ")
        f.write(f"La fase del ciclo es **{fase}**: {fase_desc}\n\n")

        # Sección de percentiles (si hay configuración y datos)
        pct_config = config.get('percentiles', {})
        pct_metrics = pct_config.get('metricas', [])
        if pct_metrics and latest_dict is not None:
            f.write("\n## 📊 Contexto histórico (percentiles)\n\n")
            for m in pct_metrics:
                nombre = m['nombre']
                etiqueta = m.get('etiqueta', nombre)
                valor = latest_dict.get(nombre)
                if valor is not None and not pd.isna(valor):
                    narrative = pct_engine.get_narrative(nombre, etiqueta, valor)
                    f.write(f"- {narrative}\n")
                else:
                    f.write(f"- {etiqueta}: *dato no disponible*\n")
            f.write("\n")
        
        # 8 preguntas clave
        f.write("## 🔍 RESPUESTA A LAS 8 PREGUNTAS CLAVE\n\n")
        f.write(f"**1. ¿Dónde entra el dinero?** {entradas}\n\n")
        f.write(f"**2. ¿Dónde sale el dinero?** {salidas}\n\n")
        f.write(f"**3. ¿Con qué fuerza?** {fuerza}\n\n")
        f.write(f"**4. ¿Con qué fiabilidad?** {fiabilidad}\n\n")
        f.write(f"**5. ¿En qué fase del ciclo?** {fase} – {fase_desc}\n\n")
        f.write(f"**6. ¿Está mejorando?** {mejora}\n\n")
        f.write(f"**7. ¿Hay acumulación?** {acumulacion}\n\n")
        f.write(f"**8. ¿Riesgo sistémico?** {riesgo_interp} (score: {riesgo_score:.2f})\n\n")
        
        f.write("## 📊 DETALLE DE MOTORES\n")
        f.write("| Motor | Score | Interpretación |\n")
        f.write("|-------|-------|----------------|\n")
        for _, row in df_motores.iterrows():
            f.write(f"| {row['Motor']} | {row['Score']:+.3f} | {row['Interpretacion']} |\n")
        f.write("\n")
        
        f.write("## 📈 MÉTRICAS DINÁMICAS\n")
        f.write(f"- **Pendiente 3d:** {pend_3d:+.4f}\n")
        f.write(f"- **Pendiente 5d:** {pend_5d:+.4f}\n")
        f.write(f"- **Pendiente 10d:** {pend_10d:+.4f}\n")
        f.write(f"- **Aceleración:** {aceleracion:+.4f}\n")
        f.write(f"- **Motores mejorando:** {motores_mejorando}/7\n\n")
        
        # Probabilidad de fase
        if phase_probs:
            f.write("## 📊 Probabilidad de fase\n")
            f.write(phase_prob.format_probabilities(phase_probs, decimals=1) + "\n\n")
        
        # Escenarios macro
        f.write("## 🌍 Escenarios macro\n")
        f.write(scenarios_text + "\n\n")
        
        # Contribución de factores
        f.write("## 📊 Contribución de factores\n")
        f.write(contrib_text + "\n\n")
        
        f.write("## 🆕 MÉTRICAS V7 Y NUEVOS MOTORES\n")
        f.write(f"- **Riesgo Sistémico:** {riesgo_score:.3f} – {riesgo_interp}\n")
        f.write(f"- **Carry Trade:** {carry_score:.3f} – {carry_interp}\n")
        f.write(f"- **Ciclo Institucional:** {ciclo_inst} – {ciclo_desc}\n")
        
        # Interpretaciones de motores adicionales
        liquidez_interp = interpretar_motor('Liquidez Global', liquidez_global_score)
        f.write(f"- **Liquidez Global (BIS):** {liquidez_global_score:.3f} – {liquidez_interp}\n")
        
        cftc_interp = interpretar_motor('CFTC Positioning', cftc_score)
        f.write(f"- **Posicionamiento CFTC:** {cftc_score:.3f} – {cftc_interp}\n")
        
        breadth_adv_interp = interpretar_motor('Breadth Avanzado', breadth_adv_score)
        f.write(f"- **Breadth Avanzado:** {breadth_adv_score:.3f} – {breadth_adv_interp}\n\n")
        
        f.write("## ⚠️ ALERTAS ACTIVAS\n")
        if alertas:
            for a in alertas:
                f.write(f"- {a}\n")
        else:
            f.write("- No hay alertas activas.\n")
        f.write("\n")
        
        f.write("## 🎯 RECOMENDACIÓN\n")
        if fase in ["EXPANSION", "EUFORIA"]:
            f.write("Mantener exposición alta, pero vigilando señales de agotamiento.\n")
        elif fase == "ACUMULACION":
            f.write("Aumentar exposición gradualmente, priorizando sectores líderes.\n")
        elif fase == "CONTRACCION":
            f.write("Exposición defensiva, esperar señales de giro.\n")
        elif fase == "CAPITULACION":
            f.write("Mantener liquidez, prepararse para compras.\n")
        elif fase == "LATE_CYCLE":
            f.write("Reducir exposición, aumentar coberturas.\n")
        else:
            f.write("Mantener exposición moderada, esperar confirmación.\n")
        f.write("\n---\n")
        f.write("_Informe generado automáticamente por el Asistente V5.0._\n")

# ------------------------------------------------------------
# FUNCIÓN DE GENERACIÓN DE INFORME PDF
# ------------------------------------------------------------
def generar_informe_pdf(df_motores, score_global, stress, breadth_score, liquidity_score, fase,
                        flujo_puro, dispersion, pend_3d, pend_5d, pend_10d,
                        aceleracion, motores_mejorando, alertas, entradas, salidas, fuerza,
                        fiabilidad, fase_desc, riesgo_score, riesgo_interp, carry_score,
                        carry_interp, ciclo_inst, ciclo_desc, fases_historicas, mejora, acumulacion,
                        liquidez_global_score, cftc_score, breadth_adv_score, pct_engine,
                        risk_quadrant, phase_probs, phase_prob, contrib_text, quick,
                        scenarios_text=None, similar_periods=None, latest_dict=None, pct_metrics=None, filename="informes/informe_radar.pdf"):
    """Genera un PDF con texto y gráficos, incluyendo escenarios y mapa de riesgo."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with PdfPages(filename) as pdf:
        # Página de resumen ejecutivo
        fig_summary = plot_summary_page(score_global, stress, breadth_score, liquidity_score, fase, ciclo_inst, risk_quadrant, flujo_puro, dispersion, riesgo_interp)
        pdf.savefig(fig_summary)
        plt.close(fig_summary)

        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')

        # Interpretaciones
        liquidez_interp = interpretar_motor('Liquidez Global', liquidez_global_score)
        cftc_interp = interpretar_motor('CFTC Positioning', cftc_score)
        breadth_adv_interp = interpretar_motor('Breadth Avanzado', breadth_adv_score)

        texto = f"""
ASISTENTE V5.0 - INFORME RADAR (V6/V7)
Fecha: {datetime.now().strftime('%d/%m/%Y')}

Fase del ciclo (7 fases): {fase}
Ciclo institucional (4 fases): {ciclo_inst}
Cuadrante de riesgo: {risk_quadrant}
Flujo puro: {flujo_puro:.3f}
Dispersión: {dispersion:.2f}
Liquidez Global (BIS): {liquidez_global_score:.3f} – {liquidez_interp}
Posicionamiento CFTC: {cftc_score:.3f} – {cftc_interp}
Breadth Avanzado: {breadth_adv_score:.3f} – {breadth_adv_interp}

RESUMEN EJECUTIVO
{entradas}
{salidas}
{fuerza}
{fiabilidad}
{fase_desc}

8 PREGUNTAS CLAVE
1. Entradas: {entradas}
2. Salidas: {salidas}
3. Fuerza: {fuerza}
4. Fiabilidad: {fiabilidad}
5. Fase: {fase} – {fase_desc}
6. Mejora: {mejora}
7. Acumulación: {acumulacion}
8. Riesgo sistémico: {riesgo_interp} (score: {riesgo_score:.2f})

MÉTRICAS DINÁMICAS
Pendiente 3d: {pend_3d:+.4f}
Pendiente 5d: {pend_5d:+.4f}
Pendiente 10d: {pend_10d:+.4f}
Aceleración: {aceleracion:+.4f}
Motores mejorando: {motores_mejorando}/7

MÉTRICAS V7 Y NUEVOS MOTORES
Riesgo Sistémico: {riesgo_score:.3f} – {riesgo_interp}
Carry Trade: {carry_score:.3f} – {carry_interp}
Ciclo Institucional: {ciclo_inst} – {ciclo_desc}
Liquidez Global (BIS): {liquidez_global_score:.3f} – {liquidez_interp}
Posicionamiento CFTC: {cftc_score:.3f} – {cftc_interp}
Breadth Avanzado: {breadth_adv_score:.3f} – {breadth_adv_interp}

ESCENARIOS MACRO
{scenarios_text}

ALERTAS ACTIVAS
{chr(10).join(alertas) if alertas else "No hay alertas activas."}

RECOMENDACIÓN
"""
        if fase in ["EXPANSION", "EUFORIA"]:
            texto += "Mantener exposición alta, pero vigilando señales de agotamiento."
        elif fase == "ACUMULACION":
            texto += "Aumentar exposición gradualmente, priorizando sectores líderes."
        elif fase == "CONTRACCION":
            texto += "Exposición defensiva, esperar señales de giro."
        elif fase == "CAPITULACION":
            texto += "Mantener liquidez, prepararse para compras."
        elif fase == "LATE_CYCLE":
            texto += "Reducir exposición, aumentar coberturas."
        else:
            texto += "Mantener exposición moderada, esperar confirmación."

        ax.text(0.5, 0.5, texto, fontsize=8, ha='center', va='center', wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

        # Página 2: Radar chart
        fig = plot_radar(df_motores)
        pdf.savefig(fig)
        plt.close(fig)

        # Página 3: Heatmap
        fig = plot_heatmap(df_motores)
        pdf.savefig(fig)
        plt.close(fig)

        # Página 4: Timeline
        if fases_historicas:
            fig = plot_timeline(fases_historicas)
            pdf.savefig(fig)
            plt.close(fig)

        # Página 5: Mapa de riesgo
        if not quick:
            fig_risk = plot_risk_map(score_global, stress)
            pdf.savefig(fig_risk)
            plt.close(fig_risk)

        # Página 6: Probabilidades de fase
        if phase_probs and not quick:
            fig_phase, ax_phase = plt.subplots(figsize=(6, 4))
            ax_phase.axis('off')
            text = "Distribución de probabilidad de fase:\n"
            for phase, prob in sorted(phase_probs.items(), key=lambda x: x[1], reverse=True):
                text += f"{phase}: {prob*100:.1f}%\n"
            ax_phase.text(0.5, 0.5, text, ha='center', va='center', fontsize=12, transform=ax_phase.transAxes)
            pdf.savefig(fig_phase)
            plt.close(fig_phase)

        # Página de percentiles
        if pct_metrics and latest_dict is not None:
            fig_pct = plot_percentile_bars(pct_engine, latest_dict, pct_metrics)
            if fig_pct:
                pdf.savefig(fig_pct)
                plt.close(fig_pct)

# ------------------------------------------------------------
# FUNCIONES DE GRÁFICOS
# ------------------------------------------------------------

def plot_summary_page(score_global, stress, breadth_score, liquidity_score, fase, 
                      ciclo_inst, risk_quadrant, flujo_puro, dispersion, riesgo_interp):
    """Crea una página de resumen ejecutivo con los indicadores clave."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    # Título
    ax.text(0.5, 0.95, '📊 RESUMEN EJECUTIVO DEL RADAR', ha='center', va='center', 
            fontsize=18, fontweight='bold', color='navy', transform=ax.transAxes)
    ax.text(0.5, 0.92, datetime.now().strftime('%d/%m/%Y'), ha='center', va='center', 
            fontsize=12, style='italic', transform=ax.transAxes)
    
    # Métricas principales en columnas
    ax.text(0.2, 0.82, 'Score Global', ha='center', va='center', fontsize=12, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.82, 'Fase del Ciclo', ha='center', va='center', fontsize=12, fontweight='bold', transform=ax.transAxes)
    ax.text(0.8, 0.82, 'Riesgo Sistémico', ha='center', va='center', fontsize=12, fontweight='bold', transform=ax.transAxes)
    
    score_color = 'green' if score_global > 0 else 'red'
    ax.text(0.2, 0.78, f'{score_global:.3f}', ha='center', va='center', fontsize=16, 
            fontweight='bold', color=score_color, transform=ax.transAxes)
    ax.text(0.5, 0.78, fase, ha='center', va='center', fontsize=14, fontweight='bold', transform=ax.transAxes)
    riesgo_color = 'red' if riesgo_interp in ['ELEVADO', 'MUY ALTO'] else 'green'
    ax.text(0.8, 0.78, riesgo_interp, ha='center', va='center', fontsize=14, fontweight='bold', color=riesgo_color, transform=ax.transAxes)
    
    # Segunda fila de métricas
    ax.text(0.2, 0.68, 'Estrés', ha='center', va='center', fontsize=11, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.68, 'Amplitud', ha='center', va='center', fontsize=11, fontweight='bold', transform=ax.transAxes)
    ax.text(0.8, 0.68, 'Liquidez', ha='center', va='center', fontsize=11, fontweight='bold', transform=ax.transAxes)
    
    stress_color = 'red' if stress < -0.2 else 'green'
    ax.text(0.2, 0.64, f'{stress:.3f}', ha='center', va='center', fontsize=14, color=stress_color, transform=ax.transAxes)
    breadth_color = 'green' if breadth_score > 0.2 else 'red'
    ax.text(0.5, 0.64, f'{breadth_score:.3f}', ha='center', va='center', fontsize=14, color=breadth_color, transform=ax.transAxes)
    liq_color = 'green' if liquidity_score > 0.2 else 'red'
    ax.text(0.8, 0.64, f'{liquidity_score:.3f}', ha='center', va='center', fontsize=14, color=liq_color, transform=ax.transAxes)
    
    # Tercera fila
    ax.text(0.2, 0.54, 'Flujo Puro', ha='center', va='center', fontsize=11, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.54, 'Dispersión', ha='center', va='center', fontsize=11, fontweight='bold', transform=ax.transAxes)
    ax.text(0.8, 0.54, 'Ciclo Inst.', ha='center', va='center', fontsize=11, fontweight='bold', transform=ax.transAxes)
    
    flujo_color = 'green' if flujo_puro > 0 else 'red'
    ax.text(0.2, 0.50, f'{flujo_puro:.3f}', ha='center', va='center', fontsize=14, color=flujo_color, transform=ax.transAxes)
    disp_color = 'orange' if dispersion > 0.6 else 'green'
    ax.text(0.5, 0.50, f'{dispersion:.3f}', ha='center', va='center', fontsize=14, color=disp_color, transform=ax.transAxes)
    ax.text(0.8, 0.50, ciclo_inst, ha='center', va='center', fontsize=12, transform=ax.transAxes)
    
    # Cuadrante de riesgo destacado
    ax.text(0.5, 0.38, f'Cuadrante de Riesgo: {risk_quadrant}', ha='center', va='center', 
            fontsize=14, fontweight='bold', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Recomendación rápida
    if fase in ["EXPANSION", "EUFORIA"]:
        rec = "Mantener exposición alta, vigilando señales de agotamiento."
    elif fase == "ACUMULACION":
        rec = "Aumentar exposición gradualmente, priorizando sectores líderes."
    elif fase == "CONTRACCION":
        rec = "Exposición defensiva, esperar señales de giro."
    elif fase == "CAPITULACION":
        rec = "Mantener liquidez, prepararse para compras."
    elif fase == "LATE_CYCLE":
        rec = "Reducir exposición, aumentar coberturas."
    else:
        rec = "Mantener exposición moderada, esperar confirmación."
    
    ax.text(0.5, 0.25, f'🎯 Recomendación: {rec}', ha='center', va='center', 
            fontsize=12, transform=ax.transAxes, wrap=True,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # Nota al pie
    ax.text(0.5, 0.10, 'Informe generado por Asistente V5.0 - Radar Macro', 
            ha='center', va='center', fontsize=9, style='italic', transform=ax.transAxes)
    
    return fig

def plot_radar(df_motores):
    """Gráfico radar de los scores por motor (versión mejorada)."""
    labels = df_motores['Motor'].tolist()
    scores = df_motores['Score'].values
    n = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    scores = np.concatenate((scores, [scores[0]]))
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
    
    # Colores degradados según el valor
    colors = plt.cm.RdYlGn((scores + 1)/2)  # mapear de [-1,1] a [0,1]
    
    ax.plot(angles, scores, 'o-', linewidth=2, color='blue', alpha=0.7)
    ax.fill(angles, scores, alpha=0.25, color='blue')
    
    # Añadir etiquetas con los valores
    for i, (ang, val) in enumerate(zip(angles[:-1], scores[:-1])):
        ax.text(ang, val + 0.1, f'{val:.2f}', ha='center', va='center', fontsize=8, color='darkblue')
    
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
    ax.set_ylim(-1, 1)
    ax.set_title('Radar de Motores - Flujo Institucional', size=14, y=1.08, fontweight='bold')
    ax.grid(True, alpha=0.3)
    return fig

def plot_heatmap(df_motores):
    """Heatmap de scores por motor (versión mejorada)."""
    data = df_motores.set_index('Motor')[['Score']]
    fig, ax = plt.subplots(figsize=(7, len(data)*0.5 + 1))
    sns.heatmap(data.T, annot=True, fmt='+.2f', cmap='RdYlGn', center=0, 
                cbar_kws={'label': 'Score'}, linewidths=1, linecolor='white', ax=ax)
    ax.set_title('Entrada/Salida por Motor', fontsize=14, fontweight='bold')
    ax.set_ylabel('')
    ax.set_xlabel('')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_timeline(fases_historicas, fechas=None):
    """Timeline de fases del ciclo."""
    if fechas is None:
        fechas = pd.date_range(end=pd.Timestamp.today(), periods=len(fases_historicas))
    df = pd.DataFrame({'Fecha': fechas, 'Fase': fases_historicas})
    
    colores = {
        'CONTRACCION': '#d62728',
        'CAPITULACION': '#e377c2',
        'ACUMULACION': '#2ca02c',
        'EXPANSION': '#1f77b4',
        'EUFORIA': '#ffbb78',
        'LATE_CYCLE': '#8c564b',
        'NEUTRAL': '#7f7f7f'
    }
    fig, ax = plt.subplots(figsize=(8,2))
    ax.bar(df['Fecha'], 1, color=[colores.get(f, 'gray') for f in df['Fase']], width=0.8)
    ax.set_yticks([])
    ax.set_title('Timeline de Fases del Ciclo', fontsize=14)
    ax.set_xlabel('Fecha')
    return fig

def plot_percentile_bars(pct_engine, latest_dict, pct_metrics):
    """Genera un gráfico de barras horizontales con los percentiles de las métricas."""
    if not pct_metrics or latest_dict is None:
        return None

    data = []
    labels = []
    colors = []

    for m in pct_metrics:
        nombre = m['nombre']
        etiqueta = m.get('etiqueta', nombre)
        valor = latest_dict.get(nombre)
        if valor is not None and not pd.isna(valor):
            pct = pct_engine.percentile(nombre, valor)
            if not np.isnan(pct):
                data.append(pct)
                labels.append(etiqueta)
                cat = pct_engine.classify(pct)
                if 'extremadamente alto' in cat:
                    colors.append('darkgreen')
                elif 'alto' in cat:
                    colors.append('green')
                elif 'normal' in cat:
                    colors.append('gray')
                elif 'bajo' in cat:
                    colors.append('orange')
                else:  # extremadamente bajo
                    colors.append('red')

    if not data:
        return None

    fig, ax = plt.subplots(figsize=(8, max(4, len(data)*0.4)))
    y_pos = np.arange(len(data))
    ax.barh(y_pos, data, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Percentil')
    ax.set_title('Contexto histórico (percentiles)')
    ax.axvline(50, color='black', linestyle='--', linewidth=0.8)  # línea de referencia
    for i, (p, col) in enumerate(zip(data, colors)):
        ax.text(p + 1, i, f'{p:.1f}%', va='center', fontsize=8)
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    main()









