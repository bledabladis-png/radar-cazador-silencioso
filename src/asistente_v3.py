#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
asistente_v3.py - Asistente de Interpretación Macro V3
Genera un informe ejecutivo en Markdown y alertas basadas en el historial del radar.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# ------------------------------------------------------------
# CONFIGURACIÓN DE UMBRALES (ajusta según experiencia)
# ------------------------------------------------------------
UMBRALES = {
    'score_extremo_alto': 0.8,
    'score_alto': 0.6,
    'score_medio': 0.2,
    'score_bajo': -0.2,
    'score_muy_bajo': -0.6,
    'score_extremo_bajo': -0.8,
    'dispersion_baja': 0.3,
    'dispersion_media': 0.6,
    'dispersion_alerta': 0.7,
    'vix_alto': 30,
    'vix_medio': 25,
    'rachas_significativas': 3,
    'pendiente_fuerte': 0.03,
    'pendiente_suave': 0.01,
    'pesos_tendencia': [0.5, 0.3, 0.2]  # para 3,5,10 días
}

# ------------------------------------------------------------
# CAPA 1: PREPARACIÓN DE DATOS (Data Preparation Layer)
# ------------------------------------------------------------
def calcular_pendientes(serie, ventanas=[3,5,10], pesos=None):
    """
    Calcula la pendiente lineal de los últimos N días usando regresión lineal simple.
    Devuelve diccionario con pendientes y tendencia ponderada.
    """
    if pesos is None:
        pesos = UMBRALES['pesos_tendencia']  # [0.5, 0.3, 0.2] para 3,5,10 días

    pendientes = {}

    for n in ventanas:
        if len(serie) >= n:
            # Tomamos los últimos n valores de la serie
            y = serie.values[-n:].astype(float)
            # Creamos el eje x: 0,1,2,...,n-1
            x = np.arange(n)
            # Fórmula de la pendiente: cov(x,y) / var(x)
            if n > 1:
                pendiente = np.cov(x, y)[0,1] / np.var(x)
            else:
                pendiente = 0.0
        else:
            pendiente = 0.0
        pendientes[f'pend{n}'] = pendiente

    # Calcular la tendencia ponderada
    tendencia_pond = 0.0
    for i, n in enumerate(ventanas):
        tendencia_pond += pendientes[f'pend{n}'] * pesos[i]
    pendientes['tendencia_ponderada'] = tendencia_pond

    return pendientes

def calcular_persistencia(serie):
    """
    Calcula días consecutivos con la misma dirección (subiendo/bajando).
    Devuelve (dias_subiendo, dias_bajando).
    """
    if len(serie) < 2:
        return 0, 0

    # Calculamos la diferencia y su signo
    diff = serie.diff().dropna()
    direccion = np.sign(diff)

    # Contar racha actual desde el final
    racha_actual = 0
    ultima_direccion = direccion.iloc[-1] if not direccion.empty else 0
    for d in reversed(direccion):
        if d == ultima_direccion and d != 0:
            racha_actual += 1
        else:
            break

    if ultima_direccion > 0:
        return racha_actual, 0
    elif ultima_direccion < 0:
        return 0, racha_actual
    else:
        return 0, 0

def calcular_momentum(serie, ventanas=[5,10]):
    """
    Diferencia entre valor actual y valor de hace N días.
    Devuelve diccionario con cambios.
    """
    momentum = {}
    for n in ventanas:
        if len(serie) > n:
            mom = serie.iloc[-1] - serie.iloc[-1-n]
        else:
            mom = 0.0
        momentum[f'mom{n}'] = mom
    return momentum

def ranking_motores(fila):
    """
    Ordena los motores por su valor (de mayor a menor).
    fila: Series con columnas score_regime, score_leadership, ...
    Devuelve lista de (nombre, valor).
    """
    motores = [
        ('Régimen', fila.get('score_regime', 0)),
        ('Liderazgo', fila.get('score_leadership', 0)),
        ('Geográfico', fila.get('score_geographic', 0)),
        ('Bonos', fila.get('score_bonds', 0)),
        ('Estrés', fila.get('score_stress', 0)),
        ('Liquidez', fila.get('score_liquidity', 0))
    ]
    # Filtrar los que tienen valor (evitar NaN)
    motores = [(nombre, valor) for nombre, valor in motores if not pd.isna(valor)]
    motores.sort(key=lambda x: x[1], reverse=True)
    return motores

def clasificar_consistencia(dispersion):
    """
    Clasifica la dispersión en señal fuerte, media o débil.
    """
    if dispersion < UMBRALES['dispersion_baja']:
        return "🟢 SEÑAL FUERTE (consenso)"
    elif dispersion < UMBRALES['dispersion_media']:
        return "🟡 SEÑAL MEDIA (dudas)"
    else:
        return "🔴 SEÑAL DÉBIL (alta dispersión)"

# ------------------------------------------------------------
# CAPA 2: ANÁLISIS DE SEÑALES (Signal Analysis Layer)
# ------------------------------------------------------------
def analizar_regimen(score_global, score_regime):
    """Analiza el régimen de mercado."""
    if score_global > UMBRALES['score_extremo_alto']:
        return "🟢 EXTREMO ALCISTA (euforia)"
    elif score_global > UMBRALES['score_alto']:
        return "🟢 ALCISTA FUERTE"
    elif score_global > UMBRALES['score_medio']:
        return "🟢 ALCISTA DÉBIL"
    elif score_global > UMBRALES['score_bajo']:
        return "⚪ NEUTRAL"
    elif score_global > UMBRALES['score_muy_bajo']:
        return "🔴 BAJISTA DÉBIL"
    elif score_global > UMBRALES['score_extremo_bajo']:
        return "🔴 BAJISTA FUERTE"
    else:
        return "🔴 EXTREMO BAJISTA (pánico)"

def analizar_tendencia(pendientes, persistencia):
    """
    Analiza la tendencia del capital usando pendiente ponderada y rachas.
    pendientes: dict con pend3, pend5, pend10, tendencia_ponderada
    persistencia: (dias_subiendo, dias_bajando)
    """
    tendencia = pendientes['tendencia_ponderada']
    racha_sub, racha_baj = persistencia
    rachas_significativas = UMBRALES['rachas_significativas']

    if tendencia > UMBRALES['pendiente_fuerte']:
        if racha_sub >= rachas_significativas:
            return "🚀 ENTRADA FUERTE DE CAPITAL (aceleración)"
        else:
            return "📈 ENTRADA FUERTE RECIENTE"
    elif tendencia > UMBRALES['pendiente_suave']:
        if racha_sub >= rachas_significativas:
            return "📊 ENTRADA GRADUAL SOSTENIDA"
        else:
            return "📉 ENTRADA INICIAL"
    elif tendencia < -UMBRALES['pendiente_fuerte']:
        if racha_baj >= rachas_significativas:
            return "🔻 SALIDA FUERTE DE CAPITAL (aceleración)"
        else:
            return "🔻 SALIDA FUERTE RECIENTE"
    elif tendencia < -UMBRALES['pendiente_suave']:
        if racha_baj >= rachas_significativas:
            return "📉 SALIDA GRADUAL SOSTENIDA"
        else:
            return "📉 SALIDA INICIAL"
    else:
        return "⚪ ESTABLE (sin presión direccional)"

def analizar_liderazgo(score_leadership):
    """Analiza liderazgo de mercado."""
    if score_leadership > UMBRALES['score_medio']:
        return "🟢 Crecimiento / tecnología lidera (fase expansiva)"
    elif score_leadership < -UMBRALES['score_medio']:
        return "🔴 Defensivos lideran (fase tardía o contracción)"
    else:
        return "⚪ Neutral (sin liderazgo claro)"
def analizar_geografico(score_geographic):
    """Analiza flujo geográfico."""
    if score_geographic > UMBRALES['score_medio']:
        return "🌍 Emergentes superan a desarrollados (expansión global, dólar débil)"
    elif score_geographic < -UMBRALES['score_medio']:
        return "🌎 Desarrollados dominan (capital defensivo, dólar fuerte)"
    else:
        return "⚪ Neutral"

def analizar_credito(score_bonds):
    """Analiza condiciones de crédito."""
    if score_bonds > UMBRALES['score_medio']:
        return "💳 Mejora del crédito, riesgo aceptado"
    elif score_bonds < -UMBRALES['score_medio']:
        return "⚠️ Miedo a impagos, huida a calidad"
    else:
        return "⚪ Neutral"

def analizar_estres(score_stress, vix):
    """Analiza estrés financiero."""
    # Si no hay vix, usamos 20 como valor por defecto
    if pd.isna(vix):
        vix = 20.0

    if score_stress > 0.3 or vix > UMBRALES['vix_alto']:
        return "🔴 ESTRÉS ELEVADO (pánico)"
    elif score_stress > 0.1 or vix > UMBRALES['vix_medio']:
        return "🟡 TENSIÓN MODERADA"
    else:
        return "🟢 ESTABILIDAD"

def analizar_liquidez(score_liquidity):
    """Analiza liquidez global."""
    if score_liquidity > UMBRALES['score_medio']:
        return "💧 Liquidez expansiva (dólar débil)"
    elif score_liquidity < -UMBRALES['score_medio']:
        return "💧 Liquidez restrictiva (dólar fuerte)"
    else:
        return "⚪ Neutral"

# ------------------------------------------------------------
# CAPA 3: INTERPRETACIÓN MACRO (Macro Interpretation Layer)
# ------------------------------------------------------------
def detectar_acumulacion_institucional(fila):
    """
    Detecta acumulación institucional (3 de 4 motores positivos).
    Motores: régime, leadership, bonds, geographic
    """
    motores_clave = [
        fila.get('score_regime', 0),
        fila.get('score_leadership', 0),
        fila.get('score_bonds', 0),
        fila.get('score_geographic', 0)
    ]
    positivos = sum(1 for valor in motores_clave if valor > 0)
    if positivos >= 3:
        return "✅ ACUMULACIÓN INSTITUCIONAL (capital entrando coordinadamente en riesgo)"
    elif positivos <= 1:
        return "🔻 DISTRIBUCIÓN INSTITUCIONAL (capital abandonando riesgo)"
    else:
        return "⚪ NEUTRAL (sin acumulación clara)"

def detectar_distribucion_institucional(fila):
    """
    Detecta distribución institucional (3 de 4 motores negativos).
    """
    # TODO: implementar
    return False

def clasificar_fase_ciclo(score_global, score_leadership, score_bonds, score_stress):
    """
    Clasifica la fase del ciclo con lenguaje suave.
    """
    if score_global > 0.2 and score_leadership > 0.2 and score_bonds > 0.2 and score_stress < 0.1:
        return "📈 Configuración compatible con EXPANSIÓN"
    elif score_global > 0 and score_leadership < -0.2 and score_bonds < 0:
        return "📉 Posible MADUREZ o desaceleración"
    elif score_global < -0.2 and score_leadership < -0.2 and score_bonds < -0.2:
        return "📉 Configuración compatible con CONTRACCIÓN"
    elif score_global < -0.5 and score_stress > 0.3:
        return "⚠️ Entorno de CRISIS"
    elif score_global < -0.2 and score_leadership > 0:
        return "🔄 Posible RECUPERACIÓN incipiente"
    else:
        return "⚪ Fase no definida claramente"

# ------------------------------------------------------------
# CAPA 4: SISTEMA DE ALERTAS
# ------------------------------------------------------------
def generar_alertas(df, fila_actual, fila_anterior=None):
    """
    Genera lista de alertas activas.
    """
    alertas = []
    # Si no hay fila anterior, no podemos comparar cambios
    if fila_anterior is None and len(df) >= 2:
        fila_anterior = df.iloc[-2]

    # Alerta 1: Cambio de régimen (score cruza 0)
    if fila_anterior is not None:
        score_actual = fila_actual['score_global']
        score_prev = fila_anterior['score_global']
        if score_prev < 0 and score_actual > 0:
            alertas.append("🔔 CAMBIO DE RÉGIMEN: de negativo a positivo (alcista)")
        elif score_prev > 0 and score_actual < 0:
            alertas.append("🔔 CAMBIO DE RÉGIMEN: de positivo a negativo (bajista)")

    # Alerta 2: Acumulación institucional
    acum = detectar_acumulacion_institucional(fila_actual)
    if "ACUMULACIÓN INSTITUCIONAL" in acum:
        alertas.append("🟢 ACUMULACIÓN INSTITUCIONAL DETECTADA")

    # Alerta 3: Riesgo sistémico
    score_stress = fila_actual.get('score_stress', 0)
    vix = fila_actual.get('vix', 20)
    score_global = fila_actual['score_global']
    if (score_stress > 0.5 or vix > UMBRALES['vix_alto']) and score_global < -0.3:
        alertas.append("🔴 RIESGO SISTÉMICO: estrés elevado y mercado bajista")

    # Alerta 4: Divergencia de motores (dispersión alta)
    dispersion = fila_actual.get('dispersion', 0)
    if dispersion > UMBRALES['dispersion_alerta']:
        alertas.append("🟡 DIVERGENCIA DE MOTORES: alta dispersión, mercado en transición")

    return alertas

# ------------------------------------------------------------
# CAPA 5: GENERACIÓN DE INFORME
# ------------------------------------------------------------
def generar_informe_markdown(datos, alertas, filename="informe_radar.md"):
    """
    Escribe el informe completo en Markdown.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# 📡 INFORME RADAR CAZADOR - {datos['fecha']}\n\n")

        # 1. Estado del régimen
        f.write("## 1. Estado del régimen de mercado\n")
        f.write(f"{datos['analisis']['regimen']}\n\n")

        # 2. Tendencia del capital
        f.write("## 2. Tendencia del capital\n")
        f.write(f"{datos['analisis']['tendencia']}\n\n")

        # 3. Flujo de capital entre activos
        f.write("## 3. Flujo de capital entre activos\n")
        f.write(f"- **Régimen:** {datos['analisis']['regimen']}\n")
        f.write(f"- **Liderazgo:** {datos['analisis']['liderazgo']}\n")
        f.write(f"- **Geográfico:** {datos['analisis']['geografico']}\n")
        f.write(f"- **Crédito:** {datos['analisis']['credito']}\n")
        f.write(f"- **Estrés:** {datos['analisis']['estres']}\n")
        f.write(f"- **Liquidez:** {datos['analisis']['liquidez']}\n\n")

        # 4. Acumulación institucional
        f.write("## 4. Acumulación institucional\n")
        f.write(f"{datos['acumulacion']}\n\n")

        # 5. Fiabilidad de la señal
        f.write("## 5. Fiabilidad de la señal\n")
        f.write(f"{datos['consistencia']} (Dispersión: {datos['fila']['dispersion']:.3f})\n\n")

        # 6. Motor dominante del mercado
        f.write("## 6. Motor dominante del mercado\n")
        if datos['ranking']:
            primero = datos['ranking'][0]
            ultimo = datos['ranking'][-1]
            f.write(f"**Motor dominante:** {primero[0]} ({primero[1]:+.3f})\n")
            f.write(f"**Motor rezagado:** {ultimo[0]} ({ultimo[1]:+.3f})\n")
        else:
            f.write("No hay suficientes datos.\n")
        f.write("\n")

        # 7. Fase del ciclo macro
        f.write("## 7. Fase del ciclo macro\n")
        f.write(f"{datos['fase']}\n\n")

        # 8. Narrativa macro (breve resumen)
        f.write("## 8. Narrativa macro\n")
        # Construimos una narrativa simple
        narrativa = f"El mercado muestra {datos['analisis']['regimen'].lower()}. "
        narrativa += f"La tendencia del capital es {datos['analisis']['tendencia'].lower()}. "
        narrativa += f"El motor dominante es {datos['ranking'][0][0]} ({datos['ranking'][0][1]:+.2f}) y el rezagado es {datos['ranking'][-1][0]} ({datos['ranking'][-1][1]:+.2f}). "
        narrativa += f"La dispersión ({datos['fila']['dispersion']:.2f}) indica {datos['consistencia'].lower()}."
        f.write(narrativa + "\n\n")

        # 9. Alertas
        f.write("## 🚨 Alertas\n")
        if alertas:
            for alerta in alertas:
                f.write(f"- {alerta}\n")
        else:
            f.write("No hay alertas activas.\n")

        f.write("\n---\n")
        f.write("_Informe generado automáticamente por el Asistente de Interpretación V3._\n")

    print(f"✅ Informe generado: {filename}")

# ------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# ------------------------------------------------------------
def main():
    print("=" * 60)
    print("ASISTENTE DE INTERPRETACIÓN MACRO V3")
    print("=" * 60)

    # 1. Cargar datos
    if not os.path.exists('historial_radar.csv'):
        print("❌ Error: No se encuentra historial_radar.csv")
        return

    df = pd.read_csv('historial_radar.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values('fecha')

    if len(df) == 0:
        print("❌ Error: El historial está vacío")
        return

    # 2. Extraer última fila y series necesarias
    fila_actual = df.iloc[-1]
    fecha_actual = fila_actual['fecha'].strftime('%d/%m/%Y')
    print(f"📅 Procesando fecha: {fecha_actual}")

    # Series para cálculos (necesitamos la serie de score_global)
    serie_score = df['score_global'].dropna().reset_index(drop=True)

    # 3. Calcular métricas de preparación
    pendientes = calcular_pendientes(serie_score)
    persistencia = calcular_persistencia(serie_score)
    momentum = calcular_momentum(serie_score)
    ranking = ranking_motores(fila_actual)
    consistencia = clasificar_consistencia(fila_actual.get('dispersion', 0))

    # 4. Mostrar resultados (para depuración)
    print("\n📊 Métricas calculadas:")
    print(f"   Pendientes (3,5,10): {pendientes['pend3']:.4f}, {pendientes['pend5']:.4f}, {pendientes['pend10']:.4f}")
    print(f"   Tendencia ponderada: {pendientes['tendencia_ponderada']:.4f}")
    print(f"   Rachas: ↑{persistencia[0]} días, ↓{persistencia[1]} días")
    print(f"   Momentum (5,10): {momentum['mom5']:.4f}, {momentum['mom10']:.4f}")
    print(f"   Consistencia: {consistencia}")
    print("   Ranking de motores:")
    # 4b. Ejecutar analizadores de señales
    print("\n🔍 Análisis de señales:")
    print(f"   Régimen: {analizar_regimen(fila_actual['score_global'], fila_actual.get('score_regime', 0))}")
    print(f"   Tendencia: {analizar_tendencia(pendientes, persistencia)}")
    print(f"   Liderazgo: {analizar_liderazgo(fila_actual.get('score_leadership', 0))}")
    print(f"   Geográfico: {analizar_geografico(fila_actual.get('score_geographic', 0))}")
    print(f"   Crédito: {analizar_credito(fila_actual.get('score_bonds', 0))}")
    print(f"   Estrés: {analizar_estres(fila_actual.get('score_stress', 0), fila_actual.get('vix', 20))}")
    print(f"   Liquidez: {analizar_liquidez(fila_actual.get('score_liquidity', 0))}")    
    # 4c. Interpretación macro
    fase = clasificar_fase_ciclo(
        fila_actual['score_global'],
        fila_actual.get('score_leadership', 0),
        fila_actual.get('score_bonds', 0),
        fila_actual.get('score_stress', 0)
    )
    acumulacion = detectar_acumulacion_institucional(fila_actual)

    # 4d. Generar alertas
    alertas = generar_alertas(df, fila_actual)

    # Mostrar resultados adicionales
    print(f"\n📈 Fase del ciclo: {fase}")
    print(f"🏦 Acumulación institucional: {acumulacion}")
    if alertas:
        print("\n🚨 ALERTAS:")
        for alerta in alertas:
            print(f"   {alerta}")
    else:
        print("\n🚨 No hay alertas activas.")

    for i, (nombre, valor) in enumerate(ranking, 1):
        print(f"      {i}. {nombre}: {valor:+.3f}")



    # 5. Guardar todo en un diccionario para el informe
    datos_informe = {
        'fecha': fecha_actual,
        'fila': fila_actual,
        'pendientes': pendientes,
        'persistencia': persistencia,
        'momentum': momentum,
        'ranking': ranking,
        'consistencia': consistencia,
        'analisis': {
            'regimen': analizar_regimen(fila_actual['score_global'], fila_actual.get('score_regime', 0)),
            'tendencia': analizar_tendencia(pendientes, persistencia),
            'liderazgo': analizar_liderazgo(fila_actual.get('score_leadership', 0)),
            'geografico': analizar_geografico(fila_actual.get('score_geographic', 0)),
            'credito': analizar_credito(fila_actual.get('score_bonds', 0)),
            'estres': analizar_estres(fila_actual.get('score_stress', 0), fila_actual.get('vix', 20)),
            'liquidez': analizar_liquidez(fila_actual.get('score_liquidity', 0))
        },
        'fase': fase,
        'acumulacion': acumulacion,
        'alertas': alertas
    }

    # 6. Generar informe (por ahora solo placeholder)
    generar_informe_markdown(datos_informe, [])

if __name__ == "__main__":
    main()