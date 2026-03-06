#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
asistente_informe_v2.py - Asistente de Interpretación Profesional V2.
Genera un informe ejecutivo detallado en Markdown con recomendaciones concretas.
Adaptado para la versión simplificada del Radar Macro Rotación Global.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# ------------------------------------------------------------
# CONFIGURACIÓN DE UMBRALES (ajústalos según experiencia)
# ------------------------------------------------------------
UMBRALES = {
    'score_alto': 0.5,
    'score_medio': 0.2,
    'score_bajo': -0.2,
    'score_muy_bajo': -0.5,
    'dispersion_alta': 0.8,
    'dispersion_media': 0.5,
    'vix_alto': 30,
    'vix_medio': 20,
    'exposicion_alta': 0.7,
    'exposicion_media': 0.4
}

def calcular_percentil(serie, valor):
    """Calcula el percentil aproximado de un valor dentro de una serie."""
    if len(serie) < 2:
        return 50
    return (serie < valor).mean() * 100

def interpretar_score(score):
    """Interpreta el score global en texto."""
    if score > UMBRALES['score_alto']:
        return "🟢 ALCISTA FUERTE", "muy positivo"
    elif score > UMBRALES['score_medio']:
        return "🟢 ALCISTA DÉBIL", "positivo"
    elif score > UMBRALES['score_bajo']:
        return "⚪ NEUTRAL", "neutral"
    elif score > UMBRALES['score_muy_bajo']:
        return "🔴 BAJISTA DÉBIL", "negativo"
    else:
        return "🔴 BAJISTA FUERTE", "muy negativo"

def interpretar_dispersion(disp):
    """Interpreta la dispersión."""
    if disp > UMBRALES['dispersion_alta']:
        return "🔴 MUY ALTA (divergencia fuerte)"
    elif disp > UMBRALES['dispersion_media']:
        return "🟡 MEDIA-ALTA (dudas)"
    else:
        return "🟢 BAJA (consenso)"

def interpretar_vix(vix):
    """Interpreta VIX."""
    if vix > UMBRALES['vix_alto']:
        return "🔴 ALTO (estrés)"
    elif vix > UMBRALES['vix_medio']:
        return "🟡 MEDIO (cautela)"
    else:
        return "🟢 BAJO (tranquilidad)"

def generar_informe():
    # Verificar existencia del historial
    if not os.path.exists('historial_radar.csv'):
        print("❌ Error: No se encuentra historial_radar.csv")
        return False

    # Cargar datos
    df = pd.read_csv('historial_radar.csv')
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values('fecha')

    if len(df) == 0:
        print("❌ Error: El historial está vacío")
        return False

    ultimo = df.iloc[-1]
    fecha_ult = ultimo['fecha'].strftime('%d/%m/%Y')

    # Calcular percentiles
    percentil_score = calcular_percentil(df['score_global'], ultimo['score_global'])
    percentil_expo = calcular_percentil(df['exposicion_final'], ultimo['exposicion_final'])
    percentil_disp = calcular_percentil(df['dispersion'], ultimo['dispersion'])

    # Últimos 5 días
    ultimos_5 = df.tail(5).copy()

    # Cambios respecto al día anterior
    if len(df) >= 2:
        penultimo = df.iloc[-2]
        cambio_score = ultimo['score_global'] - penultimo['score_global']
        cambio_expo = ultimo['exposicion_final'] - penultimo['exposicion_final']
    else:
        cambio_score = 0
        cambio_expo = 0

    # Interpretaciones
    score_texto, score_adj = interpretar_score(ultimo['score_global'])
    dispersion_texto = interpretar_dispersion(ultimo['dispersion'])
    vix_texto = interpretar_vix(ultimo['vix']) if 'vix' in ultimo else "N/A"

    # Determinar escenario
    if ultimo['score_global'] > UMBRALES['score_medio'] and ultimo['dispersion'] < UMBRALES['dispersion_media']:
        escenario = "alcista con consenso"
        confianza = "alta"
    elif ultimo['score_global'] > UMBRALES['score_medio'] and ultimo['dispersion'] > UMBRALES['dispersion_alta']:
        escenario = "alcista con divergencias"
        confianza = "baja"
    elif ultimo['score_global'] < UMBRALES['score_bajo'] and ultimo['dispersion'] < UMBRALES['dispersion_media']:
        escenario = "bajista con consenso"
        confianza = "alta"
    elif ultimo['score_global'] < UMBRALES['score_bajo'] and ultimo['dispersion'] > UMBRALES['dispersion_alta']:
        escenario = "bajista con divergencias"
        confianza = "baja"
    else:
        escenario = "neutral o mixto"
        confianza = "moderada"

    # Ranking de motores (usar las columnas disponibles)
    motores = []
    if 'score_regime' in ultimo:
        motores.append(('Régimen', ultimo['score_regime']))
    if 'score_leadership' in ultimo:
        motores.append(('Liderazgo', ultimo['score_leadership']))
    if 'score_geographic' in ultimo:
        motores.append(('Geográfico', ultimo['score_geographic']))
    if 'score_bonds' in ultimo:
        motores.append(('Bonos', ultimo['score_bonds']))
    if 'score_stress' in ultimo:
        motores.append(('Estrés', ultimo['score_stress']))
    if 'score_liquidity' in ultimo:
        motores.append(('Liquidez', ultimo['score_liquidity']))

    # Ordenar de mayor a menor
    motores_ordenados = sorted(motores, key=lambda x: x[1], reverse=True)

    # Recomendaciones según perfil
    def recomendacion_agresivo():
        expo = ultimo['exposicion_final']
        if expo > 0.6 and ultimo['score_global'] > UMBRALES['score_medio']:
            return "✅ Totalmente invertido. Aprovecha el impulso."
        elif expo > 0.4 and ultimo['score_global'] > UMBRALES['score_medio']:
            return "📈 Puedes aumentar hasta 60-70% si el score sigue subiendo."
        elif expo < 0.3 or ultimo['score_global'] < UMBRALES['score_bajo']:
            return "🔴 Reduce a mínimo (0-20%). Mal momento."
        else:
            return "🟡 Mantén exposición actual. Sin señales claras."

    def recomendacion_conservador():
        expo = ultimo['exposicion_final']
        if expo > 0.6:
            return "⚠️ Exposición alta para perfil conservador. Reduce a 40-50%."
        elif expo > 0.4 and ultimo['score_global'] > UMBRALES['score_medio']:
            return "🟢 Puedes mantener hasta 40-50%."
        elif expo < 0.3:
            return "✅ Bien, mantén liquidez."
        else:
            return "🟡 Espera señales más claras."

    def recomendacion_trader():
        if ultimo['dispersion'] < UMBRALES['dispersion_media'] and abs(ultimo['score_global']) > UMBRALES['score_medio']:
            return "📊 Tendencia clara. Puedes operar con confianza."
        elif ultimo['dispersion'] > UMBRALES['dispersion_alta']:
            return "⚠️ Evitar operar. Alta incertidumbre."
        else:
            return "🟡 Mercado lateral. Poco ruido, poco beneficio."

    # ------------------- GENERAR INFORME -------------------
    with open('informe_radar.md', 'w', encoding='utf-8') as f:
        f.write(f"# 📡 INFORME EJECUTIVO DEL RADAR - {fecha_ult}\n")
        f.write(f"_Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}_\n\n")

        # Resumen ejecutivo
        f.write("## 🧠 RESUMEN PARA TOMAR DECISIONES (LÉE ESTO PRIMERO)\n\n")
        f.write(f"**ESCENARIO ACTUAL:**  \n")
        f.write(f"✅ MERCADO {score_texto.upper()} (Score: {ultimo['score_global']:+.2f})  \n")
        f.write(f"⚠️ CONFIANZA: {confianza.upper()} (Dispersión: {ultimo['dispersion']:.2f})  \n")
        if 'vix' in ultimo:
            f.write(f"📊 VIX: {vix_texto} ({ultimo['vix']:.1f})  \n")
        f.write("\n")

        f.write("**RECOMENDACIÓN:**  \n")
        if confianza == "alta" and ultimo['score_global'] > UMBRALES['score_medio']:
            f.write(f"👉 **AUMENTAR EXPOSICIÓN HASTA {min(ultimo['exposicion_final']+0.2, 1.0)*100:.0f}%**\n")
        elif confianza == "alta" and ultimo['score_global'] < UMBRALES['score_bajo']:
            f.write(f"👉 **REDUCIR EXPOSICIÓN A {max(ultimo['exposicion_final']-0.3, 0.0)*100:.0f}% O MENOS**\n")
        else:
            f.write(f"👉 **MANTENER EXPOSICIÓN ACTUAL ({ultimo['exposicion_final']*100:.1f}%)**\n")

        f.write("**¿POR QUÉ?**  \n")
        if motores_ordenados:
            f.write(f"- El motor **{motores_ordenados[0][0]}** está muy fuerte ({motores_ordenados[0][1]:+.2f}), pero  \n")
            f.write(f"- El motor **{motores_ordenados[-1][0]}** está débil ({motores_ordenados[-1][1]:+.2f}).  \n")
        if ultimo['dispersion'] > UMBRALES['dispersion_media']:
            f.write("- La dispersión indica que los motores no están alineados → cautela.\n")
        else:
            f.write("- Hay consenso entre motores → confianza.\n")
        f.write("\n")

        # Panel de control
        f.write("## 📊 PANEL DE CONTROL\n")
        f.write("| Indicador | Valor | ¿Qué significa? |\n")
        f.write("|-----------|-------|-----------------|\n")
        f.write(f"| **Score Global** | {ultimo['score_global']:+.4f} | {score_texto} (percentil {percentil_score:.0f}%) |\n")
        f.write(f"| **Dispersión** | {ultimo['dispersion']:.4f} | {dispersion_texto} |\n")
        f.write(f"| **Exposición recomendada** | {ultimo['exposicion_final']*100:.1f}% | {('Alta' if ultimo['exposicion_final']>UMBRALES['exposicion_alta'] else 'Moderada' if ultimo['exposicion_final']>UMBRALES['exposicion_media'] else 'Baja')} |\n")
        if 'vix' in ultimo:
            f.write(f"| **VIX** | {ultimo['vix']:.1f} | {vix_texto} |\n")
        f.write("\n")

        # Análisis por motor
        f.write("## 🔍 ANÁLISIS POR MOTOR (QUÉ ESTÁ PASANDO DENTRO)\n\n")
        for nombre, valor in motores:
            f.write(f"### {nombre}\n")
            if valor > 0.3:
                f.write(f"- {nombre} es **MUY POSITIVO**. ")
                if nombre == "Régimen":
                    f.write("Tendencia macro favorable.\n")
                elif nombre == "Liderazgo":
                    f.write("Rotación hacia small caps.\n")
                elif nombre == "Geográfico":
                    f.write("Flujo hacia emergentes/desarrollados.\n")
                elif nombre == "Bonos":
                    f.write("Apetito por crédito.\n")
                elif nombre == "Estrés":
                    f.write("Sin estrés, mercado tranquilo.\n")
                elif nombre == "Liquidez":
                    f.write("Dólar débil, liquidez abundante.\n")
            elif valor > 0:
                f.write(f"- {nombre} es **LIGERAMENTE POSITIVO**.\n")
            elif valor > -0.3:
                f.write(f"- {nombre} es **NEUTRAL**.\n")
            else:
                f.write(f"- {nombre} es **NEGATIVO**. ")
                if nombre == "Régimen":
                    f.write("Deterioro macro.\n")
                elif nombre == "Liderazgo":
                    f.write("Large caps dominan (defensivo).\n")
                elif nombre == "Geográfico":
                    f.write("Flujo hacia USA (aversión al riesgo).\n")
                elif nombre == "Bonos":
                    f.write("Huida a calidad.\n")
                elif nombre == "Estrés":
                    f.write("Alerta de volatilidad.\n")
                elif nombre == "Liquidez":
                    f.write("Dólar fuerte, liquidez ajustada.\n")
            f.write("\n")

        # Evolución reciente
        f.write("## 📈 EVOLUCIÓN RECIENTE (ÚLTIMOS 5 DÍAS)\n")
        f.write("| Fecha | Score | Exposición | ¿Qué pasó? |\n")
        f.write("|-------|-------|------------|------------|\n")
        for i, (idx, row) in enumerate(ultimos_5.iterrows()):
            if i == len(ultimos_5)-1:
                comentario = "📈 Hoy"
            elif i == len(ultimos_5)-2:
                if cambio_score > 0.02:
                    comentario = "📈 Subió"
                elif cambio_score < -0.02:
                    comentario = "📉 Bajó"
                else:
                    comentario = "🔄 Estable"
            else:
                comentario = "—"
            f.write(f"| {row['fecha'].strftime('%d/%m')} | {row['score_global']:+.2f} | {row['exposicion_final']*100:.1f}% | {comentario} |\n")
        f.write("\n")

        # Señales detectadas (simplificadas)
        f.write("## 🚦 SEÑALES DE TRADING DETECTADAS\n")
        f.write("| Señal | Activada | Significado |\n")
        f.write("|-------|----------|-------------|\n")
        f.write(f"| 📈 **Tendencia alcista con consenso** | {'✅ SÍ' if ultimo['score_global'] > UMBRALES['score_medio'] and ultimo['dispersion'] < UMBRALES['dispersion_media'] else '❌ NO'} | Fuerte señal alcista |\n")
        f.write(f"| 📉 **Tendencia bajista con consenso** | {'✅ SÍ' if ultimo['score_global'] < UMBRALES['score_bajo'] and ultimo['dispersion'] < UMBRALES['dispersion_media'] else '❌ NO'} | Fuerte señal bajista |\n")
        f.write(f"| ⚠️ **Alta dispersión con score positivo** | {'✅ SÍ' if ultimo['dispersion'] > UMBRALES['dispersion_alta'] and ultimo['score_global'] > UMBRALES['score_medio'] else '❌ NO'} | Cambio de régimen alcista |\n")
        f.write(f"| ⚠️ **Alta dispersión con score negativo** | {'✅ SÍ' if ultimo['dispersion'] > UMBRALES['dispersion_alta'] and ultimo['score_global'] < UMBRALES['score_bajo'] else '❌ NO'} | Riesgo elevado |\n")
        if 'vix' in ultimo:
            f.write(f"| 😨 **VIX alto (>{UMBRALES['vix_alto']})** | {'✅ SÍ' if ultimo['vix'] > UMBRALES['vix_alto'] else '❌ NO'} | Entorno de estrés |\n")
        f.write("\n")

        # Ranking de motores
        if motores_ordenados:
            f.write("## 🏆 ¿QUÉ MOTOR ESTÁ DOMINANDO HOY?\n\n")
            f.write("**RANKING DE INFLUENCIA:**\n\n")
            for i, (nombre, valor) in enumerate(motores_ordenados, 1):
                if valor > 0.3:
                    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🎯"
                    f.write(f"{i}. {emoji} **{nombre}** - {valor:+.2f} (MUY POSITIVO)\n")
                elif valor > 0:
                    f.write(f"{i}. **{nombre}** - {valor:+.2f} (POSITIVO)\n")
                elif valor > -0.3:
                    f.write(f"{i}. **{nombre}** - {valor:+.2f} (NEUTRAL)\n")
                else:
                    f.write(f"{i}. **{nombre}** - {valor:+.2f} (NEGATIVO)\n")
            f.write("\n")

        # Recomendaciones personalizadas
        f.write("## 🎯 RECOMENDACIONES POR PERFIL\n\n")
        f.write("### 📌 PARA INVERSORES AGRESIVOS:\n")
        f.write(recomendacion_agresivo() + "\n\n")
        f.write("### 📌 PARA INVERSORES CONSERVADORES:\n")
        f.write(recomendacion_conservador() + "\n\n")
        f.write("### 📌 PARA INVERSORES QUE QUIEREN TRADING:\n")
        f.write(recomendacion_trader() + "\n\n")

        # Checklist rápido
        f.write("## 📋 CHECKLIST RÁPIDO PARA HOY\n\n")
        checklist = [
            f"☑️ ¿Mercado? → **{score_texto.split()[1]}**",
            f"☑️ ¿Confianza? → **{confianza.capitalize()}**",
            f"☑️ ¿Exposición correcta? → **{ultimo['exposicion_final']*100:.1f}%**",
            f"☑️ ¿Volatilidad? → **{vix_texto.split()[0] if 'vix' in ultimo else 'N/A'}**",
        ]
        if motores_ordenados:
            checklist.append(f"☑️ ¿Motor dominante? → **{motores_ordenados[0][0]}**")
            checklist.append(f"☑️ ¿Motor que preocupa? → **{motores_ordenados[-1][0]}**")
        for item in checklist:
            f.write(item + "\n")
        f.write("\n")

        # Frase resumen
        f.write("## 🧠 RESUMEN EN UNA FRASE\n\n")
        if confianza == "alta" and ultimo['score_global'] > UMBRALES['score_medio']:
            frase = "El mercado sube con confianza. Aumenta exposición."
        elif confianza == "alta" and ultimo['score_global'] < UMBRALES['score_bajo']:
            frase = "Mercado bajista claro. Reduce posiciones."
        elif confianza == "baja" and ultimo['score_global'] > UMBRALES['score_medio']:
            frase = "Sube pero con dudas. Mantén exposición moderada."
        elif confianza == "baja" and ultimo['score_global'] < UMBRALES['score_bajo']:
            frase = "Cae con divergencias. Mucha precaución."
        else:
            frase = "Mercado sin dirección clara. Espera."
        f.write(f"> **{frase}**\n\n")

        f.write("---\n")
        f.write("_Informe generado automáticamente por el Asistente de Interpretación V2._\n")

    print("✅ Informe generado: informe_radar.md")
    return True

if __name__ == "__main__":
    generar_informe()