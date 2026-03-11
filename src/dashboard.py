# dashboard.py - Dashboard interactivo V9 para el Radar Macro Rotación Global
# Incluye gráfico de contribución de factores y evolución de motores avanzados.

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import numpy as np
import yaml

st.set_page_config(page_title="Radar Macro Rotación Global V9", layout="wide")
st.title("📡 Radar Macro Rotación Global - Dashboard V9")
st.markdown("**Dashboard avanzado** · Datos actualizados diariamente.")

# -------------------------------------------------------------------
# Cargar datos
# -------------------------------------------------------------------
historial_path = "historial_radar.csv"
if not os.path.exists(historial_path):
    st.error("No se encuentra el archivo `historial_radar.csv`. Ejecuta primero `run_radar.py`.")
    st.stop()

df = pd.read_csv(historial_path, parse_dates=["fecha"])
df = df.sort_values("fecha")

if df.empty:
    st.error("El archivo de historial está vacío.")
    st.stop()

# -------------------------------------------------------------------
# Validación de columnas esenciales y opcionales
# -------------------------------------------------------------------
columnas_esenciales = ['fecha', 'score_global', 'score_smoothed']
columnas_opcionales = ['exposicion_final', 'dispersion', 'fase_ciclo', 'ciclo_institucional',
                       'score_riesgo_sistemico', 'score_carry', 'exposure_factor',
                       'score_regime', 'score_leadership', 'score_geographic', 'score_bonds',
                       'score_stress', 'score_liquidity', 'score_breadth',
                       'score_global_liquidity', 'score_cftc', 'score_breadth_advanced',
                       'score_etf_flow', 'score_financial_conditions']

missing_essential = [col for col in columnas_esenciales if col not in df.columns]
if missing_essential:
    st.error(f"Faltan columnas esenciales en el CSV: {missing_essential}")
    st.stop()

for col in columnas_opcionales:
    if col not in df.columns:
        st.warning(f"Columna opcional '{col}' no encontrada. Algunos gráficos pueden no mostrarse.")
        df[col] = 0  # Valor por defecto para evitar errores

# -------------------------------------------------------------------
# Información general
# -------------------------------------------------------------------
ultima_fecha = df["fecha"].max()
st.sidebar.header("📅 Filtros")
st.sidebar.info(f"Última fecha disponible: {ultima_fecha.date()}")

# Selector de rango de fechas
fecha_min = df["fecha"].min().date()
fecha_max = df["fecha"].max().date()
default_inicio = fecha_max - timedelta(days=90)
if default_inicio < fecha_min:
    default_inicio = fecha_min

fecha_inicio = st.sidebar.date_input("Fecha inicio", default_inicio,
                                      min_value=fecha_min, max_value=fecha_max)
fecha_fin = st.sidebar.date_input("Fecha fin", fecha_max,
                                   min_value=fecha_min, max_value=fecha_max)

# Filtrar
mask = (df["fecha"] >= pd.Timestamp(fecha_inicio)) & (df["fecha"] <= pd.Timestamp(fecha_fin))
df_filtrado = df.loc[mask].copy()

if df_filtrado.empty:
    st.warning("No hay datos en el rango seleccionado.")
    st.stop()

# -------------------------------------------------------------------
# Métricas principales (último día)
# -------------------------------------------------------------------
ultimo = df_filtrado.iloc[-1]
st.subheader("📊 Métricas principales")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Score global", f"{ultimo['score_global']:.3f}")
col2.metric("Score suavizado", f"{ultimo['score_smoothed']:.3f}")
col3.metric("Exposición final", f"{ultimo.get('exposicion_final', 0):.2%}")
col4.metric("Dispersión", f"{ultimo.get('dispersion', 0):.3f}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Fase ciclo", ultimo.get('fase_ciclo', 'N/A'))
col6.metric("Ciclo institucional", ultimo.get('ciclo_institucional', 'N/A'))
col7.metric("Riesgo sistémico", f"{ultimo.get('score_riesgo_sistemico', 0):.3f}")
col8.metric("Carry trade", f"{ultimo.get('score_carry', 0):.3f}")

# -------------------------------------------------------------------
# Gráfico 1: Score global y suavizado
# -------------------------------------------------------------------
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df_filtrado["fecha"], y=df_filtrado["score_global"],
                           mode="lines", name="Score global", line=dict(color="royalblue")))
fig1.add_trace(go.Scatter(x=df_filtrado["fecha"], y=df_filtrado["score_smoothed"],
                           mode="lines", name="Score suavizado", line=dict(color="orange", dash="dash")))
fig1.update_layout(title="Evolución del Score Global", xaxis_title="Fecha", yaxis_title="Score",
                   hovermode="x unified", height=400)
st.plotly_chart(fig1, use_container_width=True)

# -------------------------------------------------------------------
# Gráfico 2: Contribución de factores (motores base)
# -------------------------------------------------------------------
st.subheader("📊 Contribución de factores al score global")

# Identificar motores base (los que tienen score_...)
motores_base = ['score_regime', 'score_leadership', 'score_geographic', 'score_bonds',
                'score_stress', 'score_liquidity', 'score_breadth']
motores_presentes = [m for m in motores_base if m in df_filtrado.columns]

if motores_presentes:
    # Cargar pesos base desde configuración
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            pesos_base = config.get('weights', {}).get('base', {})
    except:
        pesos_base = {}

    # Calcular contribución aproximada (score * peso)
    contribuciones = pd.DataFrame(index=df_filtrado.index)
    for motor in motores_presentes:
        nombre_corto = motor.replace('score_', '')
        peso = pesos_base.get(nombre_corto, 1/len(motores_presentes))
        contribuciones[nombre_corto] = df_filtrado[motor] * peso

    # Área apilada
    fig2 = go.Figure()
    for col in contribuciones.columns:
        fig2.add_trace(go.Scatter(
            x=contribuciones.index,
            y=contribuciones[col],
            mode='lines',
            stackgroup='one',
            name=col.capitalize(),
            line=dict(width=0.5)
        ))
    fig2.update_layout(
        title="Contribución de motores base al score global (ponderada por pesos base)",
        xaxis_title="Fecha",
        yaxis_title="Contribución",
        hovermode="x unified",
        height=500
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("No hay datos de motores base para el gráfico de contribución.")

# -------------------------------------------------------------------
# Gráfico 3: Evolución de motores avanzados
# -------------------------------------------------------------------
st.subheader("📈 Motores avanzados")

motores_avanzados = ['score_riesgo_sistemico', 'score_carry', 'score_global_liquidity',
                     'score_cftc', 'score_breadth_advanced', 'score_etf_flow',
                     'score_financial_conditions']
nombres_avanzados = ['Riesgo Sistémico', 'Carry Trade', 'Liquidez Global (BIS)',
                     'CFTC', 'Breadth Avanzado', 'ETF Flows', 'Condiciones Financieras']

fig3 = go.Figure()
for i, motor in enumerate(motores_avanzados):
    if motor in df_filtrado.columns and not df_filtrado[motor].isnull().all():
        fig3.add_trace(go.Scatter(x=df_filtrado["fecha"], y=df_filtrado[motor],
                                   mode="lines", name=nombres_avanzados[i]))
if len(fig3.data) == 0:
    fig3.add_annotation(text="No hay datos de motores avanzados", showarrow=False)
fig3.update_layout(title="Evolución de motores avanzados", xaxis_title="Fecha", yaxis_title="Score",
                   hovermode="x unified", height=500)
st.plotly_chart(fig3, use_container_width=True)

# -------------------------------------------------------------------
# Gráfico 4: Exposición y factor de exposición
# -------------------------------------------------------------------
fig4 = go.Figure()
if 'exposicion_final' in df_filtrado.columns:
    fig4.add_trace(go.Scatter(x=df_filtrado["fecha"], y=df_filtrado["exposicion_final"],
                               mode="lines", name="Exposición final", line=dict(color="green")))
if 'exposure_factor' in df_filtrado.columns:
    fig4.add_trace(go.Scatter(x=df_filtrado["fecha"], y=df_filtrado["exposure_factor"],
                               mode="lines", name="Factor de exposición (scoring)", line=dict(color="red", dash="dot")))
fig4.update_layout(title="Exposición y Factor de Exposición", xaxis_title="Fecha", yaxis_title="Factor",
                   hovermode="x unified", height=400)
st.plotly_chart(fig4, use_container_width=True)

# -------------------------------------------------------------------
# Gráfico 5: Dispersión
# -------------------------------------------------------------------
if 'dispersion' in df_filtrado.columns:
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=df_filtrado["fecha"], y=df_filtrado["dispersion"],
                               mode="lines", name="Dispersión", fill='tozeroy', line=dict(color="purple")))
    fig5.update_layout(title="Dispersión entre Módulos", xaxis_title="Fecha", yaxis_title="Dispersión",
                       hovermode="x unified", height=300)
    st.plotly_chart(fig5, use_container_width=True)

# -------------------------------------------------------------------
# Tabla de últimos registros
# -------------------------------------------------------------------
st.subheader("📋 Últimos 20 registros")
columnas_mostrar = ['fecha', 'score_global', 'score_smoothed', 'dispersion',
                    'exposure_factor', 'exposicion_final', 'fase_ciclo', 'ciclo_institucional']
columnas_existentes = [c for c in columnas_mostrar if c in df_filtrado.columns]
st.dataframe(
    df_filtrado.tail(20)[columnas_existentes].style.format({
        'score_global': '{:.3f}',
        'score_smoothed': '{:.3f}',
        'dispersion': '{:.3f}',
        'exposure_factor': '{:.2%}',
        'exposicion_final': '{:.2%}',
    }),
    use_container_width=True,
    height=400
)

# -------------------------------------------------------------------
# Botón para descargar datos filtrados
# -------------------------------------------------------------------
csv = df_filtrado.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Descargar datos filtrados (CSV)",
    data=csv,
    file_name=f"radar_{fecha_inicio}_{fecha_fin}.csv",
    mime="text/csv"
)