# dashboard.py - Dashboard interactivo para el Radar Macro Rotación Global (versión simplificada)
# Lee el historial_radar.csv generado por run_radar.py y muestra gráficos interactivos.

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta

st.set_page_config(page_title="Radar Macro Rotación Global", layout="wide")
st.title("📡 Radar Macro Rotación Global - Dashboard Interactivo")
st.markdown("**Versión simplificada** · Datos actualizados diariamente.")

# Cargar datos
historial_path = "historial_radar.csv"
if not os.path.exists(historial_path):
    st.error("No se encuentra el archivo `historial_radar.csv`. Ejecuta primero `run_radar.py`.")
    st.stop()

df = pd.read_csv(historial_path, parse_dates=["fecha"])
df = df.sort_values("fecha")

# Información general
ultima_fecha = df["fecha"].max()
st.sidebar.header("📅 Filtros")
st.sidebar.info(f"Última fecha disponible: {ultima_fecha.date()}")

# Selector de rango de fechas
fecha_min = df["fecha"].min().date()
fecha_max = df["fecha"].max().date()

# Valor por defecto para inicio: últimos 90 días, pero sin salir del rango disponible
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

# Métricas principales (último día)
ultimo = df_filtrado.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Score global", f"{ultimo['score_global']:.3f}")
col2.metric("Exposición final", f"{ultimo['exposicion_final']:.2%}")
col3.metric("Dispersión", f"{ultimo['dispersion']:.3f}")
col4.metric("Factor exposición (scoring)", f"{ultimo['exposure_factor']:.2%}")

# Gráfico 1: Score global y suavizado
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df_filtrado["fecha"], y=df_filtrado["score_global"],
                           mode="lines", name="Score global", line=dict(color="royalblue")))
fig1.add_trace(go.Scatter(x=df_filtrado["fecha"], y=df_filtrado["score_smoothed"],
                           mode="lines", name="Score suavizado", line=dict(color="orange", dash="dash")))
fig1.update_layout(title="Evolución del Score Global", xaxis_title="Fecha", yaxis_title="Score",
                   hovermode="x unified", height=400)
st.plotly_chart(fig1, use_container_width=True)

# Gráfico 2: Exposición y factor de exposición
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_filtrado["fecha"], y=df_filtrado["exposicion_final"],
                           mode="lines", name="Exposición final", line=dict(color="green")))
fig2.add_trace(go.Scatter(x=df_filtrado["fecha"], y=df_filtrado["exposure_factor"],
                           mode="lines", name="Factor de exposición (scoring)", line=dict(color="red", dash="dot")))
fig2.update_layout(title="Exposición y Factor de Exposición", xaxis_title="Fecha", yaxis_title="Factor",
                   hovermode="x unified", height=400)
st.plotly_chart(fig2, use_container_width=True)

# Gráfico 3: Dispersión
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df_filtrado["fecha"], y=df_filtrado["dispersion"],
                           mode="lines", name="Dispersión", fill='tozeroy', line=dict(color="purple")))
fig3.update_layout(title="Dispersión entre Módulos", xaxis_title="Fecha", yaxis_title="Dispersión",
                   hovermode="x unified", height=300)
st.plotly_chart(fig3, use_container_width=True)

# Mostrar tabla de últimos registros
st.subheader("📋 Últimos 20 registros")
st.dataframe(
    df_filtrado.tail(20)[["fecha", "score_global", "score_smoothed", "dispersion", 
                           "exposure_factor", "exposicion_final"]]
    .style.format({
        "score_global": "{:.3f}",
        "score_smoothed": "{:.3f}",
        "dispersion": "{:.3f}",
        "exposure_factor": "{:.2%}",
        "exposicion_final": "{:.2%}"
    }),
    use_container_width=True,
    height=400
)

# Botón para descargar datos filtrados
csv = df_filtrado.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Descargar datos filtrados (CSV)",
    data=csv,
    file_name=f"radar_{fecha_inicio}_{fecha_fin}.csv",
    mime="text/csv"
)