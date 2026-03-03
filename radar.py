# radar.py - Versión 3.0 Ultra Simple
# Condiciones:
# 1. Filtro macro: SPY > MA200 (se evalúa aparte, pero lo incluimos)
# 2. Precio > MA150
# 3. Pendiente MA50 positiva en últimos 30 días
# 4. Compresión: vol20/vol60 < 0.65
# 5. Rango 20d < 12% del precio
# 6. Acumulación: volumen alcista/bajista últimos 30d > 1.2
# 7. Confirmación: cierre > máximo 20d

import pandas as pd
import numpy as np
import pickle
import os

# ------------------------------------------------------------
# FUNCIONES AUXILIARES
# ------------------------------------------------------------

def pendiente_ma50_positiva(df, periodos=30):
    """Retorna True si la MA50 tiene pendiente positiva en los últimos 'periodos' días."""
    ma50 = df['close'].rolling(50).mean()
    pendiente = ma50 - ma50.shift(periodos)
    # Devolvemos el último valor (puede ser positivo o negativo)
    return pendiente.iloc[-1] > 0

def ratio_volumen_alcista_bajista(df, ventana=30):
    """
    Suma de volumen en días con retorno positivo / suma de volumen en días con retorno negativo.
    """
    df = df.copy()
    df['ret'] = df['close'].pct_change()
    df['vol_pos'] = np.where(df['ret'] > 0, df['volume'], 0)
    df['vol_neg'] = np.where(df['ret'] < 0, df['volume'], 0)
    suma_pos = df['vol_pos'].rolling(ventana).sum()
    suma_neg = df['vol_neg'].rolling(ventana).sum()
    ratio = suma_pos / (suma_neg + 1e-6)
    return ratio.iloc[-1] > 1.2

def rango_20d_porcentaje(df):
    """(max_20d - min_20d) / precio_actual < 0.12"""
    max_20 = df['high'].rolling(20).max().iloc[-1]
    min_20 = df['low'].rolling(20).min().iloc[-1]
    precio = df['close'].iloc[-1]
    rango_rel = (max_20 - min_20) / precio
    return rango_rel < 0.12

def compresion_volatilidad(df):
    """vol20 / vol60 < 0.65"""
    ret = df['close'].pct_change()
    vol20 = ret.rolling(20).std().iloc[-1]
    vol60 = ret.rolling(60).std().iloc[-1]
    if vol60 == 0:
        return False
    return (vol20 / vol60) < 0.65

def precio_sobre_ma150(df):
    """Precio > MA150"""
    ma150 = df['close'].rolling(150).mean().iloc[-1]
    return df['close'].iloc[-1] > ma150

def breakout_20d(df):
    """Cierre > máximo de 20 días"""
    max_20 = df['high'].rolling(20).max().iloc[-1]
    return df['close'].iloc[-1] > max_20

# ------------------------------------------------------------
# FILTRO MACRO GLOBAL (SPY > MA200)
# ------------------------------------------------------------
def filtro_macro(df_spy):
    """Retorna True si SPY > su MA200."""
    ma200 = df_spy['close'].rolling(200).mean().iloc[-1]
    return df_spy['close'].iloc[-1] > ma200

# ------------------------------------------------------------
# FUNCIÓN PRINCIPAL DE DETECCIÓN
# ------------------------------------------------------------
def detectar_acumulacion(ticker, df, df_spy):
    """
    Evalúa todas las condiciones de la v3.0.
    Retorna un diccionario con el resultado.
    """
    if df.empty or len(df) < 200:
        return {'ticker': ticker, 'señal': False, 'razon': 'Datos insuficientes'}

    condiciones = {
        'precio > MA150': precio_sobre_ma150(df),
        'pendiente MA50 > 0 (30d)': pendiente_ma50_positiva(df),
        'vol20/vol60 < 0.65': compresion_volatilidad(df),
        'rango 20d < 12%': rango_20d_porcentaje(df),
        'volumen alcista/bajista > 1.2': ratio_volumen_alcista_bajista(df),
        'breakout 20d': breakout_20d(df),
    }

    señal = all(condiciones.values())

    return {
        'ticker': ticker,
        'fecha': df.index[-1].strftime('%Y-%m-%d'),
        'señal': señal,
        'condiciones': condiciones,
    }

# ------------------------------------------------------------
# PROCESAR TODOS LOS TICKERS
# ------------------------------------------------------------
def procesar_todos(data_clean):
    """Evalúa todos los tickers y aplica filtro macro."""
    if 'SPY' not in data_clean:
        raise ValueError("No hay datos de SPY para el filtro macro.")

    df_spy = data_clean['SPY']
    macro_ok = filtro_macro(df_spy)
    print(f"📊 Filtro macro (SPY > MA200): {'✅' if macro_ok else '❌'}")

    if not macro_ok:
        print("🔴 Filtro macro NO superado. No se generan señales.")
        return []

    resultados = []
    for ticker, df in data_clean.items():
        if ticker == 'SPY':
            continue
        print(f"   Evaluando {ticker}...")
        res = detectar_acumulacion(ticker, df, df_spy)
        if res['señal']:
            resultados.append(res)

    return resultados

# ------------------------------------------------------------
# EJECUCIÓN PRINCIPAL
# ------------------------------------------------------------
if __name__ == "__main__":
    print("📡 Radar Cazador v3.0 Ultra Simple")
    
    # Cargar datos limpios
    archivo = os.path.join(os.path.dirname(__file__), 'data_clean.pkl')
    if not os.path.exists(archivo):
        print("❌ No se encuentra data_clean.pkl. Ejecuta primero etl.py.")
        exit()

    with open(archivo, 'rb') as f:
        data_clean = pickle.load(f)

    print(f"✅ Datos cargados: {len(data_clean)} tickers.")

    señales = procesar_todos(data_clean)

    print(f"\n📊 Resumen: {len(señales)} señales de compra detectadas.")
    for s in señales:
        print(f"\n✅ {s['ticker']} - {s['fecha']}")
        for cond, cumple in s['condiciones'].items():
            print(f"   {cond}: {'✔️' if cumple else '❌'}")