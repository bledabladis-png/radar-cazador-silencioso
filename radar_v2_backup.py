# radar.py - Módulo de detección de acumulación
# Implementa las 4 condiciones del sistema v2.0

import pandas as pd
import numpy as np
import pickle
import os

# ------------------------------------------------------------
# FUNCIONES DE CÁLCULO
# ------------------------------------------------------------

def calcular_ma150(df):
    """Media móvil de 150 días."""
    return df['close'].rolling(150).mean()

def calcular_pendiente_ma50(df, periodos=20):
    """Pendiente de la MA50 en los últimos 'periodos' días."""
    ma50 = df['close'].rolling(50).mean()
    pendiente = ma50 - ma50.shift(periodos)
    return pendiente

def calcular_volatilidad(df, ventana):
    """Volatilidad histórica (desviación típica de retornos)."""
    return df['return'].rolling(ventana).std()

def calcular_volumen_positivo_negativo(df, ventana=20):
    """
    Suma de volumen en días con retorno positivo / suma de volumen en días con retorno negativo.
    """
    df = df.copy()
    df['vol_pos'] = np.where(df['return'] > 0, df['volume'], 0)
    df['vol_neg'] = np.where(df['return'] < 0, df['volume'], 0)
    suma_pos = df['vol_pos'].rolling(ventana).sum()
    suma_neg = df['vol_neg'].rolling(ventana).sum()
    # Evitar división por cero
    ratio = suma_pos / (suma_neg + 1e-6)
    return ratio

def calcular_rs(df_ticker, df_spy, ventana=60):
    """Fuerza relativa: retorno del ticker - retorno del SPY."""
    ret_ticker = df_ticker['close'].pct_change(ventana)
    ret_spy = df_spy['close'].pct_change(ventana)
    return ret_ticker - ret_spy

def calcular_rango(df, ventana):
    """Rango relativo: (max - min) / close."""
    maximo = df['high'].rolling(ventana).max()
    minimo = df['low'].rolling(ventana).min()
    return (maximo - minimo) / df['close']

# ------------------------------------------------------------
# FUNCIÓN PRINCIPAL DE DETECCIÓN
# ------------------------------------------------------------

def detectar_acumulacion(ticker, df, df_spy, fecha=None):
    """
    Evalúa las 4 condiciones de acumulación en la última fecha disponible.
    Si se proporciona 'fecha', se evalúa en esa fecha concreta.
    Retorna un diccionario con:
        - 'señal': True si cumple todas las condiciones
        - 'condiciones': detalle de cada condición
    """
    if df.empty or len(df) < 150:
        return {'señal': False, 'condiciones': {}, 'error': 'Datos insuficientes'}

    # Si se especifica fecha, limitar los datos hasta esa fecha
    if fecha is not None:
        df = df[df.index <= pd.Timestamp(fecha)].copy()
        df_spy = df_spy[df_spy.index <= pd.Timestamp(fecha)].copy()
        if df.empty or df_spy.empty:
            return {'señal': False, 'condiciones': {}, 'error': 'Sin datos hasta la fecha'}

    # Obtener la última fila
    ultimo = df.iloc[-1]
    fecha_actual = df.index[-1]

    # Calcular indicadores
    ma150 = calcular_ma150(df).iloc[-1]
    pendiente_ma50 = calcular_pendiente_ma50(df).iloc[-1]
    vol20 = calcular_volatilidad(df, 20).iloc[-1]
    vol60 = calcular_volatilidad(df, 60).iloc[-1]
    acc_ratio = calcular_volumen_positivo_negativo(df, 20).iloc[-1]
    rs60 = calcular_rs(df, df_spy, 60).iloc[-1]
    rango20 = calcular_rango(df, 20).iloc[-1]
    rango60 = calcular_rango(df, 60).iloc[-1]

    # Evaluar condiciones
    condiciones = {
        'precio > MA150': ultimo['close'] > ma150,
        'pendiente MA50 > 0': pendiente_ma50 > 0,
        'vol20 < vol60': vol20 < vol60,
        'acc_ratio > 1.2': acc_ratio > 1.2,
        'RS60 > 0': rs60 > 0,
        'compresión': rango20 < 0.6 * rango60,
    }

    señal = all(condiciones.values())

    # Preparar resultado
    resultado = {
        'ticker': ticker,
        'fecha': fecha_actual.strftime('%Y-%m-%d'),
        'señal': señal,
        'condiciones': condiciones,
        'valores': {
            'close': ultimo['close'],
            'ma150': ma150,
            'pendiente_ma50': pendiente_ma50,
            'vol20': vol20,
            'vol60': vol60,
            'acc_ratio': acc_ratio,
            'rs60': rs60,
            'rango20': rango20,
            'rango60': rango60,
        }
    }
    return resultado

# ------------------------------------------------------------
# FUNCIÓN PARA PROCESAR TODOS LOS TICKERS
# ------------------------------------------------------------

def procesar_todos(data_clean, ticker_spy='SPY'):
    """
    Evalúa la acumulación para todos los tickers en data_clean.
    Retorna una lista de resultados.
    """
    if ticker_spy not in data_clean:
        raise ValueError(f"El ticker {ticker_spy} no está en los datos.")

    df_spy = data_clean[ticker_spy]
    resultados = []

    for ticker, df in data_clean.items():
        if ticker == ticker_spy:
            continue  # No evaluamos SPY contra sí mismo
        print(f"   Evaluando {ticker}...")
        res = detectar_acumulacion(ticker, df, df_spy)
        resultados.append(res)

    return resultados

# ------------------------------------------------------------
# SI SE EJECUTA COMO SCRIPT PRINCIPAL
# ------------------------------------------------------------

if __name__ == "__main__":
    print("📡 Radar Cazador v2.0 - Detección de acumulación")
    
    # Cargar datos limpios
    archivo_datos = os.path.join(os.path.dirname(__file__), 'data_clean.pkl')
    if not os.path.exists(archivo_datos):
        print("❌ No se encuentra data_clean.pkl. Ejecuta primero etl.py.")
        exit()

    with open(archivo_datos, 'rb') as f:
        data_clean = pickle.load(f)

    print(f"✅ Datos cargados: {len(data_clean)} tickers.")

    # Procesar todos los tickers
    resultados = procesar_todos(data_clean)

    # Mostrar resumen
    señales = [r for r in resultados if r['señal']]
    print(f"\n📊 Resumen: {len(señales)} señales de compra detectadas.")

    for r in señales:
        print(f"\n✅ {r['ticker']} - {r['fecha']}")
        for cond, cumple in r['condiciones'].items():
            print(f"   {cond}: {'✔️' if cumple else '❌'}")