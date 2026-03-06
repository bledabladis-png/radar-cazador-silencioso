# backtest.py - Backtesting del Radar Macro Rotación Global
# Versión vectorizada con datos automáticos de SPY desde DataLayer.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_layer import DataLayer

def backtest(historial_file='historial_radar.csv', initial_capital=100000, comision=0.001):
    """
    Realiza el backtest del radar.
    - historial_file: archivo CSV generado por run_radar.py.
    - initial_capital: capital inicial.
    - comision: coste de transacción por cambio de exposición (fracción).
    """
    # Cargar historial
    df = pd.read_csv(historial_file, parse_dates=['fecha'])
    df = df.sort_values('fecha').set_index('fecha')
    
    # Eliminar duplicados de fecha (quedarse con la última entrada)
    df = df[~df.index.duplicated(keep='last')]
    
    if len(df) < 2:
        print("❌ No hay suficientes datos para backtest.")
        return

    # Verificar columna de exposición
    if 'exposicion_final' not in df.columns:
        print("❌ El historial no contiene la columna 'exposicion_final'.")
        return

    # Obtener precios de SPY desde DataLayer
    dl = DataLayer()
    try:
        df_prices = dl.load_latest()
    except FileNotFoundError:
        print("❌ No se encontraron datos procesados. Ejecuta primero run_radar.py.")
        return

    # Buscar columna de cierre de SPY
    if 'SPY_Close' in df_prices.columns:
        spy = df_prices['SPY_Close'].rename('spy')
    elif 'SPY' in df_prices.columns:
        spy = df_prices['SPY'].rename('spy')
    else:
        print("❌ No se encontró precio de SPY en los datos.")
        return

    spy.index = pd.to_datetime(spy.index)

    # Alinear fechas entre historial y precios de SPY
    fechas_comunes = df.index.intersection(spy.index)
    if len(fechas_comunes) < 2:
        print("❌ No hay suficientes fechas comunes entre historial y SPY.")
        return

    df = df.loc[fechas_comunes]
    spy = spy.loc[fechas_comunes]

    # Calcular retornos diarios de SPY
    returns_spy = spy.pct_change().fillna(0)

    # --- Simulación vectorizada ---
    # La exposición que se aplica hoy es la del día anterior (shift)
    exposicion_lag = df['exposicion_final'].shift(1).fillna(0)
    
    # Rendimiento diario de la cartera (antes de comisiones)
    returns_cartera_brutos = exposicion_lag * returns_spy

    # Costes de comisión por cambio de exposición
    cambio_exposicion = exposicion_lag.diff().abs().fillna(0)
    coste_comision = cambio_exposicion * comision

    # Rendimiento neto de comisiones
    returns_cartera_netos = returns_cartera_brutos - coste_comision

    # Calcular equity acumulado
    equity = (1 + returns_cartera_netos).cumprod() * initial_capital

    # DataFrame de resultados
    resultados = pd.DataFrame({
        'equity': equity,
        'exposicion': exposicion_lag,
        'retorno_bruto': returns_cartera_brutos,
        'retorno_neto': returns_cartera_netos,
        'cambio_exposicion': cambio_exposicion,
        'coste_comision': coste_comision,
        'spy_price': spy,
        'ret_spy': returns_spy
    }, index=df.index)

    # --- Métricas ---
    total_return = (equity.iloc[-1] / initial_capital - 1) * 100
    benchmark_return = ( (1 + returns_spy).prod() - 1 ) * 100

    # Sharpe ratio (anualizado)
    rets = returns_cartera_netos.dropna()
    if len(rets) > 1 and rets.std() != 0:
        sharpe = rets.mean() / rets.std() * np.sqrt(252)
    else:
        sharpe = 0.0

    # Máximo drawdown
    cumulative = equity / initial_capital
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    # Turnover total (suma de cambios de exposición)
    turnover_total = cambio_exposicion.sum()

    # --- Resultados ---
    print("=" * 60)
    print("RESULTADOS DEL BACKTEST")
    print("=" * 60)
    print(f"Período: {df.index[0].date()} a {df.index[-1].date()}")
    print(f"Días con datos: {len(df)}")
    print(f"Capital inicial: {initial_capital:,.0f}")
    print(f"Capital final: {equity.iloc[-1]:,.0f}")
    print(f"Rentabilidad total: {total_return:.2f}%")
    print(f"Rentabilidad benchmark (SPY): {benchmark_return:.2f}%")
    print(f"Exceso de retorno: {total_return - benchmark_return:.2f}%")
    print(f"Sharpe ratio (anualizado): {sharpe:.3f}")
    print(f"Máximo drawdown: {max_drawdown:.2f}%")
    print(f"Turnover total (cambios): {turnover_total:.2f}")
    print(f"Comisiones totales: {coste_comision.sum()*initial_capital:.2f} ({comision:.1%} por cambio)")

    # Guardar resultados en CSV
    resultados.to_csv('resultados_backtest.csv')
    print("\n✅ Resultados guardados en 'resultados_backtest.csv'")

    # --- Gráfico ---
    if len(resultados) > 1:
        plt.figure(figsize=(12,6))
        plt.plot(resultados.index, resultados['equity']/initial_capital, label='Cartera Radar')
        benchmark_series = (1 + returns_spy).cumprod()
        plt.plot(resultados.index, benchmark_series, '--', label='Benchmark SPY')
        plt.title('Evolución del capital (normalizado)')
        plt.xlabel('Fecha')
        plt.ylabel('Capital / Inicial')
        plt.legend()
        plt.grid(True)
        plt.savefig('backtest_resultados.png')
        plt.show()
        print("✅ Gráfico guardado como 'backtest_resultados.png'")
    else:
        print("No hay suficientes datos para generar gráfico.")

    return resultados

if __name__ == "__main__":
    backtest(historial_file='historial_radar.csv', initial_capital=100000, comision=0.001)