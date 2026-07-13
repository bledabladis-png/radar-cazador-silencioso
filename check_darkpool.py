from indicators.darkpool import compute_darkpool_signals
result = compute_darkpool_signals()
if result:
    print('Datos obtenidos correctamente.')
    print(f'Media Dark Pool: {result["media_dark_pool"]:.2f}%')
    print(f'Ticker maximo: {result["ticker_max"]}')
else:
    print('La funcion devolvio None.')
