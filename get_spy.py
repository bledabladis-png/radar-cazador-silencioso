from src.data_layer import DataLayer

dl = DataLayer()
df = dl.load_latest()
df[['SPY']].to_csv('spy_prices.csv')
print("Archivo spy_prices.csv generado correctamente.")