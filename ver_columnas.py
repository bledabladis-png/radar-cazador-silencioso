from src.data_layer import DataLayer
dl = DataLayer()
df = dl.load_latest()
print("Columnas disponibles:")
for col in df.columns:
    print(f"  '{col}'")
exit()