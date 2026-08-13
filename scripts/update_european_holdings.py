# update_european_holdings.py
import subprocess
import sys
import pandas as pd

def run(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR ejecutando: {cmd}")
        sys.exit(1)

print("Actualizando holdings europeos...")

# 1. FEZ desde SSGA
run("py scripts/parse_ssga_fez.py")

# 2. DAXEX e ISF.L desde BlackRock
run("py scripts/parse_blackrock_final.py")

# 3. LYXI desde Amundi
run("py scripts/amundi_holdings.py FR0010251744")

# 4. Fusionar en index_holdings.csv
index_path = 'data/index_holdings.csv'
index_df = pd.read_csv(index_path)
european_etfs = ['FEZ', 'DAXEX', 'ISF.L', 'LYXI']
index_df = index_df[~index_df['etf'].isin(european_etfs)]

fez = pd.read_csv('outputs/holdings/FEZ_final_holdings.csv')
daxex = pd.read_csv('outputs/holdings/DAXEX_final_holdings.csv')
isf = pd.read_csv('outputs/holdings/ISF.L_final_holdings.csv')
amundi = pd.read_csv('outputs/holdings/amundi_lyxi_holdings.csv').sort_values('weight', ascending=False).head(20)
amundi['etf'] = 'LYXI'

cols = ['etf', 'ticker', 'name', 'weight']
for df in [fez, daxex, isf, amundi]:
    df = df[cols]

nuevo_df = pd.concat([index_df, fez, daxex, isf, amundi], ignore_index=True)
nuevo_df = nuevo_df.drop_duplicates(subset=['etf','ticker'], keep='last')
nuevo_df.to_csv(index_path, index=False)
print(f"\nindex_holdings.csv actualizado con FEZ, DAXEX, ISF.L y LYXI. Total: {len(nuevo_df)}")
