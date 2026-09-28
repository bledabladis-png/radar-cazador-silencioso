# update_european_holdings.py
#
# A6.3-01 (2026-09-29): refactor para no ejecutar al importar.
# Antes: script sin `if __name__ == "__main__"` -> cualquier import
# disparaba descarga + escritura. Ahora: logica dentro de main(),
# invocacion explicita al ejecutar como script.
# A6.3-02: subprocess.run sin shell=True (comando fijo, sin metacaracteres).
import subprocess
import sys
import os

import pandas as pd


def _run(args):
    """Ejecuta un subcomando y devuelve exit code. Sin shell=True."""
    print(f"\n>>> {' '.join(args)}")
    result = subprocess.run(args)
    if result.returncode != 0:
        print(f"ERROR ejecutando: {' '.join(args)}")
    return result.returncode


def main():
    # Asegurar directorios de salida
    os.makedirs("outputs/holdings", exist_ok=True)

    print("Actualizando holdings europeos...")

    # 1. FEZ desde SSGA
    if _run([sys.executable, "scripts/parse_ssga_fez.py"]) != 0:
        return 1

    # 2. DAXEX e ISF.L desde BlackRock
    if _run([sys.executable, "scripts/parse_blackrock_final.py"]) != 0:
        return 1

    # 3. LYXI desde Amundi
    if _run([sys.executable, "scripts/amundi_holdings.py", "FR0010251744"]) != 0:
        return 1

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

    # Schema de index_holdings.csv (sin identifier)
    cols = ['etf', 'ticker', 'name', 'weight']
    fez = fez.reindex(columns=cols)
    daxex = daxex.reindex(columns=cols)
    isf = isf.reindex(columns=cols)
    amundi = amundi.reindex(columns=cols)

    nuevo_df = pd.concat([index_df, fez, daxex, isf, amundi], ignore_index=True)
    nuevo_df = nuevo_df.drop_duplicates(subset=['etf','ticker'], keep='last')
    nuevo_df.to_csv(index_path, index=False)
    print(f"\nindex_holdings.csv actualizado con FEZ, DAXEX, ISF.L y LYXI. Total: {len(nuevo_df)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
