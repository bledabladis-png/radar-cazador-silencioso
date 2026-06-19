import pandas as pd
import os

def load_macro_manual(data_dir='data/macro_manual'):
    """
    Carga todos los CSVs en data/macro_manual y los une por fecha.
    Retorna un DataFrame con columna 'date' y una columna por indicador.
    """
    if not os.path.exists(data_dir):
        return None
    
    dfs = []
    for fname in os.listdir(data_dir):
        if fname.endswith('.csv'):
            path = os.path.join(data_dir, fname)
            try:
                df = pd.read_csv(path)
                if 'date' not in df.columns:
                    continue
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                # Prefijo con el nombre del archivo (sin extension) para evitar colisiones
                prefix = os.path.splitext(fname)[0]
                df = df.add_prefix(f'{prefix}_')
                dfs.append(df)
            except Exception:
                pass
    
    if not dfs:
        return None
    
    combined = pd.concat(dfs, axis=1)
    combined.reset_index(inplace=True)
    combined.rename(columns={'index': 'date'}, inplace=True)
    return combined