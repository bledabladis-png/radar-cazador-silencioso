import pandas as pd
import numpy as np

def compute_fis(regime_score, sector_rotation_dict, leaders_csv="outputs/analisis_lideres.csv"):
    """
    Calcula el Factor Institutional Score (FIS) para cada acción.
    - regime_score: número entre 0 y 1 (del reporte diario)
    - sector_rotation_dict: diccionario con rotation_score por sector (opcional)
    - leaders_csv: ruta del CSV generado por el módulo de líderes
    """
    df = pd.read_csv(leaders_csv)
    
    # Si no hay sector_rotation, asignamos un valor neutral (0.5)
    if sector_rotation_dict is None:
        df["sector_rotation_norm"] = 0.5
    else:
        # Normalizar rotation_score de cada sector a [0,1]
        min_rot = min(sector_rotation_dict.values())
        max_rot = max(sector_rotation_dict.values())
        df["sector_rotation_norm"] = df["sector"].map(
            lambda s: (sector_rotation_dict[s] - min_rot) / (max_rot - min_rot + 1e-9)
        )
    
    # Regime alignment (mismo valor para todas las acciones)
    df["regime_alignment"] = regime_score
    
    # Calcular FIS
    df["FIS"] = (
        0.30 * df["regime_alignment"] +
        0.20 * df["sector_rotation_norm"] +
        0.30 * df["wls"] +
        0.20 * df["structure_quality"]
    )
    
    # Ordenar por FIS descendente
    df = df.sort_values("FIS", ascending=False)
    
    # Mostrar top 10
    print(df[["ticker", "sector", "wls", "structure_quality", "FIS"]].head(10))
    return df

if __name__ == "__main__":
    # Ejemplo de uso después de ejecutar el radar
    # (necesitas tener outputs/analisis_lideres.csv y saber el regime_score)
    regime = 0.72   # cámbialo por el valor que salga en el reporte
    rotation = None  # si no lo tienes, déjalo None
    compute_fis(regime, rotation)