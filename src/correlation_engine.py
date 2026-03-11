"""
correlation_engine.py - Ajuste de pesos por correlaciÃ³n usando PCA
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

def adjust_weights_pca(weights, df_motores, n_components=3, variance_threshold=0.8):
    """
    Ajusta los pesos de los motores usando PCA para reducir multicolinealidad.
    
    ParÃ¡metros:
    - weights: dict {modulo: peso} (pesos originales)
    - df_motores: DataFrame con columnas = mÃ³dulos, filas = fechas
    - n_components: nÃºmero de componentes a extraer (si es None, se elige por varianza)
    - variance_threshold: umbral de varianza explicada para elegir n_components
    
    Retorna:
    - nuevos_weights: dict con pesos ajustados (renormalizados a suma 1)
    """
    if len(df_motores) < 20:
        logger.warning("Datos insuficientes para PCA, devolviendo pesos originales")
        return weights
    
    # Estandarizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_motores)
    
    # Determinar nÃºmero de componentes
    if n_components is None:
        pca_temp = PCA()
        pca_temp.fit(X_scaled)
        cum_var = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = np.searchsorted(cum_var, variance_threshold) + 1
    
    # PCA
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    
    # Matriz de loadings (contribuciÃ³n de cada variable a los componentes)
    loadings = pca.components_.T  # shape (n_variables, n_components)
    
    # Calcular pesos basados en loadings al cuadrado (contribuciÃ³n total)
    contrib = np.sum(loadings**2, axis=1)
    contrib = contrib / contrib.sum()  # normalizar
    
    # Crear nuevos pesos
    nuevos = {}
    for i, mod in enumerate(df_motores.columns):
        if mod in weights:
            nuevos[mod] = weights[mod] * contrib[i]  # ponderar por contribuciÃ³n
        else:
            nuevos[mod] = 0
    
    # Renormalizar a suma 1
    total = sum(nuevos.values())
    if total > 0:
        for mod in nuevos:
            nuevos[mod] /= total
    
    return nuevos


# Mantener la funciÃ³n antigua por compatibilidad
def adjust_weights(weights, corr_matrix, threshold=0.8, penalty_factor=0.5):
    """
    VersiÃ³n antigua por compatibilidad.
    """
    nuevos = {}
    for mod, peso in weights.items():
        if mod in corr_matrix.columns:
            penalizacion = 0.0
            for otro in corr_matrix.columns:
                if otro != mod and otro in weights:
                    corr = abs(corr_matrix.loc[mod, otro])
                    if corr > threshold:
                        penalizacion += (corr - threshold) * penalty_factor
            nuevos[mod] = peso * (1 - penalizacion)
        else:
            nuevos[mod] = peso
    total = sum(nuevos.values())
    if total > 0:
        for mod in nuevos:
            nuevos[mod] /= total
    return nuevos
