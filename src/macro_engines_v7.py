"""
macro_engines_v7.py - Motores V7 para Radar Macro Rotación Global (v2.3.1)
Incluye:
- riesgo_sistemico (combinación lineal mejorada)
- carry_trade
- ciclo_institucional
"""

import pandas as pd
import numpy as np
import logging
from src.utils.normalization import robust_scale

logger = logging.getLogger(__name__)

def riesgo_sistemico(df):
    """
    Calcula el score de riesgo sistémico usando PCA sobre:
    - VIX (nivel)
    - Credit spread (HYG - LQD retornos) y en niveles
    - Bond volatility (volatilidad anualizada de TLT)
    - Liquidity spread (HYG - LQD en niveles)
    Retorna un score entre -1 y 1.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # 1. Preparar datos
    # VIX
    vix = df['^VIX'].ffill()
    
    # Credit spread (retornos)
    ret_hyg = df['HYG'].ffill().pct_change(fill_method=None)
    ret_lqd = df['LQD'].ffill().pct_change(fill_method=None)
    credit_ret = ret_hyg - ret_lqd
    
    # Credit spread en niveles (diferencia de precios)
    credit_level = df['HYG'].ffill() - df['LQD'].ffill()
    
    # Bond volatility
    ret_tlt = df['TLT'].ffill().pct_change(fill_method=None)
    bond_vol = ret_tlt.rolling(20).std() * np.sqrt(252)
    
    # Crear DataFrame con todos los indicadores
    pca_df = pd.DataFrame({
        'vix': vix,
        'credit_ret': credit_ret,
        'credit_level': credit_level,
        'bond_vol': bond_vol
    }).dropna()
    
    if len(pca_df) < 20:
        logger.warning("Datos insuficientes para PCA, usando combinación lineal simple")
        # Fallback a versión anterior
        vix_norm = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
        credit_norm = (credit_ret - credit_ret.rolling(252).mean()) / credit_ret.rolling(252).std()
        bond_norm = (bond_vol - bond_vol.rolling(252).mean()) / bond_vol.rolling(252).std()
        riesgo_raw = 0.4 * vix_norm + 0.4 * credit_norm + 0.2 * bond_norm
        riesgo_raw = riesgo_raw.fillna(0)
        return np.tanh(riesgo_raw).fillna(0)
    
    # 2. Estandarizar
    scaler = StandardScaler()
    pca_scaled = scaler.fit_transform(pca_df)
    
    # 3. Aplicar PCA
    pca = PCA(n_components=1)
    pca_component = pca.fit_transform(pca_scaled)
    
    # 4. Ajustar signo para que correlacione positivamente con VIX
    corr_with_vix = np.corrcoef(pca_component.flatten(), pca_df['vix'].values)[0,1]
    if corr_with_vix < 0:
        pca_component = -pca_component
    
    # 5. Convertir a score
    pca_series = pd.Series(pca_component.flatten(), index=pca_df.index)
    # Normalizar a media 0, desviación 1
    pca_std = (pca_series - pca_series.mean()) / pca_series.std()
    riesgo_score = np.tanh(pca_std)
    
    # Reindexar al índice original
    riesgo_score = riesgo_score.reindex(df.index, method='ffill').fillna(0)
    
    # Mostrar varianza explicada (debug)
    logger.info(f"PCA riesgo sistémico - varianza explicada: {pca.explained_variance_ratio_[0]:.2%}")
    
    return riesgo_score

def carry_trade(df):
    """
    Calcula el score de carry trade global (entre -1 y 1).
    Fórmula: raw = -0.4*JPY_ret + 0.3*AUD_ret + 0.3*SPY_ret
    Luego normalización robusta + tanh.
    """
    jpy_ret = df['JPY=X'].ffill().pct_change(fill_method=None)
    aud_ret = df['AUD=X'].ffill().pct_change(fill_method=None)
    spy_ret = df['SPY'].ffill().pct_change(fill_method=None)

    raw = -0.4 * jpy_ret + 0.3 * aud_ret + 0.3 * spy_ret

    scaling = robust_scale(raw, window=252).shift(1)
    scaling = scaling.replace(0, np.nan).ffill().fillna(0.5)
    score = np.tanh(raw / scaling)

    return score.fillna(0)

def ciclo_institucional(fila):
    """
    Clasifica la fase del ciclo institucional en 4 categorías.
    fila: diccionario o pd.Series con claves:
          'score_global', 'score_breadth', 'score_stress'
    Retorna una string.
    """
    score = fila.get('score_global', 0)
    breadth = fila.get('score_breadth', 0)
    stress = fila.get('score_stress', 0)

    if score > 0.4 and breadth > 0:
        return "EXPANSION"
    if score > 0 and breadth < 0:
        return "ACUMULACION"
    if score < 0 and breadth > 0:
        return "DISTRIBUCION"
    if score < -0.3 and stress < -0.3:
        return "CAPITULACION"
    return "NEUTRAL"