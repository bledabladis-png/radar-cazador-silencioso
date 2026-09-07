"""
Matriz de Evidencia v1.0 — Descriptiva, no predictiva.
Consolida 5 evidencias sectoriales + 2 de contexto global.
No genera score compuesto ni alimenta motores.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def _sign_from_value(value):
    if pd.isna(value):
        return np.nan
    if value > 0:
        return 1
    elif value < 0:
        return -1
    else:
        return 0


def _sign_from_pct_above(value, threshold=50.0):
    if pd.isna(value):
        return np.nan
    if value > threshold:
        return 1
    elif value < threshold:
        return -1
    else:
        return 0


def _wyckoff_sign(row):
    if pd.isna(row.get('pct_accumulation')) or pd.isna(row.get('pct_markup')) or pd.isna(row.get('pct_distribution')) or pd.isna(row.get('pct_markdown')):
        return np.nan
    favorable = row['pct_accumulation'] + row['pct_markup']
    unfavorable = row['pct_distribution'] + row['pct_markdown']
    if favorable > unfavorable:
        return 1
    elif favorable < unfavorable:
        return -1
    else:
        return 0


def _quality_from_coverage(cov):
    if pd.isna(cov):
        return 'Baja'
    if cov >= 80:
        return 'Alta'
    elif cov >= 50:
        return 'Media'
    else:
        return 'Baja'


def _safe_get_row(df, sector):
    if df is None or df.empty:
        return {}
    if sector in df.index:
        row = df.loc[sector]
        if isinstance(row, pd.Series):
            return row.to_dict()
        elif isinstance(row, pd.DataFrame):
            return row.iloc[-1].to_dict() if not row.empty else {}
    return {}


def compute_evidence_matrix(sector_breadth_df=None,
                           sector_concentration_df=None,
                           sector_flow_characteristics_df=None,
                           sector_wyckoff_distribution_df=None,
                           liquidity_regime=None,
                           real_liquidity_regime=None,
                           volatility_regime=None,
                           volatility_score=None,
                           liquidity_score=None):
    """
    Calcula la Matriz de Evidencia para 11 sectores.
    Devuelve DataFrame con columnas descriptivas.
    Si falta una fuente, la evidencia correspondiente se marca NaN.
    """
    # Obtener lista de sectores desde cualquier DataFrame disponible
    sectors = None
    for df in [sector_breadth_df, sector_concentration_df, sector_flow_characteristics_df, sector_wyckoff_distribution_df]:
        if df is not None and not df.empty and 'sector' in df.columns:
            sectors = df['sector'].unique()
            break
    if sectors is None:
        return None

    # Preparar índices si existen
    breadth = sector_breadth_df.set_index('sector', drop=False) if sector_breadth_df is not None else pd.DataFrame()
    flow_char = sector_flow_characteristics_df.set_index('sector', drop=False) if sector_flow_characteristics_df is not None else pd.DataFrame()
    concentration = sector_concentration_df.set_index('sector', drop=False) if sector_concentration_df is not None else pd.DataFrame()
    wyckoff = sector_wyckoff_distribution_df.set_index('sector', drop=False) if sector_wyckoff_distribution_df is not None else pd.DataFrame()

    rows = []

    for sector in sectors:
        b = _safe_get_row(breadth, sector)
        fc = _safe_get_row(flow_char, sector)
        conc = _safe_get_row(concentration, sector)
        wy = _safe_get_row(wyckoff, sector)

        price_ev = _sign_from_value(fc.get('price_ret_20d', np.nan))
        breadth_ev = _sign_from_pct_above(b.get('pct_above_ema50', np.nan))
        primary_flow_ev = _sign_from_value(fc.get('flow_20d_sum', np.nan))
        proxy_flow_ev = _sign_from_value(conc.get('flow_median', np.nan))
        wyckoff_ev = _wyckoff_sign(wy)

        # Calidad individual
        price_quality = 'Alta' if not pd.isna(price_ev) else 'Baja'
        breadth_quality = 'Alta' if not pd.isna(breadth_ev) else 'Baja'
        primary_flow_quality = 'Alta' if not pd.isna(primary_flow_ev) else 'Baja'
        proxy_flow_quality = _quality_from_coverage(conc.get('coverage_flow', np.nan))
        wyckoff_quality = _quality_from_coverage(wy.get('coverage_wyckoff', np.nan))

        # Contexto global (no participa en balance)
        credit_ev = 0
        vol_ev = 0
        if real_liquidity_regime:
            credit_ev = 1 if real_liquidity_regime in ('LOW_STRESS', 'NORMAL', 'AMPLIA', 'GOLDILOCKS', 'ABUNDANTE') else (-1 if real_liquidity_regime in ('HIGH_STRESS', 'EXTREME_STRESS', 'ESTRECHA') else 0)
        elif liquidity_regime:
            credit_ev = 1 if liquidity_regime in ('LOW_STRESS', 'NORMAL', 'AMPLIA', 'GOLDILOCKS', 'ABUNDANTE') else (-1 if liquidity_regime in ('HIGH_STRESS', 'EXTREME_STRESS', 'ESTRECHA') else 0)

        if volatility_regime:
            vol_ev = 1 if volatility_regime in ('LOW', 'LOW_VOL') else (-1 if volatility_regime in ('HIGH', 'HIGH_VOL', 'EXTREME') else 0)

        credit_quality = 'Alta' if credit_ev != 0 else 'Baja'
        volatility_quality = 'Alta' if vol_ev != 0 else 'Baja'

        # Calidad global: peor de las 5 sectoriales
        sector_qualities = [price_quality, breadth_quality, primary_flow_quality, proxy_flow_quality, wyckoff_quality]
        if 'Baja' in sector_qualities:
            evidence_quality = 'Baja'
        elif 'Media' in sector_qualities:
            evidence_quality = 'Media'
        else:
            evidence_quality = 'Alta'

        # Balance sectorial
        sector_evidences = [price_ev, breadth_ev, primary_flow_ev, proxy_flow_ev, wyckoff_ev]
        valid = [e for e in sector_evidences if not pd.isna(e)]
        n_valid = len(valid)
        n_positive = sum(1 for e in valid if e == 1)
        n_negative = sum(1 for e in valid if e == -1)
        n_neutral = sum(1 for e in valid if e == 0)

        if n_valid < 3:
            alignment = 'EVIDENCIA INSUFICIENTE'
        elif n_positive >= 3 and n_positive > n_negative:
            alignment = 'EVIDENCIA PREDOMINANTEMENTE FAVORABLE'
        elif n_negative >= 3 and n_negative > n_positive:
            alignment = 'EVIDENCIA PREDOMINANTEMENTE DESFAVORABLE'
        else:
            alignment = 'EVIDENCIA MIXTA'

        rows.append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'sector': sector,
            'price_evidence': price_ev,
            'breadth_evidence': breadth_ev,
            'primary_flow_evidence': primary_flow_ev,
            'proxy_flow_evidence': proxy_flow_ev,
            'wyckoff_evidence': wyckoff_ev,
            'credit_evidence': credit_ev,
            'volatility_evidence': vol_ev,
            'n_sector_evidence_valid': n_valid,
            'n_sector_evidence_positive': n_positive,
            'n_sector_evidence_negative': n_negative,
            'n_sector_evidence_neutral': n_neutral,
            'evidence_quality': evidence_quality,
            'alignment_reading': alignment,
            'price_quality': price_quality,
            'breadth_quality': breadth_quality,
            'primary_flow_quality': primary_flow_quality,
            'proxy_flow_quality': proxy_flow_quality,
            'wyckoff_quality': wyckoff_quality,
            'credit_quality': credit_quality,
            'volatility_quality': volatility_quality,
        })

    df = pd.DataFrame(rows)
    ev_cols = ['price_evidence','breadth_evidence','primary_flow_evidence','proxy_flow_evidence','wyckoff_evidence','credit_evidence','volatility_evidence']
    for c in ev_cols:
        df[c] = df[c].astype('Int64')
    return df


def save_evidence_matrix(df, path='outputs/history/evidence_matrix.csv'):
    if df is not None and not df.empty:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df