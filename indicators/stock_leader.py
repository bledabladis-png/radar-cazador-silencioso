import pandas as pd
import numpy as np
from indicators.wyckoff import wyckoff_score, wyckoff_confidence, classify_wyckoff_phase, detect_spring, detect_sos
from src.utils import robust_zscore, get_col

def compute_stock_metrics(df_market, df_stocks, etf_ticker, stock_list):
    results = []
    price_etf = get_col(df_market, etf_ticker, 'Close')

    for ticker in stock_list:
        try:
            close = get_col(df_stocks, ticker, 'Close')
            volume = get_col(df_stocks, ticker, 'Volume')
        except KeyError:
            continue

        if len(close.dropna()) < 60:
            continue

        # Alinear índices para evitar NaN por diferencias de calendario
        common_idx = close.index.intersection(price_etf.index)
        rs = close.loc[common_idx] / price_etf.loc[common_idx]
        rs_mom = np.log(rs).diff(20).iloc[-1]

        ret = close.pct_change(fill_method=None)
        dollar_vol = close * volume
        flow_raw = ret * dollar_vol
        flow_z = robust_zscore(flow_raw, window=60)
        flow_signal = flow_z.ewm(span=5).mean().iloc[-1]

        ret_positive = (ret > 0).astype(int)
        persistence_10d = ret_positive.rolling(10, min_periods=10).mean().iloc[-1]
        if pd.isna(persistence_10d):
            persistence_10d = 0.5

        try:
            ticker_df = pd.DataFrame({
                'Open': get_col(df_stocks, ticker, 'Open'),
                'High': get_col(df_stocks, ticker, 'High'),
                'Low': get_col(df_stocks, ticker, 'Low'),
                'Close': close,
                'Volume': volume
            }).dropna()
        except KeyError:
            continue

        if len(ticker_df) >= 60:
            wyckoff_series, struct_score, tact_score, t_n, c_n, v_n, e_n = wyckoff_score(ticker_df, ticker)
            wyckoff_sc = wyckoff_series.iloc[-1]
            wyckoff_ph = classify_wyckoff_phase(ticker_df, ticker)
            score_median = wyckoff_series.rolling(10).median().iloc[-1]
            score_mad = wyckoff_series.rolling(10).apply(lambda x: np.median(np.abs(x - np.median(x)))).iloc[-1]
            stability = np.tanh(score_median / (score_mad + 1e-9))
            spring = detect_spring(ticker_df, ticker).iloc[-1]
            sos = detect_sos(ticker_df, ticker).iloc[-1]
        else:
            wyckoff_sc = np.nan
            wyckoff_ph = 'INSUFICIENTE'
            stability = 0.0
            spring = 0
            sos = 0

        results.append({
            'ticker': ticker,
            'rs': rs.iloc[-1],
            'rs_mom': rs_mom,
            'flow_z': flow_signal,
            'wyckoff_score': wyckoff_sc,
            'wyckoff_phase': wyckoff_ph,
            'persistence_10d': persistence_10d,
            'stability': stability,
            'spring': spring,
            'sos': sos,
        })

    return pd.DataFrame(results)

def compute_wls(df_metrics, weights=None):
    df = df_metrics.copy()
    if df.empty:
        return df

    def robust_intra(s):
        median = s.median()
        mad = (s - median).abs().median()
        return (s - median) / (1.4826 * mad + 1e-9)

    df['rs_z'] = df.groupby('sector')['rs_mom'].transform(robust_intra).clip(-3, 3)
    df['flow_z_norm'] = df.groupby('sector')['flow_z'].transform(robust_intra).clip(-3, 3)

    # M8: robust_zscore intra-sectorial en lugar de percentil 70
    df['rws_z'] = df.groupby('sector')['wyckoff_score'].transform(robust_intra).clip(-3, 3)

    df['stab_z'] = df.groupby('sector')['stability'].transform(robust_intra).clip(-3, 3)

    for sector in df['sector'].unique():
        mask = df['sector'] == sector
        if mask.sum() < 3:
            continue
        rho = df.loc[mask, 'rs_mom'].corr(df.loc[mask, 'flow_z'], method='spearman')
        if pd.notna(rho) and rho > 0.7:
            factor = 1 - min(1.0, (rho - 0.7) / 0.3)
            df.loc[mask, 'flow_z_norm'] *= factor

    if weights is None:
        w_rs, w_flow, w_struct, w_stab = 0.35, 0.30, 0.25, 0.10
    else:
        w_rs = weights.get('rs', 0.35)
        w_flow = weights.get('flow', 0.30)
        w_struct = weights.get('structure', 0.25)
        w_stab = weights.get('stability', 0.10)

    df['wls'] = w_rs*df['rs_z'] + w_flow*df['flow_z_norm'] + w_struct*df['rws_z'] + w_stab*df['stab_z']
    df['wls'] *= (1 + 0.05 * np.minimum(df['persistence_10d'], 1.0))
    df['sector_rank_pct'] = df.groupby('sector')['wls'].rank(pct=True)

    # leader_confidence eliminado (no se usaba)
    for sector in df['sector'].unique():
        mask = df['sector'] == sector
        sector_df = df[mask]
        if len(sector_df) >= 20:
            current_rank = sector_df['wls'].rank()
            historical_rank = sector_df['wls'].iloc[:-20].rank()
            common_idx = current_rank.index.intersection(historical_rank.index)
            if len(common_idx) > 5:
                rho = current_rank.loc[common_idx].corr(historical_rank.loc[common_idx], method='spearman')
                # leader_confidence eliminado

    return df.sort_values('wls', ascending=False)

def generate_leader_section(df_market, df_stocks, holdings_df, fase_dict, operabilidad_dict, output_csv=None):
    lines = []
    all_data = []

    VALID_FASES = {'ACCUMULATION', 'MARKUP'}
    VALID_OPER = {'OPORTUNIDAD MODERADA'}

    for sector in ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']:
        fase = fase_dict.get(sector, 'NEUTRAL')
        oper = operabilidad_dict.get(sector, 'NO OPERAR')
        if fase not in VALID_FASES or oper not in VALID_OPER:
            continue
        stocks = holdings_df[holdings_df['etf'] == sector]['ticker'].tolist()
        if not stocks:
            continue
        metrics_df = compute_stock_metrics(df_market, df_stocks, sector, stocks)
        if metrics_df.empty:
            continue
        metrics_df['sector'] = sector
        wls_df = compute_wls(metrics_df)
        all_data.append(wls_df)

        lines.append(f'## Sector: {sector} ({fase})\n')
        lines.append('| Ticker | RS | RS Mom | Flujo (z) | WLS | Fase Wyckoff | Spring | SOS |\n')
        lines.append('|--------|----|--------|-----------|-----|---------------|--------|-----|\n')
        lines.append('*RS = RS Level (precio acción / precio sector). RS Mom = RS Momentum (cambio del RS en 20 días). El WLS combina ambas con pesos 35% y 25% respectivamente.*\n')
        for _, row in wls_df.head(3).iterrows():
            spring_flag = '✓' if row.get('spring', 0) == 1 else ''
            sos_flag = '✓' if row.get('sos', 0) == 1 else ''
            lines.append(f"| {row['ticker']} | {row['rs']:.2f} | {row['rs_mom']:.2%} | {row['flow_z']:.2f} | {row['wls']:.2f} | {row['wyckoff_phase']} | {spring_flag} | {sos_flag} |\n")
        lines.append('\n')

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        if output_csv:
            cols = ['ticker','sector','rs','rs_mom','flow_z','wyckoff_score','wyckoff_phase',
                    'persistence_10d','stability','spring','sos','wls','sector_rank_pct']
            final_df[cols].to_csv(output_csv, index=False)
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            return lines, final_df
        return lines, None
    return None, None

