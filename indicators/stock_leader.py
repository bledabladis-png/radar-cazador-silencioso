import os
import pandas as pd
from src.report.helpers import _fmt_num
from config.settings import TOP_N_CANDIDATES, TOP_N_LEADERS
from indicators.stock_metrics import compute_stock_metrics, compute_wls

# ============================================================
# N3 (2026-10-13, auditor): validacion del contrato de WLS.
# Funcion pura, sin side effects. Testeable sin DataFrame grande.
def generate_leader_section(df_market, df_stocks, holdings_df, fase_dict,
                               operabilidad_dict, output_csv=None, temporal_meta=None):
    lines = []
    leader_data = []
    full_metrics_data = []

    VALID_FASES = {'ACCUMULATION', 'MARKUP'}
    VALID_OPER = {'OPORTUNIDAD MODERADA'}


    for sector in ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']:
        fase = fase_dict.get(sector, 'NEUTRAL')
        oper = operabilidad_dict.get(sector, 'NO OPERAR')
        sector_holdings = holdings_df[holdings_df['etf'] == sector].copy()
        if 'weight' in sector_holdings.columns:
            sector_holdings = sector_holdings.sort_values('weight', ascending=False)
        stocks = sector_holdings.head(TOP_N_CANDIDATES)['ticker'].tolist()
        if not stocks:
            continue
        metrics_df = compute_stock_metrics(df_market, df_stocks, sector, stocks, temporal_meta=temporal_meta)
        if metrics_df.empty:
            continue
        metrics_df['sector'] = sector
        wls_df = compute_wls(metrics_df)
        if wls_df is None or wls_df.empty:
            continue
        full_metrics_data.append(wls_df)

        if fase not in VALID_FASES or oper not in VALID_OPER:
            continue
        leader_data.append(wls_df.head(TOP_N_LEADERS))

        lines.append(f'## Sector: {sector} ({fase})\n')
        # H17 (2026-10-15, auditoria externa): aclarar que Pers 5d/10d/20d
        # miden la persistencia del retorno POSITIVO del ticker
        # (fraccion de sesiones con retorno > 0 en la ventana).
        # Rango [0,1]. Distinta de "Persist" (Structural Ranking sectorial)
        # y de "Pers 5d/20d" del flujo primario.
        lines.append('| Ticker | RS | RS Mom | Flujo (z) | WLS | Fase Wyckoff | Pers 5d | Pers 10d | Pers 20d | Spring | SOS |\n')
        lines.append('|--------|----|--------|-----------|-----|---------------|--------|--------|---------|--------|-----|\n')
        for _, row in wls_df.head(TOP_N_LEADERS).iterrows():
            # H5 (2026-10-17): exponer 0/1 explicitos. Antes se pintaba
            # '[v]' si 1 y celda vacia si 0. El lector no podia distinguir
            # "spring=0" de "spring=NaN". El dato siempre existe; 0 es
            # un valor real.
            spring_flag = str(int(row.get('spring', 0) or 0))
            sos_flag = str(int(row.get('sos', 0) or 0))
            lines.append(f"| {row['ticker']} | {_fmt_num(row['rs'], '{:.2f}')} | {_fmt_num(row['rs_mom'], '{:.2%}')} | {_fmt_num(row['flow_proxy_z'], '{:.2f}')} | {_fmt_num(row['wls'], '{:.2f}')} | {row['wyckoff_phase']} | {_fmt_num(row['persistence_5d'], '{:.0%}')} | {_fmt_num(row['persistence_10d'], '{:.0%}')} | {_fmt_num(row['persistence_20d'], '{:.0%}')} | {spring_flag} | {sos_flag} |\n")
        lines.append('\n')

    # Filtrar métricas None/vacías y concatenar solo si hay datos válidos
    # P1 (2026-10-22, auditor externo): la componente rs_z del WLS
    # deriva de rs_mom (log-retorno relativo 20d), NO de rs nominal.
    # rs_mom es invariante a escala de precio (splits, precio nominal):
    # log(k*P_t) - log(k*P_{t-20}) = log(P_t) - log(P_{t-20}).
    # El rs nominal (ratio de precio accion / precio ETF) solo se usa
    # en LIS (slpm_v12.py), marcado como provisional.
    if leader_data:
        lines.append('*Nota: la componente rs_z del WLS usa RS Mom '
                     '(log-retorno relativo 20d), no el RS nominal. '
                     'Es invariante a escala de precio (splits, precio '
                     'nominal). El RS nominal (ratio de precio accion / '
                     'precio ETF) solo se usa en LIS, marcado como '
                     'provisional.*\n')

    full_metrics_data = [m for m in full_metrics_data if m is not None and not m.empty]
    if full_metrics_data:
        full_metrics_df = pd.concat(full_metrics_data, ignore_index=True)
    else:
        full_metrics_df = pd.DataFrame()

    if leader_data:
        leader_df = pd.concat(leader_data, ignore_index=True)
    else:
        leader_df = pd.DataFrame()

    if output_csv and not leader_df.empty:
        cols = ['ticker','sector','rs','rs_mom','flow_proxy_z','wyckoff_score','wyckoff_phase',
                'persistence_5d','persistence_10d','persistence_20d','stability','spring','sos','wls','sector_rank_pct']
        # Fix 2026-09-29: escritura atomica. Es el CSV consumido por
        # src/report/helpers.py:117 (pd.read_csv) y por scripts/generate_docs.
        # Un to_csv truncado rompe el reporte diario.
        _tmp = str(output_csv) + '.tmp'
        leader_df[cols].to_csv(_tmp, index=False, lineterminator='\n')
        os.replace(_tmp, output_csv)

    return lines, leader_df, full_metrics_df
