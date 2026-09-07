# -*- coding: utf-8 -*-
"""
Macro Sectorial v4.3 -- Sistema de analisis macro y rotacion sectorial.
Fases 1-4 + Correccion 0.5 + P1 + P2 + Mejoras 16-20.
"""
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from src.data_loader import download_market_data
from src.macro_manual_loader import load_macro_manual
from src.stock_data_loader import download_stock_prices
from data.providers.ssga_fund_data import get_etf_primary_flow_data
from data.providers.blackrock_fund_data import get_blackrock_dax_primary_flow
from data.providers.blackrock_isf_fund_data import get_blackrock_isf_primary_flow
from data.providers.blackrock_iwm_fund_data import get_blackrock_iwm_primary_flow
from data.providers.amundi_fund_data import get_amundi_lyxi_primary_flow
from data.providers.cftc_data import get_cftc_position_flow_data
from data.providers.retry_utils import retry_call
from data.providers.qqq_sec_primary_flow import get_qqq_sec_primary_flow
from data.validator import validate_market_data
from regimes.financial_conditions import compute_financial_conditions
from regimes.liquidity import compute_liquidity_score as compute_real_liquidity
from regimes.volatility_regime import compute_volatility_regime
from regimes.macro_regime import compute_macro_regime
from regimes.sector_regime import compute_sector_scores, compute_price_flow_rankings
from src.report_generator import generate_daily_report
from src.utils import get_col, detect_cross_module_conflict
from src.dependency_tracker import audit_double_counting
from indicators.sector_breadth import compute_sector_breadth
from indicators.sector_concentration import compute_sector_concentration
from indicators.sector_flow_characteristics import compute_sector_flow_characteristics
from indicators.sector_rank_history import update_rank_history
from indicators.sector_regime_matrix import build_sector_regime_matrix
from indicators.leader_representativeness import compute_leader_representativeness
from indicators.sector_wyckoff_distribution import compute_sector_wyckoff_distribution
from indicators.rs_internal import compute_rs_internal
from indicators.breadth import compute_breadth
from indicators.persistence import compute_persistence
from indicators.signal_agreement import compute_signal_agreement
from indicators.price_flow_divergence import detect_price_flow_divergence
from config.tickers import validate_sector_universe
from indicators.index_phase import compute_index_phases
from indicators.index_leaders import select_index_leaders

def main():
    validate_sector_universe()
    # Crear subcarpetas de outputs necesarias para ejecución limpia
    for subdir in ['report', 'history', 'state', 'holdings', 'audit', 'cache']:
        os.makedirs(f'outputs/{subdir}', exist_ok=True)
    print("Descargando datos de mercado...")
    df_market = download_market_data()
    if df_market is None or df_market.empty:
        print("Error: no se pudieron descargar datos.")
        return

    print("Validando datos...")
    valid, issues = validate_market_data(df_market)
    if issues:
        for t, msg in issues.items():
            print(f"  {t}: {msg}")

    if len(valid) < 5:
        print("Pocos tickers validos. Abortando.")
        return

    print("Cargando datos macro manuales (si existen)...")
    df_macro_manual = load_macro_manual()
    if df_macro_manual is not None:
        print(f"  Datos manuales cargados: {len(df_macro_manual)} filas.")

    print("Calculando regimen de Cond. Financieras...")
    financial_score, financial_regime, liq_conf = compute_financial_conditions(df_market)
    print(f"  Cond. Financieras: {financial_regime} (conf: {liq_conf:.0%})")

    print("Calculando liquidez real (FRED)...")
    result = compute_real_liquidity()
    if result[0] is not None:
        real_liq_score, real_liq_regime, real_liq_conf, real_liq_prev = result
        print(f"  Liquidez real: {real_liq_regime} (conf: {real_liq_conf:.0%})")
    else:
        real_liq_score, real_liq_regime, real_liq_conf, real_liq_prev = None, 'N/A', 0.0, None
        print("  Liquidez real: no disponible (sin datos FRED)")
    print("Calculando regimen de volatilidad...")
    try:
        vix_close = get_col(df_market, '^VIX', 'Close')
        vix_returns = vix_close.pct_change(fill_method=None)
    except KeyError:
        print("  ^VIX no disponible, usando volatilidad plana.")
        vix_returns = pd.Series(dtype=float)

    vol_score, vol_regime, vol_conf = compute_volatility_regime(vix_returns)
    print(f"  Volatilidad: {vol_regime} (conf: {vol_conf:.0%})")

    print("Calculando regimen macro...")
    macro_score, macro_regime, macro_conf, all_signals = compute_macro_regime(
        df_market, df_macro_manual, financial_score, vol_score
    )
    print(f"  Macro: {macro_regime} (conf: {macro_conf:.0%})")

    print("Calculando rankings sectoriales...")
    sector_results = compute_sector_scores(df_market)
    if sector_results:
        top3 = sector_results['ranking'][:3]
        print("  Top 3 sectores:")
        for i, (t, n, s, w) in enumerate(top3, 1):
            print(f"    {i}. {n} ({t}): {s:.2f} [{w}]")
        print(f"  Regimen sectorial: {sector_results['regime']}")

    # --- Rotación sectorial histórica reciente v1.0 ---
    try:
        from indicators.sector_rank_history import update_rank_history
        sector_rank_history_df, sector_rank_deltas_df = update_rank_history(
            sector_results, 'outputs/history/sector_rank_history.csv', date=pd.Timestamp.now().normalize()
        )
        if sector_rank_deltas_df is not None and not sector_rank_deltas_df.empty:
            print("  Rotación sectorial histórica calculada.")
        else:
            sector_rank_history_df, sector_rank_deltas_df = None, None
    except Exception as e:
        print(f"  Rotación sectorial omitida: {e}")
        sector_rank_history_df, sector_rank_deltas_df = None, None

    print("Calculando rankings de precio y flujo...")
    sector_price_rank, sector_flow_rank, otros_price_rank, otros_flow_rank = compute_price_flow_rankings(df_market)

    # Breadth ampliado
    b20, b50, b200, nh, nl = compute_breadth(df_market)

    breadth_values = {
        '% sobre EMA20': b20.iloc[-1],
        '% sobre EMA50': b50.iloc[-1],
        '% sobre EMA200': b200.iloc[-1],
        'New Highs (%)': nh.iloc[-1],
        'New Lows (%)': nl.iloc[-1],
        'EMA20 count': int(round(b20.iloc[-1] * 11)),
        'EMA50 count': int(round(b50.iloc[-1] * 11)),
        'EMA200 count': int(round(b200.iloc[-1] * 11)),
        'New Highs count': int(round(nh.iloc[-1] * 11)),
        'New Lows count': int(round(nl.iloc[-1] * 11)),
    }

    # Flujo primario ETF (SSGA)
    print("Calculando ETF Primary Flow (SSGA)...")
    try:
        etf_primary_flow_data = retry_call(get_etf_primary_flow_data)
        if etf_primary_flow_data is not None and not etf_primary_flow_data.empty:
            print("  ETF Primary Flow calculado.")
        else:
            etf_primary_flow_data = None
            print("  ETF Primary Flow sin datos.")
    except Exception as e:
        print(f"  ETF Primary Flow omitido: {e}")
        etf_primary_flow_data = None

    # --- Sector Flow Characteristics v1.0 (descriptivo) ---
    try:
        if etf_primary_flow_data is not None and not etf_primary_flow_data.empty:
            sector_flow_characteristics_df = compute_sector_flow_characteristics('outputs/history/etf_primary_flow.csv', df_market)
            sfc_path = Path('outputs/history/sector_flow_characteristics.csv')
            sfc_path.parent.mkdir(parents=True, exist_ok=True)
            if not sector_flow_characteristics_df.empty:
                if sfc_path.exists():
                    hist_sfc = pd.read_csv(sfc_path)
                    sector_flow_characteristics_df = pd.concat([hist_sfc, sector_flow_characteristics_df], ignore_index=True)
                sector_flow_characteristics_df.to_csv(sfc_path, index=False)
                print("  Sector Flow Characteristics calculado.")
        else:
            sector_flow_characteristics_df = None
    except Exception as e:
        print(f"  Sector Flow Characteristics omitido: {e}")
        sector_flow_characteristics_df = None

    # Flujo primario DAXEX (BlackRock)
    print("Calculando DAXEX Primary Flow (BlackRock)...")
    try:
        blackrock_dax_flow = retry_call(get_blackrock_dax_primary_flow)
        if blackrock_dax_flow is not None and not blackrock_dax_flow.empty:
            print("  DAXEX Primary Flow calculado.")
        else:
            blackrock_dax_flow = None
            print("  DAXEX Primary Flow sin datos.")
    except Exception as e:
        print(f"  DAXEX Primary Flow omitido: {e}")
        blackrock_dax_flow = None

    # Flujo primario ISF.L (BlackRock)
    print("Calculando ISF.L Primary Flow (BlackRock)...")
    try:
        blackrock_isf_flow = retry_call(get_blackrock_isf_primary_flow)
        if blackrock_isf_flow is not None and not blackrock_isf_flow.empty:
            print("  ISF.L Primary Flow calculado.")
        else:
            blackrock_isf_flow = None
            print("  ISF.L Primary Flow sin datos.")
    except Exception as e:
        print(f"  ISF.L Primary Flow omitido: {e}")
        blackrock_isf_flow = None

    # Flujo primario IWM (BlackRock)
    print("Calculando IWM Primary Flow (BlackRock)...")
    try:
        blackrock_iwm_flow = retry_call(get_blackrock_iwm_primary_flow)
        if blackrock_iwm_flow is not None and not blackrock_iwm_flow.empty:
            print("  IWM Primary Flow calculado.")
        else:
            blackrock_iwm_flow = None
            print("  IWM Primary Flow sin datos.")
    except Exception as e:
        print(f"  IWM Primary Flow omitido: {e}")
        blackrock_iwm_flow = None

    # Flujo primario LYXI (Amundi)
    print("Calculando LYXI Primary Flow (Amundi)...")
    try:
        amundi_lyxi_flow = retry_call(get_amundi_lyxi_primary_flow)
        if amundi_lyxi_flow is not None and not amundi_lyxi_flow.empty:
            print("  LYXI Primary Flow calculado.")
        else:
            amundi_lyxi_flow = None
            print("  LYXI Primary Flow sin datos.")
    except Exception as e:
        print(f"  LYXI Primary Flow omitido: {e}")
        amundi_lyxi_flow = None

    # Flujo primario QQQ (SEC, semestral/anual)
    print("Cargando QQQ SEC Primary Flow...")
    try:
        qqq_sec_flow = get_qqq_sec_primary_flow()
        if qqq_sec_flow is not None and not qqq_sec_flow.empty:
            print("  QQQ SEC Primary Flow cargado.")
        else:
            qqq_sec_flow = None
            print("  QQQ SEC Primary Flow sin datos.")
    except Exception as e:
        print(f"  QQQ SEC Primary Flow omitido: {e}")
        qqq_sec_flow = None

    # Posicionamiento CFTC (TFF, semanal)
    print("Calculando CFTC Position Flow (TFF)...")
    try:
        cftc_position_flow_data = retry_call(get_cftc_position_flow_data)
        if cftc_position_flow_data is not None and not cftc_position_flow_data.empty:
            print("  CFTC Position Flow calculado.")
        else:
            cftc_position_flow_data = None
            print("  CFTC Position Flow sin datos.")
    except Exception as e:
        print(f"  CFTC Position Flow omitido: {e}")
        cftc_position_flow_data = None

    # Sintesis descriptiva de flujo (sin superindicador)
    flow_synthesis = {}
    try:
        # Dirección de flow_proxy (promedio de flow_proxy_z de líderes sectoriales si existen, si no 0)
        proxy_sign = 0.0
        proxy_sign = 0.0
        flow_synthesis['flow_proxy_sign'] = proxy_sign

        # Dirección de ETF Primary Flow (promedio de primary_flow_z)
        primary_sign = 0.0
        if etf_primary_flow_data is not None and not etf_primary_flow_data.empty:
            primary_sign = float(etf_primary_flow_data['primary_flow_z'].mean())
        flow_synthesis['etf_primary_flow_sign'] = primary_sign

        # Dirección de CFTC Position Flow (promedio de flow_z)
        cftc_sign = 0.0
        if cftc_position_flow_data is not None and not cftc_position_flow_data.empty and 'flow_z' in cftc_position_flow_data.columns:
            cftc_sign = float(cftc_position_flow_data['flow_z'].mean())
        flow_synthesis['cftc_flow_sign'] = cftc_sign

        # Dirección de Europa Primary Flow (promedio de flow_zscore de DAXEX, ISF.L, LYXI)
        europe_sign = 0.0
        european_flows = []
        if blackrock_dax_flow is not None and not blackrock_dax_flow.empty and 'flow_zscore' in blackrock_dax_flow.columns:
            european_flows.append(float(blackrock_dax_flow['flow_zscore'].iloc[-1]))
        if blackrock_isf_flow is not None and not blackrock_isf_flow.empty and 'flow_zscore' in blackrock_isf_flow.columns:
            european_flows.append(float(blackrock_isf_flow['flow_zscore'].iloc[-1]))
        if amundi_lyxi_flow is not None and not amundi_lyxi_flow.empty and 'flow_zscore' in amundi_lyxi_flow.columns:
            european_flows.append(float(amundi_lyxi_flow['flow_zscore'].iloc[-1]))
        if european_flows:
            europe_sign = float(sum(european_flows) / len(european_flows))
        flow_synthesis['european_flow_sign'] = europe_sign

        # Cargar datos N-PORT más recientes para el reporte
        nport_position_change_data = None
        try:
            import pandas as pd
            nport_path = Path('outputs/history/sec_nport_position_change_quarterly.csv')
            if nport_path.exists():
                df_nport = pd.read_csv(nport_path, parse_dates=['REPORT_DATE'])
                if not df_nport.empty:
                    nport_position_change_data = df_nport.sort_values('REPORT_DATE', ascending=False).head(20)
        except Exception as e:
            print(f"  N-PORT no cargado: {e}")
            nport_position_change_data = None

        # Cargar flujo de participaciones QQQ NPORT-P
        qqq_nport_flow_data = None
        try:
            qqq_nport_path = Path('outputs/history/qqq_nport_flow.csv')
            if qqq_nport_path.exists():
                qqq_nport_flow_data = pd.read_csv(qqq_nport_path)
        except Exception as e:
            print(f"  QQQ NPORT-P no cargado: {e}")
            qqq_nport_flow_data = None

        # Cargar rendimientos QQQ desde Yahoo Finance (si existe)
        qqq_performance_data = None
        try:
            perf_path = Path('outputs/history/qqq_returns_yahoo.csv')
            if perf_path.exists():
                mtime = datetime.fromtimestamp(perf_path.stat().st_mtime)
                age = datetime.now() - mtime
                if age <= timedelta(days=7):
                    qqq_performance_data = pd.read_csv(perf_path)
                else:
                    print(f"  QQQ returns Yahoo omitidos: datos con {age.days} días")
            else:
                print("  ⚠️ QQQ returns Yahoo no disponibles: no se encontró outputs/history/qqq_returns_yahoo.csv. La sección de rendimiento QQQ se omitirá.")
        except Exception as e:
            print(f"  QQQ returns Yahoo no cargados: {e}")
            qqq_performance_data = None

        # Conteo de coincidencia de signos
        signs = []
        for s in [proxy_sign, primary_sign, cftc_sign, europe_sign]:
            if s > 0.1:
                signs.append(1)
            elif s < -0.1:
                signs.append(-1)
            else:
                signs.append(0)
        pos = sum(1 for x in signs if x > 0)
        neg = sum(1 for x in signs if x < 0)
        if pos == 3 or neg == 3:
            flow_synthesis['confidence'] = 'ALTA'
        elif pos == 2 or neg == 2:
            flow_synthesis['confidence'] = 'MEDIA'
        else:
            flow_synthesis['confidence'] = 'BAJA'
        print("  Sintesis de flujo calculada.")
    except Exception as e:
        print(f"  Sintesis de flujo omitida: {e}")
        flow_synthesis = {}

    # Modulo de lideres (solo para sectores en acumulacion/markup)
    leader_lines = None
    df_stocks = None
    try:
        df_stocks = download_stock_prices()
        if df_stocks is not None and not df_stocks.empty:
            holdings_df = pd.read_csv('data/etf_holdings.csv')
            fases = {sector: fase for sector, _, _, fase in sector_results['ranking']}
            oper = {sector: 'OPORTUNIDAD MODERADA' if fase in ['ACCUMULATION','MARKUP'] else 'NO OPERAR'
                    for sector, fase in fases.items()}
            from indicators.stock_leader import generate_leader_section
            leader_lines, leader_df = generate_leader_section(df_market, df_stocks, holdings_df, fases, oper,
                                                   output_csv='outputs/report/analisis_lideres.csv')
            if leader_lines:
                print("  Lideres sectoriales generados.")
            else:
                print("  No hay sectores favorables para lideres.")
    except Exception as e:
        print(f"  Modulo de lideres omitido: {e}")

    # --- Distribución Wyckoff sectorial v1.0 (descriptivo) ---
    try:
        if df_stocks is not None and not df_stocks.empty:
            sector_wyckoff_distribution_df = compute_sector_wyckoff_distribution(df_stocks, holdings_df)
            wyckoff_path = Path('outputs/history/sector_wyckoff_distribution.csv')
            wyckoff_path.parent.mkdir(parents=True, exist_ok=True)
            if not sector_wyckoff_distribution_df.empty:
                if wyckoff_path.exists():
                    hist_wy = pd.read_csv(wyckoff_path)
                    sector_wyckoff_distribution_df = pd.concat([hist_wy, sector_wyckoff_distribution_df], ignore_index=True)
                sector_wyckoff_distribution_df.to_csv(wyckoff_path, index=False)
                print("  Distribución Wyckoff sectorial calculada.")
        else:
            sector_wyckoff_distribution_df = None
    except Exception as e:
        print(f"  Distribución Wyckoff sectorial omitida: {e}")
        sector_wyckoff_distribution_df = None

    # --- RS Interno y Absoluto v1.0 (descriptivo) ---
    try:
        if df_stocks is not None and not df_stocks.empty:
            rs_internal_df = compute_rs_internal(df_stocks, holdings_df, df_market, benchmark='SPY')
            rs_path = Path('outputs/history/rs_internal.csv')
            rs_path.parent.mkdir(parents=True, exist_ok=True)
            if not rs_internal_df.empty:
                if rs_path.exists():
                    hist_rs = pd.read_csv(rs_path)
                    rs_internal_df = pd.concat([hist_rs, rs_internal_df], ignore_index=True)
                rs_internal_df.to_csv(rs_path, index=False)
                print("  RS Interno y Absoluto calculado.")
        else:
            rs_internal_df = None
    except Exception as e:
        print(f"  RS Interno y Absoluto omitido: {e}")
        rs_internal_df = None

    # --- Sector Concentration v1.0 (descriptivo) ---
    try:
        if df_stocks is not None and not df_stocks.empty and leader_df is not None and not leader_df.empty:
            sector_concentration_df = compute_sector_concentration(df_stocks, holdings_df, leader_df)
            sc_path = Path('outputs/history/sector_concentration.csv')
            sc_path.parent.mkdir(parents=True, exist_ok=True)
            if not sector_concentration_df.empty:
                if sc_path.exists():
                    hist_sc = pd.read_csv(sc_path)
                    sector_concentration_df = pd.concat([hist_sc, sector_concentration_df], ignore_index=True)
                sector_concentration_df.to_csv(sc_path, index=False)
                print("  Sector Concentration calculado.")
        else:
            sector_concentration_df = None
    except Exception as e:
        print(f"  Sector Concentration omitido: {e}")
        sector_concentration_df = None

    # --- Representatividad del líder v1.0 (descriptivo) ---
    try:
        if leader_df is not None and not leader_df.empty:
            leader_representativeness_df = compute_leader_representativeness(
                leader_df, 'outputs/history/sector_concentration.csv'
            )
            lr_path = Path('outputs/history/leader_representativeness.csv')
            lr_path.parent.mkdir(parents=True, exist_ok=True)
            if not leader_representativeness_df.empty:
                if lr_path.exists():
                    hist_lr = pd.read_csv(lr_path)
                    leader_representativeness_df = pd.concat([hist_lr, leader_representativeness_df], ignore_index=True)
                leader_representativeness_df.to_csv(lr_path, index=False)
                print("  Representatividad del líder calculada.")
        else:
            leader_representativeness_df = None
    except Exception as e:
        print(f"  Representatividad del líder omitida: {e}")
        leader_representativeness_df = None

    # --- Sector Breadth & Health v1.0 (descriptivo) ---
    try:
        if df_stocks is not None and not df_stocks.empty:
            from pathlib import Path as P
            sector_breadth_df = compute_sector_breadth(df_market, df_stocks, holdings_df)
            sb_path = P('outputs/history/sector_breadth.csv')
            sb_path.parent.mkdir(parents=True, exist_ok=True)
            if not sector_breadth_df.empty:
                if sb_path.exists():
                    hist_sb = pd.read_csv(sb_path)
                    sector_breadth_df = pd.concat([hist_sb, sector_breadth_df], ignore_index=True)
                sector_breadth_df.to_csv(sb_path, index=False)
                print("  Sector Breadth & Health calculado.")
        else:
            sector_breadth_df = None
    except Exception as e:
        print(f"  Sector Breadth & Health omitido: {e}")
        sector_breadth_df = None

    # --- NUEVO: Forzar lideres del sector #1 para el SLPM ---
    leader_metrics_for_slpm = []
    top_sector_ticker = sector_results['ranking'][0][0]
    top_sector_flow = 0.0
    for t, f in sector_flow_rank + otros_flow_rank:
        if t == top_sector_ticker:
            top_sector_flow = f
            break

    try:
        if leader_df is not None and not leader_df.empty:
            top_etf = top_sector_ticker
            sector1_df = leader_df[leader_df['sector'] == top_etf]
            for _, row in sector1_df.head(5).iterrows():
                leader_metrics_for_slpm.append({
                    'ticker': row['ticker'],
                    'rs': row['rs'] if pd.notna(row.get('rs')) else None,
                    'rs_momentum': row['rs_mom'] if pd.notna(row.get('rs_mom')) else None,
                    'flow_proxy_z': row['flow_proxy_z'] if pd.notna(row.get('flow_proxy_z')) else None,
                    'wyckoff_phase': row['wyckoff_phase'] if pd.notna(row.get('wyckoff_phase')) else ''
                })
            print(f"    Lideres forzados para SLPM ({top_etf}): {len(leader_metrics_for_slpm)} tickers")
    except Exception as e:
        print(f"    No se pudieron forzar lideres para SLPM: {e}")

    # --- Tactical & Structural Engines ---
    tactical_scores = {}
    structural_scores = {}
    try:
        from regimes.tactical_engine import compute_tactical_score
        from regimes.structural_engine import compute_structural_score
        for sector_etf in ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']:
            try:
                tactical_scores[sector_etf] = compute_tactical_score(df_market, sector_etf)
                structural_scores[sector_etf] = compute_structural_score(df_market, sector_etf)
            except:
                tactical_scores[sector_etf] = 0.0
                structural_scores[sector_etf] = 0.0
        print(f"    Tactical/Structural engines calculados para {len(tactical_scores)} sectores.")
    except Exception as e:
        print(f"    Tactical/Structural engines omitidos: {e}")

    # --- Persistence ---
    sector_persistence = {}
    try:
        for sector_etf in ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']:
            try:
                close_sector = get_col(df_market, sector_etf, 'Close')
                close_spy = get_col(df_market, '^GSPC', 'Close')
                rs = close_sector / close_spy
                rs20 = rs.pct_change(20)
                pers = compute_persistence(rs20, threshold=0.0, lookback=12)
                sector_persistence[sector_etf] = pers
            except:
                sector_persistence[sector_etf] = None
        print(f"    Persistence calculada para {len(sector_persistence)} sectores.")
    except Exception as e:
        print(f"    Persistence omitida: {e}")
        sector_persistence = {s: None for s in ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']}

    # --- SLPM v1.1 (legacy) ---
#     slpm_data = None
#     try:
#         from indicators.structural_leadership import evaluate_slpm
#         slpm_data = evaluate_slpm(df_market, sector_results, leader_metrics_for_slpm, top_sector_flow)
#         if slpm_data:
#             print(f"    SLPM v1.1 (legacy): {slpm_data['state']} ({slpm_data['sector']})")
#     except Exception as e:
#         print(f"    SLPM v1.1 omitido: {e}")

    # --- SLPM v1.2 (State Machine centralizada) ---
    slpm_v12_data = None
    try:
        from indicators.slpm_v12 import evaluate_slpm_v12
        slpm_v12_data = evaluate_slpm_v12(
            df_market, sector_results, leader_metrics_for_slpm, top_sector_flow,
            tactical_scores=tactical_scores,
            structural_scores=structural_scores,
            sector_persistence=sector_persistence
        )
        if slpm_v12_data:
            state = slpm_v12_data.get('state', '?')
            lis = slpm_v12_data.get('leader_integrity', {}).get('lis', 0)
            breadth = slpm_v12_data.get('leader_breadth_v2', {}).get('composite', 0)
            t_score = slpm_v12_data.get('tactical_score', 0)
            s_score = slpm_v12_data.get('structural_score', 0)
            errors = slpm_v12_data.get('validation_errors', [])
            error_msg = f" ({len(errors)} errores)" if errors else ""
            print(f"    SLPM v1.2: {state} | T={t_score:+.2f} S={s_score:+.2f} LIS={lis:.2f} Breadth={breadth:.2f}{error_msg}")
    except Exception as e:
        print(f"    SLPM v1.2 omitido: {e}")

    # --- Directional Agreement con direccion ---
    signal_agreements = {}
    signal_agreements_display = {}
    try:
        for sector_etf in ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']:
            signals = {}
            signals['tactical'] = tactical_scores.get(sector_etf, 0)
            signals['structural'] = structural_scores.get(sector_etf, 0)
            try:
                close_sector = get_col(df_market, sector_etf, 'Close')
                close_spy = get_col(df_market, '^GSPC', 'Close')
                rs = close_sector / close_spy
                rs20 = rs.pct_change(20).iloc[-1]
                signals['rs20'] = np.tanh(rs20 * 5) if pd.notna(rs20) else 0
            except:
                signals['rs20'] = 0
            flow_val = next((f for t, f in sector_flow_rank if t == sector_etf), 0)
            signals['flow'] = flow_val
            result = compute_signal_agreement(signals)
            signal_agreements[sector_etf] = result['agreement']
            signal_agreements_display[sector_etf] = result['display']
        print(f"    Directional Agreement calculado para {len(signal_agreements)} sectores.")
    except Exception as e:
        print(f"    Directional Agreement omitido: {e}")
        signal_agreements = {s: 0.5 for s in ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']}
        signal_agreements_display = {s: '50% MIXED' for s in ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']}

    # --- Price-Flow Divergence ---
    price_flow_divergences = {}
    try:
        for sector_etf in ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']:
            try:
                close_sector = get_col(df_market, sector_etf, 'Close')
                price_ret_20d = (close_sector.iloc[-1] / close_sector.iloc[-21] - 1) if len(close_sector) >= 21 else 0.0
            except:
                price_ret_20d = 0.0
            flow_val = next((f for t, f in sector_flow_rank if t == sector_etf), 0)
            price_flow_divergences[sector_etf] = detect_price_flow_divergence(price_ret_20d, flow_val)
        for sector_etf, div in price_flow_divergences.items():
            if div['status'] != 'ALIGNED':
                name = sector_etf
                print(f"    Price-Flow Divergence [{name}]: {div['status']}")
        print(f"    Price-Flow Divergence calculado para {len(price_flow_divergences)} sectores.")
    except Exception as e:
        print(f"    Price-Flow Divergence omitido: {e}")
        price_flow_divergences = {s: {'status': 'ALIGNED', 'message': ''} for s in ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']}

    # --- Shock Sensitivity ---
    shock_sensitivities = {}
    try:
        from indicators.commodity_market_correlation import compute_commodity_market_correlation
        for sector_etf in ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']:
            shock_sensitivities[sector_etf] = compute_commodity_market_correlation(df_market, sector_etf)
        print(f"    Shock Sensitivity calculada para {len(shock_sensitivities)} sectores.")
    except Exception as e:
        print(f"    Shock Sensitivity omitida: {e}")
        shock_sensitivities = {s: {} for s in ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']}

    print("Calculando sentimiento de opciones (PCR)...")
    pcr_data = None
    try:
        from indicators.options import compute_pcr_signals
        pcr_data = compute_pcr_signals()
        if pcr_data and pcr_data.get('status') == 'OK':
            print(f"  PCR Total: {pcr_data['total_pcr']:.2f} (Z: {pcr_data['z_score']:.2f}, Estado: {pcr_data['state']})")
        elif pcr_data:
            print(f"  OMS STATUS: {pcr_data['status']}")
    except Exception as e:
        print(f"  Modulo PCR omitido: {e}")

    print("Calculando Dark Pools (FINRA ATS)...")
    darkpool_data = None
    try:
        from indicators.darkpool import compute_darkpool_signals
        darkpool_data = compute_darkpool_signals()
        if darkpool_data:
            print(f"  Dark Pool medio: {darkpool_data['media_dark_pool']:.2f}% "
                  f"({darkpool_data['n_tickers_ats']}/{darkpool_data['n_tickers_total']} tickers)")
        else:
            print("  Dark Pools: no disponible")
    except Exception as e:
        print(f"  Modulo Dark Pools omitido: {e}")

    print("Calculando Market Transition Engine...")
    mte_result = None
    try:
        from indicators.mte import compute_mte
        fc_score = financial_score
        cred_signal = all_signals['credit'] if 'all_signals' in dir() and 'credit' in all_signals.columns else 0
        vol_signal = all_signals['volatility'] if 'all_signals' in dir() and 'volatility' in all_signals.columns else 0
        
        # Verificar frescura de Dark Pool antes de pasarlo al MTE
        mte_darkpool = darkpool_data
        if darkpool_data:
            week = darkpool_data.get('week', '')
            if week:
                try:
                    d = pd.Timestamp(week)
                    age = (datetime.now() - d).days
                    if age > 14:
                        print(f"    Dark Pool ARCHIVAL ({age}d). Excluido del MTE.")
                        mte_darkpool = None
                except:
                    pass
        
        mte_result = compute_mte(df_market, fc_score, cred_signal, vol_signal, pcr_data, mte_darkpool)
        if mte_result:
            print(f"  Escenario: {mte_result['scenario']} (MSI: {mte_result['msi']:.0f}, IPI: {mte_result['ipi']:.0f})")
        else:
            print("  MTE no disponible")
    except Exception as e:
        print(f"  Modulo MTE omitido: {e}")

    # --- Cross-Module Conflict Detector ---
    cross_module_conflict = detect_cross_module_conflict(
        macro_regime=macro_regime,
        financial_regime=financial_regime,
        volatility_regime=vol_regime,
        liquidity_regime=real_liq_regime if real_liq_regime != 'N/A' else None,
        mte_scenario=mte_result.get('scenario') if mte_result else None
    )
    if cross_module_conflict['conflict_level'] in ('CONFLICT', 'DIVERGENCE'):
        print(f"    CROSS-MODULE {cross_module_conflict['conflict_level']}: {cross_module_conflict['message']}")

    # --- Institutional Confirmation (Nivel 2) ---
    confirmation_data = {}

    # T10Y3M
    try:
        t10y3m_df = pd.read_csv('data/macro_manual/10y3m.csv', index_col=0, parse_dates=True)
        if not t10y3m_df.empty:
            confirmation_data['t10y3m'] = float(t10y3m_df['T10Y3M'].iloc[-1])
    except:
        confirmation_data['t10y3m'] = None

    # Vol Metrics
    try:
        from indicators.vol_metrics import compute_vol_metrics
        vol_data = compute_vol_metrics(df_market)
        confirmation_data.update(vol_data)
    except Exception as e:
        print(f"    Vol Metrics: Error - {e}")

    # Cross-Asset Ratios con tendencia
    try:
        from indicators.cross_asset import compute_cross_asset_ratios
        ratios = compute_cross_asset_ratios(df_market)
        confirmation_data['ratios'] = ratios
    except Exception as e:
        print(f"    Cross-Asset Ratios: Error - {e}")
        confirmation_data['ratios'] = {}

    # FLS
    try:
        from indicators.fls import compute_fls
        fls_data = compute_fls()
        if fls_data:
            confirmation_data['fls'] = fls_data
            stressed = fls_data.get('stressed_components', fls_data.get('components', 0))
            total_comp = fls_data.get('total_components', 5)
            print(f"    FLS: {fls_data['fls_normalized']:.2f} ({stressed}/{total_comp} componentes en estres)")
    except Exception as e:
        print(f"    FLS: Error - {e}")

    # Advance/Decline
    try:
        from indicators.breadth_equity import compute_advance_decline
        ad_data = compute_advance_decline(df_stocks) if df_stocks is not None else None
        if ad_data:
            confirmation_data['ad'] = ad_data
            print(f"    A/D: Net={ad_data['ad_net']:+d}  NH/NL={ad_data['nh_nl']:+d}  Thrust={ad_data['breadth_thrust']:.2f}")
    except Exception as e:
        print(f"    A/D: Error - {e}")

    if confirmation_data:
        print(f"  Institutional Confirmation: T10Y3M={confirmation_data.get('t10y3m', 'N/A')}%")

    # =====================================================================
    # INDICES INTERNACIONALES - FASES WYCKOFF + LIDERES
    # =====================================================================
    print("Calculando fases Wyckoff para indices internacionales...")
    index_phases, index_data = compute_index_phases(df_market)
    indices_en_acumulacion = [nombre for nombre, fase in index_phases.items() if fase in ['ACCUMULATION', 'MARKUP']]
    if indices_en_acumulacion:
        print(f"  Indices en acumulacion: {', '.join(indices_en_acumulacion)}")
        df_index_stocks = download_stock_prices()
        index_leaders = {}
        for nombre in indices_en_acumulacion:
            try:
                leaders_single = select_index_leaders(None, df_index_stocks, [nombre])
                if nombre in leaders_single and not leaders_single[nombre].empty:
                    index_leaders[nombre] = leaders_single[nombre]
                    print(f"    {nombre}: {len(leaders_single[nombre])} empresas seleccionadas")
                else:
                    print(f"    {nombre}: sin lideres disponibles")
            except Exception as e:
                print(f"    {nombre}: error al calcular lideres - {e}")
    else:
        print("  Ningun indice en fase de acumulacion.")
        index_leaders = {}

    # Exportar CSV de lideres internacionales para revision manual
    if index_leaders:
        try:
            all_leaders = []
            for nombre, df in index_leaders.items():
                df_copy = df.copy()
                df_copy['indice'] = nombre
                all_leaders.append(df_copy)
            if all_leaders:
                pd.concat(all_leaders, ignore_index=True).to_csv('outputs/report/analisis_lideres_internacionales.csv', index=False)
                print("  CSV de lideres internacionales generado.")
        except Exception as e:
            print(f"  Error al generar CSV internacional: {e}")

    # =====================================================================
    # VALIDATION GATE
    # =====================================================================
    print("Ejecutando Validation Gate...")
    validation_errors = []
    validation_checks = []

    def add_check(nombre, ok=True, detalle=""):
        if ok:
            validation_checks.append(f"{nombre}: OK {detalle}".strip())
        else:
            validation_errors.append(f"{nombre}: {detalle}".strip())

    # 1. SLPM v1.2
    if slpm_v12_data:
        slpm_errors = slpm_v12_data.get('validation_errors', [])
        if slpm_errors:
            validation_errors.extend(slpm_errors)
        add_check("SLPM v1.2", True, "estado validado")
    else:
        add_check("SLPM v1.2", True, "no disponible")

    # 2. PCR Total
    if pcr_data:
        val = pcr_data.get('total_pcr', np.nan)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            add_check("PCR Total", False, "NaN")
        else:
            add_check("PCR Total", True, f"{val:.2f}")
    else:
        add_check("PCR Total", True, "sin datos")

    # 3. Dark Pool medio
    if darkpool_data:
        val = darkpool_data.get('media_dark_pool', np.nan)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            add_check("Dark Pool medio", False, "NaN")
        else:
            add_check("Dark Pool medio", True, f"{val:.2f}")
    else:
        add_check("Dark Pool medio", True, "sin datos")

    # 4. MTE (MSI/IPI)
    if mte_result:
        msi = mte_result.get('msi', np.nan)
        ipi = mte_result.get('ipi', np.nan)
        if (msi is None or (isinstance(msi, float) and np.isnan(msi)) or
            ipi is None or (isinstance(ipi, float) and np.isnan(ipi))):
            add_check("MTE", False, "NaN en MSI/IPI")
        else:
            add_check("MTE", True, f"MSI={msi:.2f}, IPI={ipi:.2f}")
    else:
        add_check("MTE", True, "sin datos")

    # 5. Rangos tácticos/estructurales
    if tactical_scores and structural_scores:
        sectors_checked = 0
        for ticker in tactical_scores:
            if ticker in structural_scores:
                t = tactical_scores[ticker]
                s = structural_scores[ticker]
                if abs(t) > 1.0:
                    validation_errors.append(f"{ticker}: Tactical Score fuera de rango ({t:+.2f}).")
                if abs(s) > 1.0:
                    validation_errors.append(f"{ticker}: Structural Score fuera de rango ({s:+.2f}).")
                sectors_checked += 1
        add_check("Rangos tácticos/estructurales", True, f"{sectors_checked} sectores")
    else:
        add_check("Rangos tácticos/estructurales", True, "sin datos")

    # 6. Opportunity Map
    if slpm_v12_data and tactical_scores and structural_scores:
        leader_etf = slpm_v12_data.get('sector_etf', '')
        if leader_etf and leader_etf in tactical_scores:
            slpm_quadrant = slpm_v12_data.get('opportunity_quadrant', '')
            if slpm_v12_data.get('state') == 'UNRESOLVED' and slpm_quadrant != 'Transition':
                add_check("Opportunity Map", False, f"inconsistente {slpm_quadrant}")
            else:
                add_check("Opportunity Map", True, f"{slpm_v12_data.get('sector', '')} -> {slpm_quadrant}")
        else:
            add_check("Opportunity Map", True, "sin leader_etf")
    else:
        add_check("Opportunity Map", True, "sin datos")

    # 7. Data Freshness Dark Pool
    if darkpool_data:
        week = darkpool_data.get('week', '')
        if week:
            try:
                d = pd.Timestamp(week)
                age = (datetime.now() - d).days
                if age > 14:
                    add_check("Freshness Dark Pool", True, f"obsoleto {age} dias (advertencia)")
                else:
                    add_check("Freshness Dark Pool", True, f"{age} dias")
            except:
                add_check("Freshness Dark Pool", True, "sin fecha")
        else:
            add_check("Freshness Dark Pool", True, "sin fecha")
    else:
        add_check("Freshness Dark Pool", True, "sin datos")

    # 8. Data Freshness PCR
    if pcr_data:
        last_date = pcr_data.get('last_date', '')
        if last_date and last_date != 'N/A':
            try:
                d = pd.Timestamp(last_date)
                age = (datetime.now() - d).days
                if age > 5:
                    add_check("Freshness PCR", True, f"desactualizado {age} dias (advertencia)")
                else:
                    add_check("Freshness PCR", True, f"{age} dias")
            except:
                add_check("Freshness PCR", True, "sin fecha")
        else:
            add_check("Freshness PCR", True, "sin fecha")
    else:
        add_check("Freshness PCR", True, "sin datos")

    # 9. Configuración de pesos
    try:
        from config.weights import validate_weights
        validate_weights()
        add_check("Config pesos", True, "validados")
    except Exception as e:
        add_check("Config pesos", False, str(e))

    # 10. Anti-Double-Counting
    try:
        import inspect
        from indicators.state_machine import classify_leadership_state
        sig = inspect.signature(classify_leadership_state)
        lis_in_state_machine = 'lis' in sig.parameters

        dc_audit = audit_double_counting()
        critical_vars = len(dc_audit.get('critical', []))
        high_vars = len(dc_audit.get('high', []))

        if lis_in_state_machine:
            add_check("Anti-Double-Counting", False, "LIS aún en State Machine")
        elif critical_vars > 0:
            add_check("Anti-Double-Counting", True, f"corrección LIS activa, {critical_vars} criticas, {high_vars} altas")
        else:
            add_check("Anti-Double-Counting", True, f"corrección LIS activa, sin criticas, {high_vars} compartidas")
    except Exception as e:
        add_check("Anti-Double-Counting", False, str(e))


    if validation_errors:
        print(f"    VALIDATION GATE: {len(validation_errors)} errores, {len(validation_checks)} comprobaciones")
        for err in validation_errors:
            print(f"      {err}")
        sys.exit(1)
    else:
        print(f"    VALIDATION GATE: Sin errores ({len(validation_checks)} comprobaciones OK)")

    # Generar resumen de double-counting para el reporte
    dc_summary = ""
    try:
        dc_audit = audit_double_counting()
        dc_summary = dc_audit.get('summary', '')
    except:
        pass

    # --- Matriz de Régimen Sectorial v1.0 (descriptiva) ---
    try:
        from indicators.sector_regime_matrix import build_sector_regime_matrix
        sector_regime_matrix_df = build_sector_regime_matrix(
            sector_breadth_df, sector_flow_characteristics_df, sector_results
        )
        if sector_regime_matrix_df is not None and not sector_regime_matrix_df.empty:
            mp_path = Path('outputs/history/sector_regime_matrix.csv')
            mp_path.parent.mkdir(parents=True, exist_ok=True)
            sector_regime_matrix_df.to_csv(mp_path, index=False)
            print("  Matriz de régimen sectorial calculada.")
        else:
            sector_regime_matrix_df = None
    except Exception as e:
        print(f"  Matriz de régimen sectorial omitida: {e}")
        sector_regime_matrix_df = None

    print("Generando reporte...")
    generate_daily_report(macro_score, macro_regime, macro_conf,
                          financial_score, financial_regime, liq_conf,
                          vol_score, vol_regime, vol_conf,
                          sector_results,
                          sector_price_rank, sector_flow_rank, otros_price_rank, otros_flow_rank,
                          leader_lines=leader_lines, breadth_values=breadth_values,
                            etf_primary_flow_data=etf_primary_flow_data,
                            blackrock_dax_flow=blackrock_dax_flow,
                            blackrock_isf_flow=blackrock_isf_flow,
                            blackrock_iwm_flow=blackrock_iwm_flow,
                            amundi_lyxi_flow=amundi_lyxi_flow,
                            nport_position_change_data=nport_position_change_data,
                            qqq_performance_data=qqq_performance_data,
                            qqq_nport_flow_data=qqq_nport_flow_data,
                            cftc_position_flow_data=cftc_position_flow_data,
                            qqq_sec_flow=qqq_sec_flow,
                            flow_synthesis=flow_synthesis,
                          real_liquidity_regime=real_liq_regime, real_liquidity_conf=real_liq_conf,
                            real_liq_score=real_liq_score,
                          pcr_data=pcr_data, darkpool_data=darkpool_data, mte_result=mte_result,
                          confirmation_data=confirmation_data,
                          slpm_v12_data=slpm_v12_data,
                          tactical_scores=tactical_scores,
                          structural_scores=structural_scores,
                          sector_persistence=sector_persistence,
                          signal_agreements=signal_agreements,
                          signal_agreements_display=signal_agreements_display,
                          cross_module_conflict=cross_module_conflict,
                          shock_sensitivities=shock_sensitivities,
                          price_flow_divergences=price_flow_divergences,
                          dc_summary=dc_summary,
                          real_liq_prev=real_liq_prev, index_leaders=index_leaders, index_phases=index_phases, sector_breadth_data=sector_breadth_df, sector_concentration_data=sector_concentration_df, sector_flow_characteristics_data=sector_flow_characteristics_df, rs_internal_data=rs_internal_df, sector_rank_deltas_data=sector_rank_deltas_df, sector_regime_matrix_data=sector_regime_matrix_df, leader_representativeness_data=leader_representativeness_df, sector_wyckoff_distribution_data=sector_wyckoff_distribution_df,
                          all_signals=all_signals)
    print("Reporte generado en outputs/report/reporte_diario.md")

if __name__ == "__main__":
    main()

















