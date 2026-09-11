# -*- coding: utf-8 -*-
"""
Macro Sectorial v4.3 -- Sistema de analisis macro y rotacion sectorial.
Fases 1-4 + Correccion 0.5 + P1 + P2 + Mejoras 16-20.
"""
import pandas as pd
from src.utils import append_dedup
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime
from src.stock_data_loader import download_stock_prices
from src.report_generator import generate_daily_report
from src.pipeline.data_load import load_all_data
from src.pipeline.regimes import compute_all_regimes
from src.pipeline.sectors_base import compute_sectors_base
from src.pipeline.flows_primary import compute_flows_primary
from src.pipeline.flows_secondary import compute_flows_secondary
from src.pipeline.leaders import compute_leaders
from src.pipeline.sector_metrics import compute_sector_metrics
from src.utils import get_col, detect_cross_module_conflict
from src.dependency_tracker import audit_double_counting
from indicators.sector_breadth import compute_sector_breadth
from indicators.sector_breadth_momentum import compute_sector_breadth_momentum
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
    # Fix C22: limpiar CSVs de lideres previos. Si el run actual no los
    # regenera (sin sectores favorables / sin indices en fase), los
    # consumidores (report_generator, run_all_audits, verify_leader_selection)
    # no deben leer datos obsoletos de runs anteriores.
    for _csv_name in ['analisis_lideres.csv', 'analisis_lideres_internacionales.csv']:
        _csv_path = os.path.join('outputs', 'report', _csv_name)
        if os.path.exists(_csv_path):
            try:
                os.remove(_csv_path)
            except Exception:
                pass
    data = load_all_data()
    if data is None:
        return
    df_market = data['df_market']
    df_macro_manual = data['df_macro_manual']

    regimes = compute_all_regimes(df_market, df_macro_manual)
    financial_score = regimes['financial_score']
    financial_regime = regimes['financial_regime']
    liq_conf = regimes['liq_conf']
    real_liq_score = regimes['real_liq_score']
    real_liq_regime = regimes['real_liq_regime']
    real_liq_conf = regimes['real_liq_conf']
    real_liq_prev = regimes['real_liq_prev']
    vol_score = regimes['vol_score']
    vol_regime = regimes['vol_regime']
    vol_conf = regimes['vol_conf']
    macro_score = regimes['macro_score']
    macro_regime = regimes['macro_regime']
    macro_conf = regimes['macro_conf']
    all_signals = regimes['all_signals']

    sb = compute_sectors_base(df_market)
    sector_results = sb['sector_results']
    sector_rank_deltas_df = sb['sector_rank_deltas_df']
    sector_price_rank = sb['sector_price_rank']
    sector_flow_rank = sb['sector_flow_rank']
    otros_price_rank = sb['otros_price_rank']
    otros_flow_rank = sb['otros_flow_rank']
    sector_dispersion_df = sb['sector_dispersion_df']
    sector_corr_summary_df = sb['sector_corr_summary_df']
    sector_corr_matrix_df = sb['sector_corr_matrix_df']
    cross_asset_summary_df = sb['cross_asset_summary_df']
    breadth_values = sb['breadth_values']

    fp = compute_flows_primary(df_market)
    etf_primary_flow_data = fp['etf_primary_flow_data']
    sector_flow_characteristics_df = fp['sector_flow_characteristics_df']
    blackrock_dax_flow = fp['blackrock_dax_flow']
    blackrock_isf_flow = fp['blackrock_isf_flow']
    blackrock_iwm_flow = fp['blackrock_iwm_flow']
    amundi_lyxi_flow = fp['amundi_lyxi_flow']
    qqq_sec_flow = fp['qqq_sec_flow']
    cftc_position_flow_data = fp['cftc_position_flow_data']

    fs = compute_flows_secondary(
        sector_flow_rank, etf_primary_flow_data, cftc_position_flow_data,
        blackrock_dax_flow, blackrock_isf_flow, amundi_lyxi_flow,
    )
    flow_synthesis = fs['flow_synthesis']
    nport_position_change_data = fs['nport_position_change_data']
    qqq_performance_data = fs['qqq_performance_data']
    qqq_nport_flow_data = fs['qqq_nport_flow_data']

    ldr = compute_leaders(df_market, sector_results)
    df_stocks = ldr['df_stocks']
    holdings_df = ldr['holdings_df']
    leader_lines = ldr['leader_lines']
    leader_df = ldr['leader_df']
    full_metrics_df = ldr['full_metrics_df']

    sm = compute_sector_metrics(df_stocks, holdings_df, leader_df, full_metrics_df, df_market)
    sector_leader_divergence_df = sm['sector_leader_divergence_df']
    sector_wyckoff_distribution_df = sm['sector_wyckoff_distribution_df']
    rs_internal_df = sm['rs_internal_df']
    sector_concentration_df = sm['sector_concentration_df']
    leader_representativeness_df = sm['leader_representativeness_df']

    # --- Sector Breadth & Health v1.0 (descriptivo) ---
    # --- Momentum de amplitud sectorial v1.0 (descriptivo) ---
    try:
        if df_stocks is not None and not df_stocks.empty:
            sector_breadth_momentum_df = compute_sector_breadth_momentum(
                'outputs/history/sector_breadth.csv'
            )
            sbm_path = Path('outputs/history/sector_breadth_momentum.csv')
            sbm_path.parent.mkdir(parents=True, exist_ok=True)
            if not sector_breadth_momentum_df.empty:
                if sbm_path.exists():
                    hist_sbm = pd.read_csv(sbm_path)
                    sector_breadth_momentum_df = append_dedup(hist_sbm, sector_breadth_momentum_df, ["date","sector"])
                sector_breadth_momentum_df.to_csv(sbm_path, index=False)
                print("  Momentum de amplitud sectorial calculado.")
        else:
            sector_breadth_momentum_df = None
    except Exception as e:
        print(f"  Momentum de amplitud sectorial omitido: {e}")
        sector_breadth_momentum_df = None

    try:
        if df_stocks is not None and not df_stocks.empty:
            from pathlib import Path as P
            sector_breadth_df = compute_sector_breadth(df_market, df_stocks, holdings_df)
            sb_path = P('outputs/history/sector_breadth.csv')
            sb_path.parent.mkdir(parents=True, exist_ok=True)
            if not sector_breadth_df.empty:
                if sb_path.exists():
                    hist_sb = pd.read_csv(sb_path)
                    sector_breadth_df = append_dedup(hist_sb, sector_breadth_df, ["date","sector"])
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
            except Exception as e:
                print(f"  [WARN] tactical/structural engine: {e}")
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
                rs20 = rs.pct_change(20, fill_method=None)
                pers = compute_persistence(rs20, threshold=0.0, lookback=12)
                sector_persistence[sector_etf] = pers
            except Exception as e:
                print(f"  [WARN] persistence: {e}")
                sector_persistence[sector_etf] = None
        print(f"    Persistence calculada para {len(sector_persistence)} sectores.")
    except Exception as e:
        print(f"    Persistence omitida: {e}")
        sector_persistence = {s: None for s in ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']}

    # Guardar CSV histórico de persistencia sectorial
    try:
        persist_rows = []
        date_val = pd.Timestamp.now().normalize()
        for sec, val in sector_persistence.items():
            persist_rows.append({'date': date_val, 'sector': sec, 'persistence': val})
        persist_df = pd.DataFrame(persist_rows)
        p_path = Path('outputs/history/sector_persistence.csv')
        p_path.parent.mkdir(parents=True, exist_ok=True)
        if p_path.exists():
            hist_p = pd.read_csv(p_path, encoding='utf-8')
            persist_df = append_dedup(hist_p, persist_df, ['date','sector'])
        persist_df.to_csv(p_path, index=False, encoding='utf-8')
        print("  Sector Persistence CSV guardado.")
    except Exception as e:
        print(f"  Sector Persistence CSV omitido: {e}")

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
                rs20 = rs.pct_change(20, fill_method=None).iloc[-1]
                signals['rs20'] = np.tanh(rs20 * 5) if pd.notna(rs20) else 0
            except Exception as e:
                print(f"  [WARN] rs20 signal: {e}")
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
            except Exception as e:
                print(f"  [WARN] price_ret_20d: {e}")
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

    # --- Volatilidad estructural v1.0 (descriptivo) ---
    try:
        from indicators.volatility_structure import compute_volatility_structure
        pcr_vol = {}
        if pcr_data:
            pcr_vol['zscore'] = pcr_data.get('z_score', np.nan)
            # No recalcular percentil; si no existe, se deja NaN
            if 'percentile_20d' in pcr_data:
                pcr_vol['percentile_20d'] = pcr_data['percentile_20d']
        vol_structure_df = compute_volatility_structure(df_market, pcr_data=pcr_vol if pcr_vol else None)
        vs_path = Path('outputs/history/volatility_structure.csv')
        vs_path.parent.mkdir(parents=True, exist_ok=True)
        if not vol_structure_df.empty:
            if vs_path.exists():
                hist_vs = pd.read_csv(vs_path)
                vol_structure_df = append_dedup(hist_vs, vol_structure_df, ['date'])
            vol_structure_df.to_csv(vs_path, index=False)
            print("  Volatilidad estructural calculada.")
        else:
            vol_structure_df = None
    except Exception as e:
        print(f"  Volatilidad estructural omitida: {e}")
        vol_structure_df = None

    # --- Calidad, frescura y cobertura de datos v1.0 (descriptivo) ---
    try:
        from indicators.data_quality import compute_data_quality
        data_quality_df = compute_data_quality()
        dq_path = Path('outputs/history/data_quality.csv')
        dq_path.parent.mkdir(parents=True, exist_ok=True)
        if not data_quality_df.empty:
            if dq_path.exists():
                hist_dq = pd.read_csv(dq_path)
                data_quality_df = append_dedup(hist_dq, data_quality_df, ['date','source'])
            data_quality_df.to_csv(dq_path, index=False)
            print("  Calidad de datos calculada.")
        else:
            data_quality_df = None
    except Exception as e:
        print(f"  Calidad de datos omitida: {e}")
        data_quality_df = None

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
    except Exception as e:
        print(f"  [WARN] t10y3m confirmation: {e}")
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
        else:
            confirmation_data['ad'] = None
            print("    A/D: Sin datos suficientes (cobertura temporal baja). Se omite.")
    except Exception as e:
        print(f"    A/D: Error - {e}")
        confirmation_data['ad'] = None

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
    except Exception as e:
        print(f"  [WARN] audit_double_counting: {e}")

    # --- Matriz de Régimen Sectorial v1.0 (descriptiva) ---
    try:
        from indicators.sector_regime_matrix import build_sector_regime_matrix
        sector_regime_matrix_df = build_sector_regime_matrix(
            sector_breadth_df, sector_flow_characteristics_df, sector_results
        )
        if sector_regime_matrix_df is not None and not sector_regime_matrix_df.empty:
            mp_path = Path('outputs/history/sector_regime_matrix.csv')
            mp_path.parent.mkdir(parents=True, exist_ok=True)
            sector_regime_matrix_df.to_csv(mp_path, index=False, encoding='utf-8')
            print("  Matriz de régimen sectorial calculada.")
        else:
            sector_regime_matrix_df = None
    except Exception as e:
        print(f"  Matriz de régimen sectorial omitida: {e}")
        sector_regime_matrix_df = None

    # --- Matriz de Evidencia v1.0 (descriptiva) ---
    try:
        from indicators.evidence_matrix import compute_evidence_matrix
        evidence_matrix_df = compute_evidence_matrix(
            sector_breadth_df,
            sector_concentration_df,
            sector_flow_characteristics_df,
            sector_wyckoff_distribution_df,
            liquidity_regime=financial_regime,
            real_liquidity_regime=real_liq_regime,
            volatility_regime=vol_regime,
            volatility_score=vol_score,
            liquidity_score=real_liq_score if real_liq_score is not None else financial_score
        )
        if evidence_matrix_df is not None and not evidence_matrix_df.empty:
            em_path = Path('outputs/history/evidence_matrix.csv')
            em_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_matrix_df.to_csv(em_path, index=False)
            print("  Matriz de evidencia calculada.")
        else:
            evidence_matrix_df = None
    except Exception as e:
        print(f"  Matriz de evidencia omitida: {e}")
        evidence_matrix_df = None

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
                          real_liq_prev=real_liq_prev, index_leaders=index_leaders, index_phases=index_phases, sector_breadth_data=sector_breadth_df, sector_concentration_data=sector_concentration_df, sector_flow_characteristics_data=sector_flow_characteristics_df, rs_internal_data=rs_internal_df, sector_rank_deltas_data=sector_rank_deltas_df, sector_regime_matrix_data=sector_regime_matrix_df, leader_representativeness_data=leader_representativeness_df, sector_wyckoff_distribution_data=sector_wyckoff_distribution_df, sector_leader_divergence_data=sector_leader_divergence_df, sector_breadth_momentum_data=sector_breadth_momentum_df,
                          evidence_matrix_data=evidence_matrix_df,
                          sector_dispersion_data=sector_dispersion_df,
                          sector_correlation_summary_data=sector_corr_summary_df,
                          sector_correlation_matrix_data=sector_corr_matrix_df,
                          cross_asset_context_data=cross_asset_summary_df,
                          volatility_structure_data=vol_structure_df,
                          data_quality_data=data_quality_df,

                          all_signals=all_signals)
    print("Reporte generado en outputs/report/reporte_diario.md")

    # C1-10: side effects movidos desde report_generator.py
    _save_regime_history(macro_score, macro_regime, macro_conf,
                         financial_regime, vol_regime, sector_results)
    _save_sector_rankings(sector_results)
    # Reporte de cobertura europea (descriptivo; no rompe el run si falla)
    try:
        from src.european_coverage import generate_european_coverage_report
        generate_european_coverage_report()
    except Exception as e:
        print(f"  Cobertura europea omitida: {e}")

def _save_regime_history(macro_score, macro_regime, macro_conf,
                         liquidity_regime, vol_regime, sector_results):
    """Persiste la fila del regimen actual en outputs/history/macro_regime.csv.

    Movido desde report_generator.py (C1-10) para separar la generacion de
    texto de los side-effects de persistencia.
    """
    hist_path = "outputs/history/macro_regime.csv"
    new_row = pd.DataFrame({
        "date": [datetime.now()],
        "macro_regime": [macro_regime],
        "macro_score": [macro_score.iloc[-1]],
        "macro_conf": [macro_conf],
        "liquidity_regime": [liquidity_regime],
        "volatility_regime": [vol_regime],
        "sector_regime": [sector_results["regime"]],
    })
    if os.path.exists(hist_path):
        hist = pd.read_csv(hist_path)
        hist = pd.concat([hist, new_row], ignore_index=True)
    else:
        hist = new_row
    hist.to_csv(hist_path, index=False)


def _save_sector_rankings(sector_results):
    """Persiste el ranking de sectores en outputs/report/sector_rankings.csv.

    Movido desde report_generator.py (C1-10).
    """
    sector_df = pd.DataFrame(
        sector_results["ranking"],
        columns=["ticker", "name", "score", "wyckoff_phase"],
    )
    sector_df.to_csv("outputs/report/sector_rankings.csv", index=False)


if __name__ == "__main__":
    main()

















