# -*- coding: utf-8 -*-
"""
Macro Sectorial v4.3 -- Sistema de analisis macro y rotacion sectorial.
Fases 1-4 + Correccion 0.5 + P1 + P2 + Mejoras 16-20.
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime
from src.data_loader import download_market_data
from src.macro_manual_loader import load_macro_manual
from src.stock_data_loader import download_stock_prices
from data.validator import validate_market_data
from regimes.financial_conditions import compute_financial_conditions
from regimes.liquidity import compute_liquidity_score as compute_real_liquidity
from regimes.volatility_regime import compute_volatility_regime
from regimes.macro_regime import compute_macro_regime
from regimes.sector_regime import compute_sector_scores, compute_price_flow_rankings
from src.report_generator import generate_daily_report
from src.utils import get_col, detect_cross_module_conflict
from src.dependency_tracker import audit_double_counting
from indicators.breadth import compute_breadth
from indicators.persistence import compute_persistence
from indicators.signal_agreement import compute_signal_agreement
from indicators.price_flow_divergence import detect_price_flow_divergence
from config.tickers import SECTOR_NAMES, validate_sector_universe
from indicators.index_phase import compute_index_phases
from indicators.index_leaders import select_index_leaders

def main():
    validate_sector_universe()
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
    real_liq_prev = None
    try:
        result = compute_real_liquidity()
        if result[0] is not None:
            real_liq_score, real_liq_regime, real_liq_conf, real_liq_prev = result
            print(f"  Liquidez real: {real_liq_regime} (conf: {real_liq_conf:.0%})")
        else:
            real_liq_score, real_liq_regime, real_liq_conf = None, 'N/A', 0.0
            print("  Liquidez real: no disponible (sin datos FRED)")
    except ValueError:
        # Fallback si la función devuelve solo 3 valores (versión antigua)
        real_liq_score, real_liq_regime, real_liq_conf = compute_real_liquidity()
        if real_liq_score is not None:
            print(f"  Liquidez real: {real_liq_regime} (conf: {real_liq_conf:.0%})")
        else:
            real_liq_score = None
            real_liq_regime = 'N/A'
            real_liq_conf = 0.0
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
                    'flow_z': row['flow_z'] if pd.notna(row.get('flow_z')) else None,
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
    indices_en_acumulacion = [nombre for nombre, fase in index_phases.items() if fase == 'ACCUMULATION']
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
    validation_warnings = []

    if slpm_v12_data:
        slpm_errors = slpm_v12_data.get('validation_errors', [])
        if slpm_errors:
            validation_errors.extend(slpm_errors)
        else:
            validation_warnings.append("SLPM v1.2: estado validado sin errores.")

    nan_checks = {}
    if pcr_data:
        nan_checks['PCR Total'] = pcr_data.get('total_pcr', np.nan)
    if darkpool_data:
        nan_checks['Dark Pool medio'] = darkpool_data.get('media_dark_pool', np.nan)
    if mte_result:
        nan_checks['MSI'] = mte_result.get('msi', np.nan)
        nan_checks['IPI'] = mte_result.get('ipi', np.nan)
    
    for name, val in nan_checks.items():
        if val is None or (isinstance(val, float) and np.isnan(val)):
            validation_errors.append(f"NaN detectado en {name}.")
        else:
            validation_warnings.append(f"{name}: OK ({val:.2f})")

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
        validation_warnings.append(f"Rangos verificados para {sectors_checked} sectores.")

    if slpm_v12_data and tactical_scores and structural_scores:
        leader_etf = slpm_v12_data.get('sector_etf', '')
        if leader_etf and leader_etf in tactical_scores:
            slpm_quadrant = slpm_v12_data.get('opportunity_quadrant', '')
            if slpm_v12_data.get('state') == 'UNRESOLVED' and slpm_quadrant != 'Transition':
                validation_errors.append(f"Opportunity Map inconsistente: estado UNRESOLVED pero cuadrante={slpm_quadrant}.")
            else:
                validation_warnings.append(f"Opportunity Map coherente: {slpm_v12_data.get('sector', '')} -> {slpm_quadrant}.")

    # Comprobacion de Data Freshness
    if darkpool_data:
        week = darkpool_data.get('week', '')
        if week:
            try:
                d = pd.Timestamp(week)
                age = (datetime.now() - d).days
                if age > 14:
                    validation_warnings.append(f'Data Freshness: FINRA Dark Pool obsoleto ({age} dias). No se usa en clasificacion.')
            except:
                pass

    if pcr_data:
        last_date = pcr_data.get('last_date', '')
        if last_date and last_date != 'N/A':
            try:
                d = pd.Timestamp(last_date)
                age = (datetime.now() - d).days
                if age > 5:
                    validation_warnings.append(f'Data Freshness: CBOE PCR desactualizado ({age} dias).')
            except:
                pass

    # Validacion de configuracion
    try:
        from config.weights import validate_weights
        validate_weights()
        validation_warnings.append("Config: pesos validados correctamente.")
    except Exception as e:
        validation_errors.append(f"Config: error en pesos - {e}")

    # MEJORA 20: Auditoria de double-counting
    try:
        dc_audit = audit_double_counting()
        critical_vars = len(dc_audit.get('critical', []))
        high_vars = len(dc_audit.get('high', []))
        if critical_vars > 0:
            validation_warnings.append(f"Anti-Double-Counting: {critical_vars} variables criticas (4+ modulos), {high_vars} altas (3 modulos). Ver informe para detalle.")
        else:
            validation_warnings.append(f"Anti-Double-Counting: sin variables criticas. {high_vars} variables compartidas por 3 modulos.")
    except Exception as e:
        validation_warnings.append(f"Anti-Double-Counting: Error en auditoria - {e}")

    # Verificar cobertura de breadth\n    try:\n        if b20 is not None and len(b20) > 0:\n            last_b20 = b20.iloc[-1] if hasattr(b20, "iloc") else b20\n            if pd.notna(last_b20) and 0 < last_b20 < 1:\n                validation_warnings.append(f"Breadth: cobertura parcial ({last_b20:.0%}). Universo puede estar incompleto.")\n    except:\n        pass\n\n    if validation_errors:
        print(f"    VALIDATION GATE: {len(validation_errors)} errores, {len(validation_warnings)} advertencias")
        for err in validation_errors:
            print(f"      {err}")
    else:
        print(f"    VALIDATION GATE: Sin errores ({len(validation_warnings)} comprobaciones OK)")

    # Generar resumen de double-counting para el reporte
    dc_summary = ""
    try:
        dc_audit = audit_double_counting()
        dc_summary = dc_audit.get('summary', '')
    except:
        pass

    print("Generando reporte...")
    generate_daily_report(macro_score, macro_regime, macro_conf,
                          financial_score, financial_regime, liq_conf,
                          vol_score, vol_regime, vol_conf,
                          sector_results,
                          sector_price_rank, sector_flow_rank, otros_price_rank, otros_flow_rank,
                          leader_lines=leader_lines, breadth_values=breadth_values,
                          real_liquidity_regime=real_liq_regime, real_liquidity_conf=real_liq_conf,
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
                          real_liq_prev=real_liq_prev, index_leaders=index_leaders, index_phases=index_phases,
                          all_signals=all_signals)
    print("Reporte generado en outputs/report/reporte_diario.md")

if __name__ == "__main__":
    main()
















