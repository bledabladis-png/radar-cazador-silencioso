"""
Macro Sectorial v2.3 – Sistema de analisis macro y rotacion sectorial.
El sistema implementa un conjunto consistente de reglas deterministas,
con normalizacion robusta, separacion modular y metodologia documentada.
No genera recomendaciones de inversion.
"""
import pandas as pd
import os
from src.data_loader import download_market_data
from src.macro_manual_loader import load_macro_manual
from src.stock_data_loader import download_stock_prices
from data.validator import validate_market_data
from regimes.financial_conditions import compute_liquidity_score
from regimes.liquidity import compute_liquidity_score as compute_real_liquidity
from regimes.volatility_regime import compute_volatility_regime
from regimes.macro_regime import compute_macro_regime
from regimes.sector_regime import compute_sector_scores, compute_price_flow_rankings
from src.report_generator import generate_daily_report
from src.utils import get_col
from indicators.breadth import compute_breadth

def main():
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
    liquidity_score, financial_regime, liq_conf = compute_liquidity_score(df_market)
    print(f"  Cond. Financieras: {financial_regime} (conf: {liq_conf:.0%})")

    print("Calculando liquidez real (FRED)...")
    from regimes.liquidity import compute_liquidity_score as compute_real_liquidity
    real_liq_score, real_liq_regime, real_liq_conf = compute_real_liquidity()
    if real_liq_score is not None:
        print(f"  Liquidez real: {real_liq_regime} (conf: {real_liq_conf:.0%})")
    else:
        print("  Liquidez real: no disponible (sin datos FRED)")
        real_liq_score = None
        real_liq_regime = 'N/A'
        real_liq_conf = 0.0

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
    macro_score, macro_regime, macro_conf, _ = compute_macro_regime(
        df_market, df_macro_manual, liquidity_score, vol_score
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
    price_rank, flow_rank = compute_price_flow_rankings(df_market)

    # Breadth ampliado
    b20, b50, b200, nh, nl = compute_breadth(df_market)
    breadth_values = {
        '% sobre EMA20': b20.iloc[-1],
        '% sobre EMA50': b50.iloc[-1],
        '% sobre EMA200': b200.iloc[-1],
        'New Highs (%)': nh.iloc[-1],
        'New Lows (%)': nl.iloc[-1],
    }

    # Modulo de lideres
    leader_lines = None
    try:
        df_stocks = download_stock_prices()
        if df_stocks is not None and not df_stocks.empty:
            holdings_df = pd.read_csv('data/etf_holdings.csv')
            fases = {sector: fase for sector, _, _, fase in sector_results['ranking']}
            oper = {sector: 'OPORTUNIDAD MODERADA' if fase in ['ACUMULACION','MARKUP'] else 'NO OPERAR'
                    for sector, fase in fases.items()}
            from indicators.stock_leader import generate_leader_section
            leader_lines = generate_leader_section(df_market, df_stocks, holdings_df, fases, oper,
                                                   output_csv='outputs/analisis_lideres.csv')
            if leader_lines:
                print("  Lideres sectoriales generados.")
            else:
                print("  No hay sectores favorables para lideres.")
    except Exception as e:
        print(f"  Modulo de lideres omitido: {e}")

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

    print("Generando reporte...")
    generate_daily_report(macro_score, macro_regime, macro_conf,
                          liquidity_score, financial_regime, liq_conf,
                          vol_score, vol_regime, vol_conf,
                          sector_results,
                          price_rank, flow_rank,
                          leader_lines=leader_lines, breadth_values=breadth_values,
                          real_liquidity_regime=real_liq_regime, real_liquidity_conf=real_liq_conf,
                          pcr_data=pcr_data, darkpool_data=darkpool_data)
    print("Reporte generado en outputs/reporte_diario.md")

if __name__ == "__main__":
    main()
