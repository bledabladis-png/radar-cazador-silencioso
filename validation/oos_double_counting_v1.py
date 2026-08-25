
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import numpy as np

from data.providers.router import DataRouter
from config.tickers import MARKET_TICKERS
from indicators.persistence import compute_persistence
from indicators.momentum import compute_flow_proxy
from regimes.tactical_engine import compute_tactical_score
from regimes.structural_engine import compute_structural_score
from indicators.state_machine import classify_leadership_state
from src.utils import get_col


def flatten_tickers(mapping):
    tickers = set()
    for value in mapping.values():
        if isinstance(value, str):
            tickers.add(value)
        elif isinstance(value, list):
            tickers.update(value)
        elif isinstance(value, dict):
            tickers.update(flatten_tickers(value))
    return sorted(tickers)


def compute_structural_anterior(df_market, sector_etf, leader_breadth=0.5, flow_structure=0.0, persistence=0.5, benchmark="^GSPC"):
    close_sector = get_col(df_market, sector_etf, "Close")
    close_bench = get_col(df_market, benchmark, "Close")
    rs = close_sector / close_bench

    def rs_momentum(window):
        if len(rs) < window:
            return 0.0
        return (rs.iloc[-1] / rs.iloc[-window] - 1) if rs.iloc[-window] != 0 else 0.0

    rs63 = rs_momentum(63)
    rs126 = rs_momentum(126)
    rs252 = rs_momentum(252)
    rs_values = [rs63, rs126, rs252]
    rs_structural = np.mean([v for v in rs_values if pd.notna(v)]) if rs_values else 0.0
    rs_norm = np.tanh(rs_structural * 2) if pd.notna(rs_structural) else 0.0

    lb_norm = (leader_breadth - 0.5) * 2
    flow_norm = np.tanh(flow_structure)
    pers_norm = (persistence - 0.5) * 2

    score = 0.35 * rs_norm + 0.25 * lb_norm + 0.20 * flow_norm + 0.20 * pers_norm
    return float(np.clip(score, -1, 1))


def state_anterior(tactical, structural, breadth, persistence):
    t = {
        'structural_min_confirmed': 0.20,
        'structural_min_emerging': 0.20,
        'structural_max_decay': -0.20,
        'structural_max_lost': -0.40,
        'tactical_max_correction': -0.20,
        'breadth_max_decay': 0.35,
        'persistence_min_confirmed': 0.50,
        'persistence_max_emerging': 0.50,
    }
    if structural <= t['structural_max_lost'] and breadth <= t['breadth_max_decay']:
        return 'LOST'
    if structural <= t['structural_max_decay'] and breadth <= t['breadth_max_decay']:
        return 'STRUCTURAL_DECAY'
    if structural > t['structural_min_confirmed'] and tactical < t['tactical_max_correction'] and breadth > 0.35:
        return 'TACTICAL_CORRECTION'
    if structural > t['structural_min_confirmed'] and persistence >= t['persistence_min_confirmed'] and breadth > 0.35:
        return 'CONFIRMED'
    if structural > t['structural_min_emerging'] and persistence < t['persistence_max_emerging'] and breadth > 0.35:
        return 'EMERGING'
    return 'UNRESOLVED'


router = DataRouter()
all_tickers = flatten_tickers(MARKET_TICKERS)
data = router.get_market_data(all_tickers, period="5y")

# Tres períodos OOS no solapados
fechas = data.index.sort_values()
n = len(fechas)
periodos = {
    "P1": fechas[int(n*0.55):int(n*0.70)],
    "P2": fechas[int(n*0.70):int(n*0.85)],
    "P3": fechas[int(n*0.85):],
}

rows_resumen = []

for nombre_periodo, fechas_periodo in periodos.items():
    samples = fechas_periodo[::5]
    rows = []
    for fecha in samples:
        df_hasta = data.loc[:fecha]
        if len(df_hasta) < 200:
            continue
        for sector in MARKET_TICKERS['sectors']:
            try:
                close = get_col(df_hasta, sector, "Close")
                close_spy = get_col(df_hasta, "^GSPC", "Close")
                rs = close / close_spy
                rs20 = rs.pct_change(20, fill_method=None)

                persistence = compute_persistence(rs20, threshold=0.0, lookback=12)
                if persistence is None:
                    continue

                flow = compute_flow_proxy(df_hasta, sector).iloc[-1]
                tactical = compute_tactical_score(df_hasta, sector)

                structural_actual = compute_structural_score(
                    df_hasta, sector, leader_breadth=0.5,
                    flow_structure=flow, persistence=persistence,
                )
                structural_anterior = compute_structural_anterior(
                    df_hasta, sector, leader_breadth=0.5,
                    flow_structure=flow, persistence=persistence,
                )

                breadth = 0.5
                state_act = classify_leadership_state(structural_actual, breadth, persistence, coverage=1.0)["state"]
                state_ant = state_anterior(tactical, structural_anterior, breadth, persistence)

                rows.append({
                    "state_actual": state_act,
                    "state_anterior": state_ant,
                })
            except Exception:
                continue

    df_per = pd.DataFrame(rows)
    if df_per.empty:
        continue

    changed = (df_per["state_actual"] != df_per["state_anterior"]).mean()
    freq_actual = df_per["state_actual"].value_counts(normalize=True).to_dict()
    freq_anterior = df_per["state_anterior"].value_counts(normalize=True).to_dict()
    matriz = pd.crosstab(df_per["state_anterior"], df_per["state_actual"])

    rows_resumen.append({
        "periodo": nombre_periodo,
        "n_registros": len(df_per),
        "cambio_pct": changed,
        "freq_actual": freq_actual,
        "freq_anterior": freq_anterior,
    })

    print(f"\n===== {nombre_periodo} =====")
    print(f"Registros: {len(df_per)}")
    print(f"Cambio de estado: {changed:.1%}")
    print("Frecuencia actual:")
    print(freq_actual)
    print("Frecuencia anterior:")
    print(freq_anterior)
    print("Matriz anterior -> actual:")
    print(matriz.to_string())

# Guardar CSV resumen
resumen_df = pd.DataFrame(rows_resumen)
out_dir = Path("outputs/audit")
out_dir.mkdir(parents=True, exist_ok=True)
resumen_df.to_csv(out_dir / "oos_double_counting_v1.csv", index=False)
print("\nInforme guardado en outputs/audit/oos_double_counting_v1.csv")
