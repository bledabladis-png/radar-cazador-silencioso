"""Verificacion manual de 5 casos segun dictamen v1.6 5c.3.

Casos: ACS.MC (claro->cae), ANA.MC (claro->cae), BA (claro->pasa),
PCG (claro->pasa), SLB (frontera->pasa).

Detalle pedido por auditor:
  MA50, MA200, precio vs MAs, max/min recientes, struct historico,
  struct_max, fecha exacta candidate, fecha exacta SOW,
  distancia candidate-SOW, soporte roto por SOW, volumen relativo SOW.
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, ".")
from indicators import wyckoff_v1 as w1

df = pd.read_parquet("data/stock_prices.parquet")

def find_candidate_dates(struct_s, t_norm_s):
    """Devuelve lista de fechas donde candidate = True."""
    struct_c = struct_s.dropna()
    t_c = t_norm_s.dropna()
    if struct_c.empty or t_c.empty:
        return []
    # Alinear
    common = struct_c.index.intersection(t_c.index)
    s = struct_c.loc[common]
    t = t_c.loc[common]
    dates = []
    for i in range(w1.PRECEDENT_WINDOW + 1, len(s)):
        s_val = float(s.iloc[i])
        t_val = float(t.iloc[i])
        # Precedente historico hasta i (excluyendo i)
        s_win = s.iloc[max(0, i - w1.PRECEDENT_WINDOW):i]
        if s_win.empty:
            continue
        s_max = float(s_win.max())
        is_cand = (
            s_val < w1.STRUCT_DETERIORO
            and t_val > -w1.T_NORM_STRONG
            and s_max > w1.PREC_STRUCT_STRONG
        )
        if is_cand:
            dates.append(common[i])
    return dates

def find_sow_dates(df_tk, tk):
    """Devuelve fechas donde SOW = True."""
    sow = w1.detect_sow(df_tk, tk)
    return [d for d, v in sow.items() if v == 1]

def describe(tk):
    print(f"\n{'=' * 100}")
    print(f"TICKER: {tk}")
    print('=' * 100)
    try:
        tdf = w1.build_ticker_df(df, tk)
        if tdf.empty:
            print("  sin datos")
            return
        close = tdf["Close"]
        low = tdf["Low"]
        high = tdf["High"]
        volume = tdf["Volume"]

        ma50 = close.rolling(50, min_periods=50).mean()
        ma200 = close.rolling(200, min_periods=200).mean()

        # 1-3: MA50, MA200, precio
        lc = float(close.iloc[-1]); l50 = float(ma50.dropna().iloc[-1]); l200 = float(ma200.dropna().iloc[-1])
        print(f"  [1-3] close={lc:.2f}  MA50={l50:.2f}  MA200={l200:.2f}")
        print(f"        precio vs MA50: {(lc/l50-1)*100:+.2f}%   vs MA200: {(lc/l200-1)*100:+.2f}%")

        # 4: maximos/minimos recientes
        last_60_high = float(high.iloc[-60:].max())
        last_60_low = float(low.iloc[-60:].min())
        print(f"  [4]   max(High,60)={last_60_high:.2f}  min(Low,60)={last_60_low:.2f}")

        # 5-6: struct historico y struct_max
        combined, struct_s, tact_s, t_norm, c_norm, v_norm, e_norm = w1.wyckoff_score(tdf, tk)
        struct_c = struct_s.dropna()
        t_c = t_norm.dropna()
        s_last = float(struct_c.iloc[-1])
        t_last = float(t_c.iloc[-1])
        prec = w1._compute_precedent(struct_c, t_c, window=w1.PRECEDENT_WINDOW)
        s_max = prec["struct_max"] if prec else float("nan")
        print(f"  [5]   struct_t={s_last:+.4f}   t_norm_t={t_last:+.4f}")
        print(f"        struct_max (hist)= {s_max:+.4f}   umbral={w1.PREC_STRUCT_STRONG}")
        print(f"        struct_min (hist)= {prec['struct_min']:+.4f}   struct_mean={prec['struct_mean']:+.4f}" if prec else "")

        # 7: fecha candidate mas reciente
        cand_dates = find_candidate_dates(struct_s, t_norm)
        if cand_dates:
            print(f"  [7]   fechas candidate: {len(cand_dates)} totales. Ultima: {cand_dates[-1]}")
        else:
            print(f"  [7]   sin fechas candidate")

        # 8: fecha SOW mas reciente
        sow_dates = find_sow_dates(tdf, tk)
        if sow_dates:
            print(f"  [8]   fechas SOW: {len(sow_dates)} totales. Ultima: {sow_dates[-1]}")
            print(f"        ultimos 5 SOW: {sow_dates[-5:]}")
        else:
            print(f"  [8]   sin fechas SOW")

        # 9: distancia candidate-SOW
        if cand_dates and sow_dates:
            last_cand = cand_dates[-1]
            last_sow = sow_dates[-1]
            print(f"  [9]   ultimo candidate: {last_cand}")
            print(f"        ultimo SOW:       {last_sow}")
            # Distancia en sesiones bursatiles
            idx = tdf.index
            if last_cand in idx and last_sow in idx:
                i_cand = idx.get_loc(last_cand)
                i_sow = idx.get_loc(last_sow)
                print(f"        distancia (sesiones): {i_sow - i_cand}")

        # 10: soporte roto por el SOW mas reciente
        if sow_dates:
            last_sow = sow_dates[-1]
            i = low.index.get_loc(last_sow)
            if i >= w1.WYCKOFF_SOW_WINDOW_N:
                sup = float(low.iloc[i-w1.WYCKOFF_SOW_WINDOW_N:i].min())
                cl = float(close.loc[last_sow])
                print(f"  [10]  SOW en {last_sow}: close={cl:.2f} < support({w1.WYCKOFF_SOW_WINDOW_N})={sup:.2f}")
                print(f"        caida vs soporte: {(cl/sup-1)*100:+.2f}%")

        # 11: volumen relativo del SOW
        if sow_dates:
            last_sow = sow_dates[-1]
            i = volume.index.get_loc(last_sow)
            if i >= w1.WYCKOFF_SOW_WINDOW_N:
                v_baseline = float(volume.iloc[i-w1.WYCKOFF_SOW_WINDOW_N:i].mean())
                v_actual = float(volume.loc[last_sow])
                print(f"  [11]  volumen SOW: {v_actual:,.0f} vs baseline({w1.WYCKOFF_SOW_WINDOW_N})={v_baseline:,.0f}")
                print(f"        ratio: {v_actual/v_baseline:.2f}x")

        # Fase final
        phase = w1.classify_wyckoff_phase(tdf, tk)
        meta = w1.classify_wyckoff_phase_meta(tdf, tk)
        print(f"  FASE v1.6: {phase}  (candidate={meta['distribution_candidate']})")
    except Exception as e:
        import traceback
        print(f"  ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()


for tk in ["ACS.MC", "ANA.MC", "BA", "PCG", "SLB"]:
    describe(tk)

print("\n" + "=" * 100)
print("FIN")
