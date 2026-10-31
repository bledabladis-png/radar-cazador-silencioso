# -*- coding: utf-8 -*-
"""P1-H03 invariante 2: fase/eventos por ticker (post dictamen externo 2026-10-31).

Dictamen: "Demostrar que classify_phase funciona no demuestra que el
invariante 2 se cumpla en el backfill. Ejecutar la comprobacion del
invariante sobre el backfill y documentar su resultado."

Interpretacion operativa:
- Escenario A (pre-fix 4e94bf2e): lookup con ticker ORIGINAL de holdings.
  Si no existe en parquet -> ticker excluido del universo.
- Escenario B (post-fix): lookup con ticker normalizado via
  normalize_yahoo_ticker. Si existe -> ticker incluido.
- Tickers comunes (presentes en A y B): fase y event_phase deben ser
  identicas en toda fecha. Si difieren -> FAIL real.
- Tickers solo-A: regresion (antes aparecian, ahora no). Debe dar 0.
- Tickers solo-B: recuperados por el fix. Esperados.

Cobertura: 1086 fechas (mismas que el backfill) x universo A.2c
(seleccion target_weight sobre 11 sectores USA).

**Lo que este script VERIFICA:**
1. No hay tickers que desaparezcan (solo_A == 0) -> 0 regresion.
2. BRK.B (y otros duales) se recuperan via normalize_yahoo_ticker.
3. Los tickers comunes tienen fase/eventos identicos entre pasada
   unica (trivial por construccion: mismo detector, mismos datos).

**Lo que este script NO verifica:**
- Comparacion pre/post literal de fases para los 269 comunes. No
  existe snapshot pre-fix por ticker: el regenerate original produjo
  el agregado, no el snapshot. Reconstruir el pre requeriria
  revertir logicamente 4e94bf2e y otro run de ~60 min.
- El resultado es mas fuerte que el SKIP original (evidencia
  empirica de 0 regresiones), mas debil que el contrato literal
  del dictamen. Se documenta el limite.

**Deuda de diseno:** sin checkpoint. Si el proceso muere a mitad, se
pierde todo (~60 min de computo). El script original
regenerate_wyckoff_history.py si tenia checkpoint. Anadir si el
script se va a ejecutar mas de 2 veces. ROI bajo hoy.

Read-only. Escribe CSV resumen en outputs/audit/.
Uso: py scripts/audit/p1_h03_invariante2.py
"""
from __future__ import annotations

import concurrent.futures
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from indicators.wyckoff_orchestrator import classify_phase_at
from indicators.wyckoff_holdings_report import (
    MIN_BARS_THEORETICAL, _select_target_holdings,
)
from src.instrument_registry import normalize_yahoo_ticker
from config.tickers import MARKET_TICKERS

PARQUET = ROOT / "data" / "stock_prices.parquet"
HOLDINGS = ROOT / "data" / "etf_holdings.csv"
OUT_CSV = ROOT / "outputs" / "audit" / "p1_h03_invariante2.csv"

SECTORS = MARKET_TICKERS["sectors"]
N_WORKERS = 12

_DF = None
_FECHAS = None


def _init_worker(parquet_path, fechas_iso):
    global _DF, _FECHAS
    _DF = pd.read_parquet(parquet_path)
    _FECHAS = [pd.Timestamp(f) for f in fechas_iso]


def _phase_since_iso(val):
    if val is None:
        return ""
    try:
        return pd.Timestamp(val).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return ""


def _procesar_ticker(ticker_orig):
    """Devuelve filas por escenario.

    Si ticker_orig == ticker_norm: una sola pasada, marca como comun.
    Si difieren: pasada A (original) + pasada B (normalizada).
    """
    ticker_norm = normalize_yahoo_ticker(ticker_orig)
    escenarios = []
    if ticker_orig == ticker_norm:
        escenarios.append(("comun", ticker_orig))
    else:
        escenarios.append(("A_orig", ticker_orig))
        escenarios.append(("B_norm", ticker_norm))

    rows = []
    for escenario, tk in escenarios:
        for f in _FECHAS:
            try:
                fp = classify_phase_at(_DF, tk, as_of=f)
            except (KeyError, ValueError, TypeError, IndexError):
                continue
            if fp is None or fp.phase is None or fp.phase == "INSUFFICIENT_DATA":
                continue
            try:
                close_series = _DF.loc[:f, ("Close", tk)].dropna()
                n_bars = len(close_series)
            except (KeyError, TypeError):
                continue
            if n_bars < MIN_BARS_THEORETICAL:
                continue
            rows.append({
                "ticker_orig": ticker_orig,
                "ticker_used": tk,
                "escenario": escenario,
                "date": f.strftime("%Y-%m-%d"),
                "phase": fp.phase,
                "event_phase": fp.event_phase or "",
                "phase_since": _phase_since_iso(fp.phase_since),
            })
    return rows


def _get_universo():
    """Universo A.2c: seleccion target_weight sobre 11 sectores USA."""
    h = pd.read_csv(HOLDINGS)
    h = h.sort_values(["etf", "weight"], ascending=[True, False])
    partes = []
    for _etf, _g in h.groupby("etf"):
        _sel, _cob, _estado = _select_target_holdings(_g)
        partes.append(_sel)
    top = pd.concat(partes, ignore_index=True) if partes else h.iloc[0:0]
    top = top[top["etf"].isin(SECTORS)]
    return sorted(set(top["ticker"].tolist()))


def _get_fechas(df):
    close_cols = [("Close", s) for s in SECTORS if ("Close", s) in df.columns]
    if not close_cols:
        return list(df.index)
    cov = df[close_cols].notna().sum(axis=1) / len(close_cols)
    return [f for f, c in zip(df.index, cov) if c >= 0.90]


def main():
    t0 = time.time()
    tickers = _get_universo()
    df = pd.read_parquet(PARQUET)
    fechas = _get_fechas(df)
    fechas_iso = [f.strftime("%Y-%m-%d") for f in fechas]
    print(f"universo: {len(tickers)} tickers", flush=True)
    print(f"fechas: {len(fechas)} ({fechas_iso[0]} -> {fechas_iso[-1]})", flush=True)
    print(f"workers: {N_WORKERS}", flush=True)
    print()

    todas = []
    t1 = time.time()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=N_WORKERS,
        initializer=_init_worker,
        initargs=(str(PARQUET), fechas_iso),
    ) as ex:
        futures = {ex.submit(_procesar_ticker, t): t for t in tickers}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            ticker = futures[fut]
            try:
                rows = fut.result()
                todas.extend(rows)
            except Exception as e:
                print(f"[ERR] {ticker}: {e}")
            if i % 25 == 0:
                print(f"  {i}/{len(tickers)} tickers, {len(todas)} filas, "
                      f"{time.time()-t1:.0f}s", flush=True)
    t2 = time.time()
    print(f"\nprocesado en {t2-t1:.1f}s, {len(todas)} filas totales")

    df_out = pd.DataFrame(todas)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"escrito: {OUT_CSV}")

    # --- Analisis ---
    print()
    print("=" * 100)
    print("P1-H03 INVARIANTE 2: fase/eventos por ticker")
    print("=" * 100)

    tickers_comun = set(df_out[df_out["escenario"] == "comun"]["ticker_orig"])
    tickers_a = set(df_out[df_out["escenario"] == "A_orig"]["ticker_orig"])
    tickers_b = set(df_out[df_out["escenario"] == "B_norm"]["ticker_orig"])

    # Comunes: fase debe ser identica (trivial: misma pasada). Verificamos
    # contra el dual cuando el ticker aparece en A y B.
    duales = tickers_a & tickers_b
    solo_a = tickers_a - tickers_b
    solo_b = tickers_b - tickers_a

    print(f"tickers comunes (t_orig == t_norm): {len(tickers_comun)}")
    print(f"tickers duales (t_orig != t_norm): {len(duales)}")
    print(f"tickers solo en A (pre-fix): {len(solo_a)}")
    print(f"tickers solo en B (recuperados): {len(solo_b)}")

    # Comparacion dual: para tickers que existen en ambos escenarios,
    # la fase debe coincidir en cada fecha.
    violaciones = []
    for tk in duales:
        a_rows = df_out[(df_out["ticker_orig"] == tk) & (df_out["escenario"] == "A_orig")]
        b_rows = df_out[(df_out["ticker_orig"] == tk) & (df_out["escenario"] == "B_norm")]
        m = a_rows.merge(b_rows, on="date", suffixes=("_a", "_b"))
        for _, r in m.iterrows():
            if r["phase_a"] != r["phase_b"] or r["event_phase_a"] != r["event_phase_b"]:
                violaciones.append(
                    f"{tk} {r['date']}: A={r['phase_a']}/{r['event_phase_a']} "
                    f"!= B={r['phase_b']}/{r['event_phase_b']}"
                )

    print()
    print(f"violaciones (comunes con fase distinta): {len(violaciones)}")
    for v in violaciones[:5]:
        print(f"  {v}")

    if solo_b:
        print(f"\ntickers recuperados por el fix ({len(solo_b)}): {sorted(solo_b)}")
    if solo_a:
        print(f"\n[FAIL] tickers que desaparecen: {sorted(solo_a)}")

    # Resultado final
    status = "OK" if len(violaciones) == 0 and len(solo_a) == 0 else "FAIL"
    print()
    print(f"[{status}] invariante 2: comun={len(tickers_comun)}, "
          f"duales={len(duales)}, solo_A={len(solo_a)}, "
          f"solo_B={len(solo_b)}, violaciones={len(violaciones)}")
    print(f"tiempo total: {time.time()-t0:.1f}s")

    return 0 if status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
