# -*- coding: utf-8 -*-
"""Fase 3: regeneracion historica de sector_wyckoff_distribution v2.

Paralelizado por ticker. Checkpoint cada 25 tickers. Validacion obligatoria
contra el .csv activo (11 filas del 23-10). Si no coincide exactamente,
aborta sin reemplazar nada.

Uso:
    py scripts/regenerate_wyckoff_history.py --test     # 3 tickers x 10 fechas
    py scripts/regenerate_wyckoff_history.py --full     # todos los tickers
    py scripts/regenerate_wyckoff_history.py --validate # solo validacion
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators.wyckoff_orchestrator import classify_phase_at
from indicators.wyckoff_holdings_report import TOP_N_HOLDINGS, MIN_BARS_THEORETICAL
from src.instrument_registry import normalize_yahoo_ticker
from config.tickers import MARKET_TICKERS

PARQUET = ROOT / "data" / "stock_prices.parquet"
HOLDINGS = ROOT / "data" / "etf_holdings.csv"
OUT_FULL = ROOT / "outputs" / "history" / "sector_wyckoff_distribution.v2_full.csv"
OUT_ACTIVE = ROOT / "outputs" / "history" / "sector_wyckoff_distribution.csv"
CHECKPOINT = ROOT / "outputs" / "history" / "_regen_checkpoint.json"

SECTORS = MARKET_TICKERS["sectors"]
VALID_PHASES = [
    "ACCUMULATION", "ACCUMULATION_FORMING", "MARKUP", "RANGE",
    "DISTRIBUTION", "DISTRIBUTION_FORMING", "MARKDOWN", "CONFLICT",
]

# --- Globals para los workers ---
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


def _procesar_ticker(ticker):
    """Devuelve lista de dicts: una fila por (ticker, fecha).

    WYCKOFF-TICKER-NORMALIZATION (2026-10-26): ticker es el nombre
    original de holdings (ej. BRK.B). Se normaliza para parquet
    (BRK-B) pero se preserva el original en el output para que
    mapping (ticker->sector) resuelva.
    """
    ticker_norm = normalize_yahoo_ticker(ticker)
    rows = []
    for f in _FECHAS:
        try:
            fp = classify_phase_at(_DF, ticker_norm, as_of=f)
        except (KeyError, ValueError, TypeError, IndexError):
            continue
        if fp is None or fp.phase is None or fp.phase == "INSUFFICIENT_DATA":
            continue
        try:
            close_series = _DF.loc[:f, ("Close", ticker_norm)].dropna()
            n_bars = len(close_series)
            close_val = float(close_series.iloc[-1]) if n_bars > 0 else float("nan")
        except (KeyError, TypeError):
            n_bars = 0
            close_val = float("nan")
        if n_bars < MIN_BARS_THEORETICAL:
            continue
        evaluable = n_bars >= 460
        rows.append({
            "date": f.strftime("%Y-%m-%d"),
            "ticker": ticker,
            "phase": fp.phase,
            "event_phase": fp.event_phase or "",
            "phase_since": _phase_since_iso(fp.phase_since),
            "close": close_val,
            "basis": fp.current_phase_basis or "",
            "n_bars": int(n_bars),
            "evaluable": bool(evaluable),
        })
    return rows


def _get_tickers(selection_mode="nlargest"):
    from indicators.wyckoff_holdings_report import _select_target_holdings
    h = pd.read_csv(HOLDINGS)
    h = h.sort_values(["etf", "weight"], ascending=[True, False])
    if selection_mode == "nlargest":
        top = h.groupby("etf", group_keys=False).head(TOP_N_HOLDINGS)
    elif selection_mode == "target_weight":
        partes = []
        for _etf, _g in h.groupby("etf"):
            _sel, _cob, _estado = _select_target_holdings(_g)
            partes.append(_sel)
        top = pd.concat(partes, ignore_index=True) if partes else h.iloc[0:0]
    else:
        raise ValueError(
            "selection_mode desconocido: {!r}. Valores: 'nlargest', 'target_weight'.".format(
                selection_mode))
    top = top[top["etf"].isin(SECTORS)]
    # Mapear ticker -> sector
    mapping = {}
    for _, r in top.iterrows():
        mapping[r["ticker"]] = r["etf"]
    df = pd.read_parquet(PARQUET)
    validos = []
    for t in mapping.keys():
        # WYCKOFF-TICKER-NORMALIZATION (2026-10-26): holdings usa BRK.B,
        # parquet usa BRK-B. Normalizar solo para el lookup, conservar
        # el nombre original en validos (para mapping y output).
        t_norm = normalize_yahoo_ticker(t)
        try:
            n = int(df[("Close", t_norm)].dropna().shape[0])
            if n >= MIN_BARS_THEORETICAL:
                validos.append(t)
        except (KeyError, TypeError):
            pass
    return sorted(set(validos)), mapping


def _get_fechas(df):
    """Fechas con cobertura Close >= 90% sobre el universo de 11 sectores."""
    close_cols = [("Close", s) for s in SECTORS if ("Close", s) in df.columns]
    if not close_cols:
        return list(df.index)
    cov = df[close_cols].notna().sum(axis=1) / len(close_cols)
    return [f for f, c in zip(df.index, cov) if c >= 0.90]


def _construir_agregado(rows_por_ticker, mapping):
    """De lista de dicts (fila por ticker×fecha) a DataFrame de agregado por sector×fecha."""
    if not rows_por_ticker:
        return pd.DataFrame()
    df = pd.DataFrame(rows_por_ticker)
    df["sector"] = df["ticker"].map(mapping)
    df = df.dropna(subset=["sector"])
    # Solo contar tickers con evaluable (para n_valid coherente con CSVs)
    # y phase != INSUFFICIENT_DATA
    df = df[df["phase"].notna() & (df["phase"] != "INSUFFICIENT_DATA")]

    out_rows = []
    for (fecha, sector), grp in df.groupby(["date", "sector"]):
        n_total = len(grp)
        n_valid = n_total
        n_insuf = 0
        counts = {ph: 0 for ph in VALID_PHASES}
        for _, r in grp.iterrows():
            ph = r["phase"]
            if ph in VALID_PHASES:
                counts[ph] += 1
            else:
                n_insuf += 1
                n_valid -= 1
        coverage = (n_valid / n_total * 100) if n_total else float("nan")
        row = {
            "date": fecha,
            "sector": sector,
            "n_total": n_total,
            "n_valid_wyckoff": n_valid,
            "n_insufficient_wyckoff": n_insuf,
            "coverage_wyckoff": coverage,
        }
        for ph in VALID_PHASES:
            row[f"count_{ph.lower()}"] = counts[ph]
            row[f"pct_{ph.lower()}"] = (counts[ph] / n_valid * 100) if n_valid >= 5 else float("nan")
        out_rows.append(row)
    return pd.DataFrame(out_rows).sort_values(["date", "sector"]).reset_index(drop=True)


def _validate_against_active(regen_df, mode="strict"):
    """Valida el CSV regenerado contra el activo.

    mode="strict": exige coincidencia exacta en la fecha mas reciente
    (comportamiento historico, valido cuando la base no cambia).
    mode="informative": reporta diffs pero NO aborta. Se usa cuando se
    cambia la base (dictamen TOP20) y las diferencias son el objetivo.
    Aun en informative, se valida estructura (11 sectores, sin NaN en
    columnas obligatorias, n_total = n_valid + n_insufficient).
    """
    if mode == "informative":
        return _validate_informative(regen_df)
    return _validate_strict(regen_df)


def _validate_informative(regen_df):
    """Validacion estructural cuando la base cambia (no compara contra activo)."""
    if regen_df is None or regen_df.empty:
        return False, "regen vacio"
    last_date = regen_df["date"].max()
    last = regen_df[regen_df["date"] == last_date]
    n_sectores = last["sector"].nunique()
    if n_sectores != 11:
        return False, "sectores en {}: {} (esperado 11)".format(last_date, n_sectores)
    # n_total = n_valid + n_insufficient
    bad = last[last["n_total"] != last["n_valid_wyckoff"] + last["n_insufficient_wyckoff"]]
    if len(bad) > 0:
        return False, "n_total != n_valid + n_insuf en {} filas".format(len(bad))
    # count_* suma = n_valid
    count_cols = [c for c in last.columns if c.startswith("count_")]
    if count_cols:
        sum_counts = last[count_cols].sum(axis=1)
        bad = last[sum_counts != last["n_valid_wyckoff"]]
        if len(bad) > 0:
            return False, "suma count_* != n_valid en {} filas".format(len(bad))
    return True, "informative OK: {} filas, {} sectores en {}".format(
        len(last), n_sectores, last_date)


def _validate_strict(regen_df):
    """Validacion historica: coincidencia exacta con el CSV activo."""
    if not OUT_ACTIVE.exists():
        return False, "activo no existe"
    active = pd.read_csv(OUT_ACTIVE)
    last_active_date = active["date"].max()
    active_last = active[active["date"] == last_active_date].copy()
    regen_last = regen_df[regen_df["date"] == last_active_date].copy()
    if len(active_last) != len(regen_last):
        return False, f"filas distintas: activo={len(active_last)}, regen={len(regen_last)}"
    # Merge por sector
    m = active_last.merge(regen_last, on="sector", suffixes=("_a", "_r"))
    if len(m) != len(active_last):
        return False, f"merge sector fallo: {len(m)} vs {len(active_last)}"
    diffs = []
    cols = [c for c in active_last.columns if c != "sector" and c in regen_last.columns and c != "date"]
    for c in cols:
        a = m[f"{c}_a"]
        b = m[f"{c}_r"]
        if a.dtype.kind in ("f", "i") and b.dtype.kind in ("f", "i"):
            mask = ~((a.isna() & b.isna()) | (abs(a.fillna(0) - b.fillna(0)) < 1e-6))
        else:
            mask = ~((a == b) | (a.isna() & b.isna()))
        if mask.any():
            diffs.append((c, m.loc[mask, ["sector", f"{c}_a", f"{c}_r"]].to_dict("records")[:3]))
    if diffs:
        msg_lines = [f"columnas con diff ({len(diffs)}):"]
        for c, sample in diffs[:5]:
            msg_lines.append(f"  {c}: {sample}")
        return False, "\n".join(msg_lines)
    return True, f"OK: {len(active_last)} filas, {len(cols)} columnas identicas en fecha {last_active_date}"


def _load_checkpoint():
    if CHECKPOINT.exists():
        try:
            return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _save_checkpoint(state):
    CHECKPOINT.write_text(json.dumps(state), encoding="utf-8")


print("=== SCRIPT INICIADO (unbuffered) ===", flush=True)


def _dbg(msg):
    print(msg, flush=True)


def run(full=False, test=False, selection_mode="nlargest", validate_mode="strict"):
    _dbg("[1] cargando parquet...")
    df = pd.read_parquet(PARQUET)
    _dbg("[2] obteniendo tickers (selection_mode={})...".format(selection_mode))
    tickers, mapping = _get_tickers(selection_mode=selection_mode)
    _dbg("[3] obteniendo fechas...")
    fechas = _get_fechas(df)
    _dbg(f"[4] tickers={len(tickers)} fechas={len(fechas)}")

    if test:
        fechas = fechas[-1:]  # solo la ultima fecha
        print(f"[TEST] {len(tickers)} tickers x 1 fecha ({fechas[0].date()})")
    print(f"Tickers: {len(tickers)} | Fechas: {len(fechas)}")
    print(f"Total invocaciones classify_phase_at: {len(tickers)*len(fechas):,}")
    print()

    fechas_iso = [f.strftime("%Y-%m-%d") for f in fechas]
    t0 = time.time()

    all_rows = []
    done_tickers = set()
    cp = _load_checkpoint()
    if cp and not test:
        all_rows = cp.get("rows", [])
        done_tickers = set(cp.get("done_tickers", []))
        print(f"[CHECKPOINT] reanudando desde {len(done_tickers)} tickers ya hechos")

    pendientes = [t for t in tickers if t not in done_tickers]
    n_workers = min(12, len(pendientes)) if pendientes else 1
    _dbg(f"[5] Workers: {n_workers}, pendientes: {len(pendientes)}")
    _dbg("[6] lanzando ProcessPoolExecutor...")
    print(f"Workers: {n_workers}, pendientes: {len(pendientes)}", flush=True)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(str(PARQUET), fechas_iso),
    ) as ex:
        futures = {ex.submit(_procesar_ticker, t): t for t in pendientes}
        n_done = 0
        for fut in concurrent.futures.as_completed(futures):
            ticker = futures[fut]
            try:
                rows = fut.result()
                all_rows.extend(rows)
                done_tickers.add(ticker)
                n_done += 1
                if n_done % 25 == 0 or n_done == len(pendientes):
                    elapsed = time.time() - t0
                    print(f"  [{n_done}/{len(pendientes)}] {elapsed:.0f}s "
                          f"({len(all_rows):,} filas acumuladas)")
                    if not test:
                        _save_checkpoint({"rows": all_rows, "done_tickers": sorted(done_tickers)})
            except Exception as e:
                print(f"  [ERROR] {ticker}: {e}")

    elapsed = time.time() - t0
    print(f"\nTiempo total: {elapsed/60:.1f} min ({len(all_rows):,} filas ticker×fecha)")

    print("\nConstruyendo agregado por sector...")
    regen = _construir_agregado(all_rows, mapping)
    print(f"Agregado: {len(regen)} filas ({regen['date'].nunique() if not regen.empty else 0} fechas, {regen['sector'].nunique() if not regen.empty else 0} sectores)")

    print("\nValidando contra el .csv activo...")
    ok, msg = _validate_against_active(regen, mode=validate_mode)
    print(f"  Validacion [{validate_mode}]: {'[OK]' if ok else '[FALLO]'} {msg}")

    if test:
        print("\n[TEST] output:")
        print(regen.head(15).to_string(index=False))
        print("\n[TEST] no se escribe nada")
        return

    if not ok:
        print("\n[ABORT] validacion fallo. No se escribe .v2_full.csv.")
        return

    print(f"\nValidacion OK. Escribiendo {OUT_FULL.name}...")
    regen.to_csv(OUT_FULL, index=False, encoding="utf-8", lineterminator="\n")
    print(f"[OK] {OUT_FULL}: {len(regen)} filas")
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
        print("[OK] checkpoint eliminado")
    print("\nSiguiente paso: revisar diff y hacer el swap manual con .v1.csv.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--selection-mode", choices=["nlargest", "target_weight"],
                    default="nlargest", help="Criterio de seleccion de holdings")
    ap.add_argument("--validate", choices=["strict", "informative"],
                    default="strict", help="Modo de validacion contra el CSV activo")
    args = ap.parse_args()
    run(full=args.full, test=args.test,
        selection_mode=args.selection_mode, validate_mode=args.validate)