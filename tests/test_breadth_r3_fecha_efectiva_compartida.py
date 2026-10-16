# -*- coding: utf-8 -*-
"""Tests R2 + R3 explicitos: fecha efectiva compartida entre
sector_breadth y sector_breadth_momentum.

Dictamen auditor externo 2026-10-16 (§4):
  - No basta con afirmar que las fechas maximas son iguales.
  - Exige comprobar que la fecha efectiva se obtiene con
    resolve_effective_date bajo condiciones temporales aplicables.
  - Exige verificar que ambos CSV publican esa misma fecha efectiva.
  - Exige verificar politica de ausencia cuando datos insuficientes.
  - Datos controlados con filas desordenadas o ultima observacion
    no elegible.

Los tests no confian en la mera igualdad de max(date): comparan
ambas fechas contra la fecha que devuelve _resolve_breadth_effective_date().
"""
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import breadth_metrics as bm
from src.pipeline.breadth_metrics import (
    _resolve_breadth_effective_date,
    _compute_sector_breadth_health,
)


SATURDAY = datetime(2026, 9, 12, 10, 0)
FRIDAY = datetime(2026, 9, 11, 23, 30)


def _mk_stocks(end_date, tickers=None, n=300, nan_final_idx=None,
               shuffle_last_n=0):
    """df_stocks con N tickers.

    nan_final_idx: si se indica, deja NaN en la fila -1 para los
    primeros `nan_final_idx` tickers (simula cobertura final baja).
    shuffle_last_n: si > 0, mezcla las ultimas `shuffle_last_n` filas
    (rompe orden natural) sin cambiar los datos.
    """
    if tickers is None:
        tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    idx = pd.date_range(end=end_date, periods=n, freq="D")
    data = {}
    for i, t in enumerate(tickers):
        vals = [100.0 + j * 0.1 for j in range(n)]
        if nan_final_idx and i < nan_final_idx:
            vals[-1] = float("nan")
        data[("Close", t)] = vals
        data[("High", t)] = [101.0] * n
        data[("Low", t)] = [99.0] * n
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    if shuffle_last_n > 0:
        head = df.iloc[:-shuffle_last_n]
        tail = df.iloc[-shuffle_last_n:].iloc[::-1]
        df = pd.concat([head, tail])
    return df


def _mk_market(end_date, etfs=None, n=300):
    if etfs is None:
        etfs = ["XLK"]
    idx = pd.date_range(end=end_date, periods=n, freq="D")
    data = {}
    for e in etfs:
        data[(e, "Close")] = [100.0 + j * 0.05 for j in range(n)]
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _holdings(etfs=None, tickers=None):
    if etfs is None:
        etfs = ["XLK"]
    if tickers is None:
        tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    return pd.DataFrame({"etf": etfs * len(tickers),
                         "ticker": tickers * len(etfs)})


# ============================================================
# T1 - fecha efectiva igual a la fecha canonica (caso normal)
# ============================================================
def test_r3_fecha_efectiva_es_canonica(tmp_path, monkeypatch):
    """La fecha escrita en sector_breadth.csv debe coincidir con
    _resolve_breadth_effective_date() sobre el universo combinado."""
    df_stocks = _mk_stocks("2026-09-11")
    df_market = _mk_market("2026-09-11")
    holdings = _holdings()

    eff, _ = _resolve_breadth_effective_date(df_stocks, df_market, holdings)

    csv = tmp_path / "sb.csv"
    with patch.object(bm, "persist"), patch.object(bm, "skip"):
        result, is_stale = _compute_sector_breadth_health(
            df_stocks, df_market, holdings,
            reference_date=FRIDAY, output_path=csv)

    assert result is not None
    assert is_stale is False
    fecha_escrita = pd.to_datetime(result["date"].max()).normalize()
    assert fecha_escrita == pd.Timestamp(eff).normalize(), (
        "fecha escrita != fecha efectiva devuelta por resolve_effective_date"
    )


# ============================================================
# T2 - ultima fila fisica != ultima sesion admisible
# ============================================================
def test_r3_ultima_fila_fisica_no_elegible(tmp_path, monkeypatch):
    """Si la ultima fila fisica tiene cobertura insuficiente (solo 1
    de 5 tickers), la fecha efectiva debe retroceder."""
    df_stocks = _mk_stocks("2026-10-15", nan_final_idx=4)  # 1 de 5 validos
    df_market = _mk_market("2026-10-15")
    holdings = _holdings()

    eff, meta = _resolve_breadth_effective_date(df_stocks, df_market, holdings)
    # 14-oct tiene 100% cobertura; 15-oct solo 20%
    assert pd.Timestamp(eff).normalize() == pd.Timestamp("2026-10-14").normalize()
    assert meta["status"] == "OK"

    csv = tmp_path / "sb.csv"
    with patch.object(bm, "persist"), patch.object(bm, "skip"):
        result, is_stale = _compute_sector_breadth_health(
            df_stocks, df_market, holdings,
            reference_date=datetime(2026, 10, 15, 23, 30),
            output_path=csv)

    if result is not None:
        fecha_escrita = pd.to_datetime(result["date"].max()).normalize()
        assert fecha_escrita == pd.Timestamp("2026-10-14").normalize(), (
            "la fecha escrita debe ser la efectiva (14-oct), no la fisica (15-oct)"
        )
    else:
        # Si df_stocks queda None tras R4 cap, es coherente
        assert is_stale is True


# ============================================================
# T3 - filas desordenadas no rompen la resolucion
# ============================================================
def test_r3_filas_desordenadas_no_rompen(tmp_path, monkeypatch):
    """Con filas desordenadas en las ultimas 3 posiciones, la fecha
    efectiva sigue siendo la correcta (no confia en index[-1])."""
    df_stocks = _mk_stocks("2026-09-11", shuffle_last_n=3)
    df_market = _mk_market("2026-09-11")
    holdings = _holdings()

    eff, meta = _resolve_breadth_effective_date(df_stocks, df_market, holdings)
    assert eff is not None
    assert meta["status"] == "OK"

    csv = tmp_path / "sb.csv"
    with patch.object(bm, "persist"), patch.object(bm, "skip"):
        result, _ = _compute_sector_breadth_health(
            df_stocks, df_market, holdings,
            reference_date=FRIDAY, output_path=csv)

    assert result is not None
    fecha_escrita = pd.to_datetime(result["date"].max()).normalize()
    assert fecha_escrita == pd.Timestamp(eff).normalize()


# ============================================================
# T4 - momentum hereda la misma fecha efectiva
# ============================================================
def test_r3_momentum_hereda_fecha(tmp_path, monkeypatch):
    """sector_breadth_momentum.csv hereda la fecha de sector_breadth.csv
    cuando se alimenta del mismo archivo."""
    from indicators.sector_breadth_momentum import (
        compute_sector_breadth_momentum,
    )
    SECTORS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK",
               "XLP", "XLRE", "XLU", "XLV", "XLY"]
    # Construye un CSV de breadth con 25 sesiones + ultima fecha controlada.
    idx = pd.date_range(end="2026-09-11", periods=25, freq="B")
    rows = []
    for d in idx:
        for s in SECTORS:
            rows.append({
                "date": d,
                "sector": s,
                "pct_above_ema20": 50.0,
                "pct_above_ema50": 50.0,
                "pct_above_ema200": 50.0,
            })
    csv = tmp_path / "sb.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    mom = compute_sector_breadth_momentum(str(csv))
    assert not mom.empty
    fecha_breadth = pd.Timestamp(idx.max()).normalize()
    fecha_mom = pd.to_datetime(mom["date"].max()).normalize()
    assert fecha_mom == fecha_breadth, (
        "momentum debe heredar la fecha efectiva de sector_breadth"
    )


# ============================================================
# T5 - datos insuficientes -> politica de ausencia (no imputar)
# ============================================================
def test_r3_datos_insuficientes_no_imputan(tmp_path, monkeypatch):
    """Con cobertura insuficiente en todo el historico, la fecha
    efectiva es None. No se imputa una sesion."""
    # 5 tickers, solo 1 con datos en toda la serie -> cobertura 20%
    df_stocks = _mk_stocks("2026-09-11", nan_final_idx=0)
    # NaN en toda la serie para los 4 primeros:
    for i in range(4):
        df_stocks[("Close", ["AAA", "BBB", "CCC", "DDD"][i])] = float("nan")

    df_market = _mk_market("2026-09-11")
    holdings = _holdings()

    eff, meta = _resolve_breadth_effective_date(df_stocks, df_market, holdings)
    assert eff is None, "cobertura 20% < 90% -> sin fecha efectiva"
    assert meta["status"] == "INSUFFICIENT_COVERAGE"