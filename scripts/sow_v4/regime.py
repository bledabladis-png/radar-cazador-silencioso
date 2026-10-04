"""Regimen de mercado (seccion 3 del protocolo v4.1).

Calcula:
- retornos sectoriales equal-weight
- volatilidad realizada por sector (20 sesiones, ddof=1)
- mediana cross-sectorial de volatilidad
- log_vol_lag1(t) = log(vol_mediana(t-1))
- umbrales q33/q67 sobre un TRAIN
- clasificacion R(t) en {stress, normal, inter, undefined}
- asignacion R_episode = R(t0)

Regla bloqueante: los umbrales se calculan exclusivamente con datos
hasta el final del TRAIN correspondiente.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.sow_v4 import config


@dataclass(frozen=True)
class RegimeThresholds:
    q33: float
    q67: float


def compute_sector_returns(
    df: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """Retornos sectoriales equal-weight.

    df: parquet multiindex (field, ticker). Se asume level 0 = field, level 1 = ticker.
    sector_map: {ticker: sector_name}.

    Devuelve DataFrame indexado por fecha, columnas = sectores.
    NaN si el sector tiene < MIN_TICKERS_RET tickers validos en t.
    """
    if "Close" not in df.columns.get_level_values(0):
        raise ValueError("df no tiene campo 'Close' en level 0")

    close = df.xs("Close", axis=1, level=0)
    ret_all = close.pct_change(fill_method=None)

    # Filtrar a US trading days: solo sesiones donde >=50% de los
    # tickers del universo del regimen tienen Close valido.
    # Motivo: el parquet es union US+EU; los festivos US aparecen
    # con US cerrado y EU abierto. Sin este filtro, rolling(20).std()
    # se propaga 20 sesiones por cada festivo US.
    regime_tickers = [t for t in sector_map.keys() if t in close.columns]
    if not regime_tickers:
        raise ValueError("sector_map no intersecciona con columnas del df")
    coverage = close[regime_tickers].notna().sum(axis=1)
    us_mask = coverage >= (0.5 * len(regime_tickers))
    ret = ret_all.loc[us_mask]

    sectores = sorted(set(sector_map.values()))
    out = pd.DataFrame(index=ret.index, columns=sectores, dtype=float)

    min_tk = config.REGIME_MIN_TICKERS_RET
    for sector in sectores:
        tickers_sector = [tk for tk, s in sector_map.items() if s == sector and tk in ret.columns]
        if not tickers_sector:
            continue
        sub = ret[tickers_sector]
        valid_count = sub.notna().sum(axis=1)
        mean_ret = sub.mean(axis=1, skipna=True)
        mean_ret = mean_ret.where(valid_count >= min_tk, np.nan)
        out[sector] = mean_ret

    # Dropear filas all-NaN. Motivo: pct_change(fill_method=None) genera
    # NaN en el dia posterior a un festivo US (porque el Close del festivo
    # es NaN). Esos dias no aportan informacion sectorial y no deben
    # entrar en el rolling(20) de volatilidad, o propagan NaN 20 sesiones.
    return out.dropna(how="all")


def compute_vol_median(sector_returns: pd.DataFrame) -> pd.Series:
    """Mediana cross-sectorial de volatilidad realizada a 20 sesiones.

    ddof=1. Si < REGIME_MIN_SECTORES_VOL sectores tienen vol valida en t,
    devuelve NaN.
    """
    w = config.REGIME_VOL_WINDOW
    ddof = config.REGIME_VOL_DDOF
    min_sec = config.REGIME_MIN_SECTORES_VOL

    vols = sector_returns.rolling(window=w, min_periods=w).std(ddof=ddof)
    n_valid = vols.notna().sum(axis=1)
    median_vol = vols.median(axis=1, skipna=True)
    median_vol = median_vol.where(n_valid >= min_sec, np.nan)
    return median_vol


def compute_log_vol_lag1(vol_median: pd.Series) -> pd.Series:
    """log(vol_mediana(t-1)). Requiere vol > 0 y finito."""
    lag = vol_median.shift(1)
    lag = lag.where((lag > 0) & np.isfinite(lag), np.nan)
    return np.log(lag)


def fit_thresholds(log_vol: pd.Series, train_start, train_end) -> RegimeThresholds:
    """q33 y q67 sobre el rango [train_start, train_end], usando solo
    sesiones con log_vol valido."""
    mask = (log_vol.index >= pd.Timestamp(train_start)) & (
        log_vol.index <= pd.Timestamp(train_end)
    )
    sub = log_vol[mask].dropna()
    if sub.empty:
        raise ValueError(f"Sin observaciones validas en TRAIN {train_start}..{train_end}")
    return RegimeThresholds(
        q33=float(sub.quantile(0.33)),
        q67=float(sub.quantile(0.67)),
    )


def classify_regime(log_vol: pd.Series, thresholds: RegimeThresholds) -> pd.Series:
    """Devuelve Series con valores 'stress', 'normal', 'inter', o None.

    None cuando log_vol es NaN.
    """
    labels = pd.Series(index=log_vol.index, dtype=object)
    valid = log_vol.notna()
    labels[valid & (log_vol > thresholds.q67)] = "stress"
    labels[valid & (log_vol < thresholds.q33)] = "normal"
    labels[valid & (log_vol >= thresholds.q33) & (log_vol <= thresholds.q67)] = "inter"
    return labels


def regime_at_t0(regime_labels: pd.Series, t0_date) -> str | None:
    """R_episode = R(t0). Devuelve None si la fecha no esta o es NaN."""
    if t0_date not in regime_labels.index:
        return None
    val = regime_labels.loc[t0_date]
    if pd.isna(val):
        return None
    return str(val)

def load_sector_map(
    csv_path=None,
    experiment_tickers=None,
) -> dict[str, str]:
    """Carga el mapping ticker->sector desde etf_holdings.csv.

    Seccion 3 protocolo v4.1 (C-1 modificada):
    - Solo devuelve tickers presentes en el universo del experimento.
    - El resultado se usa para construir el REGIMEN; no filtra el
      experimento SOW.
    - Verificado: 0 duplicidades en el CSV actual.

    Parametros
    ----------
    csv_path : Path | None
        Ruta al CSV. Si None, usa config.SECTOR_MAP_CSV.
    experiment_tickers : iterable | None
        Universo del experimento. Si se pasa, solo se devuelven
        tickers de ese conjunto. Si None, devuelve todos.

    Devuelve
    --------
    dict {ticker: sector_etf}. Ej: {"AAPL": "XLK", "JPM": "XLF"}.
    """
    path = Path(csv_path) if csv_path is not None else config.SECTOR_MAP_CSV
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")

    df = pd.read_csv(path)
    if "ticker" not in df.columns or "etf" not in df.columns:
        raise ValueError(f"etf_holdings.csv sin columnas ticker/etf: {list(df.columns)}")

    # Verificar unicidad (bloqueante)
    dup = df.groupby("ticker")["etf"].nunique()
    conflicts = dup[dup > 1]
    if len(conflicts) > 0:
        raise ValueError(
            f"ABORT: {len(conflicts)} tickers con mas de un sector: "
            f"{conflicts.head().to_dict()}"
        )

    # Sectores validos
    sectores_validos = set(config.REGIME_REFERENCE_SECTORS)
    df = df[df["etf"].isin(sectores_validos)]

    mapping = dict(zip(df["ticker"], df["etf"]))

    if experiment_tickers is not None:
        exp_set = set(experiment_tickers)
        mapping = {tk: s for tk, s in mapping.items() if tk in exp_set}

    return mapping

