# -*- coding: utf-8 -*-
"""Render de PNGs ciegos para el panel de anotadores.

Especificacion (protocolo v7, seccion 3.4):
  - Ventana: 240 sesiones hasta t (o L para Capa 2).
  - PNG 1600x900 px.
  - Panel superior: velas OHLC.
  - Panel inferior: volumen en barras.
  - Sin fecha, ticker, medias, niveles ni anotaciones.
  - Titulo: "Caso NNNN".
  - Ejes X sin etiquetas (ciego).

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.gold_standard.constants import (
    FIELDS_REQUIRED,
    VISUAL_WINDOW,
)


WIDTH_PX = 1600
HEIGHT_PX = 900
DPI = 100
FIGSIZE = (WIDTH_PX / DPI, HEIGHT_PX / DPI)


def _slice_window(df_ticker: pd.DataFrame, t: pd.Timestamp,
                  window: int = VISUAL_WINDOW) -> pd.DataFrame:
    """Corta las ultimas `window` sesiones hasta t inclusive."""
    sliced = df_ticker.loc[:t]
    if len(sliced) < window:
        raise ValueError(
            f"Ventana incompleta: {len(sliced)} < {window} hasta {t}"
        )
    return sliced.tail(window).copy()


def render_case(
    df_ticker: pd.DataFrame,
    t: pd.Timestamp,
    blind_id: int,
    out_path: Path,
    window: int = VISUAL_WINDOW,
) -> None:
    """Renderiza un caso ciego. Escribe PNG en out_path."""
    sl = _slice_window(df_ticker, t, window)
    o = sl["Open"].to_numpy()
    h = sl["High"].to_numpy()
    l = sl["Low"].to_numpy()
    c = sl["Close"].to_numpy()
    v = sl["Volume"].to_numpy()
    n = len(sl)
    x = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=FIGSIZE,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Velas: cuerpo y mecha
    body_lo = np.minimum(o, c)
    body_hi = np.maximum(o, c)
    body_h = np.maximum(body_hi - body_lo, 1e-9)
    colors = np.where(c >= o, "#222222", "#c0392b")

    # Mechas
    for i in range(n):
        ax1.plot(
            [i, i], [l[i], h[i]],
            color="#555555", linewidth=0.6, zorder=1,
        )

    # Cuerpos (barh para mejor rendimiento)
    ax1.bar(
        x, body_h, bottom=body_lo, width=0.6,
        color=colors, edgecolor="none", zorder=2,
    )

    ax2.bar(x, v, width=0.8, color="#7f8c8d", edgecolor="none")

    # Estetica ciega
    for ax in (ax1, ax2):
        ax.set_xlim(-1, n)
        ax.tick_params(
            axis="x", which="both",
            bottom=False, top=False, labelbottom=False,
        )
        ax.grid(True, axis="y", color="#eeeeee", linewidth=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    ax1.set_ylabel("Precio", fontsize=10)
    ax2.set_ylabel("Volumen", fontsize=10)
    ax1.set_title(f"Caso {blind_id}", fontsize=13, loc="left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def render_all(
    sample: pd.DataFrame,
    dataset: pd.DataFrame,
    out_dir: Path,
    window: int = VISUAL_WINDOW,
) -> dict:
    """Renderiza todos los casos de una muestra.

    sample: DataFrame con columnas 'ticker', 't', 'blind_id'.
    dataset: stock_prices.parquet con MultiIndex (field, ticker).
    Devuelve dict con contadores.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    fail = 0
    errors = []
    for _, row in sample.iterrows():
        tk = row["ticker"]
        t = row["t"]
        bid = int(row["blind_id"])
        out_path = out_dir / f"case_{bid}.png"
        try:
            df_tk = dataset.xs(tk, axis=1, level=1)
            # Garantizar columnas requeridas
            df_tk = df_tk[list(FIELDS_REQUIRED)]
            render_case(df_tk, t, bid, out_path, window=window)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            fail += 1
            errors.append({"blind_id": bid, "error": str(exc)})
    return {"ok": ok, "fail": fail, "errors": errors}