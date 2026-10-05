# -*- coding: utf-8 -*-
"""Adjudicacion del panel de anotadores.

Especificacion (protocolo v7, secciones 2.2, 2.3):
  - Etiquetas binarias por anotador: SOW / NO_SOW.
  - technical_ineligible ∈ {yes, no}.
  - Mayoria 2/3 sobre votos validos.
  - Empate imposible en binario con 3 votos validos.
  - Casos con 2+ technical_ineligible: EXCLUIDOS.
  - Casos con 1 technical_ineligible: adjudica con 2 votos.

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd

LABEL_SOW = "SOW"
LABEL_NO_SOW = "NO_SOW"
LABEL_DIST = "DISTRIBUTIVE_CONTEXT"
LABEL_NO_DIST = "NO_DISTRIBUTIVE_CONTEXT"

COLS_ANOTADOR = ("anotador_1", "anotador_2", "anotador_3")


@dataclass
class Adjudicacion:
    df: pd.DataFrame
    composicion: dict
    n_total: int
    n_adjudicados: int
    n_excluidos: int

def _voto_valido(row: pd.Series, cols: tuple[str, ...]) -> list[str]:
    """Etiquetas validas: no vacias, no NaN, no ILEGIBLE."""
    out = []
    for c in cols:
        v = row.get(c)
        if v is None:
            continue
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s == "":
            continue
        out.append(s)
    return out


def _ineligible(row: pd.Series, cols_ti: tuple[str, ...]) -> int:
    """Cuenta cuantos anotadores marcan technical_ineligible."""
    n = 0
    for c in cols_ti:
        v = row.get(c)
        if v is None or pd.isna(v):
            continue
        s = str(v).strip().lower()
        if s in ("yes", "y", "true", "1"):
            n += 1
    return n


def _adjudicar_votos(votos: list[str]) -> str | None:
    """Mayoria simple. None si no hay mayoria."""
    if not votos:
        return None
    c = Counter(votos)
    top, n = c.most_common(1)[0]
    if n >= 2:
        return top
    return None

def adjudicate_panel(
    anotaciones: pd.DataFrame,
    label_cols: tuple[str, ...] = COLS_ANOTADOR,
    ti_cols: tuple[str, ...] | None = None,
) -> Adjudicacion:
    """Adjudica el panel completo."""
    if ti_cols is None:
        ti_cols = tuple(f"{c}_ti" for c in label_cols)

    for c in ("blind_id",) + label_cols:
        if c not in anotaciones.columns:
            raise ValueError(f"Falta columna {c}")
    for c in ti_cols:
        if c not in anotaciones.columns:
            raise ValueError(f"Falta columna {c}")

    rows = []
    for _, row in anotaciones.iterrows():
        bid = int(row["blind_id"])
        n_ti = _ineligible(row, ti_cols)
        if n_ti >= 2:
            rows.append({
                "blind_id": bid,
                "label_adjudicada": None,
                "n_votos_validos": 0,
                "n_technical_ineligible": n_ti,
                "estado": "excluido_ti",
                "composicion": None,
            })
            continue

        votos = _voto_valido(row, label_cols)
        if not votos:
            rows.append({
                "blind_id": bid,
                "label_adjudicada": None,
                "n_votos_validos": 0,
                "n_technical_ineligible": n_ti,
                "estado": "sin_votos",
                "composicion": None,
            })
            continue

        label = _adjudicar_votos(votos)
        if label is None:
            rows.append({
                "blind_id": bid,
                "label_adjudicada": None,
                "n_votos_validos": len(votos),
                "n_technical_ineligible": n_ti,
                "estado": "sin_mayoria",
                "composicion": None,
            })
            continue

        rows.append({
            "blind_id": bid,
            "label_adjudicada": label,
            "n_votos_validos": len(votos),
            "n_technical_ineligible": n_ti,
            "estado": "adjudicado",
            "composicion": _composicion(votos, label),
        })

    df = pd.DataFrame(rows)
    composicion = _resumen_composicion(df)
    n_total = len(df)
    n_excluidos = int((df["estado"] == "excluido_ti").sum())
    n_adjudicados = int((df["estado"] == "adjudicado").sum())
    return Adjudicacion(
        df=df,
        composicion=composicion,
        n_total=n_total,
        n_adjudicados=n_adjudicados,
        n_excluidos=n_excluidos,
    )

def _composicion(votos: list[str], label: str) -> str:
    """Etiqueta tipo '3-0 SOW' o '2-1 NO_SOW'."""
    n_label = sum(1 for v in votos if v == label)
    n_other = len(votos) - n_label
    return f"{n_label}-{n_other} {label}"


def _resumen_composicion(df: pd.DataFrame) -> dict:
    """Cuenta casos por composicion, votos validos y excluidos."""
    out: dict = {
        "n_total": int(len(df)),
        "n_adjudicados": int((df["estado"] == "adjudicado").sum()),
        "n_excluido_ti": int((df["estado"] == "excluido_ti").sum()),
        "n_sin_votos": int((df["estado"] == "sin_votos").sum()),
        "n_sin_mayoria": int((df["estado"] == "sin_mayoria").sum()),
        "por_composicion": {},
        "por_n_ti": {},
    }
    comp = df[df["composicion"].notna()]["composicion"]
    for c, n in comp.value_counts().items():
        out["por_composicion"][str(c)] = int(n)
    ti_counts = df["n_technical_ineligible"].value_counts().sort_index()
    for k, v in ti_counts.items():
        out["por_n_ti"][int(k)] = int(v)
    return out


def casos_con_tres_votos_validos(df_adjudicado: pd.DataFrame) -> pd.DataFrame:
    """Subconjunto con 3 votos validos. Base para Fleiss kappa principal."""
    return df_adjudicado[df_adjudicado["n_votos_validos"] == 3].copy()


def casos_unanimidad(df_adjudicado: pd.DataFrame) -> pd.DataFrame:
    """Solo casos 3-0 (analisis de alta certeza)."""
    m = df_adjudicado["composicion"].fillna("").str.startswith("3-0")
    return df_adjudicado[m].copy()


def porcentaje_technical_ineligible(df_adjudicado: pd.DataFrame) -> float:
    """Porcentaje de casos con 2+ technical_ineligible."""
    if len(df_adjudicado) == 0:
        return 0.0
    n = int((df_adjudicado["estado"] == "excluido_ti").sum())
    return 100.0 * n / len(df_adjudicado)