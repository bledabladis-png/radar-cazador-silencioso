# -*- coding: utf-8 -*-
"""Tests adjudicacion del panel.

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gold_standard.adjudication import (
    LABEL_NO_SOW,
    LABEL_SOW,
    adjudicate_panel,
    casos_con_tres_votos_validos,
    casos_unanimidad,
    porcentaje_technical_ineligible,
)


def _make_panel(rows):
    """rows: lista de (blind_id, a1, a2, a3, ti1, ti2, ti3)."""
    return pd.DataFrame([
        {
            "blind_id": r[0],
            "anotador_1": r[1], "anotador_2": r[2], "anotador_3": r[3],
            "anotador_1_ti": r[4], "anotador_2_ti": r[5],
            "anotador_3_ti": r[6],
        }
        for r in rows
    ])


def test_mayoria_3_0_sow():
    p = _make_panel([(1, "SOW", "SOW", "SOW", "no", "no", "no")])
    out = adjudicate_panel(p)
    assert out.df.loc[0, "label_adjudicada"] == LABEL_SOW
    assert out.df.loc[0, "composicion"] == "3-0 SOW"
    assert out.df.loc[0, "estado"] == "adjudicado"


def test_mayoria_2_1_sow():
    p = _make_panel([(2, "SOW", "SOW", "NO_SOW", "no", "no", "no")])
    out = adjudicate_panel(p)
    assert out.df.loc[0, "label_adjudicada"] == LABEL_SOW
    assert out.df.loc[0, "composicion"] == "2-1 SOW"


def test_mayoria_2_1_no_sow():
    p = _make_panel([(3, "NO_SOW", "NO_SOW", "SOW", "no", "no", "no")])
    out = adjudicate_panel(p)
    assert out.df.loc[0, "label_adjudicada"] == LABEL_NO_SOW
    assert out.df.loc[0, "composicion"] == "2-1 NO_SOW"


def test_1_ilegible_adjudica_con_2_votos():
    p = _make_panel([(4, "SOW", "SOW", "", "no", "no", "yes")])
    out = adjudicate_panel(p)
    assert out.df.loc[0, "label_adjudicada"] == LABEL_SOW
    assert out.df.loc[0, "n_votos_validos"] == 2
    assert out.df.loc[0, "n_technical_ineligible"] == 1
    assert out.df.loc[0, "estado"] == "adjudicado"

def test_2_ilegibles_excluye():
    p = _make_panel([(5, "SOW", "", "", "no", "yes", "yes")])
    out = adjudicate_panel(p)
    assert out.df.loc[0, "estado"] == "excluido_ti"
    assert pd.isna(out.df.loc[0, "label_adjudicada"])
    assert out.n_excluidos == 1


def test_3_ilegibles_excluye():
    p = _make_panel([(6, "", "", "", "yes", "yes", "yes")])
    out = adjudicate_panel(p)
    assert out.df.loc[0, "estado"] == "excluido_ti"


def test_sin_mayoria_imposible_en_binario():
    p = _make_panel([(7, "SOW", "NO_SOW", "SOW", "no", "no", "no")])
    out = adjudicate_panel(p)
    assert out.df.loc[0, "estado"] == "adjudicado"


def test_sin_votos_marca_estado():
    p = _make_panel([(8, "", "", "", "no", "no", "no")])
    out = adjudicate_panel(p)
    assert out.df.loc[0, "estado"] == "sin_votos"
    assert pd.isna(out.df.loc[0, "label_adjudicada"])


def test_composicion_resumen():
    p = _make_panel([
        (10, "SOW", "SOW", "SOW", "no", "no", "no"),
        (11, "SOW", "SOW", "NO_SOW", "no", "no", "no"),
        (12, "NO_SOW", "NO_SOW", "NO_SOW", "no", "no", "no"),
        (13, "", "", "", "yes", "yes", "no"),
    ])
    out = adjudicate_panel(p)
    assert out.n_total == 4
    assert out.n_adjudicados == 3
    assert out.n_excluidos == 1
    assert out.composicion["por_composicion"].get("3-0 SOW") == 1
    assert out.composicion["por_composicion"].get("2-1 SOW") == 1
    assert out.composicion["por_composicion"].get("3-0 NO_SOW") == 1

def test_casos_con_tres_votos_validos():
    p = _make_panel([
        (20, "SOW", "SOW", "SOW", "no", "no", "no"),
        (21, "SOW", "SOW", "", "no", "no", "yes"),
        (22, "NO_SOW", "NO_SOW", "NO_SOW", "no", "no", "no"),
    ])
    out = adjudicate_panel(p)
    sub = casos_con_tres_votos_validos(out.df)
    assert set(sub["blind_id"]) == {20, 22}


def test_casos_unanimidad():
    p = _make_panel([
        (30, "SOW", "SOW", "SOW", "no", "no", "no"),
        (31, "SOW", "SOW", "NO_SOW", "no", "no", "no"),
        (32, "NO_SOW", "NO_SOW", "NO_SOW", "no", "no", "no"),
    ])
    out = adjudicate_panel(p)
    u = casos_unanimidad(out.df)
    assert set(u["blind_id"]) == {30, 32}


def test_porcentaje_technical_ineligible():
    p = _make_panel([
        (40, "SOW", "SOW", "SOW", "no", "no", "no"),
        (41, "SOW", "SOW", "SOW", "no", "no", "no"),
        (42, "", "", "", "yes", "yes", "no"),
        (43, "", "", "", "yes", "yes", "yes"),
    ])
    out = adjudicate_panel(p)
    pct = porcentaje_technical_ineligible(out.df)
    assert abs(pct - 50.0) < 1e-9


def test_adjudicate_rechaza_columna_faltante():
    p = pd.DataFrame({"blind_id": [1], "anotador_1": ["SOW"]})
    with pytest.raises(ValueError, match="Falta columna"):
        adjudicate_panel(p)