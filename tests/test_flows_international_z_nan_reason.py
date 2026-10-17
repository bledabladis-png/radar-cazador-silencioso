# -*- coding: utf-8 -*-
"""Regresion H4 (2026-10-17): la columna Z-score de LYXI distingue el
motivo del N/D.

Antes: `N/D` sin explicar. El contrato de _fund_flow_utils (aprobado
2026-09-27) documenta MAD0_NAN y MAD0_SAME como estados legitimamente
no definidos: 78% de los dias Amundi publica shares_change=0, el MAD
de la ventana queda en 0 y el z no tiene escala. No es un hueco de
captura.

Ahora: el render lee `flow_zscore_regime` y anota el motivo:
- N/D (MAD0): MAD=0, sin escala.
- N/D (hist. insuf.): ventana < 20 obs.
- 0.00 (MAD0): MAD=0 pero valor == mediana.
- N/D sin sufijo: dato no presente.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.report.flows_international import render_flujos_internacionales_agregados


def _lyxi(z, regime):
    return pd.DataFrame([{
        "date": pd.Timestamp("2026-10-15"),
        "nav": 203.0569,
        "shares_change": -13760.0,
        "estimated_flow_eur": -2794062.944,
        "flow_pct_assets": -0.0066103188229789285,
        "flow_zscore": z,
        "flow_zscore_regime": regime,
    }])


def test_z_nan_con_regimen_mad0_anota_motivo():
    """flow_zscore NaN + regimen MAD0_NAN -> 'N/D (MAD0)'."""
    out = render_flujos_internacionales_agregados(
        blackrock_dax_flow=None,
        blackrock_isf_flow=None,
        amundi_lyxi_flow=_lyxi(float("nan"), "MAD0_NAN"),
        blackrock_iwm_flow=None,
    )
    text = "".join(out)
    assert "N/D (MAD0)" in text
    # La nota explicativa debe estar presente
    assert "MAD=0" in text or "MAD0" in text


def test_z_nan_con_regimen_insuficiente_anota_motivo():
    """flow_zscore NaN + regimen INSUFFICIENT -> 'N/D (hist. insuf.)'."""
    out = render_flujos_internacionales_agregados(
        blackrock_dax_flow=None,
        blackrock_isf_flow=None,
        amundi_lyxi_flow=_lyxi(float("nan"), "INSUFFICIENT"),
        blackrock_iwm_flow=None,
    )
    text = "".join(out)
    assert "N/D (hist. insuf.)" in text


def test_z_valido_no_lleva_sufijo():
    """flow_zscore valido -> se pinta el valor, sin sufijo en la fila."""
    out = render_flujos_internacionales_agregados(
        blackrock_dax_flow=None,
        blackrock_isf_flow=None,
        amundi_lyxi_flow=_lyxi(-1.23, "NORMAL"),
        blackrock_iwm_flow=None,
    )
    # Buscar solo la fila de la tabla (linea que empieza por '| LYXI '),
    # no las notas explicativas (que mencionan "N/D (MAD0)" por diseno).
    lyxi_row = next(
        (l for l in out if l.startswith("| LYXI ")), None
    )
    assert lyxi_row is not None, "fila LYXI no encontrada"
    assert "-1.23" in lyxi_row
    assert "N/D" not in lyxi_row