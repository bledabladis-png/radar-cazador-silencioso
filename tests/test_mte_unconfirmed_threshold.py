# -*- coding: utf-8 -*-
"""Centinela del umbral UNCONFIRMED en MTE (S31, 2026-11-06).

Q15 del cuestionario S30: el umbral confidence < 0.50 estaba
publicado en el render y documentado en las aclaraciones, pero
sin test que blindara el valor numerico.

Bug real latente: si alguien cambia
  src/report/volatility_mte.py:94
    _confirmed = pd.notna(mte_conf) and mte_conf >= 0.5
por > 0.5 o por >= 0.4, ningun test existente falla. El umbral
se desincroniza en silencio con:
  - src/report/helpers.py:194 'La etiqueta se publica como
    UNCONFIRMED cuando confidence < 0.50.'
  - scripts/generate_docs.py:356 'se marca como (UNCONFIRMED)
    si la confianza es inferior al 50%.'

Umbral contractual: confidence >= 0.50 => CONFIRMED
                   confidence <  0.50 => UNCONFIRMED
                   confidence NaN/None => UNCONFIRMED
"""
from __future__ import annotations

import re

import pytest

from src.report.volatility_mte import render_mte


def _mte(confidence):
    """mte_result minimo valido para render_mte. confidence puede ser
    NaN, None o float. Resto de campos con ceros."""
    return {
        "confidence": confidence,
        "scenario": "MIXED",
        "msi": 0, "ipi": 0, "srs": 0.0, "shs": 0.0,
        "cls": 0.0, "ips": 0.0,
    }


def _joined(lines):
    return "".join(lines)


class TestUmbralUnconfirmed:

    @pytest.mark.parametrize("confidence", [0.0, 0.25, 0.45, 0.49])
    def test_bajo_umbral_es_unconfirmed(self, confidence):
        out = _joined(render_mte(_mte(confidence)))
        assert "Escenario (UNCONFIRMED)" in out, (
            f"confidence={confidence} deberia ser UNCONFIRMED (< 0.50). "
            f"Salida: {out!r}"
        )

    @pytest.mark.parametrize("confidence", [0.50, 0.51, 0.75, 1.0])
    def test_en_o_sobre_umbral_es_confirmed(self, confidence):
        out = _joined(render_mte(_mte(confidence)))
        assert "Escenario (UNCONFIRMED)" not in out, (
            f"confidence={confidence} deberia ser CONFIRMED (>= 0.50). "
            f"Salida: {out!r}"
        )

    def test_frontera_0499_vs_0500(self):
        """Blindaje fino: 0.499 no redondea a 0.50 antes de comparar."""
        low = _joined(render_mte(_mte(0.499)))
        high = _joined(render_mte(_mte(0.500)))
        assert "Escenario (UNCONFIRMED)" in low
        assert "Escenario (UNCONFIRMED)" not in high

    @pytest.mark.parametrize("confidence", [float("nan"), None])
    def test_no_numerico_es_unconfirmed(self, confidence):
        out = _joined(render_mte(_mte(confidence)))
        assert "Escenario (UNCONFIRMED)" in out, (
            f"confidence={confidence!r} deberia ser UNCONFIRMED "
            f"(NaN/None no confirmado). Salida: {out!r}"
        )

    def test_texto_y_logica_usan_el_mismo_umbral(self):
        """El texto publicado declara el umbral; la logica lo respeta.

        Extraemos el umbral del texto de aclaracion (p.ej. 'confidence
        < 0.50') y verificamos la logica en la frontera declarada por
        el texto. Si alguien cambia uno de los dos sin el otro, este
        test falla. Sin hardcodear 0.50 aqui: el numero lo dicta el
        texto contractual, no el test.
        """
        out = _joined(render_mte(_mte(0.99)))
        m = re.search(r"confidence\s*<\s*(0\.\d+)", out)
        assert m, (
            "El texto de aclaracion no declara el umbral "
            "'confidence < X.XX'. Salida: " + out
        )
        umbral_texto = float(m.group(1))
        justo_bajo = round(umbral_texto - 0.01, 4)
        justo_en = umbral_texto
        assert "Escenario (UNCONFIRMED)" in _joined(render_mte(_mte(justo_bajo))), (
            f"Texto declara umbral {umbral_texto} pero logica no marca "
            f"UNCONFIRMED en {justo_bajo}."
        )
        assert "Escenario (UNCONFIRMED)" not in _joined(render_mte(_mte(justo_en))), (
            f"Texto declara umbral {umbral_texto} pero logica no marca "
            f"CONFIRMED en {justo_en}."
        )
