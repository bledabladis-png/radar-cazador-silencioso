# -*- coding: utf-8 -*-
"""Centinela: ningun render puede producir "-0.00" (negativo cero).

Bug real: REPORT-NEGATIVE-ZERO (S30, 2026-11-06). Se detecto
"z=-0.00" en Alertas de Divergencia del reporte del 06-11.

Dos capas:
  1. Integracion: llamar a los productores reales con valor que
     historicamente producia -0.00 y verificar el mensaje.
  2. AST: prohibir f-string con format_spec "+.Nf" directo en
     codigo productivo. Forzar src.formatting.format_signed_float.
"""
from __future__ import annotations

import ast
import pathlib
import re

from indicators.price_flow_divergence import detect_price_flow_divergence
from src.report.alerts import render_alerts


NEG_ZERO = re.compile(r"-0\.0+([^0-9]|$)")


class TestIntegracion:

    def test_price_flow_divergence_sin_neg_zero(self):
        r = detect_price_flow_divergence(0.06, -0.001)
        assert r["status"] == "PRICE_STRONG_FLOW_UNCONFIRMED"
        assert not NEG_ZERO.search(r["message"]), r["message"]

    def test_alerts_hyg_sin_neg_zero(self):
        cd = {"ratios": {"hyg_lqd_zscore": -0.001}}
        out = render_alerts({}, "HIGH_STRESS", {}, confirmation_data=cd)
        joined = "".join(out)
        assert not NEG_ZERO.search(joined), joined


class TestAstSentinel:

    ROOTS = ("src", "indicators", "regimes", "config")
    PATTERN = re.compile(r"^\+\.\d+f$")

    def test_sin_format_spec_neg_zero_suelto(self):
        offenders = []
        for root in self.ROOTS:
            p = pathlib.Path(root)
            if not p.exists():
                continue
            for f in p.rglob("*.py"):
                try:
                    tree = ast.parse(f.read_bytes())
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if not isinstance(node, ast.JoinedStr):
                        continue
                    for v in node.values:
                        if not isinstance(v, ast.FormattedValue):
                            continue
                        spec = v.format_spec
                        if not isinstance(spec, ast.Constant):
                            continue
                        if not isinstance(spec.value, str):
                            continue
                        if self.PATTERN.match(spec.value):
                            rel = f.relative_to(pathlib.Path(".")).as_posix()
                            offenders.append(
                                "%s:%d spec=%r"
                                % (rel, node.lineno, spec.value)
                            )
        assert not offenders, (
            "f-string con format_spec +.Nf directo. "
            "Usar src.formatting.format_signed_float. "
            "Offenders: " + ", ".join(offenders)
        )
