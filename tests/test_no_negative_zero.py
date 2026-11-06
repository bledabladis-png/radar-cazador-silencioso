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
    ROOT_FILES_GLOB = "*.py"
    PATTERN = re.compile(r"^\+\.\d+f$")

    def _iter_py_files(self):
        for root in self.ROOTS:
            p = pathlib.Path(root)
            if not p.exists():
                continue
            for f in p.rglob("*.py"):
                yield f
        for f in pathlib.Path(".").glob(self.ROOT_FILES_GLOB):
            if f.is_file():
                yield f

    def _get_spec_string(self, spec):
        """Extrae el string literal del format_spec.

        Python 3.11: ast.Constant(str) directo.
        Python 3.12+ (PEP 701): ast.JoinedStr envolviendo Constant(str),
        incluso cuando el spec no tiene interpolacion.
        Con interpolacion en el spec (spec dinamico), retorna None:
        ese caso no produce -0.00 por el mismo mecanismo.
        """
        if isinstance(spec, ast.Constant) and isinstance(spec.value, str):
            return spec.value
        if isinstance(spec, ast.JoinedStr):
            vals = spec.values
            if len(vals) == 1 and isinstance(vals[0], ast.Constant):
                v = vals[0].value
                if isinstance(v, str):
                    return v
        return None

    def _find_offenders(self):
        offenders = []
        for f in self._iter_py_files():
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
                    spec_str = self._get_spec_string(v.format_spec)
                    if spec_str is None:
                        continue
                    if self.PATTERN.match(spec_str):
                        rel = f.relative_to(pathlib.Path(".")).as_posix()
                        offenders.append(
                            "%s:%d spec=%r" % (rel, node.lineno, spec_str)
                        )
        return offenders

    def test_sin_format_spec_neg_zero_suelto(self):
        offenders = self._find_offenders()
        assert not offenders, (
            "f-string con format_spec +.Nf directo. "
            "Usar src.formatting.format_signed_float. "
            "Offenders: " + ", ".join(offenders)
        )


class TestGetSpecString:

    def test_joinedstr_pep701_spec_literal(self):
        tree = ast.parse('y = f"z={x:+.2f}"')
        fv = [n for n in ast.walk(tree) if isinstance(n, ast.FormattedValue)][0]
        assert TestAstSentinel()._get_spec_string(fv.format_spec) == "+.2f"

    def test_spec_con_interpolacion_no_matchea(self):
        tree = ast.parse('y = f"z={x:{prec}f}"')
        fv = [n for n in ast.walk(tree) if isinstance(n, ast.FormattedValue)][0]
        assert TestAstSentinel()._get_spec_string(fv.format_spec) is None

class TestAstSentinelCoversRoot:
    """Verifica que el AST guard cubre .py de raiz (run.py, conftest.py, futuros)."""

    def test_detecta_offender_en_raiz(self, tmp_path, monkeypatch):
        (tmp_path / "sospechoso_raiz.py").write_text(
            "x = 1\ny = f\"z={x:+.2f}\"\n", encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(TestAstSentinel, "ROOTS", ())
        offenders = TestAstSentinel()._find_offenders()
        assert offenders, "El guard debe detectar un +.Nf directo en raiz"
        assert "sospechoso_raiz.py" in offenders[0]