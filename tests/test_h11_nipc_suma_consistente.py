# -*- coding: utf-8 -*-
"""Centinela H11 (auditoria 2026-11-07): NIPC Total debe ser
aritmeticamente consistente con la suma de breakdowns presentados.

Bug real: el auditor observo
  Total: 7.647.182.843
  SOLE + DFND + OTR: 7.647.182.842 (+1 de diferencia)
Causa raiz: int(v) truncaba hacia cero en _fmt_int. Con floats
fraccionarios, int(a) + int(b) + int(c) != int(a+b+c). El compute_nipc
raw es correcto (raw_total == sum(raw_parts)); el bug era de
presentacion.

Fix: (1) _fmt_int usa int(round(v)); (2) el bloque NIPC deriva
Total = SOLE + DFND + OTR presentados (redondeados) para garantizar
consistencia por construccion; (3) si nipc_total raw difiere del
derivado, se anade nota al pie. Si nipc_total es NaN/None, se
preserva N/D (no se enmascara ausencia de dato con un derivado).
"""
from __future__ import annotations

from src.report.iae import render_iae_section


def _iae_full(**kwargs):
    d = {
        "status": "OK",
        "period_previous": "2026Q1",
        "period_current": "2026Q2",
        "nipc_total": 1000.0,
        "nipc_sole": 500.0,
        "nipc_dfnd": 300.0,
        "nipc_otr": 200.0,
        "catalog_coverage_declared": 0.95,
        "catalog_keys_observed": 100, "catalog_keys_total": 100,
        "coverage_status": "OK", "coverage_quality": "COMPLETE",
        "n_delta_observable": 50, "n_both": 30, "n_new": 10, "n_exit": 10,
        "n_unresolved_identity": 0,
    }
    d.update(kwargs)
    return "".join(render_iae_section(d))


def _parse_int(s):
    return int(s.replace(".", "").replace(",", ""))


class TestH11ConsistenciaAritmetica:

    def test_reproduce_bug_original_fracciones_truncadas(self):
        """Caso del auditor: raw_total == sum(raw_parts) pero
        int(a)+int(b)+int(c) != int(total). El Total presentado debe
        igualar la suma de los breakdowns presentados."""
        # Valores que reproducen el +1 con int() truncado.
        out = _iae_full(
            nipc_total=7647182843.4,
            nipc_sole=5973541781.5,
            nipc_dfnd=1698761244.4,
            nipc_otr=-25120183.5,
        )
        # Extraer valores presentados de la tabla.
        lineas = [l for l in out.split("\n") if l.startswith("| Total") or
                  l.startswith("| SOLE") or l.startswith("| DFND") or
                  l.startswith("| OTR")]
        assert len(lineas) == 4, f"esperaba 4 lineas NIPC. Out: {out}"
        vals = {}
        for ln in lineas:
            k = ln.split("|")[1].strip()
            v = ln.split("|")[2].strip()
            vals[k] = _parse_int(v)
        assert vals["Total"] == vals["SOLE"] + vals["DFND"] + vals["OTR"], (
            f"Total != suma breakdowns: {vals}")

    def test_enteros_puros_sin_regresion(self):
        """Caso normal: raw_total == sum(breakdowns) enteros."""
        out = _iae_full(
            nipc_total=1000.0,
            nipc_sole=500.0, nipc_dfnd=300.0, nipc_otr=200.0,
        )
        assert "| Total | 1.000 |" in out
        assert "| SOLE  | 500 |" in out
        assert "| DFND  | 300 |" in out
        assert "| OTR   | 200 |" in out

    def test_nipc_total_nan_preserva_nd(self):
        """Si nipc_total falta, N/D. No se enmascara con derivado."""
        out = _iae_full(nipc_total=float("nan"))
        assert "| Total | N/D |" in out

    def test_nipc_total_none_preserva_nd(self):
        out = _iae_full(nipc_total=None)
        assert "| Total | N/D |" in out

    def test_nota_redondeo_aparece_si_difiere(self):
        """Si Total derivado difiere del raw redondeado, nota visible."""
        # Configuracion que produce discrepancia int(round(total)) !=
        # round(sole)+round(dfnd)+round(otr):
        # raw total .4 -> int(round(..)) = +0.4 -> trunca a X
        # breakdowns .5 -> int(round(..)) sube a X+1
        out = _iae_full(
            nipc_total=7647182843.4,
            nipc_sole=5973541781.5,
            nipc_dfnd=1698761244.4,
            nipc_otr=-25120183.5,
        )
        # El raw redondeado: int(round(7647182843.4)) = 7647182843
        # El derivado: 5973541782 + 1698761244 + -25120184 = 7647182842
        # Difieren en 1. Nota debe aparecer.
        assert "redondeo consistente" in out

    def test_nota_redondeo_ausente_si_no_difiere(self):
        """Caso normal: no se anade ruido innecesario."""
        out = _iae_full(
            nipc_total=1000.0,
            nipc_sole=500.0, nipc_dfnd=300.0, nipc_otr=200.0,
        )
        assert "redondeo consistente" not in out
