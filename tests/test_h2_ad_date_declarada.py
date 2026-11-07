# -*- coding: utf-8 -*-
"""Centinela H2 (auditoria 2026-11-07): fecha explicita en A/D Net, NH/NL
y A/D Line.

Bug real: el render publicaba
  Advance/Decline Net (universo completo): +282 (405 avances / 123 descensos)
  New Highs/Lows (mercado): 78 maximos / 1 minimos (NH-NL: +77)
  A/D Line (acumulada): 24181
sin fecha, mientras el bloque venia del 4/11 (walk-back) y el reporte
declaraba target_session 5/11. El usuario podia leer +282 como dato
del desplome del 5/11.

Causa raiz: compute_advance_decline (indicators/breadth_equity.py:130-139)
ya anadia `effective_date`, `lag_days`, `coverage`, `n_observed`,
`n_eligible` al dict `ad`. El render (src/report/confirmation.py:66-94)
recibia el dict pero no leia esos campos.

Fix: declarar fecha (y lag si > 0) en las 3 lineas, formato:
  ... +282 (405 avances / 123 descensos) *(al 2026-11-04, lag 1d)*
  ... 78 maximos / 1 minimos (NH-NL: +77) *(al 2026-11-04, lag 1d)*
  ... 24181 *(al 2026-11-04, lag 1d)*

Sin sufijo si el dict no trae `effective_date` (compat con tests
existentes y con productores antiguos). Sin hardcode de fechas.
"""
from __future__ import annotations

import pytest

from src.report.confirmation import render_confirmation


def _render(ad_dict):
    return "".join(render_confirmation({"ad": ad_dict}))


BASE_AD = {
    "ad_net": 74, "advances": 193, "declines": 119,
    "new_highs": 5, "new_lows": 2, "nh_nl": 3,
    "breadth_thrust": 0.5, "ad_line": 14843,
}


class TestSinFechaNoAnadeSufijo:

    def test_dict_antiguo_sin_effective_date(self):
        """Compat: tests y productores que no rellenan effective_date."""
        out = _render(BASE_AD)
        assert "*(al " not in out
        assert "lag" not in out

    def test_effective_date_none(self):
        ad = {**BASE_AD, "effective_date": None, "lag_days": 1}
        out = _render(ad)
        assert "*(al " not in out

    def test_effective_date_nan(self):
        ad = {**BASE_AD, "effective_date": float("nan"), "lag_days": 1}
        out = _render(ad)
        assert "*(al " not in out


class TestConFecha:

    def test_fecha_sin_lag_aparece_en_las_tres_lineas(self):
        ad = {**BASE_AD, "effective_date": "2026-11-05", "lag_days": 0}
        out = _render(ad)
        assert out.count("*(al 2026-11-05)*") == 3, (
            "esperaba sufijo en A/D Net + NH/NL + A/D Line. Out: " + out)
        assert "lag" not in out

    def test_fecha_con_lag_aparece_en_las_tres_lineas(self):
        ad = {**BASE_AD, "effective_date": "2026-11-04", "lag_days": 1}
        out = _render(ad)
        assert out.count("*(al 2026-11-04, lag 1d)*") == 3, (
            "esperaba sufijo con lag en A/D Net + NH/NL + A/D Line. Out: " + out)

    @pytest.mark.parametrize("lag", [1, 2, 5, 30])
    def test_lag_variable(self, lag):
        ad = {**BASE_AD, "effective_date": "2026-11-04", "lag_days": lag}
        out = _render(ad)
        assert f", lag {lag}d)*" in out

    def test_lag_nan_sin_sufijo_de_lag(self):
        ad = {**BASE_AD, "effective_date": "2026-11-04",
              "lag_days": float("nan")}
        out = _render(ad)
        assert "*(al 2026-11-04)*" in out
        assert "lag" not in out

    def test_lag_negativo_no_se_declara(self):
        """Lag negativo (reloj raro): no declararlo, seria enganoso."""
        ad = {**BASE_AD, "effective_date": "2026-11-05", "lag_days": -1}
        out = _render(ad)
        assert "lag" not in out
        assert "*(al 2026-11-05)*" in out


class TestNoRompeLegacy:

    def test_contrato_advance_decline_net(self):
        """El string contractual sigue presente (test_flows_confirmation)."""
        out = _render(BASE_AD)
        assert "Advance/Decline Net (universo completo)" in out

    def test_contrato_ad_line(self):
        out = _render(BASE_AD)
        assert "A/D Line (acumulada):" in out
        assert "Recalculada sobre el universo actual" in out

    def test_contrato_universo_sigue_visible(self):
        ad = {**BASE_AD, "active_tickers": 529, "total_tickers": 530}
        out = _render(ad)
        assert "Universo: 529/530 tickers" in out
