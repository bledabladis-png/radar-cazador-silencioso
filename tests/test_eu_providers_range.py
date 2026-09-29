"""Test del rango por defecto de los providers EU (A, 2026-09-29).

Antes: BME/Xetra pedian 430 dias; Euronext 500 sesiones.
Ahora: BME/Xetra 1830 dias (5y). Euronext sigue limitado por el endpoint
(~510 sesiones, verificado 2026-09-29 con nb_session=1300).

Motivo: alinear horizonte con US/LSE (5y). Las ventanas largas
(rolling 200, percentiles multi-anio, correlaciones cross-mercado)
requieren historico equivalente entre mercados.
"""
from __future__ import annotations

from datetime import datetime

from data.providers.xetra_provider import _ws_start


def test_xetra_ws_start_cubre_5_anios():
    """_ws_start() debe devolver una fecha >= 5 anios atras."""
    s = _ws_start()
    d = datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    dias_atras = (datetime.now() - d).days
    assert dias_atras >= 1825, (
        f"_ws_start() solo cubre {dias_atras} dias (< 5 anios). "
        f"Valor: {s}"
    )