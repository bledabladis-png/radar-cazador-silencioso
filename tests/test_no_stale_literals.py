# -*- coding: utf-8 -*-
"""REP-AUDIT-20261017-capa1: ningun literal de ventana/umbral en el
render del reporte puede contradecir el valor vigente en config/settings.
"""
from pathlib import Path
import re

from config import settings


def test_cftc_nota_sin_numero_hardcoded():
    from indicators.data_quality import SOURCES
    cftc = next(s for s in SOURCES if s["source"] == "CFTC Position Flow")
    nota = cftc["notes"]
    for max_val in settings.FRESHNESS_CFTC:
        assert str(max_val) not in nota, (
            "Nota CFTC contiene el numero {} de FRESHNESS_CFTC. "
            "Si cambia el umbral, la nota queda obsoleta. Nota: {!r}".format(
                max_val, nota,
            )
        )


def test_leaders_rs_mom_sin_literal_numerico():
    from src.report import leaders
    src = Path(leaders.__file__).read_text(encoding="utf-8")
    matches = re.findall(
        r"RS Mom = variacion logaritmica de RS a \d+ d[ií]as", src,
    )
    assert not matches, (
        "Leaders contiene literal numerico en 'RS Mom = ... N dias': {}. "
        "Usar f-string con MOMENTUM_PRICE_WINDOW.".format(matches)
    )


def test_synthesis_flujo_z_sin_literal_numerico():
    from src.report import synthesis
    src = Path(synthesis.__file__).read_text(encoding="utf-8")
    matches = re.findall(r"z-score robusto sobre \d+ d[ií]as", src)
    assert not matches, (
        "Synthesis contiene literal numerico en 'z-score robusto sobre "
        "N dias': {}. Usar f-string con FLOW_ZSCORE_WINDOW.".format(matches)
    )
