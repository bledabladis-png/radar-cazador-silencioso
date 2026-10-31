"""REP-AUDIT-20261018-IPS: aclarar que IPS/IPI son proxy de mercado.

El analista externo pide explicitar que IPS no equivale a la tasa de
inflacion observada (IPC/PPI). Es proxy de mercado derivado de XLE,
^SPGSCI, TIP y ratio TIP/IEF. El renderer debe declararlo.
"""
from src.report.volatility_mte import render_mte


def test_render_mte_declara_ips_como_proxy_mercado():
    out = render_mte({
        "scenario": "MIXED", "confidence": 0.66,
        "msi": 48, "ipi": 31, "srs": 0.33, "shs": 0.15,
        "cls": 0.20, "ips": -0.39,
    })
    lineas = "".join(out)
    assert "proxy de mercado" in lineas
    assert "no una medida de la tasa de inflacion observada" in lineas
    assert "IPC/PPI" in lineas
    assert "XLE" in lineas
    assert "^SPGSCI" in lineas
    assert "TIP/IEF" in lineas
    assert "ewma" in lineas.lower() or "EWMA" in lineas
    assert "(IPS+1)/2*100" in lineas


def test_render_mte_no_rompe_bloque_de_pesos():
    """Regresion: el texto nuevo no debe alterar la nota de pesos existente."""
    out = render_mte({
        "scenario": "MIXED", "confidence": 0.66,
        "msi": 48, "ipi": 31, "srs": 0.33, "shs": 0.15,
        "cls": 0.20, "ips": -0.39,
    })
    lineas = "".join(out)
    assert "IPS=3" in lineas
    assert "IPS>0.15" in lineas
    assert "IPS>0.50 suma +2" in lineas
    # H7 B6 (2026-10-31): la nota ahora deriva de SCENARIO_RULES.
    # El contrato es que STAGFLATION documenta IPS>0.15 AND SRS>0.
    import re
    m = re.search(r"STAGFLATION: ([^;]*)", lineas)
    assert m, "STAGFLATION no aparece en la nota"
    bloque_stag = m.group(1)
    assert "IPS>0.15" in bloque_stag
    assert "SRS>0" in bloque_stag