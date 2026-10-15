"""Test H6/H10: el render de Flujo Primario SPDR expone la saturacion
del z-score con el valor bruto sin clip.

Auditoria externa 2026-10-15. H6: XLP mostro +622.85M (4.23% AUM, Z=5.00)
y el reporte no declaraba que el Z era saturado. H10: la inconsistencia
entre Z=5.00 y la acumulacion 20d=357M requeria explicitar el clip.

Contrato:
  - Z dentro del clip: se muestra "+1.50" sin asterisco.
  - Z saturado: "+5.00* (+8.31)" con asterisco y bruto entre parentesis.
  - Nota al pie explica el asterisco.
"""
import pandas as pd

from src.report.etf_flows import render_flujo_spdr


def _make_df(rows):
    return pd.DataFrame(rows)


def _base_row(ticker, z, z_raw, regime):
    return {
        "ticker": ticker, "nav": 100.0, "shares_outstanding": 1_000_000,
        "total_net_assets": 100_000_000, "primary_flow_usd": 1_000_000.0,
        "primary_flow_pct": 1.0, "primary_flow_z": z,
        "primary_flow_z_raw": z_raw, "primary_flow_z_regime": regime,
        "Date": "2026-10-13",
    }


def test_z_normal_sin_asterisco():
    """Z normal se muestra limpio, sin asterisco."""
    df = _make_df([_base_row("XLK", 1.50, 1.50, "NORMAL")])
    out = "".join(render_flujo_spdr(df))
    assert "+1.50" in out
    # No debe aparecer el patron de saturacion
    lineas_xlk = [l for l in out.split("\n") if "| XLK |" in l]
    assert len(lineas_xlk) == 1
    assert "*" not in lineas_xlk[0]


def test_z_saturado_muestra_asterisco_y_raw():
    """Z saturado se muestra con * y el bruto entre parentesis."""
    df = _make_df([_base_row("XLP", 5.0, 8.31, "SATURATED")])
    out = "".join(render_flujo_spdr(df))
    assert "+5.00*" in out
    assert "(+8.31)" in out


def test_z_saturado_negativo_muestra_raw_negativo():
    """Z saturado negativo se muestra con * y el bruto negativo."""
    df = _make_df([_base_row("XLE", -5.0, -12.5, "SATURATED")])
    out = "".join(render_flujo_spdr(df))
    assert "-5.00*" in out
    assert "(-12.50)" in out


def test_nota_explica_asterisco():
    """Debe existir una nota al pie que explique el significado del *."""
    df = _make_df([_base_row("XLP", 5.0, 8.31, "SATURATED")])
    out = "".join(render_flujo_spdr(df))
    assert "clip contractual" in out or "clip" in out.lower()
    assert "[-5, +5]" in out or "+5" in out


def test_z_saturado_sin_raw_no_falla():
    """Si raw es NaN, se muestra sin parentesis (backward-compat)."""
    df = _make_df([_base_row("XLP", 5.0, float("nan"), "SATURATED")])
    out = "".join(render_flujo_spdr(df))
    # No debe incluir parentesis
    lineas = [l for l in out.split("\n") if "| XLP |" in l]
    assert len(lineas) == 1
    assert "(" not in lineas[0]
    assert "+5.00" in lineas[0]


def test_columna_z_raw_ausente_no_falla():
    """Backward-compat: sin columna primary_flow_z_raw el render funciona."""
    row = _base_row("XLK", 1.50, 1.50, "NORMAL")
    del row["primary_flow_z_raw"]
    df = _make_df([row])
    out = "".join(render_flujo_spdr(df))
    assert "+1.50" in out


def test_xlp_reproduce_caso_auditor():
    """Reproduce el caso H6: XLP con 622.85M / Z saturado a 5."""
    df = _make_df([{
        "ticker": "XLP", "nav": 84.16831,
        "shares_outstanding": 175_021_809,
        "total_net_assets": 14_731_289_812,
        "primary_flow_usd": 622_845_494.0,
        "primary_flow_pct": 4.228,
        "primary_flow_z": 5.0,
        "primary_flow_z_raw": 8.31,
        "primary_flow_z_regime": "SATURATED",
        "Date": "2026-10-13",
    }])
    out = "".join(render_flujo_spdr(df))
    lineas = [l for l in out.split("\n") if "| XLP |" in l]
    assert len(lineas) == 1
    linea = lineas[0]
    # Debe mostrar el Z con asterisco y el bruto
    assert "+5.00*" in linea
    assert "(+8.31)" in linea
    # Y el flujo real sin modificar
    assert "622,845,494.00" in linea
    # Y el % AUM real
    assert "+4.23%" in linea
