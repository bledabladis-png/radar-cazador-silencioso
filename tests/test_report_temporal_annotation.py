# -*- coding: utf-8 -*-
"""Tests de src/report/_temporal_annotation.format_effective_date_line.

Cubre el helper que cierra parcialmente TEMPORAL-EFFECTIVE-DATE-POR-MODULO
y FRESHNESS-POR-BLOQUE (auditoria 2026-11-03, hallazgos 10.1 y 10.3).
"""
from datetime import date


from src.report._temporal_annotation import format_effective_date_line


def _tm(by_contract):
    """Construye un temporal_meta minimo con by_contract dado."""
    return {"by_contract": by_contract}


def test_temporal_meta_none_devuelve_vacio():
    assert format_effective_date_line(None, "SEC_13F") == ""


def test_temporal_meta_sin_by_contract_devuelve_vacio():
    assert format_effective_date_line({}, "SEC_13F") == ""


def test_contrato_ausente_devuelve_vacio():
    tm = _tm({"EQUITY_EOD": {"effective_date": date(2026, 11, 9),
                             "lag_days": 0, "status": "OK"}})
    assert format_effective_date_line(tm, "SEC_13F") == ""


def test_contrato_presente_lag_cero():
    tm = _tm({"EQUITY_EOD": {"effective_date": date(2026, 11, 9),
                             "lag_days": 0, "status": "OK"}})
    out = format_effective_date_line(tm, "EQUITY_EOD")
    assert out == "*Datos al 2026-11-09.*\n"


def test_contrato_presente_con_lag_positivo():
    tm = _tm({"FINRA_DARKPOOL": {"effective_date": date(2026, 10, 19),
                                 "lag_days": 22, "status": "STALE"}})
    out = format_effective_date_line(tm, "FINRA_DARKPOOL")
    assert out == "*Datos al 2026-10-19 (lag 22d).*\n"


def test_contrato_con_lag_none_omite_anotacion():
    tm = _tm({"SEC_13F": {"effective_date": date(2026, 9, 30),
                          "lag_days": None, "status": "STALE"}})
    out = format_effective_date_line(tm, "SEC_13F")
    assert out == "*Datos al 2026-09-30.*\n"


def test_include_lag_false_omite_lag():
    tm = _tm({"FRED_MACRO": {"effective_date": date(2026, 11, 6),
                             "lag_days": 4, "status": "OK"}})
    out = format_effective_date_line(tm, "FRED_MACRO", include_lag=False)
    assert out == "*Datos al 2026-11-06.*\n"


def test_prefix_custom():
    tm = _tm({"SEC_13F": {"effective_date": date(2026, 9, 30),
                          "lag_days": 90, "status": "STALE"}})
    out = format_effective_date_line(tm, "SEC_13F", prefix="Cierre")
    assert out == "*Cierre 2026-09-30 (lag 90d).*\n"


def test_multiples_contratos_interseccion():
    tm = _tm({
        "EQUITY_EOD":   {"effective_date": date(2026, 11, 9), "lag_days": 0,
                         "coverage": 0.99, "status": "OK"},
        "FRED_MACRO":   {"effective_date": date(2026, 11, 6), "lag_days": 3,
                         "coverage": 0.91, "status": "OK"},
    })
    out = format_effective_date_line(tm, ["EQUITY_EOD", "FRED_MACRO"])
    assert out == "*Datos al 2026-11-06 (lag 3d).*\n"


def test_string_es_lista_de_uno():
    tm = _tm({"SEC_13F": {"effective_date": date(2026, 9, 30),
                          "lag_days": 90, "status": "STALE"}})
    assert (format_effective_date_line(tm, "SEC_13F")
            == format_effective_date_line(tm, ["SEC_13F"]))


def test_fecha_como_string_iso():
    tm = _tm({"SEC_13F": {"effective_date": "2026-09-30",
                          "lag_days": 90, "status": "STALE"}})
    out = format_effective_date_line(tm, "SEC_13F")
    assert out == "*Datos al 2026-09-30 (lag 90d).*\n"

# --- Integracion: render_iae_section cableado ---

def test_iae_section_sin_temporal_meta_no_anota():
    """Backward compat: sin temporal_meta, render_iae_section no anade fecha."""
    from src.report.iae import render_iae_section
    out = render_iae_section({"status": "FOO"})
    assert "Datos al" not in "".join(out)


def test_iae_section_con_temporal_meta_sec_13f_anota():
    """Con SEC_13F en temporal_meta, publica 'Datos al ... (lag Nd)'."""
    from src.report.iae import render_iae_section
    tm = _tm({"SEC_13F": {"effective_date": date(2026, 9, 30),
                          "lag_days": 90, "status": "STALE"}})
    out = render_iae_section({"status": "FOO"}, temporal_meta=tm)
    text = "".join(out)
    assert "Datos al 2026-09-30" in text
    assert "(lag 90d)" in text


def test_iae_section_con_temporal_meta_sin_contrato_no_anota():
    """temporal_meta sin SEC_13F: no anota (evita fecha de otro contrato)."""
    from src.report.iae import render_iae_section
    tm = _tm({"EQUITY_EOD": {"effective_date": date(2026, 11, 9),
                             "lag_days": 0, "status": "OK"}})
    out = render_iae_section({"status": "FOO"}, temporal_meta=tm)
    assert "Datos al" not in "".join(out)

# --- Cableado: render_darkpool (FINRA_DARKPOOL) ---

def test_darkpool_sin_temporal_meta_no_anota():
    from src.report.darkpool import render_darkpool
    dp = {"week": "2026-10-19", "n_tickers": 534, "pct_volume": 25.22,
          "z_score": 0.14, "percentile": 52, "status": "normal", "z_windows": {}}
    out = "".join(render_darkpool(dp))
    assert "Datos al" not in out


def test_darkpool_con_temporal_meta_anota_finra():
    from src.report.darkpool import render_darkpool
    tm = _tm({"FINRA_DARKPOOL": {"effective_date": date(2026, 10, 19),
                                  "lag_days": 7, "status": "STALE"}})
    dp = {"week": "2026-10-19", "n_tickers": 534, "pct_volume": 25.22,
          "z_score": 0.14, "percentile": 52, "status": "normal", "z_windows": {}}
    out = "".join(render_darkpool(dp, temporal_meta=tm))
    assert "*Datos al 2026-10-19 (lag 7d).*" in out


# --- Cableado: render_regimenes (FRED_MACRO) ---

def test_regimenes_sin_temporal_meta_no_anota():
    import pandas as pd
    from src.report.header import render_regimenes
    out = "".join(render_regimenes(
        macro_score=pd.Series([0.0]), macro_regime="MIXED", macro_conf=0.5,
        liquidity_score=pd.Series([0.0]), financial_regime="NEUTRAL", liq_conf=0.5,
        volatility_score=pd.Series([0.0]), vol_regime="NORMAL", vol_conf=0.5,
        real_liquidity_regime="NEUTRAL", real_liquidity_conf=0.5,
        real_liq_score=pd.Series([0.0]), real_liq_prev=0.0,
        sector_regime="MIXED",
    ))
    assert "Datos al" not in out


def test_regimenes_con_temporal_meta_anota_fred():
    import pandas as pd
    from src.report.header import render_regimenes
    tm = _tm({"FRED_MACRO": {"effective_date": date(2026, 11, 6),
                              "lag_days": 3, "status": "OK"}})
    out = "".join(render_regimenes(
        macro_score=pd.Series([0.0]), macro_regime="MIXED", macro_conf=0.5,
        liquidity_score=pd.Series([0.0]), financial_regime="NEUTRAL", liq_conf=0.5,
        volatility_score=pd.Series([0.0]), vol_regime="NORMAL", vol_conf=0.5,
        real_liquidity_regime="NEUTRAL", real_liquidity_conf=0.5,
        real_liq_score=pd.Series([0.0]), real_liq_prev=0.0,
        sector_regime="MIXED", temporal_meta=tm,
    ))
    assert "*Datos al 2026-11-06 (lag 3d).*" in out


# --- Cableado: render_flujo_spdr (SSGA_PRIMARY_FLOW) ---

def test_flujo_spdr_sin_temporal_meta_no_anota():
    import pandas as pd
    from src.report.etf_flows import render_flujo_spdr
    df = pd.DataFrame([{
        "ticker": "XLK", "nav": 100.0, "shares_outstanding": 1000000.0,
        "delta_shares": 1000.0, "total_net_assets": 100000000.0,
        "primary_flow_usd": 100000.0, "primary_flow_pct": 0.001,
        "flow_z": 0.1, "z_score": 0.1,
    }])
    out = "".join(render_flujo_spdr(df))
    assert "Datos al" not in out


def test_flujo_spdr_con_temporal_meta_anota_ssga():
    import pandas as pd
    from src.report.etf_flows import render_flujo_spdr
    tm = _tm({"SSGA_PRIMARY_FLOW": {"effective_date": date(2026, 11, 9),
                                     "lag_days": 0, "status": "OK"}})
    df = pd.DataFrame([{
        "ticker": "XLK", "nav": 100.0, "shares_outstanding": 1000000.0,
        "delta_shares": 1000.0, "total_net_assets": 100000000.0,
        "primary_flow_usd": 100000.0, "primary_flow_pct": 0.001,
        "flow_z": 0.1, "z_score": 0.1,
    }])
    out = "".join(render_flujo_spdr(df, temporal_meta=tm))
    assert "*Datos al 2026-11-09.*" in out
