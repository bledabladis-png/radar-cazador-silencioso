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

# --- Cableado S35: fecha *Datos al* desde df['date'] (patron H3) ---

def _df_con_date(**kwargs):
    import pandas as pd
    from datetime import date
    base = {"date": pd.Timestamp(date(2026, 11, 9)), "sector": "XLK"}
    base.update(kwargs)
    return pd.DataFrame([base])


def test_sector_concentration_publica_datos_al():
    from src.report.sectorial import render_sector_concentration
    df = _df_con_date(
        top1_positive_return_concentration=0.0,
        top3_positive_return_concentration=0.0,
        top5_positive_return_concentration=0.0,
        rs_median=0.0, momentum_median=0.0, flow_median=0.0,
        wyckoff_median=0.0, wls_median=0.0, leader_ticker="XLK",
        leader_return20=0.0, n_valid_rs=20, coverage_rs=100,
        coverage_momentum=100, coverage_flow=100, coverage_wyckoff=100,
        coverage_wls=100, coverage_rs_sobre_descarga=100, coverage_peso=100,
    )
    out = "".join(render_sector_concentration(df))
    assert "*Datos al 2026-11-09.*" in out


def test_wyckoff_sectorial_publica_datos_al():
    from src.report.sector_context import render_wyckoff_sectorial
    df = _df_con_date(
        n_total=20, n_valid_wyckoff=20, coverage_wyckoff=100.0,
        pct_accumulation_forming=0.0, pct_accumulation=0.0,
        pct_markup=0.0, pct_range=100.0, pct_distribution_forming=0.0,
        pct_distribution=0.0, pct_markdown=0.0, pct_conflict=0.0,
    )
    out = "".join(render_wyckoff_sectorial(df))
    assert "*Datos al 2026-11-09.*" in out


def test_momentum_amplitud_publica_datos_al():
    from src.report.sector_context import render_momentum_amplitud
    df = _df_con_date(
        delta_1d_ema20=0.0, delta_5d_ema20=0.0, delta_20d_ema20=0.0,
        delta_5d_ema50=0.0, delta_5d_ema200=0.0,
        breadth_expansion_5d=False, breadth_deterioration_5d=False,
    )
    out = "".join(render_momentum_amplitud(df))
    assert "*Datos al 2026-11-09.*" in out


def test_sector_breadth_publica_datos_al_si_no_stale():
    from src.report.sectorial import render_sector_breadth
    df = _df_con_date(
        n_total=20, n_valid_ema20=20, n_valid_ema50=20, n_valid_ema200=20,
        n_valid_nhnl=20, n_valid_momentum=20, n_valid_ad=20,
        pct_above_ema20=50.0, pct_above_ema50=50.0, pct_above_ema200=50.0,
        pct_rs_positive=50.0, pct_momentum_positive=50.0,
        new_highs=0, new_lows=0, advances=10, declines=10, unchanged=0,
        ad_net=0, advance_pct=50.0, decline_pct=50.0,
    )
    out = "".join(render_sector_breadth(df, is_stale=False))
    assert "*Datos al 2026-11-09.*" in out


def test_sector_breadth_sin_datos_al_si_stale():
    from src.report.sectorial import render_sector_breadth
    df = _df_con_date(
        n_total=20, n_valid_ema20=20, n_valid_ema50=20, n_valid_ema200=20,
        n_valid_nhnl=20, n_valid_momentum=20, n_valid_ad=20,
        pct_above_ema20=50.0, pct_above_ema50=50.0, pct_above_ema200=50.0,
        pct_rs_positive=50.0, pct_momentum_positive=50.0,
        new_highs=0, new_lows=0, advances=10, declines=10, unchanged=0,
        ad_net=0, advance_pct=50.0, decline_pct=50.0,
    )
    out = "".join(render_sector_breadth(df, is_stale=True))
    # Si is_stale=True, ya hay WARN con fecha propia. No duplicamos con *Datos al*.
    assert "*Datos al 2026-11-09.*" not in out


def test_wyckoff_holdings_acc_publica_datos_al():
    from src.report.wyckoff_holdings import render_wyckoff_holdings_acc
    import pandas as pd
    from datetime import date
    df = pd.DataFrame([{
        "date": pd.Timestamp(date(2026, 11, 9)),
        "group": "XLK", "group_type": "sector_usa",
        "ticker": "AAPL", "name": "Apple", "weight": 5.0,
        "phase": "MARKUP", "event_phase": "", "phase_since": "",
        "close": 100.0, "basis": "", "coverage_weight_pct": 90.0,
        "n_bars": 300, "evaluable": True,
    }])
    tables = {"usa_sector_acc": df, "usa_index_acc": None, "eu_index_acc": None}
    out = "".join(render_wyckoff_holdings_acc(tables))
    assert "*Datos al 2026-11-09.*" in out
