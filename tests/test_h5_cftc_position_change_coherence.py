# -*- coding: utf-8 -*-
"""REP-AUDIT-20261017-H5: coherencia entre pos_change y net_position
en CFTC.

El auditor (2026-10-17) detecto que el reporte publicaba S&P 500
asset_mgr 918,097 con pos_change +8,796 cuando la resta real es
918,097 - 909,302 = +8,795. Diferencia de 1 unidad.

Causa: CFTC publica Change_in_Long y Change_in_Short redondeados por
lado. Su diferencia puede no coincidir con el cambio real de Net
Position (326/853 filas en el historico verificado).

Fix: la serie canonica es net_position.diff().

Nota sobre el contrato de _calculate_position_flow: la funcion devuelve
UNA fila por (contract, participant), la mas reciente (drop_duplicates
keep='first' sobre sort_values por date descendente). Los tests operan
sobre esa fila de salida.
"""
import pandas as pd

from data.providers.cftc_data import _calculate_position_flow


TARGET = "S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE"


def _make_raw_cftc(n_dates=4, single_date=False):
    """DataFrame con la estructura minima que _calculate_position_flow
    espera del CSV crudo de CFTC.

    Escenario multi-fecha (default): 4 observaciones, net = [936671,
    906259, 909302, 918097], diffs = [-30412, +3043, +8795]. Ultima
    fila con CFTC=+8796 (error de 1). Esperamos que el fix use +8795.

    Escenario single_date: 1 observacion, sin previo. Esperamos que el
    fix use el CFTC como fallback.
    """
    if single_date:
        return pd.DataFrame({
            "Market_and_Exchange_Names": [TARGET],
            "Report_Date_as_YYYY_MM_DD": ["2026-10-13"],
            "Asset_Mgr_Positions_Long_All": [1_000_000],
            "Asset_Mgr_Positions_Short_All": [1_000_000 - 918097],
            "Change_in_Asset_Mgr_Long_All": [0],
            "Change_in_Asset_Mgr_Short_All": [8796],  # CFTC ficticio
            "Dealer_Positions_Long_All": [0],
            "Dealer_Positions_Short_All": [0],
            "Change_in_Dealer_Long_All": [0],
            "Change_in_Dealer_Short_All": [0],
            "Lev_Money_Positions_Long_All": [0],
            "Lev_Money_Positions_Short_All": [0],
            "Change_in_Lev_Money_Long_All": [0],
            "Change_in_Lev_Money_Short_All": [0],
        })

    dates = ["2026-09-22", "2026-09-29", "2026-10-06", "2026-10-13"]
    nets = [936671, 906259, 909302, 918097]
    longs = [1_000_000] * 4
    shorts = [longs[i] - nets[i] for i in range(4)]
    # chg_long = 0 en todos; chg_short = -diff (porque diff(long)-diff(short) debe dar diff(net))
    # diff(net) = [-30412, +3043, +8795]
    # diffs de short = -(-30412)=+30412, -(+3043)=-3043, -(+8795)=-8795
    chg_long = [0, 0, 0, 0]
    chg_short = [0, +30412, -3043, -8796]  # ultima -8796 para introducir error de +1

    return pd.DataFrame({
        "Market_and_Exchange_Names": [TARGET] * 4,
        "Report_Date_as_YYYY_MM_DD": dates,
        "Asset_Mgr_Positions_Long_All": longs,
        "Asset_Mgr_Positions_Short_All": shorts,
        "Change_in_Asset_Mgr_Long_All": chg_long,
        "Change_in_Asset_Mgr_Short_All": chg_short,
        "Dealer_Positions_Long_All": [0, 0, 0, 0],
        "Dealer_Positions_Short_All": [0, 0, 0, 0],
        "Change_in_Dealer_Long_All": [0, 0, 0, 0],
        "Change_in_Dealer_Short_All": [0, 0, 0, 0],
        "Lev_Money_Positions_Long_All": [0, 0, 0, 0],
        "Lev_Money_Positions_Short_All": [0, 0, 0, 0],
        "Change_in_Lev_Money_Long_All": [0, 0, 0, 0],
        "Change_in_Lev_Money_Short_All": [0, 0, 0, 0],
    })


def test_h5_salida_usa_net_diff():
    """Multi-fecha: la fila de salida (ultima, 2026-10-13) usa net_diff.

    net[ultima] - net[anterior] = 918097 - 909302 = 8795.
    CFTC publicaria 8796 (error de +1). El fix debe dar 8795.
    """
    df_raw = _make_raw_cftc()
    result = _calculate_position_flow(df_raw)
    sub = result[result["participant"] == "asset_mgr"]
    assert len(sub) == 1, "La funcion devuelve 1 fila (la mas reciente)"
    row = sub.iloc[0]
    assert int(row["position_change"]) == 8795, (
        "Debe ser net_diff (8795), no CFTC (8796). Got {}".format(
            row["position_change"]
        )
    )


def test_h5_identidad_matematica_respetada():
    """position_change de la salida == net actual - net anterior (historico).

    Verificacion directa: la fila de salida tiene net=918097,
    position_change=8795. Implica net_prev=909302.
    """
    df_raw = _make_raw_cftc()
    result = _calculate_position_flow(df_raw)
    sub = result[result["participant"] == "asset_mgr"]
    row = sub.iloc[0]
    net = int(row["net_position"])
    chg = int(row["position_change"])
    assert net - chg == 909302, (
        "net({}) - chg({}) debe dar net_prev=909302. Got {}".format(
            net, chg, net - chg
        )
    )


def test_h5_single_fecha_usa_cftc_fallback():
    """Una sola observacion: sin net_prev. Se conserva el CFTC."""
    df_raw = _make_raw_cftc(single_date=True)
    result = _calculate_position_flow(df_raw)
    sub = result[result["participant"] == "asset_mgr"]
    assert len(sub) == 1
    row = sub.iloc[0]
    # CFTC dice 8796 (chg_long=0 - chg_short=8796 -> -8796? ojo con signo)
    # En _make_raw_cftc single_date, chg_long=0, chg_short=8796.
    # pos_change_cftc = chg_long - chg_short = -8796.
    # Como no hay previo, diff es NaN, fillna(-8796) -> -8796.
    assert int(row["position_change"]) == -8796, (
        "Sin previo, debe usar CFTC (-8796). Got {}".format(
            row["position_change"]
        )
    )


def test_h5_flow_z_finito():
    """flow_z se calcula sobre position_change canonica. Valores finitos."""
    df_raw = _make_raw_cftc()
    result = _calculate_position_flow(df_raw)
    sub = result[result["participant"] == "asset_mgr"]
    assert "flow_z" in sub.columns
    assert sub["flow_z"].notna().all()
