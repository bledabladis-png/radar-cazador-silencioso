"""Tests Fase 2.2: VOLATILITY_INDEX + RATE_YIELD + FUTURE_SETTLEMENT + FX_DAILY_CUT."""
from datetime import datetime

import pandas as pd

from src.temporal_contracts import (
    VolatilityIndex,
    RateYield,
    FutureSettlement,
    FxDailyCut,
    get_contract,
    resolve_all_contracts,
    list_contracts,
    VALID_STATUSES,
    STATUS_BLOCKED,
)


REF = datetime(2026, 9, 15, 22, 0)


def _ref_from_df(df):
    """REF derivada del parquet: ultima fecha + 1 dia a las 22:00.

    Evita fragilidad ante el avance del parquet local tras un run E2E
    (mismo patron que test_temporal_contracts_contracts.py, e49beda).
    En CI df_real hace skip y REF global sigue usandose con mocks.
    """
    ref = pd.Timestamp(df.index.max()) + pd.Timedelta(days=1)
    return ref.replace(hour=22, minute=0, second=0).to_pydatetime()



class TestVolatilityIndex:

    def test_metadata(self):
        c = VolatilityIndex()
        assert c.name == "VOLATILITY_INDEX"
        assert c.family == "INDEX"
        assert c.max_lag_days == 1
        assert set(c.eligible_universe) == {"^VIX", "^VIX3M", "^VXN"}

    def test_resolve(self, df_real):
        r = VolatilityIndex().resolve(df_real, _ref_from_df(df_real))
        assert r.contract_name == "VOLATILITY_INDEX"
        assert r.status in VALID_STATUSES


class TestRateYield:

    def test_metadata(self):
        c = RateYield()
        assert c.name == "RATE_YIELD"
        assert c.family == "RATE_YIELD"
        assert set(c.eligible_universe) == {"^FVX", "^TNX"}

    def test_resolve(self, df_real):
        r = RateYield().resolve(df_real, _ref_from_df(df_real))
        assert r.contract_name == "RATE_YIELD"
        assert r.status in VALID_STATUSES


class TestFutureSettlement:

    def test_metadata_activo(self):
        c = FutureSettlement()
        assert c.name == "FUTURE_SETTLEMENT"
        assert c.family == "FUTURE"
        assert c.activation_req is None
        assert c.settlement_semantics == "close_proxy"
        assert set(c.eligible_universe) == {"BZ=F", "CL=F"}
        assert c.max_lag_days == 1

    def test_gc_hg_ng_fuera_del_universo(self):
        c = FutureSettlement()
        assert "GC=F" not in c.eligible_universe
        assert "HG=F" not in c.eligible_universe
        assert "NG=F" not in c.eligible_universe

    def test_resolve_status_valido(self, df_real):
        r = FutureSettlement().resolve(df_real, _ref_from_df(df_real))
        assert r.contract_name == "FUTURE_SETTLEMENT"
        assert r.status in VALID_STATUSES
        assert r.status != STATUS_BLOCKED

    def test_resolve_sin_df_no_falla(self):
        r = FutureSettlement().resolve(None, REF)
        assert r.contract_name == "FUTURE_SETTLEMENT"
        assert r.status in VALID_STATUSES

    def test_resolve_con_df_vacio_no_falla(self):
        r = FutureSettlement().resolve(pd.DataFrame(), REF)
        assert r.contract_name == "FUTURE_SETTLEMENT"
        assert r.status in VALID_STATUSES


class TestFxDailyCut:

    def test_metadata(self):
        c = FxDailyCut()
        assert c.name == "FX_DAILY_CUT"
        assert c.family == "FX"
        assert c.session_calendar == "FX_24h"
        assert c.min_coverage == 0.5
        assert c.per_pair_max_lag["EURUSD=X"] == 0
        assert c.per_pair_max_lag["USDCNY=X"] == 1

    def test_resolve(self, df_real):
        r = FxDailyCut().resolve(df_real, _ref_from_df(df_real))
        assert r.contract_name == "FX_DAILY_CUT"
        assert r.status in VALID_STATUSES


class TestResolveAllContracts:

    def test_devuelve_diez_resoluciones(self, df_real):
        resoluciones = resolve_all_contracts(df_real, _ref_from_df(df_real))
        assert len(resoluciones) == 10

    def test_claves_iguales_a_list_contracts(self, df_real):
        resoluciones = resolve_all_contracts(df_real, _ref_from_df(df_real))
        assert set(resoluciones.keys()) == set(list_contracts())

    def test_cada_resolucion_tiene_status_valido(self, df_real):
        resoluciones = resolve_all_contracts(df_real, _ref_from_df(df_real))
        for name, r in resoluciones.items():
            assert r.status in VALID_STATUSES, f"{name}: {r.status}"
            assert r.contract_name == name

    def test_future_settlement_ya_no_blocked(self, df_real):
        resoluciones = resolve_all_contracts(df_real, _ref_from_df(df_real))
        r = resoluciones["FUTURE_SETTLEMENT"]
        assert r.status in VALID_STATUSES
        assert r.status != STATUS_BLOCKED


class TestGetContractAll10:

    def test_los_diez_implementados(self):
        names = [
            "EQUITY_EOD", "INDEX_EOD_USA", "INDEX_EOD_EUROPA",
            "INDEX_EOD_COMMODITY", "INDEX_EOD_CURRENCY",
            "VOLATILITY_INDEX", "RATE_YIELD",
            "FUTURE_SETTLEMENT", "SPOT_COMMODITY", "FX_DAILY_CUT",
        ]
        for name in names:
            c = get_contract(name)
            assert c.name == name