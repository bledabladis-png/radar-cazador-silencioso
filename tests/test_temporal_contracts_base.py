"""Tests de src/temporal_contracts/base.py (FU-021-5 Fase 1)."""
from datetime import date

import pandas as pd
import pytest

from src.temporal_contracts.base import (
    TemporalContract,
    TemporalResolution,
    MarketDataBundle,
    compute_status,
    STATUS_OK,
    STATUS_STALE,
    STATUS_INSUFFICIENT,
    STATUS_BLOCKED,
    STATUS_PENDING,
    VALID_STATUSES,
)


D_OK = date(2026, 9, 14)
D_PREV = date(2026, 9, 13)


class TestComputeStatus:

    def test_ok_frescura_perfecta(self):
        assert compute_status(D_OK, D_OK, 0, 1.0, 1, 0.9) == STATUS_OK

    def test_stale_lag_positivo_dentro_de_tolerancia(self):
        assert compute_status(D_PREV, D_OK, 1, 1.0, 1, 0.9) == STATUS_STALE

    def test_insufficient_lag_supera_max(self):
        assert compute_status(D_PREV, D_OK, 2, 1.0, 1, 0.9) == STATUS_INSUFFICIENT

    def test_insufficient_coverage_baja(self):
        assert compute_status(D_PREV, D_OK, 1, 0.5, 1, 0.9) == STATUS_INSUFFICIENT

    def test_pending_sin_resolucion(self):
        assert compute_status(None, None, None, None, None, None) == STATUS_PENDING

    def test_blocked_activation_req_no_resuelto(self):
        assert (
            compute_status(D_OK, D_OK, 0, 1.0, 1, 0.9, activation_req_resolved=False)
            == STATUS_BLOCKED
        )

    def test_max_lag_dict_se_toma_el_maximo(self):
        ml = {"^FTSE": 1, "^GDAXI": 2}
        assert compute_status(D_PREV, D_OK, 2, 1.0, ml, 0.9) == STATUS_STALE
        assert compute_status(D_PREV, D_OK, 3, 1.0, ml, 0.9) == STATUS_INSUFFICIENT

    def test_valid_statuses_contiene_los_cinco(self):
        assert set(VALID_STATUSES) == {
            STATUS_PENDING, STATUS_OK, STATUS_STALE,
            STATUS_INSUFFICIENT, STATUS_BLOCKED,
        }


class TestTemporalResolution:

    def test_construccion_completa(self):
        tr = TemporalResolution(
            contract_name="EQUITY_EOD",
            effective_date=D_OK,
            expected_date=D_OK,
            lag_days=0,
            coverage=0.9981,
            status=STATUS_OK,
        )
        assert tr.contract_name == "EQUITY_EOD"
        assert tr.status == STATUS_OK
        assert tr.per_member is None


class TestMarketDataBundle:

    def test_bundle_transporta_df_y_meta(self):
        df = pd.DataFrame({"a": [1, 2]})
        meta = {"by_contract": {"EQUITY_EOD": {"status": "OK"}}}
        bundle = MarketDataBundle(df_market=df, temporal_meta=meta)
        assert bundle.df_market is df
        assert bundle.temporal_meta is meta

    def test_bundle_meta_por_defecto_vacio(self):
        df = pd.DataFrame({"a": [1]})
        bundle = MarketDataBundle(df_market=df)
        assert bundle.temporal_meta == {}


class TestTemporalContractABC:

    def test_no_instanciable_directamente(self):
        with pytest.raises(TypeError):
            TemporalContract()

    def test_subclase_debe_implementar_resolve(self):
        class Incompleto(TemporalContract):
            name = "X"
            family = "EQUITY"
            eligible_universe = []
            session_calendar = "NYSE"
            max_lag_days = 0
            min_coverage = 1.0

        with pytest.raises(TypeError):
            Incompleto()

    def test_is_eligible(self):
        class Fake(TemporalContract):
            name = "F"
            family = "EQUITY"
            eligible_universe = ["A", "B"]
            session_calendar = "NYSE"
            max_lag_days = 0
            min_coverage = 1.0

            def resolve(self, df_market, reference_date):
                return TemporalResolution(
                    contract_name=self.name,
                    effective_date=D_OK,
                    expected_date=D_OK,
                    lag_days=0,
                    coverage=1.0,
                    status=STATUS_OK,
                )

        f = Fake()
        assert f.is_eligible("A") is True
        assert f.is_eligible("Z") is False