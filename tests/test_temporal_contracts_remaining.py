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
    STATUS_INSUFFICIENT,
    STATUS_OK,
    STATUS_STALE,
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
        assert c.settlement_semantics == "future_close"
        assert set(c.eligible_universe) == {"BZ=F", "CL=F", "GC=F", "HG=F", "NG=F"}
        assert c.max_lag_days == 1

    def test_gc_hg_ng_dentro_del_universo(self):
        c = FutureSettlement()
        assert "GC=F" in c.eligible_universe
        assert "HG=F" in c.eligible_universe
        assert "NG=F" in c.eligible_universe

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

    def test_cobertura_parcial_devuelve_insufficient(self):
        # K-FS-CI-PARITY-01: si un ticker del universo no tiene Close en la
        # fecha esperada, el contrato debe reportar INSUFFICIENT. Es un
        # estado contractual valido, no un falso positivo.
        # Universo actual: BZ=F, CL=F, GC=F, HG=F, NG=F. Aqui solo 1/5
        # tiene Close -> coverage 0.2 < 1.0 -> INSUFFICIENT.
        idx = pd.to_datetime(["2026-09-15"])
        cols = pd.MultiIndex.from_tuples(
            [("Close", "BZ=F"), ("Close", "CL=F"),
             ("Close", "GC=F"), ("Close", "HG=F"), ("Close", "NG=F")],
            names=["field", "ticker"],
        )
        df = pd.DataFrame(index=idx, columns=cols, dtype=float)
        df.loc["2026-09-15", ("Close", "BZ=F")] = 108.32
        # Resto NaN -> cobertura 0.2 < min_coverage 1.0

        r = FutureSettlement().resolve(df, REF)
        assert r.status == STATUS_INSUFFICIENT
        assert r.effective_date is None
        assert r.coverage == 0.0
        assert r.lag_days is None

    def test_cobertura_completa_no_es_insufficient(self):
        # Contraste: con los 5 tickers presentes, el contrato no debe
        # reportar INSUFFICIENT.
        idx = pd.to_datetime(["2026-09-15"])
        cols = pd.MultiIndex.from_tuples(
            [("Close", "BZ=F"), ("Close", "CL=F"),
             ("Close", "GC=F"), ("Close", "HG=F"), ("Close", "NG=F")],
            names=["field", "ticker"],
        )
        df = pd.DataFrame(index=idx, columns=cols, dtype=float)
        df.loc["2026-09-15", ("Close", "BZ=F")] = 108.32
        df.loc["2026-09-15", ("Close", "CL=F")] = 99.0
        df.loc["2026-09-15", ("Close", "GC=F")] = 4213.70
        df.loc["2026-09-15", ("Close", "HG=F")] = 6.65
        df.loc["2026-09-15", ("Close", "NG=F")] = 3.02

        r = FutureSettlement().resolve(df, REF)
        assert r.status in VALID_STATUSES
        assert r.status != STATUS_BLOCKED
        assert r.status != STATUS_INSUFFICIENT


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

    def test_devuelve_nueve_resoluciones(self, df_real):
        resoluciones = resolve_all_contracts(df_real, _ref_from_df(df_real))
        assert len(resoluciones) == 9

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


class TestGetContractAll9:

    def test_los_nueve_implementados(self):
        names = [
            "EQUITY_EOD", "INDEX_EOD_USA", "INDEX_EOD_EUROPA",
            "INDEX_EOD_COMMODITY", "INDEX_EOD_CURRENCY",
            "VOLATILITY_INDEX", "RATE_YIELD",
            "FUTURE_SETTLEMENT", "FX_DAILY_CUT",
        ]
        for name in names:
            c = get_contract(name)
            assert c.name == name


# =============================================================================
# FX-PER-PAIR (2026-10-31, Fase 2 B1)
# =============================================================================

class TestFxDailyCutPerPair:
    """Fix: FX_DAILY_CUT resolvia per-par, no agregado.

    Antes compute_status colapsaba per_pair_max_lag a max(). Bug real:
    EURUSD con lag 1 (max 0) se clasificaba STALE en el agregado. Ahora
    cada par tiene su propio status y el agregado es el peor.
    """

    def _make_df(self, fechas_por_par):
        """fechas_por_par: {pair: last_date_str}. Todos con Close no-NaN."""
        idx = pd.date_range("2026-10-01", "2026-10-31", freq="B")
        data = {}
        for pair, last in fechas_por_par.items():
            series = pd.Series(float("nan"), index=idx)
            series.loc[:last] = 1.0
            data[("Close", pair)] = series
        return pd.DataFrame(data, index=idx)

    def test_todos_mismo_lag_cero_ok(self):
        # Todos los pares con ultima fecha = expected -> OK
        df = self._make_df({
            "EURUSD=X": "2026-10-30",
            "USDJPY=X": "2026-10-30",
            "USDCNY=X": "2026-10-30",
        })
        ref = pd.Timestamp("2026-10-30").to_pydatetime()
        r = FxDailyCut().resolve(df, ref)
        assert r.status == STATUS_OK, r.status
        for pair, info in r.per_member.items():
            assert info["status"] == STATUS_OK, f"{pair}: {info['status']}"

    def test_eurusd_lag1_insufficient(self):
        """T1: EURUSD (max 0) con lag 1 -> INSUFFICIENT, no STALE."""
        df = self._make_df({
            "EURUSD=X": "2026-10-29",
            "USDJPY=X": "2026-10-30",
            "USDCNY=X": "2026-10-30",
        })
        ref = pd.Timestamp("2026-10-30").to_pydatetime()
        r = FxDailyCut().resolve(df, ref)
        # EURUSD con lag 1 y max 0 -> INSUFFICIENT
        assert r.per_member["EURUSD=X"]["status"] == STATUS_INSUFFICIENT
        assert r.per_member["EURUSD=X"]["max_lag_days"] == 0
        # Los otros OK
        assert r.per_member["USDJPY=X"]["status"] == STATUS_OK
        assert r.per_member["USDCNY=X"]["status"] == STATUS_OK
        # Agregado: peor = INSUFFICIENT
        assert r.status == STATUS_INSUFFICIENT

    def test_usdcny_lag1_stale(self):
        """T2: USDCNY (max 1) con lag 1 -> STALE, correcto."""
        df = self._make_df({
            "EURUSD=X": "2026-10-30",
            "USDJPY=X": "2026-10-30",
            "USDCNY=X": "2026-10-29",
        })
        ref = pd.Timestamp("2026-10-30").to_pydatetime()
        r = FxDailyCut().resolve(df, ref)
        assert r.per_member["USDCNY=X"]["status"] == STATUS_STALE
        assert r.per_member["USDCNY=X"]["max_lag_days"] == 1
        assert r.per_member["EURUSD=X"]["status"] == STATUS_OK
        assert r.per_member["USDJPY=X"]["status"] == STATUS_OK
        # Agregado: peor = STALE
        assert r.status == STATUS_STALE

    def test_agregado_es_peor_estado(self):
        """T3: si un par es INSUFFICIENT y otro STALE, agregado = INSUFFICIENT."""
        df = self._make_df({
            "EURUSD=X": "2026-10-28",  # lag 2, max 0 -> INSUFFICIENT
            "USDJPY=X": "2026-10-30",  # OK
            "USDCNY=X": "2026-10-29",  # lag 1, max 1 -> STALE
        })
        ref = pd.Timestamp("2026-10-30").to_pydatetime()
        r = FxDailyCut().resolve(df, ref)
        assert r.status == STATUS_INSUFFICIENT, (
            f"INSUFFICIENT debe dominar a STALE. status={r.status}"
        )

    def test_per_member_tiene_las_tres_claves(self):
        """T4: per_member expone los 3 pares con sus status."""
        df = self._make_df({
            "EURUSD=X": "2026-10-30",
            "USDJPY=X": "2026-10-30",
            "USDCNY=X": "2026-10-30",
        })
        ref = pd.Timestamp("2026-10-30").to_pydatetime()
        r = FxDailyCut().resolve(df, ref)
        assert set(r.per_member.keys()) == {"EURUSD=X", "USDJPY=X", "USDCNY=X"}
        for pair, info in r.per_member.items():
            for k in ("effective_date", "lag_days", "coverage",
                      "max_lag_days", "status"):
                assert k in info, f"{pair} sin clave {k}"
