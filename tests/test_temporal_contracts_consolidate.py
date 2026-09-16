"""Tests Fase 3: consolidate + build_temporal_meta + get_effective_meta."""
from datetime import date, datetime

import pytest

from src.temporal_contracts import (
    consolidate,
    build_temporal_meta,
    resolve_all_contracts,
    MarketDataBundle,
    TemporalResolution,
    STATUS_OK,
    STATUS_STALE,
    STATUS_INSUFFICIENT,
    STATUS_BLOCKED,
)
from src.utils import get_effective_meta
from config.settings import CURRENT_TEMPORAL_CONTRACT_VERSION


REF = datetime(2026, 9, 15, 22, 0)
D_1509 = date(2026, 9, 15)
D_1409 = date(2026, 9, 14)
D_1109 = date(2026, 9, 11)


def _res(name, eff, exp, lag, cov, status):
    return TemporalResolution(
        contract_name=name,
        effective_date=eff,
        expected_date=exp,
        lag_days=lag,
        coverage=cov,
        status=status,
    )



class TestBuildTemporalMeta:

    def test_estructura_basica(self):
        resolutions = {
            "EQUITY_EOD": _res("EQUITY_EOD", D_1409, D_1409, 0, 1.0, STATUS_OK),
        }
        meta = build_temporal_meta(resolutions, REF, "run_x")
        assert set(meta.keys()) == {
            "by_contract", "global_last_date", "reference_date", "run_id"
        }
        assert meta["run_id"] == "run_x"
        assert meta["reference_date"] is REF

    def test_by_contract_contiene_todos(self):
        resolutions = {
            "EQUITY_EOD": _res("EQUITY_EOD", D_1409, D_1409, 0, 1.0, STATUS_OK),
            "FUTURE_SETTLEMENT": _res(
                "FUTURE_SETTLEMENT", None, None, None, None, STATUS_BLOCKED
            ),
        }
        meta = build_temporal_meta(resolutions, REF)
        assert set(meta["by_contract"].keys()) == {
            "EQUITY_EOD", "FUTURE_SETTLEMENT"
        }
        assert meta["by_contract"]["FUTURE_SETTLEMENT"]["status"] == STATUS_BLOCKED

    def test_global_last_date_solo_ok_stale(self):
        resolutions = {
            "A": _res("A", D_1409, D_1409, 0, 1.0, STATUS_OK),
            "B": _res("B", D_1109, D_1409, 3, 1.0, STATUS_INSUFFICIENT),
            "C": _res("C", D_1509, D_1509, 0, 1.0, STATUS_STALE),
        }
        meta = build_temporal_meta(resolutions, REF)
        # max(1409, 1509) = 1509. 1109 excluido por INSUFFICIENT.
        assert meta["global_last_date"] == D_1509

    def test_global_last_date_none_si_nada_ok_stale(self):
        resolutions = {
            "A": _res("A", D_1109, D_1409, 3, 1.0, STATUS_INSUFFICIENT),
            "B": _res("B", None, None, None, None, STATUS_BLOCKED),
        }
        meta = build_temporal_meta(resolutions, REF)
        assert meta["global_last_date"] is None

    def test_rechaza_resolucion_no_valida(self):
        with pytest.raises(TypeError):
            build_temporal_meta({"X": "no soy resolution"}, REF)


class TestConsolidate:

    def test_devuelve_bundle(self, df_real):
        resolutions = resolve_all_contracts(df_real, REF)
        bundle = consolidate(df_real, resolutions, REF, "run_1")
        assert isinstance(bundle, MarketDataBundle)

    def test_df_es_el_mismo_objeto(self, df_real):
        resolutions = resolve_all_contracts(df_real, REF)
        bundle = consolidate(df_real, resolutions, REF, "run_1")
        assert bundle.df_market is df_real

    def test_meta_tiene_10_contratos(self, df_real):
        resolutions = resolve_all_contracts(df_real, REF)
        bundle = consolidate(df_real, resolutions, REF, "run_1")
        assert len(bundle.temporal_meta["by_contract"]) == 10

    def test_no_modifica_df_original(self, df_real):
        shape_before = df_real.shape
        cols_before = list(df_real.columns)
        resolutions = resolve_all_contracts(df_real, REF)
        consolidate(df_real, resolutions, REF, "run_1")
        assert df_real.shape == shape_before
        assert list(df_real.columns) == cols_before

    def test_global_last_date_presente(self, df_real):
        resolutions = resolve_all_contracts(df_real, REF)
        bundle = consolidate(df_real, resolutions, REF, "run_1")
        assert bundle.temporal_meta["global_last_date"] is not None


class TestGetEffectiveMeta:

    def test_un_contrato(self):
        # Mock determinista: EQUITY_EOD INSUFFICIENT con effective_date conocida.
        resolutions = {
            "EQUITY_EOD": _res(
                "EQUITY_EOD", D_1109, D_1409, 3, 1.0, STATUS_INSUFFICIENT),
        }
        meta = build_temporal_meta(resolutions, REF, "run_test")
        m = get_effective_meta(meta, ["EQUITY_EOD"])
        assert m["status"] == "INSUFFICIENT"
        assert m["effective_date"] == D_1109
        assert m["contracts"] == ["EQUITY_EOD"]

    def test_varios_contratos_peor_status(self):
        # Mock: INDEX OK + EQUITY INSUFFICIENT -> combinado INSUFFICIENT.
        resolutions = {
            "INDEX_EOD_USA": _res(
                "INDEX_EOD_USA", D_1409, D_1409, 0, 1.0, STATUS_OK),
            "EQUITY_EOD": _res(
                "EQUITY_EOD", D_1109, D_1409, 3, 1.0, STATUS_INSUFFICIENT),
        }
        meta = build_temporal_meta(resolutions, REF, "run_test")
        m = get_effective_meta(meta, ["INDEX_EOD_USA", "EQUITY_EOD"])
        assert m["status"] == "INSUFFICIENT"

    def test_min_effective_date(self, df_real):
        resolutions = resolve_all_contracts(df_real, REF)
        bundle = consolidate(df_real, resolutions, REF, "run_1")
        m = get_effective_meta(
            bundle.temporal_meta, ["INDEX_EOD_USA", "INDEX_EOD_CURRENCY"]
        )
        # ambos 14/09 en el parquet actual
        assert m["effective_date"] == D_1409

    def test_contrato_inexistente_ignorado(self, df_real):
        resolutions = resolve_all_contracts(df_real, REF)
        bundle = consolidate(df_real, resolutions, REF, "run_1")
        m = get_effective_meta(bundle.temporal_meta, ["NO_EXISTE"])
        assert m == {}

    def test_meta_vacio_devuelve_vacio(self):
        assert get_effective_meta({}, ["EQUITY_EOD"]) == {}
        assert get_effective_meta(None, ["EQUITY_EOD"]) == {}

    def test_blocked_domina(self, df_real):
        resolutions = resolve_all_contracts(df_real, REF)
        bundle = consolidate(df_real, resolutions, REF, "run_1")
        m = get_effective_meta(
            bundle.temporal_meta, ["INDEX_EOD_USA", "FUTURE_SETTLEMENT"]
        )
        assert m["status"] == "BLOCKED"

    def test_stale_menor_prioridad_que_insufficient(self):
        # Mock: STALE + INSUFFICIENT -> combinado INSUFFICIENT.
        resolutions = {
            "INDEX_EOD_EUROPA": _res(
                "INDEX_EOD_EUROPA", D_1109, D_1409, 4, 1.0, STATUS_STALE),
            "EQUITY_EOD": _res(
                "EQUITY_EOD", D_1109, D_1409, 3, 1.0, STATUS_INSUFFICIENT),
        }
        meta = build_temporal_meta(resolutions, REF, "run_test")
        m = get_effective_meta(
            meta, ["INDEX_EOD_EUROPA", "EQUITY_EOD"])
        assert m["status"] == "INSUFFICIENT"


class TestSettings:

    def test_current_version_existe(self):
        assert isinstance(CURRENT_TEMPORAL_CONTRACT_VERSION, str)
        assert CURRENT_TEMPORAL_CONTRACT_VERSION == "FU-021-5-v2"


class TestIntegracionDataLoader:

    def test_data_loader_modulo_importa_sin_romper(self):
        # El patch anadido no debe romper el import del modulo.
        import src.data_loader as dl
        assert hasattr(dl, "download_market_data")

    def test_data_load_modulo_importa_sin_romper(self):
        import src.pipeline.data_load as dlp
        assert hasattr(dlp, "load_all_data")

    def test_utils_tiene_get_effective_meta(self):
        import src.utils as u
        assert hasattr(u, "get_effective_meta")

    def test_load_all_data_dict_contiene_temporal_meta_key(self):
        # Inspeccion estatica: el dict de retorno incluye la clave.
        import inspect
        import src.pipeline.data_load as dlp
        src_text = inspect.getsource(dlp.load_all_data)
        assert "temporal_meta" in src_text