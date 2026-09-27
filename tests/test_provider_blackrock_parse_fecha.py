"""Tests para parse_fecha_es en blackrock_fund_data y blackrock_isf_fund_data.

Cubre A5-71: reemplazo de except desnudo por except (ValueError, TypeError).
Los casos aqui verifican que el contrato observable se preserva.
"""
import pandas as pd
import pytest

from data.providers.blackrock_fund_data import parse_fecha_es as parse_dax
from data.providers.blackrock_isf_fund_data import parse_fecha_es as parse_isf


PARSERS = [
    pytest.param(parse_dax, id="blackrock_fund_data"),
    pytest.param(parse_isf, id="blackrock_isf_fund_data"),
]


class TestContratoObservable:
    """Contrato comun a los dos parsers."""

    @pytest.mark.parametrize("parse", PARSERS)
    @pytest.mark.parametrize("entrada", [
        None,
        float("nan"),
        pd.NaT,
        "",
        "   ",
        "27",
        "27 dic",
        "27 dic 2000 x",
        123,
        123.45,
        True,
    ])
    def test_entradas_no_parseables_devuelven_none(self, parse, entrada):
        assert parse(entrada) is None
    @pytest.mark.parametrize("parse", PARSERS)
    @pytest.mark.parametrize("entrada, esperado", [
        ("27 dic 2000", pd.Timestamp(2000, 12, 27)),
        ("1 ene 2026", pd.Timestamp(2026, 1, 1)),
        ("27 sept 2000", pd.Timestamp(2000, 9, 27)),
        ("27 DIC 2000", pd.Timestamp(2000, 12, 27)),
        ("27 Dic 2000", pd.Timestamp(2000, 12, 27)),
        ("  27 dic 2000  ", pd.Timestamp(2000, 12, 27)),
    ])
    def test_entradas_validas_devuelven_timestamp(self, parse, entrada, esperado):
        resultado = parse(entrada)
        assert resultado == esperado
        assert isinstance(resultado, pd.Timestamp)

class TestRegresionA5_71:
    """Casos que motivaron el fix: except desnudo -> except (ValueError, TypeError).

    Verifica que el contrato observable se preserva tras el acotamiento.
    """

    @pytest.mark.parametrize("parse", PARSERS)
    @pytest.mark.parametrize("entrada", [
        "32 dic 2000",
        "0 ene 2000",
        "27 abc 2000",
        "27 dic abc",
        "abc dic 2000",
    ])
    def test_entradas_semanticamente_invalidas_devuelven_none(self, parse, entrada):
        assert parse(entrada) is None

class TestConsistenciaDaxIsf:
    """Los dos parsers deben comportarse identico (codigo duplicado)."""

    @pytest.mark.parametrize("entrada", [
        "27 dic 2000",
        "1 ene 2026",
        "27 sept 2000",
        "32 dic 2000",
        "27 abc 2000",
        None,
        "",
        "27 dic",
    ])
    def test_mismo_input_mismo_output(self, entrada):
        assert parse_dax(entrada) == parse_isf(entrada)