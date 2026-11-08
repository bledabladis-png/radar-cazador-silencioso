"""Centinela: notas de sectorial.py sin literales hardcodeados de universo.

Contexto 2026-11-07: tras el cambio a universo completo, los literales
"~220 tickers", "~313 tickers", "top-20 mostrado" y "top-20 por weight
del ETF sectorial" quedaron obsoletos. Este test verifica que no
reaparezcan.
"""
from pathlib import Path

SRC = Path("src/report/sectorial.py")


def test_sin_literal_220_tickers():
    t = SRC.read_text(encoding="utf-8")
    assert "~220 tickers" not in t, (
        "literal '~220 tickers' obsoleto: el universo cambio, derivar o "
        "usar texto generico"
    )


def test_sin_literal_313_tickers():
    t = SRC.read_text(encoding="utf-8")
    assert "~313 tickers" not in t, (
        "literal '~313 tickers' obsoleto: el universo completo ahora es "
        "mayor, derivar o usar texto generico"
    )


def test_sin_literal_top20_mostrado():
    t = SRC.read_text(encoding="utf-8")
    assert "top-20 mostrado" not in t, (
        "literal 'top-20 mostrado' obsoleto: la tabla de concentracion "
        "ya no trunca a top-20"
    )


def test_sin_literal_top20_por_weight_lider():
    t = SRC.read_text(encoding="utf-8")
    assert "Universo: top-20 por weight del ETF sectorial" not in t, (
        "literal 'top-20 por weight del ETF sectorial' obsoleto: el lider "
        "se calcula sobre todos los holdings con metricas"
    )


def test_nota_cobertura_menciona_n_total():
    t = SRC.read_text(encoding="utf-8")
    assert "n_total" in t, "la nota de cobertura debe referenciar n_total"