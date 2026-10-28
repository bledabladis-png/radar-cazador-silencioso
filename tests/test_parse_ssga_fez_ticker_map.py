"""Tests del FEZ_TICKER_MAP (bug FEZ sesion 14, commit ee7aac87).

Los 27 mapeos nuevos de ee7aac87 no tienen test. Sin test, un ticker
nuevo sin mapeo pasa crudo a Yahoo y puede resolver a un ADR
americano erroneo (DG -> Dollar General, EL -> Estee Lauder,
BN -> Brookfield, AD -> Kroger).

Este test cierra esa puerta en 3 niveles:
1. Estructura: los ~46 mapeos verificados estan presentes.
2. Resolucion: resolve_fez_ticker(raw, name) los resuelve bien.
3. End-to-end: el CSV FEZ actual no tiene tickers sin mapeo ni
   sufijo europeo.

API real (no ideal): resolve_fez_ticker(raw_ticker, name). El `name`
es necesario por FEZ_NAME_OVERRIDE (SANOFI vs BANCO SANTANDER).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import parse_ssga_fez as fez


# Mapeos verificados en ee7aac87. Si uno desaparece del dict, este
# test falla. NDA FI NO esta (excluido por D11, ver test aparte).
TICKERS_VERIFICADOS = {
    # Anteriores a ee7aac87
    'ASML': 'ASML.AS', 'SIE': 'SIE.DE', 'SAN': 'SAN.MC', 'SU': 'SU.PA',
    'SAP': 'SAP.DE', 'TTE': 'TTE.PA', 'ALV': 'ALV.DE', 'BBVA': 'BBVA.MC',
    'SAF': 'SAF.PA', 'ENR': 'ENR.DE', 'IBE': 'IBE.MC', 'UCG': 'UCG.MI',
    'AIR': 'AIR.PA', 'BNP': 'BNP.PA', 'MC': 'MC.PA', 'AI': 'AI.PA',
    'ISP': 'ISP.MI', 'DTE': 'DTE.DE', 'OR': 'OR.PA', 'INGA': 'INGA.AS',
    # Anadidos en ee7aac87
    'CS': 'CS.PA', 'ENEL': 'ENEL.MI', 'IFX': 'IFX.DE', 'ITX': 'ITX.MC',
    'DHL': 'DHL.DE', 'ENI': 'ENI.MI', 'DBK': 'DBK.DE', 'RHM': 'RHM.DE',
    'BAS': 'BAS.DE', 'ENGI': 'ENGI.PA', 'RMS': 'RMS.PA', 'BAYN': 'BAYN.DE',
    'PRX': 'PRX.AS', 'NOKIA': 'NOKIA.HE', 'SGO': 'SGO.PA', 'MBG': 'MBG.DE',
    'ADYEN': 'ADYEN.AS', 'ADS': 'ADS.DE', 'BMW': 'BMW.DE', 'ABI': 'ABI.BR',
    'DG': 'DG.PA', 'EL': 'EL.PA', 'ARGX': 'ARGX.BR', 'RACE': 'RACE.MI',
    'BN': 'BN.PA', 'AD': 'AD.AS',
}


def test_mapa_contiene_todos_los_tickers_verificados():
    faltan = [r for r in TICKERS_VERIFICADOS if r not in fez.FEZ_TICKER_MAP]
    assert not faltan, (
        "Faltan en FEZ_TICKER_MAP: " + str(sorted(faltan)) + ". "
        "Actuales: " + str(sorted(fez.FEZ_TICKER_MAP))
    )
    incorrectos = {
        r: (fez.FEZ_TICKER_MAP[r], e)
        for r, e in TICKERS_VERIFICADOS.items()
        if fez.FEZ_TICKER_MAP[r] != e
    }
    assert not incorrectos, "Mapeos incorrectos: " + str(incorrectos)


def test_resolve_ticker_usa_mapa():
    """resolve_fez_ticker(raw, '') devuelve el mapeo para cada ticker."""
    incorrectos = {}
    for raw, expected in TICKERS_VERIFICADOS.items():
        got = fez.resolve_fez_ticker(raw, '')
        if got != expected:
            incorrectos[raw] = (got, expected)
    assert not incorrectos, "resolve_fez_ticker divergente: " + str(incorrectos)


def test_resolve_ticker_override_por_nombre():
    """FEZ_NAME_OVERRIDE: SANOFI y BANCO SANTANDER comparten ticker SAN.
    El nombre desambigua: SANOFI -> SAN.PA, SANTANDER -> SAN.MC.
    """
    assert fez.resolve_fez_ticker('SAN', 'SANOFI') == 'SAN.PA'
    assert fez.resolve_fez_ticker('SAN', 'BANCO SANTANDER') == 'SAN.MC'
    # Sin nombre reconocible, cae al FEZ_TICKER_MAP
    assert fez.resolve_fez_ticker('SAN', '') == fez.FEZ_TICKER_MAP['SAN']


def test_ndafi_excluido_por_d11():
    """NDA FI excluido en ee7aac87: Yahoo lo resuelve (NDA-FI.HE) pero
    viola D11 (2 separadores: '-' y '.'). Nordea queda fuera de FEZ
    (-1.20% de peso; FEZ 96.29 -> 95.09)."""
    assert 'NDA FI' not in fez.FEZ_TICKER_MAP
    assert 'NDA' not in fez.FEZ_TICKER_MAP
    for v in ('NDA-FI', 'NDA.FI', 'NDA1V', 'NDAFI'):
        assert v not in fez.FEZ_TICKER_MAP, (
            f"{v} no deberia estar: NDA FI excluido por D11"
        )


def test_adrs_del_bug_original_resuelven_a_europa():
    """Los 4 ADRs del bug FEZ sesion 14: sin mapeo, Yahoo resuelve a
    la empresa USA del mismo ticker. Con mapeo, resuelve a la europea.
    """
    assert fez.resolve_fez_ticker('DG', '') == 'DG.PA'   # no Dollar General
    assert fez.resolve_fez_ticker('EL', '') == 'EL.PA'   # no Estee Lauder
    assert fez.resolve_fez_ticker('BN', '') == 'BN.PA'   # no Brookfield
    assert fez.resolve_fez_ticker('AD', '') == 'AD.AS'   # no Kroger


def test_csv_fez_real_todos_mapeados():
    """End-to-end: el CSV FEZ actual no tiene tickers sin mapeo ni
    sufijo europeo. Si aparece uno, este test falla con el listado
    exacto. Blinda contra el bug FEZ sesion 14.
    """
    import pandas as pd
    csv_path = ROOT / 'data' / 'index_holdings.csv'
    if not csv_path.exists():
        pytest.skip("index_holdings.csv no encontrado")
    df = pd.read_csv(csv_path)
    if 'etf' not in df.columns or 'ticker' not in df.columns:
        pytest.skip("CSV sin columnas etf/ticker")
    fez_df = df[df['etf'] == 'FEZ']
    if fez_df.empty:
        pytest.skip("CSV sin filas FEZ")
    sin_resolucion = []
    for _, row in fez_df.iterrows():
        raw_s = str(row['ticker']).strip()
        if not raw_s:
            continue
        name = str(row.get('name', '')) if 'name' in fez_df.columns else ''
        norm = fez.resolve_fez_ticker(raw_s, name)
        if raw_s not in fez.FEZ_TICKER_MAP and '.' not in norm:
            sin_resolucion.append(raw_s)
    assert not sin_resolucion, (
        "Tickers FEZ sin mapeo ni sufijo europeo (riesgo de resolver a "
        "ADR americano en Yahoo): " + str(sorted(sin_resolucion))
    )