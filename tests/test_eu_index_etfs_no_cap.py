"""Tests de EU_INDEX_ETFS_NO_CAP (bug 2026-10-27, fix d6c3ebb4).

El CSV de index_holdings tiene holdings completos (FEZ 47, DAXEX 32,
ISF.L 88, LYXI 35), pero get_stock_list() hacia .head(20) sobre TODOS
los ETFs de indice. Resultado: 112 tickers EU fuera del parquet sin
que ningun test lo detectara.

Este test blinda la excepcion en 3 niveles:
1. Estructural: EU_INDEX_ETFS_NO_CAP contiene exactamente los 4
   esperados y NO contiene los USA (SPY/QQQ/DIA/IWM).
2. Funcional (mock): con CSV sintetico de 30 tickers USA + 25 EU por
   ETF, get_stock_list() devuelve 20 por ETF USA y 25 por ETF EU.
3. End-to-end (real): si el CSV real de cualquier EU tiene >20
   tickers, todos estan en get_stock_list(). Si el cap se reintroduce,
   solo 20 estarian y el test falla con el listado exacto.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.stock_data_loader as sdl


# Contrato: los 4 ETFs EU sin cap. Si se anade uno nuevo, este test
# falla y obliga a actualizar la constante y la documentacion.
EU_ESPERADOS = ("FEZ", "DAXEX", "ISF.L", "LYXI")
# USA con cap explicito (por diseno: holdings masivos, poco aporte
# al sistema holdings-based, IWM excluido por decision de producto).
USA_CON_CAP = ("SPY", "QQQ", "DIA", "IWM")


# ---------------------------------------------------------------------
# 1. Estructural: constante correcta
# ---------------------------------------------------------------------

def test_eu_index_etfs_no_cap_contiene_los_4_esperados():
    assert sdl.EU_INDEX_ETFS_NO_CAP == EU_ESPERADOS, (
        f"EU_INDEX_ETFS_NO_CAP cambio: {sdl.EU_INDEX_ETFS_NO_CAP}. "
        f"Esperado: {EU_ESPERADOS}. Si el cambio es deliberado, "
        f"actualizar constante + documentacion + este test."
    )


def test_usa_etfs_no_estan_en_no_cap():
    for etf in USA_CON_CAP:
        assert etf not in sdl.EU_INDEX_ETFS_NO_CAP, (
            f"{etf} no debe estar en EU_INDEX_ETFS_NO_CAP: sus holdings "
            f"masivos harian el parquet inmanejable y no aportan al "
            f"sistema holdings-based (decision 2026-10-24)."
        )


# ---------------------------------------------------------------------
# 2. Funcional (mock): comportamiento exacto cap/no-cap
# ---------------------------------------------------------------------

def test_cap_usa_y_no_cap_eu_con_csv_sintetico(tmp_path, monkeypatch):
    """Con 30 tickers por ETF USA y 25 por ETF EU, get_stock_list()
    devuelve 20 de los USA (cap) y 25 de los EU (sin cap).
    Aislado del repo real via chdir a tmp_path.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Prefijos unicos por ETF para que startswith() no cruce:
    prefijos = {
        "SPY":   "AAA",   # 30 tickers, esperado 20 (cap)
        "QQQ":   "BBB",   # 30, esperado 20
        "DIA":   "CCC",   # 30, esperado 20
        "IWM":   "DDD",   # 30, esperado 20
        "FEZ":   "EEE",   # 25, esperado 25 (no cap)
        "DAXEX": "FFF",   # 25, esperado 25
        "ISF.L": "GGG",   # 25, esperado 25
        "LYXI":  "HHH",   # 25, esperado 25
    }
    rows = []
    for etf, prefijo in prefijos.items():
        n = 30 if etf in USA_CON_CAP else 25
        for i in range(n):
            rows.append({
                "etf": etf,
                "ticker": f"{prefijo}{i:03d}",
                "weight": float(n - i),
            })
    pd.DataFrame(rows).to_csv(data_dir / "index_holdings.csv", index=False)
    # etf_holdings.csv, radar_target_catalog.csv y universe_expansions
    # no existen en tmp_path -> get_stock_list los ignora via try/except.
    # Providers europeos devuelven sus tickers reales, que no empiezan
    # por ninguno de los prefijos AAA-HHH, asi que no contaminan.
    monkeypatch.chdir(tmp_path)

    result = list(sdl.get_stock_list())

    for etf, prefijo in prefijos.items():
        count = sum(1 for t in result if t.startswith(prefijo))
        esperado = 20 if etf in USA_CON_CAP else 25
        assert count == esperado, (
            f"{etf}: {count} tickers en get_stock_list() (prefijo "
            f"{prefijo}), esperado {esperado} "
            f"({'cap por TOP_N_SECTOR_COMPONENTS' if etf in USA_CON_CAP else 'sin cap por EU_INDEX_ETFS_NO_CAP'})"
        )


# ---------------------------------------------------------------------
# 3. End-to-end (real): el CSV real no pierde tickers por cap
# ---------------------------------------------------------------------

def _csv_index_por_etf(etf):
    """Devuelve tickers del ETF en el CSV real, o [] si no disponible."""
    p = ROOT / "data" / "index_holdings.csv"
    if not p.exists():
        pytest.skip("index_holdings.csv no disponible")
    df = pd.read_csv(p)
    if "etf" not in df.columns or "ticker" not in df.columns:
        pytest.skip("CSV sin columnas etf/ticker")
    return df[df["etf"] == etf]["ticker"].astype(str).tolist()


def test_csv_real_eu_tienen_mas_de_20_por_etf():
    """Sanity: si los 4 EU tuvieran <=20 en el CSV, el test siguiente
    no probaria nada (no puede haber regresion por cap)."""
    for etf in EU_ESPERADOS:
        tickers = _csv_index_por_etf(etf)
        assert len(tickers) > 20, (
            f"{etf}: {len(tickers)} tickers en el CSV. El test de "
            f"no-cap no tiene sentido si no hay >20."
        )


def test_csv_real_eu_todos_en_get_stock_list():
    """End-to-end: si el cap se reintroduce, solo 20 de los N EU del
    CSV estarian en get_stock_list(). Falla con el listado exacto."""
    from src.instrument_registry import normalize_yahoo_ticker
    listed = set(sdl.get_stock_list())
    fallos = {}
    for etf in EU_ESPERADOS:
        tickers_csv = _csv_index_por_etf(etf)
        if len(tickers_csv) <= 20:
            continue  # cubierto por el sanity anterior
        normalizados = [normalize_yahoo_ticker(t) for t in tickers_csv]
        ausentes = [t for t in normalizados if t not in listed]
        if ausentes:
            fallos[etf] = ausentes
    assert not fallos, (
        "Tickers EU del CSV no estan en get_stock_list (cap reintroducido "
        "o provider caido?): " + str(fallos)
    )


def test_index_etfs_una_sola_fuente_de_verdad():
    """USA_INDEX_ETFS y EU_INDEX_ETFS viven en config.tickers.

    FASE0 (2026-10-31): antes estaban duplicadas en
    indicators/wyckoff_holdings_report.py y src/stock_data_loader.py
    (esta ultima como EU_INDEX_ETFS_NO_CAP). Ahora las 3 fuentes
    importan de config.tickers. Si alguien reintroduce un literal en
    cualquiera de los 3 sitios, este test falla.
    """
    from config.tickers import EU_INDEX_ETFS
    import src.stock_data_loader as sdl
    # src/stock_data_loader expone el alias EU_INDEX_ETFS_NO_CAP.
    assert sdl.EU_INDEX_ETFS_NO_CAP == EU_INDEX_ETFS
    # Los consumidores importan las constantes de config, no las
    # redeclaran. AST check: buscar asignacion literal en los modulos.
    import ast
    from pathlib import Path
    for rel in ("indicators/wyckoff_holdings_report.py",
                "src/pipeline/wyckoff_holdings.py",
                "src/stock_data_loader.py"):
        src = Path(rel).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id in (
                        "USA_INDEX_ETFS", "EU_INDEX_ETFS",
                        "EU_INDEX_ETFS_NO_CAP",
                    ):
                        if isinstance(node.value, (ast.Tuple, ast.List)):
                            raise AssertionError(
                                f"{rel}:{node.lineno} redefine {t.id} "
                                f"con un literal. Debe importarse de "
                                f"config.tickers."
                            )
