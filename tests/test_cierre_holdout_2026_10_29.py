"""Contrato: los tickers del cierre SOW-HOLDOUT estan en get_stock_list.

Cierre 2026-10-29: cohort es 99.3% S&P 500 actual; los 24 tickers que
estaban bloqueados son holdings actuales de SPY/QQQ. Se promueven sin
contrato (informe 78). Este test blinda el contrato.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stock_data_loader import get_stock_list


# Los 21 que faltaban en SPY (QCOM/ADI/WDC/FTNT/ADBE/CDNS/LITE/SNPS/INTU
# ya estaban? No: verificado en raiz_*.py, faltaban los 21 + DDOG/MRNA
# que ya estaban en universe_expansions pero no en parquet).
SPY_NUEVOS = [
    'APH', 'QCOM', 'ADI', 'WDC', 'DELL', 'NOW', 'GLW', 'ACN', 'FTNT',
    'ADBE', 'CDNS', 'LITE', 'ICE', 'HPE', 'SNPS', 'MRSH', 'INTU', 'MSI', 'CI',
    'DDOG', 'MRNA',
]
QQQ_NUEVOS = ['SHOP']


def test_spy_extra_en_get_stock_list():
    lst = set(get_stock_list())
    faltan = [t for t in SPY_NUEVOS if t not in lst]
    assert not faltan, f"faltan en get_stock_list: {faltan}"


def test_qqq_extra_en_get_stock_list():
    lst = set(get_stock_list())
    faltan = [t for t in QQQ_NUEVOS if t not in lst]
    assert not faltan, f"faltan en get_stock_list: {faltan}"


def test_sin_duplicados_en_expansiones():
    """spy_extra y qqq_extra no repiten tickers entre si."""
    lst = get_stock_list()
    assert len(lst) == len(set(lst)), "get_stock_list tiene duplicados"