"""
Flujo primario ISF.L desde BlackRock.

Wrapper fino sobre `_blackrock_base`. Declara la configuracion
especifica del ETF (URL, cache, output CSV, referer, label) y
delega la logica compartida al modulo base.

API publica: MESES_ES, parse_fecha_es, get_blackrock_isf_primary_flow.

Refs: A5-70 (refactor 2026-09-27).
"""
from pathlib import Path

from . import _blackrock_base


# --- Configuracion del ETF (no API) ---
URL_ISF = (
    "https://www.blackrock.com/es/profesionales/productos/251795/"
    "ishares-ftse-100-ucits-etf-inc-fund/1515395013987.ajax"
    "?fileType=xls&fileName=iShares-Core-FTSE-100-UCITS-ETF-INC-GBP-Acc_fund&dataType=fund"
)
CACHE_FILE = Path('data/cache/blackrock_isf_history.xml')
OUTPUT_CSV = Path('outputs/history/blackrock_isf_primary_flow.csv')
_REFERER = 'https://www.blackrock.com/es/profesionales/productos/251795/'
_LABEL = 'ISF.L'


# --- API publica (re-export de _blackrock_base) ---
MESES_ES = _blackrock_base.MESES_ES
parse_fecha_es = _blackrock_base.parse_fecha_es


def get_blackrock_isf_primary_flow(force_download: bool = False):
    """Descarga/procesa el histórico de ISF.L y devuelve la última fila con flujo."""
    return _blackrock_base.get_blackrock_primary_flow(
        url=URL_ISF,
        cache_file=CACHE_FILE,
        output_csv=OUTPUT_CSV,
        referer=_REFERER,
        label=_LABEL,
        force_download=force_download,
    )


__all__ = [
    "MESES_ES",
    "parse_fecha_es",
    "get_blackrock_isf_primary_flow",
]


if __name__ == '__main__':
    df = get_blackrock_isf_primary_flow(force_download=True)
    print('\núltima fila:')
    print(df)
