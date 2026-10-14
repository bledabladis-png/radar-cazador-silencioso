import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pandas_datareader.data as web
from .base import MarketDataProvider


# N8 (2026-10-14): log de trazabilidad de descargas FRED.
# Objetivo: diagnosticar el frente de frescura del CI cuando un run
# recibe datos obsoletos. Cada descarga escribe una linea JSONL con
# metadata (URL, hora, tamano, ultima observacion).
_FRED_LOG_PATH = Path("outputs/audit/fred_downloads.jsonl")


def _log_fred_download(record: dict) -> None:
    """Append JSONL. Fallo silencioso: no debe tumbar el pipeline."""
    try:
        _FRED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_FRED_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except (OSError, TypeError, ValueError):
        pass

FRED_SERIES = {
    "fed_balance": "WALCL",
    "sofr": "SOFR",
    "reverse_repo": "RRPONTSYD",
    "fed_funds": "FEDFUNDS",
    "treasury_3m": "DTB3",
    "treasury_2y": "DGS2",
    "treasury_5y": "DGS5",
    "treasury_10y": "DGS10",
    "treasury_30y": "DGS30",
}

class FredProvider(MarketDataProvider):
    def __init__(self):
        self.name = "FRED (Federal Reserve)"
        self._cache = {}

    def get_name(self) -> str:
        return self.name

    def is_available(self) -> bool:
        return True

    def _download_series(self, series_id: str, start="2000-01-01", index=None) -> pd.Series:
        # N8: instrumentacion. Un registro por intento de descarga.
        _t0 = time.time()
        _ts = datetime.now(timezone.utc).isoformat()
        _url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        _rec = {
            "timestamp_utc": _ts,
            "series_id": series_id,
            "start": start,
            "url": _url,
            "status": "unknown",
            "n_rows": 0,
            "first_date": None,
            "last_date": None,
            "last_value": None,
            "duration_s": None,
        }
        try:
            data = web.DataReader(series_id, "fred", start=start)
            s = data.iloc[:, 0].sort_index()
            _rec["status"] = "ok"
            if len(s) > 0:
                _rec["n_rows"] = int(len(s))
                _rec["first_date"] = str(s.index[0].date())
                _rec["last_date"] = str(s.index[-1].date())
                _rec["last_value"] = float(s.iloc[-1]) if pd.notna(s.iloc[-1]) else None
            # Si se proporciona un índice externo, reindexar y rellenar
            if index is not None:
                s = s.reindex(index).ffill()
            else:
                # Extender el índice hasta hoy y rellenar
                # F5.6-06: fecha de ejecucion (extremo de rejilla
                # para reindexar). No es fecha de observacion: la
                # observacion real es s.index[-1]. El consumidor
                # (regimes/liquidity.py) usa dropna(how='all').
                # index[-1], no la longitud del indice.
                # F5.6-06b (2026-10-13): el reindex hasta 'today' generaba
                # fechas sinteticas (ffill de dias sin publicacion) que
                # los consumidores no podian distinguir de un dato real.
                # Bug observado: liquidity_state.json persistia
                # date=2026-10-13 cuando el ultimo dato real de FRED
                # era 2026-10-10. El reporte lo mostraba como 0 dias /
                # CURRENT. Fix: preservar el indice real (end=s.index[-1]).
                full_index = pd.date_range(start=s.index[0], end=s.index[-1], freq='D')
                s = s.reindex(full_index).ffill()
            _rec["duration_s"] = round(time.time() - _t0, 3)
            _log_fred_download(_rec)
            return s
        except (KeyError, ValueError, TypeError) as e:
            _rec["status"] = "error"
            _rec["error"] = str(e)
            _rec["duration_s"] = round(time.time() - _t0, 3)
            _log_fred_download(_rec)
            print(f"FRED {series_id}: {e}")
            return pd.Series(dtype=float)

    def get_prices(self, tickers: list, start: str = None, end: str = None, period: str = "10y") -> pd.DataFrame:
        raise NotImplementedError("FRED no proporciona precios de acciones. Usa Yahoo Finance.")

    def get_treasury_yields(self, maturities: list = None, index=None) -> pd.DataFrame:
        if maturities is None:
            maturities = ["treasury_3m", "treasury_2y", "treasury_10y"]
        df = pd.DataFrame()
        for key in maturities:
            if key in FRED_SERIES:
                series = self._download_series(FRED_SERIES[key], index=index)
                if not series.empty:
                    df[key] = series
        return df

    def get_fed_data(self, index=None) -> pd.DataFrame:
        series_ids = ["fed_balance", "sofr", "reverse_repo", "fed_funds"]
        df = pd.DataFrame()
        for key in series_ids:
            if key in FRED_SERIES:
                s = self._download_series(FRED_SERIES[key], index=index)
                if not s.empty:
                    df[key] = s
        return df

