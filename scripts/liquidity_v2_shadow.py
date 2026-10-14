"""scripts/liquidity_v2_shadow.py

Modo SOMBRA de liquidity_v2.py (Fase 1, dictamen N8 Fase M, 14-oct-2026).

Objetivo:
  Ejecutar la especificacion experimental B3 en paralelo a liquidity.py,
  sin modificar el estado publico, sin alterar el regimen productivo y
  sin tocar la configuracion global de FRED.

Salida:
  Append-only JSONL en outputs/audit/liquidity_v2_shadow.jsonl.
  Esquema versionado (SCHEMA_VERSION).

Requisitos del dictamen:
  - EFFR obtenido por ruta aislada (NO modificar FRED_SERIES global).
  - NY Fed p99 obtenido por ruta aislada.
  - N/D != False: datos ausentes devuelven NA, no False.
  - Guard de frescura propio por serie.
  - Sin persistencia de estado compartido.
  - Append-only con esquema versionado.

Uso:
  py scripts/liquidity_v2_shadow.py

No se invoca desde run.py ni desde ningun consumer productivo.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# Permitir ejecucion directa
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from regimes.liquidity_v2 import evaluar_serie  # noqa: E402


SCHEMA_VERSION = "liquidity_v2_shadow_v1"
LOG_PATH = _ROOT / "outputs" / "audit" / "liquidity_v2_shadow.jsonl"

# Umbrales de frescura por serie (dias calendario).
# Distintos segun frecuencia de publicacion.
FRESHNESS_MAX_DAYS = {
    "SOFR": 4,          # diaria T+1
    "EFFR": 4,          # diaria T+1
    "NYFED_P99": 4,     # diaria T+1
}

# Ventana historica de descarga
START_DATE = "2018-04-02"


# ===================================================================
# Descargadores AISLADOS (no tocan configuracion global)
# ===================================================================

def _fetch_fred_series(sid: str, start: str = START_DATE) -> Optional[pd.Series]:
    """Descarga una serie FRED sin reindexado diario.
    NO usa data.providers.fred.FredProvider ni modifica FRED_SERIES.
    """
    try:
        import pandas_datareader.data as web
        df = web.DataReader(sid, "fred", start=start)
        s = df.iloc[:, 0].sort_index().dropna()
        return s
    except Exception as e:
        print(f"  [WARN] fred {sid}: {e}")
        return None


def _fetch_nyfed_sofr_percentiles(start: str = START_DATE) -> Optional[pd.DataFrame]:
    """Descarga percentiles SOFR del NY Fed via API publica.
    Ruta aislada, no afecta al pipeline productivo.
    """
    url = "https://markets.newyorkfed.org/api/rates/secured/sofr/search.json"
    params = {"startDate": start, "endDate": str(pd.Timestamp.now().date()), "type": "rate,volume"}
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        refs = r.json().get("refRates", [])
    except Exception as e:
        print(f"  [WARN] nyfed: {e}")
        return None

    def _num(x):
        if x is None or x == "NA" or x == "":
            return np.nan
        try:
            return float(x)
        except (ValueError, TypeError):
            return np.nan

    rows = []
    for ref in refs:
        rows.append({
            "date": pd.Timestamp(ref.get("effectiveDate")),
            "sofr": _num(ref.get("percentRate")),
            "p1": _num(ref.get("percentPercentile1")),
            "p25": _num(ref.get("percentPercentile25")),
            "p75": _num(ref.get("percentPercentile75")),
            "p99": _num(ref.get("percentPercentile99")),
            "vol_bn": _num(ref.get("volumeInBillions")),
        })
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df


# ===================================================================
# Guard de frescura propio
# ===================================================================

def _check_freshness(series_name: str, s: Optional[pd.Series]) -> dict:
    """Comprueba frescura de una serie. Devuelve dict con estado."""
    if s is None or len(s) == 0:
        return {"status": "UNAVAILABLE", "last_date": None, "age_days": None}
    last = s.dropna().index[-1] if len(s.dropna()) > 0 else None
    if last is None:
        return {"status": "UNAVAILABLE", "last_date": None, "age_days": None}
    age = (pd.Timestamp.now().normalize() - pd.Timestamp(last).normalize()).days
    max_age = FRESHNESS_MAX_DAYS.get(series_name, 7)
    status = "OK" if age <= max_age else "STALE"
    return {
        "status": status,
        "last_date": str(pd.Timestamp(last).date()),
        "age_days": int(age),
        "max_age_days": max_age,
    }


# ===================================================================
# Log JSONL
# ===================================================================

def _append_log(record: dict) -> None:
    """Append-only JSONL. Fallo silencioso (no debe tumbar el script)."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except (OSError, TypeError, ValueError) as e:
        print(f"  [WARN] log: {e}")


# ===================================================================
# Ejecucion
# ===================================================================

def main() -> int:
    t0 = time.time()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(f"liquidity_v2_shadow run_id={run_id}")

    # 1) Descargar SOFR (aislado)
    print("  Descargando SOFR (ruta aislada)...")
    sofr = _fetch_fred_series("SOFR")
    freshness_sofr = _check_freshness("SOFR", sofr)

    # 2) Descargar EFFR (aislado, NO via FRED_SERIES)
    print("  Descargando EFFR (ruta aislada)...")
    effr = _fetch_fred_series("EFFR")
    freshness_effr = _check_freshness("EFFR", effr)

    # 3) Descargar NY Fed percentiles
    print("  Descargando percentiles NY Fed...")
    nyfed = _fetch_nyfed_sofr_percentiles()
    if nyfed is not None and "p99" in nyfed.columns:
        p99 = nyfed["p99"].dropna()
        freshness_nyfed = _check_freshness("NYFED_P99", p99)
    else:
        p99 = None
        freshness_nyfed = {"status": "UNAVAILABLE", "last_date": None, "age_days": None}

    print(f"  SOFR: {freshness_sofr['status']} (ult={freshness_sofr['last_date']}, age={freshness_sofr['age_days']}d)")
    print(f"  EFFR: {freshness_effr['status']} (ult={freshness_effr['last_date']}, age={freshness_effr['age_days']}d)")
    print(f"  NYFed p99: {freshness_nyfed['status']} (ult={freshness_nyfed['last_date']}, age={freshness_nyfed['age_days']}d)")

    # 4) Estado global de frescura
    all_ok = all(
        st["status"] == "OK"
        for st in [freshness_sofr, freshness_effr, freshness_nyfed]
    )
    any_unavail = any(
        st["status"] == "UNAVAILABLE"
        for st in [freshness_sofr, freshness_effr, freshness_nyfed]
    )

    if sofr is None or effr is None:
        print("  [SKIP] SOFR o EFFR no disponibles. Se registra el intento.")
        record = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(time.time() - t0, 3),
            "status": "SKIP",
            "reason": "SOFR o EFFR no disponibles",
            "freshness": {
                "sofr": freshness_sofr,
                "effr": freshness_effr,
                "nyfed_p99": freshness_nyfed,
            },
        }
        _append_log(record)
        return 0

    # 5) Calcular C1 = p99 - EFFR (si disponible)
    if p99 is not None:
        effr_aligned = effr.reindex(p99.index.intersection(effr.index))
        c1_bp = (p99.reindex(effr_aligned.index) - effr_aligned) * 100.0
    else:
        c1_bp = None

    # 6) Evaluar serie v2
    try:
        df = evaluar_serie(sofr, effr, c1_bp)
    except Exception as e:
        print(f"  [ERROR] evaluar_serie: {e}")
        traceback.print_exc()
        record = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(time.time() - t0, 3),
            "status": "ERROR",
            "reason": str(e),
            "freshness": {
                "sofr": freshness_sofr,
                "effr": freshness_effr,
                "nyfed_p99": freshness_nyfed,
            },
        }
        _append_log(record)
        return 1

    # 7) Extraer ultimo dia util con datos
    last_row = df.dropna(subset=["diff_bp"]).iloc[-1]
    last_idx = df.dropna(subset=["diff_bp"]).index[-1]

    # Convertir tipos nullable a serializable
    def _nullable_bool(v):
        if pd.isna(v):
            return None
        return bool(v)

    record = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(time.time() - t0, 3),
        "status": "OK" if all_ok else ("PARTIAL" if not any_unavail else "DEGRADED"),
        "as_of_date": str(last_idx.date()),
        "observation": {
            "diff_bp": float(last_row["diff_bp"]),
            "severidad_cruda": str(last_row["severidad_cruda"]),
            "regimen_persistente": _nullable_bool(last_row["regimen_persistente"]),
            "alerta_diferencial_extremo": _nullable_bool(last_row["alerta_diferencial_extremo"]),
            "c1_tail_divergence_candidate": _nullable_bool(last_row["c1_tail_divergence_candidate"]),
            "calendar_context": str(last_row["calendar_context"]),
            "period_context": str(last_row["period_context"]),
        },
        "freshness": {
            "sofr": freshness_sofr,
            "effr": freshness_effr,
            "nyfed_p99": freshness_nyfed,
        },
    }

    _append_log(record)

    print(f"  as_of_date={record['as_of_date']}")
    print(f"  diff_bp={record['observation']['diff_bp']:.1f}")
    print(f"  severidad={record['observation']['severidad_cruda']}")
    print(f"  persistente={record['observation']['regimen_persistente']}")
    print(f"  alerta={record['observation']['alerta_diferencial_extremo']}")
    print(f"  c1_candidate={record['observation']['c1_tail_divergence_candidate']}")
    print(f"  status={record['status']}  duration={record['duration_s']}s")
    print(f"  Log: {LOG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
