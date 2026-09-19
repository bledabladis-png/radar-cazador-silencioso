"""TARGET_UNIVERSE - resolver CUSIP_13F -> target membership.

Dado un conjunto de CUSIPs (observados en 13F),
resuelve cada uno via OpenFIGI y consulta el RADAR_TARGET_CATALOG
para decidir si pertenece al target.

Contrato (dictamen F2.3):
  - El catalogo radar es INPUT independiente. Ya existe
    (data/mappings/radar_target_catalog.csv).
  - El resolver NO es el crosswalk interno: es OpenFIGI.
  - Cruce por shareClassFIGI (identidad estable).
  - Devuelve un DataFrame con provenance por fila.

Columnas devueltas:
  cusip
  share_class_figi
  radar_ticker        (None si no pertenece al radar)
  target_membership   (bool)
  security_type
  market_sector
  exch_code
  source
  source_date
  status              (OK | NOT_IN_RADAR | NO_ID | ERROR)

No usa datetime.now(). source_date lo aporta el caller.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .openfigi_client import map_identifiers
from .radar_target_catalog import SOURCE_NAME

COLUMNS = (
    "cusip",
    "share_class_figi",
    "radar_ticker",
    "target_membership",
    "security_type",
    "market_sector",
    "exch_code",
    "source",
    "source_date",
    "status",
)


def load_catalog(path: Path) -> pd.DataFrame:
    """Carga radar_target_catalog.csv como DataFrame."""
    df = pd.read_csv(path, dtype=str)
    required = {"radar_ticker", "share_class_figi"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"catalog sin columnas: {missing}")
    return df


def build_scf_index(catalog_df: pd.DataFrame) -> dict:
    """Devuelve {share_class_figi: radar_ticker}."""
    out = {}
    for _, row in catalog_df.iterrows():
        scf = row.get("share_class_figi")
        tk = row.get("radar_ticker")
        if isinstance(scf, str) and scf and isinstance(tk, str) and tk:
            out[scf] = tk
    return out


def _pick_row(rows: list) -> dict | None:
    if not isinstance(rows, list) or not rows:
        return None
    for r in rows:
        if r.get("exchCode") == "US":
            return r
    return rows[0]


def resolve_cusips(
    cusips: list,
    catalog_df: pd.DataFrame,
    *,
    source_date: str,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Resuelve cada CUSIP contra OpenFIGI y contra el catalogo.

    Determinista dado el mismo set de CUSIPs + catalogo + respuestas API.
    """
    clean = sorted({str(c).strip() for c in cusips if c and str(c).strip()})
    scf_index = build_scf_index(catalog_df)
    raw = map_identifiers("ID_CUSIP", clean, exch_code="US", api_key=api_key)

    rows = []
    for cusip in clean:
        hit = raw.get(cusip, {"ok": False, "data": None, "error": "missing"})
        if not hit.get("ok"):
            err = (hit.get("error") or "").lower()
            status = "NO_ID" if ("no identifier" in err or "not found" in err) else "ERROR"
            rows.append({
                "cusip": cusip,
                "share_class_figi": None,
                "radar_ticker": None,
                "target_membership": False,
                "security_type": None,
                "market_sector": None,
                "exch_code": None,
                "source": SOURCE_NAME,
                "source_date": source_date,
                "status": status,
            })
            continue

        row = _pick_row(hit.get("data"))
        if row is None:
            rows.append({
                "cusip": cusip,
                "share_class_figi": None,
                "radar_ticker": None,
                "target_membership": False,
                "security_type": None,
                "market_sector": None,
                "exch_code": None,
                "source": SOURCE_NAME,
                "source_date": source_date,
                "status": "ERROR",
            })
            continue

        scf = row.get("shareClassFIGI")
        radar_tk = scf_index.get(scf) if scf else None
        rows.append({
            "cusip": cusip,
            "share_class_figi": scf,
            "radar_ticker": radar_tk,
            "target_membership": bool(radar_tk),
            "security_type": row.get("securityType"),
            "market_sector": row.get("marketSector"),
            "exch_code": row.get("exchCode"),
            "source": SOURCE_NAME,
            "source_date": source_date,
            "status": "OK" if radar_tk else "NOT_IN_RADAR",
        })

    return pd.DataFrame(rows, columns=list(COLUMNS))


def membership_summary(df: pd.DataFrame) -> dict:
    """Contadores por status + % target_membership."""
    n = len(df)
    if n == 0:
        return {
            "n_total": 0,
            "n_target": 0,
            "n_not_in_radar": 0,
            "n_no_id": 0,
            "n_error": 0,
            "pct_target": 0.0,
        }
    n_target = int(df["target_membership"].sum())
    statuses = df["status"].value_counts().to_dict()
    return {
        "n_total": int(n),
        "n_target": n_target,
        "n_not_in_radar": int(statuses.get("NOT_IN_RADAR", 0)),
        "n_no_id": int(statuses.get("NO_ID", 0)),
        "n_error": int(statuses.get("ERROR", 0)),
        "pct_target": round(float(100.0 * n_target / n), 4),
    }
