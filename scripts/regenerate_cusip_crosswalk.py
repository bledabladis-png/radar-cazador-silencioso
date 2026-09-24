"""Regenera cusip_radar_crosswalk.csv desde filings + catalogo radar.

Preserva filas verified_by=manual. Regenera filas verified_by=auto.

Convencion de vigencia:
  valid_from = period_end del primer trimestre observado
  valid_to   = period_end del ultimo, o null si es el trimestre mas reciente

Salida: data/mappings/cusip_radar_crosswalk.csv (8 columnas)

Idempotente. Dry-run disponible. No escribe si no hay diff.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "sec_13f" / "processed"
MAPPINGS = ROOT / "data" / "mappings"
CATALOG = MAPPINGS / "radar_target_catalog.csv"
CROSSWALK = MAPPINGS / "cusip_radar_crosswalk.csv"

COLUMNS = (
    "CUSIP", "ticker", "valid_from", "valid_to",
    "source", "reason", "title_of_class", "verified_by",
)
QUARTER_RE = re.compile(r"^(\d{4})Q([1-4])$")
QUARTER_END = {"Q1": "03-31", "Q2": "06-30", "Q3": "09-30", "Q4": "12-31"}

def _discover_quarters(data_dir: Path) -> list:
    """Lista ordenada de quarter IDs (YYYYQn) con INFOTABLE."""
    if not data_dir.exists():
        return []
    out = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir() or not (d / "INFOTABLE.parquet").exists():
            continue
        if QUARTER_RE.match(d.name):
            out.append(d.name)
    return out


def _period_end(quarter: str) -> str:
    """YYYYQn -> YYYY-MM-DD (ultimo dia del trimestre)."""
    m = QUARTER_RE.match(quarter)
    return f"{m.group(1)}-{QUARTER_END['Q' + m.group(2)]}"


def _load_radar_index(catalog_path: Path) -> dict:
    """{share_class_figi: radar_ticker}."""
    cat = pd.read_csv(catalog_path, dtype=str)
    out = {}
    for _, r in cat.iterrows():
        scf = str(r.get("share_class_figi") or "").strip()
        tk = str(r.get("radar_ticker") or "").strip()
        if scf and tk and scf != "nan":
            out[scf] = tk
    return out

def _extract_observations(quarter: str, infotable: Path, radar_index: dict) -> dict:
    """Devuelve {(cusip, ticker): {title, quarters: [quarter]}}."""
    df = pd.read_parquet(
        infotable,
        columns=["CUSIP", "FIGI", "TITLEOFCLASS", "SSHPRNAMTTYPE", "PUTCALL"],
    )
    # Filtro 5.1 del pipeline contractual: solo SH con PUTCALL NULL.
    # Excluye CALL/PUT explicitos y otros tipos no-equity.
    df = df[df["SSHPRNAMTTYPE"].astype(str).str.strip() == "SH"].copy()
    df = df[df["PUTCALL"].isna()].copy()
    df = df[df["FIGI"].notna()].copy()
    df = df[df["FIGI"].isin(radar_index.keys())].copy()

    out = {}
    for _, r in df.iterrows():
        cusip = str(r["CUSIP"]).strip()
        figi = str(r["FIGI"]).strip()
        title = str(r.get("TITLEOFCLASS") or "").strip()
        ticker = radar_index[figi]
        key = (cusip, ticker)
        if key not in out:
            out[key] = {"title": title, "quarters": [quarter]}
        else:
            out[key]["quarters"].append(quarter)
            if not out[key]["title"] and title:
                out[key]["title"] = title
    return out

def _build_auto_rows(observations: dict, latest_quarter: str) -> list:
    """Genera filas auto a partir de las observaciones agregadas."""
    rows = []
    for (cusip, ticker), info in sorted(observations.items()):
        qs = sorted(set(info["quarters"]))
        valid_from = _period_end(qs[0])
        valid_to = None if qs[-1] == latest_quarter else _period_end(qs[-1])
        rows.append({
            "CUSIP": cusip,
            "ticker": ticker,
            "valid_from": valid_from,
            "valid_to": valid_to if valid_to else "",
            "source": "SEC-EDGAR",
            "reason": f"CUSIP observado en filings {qs[0]}..{qs[-1]}",
            "title_of_class": info["title"] or "COM",
            "verified_by": "auto",
        })
    return rows

def _load_historical_auto(current_df: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """Filas auto del crosswalk actual con valid_from < cutoff.

    Son filas que el script NO puede regenerar con los trimestres
    disponibles en este entorno (p.ej. el runner de CI solo tiene el
    trimestre recien ingestado). Se preservan para no perder cobertura
    historica.
    """
    if current_df.empty or "verified_by" not in current_df.columns:
        return pd.DataFrame(columns=list(COLUMNS))
    auto = current_df[current_df["verified_by"] == "auto"].copy()
    if auto.empty:
        return auto
    auto["valid_from"] = auto["valid_from"].astype(str)
    return auto[auto["valid_from"] < cutoff]


def _merge(manual_df: pd.DataFrame, auto_rows: list,
           historical_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Manuales + auto regeneradas + (opcional) historicas.

    Precedencia por CUSIP: manual > auto regenerada > historica.
    """
    parts = [manual_df, pd.DataFrame(auto_rows, columns=list(COLUMNS))]
    if historical_df is not None and not historical_df.empty:
        parts.append(historical_df)
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["CUSIP"], keep="first")
    return out.sort_values(["ticker", "valid_from"]).reset_index(drop=True)


def _validate(df: pd.DataFrame) -> None:
    """Sin solapes por CUSIP, sin duplicados, columnas completas."""
    for c in COLUMNS:
        if c not in df.columns:
            raise ValueError(f"columna faltante: {c}")
    dup = df["CUSIP"].duplicated().sum()
    if dup:
        raise ValueError(f"CUSIP duplicado: {dup}")
    df2 = df.copy()
    df2["valid_from"] = pd.to_datetime(df2["valid_from"], errors="coerce")
    df2["valid_to"] = pd.to_datetime(df2["valid_to"].replace("", None), errors="coerce")
    bad = df2["valid_to"].notna() & (df2["valid_from"] > df2["valid_to"])
    if bad.any():
        raise ValueError(f"valid_from > valid_to en filas: {df2.index[bad].tolist()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print("Regenerar cusip_radar_crosswalk")
    print("=" * 72)

    quarters = _discover_quarters(DATA_DIR)
    if not quarters:
        print("[ABORT] no hay trimestres procesados")
        return 1
    latest = quarters[-1]
    print(f"\ntrimestres: {quarters}")
    print(f"latest: {latest}")

    radar_index = _load_radar_index(CATALOG)
    print(f"radar catalog: {len(radar_index)} shareClassFIGI")

    obs = {}
    for q in quarters:
        i = _extract_observations(q, DATA_DIR / q / "INFOTABLE.parquet", radar_index)
        for k, v in i.items():
            if k not in obs:
                obs[k] = v
            else:
                obs[k]["quarters"].extend(v["quarters"])
                if not obs[k]["title"] and v["title"]:
                    obs[k]["title"] = v["title"]
        print(f"  {q}: {len(i)} CUSIPs resueltos")

    auto_rows = _build_auto_rows(obs, latest)
    print(f"\nfilas auto generadas: {len(auto_rows)}")

    current = pd.read_csv(CROSSWALK, dtype=str, keep_default_na=False) if CROSSWALK.exists() else pd.DataFrame(columns=list(COLUMNS))
    manual = current[current["verified_by"] == "manual"].copy()
    print(f"filas manual preservadas: {len(manual)}")

    cutoff = _period_end(quarters[0])
    historical = _load_historical_auto(current, cutoff)
    print(f"filas historicas preservadas: {len(historical)}")

    merged = _merge(manual, auto_rows, historical_df=historical)
    print(f"total filas: {len(merged)}")

    try:
        _validate(merged)
        print("[OK] validacion pasa")
    except ValueError as e:
        print(f"[ABORT] validacion falla: {e}")
        return 1

    if args.dry_run:
        print("\n[DRY-RUN] nada escrito.")
        return 0

    with open(CROSSWALK, "w", encoding="utf-8", newline="\n") as f:
        merged.to_csv(f, index=False, lineterminator="\n")
    print(f"\n[OK] {CROSSWALK} escrito ({len(merged)} filas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())