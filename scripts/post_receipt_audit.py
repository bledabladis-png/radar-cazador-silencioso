"""Auditoria post-receipt: receipts vs filas persistidas en CSV.

Ejecutado por daily_run.yml DESPUES de write_completion_receipt.py
(con if: always()). NO bloquea: exit 0 siempre.

Detecta casos donde un writer reporto written=True con date=D pero
el CSV correspondiente no tiene fila con esa fecha. Esto captura
exactamente el bug del 09-oct (receipt emitido sin la fila real).

Escribe append-only a outputs/history/receipt_audit.csv. Health check
lo lee los lunes y alerta si las ultimas 7 sesiones tienen MISSING.

Status por receipt:
- SKIP: written=False (skip legitimo: HOLIDAY_MODE, candado B5, etc)
- NO_DATE_CHECK: written=True pero date=None (snapshot sin fecha)
- OK: written=True con date=D, CSV tiene fila D y D == target_session
- MISSING: written=True con date=D, CSV no tiene fila D
- CSV_MISSING: written=True con date=D, CSV no existe
- LAGGED: written=True con date=D, CSV tiene fila D pero D != target_session
  (writer con lag de publicacion; caso sector_flow_characteristics)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RECEIPTS_PATH = ROOT / "outputs" / "state" / "writer_receipts.json"
AUDIT_CSV = ROOT / "outputs" / "history" / "receipt_audit.csv"
AUDIT_COLUMNS = ["target_session", "writer", "status", "receipt_date", "csv_path"]


def _load_target_session() -> str | None:
    """Deriva target_session del manifest de market_data (autoridad)."""
    for name in ("market_data", "stock_prices"):
        p = ROOT / "data" / f"{name}.parquet.manifest.json"
        if not p.exists():
            continue
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
            expected = m.get("quality", {}).get("expected_session")
            if expected:
                return expected
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    return None


def _receipts(payload: dict) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    items = payload.get("receipts")
    return items if isinstance(items, list) else []


def _has_row(path: Path, session: str) -> bool:
    """True si el CSV tiene fila con columna date == session."""
    try:
        df = pd.read_csv(path, usecols=["date"])
    except (FileNotFoundError, ValueError, OSError, pd.errors.ParserError):
        return False
    if "date" not in df.columns or df.empty:
        return False
    return bool((df["date"].astype(str).str[:10] == session).any())


def audit(receipts: list[dict], target_session: str | None) -> pd.DataFrame:
    rows = []
    for r in receipts:
        name = r.get("name", "")
        written = bool(r.get("written"))
        date = r.get("date")
        path_str = r.get("path") or ""
        path = Path(path_str) if path_str else None

        if not written:
            status = "SKIP"
        elif not date:
            status = "NO_DATE_CHECK"
        elif path is None or not path.exists():
            status = "CSV_MISSING"
        elif not _has_row(path, date):
            status = "MISSING"
        elif target_session and date != target_session:
            # REP-AUDIT-20261020-RECEIPT-AUDIT-LAG: el writer escribio una
            # fila con fecha valida, pero distinta al target_session. Es
            # lag de publicacion (R1 correcto en el writer), no bug.
            status = "LAGGED"
        else:
            status = "OK"

        rows.append({
            "target_session": target_session or "",
            "writer": name,
            "status": status,
            "receipt_date": date or "",
            "csv_path": path_str,
        })
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def _append_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        existing_keys = set(zip(existing["target_session"].astype(str),
                                existing["writer"].astype(str)))
        mask = ~df.apply(
            lambda r: (str(r["target_session"]), str(r["writer"])) in existing_keys,
            axis=1,
        )
        new = df.loc[mask]
    else:
        new = df
    if new.empty:
        print(f"[audit] sin nuevas filas para {path.name}")
        return
    header = not path.exists()
    new.to_csv(path, mode="a", header=header, index=False,
               encoding="utf-8", lineterminator="\n")
    print(f"[audit] +{len(new)} filas en {path.name}")


def main() -> int:
    if not RECEIPTS_PATH.exists():
        print(f"[audit] {RECEIPTS_PATH.name} no existe, no hay auditoria")
        return 0

    try:
        payload = json.loads(RECEIPTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"[audit] {RECEIPTS_PATH.name} ilegible: {e}")
        return 0

    receipts = _receipts(payload)
    if not receipts:
        print("[audit] writer_receipts.json sin receipts, no hay auditoria")
        return 0

    target = _load_target_session()
    df = audit(receipts, target)
    _append_csv(df, AUDIT_CSV)

    counts = df["status"].value_counts().to_dict()
    missing = counts.get("MISSING", 0) + counts.get("CSV_MISSING", 0)
    lagged = counts.get("LAGGED", 0)
    print(f"[audit] target={target or 'N/D'} writer={len(df)} {counts}")
    if lagged:
        print(f"[audit] INFO: {lagged} writer(s) con lag de publicacion "
              "(date != target_session, fila presente)")
    if missing:
        print(f"[audit] ATENCION: {missing} writer(s) con fila ausente. "
              "Detalle en " + str(AUDIT_CSV.relative_to(ROOT)))
        for _, row in df[df["status"].isin(("MISSING", "CSV_MISSING"))].iterrows():
            print(f"       - {row['writer']}: {row['status']} (date={row['receipt_date']})")
    else:
        print("[audit] sin anomalias")
    return 0


if __name__ == "__main__":
    sys.exit(main())