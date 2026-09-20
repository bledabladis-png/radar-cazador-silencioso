"""Micro-probe autorizado por dictamen F2.3: identidad radar via OpenFIGI.

Direccion: 242 tickers radar -> OpenFIGI TICKER/US -> FIGI/shareClassFIGI.
Cruce con los 113 hits CUSIP del probe previo (result.json) por shareClassFIGI.

Determinista. Sin datetime.now().
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\Macro_Sectorial")
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.institutional_accumulation.identity.openfigi_client import (
    extract_stable_identity,
    map_identifiers,
)

HERE = Path(__file__).parent
PREV_RESULT = ROOT / "docs" / "auditoria" / "evidence" / "nipc_gate0_openfigi" / "result.json"
OUT_RADAR = HERE / "result_radar.json"
OUT_MATCH = HERE / "match_report.json"


def load_radar_tickers() -> list[str]:
    sp = pd.read_parquet(ROOT / "data" / "stock_prices.parquet")
    cols = sp.columns.get_level_values(-1).unique()
    radar = sorted(
        c for c in cols
        if isinstance(c, str)
        and "." not in c
        and not c.startswith("^")
        and not c.endswith("=X")
    )
    return radar


def main() -> None:
    radar = load_radar_tickers()
    print(f"Radar tickers: {len(radar)}")

    raw = map_identifiers("TICKER", radar, exch_code="US")
    OUT_RADAR.write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8", newline="\n")
    print(f"OpenFIGI radar hits: {sum(1 for v in raw.values() if v['ok'])} / {len(raw)}")

    # Extraer identidad estable por ticker
    radar_identity = {}
    for tk, hit in raw.items():
        ident = extract_stable_identity(hit)
        if ident:
            radar_identity[tk] = ident

    # Cargar previo (CUSIP -> identidad)
    prev = json.load(open(PREV_RESULT, encoding="utf-8"))
    cusip_identity = {}
    for cusip, rows in prev["results"].items():
        if not isinstance(rows, list) or not rows:
            continue
        chosen = None
        for r in rows:
            if r.get("exchCode") == "US":
                chosen = r
                break
        if chosen is None:
            chosen = rows[0]
        scf = chosen.get("shareClassFIGI")
        if scf:
            cusip_identity[cusip] = {
                "figi": chosen.get("figi"),
                "share_class_figi": scf,
                "ticker": chosen.get("ticker"),
                "name": chosen.get("name"),
            }

    # Interseccion por shareClassFIGI
    radar_scf = {tk: ident["share_class_figi"] for tk, ident in radar_identity.items() if ident.get("share_class_figi")}
    cusip_scf = {cusip: ident["share_class_figi"] for cusip, ident in cusip_identity.items()}

    scf_to_radar = {scf: tk for tk, scf in radar_scf.items()}
    matches = []
    for cusip, scf in cusip_scf.items():
        if scf in scf_to_radar:
            matches.append({
                "cusip": cusip,
                "share_class_figi": scf,
                "radar_ticker": scf_to_radar[scf],
                "cusip_ticker_from_openfigi": cusip_identity[cusip]["ticker"],
            })

    report = {
        "radar_total": len(radar),
        "radar_with_figi": sum(1 for v in radar_identity.values() if v.get("figi")),
        "radar_with_share_class_figi": len(radar_scf),
        "cusip_prev_total": len(prev["results"]),
        "cusip_prev_with_scf": len(cusip_scf),
        "matches_by_share_class_figi": len(matches),
        "matches": matches,
    }
    OUT_MATCH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8", newline="\n")

    print()
    print("=== RESULTADO ===")
    for k, v in report.items():
        if k == "matches":
            continue
        print(f"  {k}: {v}")
    print()
    print("=== EJEMPLOS DE MATCH (max 10) ===")
    for m in matches[:10]:
        print(f"  {m['cusip']} <-> {m['radar_ticker']} (scf={m['share_class_figi']})")


if __name__ == "__main__":
    main()
