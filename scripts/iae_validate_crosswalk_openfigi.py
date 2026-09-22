"""A.3 v2 - Validacion positiva del crosswalk contra OpenFIGI.

v2: la v1 fallo con IncompleteRead porque un batch de 100 CUSIPs devuelve
~3.5MB y urllib no completa la lectura en 30s. Solucion: chunks de 25
desde el script + exchCode="US" para reducir el payload del response.
"""
from __future__ import annotations
import hashlib, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\Macro_Sectorial")
sys.path.insert(0, str(ROOT))

from src.institutional_accumulation.identity.openfigi_client import map_identifiers

MAPPINGS = ROOT / "data" / "mappings"
OUT_DIR = MAPPINGS / "openfigi_requeries"
CHUNK = 25


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git_head() -> str:
    try:
        r = subprocess.run(["git","rev-parse","HEAD"], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    date_tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slug = date_tag + "_crosswalk_243_v2"

    print("=" * 72)
    print("A.3 v2 - VALIDACION POSITIVA CROSSWALK vs OpenFIGI")
    print("=" * 72)

    cw = pd.read_csv(MAPPINGS / "cusip_to_radar_figi.csv", dtype=str)
    cusips = cw["cusip"].astype(str).str.strip().tolist()
    expected = {str(r["cusip"]).strip(): {
        "radar_ticker": str(r["radar_ticker"]).strip(),
        "share_class_figi_esperado": str(r["share_class_figi"]).strip(),
    } for _, r in cw.iterrows()}
    print("\nCUSIPs a validar: " + str(len(cusips)))

    api_key = os.environ.get("OPENFIGI_API_KEY", "").strip() or None
    print("API key presente: " + str(bool(api_key)))
    print("Chunk size: " + str(CHUNK) + "  -> requests aprox: "
          + str((len(cusips) + CHUNK - 1) // CHUNK))

    input_data = {
        "run_date": date_tag, "run_ts": ts, "git_head": git_head(),
        "id_type": "ID_CUSIP", "exch_code": "US", "chunk": CHUNK,
        "n_cusips": len(cusips), "cusips": cusips,
    }
    (OUT_DIR / (slug + "_input.json")).write_text(
        json.dumps(input_data, indent=2), encoding="utf-8")

    print("\nConsultando OpenFIGI en chunks de " + str(CHUNK) + "...")
    results: dict = {}
    for i in range(0, len(cusips), CHUNK):
        chunk = cusips[i:i+CHUNK]
        n = i // CHUNK + 1
        total = (len(cusips) + CHUNK - 1) // CHUNK
        print("  chunk " + str(n) + "/" + str(total) + " (" + str(len(chunk)) + " CUSIPs)...")
        try:
            r = map_identifiers("ID_CUSIP", chunk, exch_code="US", api_key=api_key)
            results.update(r)
        except Exception as e:
            print("    ERROR en chunk: " + str(type(e).__name__) + ": " + str(e))
            for c in chunk:
                results[c] = {"ok": False, "data": None,
                              "error": type(e).__name__ + ": " + str(e)}
        time.sleep(0.5)

    raw_serializable = {k: {"ok": v.get("ok"), "error": v.get("error"),
                            "data": v.get("data")} for k, v in results.items()}
    (OUT_DIR / (slug + "_raw.json")).write_text(
        json.dumps(raw_serializable, indent=2, default=str), encoding="utf-8")

    n_ok = 0; n_err = 0; n_match = 0; n_mismatch = 0; n_no_data = 0
    mismatches = []; sin_hit = []

    for c in cusips:
        v = results.get(c) or {}
        exp = expected[c]
        if not v.get("ok"):
            n_err += 1
            sin_hit.append({"cusip": c, "error": str(v.get("error"))[:120],
                            "radar_ticker": exp["radar_ticker"]})
            continue
        rows = v.get("data") or []
        if not rows:
            n_no_data += 1
            sin_hit.append({"cusip": c, "error": "sin data",
                            "radar_ticker": exp["radar_ticker"]})
            continue
        n_ok += 1
        chosen = next((r for r in rows if r.get("exchCode") == "US"), rows[0])
        scf = str(chosen.get("shareClassFIGI") or "").strip()
        if scf == exp["share_class_figi_esperado"]:
            n_match += 1
        else:
            n_mismatch += 1
            mismatches.append({
                "cusip": c, "radar_ticker": exp["radar_ticker"],
                "esperado": exp["share_class_figi_esperado"],
                "devuelto": scf,
                "ticker_openfigi": chosen.get("ticker"),
                "name_openfigi": chosen.get("name"),
            })

    lines = [
        "A.3 v2 - Validacion positiva crosswalk vs OpenFIGI",
        "Run: " + ts, "HEAD: " + git_head(),
        "API key presente: " + str(bool(api_key)),
        "Chunk size: " + str(CHUNK),
        "",
        "CUSIPs validados: " + str(len(cusips)),
        "  con respuesta valida:     " + str(n_ok),
        "  match shareClassFIGI:     " + str(n_match),
        "  mismatch shareClassFIGI:  " + str(n_mismatch),
        "  sin data:                 " + str(n_no_data),
        "  error API:                " + str(n_err),
        "",
    ]
    if mismatches:
        lines.append("MISMATCHES:")
        for m in mismatches:
            lines.append("  " + m["cusip"] + " (" + m["radar_ticker"] + "): esperado="
                         + m["esperado"] + " devuelto=" + m["devuelto"]
                         + " (" + str(m.get("ticker_openfigi")) + ")")
        lines.append("")
    if sin_hit:
        lines.append("SIN HIT (primeros 30):")
        for s in sin_hit[:30]:
            lines.append("  " + s["cusip"] + " (" + s["radar_ticker"] + "): " + s["error"])
        lines.append("")

    summary_path = OUT_DIR / (slug + "_summary.txt")
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    lines_h = []
    for fn in [slug + "_input.json", slug + "_raw.json", slug + "_summary.txt"]:
        p = OUT_DIR / fn
        if p.exists():
            lines_h.append(sha256(p) + "  " + fn)
    (OUT_DIR / "HASHES.txt").write_text(
        "# A.3 v2 - hashes SHA-256\n# Run: " + ts + "\n\n"
        + "\n".join(lines_h) + "\n", encoding="utf-8")

    print()
    print("\n".join(lines))
    print()
    print("Artefactos en: " + str(OUT_DIR))
    return 0 if (n_mismatch == 0 and n_ok > 0) else 1


if __name__ == "__main__":
    sys.exit(main())