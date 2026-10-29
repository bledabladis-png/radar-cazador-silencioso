"""Registro de experimentos por holdout (H-05, H-06 dictamen doc 76).

Fuente operativa: data/mappings/holdout_registry.json (versionado).
Estructura por holdout:
  {
    "cohort": {
      "history": [
        {
          "candidata": "SOW-V6",
          "fecha_consumo": "2026-10-06",
          "informe": "docs/auditoria/wyckoff/51-56",
          "event_type": "EXPLORATORY_DESIGN_INFLUENCE",
          "experiment_manifest": null,
          "identity_status": "UNVERIFIABLE_PRE_MANIFEST",
          "manifest_sha256_en_consumo": null
        }
      ],
      "state": "EXPLORATORY_CONSUMED"
    },
    "r3000": { ... }
  }

Estados derivados (H-05):
  SEALED                       lista vacia
  EXPLORATORY_CONSUMED         solo EXPLORATORY_DESIGN_INFLUENCE en historial
  CONFIRMATORY_CONSUMED        al menos 1 CONFIRMATORY_EXECUTION
  RETIRED_OR_SUPERSEDED        marcas explicitas

`event_type` (H-06) distingue:
  CONFIRMATORY_EXECUTION       ejecucion de protocolo congelado
  EXPLORATORY_DESIGN_INFLUENCE evidencia historica usada para disenar
                               una candidata. NO cuenta como validacion
                               confirmatoria.

El JSON incluye `independence_note` global (H-07): los dos universos
comparten 266/268 tickers. No pueden contabilizarse como dos
confirmaciones independientes.

Uso:
  py scripts/audit/holdout_registry.py                    # muestra estado
  py scripts/audit/holdout_registry.py --verify           # valida estructura
  py scripts/audit/holdout_registry.py --add <holdout> <json>
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "data" / "mappings" / "holdout_registry.json"

VALID_EVENT_TYPES = frozenset({
    "CONFIRMATORY_EXECUTION",
    "EXPLORATORY_DESIGN_INFLUENCE",
})
VALID_IDENTITY_STATUS = frozenset({
    "UNVERIFIABLE_PRE_MANIFEST",
    "VERIFIED",
    "NOT_APPLICABLE",
})
HOLDOUTS = ("cohort", "r3000")


def _empty_registry():
    return {
        "schema_version": 1,
        "independence_note": (
            "Los universos cohort y r3000 comparten 266 de los 268 tickers "
            "de cohort (interseccion calculada por holdout_intersection.py). "
            "Sus resultados NO pueden contabilizarse como dos confirmaciones "
            "independientes ni combinarse como observaciones independientes "
            "sin tratamiento de la poblacion compartida. Ver doc 76 H-07."
        ),
        "holdouts": {h: {"history": [], "state": "SEALED"} for h in HOLDOUTS},
    }


def load_registry():
    if not REGISTRY_PATH.exists():
        return _empty_registry()
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def save_registry(reg):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps(reg, indent=2, ensure_ascii=False) + chr(10),
        encoding="utf-8",
    )


def derive_state(history):
    """Estado derivado de una lista de entradas."""
    if not history:
        return "SEALED"
    for e in history:
        if e.get("state_override") == "RETIRED_OR_SUPERSEDED":
            return "RETIRED_OR_SUPERSEDED"
    if any(e.get("event_type") == "CONFIRMATORY_EXECUTION" for e in history):
        return "CONFIRMATORY_CONSUMED"
    return "EXPLORATORY_CONSUMED"


def validate_entry(entry):
    """Devuelve lista de errores para una entrada."""
    errs = []
    for f in ("candidata", "fecha_consumo", "informe",
              "event_type", "experiment_manifest", "identity_status"):
        if f not in entry:
            errs.append("falta campo: " + f)
    if entry.get("event_type") not in VALID_EVENT_TYPES:
        errs.append(
            "event_type invalido: " + repr(entry.get("event_type"))
            + " (validos: " + ", ".join(sorted(VALID_EVENT_TYPES)) + ")"
        )
    if entry.get("identity_status") not in VALID_IDENTITY_STATUS:
        errs.append(
            "identity_status invalido: " + repr(entry.get("identity_status"))
        )
    # EXPLORATORY_DESIGN_INFLUENCE nunca debe tener experiment_manifest
    # confirmatorio, pero si lo tiene, la entrada es sospechosa
    if (entry.get("event_type") == "EXPLORATORY_DESIGN_INFLUENCE"
            and entry.get("experiment_manifest") is not None):
        errs.append(
            "EXPLORATORY_DESIGN_INFLUENCE no debe tener experiment_manifest "
            "(no es una ejecucion confirmatoria)"
        )
    # CONFIRMATORY_EXECUTION debe tener experiment_manifest o identidad explicita
    if (entry.get("event_type") == "CONFIRMATORY_EXECUTION"
            and entry.get("experiment_manifest") is None
            and entry.get("identity_status") != "UNVERIFIABLE_PRE_MANIFEST"):
        errs.append(
            "CONFIRMATORY_EXECUTION sin experiment_manifest requiere "
            "identity_status=UNVERIFIABLE_PRE_MANIFEST"
        )
    return errs


def add_event(holdout, entry):
    """Anade una entrada al historial del holdout."""
    if holdout not in HOLDOUTS:
        raise ValueError("holdout invalido: " + holdout)
    errs = validate_entry(entry)
    if errs:
        raise ValueError("entrada invalida: " + "; ".join(errs))
    reg = load_registry()
    reg["holdouts"][holdout]["history"].append(entry)
    reg["holdouts"][holdout]["state"] = derive_state(
        reg["holdouts"][holdout]["history"]
    )
    save_registry(reg)
    return reg["holdouts"][holdout]


def verify_registry(reg):
    """Valida la estructura completa. Devuelve lista de errores."""
    errs = []
    if reg.get("schema_version") != 1:
        errs.append("schema_version != 1")
    if "independence_note" not in reg:
        errs.append("falta independence_note (H-07)")
    holdouts = reg.get("holdouts", {})
    for h in HOLDOUTS:
        if h not in holdouts:
            errs.append("holdout ausente: " + h)
            continue
        info = holdouts[h]
        if "history" not in info:
            errs.append(h + ": falta history")
            continue
        for i, e in enumerate(info["history"]):
            for err in validate_entry(e):
                errs.append(f"{h}[{i}]: {err}")
        expected = derive_state(info["history"])
        if info.get("state") != expected:
            errs.append(
                f"{h}.state={info.get('state')!r} != derivado {expected!r}"
            )
    return errs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--add", nargs=2, metavar=("HOLDOUT", "JSON"),
                    help="anade entrada (JSON inline)")
    args = ap.parse_args(argv)
    if args.add:
        holdout, jtxt = args.add
        try:
            entry = json.loads(jtxt)
        except json.JSONDecodeError as e:
            print("[FAIL] JSON invalido: " + str(e))
            return 1
        try:
            info = add_event(holdout, entry)
        except ValueError as e:
            print("[FAIL] " + str(e))
            return 1
        print("[OK] " + holdout + " -> " + info["state"])
        return 0
    reg = load_registry()
    errs = verify_registry(reg)
    if errs:
        print("[FAIL] " + str(len(errs)) + " error(es):")
        for e in errs:
            print("  " + e)
        return 1
    if args.verify:
        print("[OK] registry valido")
        return 0
    print("=== holdout_registry ===")
    print("independence_note: " + reg.get("independence_note", "")[:100] + "...")
    for h in HOLDOUTS:
        info = reg.get("holdouts", {}).get(h, {})
        print(f"{h}: state={info.get('state')} n_events={len(info.get('history', []))}")
        for e in info.get("history", []):
            print(f"  - {e.get('candidata')} {e.get('fecha_consumo')} "
                  f"[{e.get('event_type')}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())