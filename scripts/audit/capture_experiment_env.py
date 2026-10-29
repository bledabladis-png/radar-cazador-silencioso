"""Captura el entorno de un experimento SOW (H-02).

Dictamen SOW-HOLDOUT-IMMUTABLE 2026-10-28 (H-02): el hash de los dos
scripts validadores no congela el experimento completo. Estos scripts
dependen de modulos, parametros, detectores y auxiliares. Un cambio
en detect_sow_v2 o en una dependencia puede cambiar el resultado sin
modificar el hash de los validadores. Ademas, registrar un hash en la
fecha de firma no demuestra que el pre-registro sea anterior a la
primera consulta de resultados.

Este script cierra H-02:

- Captura commit, rama, dirty/clean de git.
- Verifica que el commit del pre-registro es ancestro de HEAD
  (evidencia temporal: el pre-registro precede a la ejecucion).
- Congela SHA-256 de requirements.txt + lineas pinned + Python version.
- Congela SHA-256 de cada validador (glob purged_kfold_sow_*.py) y de
  cada modulo Wyckoff (glob indicators/wyckoff_*.py).
- Captura constantes WYCKOFF_SOW_* del modulo indicators.wyckoff_v1.
- Referencia el manifest actual de holdout (holdout_manifest.json):
  sha256, n_tickers, rango. Asi el experimento queda atado a un
  contenido concreto del holdout, no a "el parquet que hubiera".
- Semilla opcional (--seed).

Escribe data/mappings/experiment_manifests/<experiment_id>.json
(versionado, durable).

Exit 0: manifest escrito.
Exit 1: precondicion fallida (dirty, prereg no ancestro, holdout sin
        manifest). El experimento no debe ejecutarse.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "mappings" / "experiment_manifests"
HOLDOUT_MANIFEST = ROOT / "data" / "mappings" / "holdout_manifest.json"


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args):
    r = subprocess.run(
        ["git"] + list(args), cwd=str(ROOT),
        capture_output=True, text=True, timeout=15,
    )
    return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()


def _git_head():
    rc, out, _ = _git("rev-parse", "HEAD")
    if rc != 0:
        return None
    return out


def _git_branch():
    rc, out, _ = _git("rev-parse", "--abbrev-ref", "HEAD")
    return out if rc == 0 else None


def _git_is_dirty():
    rc, out, _ = _git("status", "--porcelain")
    if rc != 0:
        return None
    return bool(out)


def _git_head_date():
    rc, out, _ = _git("log", "-1", "--format=%cI", "HEAD")
    return out if rc == 0 else None


def _is_ancestor(commit):
    if not commit:
        return False
    rc, _, _ = _git("merge-base", "--is-ancestor", commit, "HEAD")
    return rc == 0


def _git_commit_exists(commit):
    if not commit:
        return False
    rc, _, _ = _git("cat-file", "-e", commit + "^{commit}")
    return rc == 0


def _glob_sha(paths):
    out = {}
    for p in sorted(paths):
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        out[rel] = _sha256_file(p)
    return out


def _sow_params():
    """Constantes WYCKOFF_SOW_* de config.settings.

    HANDOFF: "WYCKOFF_SOW_* = None" (candidata no activada en produccion).
    Viven en config/settings.py, no en indicators/wyckoff_v1.py.
    """
    try:
        import config.settings as cs
    except Exception as e:
        return {"_error": "no importable: " + str(e)}
    out = {}
    for name in dir(cs):
        if name.startswith("WYCKOFF_SOW_"):
            try:
                val = getattr(cs, name)
                if isinstance(val, (int, float, str, bool, type(None))):
                    out[name] = val
                else:
                    out[name] = repr(val)
            except Exception as e:
                out[name] = "_error: " + str(e)
    return out


def _validator_params(path):
    """Extrae parametros congelados de un validador via AST.

    Cada purged_kfold_sow_*.py declara:
      CANDIDATE = {"N": 40, "M": 5, "X_ATR": 1.00, "Y_VOL": 1.10, ...}
      EMBARGO = 50
      FOLDS = [...]
      H = 20
      PURGE = ...

    Parseamos sin importar el modulo (tiene side effects: lectura de
    parquets, prints, etc.). Solo valores literales. Si un valor no es
    literal (llamada a funcion, name), se guarda como repr corto.
    """
    import ast
    try:
        src = path.read_text(encoding="utf-8-sig", errors="replace")
        tree = ast.parse(src)
    except (OSError, SyntaxError) as e:
        return {"_error": "parse: " + str(e)}
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if not name.isupper():
            continue
        try:
            val = ast.literal_eval(node.value)
            if isinstance(val, (dict, list, tuple, set)):
                if isinstance(val, set):
                    val = sorted(val)
                out[name] = val
            elif isinstance(val, (int, float, str, bool, type(None))):
                out[name] = val
            else:
                out[name] = repr(val)
        except (ValueError, TypeError, SyntaxError):
            try:
                out[name] = ast.unparse(node.value)
            except Exception:
                out[name] = "<no-literal>"
    return out


def _requirements_meta():
    p = ROOT / "requirements.txt"
    if not p.exists():
        return {"exists": False}
    text = p.read_text(encoding="utf-8", errors="replace")
    pinned = [ln.strip() for ln in text.splitlines()
              if ln.strip() and "==" in ln.strip()]
    return {
        "exists": True,
        "sha256": _sha256_file(p),
        "pinned_count": len(pinned),
        "pinned": pinned,
    }


def _holdout_meta():
    if not HOLDOUT_MANIFEST.exists():
        return {"exists": False}
    try:
        d = json.loads(HOLDOUT_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {"exists": True, "error": str(e)}
    out = {"exists": True, "path": "data/mappings/holdout_manifest.json"}
    for key, entry in (d.get("holdouts") or {}).items():
        out[key] = {
            "sha256": entry.get("sha256"),
            "content_sha256": entry.get("content_sha256"),
            "tickers_sha256": entry.get("tickers_sha256"),
            "n_tickers": entry.get("n_tickers"),
            "date_min": entry.get("date_min"),
            "date_max": entry.get("date_max"),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Captura el entorno de un experimento SOW")
    ap.add_argument("--experiment-id", required=True,
                    help="identificador del experimento (ej. sow_v6_replay)")
    ap.add_argument("--candidate", required=True,
                    help="candidata evaluada (ej. C1-M5)")
    ap.add_argument("--prereg-commit", default=None,
                    help="SHA del commit del pre-registro congelado. Si se "
                         "pasa, se verifica que es ancestro de HEAD.")
    ap.add_argument("--seed", type=int, default=None,
                    help="semilla si el experimento la usa")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="permite capturar con working tree dirty. Solo para "
                         "tests. Un experimento confirmatorio debe ser clean.")
    ap.add_argument("--allow-missing-holdout", action="store_true",
                    help="permite capturar sin holdout_manifest.json. Solo "
                         "para tests.")
    ap.add_argument("--confirmatory", action="store_true",
                    help="modo confirmatorio. Exige working tree clean, "
                         "holdout verify OK y registra pip freeze efectivo. "
                         "Incompatible con --allow-dirty / --allow-missing-holdout.")
    args = ap.parse_args(argv)

    errores = []
    if args.confirmatory and args.allow_dirty:
        errores.append("--confirmatory incompatible con --allow-dirty")
    if args.confirmatory and args.allow_missing_holdout:
        errores.append("--confirmatory incompatible con --allow-missing-holdout")
    warnings = []

    head = _git_head()
    if head is None:
        errores.append("git rev-parse HEAD fallo (no es un repo git?)")
    branch = _git_branch()
    head_date = _git_head_date()
    dirty = _git_is_dirty()
    if dirty is None:
        errores.append("git status fallo")
    elif dirty and not args.allow_dirty:
        errores.append(
            "working tree dirty. Un experimento confirmatorio debe "
            "capturarse sobre un commit limpio. Commitear o stash antes."
        )
    elif dirty:
        warnings.append("working tree dirty (--allow-dirty)")

    # H-03 dictamen 2026-10-29: en modo confirmatorio se exige holdout
    # verify OK y se registra pip freeze efectivo.
    holdout_verify = {"checked": False, "ok": None, "detail": None}
    pip_freeze_list = []
    mode = "confirmatory" if args.confirmatory else "exploratory"

    if args.confirmatory and not errores:
        r = subprocess.run(
            [sys.executable, "scripts/audit/holdout_manifest.py", "--verify"],
            capture_output=True, text=True, timeout=120,
        )
        holdout_verify["checked"] = True
        holdout_verify["ok"] = (r.returncode == 0)
        holdout_verify["detail"] = (r.stdout or r.stderr)[:800]
        if r.returncode != 0:
            errores.append(
                "holdout_manifest.py --verify fallo en modo confirmatorio"
            )
        r2 = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=120,
        )
        if r2.returncode == 0:
            pip_freeze_list = [ln for ln in r2.stdout.splitlines() if ln.strip()]
        else:
            errores.append("pip freeze fallo en modo confirmatorio")

    prereg = {"commit": args.prereg_commit, "checked": False}
    if args.prereg_commit:
        prereg["checked"] = True
        prereg["exists"] = _git_commit_exists(args.prereg_commit)
        if not prereg["exists"]:
            errores.append("prereg-commit no existe: " + args.prereg_commit)
        else:
            prereg["is_ancestor_of_head"] = _is_ancestor(args.prereg_commit)
            if not prereg["is_ancestor_of_head"]:
                errores.append(
                    "prereg-commit NO es ancestro de HEAD. Evidencia "
                    "temporal invalida: el pre-registro debe preceder a "
                    "la ejecucion."
                )
    else:
        warnings.append("sin --prereg-commit: no se verifica la evidencia temporal")

    validator_paths = sorted((ROOT / "scripts").glob("purged_kfold_sow_*.py"))
    validators = _glob_sha(validator_paths)
    validator_params = {}
    for vp in validator_paths:
        rel = str(vp.relative_to(ROOT)).replace("\\", "/")
        validator_params[rel] = _validator_params(vp)
    if not validators:
        warnings.append("no se encontraron scripts/purged_kfold_sow_*.py")
    modules = _glob_sha((ROOT / "indicators").glob("wyckoff_*.py"))
    if not modules:
        warnings.append("no se encontraron indicators/wyckoff_*.py")

    holdout = _holdout_meta()
    if not holdout.get("exists") and not args.allow_missing_holdout:
        errores.append(
            "holdout_manifest.json no existe. Ejecutar "
            "scripts/audit/holdout_manifest.py antes del experimento."
        )
    elif not holdout.get("exists"):
        warnings.append("holdout_manifest.json ausente (--allow-missing-holdout)")

    manifest = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "candidate": args.candidate,
        "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": {
            "head_sha": head,
            "head_date": head_date,
            "branch": branch,
            "is_dirty": dirty,
        },
        "prereg": prereg,
        "python": {
            "version": sys.version.split()[0],
            "implementation": sys.implementation.name,
            "executable": sys.executable,
        },
        "requirements": _requirements_meta(),
        "validators": validators,
        "validator_params": validator_params,
        "modules": modules,
        "sow_params": _sow_params(),
        "seed": args.seed,
        "holdout": holdout,
        "mode": mode,
        "holdout_verify": holdout_verify,
        "pip_freeze": pip_freeze_list,
        "warnings": warnings,
        "errors": errores,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / (args.experiment_id + ".json")
    out_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="",
    )
    print("[OK] manifest: " + str(out_path))
    print("  experiment_id: " + args.experiment_id)
    print("  candidate: " + args.candidate)
    print("  head: " + (head or "?")[:12])
    print("  dirty: " + str(dirty))
    print("  validators: " + str(len(validators)))
    for vrel, vp in validator_params.items():
        n = len(vp) if isinstance(vp, dict) else 0
        cand = (vp.get("CANDIDATE") if isinstance(vp, dict) else None)
        print("    " + vrel.split("/")[-1] + ": " + str(n) + " params"
              + (", CANDIDATE=" + str(cand) if cand else ""))
    print("  modules: " + str(len(modules)))
    print("  holdout: " + ("presente" if holdout.get("exists") else "ausente"))
    if args.prereg_commit:
        print("  prereg: " + args.prereg_commit + " ancestro=" + str(
            prereg.get("is_ancestor_of_head")))
    print("  mode: " + mode)
    if holdout_verify["checked"]:
        print("  holdout_verify: " + ("OK" if holdout_verify["ok"] else "FAIL"))
    if pip_freeze_list:
        print("  pip_freeze: " + str(len(pip_freeze_list)) + " paquetes registrados")
    for w in warnings:
        print("  [WARN] " + w)

    if errores:
        print("")
        for e in errores:
            print("[ERROR] " + e)
        print("[FAIL] precondiciones. El experimento NO debe ejecutarse.")
        return 1

    print("[OK] entorno capturado. El experimento puede ejecutarse.")
    return 0


if __name__ == "__main__":
    sys.exit(main())