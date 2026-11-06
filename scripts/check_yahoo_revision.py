"""Check de revision retrospectiva de Yahoo (K-LSE-YAHOO-REVISION-01).

Ejecutado en daily_run.yml post-run.py, pre-commit. Compara el manifest
del working tree (recien escrito por run.py) contra el manifest en HEAD
(ultimo commit de main, del run anterior). Si el last_date es el mismo
pero el sha256 cambio, Yahoo reescribio el OHLC de la misma sesion: WARN.

No bloqueante: exit 0 siempre (continue-on-error: true en el YAML).
Antes vivia solo en health_check.yml, donde el working tree coincide con
HEAD (checkout limpio) y el check nunca podia disparar.

Con --with-issues (S30, Q10): abre/actualiza un issue con label
"yahoo-revision" cuando detecta WARN/FAIL, y lo cierra cuando vuelve
a OK. Requiere "issues: write" en el job (anadido a run-system en
daily_run.yml).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from health_check import (
    check_yahoo_revision,
    _run_gh,
    OK, WARN, FAIL, SKIP,
)


ISSUE_LABEL = "yahoo-revision"
ISSUE_DESCRIPTION = "Revision retrospectiva de Yahoo (K-LSE-YAHOO-REVISION-01)"
ISSUE_COLOR = "0e8a16"


def _ensure_label():
    subprocess.run(
        ["gh", "label", "create", ISSUE_LABEL, "--force",
         "--description", ISSUE_DESCRIPTION,
         "--color", ISSUE_COLOR],
        capture_output=True, text=True, timeout=15,
    )


def _find_open_issue():
    out = _run_gh([
        "issue", "list", "--label", ISSUE_LABEL, "--state", "open",
        "--json", "number", "--limit", "1",
    ])
    if not out:
        return None
    try:
        items = json.loads(out)
        return items[0]["number"] if items else None
    except Exception:
        return None


def _aggregate_status(results):
    """Agrega la lista de Result en un solo estado overall."""
    if any(r.status == FAIL for r in results):
        return FAIL
    if any(r.status == WARN for r in results):
        return WARN
    if all(r.status == SKIP for r in results):
        return SKIP
    return OK


def _render_markdown(results):
    lines = ["| Check | Estado | Detalle |", "|---|---|---|"]
    for r in results:
        lines.append(f"| {r.name} | [{r.status}] | {r.message} |")
    return "\n".join(lines)


def _update_issue(results):
    """Abre, actualiza o cierra el issue segun el estado agregado."""
    overall = _aggregate_status(results)
    if overall == SKIP:
        return
    _ensure_label()
    existing = _find_open_issue()
    if overall == OK:
        if existing:
            res = _run_gh(["issue", "close", str(existing),
                           "--comment", "Yahoo revision OK - sin reescritura."])
            if res is None:
                print(f"[yahoo] WARN: gh fallo al cerrar issue #{existing}")
            else:
                print(f"[yahoo] issue #{existing} cerrada (recuperado)")
        return
    body = _render_markdown(results)
    if overall == FAIL:
        title = "[yahoo-revision] FAIL - reescritura detectada"
    else:
        title = "[yahoo-revision] WARN - reescritura detectada"
    if existing:
        res_c = _run_gh(["issue", "comment", str(existing), "--body", body])
        res_e = _run_gh(["issue", "edit", str(existing), "--title", title])
        if res_c is None or res_e is None:
            print(f"[yahoo] WARN: gh fallo al actualizar issue #{existing}")
        else:
            print(f"[yahoo] issue #{existing} actualizada")
    else:
        r = subprocess.run(
            ["gh", "issue", "create", "--title", title, "--body", body,
             "--label", ISSUE_LABEL],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            print(f"[yahoo] issue creada: {r.stdout.strip()}")
        else:
            print(f"[yahoo] WARN: gh fallo al crear issue: {r.stderr.strip()}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-issues", action="store_true",
                    help="abre/cierra issues (requiere issues: write)")
    args = ap.parse_args(argv)
    results = check_yahoo_revision()
    for r in results:
        print(f"| {r.name} | [{r.status}] | {r.message} |")
    if args.with_issues:
        try:
            _update_issue(results)
        except Exception as e:
            print(f"[yahoo] WARN: no se pudo actualizar issue: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
