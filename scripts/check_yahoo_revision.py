"""Check de revision retrospectiva de Yahoo (K-LSE-YAHOO-REVISION-01).

Ejecutado en daily_run.yml post-run.py, pre-commit. Compara el manifest
del working tree (recien escrito por run.py) contra el manifest en HEAD
(ultimo commit de main, del run anterior). Si el last_date es el mismo
pero el sha256 cambio, Yahoo reescribio el OHLC de la misma sesion: WARN.

No bloqueante: exit 0 siempre (continue-on-error: true en el YAML).
Antes vivia solo en health_check.yml, donde el working tree coincide con
HEAD (checkout limpio) y el check nunca podia disparar.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from health_check import check_yahoo_revision


def main() -> int:
    results = check_yahoo_revision()
    for r in results:
        print(f"| {r.name} | [{r.status}] | {r.message} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
