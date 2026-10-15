"""Poda de 05_BITACORA.md segun regla §5: mantener las ultimas N
sesiones + todas las del dia en curso.

Nota: 05_BITACORA ordena las sesiones de MAS RECIENTE a MAS ANTIGUA
(§2 del propio fichero). El script respeta ese orden.

Uso:
    py scripts/poda_bitacora.py            # dry-run (no modifica)
    py scripts/poda_bitacora.py --apply    # aplica
    py scripts/poda_bitacora.py --keep 5   # N=5 en vez de 10
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BITACORA = ROOT / "Consolidacion_Documentos" / "05_BITACORA.md"
ENTRY_RE = re.compile(r"^### (\d{4}-\d{2}-\d{2})")


def load():
    raw = BITACORA.read_bytes().decode("utf-8")
    return raw.splitlines(keepends=True)


def find_section3(lines):
    sec3_start = sec3_end = None
    for i, ln in enumerate(lines):
        if sec3_start is None and re.match(r"^## 3\.", ln):
            sec3_start = i
            continue
        if sec3_start is not None and re.match(r"^## \d+\.", ln):
            sec3_end = i
            break
    if sec3_start is None or sec3_end is None:
        raise SystemExit("No se encontro seccion 3 en bitacora")
    return sec3_start, sec3_end


def find_entries(lines, start, end):
    starts = []
    for i in range(start, end):
        m = ENTRY_RE.match(lines[i])
        if m:
            starts.append((i, m.group(1)))
    ranges = []
    for j, (s, d) in enumerate(starts):
        e = starts[j + 1][0] if j + 1 < len(starts) else end
        ranges.append((s, e, d))
    return ranges


def decide(ranges, keep_n):
    # 05_BITACORA ordena las sesiones de MAS RECIENTE a MAS ANTIGUA
    # (regla §2: "Al cerrar una sesion nueva, se anade arriba").
    # Conservar: las primeras keep_n + la mas reciente (ranges[0]).
    if not ranges:
        return [], []
    newest = ranges[0][2]
    keep = set()
    for i in range(min(keep_n, len(ranges))):
        keep.add(i)
    for i, (_, _, d) in enumerate(ranges):
        if d == newest:
            keep.add(i)
    keep_sorted = sorted(keep)
    remove = [i for i in range(len(ranges)) if i not in keep]
    return keep_sorted, remove


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="aplica la poda (default: dry-run)")
    ap.add_argument("--keep", type=int, default=10,
                    help="numero de sesiones a conservar (default: 10)")
    args = ap.parse_args()

    lines = load()
    s3, e3 = find_section3(lines)
    ranges = find_entries(lines, s3, e3)
    keep, remove = decide(ranges, args.keep)

    print(f"Bitacora: {len(ranges)} entradas. Regla: keep={args.keep} + dia en curso.")
    print(f"Conservar: {len(keep)} | Eliminar: {len(remove)}")
    print()

    if not remove:
        print("Nada que podar.")
        return 0

    print("A eliminar (mas antiguas):")
    for i in remove:
        s, e, d = ranges[i]
        print(f"  - {d} (lineas {s+1}-{e})")
    print()

    if not args.apply:
        print("[DRY-RUN] No se ha modificado el fichero.")
        print("Para aplicar: py scripts/poda_bitacora.py --apply")
        return 0

    first = ranges[0][0]
    preamble = lines[s3:first]
    last_end = ranges[-1][1]
    suffix = lines[last_end:e3]
    kept_lines = []
    for i in keep:
        s, e, d = ranges[i]
        kept_lines.extend(lines[s:e])
    new_s3 = preamble + kept_lines + suffix
    new_lines = lines[:s3] + new_s3 + lines[e3:]

    BITACORA.write_bytes("".join(new_lines).encode("utf-8"))
    print(f"Aplicado. {len(remove)} entradas eliminadas.")
    print(f"Nuevo tamano: {BITACORA.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
