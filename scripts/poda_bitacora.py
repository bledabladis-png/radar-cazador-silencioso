"""Poda de 05_BITACORA.md segun regla: mantener las ultimas N
sesiones + todas las del dia en curso.

Sesion = fecha unica. Una jornada larga con varias entradas del
mismo dia cuenta como 1 sesion.

Nota: 05_BITACORA ordena las sesiones de MAS RECIENTE a MAS ANTIGUA.

Modos:
    --check     valida retencion, exit 1 si hay que podar (NO modifica)
    (default)   dry-run: muestra que haria, exit 0
    --apply     aplica la poda

Uso:
    py scripts/poda_bitacora.py --check
    py scripts/poda_bitacora.py
    py scripts/poda_bitacora.py --apply
    py scripts/poda_bitacora.py --keep 5
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BITACORA = ROOT / "Consolidacion_Documentos" / "05_BITACORA.md"
ENTRY_RE = re.compile(r"^### (\d{4}-\d{2}-\d{2})")


def load(path=None):
    p = path or BITACORA
    return p.read_bytes().decode("utf-8").splitlines(keepends=True)


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
    """[(start, end, fecha_iso)] - una entrada por heading."""
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


def agrupar_por_fecha(ranges):
    """Agrupa headings consecutivos con la misma fecha.

    Devuelve [(primera_linea, ultima_linea, fecha, [indices_headings])]
    ordenado como vienen (mas reciente primero).
    """
    grupos = []
    for idx, (s, e, d) in enumerate(ranges):
        if grupos and grupos[-1][2] == d:
            # mismo dia: extender el grupo
            prev = grupos[-1]
            grupos[-1] = (prev[0], e, d, prev[3] + [idx])
        else:
            grupos.append((s, e, d, [idx]))
    return grupos


def contar_sesiones(ranges):
    """Cuenta sesiones unicas por fecha."""
    return len(set(d for _, _, d in ranges))


def decide(ranges, keep_n):
    """Conservar: las primeras keep_n sesiones (por fecha) FUERA del dia
    en curso + todas las del dia en curso (aparte, no consumen cupo).
    Regla 5 de 05_BITACORA.md: "mantiene las ultimas 10 sesiones. Ademas,
    siempre conserva las del dia en curso, aunque superen 10".
    Devuelve (indices_headings_a_conservar, indices_headings_a_eliminar).
    """
    if not ranges:
        return [], []
    grupos = agrupar_por_fecha(ranges)
    conservar = {0}  # dia en curso siempre
    count = 0
    for i in range(1, len(grupos)):
        if count >= keep_n:
            break
        conservar.add(i)
        count += 1
    keep = sorted(h for i in conservar for h in grupos[i][3])
    remove = [i for i in range(len(ranges)) if i not in keep]
    return keep, remove


def _hash(path=None):
    p = path or BITACORA
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cmd_check(ranges, keep_n):
    sesiones = contar_sesiones(ranges)
    keep, remove = decide(ranges, keep_n)
    if not remove:
        print("[OK] Bitacora dentro de la regla "
              "({0} sesiones unicas, keep={1} + dia en curso)".format(
                  sesiones, keep_n))
        return 0
    print("[FAIL] Bitacora excede la retencion. "
          "{0} heading(s) a podar (de {1} sesiones unicas):".format(
              len(remove), sesiones))
    for i in remove:
        _, _, d = ranges[i]
        print("  - {0}".format(d))
    print("Para podar: py scripts/poda_bitacora.py --apply")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="aplica la poda (default: dry-run)")
    ap.add_argument("--check", action="store_true",
                    help="solo valida retencion, exit 1 si hay que podar")
    ap.add_argument("--keep", type=int, default=10,
                    help="numero de sesiones a conservar (default: 10)")
    args = ap.parse_args()

    lines = load()
    s3, e3 = find_section3(lines)
    ranges = find_entries(lines, s3, e3)
    keep, remove = decide(ranges, args.keep)

    if args.check:
        return cmd_check(ranges, args.keep)

    sesiones = contar_sesiones(ranges)
    print("Bitacora: {0} headings, {1} sesiones unicas. "
          "Regla: keep={2} sesiones + dia en curso.".format(
              len(ranges), sesiones, args.keep))
    print("Conservar: {0} headings | Eliminar: {1} headings".format(
        len(keep), len(remove)))
    print()
    if not remove:
        print("Nada que podar.")
        return 0

    print("A eliminar (mas antiguos):")
    for i in remove:
        s, e, d = ranges[i]
        print("  - {0} (lineas {1}-{2})".format(d, s + 1, e))
    print()

    if not args.apply:
        print("[DRY-RUN] No se ha modificado el fichero.")
        print("Para aplicar: py scripts/poda_bitacora.py --apply")
        return 0

    hash_antes = _hash()
    first = ranges[0][0]
    preamble = lines[s3:first]
    last_end = ranges[-1][1]
    suffix = lines[last_end:e3]
    kept = []
    for i in keep:
        s, e, _ = ranges[i]
        kept.extend(lines[s:e])
    new_lines = lines[:s3] + preamble + kept + suffix + lines[e3:]
    BITACORA.write_bytes("".join(new_lines).encode("utf-8"))
    print("Aplicado. {0} headings eliminados.".format(len(remove)))
    print("Nuevo tamano: {0} bytes".format(BITACORA.stat().st_size))
    print("Hash antes:  {0}".format(hash_antes[:16]))
    print("Hash despues: {0}".format(_hash()[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())