"""Genera docs/auditoria/iae/ESTADO_SISTEMA.md.

Fuente unica de hechos verificables del sistema IAE. Se regenera
con:

    py scripts/generate_estado_sistema.py

NO se edita a mano. NO se ejecuta en el pipeline productivo (daily_run.yml).

Determinismo: el unico campo no reproducible es 'generado_en', que usa la
fecha del propio commit HEAD. La fecha real de ejecucion se registra como
metadata de generacion, no como dato del sistema.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'docs' / 'auditoria' / 'iae' / 'ESTADO_SISTEMA.md'

# Modulos IAE a auditar
IAE_ROOT = ROOT / 'src' / 'institutional_accumulation'
SCRIPTS_IAE = [
    'scripts/iae_pipeline.py',
    'scripts/build_catalog_csvs.py',
]
MAPPINGS_IAE = [
    'data/mappings/radar_target_catalog.csv',
    'data/mappings/cusip_ticker_exceptions.csv',
    'data/mappings/cusip_equivalence.csv',
]
EVIDENCE_DIRS = [
    'docs/auditoria/iae/evidence/nipc_gate0_baseline',
    'docs/auditoria/iae/evidence/nipc_gate0_target_identity_top2000',
    'docs/auditoria/iae/evidence/nipc_gate0_top2000_v2',
    'docs/auditoria/iae/evidence/a64_integration_b1_p61_p38',
    'docs/auditoria/iae/evidence/b2_pit_cierre',
    'docs/auditoria/iae/evidence/nipc_p70_probe',
]

def run(cmd):
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    return r.stdout.strip(), r.returncode

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

def sha256_file(p):
    return sha256_bytes(p.read_bytes())

def loc_file(p):
    return len(p.read_text(encoding='utf-8-sig', errors='replace').splitlines())
def section_git():
    lines = ['## 1. Git', '']
    head, _ = run(['git', 'rev-parse', 'HEAD'])
    head_short = head[:7] if head else 'N/D'
    lines.append(f'- **HEAD:** `{head_short}`')
    lines.append(f'- **HEAD completo:** `{head}`')
    date, _ = run(['git', 'log', '-1', '--format=%ad', '--date=iso'])
    subject, _ = run(['git', 'log', '-1', '--format=%s'])
    lines.append(f'- **Fecha commit HEAD:** {date}')
    lines.append(f'- **Asunto commit HEAD:** {subject}')
    ahead, _ = run(['git', 'rev-list', '--count', 'origin/main..HEAD'])
    behind, _ = run(['git', 'rev-list', '--count', 'HEAD..origin/main'])
    origin, _ = run(['git', 'rev-parse', '--short', 'origin/main'])
    lines.append(f'- **Ahead:** {ahead}')
    lines.append(f'- **Behind:** {behind}')
    lines.append(f'- **origin/main:** `{origin}`')
    status, _ = run(['git', 'status', '--porcelain'])
    if status:
        lines.append('- **Working tree:** MODIFICADO')
        for line in status.splitlines()[:10]:
            lines.append(f'  - `{line}`')
    else:
        lines.append('- **Working tree:** LIMPIO')
    lines.append('')
    return lines

def section_tests():
    lines = ['## 2. Tests (pytest real)', '']
    out, rc = run(['py', '-m', 'pytest', 'tests/', 'validation/', '-q', '--tb=no', '--no-header'])
    # Buscar linea resumen "N passed, M skipped..." o similar
    summary = None
    for line in out.splitlines():
        if 'passed' in line and ('failed' in line or 'skipped' in line):
            summary = line.strip()
    if summary:
        lines.append(f'- **Resumen:** {summary}')
    else:
        lines.append(f'- **Exit code:** {rc}')
    lines.append(f'- **Exit code:** {rc}')
    lines.append('')
    return lines

def section_integridad():
    lines = ['## 3. Integridad del codigo', '']
    _, rc_c = run(['py', '-m', 'compileall', '.', '-q'])
    lines.append(f'- **compileall:** {"OK" if rc_c == 0 else "FAIL"} (exit {rc_c})')
    out_p, rc_p = run(['py', '-m', 'pyflakes', '.'])
    if rc_p == 0:
        lines.append('- **pyflakes:** LIMPIO')
    else:
        lines.append(f'- **pyflakes:** {len(out_p.splitlines())} warnings (exit {rc_p})')
    lines.append('')
    return lines

def section_modulos():
    lines = ['## 4. Modulos IAE', '']
    if not IAE_ROOT.exists():
        lines.append(f'- **{IAE_ROOT.relative_to(ROOT)}:** NO EXISTE')
        lines.append('')
        return lines
    total_py = 0
    total_loc = 0
    lines.append('| Fichero | LOC |')
    lines.append('|---|---:|')
    for p in sorted(IAE_ROOT.rglob('*.py')):
        if '__pycache__' in p.parts:
            continue
        n = loc_file(p)
        rel = p.relative_to(ROOT).as_posix()
        lines.append(f'| `{rel}` | {n} |')
        total_py += 1
        total_loc += n
    lines.append(f'| **TOTAL** | **{total_py} ficheros, {total_loc} LOC** |')
    lines.append('')
    return lines

def section_datos():
    lines = ['## 5. Datos IAE', '']
    for rel in MAPPINGS_IAE:
        p = ROOT / rel
        if not p.exists():
            lines.append(f'- `{rel}`: NO EXISTE')
            continue
        b = p.read_bytes()
        h = sha256_bytes(b)
        n_lines = b.count(b'\n')
        lines.append(f'- `{rel}`')
        lines.append(f'  - sha256: `{h}`')
        lines.append(f'  - bytes: {len(b)}')
        lines.append(f'  - lineas: {n_lines}')
    lines.append('')
    return lines

def section_scripts():
    lines = ['## 6. Scripts IAE', '']
    for rel in SCRIPTS_IAE:
        p = ROOT / rel
        if p.exists():
            lines.append(f'- `{rel}`: {loc_file(p)} LOC')
        else:
            lines.append(f'- `{rel}`: NO EXISTE')
    lines.append('')
    return lines

def section_evidencia():
    lines = ['## 7. Evidencia empirica (directorios)', '']
    for rel in EVIDENCE_DIRS:
        p = ROOT / rel
        if not p.exists():
            lines.append(f'- `{rel}`: NO EXISTE')
            continue
        files = [f for f in p.rglob('*') if f.is_file()]
        lines.append(f'- `{rel}`: {len(files)} ficheros')
    lines.append('')
    return lines
def main():
    now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    head_date, _ = run(['git', 'log', '-1', '--format=%ad', '--date=short'])
    
    header = [
        '# IAE - ESTADO DEL SISTEMA',
        '',
        'Hechos verificables del sistema. Generado por script.',
        '',
        f'**Generado en:** {now_utc}',
        f'**Snapshot de:** commit HEAD (fecha: {head_date})',
        '**NO editar a mano.** Regenerar con:',
        '',
        '    py scripts/generate_estado_sistema.py',
        '',
        'Este fichero es complementario a `ESTADO_DECLARADO.md` (declaraciones',
        'subjetivas de fase, prohibiciones, deuda). Los demas documentos del',
        'proyecto describen su tema; NO declaran el estado del sistema.',
        '',
        '---',
        '',
    ]
    
    body = []
    body += section_git()
    body += section_tests()
    body += section_integridad()
    body += section_modulos()
    body += section_datos()
    body += section_scripts()
    body += section_evidencia()
    
    footer = [
        '---',
        '',
        ''Fin del estado generado. Fuente de verdad: `ESTADO_DECLARADO.md` (declaraciones)'',
        ''+ `ESTADO_SISTEMA.md` (este fichero, hechos).'',
        '',
    ]
    
    content = '\n'.join(header + body + footer)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(content, encoding='utf-8', newline='\n')
    print(f"OK escrito: {OUT.relative_to(ROOT)} ({len(content)} caracteres, {len(content.splitlines())} lineas)")

if __name__ == '__main__':
    sys.exit(main() or 0)