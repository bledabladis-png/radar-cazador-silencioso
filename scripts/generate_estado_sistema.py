"""Genera Consolidacion_Documentos/ESTADO_SISTEMA.md.

Fuente unica de hechos verificables del sistema IAE. Se regenera
con:

    py scripts/generate_estado_sistema.py

NO se edita a mano. NO se ejecuta en el pipeline productivo (daily_run.yml).

Determinismo: el fichero describe solo hechos del arbol de trabajo.
No incluye fecha de generacion, HEAD, ahead/behind ni origin/main:
cualquier campo que dependa del commit que lo contiene queda stale
al commitear (el commit cambia HEAD). Para cuando se genero, ver
git log -1 -- Consolidacion_Documentos/ESTADO_SISTEMA.md.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'Consolidacion_Documentos' / 'ESTADO_SISTEMA.md'

# Modulos IAE a auditar
IAE_ROOT = ROOT / 'src' / 'institutional_accumulation'
SCRIPTS_IAE = [
    'scripts/iae_pipeline.py',
    'scripts/build_catalog_csvs.py',
    'scripts/iae_reconciliation_b1.py',
    'scripts/iae_validate_crosswalk_openfigi.py',
    'scripts/iae_test_census.py',
    'scripts/iae_contractual_coverage.py',
    'scripts/iae_coverage.py',
    'scripts/iae_identity_uniqueness_audit.py',
    'scripts/iae_contractual_nipc_e2e.py',
    # Fase G (2026-09-24): automatizacion de mappings del IAE
    'scripts/regenerate_radar_catalog.py',
    'scripts/regenerate_cusip_crosswalk.py',
    'scripts/update_sec_13f.py',
    'scripts/download_official_list_13f.py',
]
MAPPINGS_IAE = [
    'data/mappings/radar_target_catalog.csv',
    'data/mappings/cusip_radar_crosswalk.csv',
    'data/mappings/cusip_equivalence.csv',
]
# EVIDENCE_DIRS se descubre dinamicamente en section_evidencia() a partir
# de docs/auditoria/iae/evidence/*/. Antes era una lista hardcodeada que
# se desincronizo: incluia un directorio inexistente (nipc_p70_probe) y
# omitia tres reales. El recorrido dinamico evita esa clase de drift.
EVIDENCE_ROOT = ROOT / 'docs' / 'auditoria' / 'iae' / 'evidence'

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
    # Los campos derivados del HEAD (hash, ahead, behind, origin/main)
    # son autorreferenciales: el commit que contiene este fichero cambia
    # HEAD, asi que cualquier valor concreto queda obsoleto al commitear.
    # Se consultan con `git log` y `git status` al leer. La numeracion
    # de la seccion se conserva por compatibilidad con referencias
    # externas a "seccion 1".
    lines = ['## 1. Git', '']
    lines.append('Seccion reservada. Los campos derivados del HEAD se consultan')
    lines.append('con `git log` y `git status`. Este fichero no los incluye')
    lines.append('porque el commit que lo contiene cambia HEAD.')
    lines.append('')
    return lines

def section_tests():
    lines = ['## 2. Tests (pytest real)', '']
    out, rc = run(['py', '-m', 'pytest', 'tests/', 'validation/', '-q', '--tb=no', '--no-header'])
    # Resumen
    summary = None
    for line in out.splitlines():
        if 'passed' in line and ('failed' in line or 'skipped' in line):
            summary = line.strip()
    if summary:
        # El wall-clock de pytest no es reproducible entre runs.
        # Nos quedamos solo con los conteos (passed/failed/skipped).
        summary_counts = summary.split(' in ')[0]
        lines.append(f'- **Resumen:** {summary_counts}')
    lines.append(f'- **Exit code:** {rc}')
    # Tests fallidos con nombre completo
    failed = [l.strip() for l in out.splitlines() if l.strip().startswith('FAILED ')]
    if failed:
        lines.append('')
        lines.append('**Tests fallidos (nombre completo):**')
        for f in failed:
            lines.append(f'- `{f}`')
    # Tests skip (nombres)
    skipped = [l.strip() for l in out.splitlines() if l.strip().startswith('SKIPPED ')]
    if skipped:
        lines.append('')
        lines.append('**Tests skip:**')
        for s in skipped:
            lines.append(f'- `{s}`')
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
    if not EVIDENCE_ROOT.exists():
        lines.append(f'- `{EVIDENCE_ROOT.relative_to(ROOT)}`: NO EXISTE')
        lines.append('')
        return lines
    dirs = sorted(d for d in EVIDENCE_ROOT.iterdir() if d.is_dir())
    if not dirs:
        lines.append('(sin subdirectorios)')
        lines.append('')
        return lines
    for d in dirs:
        rel = d.relative_to(ROOT).as_posix()
        files = [f for f in d.rglob('*') if f.is_file()]
        lines.append(f'- `{rel}`: {len(files)} ficheros')
    lines.append('')
    return lines
def main():    
    header = [
        '# IAE - ESTADO DEL SISTEMA',
        '',
        'Hechos verificables del sistema. Generado por script.',
        '',
        '**Fuente:** arbol de trabajo en HEAD.',
        '**NO editar a mano.** Regenerar con:',
        '',
        '    py scripts/generate_estado_sistema.py',
        '',
        'Snapshot regenerable del sistema. Los documentos del corpus',
        'consolidado (Consolidacion_Documentos/00-06) describen su tema;',
        'NO declaran el estado. Este fichero es la fuente autoritativa de',
        'tests, integridad del codigo, modulos IAE, mappings y evidencia.',
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
        'Fin del estado generado. Fuente autoritativa: este fichero.',
        '',
    ]
    
    content = '\n'.join(header + body + footer)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(content, encoding='utf-8', newline='\n')
    print(f"OK escrito: {OUT.relative_to(ROOT)} ({len(content)} caracteres, {len(content.splitlines())} lineas)")

if __name__ == '__main__':
    sys.exit(main() or 0)