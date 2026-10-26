# -*- coding: utf-8 -*-
"""Tests de scripts/poda_bitacora.py (Fase 7)."""
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.poda_bitacora as pb


def _hash(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _ranges(lines):
    return pb.find_entries(lines, 0, len(lines))


def test_check_no_muta_bitacora():
    """--check NO modifica el fichero."""
    antes = _hash(pb.BITACORA)
    r = subprocess.run(
        [sys.executable, "scripts/poda_bitacora.py", "--check"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    despues = _hash(pb.BITACORA)
    assert antes == despues, "--check muto la bitacora"
    assert r.returncode in (0, 1)


def test_dry_run_no_muta():
    """Dry-run tampoco modifica el fichero."""
    antes = _hash(pb.BITACORA)
    subprocess.run(
        [sys.executable, "scripts/poda_bitacora.py"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert antes == _hash(pb.BITACORA)


def test_check_force_fail_con_keep_1():
    """Con --keep 1 y >1 sesion, --check debe devolver exit 1."""
    r = subprocess.run(
        [sys.executable, "scripts/poda_bitacora.py", "--check", "--keep", "1"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    # La bitacora actual tiene >1 sesion, asi que con keep=1 debe fallar
    assert r.returncode == 1
    assert "excede" in r.stdout


def test_decide_agrupa_mismo_dia():
    """Entradas del mismo dia consecutivas cuentan como sesion unica."""
    lines = [
        "### 2026-10-16 - primera\n",
        "### 2026-10-16 - segunda\n",
        "### 2026-10-15 - otra\n",
        "### 2026-10-14 - vieja\n",
    ]
    ranges = _ranges(lines)
    assert len(ranges) == 4
    keep, remove = pb.decide(ranges, keep_n=1)
    # dia en curso (16) + 1 sesion fuera (15) => headings 0,1,2
    assert keep == [0, 1, 2]
    assert remove == [3]


def test_decide_keep_n():
    """Semantica (b): keep_n sesiones fuera del dia en curso + dia en curso aparte."""
    lines = [
        "### 2026-10-16\n",
        "### 2026-10-15\n",
        "### 2026-10-14\n",
        "### 2026-10-13\n",
    ]
    ranges = _ranges(lines)
    keep, remove = pb.decide(ranges, keep_n=2)
    # 16 (dia en curso) + 15 + 14 = 3 conservadas; 13 eliminada
    assert keep == [0, 1, 2]
    assert remove == [3]


def test_decide_dia_en_curso_aparte_caso_real():
    """12 fechas unicas, keep=10 => conservar 11 grupos (dia en curso
    aparte), podar solo la mas antigua (2026-10-13)."""
    fechas = [
        "2026-10-25", "2026-10-24", "2026-10-23", "2026-10-22",
        "2026-10-21", "2026-10-20", "2026-10-18", "2026-10-17",
        "2026-10-16", "2026-10-15", "2026-10-14", "2026-10-13",
    ]
    lines = [f"### {d}\n" for d in fechas]
    ranges = _ranges(lines)
    keep, remove = pb.decide(ranges, keep_n=10)
    assert keep == list(range(11))
    assert remove == [11]


def test_decide_11_fechas_1_en_curso_keep_10():
    """11 fechas (1 dia en curso + 10 fuera), keep=10 => conservar todas."""
    fechas = [
        "2026-10-25", "2026-10-24", "2026-10-23", "2026-10-22",
        "2026-10-21", "2026-10-20", "2026-10-18", "2026-10-17",
        "2026-10-16", "2026-10-15", "2026-10-14",
    ]
    lines = [f"### {d}\n" for d in fechas]
    ranges = _ranges(lines)
    keep, remove = pb.decide(ranges, keep_n=10)
    assert keep == list(range(11))
    assert remove == []


def test_contar_sesiones_unicas():
    lines = [
        "### 2026-10-16\n",
        "### 2026-10-16\n",
        "### 2026-10-15\n",
    ]
    ranges = _ranges(lines)
    assert pb.contar_sesiones(ranges) == 2


def test_agrupar_por_fecha():
    lines = [
        "### 2026-10-16 - a\n",
        "### 2026-10-16 - b\n",
        "### 2026-10-15\n",
    ]
    grupos = pb.agrupar_por_fecha(_ranges(lines))
    assert len(grupos) == 2
    assert grupos[0][2] == "2026-10-16"
    assert len(grupos[0][3]) == 2  # dos headings en el grupo
    assert grupos[1][2] == "2026-10-15"
    assert len(grupos[1][3]) == 1