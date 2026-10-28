"""Regresiones del validador de gobernanza (bug 2026-10-28).

Caso 1: append de items sin linea en blanco -> "- id:" embebido en
siguiente_accion: |. PyYAML conserva el ultimo valor por duplicado
de clave, corrompiendo el item anterior y perdiendo el nuevo.

Caso 2: YAML convierte SHAs solo-digitos (64866361) a int.
_git_commit_exists(int) falla con TypeError capturado por except,
devolviendo False. Se reportaba como "commit no existe o huerfano".
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import verifica_gobernanza as vg


def _write_yaml(tmp_path, content):
    p = tmp_path / "REGISTRO.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def _fake_handoff(tmp_path):
    return tmp_path / "no_such_handoff.md"


def test_yaml_limpio_sin_embebidos(tmp_path):
    yaml = """version: 1
items:
  - id: ITEM-A
    tipo: deuda_tecnica
    ciclo: '2026-10-01'
    estado: FIXED
    commit: '3a682884'
    evidencia: 'algo suficientemente largo para pasar el check de 20 chars'
  - id: ITEM-B
    tipo: deuda_tecnica
    ciclo: '2026-10-01'
    estado: OPEN
    siguiente_accion: 'hacer algo'
"""
    p = _write_yaml(tmp_path, yaml)
    errores = vg.verificar(registro_path=p, handoff_path=_fake_handoff(tmp_path))
    embebidos = [e for e in errores if "embebido" in e]
    assert embebidos == [], "falsos positivos: " + str(embebidos)


def test_yaml_con_id_embebido_reporta_error(tmp_path):
    yaml = """version: 1
items:
  - id: ITEM-A
    tipo: deuda_tecnica
    ciclo: '2026-10-01'
    estado: FIXED
    commit: '3a682884'
    evidencia: 'algo suficientemente largo para pasar el check de 20 chars'
    siguiente_accion: |
      texto.- id: ITEM-B
    tipo: deuda_tecnica
    estado: OPEN
    siguiente_accion: 'hacer algo'
"""
    p = _write_yaml(tmp_path, yaml)
    errores = vg.verificar(registro_path=p, handoff_path=_fake_handoff(tmp_path))
    embebidos = [e for e in errores if "embebido" in e]
    assert len(embebidos) == 1, "esperado 1, obtenido " + str(errores)
    assert "linea" in embebidos[0]
    assert "ITEM-B" in embebidos[0]


def test_yaml_commit_int_reporta_error(tmp_path):
    yaml = """version: 1
items:
  - id: ITEM-A
    tipo: deuda_tecnica
    ciclo: '2026-10-01'
    estado: FIXED
    commit: 64866361
    evidencia: 'algo suficientemente largo para pasar el check de 20 chars'
"""
    p = _write_yaml(tmp_path, yaml)
    errores = vg.verificar(registro_path=p, handoff_path=_fake_handoff(tmp_path))
    tipo_err = [e for e in errores if "no es string" in e]
    assert len(tipo_err) == 1, "esperado 1, obtenido " + str(errores)
    assert "int" in tipo_err[0]


def test_registro_real_limpio():
    """Regresion: REGISTRO actual no tiene embebidos ni commits int."""
    errores = vg.verificar()
    embebidos = [e for e in errores if "embebido" in e]
    tipos = [e for e in errores if "no es string" in e]
    assert embebidos == [], "embebidos: " + str(embebidos)
    assert tipos == [], "commits no-str: " + str(tipos)
