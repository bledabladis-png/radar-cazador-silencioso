# -*- coding: utf-8 -*-
"""Tests del verificador de gobernanza (Fase 4)."""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.verifica_gobernanza as vg


def _write_registro(tmp_path, items):
    p = tmp_path / "REGISTRO.yaml"
    p.write_text(yaml.safe_dump({"version": 1, "items": items}),
                 encoding="utf-8")
    return p


def _base_item(**kwargs):
    d = {
        "id": "TEST-1",
        "tipo": "auditoria_interna",
        "ciclo": "2026-10-16",
        "estado": "OPEN",
        "descripcion": "x",
        "siguiente_accion": "hacer algo",
    }
    d.update(kwargs)
    return d


def _handoff_vacio(tmp_path):
    p = tmp_path / "HANDOFF.md"
    p.write_text("## 4. PENDIENTES VIVOS\n\n## 5. X\n", encoding="utf-8")
    return p


def test_registro_real_pasa():
    """El REGISTRO.yaml versionado debe pasar sin errores no-PENDING."""
    errores = vg.verificar()
    no_pending = [e for e in errores if "PENDING" not in e]
    assert not no_pending, "\n".join(no_pending)


def test_id_duplicado(tmp_path):
    p = _write_registro(tmp_path, [_base_item(), _base_item()])
    errores = vg.verificar(registro_path=p, handoff_path=_handoff_vacio(tmp_path))
    assert any("duplicado" in e for e in errores)


def test_estado_invalido(tmp_path):
    p = _write_registro(tmp_path, [_base_item(estado="FOO")])
    errores = vg.verificar(registro_path=p, handoff_path=_handoff_vacio(tmp_path))
    assert any("estado invalido" in e for e in errores)


def test_fixed_sin_commit(tmp_path):
    it = _base_item(estado="FIXED", test="tests/test_x.py")
    del it["siguiente_accion"]
    p = _write_registro(tmp_path, [it])
    errores = vg.verificar(registro_path=p, handoff_path=_handoff_vacio(tmp_path))
    assert any("FIXED sin commit" in e for e in errores)


def test_fixed_sin_test_ni_evidencia(tmp_path):
    it = _base_item(estado="FIXED", commit="abc123")
    del it["siguiente_accion"]
    p = _write_registro(tmp_path, [it])
    errores = vg.verificar(registro_path=p, handoff_path=_handoff_vacio(tmp_path))
    assert any("sin test ni evidencia" in e for e in errores)


def test_fixed_con_evidencia_sin_test(tmp_path):
    it = _base_item(
        estado="FIXED", commit="abc123",
        evidencia="Fix aplicado, sin test especifico por cambio de string",
    )
    del it["siguiente_accion"]
    p = _write_registro(tmp_path, [it])
    errores = vg.verificar(registro_path=p, handoff_path=_handoff_vacio(tmp_path))
    assert not any("sin test ni evidencia" in e for e in errores)


def test_fixed_commit_inexistente(tmp_path):
    it = _base_item(
        estado="FIXED", commit="0000000", test="tests/test_x.py",
    )
    del it["siguiente_accion"]
    p = _write_registro(tmp_path, [it])
    errores = vg.verificar(registro_path=p, handoff_path=_handoff_vacio(tmp_path))
    assert any("commit no existe" in e for e in errores)


def test_fixed_test_inexistente(tmp_path):
    it = _base_item(
        estado="FIXED", commit="abc123", test="tests/no_existe_xyz.py",
    )
    del it["siguiente_accion"]
    p = _write_registro(tmp_path, [it])
    errores = vg.verificar(registro_path=p, handoff_path=_handoff_vacio(tmp_path))
    assert any("test no existe" in e for e in errores)


def test_wont_fix_sin_razon(tmp_path):
    it = _base_item(estado="WONT_FIX")
    del it["siguiente_accion"]
    p = _write_registro(tmp_path, [it])
    errores = vg.verificar(registro_path=p, handoff_path=_handoff_vacio(tmp_path))
    assert any("razon <" in e for e in errores)


def test_wont_fix_razon_corta(tmp_path):
    it = _base_item(estado="WONT_FIX", razon="ok")
    del it["siguiente_accion"]
    p = _write_registro(tmp_path, [it])
    errores = vg.verificar(registro_path=p, handoff_path=_handoff_vacio(tmp_path))
    assert any("razon <" in e for e in errores)


def test_wont_fix_razon_generica(tmp_path):
    it = _base_item(estado="WONT_FIX", razon="wont fix")
    del it["siguiente_accion"]
    p = _write_registro(tmp_path, [it])
    errores = vg.verificar(registro_path=p, handoff_path=_handoff_vacio(tmp_path))
    assert any("razon generica" in e for e in errores)


def test_blocked_sin_dependencia(tmp_path):
    it = _base_item(estado="BLOCKED_EXTERNAL")
    del it["siguiente_accion"]
    p = _write_registro(tmp_path, [it])
    errores = vg.verificar(registro_path=p, handoff_path=_handoff_vacio(tmp_path))
    assert any("BLOCKED_EXTERNAL sin dependencia" in e for e in errores)


def test_open_sin_siguiente_accion(tmp_path):
    it = _base_item(estado="OPEN")
    del it["siguiente_accion"]
    p = _write_registro(tmp_path, [it])
    errores = vg.verificar(registro_path=p, handoff_path=_handoff_vacio(tmp_path))
    assert any("sin siguiente_accion" in e for e in errores)


def test_estado_no_fixed_aparece_pendiente_en_handoff(tmp_path):
    """Regla: item con estado != FIXED no puede figurar como pendiente."""
    handoff = tmp_path / "HANDOFF.md"
    handoff.write_text(
        "## 4. PENDIENTES VIVOS\n\n- TEST-1: pendiente\n\n## 5. X\n",
        encoding="utf-8",
    )
    it = _base_item(estado="WONT_FIX", razon="x" * 25)
    del it["siguiente_accion"]
    p = _write_registro(tmp_path, [it])
    errores = vg.verificar(registro_path=p, handoff_path=handoff)
    assert any("aparece como pendiente" in e for e in errores)