"""Tests HD-6: pyflakes es un gate real en daily_run.yml.

Dictamen auditor externo 2026-10-16: el step no debe enmascarar el
exit code de pyflakes con `|| true`. El reporte se preserva como
artifact con `if: always()`.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

YML = ROOT / ".github" / "workflows" / "daily_run.yml"


def _read():
    return YML.read_text(encoding="utf-8")


def _step_block(name_pattern):
    """Devuelve el bloque de texto de un step por su nombre.

    Encuentra la linea que matchea "- name: <pattern>" y captura
    hasta el siguiente "- name:" al MISMO nivel de indentacion (o fin
    de fichero). Robusto frente a indentaciones variables y a `|| true`
    dentro de otros steps.
    """
    lines = _read().splitlines(keepends=True)
    start = None
    indent = None
    for i, ln in enumerate(lines):
        m = re.match(r"^(\s*)- name: " + name_pattern, ln)
        if m:
            start = i
            indent = m.group(1)
            break
    if start is None:
        raise AssertionError(
            "step '{0}' no encontrado en YAML".format(name_pattern)
        )
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if re.match(r"^" + re.escape(indent) + r"- ", lines[j]):
            end = j
            break
    return "".join(lines[start:end])


def test_pyflakes_step_existe():
    block = _step_block(r"Run pyflakes.*")
    assert "python -m pyflakes" in block


def test_pyflakes_step_no_enmascara_exit_code():
    """HD-6: NO debe contener '|| true' ni continue-on-error."""
    block = _step_block(r"Run pyflakes.*")
    assert "|| true" not in block, (
        "el step enmascara el exit code con '|| true'"
    )
    assert "continue-on-error" not in block, (
        "el step tiene continue-on-error"
    )


def test_pyflakes_step_propaga_exit_code():
    """El step debe propagar el exit code de pyflakes."""
    block = _step_block(r"Run pyflakes.*")
    # Acepta cualquiera de: "exit $rc", "exit ${rc}", "exit $?"
    assert re.search(r"exit \$(rc|\{rc\}|\?)", block), (
        "el step no propaga el exit code de pyflakes"
    )


def test_pyflakes_artifact_con_always():
    """El artifact del reporte se sube con if: always() para
    preservar la evidencia incluso cuando el gate falla."""
    block = _step_block(r"Upload pyflakes report")
    assert "if: always()" in block, (
        "el step de upload no tiene if: always()"
    )
    assert "outputs/audit/pyflakes_report.txt" in block, (
        "el upload no referencia el reporte correcto"
    )


def test_pyflakes_step_captura_rc():
    """El step debe capturar el exit code en una variable antes de
    cualquier otro comando (evita perder el rc por comandos intermedios)."""
    block = _step_block(r"Run pyflakes.*")
    # Despues de la linea pyflakes, capturar rc inmediatamente
    pattern = r"python -m pyflakes \. > outputs/audit/pyflakes_report\.txt 2>&1\s*\n\s*rc=\$\?"
    assert re.search(pattern, block), (
        "el rc no se captura inmediatamente tras pyflakes"
    )
