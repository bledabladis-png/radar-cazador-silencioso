"""Contract test for .github/workflows/daily_run.yml.

Garantiza que el step Commit and push hist/state no vuelve a usar
``git add -u outputs/history`` (bug P0-SCHEMA-NUEVOS-NO-STAGED, fix
7f6d229d). No exige la solucion actual (``git add -A outputs/history``);
una solucion alternativa valida (e.g. ``git add outputs/history/*.schema``)
pasa el test.
"""
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "daily_run.yml"


def _load():
    return yaml.safe_load(WORKFLOW.read_text("utf-8-sig"))


def _commit_step_run_text():
    data = _load()
    job = data.get("jobs", {}).get("run-system")
    assert isinstance(job, dict), "job run-system no encontrado en daily_run.yml"
    for step in job.get("steps", []):
        if not isinstance(step, dict):
            continue
        name = (step.get("name") or "").lower()
        if "commit" in name and "push" in name:
            return step.get("run", "")
    raise AssertionError(
        "Step 'Commit and push hist/state' no encontrado en job run-system"
    )


def _non_comment_lines(text):
    return [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def test_daily_run_yaml_parses():
    data = _load()
    assert "jobs" in data, "daily_run.yml no contiene 'jobs'"


def test_commit_step_does_not_regress_to_git_add_u_outputs_history():
    text = _commit_step_run_text()
    bad = [
        ln for ln in _non_comment_lines(text)
        if ln.startswith("git add -u") and "outputs/history" in ln
    ]
    assert not bad, (
        "Regresion P0-SCHEMA-NUEVOS-NO-STAGED: el step Commit usa "
        "`git add -u` sobre outputs/history. Ver fix 7f6d229d. "
        f"Lineas ofensoras: {bad}"
    )


def test_commit_step_stages_outputs_history():
    text = _commit_step_run_text()
    hits = [
        ln for ln in _non_comment_lines(text)
        if ln.startswith("git add") and "outputs/history" in ln
    ]
    assert hits, (
        "El step Commit no stagea outputs/history. Debe haber al menos "
        "un `git add` que lo incluya."
    )


def test_commit_step_uses_git_add_a_outputs_state():
    """Regresion BUG-DAILY-RUN-STATE-UNTRACKED (2026-11-12): el step debe
    usar git add -A outputs/state (no -u), para stagear writers nuevos
    cuyo output es untracked la primera vez."""
    text = _commit_step_run_text()
    bad = [
        ln for ln in _non_comment_lines(text)
        if ln.startswith("git add -u") and "outputs/state" in ln
    ]
    assert not bad, (
        "Regresion BUG-DAILY-RUN-STATE-UNTRACKED: el step Commit usa "
        "`git add -u` sobre outputs/state. Ver fix del 2026-11-12. "
        f"Lineas ofensoras: {bad}"
    )
    hits = [
        ln for ln in _non_comment_lines(text)
        if ln.startswith("git add -A") and "outputs/state" in ln
    ]
    assert hits, (
        "El step Commit no stagea outputs/state con -A. Debe haber al "
        "menos un `git add -A outputs/state`."
    )


def test_artifact_daily_report_includes_temporal_contracts_result():
    """temporal_contracts_result.json debe viajar en el artifact daily-report"""
    data = _load()
    job = data.get("jobs", {}).get("run-system")
    assert isinstance(job, dict)
    for step in job.get("steps", []):
        if not isinstance(step, dict):
            continue
        name = (step.get("name") or "").lower()
        if "upload" in name and "downloadable report" in name:
            paths = step.get("with", {}).get("path", "")
            assert "outputs/state/temporal_contracts_result.json" in paths, (
                "El artifact daily-report no incluye "
                "outputs/state/temporal_contracts_result.json."
            )
            return
    raise AssertionError(
        "Step Upload artifacts (downloadable report) no encontrado."
    )
