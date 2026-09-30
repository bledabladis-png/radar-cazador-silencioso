# -*- coding: utf-8 -*-
"""D10 (2026-09-30): trazabilidad de ingesta 13F en fichero JSONL.

Antes: _write_ingest_trace anadia ingest_source + ingest_actor al
manifest del trimestre, INCLUSO en rama [SKIP]. El manifest es un
artefacto de integridad (sha256 del parquet); meterle metadatos
del run contamina el artefacto y ensucia el working tree en cada
invocacion local.

Ahora: append a data/sec_13f/ingest_traces.jsonl. Manifest intacto.
Formato JSONL: una linea JSON por invocacion, con ts UTC.
"""
import json
from datetime import datetime
from pathlib import Path

from scripts.update_sec_13f import _write_ingest_trace


FIXTURE_MANIFEST = {
    "schema_version": 1,
    "artifact": {"sha256": "abc123", "bytes": 100},
    "quality": {"last_date": "2026-08-31", "status": "VALID"},
}


def test_write_ingest_trace_crea_jsonl(tmp_path):
    """Una invocacion -> un fichero con una linea JSON valida."""
    trace = tmp_path / "ingest_traces.jsonl"
    _write_ingest_trace("2026Q2", "manual", "marta", "INGEST", trace_path=trace)

    assert trace.exists()
    lines = trace.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["quarter"] == "2026Q2"
    assert entry["source"] == "manual"
    assert entry["actor"] == "marta"
    assert entry["outcome"] == "INGEST"
    # ts ISO UTC
    assert entry["ts"].endswith("Z")
    datetime.strptime(entry["ts"], "%Y-%m-%dT%H:%M:%SZ")


def test_write_ingest_trace_append(tmp_path):
    """Dos invocaciones -> dos lineas, sin reescribir la primera."""
    trace = tmp_path / "ingest_traces.jsonl"
    _write_ingest_trace("2026Q2", "manual", "marta", "SKIP", trace_path=trace)
    _write_ingest_trace("2026Q1", "cron", "github-actions", "INGEST", trace_path=trace)

    lines = trace.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    e1 = json.loads(lines[0])
    e2 = json.loads(lines[1])
    assert e1["quarter"] == "2026Q2"
    assert e1["outcome"] == "SKIP"
    assert e2["quarter"] == "2026Q1"
    assert e2["outcome"] == "INGEST"


def test_write_ingest_trace_crea_directorio_padre(tmp_path):
    """Si data/sec_13f/ no existe, se crea."""
    trace = tmp_path / "nuevo" / "sub" / "traces.jsonl"
    _write_ingest_trace("2026Q2", "manual", "marta", "INGEST", trace_path=trace)
    assert trace.exists()


def test_manifest_intacto_tras_ingest_trace(tmp_path, monkeypatch):
    """El manifest NO se modifica al llamar _write_ingest_trace SIN trace_path.

    Regresion del bug D10: la funcion escribia ingest_source/ingest_actor
    en el manifest del trimestre. Este test llama SIN trace_path, con
    ROOT monkeypatcheado a tmp_path, para ejercitar el default real.
    Con el fix, el default es ingest_traces.jsonl -> manifest intacto.
    Con el bug, el default era el manifest -> antes != after.
    """
    import scripts.update_sec_13f as mod
    monkeypatch.setattr(mod, "ROOT", tmp_path)

    manifests = tmp_path / "data" / "sec_13f" / "manifests"
    manifests.mkdir(parents=True)
    mp = manifests / "sec_13f_2026Q2.json"
    mp.write_text(json.dumps(FIXTURE_MANIFEST, indent=2), encoding="utf-8")
    before = mp.read_bytes()

    _write_ingest_trace("2026Q2", "manual", "marta", "SKIP")

    after = mp.read_bytes()
    assert before == after, (
        "manifest modificado por _write_ingest_trace. "
        "Bug D10: la traza debe ir a ingest_traces.jsonl, no al manifest."
    )
    # Y el jsonl SI se ha creado en el default
    jsonl = tmp_path / "data" / "sec_13f" / "ingest_traces.jsonl"
    assert jsonl.exists(), "default no creo el jsonl"


def test_no_queda_codigo_que_escriba_ingest_source_en_manifest():
    """Regresion: ningun fichero fuente escribe ingest_source en el manifest."""
    src = Path(__file__).resolve().parents[1] / "scripts" / "update_sec_13f.py"
    text = src.read_text(encoding="utf-8")
    assert 'data["ingest_source"]' not in text, "codigo muerto residual"
    assert 'data["ingest_actor"]' not in text, "codigo muerto residual"


def test_workflow_13f_incluye_git_add_ingest_traces():
    """El workflow de 13F debe commitear el jsonl junto a latest_quarter."""
    p = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "update_sec_13f.yml"
    text = p.read_text(encoding="utf-8")
    assert "git add data/sec_13f/ingest_traces.jsonl" in text, (
        "el workflow no incluye ingest_traces.jsonl en git add"
    )


def test_gitignore_no_bloquea_ingest_traces():
    """El jsonl debe poder versionarse (D10)."""
    p = Path(__file__).resolve().parents[1] / ".gitignore"
    text = p.read_text(encoding="utf-8")
    assert "data/sec_13f/ingest_traces.jsonl" not in text, "bloqueado explicitamente"
    # *.log esta bloqueado, *.jsonl no.
    assert "*.jsonl" not in text, "*.jsonl bloqueado, deberia no estarlo"
