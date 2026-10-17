# -*- coding: utf-8 -*-
"""REP-AUDIT-20261017-capa3: tests H1 (timestamps) + H2 (local_run_id)."""
from datetime import datetime, timezone
from pathlib import Path

from src.report.provenance import build_provenance, render_provenance


def test_h1_generated_at_aparece_en_provenance():
    ref = datetime(2026, 10, 17, 15, 0, tzinfo=timezone.utc)
    gen = "2026-10-17T15:30:00Z"
    prov = build_provenance(ref, generated_at=gen)
    out = "".join(render_provenance(prov))
    assert f"- `generated_at`: {gen} (UTC)" in out


def test_h1_reference_y_generated_ambos_etiquetados():
    ref = datetime(2026, 10, 17, 15, 0, tzinfo=timezone.utc)
    gen = "2026-10-17T15:30:00Z"
    prov = build_provenance(ref, generated_at=gen)
    out = "".join(render_provenance(prov))
    assert "`reference_date`:" in out
    assert "`generated_at`:" in out


def test_h1_header_sin_fecha_ambigua():
    src = Path("src/report_generator.py").read_bytes()
    if src[:3] == b"\xef\xbb\xbf":
        src = src[3:]
    text = src.decode("utf-8")
    assert "lines.append(f\"**Fecha:**" not in text
    assert 'lines.append(f"**Referencia:**' in text
    assert 'lines.append(f"**Generado:**' in text


def test_h2_local_run_id_sin_ci():
    ref = datetime(2026, 10, 17, 15, 0, tzinfo=timezone.utc)
    prov = build_provenance(ref, env={})
    assert prov["run_id"].startswith("local-")
    assert len(prov["run_id"]) == len("local-") + 8


def test_h2_local_run_id_determinista():
    ref = datetime(2026, 10, 17, 15, 0, tzinfo=timezone.utc)
    gen = "2026-10-17T15:30:00Z"
    p1 = build_provenance(ref, generated_at=gen, env={})
    p2 = build_provenance(ref, generated_at=gen, env={})
    assert p1["run_id"] == p2["run_id"]


def test_h2_local_run_id_distinto_entre_ejecuciones():
    ref1 = datetime(2026, 10, 17, 15, 0, tzinfo=timezone.utc)
    ref2 = datetime(2026, 10, 17, 15, 1, tzinfo=timezone.utc)
    assert build_provenance(ref1, env={})["run_id"] != build_provenance(ref2, env={})["run_id"]


def test_h2_ci_run_id_se_preserva():
    ref = datetime(2026, 10, 17, 15, 0, tzinfo=timezone.utc)
    prov = build_provenance(ref, env={"GITHUB_RUN_ID": "12345678"})
    assert prov["run_id"] == "12345678"


def test_h2_ci_run_id_cero_genera_local():
    ref = datetime(2026, 10, 17, 15, 0, tzinfo=timezone.utc)
    prov = build_provenance(ref, env={"GITHUB_RUN_ID": "0"})
    assert prov["run_id"].startswith("local-")


def test_h2_render_provenance_local_no_dice_nd():
    ref = datetime(2026, 10, 17, 15, 0, tzinfo=timezone.utc)
    prov = build_provenance(ref, env={})
    out = "".join(render_provenance(prov))
    assert "N/D (ejecucion local)" not in out
    assert "local-" in out
    assert "(ejecucion local)" in out


def test_h2_render_provenance_ci_sin_attempt():
    ref = datetime(2026, 10, 17, 15, 0, tzinfo=timezone.utc)
    prov = build_provenance(ref, env={"GITHUB_RUN_ID": "99999"})
    out = "".join(render_provenance(prov))
    assert "`run_id`: 99999" in out
    assert "(ejecucion local)" not in out


def test_h2_render_provenance_ci_con_attempt():
    ref = datetime(2026, 10, 17, 15, 0, tzinfo=timezone.utc)
    prov = build_provenance(ref, env={
        "GITHUB_RUN_ID": "99999",
        "GITHUB_RUN_ATTEMPT": "2",
    })
    out = "".join(render_provenance(prov))
    assert "`run_id`: 99999 (attempt 2)" in out
