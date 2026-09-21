"""Tests B2-PIT (dictamen #52).

Subfase aislada: infraestructura temporal del catalogo. NO incluye
catalog_key ni validadores de asignacion (esos van en B1).
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from src.institutional_accumulation.catalog_pit import (
    CatalogAmbiguous,
    CatalogNotAvailable,
    ManifestError,
    SnapshotIntegrityError,
    list_snapshots,
    load_manifest,
    target_catalog_as_of,
    verify_snapshot_integrity,
)


def _make_snapshot(catalog_root, version_id, rows, *, valid_from, valid_to):
    """Materializa un snapshot completo (csv + sha + manifest entry)."""
    snap_dir = catalog_root / "catalog_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    csv_path = snap_dir / ("snapshot_{0}.csv".format(version_id))
    df = pd.DataFrame(rows, columns=["radar_ticker", "figi"])
    csv_path.write_text(
        df.to_csv(index=False),
        encoding="utf-8",
        newline="\n",
    )

    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    sha_path = snap_dir / ("snapshot_{0}.sha256".format(version_id))
    sha_path.write_text(sha + "\n", encoding="utf-8", newline="\n")

    return {
        "version_id": version_id,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "sha256": sha,
        "csv_path": "catalog_snapshots/snapshot_{0}.csv".format(version_id),
        "rows": len(rows),
        "producer": "test",
    }


def _write_manifest(catalog_root, entries):
    manifest = {"schema_version": 1, "snapshots": entries}
    (catalog_root / "catalog_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


@pytest.fixture
def catalog_root(tmp_path):
    root = tmp_path / "catalog_root"
    root.mkdir()
    return root


def test_as_of_0_snapshots_que_cubren_raise(catalog_root):
    _write_manifest(catalog_root, [])
    with pytest.raises(CatalogNotAvailable):
        target_catalog_as_of("2026-03-31", catalog_root=catalog_root)


def test_as_of_1_snapshot_que_cubre_ok(catalog_root):
    entry = _make_snapshot(
        catalog_root, "20260101_01",
        [["AAPL", "BBG001S5N8V8"]],
        valid_from="2026-01-01", valid_to=None,
    )
    _write_manifest(catalog_root, [entry])
    df, vid, sha = target_catalog_as_of("2026-03-31", catalog_root=catalog_root)
    assert vid == "20260101_01"
    assert sha == entry["sha256"]
    assert len(df) == 1
    assert df.iloc[0]["radar_ticker"] == "AAPL"


def test_as_of_multiples_snapshots_que_cubren_ambiguous(catalog_root):
    e1 = _make_snapshot(
        catalog_root, "20260101_01",
        [["AAPL", "BBG001S5N8V8"]],
        valid_from="2026-01-01", valid_to="2026-06-01",
    )
    e2 = _make_snapshot(
        catalog_root, "20260501_01",
        [["AAPL", "BBG001S5N8V8"]],
        valid_from="2026-05-01", valid_to=None,
    )
    _write_manifest(catalog_root, [e1, e2])
    with pytest.raises(CatalogAmbiguous):
        target_catalog_as_of("2026-05-15", catalog_root=catalog_root)

def test_as_of_snapshot_existente_pero_no_cubre_raise(catalog_root):
    entry = _make_snapshot(
        catalog_root, "20260101_01",
        [["AAPL", "BBG001S5N8V8"]],
        valid_from="2026-01-01", valid_to="2026-02-01",
    )
    _write_manifest(catalog_root, [entry])
    # period_end posterior al valid_to. Snapshot existe pero NO cubre.
    with pytest.raises(CatalogNotAvailable):
        target_catalog_as_of("2026-03-31", catalog_root=catalog_root)


def test_as_of_backdating_prohibido_q4_2025(catalog_root):
    # Snapshot vigente desde 2026-09-21. Q4 2025 no tiene snapshot.
    entry = _make_snapshot(
        catalog_root, "20260921_01",
        [["AAPL", "BBG001S5N8V8"]],
        valid_from="2026-09-21", valid_to=None,
    )
    _write_manifest(catalog_root, [entry])
    with pytest.raises(CatalogNotAvailable):
        target_catalog_as_of("2025-12-31", catalog_root=catalog_root)


def test_as_of_backdating_prohibido_q1_2026(catalog_root):
    entry = _make_snapshot(
        catalog_root, "20260921_01",
        [["AAPL", "BBG001S5N8V8"]],
        valid_from="2026-09-21", valid_to=None,
    )
    _write_manifest(catalog_root, [entry])
    with pytest.raises(CatalogNotAvailable):
        target_catalog_as_of("2026-03-31", catalog_root=catalog_root)


def test_sha256_publicado_vs_recalculado_match(catalog_root):
    entry = _make_snapshot(
        catalog_root, "20260101_01",
        [["AAPL", "BBG001S5N8V8"]],
        valid_from="2026-01-01", valid_to=None,
    )
    _write_manifest(catalog_root, [entry])
    assert verify_snapshot_integrity("20260101_01", catalog_root=catalog_root) is True


def test_sha256_mismatch_fail_closed(catalog_root):
    entry = _make_snapshot(
        catalog_root, "20260101_01",
        [["AAPL", "BBG001S5N8V8"]],
        valid_from="2026-01-01", valid_to=None,
    )
    _write_manifest(catalog_root, [entry])
    # Alterar el .sha256 para que no coincida con el CSV.
    sha_path = catalog_root / "catalog_snapshots" / "snapshot_20260101_01.sha256"
    sha_path.write_text("0" * 64 + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(SnapshotIntegrityError):
        verify_snapshot_integrity("20260101_01", catalog_root=catalog_root)


def test_corrupcion_simulada_detectada(catalog_root):
    entry = _make_snapshot(
        catalog_root, "20260101_01",
        [["AAPL", "BBG001S5N8V8"]],
        valid_from="2026-01-01", valid_to=None,
    )
    _write_manifest(catalog_root, [entry])
    # Corromper 1 byte del CSV: sha recalculado no coincidira.
    csv_path = catalog_root / "catalog_snapshots" / "snapshot_20260101_01.csv"
    data = bytearray(csv_path.read_bytes())
    data[10] = (data[10] + 1) % 256
    csv_path.write_bytes(bytes(data))
    with pytest.raises(SnapshotIntegrityError):
        verify_snapshot_integrity("20260101_01", catalog_root=catalog_root)


def test_manifest_schema_version_valido(catalog_root):
    (catalog_root / "catalog_manifest.json").write_text(
        json.dumps({"schema_version": 99, "snapshots": []}),
        encoding="utf-8",
    )
    with pytest.raises(ManifestError):
        load_manifest(catalog_root)


def test_manifest_inexistente_raise(catalog_root):
    with pytest.raises(ManifestError):
        load_manifest(catalog_root)


def test_intervalos_semiabiertos_sin_solapamiento(catalog_root):
    e1 = _make_snapshot(
        catalog_root, "20260101_01",
        [["AAPL", "BBG001S5N8V8"]],
        valid_from="2026-01-01", valid_to="2026-09-19",
    )
    e2 = _make_snapshot(
        catalog_root, "20260919_01",
        [["AAPL", "BBG001S5N8V8"]],
        valid_from="2026-09-19", valid_to=None,
    )
    _write_manifest(catalog_root, [e1, e2])

    # 2026-09-18 -> solo e1 (e2 valid_from=09-19 excluido).
    df, vid, _ = target_catalog_as_of("2026-09-18", catalog_root=catalog_root)
    assert vid == "20260101_01"

    # 2026-09-19 -> solo e2 (semiabierto: e1 excluye 09-19).
    df, vid, _ = target_catalog_as_of("2026-09-19", catalog_root=catalog_root)
    assert vid == "20260919_01"


def test_manifest_coherente_con_snapshot_sha256(catalog_root):
    entry = _make_snapshot(
        catalog_root, "20260101_01",
        [["AAPL", "BBG001S5N8V8"]],
        valid_from="2026-01-01", valid_to=None,
    )
    _write_manifest(catalog_root, [entry])
    # Manifest declara sha distinto al .sha256 publicado.
    entry_bad = dict(entry)
    entry_bad["sha256"] = "0" * 64
    _write_manifest(catalog_root, [entry_bad])
    with pytest.raises(SnapshotIntegrityError):
        verify_snapshot_integrity("20260101_01", catalog_root=catalog_root)


def test_load_manifest_catalog_root_inexistente_raise(tmp_path):
    nonexistent = tmp_path / "no_existe"
    with pytest.raises(ManifestError):
        load_manifest(nonexistent)


def test_snapshot_csv_publicado_inmutable_entre_runs(catalog_root):
    entry = _make_snapshot(
        catalog_root, "20260101_01",
        [["AAPL", "BBG001S5N8V8"]],
        valid_from="2026-01-01", valid_to=None,
    )
    _write_manifest(catalog_root, [entry])
    sha_before = hashlib.sha256(
        (catalog_root / "catalog_snapshots" / "snapshot_20260101_01.csv").read_bytes()
    ).hexdigest()
    # Ejecutar target_catalog_as_of dos veces.
    target_catalog_as_of("2026-03-31", catalog_root=catalog_root)
    target_catalog_as_of("2026-03-31", catalog_root=catalog_root)
    sha_after = hashlib.sha256(
        (catalog_root / "catalog_snapshots" / "snapshot_20260101_01.csv").read_bytes()
    ).hexdigest()
    assert sha_before == sha_after


def test_list_snapshots_devuelve_entradas(catalog_root):
    e = _make_snapshot(
        catalog_root, "20260101_01",
        [["AAPL", "BBG001S5N8V8"]],
        valid_from="2026-01-01", valid_to=None,
    )
    _write_manifest(catalog_root, [e])
    snaps = list_snapshots(catalog_root=catalog_root)
    assert len(snaps) == 1
    assert snaps[0]["version_id"] == "20260101_01"