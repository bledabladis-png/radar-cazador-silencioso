# -*- coding: utf-8 -*-
"""Tests de schema versioning en outputs (Fase 1)."""
import pandas as pd


def _make_df(date="2026-10-23"):
    return pd.DataFrame({
        "date": [date, date],
        "sector": ["XLB", "XLC"],
        "value": [1.0, 2.0],
    })


def test_persist_primera_vez_escribe_schema(tmp_path, monkeypatch):
    """Primera escritura: CSV + .schema con spec.schema_version."""
    from src.pipeline import output_writer as ow
    from src.pipeline import output_registry as oreg

    target = tmp_path / "test_output.csv"
    monkeypatch.setattr(ow, "get_spec", lambda name: oreg.OutputSpec(
        name=name, path=str(target), kind="temporal",
        date_column="date", merge_keys=("date", "sector"),
        schema_version=2,
    ))

    ow.persist("test_output", _make_df())
    assert target.exists()
    schema_f = target.with_suffix(target.suffix + ".schema")
    assert schema_f.exists()
    assert schema_f.read_text().strip() == "2"


def test_persist_append_mismo_schema(tmp_path, monkeypatch):
    """Segunda escritura mismo schema: append sin migracion."""
    from src.pipeline import output_writer as ow
    from src.pipeline import output_registry as oreg

    target = tmp_path / "test_output.csv"
    spec = oreg.OutputSpec(
        name="test_output", path=str(target), kind="temporal",
        date_column="date", merge_keys=("date", "sector"),
        schema_version=2,
    )
    monkeypatch.setattr(ow, "get_spec", lambda name: spec)

    ow.persist("test_output", _make_df("2026-10-23"))
    ow.persist("test_output", _make_df("2026-10-24"))

    df = pd.read_csv(target)
    assert len(df) == 4
    schema_f = target.with_suffix(target.suffix + ".schema")
    assert schema_f.read_text().strip() == "2"


def test_persist_migra_schema_distinto(tmp_path, monkeypatch):
    """CSV con schema viejo, spec con schema nuevo: renombra y arranca."""
    from src.pipeline import output_writer as ow
    from src.pipeline import output_registry as oreg

    target = tmp_path / "test_output.csv"
    # Simular CSV v1 existente
    pd.DataFrame({
        "date": ["2026-10-20"], "sector": ["XLB"], "value": [9.9],
    }).to_csv(target, index=False)
    schema_f = target.with_suffix(target.suffix + ".schema")
    schema_f.write_text("1")

    spec_v2 = oreg.OutputSpec(
        name="test_output", path=str(target), kind="temporal",
        date_column="date", merge_keys=("date", "sector"),
        schema_version=2,
    )
    monkeypatch.setattr(ow, "get_spec", lambda name: spec_v2)

    ow.persist("test_output", _make_df("2026-10-23"))

    # CSV nuevo solo con datos v2
    df_nuevo = pd.read_csv(target)
    assert len(df_nuevo) == 2
    assert all(df_nuevo["date"] == "2026-10-23")
    # CSV viejo renombrado
    migrated = target.with_name("test_output.v1.csv")
    assert migrated.exists()
    df_viejo = pd.read_csv(migrated)
    assert len(df_viejo) == 1
    assert df_viejo.iloc[0]["date"] == "2026-10-20"
    # schema actualizado
    assert schema_f.read_text().strip() == "2"


def test_persist_legacy_sin_schema_file(tmp_path, monkeypatch):
    """CSV sin .schema: asume v1 y, si spec=v2, migra."""
    from src.pipeline import output_writer as ow
    from src.pipeline import output_registry as oreg

    target = tmp_path / "test_output.csv"
    pd.DataFrame({
        "date": ["2026-10-20"], "sector": ["XLB"], "value": [9.9],
    }).to_csv(target, index=False)
    # sin .schema

    spec_v2 = oreg.OutputSpec(
        name="test_output", path=str(target), kind="temporal",
        date_column="date", merge_keys=("date", "sector"),
        schema_version=2,
    )
    monkeypatch.setattr(ow, "get_spec", lambda name: spec_v2)

    ow.persist("test_output", _make_df("2026-10-23"))

    df_nuevo = pd.read_csv(target)
    assert len(df_nuevo) == 2
    migrated = target.with_name("test_output.v1.csv")
    assert migrated.exists()


def test_sector_wyckoff_distribution_schema_2():
    """El spec de sector_wyckoff_distribution tiene schema_version=2."""
    from src.pipeline.output_registry import get_spec
    spec = get_spec("sector_wyckoff_distribution")
    assert getattr(spec, "schema_version", 1) == 2


def test_resto_outputs_schema_1():
    """Los demas outputs temporal mantienen schema_version=1 por defecto."""
    from src.pipeline.output_registry import REGISTRY
    for spec in REGISTRY:
        if spec.name == "sector_wyckoff_distribution":
            continue
        if getattr(spec, "kind", None) != "temporal":
            continue
        assert getattr(spec, "schema_version", 1) == 1, f"{spec.name} schema != 1"
