# -*- coding: utf-8 -*-
"""Tests de migracion de schema en output_writer.persist (Fase 2 B5.1).

Cubre la clase de bug 1fb791ab aplicada al writer:
codigo de migracion sin tests de integracion real. Antes de este
fichero, `tests/test_output_writer.py` cubria append/dedup/skip pero
NO tocaba la rama schema_file / _migrated_this_call / backup .vN.csv.

Riesgo cubierto: si la migracion falla, el CSV puede quedar con dos
schemas mezclados o perder historia.

T1. Sin CSV previo -> escribe, .schema = current.
T2. CSV con mismo schema -> append, sin backup.
T3. CSV v1, spec v2 -> backup .v1.csv creado, .schema = 2.
T4. Backup existente con contenido -> no se sobrescribe.
T5. Backup falla (shutil.copy2 raising) -> CSV viejo intacto, .schema
    no se actualiza, WARN emitido.
T6. to_csv falla -> CSV viejo intacto, .schema no se actualiza.
"""
import pandas as pd
import pytest

from src.pipeline import output_writer


def _write_csv_with_schema(path, schema, df):
    """Crea CSV + .schema con contenido."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    path.with_suffix(path.suffix + ".schema").write_text(
        str(schema), encoding="utf-8"
    )


# Spec temporal de prueba. Usa el registro real (sector_breadth, schema=1).
# Para T3 necesita un spec con schema_version=2. Uso sector_regime_matrix
# que ya esta en v2 en el registro.

def test_t1_sin_csv_previo_escribe_normal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = pd.DataFrame({"date": ["2026-10-10"], "sector": ["XLK"]})
    r = output_writer.persist(
        "sector_regime_matrix", df, date_value="2026-10-10"
    )
    assert r.written is True
    csv = tmp_path / "outputs/history/sector_regime_matrix.csv"
    schema = tmp_path / "outputs/history/sector_regime_matrix.csv.schema"
    assert csv.exists()
    assert schema.read_text(encoding="utf-8").strip() == "2"


def test_t2_mismo_schema_append_sin_backup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "outputs/history"
    base.mkdir(parents=True)
    df1 = pd.DataFrame({"date": ["2026-10-09"], "sector": ["XLK"]})
    _write_csv_with_schema(
        base / "sector_regime_matrix.csv", 2, df1
    )
    df2 = pd.DataFrame({"date": ["2026-10-10"], "sector": ["XLK"]})
    output_writer.persist(
        "sector_regime_matrix", df2, date_value="2026-10-10"
    )
    out = pd.read_csv(base / "sector_regime_matrix.csv")
    assert len(out) == 2
    # Sin backup .v2.csv (mismo schema)
    assert not (base / "sector_regime_matrix.v2.csv").exists()


def test_t3_migracion_v1_a_v2_crea_backup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "outputs/history"
    base.mkdir(parents=True)
    df_old = pd.DataFrame({
        "date": ["2026-09-01", "2026-09-02"],
        "sector": ["XLK", "XLF"],
    })
    _write_csv_with_schema(
        base / "sector_regime_matrix.csv", 1, df_old
    )
    df_new = pd.DataFrame({"date": ["2026-10-10"], "sector": ["XLK"]})
    output_writer.persist(
        "sector_regime_matrix", df_new, date_value="2026-10-10"
    )
    # Backup con el historico v1
    backup = base / "sector_regime_matrix.v1.csv"
    assert backup.exists()
    backup_df = pd.read_csv(backup)
    assert len(backup_df) == 2
    # CSV activo tiene SOLO df_new (no se mezcla v1 con v2)
    out = pd.read_csv(base / "sector_regime_matrix.csv")
    assert len(out) == 1
    assert out["date"].iloc[0] == "2026-10-10"
    # .schema actualizado a 2
    schema = base / "sector_regime_matrix.csv.schema"
    assert schema.read_text(encoding="utf-8").strip() == "2"


def test_t4_backup_existente_no_se_sobrescribe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "outputs/history"
    base.mkdir(parents=True)
    df_old = pd.DataFrame({"date": ["2026-09-01"], "sector": ["XLK"]})
    _write_csv_with_schema(
        base / "sector_regime_matrix.csv", 1, df_old
    )
    # Backup previo con contenido
    backup = base / "sector_regime_matrix.v1.csv"
    backup_df = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "sector": ["XLB", "XLE", "XLF"],
    })
    backup_df.to_csv(backup, index=False)
    contenido_previo = backup.read_text(encoding="utf-8")

    df_new = pd.DataFrame({"date": ["2026-10-10"], "sector": ["XLK"]})
    output_writer.persist(
        "sector_regime_matrix", df_new, date_value="2026-10-10"
    )
    # Backup NO se sobrescribe
    assert backup.read_text(encoding="utf-8") == contenido_previo


def test_t5_backup_falla_aborta_fail_closed(tmp_path, monkeypatch):
    """Migracion v1->v2 con backup fallido -> OSError, CSV y .schema intactos.

    Contrato fail-closed (2026-10-31, Fase 2 B5.1): antes el codigo
    silenciaba el fallo de shutil.copy2 y continuaba como append
    normal, mezclando v1 con v2 en el CSV. Receipt fail-closed del
    sistema exige abortar.
    """
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "outputs/history"
    base.mkdir(parents=True)
    df_old = pd.DataFrame({
        "date": ["2026-09-01"], "sector": ["XLK"],
    })
    _write_csv_with_schema(
        base / "sector_regime_matrix.csv", 1, df_old
    )
    contenido_csv_previo = (
        base / "sector_regime_matrix.csv"
    ).read_text(encoding="utf-8")
    schema_previo = (
        base / "sector_regime_matrix.csv.schema"
    ).read_text(encoding="utf-8")

    # Mock: shutil.copy2 falla con OSError
    def _fail_copy(*args, **kwargs):
        raise OSError("fallo simulado en shutil.copy2")
    monkeypatch.setattr(output_writer.shutil, "copy2", _fail_copy)

    df_new = pd.DataFrame({"date": ["2026-10-10"], "sector": ["XLK"]})
    with pytest.raises(OSError, match="migracion schema"):
        output_writer.persist(
            "sector_regime_matrix", df_new, date_value="2026-10-10"
        )
    # CSV viejo INTACTO (no se escribio df_new)
    assert (
        base / "sector_regime_matrix.csv"
    ).read_text(encoding="utf-8") == contenido_csv_previo
    # .schema sigue en 1 (no se actualizo)
    assert (
        base / "sector_regime_matrix.csv.schema"
    ).read_text(encoding="utf-8") == schema_previo
    # Backup NO existe (nunca se creo)
    assert not (base / "sector_regime_matrix.v1.csv").exists()


def test_t6_to_csv_falla_csv_intacto(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "outputs/history"
    base.mkdir(parents=True)
    df_old = pd.DataFrame({"date": ["2026-09-01"], "sector": ["XLK"]})
    _write_csv_with_schema(
        base / "sector_regime_matrix.csv", 2, df_old
    )
    contenido_csv_previo = (
        base / "sector_regime_matrix.csv"
    ).read_text(encoding="utf-8")

    # Mock: DataFrame.to_csv falla
    original_to_csv = pd.DataFrame.to_csv

    def _fail_to_csv(self, *args, **kwargs):
        raise OSError("fallo simulado en to_csv")
    monkeypatch.setattr(pd.DataFrame, "to_csv", _fail_to_csv)

    df_new = pd.DataFrame({"date": ["2026-10-10"], "sector": ["XLK"]})
    with pytest.raises(OSError, match="fallo simulado en to_csv"):
        output_writer.persist(
            "sector_regime_matrix", df_new, date_value="2026-10-10"
        )
    # CSV viejo intacto
    assert (
        base / "sector_regime_matrix.csv"
    ).read_text(encoding="utf-8") == contenido_csv_previo
    # Restaurar
    monkeypatch.setattr(pd.DataFrame, "to_csv", original_to_csv)
