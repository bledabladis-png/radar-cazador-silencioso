# -*- coding: utf-8 -*-
"""Wrapper unico de persistencia de outputs historicos.

Sustituye el patron read_csv -> append_dedup -> to_csv(.tmp) -> replace
duplicado en los writers del pipeline. Toda escritura a outputs/history/
pasa por persist() o skip(). Emite WriteReceipt para verificacion
post-run.

path_override: usado por tests para redirigir la escritura a un path
temporal. Cuando se pasa, el receipt usa ese path y NO se acumula en el
collector activo (los paths temporales no son parte del registro).

Collector activo: run.py llama set_collector(c) al inicio. Los writers
invocan persist(name, df) sin pasar collector y el receipt se acumula
automaticamente. Si no hay collector activo, funciona igual sin acumular.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.utils import append_dedup
from src.pipeline.output_registry import (
    get_spec, WriteReceipt, LEGITIMATE_SKIP_REASONS,
)


_active_collector = None


def set_collector(collector):
    global _active_collector
    _active_collector = collector


def get_collector():
    return _active_collector


def clear_collector():
    global _active_collector
    _active_collector = None


def _resolve_collector(collector):
    if collector is not None:
        return collector
    return _active_collector


def _fmt_date(value):
    if value is None:
        return None
    return pd.Timestamp(value).strftime('%Y-%m-%d')


def persist(name, df, *, date_value=None, sort_by=None, collector=None,
            path_override=None, replace_date=False):
    """Persiste df en el CSV del registro.

    Args:
        name: identificador en el registro (KeyError si no existe).
        df: DataFrame a persistir. Vacio/None -> skip EMPTY_DF.
        date_value: fecha de observacion (Timestamp/str/date).
        sort_by: columnas de ordenacion tras dedup (opcional).
        collector: ReceiptCollector opcional. Si None, usa el activo.
        path_override: Path alternativo (tests). Si se pasa, escribe ahi
                       y NO acumula receipt en collector.

    Returns:
        WriteReceipt
    """
    spec = get_spec(name)
    override = path_override is not None
    path = Path(path_override) if override else Path(spec.path)
    coll = None if override else _resolve_collector(collector)

    if df is None or len(df) == 0:
        return skip(name, reason='EMPTY_DF', collector=coll,
                    path_override=path_override)

    # Derivar date_value del propio df si no se pasa y hay columna.
    # Si no hay columna, se escribe igual con receipt date=None
    # (caso tests / df sin columna fecha).
    if date_value is None and spec.kind == 'temporal' and spec.date_column:
        if spec.date_column in df.columns:
            _d = df[spec.date_column].dropna()
            if len(_d) > 0:
                date_value = _d.max()

    path.parent.mkdir(parents=True, exist_ok=True)

    # --- Schema versioning (Fase 1) ---
    # Detecta la version del CSV existente y actua:
    #  - Sin CSV previo: escribe normal, anota .schema con spec.schema_version.
    #  - CSV con mismo schema: append normal.
    #  - CSV con schema distinto: renombra a .v{old}.csv, borra .schema,
    #    arranca CSV nuevo. Evita append ciego entre schemas incompatibles.
    schema_file = path.with_suffix(path.suffix + ".schema")
    _current_schema = getattr(spec, "schema_version", 1)
    _migrated_this_call = False
    if path.exists():
        if schema_file.exists():
            try:
                _existing = int(schema_file.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                _existing = 1
        else:
            _existing = 1
        if _existing != _current_schema:
            # Migracion: COPIAR CSV a .v{old}.csv (backup). El original
            # NO se toca hasta que la escritura atomica (tmp -> replace)
            # tenga exito. Preserva la garantia de "si to_csv falla, el
            # CSV viejo sigue intacto".
            _migrated_name = path.with_name(
                path.stem + ".v" + str(_existing) + path.suffix
            )
            try:
                # NO sobrescribir backup existente con contenido (preserva
                # historico). Solo crear si no existe o esta vacio.
                _should_copy = True
                if _migrated_name.exists():
                    try:
                        _backup_lines = sum(1 for _ in _migrated_name.open("r", encoding="utf-8", errors="ignore"))
                    except OSError:
                        _backup_lines = 0
                    if _backup_lines > 1:
                        _should_copy = False
                        print(f"  [SCHEMA] {name}: backup v{_existing} ya existe ({_backup_lines} lineas), no se sobrescribe")
                if _should_copy:
                    if _migrated_name.exists():
                        _migrated_name.unlink()
                    shutil.copy2(path, _migrated_name)
                    print(f"  [SCHEMA] {name}: backup v{_existing} -> {_migrated_name.name} (v{_current_schema} activo)")
                _migrated_this_call = True
            except OSError as _e:
                # FAIL-CLOSED (2026-10-31, Fase 2 B5.1): antes se
                # silenciaba y continuaba como append normal. Bug real
                # detectado por test_output_writer_schema.py::test_t5:
                # el CSV quedaba mezclando v{_existing} con v{_current_schema}
                # (schemas incompatibles). Receipt fail-closed del
                # sistema exige abortar, no mezclar.
                raise OSError(
                    f"output_writer: migracion schema de {name} fallo "
                    f"(v{_existing} -> v{_current_schema}). Backup "
                    f"{_migrated_name.name}: {_e}. CSV y .schema viejos "
                    f"intactos. Revisar permisos o disco."
                ) from _e
            # NO borrar .schema aqui. Se sobreescribe al final si el
            # write tiene exito.
        # si _existing == _current_schema: append normal

    if spec.kind == 'temporal' and spec.merge_keys:
        if path.exists() and not _migrated_this_call:
            hist = pd.read_csv(path)
            # replace_date (2026-10-26): elimina del historico TODAS las
            # filas de la fecha que se va a reescribir, antes del append.
            # Necesario para writers cuyo df es un SUBCONJUNTO filtrado
            # del estado del dia (ej. wyckoff_*_distribution solo incluye
            # DIST_PHASES). Sin esto, un ticker que sale de la fase deja
            # su fila vieja huerfana en el CSV (append_dedup solo añade/
            # reemplaza, no elimina). Ver dictamen externo 2026-10-26
            # P0-H06 y bug detectado con GE/PWR/VRT en XLI.
            if replace_date and spec.date_column:
                _dv = None
                if date_value is not None:
                    _dv = pd.Timestamp(date_value).strftime('%Y-%m-%d')
                elif spec.date_column in df.columns:
                    _d = df[spec.date_column].dropna()
                    if len(_d) > 0:
                        _dv = pd.Timestamp(_d.max()).strftime('%Y-%m-%d')
                if _dv is not None and spec.date_column in hist.columns:
                    _n_before = len(hist)
                    hist = hist[
                        hist[spec.date_column].astype(str) != _dv
                    ].copy()
                    _removed = _n_before - len(hist)
                    if _removed > 0:
                        print(f"  [PERSIST] {name}: replace_date {_dv}, "
                              f"retiradas {_removed} filas previas")
            df_out = append_dedup(
                hist, df, list(spec.merge_keys),
                sort_by=list(sort_by) if sort_by else None,
            )
        else:
            # Sin historico (primer run) o migracion (no mezclar v1 con v2).
            # El historico previo ya esta en el backup .v{old}.csv.
            df_out = df.copy()
    else:
        df_out = df.copy()

    tmp = path.with_suffix(path.suffix + '.tmp')
    df_out.to_csv(tmp, index=False, encoding='utf-8', lineterminator='\n')
    tmp.replace(path)
    try:
        path.with_suffix(path.suffix + '.schema').write_text(str(_current_schema), encoding='utf-8')
    except OSError as _e:
        # OBSERVABILIDAD (2026-10-31): antes silencioso. Si falla escribir
        # .schema, el proximo run lee default 1 y trata el CSV v2 como v1
        # -> migracion falsa positiva. El CSV queda correcto pero el
        # .schema miente.
        print(f"  [WARN] output_writer: no se pudo escribir .schema de "
              f"{name} ({type(_e).__name__}: {_e}). El proximo run asumira "
              f"schema_version=1 y migrara en falso.")

    receipt = WriteReceipt(
        name=name, written=True, reason='OK',
        date=_fmt_date(date_value), path=str(path), rows=len(df_out),
    )
    if coll is not None:
        coll.add(receipt)
    return receipt


def skip(name, reason, *, collector=None, path_override=None):
    """Emite un receipt sin escritura. Razon debe ser legitima.

    Si path_override se pasa, no acumula en collector.
    """
    spec = get_spec(name)
    if reason not in LEGITIMATE_SKIP_REASONS:
        raise ValueError("Razon de skip no legitima: " + repr(reason))
    override = path_override is not None
    coll = None if override else _resolve_collector(collector)
    receipt_path = str(path_override) if override else spec.path
    receipt = WriteReceipt(
        name=name, written=False, reason=reason,
        date=None, path=receipt_path, rows=0,
    )
    if coll is not None:
        coll.add(receipt)
    return receipt
