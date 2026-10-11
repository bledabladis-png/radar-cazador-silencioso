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
            path_override=None):
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

    if spec.kind == 'temporal' and spec.merge_keys:
        if path.exists():
            hist = pd.read_csv(path)
            df_out = append_dedup(
                hist, df, list(spec.merge_keys),
                sort_by=list(sort_by) if sort_by else None,
            )
        else:
            df_out = df.copy()
    else:
        df_out = df.copy()

    tmp = path.with_suffix(path.suffix + '.tmp')
    df_out.to_csv(tmp, index=False, encoding='utf-8', lineterminator='\n')
    tmp.replace(path)

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
