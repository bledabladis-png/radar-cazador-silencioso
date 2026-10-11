# -*- coding: utf-8 -*-
"""Wrapper unico de persistencia de outputs historicos.

Sustituye el patron read_csv -> append_dedup -> to_csv(.tmp) -> replace
duplicado en los writers del pipeline. Toda escritura a outputs/history/
pasa por persist() o skip(). Emite WriteReceipt para verificacion
post-run.

El wrapper no decide la fecha de observacion: la calcula el writer
(usando writer_observation_date / last_expected_market_date) y la pasa
como date_value. El wrapper solo escribe y emite el receipt.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import append_dedup
from src.pipeline.output_registry import (
    get_spec, WriteReceipt, LEGITIMATE_SKIP_REASONS,
)


def _fmt_date(value):
    if value is None:
        return None
    return pd.Timestamp(value).strftime('%Y-%m-%d')


def persist(name, df, *, date_value=None, sort_by=None, collector=None):
    """Persiste df en el CSV del registro.

    Args:
        name: identificador en el registro (KeyError si no existe).
        df: DataFrame a persistir. Vacio/None -> skip EMPTY_DF.
        date_value: fecha de observacion (Timestamp/str/date).
        sort_by: columnas de ordenacion tras dedup (opcional).
        collector: ReceiptCollector opcional.

    Returns:
        WriteReceipt
    """
    spec = get_spec(name)
    path = Path(spec.path)

    if df is None or len(df) == 0:
        return skip(name, reason='EMPTY_DF', collector=collector)

    if spec.kind == 'temporal' and spec.date_column:
        if spec.date_column not in df.columns:
            return skip(name, reason='NO_DATA', collector=collector)

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
    if collector is not None:
        collector.add(receipt)
    return receipt


def skip(name, reason, *, collector=None):
    """Emite un receipt sin escritura. Razon debe ser legitima."""
    spec = get_spec(name)
    if reason not in LEGITIMATE_SKIP_REASONS:
        raise ValueError("Razon de skip no legitima: " + repr(reason))
    receipt = WriteReceipt(
        name=name, written=False, reason=reason,
        date=None, path=spec.path, rows=0,
    )
    if collector is not None:
        collector.add(receipt)
    return receipt
