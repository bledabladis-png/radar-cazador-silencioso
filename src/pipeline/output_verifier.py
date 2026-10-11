# -*- coding: utf-8 -*-
"""Verificador post-run de cobertura temporal de outputs.

Recorre el registro y comprueba, contra los receipts del run y el
estado real en disco, que cada output requerido se ha escrito con la
fecha declarada o se ha saltado con una razon legitima.

Reglas:
  R1. Cada entrada del registro tiene un receipt.
  R2. Receipt written=True -> CSV existe y tiene fila con la fecha
      declarada (kind='snapshot': solo existe).
  R3. Receipt written=False -> razon legitima Y CSV existe no vacio.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pipeline.output_registry import REGISTRY, LEGITIMATE_SKIP_REASONS


def _read_dates(path, column):
    try:
        df = pd.read_csv(path, usecols=[column])
    except Exception:
        return None
    return pd.to_datetime(df[column], errors='coerce').dropna()


def verify(collector, registry=REGISTRY):
    """Verifica el estado de los outputs contra los receipts del run.

    Returns:
        dict: {'passed': bool, 'errors': list[str], 'checks': list[str]}
    """
    receipts = collector.by_name()
    errors = []
    checks = []

    for spec in registry:
        receipt = receipts.get(spec.name)

        if receipt is None:
            errors.append(spec.name + ": sin receipt en el run")
            continue

        path = Path(spec.path)

        if not receipt.written:
            if receipt.reason not in LEGITIMATE_SKIP_REASONS:
                errors.append(
                    spec.name + ": skip con razon no legitima: "
                    + repr(receipt.reason)
                )
                continue
            if not path.exists():
                errors.append(
                    spec.name + ": skip (" + receipt.reason
                    + ") pero " + spec.path + " no existe"
                )
                continue
            try:
                df = pd.read_csv(path)
            except Exception as e:
                errors.append(spec.name + ": CSV ilegible: " + str(e))
                continue
            if df.empty:
                errors.append(
                    spec.name + ": skip (" + receipt.reason
                    + ") pero " + spec.path + " esta vacio"
                )
                continue
            checks.append(
                spec.name + ": OK skip=" + receipt.reason
                + " (" + str(len(df)) + " filas)"
            )
            continue

        if spec.kind == 'snapshot':
            if not path.exists():
                errors.append(
                    spec.name + ": escrito pero " + spec.path + " no existe"
                )
                continue
            checks.append(spec.name + ": OK snapshot")
            continue

        if not path.exists():
            errors.append(
                spec.name + ": escrito pero " + spec.path + " no existe"
            )
            continue

        if receipt.date:
            dates = _read_dates(path, spec.date_column)
            if dates is None or dates.empty:
                errors.append(
                    spec.name + ": sin columna "
                    + repr(spec.date_column) + " legible"
                )
                continue
            target = pd.Timestamp(receipt.date)
            if target not in set(dates):
                errors.append(
                    spec.name + ": receipt declara " + str(receipt.date)
                    + " pero " + spec.path + " no tiene esa fecha"
                    + " (max=" + str(dates.max().date()) + ")"
                )
                continue
        checks.append(spec.name + ": OK")

    return {'passed': len(errors) == 0, 'errors': errors, 'checks': checks}
