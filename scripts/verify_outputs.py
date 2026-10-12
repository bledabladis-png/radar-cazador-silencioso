# -*- coding: utf-8 -*-
"""Verificacion extendida de outputs historicos.

Lee el registro declarativo (src/pipeline/output_registry.py) y comprueba,
para cada output temporal excepto los declarados calendar (data_quality),
que las ultimas N sesiones NYSE tienen fila en el CSV.

Uso:
    py scripts/verify_outputs.py                       # N=10, since=CONTRACT_DEPLOY
    py scripts/verify_outputs.py --window 30
    py scripts/verify_outputs.py --since 2026-09-01
    py scripts/verify_outputs.py --json
    py scripts/verify_outputs.py --ref 2026-10-12

Exit 0 si todo OK. Exit 1 si hay faltantes.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.output_registry import REGISTRY
from src.market_calendar import last_expected_market_date, previous_market_day

# Outputs excluidos del check extendido: su presencia depende de runs
# del pipeline, no de sesion bursatil. Su cobertura se verifica en
# output_verifier.verify() con la fecha del run.
EXCLUDED_FROM_HISTORICAL = frozenset({'data_quality'})

# Fecha por defecto desde la que auditar. Antes de esta fecha los CSVs
# sobreescribian (no acumulaban), por lo que gaps previos son deuda
# historica, no regresion.
DEFAULT_SINCE = '2026-10-11'


def _expected_sessions(reference_date, n, since):
    d = last_expected_market_date(reference_date)
    sessions = [d]
    for _ in range(n - 1):
        d = previous_market_day(d)
        if since and d < since:
            break
        sessions.append(d)
    return sessions


def _read_dates(path, column):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=[column])
    except Exception:
        return None
    return set(pd.to_datetime(df[column], errors='coerce').dropna().dt.date)


def verify_historical(window=10, reference_date=None, since=None,
                      registry=REGISTRY):
    if reference_date is None:
        reference_date = datetime.now()
    if since is None:
        since = datetime.strptime(DEFAULT_SINCE, '%Y-%m-%d').date()
    results = []
    for spec in registry:
        if spec.kind != 'temporal':
            continue
        if spec.name in EXCLUDED_FROM_HISTORICAL:
            continue
        path = PROJECT_ROOT / spec.path
        expected = _expected_sessions(reference_date, window, since)
        dates = _read_dates(path, spec.date_column)
        if dates is None:
            results.append({
                'name': spec.name, 'path': spec.path,
                'status': 'MISSING_FILE',
                'missing': [str(s) for s in expected],
                'present': 0, 'expected': len(expected),
            })
            continue
        missing = [s for s in expected if s not in dates]
        results.append({
            'name': spec.name, 'path': spec.path,
            'status': 'OK' if not missing else 'GAP',
            'missing': [str(s) for s in missing],
            'present': len(expected) - len(missing),
            'expected': len(expected),
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=int, default=10)
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--ref', type=str, default=None,
                    help='YYYY-MM-DD (default: ahora)')
    ap.add_argument('--since', type=str, default=DEFAULT_SINCE,
                    help='YYYY-MM-DD (default: ' + DEFAULT_SINCE + ')')
    args = ap.parse_args()

    ref = None
    if args.ref:
        ref = datetime.strptime(args.ref, '%Y-%m-%d')
    since = datetime.strptime(args.since, '%Y-%m-%d').date()

    results = verify_historical(args.window, ref, since)

    if args.json:
        print(json.dumps({'window': args.window, 'since': str(since),
                          'results': results}, indent=2))
    else:
        print(f"Verificacion extendida de outputs "
              f"(ventana={args.window} sesiones, since={since})")
        print()
        print(f"{'output':<32} {'estado':<15} {'presentes/esperadas':<22} faltantes")
        print("-" * 100)
        for r in results:
            falt = ','.join(r['missing'][:3])
            if len(r['missing']) > 3:
                falt += f"... (+{len(r['missing'])-3})"
            print(f"{r['name']:<32} {r['status']:<15} "
                  f"{r['present']}/{r['expected']:<20} {falt}")
        print()

    has_gap = any(r['status'] != 'OK' for r in results)
    sys.exit(1 if has_gap else 0)


if __name__ == '__main__':
    main()
