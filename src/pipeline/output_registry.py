# -*- coding: utf-8 -*-
"""Registro declarativo de outputs persistidos por el pipeline.

Contrato: todo writer que persiste a outputs/history/ tiene entrada aqui.
Sin entrada, no hay persistencia (KeyError fail-closed).

Ver 01_METODO seccion "Contrato de persistencia de outputs".
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OutputSpec:
    name: str
    path: str
    kind: str          # 'temporal' | 'snapshot'
    date_column: str   # '' si kind='snapshot'
    merge_keys: tuple  # () si no hay merge con historico


REGISTRY: tuple = (
    OutputSpec('sector_breadth',
               'outputs/history/sector_breadth.csv',
               'temporal', 'date', ('date', 'sector')),
    OutputSpec('sector_breadth_momentum',
               'outputs/history/sector_breadth_momentum.csv',
               'temporal', 'date', ('date', 'sector')),
    OutputSpec('sector_flow_characteristics',
               'outputs/history/sector_flow_characteristics.csv',
               'temporal', 'date', ('date', 'sector')),
    OutputSpec('sector_persistence',
               'outputs/history/sector_persistence.csv',
               'temporal', 'date', ('date', 'sector')),
    OutputSpec('sector_regime_matrix',
               'outputs/history/sector_regime_matrix.csv',
               'temporal', 'date', ('date',)),
    OutputSpec('evidence_matrix',
               'outputs/history/evidence_matrix.csv',
               'temporal', 'date', ('date',)),
    OutputSpec('macro_regime',
               'outputs/history/macro_regime.csv',
               'temporal', 'date', ('date',)),
    OutputSpec('sector_rank_deltas',
               'outputs/history/sector_rank_deltas.csv',
               'snapshot', '', ()),
    OutputSpec('sector_dispersion',
               'outputs/history/sector_dispersion.csv',
               'temporal', 'date', ('date',)),
    OutputSpec('sector_correlation_summary',
               'outputs/history/sector_correlation_summary.csv',
               'temporal', 'date', ('date', 'window')),
    OutputSpec('cross_asset_correlation',
               'outputs/history/cross_asset_correlation.csv',
               'temporal', 'date',
               ('date', 'window', 'sector', 'asset')),
    OutputSpec('cross_asset_context',
               'outputs/history/cross_asset_context.csv',
               'temporal', 'date',
               ('date', 'window', 'sector', 'asset_class')),
    OutputSpec('sector_leader_divergence',
               'outputs/history/sector_leader_divergence.csv',
               'temporal', 'date', ('date', 'sector')),
    OutputSpec('sector_wyckoff_distribution',
               'outputs/history/sector_wyckoff_distribution.csv',
               'temporal', 'date', ('date', 'sector')),
    OutputSpec('rs_internal',
               'outputs/history/rs_internal.csv',
               'temporal', 'date', ('date', 'sector', 'ticker')),
    OutputSpec('sector_concentration',
               'outputs/history/sector_concentration.csv',
               'temporal', 'date', ('date', 'sector')),
    OutputSpec('leader_representativeness',
               'outputs/history/leader_representativeness.csv',
               'temporal', 'date', ('date', 'sector', 'ticker')),
    OutputSpec('volatility_structure',
               'outputs/history/volatility_structure.csv',
               'temporal', 'date', ('date',)),
    OutputSpec('data_quality',
               'outputs/history/data_quality.csv',
               'temporal', 'date', ('date', 'source')),
)

REGISTRY_BY_NAME = {s.name: s for s in REGISTRY}


def get_spec(name):
    if name not in REGISTRY_BY_NAME:
        raise KeyError("Output no registrado: " + repr(name))
    return REGISTRY_BY_NAME[name]


LEGITIMATE_SKIP_REASONS = frozenset({
    'STALE', 'NO_DATA', 'SKIP_CONDITION', 'EMPTY_DF',
})


@dataclass
class WriteReceipt:
    name: str
    written: bool
    reason: str
    date: object
    path: str
    rows: int = 0

    def to_dict(self):
        return {
            'name': self.name,
            'written': bool(self.written),
            'reason': self.reason,
            'date': str(self.date) if self.date is not None else None,
            'path': self.path,
            'rows': int(self.rows),
        }


class ReceiptCollector:
    def __init__(self):
        self._receipts = []

    def add(self, receipt):
        self._receipts.append(receipt)

    def receipts(self):
        return list(self._receipts)

    def by_name(self):
        return {r.name: r for r in self._receipts}

    def to_dict(self):
        return {
            'schema_version': 1,
            'receipts': [r.to_dict() for r in self._receipts],
        }
