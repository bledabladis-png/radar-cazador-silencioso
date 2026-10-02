"""Tests P4 (2026-10-02): monotonia del catalogo.

Cierra P4 como no-deuda + defensa preventiva. Si algun dia el catalogo
encoge (regenerate cambia de semantica), merge_assignments avisa.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import build_catalog_csvs as gen


def _mk_asg(rows):
    return pd.DataFrame(rows, columns=list(gen.ASSIGNMENT_COLUMNS))


def _mk_snap(rows):
    return pd.DataFrame(rows, columns=["radar_ticker", "figi", "share_class_figi"])


def test_monotono_sin_warn(capsys):
    """Sin orphans: merge_assignments no imprime WARN."""
    asg = _mk_asg([
        {"catalog_key": "radar_20260101_0001", "assigned_entity_id": "radar_entity_0001",
         "radar_ticker": "AAA", "share_class_figi": "FIGI_A",
         "valid_from": "20260101", "valid_to": "", "source": "t", "reason": "t"},
    ])
    snap = _mk_snap([["AAA", "BBG_A", "FIGI_A"]])
    out = gen.merge_assignments(asg, snap, "20260201", "20260201")
    assert len(out) == 1
    captured = capsys.readouterr()
    assert "[WARN] merge_assignments" not in captured.out


def test_orphan_emite_warn(capsys):
    """Un figi de asg_prev ausente en snap: WARN con la key huerfana."""
    asg = _mk_asg([
        {"catalog_key": "radar_20260101_0001", "assigned_entity_id": "radar_entity_0001",
         "radar_ticker": "AAA", "share_class_figi": "FIGI_A",
         "valid_from": "20260101", "valid_to": "", "source": "t", "reason": "t"},
        {"catalog_key": "radar_20260101_0002", "assigned_entity_id": "radar_entity_0002",
         "radar_ticker": "BBB", "share_class_figi": "FIGI_B",
         "valid_from": "20260101", "valid_to": "", "source": "t", "reason": "t"},
    ])
    # snap solo tiene AAA: BBB (FIGI_B) queda huerfano
    snap = _mk_snap([["AAA", "BBG_A", "FIGI_A"]])
    out = gen.merge_assignments(asg, snap, "20260201", "20260201")
    # La fila huerfana se preserva (append-only)
    assert len(out) == 2
    assert set(out["radar_ticker"]) == {"AAA", "BBB"}
    captured = capsys.readouterr()
    assert "[WARN] merge_assignments" in captured.out
    assert "radar_20260101_0002" in captured.out