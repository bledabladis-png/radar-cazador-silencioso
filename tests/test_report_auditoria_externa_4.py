# -*- coding: utf-8 -*-
"""Auditoria externa #4 (2026-10-22): notas P1, P2, P6, P8.

- P1: rs_z del WLS deriva de rs_mom (invariante a escala), no de rs nominal.
- P2: denominador de cobertura = universo completo del ETF, no top-20.
- P6: temporal_coherence valida solo stock_prices y market_data.
- P8: N-PORT puede incluir posiciones residuales de filings historicos.
"""
import inspect

import pandas as pd

from src.report.provenance import render_provenance
from src.report.flows_international import render_flujo_posicional_nport


def test_p6_temporal_coherence_scope():
    out = render_provenance({
        "commit_sha": "abc12345",
        "run_id": "12345",
        "run_attempt": "1",
        "target_session": "2026-10-20",
        "reference_date": "2026-10-21T03:38:17+02:00",
        "effective_date_equity": "2026-10-20",
        "validation_gate": "10/10",
        "temporal_coherence": "OK",
        "generated_at": "2026-10-21T01:51:06Z",
    })
    body = "".join(out)
    assert "temporal_coherence" in body
    assert "stock_prices" in body
    assert "market_data" in body
    assert "Data Freshness" in body


def test_p8_nport_residual_note():
    df = pd.DataFrame([{
        "REPORT_DATE": pd.Timestamp("2026-06-30"),
        "FILING_DATE": pd.Timestamp("2026-08-28"),
        "REGISTRANT_NAME": "SELECT SECTOR SPDR TRUST",
        "ISSUER_NAME": "Norwegian Cruise Line Holdings Ltd",
        "IDENTIFIER_ISIN": "BMG667211046",
        "PREV_BALANCE": 2429969.0,
        "BALANCE": 2580423.0,
        "POSITION_CHANGE": 150454.0,
        "POSITION_CHANGE_PCT": 6.19,
    }])
    body = "".join(render_flujo_posicional_nport(df))
    assert "residuales" in body.lower()
    assert "legados" in body.lower()


def test_p2_coverage_note_in_source():
    from src.report import sectorial
    src = inspect.getsource(sectorial)
    assert "universo completo del ETF" in src
    assert "n_total" in src
