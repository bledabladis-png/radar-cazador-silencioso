# -*- coding: utf-8 -*-
"""P2 auditor externo 2026-10-20: nota temporal en provenance.

Aclara que reference_date es referencia de ejecucion, no corte de
datos. Fuentes externas (CBOE, FRED, Yahoo) pueden incorporarse
despues.
"""
from src.report.provenance import render_provenance


def test_provenance_nota_reference_date_no_es_corte():
    out = render_provenance({
        "commit_sha": "abc12345",
        "run_id": "12345",
        "run_attempt": "1",
        "target_session": "2026-10-19",
        "reference_date": "2026-10-20T03:53:28+02:00",
        "effective_date_equity": "2026-10-19",
        "validation_gate": "10/10",
        "temporal_coherence": "OK",
        "generated_at": "2026-10-20T02:08:59Z",
    })
    joined = "".join(out)
    assert "no un corte estricto de datos" in joined
    assert "CBOE" in joined
    assert "effective_date_equity" in joined


def test_provenance_sin_dict_devuelve_vacio():
    assert render_provenance(None) == []