"""B-06 (auditor #76) - prueba end-to-end de la cadena de pesos.

El auditor #76 pidio literalmente:
  "Debe existir una prueba explicita equivalente a:
   CUSIP_A=100, CUSIP_B=50, mismo FIGI -> agregado=150 ->
   max(Q4,Q1) aplicado despues, y debe quedar demostrado que cada
   peso entra una unica vez."

Este test recorre la cadena completa:
  INFOTABLE (CUSIP+SSHPRNAMT)
  -> target_builder.extract_sshprnamt_by_figi     (agrega por FIGI)
  -> distribucion a key representativa (logica del probe A.6.4)
  -> catalog_p38_adapter._records                 (peso al record)
  -> coverage.aggregate_positions_by_shareclass_figi
  -> coverage.compute_contractual_coverage        (max(Q4,Q1))

Demuestra que con N catalog_keys compartiendo el mismo FIGI, el peso
contractual entra UNA vez, no N. Si la distribucion a key
representativa no existiera (bug B-06 pre-fix), aggregate sumaria N
veces el total del FIGI.
"""
from __future__ import annotations

import pandas as pd

from src.institutional_accumulation.aggregation import catalog_p38_adapter as ca
from src.institutional_accumulation.aggregation import coverage as cov
from src.institutional_accumulation.identity import catalog_key as ck
from src.institutional_accumulation.identity import period_state as ps
from src.institutional_accumulation.identity import target_builder as tb


def _make_universe(rows):
    snap = pd.DataFrame(
        [{"radar_ticker": t, "share_class_figi": f, "name": t}
         for t, f in rows],
        columns=["radar_ticker", "share_class_figi", "name"],
    )
    uids = [ck.compute_snapshot_row_uid(snap.iloc[i], list(snap.columns))
            for i in range(len(rows))]
    keys = ["radar_20260922_" + str(i + 1).zfill(4) for i in range(len(rows))]
    mem = pd.DataFrame([
        {"version_id": "v1", "catalog_key": k, "snapshot_row_uid": u,
         "predecessor_row_uid": "", "justification": "test"}
        for k, u in zip(keys, uids)
    ], columns=list(ck.MEMBERSHIP_COLUMNS))
    asg = pd.DataFrame([
        {"catalog_key": k, "assigned_entity_id": "radar_entity_0001",
         "valid_from": "2026-09-22", "valid_to": "", "source": "test",
         "reason": "test"}
        for k in keys
    ], columns=list(ck.ASSIGNMENT_COLUMNS))
    return tb.build_target(snap, mem, asg, version_id="v1",
                           period_end="2026-03-31",
                           catalog_version_id="cat", catalog_sha256="a" * 64)


def _distribute_to_representative_keys(sh_by_figi, figi_to_keys):
    """Replica la logica del probe A.6.4 post-B-06.

    El total del FIGI se asigna a UNA key (la primera en orden
    lexicografico); las demas quedan sin peso. Asi aggregate suma el
    total UNA vez, no N.
    """
    sh_by_key = {}
    for f, v in sh_by_figi.items():
        ks = sorted(figi_to_keys.get(f, []))
        if ks:
            sh_by_key[ks[0]] = v
    return sh_by_key


def test_b06_e2e_dos_cusips_mismo_figi_agrega_una_sola_vez():
    """Auditor #76: CUSIP_A=100 + CUSIP_B=50 mismo FIGI -> 150 (una vez).

    Recorre la cadena completa. Verifica en cada paso que el total del
    FIGI aparece exactamente una vez.
    """
    # 1. INFOTABLE con 2 CUSIPs del mismo FIGI
    infotable = pd.DataFrame([
        {"CUSIP": "CUSIP_A", "SSHPRNAMT": 100},
        {"CUSIP": "CUSIP_B", "SSHPRNAMT": 50},
    ])
    figi_by_cusip = {"CUSIP_A": "FIGI_X", "CUSIP_B": "FIGI_X"}

    # 2. extract_sshprnamt_by_figi: suma 100 + 50 = 150
    sh_by_figi = tb.extract_sshprnamt_by_figi(infotable, figi_by_cusip)
    assert sh_by_figi == {"FIGI_X": 150.0}

    # 3. Universe con 2 catalog_keys que comparten FIGI_X
    # (dos tickers distintos -> dos keys distintas, un solo FIGI)
    u = _make_universe([("TICK_A", "FIGI_X"), ("TICK_B", "FIGI_X")])
    keys = sorted(u.declared_keys)
    assert len(keys) == 2

    # 4. Distribucion a key representativa: una key lleva 150, la otra 0
    figi_to_keys = {"FIGI_X": keys}
    sh_by_key = _distribute_to_representative_keys(sh_by_figi, figi_to_keys)
    assert sh_by_key == {keys[0]: 150.0}
    assert keys[1] not in sh_by_key

    # 5. period_state con el sshprnamt_evidence
    st = ps.build_period_state(
        u,
        operational_evidence={k: "VERIFIED" for k in keys},
        sshprnamt_evidence=sh_by_key,
    )
    assert st[keys[0]].sshprnamt == 150.0
    assert st[keys[1]].sshprnamt is None

    # 6. Adapter: 2 records, uno con peso 150, otro con peso 0
    recs = ca._records(u, st, period="Q1")
    assert len(recs) == 2
    weights = {r.observed_security_key: r.weight for r in recs}
    assert weights[keys[0]] == 150.0
    assert weights[keys[1]] == 0.0

    # 7. aggregate por FIGI: suma total = 150, NO 150+150=300
    agg = cov.aggregate_positions_by_shareclass_figi(recs, "Q1")
    assert agg == {"FIGI_X": 150.0}

    # 8. compute_contractual_coverage: max(Q4,Q1) aplicado despues
    #    Solo Q1 tiene datos; Q4 vacio -> coverage_previous=None
    result = cov.compute_contractual_coverage(
        target_q4=set(), target_q1={"FIGI_X"},
        records_q4=[], records_q1=recs,
    )
    assert result["coverage_current"] == 1.0
    assert result["coverage_previous"] is None
    assert result["paired_weighted_share_coverage"] is None


def test_b06_contraejemplo_sin_distribucion_doble_conteo():
    """Contraejemplo: si TODAS las keys llevan el total del FIGI,
    aggregate suma N veces. Documenta el bug B-06 pre-fix.
    """
    u = _make_universe([("TICK_A", "FIGI_X"), ("TICK_B", "FIGI_X")])
    keys = sorted(u.declared_keys)

    # Anti-patron: broadcast del total a TODAS las keys (pre-fix B-06)
    sh_by_key_bug = {k: 150.0 for k in keys}
    st = ps.build_period_state(
        u,
        operational_evidence={k: "VERIFIED" for k in keys},
        sshprnamt_evidence=sh_by_key_bug,
    )
    recs = ca._records(u, st, period="Q1")
    agg = cov.aggregate_positions_by_shareclass_figi(recs, "Q1")
    # 2 keys x 150 = 300. Bug confirmado como contraejemplo.
    assert agg == {"FIGI_X": 300.0}


def test_b06_max_q4_q1_sobre_peso_agregado_una_vez():
    """max(Q4,Q1) se aplica sobre el agregado por FIGI, no por key.

    Q4: FIGI_X = 200 (key1=200, key2=0)  -> agg=200
    Q1: FIGI_X = 150 (key1=150, key2=0)  -> agg=150
    max = 200. paired_weighted = 200/200 = 1.0
    """
    u = _make_universe([("TICK_A", "FIGI_X"), ("TICK_B", "FIGI_X")])
    keys = sorted(u.declared_keys)

    st4 = ps.build_period_state(
        u,
        operational_evidence={k: "VERIFIED" for k in keys},
        sshprnamt_evidence={keys[0]: 200.0},
    )
    st1 = ps.build_period_state(
        u,
        operational_evidence={k: "VERIFIED" for k in keys},
        sshprnamt_evidence={keys[0]: 150.0},
    )
    r4 = ca._records(u, st4, period="Q4")
    r1 = ca._records(u, st1, period="Q1")

    # Agregado por FIGI = 200 (Q4) y 150 (Q1)
    assert cov.aggregate_positions_by_shareclass_figi(r4, "Q4") == {"FIGI_X": 200.0}
    assert cov.aggregate_positions_by_shareclass_figi(r1, "Q1") == {"FIGI_X": 150.0}

    # max(Q4,Q1) = 200, numer = 200, denom = 200 -> 1.0
    result = cov.compute_contractual_coverage(
        target_q4={"FIGI_X"}, target_q1={"FIGI_X"},
        records_q4=r4, records_q1=r1,
    )
    assert result["paired_security_coverage"] == 1.0
    assert result["paired_weighted_share_coverage"] == 1.0
