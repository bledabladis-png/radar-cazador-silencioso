# Probe P65 e2e - Manager Duplication

**Objetivo:** verificar en datos reales que el pipeline P65 (pre-delta +
post-delta) se comporta fail-closed sin evidencia cruzada L3.

**Metodologia congelada antes del run** (2026-09-20).

## Que mide

1. Pipeline completo Q4 2025 -> Q1 2026:
   - filter_by_period
   - apply_amendments -> canonical_snapshot
   - resolve_batch_identities
   - compute_reported_position_units
   - build_effective_reporting_snapshot (dedup pre-delta)
   - compute_delta_shares
   - classify_reporting_transition (post-delta)
2. Metricas:
   - `n_units_q4`, `n_units_q1` (post-dedup)
   - `n_audit_rows` (deberia ser 0 con evidencia vacia)
   - `n_delta_rows`
   - `match_status_counts`
   - `reporting_transition_counts` (deberia ser todo NULL)

## Hallazgo esperado

**L3 no es materializable en v1 con los 3 TSVs estructurados.**

El campo "Other Managers Reporting for this Manager" (la declaracion del
representado de que el representante reporta por el) NO esta en columnas
estructuradas. Esta en `ADDITIONALINFORMATION` (texto libre del cover page).

Parsear texto libre esta PROHIBIDO por las reglas del sistema. Sin evidencia
cruzada L3, el pipeline debe:

- NO deduplicar (`dedup_decision = KEEP` en todo).
- NO emitir `HANDOFF` (`reporting_transition = NULL` en todo).
- NO fabricar `REPORTING_CONFLICT` ni `OVERLAP_UNRESOLVED` (sin pares con L3).

El probe confirma empiricamente este comportamiento fail-closed.

## Como ejecutarlo

    py docs/auditoria/iae/evidence/nipc_p65_probe/probe_p65_e2e.py

Determinista. NO escribe a repo. NO usa datetime.now(). Solo imprime.

## Artefactos

- `probe_p65_e2e.py`  script determinista
- `output.txt`         salida cruda (LF puro)
- `HASHES.txt`         SHA-256 de los ficheros

## Resultado (probe_p65_e2e_reduced.py, 2026-09-20)

Ejecutado sobre 200 CIKs (primeros por ACCESSION_NUMBER ascendente):

    Q4 2025: SUBMISSION 200 -> canonical_snapshot 125, INFOTABLE 325.546, units 151.222
    Q1 2026: SUBMISSION 200 -> canonical_snapshot 106, INFOTABLE 276.312, units 126.853

    DEDUP PRE-DELTA:
      effective_q4 = 151.222 (KEEP integral)
      effective_q1 = 126.853 (KEEP integral)
      dedup_audit rows = 0

    DELTA SHARES:
      delta rows = 252.341
      match_status = {UNRESOLVED_IDENTITY: 216.722,
                      BOTH: 25.734, EXIT: 7.625, NEW: 2.260}

    REPORTING TRANSITION (POST-DELTA):
      reporting_transition = {NULL: 252.341}  (100% NULL)

### Veredicto empirico

El pipeline P65 v1 se comporta **fail-closed** sin evidencia cruzada L3:

- `dedup_decision = KEEP` en todas las unidades.
- `reporting_transition = NULL` en todas las filas delta.
- `dedup_audit` vacio (no hay decisiones que auditar).

Este es el comportamiento contractual exigido por el dictamen P65 v3: sin
evidencia documental de que una observacion concreta esta siendo reportada
por otro manager en nombre del representado, NO se elimina informacion ni se
reclasifica match_status.

## Limitaciones declaradas

1. `cross_filing_evidence` se inyecta vacio (fail-closed). La dedup real
   requiere evidencia L3 cruzada entre filings, no materializable desde los
   TSVs actuales (el campo "Other Managers Reporting for this Manager"
   esta en `ADDITIONALINFORMATION` como texto libre; parsear texto libre
   esta prohibido por las reglas del sistema).
2. El probe reducido usa los 200 primeros CIKs por ACCESSION_NUMBER.
   El probe completo (todas las filas) es viable pero tarda ~10 min.
3. No se modifica ningun fichero de produccion.
4. `DROP_DUP` no se emite en v1 con 13F puro: la coexistencia de dos filings
   sobre la misma security es OVERLAP por definicion. Requiere evidencia
   cuantitativa externa (capacidad diferida v2).

## Artefactos

- `probe_p65_e2e.py`          probe completo (todas las filas, ~10 min)
- `probe_p65_e2e_reduced.py`  probe reducido (200 CIKs, ~30s)
- `output.txt`                salida del probe reducido (LF puro)
- `HASHES.txt`                SHA-256 de los 3 ficheros
