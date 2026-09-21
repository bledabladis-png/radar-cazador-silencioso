# A.6.6 - Bundle para dictamen externo (F2.4-CLOSE)

**Solicitante:** Ingeniero Supervisor del Radar de Rotacion Sectorial.
**Destinatario:** auditor externo del proyecto IAE.
**Fecha:** 2026-09-21.
**Origen:** dictamen bundle v3 (NO-GO A.6.6) + dictamen consulta H-10.1
(GO fix A2). El fix A2 se ha ejecutado completo (5/5).
**Estado del repo:** HEAD `deb9782`, ahead 291 sobre `origin/main`
(`5f1d9ab`). Push NO (local-first IAE).

---

## 1. Resumen ejecutivo

El fix A2 cierra 4 hallazgos de la auditoria externa 2026-09-21:

| ID | Descripcion | Commit |
|---|---|---|
| H-10.1 | adapter marca VERIFIED incondicionalmente | `da5fbcb` (c3) |
| H-07 | PositionRecord.weight=1.0 hardcoded | `da5fbcb` (c3) |
| H-05 | probe con mock Q4=Q1 + coverage_previous=1.0 | `157211e` (c5) |
| H-06 | probe no publica cardinalidades reales | `157211e` (c5) |

Evidencia empirica sobre 13F Q4 2025 / Q1 2026 reales (result.json):
coverage_previous=None, coverage_current=1.0 (sobre TARGET_Q1 real),
paired_*=None. Conforme a Q5 literal del auditor.

Se solicita dictamen A.6.6 (F2.4-CLOSE).

## 2. Commits del ciclo A2 (7 commits, 2026-09-21)

| # | Hash | Contenido |
|---|---|---|
| 1 | `2f98926` | period_state.py: sshprnamt + operational_mapping_status |
| 2 | `8f8ef87` | target_builder.py: extract_sshprnamt_by_figi + probe transporta |
| 3 | `da5fbcb` | catalog_p38_adapter.py: propaga VERIFIED + weight (H-10.1 + H-07) |
| 4 | `f7f7b20` | coverage.py: denominador TARGET + Q5 fail-closed |
| 5 | `157211e` | probe sin mock + Q5 literal + cardinalidades + cierre doc |
| chore | `473ce06` | regenerar ESTADO_SISTEMA.md |
| docs | `deb9782` | TRANSFER v9.1 (estado post-fix) |

Rango completo: `git log --oneline a1eaa79..HEAD`.

## 3. Evidencia empirica (result.json)

Dataset: 13F Q4 2025 + Q1 2026 (parquets locales, hashes en HASHES.txt).
Snapshot B2-PIT: `snapshot_20260921_01.csv` (valid_from=2026-09-21).

Universos:

    2025Q4: 0 filas operational, 0 tickers equity
    2026Q1: 28.371 filas operational, 22 tickers equity
    Cruce con snapshot (242 keys): Q1 shared=20, Q4 shared=0

Resultado contractual P38:

    coverage_previous              = None    (Q4 vacio)
    coverage_current               = 1.0     (18/18 VERIFIED sobre TARGET_Q1)
    paired_security_coverage       = None    (TARGET_PAIRWISE=0)
    paired_weighted_share_coverage = None
    coverage_status                = UNAVAILABLE
    unmapped_count_previous        = 0
    unmapped_count_current         = 0

Cardinalidades reales (H-06):

    target_q4_figi          = 0
    target_q1_figi          = 18
    target_pairwise_figi    = 0
    records_q4              = 0
    records_q1              = 18
    records_q4_verified     = 0
    records_q1_verified     = 18
    paired_figi             = 0

Contraste pre/post fix A2 (mismo dataset):

    Pre  A2: coverage_previous=1.0, coverage_current=1.0, paired_*=1.0
             (artefacto: mock Q4=Q1 + weight=1.0 + VERIFIED incondicional)
    Post A2: coverage_previous=None, coverage_current=1.0, paired_*=None
             (contractual: Q4 real vacio -> fail-closed; Q1 real calculado)

## 4. Respuesta 1:1 a las 5 respuestas fijas del auditor

**Q1 (alcance A2, con SSHPRNAMT real):**
Implementado. `target_builder.extract_sshprnamt_by_figi` agrega
SSHPRNAMT por shareClassFIGI sobre el canonical_snapshot
post-amendments. `period_state` lo transporta. `adapter` lo propaga
a `PositionRecord.weight`. Commit `8f8ef87` (c2).

**Q2 (coverage_previous/current = RESOLVED/TARGET):**
Implementado. `coverage.py::compute_contractual_coverage` calcula
`len(res & tq) / len(tq)`. `None` si `tq` vacio. Commit `f7f7b20` (c4).

**Q3 (adapter PROPAGA, no fabrica VERIFIED):**
Implementado. `_records` propaga `s.operational_mapping_status`. Sin
evidencia operacional en el state -> default UNRESOLVED (fail-closed).
Commit `da5fbcb` (c3). Cubierto por
`test_h101_adapter_propaga_operational_mapping_status` (3 direcciones
a/b/c: VERIFIED / TEMPORAL_UNVERIFIED / UNRESOLVED default).

**Q4 (SSHPRNAMT del canonical_snapshot, agregado por FIGI, luego
max(Q4,Q1)):**
Implementado. `extract_sshprnamt_by_figi` suma por FIGI. `coverage.py::
aggregate_positions_by_shareclass_figi` agrega y luego `_w(figi) =
max(Q4_total, Q1_total)`. Commit `8f8ef87` (c2) + `f7f7b20` (c4).
Cubierto por `test_p38_agregacion_figi_multiples_cusip_pesos_reales`
y `test_p38_paired_weighted_max_por_figi_con_cobertura_asimetrica`.

**Q5 (Q4 vacio):**
Implementado. `coverage_previous=None`, `coverage_current` calculable
sobre TARGET_Q1, `paired_*=None`. Commit `f7f7b20` (c4) + `157211e`
(c5, probe). Evidencia directa: seccion 5 y 5b de la corrida del probe.

## 5. Hallazgos cerrados por este ciclo

| ID | Sev | Cerrado en | Test de verificacion |
|---|---|---|---|
| H-05 | ALTA | c5 | `test_h101_adapter_rechaza_pairwise_vacio` + probe seccion 6 |
| H-06 | ALTA | c5 | `cardinalities` en result.json |
| H-07 | ALTA | c3 | `test_h101_adapter_propaga_weight_desde_sshprnamt` |
| H-10.1 | CRITICA | c3 | `test_h101_adapter_propaga_operational_mapping_status` |

## 6. Estado de tests y repo

    py -m pytest tests/ validation/ -q --tb=line
    -> 3 failed, 1309 passed, 2 skipped, 0 xfailed

Los 3 failed son `tests/test_freshness.py` (preexistentes; no regresion
del fix A2; market_data/stock_prices stale 5 dias > 4).

    py -m pytest tests/test_h731_adapter_p38_compat.py tests/test_p38_contract.py -q
    -> 26 passed, 0 xfailed

pyflakes silencio. compileall OK.

## 7. Prohibiciones respetadas

- NO se ha tocado `nipc.py`, `delta_shares.py`, `security_identity.py`,
  `temporal_validity.py`, MATCH_KEY, C2, contratos normativos,
  `NIPC_COVERAGE_POLICY.md`.
- NO se ha hecho push a `origin/main` (local-first IAE).
- NO se ha activado OpenFIGI masivo ni DROP_DUP.
- `unmapped_count_previous/current` conservan su semantica (observed
  sin VERIFIED). Su redefinicion contra TARGET requiere dictamen
  especifico; no se ha cambiado en este ciclo.

## 8. Pendiente (declaracion honesta)

- **H-08 (MEDIA):** test H-73.1 no cubre estados ortogonales
  (`identity=RESOLVED` + `weight=NOT_PRESENT`). Requiere autorizacion
  del auditor para cerrarlo como test.
- **Thresholds NIPC:** THRESHOLD_1/2 siguen UNDEFINED/BLOQUEADOS.
- **Gate-NIPC.2:** bloqueado.
- **OpenFIGI masivo:** no autorizado; el universo B2-PIT completo
  (242 keys) sigue sin materializacion completa via OpenFIGI.
- **Push:** local-first IAE.
- **Reclasificacion A.6.4:** el README de la evidencia se ha reescrito
  post-fix; la reclasificacion formal (smoke test -> evidencia
  contractual) es decision del auditor, no del supervisor.

## 9. Anexos (referencias, no duplicacion)

| Documento | Rol |
|---|---|
| `evidence/a64_integration_b1_p61_p38/result.json` | Evidencia empirica cruda |
| `evidence/a64_integration_b1_p61_p38/HASHES.txt` | Provenance (post-fix) |
| `evidence/a64_integration_b1_p61_p38/README.md` | Alcance del probe |
| `EXPEDIENTE_A64_FIX.md` | Expediente del fix (marcado APLICADO) |
| `NIPC_CONTRATOS_SEMANTICOS_v1.md` | Contrato P38/P60/P61/P70 |
| `ESTADO_DECLARADO.md` | Fases, hallazgos, prohibiciones |
| `ESTADO_SISTEMA.md` | Hechos autogenerados (HEAD, tests, LOC) |

## 10. Peticion al auditor

Se solicita dictamen sobre los siguientes puntos:

1. **Cierre A.6.6 (F2.4-CLOSE).** ¿El fix A2 + la evidencia empirica
   (Q5 literal sobre datos reales) son suficientes para emitir
   GO/NO-GO? En caso NO-GO, identificar hallazgo bloqueante.

2. **Reclasificacion A.6.4.** ¿Se acepta la reclasificacion de
   A.6.4 como evidencia contractual post-fix A2 (ya no smoke test
   puro), o se exige un A.6.4-v3 con un dataset adicional?

3. **H-08.** ¿Se autoriza cerrarlo como test (caso ortogonal
   `identity=RESOLVED` + `weight=NOT_PRESENT`) o se exige ciclo propio?

4. **Push.** ¿Se autoriza push a `origin/main` de los 291 commits
   locales (incluye el fix A2) o se mantiene local-first IAE hasta
   dictamen sobre thresholds?

---

Fin del bundle A.6.6. HEAD `deb9782`. 2026-09-21.
