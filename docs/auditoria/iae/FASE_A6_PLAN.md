# IAE - Fase A.6 Plan de ejecucion

**Objeto:** plan detallado de la fase A.6 (reconciliacion contrato<->codigo).

**Nota de proceso (obligatoria):** el codigo actual contiene una
implementacion anticipada autorizada internamente (PROMPT_MAESTRO
v6.43 seccion 15.33). A.6 NO presupone que dicha implementacion sea
contractualmente valida. Su objetivo es reconciliarla con el dictamen
F2.4 y corregir las divergencias que el auditor confirme.

Distincion de autorizaciones:

    autorizacion del supervisor   -> implementacion anticipada (hecha)
    dictamen F2.4                 -> aprobacion contractual
    autorizacion A.6              -> desbloqueo post-F2.4

**Generado:** 2026-09-20.
**Precondicion bloqueante:** dictamen F2.4 (GO o GO CONDICIONADO).
**Estado:** NO EJECUTABLE hasta F2.4.
**Referencia:** iae/RECONCILIACION_CONTRATO_CODIGO.md + iae/REESTRUCTURACION_MODULO.md.

---

## 1. Objetivo

Cerrar las 3 divergencias contrato<->codigo y recomputar la evidencia
empirica (baseline + TOP 2000) bajo la semantica contractual correcta.

Resultado esperado de A.6:

- Codigo alineado con contrato P38/P60/P61.
- Tests contractuales sin xfail (todos pasan).
- Baseline + TOP 2000 recalculados bajo semantica contractual.
- F2.4-CLOSE emitido con evidencia actualizada.
- Inputs para Gate-NIPC.2 disponibles.

Nota: A.6 NO desbloquea Gate-NIPC.2 por si sola. El desbloqueo requiere
ademas propuesta de thresholds sobre evidencia v2 y dictamen especifico
del auditor.

---

## 2. Sub-fases

### A.6.1 - Dictamen F2.4 (EXTERNO)

**Submission HEAD:** `2eb4dcd`.

**Entrada:**
- iae/RECONCILIACION_CONTRATO_CODIGO.md
- iae/REESTRUCTURACION_MODULO.md
- iae/NIPC_CONTRATOS_SEMANTICOS_v1.md
- iae/NIPC_COVERAGE_POLICY_V13_PROPUESTA.md
- iae/INFORME.md
- iae/DICTAMENES.md
- tests/test_p60_contract.py
- tests/test_p61_contract.py
- tests/test_p38_contract.py

**Salida:**
- Dictamen F2.4 formal, con decisiones por D1/D2/D3.

**Criterios de aceptacion:**
- Decision explicita por divergencia (A o B).
- Postura sobre OpenFIGI masivo (autorizado / no).
- Postura sobre policy v1.3 (aprobar / mantener v1.0).

**Estado:** PENDIENTE EXTERNO.

---

### A.6.2 - Fixes quirurgicos (post F2.4 = GO)

3 commits minimos, uno por divergencia.

**Commit A.6.2-P60** - P60 sin default TICKER
- Archivo: `sec_13f/identity/security_identity.py`.
- Cambio: `_find_active_equivalence` no asume default TICKER si falta
  la columna `identity_type` en `cusip_equivalence.csv`.
- NO se introduce `ValueError` para CUSIP/ISIN: el contrato vigente
  mantiene CUSIP/ISIN -> NULL.
- Test: `tests/test_p60_contract.py` (ya existe, retirar xfail).

**Commit A.6.2-P61** - P61 conectado al resolver
- Archivo: `sec_13f/identity/security_identity.py`.
- Cambio: `evidence` incluye `valid_from`/`valid_to`.
- Cambio: `resolve_security_identity` invoca `resolve_source_status`.
- Cambio: `crosswalk_internal` separado en `cusip_ticker_exceptions`
  vs `etf_holdings`.
- Test: `tests/test_p61_contract.py` (nuevo).

**Commit A.6.2-P38** - Firma nueva nipc
- Archivo: `aggregation/nipc.py` + `aggregation/coverage.py` (nuevo).
- Cambio: `compute_nipc` recibe `target_q4`/`target_q1` opcionales.
- Cambio: delegacion a `compute_contractual_coverage`.
- Cambio: mover `temporal_validity.py` a `aggregation/`.
- Test: `tests/test_p38_contract.py` (nuevo).

**Criterios de aceptacion:**
- Los 3 commits compilan.
- Pyflakes limpio.
- Tests contractuales nuevos pasan.
- Tests existentes no-regresionan.

---

### A.6.3 - Test de contrato P38 (shareClassFIGI como clave de pairing)

**Precondicion bloqueante:** decision del auditor sobre Q12. Los dos
modelos posibles son:

    Modelo A  shareClassFIGI = clave contractual de pairing
    Modelo B  canonical_security = clave contractual de pairing

El presente plan asume Modelo A (Q12/A) como direccion, pero el test
NO se escribe hasta que F2.4 lo confirme.

**Objetivo (bajo Modelo A):** verificar que PAIRED empareja por
`shareClassFIGI` contractual comun entre Q4 y Q1, con
`operational_mapping_status == VERIFIED` en ambos periodos. NO se exige
igualdad literal de `canonical_security` entre periodos.

**Regla de agregacion:** los pesos se agregan por shareClassFIGI antes
de aplicar max(Q4, Q1). Multiples CUSIPs que comparten shareClassFIGI
contribuyen a un unico peso w(X). El test debe cubrir este caso.

Escenario:

    Q4: CUSIP_A -> shareClassFIGI_X -> equity:TICKER
    Q1: CUSIP_B -> shareClassFIGI_X -> figi:BBG...

Esperado bajo Modelo A:

    TARGET_PAIRWISE = {X}
    PAIRED          = {X}   aunque canonical_security(Q4) != canonical_security(Q1)

Esperado bajo Modelo B (si F2.4 confirma Q12/B):

    TARGET_PAIRWISE = {X}
    PAIRED          = {}    porque canonical_security(Q4) != canonical_security(Q1)

Commit: `tests/test_p38_contract.py` extendido (una vez decidido Q12).

**Criterio de aceptacion:**
- Test pasa segun el modelo confirmado por F2.4.
- Se documenta explicitamente en el test bajo que modelo se valida.

---

### A.6.4 - Recalculo baseline + TOP 2000

**Precondicion:** A.6.2 y A.6.3 cerrados.

**Accion:**
- Reejecutar `probe_coverage_baseline.py` con firma nueva.
- Reejecutar `run_pilot_13f_top2000.py`.
- NO sobrescribir los ficheros historicos (`evidence/`).
- Escribir nuevos con sufijo `_v2`.

**Entregable:**
- `evidence/nipc_gate0_baseline_v2/`
- `evidence/nipc_gate0_target_identity_top2000_v2/`

**Criterio de aceptacion:**
- Ambas ejecuciones terminan exit 0.
- Se demuestra mediante evidencia de entrada/salida que el denominador
  utilizado es `TARGET_PAIRWISE` (no `observed_security_key`).
- Se documenta:
      valor historico (v1.0) = X
      valor v2               = Y
      delta                  = Z
  NO se exige que X != Y. Si X == Y, tambien es valido: el calculo se
  hizo con la semantica nueva y el resultado coincide.

---

### A.6.5 - Emitir NIPC_CONTRATOS_SEMANTICOS_v2 (si aplica)

Solo si A.6.1 autoriza cambios de contrato. Si no, el contrato v1
permanece intacto.

**Regla de versionado:** el contrato v1 esta integrado en la cadena
hash del sistema (sha256 registrado en varios documentos). NO se
modifica in-place. Se emite un artefacto v2 nuevo.

    NIPC_CONTRATOS_SEMANTICOS_v1.md    INTACTO (hash preservado)
    NIPC_CONTRATOS_SEMANTICOS_v2.md    NUEVO
      - incluye seccion "Deriva de v1"
      - cita hash de v1
      - documenta cambios materiales

Nota: esta es una excepcion documentada a la regla general "1 concepto
= 1 fichero vivo". Aplica a contratos cuyo hash ya esta en cadena
autoritativa. El auditor puede autorizar otro esquema si lo considera.

**Criterio de aceptacion:**
- v2 escrito sin modificar v1.
- v1 conserva su hash original.
- v2 referencia explicitamente a v1.

---

### A.6.6 - Emitir F2.4-CLOSE (dictamen de cierre)

**Nota de nomenclatura:** existen dos hitos distintos:

    F2.4        dictamen inicial de contratos (A.6.1)
    F2.4-CLOSE  dictamen de cierre tras implementacion + evidencia

**Entrada:**
- Codigo alineado con contrato (A.6.2).
- Tests contractuales (A.6.3).
- Nueva evidencia (A.6.4).
- Contrato v2 emitido si aplica (A.6.5).

**Salida:**
- Dictamen F2.4-CLOSE.
- Propagacion a FOLLOWUPS.md y PROMPT_MAESTRO.

**Criterio de aceptacion:**
- Inputs disponibles para Gate-NIPC.2.
- Baseline v2 + TOP 2000 v2 disponibles.
- NO se declara "Gate-NIPC.2 desbloqueado". El desbloqueo requiere
  ademas: (a) propuesta de thresholds sobre evidencia v2, (b) dictamen
  especifico del auditor con valores numericos. A.6.6 solo entrega los
  inputs.

---

## 3. Orden de dependencia

    A.6.1 (F2.4)
        |
        v
    A.6.2 (3 commits quirurgicos) --+-- A.6.3 (test shareClassFIGI)
        |                            |
        v                            v
    A.6.4 (recalculo evidencia)
        |
        v
    A.6.5 (actualizar contrato si aplica)
        |
        v
    A.6.6 (F2.4 definitivo)

Los pasos A.6.2-P60, A.6.2-P61, A.6.2-P38 pueden ejecutarse en
paralelo si se hace un backup por cada uno.

---

## 4. Bloqueos y precondiciones

    | Bloqueo                       | Afecta a    | Desbloquea con      |
    |-------------------------------|-------------|---------------------|
    | F2.4 pendiente                | Todo A.6    | Dictamen externo    |
    | OpenFIGI masivo NO AUTORIZADO | A.6.2-P38-materialize    | Autorizacion F2.4   |
    | THRESHOLD_1/2 UNDEFINED       | A.6.6 final | A.6.4 + auditor     |
    | Gate-NIPC.2 BLOQUEADO         | A.7         | A.6.6               |

Los fixes D1/D3 y el fix de cobertura (firma nueva nipc) NO dependen
de OpenFIGI. Solo el D2 completo (TARGET real) lo requiere.

**Opcion estrategica:** implementar D1+D3+coverage en A.6.2 y dejar D2
para A.6.3 o A.6.4 si F2.4 no autoriza OpenFIGI todavia.

---

## 5. Reglas de ejecucion

- **Local-first IAE.** NO push hasta Gate-NIPC.3.
- **Un cambio = una verificacion = un commit.**
- **NO tocar `delta_shares.py`, `match_key`, C2.**
- **NO modificar policy v1.0** (hash 57f2d01f...).
- **Backup + snapshot pre/post** en cada cambio estructural.
- **Commit por sub-fase.** No mezclar D1 con D3.
- **Rollback quirurgico** si algun paso falla.

---

## 6. Criterios de aceptacion globales de A.6

    | Criterio                                       | Umbral          |
    |------------------------------------------------|-----------------|
    | Codigo alineado con contrato P38/P60/P61       | 100%            |
    | Tests baseline antes de A.6                    | N (medir inicio)|
    | Tests nuevos contractuales                     | M               |
    | Tests divergencia xfail (antes del fix)        | K               |
    | Tests xfail pendientes tras A.6                | 0               |
    | Pyflakes                                       | 0 warnings      |
    | compileall                                     | OK              |
    | Baseline v2 + TOP 2000 v2                      | escritos        |
    | F2.4-CLOSE emitido                             | si              |
    | Inputs Gate-NIPC.2 disponibles                 | si              |
    | Gate-NIPC.2 desbloqueado                       | NO (fuera A.6)  |

Nota: "tests presentes" != "tests contractualmente satisfactorios". Un
xfail no es un test verde. El criterio duro es "xfail pendientes tras
A.6 == 0". Si algun xfail no puede retirarse, requiere dictamen.

---

## 7. Estimacion de esfuerzo (post F2.4=GO)

    A.6.2-P60          1 commit, 1 test nuevo              ~30 min
    A.6.2-P61          1 commit, 1 test nuevo              ~60 min
    A.6.2-P38    1 commit, 1 test nuevo + move       ~90 min
    A.6.3             extension de test P38               ~30 min
    A.6.4             reejecucion baseline + TOP 2000     ~60 min
    A.6.5             actualizar contrato si aplica       ~30 min
    A.6.6             F2.4 definitivo (externo)           -

    Total estimado (supervisor): ~5 horas de trabajo tecnico
    + dictamen externo F2.4 (variable).

---

## 8. Referencias

    | Documento                                    | Rol                |
    |----------------------------------------------|--------------------|
    | iae/RECONCILIACION_CONTRATO_CODIGO.md        | Divergencias       |
    | iae/REESTRUCTURACION_MODULO.md               | Plan arquitectonico|
    | iae/NIPC_CONTRATOS_SEMANTICOS_v1.md          | Contrato           |
    | iae/NIPC_COVERAGE_POLICY_V13_PROPUESTA.md    | Policy propuesta   |

---

Fin del plan A.6. Bloqueado hasta F2.4.
