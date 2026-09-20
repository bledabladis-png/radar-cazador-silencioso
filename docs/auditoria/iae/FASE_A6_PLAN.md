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
**Precondicion bloqueante:** F2.4 EMITIDO (GO CONDICIONADO).
**Estado:** EJECUTABLE excepto A.6.2-bis (rediseno TARGET, requiere
autorizacion OpenFIGI). A.6.0 precede a A.6.2. Ver DICTAMENES.md #24.
**Referencia:** iae/RECONCILIACION_CONTRATO_CODIGO.md + iae/REESTRUCTURACION_MODULO.md.

---

## 1. Objetivo

Cerrar las 3 divergencias contrato<->codigo y recomputar la evidencia
empirica (baseline + TOP 2000) bajo la semantica contractual correcta.

Resultado esperado de A.6:

- Codigo alineado con contrato P38/P60/P61.
- Tests contractuales: sin xfail para los requisitos que F2.4
  adopte como contractuales. Los xfail correspondientes a capacidades
  declaradas NO CONTRACTUALES o DIFERIDAS por F2.4 pueden permanecer
  como residual aceptado.
- Baseline + TOP 2000 recalculados bajo semantica contractual.
- F2.4-CLOSE emitido con evidencia actualizada.
- Inputs para Gate-NIPC.2 disponibles.

Nota: A.6 NO desbloquea Gate-NIPC.2 por si sola. El desbloqueo requiere
ademas propuesta de thresholds sobre evidencia v2 y dictamen especifico
del auditor.

---

## 2. Sub-fases

### A.6.0 - Gate 0 de los 3 bloqueantes estructurales (PREVIO)

**Origen:** dictamen F2.4 introdujo 3 bloqueantes estructurales (ver
DICTAMENES.md #24 + INFORME.md #18). Antes de tocar codigo productivo,
inventario empirico sin modificaciones.

**Objetivo:** mapear cuanto de los 3 bloqueantes esta ya cubierto y
cuanto requiere rediseno.

**Bloqueantes a auditar:**

    1. TARGET independiente del exito del mapping.
       - Donde vive la construccion de TARGET hoy.
       - Si depende de security_identity / cusip_resolver.
       - Separacion CATALOGO vs TARGET_OBSERVED.

    2. Semantica point-in-time.
       - Campos temporales en radar_target_catalog.csv.
       - Campos temporales en los mappings OpenFIGI.
       - Existencia de target_catalog_as_of().

    3. 13F != flujo en tiempo real.
       - Estados de "no aparece" en delta_shares.py.
       - Distincion corporate action vs economic accumulation.
       - Tratamiento de manager duplication.

**Entregable:** informe de inventario (sin cambios de codigo) en
iae/INFORME.md con la evidencia directa.

**Criterio de aceptacion:** inventario completo con referencias
archivo:linea. Sin tocar codigo.

**Estado:** PENDIENTE. Precede a A.6.2.

---

### A.6.1 - Dictamen F2.4 sobre divergencias y decisiones arquitectonicas (EXTERNO)

**Submission HEAD:** `3ad57b1`.

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
- Dictamen F2.4 formal.

**Decisiones requeridas de F2.4:**

    D1      P61   Conectar resolve_source_status, o retirar
                  temporal_validity.py.
    D2      P38   Construir TARGET real, o mantener proxy
                  observacional NO CONTRACTUAL.
    D3      P60   Rechazar filas sin identity_type declarado,
                  o mantener default TICKER documentado.
    Q12     Pair  Pairing por shareClassFIGI (Modelo A) o por
                  canonical_security (Modelo B).
    AGREG.        Regla de agregacion de multiples CUSIPs por
                  shareClassFIGI (ver RECONCILIACION seccion 9).
                  Incluye: (a) donde vive la agregacion (Opcion 1
                  vs Opcion 2), (b) si estado distinto de VERIFIED
                  contribuye al peso agregado.
    OpenFIGI      Autorizado / no autorizado para materializacion
                  completa del TARGET.
    Policy v1.3   Aprobada / mantener v1.0 vigente.

**Criterios de aceptacion:**
- Decision explicita por D1, D2, D3.
- Decision explicita sobre Q12.
- Decision explicita sobre AGREG. (opcion arquitectonica + regla
  de estado).
- Postura sobre OpenFIGI masivo.
- Postura sobre policy v1.3.

**Decisiones recibidas (2026-09-20):**

    D1 P61        GO - conectar resolve_source_status + evidence extendida
    D2 P38        GO CONDICIONADO - TARGET real + rediseno independencia mapping
    D3 P60        GO - fail-closed, sin default TICKER
    Q12           Modelo A - shareClassFIGI (unidad = share class)
    AGREG.        Opcion 2 - funcion separada + solo VERIFIED al peso contractual
    OpenFIGI      GO CONDICIONADO - snapshot + hash, no dependencia live
    Policy v1.3   NO - v1.0 sigue normativa
    THRESHOLD_2   BLOQUEADO
    Certificacion BLOQUEADA

**Estado:** CERRADO 2026-09-20. Referencia: DICTAMENES.md #24 + INFORME.md #18.

---

### A.6.2 - Fixes quirurgicos (post F2.4 = GO)

**Nota (F2.4 2026-09-20):** D1 (P60) y D3 (P61) siguen siendo fixes
quirurgicos. D2 (P38) se divide en dos partes:

    - A.6.2-P38        firma nueva nipc + coverage.py (quirurgico).
    - A.6.2-bis        rediseno TARGET (arquitectonico). Ver mas abajo.

3 commits minimos (uno por divergencia D1/D2/D3). Segun lo que
determine F2.4, pueden requerirse commits adicionales para materializar
Q12 (pairing) y AGREG. (agregacion shareClassFIGI). Si F2.4 adopta
Opcion 2 para AGREG., se anade un commit para
`aggregate_positions_by_shareclass_figi()`. Si adopta Opcion 1, la
logica vive dentro de `compute_contractual_coverage()`.

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

### A.6.2-bis - Rediseno arquitectonico de TARGET (post F2.4)

**Origen:** bloqueante 1 del F2.4. El TARGET no puede depender del
exito del mapping. La definicion actual
`TARGET = observaciones 13F INTERSECT RADAR_TARGET_CATALOG`
introduce sesgo de seleccion: el denominador depende del propio
proceso que se mide.

**Cambios:**

- `identity/target_builder.py`:
    - Recibe CATALOGO como entidad externa y versionada.
    - NO importa `security_identity` ni `cusip_resolver`.
    - Prohibido derivar TARGET de observaciones mapeadas.
    - Flujo: CATALOGO -> TARGET FIGIs -> mapping -> coverage.

- Introducir `catalog_version` + `catalog_valid_from` + `catalog_valid_to`
  o `target_catalog_as_of(period_end)` (bloqueante 2, P62).

- Consumidor: `aggregation/coverage.py` recibe el TARGET ya construido.

**Precondicion:** OpenFIGI masivo autorizado. Depende de A.6.0.

**Estado:** PENDIENTE. Requiere autorizacion especifica.

---

### A.6.3 - Test contractual P38 de pairing segun decision Q12

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

**Regla de agregacion (a dictamen F2.4, RECONCILIACION seccion 9):**

    Opcion 1  coverage.py hace la agregacion internamente.
    Opcion 2  aggregate_positions_by_shareclass_figi() separada,
              coverage.py recibe pesos ya agregados.

El plan asume Opcion 2 como direccion. Si F2.4 confirma Opcion 1, el
test se escribe contra compute_contractual_coverage(). Si confirma
Opcion 2, se escriben 2 tests: uno para la funcion de agregacion, uno
para compute_contractual_coverage() con pesos agregados.

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

### A.6.4 - Recalculo de evidencia (condicional a las decisiones F2.4)

**Precondicion:** A.6.2 y A.6.3 cerrados + decisiones F2.4 que
afecten a la semantica de la evidencia materializadas (D1, D2, D3,
Q12, AGREG., Policy v1.3).

**Rama condicional segun D2:**

    D2 = A (TARGET real):
        - Reejecutar probe_coverage_baseline.py con firma nueva.
        - Reejecutar run_pilot_13f_top2000.py.
        - Escribir nuevos con sufijo _v2.
        - Entregable:
              evidence/nipc_gate0_baseline_v2/
              evidence/nipc_gate0_target_identity_top2000_v2/
        - Criterio: se demuestra por evidencia entrada/salida que el
          denominador es TARGET_PAIRWISE. NO se exige que X != Y.

    D2 = B (proxy observacional NO CONTRACTUAL):
        - NO modificar ni renombrar evidencia historica existente.
        - Conservar los artefactos existentes en su ubicacion original.
        - NO recalcular como evidencia contractual.
        - NO exigir TARGET_PAIRWISE contractual.
        - Generar, cuando proceda, nuevos artefactos bajo:
              evidence/..._proxy_no_contractual_v2/
          etiquetados explicitamente como NO CONTRACTUALES.
        - Criterio: el reporte publica explicitamente que la evidencia
          es proxy y no contractual; THRESHOLD_2 permanece UNDEFINED
          por ausencia de evidencia contractual.

    Distincion preservada:
        historico    (intacto)
        nuevo        (v2 contractual o v2 proxy segun D2)
        proxy        (etiquetado NO CONTRACTUAL cuando aplique)

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

### A.6.7 - Contratos adicionales P62-P65 (derivados del F2.4)

**Origen:** el dictamen F2.4 exige formalizar semanticas que estaban
implicitas. Se integran en NIPC_CONTRATOS_SEMANTICOS_v1.md in-place
(fichero vivo, no se crea v2).

**Contratos nuevos:**

    P62  Semantica point-in-time
         - period_end != filing_date != knowledge_date (los 3 conservados).
         - catalog_version + valid_from/valid_to.
         - target_catalog_as_of(period_end).
         - OpenFIGI no expone effective_date: consulta actual != historica.

    P63  Missing != sold
         - 6 estados: ZERO_REPORTED | MISSING | BELOW_REPORTING_THRESHOLD |
           CONFIDENTIAL | UNRESOLVED | SOLD.
         - "No aparece en Q1" no es "vendio".
         - Ausencia de mapping != ausencia de posicion (NO_MATCH != not_target).

    P64  Corporate actions
         - Split, reverse split, spin-off, merger, conversion, CUSIP change.
         - delta_shares distingue economic accumulation de mechanical change.

    P65  Manager duplication
         - 13F admite other included manager + combination reports.
         - Unidad: manager / manager-group / filing / position / security.
         - Prohibido agregar universo por CUSIP sin resolver estructura filing.

**Adicionales integradas en P60/P61/P38 (seccion 3.3 + 3.4):**

    - unmapped_count int separado de unmapped_weight float.
    - compute_nipc sin degradacion silenciosa a proxy (evidence_class).
    - PositionRecord tipado en coverage.py.
    - Solo VERIFIED aporta al peso contractual; unverified aparte.

**Estado:** PENDIENTE. No ejecutable hasta A.6.0.

---

## 3. Orden de dependencia

    A.6.0 (Gate 0 bloqueantes)  <- PENDIENTE
        |
        v
    A.6.1 (F2.4)  <- CERRADO 2026-09-20
        |
        v
    A.6.2 (fixes P60 + P61 + P38)
        |
        +--- A.6.2-bis (rediseno TARGET)  [requiere OpenFIGI + A.6.0]
        |
        +--- A.6.3 (test Q12 shareClassFIGI)
        |          |
        |          v
        |      A.6.4 (recalculo evidencia)
        |          |
        |          v
        |      A.6.5 (contrato, si aplica)
        |          |
        |          v
        |      A.6.7 (contratos P62-P65)
        |
        v
    A.6.6 (F2.4-CLOSE)

Los pasos A.6.2-P60, A.6.2-P61, A.6.2-P38 pueden ejecutarse en
paralelo si se hace un backup por cada uno. A.6.2-bis requiere
autorizacion OpenFIGI y depende de A.6.0.

---

## 4. Bloqueos y precondiciones

    | Bloqueo                       | Afecta a    | Desbloquea con           |
    |-------------------------------|-------------|--------------------------|
    | F2.4                          | -           | EMITIDO 2026-09-20       |
    | A.6.0 Gate 0 bloqueantes      | A.6.2-bis   | Este ciclo               |
    | OpenFIGI masivo NO AUTORIZADO | A.6.2-bis   | Autorizacion especifica  |
    | THRESHOLD_1/2 UNDEFINED       | A.6.6 final | A.6.4 + auditor          |
    | Gate-NIPC.2 BLOQUEADO         | A.7         | A.6.6                    |

Los fixes D1/D3 (P60/P61) y el fix de cobertura (firma nueva nipc)
NO dependen de OpenFIGI. Solo D2 completo (TARGET real, rediseno)
requiere OpenFIGI + rediseno arquitectonico (A.6.0 + A.6.2-bis).

**Opcion estrategica:** implementar D1+D3+coverage en A.6.2 y dejar
A.6.2-bis (rediseno TARGET) para despues de A.6.0 y de autorizacion
OpenFIGI.

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
    | Codigo alineado con contrato P38/P60/P61       | 100% de los     |
    |                                                | req. adoptados  |
    |                                                | por F2.4        |
    | Tests baseline antes de A.6                    | N (medir inicio)|
    | Tests nuevos contractuales                     | M               |
    | Tests divergencia xfail (antes del fix)        | K               |
    | Tests xfail tras A.6 (req. contractuales)      | 0               |
    | Tests xfail residuales aceptados               | documentados    |
    | (capacidades NO CONTRACTUALES o DIFERIDAS)     |                 |
    | Pyflakes                                       | 0 warnings      |
    | compileall                                     | OK              |
    | Baseline v2 + TOP 2000 v2                      | escritos        |
    | F2.4-CLOSE emitido                             | si              |
    | Inputs Gate-NIPC.2 disponibles                 | si              |
    | Gate-NIPC.2 desbloqueado                       | NO (fuera A.6)  |

Nota: "tests presentes" != "tests contractualmente satisfactorios". Un
xfail no es un test verde.

El criterio "0 xfail" aplica SOLO a los requisitos que F2.4 adopte como
contractuales. Los xfail correspondientes a capacidades declaradas NO
CONTRACTUALES o DIFERIDAS por F2.4 pueden permanecer, registrados como
residual aceptado. En ese caso:

    - El test sigue marcado xfail.
    - El documento de estado cita explicitamente la decision F2.4 que
      lo declara fuera de contrato.
    - No se modifica el codigo solo para conseguir "0 xfail" artificial.

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
