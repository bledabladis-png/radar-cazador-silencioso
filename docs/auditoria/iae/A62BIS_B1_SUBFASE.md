# IAE - A.6.2-bis-B1 Subfase (TARGET independiente + dominio P38)

**Objeto:** documento dedicado de la subfase B1, tal como recomienda
dictamen #53 seccion 11 ("El siguiente objeto correcto de auditoria es
B1"). Sustituye la iteracion global v10+ de A.6.2-bis.

**Origen:** dictamen #51 (2 bloqueos A1/A2) + dictamen #53 (5
bloqueantes consolidados).

**Referencia historica:** A62BIS_PROPUESTA.md v9 §2 y §3 (propuesta
completa del ciclo). Este documento extrae y consolida el alcance B1
sin acoplarlo a B2-PIT (cerrado por #53).

**Fecha:** 2026-09-21.
**HEAD al redactar:** 21c30e1.
**Naturaleza:** propuesta de subfase. NO normativa. Sometida a dictamen.

---

## 0. Resumen ejecutivo

B1 responde a la pregunta "¿que TARGET contractual representa el
catalogo y como se traduce al dominio economico P38?".

**5 bloqueantes a cerrar** (2 del #51 + 3 del #53 seccion 11):

    A1  semantica correcta de catalog_key_assignment_unique
    A2  preservacion del dominio economico completo hacia P38
    B3  imposibilidad de reduccion silenciosa del denominador
        por FIGI incompleto
    B4  comportamiento fail-closed ante identidades asimetricas,
        no resueltas o conflictivas
    B5  preservacion de colisiones catalog_key -> FIGI en el
        dominio completo exigido por el contrato

**Estado previo:** la v9 de A62BIS_PROPUESTA.md ya contiene una
reformulacion de A1 y A2. Este documento consolida esa base + los 3
bloqueantes adicionales del #53 y los presenta como subfase
implementable aislada.

**Fuera de alcance:** catalog_key definitivo de las 242 filas
actuales (decision diferida a la fase de implementacion);
OpenFIGI masivo; P38 (no se modifica); DROP_DUP; certificacion.

---

## 1. Estado de partida

### 1.1. Lo que B2-PIT ya aporta (cerrado por #53)

    snapshot materializado
    catalog_manifest.json
    version_id + sha256 externo
    [valid_from, valid_to)
    target_catalog_as_of(period_end)
    integridad + corrupcion + ambiguedad
    no backdating

B1 **consume** esos artefactos. NO los modifica. NO los reabre.

### 1.2. Lo que B1 debe construir

    catalog_key (identidad administrativa estable)
    catalog_assignments.csv (registro)
    catalog_validator (assignment, continuity, collision)
    TargetUniverse (declared_keys, ticker_by_key, figi_by_key)
    target_builder (puro, sobre snapshot)
    estados por periodo (state_q4[K], state_q1[K])
    TARGET_PAIRWISE formal
    flujo normativo 9 pasos (fail-closed)
    adaptador P38 (full targets, no pairwise)

### 1.3. Prohibiciones heredadas

    NO modificar P38 (coverage.py firma y semantica).
    NO activar OpenFIGI masivo.
    NO activar DROP_DUP.
    NO modificar contratos normativos.
    NO reacoplar a B2-PIT.
---

## 2. Bloqueante A1 - asignacion de catalog_key (dictamen #51)

### 2.1. Formulacion normativa

    catalog_key_assignment_unique

    Una asignacion de catalog_key es unica globalmente.
    La misma catalog_key PUEDE aparecer en multiples snapshots
    durante su vigencia. Eso es persistencia, NO reutilizacion.

**Dos conceptos separados:**

    IDENTITY ASSIGNMENT UNIQUENESS
      Una catalog_key se asigna UNA SOLA VEZ en su ciclo de vida a
      una entidad administrativa concreta. La fecha_alta es inmutable.

    SNAPSHOT MEMBERSHIP
      La misma catalog_key PUEDE aparecer en 0..N snapshots durante
      su vigencia.

**El validator NO falla porque K1 aparezca en snapshot_1, snapshot_2,
snapshot_3.**

### 2.2. Formato de catalog_key

    catalog_key := "radar_<YYYYMMDD_alta>_<NNNN>"

Ejemplos (migracion inicial): `radar_20260919_0001` .. `radar_20260919_0242`.

- NO es el hash del snapshot (B2-PIT usa un version_id distinto).
- NO depende del ticker. El ticker es atributo versionado.
- Prefijo `radar_` identifica la fuente del catalogo.

### 2.3. Registro administrativo

    data/mappings/catalog_assignments.csv
    columnas:
        catalog_key           "radar_<YYYYMMDD>_<NNNN>"
        fecha_alta            fecha de primera asignacion (inmutable)
        source                "radar_initial_20260919" | ...
        estado                ACTIVE | RETIRED
        fecha_retiro          NULL | fecha (si RETIRED)

**Regla:** cada `catalog_key` tiene EXACTAMENTE 1 fila.
La `fecha_alta` es inmutable.

### 2.4. Validator

    def validate_assignment(assignments_df) -> dict[str, str]:
        """Errores:
             CATALOG_KEY_REASSIGNED
               catalog_key duplicado, o fecha_alta distinta a la original.
             CATALOG_KEY_RETIRED_REACTIVATED
               catalog_key RETIRED y reasignado a nueva entidad.
        """

### 2.5. Tests A1

    A1-a  persistencia correcta (K1 en multiples snapshots) -> OK
    A1-b  reasignacion (K1 con fecha_alta distinta) -> CATALOG_KEY_REASSIGNED
    A1-c  retired sin reactivar -> OK
    A1-d  retired y reactivado -> CATALOG_KEY_RETIRED_REACTIVATED
    A1-e  membership multiple OK
    A1-f  catalog_key NOT NULL / UNICO por snapshot (via build_target)

---

## 3. Bloqueante A2 - preservacion del dominio P38 (dictamen #51)

### 3.1. Regla normativa

    target_q4_figi := { figi(K) : K in TARGET_Q4, figi(K) != None }
    target_q1_figi := { figi(K) : K in TARGET_Q1, figi(K) != None }

    TARGET_PAIRWISE := usado SOLO para feasibility (fail-closed check).
                       NO filtra los sets pasados a P38.

**Firma de `compute_contractual_coverage` intacta.** Recibe
`share_class_figi`. NO recibe `catalog_key`.

### 3.2. Razon

`compute_contractual_coverage` necesita recibir los universos
completos de Q4 y Q1. P38 internamente computa:

    TARGET_PAIRWISE = target_q4_figi INTERSECT target_q1_figi
    TARGET_Q4 ^ TARGET_Q1 = simetria

Si el adaptador filtra por pairwise antes de entregar los sets, la
diferencia simetrica desaparece y se rompe la semantica contractual
(F2.4 #24: test `TARGET_Q4 ^ TARGET_Q1`).

### 3.3. Adaptador

    def catalog_to_p38_targets(
        universe_q4, universe_q1, *,
        state_q4, state_q1, pairwise_keys,
    ) -> tuple[set[str], set[str], list[PositionRecord],
               list[PositionRecord], CoverageFeasibility]:
        """Traduce TARGET administrativo a TARGET economico P38.

        IMPORTANTE:
          target_q4_figi: figi(K) para K in TARGET_Q4, figi != None.
                          NO filtrado por pairwise.
          target_q1_figi: idem para TARGET_Q1.
          TARGET_PAIRWISE se usa solo como precondicion de feasibility.
        """

### 3.4. Tests A2

    A2-a  target_q4_figi contiene TODO TARGET_Q4 (no solo pairwise).
    A2-b  target_q1_figi contiene TODO TARGET_Q1 (no solo pairwise).
    A2-c  TARGET_Q4 ^ TARGET_Q1 visible en la salida.
    A2-d  Firma de compute_contractual_coverage intacta.
---

## 4. Bloqueante B3 - imposibilidad de reduccion silenciosa del denominador

**Regla:** la metrica `paired_weighted_share_coverage` no puede reducir
silenciosamente su denominador cuando falta resolucion FIGI.

**Caso prohibido:**

    mapping FAIL -> FIGI ausente -> entrada fuera de TARGET_PAIRWISE
                 -> denominador mas pequeno -> metrica aparentemente OK

**Regla correcta:** si alguna entrada de `TARGET_PAIRWISE` no tiene
identidad economica resoluble, la metrica pasa a `UNAVAILABLE`.
El denominador NO se reduce.

**Estado:** implementado en v9 §3 via `weight_status` (RESOLVED_OBSERVED
/ ZERO_REPORTED / NOT_PRESENT / UNRESOLVED / CONFLICT) + adaptador con
`CoverageFeasibility.UNAVAILABLE`.

**Test B3:**

    entrada con mapping FAIL -> UNAVAILABLE (no denominador reducido)

---

## 5. Bloqueante B4 - fail-closed ante identidades asimetricas

**Regla:** ante identidad asimetrica entre periodos, no resuelta, o
conflictiva -> fail-closed.

**Casos:**

    Q4 = RESOLVED, Q1 = UNRESOLVED
      -> state_q1[K].identity_status == UNRESOLVED
      -> feasible(K) == False -> UNAVAILABLE

    Q4 = RESOLVED (FIGI_X), Q1 = RESOLVED (FIGI_Y)
      -> check_continuity detecta CONFLICT_FIGI_CHANGE
      -> UNAVAILABLE

    Q4 = CONFLICT (candidate_A contradictorio)
      -> UNAVAILABLE

**Regla:** `feasible_state(s)` verdadero solo si:
    s.identity_status == "RESOLVED"
    AND s.weight_status in {"RESOLVED_OBSERVED", "ZERO_REPORTED"}

**NOT_PRESENT -> UNAVAILABLE** (no se convierte a 0.0). Alineado P63.

**Tests B4:**

    B4-a  Q4 RESOLVED + Q1 UNRESOLVED -> UNAVAILABLE
    B4-b  Q4 RESOLVED + Q1 CONFLICT -> UNAVAILABLE
    B4-c  NOT_PRESENT -> UNAVAILABLE (no 0.0)
    B4-d  ZERO_REPORTED -> contribuye 0.0
    B4-e  FIGI change Q4->Q1 -> CONFLICT_FIGI_CHANGE -> UNAVAILABLE

---

## 6. Bloqueante B5 - preservacion de colisiones catalog_key -> FIGI

**Regla:** si dos `catalog_key` mapean al mismo `share_class_figi` en
cualquier periodo, no puede colapsar silenciosamente via `set()`.

**Validator:**

    def check_economic_collision(universe) -> dict[str, list[str]]:
        """Devuelve {share_class_figi: [catalog_key, ...]} para cada
        FIGI con >1 catalog_key.
        """

**Dominio (explicito por #51):**

    collision_q4 = check_economic_collision(universe_q4)
    collision_q1 = check_economic_collision(universe_q1)
    Si collision_q4 no vacio OR collision_q1 no vacio
      -> CATALOG_ECONOMIC_COLLISION -> UNAVAILABLE

**Razon:** P38 fija `share_class_figi` = unidad economica. Dos
`catalog_key` con mismo FIGI implican o duplicidad administrativa, o
agregacion economica que A.6.2-bis NO puede decidir sin modificacion
contractual. Fail-closed.

**Tests B5:**

    B5-a  2 catalog_key -> mismo FIGI -> UNAVAILABLE
    B5-b  2 catalog_key -> FIGI distintos -> OK
    B5-c  colision solo en Q4 -> UNAVAILABLE
    B5-d  colision solo en Q1 -> UNAVAILABLE
    B5-e  no hay colapso silencioso via set()

---

## 7. Flujo normativo completo (9 pasos)

    PASO 1. target_catalog_as_of(Q4), target_catalog_as_of(Q1)   [B2-PIT]
    PASO 2. target_builder.build_target x2 -> universe_q4, universe_q1
    PASO 3. TARGET_PAIRWISE := Q4.declared_keys INTERSECT Q1.declared_keys
            Si vacio -> UNAVAILABLE (STOP)
    PASO 4. state_q4[K], state_q1[K] para K in TARGET_PAIRWISE
    PASO 5. check_continuity (STOP si CONFLICT_FIGI_CHANGE)
    PASO 6. check_economic_collision(Q4) + (Q1) (STOP si colision)
    PASO 7. feasible(K) para K in TARGET_PAIRWISE
            Si alguno False -> UNAVAILABLE (STOP)
    PASO 8. catalog_to_p38_targets(...)
            -> target_q4_figi (full Q4), target_q1_figi (full Q1)
            -> records_q4, records_q1
            -> FEASIBLE
    PASO 9. compute_contractual_coverage(target_q4_figi, target_q1_figi,
                                          records_q4, records_q1)   [P38]
            -> VALID

**Invariante:** `compute_contractual_coverage` SOLO se invoca si
PASO 1-8 superados sin STOP.
---

## 8. Plan de implementacion (subfase B1)

**Subfase de 3 commits** (patron B2-PIT):

### Commit B1.1 - identidad administrativa

    src/institutional_accumulation/identity/catalog_key.py         (nuevo)
      - formato catalog_key
      - load_assignments / validate_assignment
    data/mappings/catalog_assignments.csv
      - migracion inicial: 242 filas con radar_20260919_0001..0242
    tests/test_catalog_key.py                                     (nuevo)
      - A1-a .. A1-f

### Commit B1.2 - TargetUniverse + estados por periodo

    src/institutional_accumulation/identity/target_builder.py      (nuevo)
      - build_target(snapshot, ...) -> TargetUniverse
      - declared_keys, ticker_by_key, figi_by_key, unresolved_keys
    src/institutional_accumulation/identity/period_state.py        (nuevo)
      - state_q4[K], state_q1[K]
      - weight_status enum
    tests/test_target_builder.py                                   (nuevo)
    tests/test_period_state.py                                     (nuevo)

### Commit B1.3 - flujo normativo + adaptador P38

    src/institutional_accumulation/aggregation/catalog_p38_adapter.py  (nuevo)
      - catalog_to_p38_targets
      - CoverageFeasibility
    src/institutional_accumulation/aggregation/catalog_validator.py    (nuevo)
      - check_continuity
      - check_economic_collision
    tests/test_catalog_p38_adapter.py                              (nuevo)
      - A2-a .. A2-d, B3, B5-a .. B5-e
    tests/test_p66_pipeline.py                                     (nuevo)
      - 9 pasos + STOPs + no invocacion P38

**NO se toca:** coverage.py, nipc.py, delta_shares.py,
security_identity.py, amendments.py, relationships.py,
temporal_validity.py, reporting_dedup.py, catalog_pit.py (B2-PIT).

---

## 9. Criterio de cierre B1

    A1  catalog_key_assignment_unique + 6 tests
    A2  full targets P38 + 4 tests
    B3  denominador no reducible + 1 test
    B4  fail-closed asimetrico + 5 tests
    B5  colisiones Q4+Q1 + 5 tests

    Flujo normativo 9 pasos + STOPs + no invocacion P38 + 1 test
    P38 tests PASS (existentes intactos)
    P65 tests PASS (31) + P66 tests PASS (31)
    B2-PIT tests PASS (16, ya cerrados)
    compileall OK + pyflakes LIMPIO

Cierre: **B1 CLOSED -> A.6.3 AUTHORIZED** (si el auditor lo confirma).

---

## 10. Preguntas al auditor

1. **A1 - Formulacion.** `catalog_key_assignment_unique` + membership
   multiple valida. ¿Se aprueba?

2. **A1 - Registro.** `catalog_assignments.csv` con columnas
   `catalog_key`/`fecha_alta`/`source`/`estado`/`fecha_retiro`.
   ¿Se aprueba?

3. **A1 - Migracion inicial.** 242 filas reciben
   `radar_20260919_0001..0242` en orden alfabetico por ticker.
   ¿Se aprueba?

4. **A2 - Adaptador.** `target_q4_figi` = TODO TARGET_Q4 (no solo
   pairwise). `TARGET_PAIRWISE` solo para feasibility. ¿Se aprueba?

5. **A2 - Equivalencia semantica.** Los sets que recibe P38 son
   semanticamente equivalentes a los definidos por P38 antes de
   A.6.2-bis ("universo economico completo"). ¿Se aprueba?

6. **B3/B4/B5.** Los 3 bloqueantes adicionales de #53 §11 se
   reformulan explicitamente (§4-§6). ¿Se aprueba el conjunto?

7. **Plan de implementacion.** 3 commits (B1.1 identidad admin,
   B1.2 TargetUniverse + estados, B1.3 flujo + adaptador). ¿Se aprueba?

8. **Criterio de cierre.** §9 + global. ¿Se aprueba?

9. **Cierre B1 -> A.6.3.** Tras cierre B1, ¿se autoriza A.6.3
   (test contractual P38 de pairing)? ¿O queda sujeto a otro
   dictamen?

---

## 11. Lo que NO se toca en B1

- `coverage.py` (P38) - firma y semantica INTACTAS.
- `nipc.py`, `delta_shares.py`.
- `security_identity.py`, `amendments.py`, `relationships.py`.
- `temporal_validity.py`.
- `reporting_dedup.py`.
- `catalog_pit.py` (B2-PIT ya cerrado por #53).
- Contratos: `NIPC_CONTRATOS_SEMANTICOS_v1.md`, `NIPC_COVERAGE_POLICY.md` v1.0.
- OpenFIGI masivo: NO.
- `DROP_DUP`: NO.
- Push: NO.
- Certificacion "acumulacion": NO.

---

## 12. Trazabilidad

    Dictamen #51     2 bloqueos A1 (catalog_key) + A2 (dominio P38)
    Dictamen #52     Opcion 1: B2-PIT separado, B1 continua en diseno
    Dictamen #53     B2-PIT CERRADO. Siguiente ciclo recomendado: B1
                     con 5 bloqueantes (2 heredados + 3 adicionales)
    A62BIS_PROPUESTA.md v9 §2 y §3     base reutilizada
    A62BIS_B2_PIT_SUBFASE.md           patron de subfase aislada
    FASE_A6_PLAN.md                    seccion A.6.2-bis-B1

---

Fin del documento B1. Sometido a dictamen antes de implementacion.
HEAD 21c30e1.