# IAE - A.6.2-bis Propuesta de rediseno arquitectonico TARGET (v9)

**Version:** v9. Aplicados los 2 bloqueos materiales del dictamen #51:
(A) asignacion de catalog_key (distinguir identidad unica de
membership); (B) preservacion del dominio semantico completo de P38
por el adaptador (targets completos, no filtrados por pairwise).

**HEAD base:** 3315259 (commit de la v8).

**Versiones previas:** v1 (f383932, NO-GO #44), v2 (9aa0727, GO COND
#45), v3 (fa03a97, NO-GO #46), v4 (30ed3df, NO-GO #47), v5 (6c5e33c,
NO-GO #48), v6 (641ef38, NO-GO #49), v7 (2f1f323, NO-GO #50), v8
(3315259, NO-GO #51).

**Objeto:** propuesta de diseno para materializar los 3 bloqueantes
estructurales F2.4 (B1 + B2 + B3).

**Fecha:** 2026-09-21.
**Naturaleza:** propuesta. NO normativa. Sometida a dictamen.

---

## 0. Resumen ejecutivo

v9 cierra los 2 bloqueos del #51:

**Cierre A (asignacion de catalog_key):** se sustituye "unicidad
global" por `catalog_key_assignment_unique`. La misma `catalog_key`
PUEDE aparecer en N snapshots durante su vigencia. El validator
comprueba: 1 alta por key, `fecha_alta` inmutable, sin reasignacion
post-retiro. Errores: `CatalogKeyReassigned`,
`CatalogKeyRetiredReactivated`.

**Cierre B (dominio P38):** el adaptador pasa targets economicos
COMPLETOS de Q4 y Q1 a `compute_contractual_coverage`:
  target_q4_figi = { figi(K) : K in TARGET_Q4, figi(K) != None }
  target_q1_figi = { figi(K) : K in TARGET_Q1, figi(K) != None }
`TARGET_PAIRWISE` se usa SOLO para feasibility (fail-closed check).
NO filtra los sets. P38 internamente computa TARGET_PAIRWISE y
preserva TARGET_Q4 ^ TARGET_Q1.

**Sin cambios:** weight_status, NOT_PRESENT, TARGET_PAIRWISE formal,
flujo normativo 9 pasos, check_continuity, check_economic_collision,
empty pairwise, B2, B3. Aprobados en dictamenes #50/#51.

**Fuera de alcance:** OpenFIGI masivo, recalculo de evidencia final,
modificacion de contratos, `DROP_DUP`, certificacion "acumulacion",
Policy v1.3, Gate-NIPC.2/3.

---

## 1. Estado de partida

| Bloqueante | Piezas existentes | Piezas ausentes |
|---|---|---|
| B1 | coverage.py (P38), PositionRecord, target_universe.resolve_cusips | target_builder.py, catalog_key, adaptador P38, validators |
| B2 | source_date (no contractual) | versionado + target_catalog_as_of |
| B3 | Doctrina P63/P64/P65 | knowledge_date por posicion, absence.py |

Gate 0 catalogo (2026-09-21): 242 filas, radar_ticker unico,
share_class_figi 240 unicos (0 colisiones), 2 MISS.

Migracion inicial: 242 filas reciben `catalog_key =
"radar_20260919_<NNNN>"`. `radar_ticker` versionado.

---

## 2. Bloqueo A - Asignacion de catalog_key (resuelto)

### 2.1. Dos conceptos separados

    IDENTITY ASSIGNMENT UNIQUENESS
      Una catalog_key se asigna UNA SOLA VEZ en su ciclo de vida
      a una entidad administrativa concreta. La fecha de alta no
      cambia.

    SNAPSHOT MEMBERSHIP
      La misma catalog_key PUEDE aparecer en N snapshots durante
      su vigencia. Esto es persistencia correcta, NO reutilizacion.

**El validator NO falla porque K1 aparezca en snapshot_1,
snapshot_2, snapshot_3.**

### 2.2. `catalog_key_assignment_unique` (formulacion normativa)

    Una asignacion de catalog_key es unica globalmente.
    La misma catalog_key puede aparecer en multiples snapshots
    durante su vigencia.

**Estructura de la asignacion** (registro administrativo):

    data/mappings/catalog_assignments.csv
    columnas:
        catalog_key           "radar_<YYYYMMDD>_<NNNN>"
        fecha_alta            fecha de primera asignacion (inmutable)
        source                "radar_initial_20260919" | ...
        estado                ACTIVE | RETIRED
        fecha_retiro          NULL | fecha (si RETIRED)

**Regla:** cada `catalog_key` tiene EXACTAMENTE 1 fila en
`catalog_assignments.csv`. La `fecha_alta` es inmutable.

### 2.3. `catalog_validator.validate_assignment(assignments_df)`

    def validate_assignment(assignments_df) -> dict[str, str]:
        """Devuelve {catalog_key: error_code} para conflictos.

        Errores:
          - catalog_key duplicado en assignments
            -> CATALOG_KEY_REASSIGNED
          - catalog_key re-aparece con fecha_alta distinta a la original
            -> CATALOG_KEY_REASSIGNED
          - catalog_key marcado RETIRED y reasignado a otra entidad
            -> CATALOG_KEY_RETIRED_REACTIVATED
        """

**Validaciones cruzadas con snapshots:**

    Para cada K en assignments:
      K puede aparecer en 0..N snapshots (membresia).
      K en snapshot NO actualiza fecha_alta.
      K presente en snapshot con fecha posterior al retiro
        -> CATALOG_KEY_RETIRED_REACTIVATED.

### 2.4. Tests A (bloqueo A)

**Caso A1 - persistencia correcta:**

    snapshot Q4: K1 -> ticker ABC
    snapshot Q1: K1 -> ticker XYZ
    assignments: K1 fecha_alta=20260919
    -> OK (persistencia correcta, NO CatalogKeyReused)

**Caso A2 - reasignacion (bloqueo):**

    assignments:
      K1 fecha_alta=20260919
      K1 fecha_alta=20261015  (misma key, distinta fecha)
    -> CATALOG_KEY_REASSIGNED

**Caso A3 - retired reactivated:**

    assignments:
      K1 fecha_alta=20260919, estado=RETIRED, fecha_retiro=20261001
    snapshot Q1 (2026-03-31): K1 presente
    -> OK (aun vigente en Q1)

    snapshot 2027: K1 presente, con fecha_alta=20261015
    -> CATALOG_KEY_RETIRED_REACTIVATED

**Caso A4 - membership multiple OK:**

    K1 en snapshot Q4, Q1, Q2, Q3
    assignments: 1 fila, fecha_alta=20260919
    -> OK (membresia multiple valida)
---

## 3. Bloqueo B - Preservacion del dominio P38 (resuelto)

### 3.1. Regla normativa

**El adaptador pasa a P38 los targets economicos COMPLETOS de Q4 y Q1,
NO filtrados por pairwise.**

    target_q4_figi := { figi(K) : K in TARGET_Q4, figi(K) != None }
    target_q1_figi := { figi(K) : K in TARGET_Q1, figi(K) != None }

**`TARGET_PAIRWISE` se usa SOLO para feasibility (fail-closed check).**
NO filtra los sets pasados a P38.

### 3.2. Por que

`compute_contractual_coverage` (contrato P38) necesita recibir:

    target_q4_figi  -> conjunto economico Q4
    target_q1_figi  -> conjunto economico Q1

Y P38 internamente computa:

    TARGET_PAIRWISE = target_q4_figi INTERSECT target_q1_figi
    TARGET_Q4 ^ TARGET_Q1 = simetria (via las operaciones internas)

**Si el adaptador solo pasa claves pairwise a P38**, la diferencia
simetrica `TARGET_Q4 ^ TARGET_Q1` desaparece antes de llegar a P38,
rompiendo la semantica contractual de P38 y el requisito F2.4 #24.

### 3.3. Verificacion de equivalencia semantica

**Precision obligatoria:** el adaptador debe demostrar la equivalencia:

    inputs originales P38  ==  inputs producidos por el adaptador

    Equivalencia semantica:
      Los sets target_q4_figi / target_q1_figi producidos por el
      adaptador representan exactamente lo mismo que los sets
      target_q4 / target_q1 definidos por P38 antes de A.6.2-bis:
      "el universo economico completo (share_class_figi) declarado
       por el catalogo para cada periodo".

**Documentacion obligatoria** (en docstring del adaptador):

    target_q4_figi  -> TODO TARGET_Q4 (no solo pairwise).
    target_q1_figi  -> TODO TARGET_Q1 (no solo pairwise).
    TARGET_PAIRWISE -> usado para feasibility, no para filtrar sets.

### 3.4. Flujo normativo revisado (sobre el de #50)

    PASO 1. target_catalog_as_of(Q4), target_catalog_as_of(Q1)
    PASO 2. target_builder.build_target x2
    PASO 3. TARGET_PAIRWISE := Q4_keys INTERSECT Q1_keys
            Si vacio -> UNAVAILABLE (STOP)
    PASO 4. state_q4[K], state_q1[K] para K in TARGET_PAIRWISE
    PASO 5. check_continuity (STOP si conflicto)
    PASO 6. check_economic_collision(Q4) + (Q1) (STOP si colision)
    PASO 7. feasible(K) para K in TARGET_PAIRWISE
            Si alguno False -> UNAVAILABLE (STOP)
    PASO 8. catalog_to_p38_targets(...)
            -> target_q4_figi = { figi(K) : K in TARGET_Q4, figi != None }
            -> target_q1_figi = { figi(K) : K in TARGET_Q1, figi != None }
            -> records_q4, records_q1 (por K en TARGET_Q4 / TARGET_Q1)
            -> FEASIBLE
    PASO 9. compute_contractual_coverage(target_q4_figi, target_q1_figi,
                                          records_q4, records_q1)
            -> VALID

**Diferencia con v8:** PASO 8 produce sets **completos** (TARGET_Q4 y
TARGET_Q1), no sets filtrados por TARGET_PAIRWISE.

### 3.5. Adaptador actualizado

    def catalog_to_p38_targets(
        universe_q4, universe_q1,
        *, state_q4, state_q1, pairwise_keys,
    ) -> tuple[set[str], set[str], list[PositionRecord],
               list[PositionRecord], CoverageFeasibility]:
        """Traduce TARGET administrativo a TARGET economico P38.

        Precondiciones del flujo normativo (§3.4, PASOS 3-7):
          - TARGET_PAIRWISE no vacio
          - check_continuity OK
          - check_economic_collision OK
          - Todas las K in TARGET_PAIRWISE tienen feasible(K) == True

        Salida:
          target_q4_figi: figi(K) para K in TARGET_Q4, figi != None.
          target_q1_figi: figi(K) para K in TARGET_Q1, figi != None.
          records_q4:    PositionRecord por K in TARGET_Q4.
          records_q1:    PositionRecord por K in TARGET_Q1.
          CoverageFeasibility.FEASIBLE.

        IMPORTANTE:
          Los sets NO se filtran por TARGET_PAIRWISE. Solo
          feasibility (precondicion) usa pairwise_keys.
          Esto preserva la semantica P38 de TARGET_Q4 ^ TARGET_Q1.
        """

### 3.6. Tests B (bloqueo B)

**Caso B1 - TARGET_Q4 ^ TARGET_Q1 preservado:**

    TARGET_Q4 = {K_A, K_B}
    TARGET_Q1 = {K_A, K_C}
    TARGET_PAIRWISE = {K_A}
    figi: K_A->FIGI_A, K_B->FIGI_B, K_C->FIGI_C

    -> target_q4_figi = {FIGI_A, FIGI_B}
    -> target_q1_figi = {FIGI_A, FIGI_C}
    -> P38 recibe ambos conjuntos completos.
    -> P38 puede reconstruir diferencia simetrica {FIGI_B, FIGI_C}.

**Caso B2 - No hay filtrado por pairwise:**

    Si el adaptador filtrara por pairwise, target_q4_figi seria
    {FIGI_A} (solo pairwise). El test verifica explicitamente que
    NO es asi.

**Caso B3 - Symmetric difference visible:**

    assert FIGI_B in target_q4_figi
    assert FIGI_B not in target_q1_figi
    assert FIGI_C in target_q1_figi
    assert FIGI_C not in target_q4_figi

### 3.7. Arquitectura final v9 (con el ajuste)

    period_end
        v
    target_catalog_as_of(period_end)
        v
    snapshot + manifest
        v
    target_builder.build_target(snapshot)
        v
    TargetUniverse x 2 (Q4, Q1)
        v
    TARGET_PAIRWISE := universe_q4.declared_keys INTERSECT universe_q1.declared_keys
        v
    state_q4[K], state_q1[K] para K in TARGET_PAIRWISE
        v
    check_continuity (STOP si CONFLICT_FIGI_CHANGE)
        v
    check_economic_collision(Q4) + (Q1) (STOP si colision)
        v
    feasible(K) para K in TARGET_PAIRWISE (STOP si alguno False)
        v
    target_q4_figi = { figi(K) : K in TARGET_Q4, figi != None }
    target_q1_figi = { figi(K) : K in TARGET_Q1, figi != None }
    records_q4 / records_q1 (full Q4 / Q1)
        v
    compute_contractual_coverage(target_q4_figi, target_q1_figi,
                                  records_q4, records_q1)
        v
    VALID

---

## 4. Sin cambios respecto a v8

Las siguientes piezas fueron aprobadas en #50/#51 y NO se modifican:

- flujo normativo 9 pasos (§3.4 la version v9 solo cambia PASO 8).
- check_continuity (PASO 5).
- check_economic_collision Q4+Q1 (PASO 6).
- TARGET_PAIRWISE formal (PASO 3) + empty_set -> UNAVAILABLE.
- estados por periodo (state_q4/state_q1).
- weight_status + NOT_PRESENT fail-closed.
- adaptador P38 (firma P38 intacta).
- coverage.py: sin cambio contractual.
- B2 (snapshots, manifest, intervalos, no backdating, "que cubren").
- B3 (knowledge_date, RESTATEMENT, NEW HOLDINGS, N/D).
- absence.py stub.
---

## 5. Orden de commits

    1. B2  versionado + target_catalog_as_of + catalog_validator
           (incluye validate_assignment)
    2. B1  target_builder + TargetUniverse + catalog_key
           (incluye migracion catalog_assignments.csv)
    3. B1  flujo normativo + TARGET_PAIRWISE + estados por periodo
           + check_continuity + check_economic_collision
           + adaptador P38 (targets completos)
    4. B3  Timestamps + PositionRecord + provenance + absence.py
    5. Integracion end-to-end + verificacion global

---

## 6. Tests requeridos (resumen)

| Bloque | Fichero | Cobertura |
|---|---|---|
| B1 | `tests/test_catalog_validator.py` | validate_assignment (A1-A4) + continuidad + colision Q4/Q1 |
| B1 | `tests/test_target_builder.py` | catalog_key estable, ticker versionado |
| B1 | `tests/test_target_pairwise.py` | TARGET_PAIRWISE formal + empty set + estados |
| B1 | `tests/test_p66_pipeline.py` | flujo 9 pasos + STOPs + dominio P38 |
| B1 | `tests/test_catalog_p38_adapter.py` | 9 casos A2 + B1-B3 (symmetric difference preservada) |
| B2 | `tests/test_target_catalog_as_of.py` | snapshots "que cubren", fail-closed |
| B3 | `tests/test_position_record.py` | RESTATEMENT, NEW HOLDINGS, ambiguous |
| B3 | `tests/test_absence.py` | enums + NotImplementedError |

**Casos obligatorios adicionales #51:**

    persistencia catalog_key (A1)
    reasignacion (A2)
    retired reactivated (A3)
    membership multiple OK (A4)
    symmetric difference preservada (B1-B3)

---

## 7. Criterio de cierre A.6.2-bis

### B1 - Bloqueo A (asignacion)

- `catalog_key_assignment_unique` formalizado.
- `validate_assignment` comprueba: 1 alta, fecha_alta inmutable,
  sin reasignacion, retired sin reactivar.
- Membership multiple es valida (NO es reutilizacion).
- Errores: `CatalogKeyReassigned`, `CatalogKeyRetiredReactivated`.
- Tests A1-A4.

### B1 - Bloqueo B (dominio P38)

- `target_q4_figi = { figi(K) : K in TARGET_Q4, figi != None }`
- `target_q1_figi = { figi(K) : K in TARGET_Q1, figi != None }`
- `TARGET_PAIRWISE` solo para feasibility.
- P38 recibe targets completos.
- `TARGET_Q4 ^ TARGET_Q1` preservado (test explicito B1-B3).
- Firma P38 intacta.

### B1 - Resto (aprobado #50/#51)

- Flujo 9 pasos con check_continuity en PASO 5.
- Dominio collision Q4 + Q1.
- TARGET_PAIRWISE empty -> UNAVAILABLE.
- weight_status + NOT_PRESENT fail-closed.
- Adaptador invocado solo si flujo pasa.

### B2

- 0/1/>1 snapshots **que cubren** period_end.
- Hash invalido -> FAIL-CLOSED.
- No backdating.

### B3

- knowledge_date == filing_date con ASSIGNED.
- RESTATEMENT / NEW HOLDINGS / ambiguous -> N/D.

### Global

- P38 tests PASS.
- P65 tests PASS (31).
- P66 tests PASS (31).
- A.6.2-bis tests PASS.
- compileall OK + pyflakes LIMPIO.

Solo entonces: **A.6.2-bis CLOSED -> A.6.3 AUTHORIZED.**

---

## 8. Preguntas al auditor (v9)

1. **Bloqueo A - Formulacion.** `catalog_key_assignment_unique` +
   membership multiple valida. ¿Se aprueba?

2. **Bloqueo A - Registro.** `catalog_assignments.csv` con
   `catalog_key` + `fecha_alta` + `source` + `estado` + `fecha_retiro`.
   ¿Se aprueba?

3. **Bloqueo A - Errores.** `CatalogKeyReassigned` +
   `CatalogKeyRetiredReactivated`. ¿Se aprueba el conjunto?

4. **Bloqueo B - Adaptador.** `target_q4_figi` = TODO TARGET_Q4 (no
   solo pairwise). `TARGET_PAIRWISE` solo para feasibility. ¿Se aprueba?

5. **Bloqueo B - Verificacion.** `TARGET_Q4 ^ TARGET_Q1` preservado
   hasta P38 (test explicito B1-B3). ¿Se aprueba?

6. **Resto.** weight_status, NOT_PRESENT, TARGET_PAIRWISE formal,
   check_continuity, collision Q4/Q1, empty pairwise, B2, B3: aprobados
   en #50/#51. ¿Sin cambios?

7. **Cierre.** Los 2 bloques de §7 + resto + global.

---

## 9. Lo que NO se toca en A.6.2-bis

- Contratos: `NIPC_CONTRATOS_SEMANTICOS_v1.md`, `NIPC_COVERAGE_POLICY.md` v1.0.
- Modelo economico P38 (share_class_figi, Q12 Modelo A).
- Modulos: delta_shares.py, security_identity.py, relationships.py,
  amendments.py, temporal_validity.py.
- coverage.py: firma y semantica INTACTAS.
- radar_target_catalog.csv actual: snapshot inicial + migracion.
- OpenFIGI masivo: NO.
- DROP_DUP: NO.
- Push: NO.
- Certificacion "acumulacion": NO.

---

## 10. Trazabilidad

    Dictamen #43      A.6.0 CERRADO + A.6.2-bis AUTORIZADO
    Dictamen #44      v1 NO-GO
    Dictamen #45      v2 GO COND
    Dictamen #46      v3 NO-GO
    Dictamen #47      v4 NO-GO
    Dictamen #48      v5 NO-GO
    Dictamen #49      v6 NO-GO
    Dictamen #50      v7 NO-GO
    Dictamen #51      v8 NO-GO (asignacion + dominio P38)
    Gate 0 catalogo   v9 seccion 1
    F2.4 #24          3 bloqueantes estructurales
    P38 seccion 3     unidad = share_class_figi
    P38 F2.4 #24      test TARGET_Q4 ^ TARGET_Q1
    P60 seccion 1     identity_type obligatorio
    P63 seccion 12    absence semantics
    P64 seccion 13    RESTATEMENT / NEW HOLDINGS
    P65 seccion 14    L3 booleano

---

Fin de la propuesta v9. Sometida a verificacion documental.
HEAD base 3315259 (v8).