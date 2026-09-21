# IAE - A.6.2-bis-B1 Subfase (v3)

**Version:** v3. Aplicados los 5 bloqueantes del dictamen #55:
(B1-NEW-1) snapshot_row_uid estable por contenido; (B1-NEW-2)
catalog_key inmutable; (B1-NEW-3) membership temporal; (B1-NEW-4)
full resolution por identity_status; (B1-NEW-5) assigned_entity_id
declarado como identificador administrativo artificial.

**Versiones previas:** v1 (003c5c2, NO-GO #54), v2 (95598ac, NO-GO #55).

**Referencias:** dictamenes #51, #52, #53, #54, #55.
**B2-PIT:** CLOSED (#53). B1 no lo reabre.

**Fecha:** 2026-09-21.
**HEAD al redactar:** 95598ac.
**Naturaleza:** propuesta de subfase v3. NO normativa.

---

## 0. Resumen ejecutivo

B1 responde a "¿que TARGET contractual representa el catalogo y como
se traduce al dominio economico P38?".

**5 correcciones materiales de #55:**

1. **B1-NEW-1:** identificador de fila NO posicional. Nuevo
   `snapshot_row_uid` derivado del contenido canonico de la fila.
2. **B1-NEW-2:** `catalog_key` inmutable. No hay historial multi-fila
   por key. Cambio de entidad -> nueva key. Intento de reasignacion
   -> evento de auditoria + `CATALOG_KEY_REASSIGNED`.
3. **B1-NEW-3:** `validate_membership` cruza vigencia temporal de la
   asignacion con `[valid_from, valid_to)` del snapshot.
4. **B1-NEW-4:** full resolution = `identity_status == RESOLVED` Y
   exactamente 1 `share_class_figi`. NO `figi != None`.
5. **B1-NEW-5:** `assigned_entity_id` declarado como identificador
   administrativo interno artificial, sin significado economico.

**Fuera de alcance:** P38 (no se modifica), OpenFIGI masivo,
DROP_DUP, certificacion.

---

## 1. Estado de partida

### 1.1. B2-PIT aporta (CERRADO #53)

    snapshot materializado (242 filas, SIN catalog_key)
    catalog_manifest.json
    version_id + sha256 externo
    [valid_from, valid_to)
    target_catalog_as_of(period_end)

B1 consume B2-PIT. NO lo modifica. NO reescribe snapshots.

### 1.2. B1 construye

    catalog_assignments.csv      (key <-> entidad, 1:1 inmutable)
    catalog_reassignments_attempted.csv   (auditoria de intentos)
    catalog_membership.csv       (version_id <-> key <-> row_uid)
    catalog_key.py               (formato + validators)
    target_builder.py            (puro, snapshot + membership)
    period_state.py              (state_q4, state_q1)
    catalog_p38_adapter.py       (full targets)
    catalog_validator.py         (continuity, collision, full_resolution)

### 1.3. Prohibiciones

    NO modificar P38.
    NO activar OpenFIGI masivo.
    NO activar DROP_DUP.
    NO modificar contratos normativos.
    NO reacoplar a B2-PIT.
    NO reescribir snapshots publicados.
---

## 2. A1 - catalog_key inmutable + entidad trazable

### 2.1. Semantica normativa

    catalog_key = identidad administrativa estable.
    K -> assigned_entity_id es 1:1 para toda la vida de K.
    K NO se reasigna jamas.

**Cambio de entidad -> nueva key.**

    K_old -> A
    K_new -> B

NO:
    K -> A
    K -> B   (prohibido)

### 2.2. Formato de catalog_key

    catalog_key := "radar_<YYYYMMDD_alta>_<NNNN>"

- NO es el hash del snapshot.
- NO depende del ticker.
- NO depende del orden alfabetico (solo se uso para la migracion inicial).

### 2.3. `catalog_assignments.csv` (una fila por key, inmutable)

    catalog_key           "radar_<YYYYMMDD>_<NNNN>"
    assigned_entity_id    identificador administrativo interno
                          (ver 2.4). 1:1 con catalog_key.
    valid_from            fecha de alta (inmutable)
    valid_to              fecha de retiro (NULL = vigente)
    source                "radar_initial_20260919" | ...
    reason                texto libre (notas de asignacion)

**Regla:** EXACTAMENTE 1 fila por `catalog_key`. No hay historial
multi-fila por key. Un cambio de entidad implica una key nueva.

### 2.4. Semantica de `assigned_entity_id` (B1-NEW-5)

**Declaracion explicita:**

    assigned_entity_id = "radar_entity_<NNNN>"

    Es un IDENTIFICADOR ADMINISTRATIVO INTERNO ARTIFICIAL.
    NO tiene significado economico.
    Su semantica es: "una entidad administrativa del radar,
    distinta de cualquier otra".

    Correspondencia estable 1:1 con la fila del snapshot fuente
    (via snapshot_row_uid, ver seccion 3).

**NO se interpreta como:** FIGI, CUSIP, ticker, LEI, ni cualquier
identificador economico externo.

**Migracion inicial:** 242 filas, una por key, con
`assigned_entity_id = "radar_entity_<NNNN>"` (mismo sufijo que
la key). `valid_from = 2026-09-19`. `valid_to = NULL`.

### 2.5. Registro de intentos de reasignacion

    data/mappings/catalog_reassignments_attempted.csv
    columnas:
        catalog_key           key original
        attempted_entity_id   entidad intentada (distinta)
        attempted_at          fecha del intento
        source                origen del intento
        reason                motivo documentado del intento

Registro de auditoria. NO altera `catalog_assignments.csv`.
Cada fila es un intento bloqueado. Se conserva para trazabilidad.

### 2.6. Validator

    def validate_assignment(assignments_df, attempted_df=None) -> dict:
        """Devuelve {catalog_key: error_code} para conflictos.

        Errores:
          CATALOG_KEY_REASSIGNED
            catalog_key aparece con assigned_entity_id distinto
            al original en assignments_df, o en attempted_df
            con attempted_entity_id != original.

          CATALOG_KEY_RETIRED_REACTIVATED
            key con valid_to caducado aparece con valid_from
            posterior (reactivacion).
        """

### 2.7. Tests A1

    A1-a  misma key + misma entidad + multiples snapshots -> OK
    A1-b  misma key con entidad distinta en assignments -> CATALOG_KEY_REASSIGNED
    A1-c  intento de reasignacion en attempted_df -> CATALOG_KEY_REASSIGNED
    A1-d  retired sin reactivar -> OK
    A1-e  retired y reactivado -> CATALOG_KEY_RETIRED_REACTIVATED
    A1-f  catalog_key NOT NULL + UNICO por snapshot (membership)
    A1-g  catalog_key UNICO global (una unica alta inicial)
    A1-h  K1 -> entidad A, intento K1 -> entidad B -> FAIL
    A1-i  misma K + misma entidad en historico -> OK

---

## 3. A1-membership - Relacion snapshot <-> catalog_key (B1-NEW-1)

### 3.1. Problema

El indice `snapshot_row_id = 0..N-1` es posicional por definicion.
NO cumple "independiente del orden fisico del CSV".

### 3.2. Solucion: `snapshot_row_uid` estable por contenido

    snapshot_row_uid := sha256_corto(fila_canonica)

donde `fila_canonica` es una representacion canonica e inmutable
de la fila:

    - columnas en orden alfabetico
    - valores normalizados (strip + UTF-8 + null como string vacio)
    - concatenadas con separador \x1f
    - sha256 completo, truncado a 16 hex chars

**Propiedades:**

- Independiente de la posicion fisica.
- Independiente del orden de las columnas.
- Detecta duplicados (2 filas con mismo contenido -> mismo uid
  -> error).
- Reordenar el CSV no cambia los uids.

### 3.3. `catalog_membership.csv`

    version_id            "20260921_01"
    catalog_key           "radar_20260919_<NNNN>"
    snapshot_row_uid      sha256_corto hex (16 chars)

**Publicado aparte.** NO modifica el snapshot. El snapshot permanece
inmutable.

**Invariantes:**

- Para cada `version_id`: el conjunto de `snapshot_row_uid` cubre
  exactamente el conjunto de uids calculados sobre el snapshot.
- Cada `snapshot_row_uid` mapea a exactamente 1 `catalog_key`.
- Cada `catalog_key` aparece como maximo 1 vez por `version_id`.

### 3.4. Validacion temporal (B1-NEW-3)

`validate_membership(membership_df, assignments_df, manifest)`:

    Para cada (version_id, catalog_key):
      - version_id existe en manifest B2-PIT.
      - catalog_key declarada en assignments.
      - Existe interseccion no vacia entre:
          [assignment.valid_from, assignment.valid_to)
          [snapshot.valid_from, snapshot.valid_to)
        Si no hay interseccion -> error CATALOG_KEY_NOT_VALID_FOR_SNAPSHOT.

**Ejemplo prohibido:**

    K1 assignment:  [2026-09-19, 2026-10-01)
    snapshot V3:    [2026-11-01, null)
    membership V3 -> K1
    -> CATALOG_KEY_NOT_VALID_FOR_SNAPSHOT

**Regla:** una key RETIRED no puede pertenecer a un snapshot posterior
a su `valid_to`.

### 3.5. `target_builder.build_target(...)`

    def build_target(snapshot_df, membership_df, assignments_df, *,
                     version_id, period_end,
                     catalog_version_id, catalog_sha256) -> TargetUniverse:
        """Materializa el TargetUniverse a partir del snapshot B2-PIT
        + membership B1 + assignments B1.

        Para cada fila del snapshot:
          - row_uid = sha256_corto(fila_canonica)
          - catalog_key = membership_df[row_uid].catalog_key
          - ticker = fila.radar_ticker
          - figi = fila.share_class_figi

        Vinculacion por snapshot_row_uid. NO por posicion.
        NO por ticker.

        Validaciones:
          - catalog_key NOT NULL por fila.
          - catalog_key UNICO por snapshot.
          - membership cubre TODOS los row_uid del snapshot.
          - catalog_key vigente para el snapshot (validacion temporal).
        """

### 3.6. Tests membership

    M-a  snapshot V1 + K1 -> declared_keys={K1}
    M-b  snapshot V2 + K1 (persistencia) -> declared_keys={K1}
    M-c  snapshot V2 + K1 + K2 -> declared_keys={K1, K2}
    M-d  membership con hueco -> FALLA
    M-e  membership con duplicado de catalog_key -> FALLA
    M-f  membership referencia version_id inexistente -> FALLA
    M-g  reordenacion fisica del CSV no rompe K <-> fila (B1-NEW-1)
    M-h  K1 fuera de vigencia del snapshot -> CATALOG_KEY_NOT_VALID_FOR_SNAPSHOT
    M-i  K1 retired en snapshot posterior -> FALLA
    M-j  filas con contenido identico -> mismo uid -> error duplicado
---

## 4. A2 - Dominio P38 (B1-NEW-4: full resolution por identity_status)

### 4.1. Regla normativa

**Antes de invocar P38, verificacion de resolucion completa:**

    Para todo K in (TARGET_Q4 UNION TARGET_Q1):
      si identity_status(K) != RESOLVED
        O existe != 1 share_class_figi(K)
          -> UNAVAILABLE
          -> NO invocar compute_contractual_coverage

Estados que bloquean:

    UNRESOLVED
    CONFLICT
    AMBIGUOUS
    (NOT_PRESENT en identity_status)

`figi != None` NO es suficiente. Un estado CONFLICT puede conservar
candidatos FIGI sin constituir identidad economica resuelta.

**Solo si TODOS los K cumplen RESOLVED + 1 FIGI:**

    target_q4_figi := { unico_figi(K) : K in TARGET_Q4 }
    target_q1_figi := { unico_figi(K) : K in TARGET_Q1 }

Sin filtro. Todos los K de Q4 y Q1 aparecen.

### 4.2. Adaptador

    def catalog_to_p38_targets(
        universe_q4, universe_q1, *,
        state_q4, state_q1, pairwise_keys,
    ) -> tuple[set[str], set[str], list[PositionRecord],
               list[PositionRecord], CoverageFeasibility]:
        """Traduce TARGET administrativo a TARGET economico P38.

        Precondiciones del flujo normativo (PASOS 1-9):
          - TARGET_PAIRWISE no vacio
          - check_continuity OK
          - check_economic_collision Q4+Q1 OK
          - feasible(K) para K in TARGET_PAIRWISE
          - FULL RESOLUTION de TARGET_Q4 UNION TARGET_Q1:
            identity_status == RESOLVED
            Y exactamente 1 share_class_figi por K

        Salida:
          target_q4_figi: unico_figi(K) para TODO K in TARGET_Q4.
          target_q1_figi: unico_figi(K) para TODO K in TARGET_Q1.
          records_q4, records_q1.
          CoverageFeasibility.FEASIBLE.
        """

### 4.3. Tests A2

    A2-a  K in TARGET_Q4 RESOLVED -> su figi en target_q4_figi
    A2-b  K in TARGET_Q4 UNRESOLVED -> UNAVAILABLE antes de P38
    A2-c  K in TARGET_Q4 CONFLICT con candidato FIGI -> UNAVAILABLE
    A2-d  K solo Q4 -> contribuye a target_q4_figi
    A2-e  K solo Q1 -> contribuye a target_q1_figi
    A2-f  TARGET_Q4 ^ TARGET_Q1 visible
    A2-g  Sin filtro: todos los K resueltos aparecen en el set

---

## 5. B3 - Imposibilidad de reduccion silenciosa del denominador

**Regla consolidada con A2:**

    Si cualquier K in (TARGET_Q4 UNION TARGET_Q1) no esta RESOLVED
      -> UNAVAILABLE
    Si cualquier K in TARGET_PAIRWISE no es feasible
      -> UNAVAILABLE

**NO se recalcula el denominador con menos entradas.**

### 5.1. Tests B3 (nuevos por #55)

    B3-a  entrada K solo en Q4 con FIGI no resoluble -> UNAVAILABLE
    B3-b  entrada K en Q4 CONFLICT con candidato FIGI -> UNAVAILABLE
    B3-c  entrada K en Q4 UNRESOLVED con FIGI residual -> UNAVAILABLE
    B3-d  todos RESOLVED -> continua

---

## 6. B4 - Fail-closed ante identidades asimetricas

**Regla:** `feasible_state(s)` verdadero solo si:
    s.identity_status == "RESOLVED"
    AND s.weight_status in {"RESOLVED_OBSERVED", "ZERO_REPORTED"}

**Casos:**

    Q4 RESOLVED + Q1 UNRESOLVED -> UNAVAILABLE
    Q4 RESOLVED + Q1 CONFLICT -> UNAVAILABLE
    NOT_PRESENT -> UNAVAILABLE
    ZERO_REPORTED -> contribuye 0.0
    FIGI change Q4->Q1 -> CONFLICT_FIGI_CHANGE -> UNAVAILABLE

### 6.1. Tests B4

    B4-a  Q4 RESOLVED + Q1 UNRESOLVED -> UNAVAILABLE
    B4-b  Q4 RESOLVED + Q1 CONFLICT -> UNAVAILABLE
    B4-c  NOT_PRESENT -> UNAVAILABLE
    B4-d  ZERO_REPORTED -> contribuye 0.0
    B4-e  FIGI change -> CONFLICT_FIGI_CHANGE -> UNAVAILABLE

---

## 7. B5 - Preservacion de colisiones catalog_key -> FIGI

**Regla:** si 2+ `catalog_key` mapean al mismo `share_class_figi` en
cualquier periodo -> `CATALOG_ECONOMIC_COLLISION` -> `UNAVAILABLE`.

**Validator:**

    def check_economic_collision(universe) -> dict[str, list[str]]:
        """Devuelve {share_class_figi: [catalog_key, ...]} para cada
        FIGI con >1 catalog_key.
        """

**Dominio:** Q4 + Q1. Cualquier colision -> UNAVAILABLE.

### 7.1. Tests B5

    B5-a  2 catalog_key -> mismo FIGI -> UNAVAILABLE
    B5-b  distintos FIGIs -> OK
    B5-c  colision solo en Q4 -> UNAVAILABLE
    B5-d  colision solo en Q1 -> UNAVAILABLE
    B5-e  no hay colapso silencioso via set()
---

## 8. Flujo normativo 10 pasos (v3)

    PASO 0. validate_membership (temporal + cobertura row_uid)   [v3 NUEVO]
            STOP si incoherencia.
    PASO 1. target_catalog_as_of(Q4), target_catalog_as_of(Q1)   [B2-PIT]
    PASO 2. build_target(snapshot, membership, assignments) x2
            -> universe_q4, universe_q1 (vinculacion por row_uid)
    PASO 3. TARGET_PAIRWISE := Q4.declared_keys INTERSECT Q1.declared_keys
            Si vacio -> UNAVAILABLE (STOP)
    PASO 4. state_q4[K], state_q1[K] para K in TARGET_PAIRWISE
    PASO 5. check_continuity (STOP si CONFLICT_FIGI_CHANGE)
    PASO 6. check_economic_collision(Q4) + (Q1) (STOP si colision)
    PASO 7. feasible(K) para K in TARGET_PAIRWISE
            STOP si alguno False
    PASO 8. FULL RESOLUTION de TARGET_Q4 UNION TARGET_Q1         [v2]
            Para todo K in (Q4 UNION Q1):
              identity_status == RESOLVED
              Y exactamente 1 share_class_figi
            STOP si alguno no cumple.
    PASO 9. catalog_to_p38_targets(...)
            -> target_q4_figi (full, sin filtro)
            -> target_q1_figi (full, sin filtro)
            -> records_q4, records_q1
            -> FEASIBLE
    PASO 10. compute_contractual_coverage(target_q4_figi, target_q1_figi,
                                           records_q4, records_q1)  [P38]
            -> VALID

**Invariante:** P38 solo se invoca si PASOS 0-9 superados sin STOP.

**PASO 0 nuevo:** valida membership temporal ANTES de construir
TargetUniverse. Evita que una key fuera de vigencia entre al
universo contractual.

---

## 9. Plan de implementacion (v3)

### Commit B1.1 - identidad + membership + validacion temporal

    src/institutional_accumulation/identity/catalog_key.py         (nuevo)
      - formato catalog_key
      - sha256_corto (row_uid)
      - load_assignments / validate_assignment
      - load_membership / validate_membership (temporal)
    data/mappings/catalog_assignments.csv
      - 242 filas (1 por key, valid_from=2026-09-19)
    data/mappings/catalog_membership.csv
      - 242 filas para version_id=20260921_01
    tests/test_catalog_key.py                                     (nuevo)
      - A1-a .. A1-i
    tests/test_catalog_membership.py                              (nuevo)
      - M-a .. M-j

### Commit B1.2 - TargetUniverse + estados por periodo

    src/institutional_accumulation/identity/target_builder.py      (nuevo)
      - build_target(snapshot, membership, assignments, ...)
      - vinculacion por snapshot_row_uid
    src/institutional_accumulation/identity/period_state.py        (nuevo)
      - state_q4[K], state_q1[K]
      - identity_status enum
      - weight_status enum
    tests/test_target_builder.py                                   (nuevo)
    tests/test_period_state.py                                     (nuevo)

### Commit B1.3 - flujo + adaptador P38

    src/institutional_accumulation/aggregation/catalog_p38_adapter.py  (nuevo)
      - catalog_to_p38_targets (full targets)
    src/institutional_accumulation/aggregation/catalog_validator.py    (nuevo)
      - check_continuity
      - check_economic_collision
      - check_full_resolution (identity_status == RESOLVED)
    tests/test_catalog_p38_adapter.py                              (nuevo)
      - A2-a..g, B3-a..d, B5-a..e
    tests/test_p66_pipeline.py                                     (nuevo)
      - 10 pasos + STOPs + no invocacion P38

**NO se toca:** coverage.py, nipc.py, delta_shares.py,
security_identity.py, amendments.py, relationships.py,
temporal_validity.py, reporting_dedup.py, catalog_pit.py.

---

## 10. Criterio de cierre B1 (v3)

### A1

    A1-a .. A1-i    PASS

### Membership

    M-a .. M-j      PASS
    (incluye M-g reordenacion CSV, M-h vigencia, M-i retired)

### A2

    A2-a .. A2-g    PASS
    Full resolution = identity_status RESOLVED + 1 FIGI

### B3

    B3-a .. B3-d    PASS

### B4

    B4-a .. B4-e    PASS

### B5

    B5-a .. B5-e    PASS

### Flujo

    10 pasos + STOPs + PASO 0 membership + no invocacion P38 sin
    PASO 8 PASS

### Global

    P38 tests PASS (existentes intactos)
    P65 tests PASS (31) + P66 tests PASS (31)
    B2-PIT tests PASS (16)
    compileall OK + pyflakes LIMPIO

### Advertencia de alcance (dictamen #55 seccion 16)

    B1 CLOSED != TARGET historico Q4/Q1 disponible en produccion.
    Tests pueden usar fixtures historicos explicitamente identificados.
    NO fabricar snapshots historicos presentados como evidencia real.

Cierre: **B1 CLOSED -> nuevo dictamen especifico para A.6.3**
(no automatico, segun #54 seccion 16 pregunta 9).
---

## 11. Preguntas al auditor (v3)

1. **B1-NEW-1 - snapshot_row_uid.** SHA-256 corto (16 hex chars)
   sobre representacion canonica de la fila. ¿Se aprueba? ¿Otro
   formato de uid?

2. **B1-NEW-2 - catalog_key inmutable.** K -> assigned_entity_id
   1:1 para toda la vida. Cambio -> nueva key. Intento de
   reasignacion -> evento en `catalog_reassignments_attempted.csv`.
   ¿Se aprueba?

3. **B1-NEW-3 - membership temporal.** Validacion de interseccion
   [assignment.valid_from, assignment.valid_to) con
   [snapshot.valid_from, snapshot.valid_to). Sin interseccion ->
   `CATALOG_KEY_NOT_VALID_FOR_SNAPSHOT`. ¿Se aprueba?

4. **B1-NEW-4 - full resolution.** `identity_status == RESOLVED`
   Y exactamente 1 `share_class_figi`. NO `figi != None`. Estados
   UNRESOLVED/CONFLICT/AMBIGUOUS/NOT_PRESENT bloquean. ¿Se aprueba?

5. **B1-NEW-5 - assigned_entity_id.** Declarado como identificador
   administrativo interno artificial. Correspondencia 1:1 con
   snapshot_row_uid. Sin significado economico. ¿Se aprueba?

6. **Flujo PASO 0.** Nueva validacion `validate_membership` antes
   de `build_target`. ¿Se aprueba?

7. **Plan 3 commits.** B1.1 (identidad + membership + validacion
   temporal), B1.2 (TargetUniverse + estados), B1.3 (flujo +
   adaptador). ¿Se aprueba?

8. **Criterio de cierre.** §10 + advertencia de alcance. ¿Se aprueba?

9. **Observacion de alcance.** B1 CLOSED != disponibilidad de
   TARGET historico Q4/Q1. Tests con fixtures historicos
   identificados; sin fabricar. ¿Se aprueba la advertencia?

10. **Cierre B1 -> A.6.3.** Tras cierre B1, ¿A.6.3 requiere nuevo
    dictamen? (#54 seccion 16 pregunta 9 indico que no es
    automatico.)

---

## 12. Lo que NO se toca en B1

- `coverage.py` (P38) - firma y semantica INTACTAS.
- `nipc.py`, `delta_shares.py`.
- `security_identity.py`, `amendments.py`, `relationships.py`.
- `temporal_validity.py`.
- `reporting_dedup.py`.
- `catalog_pit.py` (B2-PIT ya cerrado por #53).
- Snapshots publicados por B2-PIT (inmutables).
- Contratos: `NIPC_CONTRATOS_SEMANTICOS_v1.md`, `NIPC_COVERAGE_POLICY.md` v1.0.
- OpenFIGI masivo: NO.
- `DROP_DUP`: NO.
- Push: NO.
- Certificacion "acumulacion": NO.

---

## 13. Trazabilidad

    Dictamen #51     2 bloqueos A1 + A2
    Dictamen #52     Opcion 1: B2-PIT separado, B1 continua en diseno
    Dictamen #53     B2-PIT CERRADO. Siguiente ciclo: B1
    Dictamen #54     3 bloqueantes estructurales B1:
                     - full TARGET (no filtro None)
                     - assignment identity + historial
                     - membership snapshot <-> key
    Dictamen #55     5 bloqueantes B1 v2:
                     - B1-NEW-1 snapshot_row_uid estable
                     - B1-NEW-2 catalog_key inmutable
                     - B1-NEW-3 membership temporal
                     - B1-NEW-4 full resolution por identity_status
                     - B1-NEW-5 assigned_entity_id trazable
    A62BIS_PROPUESTA.md v9 §2-§3     base
    A62BIS_B2_PIT_SUBFASE.md         patron de subfase aislada
    FASE_A6_PLAN.md                  seccion A.6.2-bis-B1

---

Fin de la propuesta B1 v3. Sometida a dictamen. HEAD 95598ac.