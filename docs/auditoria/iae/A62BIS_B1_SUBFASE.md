# IAE - A.6.2-bis-B1 Subfase (v4)

**Version:** v4. Aplicados los 4 bloqueantes del dictamen #56:
(B1-NEW-6) cobertura temporal completa (no interseccion);
(B1-NEW-7) row_uid vs identidad longitudinal + predecessor_row_uid;
(B1-NEW-8) estados para TARGET_Q4 UNION TARGET_Q1;
(B1-NEW-9) contrato de entrada B1 explicito.
+ SHA-256 completo + serializacion length-prefixed.

**Versiones previas:** v1 (003c5c2, NO-GO #54), v2 (95598ac, NO-GO #55),
v3 (43373fd, NO-GO #56).

**Referencias:** dictamenes #51 a #56.
**B2-PIT:** CLOSED (#53).

**Fecha:** 2026-09-21.
**HEAD al redactar:** 43373fd.
**Naturaleza:** propuesta de subfase v4. NO normativa.

---

## 0. Resumen ejecutivo

B1 responde a "¿que TARGET contractual representa el catalogo y como
se traduce al dominio economico P38?".

**4 correcciones materiales de #56:**

1. **B1-NEW-6:** cobertura temporal de membership completa, no
   interseccion. Snapshot con `valid_to=NULL` no puede contener
   assignment con `valid_to` definido.
2. **B1-NEW-7:** `snapshot_row_uid` = row-instance id.
   `catalog_key` + `assigned_entity_id` = identidad longitudinal.
   `predecessor_row_uid` en membership para cadenas verificables.
3. **B1-NEW-8:** PASO 4 construye estados para
   `TARGET_Q4 ∪ TARGET_Q1` (no solo pairwise).
4. **B1-NEW-9:** `B1_REQUIRED_COLUMNS = {radar_ticker, share_class_figi}`
   con fail-closed si falta alguna.

**Aplicado adicional:** SHA-256 completo (64 hex chars) en row_uid +
serializacion length-prefixed (evita ambiguedad con `\x1f`).

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

### 1.2. B1 construye

    catalog_assignments.csv                    (key <-> entidad, 1:1 inmutable)
    catalog_reassignments_attempted.csv        (auditoria de intentos)
    catalog_membership.csv                     (con predecessor_row_uid)
    catalog_key.py                             (formato + validators)
    target_builder.py                          (puro, snapshot + membership)
    period_state.py                            (state_q4, state_q1)
    catalog_p38_adapter.py                     (full targets)
    catalog_validator.py                       (continuity, collision, full_res)

### 1.3. Prohibiciones

    NO modificar P38.
    NO activar OpenFIGI masivo.
    NO activar DROP_DUP.
    NO modificar contratos normativos.
    NO reacoplar a B2-PIT.
    NO reescribir snapshots publicados.

---

## 2. A1 - catalog_key inmutable (sin cambios v3)

### 2.1. Semantica normativa

    catalog_key = identidad administrativa estable.
    K -> assigned_entity_id es 1:1 para toda la vida de K.
    K NO se reasigna jamas.

Cambio de entidad -> nueva key:
    K_old -> A
    K_new -> B

### 2.2. Formato

    catalog_key := "radar_<YYYYMMDD_alta>_<NNNN>"

### 2.3. `catalog_assignments.csv`

    catalog_key | assigned_entity_id | valid_from | valid_to | source | reason

EXACTAMENTE 1 fila por key.

### 2.4. `assigned_entity_id` (B1-NEW-5 v3)

Identificador administrativo interno artificial, sin significado
economico. `"radar_entity_<NNNN>"`.

### 2.5. `catalog_reassignments_attempted.csv`

Registro de intentos bloqueados. NO altera assignments.

### 2.6. Validator + tests A1

Sin cambios respecto v3 (§2.6/2.7). 9 tests A1-a..i.
---

## 3. A1-membership v4 (B1-NEW-6 + B1-NEW-7)

### 3.1. Identificadores separados

    snapshot_row_uid       row-instance identifier (contenido de la fila)
    catalog_key            identidad longitudinal administrativa
    assigned_entity_id     entidad administrativa (1:1 con catalog_key)

**snapshot_row_uid** cambia si cambia el contenido de la fila.
**catalog_key** NO cambia entre snapshots.

Relacion:
    catalog_key  -> 0..N snapshot_row_uid a lo largo del tiempo.
    cada (version_id, snapshot_row_uid) -> exactamente 1 catalog_key.

### 3.2. `snapshot_row_uid` (B1-NEW-7)

    snapshot_row_uid := sha256_hex(fila_canonica)

**SHA-256 completo (64 hex chars).** No truncado.

`fila_canonica` con serializacion length-prefixed:

    Para cada columna en orden alfabetico:
      "<len>:<nombre_columna>|<len>:<valor_normalizado>|"

donde `len` = longitud en bytes UTF-8. Evita ambiguedad si un valor
contiene el separador.

**Propiedades:**
- Independiente de la posicion fisica.
- Independiente del orden de las columnas.
- Detecta duplicados exactos (2 filas con mismo contenido -> mismo uid).
- Cambia si cambia cualquier campo (row-instance, no identity).

### 3.3. `catalog_membership.csv` v4

    version_id            "20260921_01"
    catalog_key           "radar_20260919_<NNNN>"
    snapshot_row_uid      sha256 completo (64 hex chars)
    predecessor_row_uid   sha256 completo | NULL
    justification         texto (motivo del cambio de row_uid)

**`predecessor_row_uid`:**

- NULL si es la primera aparicion de K en cualquier snapshot.
- Si K ya aparecio en snapshots anteriores con otro row_uid, el
  predecessor_row_uid debe apuntar al row_uid del snapshot
  inmediatamente anterior para el mismo K.
- Cadena explicita y verificable.

**Ejemplo:**

    V1 (20260921_01):
      K1 -> row_uid_A, predecessor_row_uid = NULL

    V2 (20261101_01):
      K1 -> row_uid_B, predecessor_row_uid = row_uid_A,
          justification = "name updated in 13F filing"

    V2 no puede declarar K1 -> row_uid_Z sin predecessor. Si lo hace,
    el validator lo rechaza.

### 3.4. Validacion temporal (B1-NEW-6)

**Sustituye la interseccion por cobertura completa:**

Para cada (version_id, catalog_key) en membership:

    assignment.valid_from <= snapshot.valid_from
    AND
    (
      assignment.valid_to IS NULL
      OR (
        snapshot.valid_to IS NOT NULL
        AND snapshot.valid_to <= assignment.valid_to
      )
    )

**Si falla -> `CATALOG_KEY_NOT_VALID_FOR_SNAPSHOT_INTERVAL`.**

**Consecuencia:** snapshot con `valid_to=NULL` no puede contener
assignment con `valid_to` definido (el snapshot duraria mas que
la asignacion).

**Ejemplo prohibido:**

    assignment K1: [2026-09-19, 2026-10-01)
    snapshot V1:   [2026-09-21, NULL)

    v3: interseccion no vacia -> OK
    v4: snapshot.valid_to IS NULL y assignment.valid_to != NULL
        -> CATALOG_KEY_NOT_VALID_FOR_SNAPSHOT_INTERVAL

### 3.5. Validator `validate_membership`

    def validate_membership(membership_df, assignments_df, manifest) -> dict:
        """Errores por (version_id, catalog_key):
          MISSING_VERSION_IN_MANIFEST
          MISSING_CATALOG_KEY_IN_ASSIGNMENTS
          DUPLICATE_CATALOG_KEY_IN_SNAPSHOT
          ROW_UID_NOT_IN_SNAPSHOT
          PREDECESSOR_ROW_UID_BROKEN_CHAIN
          CATALOG_KEY_NOT_VALID_FOR_SNAPSHOT_INTERVAL
        """

Valida:
- version_id existe en manifest B2-PIT.
- catalog_key existe en assignments.
- snapshot_row_uid existe en el snapshot (calculado sobre el CSV).
- predecessor_row_uid (si != NULL) existe en algun snapshot anterior
  con el mismo catalog_key.
- Cobertura temporal completa (B1-NEW-6).

### 3.6. `target_builder.build_target(...)` v4

    def build_target(snapshot_df, membership_df, assignments_df, *,
                     version_id, period_end,
                     catalog_version_id, catalog_sha256) -> TargetUniverse:
        """Materializa el TargetUniverse.

        Validaciones:
          1. Columnas requeridas presentes (B1-NEW-9, ver §8).
          2. Para cada fila del snapshot:
             - row_uid = sha256_hex(fila_canonica)
             - catalog_key = membership[row_uid].catalog_key
          3. membership cubre TODOS los row_uid del snapshot.
          4. catalog_key vigente para el snapshot (cobertura completa).
        """

### 3.7. Tests membership v4

    M-a..M-f   (sin cambios v3)
    M-g        reordenacion fisica no rompe K <-> fila
    M-h        K fuera de vigencia -> NOT_VALID_FOR_SNAPSHOT_INTERVAL
    M-i        K retired en snapshot posterior -> FALLA
    M-j        filas identicas -> mismo uid -> error duplicado
    M-k (NEW)  predecessor_row_uid roto -> PREDECESSOR_ROW_UID_BROKEN_CHAIN
    M-l (NEW)  predecessor NULL solo en primera aparicion; no en persistencia
    M-m (NEW)  same K con row_uid cambiado + predecessor valido -> OK
    M-n (NEW)  same K con row_uid cambiado + sin predecessor -> FALLA
    M-o (NEW)  snapshot valid_to=NULL con assignment valid_to cerrado -> FALLA
    M-p (NEW)  ambos valid_to=NULL -> OK
---

## 4. A2 - Dominio P38 (full resolution por identity_status)

### 4.1. Regla normativa

Para todo K in `TARGET_Q4 ∪ TARGET_Q1`:

    identity_status(K) == RESOLVED
    AND exactamente 1 share_class_figi(K)

Si no -> `UNAVAILABLE` antes de invocar P38.

**Estados que bloquean:**

    UNRESOLVED / CONFLICT / AMBIGUOUS
    (NOT_PRESENT en identity_status)

`figi != None` NO es suficiente. Un estado CONFLICT puede conservar
candidatos FIGI sin constituir identidad resuelta.

### 4.2. Adaptador

Sin cambios respecto v3 §4.2 (full targets, sin filtro).

### 4.3. Tests A2

    A2-a..g    (sin cambios v3)

---

## 5. B3 - Denominador

Regla consolidada: si cualquier K in `TARGET_Q4 ∪ TARGET_Q1` no esta
RESOLVED -> UNAVAILABLE. Si cualquier K in `TARGET_PAIRWISE` no es
feasible -> UNAVAILABLE.

### 5.1. Tests B3

    B3-a..d    (sin cambios v3)

---

## 6. B4 - Fail-closed asimetrico

Sin cambios respecto v3 §6. `feasible_state` +
`identity_status == RESOLVED` + `weight_status in {RESOLVED_OBSERVED,
ZERO_REPORTED}`.

### 6.1. Tests B4

    B4-a..e    (sin cambios v3)

---

## 7. B5 - Colisiones catalog_key -> FIGI

Sin cambios respecto v3 §7. `check_economic_collision(Q4) + (Q1)`.
Cualquier colision -> `CATALOG_ECONOMIC_COLLISION` -> `UNAVAILABLE`.

### 7.1. Tests B5

    B5-a..e    (sin cambios v3)

---

## 8. B1-NEW-9 - Contrato de entrada B1 <-> B2-PIT

### 8.1. Columnas requeridas

    B1_REQUIRED_COLUMNS = frozenset({
        "radar_ticker",
        "share_class_figi",
    })

`build_target` verifica su presencia en el DataFrame del snapshot.

**Si falta alguna -> `B1_SCHEMA_ERROR` -> `UNAVAILABLE`.**
**Sin fallback silencioso.** NO se rellena con None.

### 8.2. Razon

B2-PIT trata el CSV como opaco. B1 necesita columnas concretas para
construir `ticker_by_key` y `figi_by_key`. El contrato debe declararlo
explicitamente para que una futura version de B2-PIT (o de cualquier
producer de snapshots) no rompa B1 silenciosamente.

### 8.3. Columnas opcionales

B1 puede tolerar columnas adicionales (name, source_date, etc.). NO
las usa para identidad.

### 8.4. Tests

    S-a  snapshot con ambas columnas -> OK
    S-b  snapshot sin radar_ticker -> B1_SCHEMA_ERROR
    S-c  snapshot sin share_class_figi -> B1_SCHEMA_ERROR
    S-d  snapshot sin ambas -> B1_SCHEMA_ERROR
    S-e  snapshot con columnas extra -> OK (ignoradas)
---

## 9. Flujo normativo 10 pasos (v4)

    PASO 0. validate_membership (cobertura temporal + linkage)
            STOP si incoherencia.
    PASO 1. target_catalog_as_of(Q4), target_catalog_as_of(Q1)   [B2-PIT]
    PASO 2. build_target(snapshot, membership, assignments) x2
            (verifica B1_REQUIRED_COLUMNS, sha256 row_uid,
             membership, vigencia)
    PASO 3. TARGET_PAIRWISE := Q4.declared_keys INTERSECT Q1.declared_keys
            Si vacio -> UNAVAILABLE (STOP)
    PASO 4. estados para TARGET_Q4 UNION TARGET_Q1              [v4 AMPLIADO]
            state_q4[K] para K in TARGET_Q4
            state_q1[K] para K in TARGET_Q1
    PASO 5. check_continuity (STOP si CONFLICT_FIGI_CHANGE)
            Dominio: TARGET_Q4 UNION TARGET_Q1
    PASO 6. check_economic_collision(Q4) + (Q1) (STOP si colision)
            Dominio: cada universo completo
    PASO 7. feasible(K) para K in TARGET_PAIRWISE               [INTERSECTION]
            STOP si alguno False
    PASO 8. FULL RESOLUTION de TARGET_Q4 UNION TARGET_Q1
            identity_status == RESOLVED
            AND exactamente 1 share_class_figi
            STOP si alguno no cumple.
    PASO 9. catalog_to_p38_targets(...)
            -> target_q4_figi (full Q4, sin filtro)
            -> target_q1_figi (full Q1, sin filtro)
            -> records_q4, records_q1
            -> FEASIBLE
    PASO 10. compute_contractual_coverage(...)                  [P38]
            -> VALID

**Invariante:** P38 solo se invoca si PASOS 0-9 superados sin STOP.

**Separacion clave:**

    ESTADOS          = UNION (TARGET_Q4 ∪ TARGET_Q1)
    FEASIBILITY      = INTERSECTION (TARGET_PAIRWISE)
    TARGETS P38      = UNION (full Q4 y full Q1)
    COLLISION CHECK  = cada universo completo (Q4 y Q1)
    CONTINUITY       = UNION

---

## 10. Plan de implementacion (v4)

### Commit B1.1 - identidad + membership + contrato entrada

    src/institutional_accumulation/identity/catalog_key.py         (nuevo)
      - formato catalog_key
      - sha256_full (64 hex chars)
      - serializacion length-prefixed
      - load_assignments / validate_assignment
      - load_membership / validate_membership (cobertura temporal)
      - B1_REQUIRED_COLUMNS + check_schema
    data/mappings/catalog_assignments.csv
      - 242 filas (1 por key, valid_from=2026-09-19)
    data/mappings/catalog_membership.csv
      - 242 filas para version_id=20260921_01
      - predecessor_row_uid (NULL en migracion inicial)
    tests/test_catalog_key.py                                     (nuevo)
      - A1-a..i
    tests/test_catalog_membership.py                              (nuevo)
      - M-a..p
    tests/test_b1_schema.py                                       (nuevo)
      - S-a..e

### Commit B1.2 - TargetUniverse + estados por periodo

    src/institutional_accumulation/identity/target_builder.py      (nuevo)
      - build_target(snapshot, membership, assignments, ...)
      - vinculacion por snapshot_row_uid
    src/institutional_accumulation/identity/period_state.py        (nuevo)
      - state_q4[K], state_q1[K] para K in Q4 ∪ Q1
      - identity_status enum
      - weight_status enum
    tests/test_target_builder.py                                   (nuevo)
    tests/test_period_state.py                                     (nuevo)

### Commit B1.3 - flujo 10 pasos + adaptador P38

    src/institutional_accumulation/aggregation/catalog_p38_adapter.py  (nuevo)
    src/institutional_accumulation/aggregation/catalog_validator.py    (nuevo)
      - check_continuity (UNION)
      - check_economic_collision (Q4 + Q1)
      - check_full_resolution (identity_status == RESOLVED)
    tests/test_catalog_p38_adapter.py                              (nuevo)
      - A2-a..g, B3-a..d, B5-a..e
    tests/test_p66_pipeline.py                                     (nuevo)
      - 10 pasos + STOPs + no invocacion P38

**NO se toca:** coverage.py, nipc.py, delta_shares.py,
security_identity.py, amendments.py, relationships.py,
temporal_validity.py, reporting_dedup.py, catalog_pit.py.

---

## 11. Criterio de cierre B1 (v4)

### A1

    A1-a .. A1-i    PASS

### Membership

    M-a .. M-p      PASS (incluye M-k/m/n/o/p nuevos)

### Schema B1

    S-a .. S-e      PASS

### A2

    A2-a .. A2-g    PASS (full resolution por identity_status)

### B3

    B3-a .. B3-d    PASS

### B4

    B4-a .. B4-e    PASS

### B5

    B5-a .. B5-e    PASS

### Flujo

    10 pasos + STOPs + PASO 0 membership cobertura completa
    + PASO 4 estados para UNION
    + no invocacion P38 sin PASO 8 PASS

### Global

    P38 tests PASS (existentes intactos)
    P65 tests PASS (31) + P66 tests PASS (31)
    B2-PIT tests PASS (16)
    compileall OK + pyflakes LIMPIO

### Advertencia de alcance (dictamen #55 seccion 16)

    B1 CLOSED != TARGET historico Q4/Q1 disponible en produccion.
    Tests con fixtures historicos explicitamente identificados.
    NO fabricar snapshots historicos presentados como evidencia real.

Cierre: **B1 CLOSED -> nuevo dictamen especifico para A.6.3**.
---

## 12. Preguntas al auditor (v4)

1. **B1-NEW-6 - Cobertura temporal.** Se sustituye interseccion por
   cobertura completa (assignment cubre todo el intervalo del
   snapshot). Snapshot con valid_to=NULL rechaza assignments con
   valid_to definido. ¿Se aprueba?

2. **B1-NEW-7 - predecessor_row_uid.** Cadenas explicitas en
   membership. Primera aparicion -> NULL. Cambio de contenido ->
   predecessor obligatorio + justification. ¿Se aprueba?

3. **B1-NEW-7 - SHA-256 completo.** 64 hex chars (no 16). Y
   serializacion length-prefixed. ¿Se aprueba?

4. **B1-NEW-8 - Estados para UNION.** PASO 4 construye
   state_q4/state_q1 para TARGET_Q4 ∪ TARGET_Q1. Feasibility
   (PASO 7) sigue siendo pairwise. ¿Se aprueba?

5. **B1-NEW-9 - B1_REQUIRED_COLUMNS.** {radar_ticker,
   share_class_figi}. Fail-closed si falta. ¿Se aprueba?

6. **Flujo v4.** 10 pasos con PASO 4 ampliado a UNION. ¿Se aprueba?

7. **Plan de commits.** B1.1 (identidad + membership + schema),
   B1.2 (TargetUniverse + estados UNION), B1.3 (flujo + adaptador).
   ¿Se aprueba?

8. **Criterio de cierre.** §11 + advertencia de alcance. ¿Se aprueba?

9. **Observacion SHA.** Aplicada: SHA-256 completo + length-prefixed.
   ¿Se aprueba la correccion?

10. **Cierre B1 -> A.6.3.** Nuevo dictamen requerido. ¿Se aprueba?

---

## 13. Lo que NO se toca en B1

- `coverage.py` (P38) - firma y semantica INTACTAS.
- `nipc.py`, `delta_shares.py`.
- `security_identity.py`, `amendments.py`, `relationships.py`.
- `temporal_validity.py`.
- `reporting_dedup.py`.
- `catalog_pit.py` (B2-PIT).
- Snapshots publicados por B2-PIT.
- Contratos: `NIPC_CONTRATOS_SEMANTICOS_v1.md`, `NIPC_COVERAGE_POLICY.md` v1.0.
- OpenFIGI masivo: NO.
- `DROP_DUP`: NO.
- Push: NO.
- Certificacion "acumulacion": NO.

---

## 14. Trazabilidad

    Dictamen #51     2 bloqueos A1 + A2
    Dictamen #52     Opcion 1: B2-PIT separado
    Dictamen #53     B2-PIT CERRADO. Siguiente ciclo: B1
    Dictamen #54     3 bloqueantes B1 (full TARGET, assignment, membership)
    Dictamen #55     5 bloqueantes B1 v2 (row_uid, key inmutable,
                     membership temporal, identity_status, entity_id)
    Dictamen #56     4 bloqueantes B1 v3 (cobertura temporal, row_uid
                     vs longitudinal, estados UNION, contrato entrada)
    A62BIS_PROPUESTA.md v9 §2-§3
    A62BIS_B2_PIT_SUBFASE.md
    FASE_A6_PLAN.md

---

Fin de la propuesta B1 v4. Sometida a dictamen. HEAD 43373fd.