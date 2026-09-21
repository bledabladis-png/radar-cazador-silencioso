# IAE - A.6.2-bis-B1 Subfase (v2)

**Version:** v2. Aplicados los 3 bloqueantes estructurales del dictamen
#54: (B1-BLK-1) full-resolution check antes de P38; (B1-BLK-2)
catalog_assignments con identidad + historial; (B1-BLK-3) relacion
formal snapshot <-> catalog_key via membership.

**Version previa:** v1 (commit 003c5c2, NO-GO #54). Backup local
`.v1.bak` hasta verificacion.

**Origen:** dictamen #51 (2 bloqueos) + dictamen #53 (5 bloqueantes) +
dictamen #54 (3 bloqueantes estructurales).

**Fecha:** 2026-09-21.
**HEAD al redactar:** 003c5c2.
**Naturaleza:** propuesta de subfase v2. NO normativa.

---

## 0. Resumen ejecutivo

B1 responde a "¿que TARGET contractual representa el catalogo y como
se traduce al dominio economico P38?".

**3 correcciones materiales del #54:**

1. **B1-BLK-1:** `figi != None` no es filtro. Antes de invocar P38 se
   verifica que TODO `TARGET_Q4 ∪ TARGET_Q1` tiene FIGI resoluble.
   Si no -> `UNAVAILABLE`.
2. **B1-BLK-2:** `catalog_assignments.csv` multi-fila con historial
   (`catalog_key`, `assigned_entity_id`, `valid_from`, `valid_to`,
   `source`, `reason`). Validator detecta reasignacion.
3. **B1-BLK-3:** `catalog_membership.csv` (`version_id`,
   `catalog_key`, `snapshot_row_id`). Publicado aparte, NO modifica
   snapshots inmutables de B2-PIT.

**Flujo ampliado** de 9 a 10 pasos: paso 8-bis de resolucion completa
del dominio.

**Fuera de alcance:** P38 (no se modifica), OpenFIGI masivo,
DROP_DUP, certificacion.

---

## 1. Estado de partida

### 1.1. Lo que B2-PIT aporta (CERRADO #53)

    snapshot materializado (242 filas, SIN catalog_key)
    catalog_manifest.json
    version_id + sha256 externo
    [valid_from, valid_to)
    target_catalog_as_of(period_end)

B1 consume B2-PIT. NO lo modifica. NO reescribe snapshots.

### 1.2. Lo que B1 construye

    catalog_assignments (identidad + historial)
    catalog_membership (snapshot <-> key)
    catalog_validator (assignment, continuity, collision)
    TargetUniverse (declared_keys, ticker_by_key, figi_by_key)
    target_builder (puro, sobre snapshot + membership)
    estados por periodo (state_q4[K], state_q1[K])
    TARGET_PAIRWISE formal
    flujo normativo 10 pasos (fail-closed)
    adaptador P38 (full targets, sin filtro None)

### 1.3. Prohibiciones heredadas

    NO modificar P38 (coverage.py firma y semantica).
    NO activar OpenFIGI masivo.
    NO activar DROP_DUP.
    NO modificar contratos normativos.
    NO reacoplar a B2-PIT.
    NO reescribir snapshots publicados.
---

## 2. A1 - Asignacion de catalog_key + identidad + historial

### 2.1. Dos conceptos separados

    IDENTITY ASSIGNMENT UNIQUENESS
      Una catalog_key se asigna a una ENTIDAD ADMINISTRATIVA concreta.
      La entidad vinculada es inmutable para una asignacion dada.
      Cambio de entidad -> nueva asignacion (nueva fila en historial).

    SNAPSHOT MEMBERSHIP
      La misma catalog_key PUEDE aparecer en N snapshots.

### 2.2. Formato de catalog_key

    catalog_key := "radar_<YYYYMMDD_alta>_<NNNN>"

Ejemplos iniciales: `radar_20260919_0001` .. `radar_20260919_0242`.

- NO es el hash del snapshot.
- NO depende del ticker.
- NO depende del orden alfabetico (que solo se uso para la migracion
  inicial).

### 2.3. `catalog_assignments.csv` (multi-fila con historial)

    catalog_key           "radar_<YYYYMMDD>_<NNNN>"
    assigned_entity_id    identificador de la entidad administrativa
                          vinculada (ej. "radar_entity_<id>")
    valid_from            fecha desde la que aplica esta asignacion
    valid_to              fecha hasta la que aplico (NULL = vigente)
    source                "radar_initial_20260919" | ...
    reason                texto libre (notas de asignacion)

**Regla:** una misma `catalog_key` puede tener multiples filas
(historial). Cada fila declara la entidad vinculada durante
`[valid_from, valid_to)`.

**Inmutable:** filas historicas no se modifican. Cambio de entidad ->
nueva fila con `valid_from` posterior.

**Migracion inicial:** 242 filas, una por key, cada una con
`assigned_entity_id = "radar_entity_<NNNN>"` (identico al sufijo de
key), `valid_from = 2026-09-19`, `valid_to = NULL`.

### 2.4. `catalog_validator.validate_assignment(assignments_df)`

    def validate_assignment(assignments_df) -> dict[str, str]:
        """Devuelve {catalog_key: error_code} para conflictos.

        Errores:
          CATALOG_KEY_REASSIGNED
            dos filas para la misma key, con distinto assigned_entity_id
            y solapamiento temporal (o sin continuidad).
          CATALOG_KEY_RETIRED_REACTIVATED
            key con valid_to caducado aparece reasignada a nueva entidad
            en un snapshot posterior.
        """

**Regla de continuidad:** dos filas consecutivas de la misma key
deben:
- tener `valid_to` de la primera < `valid_from` de la segunda
  (o `valid_to = NULL` en la primera si aun vigente);
- si `assigned_entity_id` cambia entre filas -> es un cambio
  administrativo legitimo (documentado con `reason`);
- si `assigned_entity_id` cambia sin documentar -> CATALOG_KEY_REASSIGNED.

### 2.5. Tests A1

    A1-a  misma key + misma entidad + multiples snapshots -> OK
    A1-b  misma key + entidad distinta + continuidad documentada -> OK
          (nueva fila con reason)
    A1-c  misma key + entidad distinta SIN documentar -> CATALOG_KEY_REASSIGNED
    A1-d  retired sin reactivar -> OK
    A1-e  retired y reactivado en snapshot posterior -> CATALOG_KEY_RETIRED_REACTIVATED
    A1-f  catalog_key NOT NULL + UNICO por snapshot (membership)
    A1-g  catalog_key UNICO global (una unica alta inicial)

---

## 3. A1-membership - Relacion snapshot <-> catalog_key

### 3.1. Problema

B2-PIT publico snapshots inmutables SIN catalog_key (242 filas
opacas). B1 necesita saber que catalog_key corresponde a cada fila.

**No se puede:**
- reescribir el snapshot (es inmutable, cerrado por #53);
- usar la posicion fisica de la fila;
- usar el ticker (catalog_key no depende del ticker).

### 3.2. Solucion: `catalog_membership.csv`

    data/mappings/catalog_membership.csv

    version_id            "20260921_01" (referencia al snapshot)
    catalog_key           "radar_20260919_<NNNN>"
    snapshot_row_id       entero 0..N-1 (indice de fila en el CSV)

**Publicado aparte.** NO modifica el snapshot. El snapshot permanece
inmutable.

**Invariante:** para cada `version_id`, la lista de `snapshot_row_id`
en membership cubre exactamente `0..N-1` sin huecos ni duplicados.

**Migracion inicial:** 242 filas para `version_id = "20260921_01"`,
asignando `snapshot_row_id = 0..241` (indices en el CSV del snapshot
publicado por B2-PIT).

### 3.3. `catalog_validator.validate_membership(membership_df, snapshots_index)`

    def validate_membership(membership_df, snapshots_index) -> dict:
        """Valida que el catalogo de membership sea coherente con
        los snapshots publicados.

        Comprueba:
          - version_id existe en el manifest de B2-PIT.
          - snapshot_row_id cubre 0..N-1 sin huecos ni duplicados.
          - catalog_key esta declarada en catalog_assignments.
        """

### 3.4. `target_builder.build_target(snapshot_df, membership_df, ...)`

    def build_target(snapshot_df, membership_df, *,
                     version_id, period_end,
                     catalog_version_id, catalog_sha256) -> TargetUniverse:
        """Materializa el TargetUniverse a partir del snapshot B2-PIT
        + el catalogo de membership B1.

        Para cada fila del snapshot:
          - row_id = indice (0..N-1)
          - catalog_key = membership_df[row_id].catalog_key
          - ticker = snapshot_df[row_id].radar_ticker
          - figi = snapshot_df[row_id].share_class_figi (puede ser None)

        Validaciones obligatorias (fail-closed):
          - catalog_key NOT NULL por fila.
          - catalog_key UNICO por snapshot.
          - membership cubre 0..N-1 sin huecos.
          - catalog_key declarada en assignments.
        """

**NO depende del ticker ni de la posicion fisica.** Depende del
`catalog_membership.csv` que vincula explicitamente fila <-> key.

### 3.5. Tests membership

    M-a  snapshot V1 + K1 -> build_target produce declared_keys={K1}
    M-b  snapshot V2 + K1 (persistencia) -> declared_keys={K1}
    M-c  snapshot V2 + K1 + K2 (nueva membership) -> declared_keys={K1, K2}
    M-d  membership con hueco -> validate_membership FALLA
    M-e  membership con duplicado -> FALLA
    M-f  membership referencia version_id inexistente -> FALLA
    M-g  membership NO depende del orden de filas (K segun
         snapshot_row_id, no segun orden de ticker)
---

## 4. A2 - Dominio P38 (corregido: full-resolution check)

### 4.1. Regla normativa

**Antes de invocar P38, verificacion de resolucion completa:**

    Para todo K in (TARGET_Q4 UNION TARGET_Q1):
      si figi(K) is None
        -> UNAVAILABLE
        -> NO invocar compute_contractual_coverage

**Solo si TODOS los K resuelven:**

    target_q4_figi := { figi(K) : K in TARGET_Q4 }   (sin filtro)
    target_q1_figi := { figi(K) : K in TARGET_Q1 }   (sin filtro)

**El filtro `figi != None` esta PROHIBIDO.** Si algun K no resuelve,
se bloquea antes de P38.

### 4.2. Diferencia con v1

v1 filtraba silenciosamente (eliminaba `None`). v2 bloquea.

**Caso asimetrico prohibido en v1:**

    TARGET_Q4 = {K1, K2}
    TARGET_Q1 = {K1}
    K1 -> FIGI_A, K2 -> UNRESOLVED

    v1: target_q4_figi = {FIGI_A}  <- K2 eliminada silenciosamente
    v2: UNAVAILABLE (K2 no resuelve)

### 4.3. Adaptador

    def catalog_to_p38_targets(
        universe_q4, universe_q1, *,
        state_q4, state_q1, pairwise_keys,
    ) -> tuple[set[str], set[str], list[PositionRecord],
               list[PositionRecord], CoverageFeasibility]:
        """Traduce TARGET administrativo a TARGET economico P38.

        Precondiciones del flujo normativo:
          - TARGET_PAIRWISE no vacio
          - check_continuity OK
          - check_economic_collision Q4+Q1 OK
          - feasible(K) para K in TARGET_PAIRWISE
          - RESOLUCION COMPLETA de TARGET_Q4 UNION TARGET_Q1

        Salida:
          target_q4_figi: figi(K) para TODO K in TARGET_Q4.
          target_q1_figi: figi(K) para TODO K in TARGET_Q1.
          Sin filtro None. Si algun K no resuelve, no se llega aqui.

        IMPORTANTE:
          Los sets NO se filtran por TARGET_PAIRWISE.
          `figi != None` NO se usa como filtro.
        """

### 4.4. Tests A2

    A2-a  K in TARGET_Q4 con FIGI -> target_q4_figi incluye su figi
    A2-b  K in TARGET_Q4 sin FIGI -> UNAVAILABLE antes de P38
    A2-c  K solo en TARGET_Q4 (no en Q1) -> contribuye a target_q4_figi
    A2-d  K solo en TARGET_Q1 (no en Q4) -> contribuye a target_q1_figi
    A2-e  TARGET_Q4 ^ TARGET_Q1 visible en salida
    A2-f  Sin filtro None: si todos resuelven, tamanos completos

---

## 5. B3 - Imposibilidad de reduccion silenciosa del denominador

**Regla (consolidada con A2):**

    Si cualquier K in (TARGET_Q4 UNION TARGET_Q1) no tiene FIGI
      -> UNAVAILABLE (fail-closed)
    Si cualquier K in TARGET_PAIRWISE no es feasible
      -> UNAVAILABLE (fail-closed)

**NO se recalcula el denominador con menos entradas.** Ni por
`figi=None` ni por `feasible=False`.

**Test B3 obligatorio (nuevo por #54):**

    entrada K solo en TARGET_Q4 (fuera de pairwise) con FIGI no
    resoluble -> UNAVAILABLE (no se ignora silenciosamente)

---

## 6. B4 - Fail-closed ante identidades asimetricas

**Regla:** `feasible_state(s)` verdadero solo si:
    s.identity_status == "RESOLVED"
    AND s.weight_status in {"RESOLVED_OBSERVED", "ZERO_REPORTED"}

**Casos:**

    Q4 RESOLVED + Q1 UNRESOLVED -> UNAVAILABLE
    Q4 RESOLVED + Q1 CONFLICT -> UNAVAILABLE
    NOT_PRESENT -> UNAVAILABLE (no 0.0)
    ZERO_REPORTED -> contribuye 0.0
    FIGI change Q4->Q1 -> CONFLICT_FIGI_CHANGE -> UNAVAILABLE

**Tests B4-a..e** (identicos a v1).

---

## 7. B5 - Preservacion de colisiones catalog_key -> FIGI

**Regla:** si 2+ `catalog_key` mapean al mismo `share_class_figi` en
cualquier periodo, no puede colapsar silenciosamente via `set()`.

**Validator:**

    def check_economic_collision(universe) -> dict[str, list[str]]:
        """Devuelve {share_class_figi: [catalog_key, ...]} para cada
        FIGI con >1 catalog_key.
        """

**Dominio (Q4 + Q1):**

    collision_q4 = check_economic_collision(universe_q4)
    collision_q1 = check_economic_collision(universe_q1)
    Si alguna no vacia -> CATALOG_ECONOMIC_COLLISION -> UNAVAILABLE

**Tests B5-a..e** (identicos a v1).

---

## 8. Flujo normativo 10 pasos (v2)

    PASO 1. target_catalog_as_of(Q4), target_catalog_as_of(Q1)   [B2-PIT]
    PASO 2. build_target(snapshot, membership) x2 -> universe_q4, q1
    PASO 3. TARGET_PAIRWISE := Q4.declared_keys INTERSECT Q1.declared_keys
            Si vacio -> UNAVAILABLE (STOP)
    PASO 4. state_q4[K], state_q1[K] para K in TARGET_PAIRWISE
    PASO 5. check_continuity (STOP si CONFLICT_FIGI_CHANGE)
    PASO 6. check_economic_collision(Q4) + (Q1) (STOP si colision)
    PASO 7. feasible(K) para K in TARGET_PAIRWISE
            Si alguno False -> UNAVAILABLE (STOP)
    PASO 8. RESOLUCION COMPLETA de TARGET_Q4 UNION TARGET_Q1      [v2 NUEVO]
            Para todo K in (TARGET_Q4 UNION TARGET_Q1):
              si figi(K) is None -> UNAVAILABLE (STOP)
    PASO 9. catalog_to_p38_targets(...)
            -> target_q4_figi (full, sin filtro)
            -> target_q1_figi (full, sin filtro)
            -> records_q4, records_q1
            -> FEASIBLE
    PASO 10. compute_contractual_coverage(target_q4_figi, target_q1_figi,
                                           records_q4, records_q1)   [P38]
            -> VALID

**Invariante:** P38 solo se invoca si PASO 1-9 superados sin STOP.

**Diferencia v1->v2:** PASO 8 nuevo (full-resolution). PASO 7 sigue
siendo pairwise feasibility. Ambos checks preceden a P38.
---

## 9. Plan de implementacion (v2, subfase B1)

**Subfase de 3 commits, ampliada con identidad + membership:**

### Commit B1.1 - identidad administrativa + historial + membership

    src/institutional_accumulation/identity/catalog_key.py         (nuevo)
      - formato catalog_key
      - load_assignments
      - validate_assignment (multi-fila, historial)
      - load_membership
      - validate_membership
    data/mappings/catalog_assignments.csv
      - 242 filas iniciales (una por key, valid_from=2026-09-19)
    data/mappings/catalog_membership.csv
      - 242 filas iniciales para version_id=20260921_01
    tests/test_catalog_key.py                                     (nuevo)
      - A1-a .. A1-g
    tests/test_catalog_membership.py                              (nuevo)
      - M-a .. M-g

### Commit B1.2 - TargetUniverse + estados por periodo

    src/institutional_accumulation/identity/target_builder.py      (nuevo)
      - build_target(snapshot, membership, ...) -> TargetUniverse
      - declared_keys, ticker_by_key, figi_by_key, unresolved_keys
    src/institutional_accumulation/identity/period_state.py        (nuevo)
      - state_q4[K], state_q1[K]
      - weight_status enum
    tests/test_target_builder.py                                   (nuevo)
    tests/test_period_state.py                                     (nuevo)

### Commit B1.3 - flujo 10 pasos + adaptador P38

    src/institutional_accumulation/aggregation/catalog_p38_adapter.py  (nuevo)
      - catalog_to_p38_targets (full targets)
      - CoverageFeasibility
    src/institutional_accumulation/aggregation/catalog_validator.py    (nuevo)
      - check_continuity
      - check_economic_collision
      - check_full_resolution (NUEVO v2, PASO 8)
    tests/test_catalog_p38_adapter.py                              (nuevo)
      - A2-a .. A2-f, B3, B5-a .. B5-e
    tests/test_p66_pipeline.py                                     (nuevo)
      - 10 pasos + STOPs + no invocacion P38

**NO se toca:** coverage.py, nipc.py, delta_shares.py,
security_identity.py, amendments.py, relationships.py,
temporal_validity.py, reporting_dedup.py, catalog_pit.py (B2-PIT).

---

## 10. Criterio de cierre B1 (v2)

### A1 (asignacion + historial + membership)

    A1-a .. A1-g    PASS
    M-a .. M-g      PASS

### A2 (dominio P38)

    A2-a .. A2-f    PASS
    Test explicito: K sin FIGI fuera de pairwise -> UNAVAILABLE

### B3 (denominador)

    Test explicito: no reduccion silenciosa
    Test: K solo Q4 con FIGI no resoluble -> UNAVAILABLE

### B4 (fail-closed asimetrico)

    B4-a .. B4-e    PASS

### B5 (colisiones)

    B5-a .. B5-e    PASS

### Flujo

    10 pasos + STOPs + no invocacion P38 sin PASO 8 PASS

### Global

    P38 tests PASS (existentes intactos)
    P65 tests PASS (31) + P66 tests PASS (31)
    B2-PIT tests PASS (16)
    compileall OK + pyflakes LIMPIO

Cierre: **B1 CLOSED -> nuevo dictamen especifico para A.6.3**
(segun #54 seccion 16, pregunta 9: A.6.3 no queda autorizado
automaticamente por cierre de B1).
---

## 11. Preguntas al auditor (v2)

1. **A1 - Registro con historial.** `catalog_assignments.csv` multi-fila
   (`catalog_key`, `assigned_entity_id`, `valid_from`, `valid_to`,
   `source`, `reason`). Validator detecta cambio de entidad sin
   documentar como `CATALOG_KEY_REASSIGNED`. ¿Se aprueba?

2. **A1 - Migracion inicial.** 242 filas, una por key, `assigned_entity_id
   = "radar_entity_<NNNN>"` (identico al sufijo). `valid_from =
   2026-09-19`. ¿Se aprueba?

3. **Membership - Estructura.** `catalog_membership.csv`
   (`version_id`, `catalog_key`, `snapshot_row_id`). Publicado aparte,
   no modifica snapshots. ¿Se aprueba?

4. **Membership - Invariantes.** `snapshot_row_id` cubre 0..N-1 sin
   huecos ni duplicados por `version_id`. ¿Se aprueba?

5. **build_target.** `build_target(snapshot_df, membership_df, ...)`
   usa `snapshot_row_id` para vincular fila <-> key. NO depende de
   ticker ni posicion fisica. ¿Se aprueba?

6. **A2 - Full resolution check.** Antes de P38 se verifica que TODO
   `TARGET_Q4 ∪ TARGET_Q1` tiene FIGI. Si no -> UNAVAILABLE.
   `figi != None` NO es filtro. ¿Se aprueba?

7. **Flujo 10 pasos.** Paso 8 nuevo (full resolution). ¿Se aprueba?

8. **Plan de commits.** B1.1 (identidad + membership), B1.2
   (TargetUniverse + estados), B1.3 (flujo + adaptador). ¿Se aprueba?

9. **Criterio de cierre.** §10 + global. ¿Se aprueba?

10. **Cierre B1 -> A.6.3.** Tras cierre B1, ¿se autoriza A.6.3
    automaticamente o requiere nuevo dictamen? (Dictamen #54 seccion
    16 pregunta 9 indico que no queda autorizado automaticamente).

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
    Dictamen #53     B2-PIT CERRADO. Siguiente ciclo: B1 con 5 bloqueantes
    Dictamen #54     3 bloqueantes estructurales de B1:
                     - B1-BLK-1 (full-resolution, no filtro None)
                     - B1-BLK-2 (assignment con identidad + historial)
                     - B1-BLK-3 (snapshot <-> catalog_key membership)
    A62BIS_PROPUESTA.md v9 §2 y §3     base reutilizada
    A62BIS_B2_PIT_SUBFASE.md           patron de subfase aislada
    FASE_A6_PLAN.md                    seccion A.6.2-bis-B1

---

Fin de la propuesta B1 v2. Sometida a dictamen. HEAD 003c5c2.