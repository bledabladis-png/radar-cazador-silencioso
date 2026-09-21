# IAE - A.6.2-bis Propuesta de rediseno arquitectonico TARGET (v8)

**Version:** v8. Aplicados los 3 bloqueos operacionales del dictamen
#50: (A) flujo PIT obligatorio con check_continuity; (B) dominio
explicito de check_economic_collision (Q4 + Q1); (C) TARGET_PAIRWISE
vacio con semantica explicita (no `all([])`).

**HEAD base:** 2f1f323 (commit de la v7). Este documento (v8) se
commitea por separado; su commit real consta en el historial.

**Versiones previas:** v1 (f383932, NO-GO #44), v2 (9aa0727, GO COND
#45), v3 (fa03a97, NO-GO #46), v4 (30ed3df, NO-GO #47), v5 (6c5e33c,
NO-GO #48), v6 (641ef38, NO-GO #49), v7 (2f1f323, NO-GO #50).

**Objeto:** propuesta de diseno para materializar los 3 bloqueantes
estructurales F2.4 (B1 + B2 + B3).

**Fecha:** 2026-09-21.
**Naturaleza:** propuesta. NO normativa. Sometida a dictamen.

---

## 0. Resumen ejecutivo

v8 cierra los 3 bloqueos operacionales del #50 + incidencia documental:

**Bloqueo A (flujo PIT obligatorio):** el flujo normativo incorpora
`check_continuity(Q4, Q1)` como paso obligatorio entre estados por
periodo y feasibility. `CONFLICT_FIGI_CHANGE -> UNAVAILABLE`. No
depende de precondiciones declarativas.

**Bloqueo B (dominio de colision):** `check_economic_collision` se
ejecuta sobre Q4 y Q1 por separado. Cualquier colision en cualquier
periodo -> `CATALOG_ECONOMIC_COLLISION -> UNAVAILABLE`.

**Bloqueo C (TARGET_PAIRWISE vacio):** `TARGET_PAIRWISE == empty_set`
-> `UNAVAILABLE` explicito. NO se depende de `all([]) == True`.

**Incidencia documental:** la cabecera v8 declara HEAD base 2f1f323
(v7 real), no 641ef38 (v6).

**Fuera de alcance:** OpenFIGI masivo, recalculo de evidencia final,
modificacion de contratos, `DROP_DUP`, certificacion "acumulacion",
Policy v1.3, Gate-NIPC.2/3.

---

## 1. Estado de partida

| Bloqueante | Piezas existentes | Piezas ausentes |
|---|---|---|
| B1 | `coverage.py::compute_contractual_coverage`, `PositionRecord`, `aggregate_positions_by_shareclass_figi`, `target_universe.py::resolve_cusips` | `target_builder.py`, `catalog_key`, adaptador P38, validators, flujo normativo |
| B2 | `source_date` (no contractual) | versionado + `target_catalog_as_of` |
| B3 | Doctrina P63/P64/P65 | `knowledge_date` por posicion, `absence.py` |

**Gate 0 catalogo (2026-09-21):** 242 filas, `radar_ticker` unico,
`share_class_figi` 240 unicos (0 colisiones), 2 MISS.

**Migracion inicial:** las 242 filas reciben
`catalog_key = "radar_20260919_<NNNN>"`. `radar_ticker` versionado.
---

## 2. B1 - Flujo normativo obligatorio (bloqueo A resuelto)

### 2.1. Flujo contractual completo

**El siguiente flujo es normativo.** Cada paso es obligatorio y
secuencial. No puede omitirse ni declararse como "precondicion
externa".

    PASO 1. target_catalog_as_of(period_end) para Q4 y Q1
            -> CatalogNotAvailable | CatalogAmbiguous | (df_q4, df_q1)

    PASO 2. target_builder.build_target(df_q4) -> universe_q4
            target_builder.build_target(df_q1) -> universe_q1

    PASO 3. TARGET_PAIRWISE := universe_q4.declared_keys
                               INTERSECT universe_q1.declared_keys
            Si TARGET_PAIRWISE == empty_set
              -> UNAVAILABLE (bloqueo C)
              -> STOP

    PASO 4. Construir state_q4[K] y state_q1[K] para cada K in
            TARGET_PAIRWISE

    PASO 5. check_continuity(universe_q4, universe_q1, TARGET_PAIRWISE)
            Si algun K tiene CONFLICT_FIGI_CHANGE
              -> UNAVAILABLE
              -> STOP

    PASO 6. check_economic_collision(universe_q4) -> collision_q4
            check_economic_collision(universe_q1) -> collision_q1
            Si collision_q4 no vacio OR collision_q1 no vacio
              -> CATALOG_ECONOMIC_COLLISION
              -> UNAVAILABLE
              -> STOP

    PASO 7. feasibility(K) para cada K in TARGET_PAIRWISE
            Si algun K tiene feasible(K) == False
              -> UNAVAILABLE
              -> STOP

    PASO 8. catalog_to_p38_targets(...)
            -> target_q4_figi, target_q1_figi, records_q4, records_q1
            -> FEASIBLE

    PASO 9. compute_contractual_coverage(target_q4_figi, target_q1_figi,
                                          records_q4, records_q1)
            -> VALID

**Invariante:** `compute_contractual_coverage` SOLO se invoca si los
pasos 1-8 han sido superados sin STOP. Nunca se invoca con estado
inconsistente.

### 2.2. `TARGET_PAIRWISE` formal (sin cambios respecto v7)

    TARGET_Q4 := { catalog_key : activo en snapshot vigente para Q4 }
    TARGET_Q1 := { catalog_key : activo en snapshot vigente para Q1 }
    TARGET_PAIRWISE := TARGET_Q4 INTERSECT TARGET_Q1

Interseccion sobre `catalog_key`. Determinada ANTES del mapping.

**Reglas borde:**

    K en Q4 y en Q1         -> K in TARGET_PAIRWISE
    K solo en Q4            -> fuera de pairwise
    K solo en Q1            -> fuera de pairwise
    TARGET_PAIRWISE vacio   -> UNAVAILABLE (bloqueo C)

### 2.3. `TARGET_PAIRWISE = ∅` - semantica explicita (bloqueo C resuelto)

**Regla contractual:**

    TARGET_PAIRWISE == empty_set -> UNAVAILABLE

**Razon:** la metrica pairwise sin universo contractual no tiene
significado. `paired_weighted_share_coverage` es un cociente sobre el
universo pairwise; si el universo es vacio, la metrica es
indeterminada. NO puede delegarse a `all([]) == True` ni a un
comportamiento accidental del lenguaje.

**Implementacion normativa:**

    if not TARGET_PAIRWISE:
        return CoverageFeasibility.UNAVAILABLE  # sin invocar P38
    if any(not feasible(K) for K in TARGET_PAIRWISE):
        return CoverageFeasibility.UNAVAILABLE
    # ... continuar con FEASIBLE

**Subordinacion a P38:** si el contrato P38 tuviera semantica explicita
para target vacio (`coverage = 0.0 | 1.0 | UNAVAILABLE`), se aplicaria
esa. Hoy el contrato P38 no la tiene. Fail-closed.

### 2.4. Estados por periodo (sin cambios v7)

    state_q4[K] = {
        identity_status:  RESOLVED | UNRESOLVED | CONFLICT | CONFLICT_FIGI_CHANGE
        weight_status:    RESOLVED_OBSERVED | ZERO_REPORTED | NOT_PRESENT
        weight_value:     float | None
    }
    state_q1[K] = { idem }

    feasible_state(s) :=
        s.identity_status == "RESOLVED"
        AND s.weight_status in {"RESOLVED_OBSERVED", "ZERO_REPORTED"}

    feasible(K) := feasible_state(state_q4[K]) AND feasible_state(state_q1[K])

### 2.5. `TargetUniverse` (sin cambios v7)

    @dataclass(frozen=True)
    class TargetUniverse:
        period_end: str
        catalog_version_id: str
        catalog_sha256: str
        declared_keys: frozenset[str]
        ticker_by_key: dict[str, str]
        figi_by_key: dict[str, str | None]
        unresolved_keys: frozenset[str]

### 2.6. Tests B1 (§2)

- Flujo completo con todos los pasos: FEASIBLE + P38 invocado.
- TARGET_PAIRWISE vacio -> PASO 3 STOP -> UNAVAILABLE, P38 NO invocado.
- check_continuity detecta CONFLICT -> PASO 5 STOP -> UNAVAILABLE.
- check_economic_collision Q4 no vacio -> PASO 6 STOP -> UNAVAILABLE.
- check_economic_collision Q1 no vacio -> PASO 6 STOP -> UNAVAILABLE.
- feasible(K)=False -> PASO 7 STOP -> UNAVAILABLE.
- Orden de invocacion verificado (con mocks/traza).
---

## 3. B1 - Validators y adaptador

### 3.1. `check_continuity` (bloqueo A, incorporado al flujo)

**Firma:**

    def check_continuity(
        universe_q4: TargetUniverse,
        universe_q1: TargetUniverse,
        pairwise_keys: frozenset[str],
    ) -> dict[str, str]:
        """Devuelve {catalog_key: conflict_code} para claves con conflicto.

        Reglas por K in pairwise_keys:
          figi_q4 = universe_q4.figi_by_key.get(K)
          figi_q1 = universe_q1.figi_by_key.get(K)

          None + None         -> OK
          None + figi         -> OK (mejora de mapping)
          figi + None         -> OK (regresion de mapping, no continuidad)
          figi_q4 == figi_q1  -> OK
          figi_q4 != figi_q1  -> CONFLICT_FIGI_CHANGE
        """

**Ejecucion obligatoria:** PASO 5 del flujo §2.1. No es opcional ni
delegable.

**Resultado:**

    Si dict vacio -> continuar.
    Si dict no vacio -> UNAVAILABLE (STOP).

### 3.2. `check_economic_collision` (bloqueo B, dominio explicito)

**Firma:**

    def check_economic_collision(
        universe: TargetUniverse,
    ) -> dict[str, list[str]]:
        """Devuelve {share_class_figi: [catalog_key, ...]} para cada
        FIGI con >1 catalog_key dentro del snapshot del universe.
        """

**Dominio de ejecucion (bloqueo B resuelto):**

    collision_q4 = check_economic_collision(universe_q4)
    collision_q1 = check_economic_collision(universe_q1)

    Si collision_q4 no vacio OR collision_q1 no vacio
      -> CATALOG_ECONOMIC_COLLISION -> UNAVAILABLE

**Razon:** una duplicidad administrativa puede aparecer solo en un
snapshot. Comprobar solo Q4 o solo Q1 no es suficiente. Ambos.

**Subordinacion a P38:** si el auditor decide que la colision debe
agregarse economicamente (unidad = share_class_figi agrega N entradas),
se modificara en dictamen. v8: fail-closed.

### 3.3. Adaptador P38 (flujo obligatorio)

**Modulo:** `aggregation/catalog_p38_adapter.py`.

    def catalog_to_p38_targets(
        universe_q4, universe_q1,
        *, state_q4, state_q1, pairwise_keys,
    ) -> tuple[set[str], set[str], list[PositionRecord],
               list[PositionRecord], CoverageFeasibility]:
        """Traduce TARGET administrativo a TARGET economico P38.

        Precondiciones verificadas EN EL FLUJO NORMATIVO (§2.1):
          - TARGET_PAIRWISE no vacio (PASO 3)
          - check_continuity OK (PASO 5)
          - check_economic_collision OK (PASO 6)
          - Todas las K tienen feasible(K) == True (PASO 7)

        El adaptador NO repite esas verificaciones: el flujo las
        garantiza. El adaptador asume invariantes y NO las comprueba
        (fail-fast con assert si se quiere defensa en profundidad).
        """

**Regla:** el adaptador solo se invoca desde el flujo que ha superado
los pasos 1-7. Si un consumidor externo lo invoca directamente sin
respetar el flujo, la responsabilidad es del consumidor. El flujo
normativo (§2.1) es el contrato.

### 3.4. `coverage.py` (intacto)

`compute_contractual_coverage` mantiene su firma original basada en
`share_class_figi`. No recibe `catalog_key`. Solo se invoca desde el
adaptador tras PASO 8.

### 3.5. Tests §3

- `check_continuity`: 5 casos (None+None, None+figi, figi+None,
  iguales, distintos).
- `check_economic_collision` sobre Q4: detecta colision.
- `check_economic_collision` sobre Q1: detecta colision.
- Colision solo en Q4 -> UNAVAILABLE.
- Colision solo en Q1 -> UNAVAILABLE.
- Adaptador invocado solo si flujo pasa: verificable con mocks.
---

## 4. B1 - weight_status (sin cambios v7)

### 4.1. Enum por periodo

    RESOLVED_OBSERVED   identidad OK + value explicito (incl. 0.0)
    ZERO_REPORTED       identidad OK + value == 0.0 documentado
    NOT_PRESENT         identidad OK + sin observacion en el periodo

### 4.2. Regla fail-closed

    identity_status in {UNRESOLVED, CONFLICT, CONFLICT_FIGI_CHANGE}
      -> UNAVAILABLE
    weight_status == NOT_PRESENT
      -> UNAVAILABLE
    weight_status in {RESOLVED_OBSERVED, ZERO_REPORTED}
      -> calculable (value contribuye, incl. 0.0)

**No se reintroduce** `NOT_PRESENT -> 0.0` (cerrado en #49).

### 4.3. Tests weight_status

- RESOLVED_OBSERVED con value=1000 -> contribuye 1000.
- ZERO_REPORTED con value=0 -> contribuye 0.
- NOT_PRESENT -> UNAVAILABLE.
- UNRESOLVED -> UNAVAILABLE.
- CONFLICT -> UNAVAILABLE.

---

## 5. B2 - Point-in-time (aprobado #49, sin cambios)

### 5.1. Estructura

    data/mappings/catalog_snapshots/
        snapshot_<version_id>.csv
        snapshot_<version_id>.sha256
    data/mappings/catalog_manifest.json

### 5.2. Inmutabilidad

`.csv` y `.sha256` inmutables. Manifest mutable.

### 5.3. Intervalos semiabiertos

`[valid_from, valid_to)`. `null` = vigente.

### 5.4. `target_catalog_as_of` (con "que cubren")

    0 snapshots QUE CUBREN period_end -> CatalogNotAvailable
    1 snapshot valido QUE CUBRE        -> (df, vid, sha256)
    >1 snapshots validos QUE CUBREN    -> CatalogAmbiguous

### 5.5. Backdating prohibido

`valid_from = 2026-09-19`. Q4 2025 / Q1 2026 -> CatalogNotAvailable.

### 5.6. Tests B2

- 0 snapshots que cubren -> CatalogNotAvailable.
- 1 snapshot que cubre -> OK.
- >1 snapshots que cubren -> CatalogAmbiguous.
- Snapshot existe pero no cubre -> CatalogNotAvailable.
- Hash invalido -> FAIL-CLOSED.
- Corrupcion -> detectada.

---

## 6. B3 - knowledge_date (aprobado #49, sin cambios)

### 6.1. Tres timestamps

`period_end` / `filing_date` / `knowledge_date`.

### 6.2. RESTATEMENT

Estado sustituido -> fecha del restatement.

### 6.3. NEW HOLDINGS

Heredada -> original. Nueva -> amendment. Ambiguo -> N/D.

### 6.4. `knowledge_date_status`

    ASSIGNED | N/D | LEGACY

### 6.5. `PositionRecord`

Sin cambios respecto v7.

### 6.6. `absence.py` stub

Enums + NotImplementedError. `P63 absence classifier = DEFERRED`.

### 6.7. Regla dura preservada

`delta_shares` NO crea `SOLD`. `P64_EVENTS_DEFERRED`.

### 6.8. Tests B3

Sin cambios respecto v7.
---

## 7. Orden de commits (sin cambios)

    1. B2  versionado + target_catalog_as_of + catalog_validator
    2. B1  target_builder + TargetUniverse + catalog_key
    3. B1  Flujo normativo + TARGET_PAIRWISE + estados por periodo
           + check_continuity + check_economic_collision + adaptador
           (gate 0 consumidores primero)
    4. B3  Timestamps + PositionRecord + provenance + absence.py
    5. Integracion end-to-end + verificacion global

---

## 8. Tests requeridos (resumen)

| Bloque | Fichero | Cobertura |
|---|---|---|
| B1 | `tests/test_target_builder.py` | catalog_key estable, ticker versionado |
| B1 | `tests/test_catalog_validator.py` | unicidad global + continuidad + colision Q4/Q1 |
| B1 | `tests/test_target_pairwise.py` | TARGET_PAIRWISE formal + empty set + estados Q4/Q1 |
| B1 | `tests/test_p66_pipeline.py` (nuevo) | flujo normativo 9 pasos + STOPs + no invocacion de P38 |
| B1 | `tests/test_catalog_p38_adapter.py` | 8 casos A2 + TARGET_PAIRWISE vacio + colision Q4/Q1 |
| B2 | `tests/test_target_catalog_as_of.py` | snapshots "que cubren", fail-closed |
| B3 | `tests/test_position_record.py` | RESTATEMENT, NEW HOLDINGS, ambiguous |
| B3 | `tests/test_absence.py` | enums + NotImplementedError |

**Casos obligatorios adicionales #50 seccion 14:**

    empty target
    Q4/Q1 symmetric difference
    FIGI change (Q4 -> Q1)
    Q4 collision
    Q1 collision
    UNRESOLVED
    CONFLICT
    NOT_PRESENT
    ZERO_REPORTED

---

## 9. Criterio de cierre A.6.2-bis

### B1

**A1 - catalog_key:** inmutable + unicidad intra + unicidad global.

**Flujo normativo:** 9 pasos ejecutados obligatoriamente. STOPs en
PASO 3 (pairwise vacio), PASO 5 (continuity), PASO 6 (collision),
PASO 7 (feasibility). P38 solo invocado desde PASO 9.

**TARGET_PAIRWISE:** definido sobre catalog_key. Empty set ->
UNAVAILABLE explicito. Nunca `all([])`.

**Dominio collision:** Q4 + Q1, ambos. Cualquier colision -> UNAVAILABLE.

**NOT_PRESENT:** fail-closed.

**Adaptador P38:** invocado solo si flujo pasa. Firma P38 intacta.

**9 casos A2 + empty target + FIGI change + colisiones Q4/Q1.**

### B2

- 0/1/>1 snapshots **que cubren** period_end.
- Hash invalido -> FAIL-CLOSED.
- No backdating.
- Corrupcion detectada.

### B3

- knowledge_date == filing_date con ASSIGNED.
- RESTATEMENT / NEW HOLDINGS / ambiguous -> N/D.

### Global

- P38 tests PASS.
- P65 tests PASS (31).
- P66 tests PASS (31).
- A.6.2-bis tests PASS.
- `compileall` OK + `pyflakes` LIMPIO.

Solo entonces: **A.6.2-bis CLOSED -> A.6.3 AUTHORIZED.**

---

## 10. Preguntas al auditor (v8)

1. **Flujo normativo 9 pasos.** Se propone flujo literal §2.1 con
   PASO 5 (continuity), PASO 6 (collision Q4+Q1), PASO 3 (pairwise
   vacio). ¿Se aprueba?

2. **TARGET_PAIRWISE = ∅.** UNAVAILABLE explicito. Subordinado a P38
   si este tiene semantica aprobada. ¿Se aprueba fail-closed?

3. **Dominio collision.** Q4 y Q1 independientemente. ¿Se aprueba?
   ¿O solo TARGET_PAIRWISE?

4. **check_continuity en flujo.** PASO 5 obligatorio. ¿Se aprueba la
   posicion en el flujo (antes de collision, despues de estados)?

5. **Adaptador y flujo.** El adaptador NO repite verificaciones: el
   flujo las garantiza. ¿Se aprueba este contrato de
   responsabilidades?

6. **Tests §8.** 9 casos basicos + 5 adicionales. ¿Se aprueba el
   conjunto?

7. **HEAD fix.** v8 declara HEAD base 2f1f323 (v7). ¿Se aprueba?

8. **Cierre.** Los 3 bloques de §9 + global.

---

## 11. Lo que NO se toca en A.6.2-bis

- Contratos: `NIPC_CONTRATOS_SEMANTICOS_v1.md`, `NIPC_COVERAGE_POLICY.md` v1.0.
- Modelo economico P38 (`share_class_figi` como unidad, Q12 Modelo A).
- Modulos: `delta_shares.py`, `security_identity.py`, `relationships.py`,
  `amendments.py`, `temporal_validity.py`.
- `coverage.py`: firma y semantica INTACTAS.
- `radar_target_catalog.csv`: snapshot inicial + migracion.
- OpenFIGI masivo: NO.
- `DROP_DUP`: NO.
- Push: NO.
- Certificacion "acumulacion": NO.

---

## 12. Trazabilidad

    Dictamen #43      A.6.0 CERRADO + A.6.2-bis AUTORIZADO
    Dictamen #44      v1 NO-GO; 4 correcciones
    Dictamen #45      v2 GO COND
    Dictamen #46      v3 NO-GO; A + B
    Dictamen #47      v4 NO-GO; A1 catalog_key + A2 denominador
    Dictamen #48      v5 NO-GO; weight_status + adaptador + PIT + validator
    Dictamen #49      v6 NO-GO; NOT_PRESENT + TARGET_PAIRWISE + colision
    Dictamen #50      v7 NO-GO; flujo PIT + dominio collision + empty pairwise
    Gate 0 catalogo   v8 seccion 1
    F2.4 #24          3 bloqueantes estructurales
    P38 seccion 3     unidad = share_class_figi
    P60 seccion 1     identity_type obligatorio
    P63 seccion 12    absence semantics
    P64 seccion 13    RESTATEMENT / NEW HOLDINGS
    P65 seccion 14    L3 booleano

---

Fin de la propuesta v8. Sometida a verificacion documental.
HEAD base 2f1f323 (v7).