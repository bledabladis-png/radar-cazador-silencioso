# IAE - A.6.2-bis Propuesta de rediseno arquitectonico TARGET (v7)

**Version:** v7. Aplicados los 3 bloqueos materiales del dictamen #49:
(1) NOT_PRESENT sin semantica autorizada -> fail-closed; (2)
TARGET_PAIRWISE formal sobre catalog_key + estados por periodo; (3)
colision multiples catalog_key -> mismo FIGI -> estado explicito.
+ precision editorial B2 (snapshots que cubren period_end).

**Versiones previas:** v1 (f383932, NO-GO #44), v2 (9aa0727, GO COND
#45), v3 (fa03a97, NO-GO #46), v4 (30ed3df, NO-GO #47), v5 (6c5e33c,
NO-GO #48), v6 (641ef38, NO-GO #49).

**Objeto:** propuesta de diseno para materializar los 3 bloqueantes
estructurales F2.4 (B1 + B2 + B3).

**Fecha:** 2026-09-21.
**HEAD al redactar:** 641ef38.
**Naturaleza:** propuesta. NO normativa. Sometida a dictamen.

---

## 0. Resumen ejecutivo

v7 cierra los 3 bloqueos del #49 + precision editorial:

**Cierre 1 (NOT_PRESENT fail-closed):** `NOT_PRESENT` -> `UNAVAILABLE`.
No se convierte a `0.0`. Solo `ZERO_REPORTED` (observacion documental
de cero) y `RESOLVED_OBSERVED` (con value explicito) contribuyen
numericamente. Alineado con P63 (ausencia != cero).

**Cierre 2 (TARGET_PAIRWISE formal):** definicion matematica explicita:

    TARGET_Q4       = {catalog_key vigente en snapshot Q4}
    TARGET_Q1       = {catalog_key vigente en snapshot Q1}
    TARGET_PAIRWISE = TARGET_Q4 INTERSECT TARGET_Q1

Estados por periodo (no un unico `weight_status` por entrada):

    state_q4[K] = (identity_status, weight_status, weight_value)
    state_q1[K] = (identity_status, weight_status, weight_value)

Feasibility = f(state_q4[K], state_q1[K]) para cada K.

**Cierre 3 (colision catalog_key -> FIGI):** `catalog_validator.check_economic_collision(...)`
detecta 2+ `catalog_key` -> mismo `share_class_figi`. Resultado:
`CATALOG_ECONOMIC_COLLISION` -> `UNAVAILABLE`. Nunca `set()` decide.

**Precision editorial B2:** "0 snapshots QUE CUBREN period_end ->
UNAVAILABLE". Un snapshot existente que no cubre el period_end no
convierte la consulta en valida.

**Fuera de alcance:** OpenFIGI masivo, recalculo de evidencia final,
modificacion de contratos, `DROP_DUP`, certificacion "acumulacion",
Policy v1.3, Gate-NIPC.2/3.

---

## 1. Estado de partida

| Bloqueante | Piezas existentes | Piezas ausentes |
|---|---|---|
| B1 | `coverage.py::compute_contractual_coverage`, `PositionRecord`, `aggregate_positions_by_shareclass_figi`, `target_universe.py::resolve_cusips` | `target_builder.py`, `catalog_key`, adaptador P38, validators |
| B2 | `source_date` (no contractual) | versionado + `target_catalog_as_of` |
| B3 | Doctrina P63/P64/P65 | `knowledge_date` por posicion, `absence.py` |

**Gate 0 catalogo (2026-09-21):** 242 filas, `radar_ticker` unico,
`share_class_figi` 240 unicos (0 colisiones), 2 MISS.

**Migracion inicial:** las 242 filas reciben
`catalog_key = "radar_20260919_<NNNN>"`. `radar_ticker` versionado.
---

## 2. B1 - TARGET_PAIRWISE formal (bloqueo B resuelto)

### 2.1. Definiciones matematicas

**Conjunto contractual declarado por periodo:**

    TARGET_Q4 := { catalog_key : activo en snapshot vigente para Q4 }
    TARGET_Q1 := { catalog_key : activo en snapshot vigente para Q1 }

**Conjunto pairwise contractual:**

    TARGET_PAIRWISE := TARGET_Q4 INTERSECT TARGET_Q1

**Regla:** la interseccion se calcula sobre `catalog_key`. NO sobre
`share_class_figi`. Se determina ANTES de cualquier conversion
administrativa -> economica.

**Reglas explicitas para casos borde:**

    K en TARGET_Q4 y en TARGET_Q1      -> K in TARGET_PAIRWISE
    K en TARGET_Q4, no en TARGET_Q1    -> K fuera de TARGET_PAIRWISE
    K en TARGET_Q1, no en TARGET_Q4    -> K fuera de TARGET_PAIRWISE
    K en ninguno                       -> K irrelevante

**Multiples snapshots validos que cubren un period_end:**

    Si >1 snapshots validos cubren el period_end -> CatalogAmbiguous
    (no se elige). Bloquea la construccion de TARGET_Q*.

**Sin snapshot valido que cubra el period_end:**

    0 snapshots que cubren -> CatalogNotAvailable.
    TARGET_Q* no construible. Metrica -> UNAVAILABLE.

### 2.2. Estados por periodo (no un unico weight_status)

**Correccion material:** el `weight_status` de una entrada pairwise NO
es unico. Depende del periodo.

**Estructura por entrada K en TARGET_PAIRWISE:**

    state_q4[K] = {
        identity_status:  RESOLVED | UNRESOLVED | CONFLICT | CONFLICT_FIGI_CHANGE
        weight_status:    RESOLVED_OBSERVED | ZERO_REPORTED | NOT_PRESENT
        weight_value:     float | None
    }

    state_q1[K] = { idem }

**Feasibility de la metrica pairwise:**

    feasible(K) = feasible_state(state_q4[K]) AND feasible_state(state_q1[K])

    feasible_state(s) :=
        s.identity_status == "RESOLVED"
        AND s.weight_status in {"RESOLVED_OBSERVED", "ZERO_REPORTED"}

    Si cualquier K tiene feasible(K) == False:
        paired_weighted_share_coverage -> UNAVAILABLE
    SINO:
        calcular con w(K) = max(state_q4[K].weight_value,
                                  state_q1[K].weight_value)
        (una vez por security, segun P38 §3.3)

**Caso Q4=RESOLVED + Q1=UNRESOLVED:** feasible(K) == False -> UNAVAILABLE.
Reconstruible sin ambiguedad porque los estados viven por periodo.

### 2.3. `TargetUniverse` v7

    @dataclass(frozen=True)
    class TargetUniverse:
        period_end: str
        catalog_version_id: str
        catalog_sha256: str
        declared_keys: frozenset[str]          # catalog_key del snapshot
        ticker_by_key: dict[str, str]          # key -> radar_ticker actual
        figi_by_key: dict[str, str | None]     # key -> share_class_figi o None
        unresolved_keys: frozenset[str]        # sin FIGI

### 2.4. `target_builder.build_target()` (sin cambios semanticos)

    def build_target(snapshot_df, *, period_end,
                     catalog_version_id, catalog_sha256) -> TargetUniverse:
        """Valida invariantes DENTRO del snapshot.
          - catalog_key NOT NULL
          - catalog_key UNICO en el snapshot
          - radar_ticker NOT NULL
          - radar_ticker UNICO en el snapshot
        NO valida unicidad global (eso es catalog_validator).
        """

Violacion -> `raise SnapshotInvariantViolated`.

### 2.5. Tests B1 §2

- `TARGET_PAIRWISE = TARGET_Q4 ∩ TARGET_Q1` sobre catalog_key.
- K solo en Q4 -> fuera de pairwise.
- K solo en Q1 -> fuera de pairwise.
- 0 snapshots que cubren -> CatalogNotAvailable.
- >1 snapshots que cubren -> CatalogAmbiguous.
- `feasible(K)` con Q4=RESOLVED + Q1=UNRESOLVED -> False -> UNAVAILABLE.
- `feasible(K)` con Q4=RESOLVED_OBSERVED + Q1=ZERO_REPORTED -> True.
---

## 3. B1 - A2: weight_status + adaptador + colision

### 3.1. NOT_PRESENT -> fail-closed (bloqueo A resuelto)

**Regla contractual:**

    NOT_PRESENT -> UNAVAILABLE (no 0.0)

**Justificacion:** P63 distingue hecho observado de causa de ausencia.
`ZERO_REPORTED` es un cero documentado (SSHPRNAMT = 0 en el filing).
`NOT_PRESENT` es ausencia de observacion: no existe cero contractual,
existe ausencia. No se puede convertir ausencia en cero sin autorizacion
P38 explicita, que no existe hoy.

**Consecuencia:** si `state_q4[K].weight_status == "NOT_PRESENT"` o
`state_q1[K].weight_status == "NOT_PRESENT"` -> `feasible(K) == False`
-> metrica `UNAVAILABLE`.

**Nota de alcance:** si el auditor autoriza (via dictamen o via
contrato P38) que `NOT_PRESENT -> 0.0` con una semantica concreta, se
puede revisar. En esta version: fail-closed.

### 3.2. `weight_status` (por periodo)

    Enum:
      RESOLVED_OBSERVED   identidad OK + value explicito (incl. 0.0)
      ZERO_REPORTED       identidad OK + value == 0.0 documentado
      NOT_PRESENT         identidad OK + sin observacion en el periodo

**Regla fail-closed:**

    UNRESOLVED / CONFLICT / CONFLICT_FIGI_CHANGE en identity_status
      -> UNAVAILABLE
    NOT_PRESENT en weight_status
      -> UNAVAILABLE
    RESOLVED_OBSERVED / ZERO_REPORTED
      -> calculable (value contribuye, incl. 0.0)

**Diferencia con v6:** v6 trataba `NOT_PRESENT -> 0.0`. v7 no lo hace.

### 3.3. Adaptador P38

**Modulo:** `aggregation/catalog_p38_adapter.py`.

    def catalog_to_p38_targets(
        universe_q4: TargetUniverse,
        universe_q1: TargetUniverse,
        *,
        state_q4: dict[str, PeriodState],
        state_q1: dict[str, PeriodState],
    ) -> tuple[
        set[str],                # target_q4_figi
        set[str],                # target_q1_figi
        list[PositionRecord],    # records_q4
        list[PositionRecord],    # records_q1
        CoverageFeasibility,
    ]:
        """Traduce TARGET administrativo a TARGET economico P38.

        Precondiciones (verificadas antes de invocar):
          - check_economic_collision: no hay 2+ catalog_key -> mismo FIGI.
          - check_continuity: no hay CONFLICT_FIGI_CHANGE cross-snapshot.
          - Todos los K in TARGET_PAIRWISE tienen feasible(K) == True.
        """

**Reglas:**

- Si TODAS las entradas de `TARGET_PAIRWISE` tienen `feasible(K) == True`:
  -> `CoverageFeasibility.FEASIBLE`. Se invoca
  `compute_contractual_coverage(target_q4_figi, target_q1_figi, ...)`.

- Si ALGUNA tiene `feasible(K) == False`:
  -> `CoverageFeasibility.UNAVAILABLE`. **NO se invoca
  `compute_contractual_coverage`.**

**Firma P38 intacta.**

### 3.4. Colision `catalog_key -> FIGI` (bloqueo C resuelto)

**Regla:** multiples `catalog_key` mapeando al mismo `share_class_figi`
no pueden colapsar silenciosamente via `set()`.

**Validator:** `catalog_validator.check_economic_collision(universe)`.

    def check_economic_collision(universe: TargetUniverse) -> dict[str, list[str]]:
        """Devuelve {share_class_figi: [catalog_key1, catalog_key2, ...]}
        para cada FIGI con >1 catalog_key.
        """

**Si colisiones no vacias:**

    CATALOG_ECONOMIC_COLLISION -> UNAVAILABLE

**Razon:** el contrato P38 fija `share_class_figi` como unidad
economica. Dos `catalog_key` con mismo FIGI implican o bien duplicidad
administrativa, o bien agregacion economica que A.6.2-bis NO puede
decidir sin modificacion contractual. Fail-closed.

**Subordinacion a P38:** si el auditor decide en dictamen que la
colision debe agregarse (ej. la unidad economica agrega ambas
entradas), se modificara la politica. A.6.2-bis no la inventa.

**Implementacion:** el adaptador comprueba colisiones ANTES de
construir los sets.

**Tests C:**

- 2 catalog_key -> mismo FIGI -> `CATALOG_ECONOMIC_COLLISION` -> UNAVAILABLE.
- 2 catalog_key -> FIGIs distintos -> OK.
- 1 catalog_key -> FIGI None + 1 catalog_key -> FIGI X -> OK (no hay colision).
- Colision + metrica -> UNAVAILABLE, no reduccion silenciosa.

### 3.5. `coverage.py` (sin cambios)

`compute_contractual_coverage` mantiene su firma original basada en
`share_class_figi`. NO recibe `catalog_key`. La traduccion ocurre en
el adaptador. Tests P38 existentes pasan sin cambios.

### 3.6. Arquitectura final v7

    period_end
        v
    target_catalog_as_of(period_end)           [B2]
        v
    snapshot + manifest
        v
    target_builder.build_target(snapshot)      [B1]
        v
    TargetUniverse x 2 (Q4, Q1)
        v
    state_q4, state_q1 (por catalog_key)       [B1]
        v
    TARGET_PAIRWISE = Q4_keys INTERSECT Q1_keys
        v
    feasible(K) para cada K
        v
    si TODAS feasible:
      check_economic_collision -> OK
        v
      catalog_to_p38_targets
        v
      compute_contractual_coverage(target_figi, records)  [P38 intacto]
        v
      VALID
    si NO:
      UNAVAILABLE (nunca se llama a compute_contractual_coverage)

### 3.7. Tests A2 (6 casos del #48 + #49)

**1. Identidad OK + peso positivo:**
Q4 RESOLVED_OBSERVED value=1000, Q1 idem -> feasible -> VALID.

**2. Identidad OK + peso 0 explicito:**
Q4 ZERO_REPORTED value=0, Q1 idem -> feasible -> contribuye 0.

**3. Identidad unresolved:**
Q4 UNRESOLVED -> feasible(K)=False -> UNAVAILABLE.

**4. Identity conflict:**
Q4 CONFLICT -> UNAVAILABLE.

**5. Q4 OK + Q1 UNRESOLVED:**
state_q1[K].identity_status=UNRESOLVED -> feasible(K)=False -> UNAVAILABLE.

**6. Q4 RESOLVED + Q1 RESOLVED con FIGI distinto:**
check_continuity detecta CONFLICT_FIGI_CHANGE -> UNAVAILABLE.

**7. (nuevo #49) NOT_PRESENT:**
Q4 NOT_PRESENT -> feasible(K)=False -> UNAVAILABLE.

**8. (nuevo #49) Colision:**
K1->FIGI_X y K2->FIGI_X -> CATALOG_ECONOMIC_COLLISION -> UNAVAILABLE.
---

## 4. B2 - Point-in-time (con precision editorial #49)

### 4.1. Estructura de disco (sin cambios)

    data/mappings/catalog_snapshots/
        snapshot_<version_id>.csv
        snapshot_<version_id>.sha256
    data/mappings/catalog_manifest.json

### 4.2. Inmutabilidad

`.csv` y `.sha256` inmutables in-place. `catalog_manifest.json` mutable.

### 4.3. Intervalos semiabiertos

`[valid_from, valid_to)`. `null` = vigente.

### 4.4. `target_catalog_as_of` (correccion editorial #49)

**Reformulacion obligatoria:**

    0 snapshots QUE CUBREN period_end -> raise CatalogNotAvailable
    1 snapshot valido QUE CUBRE        -> devuelve (df, vid, sha256)
    >1 snapshots validos QUE CUBREN    -> raise CatalogAmbiguous

**Precision:** un snapshot existente que NO cubre el `period_end`
solicitado no convierte la consulta en valida. La consulta pregunta
"que catalogo era valido para period_end"; la respuesta no puede ser
"un catalogo cuyo rango no incluye period_end".

### 4.5. Backdating prohibido

`valid_from = 2026-09-19` para snapshot inicial. Q4 2025 / Q1 2026 ->
`CatalogNotAvailable`.

### 4.6. Test de integridad

sha256 recalculado == publicado (.sha256) + coherencia manifest.

### 4.7. Tests B2

- 0 snapshots que cubren -> CatalogNotAvailable.
- 1 snapshot que cubre -> OK.
- >1 snapshots que cubren -> CatalogAmbiguous.
- Snapshot existente pero que NO cubre -> CatalogNotAvailable.
- as_of("2025-12-31") sin snapshot -> CatalogNotAvailable.
- Integridad: sha256.
- Corrupcion: byte alterado -> fail-closed.

---

## 5. B3 - Semantica temporal (sin cambios respecto v6)

### 5.1. Tres timestamps

    period_end / filing_date / knowledge_date
    Contrato: knowledge_date == filing_date para observaciones ASSIGNED.

### 5.2. RESTATEMENT

Estado sustituido -> fecha del restatement.

### 5.3. NEW HOLDINGS no ambiguo

Heredada -> fecha original. Nueva -> fecha amendment.

### 5.4. NEW HOLDINGS ambiguous

`knowledge_date = None` + `provenance.knowledge_date_status = "N/D"`.

### 5.5. `knowledge_date_status`

    ASSIGNED  fecha asignada (RESTATEMENT o NEW HOLDINGS inequivoca)
    N/D       atribucion no determinable (fail-closed)
    LEGACY    registro pre-B3 sin 3 timestamps

### 5.6. `PositionRecord`

    @dataclass(frozen=True)
    class PositionRecord:
        period: str
        observed_security_key: str
        share_class_figi: Optional[str]
        canonical_security: Optional[str]
        resolution_status: str
        operational_mapping_status: str
        weight: float
        provenance: dict = field(default_factory=dict)
        period_end: Optional[str] = None
        filing_date: Optional[str] = None
        knowledge_date: Optional[str] = None

    def is_contractual_b3(rec) -> bool: ... (sin cambios v6)

### 5.7. `absence.py` stub

Enums + NotImplementedError. `P63 absence classifier = DEFERRED`.

### 5.8. Regla dura preservada

`delta_shares` NO crea `SOLD`. `P64_EVENTS_DEFERRED` se mantiene.

### 5.9. Tests B3

Sin cambios respecto a v6.
---

## 6. Orden de commits (sin cambios)

    1. B2  versionado + target_catalog_as_of + catalog_validator
    2. B1  target_builder + TargetUniverse + catalog_key
    3. B1  TARGET_PAIRWISE + estados por periodo + adaptador P38
           (gate 0 consumidores primero)
    4. B3  Timestamps + PositionRecord + provenance + absence.py
    5. Integracion end-to-end + verificacion global

Razon: B1 depende de B2. B3 depende de B1.

---

## 7. Tests requeridos (resumen)

| Bloque | Fichero | Cobertura |
|---|---|---|
| B1 | `tests/test_target_builder.py` | catalog_key estable, ticker versionado, unicidad snapshot |
| B1 | `tests/test_catalog_validator.py` | unicidad global, continuidad cross-snapshot, colision economica |
| B1 | `tests/test_target_pairwise.py` (nuevo) | TARGET_PAIRWISE formal, estados por periodo, feasible(K) |
| B1 | `tests/test_catalog_p38_adapter.py` | 8 casos A2, feasibility, no invocacion si UNAVAILABLE |
| B2 | `tests/test_target_catalog_as_of.py` | snapshots, "que cubren", fail-closed |
| B3 | `tests/test_position_record.py` | RESTATEMENT, NEW HOLDINGS, ambiguous |
| B3 | `tests/test_absence.py` | enums + NotImplementedError |

**Cero regresion P38/P65/P66.**

---

## 8. Criterio de cierre A.6.2-bis

### B1

**A1 - catalog_key:**
- Inmutable, no derivado del ticker.
- Unicidad intra-snapshot (build_target).
- Unicidad global (catalog_validator).

**TARGET_PAIRWISE formal:**
- Definido sobre catalog_key.
- Estados por periodo (`state_q4[K]`, `state_q1[K]`).
- `feasible(K)` por periodo.
- Determinado ANTES del mapping.

**NOT_PRESENT:**
- Fail-closed (no se convierte a 0.0).
- Test explicito: NOT_PRESENT -> UNAVAILABLE.

**Colision:**
- `check_economic_collision` detecta 2+ catalog_key -> mismo FIGI.
- `CATALOG_ECONOMIC_COLLISION` -> UNAVAILABLE.
- Nunca colapso silencioso via set().

**Adaptador P38:**
- Firma P38 intacta (share_class_figi, PositionRecord).
- `compute_contractual_coverage` no recibe catalog_key.
- No se invoca si `feasible(K) == False` para algun K.

**8 casos A2 cubiertos** (incluye NOT_PRESENT + colision).

### B2

- 0/1/>1 snapshots **que cubren** period_end -> UNAVAILABLE/OK/AMBIGUOUS.
- Hash invalido -> FAIL-CLOSED.
- No backdating.
- Integridad verificable.
- Corrupcion detectada.

### B3

- `knowledge_date == filing_date` con status ASSIGNED.
- RESTATEMENT / NEW HOLDINGS / ambiguous -> N/D explicito.
- `is_contractual_b3` acepta ASSIGNED y N/D.

### Global

- P38 tests PASS (existentes intactos).
- P65 tests PASS (31).
- P66 tests PASS (31).
- A.6.2-bis tests PASS.
- `compileall` OK + `pyflakes` LIMPIO.

Solo entonces: **A.6.2-bis CLOSED -> A.6.3 AUTHORIZED.**
---

## 9. Preguntas al auditor (v7)

1. **NOT_PRESENT -> fail-closed.** Se ha elegido no convertir
   NOT_PRESENT a 0.0. Toda entrada con NOT_PRESENT en algun periodo ->
   UNAVAILABLE. ¿Se aprueba? ¿O el auditor prefiere autorizar via
   contrato P38 una semantica explicita para NOT_PRESENT?

2. **TARGET_PAIRWISE formal.** Definido sobre catalog_key
   (`TARGET_Q4 ∩ TARGET_Q1`). Independiente del mapping. ¿Se aprueba?

3. **Estados por periodo.** `state_q4[K]` + `state_q1[K]` con
   `identity_status` + `weight_status` + `weight_value`. Feasibility =
   f(ambos). ¿Se aprueba?

4. **`feasible(K)`.** True solo si Q4 y Q1 tienen
   `identity_status=RESOLVED` y `weight_status in {RESOLVED_OBSERVED,
   ZERO_REPORTED}`. ¿Se aprueba?

5. **Colision catalog_key -> FIGI.** `check_economic_collision`
   detecta 2+ `catalog_key` -> mismo FIGI. Resultado:
   `CATALOG_ECONOMIC_COLLISION` -> UNAVAILABLE. ¿Se aprueba?

6. **Subordinacion P38.** La politica de colision queda fail-closed.
   Si el auditor prefiere agregacion economica (que la unidad P38 se
   agregue sobre multiples catalog_key), se modificara en dictamen.
   ¿Se aprueba fail-closed por defecto?

7. **Adaptador.** `catalog_to_p38_targets` no invoca
   `compute_contractual_coverage` si `feasible(K) == False` o si hay
   colision. Firma P38 intacta. ¿Se aprueba?

8. **B2 precision.** "0 snapshots QUE CUBREN period_end -> UNAVAILABLE".
   Snapshot existente que no cubre no es valido. ¿Se aprueba?

9. **8 casos A2.** Los 6 casos de #48 + NOT_PRESENT + colision.
   ¿Se aprueba el conjunto?

10. **Cierre.** Los 3 bloques de la seccion 8 + global.

---

## 10. Lo que NO se toca en A.6.2-bis

- Contratos: `NIPC_CONTRATOS_SEMANTICOS_v1.md`, `NIPC_COVERAGE_POLICY.md` v1.0.
- Modelo economico P38 (`share_class_figi` como unidad, Q12 Modelo A).
- Modulos: `delta_shares.py`, `security_identity.py`, `relationships.py`,
  `amendments.py`, `temporal_validity.py`.
- `coverage.py`: firma y semantica INTACTAS.
- `radar_target_catalog.csv` actual: snapshot inicial + migracion.
- OpenFIGI masivo: NO.
- `DROP_DUP`: NO.
- Push a `origin/main`: NO.
- Certificacion "acumulacion": NO.

---

## 11. Trazabilidad

    Dictamen #43      A.6.0 CERRADO + A.6.2-bis AUTORIZADO
    Dictamen #44      v1 NO-GO; 4 correcciones
    Dictamen #45      v2 GO COND; F1/F2/F3 + amendments
    Dictamen #46      v3 NO-GO; A (B1) + B (B3)
    Dictamen #47      v4 NO-GO; A1 catalog_key + A2 denominador
                      + NEW HOLDINGS ambiguous
    Dictamen #48      v5 NO-GO; weight_status + adaptador + PIT + validator
    Dictamen #49      v6 NO-GO; NOT_PRESENT + TARGET_PAIRWISE + colision
    Gate 0 catalogo   v7 seccion 1 (242 filas, unicidad)
    F2.4 #24          3 bloqueantes estructurales
    P38 seccion 3     unidad = share_class_figi
    P60 seccion 1     identity_type obligatorio
    P63 seccion 12    absence semantics (ausencia != cero)
    P64 seccion 13    RESTATEMENT / NEW HOLDINGS
    P65 seccion 14    L3 booleano

---

Fin de la propuesta v7. Sometida a verificacion documental.
HEAD 641ef38.