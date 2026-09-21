# IAE - A.6.2-bis Propuesta de rediseno arquitectonico TARGET (v6)

**Version:** v6. Aplicados los 3 bloqueos materiales del dictamen #48:
(1) separacion weight_value / weight_status; (2) adaptador explicito
P38 sin cambio de interfaz contractual; (3) fail-closed PIT si
catalog_key conserva pero share_class_figi cambia cross-snapshot.
+ validator global de unicidad separado de build_target.

**Versiones previas:** v1 (f383932, NO-GO #44), v2 (9aa0727, GO COND
#45), v3 (fa03a97, NO-GO #46), v4 (30ed3df, NO-GO #47), v5 (6c5e33c,
NO-GO #48).

**Objeto:** propuesta de diseno para materializar los 3 bloqueantes
estructurales F2.4 (B1 + B2 + B3).

**Fecha:** 2026-09-21.
**HEAD al redactar:** 6c5e33c.
**Naturaleza:** propuesta. NO normativa. Sometida a dictamen.

---

## 0. Resumen ejecutivo

v6 aplica los 3 bloqueos materiales + correccion adicional del #48:

**Bloqueo 1 (A2 - weight_status):** se separa `weight_value` /
`weight_status`. Fail-closed sobre `status` (no sobre valor 0).
Estados: `RESOLVED_OBSERVED` | `ZERO_REPORTED` | `NOT_PRESENT` |
`UNRESOLVED` | `CONFLICT`. Solo `UNRESOLVED` y `CONFLICT` producen
`UNAVAILABLE`.

**Bloqueo 2 (A2/P38 - adaptador):** capa adaptadora
`catalog_to_p38_targets(...)` traduce `catalog_key -> share_class_figi
-> PositionRecord`. La API contractual `compute_contractual_coverage`
mantiene su firma original (`share_class_figi`). NO se cambia
silenciosamente el dominio.

**Bloqueo 3 (A1/PIT - cross-snapshot FIGI change):** si el mismo
`catalog_key` conserva pero su `share_class_figi` cambia entre
snapshots sin evidencia de continuidad -> `CONFLICT_FIGI_CHANGE` ->
`UNAVAILABLE`.

**Correccion adicional:** validator global de unicidad de
`catalog_key` (entre snapshots) separado de `build_target()`. La
funcion pura solo valida unicidad dentro del snapshot.

**B3 - N/D explicito:** representacion explicita via
`provenance["knowledge_date_status"] = "N/D"`.

**Fuera de alcance:** OpenFIGI masivo, recalculo de evidencia final,
modificacion de contratos, `DROP_DUP`, certificacion "acumulacion",
Policy v1.3, Gate-NIPC.2/3.

---

## 1. Estado de partida

| Bloqueante | Piezas existentes | Piezas ausentes |
|---|---|---|
| B1 | `coverage.py::compute_contractual_coverage`, `PositionRecord`, `aggregate_positions_by_shareclass_figi`, `target_universe.py::resolve_cusips` | `target_builder.py`, `catalog_key` estable, adaptador P38, validator global |
| B2 | `source_date` (no contractual) | versionado + `target_catalog_as_of` |
| B3 | Doctrina P63/P64/P65 | `knowledge_date` por posicion, `absence.py` |

**Gate 0 catalogo (2026-09-21):** 242 filas, `radar_ticker` unico,
`share_class_figi` 240 unicos, 2 MISS.

**Migracion inicial v6:** las 242 filas reciben
`catalog_key = "radar_20260919_<NNNN>"`. `radar_ticker` pasa a
atributo versionado.
---

## 2. B1 - A1: catalog_key estable + unicidad

### 2.1. Identidad

    catalog_key := "radar_<YYYYMMDD_alta>_<NNNN>"

Inmutable. Asignado una sola vez en el alta. NO depende del ticker.

**Nota de auditoria (#48 seccion 8):** el prefijo NO es garantia
criptografica. La garantia de unicidad viene del registro
administrativo global (impedir reutilizacion). Se documenta asi.

### 2.2. `radar_ticker` como atributo versionado

- Columna del catalogo. Puede cambiar entre snapshots.
- El `catalog_key` no cambia por cambio de ticker.
- Invariante `radar_ticker NOT NULL` + `UNICO` por snapshot.

Test A1: cambio FB -> META conserva `catalog_key`.

### 2.3. Unicidad: separacion de responsabilidades

**`build_target()` (funcion pura, recibe 1 snapshot):**

    valida unicidad DENTRO del snapshot:
      - catalog_key NOT NULL
      - catalog_key UNICO por snapshot
      - radar_ticker NOT NULL por snapshot
      - radar_ticker UNICO por snapshot
    Violacion -> raise SnapshotInvariantViolated

**`catalog_validator.validate_global_uniqueness(catalog_root)`:**
(fuera de build_target)

    valida unicidad GLOBAL entre snapshots:
      - Lee catalog_manifest.json
      - Carga todos los snapshots
      - Verifica que ningun catalog_key se reutiliza
      - Verifica que ningun catalog_key cambia de "alta" (YYYYMMDD)
    Violacion -> raise CatalogKeyReused

**Razon:** `build_target` no puede ver el universo historico completo
(solo ve 1 snapshot). La unicidad global requiere acceso al manifest
y a todos los snapshots.

### 2.4. Validacion de continuidad cross-snapshot

`catalog_validator.check_continuity(catalog_root, period_q4, period_q1)`:

    Para cada catalog_key presente en ambos snapshots:
      figi_q4 = figi_by_key_q4[K]
      figi_q1 = figi_by_key_q1[K]

      si figi_q4 is None y figi_q1 is None -> OK (unresolved en ambos)
      si figi_q4 is None y figi_q1 is not None -> OK (mejora de mapping)
      si figi_q4 is not None y figi_q1 is None -> OK (regresion de mapping, no continuidad)
      si figi_q4 == figi_q1 -> OK (misma unidad economica)
      si figi_q4 != figi_q1 -> CONFLICT_FIGI_CHANGE
                              (sin evidencia de continuidad economica)

**Regla PIT (bloqueo 3 resuelto):** `CONFLICT_FIGI_CHANGE` -> la
entrada afectada se marca `UNRESOLVED` -> metrica `UNAVAILABLE`.

NO se convierte automaticamente cambio de FIGI en continuidad
economica. Se requiere evidencia adicional (fuera de A.6.2-bis:
corporate actions, P64). Sin evidencia: fail-closed.

### 2.5. Tests A1

- Cambio de ticker entre snapshots conserva `catalog_key`.
- `build_target`: unicidad dentro del snapshot.
- `build_target`: `radar_ticker` duplicado -> raise.
- `validate_global_uniqueness`: `catalog_key` duplicado entre snapshots -> raise.
- `check_continuity`: mismo FIGI -> OK.
- `check_continuity`: FIGI distinto -> `CONFLICT_FIGI_CHANGE`.
- `check_continuity`: uno None, otro no -> OK (mejora/regresion de mapping, no continuidad cambiada).
---

## 3. B1 - A2: weight_status + adaptador P38

### 3.1. Separacion weight_value / weight_status (bloqueo 1 resuelto)

**Por entrada de `TARGET_PAIRWISE`, dos campos:**

    weight_value:  float | None
    weight_status: enum

**Enum `weight_status`:**

    RESOLVED_OBSERVED   identidad OK + SSHPRNAMT disponible (incl. 0)
    ZERO_REPORTED       identidad OK + SSHPRNAMT = 0 explicito
                        (en la practica es un subcaso de RESOLVED_OBSERVED)
    NOT_PRESENT         identidad OK + sin observacion en el periodo
    UNRESOLVED          identidad no resuelta (mapping fallido)
    CONFLICT            contradiccion de identidad (candidate_A, etc.)

**Nota de notacion:** `ZERO_REPORTED` y `RESOLVED_OBSERVED` con
`value == 0.0` son equivalentes para el calculo. Se conservan como
distintos por trazabilidad de provenance (contrato P63).

**Regla fail-closed (resuelve bloqueo 1):**

    paired_weighted_share_coverage:
      Si ALGUNA entrada de TARGET_PAIRWISE tiene
        weight_status in {UNRESOLVED, CONFLICT}
          -> UNAVAILABLE
      SINO:
        calcular con w(s) segun:
          RESOLVED_OBSERVED / ZERO_REPORTED -> value (incl. 0)
          NOT_PRESENT                       -> 0.0
            (max(Q4_value, 0.0) = Q4_value si Q4 presente;
             si ambos NOT_PRESENT -> 0.0)

**Diferencia con v5:** v5 usaba `w(s) == 0` como detector de fallo.
v6 usa `weight_status` explicito. **`UNRESOLVED` NO se convierte en
`ZERO_REPORTED`.**

### 3.2. Adaptador explicito P38 (bloqueo 2 resuelto)

**Regla:** `compute_contractual_coverage` mantiene su firma original
basada en `share_class_figi`. El TARGET administrativo (`catalog_key`)
se traduce en una **capa adaptadora explicita**.

    catalog_key -> share_class_figi -> PositionRecord -> coverage

**Modulo nuevo:** `aggregation/catalog_p38_adapter.py`.

    def catalog_to_p38_targets(
        universe_q4: TargetUniverse,
        universe_q1: TargetUniverse,
        *,
        periods=("Q4", "Q1"),
    ) -> tuple[
        set[str],                # target_q4_figi
        set[str],                # target_q1_figi
        list[PositionRecord],    # records_q4
        list[PositionRecord],    # records_q1
        CoverageFeasibility,     # flag de viabilidad
    ]:
        """Traduce TARGET administrativo a TARGET economico P38.

        Salida:
          target_q4_figi / target_q1_figi: sets de share_class_figi.
          records_q4 / records_q1: PositionRecord por entrada resoluble.
          CoverageFeasibility: enum
            FEASIBLE     -> compute_contractual_coverage invocable
            UNAVAILABLE  -> fallo de identidad (UNRESOLVED/CONFLICT)
                            o CONFLICT_FIGI_CHANGE
        """

**Regla:**

- Si TODAS las entradas de `TARGET_PAIRWISE` tienen
  `weight_status in {RESOLVED_OBSERVED, ZERO_REPORTED, NOT_PRESENT}`:
  -> `CoverageFeasibility.FEASIBLE`. Se invoca
  `compute_contractual_coverage(target_q4_figi, target_q1_figi, records_q4, records_q1)`.

- Si ALGUNA tiene `weight_status in {UNRESOLVED, CONFLICT}`:
  -> `CoverageFeasibility.UNAVAILABLE`. **NO se invoca
  `compute_contractual_coverage`.** El caller devuelve `UNAVAILABLE`.

**Semantica preservada:**

- `compute_contractual_coverage` NO recibe `catalog_key`.
- Su firma, su docstring, sus tests P38 quedan intactos.
- La traduccion ocurre aguas arriba, en la capa adaptadora.
- Los tests P38 existentes siguen pasando.

### 3.3. Arquitectura final v6

    period_end
        v
    target_catalog_as_of(period_end, catalog_root)   [B2]
        v
    snapshot + manifest
        v
    target_builder.build_target(snapshot, ...)       [B1]
        v
    TargetUniverse (administrativo: catalog_key)
        v
    catalog_p38_adapter.catalog_to_p38_targets(...)  [B1]
        v
    (target_q4_figi, target_q1_figi, records_q4, records_q1, feasibility)
        v
    SI feasibility == FEASIBLE:
      coverage.compute_contractual_coverage(...)     [P38 intacto]
    SINO:
      return UNAVAILABLE

### 3.4. Tests A2 (6 casos del #48 seccion 9)

**1. Identidad OK + peso positivo:**

    Target: {A} con FIGI_A.
    A: RESOLVED_OBSERVED, value=1000.
    -> feasibility FEASIBLE, cobertura calculada.

**2. Identidad OK + peso 0:**

    A: RESOLVED_OBSERVED con value=0 (o ZERO_REPORTED).
    -> feasibility FEASIBLE, contribuye con 0.

**3. Identidad unresolved + peso no disponible:**

    A: UNRESOLVED, value=None.
    -> feasibility UNAVAILABLE, no se llama compute_contractual_coverage.

**4. Identity conflict:**

    A: CONFLICT.
    -> feasibility UNAVAILABLE.

**5. Identity OK en Q4 + unresolved en Q1:**

    A(Q4): RESOLVED_OBSERVED.
    A(Q1): UNRESOLVED.
    -> feasibility UNAVAILABLE (fallo de identidad en algun periodo).

**6. Identity OK en ambos + distinta share_class_figi:**

    A(Q4): RESOLVED_OBSERVED, FIGI_X.
    A(Q1): RESOLVED_OBSERVED, FIGI_Y.
    -> CONFLICT_FIGI_CHANGE en check_continuity.
    -> feasibility UNAVAILABLE.

**Precision sobre caso 6:** es el bloqueo 3 del #48. Sin evidencia
adicional de continuidad economica (corporate actions, P64), el
fail-closed es la respuesta correcta.

### 3.5. Tests B1 adicionales

- `compute_contractual_coverage` NO recibe `catalog_key` (verificado por firma).
- Tests P38 existentes pasan sin cambios.
- `catalog_to_p38_targets` con `FEASIBLE` produce sets de FIGI.
- `catalog_to_p38_targets` con `UNAVAILABLE` NO invoca la funcion contractual.
---

## 4. B2 - Point-in-time (aprobado #48, sin cambios)

### 4.1. Estructura de disco

    data/mappings/catalog_snapshots/
        snapshot_<version_id>.csv
        snapshot_<version_id>.sha256
    data/mappings/catalog_manifest.json

`version_id = <YYYYMMDD>_<NN>`. No autorreferencial.

### 4.2. Inmutabilidad

- `.csv` y `.sha256`: inmutables in-place.
- `catalog_manifest.json`: mutable (indice).

### 4.3. Intervalos semiabiertos

`[valid_from, valid_to)`. `null` = vigente.

    0 snapshots -> CatalogNotAvailable
    1 snapshot  -> OK
    >1 solapan  -> CatalogAmbiguous

### 4.4. Backdating prohibido

`valid_from = 2026-09-19` para snapshot inicial. Q4 2025 / Q1 2026 ->
`CatalogNotAvailable`.

### 4.5. Test de integridad

sha256 recalculado == publicado (.sha256) + coherencia manifest.
Mismatch -> FAIL-CLOSED. Corrupcion simulada detectada.

### 4.6. Tests B2

- 0/1/>1 snapshots -> fail-closed/OK/ambiguous.
- `as_of("2025-12-31")` -> `CatalogNotAvailable`.
- Integridad: sha256.
- Corrupcion: byte alterado -> detectada.
- Intervalos sin solapamiento.

---

## 5. B3 - Semantica temporal + N/D explicito

### 5.1. Tres timestamps

    period_end       cierre del trimestre
    filing_date      fecha del filing del filing_manager
    knowledge_date   fecha de publicacion de la observacion efectiva

Contrato: `knowledge_date == filing_date`.

### 5.2. RESTATEMENT

Estado sustituido -> fecha del restatement. Coherente con P64.

### 5.3. NEW HOLDINGS no ambiguo

    original  X (2026-01-30)
    amendment Y (2026-02-15)
      X -> knowledge_date 2026-01-30
      Y -> knowledge_date 2026-02-15

### 5.4. NEW HOLDINGS ambiguous

    original  X (2026-01-30)
    amendment X (2026-02-15) - sin evidencia adicional

    knowledge_date = None + knowledge_date_status = "N/D" (ver 5.5)

No heuristica.

### 5.5. N/D explicito (precision #48 seccion 7)

**Representacion obligatoria:** `knowledge_date = None` +
`provenance["knowledge_date_status"] = "N/D"`.

**Regla:** `N/D` NO se representa con:
- `0` numerico
- cadena vacia
- `NaT` sin provenance
- fecha ausente por error tecnico

Un registro con `knowledge_date_status = "N/D"` es contractual
(se conoce la causa). Un registro sin `knowledge_date_status` es
legacy o tecnico.

Enum:

    knowledge_date_status:
      ASSIGNED  fecha asignada (RESTATEMENT o NEW HOLDINGS inequivoca)
      N/D       atribucion no determinable (fail-closed)
      LEGACY    registro pre-B3 sin 3 timestamps

### 5.6. `PositionRecord` extendido

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

**Invariantes:**

- `period` <-> `period_end` coherentes.
- `knowledge_date == filing_date` si ambos presentes.
- `provenance.effective_filing_accession` presente en contractual B3.
- `provenance.knowledge_date_status` presente en contractual B3.

    def is_contractual_b3(rec) -> bool:
        return (
            rec.period is not None
            and rec.period_end is not None
            and "knowledge_date_status" in rec.provenance
            and rec.provenance["knowledge_date_status"] in ("ASSIGNED", "N/D")
            and (
                rec.provenance["knowledge_date_status"] == "N/D"
                or (
                    rec.filing_date is not None
                    and rec.knowledge_date == rec.filing_date
                    and "effective_filing_accession" in rec.provenance
                    and not rec.provenance.get("ambiguity_flag", False)
                )
            )
        )

### 5.7. `absence.py` stub

Enums + `NotImplementedError`. `P63 absence classifier = DEFERRED`.

### 5.8. Regla dura preservada

`delta_shares` NO crea `SOLD`. `P64_EVENTS_DEFERRED` se mantiene.

### 5.9. Tests B3

- 3 timestamps -> OK.
- Sin timestamps -> legacy.
- `period <-> period_end`.
- `knowledge_date == filing_date` cuando `ASSIGNED`.
- RESTATEMENT -> ASSIGNED con fecha restatement.
- NEW HOLDINGS no ambiguo -> ASSIGNED por posicion.
- NEW HOLDINGS ambiguous -> N/D explicito.
- `is_contractual_b3` acepta ASSIGNED y N/D, rechaza LEGACY o ausencia de status.
- `absence.py::classify_absence` -> NotImplementedError.
- `delta_shares` no produce SOLD (test P63).
---

## 6. Orden de commits (sin cambios)

    1. B2  Modelo de versionado + target_catalog_as_of + catalog_validator
    2. B1  target_builder + TargetUniverse + catalog_key
    3. B1  catalog_p38_adapter + integracion coverage (gate 0 consumidores)
    4. B3  Timestamps + PositionRecord + provenance + absence.py
    5. Integracion end-to-end + verificacion global

Razon: B1 depende de B2. B3 depende de B1.

---

## 7. Tests requeridos (resumen)

| Bloque | Fichero | Cobertura |
|---|---|---|
| B1 | `tests/test_target_builder.py` (nuevo) | catalog_key estable, ticker versionado, declared_keys, unicidad snapshot |
| B1 | `tests/test_catalog_validator.py` (nuevo) | unicidad global, continuidad cross-snapshot (FIGI change -> CONFLICT) |
| B1 | `tests/test_catalog_p38_adapter.py` (nuevo) | weight_status, 6 casos A2, feasibility |
| B1 | `tests/test_sec_13f_nipc.py` (extender) | wrapper con/sin TARGET |
| B2 | `tests/test_target_catalog_as_of.py` (nuevo) | snapshots, fail-closed, integridad |
| B3 | `tests/test_position_record.py` (nuevo) | RESTATEMENT, NEW HOLDINGS, ambiguous -> N/D explicito |
| B3 | `tests/test_absence.py` (nuevo) | enums + NotImplementedError |

**Cero regresion P38/P65/P66 esperada.** Tests P38 existentes intactos.

---

## 8. Criterio de cierre A.6.2-bis

### B1

**A1 - catalog_key + unicidad:**
- Inmutable, no derivado del ticker (test FB -> META).
- Unicidad dentro del snapshot (build_target).
- Unicidad global entre snapshots (catalog_validator).
- Continuidad cross-snapshot: FIGI cambiado -> `CONFLICT_FIGI_CHANGE` -> UNAVAILABLE.

**A2 - weight_status + adaptador:**
- Separacion weight_value / weight_status.
- UNRESOLVED y CONFLICT -> UNAVAILABLE.
- RESOLVED_OBSERVED / ZERO_REPORTED / NOT_PRESENT -> calculables.
- Adaptador P38 explicito: `catalog_key -> share_class_figi -> PositionRecord`.
- `compute_contractual_coverage` firma intacta.
- 6 casos A2 cubiertos.

**General:**
- Unidad economica P38 (`share_class_figi`) intacta.
- Tests P38 existentes sin cambios.

### B2

- 0/1/>1 snapshots -> fail-closed/OK/ambiguous.
- Hash invalido -> fail-closed.
- No backdating.
- Integridad verificable.
- Corrupcion detectada.

### B3

- `period_end != filing_date` cuando difieran.
- `knowledge_date == filing_date` con status ASSIGNED.
- RESTATEMENT -> fecha del restatement.
- NEW HOLDINGS no ambiguo -> fechas por posicion.
- NEW HOLDINGS ambiguo -> `knowledge_date = None` + `status = N/D`.
- `N/D` no confundible con 0 / vacio / error tecnico.
- `is_contractual_b3` acepta ASSIGNED y N/D, rechaza LEGACY sin status.

### Global

- P38 tests PASS (existentes).
- P65 tests PASS (31).
- P66 tests PASS (31).
- A.6.2-bis tests PASS (nuevos).
- `compileall` OK + `pyflakes` LIMPIO.
- Sin regresion nueva.

Solo entonces: **A.6.2-bis CLOSED -> A.6.3 AUTHORIZED.**
---

## 9. Preguntas al auditor (v6)

1. **A1 - catalog_key.** `radar_<YYYYMMDD_alta>_<NNNN>` + registro
   administrativo global. ¿Se aprueba?

2. **A1 - Separacion build_target / catalog_validator.**
   `build_target`: unicidad por snapshot. `catalog_validator`:
   unicidad global entre snapshots. ¿Se aprueba?

3. **A1 - Continuidad cross-snapshot.** Mismo catalog_key + distinto
   FIGI -> `CONFLICT_FIGI_CHANGE` -> UNAVAILABLE. ¿Se aprueba?

4. **A2 - weight_status.** Enum RESOLVED_OBSERVED | ZERO_REPORTED |
   NOT_PRESENT | UNRESOLVED | CONFLICT. Fail-closed sobre status.
   ¿Se aprueba?

5. **A2 - Caso identity OK + weight 0.** Se trata como
   RESOLVED_OBSERVED (contribuye con 0). ¿Se aprueba? ¿O se prefiere
   ZERO_REPORTED como estado separado con tratamiento distinto?

6. **A2 - Caso NOT_PRESENT.** Identidad OK + sin observacion en el
   periodo. w(s) = 0.0 en el periodo sin observacion. ¿Se aprueba?

7. **A2 - Adaptador.** Capa `catalog_p38_adapter.catalog_to_p38_targets`.
   `compute_contractual_coverage` firma intacta. ¿Se aprueba?

8. **B3 - N/D explicito.** `knowledge_date = None` +
   `provenance.knowledge_date_status = "N/D"`. ¿Se aprueba?

9. **B3 - `is_contractual_b3`.** Acepta ASSIGNED y N/D; rechaza
   LEGACY sin status. ¿Se aprueba?

10. **Cierre.** Los 3 bloques de la seccion 8 + global.

---

## 10. Lo que NO se toca en A.6.2-bis

- Contratos: `NIPC_CONTRATOS_SEMANTICOS_v1.md`, `NIPC_COVERAGE_POLICY.md` v1.0.
- Modelo economico P38 (`share_class_figi` como unidad, Q12 Modelo A).
- Modulos: `delta_shares.py`, `security_identity.py`, `relationships.py`,
  `amendments.py`, `temporal_validity.py`.
- `coverage.py`: firma y semantica INTACTAS. Se invoca desde el
  adaptador.
- `radar_target_catalog.csv` actual: snapshot inicial + migracion
  administrativa.
- OpenFIGI masivo: NO.
- `DROP_DUP`: NO.
- Push a `origin/main`: NO.
- Certificacion "acumulacion": NO.

---

## 11. Trazabilidad

    Dictamen #43                 A.6.0 CERRADO + A.6.2-bis AUTORIZADO
    Dictamen #44                 v1 NO-GO; 4 correcciones
    Dictamen #45                 v2 GO COND; F1/F2/F3 + amendments
    Dictamen #46                 v3 NO-GO; A (B1) + B (B3)
    Dictamen #47                 v4 NO-GO; A1 catalog_key + A2 denominador
                                 + NEW HOLDINGS ambiguous
    Dictamen #48                 v5 NO-GO; weight_status + adaptador P38
                                 + PIT cross-snapshot + validator global
    Gate 0 catalogo              v6 seccion 1 (242 filas, unicidad)
    F2.4 (dictamen #24)          3 bloqueantes estructurales
    P38 (contrato seccion 3)     unidad = share_class_figi
    P60 (contrato seccion 1)     identity_type obligatorio
    P63 (contrato seccion 12)    absence semantics
    P64 (contrato seccion 13)    RESTATEMENT / NEW HOLDINGS
    P65 (contrato seccion 14)    L3 booleano

---

Fin de la propuesta v6. Sometida a verificacion documental.
HEAD 6c5e33c.