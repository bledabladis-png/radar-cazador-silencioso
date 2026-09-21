# IAE - A.6.0 Expediente para auditoria (Gate 0 de los 3 bloqueantes F2.4)

**Objeto:** entrega al auditor externo del inventario A.6.0 (Gate 0 de los
3 bloqueantes estructurales introducidos por el dictamen F2.4).

**Origen:** FASE_A6_PLAN.md seccion A.6.0, marcada PENDIENTE hasta
2026-09-21. Dictamen F2.4 (DICTAMENES.md #24).

**Fecha:** 2026-09-21.
**HEAD al redactar:** 22a92ae.
**Naturaleza:** documento de entrega. NO normativo. En caso de conflicto,
gana FASE_A6_PLAN.md, RECONCILIACION_CONTRATO_CODIGO.md o el dictamen
F2.4.

---

## 0. Resumen ejecutivo

A.6.0 (inventario documental, sin cambios de codigo) esta completado.
Evidencia en `iae/INFORME.md` seccion 21 (commit `22a92ae`).

Resultado del inventario:

| Bloqueante | Estado global |
|---|---|
| B1 - TARGET independiente del mapping | PARCIAL |
| B2 - Point-in-time | AUSENTE |
| B3 - 13F != flujo en tiempo real | DOCUMENTAL (doctrina si, materializacion no) |

Se solicita al auditor:

1. Validacion del inventario A.6.0 (criterio de aceptacion cumplido).
2. Autorizacion (o desestimacion) para A.6.2-bis (rediseno TARGET).
3. Dictamen sobre las 6 preguntas de la seccion 7.

---

## 1. Alcance y metodo

**Alcance A.6.0:** inventario empirico con referencias archivo:linea.
Sin cambios de codigo productivo. Sin OpenFIGI. Sin tocar contratos.

**Metodo:** lectura directa de modulos y CSVs en disco. Grep sobre
`src/institutional_accumulation/**/*.py`.

**Criterio de aceptacion (FASE_A6_PLAN.md):** inventario completo con
referencias archivo:linea. Sin tocar codigo. CUMPLIDO.

**Fuente primaria:** `iae/INFORME.md` seccion 21.
---

## 2. Resultado por bloqueante

### 2.1. B1 - TARGET independiente del exito del mapping

**Enunciado F2.4:** el denominador de `paired_weighted_share_coverage`
no puede depender del exito del mapping. Separar CATALOGO (externo,
versionado) de TARGET_OBSERVED.

**Piezas existentes (archivo:linea):**

    coverage.py::compute_contractual_coverage          L64
    coverage.py::PositionRecord                         L25
    coverage.py::aggregate_positions_by_shareclass_figi L41
    target_universe.py::resolve_cusips                  L81
    radar_target_catalog.csv                            242 filas

**Piezas ausentes:**

    identity/target_builder.py                         NO EXISTE
    Integracion en nipc.py::compute_coverage_pairwise  NO EXISTE
    Refactor de nombres (target_pairwise es nominal)   NO EXISTE

**Diagnostico:** coexisten 2 APIs.
- API contractual P38 (`coverage.py`): completa, recibe TARGET externo.
- API operativa (`nipc.py::compute_coverage_pairwise`): usa
  `observed_security_key` (CUSIPs) como proxy; NO invoca la contractual.

**Consecuencia:** el nombre `target_pairwise` en `nipc.py` (L213) es
nominal. La implementacion real es `observed_pairwise`.

### 2.2. B2 - Point-in-time

**Enunciado F2.4:** toda entidad aplicada a un periodo historico debe
declarar validez temporal. Aplicar la version de hoy retroactivamente
produce survivorship y look-ahead bias.

**Piezas existentes:**

    radar_target_catalog.csv::source_date   (snapshot, no vigencia)
    cusip_equivalence.csv::valid_from/to    (0 filas)
    cusip_ticker_exceptions.csv::valid_from/to (3 filas)

**Piezas ausentes (grep 0 hits en src/):**

    catalog_version
    catalog_valid_from
    catalog_valid_to
    target_catalog_as_of()
    Snapshot + hash por consulta OpenFIGI

**Diagnostico:** B2 esta ausente excepto `source_date` (no cumple la
semantica point-in-time).

### 2.3. B3 - 13F != flujo en tiempo real

**Enunciado F2.4:** "posiciones al cierre de trimestre + publicacion
hasta 45 dias. Sin cortos, con minimis, con confidencialidad."

**Piezas existentes:**

    delta_shares.py::DELTA_SEMANTICS_GROSS_OBSERVED     L56
    delta_shares.py::P64_EVENTS_DEFERRED                L57
    Doctrina P63/P64/P65 en docstrings
    SUBMISSION.FILING_DATE + PERIODOFREPORT (en bruto)

**Piezas ausentes (grep 0 hits en src/):**

    knowledge_date
    period_end/filing_date separados en PositionRecord
    absence_reason (clasificador)
    ZERO_REPORTED / NOT_PRESENT (enums)
    Distincion corporate action vs economic en delta_shares

**Diagnostico:** doctrina documentada; materializacion ausente.

---

## 3. Matriz de cobertura

| Bloqueante | Existentes | Ausentes | Estado |
|---|---|---|---|
| B1 | coverage.py (API P38), target_universe.py, catalogo 242 filas | target_builder.py, integracion nipc.py, refactor nombres | PARCIAL |
| B2 | source_date (no contractual), valid_from/to en 2 tablas | catalog_version, catalog_valid_from/to, target_catalog_as_of, snapshot+hash | AUSENTE |
| B3 | Doctrina P63/64/65, FILING_DATE+PERIODOFREPORT brutos | knowledge_date, period_end+filing_date, absence_reason, ZERO_REPORTED/NOT_PRESENT | DOCUMENTAL |
---

## 4. Lo que NO se ha tocado en A.6.0

- Codigo productivo: `nipc.py`, `coverage.py`, `delta_shares.py`,
  `security_identity.py`, `relationships.py`.
- Contratos: `NIPC_CONTRATOS_SEMANTICOS_v1.md`, `NIPC_COVERAGE_POLICY.md` v1.0.
- Catalogos en disco: `radar_target_catalog.csv`,
  `cusip_equivalence.csv`, `cusip_ticker_exceptions.csv`.
- OpenFIGI: NO ejecutado.
- `DROP_DUP`: NO activado.
- Push a `origin/main`: NO.

---

## 5. Bloqueos actuales

| Bloqueo | Estado |
|---|---|
| THRESHOLD_1 | UNDEFINED |
| THRESHOLD_2 | BLOQUEADO |
| Gate-NIPC.2 | BLOQUEADO |
| Gate-NIPC.3 | NO AUTORIZADO |
| OpenFIGI masivo | NO AUTORIZADO |
| Policy v1.3 aplicacion | NO AUTORIZADA |
| Push a `origin/main` | NO |
| Activacion `DROP_DUP` | NO AUTORIZADA |
| Certificacion "acumulacion" | BLOQUEADA |
| A.6.2-bis (rediseno TARGET) | NO AUTORIZADO |

---

## 6. Estado de las sub-fases A.6

| Sub-fase | Descripcion | Estado |
|---|---|---|
| A.6.0 | Gate 0 de los 3 bloqueantes | CERRADO (este expediente) |
| A.6.1 | Dictamen F2.4 | CERRADO 2026-09-20 |
| A.6.2 | Fixes quirurgicos | CERRADO 2026-09-20 |
| A.6.2-bis | Rediseno arquitectonico TARGET | PENDIENTE - requiere autorizacion |
| A.6.3 | Test P38 Q12 | PENDIENTE |
| A.6.4 | Recalculo evidencia | PENDIENTE (condicional F2.4) |
| A.6.5 | Edicion in-place contrato | EJECUTADO 2026-09-20 |
| A.6.6 | F2.4-CLOSE | PENDIENTE |
| A.6.7 | Contratos P62-P65 | PARCIAL (contratos formalizados, materializacion pendiente) |
---

## 7. Preguntas al auditor

1. **Validacion de A.6.0.** ¿El inventario cumple el criterio de
   aceptacion de FASE_A6_PLAN.md (referencias archivo:linea, sin
   tocar codigo)?

2. **B1 - Coexistencia de APIs.** ¿Se autoriza unificar la API
   operativa (`nipc.py::compute_coverage_pairwise`) con la contractual
   (`coverage.py::compute_contractual_coverage`), o se mantienen
   separadas con proposito distinto?

3. **B1 - `target_builder.py`.** ¿Se autoriza su creacion (materializa
   TARGET_P desde catalogo + observacion 13F)? Es requisito para
   materializar P38 contractualmente.

4. **B2 - Versionado del catalogo.** ¿Se autoriza anadir
   `catalog_version` + `catalog_valid_from` + `catalog_valid_to` al
   catalogo y crear `target_catalog_as_of(period_end)`? ¿O se
   considera suficiente el `source_date` actual?

5. **B3 - `knowledge_date`.** ¿Se autoriza anadir `knowledge_date` al
   pipeline de ingest como tercer timestamp (junto a `period_end` y
   `filing_date`)?

6. **A.6.2-bis.** ¿Se autoriza abrir el rediseno arquitectonico de
   TARGET ahora, o se mantiene bloqueado hasta nuevo dictamen
   especifico?

---

## 8. Trazabilidad

### 8.1. Commit de A.6.0

    22a92ae  docs(iae): A.6.0 - Gate 0 de los 3 bloqueantes F2.4

### 8.2. Evidencia primaria

    iae/INFORME.md seccion 21    inventario completo
    iae/FASE_A6_PLAN.md          plan de la fase A.6
    iae/RECONCILIACION_CONTRATO_CODIGO.md  D1/D2/D3
    iae/REESTRUCTURACION_MODULO.md         arquitectura objetivo

### 8.3. Dictamenes de referencia

    iae/DICTAMENES.md #24    F2.4 (GO CONDICIONADO + 3 bloqueantes)

### 8.4. Contratos de referencia

    NIPC_CONTRATOS_SEMANTICOS_v1.md seccion 3    P38
    NIPC_CONTRATOS_SEMANTICOS_v1.md seccion 11   P62 point-in-time
    NIPC_CONTRATOS_SEMANTICOS_v1.md seccion 12   P63 missing != sold
    NIPC_CONTRATOS_SEMANTICOS_v1.md seccion 13   P64 corporate actions
    NIPC_CONTRATOS_SEMANTICOS_v1.md seccion 14   P65 manager duplication

---

## 9. Estado final del repositorio

    HEAD local          22a92ae (o posterior)
    origin/main         9d4a81e
    Ahead               197 commits
    Behind              3 (bot CI)
    Working tree        limpio
    Push                NO

---

Fin del expediente. Redactado 2026-09-21. HEAD 22a92ae.