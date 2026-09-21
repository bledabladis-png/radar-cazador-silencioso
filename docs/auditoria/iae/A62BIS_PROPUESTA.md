# IAE - A.6.2-bis Propuesta de rediseno arquitectonico TARGET (v2)

**Version:** v2. Aplicadas las 4 correcciones materiales + precisiones
del dictamen #44 (2026-09-21). Sometida a nueva verificacion documental.

**Version previa:** v1 (commit f383932, dictamen #44). El backup esta
en `A62BIS_PROPUESTA.md.v1.bak` solo para trazabilidad local; se
elimina tras la verificacion.

**Objeto:** propuesta de diseno para materializar los 3 bloqueantes
estructurales F2.4 (B1 TARGET independiente + B2 point-in-time + B3
13F != flujo en tiempo real), autorizada por dictamen #43.

**Fecha:** 2026-09-21.

**HEAD al redactar:** 1e83d70.

**Naturaleza:** propuesta. NO normativa. Sometida a dictamen.

---

## 0. Resumen ejecutivo

v2 aplica las 4 correcciones materiales del dictamen #44:

1. **F2 (autorreferencia hash):** el `version_id` NO se deriva del
   contenido que lo contiene. Manifiesto externo lleva sha256 completo.
2. **F3 (inmutabilidad real):** ningun snapshot publicado se modifica
   in-place. Los intervalos los determina un `catalog_manifest.json`
   externo, no el `.meta.json` historico.
3. **F4 (backdating prohibido):** el catalogo actual NO se asigna
   retroactivamente a periodos historicos. Sin snapshot historico ->
   `UNAVAILABLE`. Q4 2025 / Q1 2026 quedan sin snapshot.
4. **F5 (knowledge_date):** no se equipara a `max(FILING_DATE)`.
   Se elige explicitamente la semantica contractual (opcion A: fecha
   de publicacion del filing que origino la observacion).

Ademas, precisiones del dictamen:

- B1: `build_target` no acoplado a `period_end` + `catalog_version`.
  La resolucion PIT precede a la construccion TARGET.
- B1: `TARGET_UNRESOLVED` como estado explicito. No descartar filas
  del catalogo sin `share_class_figi`.
- `compute_coverage_pairwise`: inventario de consumidores antes de
  refactor.
- Orden de commits: B2 primero, luego B1, luego B3.
- Criterio de cierre fijado (seccion 7).

**Fuera de alcance:** OpenFIGI masivo, recalculo de evidencia final,
modificacion de contratos, `DROP_DUP`, certificacion "acumulacion",
Policy v1.3, Gate-NIPC.2/3.

---

## 1. Estado de partida (referencia A.6.0)

| Bloqueante | Piezas existentes | Piezas ausentes |
|---|---|---|
| B1 | `coverage.py::compute_contractual_coverage`, `PositionRecord`, `aggregate_positions_by_shareclass_figi`, `target_universe.py::resolve_cusips` | `target_builder.py`, integracion `nipc.py` |
| B2 | `source_date` (no contractual) | versionado de snapshots + `target_catalog_as_of` |
| B3 | Doctrina P63/P64/P65 | `knowledge_date` con semantica explicita, `absence.py` |

Hecho estructural: `coverage.py` ya define la API contractual P38 con
TARGET externo. `nipc.py::compute_coverage_pairwise` mantiene una
implementacion paralela con `observed_security_key` (CUSIPs) como
proxy. La v2 unifica semanticamente sin fusionar fisicamente los
modulos.
---

## 2. B1 - TARGET independiente del exito del mapping

**Cautelas v2:**
- Separar resolucion PIT de construccion TARGET (dictamen #44 seccion 1.1).
- Filas sin `share_class_figi` -> `TARGET_UNRESOLVED`, no descartar.
- No acoplar `build_target` a `catalog_version`: el snapshot lo resuelve
  primero `target_catalog_as_of`.

### 2.1. Arquitectura

    period_end
        v
    target_catalog_as_of(period_end, catalog_root)   [B2]
        v
    snapshot resuelto (inmutable)
        v
    target_builder.build_target(snapshot, period_end)  [B1]
        v
    TARGET_P = {share_class_figi: ...} + {UNRESOLVED: ...}
        v
    coverage.compute_contractual_coverage(TARGET_P, RESOLVED_P)

**Invariante B1:** `TARGET_P` se construye exclusivamente desde el
snapshot del catalogo. NO depende de:
- El resultado del mapping CUSIP_13F -> share_class_figi.
- La presencia/ausencia de un CUSIP concreto en el filing 13F.
- El exito de OpenFIGI sobre el filing concreto.

### 2.2. `identity/target_builder.py`

    @dataclass(frozen=True)
    class TargetUniverse:
        period_end: str
        catalog_version_id: str
        catalog_sha256: str
        resolved: frozenset[str]        # share_class_figi
        unresolved: frozenset[str]      # radar_ticker sin FIGI (TARGET_UNRESOLVED)

    def build_target(snapshot_df, *, period_end, catalog_version_id, catalog_sha256):
        """Materializa el TARGET para el periodo a partir del snapshot.

        El snapshot ya viene resuelto por target_catalog_as_of.
        Este modulo NO consulta catalogo ni sistema de ficheros.
        """

**Estados:**

    OK           -> figi + share_class_figi presentes -> suma a .resolved
    PARTIAL      -> figi sin share_class_figi          -> suma a .unresolved (TARGET_UNRESOLVED)
    MISS         -> sin hit en OpenFIGI                -> suma a .unresolved (TARGET_UNRESOLVED)

**Nota:** `TARGET_UNRESOLVED` mantiene el universo contractual en su
magnitud declarada. Un consumidor que necesite el subconjunto
efectivamente resoluble usa `target.resolved`. Un consumidor que
necesite la magnitud declarada usa `target.resolved | target.unresolved`.

### 2.3. `target_universe.resolve_cusips` (sin cambios)

Sigue siendo el resolver inverso (CUSIP observado -> membership). No
se toca. El diccionario `scf_index` del catalogo se construye sobre
`resolved` (FIGIs presentes).

### 2.4. `compute_coverage_pairwise` (wrapper de compatibilidad)

**Gate 0 obligatorio previo al refactor:** inventariar todos los
consumidores y fijar el contrato de retorno tipado. La v2 declara el
contrato objetivo:

    def compute_coverage_pairwise(units_current, units_previous, *,
                                   target_q4=None, target_q1=None,
                                   records_q4=None, records_q1=None):
        """
        Wrapper de compatibilidad P38.
        Si target_q4/target_q1/records_* se aportan -> delega a
        coverage.compute_contractual_coverage.
        Si NO -> devuelve dict con coverage_status = "UNAVAILABLE" y
        campos numericos a None. NO devuelve proxy silencioso.
        Tipo de retorno: dict (mismo shape que compute_contractual_coverage).
        """

**Contrato de retorno:** dict con las mismas claves que
`compute_contractual_coverage` (los 7 campos). Los numericos son
`float | None`; `coverage_status` es `"VALID" | "UNAVAILABLE"`.

**Accion previa (commit 1 de B1):** `grep -rn "compute_coverage_pairwise"
tests/ src/ scripts/` para inventariar. Si algun consumidor asume float
no-nulo, se anota y se decide en dictamen.

### 2.5. Tests B1

- `build_target` con snapshot 3 filas OK -> `resolved == 3 FIGIs`, `unresolved == 0`.
- `build_target` con snapshot 2 OK + 1 MISS -> `resolved == 2`, `unresolved == 1`.
- `build_target(mapping OK) == build_target(mapping fallido)`: mismo snapshot, distinto observed -> mismo TARGET_P.
- `build_target` NO consulta catalogo (puro sobre snapshot).
- `compute_coverage_pairwise` con TARGET -> llama a `compute_contractual_coverage`.
- `compute_coverage_pairwise` sin TARGET -> dict con `coverage_status == "UNAVAILABLE"`, sin `float` implicitos.

---

## 3. B2 - Point-in-time (modelo corregido)

**Correcciones aplicadas:** F2 (autorreferencia hash) + F3 (inmutabilidad)
+ F4 (backdating prohibido) + intervalos semiabiertos.

### 3.1. Estructura de disco

    data/mappings/catalog_snapshots/
        snapshot_<version_id>.csv         # contenido del catalogo
        snapshot_<version_id>.sha256      # hash completo, fichero aparte
    data/mappings/catalog_manifest.json   # indice mutable, es el "current" historico

**Reglas:**

- El `.csv` NO contiene su propio hash.
- El `.sha256` (fichero separado) contiene el SHA-256 completo del `.csv`.
- `version_id`: identificador unico, derivado de la fecha de creacion del
  snapshot + secuencia (e.g., `20260919_01`). Puede ser cualquier cadena
  unica; NO tiene que ser el hash (evita autorreferencia).
- El `catalog_manifest.json` es el unico objeto que declara intervalos:

      {
        "schema_version": 1,
        "snapshots": [
          {"version_id": "20260919_01",
           "valid_from": "2026-09-19",
           "valid_to": null,
           "sha256": "<hash completo>",
           "csv_path": "catalog_snapshots/snapshot_20260919_01.csv"}
        ]
      }

### 3.2. Inmutabilidad real

- Un `snapshot_*.csv` publicado NO se modifica nunca.
- Un `snapshot_*.sha256` publicado NO se modifica nunca.
- El `catalog_manifest.json` SI puede cambiar (es un indice).
- El manifiesto conserva su propia historia en git. Ademas, cada cambio
  al manifiesto se puede acompanar de un `catalog_manifest.<timestamp>.json`
  archivado si se necesita trazabilidad explicita.

**Regla contractual:** ninguna evidencia historica publicada se modifica
in-place. Cualquier correccion se publica como snapshot nuevo.

### 3.3. Intervalos semiabiertos

`valid_from` inclusivo, `valid_to` exclusivo: `[valid_from, valid_to)`.
Ejemplo sin solapamiento:

    V1 [2026-01-01, 2026-09-19)
    V2 [2026-09-19, NULL)      # NULL = vigente

`target_catalog_as_of(period_end)`:

    0 snapshots cubren period_end -> raise CatalogNotAvailable
    1 snapshot cubre period_end  -> devuelve el DataFrame
    >1 snapshots cubren          -> raise CatalogAmbiguous

### 3.4. Backdating prohibido (F4)

**No se permite** asignar `valid_from` pasado al catalogo actual.
Consecuencia: **Q4 2025 y Q1 2026 no tienen snapshot**.

    target_catalog_as_of("2025-12-31") -> raise CatalogNotAvailable
    target_catalog_as_of("2026-03-31") -> raise CatalogNotAvailable
    target_catalog_as_of("2026-09-19") -> OK (snapshot 20260919_01)

**Implicacion para A.6.4:** el recalculo de evidencia Q4/Q1 quedara
como `historical target = UNAVAILABLE / N/D`. La v2 declara esta
consecuencia como decision consciente.

**Alternativa (a dictamen):** si el auditor requiere un snapshot historico
para Q4 2025 / Q1 2026, debe existir evidencia externa (probe OpenFIGI
historico, auditoria previa, otro dataset). No se fabrica retrospectivamente.

### 3.5. Tests B2

- `target_catalog_as_of` con 0 snapshots -> raise.
- `target_catalog_as_of` con 1 snapshot que cubre -> DataFrame.
- `target_catalog_as_of` con 2 snapshots solapados -> raise CatalogAmbiguous.
- `target_catalog_as_of("2025-12-31")` sin snapshot -> raise (backdating prohibido).
- SHA-256 del `.csv` NO coincide con el declarado -> raise.
- Intervalos `[from, to)` no solapan entre snapshots consecutivos.
- Inmutabilidad: dos "runs" consecutivos de un test de integridad
  verifican que el `.csv` no cambia entre invocaciones.
---

## 4. B3 - 13F != flujo en tiempo real

**Correccion aplicada:** F5 - `knowledge_date` NO se equipara a
`max(FILING_DATE)`. Semantica explicita elegida.

### 4.1. Tres timestamps, tres semanticas

    period_end       cierre del trimestre (13F: 31-mar, 30-jun, ...)
    filing_date      fecha del filing (SUBMISSION.FILING_DATE)
    knowledge_date   fecha en que el hecho fue publico (ver 4.2)

### 4.2. Semantica elegida para `knowledge_date`

**Opcion A (propuesta):** `filing_date` del filing que origino la
observacion concreta.

Razon: es la unica semantica point-in-time estricta que no introduce
look-ahead dentro del periodo. Cada `PositionRecord` lleva la fecha en
que su filing fue presentado. Un analisis a "fecha de febrero" ve solo
los filings publicados hasta febrero.

**Descartadas:**

- B (fecha de ingesta por pipeline): requiere un log externo de ingesta
  que hoy no existe. Es determinista pero depende de estado externo al
  dataset. No es point-in-time en sentido de disponibilidad publica.
- C (max del periodo): introduce look-ahead (posicion de enero marcada
  como conocida en abril).

**Campo derivado opcional:** `period_consolidation_date = max(FILING_DATE)`
sobre el periodo, como metadato del snapshot consolidado, separado de
`knowledge_date`. Se puede anadir si un consumidor lo necesita. No es
el contrato de `knowledge_date`.

### 4.3. Ampliacion de `PositionRecord`

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
        # Nuevos campos v2:
        period_end: Optional[str] = None
        filing_date: Optional[str] = None
        knowledge_date: Optional[str] = None

Retrocompatible: los 3 campos son opcionales por defecto. `period` se
mantiene.

### 4.4. `absence.py` (stub diferido)

Se declara el modulo con enums + NotImplementedError:

    MISSING
    BELOW_REPORTING_THRESHOLD
    CONFIDENTIAL
    OTHER_MANAGER
    UNKNOWN
    ZERO_REPORTED
    NOT_PRESENT

Docstring con reglas P63 (R1-R8). `classify_absence(...)` lanza
`NotImplementedError`. Ninguna ruta productiva lo invoca.

### 4.5. Regla dura preservada

`delta_shares` NO crea `SOLD`. `P64_EVENTS_DEFERRED` se mantiene.
Distincion corporate action vs economic accumulation sigue diferida
a fuente externa.

### 4.6. Tests B3

- `PositionRecord` con 3 timestamps -> OK.
- `PositionRecord` sin timestamps -> OK (retrocompat).
- `knowledge_date == filing_date` de la observacion concreta
  (por construccion).
- Test explicito: dos observaciones del mismo periodo con filing_date
  distinta -> knowledge_date distinta.
- `absence.py::classify_absence` -> NotImplementedError.
- `delta_shares` no produce SOLD (test existente P63).

---

## 5. Orden de commits (revisado por dictamen #44 seccion 10)

    1. B2  Modelo de identidad/versionado
           - Estructura catalog_snapshots/ + catalog_manifest.json
           - target_catalog_as_of (fail-closed, semiabierto)
           - tests B2

    2. B1  target_builder + TargetUniverse
           - build_target (puro sobre snapshot)
           - tests B1 target_builder

    3. B1  Integracion coverage
           - Gate 0: inventario consumidores compute_coverage_pairwise
           - Refactor wrapper con contrato de retorno dict
           - tests B1 integracion

    4. B3  Timestamps
           - PositionRecord extendido (opcional)
           - knowledge_date = filing_date de la observacion
           - absence.py stub
           - tests B3

    5. Integracion end-to-end + verificacion global
           - P65 PASS + P66 PASS + nuevos PASS
           - compileall + pyflakes
           - Sin regresion nueva

**Razon del orden:** B1 depende de B2 para que TARGET sea temporalmente
valido. B2 primero evita construir TARGET contra un modelo de catalogo
que aun no tiene versionado.
---

## 6. Tests requeridos (resumen)

| Bloque | Fichero | Cobertura |
|---|---|---|
| B1 | `tests/test_target_builder.py` (nuevo) | build_target puro, resolved/unresolved, independencia del mapping |
| B1 | `tests/test_sec_13f_nipc.py` (extender) | wrapper con/sin TARGET, contrato tipado |
| B2 | `tests/test_target_catalog_as_of.py` (nuevo) | snapshots, fail-closed, ambiguedad, inmutabilidad, semiabierto |
| B3 | `tests/test_position_record.py` (nuevo) | 3 timestamps, retrocompat, knowledge_date == filing_date |
| B3 | `tests/test_absence.py` (nuevo) | enums declarados + NotImplementedError |

**Cero regresion P65/P66 esperada.** 31 + 31 tests existentes intactos.

---

## 7. Criterio de cierre A.6.2-bis (fijado por dictamen #44 seccion 11)

### B1

- `build_target(mapping OK) == build_target(mapping fallido)` para mismo
  snapshot (test explicito).
- `TARGET_P != f(CUSIP observado)` (test explicito).
- `compute_coverage_pairwise` sin TARGET -> comportamiento tipado
  documentado (`dict` con `coverage_status == "UNAVAILABLE"`).

### B2

- `target_catalog_as_of(periodo)`:
  - 0 snapshots -> fail-closed.
  - 1 snapshot -> exito.
  - >1 solapados -> `CatalogAmbiguous`.
  - hash invalido -> fail-closed.
- Ningun snapshot historico inventado.
- Snapshot publicado es inmutable (test de integridad entre runs).
- Intervalos `[valid_from, valid_to)` sin solapamiento.

### B3

- `period_end != filing_date != knowledge_date` cuando semánticamente
  difieran (test explicito).
- Ninguna ruta interpreta filing posterior como conocimiento anterior.

### Global

- P65 tests PASS (31 actuales).
- P66 tests PASS (31 actuales).
- Nuevos tests A.6.2-bis PASS.
- `compileall` OK.
- `pyflakes` LIMPIO.
- Suite sin regresion nueva.

Solo entonces: **A.6.2-bis CLOSED -> A.6.3 AUTHORIZED**.
---

## 8. Preguntas al auditor (v2)

1. **B1 - Firma de `build_target`.** Confirmar la separacion
   `target_catalog_as_of` -> `build_target(snapshot, ...)`.

2. **B1 - `TargetUniverse`.** Se propone `resolved` + `unresolved`
   (TARGET_UNRESOLVED). ¿Se aprueba este contrato explicito para
   filas sin FIGI?

3. **B1 - Contrato de retorno de `compute_coverage_pairwise`.**
   `dict` con `coverage_status = "VALID" | "UNAVAILABLE"` y numericos
   `float | None`. ¿Se aprueba? Nota: hay una accion previa (gate 0
   de consumidores) antes de refactorizar.

4. **B2 - Estructura.** `catalog_snapshots/` + `catalog_manifest.json`.
   `version_id` desacoplado del hash. SHA-256 completo en fichero aparte.
   ¿Se aprueba?

5. **B2 - Intervalos.** Semiabierto `[valid_from, valid_to)`. Sin
   solapamiento. ¿Se aprueba?

6. **B2 - Backdating.** Prohibido. Q4 2025 / Q1 2026 quedaran sin
   snapshot -> `UNAVAILABLE` en A.6.4. ¿Se acepta esta consecuencia
   o se requiere evidencia adicional?

7. **B3 - Semantica `knowledge_date`.** Opcion A: fecha del filing que
   origino la observacion. ¿Se aprueba? `period_consolidation_date`
   (max) queda como campo derivado opcional.

8. **B3 - `absence.py` stub.** Enums + NotImplementedError. Ninguna
   ruta productiva lo invoca. ¿Se aprueba?

9. **Orden de commits.** B2 -> B1 -> B3 -> integracion. ¿Se aprueba?

10. **Criterio de cierre.** Los 3 bloques de la seccion 7 + global.

---

## 9. Lo que NO se toca en A.6.2-bis

- Contratos: `NIPC_CONTRATOS_SEMANTICOS_v1.md`, `NIPC_COVERAGE_POLICY.md` v1.0.
- Modulos: `delta_shares.py`, `security_identity.py`, `relationships.py`,
  `amendments.py`, `temporal_validity.py`, `coverage.py` (se invoca, no
  se modifica).
- `radar_target_catalog.csv` actual: se conserva como snapshot inicial.
- OpenFIGI masivo: NO ejecutado.
- `DROP_DUP`: NO activado.
- Push a `origin/main`: NO.
- Certificacion "acumulacion": NO.

---

## 10. Trazabilidad

    Dictamen #43                 A.6.0 CERRADO + A.6.2-bis AUTORIZADO
    Dictamen #44                 v1 NO-GO; 4 correcciones materiales
    A.6.0 inventario             INFORME.md seccion 21
    F2.4 (dictamen #24)          3 bloqueantes estructurales
    RECONCILIACION D1/D2/D3      divergencias contrato <-> codigo
    Contratos P38/P62/P63         secciones 3, 11, 12

---

Fin de la propuesta v2. Sometida a nueva verificacion documental.
HEAD 1e83d70.