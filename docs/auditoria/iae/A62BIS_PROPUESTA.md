# IAE - A.6.2-bis Propuesta de rediseno arquitectonico TARGET (v3)

**Version:** v3. Aplicadas las 3 correcciones materiales + la precision
sobre amendments + las precisiones sobre PositionRecord del dictamen
#45 (2026-09-21).

**Versiones previas:** v1 (commit f383932, NO-GO #44) y v2 (commit
9aa0727, GO CONDICIONAL #45). Backups locales como .v1.bak / .v2.bak
hasta verificacion; se eliminan despues.

**Objeto:** propuesta de diseno para materializar los 3 bloqueantes
estructurales F2.4 (B1 + B2 + B3).

**Fecha:** 2026-09-21.
**HEAD al redactar:** 9aa0727.
**Naturaleza:** propuesta. NO normativa. Sometida a dictamen.

---

## 0. Resumen ejecutivo

v3 aplica las 3 correcciones materiales del dictamen #45 + precisiones:

1. **F1 (B1, semantica del denominador):** eleccion explicita de la
   Opcion 2. El denominador contractual se calcula sobre la
   cardinalidad declarada del catalogo. `resolved` es el subconjunto
   apto para operaciones que requieren FIGI. Separacion estricta:
   TARGET declarado != TARGET resoluble != TARGET observado.

2. **F2 (B2, test de inmutabilidad):** prueba de integridad por
   hash historico. Deteccion fail-closed de modificacion posterior.
   Validacion de que snapshot publicado != objeto mutable de trabajo.

3. **F3 (B3, contradiccion):** criterio de cierre corregido:
   `knowledge_date == filing_date` por contrato;
   `period_end != filing_date` cuando difieren;
   `knowledge_date` nunca anterior al filing origen.

4. **Precision amendments:** `knowledge_date` = filing_date del
   filing que realmente aporta el estado observado al snapshot
   efectivo (no el primer filing historico de la cadena).

5. **PositionRecord:** invariante `period == period_end` verificable.
   Distincion entre legacy record (3 timestamps None) y contractual.

**Fuera de alcance:** OpenFIGI masivo, recalculo de evidencia final,
modificacion de contratos, `DROP_DUP`, certificacion "acumulacion",
Policy v1.3, Gate-NIPC.2/3.

---

## 1. Estado de partida (referencia A.6.0)

| Bloqueante | Piezas existentes | Piezas ausentes |
|---|---|---|
| B1 | `coverage.py::compute_contractual_coverage`, `PositionRecord`, `aggregate_positions_by_shareclass_figi`, `target_universe.py::resolve_cusips` | `target_builder.py`, integracion `nipc.py` |
| B2 | `source_date` (no contractual) | versionado + `target_catalog_as_of` |
| B3 | Doctrina P63/P64/P65 | `knowledge_date` con semantica explicita, `absence.py` |

Hecho estructural: `coverage.py` ya define la API contractual P38 con
TARGET externo. `nipc.py::compute_coverage_pairwise` mantiene una
implementacion paralela con `observed_security_key` (CUSIPs) como
proxy. La v3 unifica semanticamente sin fusionar fisicamente los
modulos.
---

## 2. B1 - TARGET independiente del exito del mapping (F1 resuelta)

**Correccion F1 aplicada:** eleccion explicita de la Opcion 2 del
dictamen #45.

### 2.1. Tres universos separados

    TARGET declarado     = {entradas del catalogo (identidad estable)}
    TARGET resoluble     = {share_class_figi presentes}    subset declarado
    TARGET observado     = {share_class_figi con 13F observado}

**Invariante estructural:**

    TARGET declarado NO depende del mapping.
    TARGET declarado NO depende del 13F observado.
    TARGET declarado NO depende de OpenFIGI.

Un fallo de mapping produce ausencia en `TARGET observado`, nunca en
`TARGET declarado`.

### 2.2. Semantica del denominador contractual

**Eleccion: Opcion 2.**

    Denominador contractual = |TARGET declarado|

No depende de `share_class_figi`. Depende de la cardinalidad declarada
del catalogo. `resolved` es el subconjunto de `declarado` apto para
operaciones que requieren FIGI.

Razon: la magnitud declarada del catalogo es el universo contractual
definido por el radar. El exito o fallo del mapping no debe alterar
esa magnitud.

### 2.3. Arquitectura

    period_end
        v
    target_catalog_as_of(period_end, catalog_root)   [B2]
        v
    snapshot resuelto (inmutable)
        v
    target_builder.build_target(snapshot, *, period_end,
                                 catalog_version_id, catalog_sha256)
        v
    TargetUniverse:
        declared   frozenset[catalog_key]      # identidad de catalogo
        resolved   frozenset[share_class_figi] # subset con FIGI
        unresolved frozenset[catalog_key]      # subset sin FIGI

**Identidad de catalogo:** el campo `radar_ticker` es la identidad
primaria declarada (presente en todas las filas del catalogo por
construccion). No se mezcla con `share_class_figi`.

**Denominador contractual** (para `compute_contractual_coverage`):
`len(universe.declared)`.

### 2.4. `identity/target_builder.py`

    @dataclass(frozen=True)
    class TargetUniverse:
        period_end: str
        catalog_version_id: str
        catalog_sha256: str
        declared: frozenset[str]        # radar_ticker (identidad declarada)
        resolved: frozenset[str]        # share_class_figi
        unresolved: frozenset[str]      # radar_ticker sin FIGI

    def build_target(snapshot_df, *, period_end,
                     catalog_version_id, catalog_sha256) -> TargetUniverse:
        """Materializa el TargetUniverse a partir del snapshot.

        El snapshot viene resuelto por target_catalog_as_of.
        Funcion pura. NO consulta catalogo ni sistema de ficheros.
        """

**Estados por fila del catalogo:**

    OK        -> figi + share_class_figi presentes -> declared + resolved
    PARTIAL   -> figi sin share_class_figi          -> declared + unresolved
    MISS      -> sin hit en OpenFIGI                -> declared + unresolved

**Identidad declarada:** `radar_ticker` (siempre presente).
`declared = {row.radar_ticker for row in snapshot}`.

### 2.5. Contrato con `compute_contractual_coverage`

La funcion contractual de `coverage.py` recibe:
- `target_q4` / `target_q1`: `set` de identidades de catalogo
  (`radar_ticker`) -> cardinalidad declarada.
- `records_q4` / `records_q1`: `PositionRecord` con `share_class_figi`
  (subconjunto resoluble + observado).

**Nota tecnica:** `coverage.py::compute_contractual_coverage` actual
opera por `share_class_figi`. Para soportar la Opcion 2 requiere un
ajuste minimo: el denominador se calcula sobre `target_q4 & target_q1`
como identidades declaradas; el numerador `paired` requiere que ambos
lados tengan `share_class_figi` comun. Este ajuste se implementara en
el commit 3 de B1 (integracion) previa revision.

**Alternativa si se prefiere no tocar `coverage.py`:** introducir un
`catalog_key -> share_class_figi` mapping declarado y computar el
denominador sobre el rango del mapping (con `None` mapeado a
`TARGET_UNRESOLVED`). Esta opcion es funcionalmente equivalente; se
propone la primera por simplicidad.

### 2.6. `target_universe.resolve_cusips` (sin cambios)

Sigue siendo el resolver inverso (CUSIP observado -> membership).
No se toca.

### 2.7. `compute_coverage_pairwise` (wrapper)

**Gate 0 obligatorio previo al refactor** (dictamen #45 mantiene):

    grep -rn "compute_coverage_pairwise" tests/ src/ scripts/

Contrato de retorno declarado:

    dict con las mismas 7 claves que compute_contractual_coverage.
    Numericos: float | None.
    coverage_status: "VALID" | "UNAVAILABLE".
    Sin TARGET aportado -> "UNAVAILABLE", sin proxy silencioso.

### 2.8. Tests B1

- `declared = {radar_ticker}` para snapshot de N filas.
- `resolved ⊆ declared` (por construccion).
- `unresolved = declared - resolved` en cardinalidad.
- `mapping OK` vs `mapping fallido` -> mismo `declared`.
- Denominador contractual = `len(declared)` en ambos casos.
- `compute_coverage_pairwise` con TARGET -> delegacion; sin TARGET -> dict tipado.
---

## 3. B2 - Point-in-time (F2 resuelta)

**Correccion F2 aplicada:** test de inmutabilidad reforzado.
Deteccion fail-closed de modificacion posterior. Hash historico
verificable.

### 3.1. Estructura de disco (sin cambios respecto v2)

    data/mappings/catalog_snapshots/
        snapshot_<version_id>.csv
        snapshot_<version_id>.sha256       # hash completo, separado
    data/mappings/catalog_manifest.json    # indice mutable

**version_id** = `<YYYYMMDD>_<NN>` (secuencia diaria). No depende del
contenido del CSV.

**catalog_manifest.json** (unico objeto mutable):

    {
      "schema_version": 1,
      "snapshots": [
        {
          "version_id": "20260919_01",
          "valid_from": "2026-09-19",
          "valid_to": null,
          "sha256": "<hash completo>",
          "csv_path": "catalog_snapshots/snapshot_20260919_01.csv"
        }
      ]
    }

### 3.2. Inmutabilidad real

- `snapshot_*.csv` publicado: no se modifica nunca in-place.
- `snapshot_*.sha256` publicado: no se modifica nunca in-place.
- `catalog_manifest.json`: mutable, es indice. Su historia vive en git.

**Regla contractual:** ninguna evidencia historica publicada se
modifica in-place. Correcciones = snapshot nuevo.

### 3.3. Intervalos semiabiertos

`[valid_from, valid_to)`. `valid_to = null` = vigente.

`target_catalog_as_of(period_end, *, catalog_root)`:

    0 snapshots cubren period_end -> raise CatalogNotAvailable
    1 snapshot cubre period_end   -> devuelve (DataFrame, version_id, sha256)
    >1 snapshots cubren           -> raise CatalogAmbiguous

### 3.4. Backdating prohibido (F4 de #44, mantenido)

`valid_from = 2026-09-19` para el snapshot inicial. Q4 2025 / Q1 2026
-> `CatalogNotAvailable`. No se fabrica retrospectivamente.

**Implicacion A.6.4:** recalculo Q4/Q1 -> `UNAVAILABLE`. Decision
consciente. Si el auditor requiere evidencia historica, debe aportarse
externa (probe OpenFIGI historico, dataset externo, auditoria previa).

### 3.5. Test de inmutabilidad reforzado (F2 resuelta)

**Test de integridad contractual (obligatorio):**

    def test_snapshot_integrity_fail_closed():
        # 1. Cargar snapshot publicado.
        snap_path = catalog_snapshots / f"snapshot_{vid}.csv"
        sha_path  = catalog_snapshots / f"snapshot_{vid}.sha256"

        # 2. Leer el hash publicado (fichero separado).
        published_sha = sha_path.read_text().strip()

        # 3. Recalcular sobre el .csv en disco.
        recalculated = hashlib.sha256(snap_path.read_bytes()).hexdigest()

        # 4. Fail-closed si discrepan.
        assert recalculated == published_sha, (
            "SNAPSHOT MODIFICADO O CORRUPTO: "
            "hash recalculado != hash publicado"
        )

        # 5. Validar contra el manifiesto.
        manifest = json.loads(manifest_path.read_text())
        entry = next(s for s in manifest["snapshots"] if s["version_id"] == vid)
        assert entry["sha256"] == published_sha, (
            "MANIFIESTO DESINCRONIZADO con snapshot_*.sha256"
        )

**Cualquier modificacion posterior del contenido -> detectable y
fail-closed.** El test no demuestra solo "no se ha modificado en esta
ejecucion"; demuestra integridad verificable contra evidencia
publicada.

**Corrupcion simulada (test adicional):**

    def test_snapshot_corruption_detected():
        # Copia temporal con 1 byte alterado.
        # Verifica que recalcular el hash no coincide con el publicado.
        # Verifica que compute_as_of lanza (o aborta) antes de usar el
        # snapshot corrupto.

**Validacion de separacion objeto publicado != objeto de trabajo:**

    - Los tests leen del directorio `catalog_snapshots/`.
    - `target_catalog_as_of` NO escribe ficheros.
    - Ninguna funcion de B2 abre un snapshot con modo escritura.

### 3.6. Tests B2

- 0 snapshots -> `CatalogNotAvailable`.
- 1 snapshot -> exito, devuelve (df, vid, sha256).
- >1 solapados -> `CatalogAmbiguous`.
- `as_of("2025-12-31")` sin snapshot -> `CatalogNotAvailable` (no backdating).
- Integridad: sha256 recalculado == publicado (fail-closed).
- Corrupcion: sha256 divergente -> fail-closed.
- Intervalos `[from, to)` sin solapamiento.
- Inmutabilidad entre "runs": 2 invocaciones del test de integridad.
---

## 4. B3 - Semantica temporal (F3 resuelta + amendments)

**Correcciones aplicadas:**

- F3: contradiccion del criterio de cierre corregida.
- Precision: semantica de `knowledge_date` con cadena de amendments.

### 4.1. Tres timestamps, tres semanticas

    period_end      cierre del trimestre (13F: 31-mar, 30-jun, ...)
    filing_date     fecha del filing (SUBMISSION.FILING_DATE)
    knowledge_date  fecha en que la observacion fue publica

### 4.2. Semantica de `knowledge_date` (corregida)

**Contrato:**

    knowledge_date == filing_date   para la observacion concreta.

Es decir: `knowledge_date` NO es un campo independiente que pueda
diferir de `filing_date` por decision del caller. Es la misma fecha.

**Regla dura:**

    knowledge_date NUNCA puede ser anterior al filing que origina la
    observacion.

Si existe una cadena de amendments, `knowledge_date` es la fecha del
filing que **realmente aporta el estado observado** al snapshot
efectivo (ver 4.3).

### 4.3. Amendments (precision adicional)

Cuando la cadena documental tiene RESTATEMENT o NEW HOLDINGS:

- El filing efectivo puede ser el amendment (no el filing original).
- La `PositionRecord` que entra al snapshot efectivo lleva
  `filing_date` = `FILING_DATE` del filing que realmente aporta el
  estado observado al snapshot.
- `knowledge_date` = misma fecha.

Ejemplo:

    filing original   2026-01-30   RESTATEMENT
    amendment 1       2026-02-15   RESTATEMENT (reemplaza al anterior)
    amendment 2       2026-03-10   NEW HOLDINGS

Si el snapshot efectivo se construye desde amendment 2 -> `filing_date
= 2026-03-10`, `knowledge_date = 2026-03-10`.

Razon: es el filing que efectivamente aporta el estado observado. No
se usa el primer filing historico como referencia.

Coherencia con P64: `compute_effective_snapshot` produce el snapshot
efectivo; la `PositionRecord` se construye desde sus lineas.

### 4.4. `PositionRecord` extendido

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
        # Nuevos:
        period_end: Optional[str] = None
        filing_date: Optional[str] = None
        knowledge_date: Optional[str] = None

**Invariantes verificables (F de #45 seccion 7):**

- Si `period` y `period_end` estan ambos presentes:
  `period` es la representacion trimestral de `period_end`
  (`period_end` = ultimo dia del trimestre referenciado por `period`).
  `2025Q4` <-> `2025-12-31`; `2026Q1` <-> `2026-03-31`.
- Si `filing_date` y `knowledge_date` estan ambos presentes:
  `knowledge_date == filing_date`.
- Si `knowledge_date` presente y `filing_date` no: no aplica (registro
  no es contractual B3).

**Distincion legacy vs contractual:**

- **Legacy record:** los 3 timestamps son `None`. Proveniente de codigo
  pre-B3. Válido para retrocompatibilidad.
- **Contractual B3 record:** los 3 timestamps presentes y coherentes
  con las invariantes. Válido para rutas B3.

Una ruta contractual B3 NO puede aceptar un registro con los 3
timestamps a `None`: debe fallar antes.

**Helper:**

    def is_contractual_b3(rec) -> bool:
        return (
            rec.period is not None
            and rec.period_end is not None
            and rec.filing_date is not None
            and rec.knowledge_date is not None
            and rec.knowledge_date == rec.filing_date
        )

### 4.5. `absence.py` (stub diferido, aprobado por #45 seccion 8)

Modulo con enums `MISSING | BELOW_REPORTING_THRESHOLD | CONFIDENTIAL |
OTHER_MANAGER | UNKNOWN | ZERO_REPORTED | NOT_PRESENT`. Docstring con
reglas P63 R1-R8. `classify_absence(...)` -> `NotImplementedError`.
Ninguna ruta productiva lo invoca.

Estado: `P63 absence classifier = DEFERRED`.

### 4.6. Regla dura preservada

`delta_shares` NO crea `SOLD`. `P64_EVENTS_DEFERRED` se mantiene.

### 4.7. Tests B3

- `PositionRecord` con 3 timestamps -> OK.
- `PositionRecord` sin timestamps -> OK legacy.
- `knowledge_date == filing_date` para la observacion (por construccion).
- Invariante `period <-> period_end`: `2025Q4 <-> 2025-12-31`.
- Invariante violada -> fail.
- `is_contractual_b3` -> True solo si 3 timestamps + coherencia.
- Amendment RESTATEMENT: `filing_date` = fecha del amendment que
  aporta el estado efectivo (no la del original).
- `absence.py::classify_absence` -> `NotImplementedError`.
- `delta_shares` no produce SOLD (test existente P63).

---

## 5. Orden de commits (sin cambios respecto v2)

    1. B2  Modelo de versionado + target_catalog_as_of
    2. B1  target_builder + TargetUniverse
    3. B1  Integracion coverage (gate 0 consumidores primero)
    4. B3  Timestamps + PositionRecord extendido + absence.py stub
    5. Integracion end-to-end + verificacion global

Razon: B1 depende de B2 para validez temporal. B3 depende de B1
(necesita PositionRecord con FIGIs resueltos).
---

## 6. Tests requeridos (resumen)

| Bloque | Fichero | Cobertura |
|---|---|---|
| B1 | `tests/test_target_builder.py` (nuevo) | TargetUniverse, declared/resolved/unresolved, independencia del mapping, denominador sobre declared |
| B1 | `tests/test_sec_13f_nipc.py` (extender) | wrapper con/sin TARGET, contrato tipado |
| B2 | `tests/test_target_catalog_as_of.py` (nuevo) | snapshots, fail-closed, ambiguedad, integridad reforzada (F2) |
| B3 | `tests/test_position_record.py` (nuevo) | 3 timestamps, invariantes, `is_contractual_b3`, legacy vs contractual |
| B3 | `tests/test_absence.py` (nuevo) | enums declarados + NotImplementedError |

**Cero regresion P65/P66 esperada.** 31 + 31 tests existentes intactos.

---

## 7. Criterio de cierre A.6.2-bis (corregido F3)

### B1

- Tres universos separados: declarado != resoluble != observado.
- Denominador contractual = `len(TARGET declarado)`.
- `mapping OK` vs `mapping fallido` -> **mismo `declared`** (test explicito).
- `TARGET declarado != f(CUSIP observado)` (test explicito).
- `compute_coverage_pairwise` sin TARGET -> contrato tipado
  (`dict` con `coverage_status == "UNAVAILABLE"`).

### B2

- `target_catalog_as_of(periodo)`:
  - 0 snapshots -> fail-closed.
  - 1 snapshot -> exito.
  - >1 solapados -> `CatalogAmbiguous`.
  - hash invalido -> fail-closed.
- Backdating prohibido: `as_of("2025-12-31")` -> `CatalogNotAvailable`.
- **Test de integridad reforzado (F2):**
  - `sha256` recalculado == `sha256` publicado (`.sha256` fichero aparte).
  - Manifiesto coherente con `sha256` publicado.
  - Corrupcion simulada (byte alterado) -> deteccion y fail-closed.

### B3

**Correccion F3 aplicada:**

    period_end != filing_date       cuando difieren temporalmente
    knowledge_date == filing_date   por contrato (opcion A)
    knowledge_date >= filing_date   invariante dura
    knowledge_date NUNCA anterior al filing que origina la observacion

- Test explicito: `knowledge_date == filing_date` para la observacion.
- Test explicito: `period_end != filing_date` cuando aplica.
- Test explicito: ninguna ruta interpreta filing posterior como
  conocimiento anterior.
- Amendments: `knowledge_date` = fecha del filing efectivo (RESTATEMENT
  aporta el estado), no del filing original.

### Global

- P65 tests PASS (31 actuales).
- P66 tests PASS (31 actuales).
- Nuevos tests A.6.2-bis PASS.
- `compileall` OK.
- `pyflakes` LIMPIO.
- Suite sin regresion nueva.

Solo entonces: **A.6.2-bis CLOSED -> A.6.3 AUTHORIZED.**
---

## 8. Preguntas al auditor (v3)

1. **F1 - Opcion elegida.** Se ha elegido la Opcion 2 del dictamen
   #45: denominador = cardinalidad declarada del catalogo; `resolved`
   es subconjunto. ¿Se aprueba?

2. **F1 - Identidad declarada.** Se propone `radar_ticker` como
   identidad primaria declarada. ¿Se aprueba? Alternativa: `composite_figi`
   o combinacion `radar_ticker + source_date`.

3. **F1 - `coverage.py`.** Para soportar Opcion 2 se requiere ajuste
   minimo en `compute_contractual_coverage` (denominador sobre
   identidades declaradas; `paired` requiere FIGI comun). ¿Se aprueba
   este ajuste o se prefiere mantener `coverage.py` intacto y
   envolverlo con una capa `catalog_key -> figi`?

4. **F2 - Test de inmutabilidad.** Se propone test de integridad por
   hash historico + corrupcion simulada + validacion objeto publicado
   != objeto de trabajo. ¿Se aprueba el diseno?

5. **F3 - Criterio B3 corregido.** `knowledge_date == filing_date`
   por contrato; `period_end != filing_date` cuando difieran;
   `knowledge_date >= filing_date`. ¿Se aprueba?

6. **Amendments.** `knowledge_date` = fecha del filing efectivo
   (RESTATEMENT aporta el estado) -> no la del original. ¿Se aprueba?

7. **PositionRecord - invariantes.** `period <-> period_end` con
   verificacion; distincion legacy vs contractual via `is_contractual_b3`.
   ¿Se aprueba?

8. **`absence.py` stub.** Enums + NotImplementedError. ¿Se mantiene
   aprobado (v2 ya lo estaba)?

9. **Orden de commits.** B2 -> B1 -> B3 -> integracion. ¿Se mantiene
   aprobado (v2 ya lo estaba)?

10. **Criterio de cierre.** Los 3 bloques de la seccion 7 + global.

---

## 9. Lo que NO se toca en A.6.2-bis

- Contratos: `NIPC_CONTRATOS_SEMANTICOS_v1.md`, `NIPC_COVERAGE_POLICY.md` v1.0.
- Modulos: `delta_shares.py`, `security_identity.py`, `relationships.py`,
  `amendments.py`, `temporal_validity.py`.
- `coverage.py`: se ajusta minimamente si se aprueba pregunta 3; si no,
  se envuelve sin modificar.
- `radar_target_catalog.csv` actual: se conserva como snapshot inicial.
- OpenFIGI masivo: NO.
- `DROP_DUP`: NO.
- Push a `origin/main`: NO.
- Certificacion "acumulacion": NO.

---

## 10. Trazabilidad

    Dictamen #43                 A.6.0 CERRADO + A.6.2-bis AUTORIZADO
    Dictamen #44                 v1 NO-GO; 4 correcciones
    Dictamen #45                 v2 GO CONDICIONAL; F1/F2/F3 + amendments
    A.6.0 inventario             INFORME.md seccion 21
    F2.4 (dictamen #24)          3 bloqueantes estructurales
    RECONCILIACION D1/D2/D3      divergencias contrato <-> codigo
    Contratos P38/P62/P63/P64    secciones 3, 11, 12, 13

---

Fin de la propuesta v3. Sometida a verificacion documental.
HEAD 9aa0727.