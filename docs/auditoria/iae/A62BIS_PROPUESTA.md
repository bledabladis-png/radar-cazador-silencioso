# IAE - A.6.2-bis Propuesta de rediseno arquitectonico TARGET (v5)

**Version:** v5. Aplicados los 2 bloqueos materiales del dictamen #47
(A1 catalog_key estable + A2 denominador independiente del mapping) +
test reforzado NEW HOLDINGS ambiguous.

**Versiones previas:** v1 (f383932, NO-GO #44), v2 (9aa0727, GO COND
#45), v3 (fa03a97, NO-GO #46), v4 (30ed3df, NO-GO #47).

**Objeto:** propuesta de diseno para materializar los 3 bloqueantes
estructurales F2.4 (B1 + B2 + B3).

**Fecha:** 2026-09-21.
**HEAD al redactar:** 30ed3df.
**Naturaleza:** propuesta. NO normativa. Sometida a dictamen.

---

## 0. Resumen ejecutivo

v5 cierra los 2 bloqueos materiales del dictamen #47 + test reforzado:

**A1 (catalog_key estable):** `catalog_key` es un identificador inmutable
asignado en el alta, NO derivado del ticker. Formato:
`<YYYYMMDD_alta>_<NNNN>` (secuencial administrativo). El `radar_ticker`
pasa a ser un ATRIBUTO versionado del catalogo. Migracion inicial
asigna keys a las 242 filas actuales.

**A2 (denominador independiente del mapping):** eleccion del
**Camino B (fail-closed)** del dictamen #47 seccion 4.
`TARGET_PAIRWISE` es el universo contractual del catalogo
(independiente del mapping). El peso `w(s)` viene del 13F y requiere
mapping. Si **cualquier** entrada de `TARGET_PAIRWISE` tiene `w(s) = 0`
(mapping fallido, sin SSHPRNAMT), la metrica pasa a `UNAVAILABLE`.
PROHIBIDO recalcular con denominador reducido.

**Test reforzado NEW HOLDINGS ambiguous:** original `security X` +
NEW HOLDINGS `security X` sin evidencia de si es nueva/correccion/
duplicado -> `knowledge_date = N/D`. Sin heuristica.

**Fuera de alcance:** OpenFIGI masivo, recalculo de evidencia final,
modificacion de contratos, `DROP_DUP`, certificacion "acumulacion",
Policy v1.3, Gate-NIPC.2/3.

---

## 1. Estado de partida

| Bloqueante | Piezas existentes | Piezas ausentes |
|---|---|---|
| B1 | `coverage.py::compute_contractual_coverage`, `PositionRecord`, `aggregate_positions_by_shareclass_figi`, `target_universe.py::resolve_cusips` | `target_builder.py`, `catalog_key` estable, integracion `nipc.py` |
| B2 | `source_date` (no contractual) | versionado + `target_catalog_as_of` |
| B3 | Doctrina P63/P64/P65 | `knowledge_date` por posicion, `absence.py` |

**Gate 0 sobre `radar_target_catalog.csv` (2026-09-21):**

- 242 filas. `radar_ticker` 242 unicos, 0 nulos.
- `share_class_figi`: 240 no nulos unicos, 0 duplicados, 2 MISS (BRK-B, MOG-A).
- Status: 240 OK + 2 MISS.

**Migracion inicial v5:** las 242 filas actuales reciben
`catalog_key = "radar_20260919_<NNNN>"` (asignacion administrativa
secuencial). El atributo `radar_ticker` se conserva como columna
mutable.
---

## 2. B1 - TARGET independiente del mapping (A1+A2 resueltos)

### 2.1. A1 - `catalog_key` estable (no derivado del ticker)

**Identidad del catalogo:** inmutable, asignada en el alta, sin
dependencia del ticker, `source_date`, `share_class_figi` ni OpenFIGI.

    catalog_key := "radar_<YYYYMMDD_alta>_<NNNN>"

Ejemplos (migracion inicial del catalogo actual):

    radar_20260919_0001   (AAPL)
    radar_20260919_0002   (ABBV)
    ...
    radar_20260919_0242   (ultima entrada)

**Propiedades:**

- Asignado una sola vez cuando la entrada entra al radar.
- Inmutable: no cambia aunque cambie el `radar_ticker`.
- Secuencial dentro del dia de alta; dos altas el mismo dia -> `_0001`, `_0002`.
- El prefijo `radar_` identifica la fuente del catalogo.

**Columna `radar_ticker`:** pasa a ser **atributo versionado**.
- Puede cambiar entre snapshots (ej. FB -> META).
- El `catalog_key` no cambia.
- El invariante `radar_ticker NOT NULL` se verifica por snapshot.
- El invariante `radar_ticker UNICO` se verifica por snapshot.

**El invariante `catalog_key UNICO` se verifica globalmente** (a lo
largo de snapshots), no solo por snapshot.

**Test explicito A1:**

    Ticker cambio FB -> META:
      snapshot_2026_09_19.csv:  radar_20240101_0050 | FB  | <figi_fb>
      snapshot_2027_01_15.csv:  radar_20240101_0050 | META| <figi_meta>

    assert catalog_key identico en ambos snapshots.
    assert radar_ticker cambia (informacion de provenance).

### 2.2. A2 - Denominador independiente del mapping (Camino B)

**Eleccion: Camino B (fail-closed) del dictamen #47 seccion 4.**

**Regla contractual:**

    TARGET_PAIRWISE := entradas del catalogo cuyo periodo de vigencia
                        cubre Q4 y Q1
                        (independiente del mapping CUSIP_13F -> FIGI)

    w(s) := max( SSHPRNAMT_Q4(s), SSHPRNAMT_Q1(s) )
            si ambos disponibles.
            max(SSHPRNAMT disponible, NaN) si solo uno.
            0 si ninguno (mapping fallido en ambos periodos).

    paired_weighted_share_coverage:
        SI TODAS las entradas de TARGET_PAIRWISE tienen w(s) > 0:
            numer / denom con denom = sum(w(s) para s in TARGET_PAIRWISE)
        SINO:
            UNAVAILABLE

**Invariante:**

    mapping OK  -> todas las entradas tienen w(s) > 0 -> metrica VALID
    mapping FAIL parcial -> alguna entrada con w(s) = 0 -> UNAVAILABLE
    mapping FAIL total   -> UNAVAILABLE

**PROHIBIDO:** recalcular el denominador con menos entradas cuando el
mapping falla. Eso es el sesgo de seleccion que F2.4 exigio eliminar.

**Razon:** la metrica mide la calidad de la identidad sobre el
universo contractual completo. Si el universo no puede evaluarse
completo (falta peso), no se mide; se declara `UNAVAILABLE`. Fail-closed.

**Test explicito A2:**

    Catalogo: {A, B, C} con FIGIs.
    Mapping:
      A -> FIGI_A  +  SSHPRNAMT_A
      B -> FIGI_B  +  SSHPRNAMT_B
      C -> FAIL    +  sin SSHPRNAMT

    assert TARGET_PAIRWISE == {A, B, C}  # cardinalidad intacta
    assert w(C) == 0
    assert paired_weighted_share_coverage == "UNAVAILABLE"

    # Contra-caso: mapping OK para A, B, C
    assert paired_weighted_share_coverage != "UNAVAILABLE"
    assert cardinalidad TARGET_PAIRWISE == 3 en ambos casos

### 2.3. Cuatro conceptos separados

    catalog_key                identidad administrativa inmutable
    radar_ticker               atributo versionado (puede cambiar)
    share_class_figi           unidad economica P38 (Modelo A / Q12)
    TARGET_PAIRWISE            universo contractual del catalogo
    TARGET_OBSERVED            subset con SSHPRNAMT obtenido via mapping

**Regla dura:**

    cardinalidad TARGET_PAIRWISE != denominador ponderado
    fallo de mapping NO modifica cardinalidad TARGET_PAIRWISE
    fallo de mapping produce UNAVAILABLE (no denom reducido)

### 2.4. Unidad economica P38 (sin cambios)

Se mantiene la decision Q12 = Modelo A:

    unidad economica = share_class_figi (share class)

`coverage.py::compute_contractual_coverage` opera por `share_class_figi`.
NO se modifica la semantica del contrato P38.

### 2.5. Arquitectura

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
        period_end: str
        catalog_version_id: str
        catalog_sha256: str
        declared_keys: frozenset[str]         # catalog_key
        ticker_by_key: dict[str, str]         # key -> radar_ticker actual
        figi_by_key: dict[str, str | None]    # key -> figi o None
        unresolved_keys: frozenset[str]       # subset sin figi

**Invariantes:**

    declared_keys NO depende del mapping.
    declared_keys NO depende del 13F observado.
    declared_keys NO depende de OpenFIGI.
    cardinalidad declared_keys == |snapshot|.

### 2.6. `identity/target_builder.py`

    @dataclass(frozen=True)
    class TargetUniverse:
        period_end: str
        catalog_version_id: str
        catalog_sha256: str
        declared_keys: frozenset[str]
        ticker_by_key: dict[str, str]
        figi_by_key: dict[str, str | None]
        unresolved_keys: frozenset[str]

    def build_target(snapshot_df, *, period_end,
                     catalog_version_id, catalog_sha256) -> TargetUniverse:
        """Materializa el TargetUniverse a partir del snapshot.

        Validaciones obligatorias (fail-closed):
          - catalog_key NOT NULL en todas las filas.
          - catalog_key UNICO global (entre snapshots).
          - catalog_key UNICO por snapshot.
          - radar_ticker NOT NULL por snapshot.
          - radar_ticker UNICO por snapshot.

        Funcion pura. NO consulta catalogo ni sistema de ficheros.
        """

**Violacion -> `raise CatalogKeyInvariantViolated`.**

### 2.7. `coverage.py` - ajuste minimo

`compute_contractual_coverage` recibe:
- `target_q4` / `target_q1`: sets de `catalog_key` (universo contractual).
- `weights_q4` / `weights_q1`: dicts `{catalog_key -> sshprnamt}`,
  derivados del 13F via mapping.

**Cambio minimo en la formula P38:** el denominador se calcula sobre
`TARGET_PAIRWISE` (catalog_key). Si alguna entrada de `TARGET_PAIRWISE`
no tiene peso en ningun periodo, `coverage_status = "UNAVAILABLE"`.

El numerador `PAIRED` sigue requiriendo `share_class_figi` comun
(unidad economica), pero **el universo del denominador no se reduce
por mapping fallido**.

**NO se modifica la semantica de P38 §3.3** (formula del cociente); se
modifica el dominio sobre el que se evalua la validez (fail-closed
si falta peso).

### 2.8. `target_universe.resolve_cusips` (sin cambios)

Sigue siendo el resolver inverso. El `scf_index` se construye sobre
`figi_by_key`.

### 2.9. `compute_coverage_pairwise` (wrapper)

**Gate 0 obligatorio previo al refactor.**

Contrato de retorno:

    dict con las 7 claves de compute_contractual_coverage.
    Numericos: float | None.
    coverage_status: "VALID" | "UNAVAILABLE".
    Sin TARGET -> "UNAVAILABLE".

### 2.10. Tests B1 (resumen)

- A1: ticker cambio -> catalog_key identico.
- A1: `catalog_key NOT NULL`, `catalog_key UNICO` (snapshot + global).
- A2: `mapping OK` vs `mapping FAIL` -> cardinalidad TARGET_PAIRWISE identica.
- A2: mapping FAIL parcial -> `UNAVAILABLE` (no denom reducido).
- A2: mapping OK total -> metrica VALID, cardinalidad == |catalogo|.
- `declared_keys == |snapshot|`.
- `compute_coverage_pairwise` sin TARGET -> dict tipado UNAVAILABLE.
---

## 3. B2 - Point-in-time (aprobado #47, sin cambios)

### 3.1. Estructura de disco

    data/mappings/catalog_snapshots/
        snapshot_<version_id>.csv
        snapshot_<version_id>.sha256
    data/mappings/catalog_manifest.json

`version_id = <YYYYMMDD>_<NN>`. No autorreferencial.

### 3.2. Inmutabilidad real

- `.csv` y `.sha256`: inmutables in-place.
- `catalog_manifest.json`: mutable (indice).

**Regla:** ninguna evidencia historica publicada se modifica in-place.

### 3.3. Intervalos semiabiertos

`[valid_from, valid_to)`. `null` = vigente.

`target_catalog_as_of(period_end, *, catalog_root)`:

    0 -> raise CatalogNotAvailable
    1 -> devuelve (df, version_id, sha256)
    >1 -> raise CatalogAmbiguous

### 3.4. Backdating prohibido

`valid_from = 2026-09-19` para el snapshot inicial. Q4 2025 / Q1 2026
-> `CatalogNotAvailable`.

### 3.5. Test de integridad

    sha256 recalculado == sha256 publicado (.sha256)
    + coherencia con catalog_manifest.json
    mismatch -> FAIL-CLOSED

Corrupcion simulada: byte alterado -> deteccion.

### 3.6. Tests B2

- 0/1/>1 snapshots -> fail-closed/OK/ambiguous.
- `as_of("2025-12-31")` -> `CatalogNotAvailable`.
- Integridad: sha256 recalculado == publicado.
- Corrupcion: byte alterado -> fail-closed.
- Intervalos `[from, to)` sin solapamiento.

---

## 4. B3 - Semantica temporal (test NEW HOLDINGS reforzado)

### 4.1. Tres timestamps

    period_end       cierre del trimestre
    filing_date      fecha del filing del filing_manager
    knowledge_date   fecha de publicacion de la observacion efectiva

Contrato: `knowledge_date == filing_date`.

### 4.2. RESTATEMENT

Estado sustituido -> fecha del restatement. Coherente con P64
(`RESTATEMENT -> REPLACE`).

### 4.3. NEW HOLDINGS - caso no ambiguo

    original    2026-01-30   Holdings A (security X)
    amendment   2026-02-15   NEW HOLDINGS: security Y

Estado efectivo: X + Y.
  X (heredada) -> knowledge_date = 2026-01-30.
  Y (nueva)    -> knowledge_date = 2026-02-15.

### 4.4. NEW HOLDINGS - caso ambiguous (test reforzado #47 seccion 7)

    original    2026-01-30   Holdings A (security X)
    amendment   2026-02-15   NEW HOLDINGS: security X

Sin evidencia adicional, X del amendment puede ser:
  - nueva entry (reaparicion),
  - correccion de la entry original,
  - duplicidad documental.

**Regla dura:** si la atribucion documental no es inequivoca ->
`knowledge_date = N/D` para la entry afectada.

NO heuristica. NO asignar la fecha del amendment por defecto.

**Test explicito:**

    def test_new_holdings_ambiguous_knowledge_date_nd():
        # original + amendment con la misma security
        # sin metadata adicional (lineage) que desambigue
        recs = compute_effective_snapshot(...)
        affected = [r for r in recs if r.observado_security_key == "X"]
        # Ninguna con knowledge_date asignable
        assert all(r.knowledge_date is None for r in affected)

### 4.5. Regla general

    knowledge_date = filing_date del filing que REALMENTE aporta el
                     estado observado al snapshot efectivo, por
                     posicion.

    RESTATEMENT: el estado sustituido lleva fecha del restatement.
    NEW HOLDINGS:
        Posiciones heredadas inequivocas: fecha del filing original.
        Posiciones nuevas inequivocas:    fecha del amendment.
        Atribucion ambigua:               N/D.
    Sin reconstruccion documental:        N/D.

### 4.6. Provenance por posicion

`PositionRecord.provenance` incluye:

    effective_filing_accession   accession del filing que aporta la linea
    effective_filing_date        FILING_DATE de ese filing
    effective_amendment_type     RESTATEMENT | NEW_HOLDINGS | None
    ambiguity_flag               bool (True si atribucion ambigua)

`knowledge_date` se deriva de `effective_filing_date`. Nunca se
disocia del origen documental.

### 4.7. `PositionRecord` extendido

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

**Distincion legacy vs contractual:**

    def is_contractual_b3(rec) -> bool:
        return (
            rec.period is not None
            and rec.period_end is not None
            and rec.filing_date is not None
            and rec.knowledge_date is not None
            and rec.knowledge_date == rec.filing_date
            and "effective_filing_accession" in rec.provenance
            and not rec.provenance.get("ambiguity_flag", False)
        )

### 4.8. `absence.py` stub

Enums + `NotImplementedError`. `P63 absence classifier = DEFERRED`.

### 4.9. Regla dura preservada

`delta_shares` NO crea `SOLD`. `P64_EVENTS_DEFERRED` se mantiene.

### 4.10. Tests B3

- `PositionRecord` con 3 timestamps -> OK.
- Sin timestamps -> legacy.
- Invariante `period <-> period_end`.
- `knowledge_date == filing_date`.
- RESTATEMENT: fecha del restatement.
- NEW HOLDINGS no ambiguo: heredada -> original; nueva -> amendment.
- **NEW HOLDINGS ambiguous -> `knowledge_date = N/D`** (test reforzado).
- `is_contractual_b3` rechaza registros con `ambiguity_flag=True`.
- `absence.py::classify_absence` -> `NotImplementedError`.
- `delta_shares` no produce SOLD (test P63).
---

## 5. Orden de commits (sin cambios)

    1. B2  Modelo de versionado + target_catalog_as_of
    2. B1  target_builder + TargetUniverse + catalog_key
    3. B1  Integracion coverage (gate 0 consumidores primero)
    4. B3  Timestamps + PositionRecord + provenance + absence.py
    5. Integracion end-to-end + verificacion global

Razon: B1 depende de B2. B3 depende de B1.

---

## 6. Tests requeridos (resumen)

| Bloque | Fichero | Cobertura |
|---|---|---|
| B1 | `tests/test_target_builder.py` (nuevo) | catalog_key estable, ticker versionado, declared_keys, cardinalidad intacta bajo mapping fail |
| B1 | `tests/test_sec_13f_nipc.py` (extender) | wrapper con/sin TARGET, UNAVAILABLE bajo mapping fail |
| B2 | `tests/test_target_catalog_as_of.py` (nuevo) | snapshots, fail-closed, integridad, corrupcion |
| B3 | `tests/test_position_record.py` (nuevo) | RESTATEMENT, NEW HOLDINGS no ambiguo, NEW HOLDINGS ambiguo -> N/D |
| B3 | `tests/test_absence.py` (nuevo) | enums + NotImplementedError |

**Cero regresion P65/P66 esperada.**

---

## 7. Criterio de cierre A.6.2-bis

### B1

**A1 - catalog_key:**
- Inmutable y no derivado del ticker (test: cambio de ticker -> key identico).
- `catalog_key NOT NULL` + `UNICO` por snapshot + `UNICO` global.
- Violacion -> `raise CatalogKeyInvariantViolated`.

**A2 - denominador:**
- `TARGET_PAIRWISE` cardinalidad == |catalogo contractual|.
- `mapping OK` vs `mapping FAIL` -> **misma cardinalidad** TARGET_PAIRWISE.
- mapping FAIL -> `paired_weighted_share_coverage = UNAVAILABLE`.
- NUNCA recalcular con denominador reducido.

**General:**
- Unidad economica P38 (`share_class_figi`) intacta.
- `compute_coverage_pairwise` sin TARGET -> dict tipado UNAVAILABLE.

### B2

- 0/1/>1 snapshots -> fail-closed/OK/ambiguous.
- Hash invalido -> fail-closed.
- No backdating.
- Integridad verificable (sha256 publicado vs recalculado).
- Corrupcion detectada.

### B3

- `period_end != filing_date` cuando difieran.
- `knowledge_date == filing_date`.
- RESTATEMENT -> fecha del restatement.
- NEW HOLDINGS no ambiguo -> fechas por posicion.
- **NEW HOLDINGS ambiguo -> `knowledge_date = N/D`** (test reforzado).
- `provenance.effective_filing_accession` presente.
- `is_contractual_b3` rechaza `ambiguity_flag=True`.

### Global

- P65 PASS + P66 PASS + A.6.2-bis PASS.
- `compileall` OK + `pyflakes` LIMPIO.
- Sin regresion nueva.

Solo entonces: **A.6.2-bis CLOSED -> A.6.3 AUTHORIZED.**
---

## 8. Preguntas al auditor (v5)

1. **A1 - Formato `catalog_key`.** Se propone
   `radar_<YYYYMMDD_alta>_<NNNN>` (secuencial). ¿Se aprueba? ¿ULID?
   ¿Hash de la primera observacion?

2. **A1 - Migracion inicial.** Las 242 filas actuales reciben
   `radar_20260919_<NNNN>` (secuencial administrativo). ¿Se aprueba?

3. **A1 - `radar_ticker` como atributo versionado.** Puede cambiar
   entre snapshots sin alterar `catalog_key`. ¿Se aprueba?

4. **A2 - Camino B (fail-closed).** Se ha elegido Camino B: si alguna
   entrada de `TARGET_PAIRWISE` tiene `w(s) = 0`, la metrica pasa a
   `UNAVAILABLE`. NO se recalcula con denominador reducido. ¿Se aprueba?

5. **A2 - Universo de `TARGET_PAIRWISE`.** Definido por el catalogo
   (independiente del mapping). Cardinalidad = |catalogo contractual|.
   ¿Se aprueba?

6. **A2 - Ajuste `coverage.py`.** Formula P38 §3.3 intacta. Se anade
   regla de validez: si falta peso en `TARGET_PAIRWISE` -> `UNAVAILABLE`.
   ¿Se aprueba?

7. **B2 - Layout snapshots + manifest.** Sin cambios respecto v4
   (aprobado #46/#47). ¿Se mantiene?

8. **B3 - RESTATEMENT.** Fecha del filing que sustituye. ¿Se mantiene
   (aprobado #47)?

9. **B3 - NEW HOLDINGS ambiguous.** `knowledge_date = N/D` (test
   reforzado #47 seccion 7). ¿Se aprueba?

10. **Criterio de cierre.** Los 3 bloques de la seccion 7 + global.

---

## 9. Lo que NO se toca en A.6.2-bis

- Contratos: `NIPC_CONTRATOS_SEMANTICOS_v1.md`, `NIPC_COVERAGE_POLICY.md` v1.0.
- Modelo economico P38 (`share_class_figi` como unidad, Q12 Modelo A).
- Modulos: `delta_shares.py`, `security_identity.py`, `relationships.py`,
  `amendments.py`, `temporal_validity.py`.
- `coverage.py`: ajuste minimo (regla de validez fail-closed); formula
  P38 §3.3 intacta.
- `radar_target_catalog.csv` actual: snapshot inicial + migracion
  administrativa de `catalog_key`.
- OpenFIGI masivo: NO.
- `DROP_DUP`: NO.
- Push a `origin/main`: NO.
- Certificacion "acumulacion": NO.

---

## 10. Trazabilidad

    Dictamen #43                 A.6.0 CERRADO + A.6.2-bis AUTORIZADO
    Dictamen #44                 v1 NO-GO; 4 correcciones
    Dictamen #45                 v2 GO COND; F1/F2/F3 + amendments
    Dictamen #46                 v3 NO-GO; bloqueos A (B1) + B (B3)
    Dictamen #47                 v4 NO-GO; A1 catalog_key + A2 denominador
                                 + test NEW HOLDINGS ambiguous
                                 + correccion DICTAMENES.md §13
    Gate 0 catalogo              v5 seccion 1 (242 filas, unicidad)
    F2.4 (dictamen #24)          3 bloqueantes estructurales
    P38 (contrato seccion 3)     unidad = share_class_figi
    P64 (contrato seccion 13)    RESTATEMENT / NEW HOLDINGS
    P65 (contrato seccion 14)    L3 booleano

---

Fin de la propuesta v5. Sometida a verificacion documental.
HEAD 30ed3df.