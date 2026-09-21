# IAE - A.6.2-bis Propuesta de rediseno arquitectonico TARGET (v4)

**Version:** v4. Aplicados los 2 bloqueos materiales del dictamen #46:
A (B1 unidad contractual del TARGET) + B (B3 knowledge_date por
observacion, no por snapshot).

**Versiones previas:** v1 (f383932, NO-GO #44), v2 (9aa0727, GO COND
#45), v3 (fa03a97, NO-GO #46). Backups locales como .vN.bak hasta
verificacion.

**Objeto:** propuesta de diseno para materializar los 3 bloqueantes
estructurales F2.4 (B1 + B2 + B3).

**Fecha:** 2026-09-21.
**HEAD al redactar:** fa03a97.
**Naturaleza:** propuesta. NO normativa. Sometida a dictamen.

---

## 0. Resumen ejecutivo

v4 aplica los 2 bloqueos materiales del dictamen #46:

**A (B1, unidad contractual del TARGET):** se introduce `catalog_key`
(identidad administrativa del catalogo) separado de `share_class_figi`
(unidad economica P38). Se separan cardinalidad de universo
(`len(catalog_key)`) y denominador ponderado de
`paired_weighted_share_coverage` (suma de pesos sobre
`TARGET_PAIRWISE`, segun P38 §3.3). La unidad economica del contrato
P38 (`share_class_figi`, Modelo A / Q12) NO se modifica.

**B (B3, knowledge_date con NEW HOLDINGS):** `knowledge_date` es
propiedad de la observacion efectiva / posicion, no del snapshot
consolidado. RESTATEMENT sustituye; NEW HOLDINGS suplementa. Posiciones
heredadas conservan su fecha original; nuevas entradas reciben la
fecha del amendment. Si no puede reconstruirse la atribucion
documental -> `knowledge_date = N/D`.

**Bloqueo previo resuelto (F3 #45):** criterio corregido:
`knowledge_date == filing_date` por contrato;
`period_end != filing_date` cuando difieran.

**Fuera de alcance:** OpenFIGI masivo, recalculo de evidencia final,
modificacion de contratos, `DROP_DUP`, certificacion "acumulacion",
Policy v1.3, Gate-NIPC.2/3.

---

## 1. Estado de partida

| Bloqueante | Piezas existentes | Piezas ausentes |
|---|---|---|
| B1 | `coverage.py::compute_contractual_coverage`, `PositionRecord`, `aggregate_positions_by_shareclass_figi`, `target_universe.py::resolve_cusips` | `target_builder.py`, `catalog_key`, integracion `nipc.py` |
| B2 | `source_date` (no contractual) | versionado + `target_catalog_as_of` |
| B3 | Doctrina P63/P64/P65 | `knowledge_date` por posicion, `absence.py` |

**Gate 0 sobre `radar_target_catalog.csv` (2026-09-21, empírico):**

- 242 filas totales.
- `radar_ticker`: 0 nulos, 0 vacios, 242 unicos, 0 duplicados.
- `share_class_figi`: 2 nulos (BRK-B, MOG-A), 240 unicos (no nulos), 0 duplicados.
- `figi`: 240 unicos (no nulos).
- Status: 240 OK + 2 MISS.

Propiedades demostradas para el catalogo actual:

    radar_ticker         NOT NULL, UNICO en el snapshot
    share_class_figi     unico cuando presente (240/240)
    figi                 unico cuando presente (240/240)
---

## 2. B1 - TARGET independiente del mapping (bloqueo A resuelto)

**Correccion A aplicada:** reconciliacion con P38.

### 2.1. Cuatro conceptos separados

    catalog_key                identidad administrativa del catalogo
    share_class_figi           unidad economica P38 (Modelo A / Q12)
    TARGET declarado           conjunto de catalog_key del snapshot
    TARGET_PAIRWISE (P38)      interseccion de share_class_figi Q4 & Q1

**Regla dura:**

    catalog_key != share_class_figi
    cardinalidad TARGET != denominador ponderado del indicador

### 2.2. `catalog_key`

    catalog_key := "radar:" + radar_ticker

- Prefijo `radar:` evita colision con otras fuentes.
- Estable entre snapshots: `radar:AAPL` siempre se refiere a la
  entrada del radar para AAPL, independientemente del FIGI resuelto.
- `radar_ticker` cumple NOT NULL + UNICO (evidencia Gate 0 sobre el
  snapshot actual: 242/242).
- El invariante NOT NULL + UNICO es **verificado en cada build**.
  Si un snapshot futuro violara el invariante -> error (no colapso
  silencioso de set()).

### 2.3. Unidad economica P38

Se mantiene la decision Q12 = Modelo A:

    unidad economica = share_class_figi (share class)

`coverage.py::compute_contractual_coverage` opera por `share_class_figi`.
NO se modifica la semantica del contrato P38.

### 2.4. Separacion cardinalidad vs denominador

    Cardinalidad del TARGET declarado
      = |{ catalog_key }|          -> 242 en el catalogo actual

    Denominador de paired_weighted_share_coverage (P38 seccion 3.3)
      = sum( w(s) for s in TARGET_PAIRWISE )
      con:
        TARGET_PAIRWISE  = TARGET_Q4 INTERSECT TARGET_Q1
                           por share_class_figi
        w(s)             = max(SSHPRNAMT_Q4(s), SSHPRNAMT_Q1(s))
                           una vez por security

Son magnitudes distintas. `len(declared)` NO es el denominador del
indicador ponderado.

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
        declared_keys: frozenset[str]       # catalog_key (242)
        figi_by_key: dict[str, str | None]  # catalog_key -> figi
        unresolved_keys: frozenset[str]     # catalog_key sin figi

**Invariante estructural:**

    TARGET declarado (declared_keys) NO depende del mapping.
    TARGET declarado NO depende del 13F observado.
    TARGET declarado NO depende de OpenFIGI.

Un fallo de mapping produce ausencia en el pairing ponderado, nunca
en `declared_keys`.

### 2.6. `identity/target_builder.py`

    @dataclass(frozen=True)
    class TargetUniverse:
        period_end: str
        catalog_version_id: str
        catalog_sha256: str
        declared_keys: frozenset[str]       # 242 para el catalogo actual
        figi_by_key: dict[str, str | None]  # key -> figi o None
        unresolved_keys: frozenset[str]     # subset sin figi

    def build_target(snapshot_df, *, period_end,
                     catalog_version_id, catalog_sha256) -> TargetUniverse:
        """Materializa el TargetUniverse a partir del snapshot.

        Validaciones obligatorias (fail-closed):
          - radar_ticker NOT NULL en todas las filas.
          - radar_ticker UNICO: len(set) == len(df).
          - ningun catalog_key colisiona.

        Funcion pura. NO consulta catalogo ni sistema de ficheros.
        """

**Si el snapshot violara NOT NULL o UNICO:**
`raise CatalogKeyInvariantViolated`.

### 2.7. Relacion `catalog_key` <-> `share_class_figi`

    | TARGET declarado  |            242 (catalog_key)
    | TARGET resoluble  |            240 (share_class_figi)
    | TARGET_UNRESOLVED |              2 (BRK-B, MOG-A)
    | TARGET_PAIRWISE   |  variable (interseccion Q4 ∩ Q1 por figi)

`TARGET_UNRESOLVED` no entra al pairing ponderado (no tiene unidad
economica). Pero si esta en la cardinalidad declarada del catalogo.

### 2.8. `coverage.py` - ajuste minimo autorizado

`compute_contractual_coverage` se ajusta para:
- Recibir `target_q4` / `target_q1` como sets de `share_class_figi`.
- El caller construye esos sets desde `TargetUniverse` filtrando
  `unresolved_keys` (sin FIGI, no aportan unidad economica).

**NO se modifica la formula contractual.** El denominador ponderado
sigue siendo `sum(w(s))` sobre `TARGET_PAIRWISE` (interseccion de
FIGIs).

### 2.9. `target_universe.resolve_cusips` (sin cambios)

Sigue siendo el resolver inverso (CUSIP observado -> membership). El
`scf_index` se construye sobre `figi_by_key`.

### 2.10. `compute_coverage_pairwise` (wrapper)

**Gate 0 obligatorio previo al refactor:**

    grep -rn "compute_coverage_pairwise" tests/ src/ scripts/

Contrato de retorno declarado:

    dict con las 7 claves de compute_contractual_coverage.
    Numericos: float | None.
    coverage_status: "VALID" | "UNAVAILABLE".
    Sin TARGET -> "UNAVAILABLE", sin proxy silencioso.

### 2.11. Tests B1

- `declared_keys = {radar:ticker}` con 242 filas -> 242 keys.
- `unresolved_keys = {radar:BRK-B, radar:MOG-A}` -> 2 keys.
- `figi_by_key` con 240 valores no-None + 2 None.
- Invariante NOT NULL / UNICO: snapshot sintetico con ticker duplicado -> raise.
- `mapping OK` vs `mapping fallido` -> mismo `declared_keys` (test explicito).
- `TARGET_PAIRWISE` (por figi) != cardinalidad declarada (test explicito).
- `compute_coverage_pairwise` sin TARGET -> dict con `coverage_status == "UNAVAILABLE"`.
---

## 3. B2 - Point-in-time (aprobado #46, precisiones aplicadas)

Arquitectura aprobada por #46. Precisiones no bloqueantes incorporadas.

### 3.1. Estructura de disco (sin cambios respecto v3)

    data/mappings/catalog_snapshots/
        snapshot_<version_id>.csv
        snapshot_<version_id>.sha256
    data/mappings/catalog_manifest.json

`version_id = <YYYYMMDD>_<NN>`. No autorreferencial.

### 3.2. Inmutabilidad real

- `snapshot_*.csv` publicado: inmutable in-place.
- `snapshot_*.sha256` publicado: inmutable in-place.
- `catalog_manifest.json`: mutable (indice).

**Regla:** ninguna evidencia historica publicada se modifica in-place.
Correcciones -> snapshot nuevo.

### 3.3. Intervalos semiabiertos

`[valid_from, valid_to)`. `null` = vigente.

`target_catalog_as_of(period_end, *, catalog_root)`:

    0 snapshots -> raise CatalogNotAvailable
    1 snapshot  -> devuelve (DataFrame, version_id, sha256)
    >1 solapan  -> raise CatalogAmbiguous

### 3.4. Backdating prohibido

`valid_from = 2026-09-19` para el snapshot inicial. Q4 2025 / Q1 2026
-> `CatalogNotAvailable`. No se fabrica retrospectivamente.

### 3.5. Test de inmutabilidad (aprobado #46)

    contenido publicado -> sha256 publicado -> sha256 recalculado -> comparar
    mismatch -> FAIL-CLOSED

Test adicional: corrupcion simulada (byte alterado -> detectada).

### 3.6. Precision #46 - conservar revision del manifest en provenance

Recomendacion no bloqueante aplicada: la provenance incluye el commit
de git del repositorio (o hash del propio `catalog_manifest.json`)
para reproducibilidad completa de consultas historicas.

### 3.7. Tests B2

- 0 snapshots -> `CatalogNotAvailable`.
- 1 snapshot -> OK.
- >1 solapados -> `CatalogAmbiguous`.
- `as_of("2025-12-31")` -> `CatalogNotAvailable` (backdating prohibido).
- Integridad: sha256 recalculado == publicado.
- Corrupcion: byte alterado -> deteccion + fail-closed.
- Intervalos `[from, to)` sin solapamiento.
---

## 4. B3 - Semantica temporal (bloqueo B resuelto)

**Correccion B aplicada:** `knowledge_date` es propiedad de la
observacion efectiva / posicion, no del snapshot consolidado.

### 4.1. Tres timestamps

    period_end       cierre del trimestre
    filing_date      fecha del filing del filing_manager
    knowledge_date   fecha de publicacion de la observacion efectiva

Contrato: `knowledge_date == filing_date` de la observacion concreta.

### 4.2. RESTATEMENT

    filing original    2026-01-30
    amendment RESTATEMENT 2026-02-15 (sustituye)

Estado efectivo: el filing del amendment sustituye al original.
`knowledge_date = 2026-02-15` para todas las posiciones del snapshot
efectivo. Coherente con P64 (`RESTATEMENT -> REPLACE`).

### 4.3. NEW HOLDINGS

    filing original    2026-01-30   Holdings A
    amendment NEW HOLDINGS 2026-02-15   Holdings B (suplementa)

Estado efectivo consolidado: A + B.

**Regla corregida:**

    A (heredada del original)   -> knowledge_date = 2026-01-30
    B (nueva del amendment)     -> knowledge_date = 2026-02-15

**NO** se asigna uniformemente la fecha del amendment a A y B.
Eso introduciria look-ahead para A.

### 4.4. Regla general

    knowledge_date = filing_date del filing que REALMENTE aporta el
                     estado observado al snapshot efectivo, por
                     posicion.

    RESTATEMENT: el estado sustituido lleva la fecha del restatement.
    NEW HOLDINGS:
        posiciones heredadas: fecha del filing original.
        posiciones nuevas:    fecha del amendment.
    Sin reconstruccion documental posible: knowledge_date = N/D.

Coherente con P64 (`NEW HOLDINGS -> ADD`).

### 4.5. Provenance por posicion

La `PositionRecord` incluye en provenance:

    effective_filing_accession   accession del filing que aporta la linea
    effective_filing_date        FILING_DATE de ese filing
    effective_amendment_type     RESTATEMENT | NEW_HOLDINGS | None

`knowledge_date` se deriva de `effective_filing_date`. Nunca se
disocia del origen documental.

### 4.6. `PositionRecord` extendido

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
        # v4:
        period_end: Optional[str] = None
        filing_date: Optional[str] = None
        knowledge_date: Optional[str] = None

**Invariantes verificables:**

- Si `period` y `period_end` presentes: `period_end` = ultimo dia
  del trimestre referenciado por `period`.
- Si `filing_date` y `knowledge_date` presentes:
  `knowledge_date == filing_date`.
- `provenance["effective_filing_accession"]` identifica el filing que
  aporta la linea.

**Distincion legacy vs contractual:**

- Legacy: los 3 timestamps `None`. Retrocompat.
- Contractual B3: 3 timestamps + `effective_filing_accession` en
  provenance + invariantes.

    def is_contractual_b3(rec) -> bool:
        return (
            rec.period is not None
            and rec.period_end is not None
            and rec.filing_date is not None
            and rec.knowledge_date is not None
            and rec.knowledge_date == rec.filing_date
            and "effective_filing_accession" in rec.provenance
        )

### 4.7. `absence.py` (stub diferido, aprobado #45/#46)

Enums + `NotImplementedError`. `P63 absence classifier = DEFERRED`.

### 4.8. Regla dura preservada

`delta_shares` NO crea `SOLD`. `P64_EVENTS_DEFERRED` se mantiene.

### 4.9. Tests B3

- `PositionRecord` con 3 timestamps -> OK.
- `PositionRecord` sin timestamps -> legacy.
- Invariante `period <-> period_end`.
- `knowledge_date == filing_date` por construccion.
- **RESTATEMENT:** 1 filing original + 1 restatement -> todas las
  posiciones efectivas llevan fecha del restatement.
- **NEW HOLDINGS:** original A (enero) + amendment B (febrero) ->
  A lleva enero, B lleva febrero (test explicito).
- Sin reconstruccion -> `knowledge_date = N/D`.
- `is_contractual_b3` retorna True solo con 3 timestamps + provenance.
- `absence.py::classify_absence` -> `NotImplementedError`.
- `delta_shares` no produce SOLD (test P63 existente).
---

## 5. Orden de commits (aprobado #45/#46)

    1. B2  Modelo de versionado + target_catalog_as_of
    2. B1  target_builder + TargetUniverse + catalog_key
    3. B1  Integracion coverage (gate 0 consumidores primero)
    4. B3  Timestamps + PositionRecord extendido + provenance + absence.py
    5. Integracion end-to-end + verificacion global

Razon: B1 depende de B2 para validez temporal. B3 depende de B1.

---

## 6. Tests requeridos (resumen)

| Bloque | Fichero | Cobertura |
|---|---|---|
| B1 | `tests/test_target_builder.py` (nuevo) | catalog_key, declared/resolved/unresolved, invariante NOT NULL/UNICO, independencia del mapping, TARGET_PAIRWISE != cardinalidad |
| B1 | `tests/test_sec_13f_nipc.py` (extender) | wrapper con/sin TARGET, contrato tipado |
| B2 | `tests/test_target_catalog_as_of.py` (nuevo) | snapshots, fail-closed, ambiguedad, integridad (hash), corrupcion simulada |
| B3 | `tests/test_position_record.py` (nuevo) | 3 timestamps, invariantes, `is_contractual_b3`, RESTATEMENT vs NEW HOLDINGS |
| B3 | `tests/test_absence.py` (nuevo) | enums + NotImplementedError |

**Cero regresion P65/P66 esperada.** 31 + 31 tests existentes intactos.

---

## 7. Criterio de cierre A.6.2-bis (revisado #46 seccion 9)

### B1

- `catalog_key = "radar:<radar_ticker>"` unico y NOT NULL (test).
- Invariante verificado en cada build: si duplicado -> raise.
- Cardinalidad declarada (`len(declared_keys)`) independiente del mapping.
- Unidad economica P38 (`share_class_figi`) intacta.
- Separacion cardinalidad / denominador ponderado (test explicito).
- `mapping OK` vs `mapping fallido` -> mismo `declared_keys`.

### B2

- `target_catalog_as_of(periodo)`:
  - 0 snapshots -> `CatalogNotAvailable`.
  - 1 snapshot -> OK.
  - >1 solapados -> `CatalogAmbiguous`.
  - hash invalido -> fail-closed.
- Backdating prohibido.
- Test integridad: sha256 recalculado == publicado.
- Test corrupcion: byte alterado -> deteccion fail-closed.
- Intervalos `[valid_from, valid_to)` sin solapamiento.

### B3

- `period_end != filing_date` cuando difieran.
- `knowledge_date == filing_date` por contrato.
- `knowledge_date >= filing_date` (invariante dura).
- **RESTATEMENT:** estado sustituido lleva fecha del restatement.
- **NEW HOLDINGS:** posiciones heredadas -> fecha original;
  nuevas -> fecha del amendment. `N/D` si no reconstruible.
- `provenance` incluye `effective_filing_accession`.

### Global

- P65 tests PASS (31 actuales).
- P66 tests PASS (31 actuales).
- Nuevos tests A.6.2-bis PASS.
- `compileall` OK.
- `pyflakes` LIMPIO.
- Suite sin regresion nueva.

Solo entonces: **A.6.2-bis CLOSED -> A.6.3 AUTHORIZED.**
---

## 8. Preguntas al auditor (v4)

1. **A - `catalog_key`.** Se propone `catalog_key = "radar:<radar_ticker>"`.
   Gate 0 demuestra 242/242 unicos, 0 nulos en el catalogo actual.
   ¿Se aprueba? ¿Prefijo distinto?

2. **A - Invariante NOT NULL/UNICO.** Se valida en cada build; violacion
   -> `raise CatalogKeyInvariantViolated`. ¿Se aprueba?

3. **A - `coverage.py` ajuste minimo.** El caller construye
   `target_q4/q1` como sets de `share_class_figi` filtrando
   `unresolved_keys`. La formula P38 se mantiene intacta. ¿Se aprueba?

4. **A - `TargetUniverse`.** `declared_keys` (catalog_key) +
   `figi_by_key` (dict) + `unresolved_keys`. ¿Se aprueba este contrato?

5. **A - Cardinalidad vs denominador.** `len(declared_keys)` es
   cardinalidad; `paired_weighted_share_coverage` denominador ponderado
   sobre `TARGET_PAIRWISE` (por figi). ¿Se aprueba la separacion?

6. **B - RESTATEMENT.** Estado sustituido lleva fecha del restatement.
   ¿Se aprueba?

7. **B - NEW HOLDINGS.** Heredadas -> fecha original; nuevas ->
   fecha amendment. `N/D` si no reconstruible. ¿Se aprueba?

8. **B - `provenance.effective_filing_accession`.** Obligatorio para
   `is_contractual_b3`. ¿Se aprueba?

9. **B - `is_contractual_b3`.** Requiere 3 timestamps + provenance +
   invariantes. ¿Se aprueba?

10. **Cierre.** Los criterios de la seccion 7 + global.

---

## 9. Lo que NO se toca en A.6.2-bis

- Contratos: `NIPC_CONTRATOS_SEMANTICOS_v1.md`, `NIPC_COVERAGE_POLICY.md` v1.0.
- Modelo economico P38: `share_class_figi` como unidad (Q12 Modelo A).
- Modulos: `delta_shares.py`, `security_identity.py`, `relationships.py`,
  `amendments.py`, `temporal_validity.py`.
- `coverage.py`: se ajusta minimamente (punto 3 preguntas); la formula
  de `paired_weighted_share_coverage` NO cambia.
- `radar_target_catalog.csv` actual: snapshot inicial.
- OpenFIGI masivo: NO.
- `DROP_DUP`: NO.
- Push a `origin/main`: NO.
- Certificacion "acumulacion": NO.

---

## 10. Trazabilidad

    Dictamen #43                 A.6.0 CERRADO + A.6.2-bis AUTORIZADO
    Dictamen #44                 v1 NO-GO; 4 correcciones
    Dictamen #45                 v2 GO COND; F1/F2/F3 + amendments
    Dictamen #46                 v3 NO-GO; bloqueos A + B
    A.6.0 inventario             INFORME.md seccion 21
    Gate 0 catalogo              este documento seccion 1 (242 filas, unicidad)
    F2.4 (dictamen #24)          3 bloqueantes estructurales
    P38 (contrato seccion 3)     unidad = share_class_figi
    P64 (contrato seccion 13)    RESTATEMENT -> REPLACE; NEW HOLDINGS -> ADD
    P65 (contrato seccion 14)    L3 booleano

---

Fin de la propuesta v4. Sometida a verificacion documental.
HEAD fa03a97.