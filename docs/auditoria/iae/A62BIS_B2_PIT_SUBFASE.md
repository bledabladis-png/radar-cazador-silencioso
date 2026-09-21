# IAE - A.6.2-bis-B2-PIT Subfase implementable

**Objeto:** documento dedicado de la subfase B2-PIT, autorizada a
implementacion aislada por dictamen estrategico #52 (2026-09-21).

**Origen:** dictamen #52, seccion 17: "El siguiente documento de B2
debera tratarse como una subfase implementable independiente, no como
una nueva v10 de todo A.6.2-bis."

**Fecha:** 2026-09-21.
**HEAD al redactar:** 5f4fa23.
**Naturaleza:** documento de subfase. Autoriza implementacion de
B2-PIT bajo sus estrictos limites.

---

## 0. Resumen ejecutivo

B2-PIT queda desacoplado de B1 y B3.

    B2-PIT = infraestructura temporal del catalogo
             ("que catalogo era valido en una fecha")

**Alcance:**

    snapshot materializado
    catalog_manifest.json
    version_id
    sha256 externo
    valid_from / valid_to
    target_catalog_as_of(period_end)
    integridad + corrupcion + ambiguedad
    no backdating

**NO incluye:**

    catalog_key
    catalog_key_assignment_unique
    resolucion economica FIGI
    continuidad catalog_key -> FIGI
    TARGET_PAIRWISE
    adaptador P38
    colision catalog_key -> FIGI
    catalog_validator de asignacion

Todo eso pertenece a B1 (bloqueado por #51) o B3 (aprobado
condicionalmente, no autorizado a implementar).

**Autorizacion:** implementacion AUTHORIZED. Cierre operativo PENDING
TESTS Y EVIDENCIA.

**Prohibido:** P38, coverage.py, OpenFIGI masivo, DROP_DUP, recálculo
final, certificación, Gate-NIPC.2/3, push.

---

## 1. Frontera arquitectonica

La separacion es estricta:

    B2-PIT  ->  "¿que catalogo era valido en t?"
                    ↓ (input: catalogo sin identidad administrativa)
    B1      ->  "¿que TARGET contractual representa ese catalogo?"
                    ↓
    B3      ->  "¿cuando era conocida cada observacion 13F?"

B2-PIT NO depende de B1 ni B3. B2-PIT NO cierra la identidad
administrativa (`catalog_key`). B2-PIT NO toca el dominio economico.

---

## 2. Modelo de disco

    data/mappings/catalog_snapshots/
        snapshot_<version_id>.csv           # contenido del catalogo
        snapshot_<version_id>.sha256        # hash completo, fichero aparte
    data/mappings/catalog_manifest.json     # indice (unico objeto mutable)

### 2.1. `version_id`

Formato: `<YYYYMMDD>_<NN>` (ej. `20260921_01`).
- NO es el hash del CSV (evita autorreferencia).
- Asignado en el momento de publicacion del snapshot.
- Unico globalmente (dos snapshots en el mismo dia -> sufijos _01, _02).

### 2.2. Contenido del CSV del snapshot

**B2-PIT NO prescribe columnas de identidad administrativa.** El CSV
contiene lo que el productor (probe OpenFIGI, catalogo actual) ha
materializado. Puede contener columnas informativas (`radar_ticker`,
`share_class_figi`, `figi`, `name`, etc.) pero B2-PIT las trata como
opacas.

**NO incluye** una columna `catalog_key` como parte contractual de
B2-PIT. Si en el futuro B1 define un `catalog_key`, sera una
columna mas; el snapshot no se reescribe.

### 2.3. `.sha256`

Un fichero de texto con el SHA-256 completo (64 hex chars) del
`snapshot_<version_id>.csv`.

### 2.4. `catalog_manifest.json`

    {
      "schema_version": 1,
      "snapshots": [
        {
          "version_id": "20260921_01",
          "valid_from": "2026-09-21",
          "valid_to": null,
          "sha256": "<hash completo>",
          "csv_path": "catalog_snapshots/snapshot_20260921_01.csv",
          "rows": 242,
          "producer": "radar_target_catalog.build_from_probe_result"
        }
      ]
    }
---

## 3. Reglas contractuales

### 3.1. `target_catalog_as_of(period_end)`

    def target_catalog_as_of(period_end, *, catalog_root) -> (df, version_id, sha256):
        """Devuelve el snapshot del catalogo cuyo intervalo cubre
        period_end.

        Reglas:
          0 snapshots QUE CUBREN period_end -> raise CatalogNotAvailable
          1 snapshot valido QUE CUBRE        -> devuelve (df, vid, sha256)
          >1 snapshots validos QUE CUBREN    -> raise CatalogAmbiguous

        Un snapshot existente que NO cubre period_end no convierte la
        consulta en valida.
        """

### 3.2. Intervalos

Semiabiertos: `[valid_from, valid_to)`. `null` en `valid_to` = vigente.

Ejemplo sin solapamiento:

    V1 [2026-01-01, 2026-09-19)
    V2 [2026-09-19, null)              # vigente

### 3.3. Inmutabilidad

- `snapshot_*.csv` publicado: inmutable in-place.
- `snapshot_*.sha256` publicado: inmutable in-place.
- `catalog_manifest.json`: mutable (indice).

**Regla:** ninguna evidencia historica publicada se modifica in-place.
Correcciones = snapshot nuevo.

### 3.4. Backdating prohibido

**No se permite asignar `valid_from` pasado al catalogo actual.**

Consecuencia: **Q4 2025 / Q1 2026 no tienen snapshot.**

    target_catalog_as_of("2025-12-31") -> raise CatalogNotAvailable
    target_catalog_as_of("2026-03-31") -> raise CatalogNotAvailable
    target_catalog_as_of("2026-09-21") -> OK

Si el auditor requiere snapshot historico para Q4/Q1, debe aportarse
evidencia externa. No se fabrica retrospectivamente.

### 3.5. Test de integridad

    contenido publicado (.csv)
      -> sha256 publicado (.sha256)
      -> sha256 recalculado
      -> comparar
    mismatch -> FAIL-CLOSED

**Test de corrupcion simulada:** byte alterado -> deteccion ->
fail-closed.

**Validacion de separacion objeto publicado != objeto de trabajo:**
- Tests leen de `catalog_snapshots/`.
- `target_catalog_as_of` no escribe ficheros.
- Ninguna funcion de B2-PIT abre snapshot en modo escritura.

---

## 4. API publica de B2-PIT

**Modulo nuevo:** `src/institutional_accumulation/catalog_pit.py`.

Funciones publicas:

    def target_catalog_as_of(period_end, *, catalog_root) -> ...
    def load_manifest(catalog_root) -> dict
    def verify_snapshot_integrity(version_id, *, catalog_root) -> bool
    def list_snapshots(*, catalog_root) -> list[dict]

Clases / errores:

    class CatalogNotAvailable(Exception): ...
    class CatalogAmbiguous(Exception): ...
    class SnapshotIntegrityError(Exception): ...
    class ManifestError(Exception): ...

**Contrato de pureza:**
- No usar `datetime.now()`. `period_end` es parametro explicito.
- Deterministas.
- Lectura de disco permitida; escritura NO (salvo en fases
  administrativas fuera de B2-PIT).

**NO incluye:** `catalog_validator.validate_assignment`,
`catalog_validator.check_continuity`,
`catalog_validator.check_economic_collision`. Esas viven en B1.
---

## 5. Tests B2-PIT

**Fichero:** `tests/test_catalog_pit.py`.

Casos obligatorios:

    test_as_of_0_snapshots_que_cubren_raise
    test_as_of_1_snapshot_que_cubre_ok
    test_as_of_multiples_snapshots_que_cubren_ambiguous
    test_as_of_snapshot_existente_pero_no_cubre_raise
    test_as_of_backdating_prohibido_q4_2025
    test_as_of_backdating_prohibido_q1_2026
    test_sha256_publicado_vs_recalculado_match
    test_sha256_mismatch_fail_closed
    test_corrupcion_simulada_detectada
    test_manifest_schema_version_valido
    test_intervalos_semiabiertos_sin_solapamiento
    test_manifest_coherente_con_snapshot_sha256
    test_load_manifest_catalog_root_inexistente_raise
    test_snapshot_csv_publicado_inmutable_entre_runs

**Cero regresion** P38/P65/P66 esperada.

---

## 6. Plan de implementacion (B2-PIT)

**Subfase de 2 commits:**

### Commit 1 - modelo + funciones puras

    src/institutional_accumulation/catalog_pit.py       (nuevo)
      - target_catalog_as_of
      - load_manifest
      - verify_snapshot_integrity
      - list_snapshots
      - CatalogNotAvailable / CatalogAmbiguous /
        SnapshotIntegrityError / ManifestError

    data/mappings/catalog_snapshots/snapshot_20260921_01.csv
      - copia del radar_target_catalog.csv actual
      - snapshot inicial sin catalog_key
    data/mappings/catalog_snapshots/snapshot_20260921_01.sha256
    data/mappings/catalog_manifest.json
      - valido desde 2026-09-21

    tests/test_catalog_pit.py                            (nuevo)

### Commit 2 - integracion end-to-end + verificacion global

    - Verificacion: compileall + pyflakes + suite completa
    - Cero regresion P38/P65/P66
    - Evidencia: docs/auditoria/iae/evidence/b2_pit_cierre/ (opcional)

**NO se toca:** `coverage.py`, `nipc.py`, `delta_shares.py`,
`security_identity.py`, `amendments.py`, `relationships.py`,
`temporal_validity.py`, `reporting_dedup.py`.

**NO se incluye:** `catalog_validator` de asignacion. Va en B1.

---

## 7. Criterio de cierre B2-PIT

Cierre de la subfase = todos los casos de §5 pasan + cero regresion
en P38/P65/P66 + compileall + pyflakes.

    A.6.2-bis-B2-PIT CLOSED
      requiere:
        - tests de §5 PASS (14 casos enumerados; 16 tests implementados,
          +2 adicionales: manifest_inexistente_raise,
          list_snapshots_devuelve_entradas)
        - P38 tests PASS (existentes intactos)
        - P65 tests PASS (31)
        - P66 tests PASS (31)
        - compileall OK
        - pyflakes LIMPIO

**Lo que NO habilita el cierre de B2-PIT:**

    A.6.3                 sigue BLOCKED (requiere B1)
    A.6.4                 sigue BLOCKED (requiere A.6.3)
    F2.4-CLOSE            sigue BLOCKED
    B1 / B3               siguen en diseno

---

## 8. Lo que NO se toca en B2-PIT

- Contratos: `NIPC_CONTRATOS_SEMANTICOS_v1.md`, `NIPC_COVERAGE_POLICY.md` v1.0.
- Modelo economico P38 (share_class_figi).
- `coverage.py`, `nipc.py`, `delta_shares.py`.
- `catalog_key` (pertenece a B1, sigue bloqueado por #51).
- Adaptador P38 (pertenece a B1).
- TARGET_PAIRWISE formal (pertenece a B1).
- Validadores de asignacion, continuidad, colision (B1).
- Identidad de las 242 filas actuales (B1).
- OpenFIGI masivo: NO.
- DROP_DUP: NO.
- Push: NO.
- Certificacion "acumulacion": NO.

---

## 9. Trazabilidad

    Dictamen #52          Opcion 1 aprobada, B2-PIT autorizado
    Dictamen #46          B2 aprobado arquitectura (primera vez)
    Dictamen #47-#51      B2 sin regresion (6 rondas)
    A62BIS_PROPUESTA.md   v9 (referencia completa del ciclo)
    F2.4 #24              bloqueante 2 (point-in-time)
    P62 (contrato §11)    point-in-time
    NIPC_CONTRATOS_SEMANTICOS_v1.md secciones 3 y 11

---

Fin del documento B2-PIT. Sometido a implementacion segun plan §6.