# IAE - B2-PIT cierre operativo

**Objeto:** evidencia de cierre operativo de la subfase B2-PIT,
autorizada a implementacion aislada por dictamen estrategico #52.

**Fecha:** 2026-09-21.
**HEAD al cerrar:** 5eee203.
**Commit de implementacion:** 5eee203 (feat(b2-pit): infraestructura
temporal del catalogo).

**Naturaleza:** evidencia directa, no normativa.

---

## 1. Alcance autorizado

Dictamen #52, seccion 5:

    snapshot materializado
    manifest
    version_id
    sha256
    valid_from / valid_to
    target_catalog_as_of(period_end)
    integridad
    corrupcion
    ambiguedad
    no backdating

**NO incluido** (pertenece a B1 / B3):

    catalog_key
    catalog_key_assignment_unique
    resolucion economica FIGI
    continuidad catalog_key -> FIGI
    TARGET_PAIRWISE
    adaptador P38
    collision catalog_key -> FIGI
    catalog_validator de asignacion

---

## 2. Artefactos materializados

    src/institutional_accumulation/catalog_pit.py     (6991 bytes)
      - target_catalog_as_of
      - load_manifest
      - verify_snapshot_integrity
      - list_snapshots
      - CatalogNotAvailable / CatalogAmbiguous /
        SnapshotIntegrityError / ManifestError

    tests/test_catalog_pit.py                         (~9000 bytes, 16 tests)

    data/mappings/catalog_snapshots/
        snapshot_20260921_01.csv      (29849 bytes, 242 filas)
        snapshot_20260921_01.sha256   (65 bytes)
    data/mappings/catalog_manifest.json  (schema_version=1)

---

## 3. Resultado

    pytest tests/test_catalog_pit.py -v
        -> 16 passed

    pyflakes
        -> LIMPIO

    compileall
        -> OK

    Suite completa
        -> 1083 passed + 2 skipped + 3 failed preexistentes

Los 3 failed son los mismos `test_freshness.py` documentados en
`iae/evidence/p66_baseline_pre/` (parquets desactualizados a
2026-09-16). Cero regresion nueva introducida por B2-PIT.

---

## 4. Tests cubiertos

    1.  test_as_of_0_snapshots_que_cubren_raise
    2.  test_as_of_1_snapshot_que_cubre_ok
    3.  test_as_of_multiples_snapshots_que_cubren_ambiguous
    4.  test_as_of_snapshot_existente_pero_no_cubre_raise
    5.  test_as_of_backdating_prohibido_q4_2025
    6.  test_as_of_backdating_prohibido_q1_2026
    7.  test_sha256_publicado_vs_recalculado_match
    8.  test_sha256_mismatch_fail_closed
    9.  test_corrupcion_simulada_detectada
    10. test_manifest_schema_version_valido
    11. test_manifest_inexistente_raise
    12. test_intervalos_semiabiertos_sin_solapamiento
    13. test_manifest_coherente_con_snapshot_sha256
    14. test_load_manifest_catalog_root_inexistente_raise
    15. test_snapshot_csv_publicado_inmutable_entre_runs
    16. test_list_snapshots_devuelve_entradas

Todos los casos obligatorios del documento de subfase §5 estan
cubiertos.

---

## 5. Criterio de cierre B2-PIT

Segun documento de subfase §7:

    - 16 tests PASS                                   CUMPLIDO
    - P38 tests PASS (existentes intactos)            CUMPLIDO
    - P65 tests PASS (31)                             CUMPLIDO
    - P66 tests PASS (31)                             CUMPLIDO
    - compileall OK                                   CUMPLIDO
    - pyflakes LIMPIO                                 CUMPLIDO

**B2-PIT: CLOSED.**

---

## 6. Lo que NO habilita el cierre B2-PIT

    A.6.3                 sigue BLOCKED (requiere B1)
    A.6.4                 sigue BLOCKED (requiere A.6.3)
    F2.4-CLOSE            sigue BLOCKED
    B1                    sigue en diseno (dictamen #51 abierto)
    B3                    sigue en diseno
    A.6.2-bis completo    NO CERRADO

---

## 7. Referencias

    Dictamen #52          Opcion 1 aprobada
    A62BIS_B2_PIT_SUBFASE.md   documento de subfase
    A62BIS_PROPUESTA.md   v9 (referencia completa del ciclo)
    FASE_A6_PLAN.md       plan de la fase A.6
    NIPC_CONTRATOS_SEMANTICOS_v1.md secciones 3 y 11
    F2.4 #24              bloqueante 2 (point-in-time)

---

Fin de la evidencia B2-PIT. Cierre operativo 2026-09-21.