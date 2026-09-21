# B-02 - Expediente P62/PIT (correccion de diagnostico al dictamen #76)

**Fecha:** 2026-09-22.
**Origen:** dictamen #76 seccion 4 (B-02) + verificacion tecnica.
**Naturaleza:** expediente para dictamen. NO modifica codigo. Acompana a
`A67_CONSULTA.md`.

---

## 1. Correccion material al dictamen #76

El dictamen #76 seccion 4 afirma:

> "P62/PIT sigue sin estar implementado y el snapshot usado para el
> resultado Q1 tiene valid_from=2026-09-21, posterior a Q1 2026."

Tras verificacion directa sobre el codigo en HEAD actual:

**P62 SI esta implementado.** `src/institutional_accumulation/catalog_pit.py`
(232 lineas, 6.991 B) contiene la infraestructura PIT completa:

    load_manifest(catalog_root)
    _interval_covers(valid_from, valid_to, period_end)   # semiabierto
    _select_snapshot_for_period(manifest, period_end)
    verify_snapshot_integrity(version_id, catalog_root)
    list_snapshots(catalog_root)
    target_catalog_as_of(period_end, catalog_root)

**Tests dedicados:** `tests/test_catalog_pit.py`, 16 tests, incluyendo:

    test_as_of_backdating_prohibido_q4_2025
    test_as_of_backdating_prohibido_q1_2026
    test_as_of_0_snapshots_que_cubren_raise
    test_as_of_1_snapshot_que_cubre_ok
    test_as_of_multiples_snapshots_que_cubren_ambiguous
    test_as_of_snapshot_existente_pero_no_cubre_raise

El sistema **ya rechaza** aplicar el snapshot actual a Q1 2026: existe
un test explicito que lo comprueba (`test_as_of_backdating_prohibido_q1_2026`).

## 2. Estado real del snapshot

`data/mappings/catalog_manifest.json` (unico snapshot registrado):

    version_id:  20260921_01
    valid_from:  2026-09-21
    valid_to:    null
    sha256:      e5d8f9c8... (242 filas)

`radar_target_catalog.csv`: 242 filas, `source_date=2026-09-19`.
`status`: 240 OK, 2 MISS (BRK-B, MOG-A).

## 3. Que ocurre si invocamos target_catalog_as_of para Q1 2026

    target_catalog_as_of("2026-03-31", catalog_root=...)

- `_select_snapshot_for_period` con `period_end=2026-03-31`
- `_interval_covers(2026-09-21, None, 2026-03-31)` -> False (p < vf)
- `hits = []`
- `raise CatalogNotAvailable`

**Conclusion: la via PIT rechaza correctamente Q1 2026.** El sistema no
aplica retroactivamente el catalogo actual a un periodo historico.

## 4. Por que el probe A.6.4 publica 20/20 entonces

El probe `probe_integration_b1_p61_p38.py` **no usa la via PIT**.
Su docstring lo declara expresamente:

> "El snapshot B2-PIT (valid_from=2026-09-21) NO cubre Q1/Q4 por PIT.
> La integracion se ejecuta invocando build_target directamente (no
> target_catalog_as_of). La parte PIT del catalog no se evalua aqui:
> requiere snapshots historicos inexistentes."

Es decir: el probe demuestra que la **cadena tecnica** funciona sobre un
subconjunto de claves del catalogo vigente, pero **no demuestra una
aplicacion PIT valida** para Q1 2026. La correccion material al dictamen
#76 no cambia el veredicto (A.6.6 NO-GO), solo la causa:

  - #76: PIT no implementado.
  - Real: PIT implementado y rechaza correctamente; el probe bypassa PIT
    y no existe snapshot historico que cubra Q1 2026.

## 5. Diagnostico B-02 reformulado

B-02 se desdobla en dos sub-problemas independientes:

**B-02.1 - El probe no usa PIT.**
El probe actual llama `build_target` directamente, saltando la
validacion PIT. Esto es un bypass administrativo documentado, no un bug
de codigo.

**B-02.2 - No existe snapshot historico que cubra Q1 2026.**
El unico snapshot registrado tiene `valid_from=2026-09-21`. Para obtener
un resultado PIT valido para Q1 2026 se necesita un snapshot con
`valid_from <= 2026-03-31 < valid_to`. No existe hoy.

## 6. Opciones tecnicas para B-02.2

**Opcion A - Declarar la limitacion y no buscar snapshot historico.**
El probe se reclasifica explicitamente como evidencia tecnica pre-PIT.
`P38` mantiene `GO CONDICIONADO`. No se pretende cobertura contractual
historica hasta que exista snapshot real post-Q1.

**Opcion B - Snapshot retroactivo con justificacion documental.**
Construir un snapshot con `valid_from <= 2026-03-31` a partir de
`source_date=2026-09-19` del `radar_target_catalog.csv`, mas una
declaracion de que el catalogo era vigente para Q1 2026 segun fuente
documental externa.
Requiere:
  - evidencia documental de la composicion del catalogo en Q1 2026;
  - firma explicita del auditor aceptando el retroactivo;
  - actualizacion del manifest con `valid_from` retroactivo y
    `producer` que identifique la fuente.
Riesgo: convierte el catalogo en un objeto retroactivo, lo cual es
precisamente lo que P62 prohibe si no hay evidencia suficiente.

**Opcion C - Diferir hasta tener snapshot real post-Q1.**
Declarar B-02.2 como bloqueante hasta que exista un snapshot con
`valid_from` anterior a `period_end` objetivo. Implica **no emitir
cobertura contractual historica de Q1 2026** en el ciclo actual.

## 7. Recomendacion del supervisor

**Opcion A + C combinadas:**

1. Reclasificar el probe A.6.4 explicitamente como evidencia **pre-PIT**
   (no invoca `target_catalog_as_of`).
2. Mantener `P38 = GO CONDICIONADO` sin reclamar cobertura historica.
3. Reservar `P62` (infraestructura) como **CERRADO** a nivel
   implementacion; el bloqueo real es la **ausencia de snapshot
   historico**, no el codigo PIT.
4. Fijar como condicion de desbloqueo futuro: existe un snapshot con
   `valid_from <= period_end` y el probe lo usa via
   `target_catalog_as_of`.

Justificacion: no inventar snapshots retroactivos (viola P62 en
espiritu). El objetivo del sistema es auditabilidad, no cerrar fases
con datos reconstruidos.

## 8. Preguntas concretas al auditor

  1. Se acepta la correccion material al dictamen #76 (P62
     implementado; el bloqueo real es ausencia de snapshot historico)?
  2. Se acepta reclasificar el probe A.6.4 como evidencia tecnica
     pre-PIT sin reclamar cobertura contractual historica, mientras no
     exista snapshot que cubra Q1 2026?
  3. Se autoriza la Opcion B (snapshot retroactivo con justificacion
     documental) o se prefiere Opcion C (diferir hasta snapshot real)?
  4. El bloqueo de B-02 se registra como pendiente de materializar
     snapshot historico en lugar de P62 no implementado en el
     proximo dictamen?

## 9. Anexos

- Codigo PIT: `src/institutional_accumulation/catalog_pit.py`
- Tests PIT: `tests/test_catalog_pit.py` (16 tests)
- Manifest: `data/mappings/catalog_manifest.json`
- Probe A.6.4: `evidence/a64_integration_b1_p61_p38/probe_integration_b1_p61_p38.py`
- Consulta principal: `A67_CONSULTA.md`

---

Fin del expediente B-02. HEAD al redactar: `4a802b4`. 2026-09-22.
