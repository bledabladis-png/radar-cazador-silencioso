# A.6.7 - Consulta al auditor: desbloqueo de B-02 / B-03 / B-04

**Solicitante:** Ingeniero Supervisor del Radar de Rotacion Sectorial.
**Destinatario:** auditor externo del proyecto IAE.
**Fecha:** 2026-09-22.
**Origen:** dictamen #76 (A.6.6 = NO-GO). Saneamiento B-01/B-05/B-06/B-07
y cierre H-08 EJECUTADOS y cerrados. Los unicos bloqueos residuales para
avanzar son B-02, B-03 y B-04. Esta consulta los desarrolla uno a uno con
propuesta tecnica y peticion concreta.

---

## 1. Contexto

Tras el dictamen #76 el supervisor ejecuto, dentro de lo autorizado:

- Cierre tecnico de H-05/H-06/H-07/H-10.1 (ciclo A2, 5 commits).
- Saneamiento documental B-01 + precision B-05 + fix B-06 + documentacion
  de B-07 + cierre H-08 como test (commit `6d2f17f`).
- Anexo de diffs `A66_DIFFS.txt` (auditable linea a linea sin push).

El repositorio esta limpio, la suite pasa a `1312 passed + 2 skipped +
3 failed` (los 3 failed son `test_freshness`, preexistentes). No hay
trabajo tecnico pendiente sin dictamen nuevo.

Los tres bloqueos residuales son estructurales, no bugs: requieren una
decision del auditor antes de tocar nada. A continuacion se desarrollan.

## 2. B-02 - P62/PIT (look-ahead temporal del snapshot)

### 2.1. Descripcion del bloqueo

El resultado empirico `coverage_current = 1.0` se calcula aplicando el
snapshot B2-PIT `snapshot_20260921_01.csv` (valid_from = 2026-09-21) al
periodo Q1 2026 (period_end = 2026-03-31). El contrato P62 prohibe
aplicar retroactivamente la version de hoy a un periodo historico.

### 2.2. Propuesta tecnica

Ciclo A.6.7-P62 con los siguientes pasos:

  1. Auditoria de snapshots disponibles en `data/mappings/catalog_snapshots/`.
     Inventario de fechas valid_from/valid_to por snapshot.
  2. Determinar si existe (o puede construirse) un snapshot valido para
     Q1 2026 (valid_from <= 2026-03-31 < valid_to).
  3. Si existe: re-ejecutar el probe con ese snapshot. Si no existe:
     evaluar si el catalogo actual puede retro-fecharse con fuente
     documental (radar_target_catalog.csv tiene source_date).
  4. Exponer `target_catalog_as_of(period_end)` en `catalog_pit.py`
     (capacidad ya disenada, no invocada).
  5. Re-ejecutar el probe con invocacion PIT completa.

Impacto estimado: 1-2 commits, sin tocar contrato ni policy. Solo
infraestructura PIT + probe.

### 2.3. Preguntas al auditor

  1. ¿Se autoriza el ciclo A.6.7-P62 con el alcance descrito?
  2. Si NO existe snapshot valido para Q1 2026: ¿se acepta retro-fechar
     el catalogo con `source_date` como evidencia documental, o se exige
     un snapshot historico adicional?
  3. ¿El probe post-P62 debe publicar el resultado PIT como metrica
     contractual definitiva, o se mantiene como evidencia parcial hasta
     cerrar B-03/B-04?

## 3. B-03 - TARGET completo (OpenFIGI masivo)

### 3.1. Descripcion del bloqueo

`TARGET_Q1 = 20 FIGIs` sobre un universo B2-PIT de 242 keys. El TARGET
contractual completo requiere resolver los ~24.838 CUSIPs del universo
13F contra OpenFIGI, operacion no autorizada. Mientras no se resuelva,
el `coverage_current = 1.0` es correcto sobre el subconjunto materializado
pero no certifica el universo contractual completo.

### 3.2. Dos opciones tecnicas

**Opcion A - OpenFIGI masivo autorizado.**
  - Resolver los 24.838 CUSIPs contra la API OpenFIGI en batches.
  - Requiere API key (no hay hoy) + rate limiting + snapshot con hash.
  - El contrato P62 exige que la resolucion OpenFIGI se acompañe de
    snapshot + hash (seccion 11.4).
  - Riesgo: coste operativo, dependencia externa, rate limits.

**Opcion B - Acotar el universo contractual y justificarlo.**
  - Definir el universo contractual como un subconjunto explicito
    (p.ej. TOP 2000 por AUM, o universo con crosswalk interno curado).
  - Documentar el criterio de acotacion en `NIPC_COVERAGE_POLICY.md`.
  - El `coverage_current` resultante se etiqueta como 'sobre universo
    acotado (criterio X)'.
  - No requiere OpenFIGI masivo. Elimina el bloqueo estructural.

### 3.3. Preguntas al auditor

  1. ¿Se autoriza OpenFIGI masivo (Opcion A)? Si si: ¿con que API key y
     con que limites de rate? ¿Se exige snapshot+hash antes del primer
     batch o se acepta snapshot al cierre del ciclo?
  2. Si NO se autoriza OpenFIGI masivo: ¿se acepta la Opcion B (acotar
     universo con criterio documentado)? Si si: ¿que criterio de
     acotacion acepta el auditor como contractual?
  3. ¿La eleccion A o B afecta al estado de THRESHOLD_1/THRESHOLD_2
     (hoy UNDEFINED/BLOQUEADO) o son dimensiones independientes?

## 4. B-04 - Pairwise real (TARGET_PAIRWISE > 0)

### 4.1. Descripcion del bloqueo

El resultado empirico tiene `TARGET_PAIRWISE = 0` porque Q4 2025 esta
vacio en el subconjunto materializado. La rama fail-closed esta validada
(`coverage_previous = None`, `paired_* = None`), pero la rama pairwise
real (calculo `paired_security_coverage` y `paired_weighted_share_coverage`
sobre un conjunto no vacio) NO ha sido ejercitada end-to-end sobre datos
reales.

### 4.2. Opciones de dataset

**Opcion 1 - Snapshot Q4 2025 valido (depende de B-02).**
  - Requiere primero resolver B-02 (snapshot PIT).
  - Con snapshot Q4 valido, `TARGET_Q4` podria ser > 0.
  - Si Q4 tiene al menos un FIGI comun con Q1 -> pairwise real.

**Opcion 2 - Dataset sintetico controlado.**
  - Construir un dataset minimo (2-5 FIGIs con overlap Q4<->Q1) a mano.
  - Documentar como fixture, no como evidencia empirica del universo real.
  - El auditor ya autorizo tests unitarios que cubren esto. La pregunta
    es si acepta un probe con fixture sintetico como cierre de B-04, o
    exige datos reales.

**Opcion 3 - Diferir B-04 hasta que exista overlap real.**
  - Declarar P38 empiricamente validado solo en rama Q1 + fail-closed.
  - Mantener `P38 = GO CONDICIONADO` sin cambio.
  - No bloquea A.6.6 si el auditor acepta la limitacion declarada.

### 4.3. Preguntas al auditor

  1. ¿Cual de las 3 opciones acepta el auditor como via de cierre B-04?
  2. Si Opcion 2: ¿que tamano minimo de dataset sintetico se considera
     suficiente? ¿Se exige provenance documental del fixture?
  3. Si Opcion 3: ¿se puede emitir A.6.6 con P38 GO CONDICIONADO sin
     pairwise empirico, siempre que la limitacion quede declarada?

## 5. Orden propuesto de resolucion

Si el auditor autoriza los tres ciclos:

  1. B-02 (P62/PIT): prerequisito de B-04 en Opcion 1.
  2. B-03 (TARGET): define el universo contractual de todos los calculos.
  3. B-04 (pairwise): cierra la rama empirica pendiente.
  4. Re-ejecutar probe con todo aplicado.
  5. Bundle A.6.7 con diffs (`A66_DIFFS.txt` regenerado).
  6. Nueva solicitud A.6.6.

Si el auditor acepta Opcion 3 en B-04 (diferir):

  1. B-02 (P62/PIT).
  2. B-03 (TARGET).
  3. Bundle con limitacion pairwise declarada.
  4. Solicitud A.6.6 con P38 GO CONDICIONADO.

## 6. Peticion

Se solicita:

- Dictamen sobre B-02 (autorizacion ciclo A.6.7-P62 + criterio de
  snapshot historico).
- Dictamen sobre B-03 (OpenFIGI masivo o acotacion documentada).
- Dictamen sobre B-04 (opcion de cierre de la rama pairwise).
- Indicacion del orden de ejecucion preferido.

Con estos tres autorizados/decididos, el supervisor podra ejecutar los
ciclos tecnicos y preparar el bundle A.6.7.

## 7. Expedientes tecnicos adjuntos (2026-09-22)

Verificacion directa sobre codigo y datos en HEAD actual corrige el
diagnostico de #76 en los tres sub-frentes. Documentos complementarios:

| Expediente | Contenido | Correccion material |
|---|---|---|
| `B02_EXPEDIENTE.md` | P62/PIT | P62 SI implementado (catalog_pit.py, 16 tests). Bloqueo real: ausencia de snapshot historico. |
| `B034_EXPEDIENTE.md` | B-03 + B-04 | B-03: universo contractual = 242 keys (no 35.649). Hueco real = 2 MISS (BRK-B, MOG-A). B-04: Q4 SI tiene datos; cuello = vigencia temporal de fuentes. |

Conclusiones operativas:

- No es necesaria OpenFIGI masiva (35.127 CUSIPs son OUT_OF_TARGET).
- 2 re-queries OpenFIGI sobre BRK-B/MOG-A cierran el hueco real.
- 20-30 exceptions Q4 desbloquean la rama pairwise empirica.
- P62 no requiere implementacion; requiere decision sobre snapshot
  historico (declarar limitacion o construir retroactivo).

---

Fin de la consulta A.6.7. HEAD `dae492c`. 2026-09-22.
