# IAE - A.6.2-bis Propuesta de rediseno arquitectonico TARGET

**Objeto:** propuesta de diseno para materializar los 3 bloqueantes
estructurales F2.4 (B1 TARGET independiente + B2 point-in-time + B3
13F != flujo en tiempo real), autorizada por dictamen #43 (GO
CONDICIONADO).

**Origen:** dictamen #43 (A.6.0 validado + A.6.2-bis AUTORIZADO).
A.6.0 CERRADO. Inventario en `INFORME.md` seccion 21.

**Fecha:** 2026-09-21.

**HEAD al redactar:** 1e83d70.

**Naturaleza:** propuesta. NO normativa. Sometida a dictamen. No
autoriza cambios de codigo por si misma.

---

## 0. Resumen ejecutivo

La propuesta materializa los 3 bloqueantes F2.4 como un unico rediseno
arquitectonico coherente, con 3 piezas acopladas por orden estricto:

    B1 TARGET independiente    ->    B2 point-in-time    ->    B3 semantica temporal

**Alcance autorizado (dictamen #43):**

- B1: TARGET independiente del exito del mapping. Una unica semantica
  contractual + una unica fuente de TARGET.
- B2: versionado historico del catalogo. Fail-closed.
- B3: `knowledge_date` + timestamps separados. Estados separados
  hasta estar implementados y probados.

**Fuera de alcance (dictamen #43):** OpenFIGI masivo, recalculo de
evidencia final, modificacion de contratos, `DROP_DUP`, certificacion
"acumulacion", Policy v1.3, Gate-NIPC.2/3.

**Restriccion transversal:** toda pieza nueva debe ser determinista,
sin `datetime.now()`, sin side effects ocultos, con tests contractuales.

---

## 1. Estado de partida (referencia A.6.0)

| Bloqueante | Piezas existentes | Piezas ausentes |
|---|---|---|
| B1 | `coverage.py::compute_contractual_coverage`, `PositionRecord`, `aggregate_positions_by_shareclass_figi`, `target_universe.py::resolve_cusips`, `radar_target_catalog.csv` | `target_builder.py`, integracion en `nipc.py::compute_coverage_pairwise` |
| B2 | `source_date` (no contractual) | `catalog_version`, `catalog_valid_from`, `catalog_valid_to`, `target_catalog_as_of` |
| B3 | Doctrina P63/P64/P65 | `knowledge_date`, `period_end`+`filing_date` en `PositionRecord`, `absence_reason` |

**Hecho estructural relevante:** `coverage.py` ya define la API
contractual P38 con TARGET externo. `nipc.py::compute_coverage_pairwise`
mantiene una implementacion paralela que usa `observed_security_key`
(CUSIPs observados) como proxy. **La propuesta unifica semanticamente
sin fusionar fisicamente los modulos.**
---

## 2. B1 - TARGET independiente del exito del mapping

**Cautela del dictamen #43:** "el fallo del mapping debe poder producir
ausencia/incertidumbre de identidad SIN alterar el universo TARGET."

### 2.1. Arquitectura propuesta

    CATALOGO (radar_target_catalog.csv)
        |
        v   [catalogo = INPUT externo, versionado]
    TARGET_P = {share_class_figi: ...}    [target_builder.py]
        |
        v   [TARGET construido independiente del 13F y del mapping]
    OBSERVED_13F = {cusip: ..., sshprnamt: ...}    [13F Q4/Q1]
        |
        v   [mapping CUSIP_13F -> share_class_figi: OpenFIGI o exceptions]
    RESOLVED = {share_class_figi: ...}    [post-mapping]
        |
        v
    coverage = f(TARGET_P, RESOLVED)    [coverage.py::compute_contractual_coverage]

**Invariante B1:** el conjunto `TARGET_P` se construye exclusivamente
desde el catalogo + una fuente de periodo (que se materializa en B2).
NO depende de:
- El resultado del mapping CUSIP_13F -> share_class_figi.
- La presencia o ausencia de un CUSIP concreto en el filing 13F.
- El exito de OpenFIGI sobre el filing concreto.

Un CUSIP que falla el mapping produce `NO_ID` en `RESOLVED`, pero
`TARGET_P` no cambia.

### 2.2. Modulo nuevo: `identity/target_builder.py`

    def build_target(period_end, catalog_version, *, catalog_root, observed=None):
        """Materializa TARGET_P (set de share_class_figi) para el periodo.

        Args:
          period_end: fecha ISO del cierre del trimestre (e.g. "2025-12-31").
          catalog_version: identificador de version del catalogo (B2).
          catalog_root: directorio de snapshots historicos (B2).
          observed: DataFrame opcional con {cusip, sshprnamt, period} del 13F.
                    Si se aporta, sirve para trazabilidad (provenance),
                    pero NO altera el conjunto devuelto.

        Returns:
          set[str] de share_class_figi que son TARGET para el periodo.
        """

**Restricciones del modulo:**

- Deterministico. Sin `datetime.now()`. Sin IO de red.
- `catalog_version` y `catalog_root` vienen del caller.
- Si `catalog_version` no cubre `period_end` -> error (fail-closed, ver B2).
- Si `observed` se aporta, se usa para generar provenance por fila,
  NUNCA para reducir o expandir TARGET_P.

**Relacion con `target_universe.py`:**

`target_universe.resolve_cusips(cusips, catalog_df)` es el resolver
inverso (CUSIP observado -> target membership por FIGI). Se mantiene.
`target_builder.build_target(...)` es la pieza complementaria
(catalogo -> TARGET_P). Se anaden los dos:

    TARGET_P = target_builder.build_target(...)      # B1
    RESOLVED = target_universe.resolve_cusips(...)   # preexistente

### 2.3. Refactor de `nipc.py::compute_coverage_pairwise`

**Cambio propuesto:** la funcion mantiene firma publica pero delega el
calculo contractual a `coverage.py`. La logica de `target_pairwise =
sec_c & sec_p` (CUSIPs) se retira.

    def compute_coverage_pairwise(units_current, units_previous, *,
                                   target_q4=None, target_q1=None,
                                   records_q4=None, records_q1=None):
        """
        Wrapper de compatibilidad. Delega a coverage.compute_contractual_coverage.
        Si target_q4/target_q1/records_* no se aportan, devuelve
        coverage_status = "UNAVAILABLE" (Q5 del contrato P38).
        """

**Compatibilidad:** los consumidores actuales de
`compute_coverage_pairwise(units_current, units_previous)` siguen
funcionando. La diferencia es que sin TARGET aportado el resultado es
`UNAVAILABLE` en vez de un proxy silencioso.

**Eliminacion de la doble semantica:** el nombre `target_pairwise`
pasa a tener una unica definicion (interseccion de TARGET_P, no
interseccion de observadas). El comentario actual L211-213 se sustituye
por una referencia directa a la funcion contractual.

### 2.4. Tests B1

- `build_target` con catalogo de 3 entradas y observed vacio -> 3 FIGIs.
- `build_target` con catalogo de 3 entradas y observed con CUSIPs que
  fallan mapping -> TARGET_P identico.
- `build_target` fail-closed si `catalog_version` no cubre `period_end`.
- `compute_coverage_pairwise` con TARGET aportado -> llama a
  `compute_contractual_coverage`.
- `compute_coverage_pairwise` sin TARGET -> `UNAVAILABLE`, no proxy.
---

## 3. B2 - Point-in-time

**Cautela del dictamen #43:** "anadir columnas no basta. Debe existir
politica de snapshots historicos inmutables o equivalente
(hash/versionado verificable). Comportamiento fail-closed."

### 3.1. Modelo de versionado propuesto

**Snapshots inmutables en disco:**

    data/mappings/catalog_snapshots/
        catalog_v_<sha256_prefix>.csv     # snapshot inmutable
        catalog_v_<sha256_prefix>.meta.json  # {valid_from, valid_to, sha256, rows, source_date}
    data/mappings/catalog_current.txt     # apunta al snapshot vigente (nombre)

- El nombre del fichero de snapshot incluye los primeros 12 caracteres
  del SHA-256 -> su contenido es verificable.
- El `.meta.json` declara `valid_from` / `valid_to` (nullable = vigente).
- `catalog_current.txt` es un puntero mutable al snapshot vigente.
  Sobrescribirlo NO modifica snapshots historicos.

**Razon del diseno:** el `radar_target_catalog.csv` actual es un unico
fichero sobrescribible. Perdida de trazabilidad historica.

### 3.2. Funcion contractual

    def target_catalog_as_of(period_end, *, catalog_root):
        """Devuelve el snapshot del catalogo que cubre period_end.

        Reglas (fail-closed):
          - 0 snapshots cubren period_end -> raise CatalogNotAvailable.
          - >1 snapshots cubren period_end -> raise CatalogAmbiguous.
          - 1 snapshot cubre period_end -> devuelve DataFrame del snapshot.
        """

**Sin fallback silencioso.** Si ningun snapshot cubre `period_end`, la
funcion lanza. No se devuelve la version actual por defecto.

### 3.3. Columnas anadidas al catalogo (formato de cada snapshot)

    catalog_version      sha256_prefix del snapshot (identificador unico)
    catalog_valid_from   fecha ISO, no nula
    catalog_valid_to     fecha ISO o NULL (= vigente)

Las columnas `source_date` (fecha del probe OpenFIGI) se mantienen:
son informacion de provenance, NO la validez contractual.

### 3.4. Regla de inmutabilidad

    Un snapshot publicado es inmutable.
    Cualquier correccion se publica como snapshot nuevo con valid_from
    posterior. NO se edita in-place.

**Verificacion:** `catalog_current.txt` + hash del fichero declarado en
`.meta.json` permiten comprobar integridad antes de usarlo.

### 3.5. Politica de cadencia

- Snapshot inicial: el catalogo actual (`2026-09-19`) se convierte en
  `catalog_v_<hash>.csv` con `valid_from = 2026-09-19` y `valid_to = NULL`.
- Nuevo snapshot: cuando cambie el radar (reconstruccion de la lista
  de 242 tickers) o cuando se actualice OpenFIGI sobre tickers existentes.
- Actualizacion de `valid_to`: cuando se publique un snapshot nuevo,
  el anterior recibe `valid_to = valid_from_del_nuevo`. Se modifica el
  `.meta.json` del anterior (metadato, no contenido del `.csv`). El
  `.csv` permanece inmutable.

### 3.6. Tests B2

- `target_catalog_as_of("2026-03-31")` con snapshot `[2026-01-01, NULL]`
  -> devuelve el snapshot.
- `target_catalog_as_of("2025-12-31")` sin snapshot que cubra -> raise.
- Dos snapshots solapados -> raise `CatalogAmbiguous`.
- Snapshot con hash que no coincide con `meta.json` -> raise.
- Inmutabilidad: el `.csv` del snapshot no se modifica entre builds.
---

## 4. B3 - 13F != flujo en tiempo real

**Cautela del dictamen #43:**

- `knowledge_date` debe representar la fecha en que la informacion
  estaba disponible segun el contrato aplicable, no "fecha actual del
  proceso".
- Los elementos (absence_reason, ZERO_REPORTED, NOT_PRESENT,
  corporate action vs economic) deben permanecer explicitamente
  separados hasta que cada semantica este implementada y probada.

### 4.1. Tres timestamps

    period_end      cierre del trimestre (13F: 31-mar, 30-jun, ...)
    filing_date     fecha del filing (SUBMISSION.FILING_DATE)
    knowledge_date  fecha en que el dato entro en la cadena local
                    (= max(FILING_DATE de las submissions del periodo))

**Definicion de `knowledge_date`:** el maximo de `FILING_DATE` de las
submissions consideradas. Es determinista: se deriva del dataset, no
del reloj del sistema. No cambia si el pipeline se reejecuta.

### 4.2. Ampliacion de `PositionRecord`

Anadir campos:

    period_end:    str    # "2025-12-31"
    filing_date:   str    # ISO del filing del filing_manager
    knowledge_date: str   # ver 4.1

`period` (campo actual) se conserva por retrocompatibilidad. En la
practica `period` == `period_end` tras el rediseno.

### 4.3. Clasificador de ausencia (DIFERIDO)

El dictamen #43 exige que los estados permanezcan separados hasta
estar implementados y probados. **La propuesta NO materializa el
clasificador en A.6.2-bis.** Solo prepara la infraestructura:

- Enum declarado en `aggregation/absence.py` (nuevo): `MISSING`,
  `BELOW_REPORTING_THRESHOLD`, `CONFIDENTIAL`, `OTHER_MANAGER`,
  `UNKNOWN`, `ZERO_REPORTED`, `NOT_PRESENT`.
- Docstring con las reglas del contrato P63 (R1-R8).
- `NotImplementedError` en el clasificador.

Su materializacion sera otro ciclo (A.6.7 o similar), tras dictamen
especifico.

### 4.4. Regla dura preservada

`delta_shares` NO crea `SOLD`. La etiqueta `P64_EVENTS_DEFERRED` se
mantiene: la distincion corporate action vs economic accumulation
requiere fuente externa y queda diferida.

### 4.5. Tests B3

- `PositionRecord` con los 3 timestamps -> OK.
- `knowledge_date = max(FILING_DATE)` sobre un DataFrame sintetico -> OK.
- `absence.py::classify_absence(...)` -> NotImplementedError.
- `delta_shares` sigue sin producir SOLD (test existente P63).

---

## 5. Orden de implementacion

El auditor fijo el orden:

    A.6.0  INVENTARIO                             CERRADO
       v
    A.6.2-bis  REDISENO / MATERIALIZACION B1-B3   ESTA PROPUESTA
       v
    A.6.3  TEST P38 Q12
       v
    A.6.4  RECALCULO DE EVIDENCIA
       v
    A.6.6  F2.4-CLOSE

**Dentro de A.6.2-bis, sub-orden propuesto:**

    Commits 1-2   B1  target_builder.py + refactor nipc.py
    Commits 3-4   B2  catalog_snapshots/ + target_catalog_as_of
    Commit  5     B3  PositionRecord extendido + absence.py (stub)
    Commit  6     Integracion end-to-end + tests

**Regla:** un commit = una verificacion = un commit. Cada commit pasa
`compileall` + `pyflakes` + suite antes de continuar.
---

## 6. Tests requeridos (resumen)

| Bloque | Fichero | Cobertura |
|---|---|---|
| B1 | `tests/test_target_builder.py` (nuevo) | build_target puro, fail-closed, independencia del mapping |
| B1 | `tests/test_sec_13f_nipc.py` (extender) | compute_coverage_pairwise delega, sin TARGET -> UNAVAILABLE |
| B2 | `tests/test_target_catalog_as_of.py` (nuevo) | snapshots, fail-closed, ambiguedad, inmutabilidad |
| B3 | `tests/test_position_record.py` (nuevo) | 3 timestamps, knowledge_date derivada |
| B3 | `tests/test_absence.py` (nuevo) | NotImplementedError + enums declarados |

**Cero regresion P65/P66 esperada.** Los tests P65 actuales
(31 passed) y P66 (31 passed) siguen sin cambios.

---

## 7. Preguntas al auditor

1. **B1 - Estructura de `target_builder.py`.** ¿Se aprueba la firma
   `build_target(period_end, catalog_version, catalog_root, observed)`?
   ¿O se prefiere un contrato distinto (por ejemplo, un objeto
   `TargetUniverse` en vez de una funcion pura)?

2. **B1 - `compute_coverage_pairwise` como wrapper.** ¿Se aprueba que
   mantenga la firma publica pero delegue a `compute_contractual_coverage`
   y devuelva `UNAVAILABLE` sin TARGET? ¿O se prefiere deprecarla y
   eliminar sus consumidores?

3. **B2 - Layout de snapshots.** ¿Se aprueba
   `data/mappings/catalog_snapshots/catalog_v_<sha256_prefix>.csv` +
   `.meta.json` + `catalog_current.txt`? ¿O se prefiere un formato
   distinto (por ejemplo, JSON single-file con historico embebido)?

4. **B2 - Politica de publicacion.** ¿Se aprueba que la correccion de
   un snapshot se publique como snapshot nuevo con `valid_from`
   posterior, sin tocar el original? ¿O se admite `amend` con
   `version_suffix`?

5. **B3 - Definicion de `knowledge_date`.** ¿Se aprueba
   `max(FILING_DATE)` sobre las submissions del periodo? ¿O se
   prefiere otra definicion (por ejemplo, `fecha del run_id` de la
   primera ingesta del periodo, conservada en un log externo)?

6. **B3 - Clasificador de ausencia.** ¿Se aprueba declarar los enums
   + `NotImplementedError` en A.6.2-bis, y materializar el clasificador
   en un ciclo posterior (A.6.7 o nuevo)? ¿O se prefiere no declarar
   nada hasta tenerlo todo?

7. **Cierre de A.6.2-bis.** ¿Que criterio de aceptacion fija el auditor
   para cerrar A.6.2-bis y autorizar A.6.3?

---

## 8. Lo que NO se toca en A.6.2-bis

- Contratos: `NIPC_CONTRATOS_SEMANTICOS_v1.md`, `NIPC_COVERAGE_POLICY.md` v1.0.
- Modulos: `delta_shares.py`, `security_identity.py`, `relationships.py`,
  `amendments.py`, `temporal_validity.py`, `coverage.py` (solo se invoca,
  no se modifica).
- Catalogo actual: `radar_target_catalog.csv` (se conserva como input;
  los snapshots son ficheros adicionales, no sustitutos).
- OpenFIGI masivo: NO ejecutado.
- `DROP_DUP`: NO activado.
- Push a `origin/main`: NO.
- Certificacion "acumulacion": NO.

---

## 9. Trazabilidad

    Dictamen #43                 A.6.0 CERRADO + A.6.2-bis AUTORIZADO
    A.6.0 inventario             INFORME.md seccion 21
    F2.4 (dictamen #24)          3 bloqueantes estructurales
    RECONCILIACION D1/D2/D3      divergencias contrato <-> codigo
    REESTRUCTURACION §4.1-4.5    arquitectura objetivo previa
    Contratos P38/P62/P63/P64/P65  secciones 3, 11, 12, 13, 14

---

Fin de la propuesta. Sometida a dictamen del auditor. HEAD 1e83d70.