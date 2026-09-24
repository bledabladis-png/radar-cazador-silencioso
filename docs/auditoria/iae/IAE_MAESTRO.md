# IAE_MAESTRO - Modulo Institutional Accumulation Engine

Documento unico del modulo IAE. Estado, arquitectura, verificacion.

**Actualizado:** 2026-09-24
**HEAD operativo:** ver `docs/auditoria/iae/ESTADO_SISTEMA.md` (campo HEAD).
**Snapshot de mediciones:** §2.1 y §12 sobre `7caa86b` (2026-09-23); §3.2 sobre HEAD actual (post-fix `89b98e9`).


**Que es este documento.** Describe el modulo IAE tal como esta
implementado y verificado. No certifica cumplimiento de contratos
externos ni periodos historicos. Los contratos P60-P70 en
`NIPC_CONTRATOS_SEMANTICOS_v1.md` se mantienen como documentacion
historica del diseno, no como normativa vigente.

### Estado de puntos (indice de auditoria)

Este documento ha pasado por varios dictamenes externos. Indice para
lectores nuevos:

- **Dictamen v3** (6 puntos): cerrado. Cobertura 92% reproducible via
  `scripts/iae_coverage.py`; separacion contractual/PROXY en §10.7;
  umbral 0.95 declarado operativo; fail-closed denominador ponderado=0
  en §10.5.
- **Dictamen v4** (11 puntos): cerrado. Ref cruzada §10.7, error
  historico §13.3, frase OpenFIGI §12.4, invariante C9 §10.4,
  fail-closed ponderado §10.5, `reporting_dedup` §10.7, HEAD
  operativo. Punto 7 (evidencia OpenFIGI reproducible) cerrado
  2026-09-23 con `script_version` + tests + artefactos versionados
  (§12.4). Deuda #8 eliminada de `ESTADO_DECLARADO §3`.
- **Dictamen v5** (auditor fresco, 15 puntos): GATE 1 cerrado
  (nomenclatura catalog/TARGET, §1, §13.5). GATE 2 cerrado
  empiricamente para la configuracion E2E auditada: 0 fragmentacion
  de identidad por shareClassFIGI, 0 uso de la rama `figi:*`, delta
  NIPC = 0 al normalizar (§13.10, auditoria
  `scripts/iae_identity_uniqueness_audit.py`). Resto de puntos
  cubiertos por dictamenes previos o declarados como limitacion.

**Limitaciones declaradas** (no bugs): PIT historico Q4 2025 (§13.9,
ASSUMPTION de reconstruccion retrospectiva), NIPC como variacion
reportada, no flujo economico (§10.3, §13.7), cobertura 12,2% FIGI en
INFOTABLE (§13.1). La dualidad `equity:` / `figi:` del resolver
(§13.10) es una capacidad implementada y cubierta por tests, no
activada en la configuracion E2E auditada.

**Deuda activa:** ver `ESTADO_DECLARADO.md` seccion 3.

---

## 1. Resumen ejecutivo

### Que es el IAE

El Institutional Accumulation Engine (IAE) es el modulo del sistema que
analiza posiciones institucionales a partir de los filings 13F de la SEC.
Responde a una pregunta concreta:

> Dado un universo de tickers del radar, que instituciones han aumentado
> o reducido su posicion entre dos trimestres consecutivos, y con que
> cobertura de datos se puede afirmar.

### Estado actual

El modulo esta implementado, testeado y verificado sobre datos reales.
Opera integrado en el pipeline productivo del radar sectorial
mediante una dependencia unidireccional (run.py -> IAE).

Las mediciones estructurales siguientes corresponden al estado del
repositorio a 2026-09-24 (post F.1-F.4 + F-IAE-HOLIDAY-01); el conteo
de tests y la cobertura se actualizan tras cada commit (ver cabecera
del documento).

| Aspecto | Valor |
|---|---|
| Ficheros de produccion | 36 |
| LOC produccion | 8.008 |
| Funciones publicas | 110 |
| Ficheros de test | 44 |
| Tests que pasan | 819 |
| Cobertura de lineas | 90% |
| Integrado en produccion | SI (integracion unidireccional) |

El conteo de tests sigue el criterio: ficheros `tests/test_*.py` que
importan `src.institutional_accumulation` (verificado por AST).
Reproducible con `scripts/iae_test_census.py`.

### Verificacion clave

Sobre los filings reales de Q4 2025 y Q1 2026, con el crosswalk CUSIP
extendido a 246 filas (242 tickers unicos), el motor produce:

- Q4 2025: 240 tickers contribuyen al TARGET · Q1 2026: 240.
- TARGET_PAIRWISE = 240 · cobertura catalogo declarado 240/242 = 99,17%
  · cobertura interna del TARGET construido = 100% (coverage_status = VALID,
  coverage_quality = COMPLETE). Las dos metricas son distintas: la
  segunda es endogena al TARGET por construccion. Ver §13.5.
- Delta radar: 553.321 filas (BOTH 444.276 · NEW 60.324 · EXIT 48.721).
- NIPC radar: -4.316.734.936 (SOLE -32.485B, DFND +28.318B, OTR -0.150B).

Detalle de sub-universos y contadores C9/C10 en §12.3. OKE y SPCX no
contribuyen al TARGET (§13.2).

Los pesos agregados son reproducibles a partir de los parquets de
entrada. La cadena completa de calculo esta en la seccion 12 de este
documento.

---

## 2. Arquitectura del modulo

### 2.1 Estructura de ficheros

    src/institutional_accumulation/
    ├── __init__.py                          28 LOC
    ├── absence.py                           77 LOC
    ├── catalog_pit.py                      232 LOC
    ├── operational_universe.py             194 LOC
    ├── pipeline_contractual.py             211 LOC
    ├── security_type.py                    397 LOC
    ├── temporal_validity.py                 92 LOC
    ├── timestamps.py                       153 LOC
    ├── aggregation/
    │   ├── __init__.py                      97 LOC
    │   ├── catalog_p38_adapter.py          170 LOC
    │   ├── catalog_validator.py             94 LOC
    │   ├── coverage.py                     258 LOC
    │   ├── delta_shares.py                 380 LOC
    │   ├── nipc.py                         395 LOC
    │   └── reporting_dedup.py              880 LOC
    ├── identity/
    │   ├── __init__.py                      13 LOC
    │   ├── catalog_key.py                  375 LOC
    │   ├── openfigi_client.py              180 LOC
    │   ├── period_state.py                 212 LOC
    │   ├── radar_target_catalog.py         145 LOC
    │   ├── target_builder.py               181 LOC
    │   └── target_universe.py              171 LOC
    └── sec_13f/
        ├── __init__.py                      22 LOC
        ├── downloader.py                   224 LOC
        ├── ingest.py                       131 LOC
        ├── manifest.py                     150 LOC
        ├── parser.py                       142 LOC
        ├── schema.py                       185 LOC
        ├── storage.py                       89 LOC
        └── identity/
            ├── __init__.py                 161 LOC
            ├── amendments.py               444 LOC
            ├── cusip_resolver.py           149 LOC
            ├── relationships.py            367 LOC
            ├── sec13f_list.py              255 LOC
            ├── security_identity.py        654 LOC
            └── temporal_filter.py          100 LOC
**Totales:** 36 ficheros, 8.008 LOC produccion, ~8.300 LOC test.

LOC medidos el 2026-09-24 sobre el estado del repositorio en ese
momento. Se cuentan todas las lineas del fichero (codigo, comentarios
y lineas en blanco). Pueden desfasarse con cada commit; el valor de
este bloque es orden de magnitud, no cifra contractual.

### 2.2 Capas funcionales

**Ingestion (`sec_13f/`)** - Descarga, parsea y almacena filings 13F.
Componentes: downloader, parser, schema, storage, manifest, ingest.

**Identity (`sec_13f/identity/`)** - Resuelve identidad de securities:
CUSIP a FIGI, relaciones entre managers, enmiendas, elegibilidad SEC.

**Catalog (`identity/`)** - Mantiene el catalogo radar: que tickers
forman el universo operacional, con provenance.

**Aggregation (`aggregation/`)** - Calcula NIPC y cobertura. Paquete
puro: no lee ni escribe ficheros, no usa datetime.now(), determinista.

**Modulos transversales** - `catalog_pit.py`, `operational_universe.py`,
`security_type.py`, `temporal_validity.py`, `timestamps.py`,
`absence.py`.

### 2.3 Integracion unidireccional

**Estado 2026-09-23:** el IAE se ha integrado a `run.py` y al reporte
diario como una seccion adicional. La integracion es unidireccional
y controlada.

Flujo permitido (unico):

    run.py -> src.pipeline.iae_section -> IAE -> reporte

Restriccion inversa (prohibida):

    IAE -X-> run.py
    IAE -X-> src.report.*
    IAE -X-> src.regimes.*
    IAE -X-> src.indicators.*

El modulo mantiene su encapsulamiento interno. Se permite que la
capa de ejecucion (run.py) y de reporte (src.report.iae) consuman el
IAE; se prohibe que el IAE dependa de ellas. Los scripts de
`scripts/*.py` que invocan IAE son scripts reproducibles de auditoria,
no consumidores productivos.

Verificacion inversa (2026-09-23): no existe ningun import desde
`src/institutional_accumulation/*` hacia `run.py`, `src.report`,
`src.pipeline`, `src.regimes` o `src.indicators`.

---

## 3. Estado de implementacion

### 3.1 Superficie publica

    Funciones publicas:            110
    Clases / excepciones:          14
    Total simbolos publicos:       124

Las 110 funciones estan invocadas al menos una vez en la suite de tests.
Esto no equivale a verificacion exhaustiva. La cobertura de lineas es
90% (ver §3.2); ningun fichero queda por debajo del 80%. Los modulos
con menor cobertura siguen siendo riesgo pendiente de mejora antes de
su integracion en produccion, no deuda que "subira sola".

### 3.2 Cobertura de tests

    Cobertura de lineas:           90%  (3093 stmts, 304 miss)
    Ficheros >= 95%:               17
    Ficheros 80-94%:               15
    Ficheros < 80%:                 0

Nota de actualizacion (2026-09-24). La cobertura bajo de 92% a 90%
tras anadir `pipeline_contractual.py` (+129 stmts instrumentables).
De esos, 76 no estan cubiertos por tests unitarios: `run_contractual_nipc`
se verifica via E2E real (`scripts/iae_contractual_nipc_e2e.py`, PASS),
no via tests unitarios con mock. Recuperar 92% requeriria tests unitarios
adicionales con mock de la cadena completa; se considera deuda de
cobertura residual.

Ningun fichero del modulo queda por debajo del 80% de cobertura de
lineas. El minimo actual es catalog_key.py (82%). Los ficheros con
cobertura mas baja siguen sin estar integrados en produccion.

Nota de alcance. La cobertura se calcula sobre 33 ficheros con
statements instrumentables. Los 3 ficheros restantes hasta 36 son
`__init__.py` sin statements: `src/institutional_accumulation/__init__.py`,
`identity/__init__.py`, `sec_13f/__init__.py`. Los otros 2
`__init__.py` (`aggregation/` 6 stmts, `sec_13f/identity/` 7 stmts)
si contienen statements y se cuentan en el total.

### 3.3 Estado de integracion

El IAE esta integrado en `run.py` como fase adicional del pipeline
productivo, y expone una seccion en el reporte diario. La integracion
es unidireccional (ver §2.3): `run.py` consume el IAE; el IAE no
importa `run.py` ni las capas de reporte/regimes/indicators.

Componentes integrados:
- `src/pipeline/iae_section.py::compute_iae_section` invocado desde
  `run.py` tras el calculo de matrices finales.
- `src/report/iae.py::render_iae_section` invocado desde
  `src/report_generator.py` al final del reporte.

Deuda residual:
- `compute_nipc_contractual`: sin callers productivos directos (lo
  consume `pipeline_contractual.run_contractual_nipc`, que si tiene
  caller productivo).
- `build_effective_reporting_snapshot`: sin callers productivos.

---

## 4. Verificacion empirica

Toda la evidencia de esta seccion ha sido medida el 2026-09-22
mediante comandos reproducibles. Los comandos estan en el anexo.

**Aviso de alcance.** Esta seccion documenta un estado INTERMEDIO del
motor: post-crosswalk parcial (`cusip_to_radar_figi.csv` con 243
filas), PRE-fix 7609a56 (que extendio `valid_from` a Q4 2025 para
DD/HON/XOM). Las mediciones aqui no son representativas del estado
actual: el crosswalk tiene 246 filas y el fix 7609a56 corrigio la
resolucion Q4. Los valores concretos que aqui aparecen (242 tickers
del catalogo, TARGET_PAIRWISE = 239) se conservan como referencia
historica del estado intermedio. Para el estado final ver §12.

Fechas relevantes: el catalogo radar nacio el 2026-09-21 (§13.8).
Las mediciones de esta seccion se tomaron el 2026-09-22. El rename
`cusip_ticker_exceptions.csv` -> `cusip_radar_crosswalk.csv` se hizo
el 2026-09-23 (§4.3, nota de nomenclatura).

### 4.1 Datos disponibles

    data/sec_13f/processed/2025Q4/
      INFOTABLE.parquet        96,5 MB   3.473.209 filas   33.780 CUSIPs
      COVERPAGE.parquet
      OTHERMANAGER.parquet
      OTHERMANAGER2.parquet
      SIGNATURE.parquet
      SUBMISSION.parquet
      SUMMARYPAGE.parquet

    data/sec_13f/processed/2026Q1/
      INFOTABLE.parquet       104,4 MB   3.822.885 filas   35.649 CUSIPs
      (resto de tablas analogas)

Dos trimestres completos, parseados y almacenados. La fuente primaria
(SEC EDGAR) esta integra.

### 4.2 El catalogo radar es una regla determinista

    load_radar_tickers() devuelve:        242 tickers
    radar_target_catalog.csv tiene:       242 tickers
    Solo en regla:                          0
    Solo en catalogo:                       0
    En ambos:                             242

La funcion `load_radar_tickers()` aplica la regla: tickers USA, sin
sufijo europeo, sin prefijo ^, sin sufijo =X. Aplicada al
`stock_prices.parquet` actual, reproduce exactamente el catalogo
materializado.

### 4.3 CUSIPs con excepcion documentada

`cusip_radar_crosswalk.csv` contiene 246 filas con vigencia
`2025-12-31` a `2026-03-31`. El campo `source` con valor `SEC-EDGAR`
identifica la fuente primaria de evidencia (CUSIPs observados en
filings SEC); el mapping `CUSIP -> radar ticker` es una **derivacion**
que combina filings SEC + `radar_target_catalog.csv` (ver §12.2). No
debe interpretarse como si SEC proporcionara directamente el mapping.
Las 246 filas
corresponden a 242 tickers unicos. La diferencia se debe a 4 tickers
con doble CUSIP (AMCR, AMZN, LRCX, MU, ver §12.3).
El crosswalk incluye tambien 2 tickers que no estan en el radar
canonico actual (DD, HON), y NO incluye 2 tickers del radar (OKE,
SPCX). Ver §13.2 para el detalle.

Las 24 originales (valid_from=2026-03-31) fueron extendidas a Q4 y
se anadieron 222 nuevas desde el crosswalk CUSIP->radar (construido
cruzando filings con el catalogo radar, ver seccion 12.2).

Nota de nomenclatura. El fichero se renombro de
`cusip_ticker_exceptions.csv` a `cusip_radar_crosswalk.csv` el
2026-09-23. El string `source` `cusip_ticker_exceptions` que
aparece en los registros de identidad se conserva como
provenance historica y NO se renombra: los registros resueltos
en el pasado deben seguir declarando la fuente original.

### 4.4 Overlap sobre datos reales

Setup:

    FIGIs observados Q4:       22.127
    FIGIs observados Q1:       22.917
    radar_scf:                 share_class_figi del catalogo radar

    TARGET_Q4       = FIGIs_Q4 intersect radar_scf
    TARGET_Q1       = FIGIs_Q1 intersect radar_scf
    TARGET_PAIRWISE = TARGET_Q4 intersect TARGET_Q1

Resultado:

    TARGET_Q4:            239
    TARGET_Q1:            239
    TARGET_PAIRWISE:      239

Existe overlap real de 239 FIGIs compartidos entre el radar y los
filings observados. La rama pairwise del calculo se ejercita sobre
datos reales, no solo sobre fixture sintetico.

**Nota de alcance.** Este resultado es el overlap observable por FIGI
directamente presente en los filings (12,2% de las filas de INFOTABLE
llevan FIGI poblado). El overlap completo requeriria resolver los
CUSIPs restantes por otra via, lo que no esta implementado actualmente.

### 4.5 Verificacion anti-artefacto

Cuentas de filings por FIGI, Q4 vs Q1:

    AAPL:    794 filings Q4  vs  943 filings Q1   (+18.7%)
    Cuentas identicas Q4 <-> Q1 por FIGI: False

Las cuentas difieren entre trimestres. Los datasets subyacentes no
tienen distribucion identica. El overlap de 239 es real.

### 4.6 Analisis de pesos agregados

    Suma SSHPRNAMT agregada por FIGI:
      Q4 2025:   7.614.222.283
      Q1 2026:  10.578.924.699     (+38.9%)

    Top-5 Q4 por SSHPRNAMT agregado:
      NVDA:   490.915.097
      AAPL:   269.353.275
      AMZN:   226.279.466
      CMCSA:  218.445.902
      INTC:   190.031.200

    Top-5 Q1 por SSHPRNAMT agregado:
      NVDA:   819.785.015
      AAPL:   425.718.223
      AMZN:   364.363.236
      MSFT:   267.184.259
      CMCSA:  251.737.199

Los pesos agregados son reproducibles a partir de los parquets de
entrada.

### 4.7 Anomalias identificadas (estado intermedio)

En el momento de estas mediciones (2026-09-22), 3 securities del
catalogo radar (tickers SPCX, XOM, OKE) no aparecian en ningun filing
via matching directo por FIGI:

**SPCX** (BBG001SQPN65). Emisor privado (SpaceX). Sin filings 13F.
Presencia en el catalogo cuestionable. Sigue sin fuente de identidad
a 2026-09-23. Ver §13.2.

**XOM** (BBG023CY9NL0 en catalogo). Filings Q1 2026 contienen CUSIP
30231G102 con 7.575 filas. Los FIGIs historicos en filings son
BBG001S69V32, BBG000GZQBJ1, BBG000GZQ728. Ninguno coincide con el
FIGI del catalogo. **Resuelto por CUSIP** a partir del fix 7609a56:
XOM esta en `cusip_radar_crosswalk.csv` con CUSIP 30231G102 y
contribuye al TARGET contractual (§13.2).

**OKE** (BBG024TZWVS6 en catalogo). Misma situacion que XOM. CUSIP
682680103, 2.673 filas en Q1. Se resuelve via `etf_holdings.csv`
sin vigencia temporal, por lo que P61 lo marca TEMPORAL_UNVERIFIED
y queda fuera del filtro §5.5. No contribuye al TARGET (§13.2).

### 4.8 Que se ha verificado y que no

**Verificado:**
- Estructura del modulo coincide con la declarada.
- Los 242 tickers del catalogo son reproducibles por la regla.
- Existe overlap real de 239 FIGIs entre radar y filings.
- Los 24 CUSIPs de la excepcion original (pre-extension del 22-09)
  aparecen en los filings.
- Los pesos agregados son reproducibles.
- El modulo se integra unidireccionalmente en produccion.

**No verificado:**
- Que el overlap completo (resolviendo el 88% de filas sin FIGI)
  coincida con el overlap observable.
- Que el pipeline productivo vaya a integrar el modulo.

**Fuera de alcance:**
- El sistema nacio en septiembre de 2026. No hay periodos anteriores
  auditables.

---

## 5. Deuda tecnica

### 5.1 Deuda funcional

- `compute_nipc_contractual`: **ya tiene caller productivo** desde
  2026-09-24 via `pipeline_contractual.run_contractual_nipc`, invocado
  por `compute_iae_section` desde `run.py`. Reconciliacion exacta con
  §12.5 verificada via `scripts/iae_contractual_nipc_e2e.py` (PASS).
- `build_effective_reporting_snapshot` sin callers productivos.
- `scripts/iae_contractual_coverage.py` reproduce §12.3 con la cadena
  contractual completa. Su logica se consume ahora a traves de
  `pipeline_contractual` (ver §13.11).
- Cobertura unitaria de `run_contractual_nipc`: 76 stmts no cubiertos
  por tests con mock (ver §3.2). Deuda residual de cobertura.

### 5.2 Deuda de cobertura

Estado actual: 90% global, ningun fichero por debajo del 80%
(ver §3.2). El minimo es `catalog_key.py` con 82%.

Deuda residual: mejorar la cobertura de los modulos con menor
cobertura ANTES de su integracion en produccion. El criterio
concreto (que umbral, que ficheros) se fijara cuando la
integracion sea una decision activa, no hoy.

### 5.3 Deuda de datos

- Solo el 12,2% de las filas de INFOTABLE llevan FIGI poblado. Un
  fallback CUSIP -> FIGI por OpenFIGI podria ampliar la cobertura de
  identidad, pero no se ha demostrado que resuelva exhaustivamente
  los CUSIPs restantes (ver §13.1 y §12.4).
- SPCX esta en el catalogo radar pero no tiene fuente de identidad
  (ni crosswalk ni etf_holdings). Emisor privado sin filings 13F.
  No contribuye al TARGET. Ver §13.2.
- OKE se resuelve via etf_holdings sin vigencia temporal, por lo que
  el resolver lo marca TEMPORAL_UNVERIFIED y queda fuera del filtro
  §5.5 (CANONICAL AND VERIFIED). No contribuye al TARGET. Ver §13.2.

### 5.4 Deuda de trazabilidad

- Registro de bundles enviados a terceros: `outputs/BUNDLES_SENT.log`.
- Deuda semantica del nombre `NIPC`. Si el modulo se integra a
  `run.py`, la superficie visible del reporte debe usar un nombre no
  ambiguo (`observed_position_change` o similar) para evitar confusion
  con las capas de flujo del radar. Ver §10.3.
- Manifest `stock_prices.parquet.manifest.json` stale tras F-IAE-HOLIDAY-01
  (parquet reescrito el 2026-09-24 sin regenerar el manifest). El proximo
  `run.py` lo detectara como INVALID y lo regenerara.

---

## 6. Reglas de operacion

- El modulo IAE no se integra al pipeline productivo sin validacion
  funcional + beneficio del reporte.
- No se hace push a `origin/main` del codigo IAE hasta que este
  integrado y verificado en produccion.
- No se ejecuta OpenFIGI masivo. Solo consultas dirigidas.
- No se modifica el codigo sin ciclo de validacion previo.
- Toda escritura de ficheros pasa por `write_artifact_with_manifest`.

---

## 7. Anexo: comandos reproducibles

### A.1 Estado del repo

    git log --oneline -5
    git status -sb

### A.2 Inventario del modulo

    Get-ChildItem -Path src\institutional_accumulation -Recurse -File -Filter *.py

### A.3 Cobertura de tests

Criterio AST (44 ficheros del modulo, 819 casos):

    py scripts/iae_test_census.py

Cobertura de lineas sobre el universo canonico (los 819 casos):

    py scripts/iae_coverage.py

El script resuelve los 44 ficheros de test por AST (mismo criterio
que `iae_test_census.py`) y ejecuta pytest con
`--cov=src/institutional_accumulation` sobre el universo completo.
Resultado esperado a 2026-09-24: 819 passed, 90% coverage.

### A.4 Overlap sobre datos reales

    py -c "
    import pandas as pd
    cat = pd.read_csv('data/mappings/radar_target_catalog.csv', dtype=str)
    radar_scf = set(cat[cat['status']=='OK']['share_class_figi'].dropna())
    def figis(q):
        df = pd.read_parquet(f'data/sec_13f/processed/{q}/INFOTABLE.parquet',
                              columns=['FIGI'])
        return set(df['FIGI'].dropna())
    q4, q1 = figis('2025Q4'), figis('2026Q1')
    target_q4 = q4 & radar_scf
    target_q1 = q1 & radar_scf
    print(f'TARGET_PAIRWISE: {len(target_q4 & target_q1)}')
    "

### A.5 Verificacion de aislamiento

    Select-String -Path (Get-ChildItem -Path src,run.py,scripts -Recurse -File -Filter *.py).FullName `
      -Pattern "from\s+.*institutional_accumulation|import\s+.*institutional_accumulation"

---

## 8. Referencias

- Evidencia empirica: `docs/auditoria/iae/evidence/`.
- Estado del sistema: `docs/auditoria/iae/ESTADO_SISTEMA.md`.
- Estado declarado: `docs/auditoria/iae/ESTADO_DECLARADO.md`.
- Normativa general: `docs/auditoria/PROMPT_MAESTRO.md`.
- Onboarding: `docs/auditoria/TRANSFER.md`.

---

## 9. Inventario de funciones publicas

Cada funcion publica del modulo, agrupada por capa.
Formato: firma completa + primera linea del docstring.
Detalle completo en el codigo fuente.

**Nota sobre referencias 14.3.x.** Las menciones a `14.3.1`, `14.3.2`,
`14.3.4`, `14.3.5`, `14.3.6` en docstrings y firmas corresponden al
contrato P65 historico (ver `NIPC_CONTRATOS_SEMANTICOS_v1.md`), no a
una seccion de este documento. Este documento llega solo hasta §13.


### 9.1. Ingestion (sec_13f/)


#### `sec_13f\downloader.py`

- `download_13f_zip(period, dest_dir, user_agent=DEFAULT_USER_AGENT, force=False)` — Descarga el ZIP trimestral 13F de SEC a dest_dir.
- `extract_13f_zip(zip_path, dest_dir, force=False)` — Extrae los 7 TSVs esperados del ZIP a dest_dir.


#### `sec_13f\ingest.py`

- `ingest_13f(quarter, source_period, base_dir=DEFAULT_BASE_DIR, user_agent=DEFAULT_USER_AGENT, force_download=False, project_root=DEFAULT_PROJECT_ROOT)` — Ingesta completa del trimestre 13F.


#### `sec_13f\manifest.py`

- `build_manifest(*, quarter, source_period, source_url, source_filename, source_sha256, tsv_files, parquet_files, row_counts, validation, project_root=DEFAULT_PROJECT_ROOT, parser_version=PARSER_VERSION, schema_version=SCHEMA_VERSION)` — Construye dict de manifest con lineage completo a 3 niveles.
- `write_manifest(manifest, path)` — Escribe manifest a JSON indentado. Escritura atomica via .tmp.
- `read_manifest(path)` — Lee un manifest JSON. Devuelve dict.


#### `sec_13f\parser.py`

- `parse_one(tsv_path, tsv_name, *, validate=True)` — Lee un TSV y devuelve DataFrame con dtypes aplicados.
- `parse_13f(extracted_dir, *, validate=True)` — Lee los 7 TSVs de un directorio extraido.


#### `sec_13f\schema.py`

- `get_expected_columns(tsv_name)` — Devuelve la tupla de columnas esperadas para un TSV.
- `get_primary_key(tsv_name)` — Devuelve la tupla de columnas que forman la clave primaria.
- `validate_columns(tsv_name, actual_columns)` — Valida que las columnas reales coinciden con las esperadas.


#### `sec_13f\storage.py`

- `get_manifest_path(quarter, base_dir=DEFAULT_BASE_DIR)` — Ruta esperada del manifest para un trimestre.
- `write_parquets(dfs, quarter, base_dir=DEFAULT_BASE_DIR, compression='snappy')` — Escribe cada DataFrame como parquet via .tmp + rename atomico.
- `load_parquets(quarter, base_dir=DEFAULT_BASE_DIR)` — Carga los 7 parquets de un trimestre.


### 9.2. Identity 13F (sec_13f/identity/)


#### `sec_13f\identity\amendments.py`

- `order_filings(sub_df, cov_df=None)` — Devuelve SUBMISSION + AMENDMENTNO/AMENDMENTTYPE/ISAMENDMENT + _order.
- `classify_strategy(filings_group)` — Clasifica un grupo (CIK, PERIOD) segun SUBMISSIONTYPE + AMENDMENTTYPE.
- `detect_base_filing(filings_group)` — Detecta el filing base de holdings.
- `apply_amendments(dfs, *, period)` — Orquesta la canonicalizacion. Devuelve dict.
- `compute_strategy_counts(per_cik_period_df)` — Devuelve dict estrategia -> count.
- `compute_status_counts(per_cik_period_df)` — Devuelve dict status -> count.


#### `sec_13f\identity\cusip_resolver.py`

- `load_exceptions(path=DEFAULT_EXCEPTIONS_PATH)` — Carga y valida tabla de excepciones CUSIP.
- `resolve_cusip(cusip, report_period, exceptions_df)` — Resuelve un CUSIP a ticker segun report_period.
- `resolve_batch(df, report_period, exceptions_df, *, cusip_col='CUSIP')` — Anade columna `ticker_resolved` a un DataFrame con CUSIPs.


#### `sec_13f\identity\relationships.py`

- `build_filing_manager_index(submission_df, coverpage_df=None)` — Indice ACCESSION -> filing_manager_cik (SUBMISSION.CIK).
- `build_om2_seq_index(othermanager2_df)` — Indice (ACCESSION, SEQ) -> CIK desde OTHERMANAGER2.
- `build_provenance_index(othermanager_df)` — Indice de provenance desde OTHERMANAGER (SK interno).
- `classify_token(raw_token, accession, om2_index)` — Clasifica un token individual de Column 7.
- `explode_othermanager_edges(infotable_df, om2_df, *, return_metrics=False)` — Expande INFOTABLE a nivel de edge (source_line, manager).
- `compute_edge_metrics(edges_df, infotable_df=None)` — Metricas de cobertura y clasificacion (dictamen FA-2.3).
- `check_edge_uniqueness(edges_df)` — Verifica unicidad de (ACCESSION, INFOTABLE_SK, manager_sequence).
- `build_canonical_relationship(infotable_df, om2_df, submission_df, *, report_period, coverpage_df=None)` — Pipeline completo: edges + filing_manager_cik + canonical key.
- `compute_source_line_count(infotable_df)` — Cuenta source_line_id unicos (ACCESSION, INFOTABLE_SK).


#### `sec_13f\identity\sec13f_list.py`

- `parse_line(line)` — Parsea una linea fixed-width 80. Devuelve dict o None.
- `parse_official_list_text(text)` — Parsea el contenido completo del TXT oficial.
- `load_official_list(path)` — Carga un fichero TXT oficial SEC 13(f).
- `resolve_eligibility(cusips, official_df)` — Resuelve section13f_eligible (5 estados) para una lista de CUSIPs.
- `compute_eligibility_coverage(cusips, official_df)` — Metricas agregadas de elegibilidad.


#### `sec_13f\identity\security_identity.py`

- `observed_security_key(cusip)` — Devuelve la clave tecnica por periodo: cusip:<CUSIP>.
- `load_cusip_equivalence(path=DEFAULT_EQUIVALENCE_PATH)` — Carga y valida la tabla de equivalencias CUSIP.
- `load_crosswalk_internal(exceptions_path=DEFAULT_EXCEPTIONS_PATH, etf_holdings_path=DEFAULT_ETF_HOLDINGS_PATH)` — Carga el crosswalk interno combinado.
- `resolve_security_identity(cusip, report_period, *, equivalence_df=None, crosswalk_internal_df=None, figi_lookup=None)` — Resuelve identidad canonica + operational_mapping_status (P61).
- `resolve_batch_identities(cusips, report_period, *, equivalence_df=None, crosswalk_internal_df=None, figi_lookup=None)` — Version batch. Devuelve dict {cusip: resultado}.
- `compute_identity_coverage(cusips, report_period, *, equivalence_df=None, crosswalk_internal_df=None, figi_lookup=None)` — Metricas agregadas de resolucion de identidad.


#### `sec_13f\identity\temporal_filter.py`

- `filter_by_period(dfs, period=FULL_PERIOD, *, return_stats=False)` — Filtra los 7 TSVs al periodo canonico.


### 9.3. Catalog radar (identity/)


#### `identity\catalog_key.py`

- `is_valid_catalog_key(s)` — (sin docstring)
- `is_valid_entity_id(s)` — (sin docstring)
- `is_valid_sha256(s)` — (sin docstring)
- `canonical_serialization(row, columns)` — Serializa una fila como "len:name|len:value|..." por columna
- `compute_snapshot_row_uid(row, columns)` — sha256 hex (64 chars) del canonical_serialization.
- `check_schema(df)` — Verifica B1_REQUIRED_COLUMNS. Raise ValueError si falta alguna.
- `load_assignments(path)` — (sin docstring)
- `load_membership(path)` — (sin docstring)
- `load_attempted(path)` — (sin docstring)
- `validate_assignment(assignments_df, attempted_df=None)` — Devuelve dict {catalog_key: error_code}.
- `validate_membership(membership_df, assignments_df, manifest, *, snapshot_loader=None)` — Devuelve dict {(version_id, catalog_key): error_code}.


#### `identity\openfigi_client.py`

- `map_identifiers(id_type: str, values: list[str], *, exch_code: str | None=None, api_key: str | None=None, max_retry: int=2)` — Mapea una lista de identificadores a traves de OpenFIGI /v3/mapping.
- `extract_stable_identity(hit: dict | None)` — Normaliza un hit de OpenFIGI a la identidad estable que guarda el catalogo.


#### `identity\period_state.py`

- **class `PeriodState`** — Estado de una catalog_key en un periodo.
- **class `StateBuildError`** — Error en la construccion de estados.
- `build_period_state(universe, *, figi_evidence=None, weight_evidence=None, operational_evidence=None, sshprnamt_evidence=None)` — Construye state[K] para K in universe.declared_keys.
- `feasible_state(state)` — B4: identity_status == RESOLVED AND weight_status in
- `unique_figi(state)` — Devuelve el FIGI unico si identity_status==RESOLVED y figi no vacio.


#### `identity\radar_target_catalog.py`

- `load_radar_tickers(stock_prices_path: Path)` — Devuelve los tickers del radar USA (242 por construccion actual).
- `build_from_probe_result(result_json_path: Path, *, source_date: str)` — Construye el catalog leyendo un result_radar.json ya generado.
- `write_catalog(df: pd.DataFrame, out_path: Path)` — Escribe el catalog a CSV con UTF-8 sin BOM y LF.
- `coverage_summary(df: pd.DataFrame)` — Cuenta filas por status y % con shareClassFIGI.


#### `identity\target_builder.py`

- **class `TargetUniverse`** — Universo administrativo del catalogo para un periodo.
- **class `BuildTargetError`** — Error de construccion del TargetUniverse.
- `build_target(snapshot_df, membership_df, assignments_df, *, version_id, period_end, catalog_version_id, catalog_sha256)` — Materializa el TargetUniverse (§3.6).
- `extract_sshprnamt_by_figi(infotable_df, figi_by_cusip)` — Extrae SSHPRNAMT efectivo agregado por shareClassFIGI.


#### `identity\target_universe.py`

- `load_catalog(path: Path)` — Carga radar_target_catalog.csv como DataFrame.
- `build_scf_index(catalog_df: pd.DataFrame)` — Devuelve {share_class_figi: radar_ticker}.
- `resolve_cusips(cusips: list, catalog_df: pd.DataFrame, *, source_date: str, api_key: str | None=None)` — Resuelve cada CUSIP contra OpenFIGI y contra el catalogo.
- `membership_summary(df: pd.DataFrame)` — Contadores por status + % target_membership.


### 9.4. Aggregation (aggregation/)


#### `aggregation\catalog_p38_adapter.py`

- **class `AdapterError`** — Fallo en las precondiciones del flujo normativo.
- `catalog_to_p38_targets(universe_q4, universe_q1, *, state_q4, state_q1, pairwise_keys)` — Traduce TARGET administrativo a TARGET economico P38.


#### `aggregation\catalog_validator.py`

- **class `CoverageFeasibility`** — Resultado de la cadena de validaciones pre-P38.
- `check_continuity(universe_q4, universe_q1)` — PASO 5: verifica continuidad Q4 -> Q1 para K in UNION.
- `check_economic_collision(universe)` — PASO 6: detecta colisiones catalog_key -> FIGI.
- `check_full_resolution(universe, state)` — PASO 8: para todo K in universe.declared_keys,


#### `aggregation\coverage.py`

- **class `PositionRecord`** — Registro tipado de una posicion observada (F2.4 regla #4).
- `aggregate_positions_by_shareclass_figi(records, period, *, return_stats=False)` — Agrega weights por shareClassFIGI dentro de un periodo.
- `compute_contractual_coverage(target_q4, target_q1, records_q4, records_q1)` — Cobertura contractual P38 (6 metricas base) + contadores C9 de agregacion (n_received/n_verified/n_excluded) + campos C10 (coverage_available, coverage_quality). Ver §10.5.


#### `aggregation\delta_shares.py`

- `compute_reported_position_units(infotable_df, submission_df, coverpage_df=None, *, report_period, identity_results=None, return_stats=False)` — Agrega SSHPRNAMT por reported_position_unit.
- `compute_delta_shares(units_current, units_previous)` — FULL OUTER JOIN por MATCH_KEY. Devuelve DataFrame con DELTA_COLUMNS.


#### `aggregation\nipc.py`

- `compute_nipc(delta_df, *, discretion_breakdown=True)` — Calcula NIPC total y breakdown por discretion.
- `compute_coverage_pairwise(units_current, units_previous)` — 6 metricas de cobertura pairwise (spec 3.14).
- `compute_nipc_and_coverage(delta_df, units_current=None, units_previous=None, *, threshold_1=None, threshold_2=None)` — Wrapper: NIPC + breakdown + coverage pairwise + status derivado.
- `compute_nipc_contractual(delta_df, *, target_q4, target_q1, records_q4, records_q1, threshold_1=None, threshold_2=None)` — Ruta contractual P38 (F2.4).


#### `aggregation\reporting_dedup.py`

- **class `ReportingEvidence`** — Evidencia de reporting entre managers para una linea concreta.
- `classify_evidence_level(filing_manager_cik: Optional[str], reporting_for_manager_cik: Optional[str], reference_status: Optional[str], accession_representante: Optional[str]=None, accession_representado: Optional[str]=None, reference_seq: Optional[int]=None)` — Clasifica una relacion en L1 / L2 / L3 / None.
- `build_effective_reporting_snapshot(units_q4, units_q1, relationships_q4, relationships_q1, cross_filing_evidence, *, period_q4: str, period_q1: str)` — PRE-delta (Commit 2).
- `classify_reporting_transition(delta_df, effective_q4, effective_q1)` — POST-delta (Commit 3).
- `canonicalize_form13f_filenumber(value)` — 14.3.6. Canonicaliza un Form 13F File Number.
- `classify_filing_family(submissiontype, reporttype)` — 14.3.1 PASO 1. Familia documental (NOTICE / COMBINATION).
- `classify_filing_role(submissiontype)` — 14.3.1 PASO 2. Rol BASE / AMENDMENT por SUBMISSIONTYPE.
- `build_formnum_cik_mapping(coverpage_df, period)` — 14.3.5. Construye mapping FormNum normalizado -> CIK.
- `resolve_formnum_to_cik(formnum, mapping)` — 14.3.5. Resuelve un FormNum a su CIK contractual.
- `validate_amendment_chain(amendments)` — 14.3.2. Valida la cadena de amendments de una base.
- `resolve_r3(filings, *, period, scope_completeness_verified)` — 14.3.1. Determina R3(B, period) tri-state.
- `resolve_r4(a_cik, othermanager_rows, formnum_mapping)` — 14.3.4. Determina R4(A, B).
- `evaluate_l3(r1, r2, r3, r4, r5)` — 14.3. L3(A, B, S, period) = R1 AND R2 AND (R3==TRUE)


### 9.5. Modulos raiz


#### `absence.py`

- **class `AbsenceClassificationNotImplemented`** — Inferencia automatica de absence_reason diferida (contrato seccion 12.4).
- `classify_absence(*args, **kwargs)` — Inferencia automatica de absence_reason.


#### `catalog_pit.py`

- **class `CatalogPitError`** — Base de errores de B2-PIT.
- **class `CatalogNotAvailable`** — No existe snapshot que cubra period_end.
- **class `CatalogAmbiguous`** — Multiples snapshots validos cubren period_end.
- **class `SnapshotIntegrityError`** — El sha256 del snapshot no coincide con el declarado.
- **class `ManifestError`** — Manifest invalido (schema, campos, coherencia).
- `load_manifest(catalog_root)` — Carga y valida estructuralmente el catalog_manifest.json.
- `verify_snapshot_integrity(version_id, *, catalog_root)` — Verifica que sha256 recalculado == sha256 publicado.
- `list_snapshots(*, catalog_root)` — Devuelve la lista de entradas del manifest (copia superficial).
- `target_catalog_as_of(period_end, *, catalog_root)` — Selecciona el snapshot que cubre period_end.


#### `operational_universe.py`

- `build_operational_universe(infotable_df, official_df, period_iso, *, identity_results, identity_period_iso)` — Construye el operational_universe §5.5 (dictamen #65).


#### `security_type.py`

- `classify_title_of_class(title_of_class)` — Clasifica TITLEOFCLASS segun spec 3.3 + dictamen #61 Nivel B.
- `resolve_security_type(title_of_class, external_evidence=None)` — Combina TITLEOFCLASS con evidencia externa (OpenFIGI u otra).
- `coverage_stats(records)` — Cuenta estados para reporte pct_unresolved (dictamen #61 seccion 8).
- `is_operational_candidate(security_type_value, security_type_status)` — Filtro contractual §5.4 -> §5.5 (dictamen #63 seccion 9).
- `operational_universe_candidate(toc)` — Aplica el filtro de handoff a un TITLEOFCLASS crudo.


#### `temporal_validity.py`

- `resolve_source_status(source, valid_from, valid_to, period)` — Estado de una fuente para un periodo.
- `aggregate_status(entries)` — Agrega multiples (value, status) aplicando la regla Q7.


#### `timestamps.py`

- `derive_timestamps(snapshot, accession)` — Dado un snapshot y un accession, devuelve los 3 timestamps.
- `enrich_positions_with_timestamps(positions, snapshot, accession_col='ACCESSION_NUMBER')` — Enriquece un DataFrame de posiciones con los 3 timestamps.
- `build_provenance(accession=None, knowledge_date_status=None)` — Construye un dict de provenance canonico para PositionRecord.

---

## 10. Formulas exactas

Cada formula documentada aquí está extraída directamente del código fuente
y verificada contra una ejecución real sobre Q4 2025 y Q1 2026. Los números
de ejemplo provienen de esa ejecución.

Notación:
  S        security canonica cuando esta resuelta (identificada por
           shareClassFIGI). Las filas no resueltas conservan
           observed_security_key=cusip:XXX y no participan en el NIPC.
  M        manager institucional identificado por filing_manager_cik.
  D        tipo de discreción (SOLE | DFND | OTR).
  P        periodo (2025-12-31 o 2026-03-31).
  TARGET_P universo contractual para el periodo P.
  UNITS_P  (M, S, D) -> SSHPRNAMT agregado.

---

### 10.1. Unidades reportadas por posición

Función: `delta_shares.compute_reported_position_units(infotable_df, submission_df, coverpage_df, *, report_period, identity_results)`.

Entrada:
  INFOTABLE: filas de posiciones 13F con columnas ACCESSION_NUMBER,
             INFOTABLE_SK, CUSIP, SSHPRNAMT, INVESTMENTDISCRETION.
  SUBMISSION: mapping ACCESSION -> filing_manager_cik.
  identity_results: dict {CUSIP: resultado_P61} del resolver.

Procedimiento:
  1. Filtrar filas canónicas (descartar SH, PRN, CALL, PUT, no-equity).
     Variables: SSHPRNAMT_f = SSHPRNAMT * 1000 (SEC reporta en miles).
  2. Attach filing_manager_cik via ACCESSION_NUMBER.
  3. Attach identidad via CUSIP (canonical_security, security_resolution_status,
     operational_mapping_status).
  4. Filtrar por D en {SOLE, DFND, OTR}.
  5. Agrupar por (M, CUSIP, status, kind, canonical, op_status, D) y sumar.

Formula:
  units(M, S, D; P) = sum_{lineas L del manager M en P con
                            security_resolution_status(L) = estado
                            and discretion(L) = D}  SSHPRNAMT_f(L)

Salida: una fila por (report_period, filing_manager_cik, observed_security_key,
canonical_security_kind, canonical_security, operational_mapping_status,
discretion_type) con las columnas `sshprnamt_total` y `n_source_lines`.

Nota contractual: no se multiplica por el número de edges OTHERMANAGER.
La agregación es sobre source lines del filing manager.

**Ejemplo Q4 2025:** 2.360.246 filas de units, de las cuales 808.975 tienen
`canonical_security` no nulo. **Ejemplo Q1 2026:** 2.420.326 filas,
820.701 con canonical.

---

### 10.2. Delta de posiciones por manager y security

Función: `delta_shares.compute_delta_shares(units_current, units_previous)`.

MATCH_KEY = (filing_manager_cik, canonical_security, discretion_type)

Procedimiento:
  1. Split: solo filas con security_resolution_status = CANONICAL participan
     del match. El resto se emite como UNRESOLVED_IDENTITY.
  2. Agregar por MATCH_KEY sumando `sshprnamt_total`:
       cur_agg[M, S, D] = sum sshprnamt_total sobre units_current
       prev_agg[M, S, D] = sum sshprnamt_total sobre units_previous
     (esto colapsa multiples CUSIPs del mismo canonical: por ejemplo,
     CUSIP historico y CUSIP actual del mismo ticker, o CUSIP de la
     misma security con distintos codigos de mercado).

     Nota sobre identidad canonica. `canonical_security` puede tomar
     dos formas disjuntas:
       - `equity:<TICKER>` cuando la resolucion pasa por crosswalk o
         tabla de equivalencias.
       - `figi:<FIGI>` cuando la resolucion pasa por figi_lookup
         directo (p.ej. INFOTABLE con FIGI poblado).
     Ambas formas coexisten en el delta y NO se colapsan entre si por
     construccion del MATCH_KEY. GOOG (shareClassFIGI BBG009S3NB21) y
     GOOGL (shareClassFIGI BBG009S39JY5) son tickers distintos, cada
     uno con su propio `equity:` — no son un caso de colapso. Ver §13.10
     para el detalle y el impacto medido.
  3. Full outer join por MATCH_KEY.

Reglas por origen del join:
  BOTH      -> delta_shares = sshprnamt_current - sshprnamt_previous
  NEW       -> delta_shares = sshprnamt_current            (previous = 0)
  EXIT      -> delta_shares = -sshprnamt_previous           (current = 0)
  UNRESOLVED_IDENTITY -> delta_shares = None (no se fabrica valor)

Corrupción: si una fila BOTH tiene NaN en cualquiera de los dos lados, se
lanza ValueError. No se imputa silenciosamente.

**Ejemplo Q4 -> Q1 2026:**
  Delta completo: 4.066.694 filas
    BOTH: 713.865 · NEW: 106.828 · EXIT: 95.105 · UNRESOLVED: 3.150.896
  Delta filtrado al radar: 553.321 filas
    BOTH: 444.276 · NEW: 60.324 · EXIT: 48.721 · UNRESOLVED: 0

---

### 10.3. NIPC (Net Institutional Position Change)

**Nota semantica.** A pesar del nombre, NIPC NO mide flujo economico.
Mide la variacion neta de posiciones **reportadas** en 13F entre dos
periodos consecutivos (gross observed delta). No distingue apertura
o cierre de posiciones de cambios de tamano dentro de una posicion
existente. Ver tambien §13.7.

Función: `nipc.compute_nipc(delta_df, *, discretion_breakdown=True)`.

Observable = filas con match_status in {BOTH, NEW, EXIT}.

Formula:
  n_delta_observable = |{(M, S, D) : observable}|

  nipc_total = sum_{(M,S,D) observable}  delta_shares(M, S, D)

  nipc_sole  = sum_{(M,S,D) observable, D=SOLE}  delta_shares
  nipc_dfnd  = sum_{(M,S,D) observable, D=DFND}  delta_shares
  nipc_otr   = sum_{(M,S,D) observable, D=OTR}   delta_shares

Invariantes:
  nipc_total_available = (n_delta_observable > 0)
  n_unresolved_identity se reporta separado, NUNCA entra en nipc_total.
  evidence_class = "PROXY" (ruta libre).

**Ejemplo Q4 -> Q1 2026 (radar):**
  nipc_total = -4.316.734.936
  nipc_sole  = -32.484.713.330
  nipc_dfnd  = +28.317.652.408
  nipc_otr   =      -149.674.014

  n_both = 444.276 · n_new = 60.324 · n_exit = 48.721
  n_unresolved_identity = 0
  n_delta_observable = 553.321

Verificación de coherencia:
  nipc_sole + nipc_dfnd + nipc_otr = nipc_total  (dentro del error de float)

---

### 10.4. Agregación de pesos por shareClassFIGI

Función: `coverage.aggregate_positions_by_shareclass_figi(records, period, *,
return_stats=False)`.

Entrada: lista de PositionRecord, cada uno con (share_class_figi, weight,
period, operational_mapping_status).

Formula:
  agg(S; P) = sum_{r in records : r.period = P,
                           r.operational_mapping_status = VERIFIED,
                           r.share_class_figi = S}  r.weight

Solo contribuyen al peso contractual los records con status VERIFIED.
Records con status distinto de VERIFIED (TEMPORAL_UNVERIFIED, UNRESOLVED,
CONFLICT) se excluyen del agregado, y con `return_stats=True` se
publican como contadores observables (ver mas abajo). El descarte deja
de ser silencioso.

Salida:
  - return_stats=False (default): dict {shareClassFIGI: weight_total}.
  - return_stats=True: tupla (agg, stats), donde stats es un dict con
    n_received, n_period_mismatch, n_missing_figi, n_verified,
    n_excluded, excluded_by_status (dict status->count) y
    n_temporal_unverified (atajo). Los contadores cuentan RECORDS de
    entrada al filtro, no FIGIs unicos, no CUSIPs, ni pesos.

    Invariante de reconciliacion (categorias disjuntas, aplicadas en
    este orden):
      n_received = n_period_mismatch + n_missing_figi
                 + n_verified + n_excluded
    Si en algun caso n_missing_figi pudiera pertenecer tambien a
    n_excluded (no es el caso en esta implementacion), la identidad
    no debe aplicarse.

---

### 10.5. Cobertura contractual P38

Función: `coverage.compute_contractual_coverage(target_q4, target_q1, records_q4, records_q1)`.

Entradas:
  target_q4, target_q1: sets de shareClassFIGI (universo contractual por
                        periodo, construido por target_builder).
  records_q4, records_q1: listas de PositionRecord.

Definiciones intermedias:
  res_q4(S) = {r.share_class_figi : r in records_q4,
                 r.share_class_figi no vacío,
                 r.operational_mapping_status = VERIFIED}
  res_q1(S) análogo.

Formulas de cobertura por periodo (denominador TARGET, no observed):
  coverage_previous = |res_q4 ∩ TARGET_Q4| / |TARGET_Q4|   (None si vacío)
  coverage_current  = |res_q1 ∩ TARGET_Q1| / |TARGET_Q1|   (None si vacío)

Formulas pairwise:
  TARGET_PAIRWISE = TARGET_Q4 ∩ TARGET_Q1
  PAIRED          = TARGET_PAIRWISE ∩ res_q4 ∩ res_q1

  paired_security_coverage =
      |PAIRED| / |TARGET_PAIRWISE|   (None si TARGET_PAIRWISE vacío)

  paired_weighted_share_coverage =
      sum_{S in PAIRED} w(S) / sum_{S in TARGET_PAIRWISE} w(S)

  donde:
      w(S) = max(agg(S; Q4), agg(S; Q1))

  Fail-closed: si TARGET_PAIRWISE vacío, se devuelve None, NO 0.0.
  Fail-closed ponderado: si TARGET_PAIRWISE no vacío pero
    sum_{S in TARGET_PAIRWISE} w(S) = 0, entonces
    paired_weighted_share_coverage = None y
    coverage_quality = UNAVAILABLE. Nunca division por cero, nunca
    NaN, nunca interpretacion de 0%.

Estado agregado (legacy):
  coverage_status = "VALID"      si TARGET_Q4, TARGET_Q1 y TARGET_PAIRWISE
                                 tienen valores no nulos y el cálculo
                                 produce cobertura > 0 en ambos.
  coverage_status = "UNAVAILABLE" en otro caso.

Estado agregado (C10, aditivo). Se anaden dos campos que separan
"medible" de "bueno":
  coverage_available = True      si TARGET_PAIRWISE no vacio (la
                                 cobertura es medible, aunque sea 0%).
  coverage_available = False     si TARGET_PAIRWISE vacio.
  coverage_quality = "UNAVAILABLE"  si TARGET_PAIRWISE vacio, o si
                                    paired_weighted_share_coverage
                                    es None (target no vacio pero
                                    denominador ponderado = 0: no
                                    medible, no parcial).
  coverage_quality = "COMPLETE"     si paired_security_coverage >= 0.95
                                    Y paired_weighted_share_coverage >= 0.95.
  coverage_quality = "PARTIAL"      en cualquier otro caso medible.
Threshold: COVERAGE_COMPLETE_THRESHOLD = 0.95. Umbral operativo de
gobernanza del sistema (alineado con guard_coverage.py), NO una
derivacion matematica de las formulas base de P38. `coverage_status`
se conserva como campo legacy.

**Ejemplo Q4 -> Q1 2026 (radar, con crosswalk extendido):**
  TARGET_Q4 = 240 · TARGET_Q1 = 240 · TARGET_PAIRWISE = 240
  coverage_previous = 1.0
  coverage_current = 1.0
  paired_security_coverage = 1.0
  paired_weighted_share_coverage = 1.0
  coverage_status = VALID
  coverage_available = True
  coverage_quality = COMPLETE

---

### 10.6. Cobertura pairwise legacy (informe)

Función: `nipc.compute_coverage_pairwise(units_current, units_previous)`.

Versión legacy sobre units (no sobre records). Se conserva para el informe.

Definiciones:
  sec_c = {observed_security_key : presente en units_current}
  sec_p = {observed_security_key : presente en units_previous}
  mapped_c = {observed_security_key : CANONICAL AND VERIFIED en current}
  mapped_p = análogo en previous

Formulas:
  coverage_current  = |mapped_c| / |sec_c|
  coverage_previous = |mapped_p| / |sec_p|

  target_pairwise = sec_c ∩ sec_p
  both_mapped_sec = mapped_c ∩ mapped_p
  paired_security_coverage = |both_mapped_sec| / |target_pairwise|

  Para el ponderado por SSHPRNAMT: se toma max entre periodos por security.

Diferencia clave con §10.5:
  - §10.5 usa TARGET como denominador (contractual).
  - §10.6 usa observed como denominador (diagnóstico).

Se conserva §10.6 como métrica de diagnóstico del pipeline. La versión
contractual es §10.5.

---

### 10.7. Cadena contractual (diseñada, no ejecutada en producción)

  filings 13F (parquets)
        |
        v
  filter_by_period -> apply_amendments -> canonical_snapshot
        |
        v
  resolve_batch_identities (CUSIP -> canonical_security + op_status)
        |
        v
  compute_reported_position_units  (§10.1)
        |
        +--------------+
        |              |
        v              v
  compute_delta_shares (§10.2)    build_target + catalog_to_p38_targets
        |              |
        v              v
  compute_nipc_contractual (§10.5, ruta contractual)  compute_contractual_coverage (§10.5)
        |              |
        +------+-------+
               |
               v
           Resultado contractual (diseñado):
             nipc_total + coverage_*

El resultado contractual se emite como dict con:
  nipc_total, nipc_sole, nipc_dfnd, nipc_otr
  n_both, n_new, n_exit, n_unresolved_identity, n_delta_observable
  coverage_previous, coverage_current
  paired_security_coverage, paired_weighted_share_coverage
  coverage_status
  evidence_class = "CONTRACTUAL"

**Cadena E2E ejecutada en §12: PROXY (estado 2026-09-22).** La
cadena mostrada arriba describe el diseno contractual, no la
ejecucion. La ejecucion actual del motor (§12.5 y §12.6) usa
`compute_nipc` sobre delta filtrado al radar, con
`evidence_class = "PROXY"`. `compute_nipc_contractual` existe y esta
testeado pero no tiene callers productivos (ver §5.1).

**Capa auxiliar: reporting_dedup.** `build_effective_reporting_snapshot`
(`reporting_dedup.py`) opera PRE-delta, con 35 funciones de test AST /
64 casos pytest expandidos y 0 callers productivos. NO forma parte de
la cadena contractual y NO modifica
MATCH_KEY, delta_shares, NIPC ni coverage. Su frontera semantica es
REPORTING RELATIONSHIP != REPORTING NETWORK != DEDUP AUTHORIZATION
!= ECONOMIC OWNERSHIP. Se mantiene como capa de diagnostico/auditoria
separada, no como etapa del flujo productivo. Los numeros de §12.5 y
§12.6 corresponden a la cadena sin esta capa.


---

---

## 11. Mapa de tests por funcion

Cada fichero de test cubre una parte del modulo. Se listan las
funciones de test con su docstring cuando existe.

Ejecucion actual: **819 casos de test pasan** (0 fallos). El modulo
completo (44 ficheros que importan `src.institutional_accumulation`)
tiene 714 funciones de test detectables por AST. La diferencia
(819 vs 714) corresponde a casos parametrizados via
@pytest.mark.parametrize, que pytest expande a multiples casos por
funcion.

Censo del listado. La seccion 11 tiene 29 bloques `### tests/test_*.py`.
De ellos, 27 corresponden a ficheros que importan el modulo
(criterio AST) y 2 son extra: `test_build_catalog_csvs.py` (importa
`scripts/build_catalog_csvs`) y `test_iae_pipeline_report.py` (fuera
del AST por dependencia indirecta). Del universo de 42 ficheros del
modulo, 27 estan listados en §11 y 15 no aparecen:
`test_absence`, `test_b06_e2e_aggregation`, `test_b1_schema`,
`test_h692_temporal_precedence`, `test_openfigi_client`,
`test_operational_universe`, `test_p60_contract`, `test_p61_contract`,
`test_p66_contract`, `test_p66_pipeline`, `test_period_state`,
`test_position_record`, `test_security_type`, `test_temporal_validity`,
`test_timestamps`. Comprobacion: 27 listados + 15 ausentes = 42.
El conteo autoritativo del modulo completo esta en el resumen
ejecutivo (§1) y se reproduce con `scripts/iae_test_census.py`.

Nota historica (2026-09-23). El conteo anterior declaraba 641 tests,
no reproducible con ningun criterio mecanico. El conteo actual usa
el criterio AST declarado en el resumen ejecutivo.


### `tests/test_build_catalog_csvs.py` (20 tests)

- `test_assignments_existe`
- `test_membership_existe`
- `test_assignments_242_filas`
- `test_membership_242_filas`
- `test_assignments_columnas`
- `test_membership_columnas`
- `test_catalog_key_formato`
- `test_assigned_entity_id_formato`
- `test_catalog_key_unico`
- `test_assigned_entity_id_unico`
- `test_snapshot_row_uid_sha256_completo`
- `test_snapshot_row_uid_unico`
- `test_membership_catalog_keys_en_assignments`
- `test_predecessor_solo_en_filas_cambiadas` — Migracion inicial: predecessor vacio. Tras cambios de snapshot,
- `test_justification_por_fila` — 240 filas intactas = initial migration; 2 modificadas llevan
- `test_valid_to_null`
- `test_idempotencia` — Re-ejecutar main() produce el mismo output (idempotente).
- `test_canonical_serialization_estable` — Misma fila -> mismo uid, dos veces.
- `test_canonical_serialization_columna_orden_independiente` — Reordenar columnas no cambia el uid.
- `test_canonical_serialization_cambio_valor_cambia_uid`


### `tests/test_catalog_key.py` (9 tests)

- `test_A1a_misma_key_misma_entidad_ok`
- `test_A1b_key_duplicada_entidad_distinta`
- `test_A1c_intento_reasignacion`
- `test_A1d_retired_sin_reactivar_ok`
- `test_A1e_retired_reactivado`
- `test_A1f_catalog_key_unico_por_snapshot`
- `test_A1g_catalog_key_unico_global`
- `test_A1h_reescritura_falla`
- `test_A1i_historico_ok`


### `tests/test_catalog_membership.py` (14 tests)

- `test_Ma_snapshot_v1_k1`
- `test_Mb_persistencia`
- `test_Mc_dos_keys`
- `test_Md_hueco_key_fuera_assignments`
- `test_Me_duplicado_key_en_snapshot`
- `test_Mf_version_inexistente`
- `test_Mg_reordenacion`
- `test_Mh_fuera_vigencia`
- `test_Mi_retired_snapshot_posterior`
- `test_Mj_mismo_uid`
- `test_Mk_predecessor_roto`
- `test_Mm_predecessor_valido`
- `test_Mo_snapshot_abierto_assignment_cerrado`
- `test_Mp_ambos_abiertos_ok`


### `tests/test_catalog_p38_adapter.py` (16 tests)

- `test_A2a_figi_en_target`
- `test_A2b_unresolved_falla`
- `test_A2c_conflict_con_figi_falla`
- `test_A2d_solo_q4_contribuye_a_q4`
- `test_A2e_full_q4_y_q1`
- `test_A2f_pairwise_vacio_falla`
- `test_B5a_colision_falla`
- `test_B5b_sin_colision_ok`
- `test_B5_check_collision_dict`
- `test_check_continuity_ok`
- `test_check_continuity_conflicto`
- `test_full_resolution_ok`
- `test_full_resolution_unresolved_falla`
- `test_full_resolution_conflict_falla`
- `test_coverage_feasibility_enum`
- `test_records_producidos`


### `tests/test_catalog_pit.py` (16 tests)

- `test_as_of_0_snapshots_que_cubren_raise`
- `test_as_of_1_snapshot_que_cubre_ok`
- `test_as_of_multiples_snapshots_que_cubren_ambiguous`
- `test_as_of_snapshot_existente_pero_no_cubre_raise`
- `test_as_of_backdating_prohibido_q4_2025`
- `test_as_of_backdating_prohibido_q1_2026`
- `test_sha256_publicado_vs_recalculado_match`
- `test_sha256_mismatch_fail_closed`
- `test_corrupcion_simulada_detectada`
- `test_manifest_schema_version_valido`
- `test_manifest_inexistente_raise`
- `test_intervalos_semiabiertos_sin_solapamiento`
- `test_manifest_coherente_con_snapshot_sha256`
- `test_load_manifest_catalog_root_inexistente_raise`
- `test_snapshot_csv_publicado_inmutable_entre_runs`
- `test_list_snapshots_devuelve_entradas`


### `tests/test_h731_adapter_p38_compat.py` (10 tests)

- `test_compat_a_records_del_adapter_propagan_verified` — A2 c3/5: con state VERIFIED, el adapter PROPAGA VERIFIED.
- `test_compat_b_propaga_no_inventa_operational` — A2 c3/5: con state default (UNRESOLVED), el adapter PROPAGA
- `test_compat_c_p38_acepta_records_del_adapter` — COMPATIBILIDAD: con state VERIFIED, P38 acepta los records.
- `test_compat_d_regresion_bug_original` — Regresion del bug: si adapter mapeara weight_status como
- `test_compat_e_state_no_resolved_excluido` — Si state[k].identity_status != RESOLVED, el record no se
- `test_h08_adapter_no_confunde_weight_status_con_operational` — H-08 (GO #76): ortogonalidad P61.
- `test_h101_adapter_rechaza_pairwise_vacio` — Propiedad correcta: con pairwise vacio, el adapter lanza
- `test_h101_adapter_rechaza_state_no_resolved` — Propiedad correcta: si una K del pairwise tiene identity_status
- `test_h101_adapter_propaga_operational_mapping_status` — A2 c3/5 (H-10.1 CERRADO): el adapter PROPAGA
- `test_h101_adapter_propaga_weight_desde_sshprnamt` — A2 c3/5 (H-07 CERRADO): el adapter PROPAGA el peso contractual


### `tests/test_iae_gap_coverage.py` (20 tests)

- `test_extract_sshprnamt_by_figi_df_vacio`
- `test_extract_sshprnamt_by_figi_df_sin_columnas`
- `test_extract_sshprnamt_by_figi_caso_feliz`
- `test_extract_sshprnamt_by_figi_agrega_por_figi`
- `test_extract_sshprnamt_by_figi_cusip_no_mapeado_se_ignora`
- `test_extract_sshprnamt_by_figi_valores_invalidos_descartados`
- `test_extract_sshprnamt_by_figi_cusip_vacio_se_ignora`
- `test_is_valid_catalog_key`
- `test_is_valid_entity_id`
- `test_is_valid_sha256`
- `test_canonical_serialization_ordenado_y_length_prefixed`
- `test_canonical_serialization_columna_ausente_produce_vacio`
- `test_canonical_serialization_utf8_bytes_length`
- `test_load_membership_ok`
- `test_load_membership_falta_columna_raises`
- `test_load_attempted_ok`
- `test_load_attempted_falta_columna_raises`
- `test_load_radar_tickers_filtra_sufijos_no_usa`
- `test_load_catalog_ok`
- `test_load_catalog_falta_columnas_raises`


### `tests/test_iae_gap_semantic.py` (8 tests)

- `test_compute_edge_metrics_vacio`
- `test_compute_edge_metrics_cuenta_por_status`
- `test_compute_edge_metrics_denominador_cero`
- `test_compute_edge_metrics_con_infotable_calcula_source_lines`
- `test_compute_edge_metrics_source_line_invalida`
- `test_classify_title_of_class_caracterizacion` — Spec 3.3 + dictamen #61 Nivel B. Precedencia: NON_EQUITY > fondo
- `test_classify_title_of_class_precedencia_non_equity_sobre_equity` — NON_EQUITY tiene precedencia sobre EQUITY anchor.
- `test_classify_title_of_class_no_detecta_conflict` — Docstring: esta funcion NO detecta CONFLICT. Solo resolve_security_type.


### `tests/test_iae_pipeline_report.py` (4 tests)

- `test_report_snapshot_vacio` — Caso borde: snapshot sin filas. No debe romper.
- `test_report_con_datos_muestra_pct_operational`
- `test_report_u51_vacio_no_muestra_pct` — Si u51 vacio (ningun SH + PUTCALL NULL), no imprime pct.
- `test_report_identities_status_ordenados_descendente` — Etapa 2 ordena por frecuencia descendente.


### `tests/test_p38_contract.py` (33 tests)

- `test_p38_legacy_denominador_cero_unavailable` — Ruta PROXY: denominador cero -> UNAVAILABLE.
- `test_p38_legacy_usa_observed_key` — Ruta PROXY: opera sobre observed_security_key, no sobre TARGET.
- `test_p38_existe_compute_contractual_coverage` — Contrato P38: la funcion contractual existe en coverage.py.
- `test_p38_contractual_recibe_target_externo` — Contrato P38 seccion 8.3: TARGET se recibe, no se construye internamente.
- `test_p38_agrega_pesos_por_shareclass_figi` — Contrato P38 seccion 3.3: agregar ANTES de max(Q4,Q1).
- `test_p38_solo_verified_contribuye_al_peso` — F2.4 AGREG.: solo VERIFIED aporta al peso contractual.
- `test_p38_target_pairwise_interseccion_no_union` — Contrato P38: TARGET_PAIRWISE = TARGET_Q4 INTERSECT TARGET_Q1.
- `test_p38_cusip_distinto_figi_igual_paired` — Modelo A (shareClassFIGI): CUSIP distinto con FIGI comun -> PAIRED.
- `test_p38_figi_distinto_not_paired` — FIGI distinto -> TARGET_PAIRWISE vacio -> None (auditor Q5).
- `test_p38_q5_target_q4_vacio_coverage_previous_none` — Auditor Q5: TARGET_Q4 vacio -> coverage_previous UNAVAILABLE.
- `test_p38_q5_target_q1_vacio_coverage_current_none` — Auditor Q5 simetrico: TARGET_Q1 vacio -> coverage_current UNAVAILABLE.
- `test_p38_unmapped_count_es_int` — F2.4 regla #2: unmapped_count es int, no float.
- `test_p38_paired_weighted_max_por_figi_con_cobertura_asimetrica` — A2 c5/5 (H-06): paired_weighted calcula max(Q4,Q1) POR FIGI.
- `test_p38_agregacion_figi_multiples_cusip_pesos_reales` — A2 c5/5: agregacion FIGI con SSHPRNAMT reales (no weight=1.0).
- `test_p38_sshprnamt_no_doble_conteo_multiples_keys_mismo_figi` — B-06 (auditor #76): cuando N catalog_keys comparten FIGI, el
- `test_p38_sshprnamt_doble_conteo_es_reproducible_sin_fix_b06` — B-06: documenta el bug que el fix evita.
- `test_p38_compute_nipc_contractual_evidence_class` — Contrato: la ruta contractual marca evidence_class=CONTRACTUAL.
- `test_p38_compute_nipc_contractual_sin_target_error` — F2.4 regla #3: target None en ruta contractual -> error duro.
- `test_p38_compute_nipc_legacy_evidence_class_proxy` — Regla #3: la ruta legacy marca evidence_class=PROXY.
- `test_p38_coverage_quality_unavailable_si_pesos_cero` — C10 fail-closed: target pairwise no vacio + peso 0 -> UNAVAILABLE (no PARTIAL).


### `tests/test_p38_pairwise_fixture.py` (6 tests)

- `test_fixture_esta_marcado_non_production`
- `test_paired_security_coverage_con_pairwise_no_vacio` — TARGET_PAIRWISE = {A, D}; PAIRED = {A}; cobertura = 1/2.
- `test_paired_weighted_con_pesos_asimetricos` — w(A)=max(1000,1200)=1200; w(D)=max(200,200)=200.
- `test_max_q4_q1_tras_agregacion_por_figi` — Dos CUSIPs comparten FIGI_A: agregado Q4=1000, Q1=1200,
- `test_temporal_unverified_deja_de_contribuir` — FIGI_D en Q1 esta TEMPORAL_UNVERIFIED -> no entra en PAIRED.
- `test_determinismo_fixture` — Misma entrada -> mismo resultado, dos veces.


### `tests/test_radar_target_catalog.py` (8 tests)

- `test_build_from_probe_result_columnas`
- `test_build_status_ok_partial_miss`
- `test_build_source_poblado`
- `test_build_ordenado_por_ticker`
- `test_coverage_summary`
- `test_coverage_summary_vacio`
- `test_write_catalog_round_trip`
- `test_write_catalog_crea_directorio`


### `tests/test_sec_13f_amendments.py` (31 tests)

- `test_strategies_definidas`
- `test_status_definidos`
- `test_anomalias_definidas`
- `test_order_original_antes_que_amendment`
- `test_order_por_amendment_no`
- `test_order_amendment_no_gana_a_fecha` — AMENDMENTNO=2 con fecha anterior debe ir DESPUES de AMENDMENTNO=1.
- `test_base_hr_unico`
- `test_base_notice_sin_holdings`
- `test_base_ambiguo`
- `test_strategy_single_hr`
- `test_strategy_single_notice`
- `test_strategy_hr_plus_restatement`
- `test_classify_strategy_guarda_invariante_hr_plus_restatement` — P70: si base==HR_PLUS_RESTATEMENT con len!=2, abortar.
- `test_strategy_hr_plus_new_holdings`
- `test_strategy_hr_chain_restatement`
- `test_strategy_hr_composite`
- `test_strategy_notice_amended`
- `test_apply_single_hr`
- `test_apply_restatement_reemplaza`
- `test_apply_new_holdings_compone`
- `test_apply_nt_no_contribuye`
- `test_apply_nt_a_new_holdings_anomalia`
- `test_apply_hr_with_amendment_flags_anomalia`
- `test_apply_ambiguous_amendment_order`
- `test_apply_per_cik_period_estructura`
- `test_apply_lineage_estructura`
- `test_apply_chain_restatement`
- `test_apply_hr_composite`
- `test_strategy_counts`
- `test_status_counts`
- `test_apply_no_muta_inputs`


### `tests/test_sec_13f_cusip_resolver.py` (17 tests)

- `test_columnas_requeridas_constante`
- `test_load_exceptions_fichero_no_existe_devuelve_vacio`
- `test_load_exceptions_columnas_faltantes_lanza`
- `test_valid_from_mayor_que_valid_to_lanza`
- `test_source_prohibido_lanza`
- `test_verified_by_invalido_lanza`
- `test_solape_lanza`
- `test_resolve_cusip_match_exacto`
- `test_resolve_cusip_sin_match_devuelve_none`
- `test_resolve_cusip_periodo_fuera_de_vigencia_devuelve_none`
- `test_resolve_cusip_vigencia_abierta_valid_to_null`
- `test_resolve_cusip_multiples_variantes_temporales`
- `test_resolve_cusip_ambiguo_lanza`
- `test_resolve_cusip_empty_df_devuelve_none`
- `test_resolve_batch_anade_columna`
- `test_resolve_batch_no_muta_original`
- `test_resolve_batch_sin_columna_cusip_lanza`


### `tests/test_sec_13f_delta_shares.py` (28 tests)

- `test_match_key_y_estados_definidos`
- `test_discretions_validas`
- `test_multi_manager_no_multiplica_shares` — OTHERMANAGER con varios valores -> SSHPRNAMT entra UNA SOLA VEZ.
- `test_discretion_separada` — SOLE + DFND en mismo filing -> 2 units.
- `test_reported_position_unit_key_estable_entre_periodos` — Misma (filing_manager, canonical_security, discretion) entre Q4/Q1.
- `test_filtro_sh_null_excluye_prn_y_putcall`
- `test_discretion_no_excluye_dfnd`
- `test_discretion_invalida_se_excluye`
- `test_sin_datetime_now` — Verifica por AST que no se llama a now()/today().
- `test_reproducible_mismo_input_mismo_output`
- `test_cusip_change_same_canonical_security` — Q4 CUSIP_A y Q1 CUSIP_B -> mismo canonical_security -> BOTH, delta real.
- `test_multi_edge_does_not_duplicate_delta` — Varias source lines del mismo canonical -> 1 unit, delta unico.
- `test_unresolved_mapping_blocks_pair_match` — Security mapeada en Q1 pero no en Q4 -> UNRESOLVED_IDENTITY, no NEW.
- `test_new_and_exit_zero_baseline_current` — NEW -> previous=0. EXIT -> current=0. Nunca imputar.
- `test_sin_columnas_requeridas_lanza`
- `test_no_muta_inputs`
- `test_reported_position_unit_columns_presentes`
- `test_filter_canonical_stats_presentes` — P32: _filter_canonical devuelve (df, stats) con 5 claves.
- `test_filter_canonical_cuenta_invalid_sshprnamt` — P32: SSHPRNAMT no parseable se cuenta en n_dropped_invalid_sshprnamt.
- `test_compute_reported_position_units_return_stats` — P32: kwarg return_stats=True devuelve (DataFrame, stats).
- `test_delta_shares_both_con_nan_aborta` — P14-BIS: fila BOTH con NaN -> ValueError, no imputar 0 silencioso.
- `test_delta_shares_columns_presentes`
- `test_p63_amendment_evita_exit_falso` — P63 / F2.4 Regla 6: amendments resueltos ANTES de delta.
- `test_p63_othermanager_produce_exit_mas_new` — P63 / F2.4 Regla 7: 13F admite Other Manager / Combination Report.
- `test_p64_semantica_gross_observed_delta` — P64 / F2.4: delta_shares es GROSS OBSERVED DELTA.
- `test_p64_delta_no_lleva_columnas_economic_mechanical` — P64 / F2.4: DELTA_COLUMNS no expone economic ni mechanical.
- `test_p64_split_no_se_etiqueta_como_economico` — P64 / F2.4: escenario split 1:2 -> delta=+100 sin etiqueta economica.
- `test_p64_cusip_change_es_identidad_no_corporate_action` — P64 / F2.4: CUSIP change -> continuidad de IDENTIDAD, no deteccion


### `tests/test_sec_13f_downloader.py` (15 tests)

- `test_build_url_valido`
- `test_build_url_vacio_lanza`
- `test_validate_user_agent_valido`
- `test_validate_user_agent_vacio_lanza`
- `test_validate_user_agent_whitespace_lanza`
- `test_sha256_conocido`
- `test_http_4xx_no_retry`
- `test_http_5xx_retry_agotado`
- `test_http_5xx_recupera`
- `test_download_cache_hit`
- `test_download_cache_hit_force_redescarga`
- `test_extract_ok`
- `test_extract_falta_tsv`
- `test_extract_extras_ignorados`
- `test_extract_zip_no_existe`


### `tests/test_sec_13f_ingest.py` (8 tests)

- `test_ingest_end_to_end_estructura`
- `test_ingest_crea_7_parquets`
- `test_ingest_crea_manifest_en_disco`
- `test_ingest_manifest_3_niveles_hash`
- `test_ingest_manifest_validation_flags`
- `test_ingest_manifest_quarter_y_period_separados`
- `test_ingest_segunda_llamada_reutiliza_cache`
- `test_ingest_estructura_directorios`


### `tests/test_sec_13f_list.py` (26 tests)

- `test_estados_definidos`
- `test_layout_constantes`
- `test_parse_line_basico`
- `test_parse_line_status_added`
- `test_parse_line_status_deleted`
- `test_parse_line_status_desconocido_devuelve_none`
- `test_parse_line_vacio_devuelve_none`
- `test_parse_line_ignora_pos_80`
- `test_parse_line_padding_si_short` — P51: linea corta (< 70): no crashea, se preserva como anomalia.
- `test_parse_line_demasiado_corta_status_none` — P51: linea < 70 chars -> status=None (anomalia), no ACTIVE.
- `test_parse_line_exactamente_70_chars_parsable` — P51: linea de exactamente 70 chars se parsea con normalidad.
- `test_parse_official_list_text_varias_lineas`
- `test_parse_official_list_text_preserva_order_y_line_no`
- `test_load_official_list_no_existe`
- `test_load_official_list_roundtrip`
- `test_resolve_not_in_list`
- `test_resolve_active`
- `test_resolve_added`
- `test_resolve_deleted`
- `test_resolve_conflict_a_y_d` — Mismo CUSIP con *A* y *D* -> CONFLICT (no 'ultimo gana').
- `test_resolve_conflict_por_status_desconocido` — Linea con status raw invalido -> status=None -> CONFLICT.
- `test_resolve_official_df_empty`
- `test_resolve_official_df_none`
- `test_resolve_option_indicator_capturado`
- `test_compute_eligibility_coverage_conteos`
- `test_compute_eligibility_coverage_vacio`


### `tests/test_sec_13f_manifest.py` (13 tests)

- `test_build_manifest_estructura`
- `test_build_manifest_tres_niveles_hash`
- `test_build_manifest_validation_flags`
- `test_build_manifest_rutas_relativas`
- `test_build_manifest_falla_si_parquet_no_existe`
- `test_build_manifest_falla_si_tsv_no_existe`
- `test_write_manifest_crea_dirs`
- `test_write_manifest_atomico_sin_tmp`
- `test_write_manifest_json_valido`
- `test_read_manifest_roundtrip`
- `test_default_project_root_apunta_a_raiz_repo` — P18/P26: el default no es literal; apunta a la raiz del repo.
- `test_default_project_root_no_es_literal_hardcoded` — P18/P26: el codigo fuente no contiene path absoluto hardcoded.
- `test_read_manifest_no_existe`


### `tests/test_sec_13f_nipc.py` (27 tests)

- `test_status_nipc_definidos`
- `test_discretion_types`
- `test_nipc_total_suma_solo_observable`
- `test_nipc_breakdown_por_discretion`
- `test_nipc_sin_breakdown`
- `test_nipc_vacio`
- `test_nipc_unresolved_no_contribuye`
- `test_coverage_pairwise_ambos_mapeados`
- `test_coverage_pairwise_parcial` — P38: denominador = interseccion de TARGET, no union de observadas.
- `test_coverage_pairwise_ambos_vacios`
- `test_coverage_pairwise_solo_current`
- `test_status_insufficient_sin_thresholds`
- `test_status_ready_requiere_ambos_thresholds`
- `test_status_no_ready_si_solo_uno_pasa`
- `test_status_conflict_si_unidades_tienen_conflict`
- `test_status_unresolved_sin_unidades`
- `test_compute_nipc_and_coverage_devuelve_todas_las_claves`
- `test_nipc_total_available_true_con_datos` — P31: nipc_total_available=True cuando hay al menos un delta observable.
- `test_nipc_total_available_false_sin_datos` — P31: nipc_total_available=False cuando no hay observables.
- `test_sin_datetime_now`
- `test_end_to_end_units_delta_nipc` — Q4->Q1 con dos managers y tres CUSIPs. NIPC observable.
- `test_p38_denominador_interseccion_no_union` — P38: dos securities solo en Q4 / solo en Q1 no entran al denominador.
- `test_p38_denominador_ambos_presentes` — P38: cuando A esta en ambos, A es TARGET_PAIRWISE.
- `test_q5_coverage_status_unavailable_cuando_vacio` — Q5: cobertura no medible -> None + UNAVAILABLE, no 0.0.
- `test_p61_temporal_unverified_excluido_del_conjunto_operacional` — P61: TEMPORAL_UNVERIFIED no entra a RESOLVED ni a cobertura.
- `test_p61_verified_entra_al_conjunto_operacional`
- `test_q6_unmapped_count_nombres`


### `tests/test_sec_13f_parser.py` (15 tests)

- `test_parse_one_submission_ok`
- `test_parse_one_fechas_convertidas`
- `test_parse_one_fecha_invalida_coerce_nat`
- `test_parse_one_dtypes_int64_nullable`
- `test_parse_one_dtypes_float64_nullable`
- `test_parse_one_tsv_faltante`
- `test_parse_one_tsv_no_reconocido`
- `test_parse_one_columnas_incorrectas_validate_true`
- `test_parse_one_columnas_incorrectas_validate_false`
- `test_parse_13f_completo`
- `test_parse_13f_dir_no_existe`
- `test_parse_13f_falta_tsv`
- `test_parse_13f_validate_false`
- `test_parse_one_valor_no_numerico_en_int_coerce_nan` — Valor no parseable en columna Int64 -> NaN, no ValueError.
- `test_parse_one_valor_no_numerico_en_float_coerce_nan` — Valor no parseable en columna Float64 -> NaN, no ValueError.


### `tests/test_sec_13f_relationships.py` (33 tests)

- `test_estados_definidos`
- `test_seq_max_domain_number_3`
- `test_terminos_prohibidos`
- `test_classify_token_resolved`
- `test_classify_token_unmapped`
- `test_classify_token_zero_embebido`
- `test_classify_token_non_numeric`
- `test_classify_token_out_of_domain`
- `test_classify_token_none_literal`
- `test_explode_no_reference_cero`
- `test_explode_no_reference_none`
- `test_explode_null`
- `test_explode_single_resolved`
- `test_explode_multi_manager_tres_edges`
- `test_explode_zero_embebido_en_lista`
- `test_explode_non_numeric_en_lista`
- `test_explode_out_of_domain`
- `test_explode_unmapped_missing`
- `test_explode_source_line_id_preservado`
- `test_explode_no_divide_economicamente` — La expansion multi-edge NO crea filas con VALUE/SSHPRNAMT.
- `test_explode_sin_columnas_lanza`
- `test_metrics_basico`
- `test_metrics_resolution_rate`
- `test_check_edge_uniqueness_sin_duplicados`
- `test_build_canonical_pipeline_completo`
- `test_build_canonical_key_formato`
- `test_build_canonical_no_muta`
- `test_build_canonical_sin_terminos_prohibidos`
- `test_dfnd_preservado`
- `test_build_filing_manager_index`
- `test_build_om2_seq_index`
- `test_build_provenance_index`
- `test_compute_source_line_count`


### `tests/test_sec_13f_reporting_dedup.py` (35 tests)

- `test_transitions_definidas`
- `test_dedup_reasons_definidos`
- `test_dedup_decisions_definidas`
- `test_evidence_levels_definidos`
- `test_evidence_sources_definidos`
- `test_forbidden_terms_no_economic_owner` — P65 v3: economic_owner_cik PROHIBIDO.
- `test_reporting_evidence_dataclass_no_economic_owner` — ReportingEvidence NO tiene campo economic_owner.
- `test_dedup_audit_columns_completas` — dedup_audit debe incluir las columnas obligatorias P65 v3.
- `test_classify_l1_solo_resolved` — RESOLVED sin representante -> L1.
- `test_classify_l2_representante` — RESOLVED + accession representante -> L2.
- `test_classify_l3_evidencia_cruzada` — RESOLVED + representante + representado -> L3.
- `test_classify_none_si_no_resolved` — reference_status != RESOLVED -> None.
- `test_p65_sin_l3_keep_silencioso` — Sin evidencia cruzada L3, dos units mismo security -> KEEP ambos.
- `test_p65_l3_unidireccional_es_overlap_unresolved` — P65 v3: L3 unidireccional + coexistencia -> OVERLAP_UNRESOLVED + KEEP.
- `test_p65_l3_reciproco_reporting_conflict` — A->B y B->A -> ambiguedad -> KEEP ambos + REPORTING_CONFLICT.
- `test_p65_distinta_discretion_no_interactua` — SOLE vs DFND -> distintos grupos, sin dedup.
- `test_p65_misma_manager_no_dedup` — Mismo filing_manager_cik -> sin dedup.
- `test_p65_dedup_audit_columnas` — El audit trail tiene exactamente las columnas obligatorias.
- `test_p65_units_vacio_no_revienta` — Con units vacios, devuelve vacios sin error.
- `test_p65_handoff_positivo` — Q4 (A filing, B reporting) + Q1 (B filing, B reporting) -> HANDOFF.
- `test_p65_sin_reporting_for_es_null` — Sin reporting_for_manager_cik -> NULL.
- `test_p65_reporting_for_cambia_null` — Q4 reporting=B, Q1 reporting=C -> NULL.
- `test_p65_both_no_se_toca` — BOTH no debe marcarse HANDOFF.
- `test_p65_handoff_no_cambia_match_status_ni_delta` — El HANDOFF no toca match_status ni delta_shares.
- `test_p65_delta_vacio_no_revienta` — delta vacio -> devuelve con columnas nuevas.
- `test_p65_handoff_ambiguedad_null` — Q4 con dos managers (A->B y D->B) + Q1 con B->B -> ambiguedad -> NULL.
- `test_p65_matrix_4_same_network_separate_hr_keep` — Test 4: HR separados bajo mismo control comun -> KEEP.
- `test_p65_matrix_5_resolved_sin_scope_keep` — Test 5: RESOLVED sin scope de security (sin evidencia L3) -> KEEP.
- `test_p65_matrix_12_exit_new_network_only_null` — Test 12: EXIT+NEW con network pero sin reporting_for -> NULL.
- `test_p65_matrix_13_l3_sin_overlap_resuelto_no_drop` — Test 13: Column 7 -> B + B HR mismo security + overlap -> KEEP.
- `test_p65_matrix_15_partition_not_deduped` — Test 15: partition no resuelta -> KEEP (no DROP).
- `test_classify_filing_family_caracterizacion` — 14.3.1 PASO 1 - familia documental (NOTICE / COMBINATION / None).
- `test_classify_filing_role_caracterizacion` — 14.3.1 PASO 2 - rol BASE / AMENDMENT / None.
- `test_evaluate_l3_true_cuando_todas_las_condiciones_se_cumplen` — 14.3 - L3 True cuando R1 AND R2 AND (R3==TRUE) AND (R4==MATCH) AND R5.
- `test_evaluate_l3_false_y_no_colapso_estados` — 14.3 - L3 False cuando cualquier R falla. N/D y CONFLICT no se


### `tests/test_sec_13f_schema.py` (13 tests)

- `test_expected_files_son_7`
- `test_expected_columns_tiene_7_claves`
- `test_primary_keys_tiene_7_claves`
- `test_infotable_primary_key`
- `test_othermanager_primary_key`
- `test_validate_columns_ok`
- `test_validate_columns_faltantes`
- `test_validate_columns_extras`
- `test_get_expected_columns_sin_extension`
- `test_get_expected_columns_con_extension`
- `test_get_expected_columns_inexistente`
- `test_get_primary_key_inexistente`
- `test_schema_version`


### `tests/test_sec_13f_security_identity.py` (42 tests)

- `test_estados_definidos`
- `test_kinds_definidos`
- `test_no_ticker_como_canonical_kind` — KIND no incluye 'TICKER' ni 'CANONICAL_TICKER'.
- `test_observed_security_key_formato`
- `test_normalize_canonical`
- `test_load_cusip_equivalence_no_existe_devuelve_vacio`
- `test_load_cusip_equivalence_columnas_faltantes`
- `test_load_cusip_equivalence_valid_from_mayor`
- `test_load_cusip_equivalence_verified_by_invalido`
- `test_load_cusip_equivalence_source_prohibido`
- `test_load_cusip_equivalence_solape`
- `test_load_cusip_equivalence_ok`
- `test_resolve_cusip_vacio_unresolved`
- `test_resolve_sin_fuentes_observed_only`
- `test_resolve_equivalence_unica`
- `test_resolve_equivalence_periodo_fuera`
- `test_resolve_equivalence_ambigua`
- `test_resolve_equivalence_normaliza_ticker_a_equity`
- `test_resolve_equivalence_respeta_figi_prefix`
- `test_resolve_crosswalk_internal`
- `test_resolve_crosswalk_sin_vigencia_activo` — etf_holdings no tiene vigencia; debe ser activo siempre.
- `test_resolve_crosswalk_conflicto_varios_tickers`
- `test_resolve_precedencia_equivalence_sobre_crosswalk`
- `test_resolve_figi_lookup`
- `test_resolve_figi_lookup_falla_se_ignora`
- `test_resolve_figi_no_se_usa_si_equivalence_resuelve`
- `test_resolve_no_muta_inputs`
- `test_resolve_batch`
- `test_compute_identity_coverage`
- `test_compute_identity_coverage_vacio`
- `test_p60_normalize_canonical_ticker`
- `test_p60_normalize_canonical_figi`
- `test_p60_normalize_canonical_cusip_no_produce`
- `test_p60_normalize_canonical_isin_no_produce`
- `test_p60_identity_type_obligatorio`
- `test_p60_identity_type_invalido`
- `test_p61_equivalence_da_verified`
- `test_p61_crosswalk_da_temporal_unverified` — P61 / F2.4: etf_holdings (sin vigencia) -> TEMPORAL_UNVERIFIED.
- `test_p61_sin_resolucion_da_unresolved`
- `test_p61_figi_lookup_da_temporal_unverified`
- `test_p60_csv_sin_identity_type_error` — P60 / F2.4: sin columna identity_type, la fuente es invalida.
- `test_q8_identity_type_figi` — Con identity_type=FIGI, kind y canonical correctos.


### `tests/test_sec_13f_storage.py` (10 tests)

- `test_write_parquets_devuelve_metadata`
- `test_write_parquets_dict_vacio_lanza`
- `test_write_parquets_no_dataframe_lanza`
- `test_write_parquets_crea_directorios`
- `test_write_parquets_no_deja_tmp`
- `test_write_parquets_idempotente`
- `test_load_parquets_roundtrip`
- `test_load_parquets_dir_no_existe`
- `test_load_parquets_falta_uno`
- `test_get_manifest_path`


### `tests/test_sec_13f_temporal_filter.py` (13 tests)

- `test_imports_publicos`
- `test_sin_submission_lanza_keyerror`
- `test_submission_sin_periodofreport_lanza`
- `test_period_invalido_lanza_valueerror`
- `test_filtro_conserva_solo_periodo`
- `test_filtro_propaga_a_los_6_tsvs_derivados`
- `test_filtro_sin_filings_devuelve_dict_vacio_pero_con_claves`
- `test_filtro_idempotente`
- `test_filtro_no_muta_dfs_originales`
- `test_stats_conteos_basicos`
- `test_stats_submissiontype_breakdown`
- `test_return_stats_false_devuelve_solo_dict`
- `test_uses_periodofreport_no_reporcalendar` — El filtro usa SUBMISSION.PERIODOFREPORT (dictamen Q-A).


### `tests/test_target_builder.py` (8 tests)

- `test_build_target_basico`
- `test_build_target_version_metadata`
- `test_membership_no_cubre_uid`
- `test_catalog_key_no_en_assignments`
- `test_membership_vacio_para_version`
- `test_schema_error_snapshot_sin_columnas`
- `test_vinculacion_por_uid_no_por_posicion`
- `test_target_universe_es_frozen`


### `tests/test_target_universe.py` (8 tests)

- `test_build_scf_index_ignora_nulos`
- `test_resolve_cusip_target`
- `test_resolve_cusip_no_en_radar`
- `test_resolve_cusip_no_id`
- `test_resolve_cusip_error`
- `test_resolve_ordena_y_dedup`
- `test_membership_summary`
- `test_membership_summary_vacio`


**Cobertura de este listado (parcial).** §11 tiene 29 bloques. De
ellos, 27 corresponden a ficheros del modulo (criterio AST) y 2 son
extra. La suma de funciones de test detectables por AST en los 29
bloques es 506 (492 hasta la revision 2026-09-22; +14 en
`test_p38_contract.py` por C9+C10+fix fail-closed).

El modulo completo tiene 44 ficheros que importan
`src.institutional_accumulation`. 17 no aparecen en §11:
`test_absence`, `test_b06_e2e_aggregation`, `test_b1_schema`,
`test_h692_temporal_precedence`, `test_openfigi_client`,
`test_operational_universe`, `test_p60_contract`, `test_p61_contract`,
`test_p66_contract`, `test_p66_pipeline`, `test_period_state`,
`test_pipeline_contractual`, `test_position_record`,
`test_security_type`, `test_temporal_validity`, `test_timestamps`,
`test_update_sec_13f`.

Este listado tiene valor como inventario nominal de funciones de test,
no como conteo exhaustivo. Conteo autoritativo del modulo (44 ficheros,
819 casos pytest, 714 funciones AST): §1 (resumen ejecutivo) y
`scripts/iae_test_census.py`.


---

## 12. Validacion end-to-end Q4 2025 / Q1 2026

**Nota (2026-09-24).** La validacion documentada en esta seccion cubre
el par Q4 2025 -> Q1 2026. Q2 2026 esta publicado por SEC pero pendiente
de descarga (bloqueo rate limit 2026-09-23; reintento programado).
Cuando se descargue, el motor detectara automaticamente el par mas
reciente (Q1 2026 -> Q2 2026) y la seccion IAE del reporte reflejara
ese nuevo delta. Ver §13.11 para el flujo automatico.

Esta seccion documenta la validacion completa del motor sobre datos reales
de los dos trimestres disponibles. Todos los números fueron medidos el
2026-09-22/23 sobre el repositorio en HEAD 7caa86b, con pandas 2.3.3.
La primera ejecución se hizo sobre b817858 (2026-09-22T22:20:44Z);
la cadena A+B+C posterior no modifica la ruta E2E (delta_shares + nipc)
y el motor fue reejecutado sobre 7caa86b (2026-09-22T23:48:30Z) con
resultado identico. Los sha256 de INFOTABLE no cambian entre ejecuciones.

Cadena forense (A.1, ver `scripts/iae_reconciliation_b1.py`):

    sha256 INFOTABLE 2025Q4: 207621CF7D0C0DB44B3BEE6232501AC9FD07B4AF9ABF6EDF7946C6BF9E009536
    sha256 INFOTABLE 2026Q1: DF4A4A5C25CE45E68768CE037B97372EF46B059C782F6C335F0B3FA37950C7D2
    timestamp UTC:           2026-09-22T23:48:30Z (reejecucion sobre 7caa86b)
    sha256 json B1:          outputs/audit/b1_reconciliation/20260922T234830Z_b1_final.json

### 12.1. Estado inicial (antes de los fixes)

Ejecucion del probe end-to-end antes de extender el crosswalk:

    Q4 2025: 0 filas oper, 0 tickers equity
    Q1 2026: 29.549 filas oper, 24 tickers equity
    TARGET_Q4 = 0 · TARGET_Q1 = 22 · TARGET_PAIRWISE = 0
    coverage_status = UNAVAILABLE

Causa: el crosswalk interno solo cubria 24 CUSIPs (cusip_ticker_exceptions)
mas 519 ETFs (etf_holdings). Los 242 tickers del radar no resolvian.

### 12.2. Crosswalk extendido

Se construyo `data/mappings/cusip_to_radar_figi.csv` cruzando filings
(CUSIP+FIGI) contra el catalogo radar (shareClassFIGI). Resultado:
**243 filas, 239 tickers unicos**.

Se extendio `cusip_radar_crosswalk.csv` de 24 a 246 filas, asignando
`valid_from=2025-12-31` y `valid_to=2026-03-31` a todas las entradas del
radar.

**Ficheros de mappings relevantes.** Tres ficheros con nombres
parecidos, contenido distinto:

    fichero                              filas  contenido
    -----------------------------------  -----  ----------------------------
    cusip_to_radar_figi.csv                243  CUSIP -> radar_ticker + FIGI
    cusip_radar_crosswalk.csv              246  CUSIP -> ticker + vigencia
                                                 (crosswalk del resolver P61)
    radar_target_catalog.csv               242  catalogo radar canonico

`cusip_to_radar_figi.csv` excluye DD/HON/XOM (3 CUSIPs que si estan en
el crosswalk). `cusip_radar_crosswalk.csv` incluye todos los mapeos
activos, con vigencia temporal y provenance.

**Tabla de reconciliacion de universos.** Los distintos numeros que
aparecen en esta seccion y en §12.5 transicionan asi:

    etapa                                  valor   razon
    -------------------------------------  ------  ----------------------------------
    crosswalk filas                          246   fichero cusip_radar_crosswalk.csv
    crosswalk tickers unicos                 242   AMCR/AMZN/LRCX/MU con 2 CUSIPs (-4)
    radar_figi filas                         243   excluye DD/HON/XOM (-3)
    radar_figi tickers unicos                239   AMCR/AMZN/LRCX/MU duplicados (-4)
    catalogo radar                           242   radar_target_catalog.csv (canonico)

Los 3 CUSIPs del crosswalk que no estan en `cusip_to_radar_figi.csv`
son DD (26614N102), HON (438516106) y XOM (30231G102), extendidos a
Q4 2025 por el fix 7609a56. Los 4 tickers con doble CUSIP en el radar
son AMCR (G0250X149, G0250X107), AMZN (023135906, 023135106),
LRCX (512807108, 512807306) y MU (595112103, 595112903).

### 12.3. Resultado del probe end-to-end

Reproducido con `scripts/iae_contractual_coverage.py`. El script
construye los TargetUniverse por periodo con las keys observadas en
cada uno (VERIFIED + con CUSIP resoluble por crosswalk), puebla
`build_period_state` con `operational_evidence` y `sshprnamt_evidence`,
invoca el adapter y `compute_contractual_coverage`.

    === Operational universes por periodo ===
    Q4 2025: 632.339 filas oper
    Q1 2026: 641.457 filas oper

    === Sub-universos por periodo ===
    universe catalogo:            242 keys
    Q4 keys observadas:           240
    Q1 keys observadas:           240
    pairwise (interseccion):      240
    ausentes:                     OKE, SPCX (ver §13.2)

    === Adapter (catalog_to_p38_targets) ===
    feasibility:                  FEASIBLE
    TARGET_Q4:                    240
    TARGET_Q1:                    240
    records_q4:                   240
    records_q1:                   240

    === compute_contractual_coverage ===
    coverage_previous:            1.0
    coverage_current:             1.0
    paired_security_coverage:     1.0
    paired_weighted_share_coverage: 1.0
    coverage_status:              VALID
    coverage_available:           True
    coverage_quality:             COMPLETE

    === Contadores C9 (observabilidad del filtrado) ===
    n_received_q4:                240
    n_verified_q4:                240
    n_temporal_unverified_q4:     0
    n_excluded_q4:                0
    excluded_by_status_q4:        {}
    n_received_q1:                240
    n_verified_q1:                240
    n_temporal_unverified_q1:     0
    n_excluded_q1:                0
    excluded_by_status_q1:        {}

    === unmapped_count legacy ===
    unmapped_count_previous:      0
    unmapped_count_current:       0

Nota de alcance (C9). En esta ejecucion los contadores de exclusion
(n_excluded_q4, n_excluded_q1, excluded_by_status_*) valen 0: los 240
records de cada periodo son VERIFIED, sin TEMPORAL_UNVERIFIED ni otros
estados. La E2E demuestra que el flujo nominal no excluye registros,
pero NO ejercita el camino de exclusion sobre datos reales. El
comportamiento del contador cuando hay registros excluidos esta
cubierto por tests con fixture (test_p38_*), no por esta ejecucion.

Semantica del TARGET. El TARGET contractual de cada periodo es el
sub-universo de keys del catalogo radar con observacion VERIFIED en el
periodo. Las 2 keys sin observacion (OKE, SPCX) se excluyen del
sub-universo antes del adapter. Esto alinea la cobertura con lo que el
motor realmente construye (§13.5).

Distincion importante: la cobertura 1.0 declarada en esta seccion es
`target_internal_coverage` (endogena al TARGET construido: todo
elemento del TARGET tiene evidencia VERIFIED por construccion). NO es
lo mismo que `catalog_coverage_declared`, la cobertura del catalogo
radar declarado (240 de 242 = 99,17%). Ver §13.5 para la tabla y la
nomenclatura canonica.

Nota historica (2026-09-23). Los valores anteriores de esta seccion
(Q4 621.046 filas oper, catalog_keys Q4 239, TARGET_PAIRWISE 239)
provenian de una ejecucion pre-fix del crosswalk, antes de extender
`valid_from` a Q4 2025 para DD/HON/XOM (commit 7609a56). No eran
reproducibles con el codigo actual. Los valores actuales se reproducen
con `scripts/iae_contractual_coverage.py` sobre HEAD 7caa86b.

### 12.4. Validacion externa con OpenFIGI

La validacion externa se hizo en dos pasos complementarios.

**Paso 1 - descarte de perdidas (2026-09-22).** Sample aleatorio
(seed=42) de 100 CUSIPs entre los 35.123 sin cobertura en el crosswalk.
Consulta a OpenFIGI por ID_CUSIP.

    Con hit:          44/100
    Errores:          56/100
    En tickers del radar: 0/100

Los 56 errores se desglosan:
  26x "Invalid idValue format" (formato CUSIP invalido)
  30x "No identifier found" (bonos, munis, foreign issuers)

En la muestra aleatoria, ninguno de los 100 CUSIPs no cubiertos por
el crosswalk resolvio mediante OpenFIGI a un ticker del radar. Esto
aporta evidencia negativa sobre la muestra, pero NO demuestra
cobertura exhaustiva fuera del crosswalk ni la correccion individual
de los 246 mapeos del crosswalk.

**Paso 2 - validacion positiva de los mapeos (2026-09-22).**
Consulta dirigida de los 243 CUSIPs de `cusip_to_radar_figi.csv` a
OpenFIGI (ID_CUSIP, exchCode=US). Script:
`scripts/iae_validate_crosswalk_openfigi.py`.

    CUSIPs validados:          243
    con respuesta valida:      227
    match shareClassFIGI:      227  (100% de los resolubles)
    mismatch shareClassFIGI:     0
    sin data:                    0
    error API:                  16

Los 16 sin hit son CUSIPs de emisores con prefijo no-USA (G, H, M, V
- Irlanda, Jersey, Suiza, Islas Caiman) o con formato que OpenFIGI no
reconoce por ID_CUSIP. No son contra-evidencia: quedan fuera del
alcance de esta validacion.

Conclusion: la validacion externa confirma la identidad CUSIP ->
shareClassFIGI de los 227 mapeos resolubles. Los 16 restantes no son
verificables por este canal. La cobertura interna del TARGET construido
es 1,0, con el alcance declarado; la cobertura observable del catalogo
radar declarado es 240/242 = 99,17% (§13.5).

**Evidencia archivada.** Cada ejecucion de
`scripts/iae_validate_crosswalk_openfigi.py` versiona sus artefactos en
`data/mappings/openfigi_requeries/`:

    <slug>_input.json     parametros, timestamp UTC, git_head,
                          script_version, lista de CUSIPs.
    <slug>_raw.json       respuesta cruda por CUSIP.
    <slug>_summary.txt    resumen legible + script_version + HEAD.
    HASHES_<slug>.txt     sha256 de los 3 ficheros anteriores +
                          timestamp + script_version.

El artefacto de la validacion 2026-09-22 (run
`2026-09-22_crosswalk_243_v2`, HEAD `c6185e1a`, script_version `2.1` en
la version actual) esta versionado en git. Esto cierra el punto 7 del
dictamen v4: la evidencia externa es reproducible y auditable. Los
raws quedan versionados (tamano moderado); si en el futuro crecen
significativamente, la politica puede pasar a conservar solo `HASHES_*`
y `summary`.

### 12.5. Resultado de la cadena de delta y NIPC

Nota semantica: NIPC = Observed Reported Position Change. NO es flujo
economico. Ver §10.3.

Ejecucion con el crosswalk completo y el filtro correcto (split oficial: por `canonical_security`, con `canonical_security in {equity:<TICKER>, figi:<FIGI>}`; en el E2E observado solo aparece la forma `equity:<TICKER>`, ver §13.10; el
split por CUSIP observado NO es viable porque las filas con
observed_security_key=`cusip:XXX` son UNRESOLVED_IDENTITY y no
contribuyen al NIPC):

    Delta full:   4.066.694 filas
      BOTH:          713.865
      NEW:           106.828
      EXIT:           95.105
      UNRESOLVED:  3.150.896

    Delta radar:    553.321 filas
      BOTH:          444.276
      NEW:            60.324
      EXIT:           48.721
      UNRESOLVED:          0

    NIPC radar:
      nipc_total       -4.316.734.936
      nipc_sole       -32.484.713.330
      nipc_dfnd       +28.317.652.408
      nipc_otr             -149.674.014

Verificacion de coherencia:
    n_both + n_new + n_exit = 553.321 = n_delta_observable
    n_unresolved_identity = 0

### 12.6. Validacion cruzada radar vs complemento

El delta full se descompone en radar (553.321) y complemento (3.513.373).
La suma de sus NIPC coincide con el NIPC full en todas las metricas:

    metrica             radar           complemento     full
    nipc_total         -4.316.734.936   -696.974.111    -5.013.709.047
    nipc_sole         -32.484.713.330  -11.359.019.235  -43.843.732.565
    nipc_dfnd         +28.317.652.408  +10.633.699.724  +38.951.352.132
    nipc_otr             -149.674.014      +28.345.400     -121.328.614

Verificacion: radar + complemento = full en todas las metricas.

Nota metodologica (A.1 v3, 2026-09-22). El split radar/complemento se
hace por `canonical_security`, con `canonical_security in
{equity:<TICKER>, figi:<FIGI>}`. En el E2E auditado: observados =
`equity:<TICKER>`, `figi:*` = 0 (la segunda rama requiere
`figi_lookup` no nulo, no activado en el pipeline auditado, ver §13.10).
El delta esta
particionado en dos subpoblaciones disjuntas: filas UNRESOLVED con
`observed_security_key=cusip:XXX` y `canonical_security=None`
(3.150.896 filas, no contribuyen al NIPC); filas CANONICAL con
`observed_security_key=None` y `canonical_security in {equity:XXX,
figi:BBG...}` (915.798 filas; la forma `figi:` puede aparecer
teoricamente si el resolver recibe `figi_lookup`, no en el E2E
auditado). No existe ninguna fila con ambos campos poblados, por lo
que un filtro por CUSIP sobre el delta no puede recuperar ninguna
fila contribuyente.

Los valores anteriores de esta tabla (complemento = +164.048.507,
full = -1.404.191.665) provenian de una ejecucion pre-fix del crosswalk
(antes de extender valid_from a Q4 para DD, HON, XOM) y no eran
reproducibles. Ver `scripts/iae_reconciliation_b1.py` y
`outputs/audit/b1_reconciliation/` para la cadena completa.

---

## 13. Limitaciones y alcance

Honestidad sobre lo que el motor cubre y lo que no. Ninguna de estas
limitaciones es un defecto de implementacion: son propiedades del universo
de datos o del diseño.

### 13.1. Cobertura FIGI en INFOTABLE es 12.2%

Solo 468.237 de 3.822.885 filas de INFOTABLE Q1 2026 tienen FIGI poblado.
El resto solo tienen CUSIP.

Impacto: el motor resuelve identidad por CUSIP via crosswalk, no por FIGI.
La cobertura contractual del TARGET construido por periodo es 1.0 (ver
§12.3 y las cautelas de §13.5). La cobertura de securities pequeños
fuera del radar sigue siendo baja.

Un fallback CUSIP -> FIGI via OpenFIGI podria ampliar la cobertura de
identidad, pero NO se ha demostrado que resuelva exhaustivamente los
CUSIPs restantes: en la validacion de §12.4, 56 de 100 CUSIPs de la
muestra aleatoria y 16 de 243 del crosswalk no obtuvieron hit de
OpenFIGI. Por tanto, OpenFIGI es una via de mejora, no una garantia
de cierre.

### 13.2. OKE y SPCX no contribuyen al TARGET contractual

De los 242 tickers del catalogo radar, 240 contribuyen al TARGET
contractual de Q4 2025 y Q1 2026. Los 2 ausentes:

- **OKE**. CUSIP 682680103 aparece en `etf_holdings.csv` (sin
  vigencia temporal). El resolver P61 lo clasifica como
  `TEMPORAL_UNVERIFIED`, y el filtro §5.5 (CANONICAL AND VERIFIED)
  lo excluye del operational universe. No contribuye al delta ni al
  NIPC.

- **SPCX**. No esta en ninguna fuente de identidad (ni
  `cusip_radar_crosswalk.csv` ni `etf_holdings.csv`). Emisor privado
  (SpaceX), sin filings 13F. Tampoco contribuye.

Adicionalmente, `cusip_radar_crosswalk.csv` contiene 2 tickers que NO
estan en el radar canonico actual: DD y HON. Fueron extendidos a Q4
2025 por el fix 7609a56 (ver §12.3, nota historica). Son entradas
validas del crosswalk pero no pertenecen al universo radar.

### 13.3. Cobertura de tests: 90% global, 0 ficheros < 80%

Ningun fichero del modulo queda por debajo del 80% de cobertura de
lineas. Los 6 ficheros que en la revision anterior estaban por debajo
del 80% eran `timestamps.py` (20%), `openfigi_client.py` (31%),
`target_universe.py` (31%), `reporting_dedup.py` (55%),
`security_type.py` (64%) y `temporal_validity.py` (79%). Hoy sus
coberturas respectivas son 87%, 89%, 93%, 91%, 98% y 96%, todas por
encima del umbral.

El minimo actual es `catalog_key.py` con 82%. La deuda de cobertura
como tal queda cerrada; los ficheros con cobertura mas baja siguen
sin estar integrados en produccion (13.4).

### 13.4. Integracion unidireccional a produccion

**Estado 2026-09-23:** el IAE esta integrado en `run.py` como fase
adicional del pipeline. Se ejecuta en cada run productivo.

Restriccion que se mantiene: la integracion es unidireccional
(`run.py -> IAE`). El IAE no importa `run.py`, `src.report`,
`src.regimes` ni `src.indicators`. Se conserva el encapsulamiento
interno del modulo.

Verificacion inversa (2026-09-23): 0 imports desde
`src/institutional_accumulation/*` hacia las capas consumidoras.

### 13.5. Alcance de la cobertura contractual 1.0

**El 100% NUNCA debe leerse como cobertura del radar declarado.** La
cobertura 1.0 de §12.3 es la cobertura interna del TARGET que el
sistema construye en el periodo, NO del catalogo radar declarado. El
TARGET se construye a partir de las keys del catalogo con observacion
VERIFIED, por lo que todos sus elementos tienen evidencia por
construccion. La cobertura 1.0 es, en ese sentido, endogena al TARGET.

Nomenclatura canonica (adoptada a partir del dictamen v5):

    catalog_coverage_declared = TARGET_OBSERVABLE / CATALOG_DECLARED
                              = 240 / 242 = 99,17%
    target_internal_coverage  = VERIFIED / TARGET_CONSTRUIDO
                              = 240 / 240 = 100%

`catalog_coverage_declared` es la metrica independiente del radar. Es
la que debe citarse cuando se hable de "cobertura del radar".

`target_internal_coverage` es la metrica interna del pipeline. Solo
declara que todos los elementos del TARGET construido tienen evidencia
verificada. Es endogena, no es evidencia de cobertura del radar.

| Metrica                          | Resultado | Que significa                                              |
| -------------------------------- | --------: | ---------------------------------------------------------- |
| catalog_coverage_declared Q4     |    99,17% | 240 / 242 keys con observacion VERIFIED                    |
| catalog_coverage_declared Q1     |    99,17% | 240 / 242 keys con observacion VERIFIED                    |
| TARGET operacional Q4            |       240 | sub-universo construido con observacion VERIFIED           |
| TARGET operacional Q1            |       240 | sub-universo construido con observacion VERIFIED           |
| target_internal_coverage         |      100% | todos los elementos del TARGET tienen evidencia verificada |

No equivale a:

- Reconstruccion PIT historica del radar de Q4 2025 con fidelidad
  total. El catalogo radar nacio en 2026-09-21 (§13.8) y la pertenencia
  historica al radar no esta demostrada (§13.9).
- Cobertura del universo 13F completo. El universo observado (35.649
  CUSIPs en INFOTABLE Q1) es mucho mayor que el radar (242). El motor
  mide acumulacion institucional **sobre el radar**, no sobre el
  universo 13F completo.
- Inclusión efectiva de los 242 tickers del catalogo. 240 contribuyen
  (ver §13.2); OKE y SPCX quedan fuera del TARGET por falta de
  observacion VERIFIED en los periodos analizados.

Lo que si esta respaldado: el TARGET contractual construido para Q4
2025 y Q1 2026 tiene cobertura 1.0 en las dos dimensiones (security
y weighted share), y la validacion externa con OpenFIGI (§12.4)
confirma la identidad CUSIP->shareClassFIGI de los 227 mapeos
resolubles del crosswalk.

### 13.6. n_unresolved_identity en delta full es 77.5%

Del delta full (4.066.694 filas), 3.150.896 son UNRESOLVED_IDENTITY.
Corresponden a securities fuera del crosswalk, mayoritariamente
emisores pequeños.

Al filtrar al radar, n_unresolved_identity = 0. El motor contractual
opera solo sobre securities resueltas.

### 13.7. Alcance de posiciones del NIPC

El NIPC utiliza unicamente las lineas 13F que superan el filtro
operativo definido en §10.1. El calculo actual esta restringido al
universo de securities elegibles para el radar: las filas CALL, PUT
y no-equity se excluyen y, por tanto, no forman parte del NIPC.

Aclaracion de alcance: los filings 13F no proporcionan una medida
integral de exposicion long/short. El NIPC utiliza las posiciones
reportadas por 13F que superan el filtro operativo definido en §10.1;
no calcula exposicion neta long/short.

La metrica NIPC (Net Institutional Position Change) suma deltas de
SSHPRNAMT entre trimestres. No distingue entre apertura/cierre de
posiciones y cambios de tamaño dentro de una posición existente.
El desglose por discretion_type (SOLE / DFND / OTR) permite leer
el flujo neto por tipo de discreción, pero no hay analisis de
turnover por manager.

### 13.8. El sistema nacio en septiembre de 2026

No hay historico de snapshots del radar anteriores a 2026-09-21.
El catalogo radar es un artefacto derivado de `stock_prices.parquet`
que se regenera cada dia. No hay versionado PIT de periodos anteriores.

Esto significa: la reconstruccion de un TARGET para periodos anteriores
al nacimiento del sistema no es posible con los datos actuales.

### 13.9. ASSUMPTION / Historical reconstruction limitation

El crosswalk CUSIP -> ticker se construyo cruzando filings Q4 2025 +
Q1 2026 contra el catalogo radar. Se aplico retroactivamente a Q4 2025
con `valid_from=2025-12-31`.

Distincion critica entre dos conceptos que el diseno actual no puede
separar con la evidencia disponible:

- **Identidad historica** de la security: probablemente estable. Un
  CUSIP X corresponde al mismo ticker entre periodos consecutivos
  (evidencia: la validacion positiva del §12.4 sobre los 227 mapeos
  resolubles no muestra ningun mismatch).
- **Pertenencia historica** al universo radar: NO demostrado. Un
  ticker puede estar en el radar en Q1 2026 pero no en Q4 2025 (o al
  reves). El sistema no dispone de snapshots PIT del catalogo radar
  anteriores a 2026-09-21.

La reconstruccion de identidad es plausible. La pertenencia historica
no es verificable con la evidencia actual. Se declara como limitacion
de alcance, no como bug. Reabrir si aparece un snapshot PIT historico
o si auditoria externa exige reconstruir el radar en fechas previas a
2026-09-21.

### 13.10. Dualidad de representacion de canonical_security - rama figi:* no activada en E2E

El resolver (`security_identity.py::_normalize_canonical`) admite dos
formas: `equity:<TICKER>` (crosswalk/equivalence) y `figi:<FIGI>`
(figi_lookup directo). La segunda requiere `figi_lookup` no nulo.

**Auditoria empirica (2026-09-23).** Sobre los 4.780.572 `units` del
E2E auditado, con `figi_lookup=None`:

| Metrica                                                  | Resultado |
| -------------------------------------------------------- | --------: |
| `canonical_security = equity:*`                          | 1.629.676 |
| - dentro del radar                                       |   997.610 |
| - fuera del radar (`unresolved_ref`)                     |   632.066 |
| `canonical_security = figi:*`                            |     **0** |
| `shareClassFIGI` bajo dos `canonical_security` distintos |     **0** |
| Tickers del catalogo con >1 `shareClassFIGI`             |     **0** |
| Delta NIPC al normalizar por `shareClassFIGI`            |     **0** |

**Conclusion:** en la configuracion E2E auditada no se observa
fragmentacion de identidad. La dualidad es una **capacidad del
resolver** (implementada y cubierta por tests); su segunda rama no se
activa en el pipeline auditado porque `figi_lookup=None`. Los
`unresolved_ref` corresponden a `equity:X` cuyo ticker X esta fuera
del radar y no constituyen fragmentacion.

Reabrir esta auditoria si:

  - se activa `figi_lookup` en algun punto del pipeline;
  - aparece un caso `figi:*` en `units` o `delta`;
  - se modifica el modelo de resolucion de identidad;
  - se decide integrar el IAE en produccion.

Script reproducible: `scripts/iae_identity_uniqueness_audit.py`.
Salida: `outputs/audit/iae_identity_audit/`.

### 13.11. Operacion autonoma en GitHub Actions

El modulo IAE se mantiene actualizado sin intervencion manual. Dos
workflows lo soportan:

  - `update_sec_13f.yml` (trimestral, cron `0 6 20 2,5,8,11 *`):
    descarga el dataset SEC 13F del trimestre cerrado, lo ingesta
    (7 parquets + manifest) y descarga la Official List 13(f).
    Cachea `data/sec_13f/processed/` con key
    `13f-processed-v1-<latest_quarter>`.
  - `daily_run.yml` (diario): restaura la cache 13F antes de ejecutar
    `run.py`. La seccion IAE del reporte se calcula con lo disponible.

Artefactos versionados en git:
  - `data/sec_13f/latest_quarter.txt` (2 bytes): identifica el ultimo
    trimestre ingestado. Usado como key de cache.
  - `data/sec_13f/official_list_13f/13flist_*.txt` (~1-2 MB/trimestre):
    versionado para auditabilidad.
  - `data/sec_13f/manifests/`: lineage de cada ingesta.
  - `data/sec_13f/processed/`: **NO versionado**. Vive solo en cache
    de Actions (~195 MB/trimestre). Un clone local fresco no lo tiene
    y la seccion IAE devolvera `STALE` hasta que se ejecute
    `scripts/update_sec_13f.py`.

Comportamiento ante fallos:
  - Cache miss en `daily_run`: warning + seccion IAE `STALE`. El radar
    diario NO se bloquea.
  - Error en `update_sec_13f`: workflow rojo, sin commit. La cache
    previa se mantiene.
  - SEC rate limit (404/408/429): `downloader._http_get_with_retry`
    reintenta con backoff exponencial (5s, 15s, 45s).

Scripts reproducibles:
  - `scripts/update_sec_13f.py` - orquesta descarga + ingesta.
  - `scripts/download_official_list_13f.py` - descarga Official List.
  - `scripts/cleanup_stock_prices_nyse_holidays.py` - saneamiento
    puntual de parquets (F-IAE-HOLIDAY-01).

## FIN DEL IAE_MAESTRO

Documento unico del modulo IAE. No depende de expedientes previos,
contratos externos ni dictamenes.

2026-09-23.