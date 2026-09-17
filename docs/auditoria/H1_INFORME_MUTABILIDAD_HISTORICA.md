# H1 - Mutabilidad del dataset historico

**Fase:** investigacion arquitectonica (no bug)
**Fecha:** 2026-09-17
**HEAD de referencia:** 04b8e34 (origin/main)
**Alcance:** dataset historico versionado (`data/*.parquet`, `data/*.manifest.json`, `outputs/history/*.csv`)
**Autorizado por:** sesion 2026-09-17 (post Gate 0 sistematico BAJA)
**Estado:** investigacion cerrada. Requiere decision arquitectonica del auditor.
**Autor:** Ingeniero Supervisor

---

## 0. Motivo del informe

H1 no es un bug. Es la observacion de que el dataset historico del sistema puede variar entre runs por tres mecanismos independientes. La pregunta no es "como arreglarlo" sino "que politica de mutabilidad adoptar".

Origen: durante el ciclo K-FS-CI-PARITY-01 (2026-09-17) se detecto que OilPriceAPI reviso `CL=F 2026-09-16` de 100.40 a 97.21 entre dos capturas separadas por horas. Inicialmente catalogado como "mutabilidad de proveedor". Gate 0 posterior confirmo que:
- (a) el mismo efecto existe en Yahoo (`USDJPY=X 2026-09-16`),
- (b) el pipeline mismo ha regenerado historicos masivamente (commit `35af4ba`),
- (c) `append_dedup` aplica `keep='last'` sobre cualquier clave duplicada.

Este informe no modifica codigo de produccion. Es precondicion para decidir si se modifica. **No incluye recomendacion del ingeniero**: la decision es de politica de datos y corresponde al auditor.

---

## 1. Hechos medidos

### 1.1. H1-a - Mutabilidad externa (proveedor)

Proveedor externo revisa valores ya capturados, sin notificacion al consumidor.

**Evidencia 1 - OilPriceAPI (`CL=F`):**
- `2026-09-16`: `100.40` (parquet commiteado en `38092ed`)
- `2026-09-17`: `97.21` (probe contra OilPriceAPI)
- Delta: `-3.19` (`-3.18%`)

**Evidencia 2 - Yahoo (`USDJPY=X`):**
- `2026-09-16`: `156.1880` (parquet local)
- `2026-09-17`: `155.2660` (yfinance fresco)
- Delta: `-0.9220` (`-0.594%`)
- Fechas anteriores (`09-10` a `09-15`): `0.000%` diff.
**Nota:** la mutacion observada NO es uniforme. Los valores de dias anteriores al ultimo se mantienen estables. El dia mas reciente es el que sufre revision. Los proveedores parecen "asentar" sus valores en ventanas temporales cortas tras la publicacion inicial.

**Metodologia del probe:** se compara el valor almacenado en el parquet local con el valor devuelto por el mismo proveedor en una segunda captura. La disparidad es, por definicion, revision del proveedor (mismo endpoint, mismo ticker, misma fecha de observacion, distinto valor).

### 1.2. H1-b - Mutabilidad introducida por pipeline

El pipeline ha reescrito historicos versionados en git por decision interna, no por cambios del proveedor.

**Evidencia - Commit `35af4ba` (2026-09-12):**
- Mensaje: `Data: regenerar etf_primary_flow.csv con Z clipado (1485 valores fuera de [-5,+5] corregidos).`
- Diff stat: `outputs/history/etf_primary_flow.csv` -> 1485 inserciones / 1485 eliminaciones.
- Alcance: el CSV tracked completo se reescribio para aplicar clip a `[-5, +5]` sobre Z-scores previamente almacenados sin clip.

**Implicacion:** el dataset historico puede cambiar por correccion de logica del pipeline, no solo por revision de la fuente. Cualquier auditoria que compare CSV actual con CSV de un HEAD anterior puede encontrar diferencias que no provienen de cambios en los datos crudos.
### 1.3. H1-c - Sobrescritura por utilidad interna

`src/utils.py:25`:

    return combined.drop_duplicates(subset=subset, keep='last')

**Descripcion neutral:** `append_dedup(..., keep='last')` permite sustituir una observacion historica existente cuando llega una nueva observacion para la misma clave, sin registrar explicitamente el hecho de la revision.

**Consecuencia observable:** si el proveedor revisa el valor de `(ticker, fecha)` y el pipeline recibe la nueva observacion, la fila historica se sustituye. El CSV o parquet resultante es coherente con la ultima fuente consultada, pero no conserva traza de la observacion previa.

**Aplicado a:** `outputs/history/*.csv` (~25 ficheros), `data/commodities_*.parquet`, y cualquier writer que use `append_dedup` con clave `(ticker, date)` o similar.

### 1.4. Tracking heterogeneo

| Fichero | Parquet tracked | Manifest tracked |
|---|---|---|
| `data/market_data.parquet` | NO (gitignored) | SI |
| `data/stock_prices.parquet` | NO (gitignored) | SI |
| `data/commodities_futures.parquet` | SI | SI |
| `data/commodities_spot.parquet` | SI | SI |
| `data/cboe_vix3m.parquet` | SI | SI |

El manifest de `market_data.parquet` y `stock_prices.parquet` esta versionado, pero los parquets referenciados no. Resultado: el repositorio conserva el sha256 historico del artefacto, pero no el artefacto mismo.

**Evidencia:** `git log --oneline -- data/market_data.parquet.manifest.json` -> 12 commits. Cada uno con sha256 distinto. Ninguno de esos sha256 puede verificarse contra el fichero correspondiente desde el repo.

### 1.5. FU-002 - Scope del sha256

`src/utils.py:405-409`:

    h = hashlib.sha256()
    with open(tmp_parquet, 'rb') as fh:
        for chunk in iter(lambda: fh.read(65536), b''):
            h.update(chunk)
    sha256 = h.hexdigest()

**Descripcion neutral:** FU-002 calcula el hash del artefacto en el momento de escritura, pero no existe actualmente un mecanismo especifico de deteccion o registro de revisiones posteriores del valor para una misma clave.

**Alcance del hash actual:**
- Cubre integridad del fichero en el instante de escritura.
- No cubre comparacion con un manifest previo.
- No cubre deteccion de revisiones de clave `(ticker, fecha)` entre runs.

La funcion cumple lo que declara. El gap es de cobertura conceptual, no de implementacion.

### 1.6. Deteccion de revisiones en el pipeline

Busqueda en `src/utils.py` de los patrones `revision`, `mutation`, `compare.*prev`, `prev.*hash`, `diff.*manifest`. Resultado: 0 hits.

No existe mecanismo que:
- Compare un manifest nuevo con el anterior.
- Alerte cuando sha256 cambia para un mismo artefacto sin cambio de esquema.
- Registre revisiones de valor para una clave `(ticker, fecha)`.

FU-002 es un contrato de integridad intra-run, no un sistema de auditoria cross-run.

---

## 2. Impacto

### 2.1. Reproducibilidad historica

El efecto arquitectonico central de H1 es la imposibilidad de reconstruir el estado exacto del dataset en un HEAD pasado usando solo el repositorio.

**Casos:**

1. `data/market_data.parquet` gitignored + manifest tracked. El manifest de `HEAD~N` declara un sha256 que ya no es verificable. El parquet actual no coincide con ese sha256.

2. `data/commodities_futures.parquet` tracked. `git diff` muestra revisiones del proveedor como cambios de datos. No hay metadato que distinga revision de proveedor de correccion de pipeline de bug de escritura.

3. `outputs/history/*.csv` tracked con alta rotacion (20-25 commits por fichero). Cada commit puede contener revisiones o regeneraciones. No hay changelog por fichero.

**Consecuencia practica:** si el reporte diario del `2026-09-10` se audita hoy, el auditor no puede determinar con certeza si el CSV que ve corresponde al estado del dia o a una version posterior sobrescrita.

**Pregunta abierta:** es la reproducibilidad historica un requisito del sistema? La respuesta pertenece al auditor.

### 2.2. FU-002 - sha256 huerfano para `market_data` y `stock_prices`

El manifest versionado actua como firma del artefacto, pero el artefacto firmado no esta en el repo. La firma no es verificable hoy contra el fichero original.

**Escenarios donde importa:**
- Auditoria de integridad historica: no se puede validar que el parquet de `HEAD~N` es el que produjo el manifest de `HEAD~N`.
- Debug de regresiones: comparar manifests antiguos con actuales solo revela cambio de hash, no cambio de contenido.

**Escenarios donde NO importa:**
- Uso operativo diario: el manifest se lee en el mismo run.
- Deteccion de corrupcion intra-run: sigue funcionando.
- FSM de readers (VALID/INVALID/UNAVAILABLE): sigue funcionando con el parquet presente en disco.

### 2.3. Contratos temporales R1-R4

Los contratos validan (fecha, cobertura, lag) por ticker/clase. NO validan valor.

**Consecuencia:** una revision de valor por el proveedor no dispara STALE ni INSUFFICIENT. El contrato temporal no es el mecanismo adecuado para detectar mutabilidad de valor. No esta disenado para eso y no es un bug del contrato.

FU-021-5 queda fuera del alcance de H1. La separacion es limpia: temporalidad vs integridad de valor.

### 2.4. `outputs/history/*.csv` como fuente historica paralela

Los CSV en `outputs/history/` tienen 20-25 commits cada uno. Se versionan en cada run automatico (Daily hist/state). Cada version refleja el estado acumulado del dataset hasta ese run.

**Implicacion:** existen dos fuentes de verdad del historico:
- `data/*.parquet` (local, gitignored o tracked segun fichero)
- `outputs/history/*.csv` (tracked)

Ambas pueden discrepar por revisiones, correcciones o regeneraciones. No hay mecanismo que sincronice ni que detecte discrepancias entre ellas.

---

## 3. Lo que H1 NO es

- No es un bug del pipeline. El sistema opera como esta disenado.
- No viola Gate 10/10.
- No rompe tests (615 + 2 skipped estables).
- No viola R1-R6 (contratos temporales ortogonales).
- No viola FU-002 (el contrato declara hash en momento de escritura, y eso hace).
- No afecta a la operativa diaria del reporte.
- No produce perdida de informacion en el momento del run.
- No implica que el dataset este corrupto.

H1 es una caracteristica arquitectonica del dataset historico, no un defecto de implementacion.


---

## 4. Opciones

Cuatro opciones, presentadas sin recomendacion del ingeniero.

### A - Snapshot/inmutabilidad por periodo

Mantener snapshots inmutables del dataset a intervalos definidos (por ejemplo, mensual). El snapshot se versiona en git o en almacenamiento externo. Los runs diarios siguen mutando el dataset vivo, pero los snapshots son referencias estables.

**Ejemplo:** `data/snapshots/2026-09-30/market_data.parquet` con su manifest.

### B - Congelacion de valores publicados

Una vez que un valor `(ticker, fecha)` se ha publicado en un reporte, se congela. Cualquier revision posterior se registra como metadato adicional (revision_count, original_value, current_value), no sobrescribe.

**Requiere:** cambio de `append_dedup` y de los writers.

### C - Aceptar mutabilidad documentada

Declarar explicito en el prompt y en los contratos que el dataset historico es mutable por diseno. Registrar la politica en `docs/`. No modificar codigo.

**Se apoya en:** `outputs/history/*.csv` ya funcionan como fuente congelada por commit. El commit de cada run fija el estado del dia.

### D - Deteccion/registro de revisiones

Anadir un mecanismo de deteccion de revisiones: al escribir un artefacto, comparar con el manifest previo. Si sha256 cambia sin cambio de schema, registrar evento (`revision_log.json` o similar).

**No previene** la mutabilidad, pero la hace visible.

---

## 5. Comparativa

Sin recomendacion. La decision corresponde al auditor.

| Criterio | A (snapshot) | B (congelacion) | C (aceptar doc) | D (deteccion) |
|---|---|---|---|---|
| Esfuerzo implementacion | Medio | Alto | Bajo | Medio |
| Cambio de codigo | No (nuevo proceso) | Si (append_dedup + writers) | No | Si (write_artifact) |
| Impacto operativo | Bajo | Alto (cambia semantica) | Cero | Bajo |
| Riesgo de regresion | Bajo | Medio | Cero | Bajo |
| Resuelve reproducibilidad | Parcial (periodos) | Total | No (solo documenta) | No (solo visibiliza) |
| Coste almacenamiento | Alto (snapshots) | Bajo | Cero | Bajo |
| Complejidad auditoria | Baja | Media | Nula | Baja |
| Compatible con FU-002 actual | Si | Requiere extension | Si | Requiere extension |

**Combinaciones posibles:** A+D, B+D, C+D.


---

## 6. Preguntas al auditor

1. Debe el sistema garantizar reproducibilidad historica byte-exacta? Si no, H1 puede cerrarse como WONT FIX documentado (opcion C).

2. Si debe garantizarla, que granularidad es aceptable? Diaria / semanal / mensual / por run.

3. Los 1485 valores regenerados en el commit `35af4ba` deben tratarse como baseline nuevo o deben conservarse como diff con el original?

4. FU-002 debe extenderse a deteccion cross-run de revisiones, o su alcance actual (integridad intra-run) es suficiente?

5. `market_data.parquet` y `stock_prices.parquet` deben dejar de estar gitignored? Alternativa: mantener gitignored + manifest tracked como firma no verificable.

6. Los CSV en `outputs/history/` son la fuente historica oficial, con los parquets como datos de trabajo? O al reves?

7. Existe un contexto externo (auditoria regulatoria, compliance) que imponga requisitos de inmutabilidad?

---

## 7. Evidencia reproducible

### 7.1. Comandos git

    git log --oneline --follow -- data/commodities_futures.parquet
      -> 38092ed Daily hist/state (2026-09-16 22:30 UTC)
      -> b088ff6 feat(workflows): update commodities en daily_run (2026-09-16)

    git show 35af4ba --stat
      -> outputs/history/etf_primary_flow.csv | 2970 ++++++++++++------------
      -> 1485 insertions(+), 1485 deletions(-)

    git log --oneline -- data/market_data.parquet.manifest.json | wc -l
      -> 12

    git ls-files data/*.parquet data/*.manifest.json
      -> 5 pares (cboe, commodities_futures, commodities_spot + 2 manifests
         huerfanos para market_data, stock_prices)

### 7.2. Probes ejecutados

`py _probe_h1a.py` (2026-09-17, HEAD `04b8e34`):
- Manifests `data/*.manifest.json`: 5 presentes (cboe 941 B, commodities_futures 968 B, commodities_spot 951 B, market_data 971 B, stock_prices 972 B).
- `market_data.manifest`: sha256=0114d6...cf05, rows=2604, cols=2810, n_tickers=562, status=VALID_WITH_MISSING.
- `commodities_futures.manifest`: sha256=0b909d...3d525, rows=2, cols=10, status=VALID_WITH_MISSING.

Comparativa contra yfinance (`USDJPY=X`):
- 2026-09-10 a 2026-09-15: 0.000% diff.
- 2026-09-16: parquet=156.1880, yahoo=155.2660, diff=+0.594%.

### 7.3. Ficheros inspeccionados

- `docs/auditoria/PROMPT_MAESTRO.md` (v6.25, secciones 12 + 15.16 + 15.18).
- `src/utils.py:5-25` (`append_dedup`).
- `src/utils.py:405-409` (sha256 scope).
- `src/utils.py:492-496` (manifest artifact).
- `.gitignore:5-8` (exclusion `data/*.parquet`).
- `data/*.manifest.json` (5 ficheros).
- `outputs/history/etf_primary_flow.csv` (revision `35af4ba`).

---

**Fin del informe H1.**

**Fecha:** 2026-09-17.
**HEAD:** 04b8e34.
**Proximo paso:** dictamen del auditor.
