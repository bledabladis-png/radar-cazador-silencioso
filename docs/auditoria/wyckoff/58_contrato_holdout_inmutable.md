# CONTRATO SOW-HOLDOUT-IMMUTABLE v1.1

**Estado:** BORRADOR v1.1. Pendiente de revision y firma del auditor
externo. No normativo hasta firma.
**Fecha:** 2026-10-28.
**Ciclo:** sesion 15.
**Frente:** SOW-HOLDOUT-IMMUTABLE (REGISTRO.yaml, IN_PROGRESS).
**Base:** v1.0 (commit 46b36409) + dictamen externo 2026-10-28 +
doc 74 (informe H-04/H-07).

---

## 1. Proposito

Declarar formalmente que dos parquets de datos son HOLDOUT para la
validacion SOW, fijar las condiciones bajo las que (si acaso) un
ticker puede promoverse desde ellos a produccion, y garantizar que
el estado del holdout (que se ha consumido, con que candidata, con
que codigo y con que datos) queda registrado de forma auditable.

Sin este contrato, promover tickers desde holdout a produccion es
ambiguo y el estado del holdout no es trazable. El dictamen externo
2026-10-28 identifico esta doble ambiguedad (inmutabilidad fisica vs
independencia estadistica) como el riesgo central.

La inmutabilidad fisica y la independencia estadistica son
propiedades distintas. Un parquet puede permanecer intacto y, sin
embargo, dejar de ser una muestra independiente para futuras
decisiones si sus resultados ya se han utilizado para seleccionar o
modificar una candidata.

---

## 2. Definiciones

### 2.1. Parquet holdout cohort

`data/stock_prices_reserved_cohort.parquet`

- **Contenido real:** 268 tickers de la composicion **actual** del
  S&P 500 (descargada el 2026-10-24 desde
  `https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv`)
  que NO estan en `stock_prices_extended.parquet`.
- **Poblacion:** composicion actual del S&P 500 aplicada
  retrospectivamente a precios 2015-01-02 .. 2026-10-23. NO es la
  composicion historica del indice. Survivorship bias explicito.
- **Extrapolacion:** los resultados NO son extrapolables a la
  composicion historica del S&P 500.
- **Rango:** 2015-01-02 -> 2026-10-23.
- **Generado por:** `scripts/fetch_reserved_cohort.py`.
- **Gitignored:** `.gitignore:96`. Regenerable, no versionado.

### 2.2. Parquet holdout r3000

`data/stock_prices_reserved_r3000.parquet`

- **Contenido real:** 3423 tickers de la lista `all_tickers.txt` de
  `rreichel3/US-Stock-Symbols` (descarga 2026-10-24), menos los
  presentes en `stock_prices_extended.parquet`.
- **Poblacion:** lista generica de tickers estadounidenses. **NO
  representa la composicion del Russell 3000.** El nombre `r3000`
  se conserva como nombre heredado, no como identificacion de la
  poblacion.
- **Limitaciones documentadas:** 2 tickers de la composicion actual
  del S&P 500 no estan en `all_tickers.txt`. La lista tiene huecos.
- **Rango:** 2015-01-02 -> 2026-10-23.
- **Generado por:** `scripts/fetch_reserved_cohort_r3000.py`.
- **Gitignored:** `.gitignore:97`. Regenerable, no versionado.

### 2.3. Interseccion de universos

Los dos universos NO son disjuntos. Interseccion calculada
2026-10-28: **266 tickers de 268 de cohort estan tambien en r3000.**
Solo 2 tickers de cohort no estan en r3000.

Implicacion contractual: los resultados de cohort y r3000 **no son
dos confirmaciones independientes** para 266 de los 268 tickers de
cohort. Comparten observaciones. La validacion "en dos universos
independientes" no se sostiene estadisticamente para 266/268.

### 2.4. Inmutabilidad fisica

Un parquet es fisicamente inmutable si:

- Su contenido no se modifica por el pipeline diario.
- Su contenido no se modifica por scripts de regeneracion historica.
- Los generadores no sobrescriben silenciosamente un holdout ya
  sellado. Regenerar requiere procedimiento explicito de nueva
  version (§7).
- La huella del contenido aprobado se verifica antes de cada
  ejecucion confirmatoria (§6).

### 2.5. Independencia estadistica

Un holdout es estadisticamente independiente para una candidata
concreta si:

- Sus resultados no se han utilizado previamente para disenar,
  seleccionar o modificar esa candidata.
- El pre-registro de la candidata es anterior a la primera consulta
  de resultados sobre ese holdout.
- Existe evidencia temporal verificable (commit ancestro de HEAD,
  `capture_experiment_env.py` exit 0).

Inmutabilidad fisica NO implica independencia estadistica. Un
parquet puede estar intacto y haber sido consumido.

---

## 3. Consumidores declarados

### 3.1. Generadores

- `scripts/fetch_reserved_cohort.py`: genera `reserved_cohort.parquet`.
- `scripts/fetch_reserved_cohort_r3000.py`: genera `reserved_r3000.parquet`.

Los generadores son read-only respecto a produccion. No tocan
`stock_prices*.parquet` ni `config/`. No sobrescriben un holdout
sellado sin procedimiento explicito de nueva version (§7).

### 3.2. Validadores con pre-registro congelado

- `scripts/purged_kfold_sow_c1_reserved.py`: consume `reserved_cohort`.
- `scripts/purged_kfold_sow_c1_r3000.py`: consume `reserved_r3000`.

Ambos declaran en su codigo parametros congelados verificables por
`capture_experiment_env.py`:

- CANDIDATE = {"N": 40, "M": 5, "X_ATR": 1.0, "Y_VOL": 1.1}.
- EMBARGO = 50, H = 20, 23 folds.
- Criterio FUNCIONA (definido en el propio script): mean(lift) > 0.05
  AND IC95_lower > 0 AND folds_positive >= 7/9 AND
  folds_beat_placebo_B >= 7/9.

Cualquier cambio en esos scripts invalida el holdout como tal y
requiere nueva version del contrato (§11).

### 3.3. Consumidores NO declarados

Cualquier otro script o modulo que lea de estos parquets debe:

- Declararse en este contrato antes de usarlo.
- Justificar por que necesita el holdout y no produccion.
- No modificar el parquet.

---

## 4. Reglas de inmutabilidad operativa

I1. El pipeline diario (`py run.py`) NO lee de los parquets holdout.

I2. Los scripts de regeneracion historica
(`regenerate_wyckoff_history.py`, `build_extended_dataset.py`) NO
leen de los parquets holdout.

I3. Los parquets holdout NO se versionan en git (gitignored). Su
huella byte a byte queda registrada en
`data/mappings/holdout_manifest.json` (§6). Si se pierden, la
regeneracion debe producir un parquet con el mismo `content_sha256`
y `tickers_sha256` que el manifest, o requerir version bump (§11).

I4. Cualquier modificacion del contenido (anadir tickers, extender
rango, corregir OHLC) requiere version bump del contrato y re-firma
del auditor.

I5. La huella del manifest registra el contenido **desde 2026-10-28**.
No es la huella historica de los datos consumidos por SOW-V6 (§5).
Cualquier afirmacion sobre identidad byte a byte anterior a esa
fecha es imposible de verificar.

---

## 5. Estado del holdout (H-04)

El estado del holdout se registra en `REGISTRO.yaml` con **un item
por holdout** y una lista `experimentos_consumidos`. El estado se
deriva:

- Lista vacia -> `SEALED_UNRUN`.
- Lista con >=1 -> `CONSUMED_FOR_CANDIDATE`.
- Si algun experimento invalida la reserva para versiones nuevas ->
  `RETIRED_OR_SUPERSEDED`.

Cada entrada de `experimentos_consumidos` registra:

| Campo | Descripcion |
|---|---|
| `candidata` | identificador de la candidata evaluada (ej. SOW-V6, SOW-BEAR-CONFIRMATION v1.10.1) |
| `fecha_consumo` | fecha de la ejecucion confirmatoria |
| `informe` | referencia al documento de resultados |
| `experiment_manifest` | path a `data/mappings/experiment_manifests/<id>.json` |
| `manifest_sha256_en_consumo` | SHA-256 del parquet en el momento del consumo. `null` si el manifest no existia (todos los consumos anteriores a 2026-10-28) |

### 5.1. Estado inicial (2026-10-28)

- **cohort**: `CONSUMED_FOR_CANDIDATE`. Entrada:
  `{candidata: SOW-V6, fecha_consumo: 2026-10-06, informe: docs/auditoria/wyckoff/51-56, manifest_sha256_en_consumo: null}`.
- **r3000**: `CONSUMED_FOR_CANDIDATE`. Misma entrada.

### 5.2. Condicion especial sobre SOW-BEAR-CONFIRMATION

SOW-BEAR-CONFIRMATION **NO puede figurar como `SEALED_UNRUN`**. Su
diseno ya esta informado por los resultados de SOW-V6 (el patron
bear es real precisamente porque SOW-V6 los encontro). Debe
distinguirse:

- **Uso exploratorio**: analizar el holdout para disenar una
  candidata. NO cuenta como confirmacion independiente.
- **Ejecucion confirmatoria**: validar una candidata con protocolo
  pre-registrado y congelado, sobre datos no usados para disenarla.
  Solo esto cuenta como confirmacion.

Los datos actuales pueden servir para analisis exploratorio de
SOW-BEAR-CONFIRMATION, pero **no** como confirmacion independiente
de una hipotesis derivada de ellos.

---

## 6. Congelacion del experimento (H-02)

Antes de cada ejecucion confirmatoria sobre un holdout, el
experimento debe quedar congelado y verificable:

### 6.1. Huella del holdout (H-01)

`data/mappings/holdout_manifest.json` (versionado) contiene, por
cada parquet:

- SHA-256 del fichero entero (`sha256`).
- Hash canonico del esqueleto (`content_sha256`): rango temporal,
  n_rows, n_tickers, n_fields, fields ordenados, tickers ordenados.
- Hash de la lista de tickers (`tickers_sha256`).
- n_tickers, date_min, date_max, generator, source.

Antes de cada ejecucion confirmatoria:
`py scripts/audit/holdout_manifest.py --verify` debe pasar (exit 0).
Si falla, el experimento NO se ejecuta.

### 6.2. Congelacion del entorno

`scripts/audit/capture_experiment_env.py` captura por experimento:

- Commit (HEAD SHA, fecha, rama, dirty/clean).
- Evidencia temporal: `--prereg-commit` debe ser ancestro de HEAD.
  Si no, exit 1.
- SHA-256 de `requirements.txt` + versiones pinned + Python version.
- SHA-256 de cada validador y de cada modulo Wyckoff.
- Parametros congelados de cada validador (CANDIDATE, EMBARGO, H,
  FOLDS) via AST.
- Constantes `WYCKOFF_SOW_*` de `config.settings`.
- Referencia al `holdout_manifest.json` (sha256 + n_tickers + rango
  por parquet).
- Semilla si aplica.

Manifest en `data/mappings/experiment_manifests/<id>.json`
(versionado). Un manifest por ejecucion.

### 6.3. Evidencia temporal pre-registro

Requisito: el commit del pre-registro debe ser ancestro de HEAD en
el momento de la ejecucion. Esto se verifica automaticamente en
`capture_experiment_env.py`. Un commit no ancestro (por ejemplo,
posterior a la ejecucion) invalida el experimento.

Si el pre-registro es un documento sin commit propio (por ejemplo,
un fichero de texto en el working tree), debe commitearse antes de
la ejecucion y su SHA debe pasarse a `--prereg-commit`.

---

## 7. Condiciones de promocion holdout -> produccion (H-08)

Un ticker del holdout puede promoverse a produccion (anadirse a
`data/stock_prices.parquet` o a `data/universe_expansions.yaml`) si y
solo si se cumplen TODAS:

P1. El pre-registro congelado del validador correspondiente ya ha
    emitido su veredicto (§8). No se promueve antes de conocer el
    resultado.

P2. El auditor externo ha firmado explicitamente la promocion. La
    firma debe delimitar:

    - **Tickers concretos** (lista explicita, no "los 35").
    - **Grupo(s) productivo(s) de destino** (e.g. XLI, SPY).
    - **Uso permitido** (e.g. "descarga diaria y clasificacion
      Wyckoff", no "rediseno del detector").
    - **Version del contrato** y **version del codigo**
      (`HEAD` SHA) vigentes en la promocion.
    - **Estado de la validacion SOW** y referencia al informe
      (`experiment_manifest` + `informe`).

P3. La promocion no invalida el holdout para futuros usos. Si el
    ticker se usa en produccion, deja de ser holdout para ese
    ticker concreto. El resto del parquet sigue siendo holdout
    **fisicamente**, pero no necesariamente independiente para
    futuras candidatas (§5).

P4. La promocion se registra en `REGISTRO.yaml` con: SHA del commit
    que la aplica, lista de tickers promovidos, grupo destino, uso
    permitido, version contrato, version codigo, `experiment_id`
    del manifest, informe, y referencia a la firma del auditor.

P5. La firma de este contrato **no constituye autorizacion
    anticipada** para incorporar los 35 tickers a produccion. Esa
    aprobacion debe emitirse separadamente.

---

## 8. Veredictos posibles (H-05)

Una ejecucion confirmatoria sobre un holdout puede terminar en uno
de estos estados. La definicion exacta debe coincidir con la que ya
existe en los validadores congelados; no se introduce un criterio
nuevo retrospectivo.

- **FUNCIONA**: se satisface integramente el criterio
  pre-registrado (e.g. mean(lift) > 0.05 AND IC95_lower > 0 AND
  folds_positive >= 7/9 AND folds_beat_placebo_B >= 7/9).
- **NO_FUNCIONA**: se satisface el criterio pre-registrado de
  rechazo, si esta definido en el validador.
- **INCONCLUSO**: ejecucion valida que no satisface ni el criterio
  de aceptacion ni el de rechazo.
- **INSUFFICIENT_SAMPLE** o error de validacion: ejecucion sin
  resultado inferencial valido (e.g. menos folds de los exigidos,
  holdout sin suficiente historia).

En todos los casos, la ejecucion consume el holdout para la
candidata evaluada (ver §5). Un `INSUFFICIENT_SAMPLE` **no** debe
presentarse como resultado estadistico negativo. Una ejecucion
`INCONCLUSO` puede documentarse como consumo de la reserva para la
candidata examinada, pero **no** como validacion del detector.

La decision de promocion (§7) debe especificar que estados permiten
continuar. En la primera version, solo `FUNCIONA` habilita
promocion.

---

## 9. Tests de contrato requeridos (H-06)

### 9.1. Tests T1-T5 (v1.0, se mantienen)

T1. El pipeline diario no lee de los parquets holdout. Verificable
    por AST o grep controlado en `src/pipeline/` e `indicators/`.

T2. Los scripts de regeneracion no leen de los parquets holdout.

T3. Cualquier script que lea de los parquets holdout esta declarado
    en la seccion 3. Verificable por comparacion entre consumidores
    reales (grep) y consumidores declarados.

T4. Los parquets holdout siguen gitignored **y no estan
    versionados**. Verificable por `git check-ignore` **y**
    `git ls-files` (el patron de exclusion no sustituye a la
    verificacion del estado del repositorio).

T5. El pre-registro congelado de los validadores no ha cambiado
    desde la firma del contrato. Verificable por SHA-256 de los
    validadores via `capture_experiment_env.py`.

### 9.2. Tests T6-T10 (v1.1, nuevos)

T6. **Generadores no sobrescriben sellado.** Una regeneracion no
    sobrescribe el artefacto sellado ni altera produccion. Si el
    manifest existe y el hash cambia, el generador debe abortar
    (no sobrescribir). Verificable por test que ejecuta el
    generador con `holdout_manifest.json` preexistente y hash
    distinto: debe fallar sin escribir.

T7. **Dependencias del experimento.** Una modificacion de modulos o
    parametros efectivos cambia el `experiment_manifest` o invalida
    la ejecucion. Verificable por test que altera un modulo
    Wyckoff y comprueba que el SHA-256 del modulo en el manifest
    cambia.

T8. **Universo.** Interseccion entre cohortes, lista de simbolos y
    huella del universo verificadas. Verificable por
    `holdout_intersection.py` + asserts sobre
    `holdout_intersection.json` (n_intersection, n_only_cohort,
    cohort_subset_of_r3000).

T9. **Estado del holdout.** Despues de conocer los resultados, el
    conjunto cambia al estado correspondiente a la candidata
    evaluada. Verificable por test que lee
    `REGISTRO.yaml` y comprueba que `SOW-HOLDOUT-*` tiene
    `experimentos_consumidos` con la candidata en cuestion.

T10. **Git.** Los parquets estan ignorados y no estan versionados.
     Verificable por `git check-ignore` (ignorado) Y
     `git ls-files --error-unmatch` (NO versionado, exit != 0).

---

## 10. Procedimiento de promocion (cuando se apruebe)

1. Confirmar P1 (veredicto §8) y P2 (firma del auditor §7).
2. Verificar `holdout_manifest.py --verify` exit 0 y
   `capture_experiment_env.py --prereg-commit <SHA>` exit 0 para la
   candidata.
3. Anadir el ticker a `data/universe_expansions.yaml` en el bloque
   del grupo correspondiente, con `fecha` y `motivo`.
4. Ejecutar `py run.py` para descargar el ticker al parquet
   productivo.
5. Verificar cobertura efectiva del grupo con
   `py scripts/audit/p1_h02_cobertura_efectiva.py`.
6. Registrar en `REGISTRO.yaml`: tickers promovidos, grupo destino,
   uso permitido, SHA del commit, `experiment_id` del manifest,
   informe del validador, referencia a la firma del auditor.
7. Confirmar que el ticker desaparece de
   `outputs/audit/p1_h02_tickers_sin_datos.csv`.
8. No borrar el ticker del parquet holdout: deja de ser holdout para
   ese ticker, pero el parquet conserva la historia.

---

## 11. Condiciones de firma

El contrato se considera firmado si el auditor externo responde
explicitamente con:

- **Aprobado:** la v1.1 pasa a `FIXED` con SHA del commit del
  contrato. SOW-HOLDOUT-IMMUTABLE cierra. Los 35 tickers siguen sin
  promocion hasta la firma especifica de P2 (§7).
- **Aprobado con cambios:** se anotan los cambios, se versiona como
  v1.2, se re-revisa. Sin firma v1.2, no se promociona ningun
  ticker.
- **Rechazado:** el frente se cierra como `WONT_FIX` con la razon
  del auditor. Los parquets siguen usandose por los validadores
  existentes, pero no se declaran holdout formalmente. El frente
  `TOP20-FALTANTES-DATOS` queda bloqueado en su parte de holdout.

Sin respuesta explicita, el ticket sigue `IN_PROGRESS` y no se
promueve ningun ticker desde holdout.

---

## 12. Impacto en TOP20-FALTANTES-DATOS

Una vez firmado este contrato, `TOP20-FALTANTES-DATOS` puede
reformularse como:

- **35 pendientes de decision de promocion** (bloqueado por auditor
  §7 P2).
- **2 pendientes de descarga** (DDOG, MRNA; sin bloqueo externo).
- **3 pendientes de historico suficiente** (WBD, HONA, SPCX; sin
  accion posible).

---

## 13. Estado normativo

Borrador v1.1. Sin validez normativa hasta firma del auditor
externo. Cualquier promocion de tickers desde holdout antes de la
firma es una violacion del contrato del sistema.

Hasta la firma de la v1.1, `SOW-HOLDOUT-IMMUTABLE` permanece
`IN_PROGRESS`. No se autoriza ninguna promocion de tickers desde
holdout a produccion.

---

## 14. Resumen de cambios v1.0 -> v1.1

| Seccion | Cambio |
|---|---|
| 1. Proposito | Amplia con distincion inmutabilidad fisica vs independencia estadistica (H-04). |
| 2. Definiciones | H-07: poblacion real, limitaciones, huecos, interseccion 266/268. |
| 3. Consumidores | Sin cambios funcionales. Referencia a `capture_experiment_env.py`. |
| 4. Reglas I1-I5 | I3 reescrita (huella en manifest, no "reproducible"). I5 nueva (huella desde 2026-10-28, no historica). |
| 5. Estado del holdout | NUEVA. H-04 opcion B (item por holdout con `experimentos_consumidos`). Estado inicial: cohort y r3000 = CONSUMED(SOW-V6). Condicion sobre SOW-BEAR. |
| 6. Congelacion experimento | NUEVA. H-01, H-02. Manifest + `capture_experiment_env.py` + evidencia temporal. |
| 7. Condiciones promocion | P1-P5. P2 delimita unidad exacta (H-08). |
| 8. Veredictos | NUEVA. H-05. FUNCIONA / NO_FUNCIONA / INCONCLUSO / INSUFFICIENT_SAMPLE. |
| 9. Tests T1-T10 | T4 ampliado (git ls-files). T6-T10 nuevos (H-06). |
| 10. Procedimiento | Ampliado con `capture_experiment_env.py` y mas campos en REGISTRO. |
| 11-13 | Sin cambios funcionales. |
| 14. Resumen cambios | NUEVA. Este bloque. |

---

Fin del contrato SOW-HOLDOUT-IMMUTABLE v1.1 (borrador).
