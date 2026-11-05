# 07 - RUNBOOK

**Procedimiento operativo de los crons. NO normativo sobre diseno.**
**Se amplia con cada cron observado. Se poda cuando un workflow se retira.**

---

## 1. PROPOSITO

Este documento describe **que hacer** cuando un cron del sistema dispara (o
no dispara). No explica el diseno del sistema (eso es 02_ARQUITECTURA) ni
el metodo de patch (eso es 01_METODO). Es procedimiento puro.

**Cuando ampliarlo.** Cada vez que se observa un cron por primera vez, se
anade su resultado. Si dispara OK, la entrada sirve de baseline. Si falla,
se anade la contingencia aplicada y se decide si merece fix o WATCHED.

**Cuando podarlo.** Si un workflow se retira del repo, su seccion se
elimina. Si un procedimiento se vuelve obsoleto, se tacha con fecha.

---

## 2. INVENTARIO DE CRONS

### 2.1. Cron diario

Pipeline principal:

    Workflow:      daily_run.yml
    Cron (UTC):    5 slots: 23:17, 03:17, 05:17, 07:17, 11:17
    Proposito:     Pipeline principal

health_check los verifica con severidad 0->OK, 1-2->WARN, 3+->FAIL.

Actualizacion de macros FRED (independiente del pipeline):

    update_macro_manual.yml         17 6 * * *            CEST 08:17
      -> data/macro_manual/ (diario)

### 2.2. Cron semanal

    Workflow:      health_check.yml
    Cron (UTC):    lunes 06:47
    Proposito:     Termometro del sistema

    update_sec_nport.yml            17 6 * * 3            CEST 08:17
      (miercoles; reacciona en <=7 dias a la SEC)
      -> outputs/history/sec_nport_*, qqq_nport_flow.csv

### 2.3. Cron trimestral (meses 1, 4, 7, 10)

Disparan el dia 1 del mes (holdings):

    update_sector_holdings.yml      47 2 1 1,4,7,10 *    CEST 04:47
      -> data/etf_holdings.csv
    update_index_holdings.yml       17 4 1 1,4,7,10 *    CEST 06:17
      -> data/index_holdings.csv
    update_european_holdings.yml    17 5 1 1,4,7,10 *    CEST 07:17
      -> data/index_holdings.csv

### 2.3.1. Cron semestral (15 ene, 15 jul)

    update_qqq_sec_flow.yml         17 6 15 1,7 *        CEST 08:17
      -> outputs/history/qqq_sec_primary_flow.csv

### 2.4. Cron trimestral (meses 2, 5, 8, 11)

    update_sec_13f.yml              17 6 20 2,5,8,11 *   CEST 08:17
      -> ingesta SEC 13F (lag 50d, ver D4 2026-09-30)

#### 2.4.1. Verificacion H5.3 (cron 13F nov 2026)

    gh run list --workflow=update_sec_13f.yml --event schedule --limit 5
    Get-Content data\sec_13f\ingest_traces.jsonl -Encoding UTF8 | Select-Object -Last 5

Esperado: entrada con {"source":"cron", "outcome":"INGEST", ...} y
sin rama SKIP inesperada. Fecha de referencia: 20-nov-2026 06:17 UTC.

Si no hay entrada: revisar seccion 4.1 (cron no disparo). Si el
outcome es SKIP: determinar causa (quarter ya ingestado, feriado,
fallo upstream).

Trazabilidad: H5.3 implementada 2026-09-30. Ver 03_IAE.md seccion
H5 y 04_HISTORICO.md.

### 2.5. Deuda estructural detectada 2026-09-30

Los 4 workflows de §2.3 nunca se han disparado por schedule en el
historial visible. Todos los runs conocidos son workflow_dispatch
(9-sep, 25-ago).

**Matiz operativo (historico 1-oct-2026, corregido 2026-11-03):** los
3 workflows sector/index/european disparan el dia 1. `update_sec_nport`
es SEMANAL (miercoles 06:17 UTC), no trimestral dia 20: la SEC publica
N-PORT con latencia irregular y el cron semanal reacciona en <=7 dias
(dictamen H5, 2026-10-16; YAML cambiado entonces, runbook actualizado
2026-11-03 tras revision de workflows). Ver `update_sec_nport.yml`
para el cron vigente. Test centinela: `tests/test_runbook_vs_workflows.py`.

**Riesgo de colision el 1-oct:** `update_macro_manual` dispara DIARIO
a `17 6 UTC` (06:17 CEST), justo cuando `european` lleva 1h corriendo.
Sus `concurrency.group` son distintos (`update_index_holdings_csv` vs
`update_macro_manual`), asi que no se serializan. Ambos hacen push a
main. Con delays de GitHub (2-8h documentados), `european` puede seguir
activo cuando arranca `macro_manual`. Si el push de uno choca, ver
seccion 4.3.

Esto anade un riesgo que el health_check no puede medir: si el cron esta
mal escrito, si permissions no basta, o si un path no existe en el
runner, se vera por primera vez manana.

---

## 3. VERIFICACION POST-CRON

Ejecutar en este orden. Solo el paso 1 es imprescindible; los demas solo
si el paso 1 muestra problema.

### 3.1. Se disparo?

    gh run list --workflow=NOMBRE.yml --event schedule --limit 3

Criterios:
- Sin runs por schedule: el cron no disparo. Ver §4.1.
- Conclusion success: paso a 3.2.
- Conclusion failure o cancelled: paso 3.3 + §4.2.

### 3.2. Actualizo el fichero?

    git log origin/main --oneline -3
    git log -1 --format="%h %ad %s" --date=iso -- FICHERO

Criterios:
- Commit del bot con mensaje de actualizacion en origin/main: OK.
- Sin commit: puede ser normal (no habia cambios). Ver 3.3.

### 3.3. Inspeccion del log

    gh run view RUN_ID --log-failed

Criterios:
- No hay cambios: OK. El workflow es idempotente.
- Push fallido tras 3 intentos: conflicto de rebase. Ver §4.3.
- startup_failure (sin jobs): ver §4.4.
- Error en un step concreto: leer el log del step y decidir.

### 3.4. Contenido del fichero

Solo si hay sospecha. Para los CSV de holdings:

    Get-Content data\etf_holdings.csv | Select-Object -First 2
    Get-Content data\etf_holdings.csv | Select-Object -Last 2

Coherencia: cabecera etf,ticker,identifier,weight, sin tickers con guion
suelto, sin pesos negativos. **Nota:** 2026-09-30 se observo
etf_holdings.csv con ticker guion (placeholder) y XASU6 con peso negativo.
Los consume data_loader._ticker_list filtrando por INVALID_TICKERS.
Deuda documentada, no bug activo.

---

## 4. CONTINGENCIAS

### 4.1. El cron no disparo

Sintoma: gh run list --event schedule no devuelve runs en la ventana
esperada.

Causas tipicas:
1. Delay del scheduler de GitHub. Documentado 2-8h. Esperar.
2. Scheduler best-effort. GitHub no garantiza todos los slots. Ver los
   slots de daily_run para precedente (finding 2026-09-27).
3. Workflow mal escrito. Verificar el on.schedule.cron en el YAML. Cinco
   campos, formato POSIX. Validador: crontab.guru.

Accion: esperar 8h. Si sigue sin disparar, workflow_dispatch manual para
recuperar el dia. Si es daily_run, la cache de slots lo detectara.

### 4.2. Fallo con jobs

Sintoma: conclusion failure, hay jobs y steps con X.

Accion:
1. gh run view RUN_ID -> identificar step fallido.
2. gh run view RUN_ID --log-failed -> linea exacta del error.
3. Si es HTTP 4xx/5xx de una fuente externa: puede ser transitorio.
   Retry manual via workflow_dispatch.
4. Si es un AssertionError o un path inexistente: bug de codigo. Abrir
   frente de auditoria.

Distinguir de startup_failure: si gh run view RUN_ID no muestra jobs
(lista vacia), es startup_failure. Ver §4.4.

### 4.3. Conflicto de rebase en push

Sintoma: log muestra Push fallido tras 3 intentos.

Causa tipica: dos workflows tocan el mismo fichero con concurrency
distinto. Caso observado 2026-09-30: update_index_holdings y
update_european_holdings ambos escriben data/index_holdings.csv. Si
coinciden en vuelo, el segundo rebase falla.

Accion:
1. gh run view RUN_ID --log -> ver el conflicto.
2. Resolver en local:
       git pull --rebase origin main
       # resolver conflicto
       git rebase --continue
       git push
3. Revisar si el concurrency.group de ambos deberia ser el mismo (fix
   estructural pendiente de decidir).

### 4.4. startup_failure

Sintoma: gh run view RUN_ID muestra X en el run pero cero jobs.
gh run view --log no devuelve nada.

Causas tipicas:
1. Cancelacion manual. El caso mas comun. conclusion cancelled, sin jobs.
   No es fallo del workflow.
2. YAML invalido. El workflow no parsea. Revisar el commit del YAML
   contra el run.
3. head_sha inexistente. El commit ya no esta en la rama (rebase o
   force-push posterior).

Accion: si es cancelled, ignorar. Si es failure sin jobs, revisar el
YAML. Si sigue sin explicacion, workflow_dispatch manual.

### 4.5. workflow_dispatch force=true fuera de horario de mercado

Sintoma: run lanzado con `gh workflow run daily_run.yml -f force=true`
a media sesion USA contamina `outputs/` y `data/` con un snapshot
distinto al que publicara el scheduled de la noche. El snapshot
contaminado se **pushea a main** automaticamente (step "Commit and
push hist/state"), asi que el siguiente scheduled no lo revierte:
lo encuentra como base.

Causa real (verificada 2026-11-05): el gate `pipeline_gate.py` corre
siempre y decide `should_run` (CURRENT -> False, READY -> True). El
input `force` NO se pasa al gate (sus args son `--now`,
`--target-session`, `--slot`). Lo que hace `force=true` es saltarse
el resultado del gate en el `if:` del job `run-system`:

    if: needs.gate.outputs.should_run == 'true' || github.event.inputs.force == 'true'

El gate no se bypassea: el job de ejecucion ignora su veredicto. Por
eso el run corre aunque el gate diga CURRENT (receipt ya emitido para
la target_session) o NOT_READY (probe coverage insuficiente).

Por que corrompe. Si el run corre entre 13:30 y 20:00 UTC (apertura y
cierre NYSE), procesa datos a medio dia: precios intradiarios, Yahoo
sin cierre definitivo, FRED con lag indeterminado. Los writers
publican con la fecha efectiva resuelta (habitualmente la sesion
anterior), pero los CSV y manifests reflejan el snapshot intradiario.
El step `Commit and push hist/state` hace:

    git add -u
    git add -A outputs/history
    git add -u outputs/state
    git add data/cboe_vix3m.parquet data/cboe_vix3m.parquet.manifest.json
    git add data/market_data.parquet.manifest.json data/stock_prices.parquet.manifest.json
    git add data/mappings/radar_target_catalog.csv data/mappings/catalog_manifest.json
    git add data/mappings/catalog_snapshots/
    git add data/lse_close_provenance.json
    git commit -m "Daily hist/state "
    git pull --rebase origin main
    git push   # 3 intentos con backoff

El commit contaminado llega a `main`. Ficheros que se contaminan y
se pushean:

- `outputs/history/*.csv` (appends y reescrituras del run).
- `outputs/state/*.json` excepto los gitignored
  (`completion_receipt.json`, `validation_gate_result.json`).
- `data/market_data.parquet.manifest.json` (sha256 del parquet
  intradiario).
- `data/stock_prices.parquet.manifest.json` (idem).
- `data/cboe_vix3m.parquet` + `data/cboe_vix3m.parquet.manifest.json`.
- `data/lse_close_provenance.json`.
- `data/mappings/radar_target_catalog.csv`.
- `data/mappings/catalog_manifest.json`.
- `data/mappings/catalog_snapshots/`.

Backup pre-force. Como el step auto-pushea, el backup no es opcional:
es el SHA de `main` antes de lanzar.

    Set-Location D:\Macro_Sectorial
    git fetch
    git rev-parse origin/main   # -> PRE_SHA, anotar

Opcional: tag de referencia local y remoto.

    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    git tag "pre-force-$stamp" origin/main
    git push origin "pre-force-$stamp"

Procedimiento de emergencia (paso a paso):

1. Verificar ventana UTC. Si 13:30-20:00 UTC, NO lanzar. Esperar a
   pre-apertura (< 13:30) o post-cierre (> 20:00).

       (Get-Date).ToUniversalTime()

2. Anotar el SHA de retorno.

       git fetch
       git rev-parse origin/main   # -> PRE_SHA

3. Lanzar el dispatch y seguirlo.

       gh workflow run daily_run.yml -f force=true
       gh run list --workflow=daily_run.yml --limit 1
       gh run watch <run-id>

4. Verificar el resultado. Si el run termino OK, mirar el commit
   `Daily hist/state ` que acaba de pushear el bot:

       git fetch
       git log --oneline -5 origin/main

   Si aparece un commit nuevo del bot con timestamp del force
   dispatch, comparar los `outputs/history/*.csv` contra el
   snapshot previo (PRE_SHA).

5. Si el snapshot es correcto (raro en horario intradiario, posible
   en post-cierre): nada que hacer. El bot ya lo pusheo y es el
   nuevo estado.

6. Si el snapshot es incorrecto (habitual en horario intradiario):

       # Opcion A: revert completo del commit contaminado
       git revert <sha_del_bot> --no-edit
       git push

       # Opcion B: restauracion quirurgica (mas preciso si el bot
       # commitio otros cambios legitimos entre medias)
       git checkout PRE_SHA -- outputs/history outputs/state \
           data/market_data.parquet.manifest.json \
           data/stock_prices.parquet.manifest.json \
           data/cboe_vix3m.parquet \
           data/cboe_vix3m.parquet.manifest.json \
           data/lse_close_provenance.json \
           data/mappings/radar_target_catalog.csv \
           data/mappings/catalog_manifest.json \
           data/mappings/catalog_snapshots
       git add -A
       git commit -m "revert: descarta snapshot intra-sesion del force dispatch <stamp>"
       git push

7. Regenerar ESTADO_SISTEMA y verificar gobernanza.

       py scripts\generate_estado_sistema.py
       py scripts\verifica_gobernanza.py

8. El proximo scheduled (slot 23:17 UTC) reescribira el snapshot
   correcto. Verificar en `outputs/history/data_quality.csv` que la
   fila de la fecha efectiva es la esperada y que no hay filas con
   fecha no-bursatil.

Caso real: 2026-10-29 19:19 UTC. Lanzado y cancelado a los pocos
minutos tras detectar el error. Sin side effects: el step Commit no
llego a ejecutarse. Si hubiera llegado, el commit `Daily hist/state `
habria pusheado un snapshot intra-sesion a `main`; el slot 23:17 UTC
lo habria sobreescrito con el correcto, pero el commit intermedio
queda en historia (ruido de auditoria).

### 4.6. git push sin verificacion de exit code

Sintoma: commit exitoso en local, push fallido silencioso (el bot
pusheo concurrentemente y el push local fue rechazado con "fetch
first"). El commit queda huerfano en local, CI no corre.

Causa: `git push` sin `if ($LASTEXITCODE -ne 0)` trata el rechazo como
warning. El script sigue. La sesion cierra con `git log --oneline -1`
mostrando el commit local, pero `origin/main` apunta al commit del bot.

Accion: cualquier bloque de push debe usar:

    git push
    if ($LASTEXITCODE -ne 0) {
        git fetch origin
        git pull --rebase origin main
        if ($LASTEXITCODE -ne 0) { throw "rebase conflicto" }
        git push
        if ($LASTEXITCODE -ne 0) { throw "push fallido" }
    }

Caso real: 2026-10-30, push de `c879df71` rechazado por el bot.
El bloque tenia el guard; fetch + rebase + retry resolvio. Sin el guard
habria quedado huerfano.

---

## 5. COMPLETION RECEIPT (contrato v1, 2026-10-01)

El gate del `daily_run` declara `CURRENT` cuando existe un
`completion receipt` valido para la `target_session`. El receipt es
un artifact inmutable en GitHub Actions, subido por el job
`run-system` con `if: success()` tras `guard_coverage` y push.

**Ficheros:**
- `scripts/pipeline_gate.py::find_completion_receipt` (lee).
- `scripts/write_completion_receipt.py` (escribe).
- `docs/auditoria/daily_run_gate_contrato_v1.md` (contrato).
- `docs/auditoria/daily_run_gate_dictamen.md` (dictamen externo).

**Cuando verificar el receipt:**

- Tras un `daily_run` verde, comprobar que el artifact existe:
  FENCE
  gh api "repos/bledabladis-png/radar-cazador-silencioso/actions/runs/<RUN_ID>/artifacts" --jq '.artifacts[] | [.name, .size_in_bytes, .created_at] | @tsv'
  FENCE
  Debe aparecer `completion-receipt-<target_session>`.
- En el siguiente slot, el gate debe devolver `CURRENT`. Verificacion:
  FENCE
  gh run view <RUN_ID> --json jobs
  FENCE
  `run-system` debe aparecer con `conclusion=skipped`.

**Sintomas tipicos y diagnostico:**

- **Gate devuelve READY tras un pipeline verde**: el receipt no se
  detecta. Causas observadas 2026-10-01:
  - `RECEIPT_FILENAME` desalineado con el basename del artifact
    (upload-artifact@v4 sube el basename).
  - `GH_TOKEN` ausente en el env del step `Run gate`. Sin token,
    `_download_receipt_json` sale antes de la llamada.
- **`Commit and push hist/state` falla con "untracked files in
  outputs/history or outputs/state"**: algun JSON nuevo en
  `outputs/state/` no esta en `.gitignore` con exclusion explicita.
- **Receipt expirado (90 dias)**: el gate cae al probe -> READY ->
  rerun. Es fail-safe, no bug.

**Invariantes (I1-I8 del contrato).** Si alguna falla, abrir frente
con el dictamen como referencia. Los 12 tests en
`tests/test_pipeline_gate_receipt.py` las cubren.

---

## 6. DEUDAS OBSERVADAS 2026-09-30


Hallazgos del reconocimiento previo al cron del 1-oct. No abren frentes
todavia. Se anotan para cuando se decida actuar.

- etf_holdings.csv con residuos. Ticker guion suelto y XASU6 con peso
  negativo. Filtrados por INVALID_TICKERS en data_loader. Deuda
  estructural: el CSV contiene datos que el consumidor descarta.
- index_holdings.csv con mtime desajustado. mtime 2026-09-20, ultimo
  dispatch 2026-09-09. 11 dias de diferencia. Sin verificar si otro
  workflow o ejecucion local lo modifico.
- sec_nport_positions_2026q2.csv con mtime desajustado. mtime
  2026-09-24, ultimo dispatch 2026-09-10. 14 dias de diferencia. Mismo
  caso.
- concurrency.group desalineado. Los 4 workflows de §2.3 tienen grupos
  distintos, pero dos de ellos escriben el mismo fichero. Sin proteccion
  contra colision.

---

**Fin del runbook.**
