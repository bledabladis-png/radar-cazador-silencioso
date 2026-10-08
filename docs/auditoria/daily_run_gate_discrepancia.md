
# Discrepancia: el gate del daily_run nunca devuelve CURRENT en CI

**Expediente para auditor externo. Autocontenido.**
**Fecha:** 2026-10-01
**Autor:** equipo interno.
**Pregunta:** decidir entre opciones §9 o proponer otra.

---

## 1. RESUMEN EJECUTIVO

El job `gate` del workflow `daily_run.yml` implementa idempotencia
entre los 5 slots diarios del pipeline. Si el slot N ya produjo la
`target_session` con cobertura suficiente, los slots posteriores deben
skipear el pipeline.

**Hoy (2026-10-01), los 4 slots que dispararon ejecutaron el pipeline
completo.** Los 4 `gate` devolvieron `READY`, los 4 `run-system`
corrieron, los 4 duraron ~18-21 min. Coste: ~80 min de runner mas 4
oportunidades de commit del bot.

**Causa raiz identificada:** `_manifest_satisfies` exige verificar el
sha256 del parquet real contra el manifest. En CI, el parquet esta
gitignored y no existe en el runner; el job `gate` no lo restaura. La
funcion devuelve `False` incondicionalmente. El gate cae al probe
(Yahoo), el probe pasa (100%), el gate devuelve `READY`.

El chequeo sha256 se introdujo en `2a640f2` (2026-09-29) para proteger
contra falso-CURRENT en local (parquet local vs manifest de CI). En
local es correcto. En CI convierte el gate en un no-op.

**Verificable sin GitHub:** apendice A.

---

## 2. CONTEXTO OPERATIVO

### 2.1. Slots del daily_run

`daily_run.yml` tiene 5 slots de cron (fuente unica en
`scripts/pipeline_gate.py`):

```
CRON_SLOTS = (
    "17 23 * * *",
    "17 3 * * *",
    "17 5 * * *",
    "17 7 * * *",
    "17 11 * * *",
)
```

Los 5 apuntan a la misma `target_session`: la ultima sesion bursatil
cerrada de USA. El proposito es tolerar la latencia de Yahoo, que
puede tardar horas en consolidar el Close.

### 2.2. Contrato del gate

`evaluate(target_session)` devuelve uno de 4 estados:

| Estado | should_run | Significado |
|---|---|---|
| CURRENT | False | Ya hay artefacto con cobertura para target_session. No correr. |
| READY | True | No hay artefacto pero la fuente ya publico. Correr. |
| NOT_READY | False | Fuente no publica aun. Skip, esperar slot siguiente. |
| ERROR | False | Fallo tecnico del gate. Skip. |

El gate viaja por `GITHUB_OUTPUT`. `run-system` corre solo si
`should_run == 'true'`.

### 2.3. Job gate en el yml (extracto)

```
jobs:
  gate:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    permissions:
      contents: read
    outputs:
      state: ${{ steps.gate.outputs.state }}
      should_run: ${{ steps.gate.outputs.should_run }}
      expected_session: ${{ steps.gate.outputs.expected_session }}
      is_last_slot: ${{ steps.gate.outputs.is_last_slot }}
      reason: ${{ steps.gate.outputs.reason }}
    steps:
      - name: Checkout repo
        uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.14'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run gate
        id: gate
        env:
          PYTHONIOENCODING: utf-8
        run: python scripts/pipeline_gate.py --slot "${{ github.event.schedule }}"
```

Observacion: no hay `actions/cache` ni `actions/download-artifact` en el
job `gate`. El runner arranca limpio: solo lo que viene en el checkout
Git.

---

## 3. EVIDENCIA EMPIRICA

### 3.1. Los 4 slots del 2026-10-01

| Run ID | Created (UTC) | gate | run-system | manifest_coverage |
|---|---|---|---|---|
| 36804481750 | 02:09 | success READY | success | null |
| 36847604125 | 10:11 | success READY | success | null |
| 36856527967 | 11:37 | success READY | success | null |
| 36876380805 | 14:26 | success READY | success | null |

`manifest_coverage: null` en el output JSON del gate significa que la
rama `CURRENT` de `evaluate()` **no se alcanzo nunca**.

Extracto literal del log del gate de 36804481750:

```
"state": "READY",
"should_run": true,
"reason": "probe coverage 100.00% >= 90%",
"expected_session": "2026-09-30",
"is_last_slot": false
```

Idem para 36876380805. Los 4 slots coincidieron en `expected_session`
2026-09-30 y `probe coverage 100.00%`.

### 3.2. Verificacion directa

Cualquiera puede reproducir el bug sin GitHub usando el script del
apendice A.

---

## 4. CODIGO RELEVANTE

### 4.1. `_manifest_satisfies` (scripts/pipeline_gate.py L86-112)

```
def _manifest_satisfies(manifest, target_session):
    q = manifest.get("quality")
    if not isinstance(q, dict):
        return False
    if q.get("expected_session") != target_session:
        return False
    cov = q.get("coverage_pct_last")
    if not isinstance(cov, (int, float)):
        return False
    if cov < MANIFEST_THRESHOLD:
        return False
    declared_sha = (manifest.get("artifact") or {}).get("sha256")
    if not isinstance(declared_sha, str) or not declared_sha:
        return False
    p = PROJECT_ROOT / MANIFEST_PATH.replace(".manifest.json", "")
    if not p.exists():
        return False
    return _sha256_file(p).lower() == declared_sha.lower()
```

`MANIFEST_PATH = "data/stock_prices.parquet.manifest.json"`.
`PROJECT_ROOT` = raiz del repo. El parquet `data/stock_prices.parquet`
esta en `.gitignore`.

### 4.2. `evaluate` (L202-...)

```
def evaluate(target_session):
    manifest = _read_manifest(MANIFEST_PATH)
    if _manifest_satisfies(manifest, target_session):
        return {
            "state": "CURRENT",
            "should_run": False,
            "reason": "manifest already covers {0}".format(target_session),
            "expected_session": target_session,
            "probe_coverage": None,
            "manifest_coverage": manifest["quality"]["coverage_pct_last"],
        }
    probe_cov, probe_err = _probe_panel(target_session)
    if probe_err is not None:
        return {"state": "ERROR", "should_run": False, ...}
    if probe_cov >= PROBE_MIN_COVERAGE:
        return {
            "state": "READY",
            "should_run": True,
            "reason": "probe coverage {0:.2%} >= {1:.0%}".format(
                probe_cov, PROBE_MIN_COVERAGE),
            ...
        }
    return {"state": "NOT_READY", "should_run": False, ...}
```

---

## 5. DIAGNOSTICO

En CI, cada corrida del job `gate`:

1. `actions/checkout@v4` trae el repo tracked: incluye
   `data/stock_prices.parquet.manifest.json` (commitado).
2. **No** trae `data/stock_prices.parquet` (gitignored).
3. `actions/setup-python@v5` + `pip install` no lo generan.
4. `python scripts/pipeline_gate.py`:
   - `_read_manifest` lee el manifest commiteado -> dict OK.
   - `_manifest_satisfies`:
     - `expected_session` del manifest == target -> OK.
     - `coverage_pct_last` >= 0.95 -> OK.
     - `declared_sha` presente -> OK.
     - `p = PROJECT_ROOT / "data/stock_prices.parquet"` -> no existe.
     - `p.exists()` -> False -> `return False`.
   - `_probe_panel`: consulta Yahoo. Tras cierre USA, devuelve 100%.
   - `probe_cov >= 0.90` -> `READY`, `should_run=True`.

**La rama `CURRENT` es inalcanzable en CI.** El chequeo sha256,
correcto en local, es un bloqueo incondicional en CI por construccion
(mide un fichero que no viaja por git).

---

## 6. HISTORIA DEL CHEQUEO SHA256

Commit que lo introdujo: `2a640f2` (2026-09-29). Mensaje original
(resumen):

```
Bug de diseno detectado 2026-09-29: los .parquet estan gitignored
pero los .manifest.json estan tracked. En cada git pull el manifest
viene de CI mientras el parquet es del ultimo run local. Ambos
guard_coverage y pipeline_gate leian solo el manifest, nunca el
parquet.

Fix: ambos scripts calculan sha256 del parquet sibling y abortan
si no coincide con artifact.sha256 del manifest. Fail-closed:
manifest sin sha256, parquet ausente, o mismatch -> fallo.

No rompe CI: pipeline_gate corre antes del run (parquet no existe
-> READY, correcto); guard_coverage corre despues (parquet+manifest
recien generados -> coinciden).
```

**El propio mensaje declara la situacion como "no rompe CI".** No
rompe la ejecucion del CI, pero si inutiliza el gate: en CI jamas
puede alcanzar CURRENT.

---

## 7. COMPORTAMIENTO DEL PROBE

El probe descarga un panel fijo de 20 tickers USA
(`GATE_PANEL_USA`) para `target_session`, usando `curl_cffi` con
`impersonate=chrome`. Calcula `n_valid / n_total`. Cobertura de
100% del panel implica `READY`.

**Implicacion del diseno actual:** en CI, el gate decide
exclusivamente en funcion del probe. El manifest (y su sha256) no
participa en la decision. La idea original — "no correr si ya
corrimos con exito esta sesion" — no existe en CI: el gate solo
sabe preguntar a Yahoo "¿ya tienes el Close?", y Yahoo siempre dice
que si desde pocas horas tras el cierre.

Esto explica el sintoma: los 4 slots del 2026-10-01 corrieron
completo, en secuencia, sobre la misma `target_session`.

---

## 8. IMPACTO OPERATIVO

**Nota sobre cuota de minutos:** el repo es publico. En GitHub Actions,
repos publicos tienen minutos de runner ilimitados y gratuitos. El
limite de 2000 min/mes aplica solo a repos privados. Por tanto el
argumento de coste NO aplica en este repo.

Quedan los siguientes impactos, que si son reales:

- **Ruido en el repo:** cada run `chore(daily)` commitea outputs con
  variacion de floats. Ejemplo 2026-10-01: `etf_primary_flow.csv`
  +/- 12 lineas de ruido de precision. Multiplicado por 4-5 slots/dia.
  Contamina la historia de git con cambios no semanticos.
- **Presion sobre proveedores externos:** Yahoo, SSGA, BlackRock,
  CFTC, Invesco, Amundi, Xetra, BME, Euronext consultados 5 veces
  al dia en lugar de 1. Riesgo teorico de rate limiting, no observado
  hasta hoy.
- **Serializacion:** `run-system` tiene `timeout-minutes: 90`. El
  `concurrency.group: daily-run` con `queue: max` serializa los
  slots. Si la latencia de GitHub retrasa varios slots y coinciden
  en vuelo, la ventana diaria de procesamiento se estrecha. En el
  peor caso, el slot 5 puede arrancar de madrugada del dia siguiente.
- **Consumo de recursos del sistema:** 5 pipelines completos diarios
  multiplican por 5 el desgaste sobre la base de datos de outputs
  (`outputs/history/*.csv`) via append_dedup, con riesgo de
  duplicacion si algun writer no es idempotente.

Riesgo actual: no hay evidencia de dano material (cuota, rate limit,
corrupcion) todavia. El unico dano observable es el ruido de commits
del bot. La correccion del bug es una mejora de higiene y de
contratos, no un incendio.

---

## 9. OPCIONES CONSIDERADAS

Sin decidir. En orden de menor a mayor intervencion.

### (a) Cachear parquet+manifest al final de run-system

Anadir un `actions/cache/save` en `run-system` (post-pipeline) con
clave derivada de `target_session`. En `gate`, un `actions/cache/restore`
previo al run. Si la cache existe y el sha256 coincide con el
manifest, `_manifest_satisfies` devuelve `CURRENT`.

- **Pros:** comportamiento intacto del chequeo. Idempotencia real.
- **Contras:** acopla el gate a un artefacto de cache. Si el save
  falla en el slot 1, los slots 2-5 corren. Coste: ~10 MB de cache
  por dia.

### (b) Degradar sha256 a best-effort en CI

`_manifest_satisfies` lee `expected_session` y `coverage_pct_last`
como fuente principal; solo verifica sha256 si el parquet existe.

```
p = PROJECT_ROOT / MANIFEST_PATH.replace(".manifest.json", "")
if not p.exists():
    return True
return _sha256_file(p).lower() == declared_sha.lower()
```

- **Pros:** cambio minimo. Idempotencia restaurada en CI por
  confianza en el manifest commiteado.
- **Contras:** en CI acepta el manifest sin verificar. Ruptura de la
  regla "fail-closed".

### (c) Invertir el contrato: el gate pregunta al pipeline

En lugar de leer un artefacto local, que lea un marcador publicado en
la rama:

```
# escrito por run-system al final OK:
# outputs/state/daily_completed.json
{"target_session": "2026-09-30", "gate": "10/10", "manifest_sha": "..."}
```

`gate` lee ese fichero si existe y si `target_session` coincide ->
`CURRENT`.

- **Pros:** no acopla al parquet. Un solo artefacto (small, JSON) que
  si viaja por git.
- **Contras:** nueva pieza en el repo. Otro escritor atomico que
  auditar.

### (d) Reducir los slots a 1

Si el timing de Yahoo ya no se manifiesta, los 5 slots son
redundancia innecesaria. Reducir a 2 (o 1) elimina el problema sin
tocar el gate.

- **Pros:** simplifica. Elimina coste y ruido.
- **Contras:** pierde resiliencia contra retrasos de Yahoo. Requiere
  evidencia de que 1 slot basta a lo largo del ano.

---

## 10. PREGUNTAS AL AUDITOR

1. ¿Cual de (a)-(d) es mas robusta? ¿Otra?
2. En el caso (b), ¿el cambio de politica "fail-closed -> confiar en
   manifest en CI" introduce riesgo inaceptable?
3. ¿Tiene sentido redisenar el contrato para que el gate no dependa
   nunca del parquet (opcion c)?
4. ¿Hay riesgo de que la opcion (c) genere otro "contrato que miente"
   (mismo patron A6.1-02 / D16 ya visto en el repo)?

---

## APENDICE A - Script reproducible

`docs/auditoria/daily_run_gate_repro.py` demuestra el bug sin GitHub
ni yfinance. Ejecutar:

```
py docs\auditoria\daily_run_gate_repro.py
```

El script monkeypatchea `MANIFEST_PATH` y `PROJECT_ROOT` a un
directorio temporal con manifest sintetico pero sin parquet, llama
`_manifest_satisfies`, e imprime el resultado. No depende de red.
