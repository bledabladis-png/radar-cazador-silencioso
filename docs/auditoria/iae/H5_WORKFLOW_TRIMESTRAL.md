# H5 - Workflow trimestral SEC 13F: analisis forense y plan

**Fecha:** 2026-09-27
**Origen:** hallazgo H5 del dictamen externo (`AUDITORIA_EXTERNA_2026-09-27.md`)
**Estado:** AUDITADO / HALLAZGOS ABIERTOS
**Bloqueante activo:** H5.3 (Q2 2026 ingestado manualmente)
**Referencia:** `update_sec_13f.yml`, `scripts/update_sec_13f.py`, commit `c5e3ee0`

---

## 1. Estado de los sub-hallazgos

| ID | Hallazgo | Severidad | Estado |
|---|---|---|---|
| H5.1 | Criterio de seleccion duplicado (workflow 51d vs script 60d) | MEDIO | ABIERTO |
| H5.2 | Fallo de ingesta sin alerta (no retry, no Issue) | MEDIO | ABIERTO |
| H5.3 | Q2 2026 ingestado manualmente (commit c5e3ee0) | BLOQUEANTE | ABIERTO |
| H5.4 | Workflow no valida formato de inputs.quarter | BAJO | ABIERTO |
| H5.5 | Cache no invalida ante republicaciones SEC | BAJO | ABIERTO |
| H5.6 | Official List Q2 404 resuelto con sufijo -txt | HISTORICO | CERRADO |
| H5.7 | Commit mixto c5e3ee0 (13F + outputs pipeline) | BAJO | HISTORICO |

**No declarar H5 "CERRADO"** mientras H5.3 sea bloqueante.

---

## 2. Evidencia forense del commit c5e3ee0

### 2.1. Anatomia del commit

| Campo | Valor |
|---|---|
| SHA | `c5e3ee061b1cce60edeb6339aca6cec6fa26d6d8` |
| Autor | `bledabladis <bledabladis@gmail.com>` |
| Fecha | Thu Sep 24 04:51:57 2026 +0200 |
| Mensaje | `feat(iae): Q2 2026 ingestado (7 parquets + manifest + latest_quarter)` |

### 2.2. Ficheros del commit

| Fichero | Cambio | Tipo |
|---|---|---|
| `data/sec_13f/latest_quarter.txt` | 1 linea | Ingesta |
| `data/sec_13f/manifests/sec_13f_2026Q2.json` | 106 lineas (nuevo) | Ingesta |
| `outputs/history/macro_regime.csv` | 1 linea | Pipeline (fuera de ingesta) |
| `outputs/state/mte_state.json` | 6 lineas | Pipeline (fuera de ingesta) |
| `outputs/state/slpm_state.json` | 4 lineas | Pipeline (fuera de ingesta) |
| `.github/workflows/update_sec_13f.yml` | NO tocado | - |

### 2.3. Diagnostico

**H5.3 - Q2 ingestado manualmente.** El commit lo firma un autor humano
en horario laboral (04:51 local, antes del cron tipico de 17:06 UTC).
No hay traza de ejecucion de `update_sec_13f.yml` para Q2 2026. Los
manifests Q4 y Q1 fueron regenerados a las 16:46:09 del mismo dia
(24/09), probablemente por una reejecucion manual de
`update_sec_13f.py --backfill 2`.

**H5.7 - Commit mixto.** El commit combina la ingesta 13F con outputs
del pipeline (`macro_regime.csv`, `mte_state.json`, `slpm_state.json`)
que deberian haber sido commiteados por `daily_run.yml`. La ingesta
trimestral y el pipeline diario son procesos separados; mezclarlos en
un mismo commit impide auditar cada cambio por separado y ensucia el
historial.

---

## 3. Causa raiz de H5.3

Tres posibles causas, no mutuamente excluyentes:

**Causa A - El workflow no disparo.** El cron es `17 6 20 2,5,8,11 *`.
Para Q2 2026 (periodo terminado 30/06), el mes de disparo es agosto.
Si GitHub Actions no ejecuto el cron ese dia, la ingesta no ocurre.

**Causa B - El workflow disparo pero fallo silenciosamente.** El
workflow no tiene retry, no abre Issue, no bloquea el siguiente step.
Un fallo pasa desapercibido hasta la proxima ingesta.

**Causa C - El mantenedor prefirio ingesta manual.** Sin traza en el
log de Actions, no hay evidencia de que el workflow corriera.

Con la informacion disponible no se puede distinguir A de B de C. Esa
indistincion es en si misma parte del problema: **falta observabilidad
sobre el cron trimestral**.

---
## 4. Fixes propuestos

### 4.1. H5.3 - Trazabilidad de ejecuciones trimestrales

**Objetivo:** que cualquier ingesta quede registrada con su origen.

**Cambio:** `update_sec_13f.py` acepta `--source` con valores
`cron` | `dispatch` | `manual`, escribe en el manifest de cada
trimestre el campo `ingest_source` + `ingest_actor` (env var
`GITHUB_ACTOR` o `USERNAME` del entorno local). El commit automatico
del workflow incluye `[source=cron]` o `[source=dispatch]` en el
mensaje, distinguible del commit manual.

**Verificacion:** cualquier ingesta Q3 2026 en adelante lleva el campo.
Para Q2 2026 (ya ingestado manualmente) se anade retroactivamente un
campo `ingest_source: "manual_initial"` en su manifest, con nota.

### 4.2. H5.1 - Unificar criterio de seleccion de trimestre

**Problema:** workflow filtra por 51 dias, script por 60 dias. Dos
umbrales distintos para el mismo concepto.

**Cambio:** mover el umbral a `config/settings.py`
(`SEC_13F_QUARTER_LAG_DAYS`) y consumirlo desde ambos sitios. Valor
unico: 60 dias. El 51 del workflow era un margen conservador
(evitar disparo prematuro) sin justificacion estadistica.

### 4.3. H5.2 - Retry + alerta en fallo de ingesta

**Cambio en `update_sec_13f.yml`:**
- Step de ingesta con `retry` a nivel de step via wrapper bash
  (3 intentos, sleeps 30/90/180s) para errores de red.
- Tras el step, si `outcome == failure`, abrir Issue con label
  `sec-13f-failure` usando el mismo patron de `issue_manager.py`.
- El workflow NO abre Issue si el fallo es por rate-limit
  (detectable por HTTP 429 en el log).

### 4.4. H5.4 - Validacion de inputs.quarter

**Cambio en `update_sec_13f.yml`:** step inicial de validacion que
rechaza `inputs.quarter` que no matchee `^\d{4}Q[1-4]$`. Falla el
workflow con mensaje claro antes de intentar descarga.

### 4.5. H5.5 - Cache invalidation

**Problema:** SEC puede republicar un trimestre (correcciones,
cambios de formato TXT). La cache por `latest_quarter` no lo detecta.

**Cambio:** la key de cache incluye sha256 del manifest del trimestre
mas reciente. Si SEC republica, el sha256 cambia, la key cambia, la
cache se invalida. La politica de "republicacion no detectada" queda
cubierta sin trabajo adicional.

### 4.6. H5.7 - Separacion de commits

**Cambio en `update_sec_13f.yml`:** el step de commit hace `git add`
selectivo:
- `data/sec_13f/latest_quarter.txt`
- `data/sec_13f/manifests/sec_13f_*.json`
- `data/mappings/cusip_radar_crosswalk.csv`
- `data/sec_13f/official_list_13f/13flist_*.txt`

Nunca incluye `outputs/`. Los outputs los commitea `daily_run.yml`.
El workflow trimestral no toca `outputs/`.

**Nota:** este cambio ya podria estar aplicado. Verificar contra el
workflow actual antes de reimplementar.

---
## 5. Plan de cierre de H5

**Orden recomendado:**

1. **H5.3 + H5.7** (bloqueante + commit mixto). Un solo commit.
   Trazabilidad de ingestas + separacion de `outputs/`. Verificable
   en la proxima ejecucion del workflow (Q3 2026: agosto 2026 ya paso,
   disparo el 20 de noviembre).
2. **H5.1 + H5.4** (criterio unificado + validacion input). Un commit.
   Bajo riesgo, tests unitarios.
3. **H5.2** (retry + alerta). Un commit. Mayor superficie, requiere
   tests de workflow.
4. **H5.5** (cache invalidation). Un commit. Bajo riesgo.

**Verificacion transversal:** tras los 4 commits, lanzar
`update_sec_13f.yml` en `workflow_dispatch` con `quarter=2025Q4`
(sin tocar nada, solo validacion). El workflow debe:
- Validar input OK.
- Detectar que la cache ya cubre Q4.
- No abrir Issue.
- No commitear.

**No bloquea H1-B.** H5 es operacional; H1-B es de logica de calculo.
Se pueden cerrar en paralelo.

---

## 6. Que NO se hace

- No se reescribe el workflow desde cero.
- No se toca `run.py` ni la logica de calculo del NIPC.
- No se modifica el commit `c5e3ee0` (es historico).
- No se elimina la posibilidad de ingesta manual; solo se traza.

---

## 7. Rollback

Cada fix es un commit independiente. `git revert <sha>` limpio.

---

## 8. Pendientes de decision

1. **Umbral unificado de trimestre:** 60 dias (mi propuesta) o el
   mantenedor fija otro valor.
2. **Formato de `ingest_actor`:** `GITHUB_ACTOR` en CI, `getpass.getuser()`
   en local. Alternativa: campo siempre vacio en local, `"<local>"`
   en CI no aplica. El mantenedor decide.
3. **Label de Issue:** `sec-13f-failure` (nueva) o reutilizar
   `health-check` (existente).

---

**Fin del documento.**
Estado H5: AUDITADO / HALLAZGOS ABIERTOS. H5.3 no se declara cerrado
hasta que exista al menos una ingesta por cron con trazabilidad
completa.