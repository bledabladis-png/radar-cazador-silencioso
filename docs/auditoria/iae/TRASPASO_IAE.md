# TRASPASO DEL MODULO IAE - Documento maestro para asistente entrante

**Fecha:** 2026-09-27
**HEAD de referencia:** `de8b8ffa34102a19a6005fce584a34a8268dae87`
**Proposito:** onboarding completo del asistente que retoma el trabajo del Ingeniero Supervisor sobre el modulo IAE del Radar de Rotacion Sectorial. Este documento es autocontenido.

---

## 0. Como usar este documento

Si eres un asistente entrante (otro LLM o humano):

1. **Lee este documento completo antes de tocar nada.** Tiene 12 secciones; la seccion 5 (trabajo en curso) es la operativa critica.
2. **Consulta la seccion 8 (mapa de documentos)** para saber donde vive cada tipo de informacion.
3. **Respeta las reglas de la seccion 7 (metodologia y prohibiciones).** Son restricciones duras del sistema, no sugerencias.
4. **No toques codigo productivo del IAE sin dictamen del auditor externo.** Hay una auditoria abierta (seccion 5).
5. **Al arrancar**, ejecuta los comandos de la seccion 10 y compara con lo esperado.

### 0.1. Lecturas obligatorias antes de operar

En este orden:

1. **`PROMPT_MAESTRO.md`** - norma vigente del sistema (rol, metodologia, arquitectura).
2. **`TRASPASO_IAE.md`** - este documento (contexto operativo del IAE).
3. **`IAE_MAESTRO.md`** - referencia unica del modulo IAE. 110 KB, 2400+ lineas.
   Contiene la arquitectura completa, formulas exactas del NIPC, superficie
   publica (115 funciones), evidencia empirica E2E y limitaciones declaradas.
   **Imprescindible para entender como funciona el modulo.**
4. **`AUDITORIA_EXTERNA_2026-09-27.md`** - dictamen del auditor externo.
   Estado de los hallazgos H1-A / H1-B / H4 / H5.
5. **`DISENO_H1B.md`** - diseno tecnico del fix bloqueante principal.
   Pendiente de aprobacion del auditor antes de implementar.
6. **`ESTADO_DECLARADO.md`** - fases del IAE, deuda activa, prohibiciones.
7. **`ESTADO_SISTEMA.md`** - hechos verificables (HEAD, tests, integridad).
   Autogenerado, no editar a mano.
8. **`EVIDENCIA_FORENSE_H1A.md`** - cadena forense del drift historico (opcional,
   solo si necesitas reproducir la prueba del cierre H1-A).

---

## 1. El sistema en una frase

El Radar de Rotacion Sectorial es un sistema determinista y descriptivo de analisis macro-sectorial. Ejecuta un pipeline diario (`run.py`) que descarga datos de mercado, calcula regimenes, indicadores y scores sectoriales, y genera un reporte Markdown con validacion automatica 10/10. No predice, no opera, no usa ML.

**Entorno:** Windows, PowerShell, Python 3.14 (`py`).
**Repo:** `https://github.com/bledabladis-png/radar-cazador-silencioso`
**Raiz local:** `D:\Macro_Sectorial`

---

## 2. Que es el IAE (modulo que retomas)

El **Institutional Accumulation Engine (IAE)** es el modulo del radar que analiza posiciones institucionales a partir de filings 13F de la SEC. Responde a:

> Dado el universo de tickers del radar (catalogo radar de 242 tickers USA), que instituciones han aumentado o reducido su posicion entre dos trimestres consecutivos, y con que cobertura de datos se puede afirmar.

**Caracteristicas:**
- Descriptivo (no predice).
- Determinista (misma entrada -> mismo output).
- Auditable (trazable a filings SEC).
- Publica **una sola seccion** en el reporte diario: `## Acumulacion Institucional (13F)`.
- Metrica principal: **NIPC** (Net Institutional Position Change). Nota semantica: NO es flujo economico; es variacion neta de posiciones reportadas.

**Ubicacion:** `src/institutional_accumulation/` (36 ficheros, 8.009 LOC).
**Integracion:** `run.py` invoca `compute_iae_section`; `src/report_generator.py` invoca `render_iae_section`.

---

## 3. Estado del IAE (verificado 2026-09-27)

| Aspecto | Valor |
|---|---|
| Suite global | **2205 passed + 2 skipped + 0 failed** |
| Suite IAE (45 ficheros) | 845 tests |
| Cobertura IAE | 90% (3.088 stmts, 298 miss) |
| Aislamiento arquitectonico | 0 violaciones |
| Integracion produccion | Verificada (run.py + report_generator.py) |
| Ultimo run local | exit 0, 953s, Validation Gate 10/10 |
| Cache 13F | 3 trimestres (Q4 2025, Q1 2026, Q2 2026) |
| Official List 13(f) | Q4 2025, Q1 2026, Q2 2026 descargadas |
| Seccion IAE en reporte | OPERATIVA (NIPC Q1->Q2: 8.254.818.120) |
---

## 4. Cronologia de la sesion anterior (2026-09-27)

Sesion larga de cierre de backlog. 17 commits pusheados. Los mas relevantes:

| Commit | Que hizo |
|---|---|
| `a0c0ce4`, `3aff5be` | K-BLACKROCK-CSV-STALE-01 (reorder CSV + docs) |
| `ff782d7`, `3e967ed` | save_scenario eliminado (codigo muerto) |
| `1805766`, `dc6aea2` | Cat 4 desmontado + F5.6-01 + F5.7-01 + F7-03 |
| `9cce7dd` | F5.6-02b FRED cleanup (DGS2, put_call, get_options_data muerto) |
| `4fd5d57` | F7-04 freshness -> settings.py |
| `bfe030c` | F7-01 sector_correlation_matrix dead end eliminado |
| `743896e` | F5.6-03 CboeProvider guardas |
| `62d4992` | F5.7-14 CFTC thresholds -> settings.py |
| `b7d1c31` | F5.7-19 N-PORT dinamico (trimestres, XML) |
| `3376bb3` | F5.7-05 FINRA get_latest_week memoize |
| `d84902b` | F5.6-06 datetime.now como fecha de ejecucion (NO BUG) |
| `c233855` | Cierre documental de la sesion (v7.12, rev.7) |
| `08a6c6a` | **SEC Official List Q2 desbloqueada (sufijo -txt)** |
| `de8b8ff` | Cierre documental del desbloqueo (v7.13, rev.8) |

**Estado:** todo pusheado a `origin/main`. Working tree limpio.

---

## 5. TRABAJO EN CURSO - Auditoria externa del IAE

### 5.1. Contexto

La "Fase D" del plan original del IAE estaba pendiente desde antes del 2026-09-23: enviar el modulo a un **auditor externo** para dictamen independiente. El 2026-09-27 se materializo:

1. Se preparo un **informe de auditoria externa completo** (ver `AUDITORIA_EXTERNA_2026-09-27.md`).
2. Se entrego a un auditor independiente (LLM externo con acceso al repo).
3. El auditor emitio dictamen: **APROBADO CON CONDICIONES** con hallazgos H1 a H5.

### 5.2. Estado de los hallazgos (dictamen final)

| Finding | Estado | Descripcion breve |
|---|---|---|
| **H1-A** | CERRADO | Drift historico del E2E explicado por regeneracion de crosswalk (prueba forense CUSIP-por-CUSIP). |
| **H1-B** | ABIERTO / BLOQUEANTE | Clasificacion CALL/PUT incompleta. Filas con `PUTCALL=NULL` pero `TITLEOFCLASS` de opcion pasan el filtro. Dano medido: **+53.228.375** sobre el NIPC actual. |
| **H4** | ABIERTO, pendiente H1-B | Golden no versionado como artefacto independiente. |
| **H5.1** | ABIERTO | Criterio de seleccion duplicado (workflow 51d vs script 60d). |
| **H5.2** | ABIERTO | Fallo de ingesta trimestral sin alerta (no retry, no Issue). |
| **H5.3** | ABIERTO / BLOQUEANTE | Q2 2026 ingestado manualmente (commit `c5e3ee0`, autor humano, fuera de cron). |
| **H5.4** | ABIERTO | Workflow no valida formato de `inputs.quarter`. |
| **H5.5** | ABIERTO | Cache de parquets no invalida ante republicaciones SEC. |
| **H5.6** | CERRADO | Official List Q2 fail (404) resuelto con sufijo `-txt` (commit `08a6c6a`). |
| **H5.7** | HISTORICO | Commit mixto (`c5e3ee0` mezclo 13F + outputs pipeline). |

### 5.3. Instruccion del auditor sobre H1-B (BLOQUEANTE)

El auditor ha sido explicito:

> **Implementar el fix definitivo de H1-B basado en identificacion por CUSIP + tipo de security temporalmente apropiado, usando la Official List como fuente primaria.**

Reglas que exige:

1. **Fuente primaria de tipo:** Official List de la SEC (Option Indicator en posicion 10).
2. **Resolucion temporal:** Q4 2025 usa lista Q4 2025; Q1 2026 usa lista Q1 2026; etc.
3. **Aplicacion en capa de identidad**, no en `_filter_canonical`.
4. **Regex `TITLEOFCLASS`** como defensa secundaria, no como autoridad principal.
5. **Auditoria explicita** de exclusiones: contadores `excluded_by_official_list_type`, `excluded_by_titleofclass`, `unresolved_type`, `conflict_filer_vs_external`.
6. **No congelar NIPC baseline** hasta despues del fix. `-4.317.678.307` = `PRE_H1B_OBSERVATION`; `-4.264.449.932` = "resultado de mitigacion".
7. **Rehacer** E2E + reconciliacion + cobertura + determinismo tras el fix.

### 5.4. Lo que NO debes hacer (prohibiciones consolidadas del auditor)

- **No tocar codigo productivo del IAE** hasta cerrar H1-B con el auditor.
  El diseno debe aprobarse antes de implementar (ver `DISENO_H1B.md`).
- **No congelar ningun NIPC como baseline.** Ni `-4.317.678.307` ni
  `-4.264.449.932`. Los nombres validos son `PRE_H1B_OBSERVATION` y
  `MITIGATION_RESULT`. El baseline contractual final se creara solo
  despues de cerrar H1-B y con aprobacion del auditor.
- **No actualizar los `EXPECTED` hardcoded del E2E** para "hacer pasar" el
  test. El E2E fallido es evidencia de un drift real, no un bug del test.
- **No elevar cobertura de tests artificialmente.** El auditor exige
  verificacion funcional, no numeros de cobertura.
- **No commitear outputs del pipeline** mezclados con cambios de 13F
  (leccion H5.7 del auditor).
- **No usar `--quarter` con formato distinto a `YYYYQn`** (mayusculas).
  Derivado de H5.4: el workflow no valida formato.
- **No modificar `_filter_canonical` sin coordinar con el diseno H1-B.**
  La regex `TITLEOFCLASS` es defensa secundaria, no autoridad principal.
  El filtro primario va en `operational_universe.py::_apply_5_3b`.
- **No introducir `fillna`/`ffill`/`bfill`/`interpolate`** en los caminos
  del IAE sin declaracion explicita. Derivado de O1 del auditor.

### 5.5. Siguiente paso inmediato

**Gate 0 de H1-B** - verificacion previa al fix:

1. Verificar que las 3 Official List locales (`data/sec_13f/official_list_13f/13flist_2025Q4.txt`, `13flist_2026Q1.txt`, `13flist_2026Q2.txt`) contienen el Option Indicator correctamente parseable.
2. Confirmar que `023135906` aparece como CALL y `023135106` como COMMON en la lista Q1 2026.
3. Confirmar que `595112903` aparece como CALL y `595112103` como COMMON en la lista Q1 2026.
4. Disenar la integracion en `identity/` (nueva funcion `resolve_security_type_from_official_list(cusip, quarter)`).


---

## 6. Hallazgo H1-A - prueba forense (cerrada, no reabrir)

**Cadena demostrada:**

```
7caa86b  (crosswalk 246 filas: AMZNx2, MUx2)
    |
    | Fase G regenera
    v
HEAD     (crosswalk 245 filas: AMZNx1, MUx1, SPCXx1)
    |
    v
delta_radar_rows = -2    nipc_total = -943.371

Descomposicion: AMZN -943.169  -  MU -202
```

**Prueba:** al restaurar el crosswalk `7caa86b` sobre los parquets actuales, el E2E reproduce **exactamente** el golden seccion 12.5 con tolerancia 0 (553.321 / -4.316.734.936).

**No hay regresion de codigo.** El drift es atribuible integramente a la regeneracion del crosswalk por Fase G.

---

## 7. Reglas de trabajo

### 7.1. Metodologia

- **Gate 0 read-only antes de invertir ciclos.** ~50% de los findings revisados recientemente estaban subrogados o eran falso positivo.
- **Un cambio = una verificacion = un commit.** No mezclar cambios.
- **Local-first.** Refactors grandes: commits locales, verificacion exhaustiva, push unico.
- **Ver antes del patch.** Nunca aplicar sin inspeccionar el bloque exacto.
- **Saber parar.** ROI < 1 -> WONT FIX razonado.

### 7.2. Estructura estandar de un patch (Python)

1. Backup `.orig` / `.bak` (o escribir patch a `_patch_XXX.py`).
2. Detectar BOM: `raw[:3] == b'\xef\xbb\xbf'`.
3. Detectar LF/CRLF: `text.count('\r\n') vs text.count('\n')`.
4. Aplicar con `read_bytes()` / `write_bytes()`.
5. Validar sintaxis con `ast.parse()`.
6. **Todos los asserts ANTES de cualquier write.** Patron: acumular reemplazos en memoria, escribir una sola vez.
7. `compileall . -q` + `pyflakes .` + `pytest tests/ validation/ -q`.

### 7.3. Verificaciones obligatorias antes de commit

```powershell
py -m compileall . -q
py -m pyflakes . 2>&1
py -m pytest tests/ validation/ -q --tb=short
```

Esperado: `compileall OK` - `pyflakes LIMPIO` - `2205 passed + 2 skipped`.

### 7.4. Prohibiciones duras

- No push sin validacion local completa.
- No tocar pesos ni parametros sin justificacion estadistica.
- No inventar datos. Si no hay suficiente -> `N/D`.
- No usar `datetime.now()` como fecha de observacion. Solo como log.
- No mezclar capas de flujo.
- No mezclar commits de ingesta 13F con outputs del pipeline (leccion H5.7).

### 7.5. Trampas de PowerShell

- Here-strings `@'...'@` no expanden `$`. `@"..."@` si, pero hay que escapar.
- Ficheros con `$`, backticks o >20 lineas: escribir a `_patch_XXX.py` con `[System.IO.File]::WriteAllText`.
- Ficheros sin newline final -> anchor con `\n` no cuadra. Usar `repr()` para verificar.
- Mojibake en acentos: anchor ASCII-safe o extraccion por substring.

### 7.6. Personalidad

- Directo, estructurado, orientado a la accion.
- Autocritico, sin grandilocuencia.
- Idioma: espanol tecnico, tuteo neutro.
- Terminas cada mensaje con pregunta accionable.

---

## 8. Mapa de documentos

| Fichero | Proposito |
|---|---|
| `docs/auditoria/PROMPT_MAESTRO.md` | Norma vigente del sistema. **No declara estado.** |
| `docs/auditoria/TRANSFER.md` | Guia de onboarding. **No es fuente de estado.** |
| `docs/auditoria/README.md` | Navegacion. |
| `docs/auditoria/iae/IAE_MAESTRO.md` | **Referencia unica del modulo IAE.** |
| `docs/auditoria/iae/ESTADO_DECLARADO.md` | Fases IAE, deuda activa, prohibiciones. |
| `docs/auditoria/iae/ESTADO_SISTEMA.md` | Hechos verificables. Autogenerado. |
| `docs/auditoria/iae/AUDITORIA_EXTERNA_2026-09-27.md` | Registro consolidado del dictamen. |
| `docs/auditoria/iae/DISENO_H1B.md` | Diseno tecnico del fix bloqueante H1-B. Pendiente de aprobacion. |
| `docs/auditoria/iae/EVIDENCIA_FORENSE_H1A.md` | Cadena forense del cierre H1-A. Reproducible. |
| `docs/auditoria/iae/TRASPASO_IAE.md` | Este documento. |
| `docs/auditoria/iae/evidence/` | Evidencia empirica por ciclo. |

**Documentos historicos (no normativos):** `NIPC_CONTRATOS_SEMANTICOS_v1.md`, `NIPC_COVERAGE_POLICY.md`, `INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md`.

---

## 9. Scripts reproducibles clave

| Script | Proposito |
|---|---|
| `scripts/iae_test_census.py` | Censo AST. Esperado: 45 ficheros, 845 tests. |
| `scripts/iae_coverage.py` | Cobertura de lineas. Esperado: 90%. |
| `scripts/iae_contractual_coverage.py` | Cadena contractual completa. Esperado: exit 0, coverage 1.0. |
| `scripts/iae_contractual_nipc_e2e.py` | E2E contra seccion 12.5. **Falla hoy con HEAD** (H1-A/H1-B). |
| `scripts/iae_reconciliation_b1.py` | Reconciliacion radar + complemento = full. Esperado: PASS. |
| `scripts/iae_identity_uniqueness_audit.py` | Auditoria unicidad shareClassFIGI. Esperado: PASS. |
| `scripts/iae_validate_crosswalk_openfigi.py` | Validacion externa OpenFIGI. Requiere API key. |

---

## 10. Comandos de arranque

```powershell
Set-Location D:\Macro_Sectorial

git log --oneline -5
git status -sb
py scripts\generate_estado_sistema.py
py -m pytest tests/ validation/ -q --tb=short
py scripts\iae_coverage.py
py scripts\iae_contractual_nipc_e2e.py
py scripts\iae_reconciliation_b1.py
```

Esperado:
- HEAD = `de8b8ff`.
- ahead 0, behind 0.
- working tree limpio.
- **2205 passed + 2 skipped + 0 failed.**
- IAE coverage 90%.
- E2E contractual: **FAIL** (esperado).
- Reconciliacion B1: **PASS**.

---

## 11. Contacto con el auditor

El auditor externo es otro LLM. Se le entrega el material en formato Markdown autocontenido. Sus dictamenes son formales y numerados. Los hallazgos reciben IDs (`H1-A`, `H1-B`, `H4`, `H5.x`) y severidad explicita.

**Cuando contactarle:**
- Antes de congelar cualquier NIPC baseline.
- Antes de aplicar el fix de H1-B.
- Antes de cerrar H4 (golden versionado).
- Antes de tocar el workflow `update_sec_13f.yml`.

**Material a preparar:**
- Resumen ejecutivo del ciclo.
- Evidencia forense reproducible.
- Nuevas preguntas explicitas.
- Estado actualizado de hallazgos.

---

## 12. Resumen operativo

**Si el usuario pide "seguir con la auditoria":**
1. Lee `AUDITORIA_EXTERNA_2026-09-27.md`.
2. Confirma estado con comandos de arranque (seccion 10).
3. Arranca el **Gate 0 de H1-B** (seccion 5.5).

**Si el usuario pide "arreglar H1-B":**
1. Verifica que el usuario ha autorizado tocar codigo productivo.
2. Ejecuta Gate 0 primero.
3. Disena el fix segun las 7 reglas del auditor (seccion 5.3).
4. Envia el diseno al auditor antes de codificar.

**En cualquier caso:**
- No inventes datos ni resultados.
- No digas "voy a hacer X" sin hacerlo.
- Terminas cada mensaje con pregunta accionable.

---

**Fin del traspaso.**
Ultima actualizacion: 2026-09-27, HEAD `de8b8ff`.

Documentos anadidos tras la primera version:
- `DISENO_H1B.md` (13547 bytes) - diseno tecnico del fix bloqueante.
- `EVIDENCIA_FORENSE_H1A.md` (7476 bytes) - cadena forense del cierre H1-A.
