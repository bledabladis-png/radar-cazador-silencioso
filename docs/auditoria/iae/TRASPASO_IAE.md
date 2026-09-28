# TRASPASO DEL MODULO IAE - Documento maestro para asistente entrante

**Actualizado:** 2026-09-28
**Estado:** modulo IAE con dictamen externo CERRADO (C2 residual abierto no bloqueante).
**Referencia unica del modulo:** `docs/auditoria/iae/IAE_MAESTRO.md`.
**Estado autoritativo:** `docs/auditoria/iae/ESTADO_SISTEMA.md` + `ESTADO_DECLARADO.md`.

---

## 0. Como usar este documento

Este documento es **capa complementaria, no normativa**. La norma vigente
es `docs/auditoria/PROMPT_MAESTRO.md` (v7.15). Si hay conflicto, gana el
PROMPT.

Cuando entres a trabajar:

1. Lee este documento completo.
2. Lee `ESTADO_SISTEMA.md` (hechos autogenerados).
3. Lee `ESTADO_DECLARADO.md` (fases y deuda activa).
4. No empieces a proponer trabajo sin haber confirmado la asimilacion
   completa al usuario.

### 0.1. Lecturas obligatorias antes de operar

1. **`PROMPT_MAESTRO.md`** v7.15 - rol, metodologia, prohibiciones.
2. **`ESTADO_SISTEMA.md`** - HEAD, tests, integridad. Se regenera con
   cada commit.
3. **`ESTADO_DECLARADO.md`** - estado de hallazgos H1-A a H5, deuda.
4. **`IAE_MAESTRO.md`** - referencia unica del modulo.
5. **`golden/current.json`** + **`golden/12_5_historic.json`** -
   baselines.

---

## 1. El sistema en una frase

Radar de rotacion sectorial: sistema determinista, descriptivo, auditable.
Sin ML predictivo, sin optimizacion de parametros, sin automatizacion de
trading. Todos los outputs son diagnosticos, no recomendaciones.

---

## 2. Que es el IAE (modulo que retomas)

El Institutional Accumulation Engine (IAE) analiza posiciones
institucionales a partir de filings 13F de la SEC. Responde:

> Dado un universo de tickers del radar, que instituciones han aumentado
> o reducido su posicion entre dos trimestres consecutivos, y con que
> cobertura de datos se puede afirmar.

Opera integrado en el pipeline productivo del radar con dependencia
unidireccional (`run.py -> IAE`). Prohibido lo inverso.

Documentacion completa: `IAE_MAESTRO.md`.

---

## 3. Estado del IAE (verificado 2026-09-28)

    Suite global       2226 passed + 2 skipped + 0 failed
    Suite IAE          845 passed (criterio AST, 45 ficheros)
    Validation Gate    10/10
    Cobertura radar    313/313 (100%)
    Working tree       LIMPIO
    Push               SI (sincronizado con origin/main)

Auditoria externa del IAE (2026-09-27): **APROBADO CON CONDICIONES**.
Estado final tras cierre 2026-09-28:

| Hallazgo | Estado |
|---|---|
| H1-A | CERRADO |
| H1-B | CERRADO con C2 abierto no bloqueante (920) |
| H2   | CERRADO |
| H3   | CERRADO |
| H4   | CERRADO (golden/current.json ACTIVE_WITH_OPEN_DISCREPANCY) |
| H5.1 | CERRADO |
| H5.2 | CERRADO |
| H5.3 | TRAZABILIDAD IMPLEMENTADA - pendiente cron nov 2026 |
| H5.4 | CERRADO |
| H5.5 | CERRADO |
| H5.6 | CERRADO (historico) |
| H5.7 | HISTORICO |
| O1   | CERRADO |

**Baseline H1-B:**

- `local_observed_nipc_total` = **-4.264.449.012** (reproducible por
  nosotros con HEAD `72fa824`).
- `external_auditor_reference` = **-4.264.449.932** (declarado por
  auditoria, sin cadena de custodia reproducible).
- `reconciliation_delta` = **920**.
- `reconciliation_status` = **OPEN**.

Ver `INFORME_H1B_RECONCILIACION_FINAL.md` y `golden/current.json`.

---

## 4. Cronologia de sesiones recientes

### 2026-09-27

- Auditoria externa del IAE abierta. Dictamen APROBADO CON CONDICIONES.
- H1-A cerrado por el auditor.
- H1-B / H5.3 bloqueantes.
- Desbloqueo SEC Official List Q2 2026 (commit `08a6c6a`): NIPC Q1 2026
  -> Q2 2026 calculado por primera vez = 8.254.818.120.

### 2026-09-28 (sesion de cierre)

12 commits encadenados:

1. Reconciliacion H1-B: descubrimiento de que la Official List Q4 2025
   existe pero no contiene todos los option CUSIPs observados en filings.
2. Implementacion H1-B v2 (clasificacion CALL/PUT por Official List).
3. Correccion H1-B v2.3 (solo excluir OPTION confirmada; UNRESOLVED
   mantenido).
4. H5.1-H5.5 cerrados (workflow trimestral).
5. H2 (cifra 45 ficheros) + H3 (clasificacion de scripts) cerrados.
6. O1 (ausencia de imputacion) cerrado.
7. H1-B cerrado con C2 abierto. H4 cerrado (`current.json`).
8. Hallazgo metodologico importante: el valor `-4.264.449.932` citado
   por el auditor **no es reproducible en local**. Tras multiples
   variantes, ninguno coincide. Se registra como discrepancia abierta.

---

## 5. Auditoria externa del IAE - CERRADA

### 5.1. Contexto

Fase D del plan original: envio del modulo a auditor externo (LLM
independiente con acceso al repo). El auditor emitio dictamen con
hallazgos H1-A, H1-B, H2, H3, H4, H5, O1.

### 5.2. Estado final

Todos los hallazgos del dictamen estan cerrados, salvo:

- **C2 (920):** discrepancia entre `-4.264.449.012` (nuestro) y
  `-4.264.449.932` (del auditor). El auditor lo acepto como
  discrepancia abierta no bloqueante y autorizo el cierre de H1-B con
  esa condicion.
- **H5.3:** trazabilidad implementada, pendiente de verificacion en el
  cron real de noviembre 2026.

### 5.3. Hallazgo residual C2 (920)

**Estado:** OPEN, no bloqueante.

**Resumen:**

- `local_observed_nipc_total = -4.264.449.012`
- `external_auditor_reference = -4.264.449.932`
- `reconciliation_delta = 920`

**Leccion metodologica:** en el primer informe se presento el valor del
auditor como si fuera un golden medido, con "coincidencia dentro del
margen de redondeo". El auditor lo rechazo correctamente: el valor
`-4.264.449.932` no tiene cadena de custodia reproducible desde este
repositorio. Se corrigio a "no reproducido bajo las configuraciones y
el entorno controlados actualmente auditados".

**Si el auditor aporta comando + HEAD:** reproducir y cerrar por
atribucion.
**Si no lo aporta:** mantener `reconciliation_status = OPEN`. No
convertir el 920 en "tolerancia aceptada" sin decision explicita.

### 5.4. Prohibiciones consolidadas

- **No congelar como golden contractual** ningun valor que no sea
  reproducible por el sistema local.
- **No presentar cifras declaradas por el auditor como hechos
  verificados.** Marcarlas siempre como "declaradas, sin cadena de
  custodia".
- **No mezclar breakdowns de estados distintos** en la misma tabla
  (el error que produjo el "943.371 fantasma").
- **No inventar explicaciones** para discrepancias no reproducidas.
  Declararlas y preguntar.
- **No modificar codigo productivo sin ciclo previo.**
- **No integrar a produccion sin validacion funcional** + dictamen del
  auditor cuando aplique.
- **No push sin validacion local completa.**

---

## 6. Hallazgo H1-A - prueba forense (cerrada, no reabrir)

**Estado:** CERRADO.

`scripts/iae_contractual_nipc_e2e.py` fallaba contra el golden §12.5
(`553.319` vs `553.321`). Causa: regeneracion del crosswalk por Fase G.
Prueba forense: restaurando el crosswalk `7caa86b`, el E2E reproduce
el golden con tolerancia cero.

**No reabrir.**

---

## 7. Reglas de trabajo

### 7.1. Metodologia

- Ver el contenido real antes del patch.
- Un cambio = una verificacion = un commit.
- Local-first.
- Deteccion por contenido > por indices.
- Rollback quirurgico.
- Saber parar (ROI < 1 -> WONT FIX).
- Auditor externo antes de decisiones irreversibles.
- Gate 0 antes de tocar datos.
- Un fix destapa el siguiente.

### 7.2. Estructura estandar de un patch (Python)

1. Backup `.orig` (si aplica).
2. Detectar BOM: `data[:3] == b'\xef\xbb\xbf'`.
3. Detectar LF/CRLF: `count('\r\n') vs count('\n')`. Preservar EOL.
4. Aplicar cambio con `read_bytes()` / `write_bytes()`.
5. Validar sintaxis con `ast.parse()`.
6. Si falla -> restaurar backup.
7. Escribir con `encode()` correcto.
8. Asserts `== 1` acumulados antes del write unico.
9. `pyflakes` + `compileall` + suite antes de commit.

### 7.3. Verificaciones obligatorias antes de commit

    py -m compileall . -q
    py -m pyflakes . 2>&1
    py -m pytest tests/ validation/ -q --tb=short

Esperado: `compileall OK` - `pyflakes LIMPIO` - `2226 passed + 2 skipped`.

### 7.4. Prohibiciones duras

- No tocar datos historicos sin snapshot pre/post.
- No limpiar fecha sospechosa sin doble candado
  (`is_market_day(date)==False` AND `date in CONFIRMED_SET`).
- No mezclar commits de ingesta 13F con outputs del pipeline
  (leccion H5.7).
- No imputar valores (ni en IAE ni en radar).
- No convertir ausencia de evidencia en evidencia negativa.

### 7.5. Trampas de PowerShell

- Here-strings >20 lineas o >5 `$` van a `_patch_XXX.py` con
  `[System.IO.File]::WriteAllText`.
- Backticks Markdown en here-string se corrompen: usar placeholders
  ASCII + `chr(96)`.
- `[System.IO.File]::WriteAllText` usa el CWD del proceso .NET, no el
  de PowerShell: usar `Join-Path $PWD`.
- `py -c "..."` con comillas dobles anidadas rompe el parser. Escribir
  a fichero temporal.
- BOM: si el fichero empieza con `\xef\xbb\xbf`, leer con
  `encoding="utf-8-sig"` y escribir con `utf-8-sig`.
- CRLF vs LF: normalizar a LF para matching de anchors, restaurar al
  escribir si el fichero original era CRLF.

### 7.6. Personalidad

Directo, estructurado, orientado a la accion. Autocritico. Sin
grandilocuencia. Documentar todo. Reconocer cuando una investigacion
no merece la pena. Espanol tecnico, tuteo neutro. Sin emojis
decorativos.

---

## 8. Mapa de documentos

| Documento | Proposito | Estado |
|---|---|---|
| `PROMPT_MAESTRO.md` | Norma vigente | v7.15 |
| `ESTADO_SISTEMA.md` | Hechos autogenerados (HEAD, tests) | regenerado por script |
| `ESTADO_DECLARADO.md` | Fases y deuda activa | actualizado 2026-09-28 |
| `IAE_MAESTRO.md` | Referencia unica del modulo IAE | vigente |
| `AUDITORIA_EXTERNA_2026-09-27.md` | Dictamen original del auditor | historico |
| `DISENO_H1B.md` | Diseno tecnico H1-B v2 | aprobado e implementado |
| `CONSULTA_H1B.md` | Consulta Gate 0 al auditor | historico |
| `H5_WORKFLOW_TRIMESTRAL.md` | Analisis y cierre H5 | cerrado salvo H5.3 |
| `INFORME_H1B_POST_IMPLEMENTACION.md` | Resultado empirico H1-B | cerrado |
| `INFORME_H1B_RECONCILIACION_FINAL.md` | C1+C2+C3 final | cerrado |
| `INFORME_O1_AUSENCIA_IMPUTACION.md` | Barrido imputaciones | cerrado |
| `golden/12_5_historic.json` | Snapshot historico (FROZEN) | inmutable |
| `golden/current.json` | Baseline local (ACTIVE_WITH_OPEN_DISCREPANCY) | vigente |
| `golden/README.md` | Explicacion del directorio golden | vigente |
| `TRASPASO_IAE.md` | Este documento | actualizado 2026-09-28 |

---

## 9. Scripts reproducibles clave

- `scripts/iae_test_census.py` - censo AST de tests.
- `scripts/iae_coverage.py` - cobertura reproducible.
- `scripts/iae_contractual_nipc_e2e.py` - E2E contractual.
- `scripts/iae_contractual_coverage.py` - coverage contractual.
- `scripts/iae_reconciliation_b1.py` - reconciliacion radar+complemento.
- `scripts/iae_validate_crosswalk_openfigi.py` - validacion externa.
- `scripts/iae_identity_uniqueness_audit.py` - unicidad por shareClassFIGI.
- `scripts/update_sec_13f.py` - ingesta trimestral.
- `scripts/download_official_list_13f.py` - Official List SEC.

---

## 10. Comandos de arranque

    Set-Location D:\Macro_Sectorial
    git log --oneline -5
    git status -sb
    py scripts\generate_estado_sistema.py
    py -m pytest tests/ -q --tb=line
    py -m pyflakes src\ scripts\
    py -m compileall . -q

Esperado:

- `ahead 0, behind 0`, working tree limpio.
- **2226 passed + 2 skipped + 0 failed.**
- pyflakes silencio, compileall OK.

Censo IAE (tarda unos segundos):

    py scripts\iae_test_census.py

Esperado: 45 ficheros, 845 tests.

---

## 11. Contacto con el auditor

El auditor externo es otro LLM. Se le entrega material en formato
Markdown autocontenido. Sus dictamenes son vinculantes para cerrar
hallazgos.

**Cuando consultar al auditor:**

- Antes de cerrar hallazgos bloqueantes.
- Antes de congelar baselines.
- Si hay discrepancia entre valores locales y externos.

**Cuando NO consultar:**

- Para decisiones de implementacion sin impacto contractual.
- Para refactors internos sin cambio de contrato observable.

**Formato:** documento Markdown con contexto, evidencia empirica y
preguntas concretas. Nada de "que opinas", si "aceptas X dado Y".

---

## 12. Resumen operativo

1. El modulo IAE esta implementado, testeado e integrado en produccion.
2. La auditoria externa esta cerrada. Queda C2 (920) como discrepancia
   abierta no bloqueante.
3. El baseline local es `-4.264.449.012`. La referencia externa
   `-4.264.449.932` es declarada, no reproducible.
4. H5.3 espera verificacion en el cron de noviembre 2026.
5. No quedan bloqueantes activos.

**Proximo trabajo real (no hay bloqueantes):**

- Esperar verificacion H5.3 en cron nov 2026.
- Si el auditor aporta comando + HEAD del 932, cerrar C2 por atribucion.
- Cualquier ciclo nuevo requiere consulta previa al usuario.
