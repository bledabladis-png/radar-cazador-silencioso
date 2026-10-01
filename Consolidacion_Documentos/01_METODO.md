# 01 - METODO

**Referencia on-demand. Se consulta antes de tocar codigo.**
**No se pega al arrancar. El arranque es 00_ARRANQUE.md.**

---

## 1. PRINCIPIOS

- **"Ver el contenido real antes del patch."** Nunca aplicar un patch sin haber inspeccionado el bloque exacto con repr() o dump numerado.
- **"Un cambio = una verificacion = un commit."** No mezclar cambios.
- **"Local-first."** Para refactors grandes: 15-20 commits locales, verificacion exhaustiva, push unico al final.
- **"Local-first IAE."** Modulo nuevo, no consolidado: NO push a main hasta validar funcionalidad.
- **"Deteccion por contenido > por indices."** Los indices cambian tras cada extraccion. Usar strings unicos como anclas.
- **"Rollback quirurgico."** Si un patch falla, revertir solo la parte rota.
- **"Saber parar."** Si una tarea tiene ROI < 1, cerrarla como WONT FIX.
- **"Auditor externo antes de decisiones irreversibles."** Gates para arquitectura, contratos y limpieza de datos.
- **"Gate 0 antes de tocar datos o codigo."** Inventario previo completo.
- **"Un fix destapa el siguiente."** Cuando corriges un bug, revisa si el patron se repite en writers/readers hermanos.
- **"Verificar en produccion real, no solo en tests."** Los fixes que tocan writers/readers temporales deben validarse con un run manual.
- **"NO confundir VALID con UNAVAILABLE."** Un artefacto "no pude validarlo" no es lo mismo que "lo valide y paso".

---

## 2. ESTRUCTURA ESTANDAR DE UN PATCH (PYTHON)

1. **Backup** `.orig` / `.bak` (primera vez). Eliminar tras verificar.
2. **Detectar BOM:** `data[:3] == b'\xef\xbb\xbf'`.
3. **Detectar LF/CRLF:** `count('\r\n') vs count('\n')`. Preservar EOL original.
4. **Aplicar cambio con `read_bytes()` / `write_bytes()`.**
5. **Validar sintaxis con `ast.parse()` ANTES de escribir.**
6. **Si falla -> restaurar backup automaticamente.**
7. **Escribir con `encode()` correcto** (`utf-8-sig` si habia BOM, `utf-8` si no).
8. **Para here-strings PowerShell que contienen Python:** usar `@'...'@` (single-quote, no expande variables). Si el contenido lleva `$` o comillas dobles, escribir a `_patch_XXX.py` con `WriteAllText` y ejecutar con `py`. Nunca `py -c` con comillas anidadas.
9. **No usar caracteres especiales (á, é, í, —, →) en patrones de busqueda.** Preferir ASCII-safe.
10. **En here-strings PowerShell con `py -c`**, los escapes `\"` dentro de f-strings rompen el parser.
11. **Un script de patch con multiples `assert text.count(anchor) == 1` debe abortar ANTES de escribir si cualquier assert falla.**
    Para scripts multi-fix: acumular TODOS los reemplazos en memoria y escribir UNA SOLA VEZ al final. NO imprimir `[OK]` de operaciones individuales antes del write final.
12. **Si un here-string contiene muchos `@'` o caracteres `$`, PowerShell puede fallar silenciosamente.**
    Verificar con `Test-Path` + `(Get-Content file | Measure-Object -Line).Lines` antes de ejecutar el patch.
13. **Here-strings PowerShell >20 lineas o >5 `$`: escribir a archivo Python temporal, no pegar en consola interactiva.**
    Patron seguro: `[System.IO.File]::WriteAllText` a `_patch_XXX.py`, luego `py _patch_XXX.py`.
14. **Para checks triviales (`"X" in text`), escribir a `_check_*.py` con `[System.IO.File]::WriteAllText` + `py _check.py`.**
    Nunca `py -c` con comillas dobles anidadas: los escapes `\"` rompen el parser. Tropiezo confirmado 3x.
15. **Backticks en here-string PowerShell se corrompen silenciosamente.**
    Al escribir contenido con triple-backtick (fences Markdown), PowerShell los interpreta como escape.
    Solucion: escribir a script Python con placeholder y aplicar `replace` antes de escribir.
16. **Contenido largo (>20 lineas) se escribe en chunks de maximo 25 lineas.**
    Usar `[System.IO.File]::WriteAllText` para el primer chunk y `[System.IO.File]::AppendAllText` para los siguientes.
17. **Al corregir un valor que puede repetirse en varios sitios del mismo documento, buscar TODAS las ocurrencias antes del patch.**
    Un assert `== 1` solo valida el anchor, no la ausencia de otras ocurrencias.

---

## 3. VERIFICACION OBLIGATORIA PRE-COMMIT

Pega este bloque antes de cualquier commit:

    py -m compileall . -q
    py -m pyflakes src scripts regimes indicators config data\providers validation
    py -m pytest tests/ validation/ -q --tb=short

Esperado:
- compileall OK
- pyflakes LIMPIO (silencio total)
- 2945 passed + 2 skipped + 0 failed (2026-09-30)

Si algun test falla: NO commitear. Diagnosticar primero.

**Tests opt-in (skipped por diseno):** dos familias, ambas excluidas por defecto.

- `@pytest.mark.network` (2 tests en `test_freshness.py`): consultan CBOE/FINRA reales. Excluidos salvo `--run-network`. Implementado en `conftest.py` (`pytest_addoption` + `pytest_collection_modifyitems`). No es deuda: el test comprueba frescura real y no se puede mockear sin perder su objeto.
- `integration_real_data` (8 sitios): requieren parquets locales (gitignored en CI). Skip si no existen. En CI corren en el paso "Validate freshness post-run" tras `run.py`.

Ambos son diseno, no deuda. El skip silencioso no aparece en el resumen; se ve con `-ra` (ya en `pytest.ini`).

---

## 4. VERIFICACION END-TO-END (REFACTORS GRANDES)

Aplicable a cambios que tocan writers, readers, nucleo temporal, o el reporte.

**Criterio fino de aplicacion (D39, 2026-09-30):**

- Fix que puede alterar el output de `run.py` de forma que la suite no lo detecta -> E2E obligatorio. Incluye `src/pipeline/*`, `regimes/*`, `indicators/*` (si se propaga a reporte), `run.py`, `report_generator.py`, `data_loader.py`, `stock_data_loader.py`.
- Fix que cambia una firma, un valor por defecto o un path de fichero con consumidores en `src/pipeline/` -> E2E.
- Fix que solo cambia un comportamiento de fallback (rama defensiva) -> verificacion ligera (tests + probe sobre artefacto real). No `run.py`.
- Helpers puros (`src/utils.py` salvo `write_artifact_with_manifest`), tests, scripts de CI, corpus -> no E2E.
- Checks de `health_check` que solo leen artefactos -> verificacion ligera en produccion real, no `run.py`.

1. **Snapshot pre:** copiar outputs/report/, outputs/state/, outputs/history/ relevantes a `outputs/audit/pre_<bloque>_<stamp>/`.
2. **`py run.py` real (~10-15 min).** No sustituto.
3. **Snapshot post:** copiar los mismos a `outputs/audit/post_<bloque>_<stamp>/`.
4. **Criterios de aceptacion:**
   - exit=0.
   - Validation Gate 10/10.
   - Secciones `##` del reporte: identicas o con delta justificado.
   - Lineas del reporte: solo ruido conocido (timestamps, liquidez FRED).
   - CSVs historicos: solo append de la fecha del run, 0 reescritura retroactiva.
   - State files (mte, slpm, liquidity): coherentes con la maquina de estados.
5. **Si algo falla:** rollback quirurgico. No push.

---

## 5. COMMIT

Un commit por cambio verificado. Mensaje en formato:

    tipo(area): descripcion corta

    Contexto: que problema resuelve, en que hallazgo/auditoria se enmarca.

    - Cambio 1
    - Cambio 2

    Verificacion: tests, probes, run end-to-end si aplica.

    Auditoria <bloque>, hallazgos <IDs>.

Tipos: `fix`, `refactor`, `docs`, `data`, `test`, `chore`, `perf`.

---

## 6. POWERSHELL - TRAMPAS CONOCIDAS

- **Artefacto CP1252 de consola:** `Get-Content` puede mostrar `â€"` en lugar de `—`. Es artefacto de visualizacion, no del fichero. **Verificar con `read_bytes()` + `decode('utf-8')` o `repr()`.**
- **`Get-ChildItem -Include *.py`** no funciona sin `-Recurse` con `path\*`. Preferir `-Filter *.py`.
- **`git status -sb`** (un guion).
- **`@'...'@`** no expande variables. `@"..."@` si, pero hay que escapar `$` como `` `$ `` y `"` como `""`.
- **`Select-String -SimpleMatch`** desactiva regex -> el `|` se trata como literal. No usar `-SimpleMatch` con patrones que contengan `|`.
- **En here-strings PowerShell:** no hay heredoc. Para pasar contenido a Python: escribir a fichero temporal con `[System.IO.File]::WriteAllText`.
- **`New-Item -ItemType Directory -Force -Path "Consolidacion_Documentos"`:** crear directorio sin fallo si ya existe.
- **`@' ... '@` dentro de un `$py = @' ... '@`:** no anidar here-strings. Extraer el contenido a una variable previa.

---

## 7. AUDITORIA - METODO

Cuando se audita codigo linea a linea:

**Fase 0 (inventario, read-only):**
- Volcado con firmas + docstrings + imports + patrones sospechosos.
- Consumidores reales por cada funcion publica.
- Identificar dead code.
- **Gate 0 quirurgico:** para cada patron a auditar (except, datetime.now, robust_zscore, etc.), volcar contexto real con `repr()`.

**Clasificacion de hallazgos:**

| Sev | Criterio |
|---|---|
| ALTA | Bug activo en produccion, o error invisible enmascarado |
| MEDIA | Bug latente, deuda estructural, inconsistencias de contrato |
| BAJA | Cosmetico, redundancia, estilo |
| INFO | Anotacion sin accion requerida |

**Clasificacion de acciones:**

- **CERRAR:** fix con contrato verificado.
- **WONT FIX razonado:** decision documentada en el commit.
- **FALSO POSITIVO:** no era hallazgo (artefacto consola, regex del auditor mal, etc.).

**Regla critica:** antes de proponer un patch, verificar que ningun test ancla el comportamiento actual. Si un test falla por el narrowing, el test es el contrato — o se adapta al contrato nuevo, o se mantiene el comportamiento. Nunca ambos.

---

## 8. LECCIONES ACUMULADAS (A1-A5)

Estas son las reglas que hemos aprendido rompiendo cosas. Se aplican.

**Sobre el entorno PowerShell:**
- **Un here-string no es un archivo.** Si tiene >20 lineas o >5 `$`, va a `_patch_XXX.py` con `[System.IO.File]::WriteAllText`.
- **`py -c "..."` con comillas dobles anidadas rompe.** Escribir a fichero temporal.
- **Backticks en here-string se corrompen.** Placeholders + `chr(96)` o escribir a fichero.
- **No-ASCII en anchors de scripts temporales se corrompe.** `§`, tildes y similares pueden llegar mal del here-string al `_script.py` (PowerShell 5.x + UTF-8 sin BOM). Evitar no-ASCII en anchors: usar ASCII, o `chr()` explicito. Caso observado 2 veces el 2026-09-30 (anchor con seccion 5 de RUNBOOK, y `py -c` con comillas anidadas).

**Sobre Windows / consola:**
- **`Get-Content` puede mostrar UTF-8 como CP850.** Artefacto de consola, no corrupcion del fichero. 4 falsos positivos de "mojibake" en la sesion por esto. **Verificar SIEMPRE con `read_bytes().decode('utf-8')` + `repr()` antes de declarar mojibake.**
- **`read_text("utf-8-sig")` + `write_bytes(encode("utf-8"))` consume el BOM.** Para preservar BOM, escribir con `utf-8-sig`.
- **Para verificar BOM:** `p.read_bytes()[:3] == b'\xef\xbb\xbf'`.

**Sobre tests:**
- **Un test que asume un comportamiento generico (p.ej. `except Exception`) es un contrato.** Antes de acotar, verificar si el test lo ejercita con excepcion generica. Si lo hace, adaptar la tupla o mantener el generico.
- **Los tests anclados a numero de linea son fragiles.** Cualquier refactor de whitespace los rompe. Preferir busqueda por estructura.
- **En el pipeline, `RuntimeError` va SIEMPRE en la tupla de except acotados.** Los tests lo usan como idiom para "fallo simulado en provider/loader/CSV".

**Sobre refactors:**
- **Antes de unificar una constante, grep de consumidores indirectos via config, no solo literales.** Si un modulo importa `MARKET_TICKERS['sectors']`, no aparece buscando `SECTORS = [...]`. Dos fallos en A3.2 por esto.
- **Al regenerar un golden, probe de invariancia ANTES.** Si el test no es invariante al cambio, el test esta mal diseñado — se arregla el test, no se regenera a ciegas.
- **AST check bloqueante para refactors de whitespace.** `ast.dump(ast.parse(antes)) == ast.dump(ast.parse(despues))`. Sin eso, un colapso puede alterar semantica silenciosamente.

**Sobre auditoria:**
- **"El sistema prevalece sobre la documentacion."** Un comentario que dice una cosa y el codigo otra: manda el codigo.
- **Un dump de consola no es evidencia.** Para bugs, probes reproducibles. Para encoding, `read_bytes` + `repr`.
- **Un falso positivo es una leccion.** Documentarlo evita repetirlo.

**Sobre herramientas concretas (2026-09-28/29):**
- **`ast.parse` solo aplica a Python.** Nunca invocar sobre markdown. `ast.parse(markdown)` lanza SyntaxError. Para validar markdown, no hay parser: se revisa a mano o con regex.
- **`Write-Host` no soporta `-f` del mismo modo que `Write-Output`.** Para formatear strings en `Write-Host`, usar `[string]::Format(...)` o concatenacion directa. Fallo cometido 3x en la sesion de consolidacion documental.
- **Here-string de 500+ lineas se descarta silenciosamente al pegar en PowerShell.** Incluso con el patron `WriteAllText` correcto. Sintoma: no se escribe nada, `git status` no muestra el fichero. Solucion: escribir en chunks de ~25-50 lineas (seccion 10). Verificar con `Test-Path` + `(Get-Item $path).Length` tras cada chunk.

---

## 8.5. CATALOGO DE PATRONES DE BUG

Patrones recurrentes detectados en las auditorias A1-A5. Aplicables a A6, B y C. **Cada vez que se audita un modulo, buscar estos patrones antes de leer linea por linea.**

**Patron 1 - `except Exception` outer que enmascara bug interno.**
El bloque entero envuelto en `except Exception` con mensaje generico. Un bug interno (`NameError`, `AttributeError`, `IndexError`) queda silenciado como "modulo omitido". **Caso A5-07:** `leaders.py`, `NameError` en rama INSUFFICIENT_COVERAGE oculto como "Modulo de lideres omitido". **Como detectar:** indentacion visible sospechosa; variables usadas fuera del scope donde se definen.

**Patron 2 - Check muerto (siempre True / siempre False).**
`if 'x' in dir()` (siempre True para parametros). `if x is not None` (si x es Series vacia, nunca None). `if len(x) > 0` tras filtrar por `.empty`. **Caso A5-08:** `'all_signals' in dir()` siempre True.

**Patron 3 - tz-naive vs tz-aware en comparaciones.**
Restar `datetime.now()` (naive) contra timestamp tz-aware. Rinde `TypeError`, a menudo silenciado por `except`. **Caso A2.2-02/03:** `european_coverage` y `data_quality`.

**Patron 4 - Aridad variable de retorno entre ramas.**
`return a, b, c` en early returns; `return a, b, c, d` en exito. El caller sortea por coincidencia (`result[0] is not None`). **Caso A3.1-03:** `compute_liquidity_score`.

**Patron 5 - Bucle sin guard.**
`while not is_market_day(d): d -= timedelta(days=1)`. Sin limite. Si `NYSE_HOLIDAYS` corrupto o rango no cubierto → bucle infinito. **Caso A1-11.**

**Patron 6 - Variante local de funcion canonica.**
`robust_zscore` definido dos veces con contratos distintos. `tanh` local. `_robust_z` con ffill. **Casos A3.4-03/04:** `options.robust_zscore`, `fls.manual_robust_zscore`. **Como detectar:** grep por `def robust_zscore|def _robust|def tanh`.

**Patron 7 - Test anclado a numero de linea.**
`for node in ast.walk(tree): if node.lineno == 89: ...`. Rompe con cualquier refactor de whitespace. **Caso A3.3-13.**

**Patron 8 - Dead code por refactor.**
`def f(): return wrapper()` sin consumidor. Alias conservados por "compatibilidad" sin importadores. Constantes sin uso. **Casos:** `compute_liquidity_score` alias, `_n_close` fuera de scope, `robust_zscore_series` en MTE.

**Patron 9 - Fallback silencioso en provider.**
`except Exception: return self._load_cache()`. Devuelve datos sin marcar staleness. **Caso A5-13:** Yahoo devolvia parquet completo.

**Patron 10 - Documento vs codigo.**
Docstring dice "9 contratos" pero hay 10. Comentario "esto hace X" pero el codigo hace Y. **Caso A3.2-01:** 7 literales `SECTORS = [...]` vs `MARKET_TICKERS['sectors']`.

**Patron 11 - `mtime` como criterio de freshness.**
`if datetime.now() - mtime < timedelta(hours=23): return cache`.
Confunde "cache reciente en disco" con "fuente sin dato nuevo". Si la
fuente publica despues del run, el siguiente run acepta cache de 22h
como fresco. **Caso Fix K (2026-10-01):** 13 Euronext + 19 BME con
cache de 29-sep mientras las fuentes tenian 30-sep. **Caso Fix N
(2026-10-01):** 5 writers de fund flow (ssga, amundi, blackrock_base,
blackrock_iwm, cftc) con el mismo patron. **Como detectar:** grep
`datetime.now() - mtime|timedelta(hours=23)|age <= timedelta`.
**Fix correcto:** intentar descarga SIEMPRE. Cache solo como fallback
si la descarga falla (WARN explicito con mtime). Coste medido: 13s
descargar todos los providers vs 12min del run.

**Patron 12 - Cache-hit valida forma, no calidad.**
`if _df_last >= _last_exp: return cache`. Comprueba la fecha pero no
la cobertura/estado del dato. Un parquet con la fecha correcta y 89.8%
de cobertura se acepta como valido. **Caso Fix M (2026-10-01):**
`stock_data_loader` y `data_loader` con `CACHE_VALIDATE_TRADING_DATE`.
El cascade europeo no se ejecutaba porque la fecha era correcta pero
faltaban 32 tickers. **Como detectar:** grep `_df_last < _last_exp|
CACHE_VALIDATE_TRADING_DATE`. **Fix correcto:** validar cobertura de
la ultima fila ademas de la fecha. Fail-closed si no se puede medir.

**Patron 13 - Consumidor accede por indice sin guard.**
`ranking[0][0]` sobre lista que puede quedar vacia. `df.iloc[-1]` sin
comprobar `len(df) > 0`. **Caso `engines.py:25`** (auditoria
sector_regime): `sector_results['ranking'][0][0]` sin guard. Bug
latente: si `ranking` queda vacio, IndexError. **Caso `qqq_returns`
(pre-Fix H):** `prices.index[-1]` sin comprobar la sesion esperada.
**Como detectar:** grep `\[0\]\[0\]|\.iloc\[-1\]|\.index\[-1\]` y
comprobar si el consumidor asume no-vacio. **Fix correcto:** guard
explicito o derivar por `resolve_effective_date` (R2).
**Como aplicar:** al abrir un modulo nuevo, primer paso: `grep` de estos 10 patrones con regex. Los hits se convierten en hallazgos preliminares. Se verifican uno a uno. No se declaran sin evidencia directa.

---

## 8.6. LISTA NEGRA DE FALSOS POSITIVOS

Cosas que **parecen** un hallazgo pero no lo son. Antes de declarar, aplicar el check.

| Falso positivo | Check obligatorio |
|---|---|
| "Mojibake en el fichero" | `p.read_bytes().decode('utf-8')` + `repr()`. Si los codepoints son UTF-8 validos, no es mojibake; es artefacto CP850 de la consola. |
| "Variable no definida" en un except | ¿El `except` captura el error? Puede ser red de seguridad deliberada. Ver contexto completo. |
| "Duplicacion de constante" | Grep de **consumidores indirectos via config**. Si un modulo importa `MARKET_TICKERS['sectors']` y otro tiene literal `SECTORS = [...]`, **no es duplicacion**: es una divergencia real que hay que unificar. |
| "Test fragil porque anclaba linea X" | Ver si el test tiene estructura alternativa. A veces el numero de linea es porque la funcion es unica en el fichero. |
| "except Exception sin acotar" | Puede ser contrato defensivo (mte/engine.py, flows_secondary:116). Comprobar si hay tests que dependen de la genericidad. |
| "Dead code porque no aparece el nombre en grep" | Buscar por patron alternativo. `compute_liquidity_score` en `financial_conditions.py` era alias; el consumidor importaba el de `liquidity.py`. |
| "Commit sin autor conocido" | `git log --format='%an <%ae>'`. Si es de github-actions, es bot. |
| "CSV modificado sin razon" | Verificar si es append (fecha nueva) o reescritura (mismas fechas, valores cambiados). Los runs legitimos anaden filas. |
| "Test que pasa pero no verifica nada" | Buscar `assert` con valor constante, `pass`, `# no lanza`, `skip` mal configurado. Verificar con cobertura. |


**Regla anti-falso-positivo del test (D38, 2026-09-30).** Antes de aceptar un verde, comprobar que el test falla con el fix revertido. Si pasa con y sin fix, el test no mide el cambio. Evidencia empirica: `test_load_radar_index_descarta_vacios_y_nan` (rojo sin fix -> verde con fix, destapo bug latente de simetria en `_load_radar_index`); `test_calculate_returns_ytd_nan_sin_prev_year` (mal disenado, rojo por error propio, corregido antes de commitear). Los casos frontera son obligatorios: toda frontera (lag, threshold, borde de ventana, primer/ultimo valor) necesita un caso que la cruce y otro que no.

**Regla dura:** antes de proponer un patch por un "hallazgo", verificar que no cae en uno de estos casos. Un falso positivo documentado vale mas que un patch a ciegas.

---

## 9. FRASES GUIA

"Determinista, descriptivo, auditado. Paso a paso. Documentar. Saber parar."

Complementos:
- "Ver antes del patch."
- "Un cambio = una verificacion = un commit."
- "Local-first: push solo cuando el sistema este verificado solido."
- "Si el ROI < 1, cerrar."
- "Ver antes de concluir: si no hay evidencia directa, es hipotesis, no hecho."
- "Auditor externo antes de decisiones irreversibles."
- "Gate 0 antes de tocar datos."
- "Doble candado: is_market_day==False AND date in CONFIRMED_SET."
- "Si una cifra del prompt no coincide con la realidad medida, se corrige el prompt, no la realidad."
- "Un fix destapa el siguiente."
- "Verificar en produccion real (CI), no solo en tests locales."
- "No confundir VALID con UNAVAILABLE."
- "Un here-string no es un archivo: si tiene >20 lineas o 5 $, va a _patch_XXX.py."
- "Antes de autorizar un patch sobre Pandas, verificar la semantica exacta del objeto."
- "Antes de anadir una capa de descarga, verificar si ya existe."
- "El sistema prevalece sobre la documentacion. Si el codigo y el doc discrepan, gana el codigo."

---

## 10. SCRIPTS TEMPORALES

Los patches, probes y checks se hacen con ficheros temporales en `%TEMP%`. Patron:

    $py = @'
    # contenido del script Python
    '@
    $tmp = Join-Path $env:TEMP "_nombre_XXX.py"
    [System.IO.File]::WriteAllText($tmp, $py, [System.Text.UTF8Encoding]::new($false))
    $salida = py $tmp 2>&1
    $exitcode = $LASTEXITCODE
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    Write-Host "exit=$exitcode"
    $salida | ForEach-Object { Write-Host $_ }

Regla: **ver la salida del patch ANTES de la verificacion.** Un patch multi-anchor que aborta silenciosamente en PowerShell es comun. Sin ver el `exit=` y el `[OK]`/`[ABORT]`, la verificacion posterior mide un estado que no es el esperado.

Ademas:
- **Si el `$py` here-string tiene >50 lineas, escribirlo en chunks.** Un here-string grande puede descartarse silenciosamente al pegar. Verificar con `Test-Path $tmp` + `(Get-Item $tmp).Length` tras escribir.
- **Verificar la salida del patch inmediatamente.** Antes de cualquier otra accion. Si no aparece `[OK]`, el fichero no se toco.

---

## 11. CORPUS DOCUMENTAL

El corpus consolidado vive en `Consolidacion_Documentos/` (6 ficheros + snapshot auto-generado). Reglas de actualizacion:

**`00_ARRANQUE.md`:** se actualiza cuando cambia el estado del sistema (HEAD, tests, pendientes) o las reglas duras. Es el unico documento que se pega al arrancar — su tamano importa (<=10 KB).

**`01_METODO.md` (este documento):** se actualiza cuando se aprende una leccion nueva. Crece lentamente. Si supera 20 KB, dividir.

**`02_ARQUITECTURA.md`:** se actualiza cuando cambia la estructura de directorios, los contratos de modulo, las decisiones arquitectonicas vigentes, o el pipeline. Es referencia on-demand. Si supera 40 KB, revisar.

**`03_IAE.md`:** se actualiza cuando cambia el subsistema IAE. Independiente del resto.

**`04_HISTORICO.md`:** se actualiza **por acumulacion**. Cada cierre de sesion anade una entrada tematica. No se poda.

**`05_BITACORA.md`:** se actualiza al **cierre de cada sesion**. Formato definido en el propio fichero. Mantener solo las ultimas **5 sesiones**. La 6ª se elimina (o se resume en `04_HISTORICO.md`).

**`ESTADO_SISTEMA.md`:** NO se edita a mano. Se regenera con `py scripts/generate_estado_sistema.py`.

**Reglas duras del corpus:**
- Nada duplica lo que otro documento ya cubre. Si un concepto aparece en dos sitios, uno de los dos sobra.
- Los documentos del corpus NO se citan a ficheros borrados. Los unicos que pueden mencionar el corpus antiguo son `04_HISTORICO.md` (cronologia) y `05_BITACORA.md` (referencia historica).
- Cuando se cierra una sesion: actualizar `05_BITACORA.md` (anadir entrada) + `04_HISTORICO.md` (si hubo decision estructural). El resto solo cambia si el contenido lo exige.
- **Toda afirmacion sin evidencia directa se marca como hipotesis.** No "el sistema hace X" si no se ha verificado.
- **El sistema prevalece sobre la documentacion.** Si el codigo y el documento discrepan, gana el codigo y se corrige el documento.