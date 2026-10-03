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
    py -m pyflakes .
    py -m pytest tests/ validation/ -q --tb=short

Esperado:
- compileall OK
- pyflakes LIMPIO (silencio total)
- 3033 passed + 2 skipped + 0 failed (2026-10-01)

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
- **`@(...)` con `[` o `(` sin pareja:** rompe el parser de PS. Usar `list()` / `dict()`, o escribir con `[IO.File]::WriteAllText` desde un array externo.
- **`[IO.File]::ReadAllBytes` con ruta relativa:** resuelve el CWD de .NET, no el de PS. Usar `Resolve-Path` siempre.
- **`Select-String | Out-Null; if ($?) { throw }`:** falso positivo. Usar `if (Select-String ...) { throw }`.
- **`git commit -F` con `Set-Content -Encoding UTF8`:** mete BOM en PS 5.1. Usar `[IO.File]::WriteAllText` + `UTF8Encoding($false)`.
- **`path.read_text("utf-8")` no elimina BOM:** usar `"utf-8-sig"` cuando se parsea con `ast`.

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

**Reglas adicionales (2026-10-03):**

- **Alineacion protocolo<->script.** Un script con hash congelado puede no implementar el protocolo que dice congelar. Verificar la alineacion antes de escribir tests contractuales.
- **Ficheros con autorreferencia.** Un fichero versionado no puede contener campos derivados del commit que lo contiene (bucle). Aplicado a `ESTADO_SISTEMA.md` y manifiestos de freeze.
- **Tests centinela.** Algunos tests existen para romper si un artefacto cambia (T7/T7b en `test_wyckoff_sow_5bX_contract.py`). Modificar el artefacto produce test rojo, no silencio. No "arreglar" el test sin revisar el contrato.
- **Auditoria mecanica complementa manual.** Un barrido AST detecto 2 docstrings desalineados que se escaparon en auditoria manual (breadth_metrics, iae_section). Convertidos en tests contractuales (`tests/test_audit_contracts.py`). Ninguna de las dos verificaciones basta por si sola.
- **Clasificar tras verificar, no antes.** Clasifique F-01/F-02 de `validation_gate.py` como MEDIA antes de leer tests; tras contraste bajaron a BAJA/INFO. Mismo patron con el parser de docstrings: clasifico 5 falsos positivos antes de verificar su tokenizador. Primero la evidencia, luego la severidad.
- **Regla de estilo: `EXPECTED_SECTOR_COUNT`.** Todo conteo o iteracion sobre los 11 sectores usa `EXPECTED_SECTOR_COUNT` de `config.settings`. Barrido 2026-10-03: 9 ficheros, 27 sitios corregidos. Test mecanico en `tests/test_audit_contracts.py`.
- **Regla de estilo: docstring `Returns: dict con keys:` alineado.** El bloque debe coincidir con las keys del return real. 5 ficheros corregidos en S-03-deep. Test mecanico en `tests/test_audit_contracts.py`.


---

## 8. LECCIONES ACUMULADAS (A1-A5)

**Movido a `01c_LECCIONES.md`** (2026-10-03, division por tamano).

Las lecciones acumuladas de entorno, tests, refactors y auditoria
viven en `Consolidacion_Documentos/01c_LECCIONES.md`. Consulta
obligatoria antes de tocar PowerShell, tests o refactors.

## 8.5. CATALOGO DE PATRONES DE BUG

**Movido a `01b_PATRONES.md`** (2026-10-02, division por tamano).

El catalogo completo de 13 patrones de bug recurrentes, la lista
negra de falsos positivos y la regla anti-falso-positivo del test
viven en `Consolidacion_Documentos/01b_PATRONES.md`.

Consulta obligatoria al auditar un modulo nuevo.

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
- **Verificar unicidad del ancla antes de escribir el script.** Si el anchor del `.Replace()` aparece N>1 veces, el script aborta antes de escribir. Caso observado 2026-10-03 (H5.3): `- H5.3: cron 13F 20-nov-2026.` aparecia 2 veces. Sin dano. La unicidad se verifica al disenar el script, no solo al ejecutarlo.


---

## 11. CORPUS DOCUMENTAL

El corpus consolidado vive en `Consolidacion_Documentos/`. Indice vigente en `00_ARRANQUE.md` seccion 6 (+ snapshot auto-generado). Reglas de actualizacion:

**`00_ARRANQUE.md`:** se actualiza cuando cambia el estado del sistema (tests, integridad, pendientes) o las reglas duras. El HEAD se consulta con `git log`; el arranque no lo declara. Es el unico documento que se pega al arrancar — su tamano importa (<=10 KB = 10240 B).

**`01_METODO.md` (este documento):** se actualiza cuando se aprende una leccion nueva. Crece lentamente. Si supera 20 KB, dividir.

**`01c_LECCIONES.md`:** lecciones acumuladas (entorno, tests, refactors, auditoria). Se amplia cuando se aprende una leccion nueva. (2026-10-03, division por tamano de `01_METODO.md`.)

**`02_ARQUITECTURA.md`:** se actualiza cuando cambia la estructura de directorios, los contratos de modulo, las decisiones arquitectonicas vigentes, o el pipeline. Es referencia on-demand. Si supera 40 KB, revisar.

**`03_IAE.md`:** se actualiza cuando cambia el subsistema IAE. Independiente del resto.

**`04_HISTORICO.md`:** se actualiza **por acumulacion**. Cada cierre de sesion anade una entrada tematica. No se poda.

**`05_BITACORA.md`:** se actualiza al **cierre de cada sesion**. Formato definido en el propio fichero. Mantener las ultimas **10 sesiones** mas todas las del dia en curso. La que exceda se elimina (o se resume en `04_HISTORICO.md`).

**`ESTADO_SISTEMA.md`:** NO se edita a mano. Se regenera con `py scripts/generate_estado_sistema.py`.

**Reglas duras del corpus:**
- Nada duplica lo que otro documento ya cubre. Si un concepto aparece en dos sitios, uno de los dos sobra.
- Los documentos del corpus NO se citan a ficheros borrados. Los unicos que pueden mencionar el corpus antiguo son `04_HISTORICO.md` (cronologia) y `05_BITACORA.md` (referencia historica).
- Cuando se cierra una sesion: actualizar `05_BITACORA.md` (anadir entrada) + `04_HISTORICO.md` (si hubo decision estructural). El resto solo cambia si el contenido lo exige.
- **Toda afirmacion sin evidencia directa se marca como hipotesis.** No "el sistema hace X" si no se ha verificado.
- **El sistema prevalece sobre la documentacion.** Si el codigo y el documento discrepan, gana el codigo y se corrige el documento.
