# 01d - POWERSHELL Y AUDITORIA

**Referencia on-demand. Se consulta antes de tocar PowerShell o al**
**auditar un modulo. No se pega al arrancar. El arranque es**
**`00_ARRANQUE.md`.**

*Extraido de `01_METODO.md` el 2026-11-01 (division por tamano:
METODO superaba 20 KB). Secciones 6, 6bis, 6ter y 7.*

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
- **`$var:` en strings PS es ambiguo.** `Write-Host "x $ln: y"`
  produce `La referencia de variable no es valida`. El parser
  interpreta `$ln:` como drive-variable. Fix: `Write-Host ("x {0}: y" -f $ln)`
  o `${ln}:`.
- **`<STAMP>` en bloques de comandos.** No dejar placeholders
  angulares en bloques que el usuario pegara. Cuando un bloque depende
  de un valor de salida del bloque anterior, escribir el valor literal
  o encadenar por variable en el mismo paste.
- **`<...>` en bloques de comando:** PowerShell interpreta `<` como
  redireccion reservada. `gh run view <RUN_ID>` falla con error de
  parser. Usar variable previa: `$rid = "..."`; `gh run view $rid`.
  Lo mismo con `<STAMP>`, `<DATE>`, etc.
- **`-ErrorAction` no existe en `git`:** es parametro de cmdlet PS, no
  de binario. `git checkout -- path -ErrorAction SilentlyContinue`
  falla como "unknown option". Usar `2>$null` o confiar en que
  `git status` muestra el estado real tras el intento.
- **`--jq` en `gh` con PowerShell se come las comillas:** el parser
  PS interpreta mal los `"..."` dentro del `--jq '...'`. Usar
  `ConvertFrom-Json` en PS: `$runs = gh run list --json ... | ConvertFrom-Json`.
  Lo mismo con `>` en PS: es redireccion, no argumento. Filtrar en PS,
  no en la linea de comandos.
- **Mojibake por here-string:** el paso de here-string PS 5.1 a
  `_script.py` corrompe caracteres unicode (dibujo de caja U+251C,
  tildes). Distinto del artefacto de consola (L159): aqui el fichero
  escrito queda corrupto. Verificar tras escribir: `'\u251c' in texto`.
  No fiarse de la vista de consola.
- **SHA puramente numerico en YAML se parsea como int.** `commit:
  71371872` sin comillas -> `verifica_gobernanza` falla con "SHA
  entrecomillado en YAML: ... (int)". Los SHAs con letras (p.ej.
  `090a3439`) no fallan porque no son parseables como int. Regla: en
  REGISTRO, todo `commit:` se entrecomilla SIEMPRE (`commit:
  '71371872'`). Aplicado 2026-10-30 en commit cierre-19.
- **YAML con `:` en `descripcion` rompe `verifica_gobernanza`:** un
  `:` inline sin comillas altera el parseo. Usar block scalar `|` con
  sangria 4 espacios, o citar con `'...'`. Evitar `:` sin escapar en
  valores inline de `REGISTRO.yaml`.
- **`Where-Object` desenvuelve arrays de 1 elemento a escalar.** Si
  `$arr = (cmd) | Where-Object { $_ }` devuelve un solo elemento,
  `$arr` es string, no array. `$arr[0]` devuelve el primer caracter,
  no el elemento. `$arr.Count` da longitud de la string.
  Fix: forzar `@(...)` -> `$arr = @(cmd | Where-Object { $_ })`.
  Uso tipico: guard de staging que comprueba `$staged.Count -ne 1`
  y `$staged[0] -match ...` falla con un solo fichero sin el
  `@(...)`. Tropiezo confirmado 2026-10-22.

---

- **`cancelled` por concurrency NO es `failure`.** GitHub cancela
  runs con `cancel-in-progress` cuando entra otro push al mismo
  branch. El icono X en `gh run list` es ambiguo. Regla: `gh run
  view <rid> --json conclusion` antes de declarar rojo. Y verificar
  cobertura por ancestro: `git merge-base --is-ancestor <cancelado>
  <sucesor_success>`. Pisado 4 veces (b9d7aee7, e3400a88, 5b96192f,
  e8a2a4ba). Leccion sesion 21.

### 6bis. TRAMPAS ADICIONALES (2026-10-08)

- **`git show <sha>:<path> > file` en PowerShell escribe UTF-16 con
  BOM** (`0xFF 0xFE`). `pd.read_csv` falla con `UnicodeDecodeError`.
  Fix: `[IO.File]::WriteAllLines` con `UTF8Encoding($false)`.
- **Copiar un CSV *después* de un E2E contamina la comparación
  pre/post.** Si haces snapshot "pre", corres el E2E, y luego copias
  el fichero como "frozen": el frozen ya tiene los cambios del E2E.
  Caso real 2026-10-08 con `sector_wyckoff_distribution.csv`.
- **El corpus tiene EOL mixto en disco.** Verificado 2026-10-22:
  `01_METODO.md` (352 CRLF) y `01c_LECCIONES.md` (61 CRLF) en CRLF;
  `00_ARRANQUE.md` (257 LF) y `05_BITACORA.md` (974 LF) en LF. El EOL
  puede cambiar entre sesiones por edicion manual o checkout con
  `autocrlf`; re-verificar antes de cada patch. `.gitattributes` fuerza
  LF al commitear pero el working tree conserva el EOL original.
  Detectar con `ReadAllText` + `[regex]::Matches` (no `Get-Content`,
  que puede normalizar).
- **`foreach (...) { ... } | pipe` falla en PowerShell 5.1** con
  `No se permiten elementos de canalización vacíos`. Usar
  `... | ForEach-Object { ... }` o envolver en `$(...)`.
- **`git cherry main <rama>` antes de borrar una rama.** Los `+` son
  commits unicos. Los `-` son duplicados por rebase. Verificar los
  `A` (added) del `git diff --name-status` antes de borrar.

### 6ter. LIMITES DE LOS GATES AUTOMATICOS (2026-10-22)

Los verificadores (`verifica_gobernanza`, `poda_bitacora`, `pyflakes`,
`compileall`) comprueban **estructura**, no **semantica**. `[OK]` en
cada uno significa lo que el script declara que comprueba; no implica
coherencia con la intencion del autor. Lecciones observadas 2026-10-22:

- **Un `[OK]` no valida orden ni contenido semantico.** `poda_bitacora
  --check` cuenta fechas unicas, no verifica orden de entradas.
  `verifica_gobernanza` valida que un item FIXED no aparezca en
  `HANDOFF §4`, pero **no** audita §3 (menciones desactualizadas,
  lineas perdidas). Asumir "verde" como "correcto" es el error.
- **No predecir conteos exactos de lineas.** `ReadAllLines` cuenta
  trailing newline distinto al conteo de `\n` o `\r\n`. Un `throw`
  con un `esperado 329` (cuando son 330) es ruido que oculta errores
  reales. Verificar solo "igual" o "mayor", nunca cifra exacta.
- **Verificar al verificador.** Un check que devuelve "orden
  descendente OK (1 headers)" cuando hay 11 headers no verifica nada.
  Antes de confiar en `[OK]`, confirmar que el check se ejecuto sobre
  el numero esperado de elementos. `-AllMatches` falta en
  `Select-String ... .Matches.Groups`, `Where-Object` desenvuelve
  escalar, etc.
- **`.Replace` no verifica el resultado.** `Count-Occ == 1` prueba
  que el anchor existe, no que el replacement contiene lo que debe.
  Post-escritura obligatoria: `if ($text.IndexOf($marker) -lt 0)
  { throw }` con `$marker` unico del nuevo contenido. Bug 2026-10-22:
  `.Replace` con here-string vacio borro dos bloques `---` del fichero.
- **Sintaxis PowerShell rota = no ejecuta nada, pero no se detecta
  hasta pegar.** Parentesis anidados en expresiones `[regex]::...`
  son propensos. Comprobar conteo antes de enviar el bloque.
  Tropiezo confirmado 2026-10-22.
- **`git checkout HEAD -- file` puede cambiar el EOL del working
  tree** si `.gitattributes` declara `eol=lf`. No asumir que restore
  preserva el EOL pre-restore. Re-detectar EOL despues de restore.
- **Bucle de busqueda con break en la propia condicion de inicio.**
  `for ($j = $idxItem; ...) { if ($lines[$j] -match '^\- id:') break }`
  dispara el break en la linea de origen ($lines[$idxItem]) porque
  tambien matchea el patron. El loop nunca avanza, el resultado queda
  en -1. Fix: empezar en $idxItem + 1 o excluir explicitamente la
  linea de origen del break. Bug 2026-10-22: busqueda de
  `siguiente_accion:` dentro de un item REGISTRO fallo por esto.
- **Prioridad de fuentes sin comparación de frescura.** Cuando un
  consumidor tiene múltiples fuentes (API externa con lag no
  controlado vs fichero local con SLA conocido), priorizar ciegamente
  la API es bug de diseño. El consumidor calcula frescura contra
  datos viejos y produce N/A, mientras otro módulo que lee la fuente
  fresca publica valores válidos. Contradicción visible en el mismo
  reporte. Fix: priorizar la fuente con SLA conocido (CSVs
  versionados, pusheados por bot diario) y usar la API como fallback.
  Evidencia 2026-10-22: `router.get_fed_data` priorizaba FRED API
  (lag 5d) sobre CSVs diarios → `liquidity.py` N/A + `fls.py` OK.
  Item: REP-AUDIT-20261022-LIQUIDITY-ROUTER-PRIORITY.

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

