# 01c - LECCIONES ACUMULADAS

**Referencia on-demand. Se consulta antes de tocar PowerShell, tests, refactors o auditoria.**
**No se pega al arrancar. El arranque es 00_ARRANQUE.md.**

Movido desde `01_METODO.md` seccion 8 el 2026-10-03 (division por
tamano; `01_METODO.md` estaba a 64 B del limite autoimpuesto de
20 KB).

---

## 8. LECCIONES ACUMULADAS (A1-A5)

Estas son las reglas que hemos aprendido rompiendo cosas. Se aplican.

**Sobre el entorno PowerShell:**
- **Un here-string no es un archivo.** Si tiene >20 lineas o >5 `$`, va a `_patch_XXX.py` con `[System.IO.File]::WriteAllText`.
- **`py -c "..."` con comillas dobles anidadas rompe.** Escribir a fichero temporal.
- **Backticks en here-string se corrompen.** Placeholders + `chr(96)` o escribir a fichero.
- **No-ASCII en anchors de scripts temporales se corrompe.** `§`, tildes y similares pueden llegar mal del here-string al `_script.py` (PowerShell 5.x + UTF-8 sin BOM). Evitar no-ASCII en anchors: usar ASCII, o `chr()` explicito. Caso observado 2 veces el 2026-09-30 (anchor con seccion 5 de RUNBOOK, y `py -c` con comillas anidadas).

- **Un patch multilinea pegado en consola puede perderse sin error.**
  Caso 2026-10-03: el patch de MR-8 (5 lineas) desaparecio entre
  pegar y ejecutar. `git commit` corrio con working tree limpio y el
  fichero quedo como estaba. Sintoma: silencio, `exit=0` de la nada,
  `git log` sin el commit esperado. Regla: `git status` justo ANTES
  de `git commit`, no despues. Si el fichero que esperabas modificar
  no aparece, abortar el commit y reaplicar el patch.
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

- **Confirmar ruta en disco antes de heredar nombres de un handoff.**
  Caso 2026-10-03: el handoff citaba `src/regimes/macro_regime.py`;
  el modulo real esta en `regimes/macro_regime.py` (raiz del repo).
  Regla: `Get-Item` sobre la ruta antes de disenar el plan; si no
  existe, `Get-ChildItem -Recurse -Directory` para localizar el
  paquete. Aplicable a cualquier nombre propio (modulo, fichero,
  clase) heredado de una sesion previa.
**Sobre herramientas concretas (2026-09-28/29):**
- **`ast.parse` solo aplica a Python.** Nunca invocar sobre markdown. `ast.parse(markdown)` lanza SyntaxError. Para validar markdown, no hay parser: se revisa a mano o con regex.
- **`Write-Host` no soporta `-f` del mismo modo que `Write-Output`.** Para formatear strings en `Write-Host`, usar `[string]::Format(...)` o concatenacion directa. Fallo cometido 3x en la sesion de consolidacion documental.
- **Here-string de 500+ lineas se descarta silenciosamente al pegar en PowerShell.** Incluso con el patron `WriteAllText` correcto. Sintoma: no se escribe nada, `git status` no muestra el fichero. Solucion: escribir en chunks de ~25-50 lineas (seccion 10). Verificar con `Test-Path` + `(Get-Item $path).Length` tras cada chunk.

---
