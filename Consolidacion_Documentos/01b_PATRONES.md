# 01b - PATRONES DE BUG Y FALSOS POSITIVOS

**Referencia on-demand. Se consulta al auditar modulos.**
**No se pega al arrancar. Hijo de `01_METODO.md` §8.**

Este documento se extrajo de `01_METODO.md` el 2026-10-02 para
mantener ambos por debajo del umbral autoimpuesto de 20 KB.

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
**Como aplicar:** al abrir un modulo nuevo, primer paso: `grep` de estos 13 patrones con regex. Los hits se convierten en hallazgos preliminares. Se verifican uno a uno. No se declaran sin evidencia directa.

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
