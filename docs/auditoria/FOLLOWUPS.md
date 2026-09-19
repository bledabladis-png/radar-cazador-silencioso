# Follow-ups tecnicos registrados

## FU-001 — ffill global sobre DataFrame consolidado (RESUELTO)

- **Origen:** C1, sesion 2026-09-12. Dictamen auditor.
- **Descripcion:** `stock_data_loader.py:352` aplica `ffill(limit=3)` sobre el DataFrame consolidado (Yahoo USA + Euronext + Xetra + BME). Ese ffill puede reintroducir valores imputados en tickers fuera del calendario NYSE, pese a haber sido clasificados como DATA_ISSUE en su etapa.
- **Evidencia:** tras el run C1 (2026-09-12), 20 tickers .L marcados como `DATA_ISSUE / MISSING_CLOSE_EXPECTED_SESSION` vuelven a mostrar `close[-1]==close[-2]` en el parquet final por el ffill de L352.
- **Impacto:** no afecta al pipeline USA actual ni invalida C1. Afecta a la coherencia de la marca DATA_ISSUE para tickers no-USA.
- **Clasificacion:** P2/P3 integridad tecnica localizada (auditor).
- **Accion:** disenar capa de proteccion por calendario (NYSE/BME/Xetra/Euronext) antes de generalizar la arquitectura a Europa.
- **Bloqueante:** no para C1/C2/C3. Obligatorio antes de declarar cerrada la nueva arquitectura de validacion multi-calendario.

**Resolucion (2026-09-15):**
- **Diagnostico confirmado:** `src/stock_data_loader.py:480` aplicaba `data.ffill(limit=3)` sobre el DataFrame consolidado, rellenando los NaN que B1 habia preservado correctamente. Esto producia `pct_dup_last=0.837` en runs pre-PUBLISH_HOUR.
- **Fix aplicado:** commit `38f9ce1` elimina la operacion. Los NaN preservados por B1 llegan al parquet final.
- **Cron ajustado:** commit `8a76380` mueve `daily_run.yml` de 20:00 UTC (22:00 Madrid) a 06:00 UTC (08:00 Madrid verano / 07:00 invierno), dando tiempo a Yahoo a publicar los cierres.
- **Verificacion en produccion:** run automatico del 2026-09-15 06:00 UTC. `close_nan=0` en 24/27 lotes, `pct_dup=0.026`, manifest `status=VALID`, cobertura 313/313, Gate 10/10.
- **Tests de regresion:** 3 en `tests/test_fu001_sin_ffill_global.py`.
- **Estado:** **RESUELTO 2026-09-15**.

## FU-002 — Validacion circular en BackupProvider (RESUELTO)

- **Origen:** informe v2, hallazgo B4. Dictamen auditor.
- **Descripcion original:** `_validate_with_cache` compara el nuevo dato contra los mismos parquets que luego sobreescribe. Si el cache esta contaminado, la validacion pasa.
- **Reformulacion (2026-09-15):** la circularidad es **temporalmente diferida**, no activa. BackupProvider instancia el cache al inicio del pipeline (fase 1) y lee los parquets del run N-1. Si N-1 estaba corrupto, N valida contra corrupcion.
- **Evidencia:** T7 del informe v2: 41/112 tickers contaminados en reference_cache por B1.
- **Fix aplicado (2 commits):**
  - `45f29c2` feat: writer genera `<parquet>.manifest.json` atomico con sha256, expected_session, pct_dup_last, quality.status y run_id. Modifica `src/utils.py` (nuevo helper `write_artifact_with_manifest`), `src/data_loader.py`, `src/stock_data_loader.py`, `src/pipeline/data_load.py`, `src/pipeline/leaders.py`, `run.py`, `config/settings.py` (`MANIFEST_DUP_THRESHOLD=0.5`).
  - `2da234f` fix: consumer rechaza cache no verificado. `_load_reference_cache` verifica manifest (sha256 + schema_version + quality.status) y devuelve `(df, status)` con status en `{VALID, INVALID, UNAVAILABLE}`. `_validate_with_cache` cambia a `True/False/None`. Los 5 `return True` silenciosos eliminados.
- **Tests nuevos:** 10 (`test_artifact_manifest.py` + `test_backup_provider_reference.py`). Suite pasa de 202 a 212.
- **Clasificacion:** P2 Technical Debt estructural - **RESUELTO 2026-09-15**.
- **Bloqueante:** no.

## FU-003 — Cosmetico: signos +0.00 y flechas -> (OBSOLETO 2026-09-17)

- **Origen:** prompt maestro v6.8 seccion 15.2.
- **Descripcion:** formato de +0.00 en scores pequenos; flecha -> en algunos textos.
- **Clasificacion:** P3 cosmetico.
- **Bloqueante:** no.

**Resolucion (2026-09-17):**
- **Gate 0 (parte +0.00):** `outputs/report/reporte_diario.md` (reporte vivo) tiene **0 matches** de `+0.00`. Los 43 matches en `outputs/audit/*` son snapshots historicos no versionados. FU-003b (`slpm.py:50`) ya corrigio los casos donde `+0.00` era enganoso (n=0 -> N/D). El resto de `{:+.2f}` es convencion intencional (signo explicito).
- **Gate 0 (parte flechas ->):** 5 matches en reporte vivo, todos semanticos (`UNRESOLVED -> Transition` FSM, `Persistence -> Structural Score -> SLPM`, etc.). Sustituir por `->` unicode (`→`) romperia grep/ASCII sin beneficio funcional. `slpm.py:37` (` -> {quadrant}`) no aparece en el reporte vivo (quadrant probablemente vacio).
- **Estado:** **OBSOLETO 2026-09-17**. FU-003b ya resolvio lo relevante; el resto es convencion/diseño.

## FU-004 — Origen del colapso `macro_regime.csv` (315 -> 1) sin identificar

- **Origen:** sesion 2026-09-12, verificacion E2E-pre.
- **Descripcion:** `outputs/history/macro_regime.csv` paso de 314 filas (HEAD `829f9c6`) a 1 fila (working tree). El codigo actual de `save_regime_history` (C4-code) es coherente con el resultado final, pero no explica el colapso inicial.
- **Evidencia:**
  - Snapshot `outputs/audit/c4_code_verif/macro_regime_pre.csv` (06:40:45) ya mostraba 1 fila antes del run.
  - HEAD `829f9c6` tenia 314 filas con strings de fecha distintos (`2026-06-19 16:38:26.367013`, etc.), todas B2-contaminadas.
  - `save_regime_history` con `dtype=str` + `drop_duplicates(subset=['date'])` no habria colapsado 314 strings distintos.
- **Impacto:** ninguno funcional. El estado actual (1 fila, `2026-09-11`, formato `YYYY-MM-DD`) es el deseado.
- **Hipotesis:** algun paso de la bateria de verificacion C4-code (pre-06:40) escribio al path real en lugar de tmp, o hubo `git checkout` selectivo. No confirmado.
- **Clasificacion:** P3 documental.
- **Accion:** ninguna urgente. Si reaparece un colapso similar en otro writer, investigar el origen con mas instrumentacion.
- **Bloqueante:** no.

## FU-005 — WARN analisis_lideres.csv cuando no hay sectores favorables

- **Origen:** run manual 2026-09-12, log del workflow daily_run.
- **Descripcion:** `report_generator` intenta leer `outputs/report/analisis_lideres.csv` sin comprobar existencia. Cuando no hay sectores en fase favorable, el fichero no se genera y se emite `[WARN] [Errno 2] No such file or directory`.
- **Impacto:** cosmetico. El reporte se genera igual.
- **Clasificacion:** P3 cosmetico.
- **Accion:** `if path.exists()` antes de leer, o `try/except FileNotFoundError`.
- **Bloqueante:** no.

## FU-006 — save_regime_history escribia filas B2 (RESUELTO)

- **Origen:** run manual 2026-09-12, commit `19b6f30` del bot Daily hist/state.
- **Descripcion:** `save_regime_history` derivaba `obs_date` de `df_macro_manual['date'].max()`. El bot FRED publica `iorb.csv` con fecha del dia natural (incluye sabado/domingo). En fin de semana, `obs_date` caia en dia no bursatil y se escribia una fila B2 en `outputs/history/macro_regime.csv`.
- **Evidencia:** commit `19b6f30` anadio fila `2026-09-12` (sabado). Detectado en workflow manual, no en tests locales.
- **Fix aplicado:** segundo candado `is_market_day(obs_date)` antes de escribir. Commit del fix junto con test de regresion en `test_c4_code.py`.
- **Clasificacion:** P2 estructural - **RESUELTO 2026-09-12**.
- **Bloqueante:** no (ya resuelto).

## FU-010 — Matriz de Evidencia cambia clasificaciones al poblar Flow Proxy

- **Origen:** sesion 2026-09-13, run manual tras fix FU-008-b.
- **Descripcion:** al corregir FU-008-b (el writer de sector_concentration se ejecuta sin depender de leader_df), el Flow Proxy dejo de ser NA en la Matriz de Evidencia. Esto rebalancea la evidencia sectorial y puede cambiar la clasificacion (MIXTA -> DESFAVORABLE, etc.).
- **Evidencia:** run del 2026-09-12 23:28 vs 21:53. Sectores XLF, XLK, XLP, XLU, XLV cambiaron de EVIDENCIA MIXTA a EVIDENCIA PREDOMINANTEMENTE DESFAVORABLE.
- **Impacto:** ninguno funcional. El sistema rebalancea correctamente al tener mas evidencia disponible. Pero reportes consecutivos no son comparables byte-a-byte cuando cambia la disponibilidad de una capa de evidencia.
- **Clasificacion:** P3 documental.
- **Accion:** ninguna. Documentado para evitar sorpresas al comparar reportes historicos.
- **Bloqueante:** no.

## FU-012 — A/D Line acumulada varia +/-1 entre runs sin sesion nueva (WONT FIX)

- **Origen:** sesion 2026-09-13, comparacion de reportes 23:28 vs 23:57.
- **Descripcion:** `A/D Line (acumulada)` paso de 13926 a 13925 entre dos runs consecutivos, ambos con ultima sesion 2026-09-11.
- **Causa:** `indicators/breadth_equity.py:75` calcula `ad_line = ad_net.cumsum()` sobre todo el historico de tickers presentes. Cualquier variacion marginal en precios (re-descarga Yahoo, ajuste de cierre, ticker que cambia de estado) altera el acumulado final.
- **Impacto:** ninguno funcional. El valor es informativo, no participa en scores ni motores.
- **Clasificacion:** WONT FIX. Comportamiento correcto de un acumulador sobre datos variables.
- **Alternativa descartada:** persistir `ad_line` en state para hacerlo determinista. Anade complejidad sin valor analitico.
- **Bloqueante:** no.

## FU-014 — Xetra/BME perdian 38 tickers cuando gap>=2 dias (RESUELTO)

- **Origen:** sesion 2026-09-13, run manual 17:29.
- **Descripcion:** `xetra_provider.py:296` y `bme_provider.py:213` calculaban `start = last_date + timedelta(days=1)` sin aplicar `is_market_day`. Con `last_date = viernes 11/09`, el start caia en sabado 12/09 y la query no devolvia datos. Los 19 tickers Xetra y 19 BME perdian la cobertura -> 275/313 en lugar de 313/313.
- **Evidencia:** log del run 34771231787 -> `SIN COBERTURA EUROPEA: 38 tickers -> [SIE.DE, SAP.DE, ..., SAN.MC, ...]`. Euronext no se vio afectado porque usa `nb_session` en vez de fechas explicitas.
- **Fix aplicado:** `3105689`. Ahora avanzan start al siguiente dia bursatil con `while not is_market_day(next_day.date())`. Si `next_day > today`, se salta la query y se usa cache.
- **Clasificacion:** P1 temporal (bug latente que solo se manifiesta cuando gap>=2 dias y hoy no es bursatil).
- **Bloqueante:** no tras el fix. Era bloqueante para el E2E del lunes sin el.


## FU-015 — Clasificar antes de all_data.append (RESUELTO)

- **Origen:** informe de implementaciones pendientes, 2026-09-15.
- **Descripcion:** `src/stock_data_loader.py:320` ejecutaba `all_data.append(data_batch)` antes de clasificar los tickers del lote. Los tickers FAILED entraban al concat global, y el retry individual (L373) los volvia a anadir. El resultado eran columnas duplicadas que la dedup defensiva de L484 limpiaba con `keep='last'`. Funcionaba, pero dependia del orden de insercion y silenciaba discrepancias entre batch y retry.
- **Evidencia:** AVISO recurrente en logs cuando un retry tiene exito: `AVISO: N columnas duplicadas detectadas, deduplicando (keep=last)`. Run 2026-09-13 22:12 con 5 columnas tras retry de LOW.
- **Fix aplicado:** `afcf095` fix(FU-015). Clasificacion movida antes de `all_data.append`. Nuevo helper `_filter_failed_from_batch` que elimina las columnas de tickers fallidos antes del append. Dedup de L484 se mantiene como red de seguridad (defensa en profundidad).
- **Tests nuevos:** 5 (`tests/test_fu015_dedup.py`). Suite local 212 -> 217.
- **Clasificacion:** P2 estructural — **RESUELTO 2026-09-15**.
- **Bloqueante:** no.

## FU-016 — Desfase 1 dia entre writers cuando run < PUBLISH_HOUR (OBSOLETO 2026-09-17)

- **Origen:** informe de implementaciones pendientes, 2026-09-15.
- **Descripcion:** cuando un run se ejecuta antes de PUBLISH_HOUR=23 (hora Madrid), `last_expected_market_date()` retrocede al ultimo dia con cierre publicado. Los writers derivan su fecha de fuentes distintas: `macro_regime` de `df_macro_manual['date'].max()` (FRED, publicado antes de las 23); `sector_breadth` de `df_stocks.index[-1]` (Yahoo, sesion con cierre ya publicado). Resultado: las dos tablas quedan desfasadas entre si.
- **Evidencia:** run del 2026-09-14 23:00. `macro_regime.csv` -> max 2026-09-14 (40 filas). `sector_breadth.csv` -> max 2026-09-11 (979 filas). Log: `[YAHOO_RAW] expected_session=2026-09-11 raw_last_date=2026-09-14`.
- **Impacto:** observabilidad, no funcionalidad. Un usuario que compare CSVs puede pensar que hay un bug donde no lo hay.
- **Clasificacion:** P3 documental.
- **Accion:** ninguna. Documentado para evitar falsos positivos en futuras auditorias de integridad.
- **Bloqueante:** no.

**Resolucion (2026-09-17):**
- **Gate 0:** `PUBLISH_HOUR=23` vive en `src/market_calendar.py:38`. El cron real de `daily_run.yml` es `0 4 * * *` (04:00 UTC = 06:00 Madrid verano / 05:00 invierno), muy despues de las 23h. La condicion "run < PUBLISH_HOUR" ya no se da en produccion.
- **Clasificacion original:** P3 documental con `Accion: ninguna`. No era deuda activa, era registro para evitar falsos positivos.
- **Estado:** **OBSOLETO 2026-09-17**. No hay fix que aplicar.

## FU-017 — El doble candado B2 no esta codificado (documental)

- **Origen:** verificacion 2026-09-15 durante la integracion del informe tecnico y del roadmap de implementaciones.
- **Descripcion:** el prompt v6.12 (secciones 1.3, 11.7, 16) y el informe tecnico invocan un "doble candado" para sanear fechas sospechosas: `is_market_day(date) == False AND date in CONFIRMED_SET`. La busqueda en todo el repositorio (excluyendo docs) no encuentra ninguna definicion de `CONFIRMED_SET` ni `CONFIRMED_B2_DATES` en codigo. Solo aparece en `docs/auditoria/PROMPT_MAESTRO.md` (lineas 55, 455, 726).
- **Discrepancia adicional:** el informe tecnico lista 7 fechas B2 (incluye 2026-09-13, domingo). El prompt v6.12 lista 6 (sin 2026-09-13). Ambas fuentes son documentales y ninguna esta respaldada por codigo verificable.
- **Interpretacion:** el candado se aplico manualmente durante el saneamiento C4-data (commit `db03b6c`, 2026-09-13). No hay mecanismo automatico que lo aplique en adelante.
- **Impacto:** si en el futuro reaparece una fecha no bursatil en un historico, el saneamiento depende de que el operador recuerde la regla. No hay test que lo verifique.
- **Clasificacion:** P3 documental.
- **Accion:** decidir en ciclo separado si se codifica (helper en `src/market_calendar.py` tipo `is_confirmed_b2(date)` con set versionado) o si se degrada a regla documental explicita en el prompt. No bloqueante.
- **Bloqueante:** no.


## FU-018 - El pipeline EOD mezclaba precios vivos con cierres (RESUELTO)

- **Origen:** deteccion durante FU-002-bug, 2026-09-15.
- **Descripcion:** el pipeline descargaba datos de proveedores con semantica temporal heterogenea. Yahoo y Xetra devolvian la vela del dia en curso cuando su mercado estaba abierto (precio vivo, no cierre). Euronext y BME devolvian EOD por construccion. Al concatenar, el analisis usaba una mezcla de cierres y precios intradia como si fueran la misma observacion.
- **Evidencia:** A/D Net del 15/09 oscilo -8 -> -120 -> -131 en el mismo dia segun la hora del run, sin cambio de mercado. La causa era la mezcla de sesiones cerradas y abiertas.
- **Diagnostico (FU-018-1):** inventario de semantica por provider (commit `e645da4`). Yahoo USA/UK y Xetra requieren filtro; Euronext y BME no.
- **Contrato (FU-018-2):** modelo temporal minimo aprobado por auditor (commit `743242d`). 4 funciones: `get_market`, `is_trading_session`, `get_session_close`, `is_session_closed`. `reference_date` obligatoriamente tz-aware. `UNKNOWN` = no elegible, nunca fail-open.
- **Implementacion:**
  - `6433cdb` FU-018-3a: `src/market_hours.py`, `config/market_close_regular.csv`, `config/market_close_exceptions.csv` (vacio), `get_market` en `instrument_registry.py`. 28 tests.
  - `935de55` + `1a8ce1b` FU-018-3b: filtro Xetra (fetch + cache-hit), `_cache_is_fresh(reference_date)`, `run.py` con `ZoneInfo("Europe/Madrid")`. 15 tests.
  - `4cb5d7d` FU-018-3c: filtro Yahoo en lotes (USA + UK), UNKNOWN no elegible. 12 tests.
- **Verificacion E2E (workflow_dispatch 2026-09-15 15:38 UTC):**
  - `last_date=2026-09-14` (antes 15/09) y `last_date_is_expected_session=True`.
  - `close_nan=0` y `status=VALID`.
  - A/D Net vuelve a -8, coherente con el run nocturno.
  - Logs `[FU-018]` en cada lote Yahoo y cada ticker Xetra indican la eliminacion de la vela no EOD.
- **Tests:** 55 nuevos acumulados (28 + 15 + 12). Suite total: 296 passed + 2 skipped.
- **Clasificacion:** P1 estructural - **RESUELTO 2026-09-15**.
- **Bloqueante:** no tras el fix. Era bloqueante para la fiabilidad del A/D y de cualquier metrica calculada sobre la fila superior del parquet.
- **Notas:**
  - Fuera de scope de FU-018: indices, futuros, FX. Requieren definicion temporal propia.
  - Fuera de scope: calendarios oficiales europeos completos. `is_trading_session` para LSE/XETRA usa lunes-viernes como PROVISIONAL; se ampliara con calendarios oficiales verificados.
  - La lista `config/market_close_exceptions.csv` empieza vacia. Solo se anaden filas con documento oficial.


## FU-019 - Xetra perdia 2 tickers esporadicamente (BAYN.DE, DTG.DE) (RESUELTO)

- **Origen:** deteccion durante la verificacion E2E de FU-018, run `workflow_dispatch` del 2026-09-15 15:38 UTC.
- **Descripcion:** el provider Xetra (WebSocket MDS) no respondia para algunos ISIN de forma esporadica. El `_query_one` realizaba un unico intento y, si tras `WS_QUERY_TIMEOUT=25s` no llegaba respuesta, devolvia lista vacia. El ticker quedaba sin datos y no habia fallback a cache, generando `SIN COBERTURA EUROPEA: 2 tickers`.
- **Evidencia:** run del 15/09 15:38 -> `SIN COBERTURA EUROPEA: 2 tickers -> ['BAYN.DE', 'DTG.DE']`. Ambos tenian ISIN correcto en `config/xetra_ticker_map.csv` y cache local del run de las 02:24.
- **Causa raiz:** WS de Xetra no determinista. Sin reintento ni fallback.
- **Fix aplicado (`1f28bad`):**
  - `_query_one(..., max_attempts=2)`: reintenta con `requestId` distinto por intento, filtrando respuestas huerfanas.
  - `get_prices`: si tras los reintentos `rows` sigue vacio, usa cache local con WARN explicito. Si tampoco hay cache, conserva el comportamiento anterior (ticker sin cobertura).
- **Verificacion E2E (run 2026-09-15 16:04 UTC):** `[XETRA] BAYN.DE OK (303 filas, 2025-07-08 -> 2026-09-15)`, `[XETRA] DTG.DE OK (...)` y `Cascada cubrio 51 tickers europeos`. Sin `SIN COBERTURA EUROPEA`.
- **Tests:** 4 nuevos (`tests/test_fu019_xetra_retry.py`). Suite total: 300 passed + 2 skipped.
- **Clasificacion:** P2 (bug de proveedor externo con mitigacion local) - **RESUELTO 2026-09-15**.
- **Bloqueante:** no tras el fix. Afectaba a la cobertura 313/313 declarada como invariante.


## FU-020 - Motores agregados sin cobertura efectiva declarada (RESUELTO Fase 1)

- **Origen:** deteccion durante la verificacion E2E de FU-019, run del 2026-09-15 16:04 UTC.
- **Descripcion:** los motores agregados (A/D, NH/NL, thrust) calculaban sobre `.iloc[-1]` del DataFrame sin exigir ni declarar la cobertura efectiva de la fecha usada. Tras FU-018, la fila superior puede tener un numero variable de tickers con dato (no siempre 313). El A/D podia medir 42 tickers en un run y 313 en otro, sin que el operador pudiera saberlo.
- **Causa raiz:** `trim_to_last_valid_date_for_tickers(df_stocks, usa_tickers, min_coverage=0.8)` solo exigia cobertura de USA. Si USA estaban completos pero Europa no, conservaba una fila parcial. Si USA no estaban completos, recortaba a la ultima fecha con USA >= 80%, sin garantizar cobertura del universo completo. En ningun caso declaraba la cobertura efectiva.
- **Fix aplicado (Fase 1, commits 306555f + 8e0cbb3):**
  - `src/effective_date.py::resolve_effective_date(prices, eligible_tickers, min_coverage=0.90)` -> resuelve la fecha mas reciente con cobertura suficiente sobre el universo elegible. Devuelve `date`, `requested_date`, `lag_days`, `n_eligible`, `n_observed`, `coverage`, `status`.
  - `config/instrument_exclusions.csv` (vacio inicialmente) para exclusiones permanentes explicitas.
  - `leaders.py`: sustitucion de `trim_to_last_valid_date_for_tickers` por `resolve_effective_date` con universo completo y `min_coverage=0.90`.
  - Propagacion de metadata (`df_stocks_effective_meta`) desde `compute_leaders` a `compute_mte_confirmation`.
  - `compute_advance_decline(df_stocks, effective_meta=None)` declara `effective_date`, `coverage`, `n_observed`, `n_eligible`, `lag_days`.
- **Verificacion E2E (workflow_dispatch 2026-09-15 16:35 UTC):**
  - `[FU-020] effective=2026-09-14 requested=2026-09-15 lag=1d coverage=100.00% (313/313)`
  - `A/D: Net=-8  NH/NL=+0  Thrust=0.46`
  - `effective=2026-09-14  coverage=100.00% (313/313)  lag=1d`
  - Gate 10/10.
- **Tests:** 17 nuevos (`test_fu020_resolve_effective_date.py` + `test_fu020_breadth_meta.py`). Suite total: 317 passed + 2 skipped.
- **Clasificacion:** P1 estructural - **RESUELTO Fase 1** 2026-09-15.
- **Fase 2 (pendiente):** `compute_sector_breadth` y derivados. Puede requerir `resolve_effective_date` por sector.
- **Fuera de alcance:** recalculo de historicos, indices/futuros/FX, evolucion adicional del manifest FU-002.

## FU-021-3A - Filtro EOD en market_data (RESUELTO correction v2)

- **Origen:** deteccion durante la verificacion E2E de FU-020, run `workflow_dispatch` del 2026-09-15 17:04 UTC.
- **Descripcion:** `market_data.parquet` publicaba una fila intradia como si fuera EOD cuando el run caia en horario de sesion USA abierta. El manifest lo detectaba (`last_date_is_expected_session=False`) pero no lo bloqueaba.
- **Evidencia:** run 17:04Z -> `[FU-021-3A] EQUITY_EOD effective=2026-09-15 requested=2026-09-15 lag=0d coverage=100.00% (539/539)`. Manifest `last_date=2026-09-15, expected=2026-09-14, last_date_is_expected_session=False`. Contradice el diseno declarado de FU-021-2.
- **Causa raiz:** FU-021-3A replico el control de cobertura de FU-020 (`resolve_effective_date`) pero no el control de sesion de FU-018. La cobertura mide presencia de NaN, no cierre de sesion. `resolve_effective_date` no consulta calendario.
- **Diagnostico adicional:** `instrument_registry.get_market()` clasifica 22 de 23 tickers no-equity como `US_EQUITY`. Futuros, indices no-USA y FX incluidos. Invalida el port directo de FU-018 a market_data.
- **Fix aplicado (correction v2, commits 78b7583 + 747cfb1 + fded14b):**
  - `src/data_loader.py`: nueva `_filter_non_eod_equity(data, reference_date)` aplica mecanismo FU-018 SOLO al universo EQUITY_EOD (539 tickers). No-equity queda fuera (3B/3C pendientes).
  - Fallback `reference_date` tz-aware (Europe/Madrid) coherente con FU-018-3c.
  - Log ampliado: `function_lag` (propiedad de `resolve_effective_date`) y `reference_lag` (calculado contra `reference_date.date()`). Separacion conceptual exigida por dictamen.
  - `resolve_effective_date` intacta. `trim_to_last_valid_date` no tocado (A3.1 diferido). Manifest FU-002 no tocado.
- **Tests:** 5 nuevos (`test_fu021_3a_equity_eod.py`). Incluye `test_filter_non_eod_equity_100pct_intraday_still_trims` (test clave del auditor: 100% cobertura != EOD). Suite total: 334 passed + 2 skipped.
- **Verificacion E2E (workflow_dispatch 2026-09-15 18:04 UTC sobre rama aislada + 18:20 UTC sobre main):**
  - `[FU-021-3A] EQUITY_EOD: 539/539 equity con vela no EOD en 2026-09-15 -> eliminar ultima fila`
  - `[FU-021-3A] EQUITY_EOD effective=2026-09-14 requested=2026-09-14 function_lag=0d reference_lag=1d coverage=99.81% (538/539)`
  - Manifest: `last_date=2026-09-14, expected=2026-09-14, last_date_is_expected_session=True`
  - `[FU-020] effective=2026-09-14 requested=2026-09-15 lag=1d coverage=100.00% (313/313)` (Ruta A intacta)
  - Gate 10/10.
- **Clasificacion:** P1 estructural - **RESUELTO 2026-09-15**.
- **Informe y dictamen completo:** `docs/auditoria/FU-021-3A_INFORME_Y_DICTAMEN.md`.
- **Deudas derivadas abiertas:**
  - FU-002-bis: endurecer regla INVALID para `last_date > expected_session`.
  - FU-021-5: metadata temporal por clase en manifest.
  - FU-021-3B: INDEX_EOD + RATE_YIELD (verificacion empirica Close Yahoo).
  - FU-021-3C: FUTURE_SETTLEMENT + FX_DAILY_CUT (provider dedicado).
  - A2.3: corregir `get_market` para `^`/`=F`/`=X`.
  - A3.1: retirar `trim_to_last_valid_date` de `data_load.py` (diferido a sub-informe de 13 consumidores).


## FU-002-bis - temporal_contract declarativo en writer de manifest (RESUELTO)

- **Origen:** derivado del dictamen de FU-021-3A. El manifest FU-002 no bloqueaba `last_date > expected_session` (deteccion sin accion).
- **Descripcion:** la regla `quality.status` fue reescrita por FU-002-evo2 (commit `0ed695f`) y dejo de consultar `last_date_is_expected_session`. Resultado: una fila futura (intradia tratada como EOD) podia declararse VALID o VALID_WITH_MISSING segun close_nan.
- **Hallazgo adicional:** el analisis USA+UK revelo que `expected_session = last_expected_market_date(reference_date)` es autoridad temporal solo para universos USA puros. `market_data.parquet` (562 mixtos) y `stock_prices.parquet` (313 USA+UK) son heterogeneos y quedan exentos de validacion temporal hasta FU-021-5.
- **Diseno aprobado (dictamen v3):**
  - `temporal_contract` declarativo, opcional, keyword-only.
  - Bloque `temporal: {contract: null | string}` siempre presente en el manifest.
  - Regla temporal con precedencia maxima cuando hay contrato declarado: `last_date > expected_session` -> `INVALID`, sin depender de close_nan (temporalidad y completitud son dimensiones independientes).
  - Writer no descubre mercados por `get_market()`. Resolucion por clase queda en FU-021-5.
  - Reader intacto. `schema_version = 1`.
- **Fix aplicado (`d068c99`):**
  - `src/utils.py::write_artifact_with_manifest`: parametro keyword-only `temporal_contract=None`. Rama temporal al frente de la regla de status. Bloque `temporal` en el JSON. Log INFO cuando `None`.
  - `tests/test_artifact_manifest.py`: renombrado `test_fu_002_evo2_valid_with_missing_pre_publish` -> `..._no_contract`. 5 tests nuevos.
  - `tests/test_backup_provider_reference.py`: 4 tests writer<->reader.
- **Tests:** 9 nuevos (5 writer + 4 reader). Suite total: 343 passed + 2 skipped.
- **Verificacion E2E (workflow_dispatch 2026-09-15 20:20 UTC sobre main):**
  - `[MANIFEST] data/market_data.parquet: temporal_contract=None (validacion temporal no aplicada)`.
  - `[MANIFEST] data/stock_prices.parquet: temporal_contract=None (validacion temporal no aplicada)`.
  - Reader acepta ambos manifests con bloque temporal.
  - Gate 10/10.
- **Clasificacion:** P1 estructural (preparacion contractual) - **RESUELTO 2026-09-15**.
- **Estado efectivo:** el mecanismo existe pero ningun caller lo activa. `temporal_contract=None` en todos los runs. Validacion temporal queda desactivada hasta que FU-021-5 defina contratos por clase.
- **Deudas derivadas abiertas:**
  - FU-021-5: definir semantica temporal por clase y activar `temporal_contract` desde los callers.
  - A2.3: corregir `get_market` para `^`/`=F`/`=X`.


## FU-015 - Columnas duplicadas tras retry Yahoo (RESUELTO)

- **Origen:** deteccion durante FU-002-bug, 2026-09-15.
- **Descripcion:** el AVISO "N columnas duplicadas detectadas, deduplicando" aparecia en el concat final de `stock_data_loader.py`. Causa: un ticker fallido en un batch podia triunfar en el retry individual, y ambos se anadian a `all_data`. El concat producia columnas duplicadas que la dedup defensiva debia limpiar (fragil y ruidoso).
- **Causa raiz:** `all_data.append(data_batch)` se ejecutaba ANTES del bucle de clasificacion que marcaba los tickers FAILED. Los FAILED del batch no se excluian antes del append; el retry posterior anadia las mismas columnas.
- **Fix aplicado (`afcf095`):**
  - Bucle de clasificacion movido ANTES de `all_data.append(data_batch)`.
  - Nueva `_filter_failed_from_batch(data_batch, failed_in_batch)`: excluye columnas de tickers FAILED del batch antes del append.
  - La dedup defensiva (L583-588) se mantiene como red de seguridad, no como mecanismo primario.
- **Tests (`tests/test_fu015_dedup.py`):** 5 tests.
  - 4 unitarios del helper (lista vacia, no-MultiIndex, remove columnas, ticker desconocido).
  - 1 test estructural: verifica en el fuente que `failed_in_batch = []` aparece antes de `all_data.append(data_batch)`.
- **Verificacion en produccion:** en los runs del 2026-09-15 (17:04Z, 17:51Z, 18:04Z, 18:20Z, 20:20Z) no aparece el AVISO "columnas duplicadas detectadas". La dedup defensiva no se ha activado.
- **Clasificacion:** P2 (integridad tecnica localizada) - **RESUELTO 2026-09-15**.
- **Notas:**
  - `data_loader.py` no tiene el patron del bug: no hay retry individual de tickers. No requiere fix equivalente.
  - El prompt v6.14 marcaba FU-015 como pendiente P2 por desfase documental. Cierre corregido.


## A2.3 - get_market clasificaba no-equity como US_EQUITY (RESUELTO)

- **Origen:** hallazgo colateral durante FU-021-3A. `get_market` clasificaba futuros, indices no-USA y FX como `US_EQUITY` (30 de 31 tickers no-equity mal clasificados).
- **Descripcion:** `get_market(ticker)` responde "que calendario bursatil aplica", no "que tipo de instrumento es". La confusion surgio al usarla como clasificador de universo en el analisis USA+UK de FU-002-bis.
- **Causa raiz:** `get_market` nunca tuvo la responsabilidad de clasificar economicamente. Los callers actuales la usan correctamente (solo calendario). El bug es latente: ningun caller le pasa hoy tickers no-equity.
- **Riesgo de un fix in-place:** cambiar `get_market("CL=F")` de `US_EQUITY` a `FUTURE` romperia `is_trading_session("FUTURE", ...)` (UnknownMarketError) y los callers actuales.
- **Fix aplicado (`fd12ea1`):**
  - Nueva `get_instrument_class(ticker)` en `src/instrument_registry.py`. Paralela a `get_market`, no la sustituye.
  - Devuelve: `EQUITY`, `INDEX`, `VOLATILITY_INDEX`, `RATE_YIELD`, `FUTURE`, `FX`, `UNKNOWN`.
  - Subtipos (`COMMODITY_INDEX`, `CURRENCY_INDEX`, `EQUITY_INDEX`) diferidos a FU-021-5.
  - `get_market` sin cambios. Callers actuales sin cambios.
  - Principio fijado: `INSTRUMENT CLASS != MARKET != TEMPORAL CONTRACT`.
- **Tests (`tests/test_a2_3_instrument_class.py`):** 19 tests.
  - 15 funcionales de `get_instrument_class` (una por clase + bordes).
  - 4 de regresion: confirman que `get_market` NO cambia su semantica, ni para equity ni para no-equity.
- **Verificacion:** compileall OK, pyflakes 0 warnings, 362 passed + 2 skipped. Sin E2E (el camino de produccion no cambia: `get_market` sigue igual).
- **Clasificacion:** P3 (deuda tecnica estructural latente) - **RESUELTO 2026-09-15**.
- **Consumidor futuro:** `get_instrument_class` sera utilizada por FU-021-5 para asignar contratos temporales por clase (EQUITY_EOD, INDEX_EOD, RATE_YIELD, FUTURE_SETTLEMENT, FX_DAILY_CUT).


## FU-021-3B - Verificacion empirica INDEX_EOD y RATE_YIELD (INFORME CERRADO)

- **Origen:** investigacion empirica previa a Parte B de FU-021-5. Autorizado por dictamen del contrato temporal (Q-C2.6, camino alpha).
- **Informe principal:** `docs/auditoria/FU-021-3B_INFORME_INDEX_EOD_RATE_YIELD.md` (`b645de4`, HEAD ref `61aa99f`).
- **Anexo:** `docs/auditoria/FU-021-3B_ANEXO_VOLATILITY_INDEX.md` (`794f0e9`). Cierra la laguna de VOLATILITY_INDEX.
- **Fecha:** 2026-09-15 (informe) / 2026-09-16 (anexo).
- **Alcance:** 10 indices + 2 yields + 3 indices de volatilidad del universo `df_market`.
- **Objetivo:** medir empiricamente la semantica temporal de las clases no verificadas en Parte A (INDEX_EOD, RATE_YIELD), necesaria para redactar Parte B.
- **Hallazgos clave:**
  - Causa raiz del lag europeo medida (INDEX_EOD_EUROPA con `per_ticker_lag`, `max_lag=5`).
  - VOLATILITY_INDEX confirmada como clase independiente que hereda de INDEX_EOD_USA (subclase, no asumible desde INDEX_EOD_USA).
  - Universo no-equity: 6 clases confirmadas (Parte A documentaba 5).
- **Clasificacion:** informe empirico cerrado. Alimenta Parte B y Plan de FU-021-5.
- **Notas:** el descubrimiento '5 vs 6 clases' origino la deuda documental K-FU-021-5-01.
- **Hallazgo colateral:** E5 (anomalia yfinance en `^VIX3M`, individual vs batch) detectado durante el anexo.


## FU-021-3C - Verificacion empirica FUTURE_SETTLEMENT y FX_DAILY_CUT (INFORME CERRADO)

- **Origen:** contraparte de FU-021-3B para las dos clases restantes bloqueadas por provider.
- **Informe:** `docs/auditoria/FU-021-3C_INFORME_FUTURE_FX.md` (`794f0e9`, HEAD ref `b645de4`).
- **Fecha:** 2026-09-16.
- **Alcance:** 5 futuros de commodities (BZ=F, CL=F, GC=F, HG=F, NG=F) + 3 pares FX (EURUSD=X, USDCNY=X, USDJPY=X) del universo `df_market`.
- **Objetivo:** confirmar empiricamente si la clasificacion de Parte A ('bloqueadas por provider dedicado') se sostiene.
- **Hallazgos clave:**
  - H-3C-7: historicos de futuros via Yahoo no reproducibles (contratos explicitos reescriben historicos).
  - CME/ICE bloqueados por IP tras uso intensivo.
  - FX_DAILY_CUT: cutoff exacto provisional (17:00 ET).
  - Ambas clases confirmadas como BLOCKED por provider en el momento del informe.
- **Cierre sin causa raiz exacta:** H-3C-7 no tiene causa raiz reproducible. Investigacion detenida por ROI negativo, documentada en el propio informe.
- **Desbloqueo posterior:**
  - FUTURE_SETTLEMENT: `78f18a8` (FU-021-3C-bis, OilPriceAPI close_proxy).
  - FX_DAILY_CUT: contrato activo en FU-021-5 con `per_pair_max_lag`.
- **Clasificacion:** informe empirico cerrado. Alimenta Parte B de FU-021-5.
- **Notas:** desbloqueo llega por via alterna (OilPriceAPI). Ver entrada FU-021-3C-bis.


## FU-021-5 - Contratos temporales explicitos sobre df_market (RESUELTO)

- **Origen:** el pipeline usaba fechas implicitas (ultima fila, max index). R1-R3 (FU-020) exigian declarar fecha efectiva y cobertura por metrica. Sin contrato temporal explicito, la trazabilidad era fragil.
- **Objetivo:** definir contratos temporales declarativos por clase de instrumento, con FSM y autoridad explicita, propagados a todo consumidor.
- **Alcance:** 9 contratos en 5 familias (EQUITY, INDEX, VOLATILITY_INDEX, RATE_YIELD, FUTURE_SETTLEMENT, FX_DAILY_CUT). 20 commits (`c1a1df9..7560e12`).
- **Tag:** `fu-021-5-completo` en `7560e12`.
- **Arquitectura nueva:** `src/temporal_contracts/` (10 modulos): `base.py` (FSM PENDING|OK|STALE|INSUFFICIENT|BLOCKED), `registry.py` (catalogo), `consolidate.py` (build_temporal_meta), + 7 contratos.
- **Contratos implementados:**
  - EQUITY_EOD (539 tickers, NYSE, max_lag=0, min_cov=0.90).
  - INDEX_EOD_USA (^GSPC ^DJI ^NDX ^RUT).
  - INDEX_EOD_EUROPA (^FTSE ^GDAXI ^IBEX ^STOXX50E, per_ticker_lag, max_lag=5).
  - INDEX_EOD_COMMODITY (^SPGSCI).
  - INDEX_EOD_CURRENCY (DX-Y.NYB, ICE).
  - VOLATILITY_INDEX (^VIX ^VIX3M ^VXN, hereda de INDEX_EOD_USA).
  - RATE_YIELD (^FVX ^TNX).
  - FUTURE_SETTLEMENT (BZ=F CL=F GC=F HG=F NG=F). BLOCKED al cierre del ciclo (Q-B.4).
  - FX_DAILY_CUT (EURUSD=X USDCNY=X USDJPY=X, per_pair_max_lag).
- **Autoridad (Q-P.3):** `temporal_meta` es dict explicito. `df.attrs` es espejo auxiliar, nunca autoridad.
- **Transporte:** `MarketDataBundle` (dataclass).
- **Propagacion:** 13 consumidores pipeline + run.py + 3 regimes + 16 indicadores.
- **Writers migrados:** 20 (patrones P0-P8). Helper `src/utils.py::writer_observation_date`.
- **MTE state:** schema versionado (schema_version=1, temporal_contract_version, effective_date, expected_date, coverage, futures_status=BLOCKED). Reset automatico si contrato cambia.
- **Darkpool:** `compute_darkpool_signals(df_market, df_stocks)` con fallback parquet (Q-P.7).
- **A3.1 desbloqueada:** `trim_to_last_valid_date` retirado de `data_load.py`. Verificacion empirica: diff filas = 0 con/sin trim (redundante con FU-021-3A). Funcion marcada DEPRECATED en `src/utils.py`.
- **Tests:** `test_temporal_contracts_base.py` (14, FSM 5 estados), `test_temporal_contracts_registry.py` (13), `test_temporal_contracts_contracts.py` (37), `test_temporal_contracts_remaining.py` (15), `test_temporal_contracts_consolidate.py` (22). Total: ~101 tests.
- **Verificacion:** compileall OK, pyflakes 0 warnings, 469 passed + 2 skipped al cierre. Gate 10/10.
- **Clasificacion:** ciclo de refactor arquitectonico. RESUELTO 2026-09-16.
- **Deudas residuales del ciclo:** K7 (Plan §2.4 vs §2.5), K-FU-021-5-01..06.
- **Sucesor:** FU-021-3C-bis resuelve el unico BLOCKED pendiente (FUTURE_SETTLEMENT). Ver entrada siguiente.


## FU-021-3C-bis - Commodities via OilPriceAPI + SPOT_COMMODITY (RESUELTO)

- **Origen:** FU-021-3C cerro con FUTURE_SETTLEMENT y FX_DAILY_CUT en BLOCKED. Yahoo agotado, CME/ICE bloqueados por IP. FUTURE_SETTLEMENT era el unico contrato BLOCKED del sistema.
- **Objetivo:** desbloquear FUTURE_SETTLEMENT con un provider dedicado. Cierra FU-021-3C por via alterna.
- **Solucion:** OilPriceAPI como provider oficial de commodities. Dos contratos diferenciados por semantica.
- **Contratos temporales (10 en total, 9 -> 10):**
  - `FUTURE_SETTLEMENT` (reducido a BZ=F, CL=F). `settlement_semantics=close_proxy`. `close` de OilPriceAPI como proxy del settlement oficial ICE/NYMEX (<0.5% diff).
  - `SPOT_COMMODITY` (nuevo, GC=F, HG=F, NG=F). `settlement_semantics=spot_reference`. Spot, no futuros.
- **Regla nueva:** R5. Los tickers de commodities se alimentan exclusivamente via OilPriceAPI. Prohibido mezclar con Yahoo.
- **Provider:** `data/providers/futures.py::FuturesProvider` (~250 LOC). Endpoints `/v1/futures/ice-brent`, `/v1/futures/ice-wti`, `/v1/prices/latest`. API key env `OIL_PRICE_API` con fallback local. Reintentos: 1 en timeout, 0 en 401/429. Presupuesto: 3 req/dia (90/mes sobre 200 plan free).
- **Merge:** `src/commodities_merge.py::merge_commodities_into_market` (~100 LOC). Solo merge en fechas comunes (no anade filas, respeta FU-021-3A).
- **Workflow:** step 'Update commodities' en `daily_run.yml`, antes de 'Run Macro Sectorial'. `scripts/update_futures.py` con skip si parquets al dia (no quema requests).
- **Artefactos nuevos:** `data/commodities_futures.parquet` y `data/commodities_spot.parquet`, con manifest FU-002.
- **Bug latente corregido (FU-002):** `write_artifact_with_manifest` con df de 1 fila lanzaba `UnboundLocalError` en `close_cols`. Fix: inicializar `close_cols=[]` antes del bloque condicional. Pre-existente.
- **Tests:** `test_provider_futures.py` (17: parsing, merge, API key, reintentos), `test_commodities_merge.py` (10: merge idempotente, degradacion). `test_artifact_manifest.py` +1 (df 1 fila, bug close_cols). Total: 469 -> 498 (+29).
- **Commits del ciclo (13):** `303db48..1b73b02` + fix docs `bf9c6fe`, `03fcb3e`, `1b73b02`, `24b3bf6`, `e446fe4`.
- **Verificacion:** Gate 10/10. 47/47 secciones `##` identicas pre/post. BZ=F, CL=F integrados correctamente. `FUTURE_SETTLEMENT=STALE` (era BLOCKED). `SPOT_COMMODITY=STALE` (nuevo).
- **Nota metodologica** en reporte: seccion 'Momentum de Precio - Otros Activos' indica semantica de los 5 tickers.
- **Clasificacion:** ciclo de integracion de provider. RESUELTO 2026-09-16.
- **Deudas nuevas (K-FU-021-3C-bis-01..09):** K-01 (git pull --rebase, resuelto `03fcb3e`), K-02 (SPOT_COMMODITY STALE sistematico), K-03 (CI real, pendiente), K-04 (tests fragiles residuales), K-05 (transfer doc), K-06 (prompt v6.16, resuelto `bf9c6fe`), K-07 (informe formal), K-08 (contador dinamico, resuelto `03fcb3e`), K-09 (6 workflows pull --rebase, resuelto `e446fe4`).


## K-01 / K-06 / K-08 / K-09 - Fixes CI post FU-021-3C-bis (RESUELTOS)

- **Origen:** el ciclo FU-021-3C-bis expuso cuatro deudas operativas. K-01 (push rechazado por falta de rebase), K-06 (prompt desactualizado), K-08 (contador hardcodeado), K-09 (6 workflows con mismo patron que K-01).
- **Contexto:** el 2026-09-16 un push de `daily_run.yml` fue rechazado en produccion. Mitigado a mano. Origen: ausencia de `git pull --rebase origin main` antes del push del commit automatico.
- **Fixes aplicados:**
  - **K-01** (`03fcb3e`): `daily_run.yml` anade `git pull --rebase origin main` entre `git commit` y `git push`.
  - **K-06** (`bf9c6fe`): `PROMPT_MAESTRO.md` actualizado a v6.16 con ciclo FU-021-3C-bis completo.
  - **K-08** (`03fcb3e`): `data_loader.py` usa `len(_resolutions)` en lugar de '9 contratos' hardcodeado.
  - **K-09** (`e446fe4`): mismo fix que K-01 aplicado a `update_european_holdings.yml`, `update_index_holdings.yml`, `update_macro_manual.yml`, `update_qqq_sec_flow.yml`, `update_sector_holdings.yml`, `update_sec_nport.yml`.
- **Fix adicional de mantenimiento del prompt** (`24b3bf6`): Seccion 9 decia `0 20 * * *` para `daily_run.yml`; el YAML real es `0 4 * * *`. Corregido. Bump v6.16 -> v6.17.
- **Tests:** ninguno especifico (cambios de CI + docs). Verificacion: compileall OK, pyflakes 0 warnings, suite completa 498+2.
- **Verificacion en produccion:** pendiente de confirmar el proximo cron sin workflow_dispatch (K-03).
- **Clasificacion:** P1 (integridad CI). RESUELTOS 2026-09-17.
- **Deudas residuales del ciclo FU-021-3C-bis (abiertas):**
  - K-02 (MEDIA): SPOT_COMMODITY en STALE sistematico por desalineacion spot/market (1 dia natural).
  - K-03 (ALTA): confirmar cron real sin workflow_dispatch.
  - K-04 (MEDIA): `test_temporal_contracts_remaining.py` y `test_temporal_contracts_consolidate.py` con REF hardcodeado.
  - K-05 (BAJA): transfer doc original desactualizado.
  - K-07 (BAJA): informe formal FU-021-3C-bis en `docs/auditoria/` no redactado.

## FU-021-3D - CBOE para ^VIX3M (RESUELTO)

- **Origen:** E5 (anomalia ^VIX3M en Yahoo). Yahoo dejo de servir historico de indices de term structure. Solo devuelve cotizacion actual. VOLATILITY_INDEX cae a INSUFFICIENT en cada run.
- **Informe previo:** `docs/auditoria/E5_INFORME_VIX3M.md` (diagnostico, opciones, decision B).
- **Dictamen auditor:** opcion B aprobada. CboeIndexProvider paralelo (no tocar CboeProvider existente). R6 aprobada. ^VIX9D fuera de alcance. NO-GO hasta prueba de aceptacion.
- **Prueba de aceptacion:** CBOE HTTP 200, CSV DATE,OPEN,HIGH,LOW,CLOSE, 4273 filas (2009-09-18 -> 2026-09-15), 0 NaN, 0 duplicados, reproducible byte-exacto. Inyeccion en df_market: VOLATILITY_INDEX INSUFFICIENT -> OK/STALE con coverage 1.0.
- **Componentes:**
  - `data/providers/cboe_index.py::CboeIndexProvider` (153 LOC, sin auth).
  - `src/cboe_merge.py::merge_cboe_into_market` (85 LOC, idempotente, solo fechas comunes).
  - `scripts/update_cboe.py` (65 LOC, skip si al dia, best-effort).
  - `data/cboe_vix3m.parquet` + manifest FU-002.
  - Step Update CBOE en `daily_run.yml`.
  - R6 en PROMPT_MAESTRO v6.18.
- **Commits:** ca223c0, 3c25262, 904947f, eb071db, a608605, a560708.
- **Tests:** `test_provider_cboe_index.py` (17), `test_cboe_merge.py` (11). Total +28.
- **Verificacion:** compileall OK, pyflakes limpio. Gate 10/10 en runs reales pendiente del proximo cron.
- **Clasificacion:** ciclo de integracion de provider. RESUELTO 2026-09-17.
- **Deudas generadas:** ninguna nueva. BOM de `cboe.py` y limpieza ya gestionados como K-ID separado.

## FU-021-3D-cierre - Cierre documental del ciclo (RESUELTO)

- **Origen:** cierre del ciclo FU-021-3D tras run CI `38092ed`.
- **Correccion a entrada historica:** la entrada `FU-021-3D` afirmaba "deudas generadas: ninguna nueva". Era prematuro: se redacto antes del run CI que revelo el estado real.
- **Informe E5 persistido:** `docs/auditoria/E5_INFORME_VIX3M.md`. La entrada historica referenciaba este fichero antes de que existiera. Ya reparado.
- **Verificacion real:** run CI `38092ed` (workflow_dispatch, 2026-09-17). Resultado: Gate 10/10, VOLATILITY_INDEX=OK, 10/10 contratos temporales en OK/STALE.
- **Deudas generadas por el ciclo (cierre real):**
  - K-FU-021-3D-01 (manifests en git add): RESUELTO `5caceda`.
  - K-FU-021-3D-02 (git add -u + diagnostico): RESUELTO `7035a1e`.
  - K-FU-021-3D-03 (BAJA): FutureWarning commodities_merge.py:75 (dtype incompatible, Pandas 3.0).
  - K-FU-021-3D-04 (BAJA): FutureWarning momentum.py .ffill (downcasting object dtype, Pandas 3.0).
  - K-FU-021-3D-05 (BAJA): `docs/auditoria/auditoria_arquitectura.md` se regenera en cada run. Entra por `git add -u`. Documentar.
- **K-FU-021-3C-bis-03:** RESUELTO. Confirmado con workflow_dispatch en CI real.
- **Clasificacion:** cierre documental. RESUELTO 2026-09-17.
- **Nota:** la entrada historica `FU-021-3D` se preserva sin reescribir. Este bloque refleja el estado final verificado.

## DT2 - Refactor MTE (CERRADO)

- **Origen:** deuda tecnica ALTA. `indicators/mte.py` (1963 LOC, 21 funciones).
- **Dictamen auditor:** B (refactor por fases) + encoding FUERA + golden HOY.
- **Fases ejecutadas:**
  - Fase 0 (`6433bc1`): golden reference reproducible. 7 fixtures + 1 golden JSON. Tolerancias beta: exact para `scenario`, 1e-9 para srs/ipi/ips, 1e-3 para cls/shs/msi/confidence (derivada de variacion FRED, documentada).
  - Fase 1 (`9eeaaa8`, `9473897`, `3407909`): 25 tests nuevos MTE (engine, state, scoring).
  - Fase 2 (`c717e56`): modulo -> paquete con alias sys.modules.
  - Fase 3 (`2b7363a`): state.py extraido. Paquete real (sin alias).
  - Fase 4 (`43d52a3`): scoring.py extraido (7 funciones + helpers tanh/_get_last + SCENARIO_WEIGHTS).
  - Fase 5 (`f427e80`): decision.py extraido (5 funciones + NORMAL_TRANSITIONS + EXCEPTION_TRANSITIONS).
  - Fase 6 (`571f165`): engine.py extraido, `mte_legacy.py` eliminado.
- **Arquitectura final:** `indicators/mte/` con `__init__.py`, `engine.py`, `state.py`, `scoring.py`, `decision.py`.
- **Verificacion:** 25 tests MTE + 551 suite global verde. compileall OK, pyflakes limpio. Gate 10/10 en E2E local. Golden sin cambios semanticos. Import limpio desde proceso independiente.
- **U+FFFD:** 28 -> 26. Los 2 desaparecidos pertenecian a codigo eliminado durante la extraccion (docstring del modulo legacy + header de seccion huerfano). No fueron corregidos. K-ID de encoding sigue separado.
- **Hallazgo colateral preexistente (NO regresion DT2):** cache-hit path de `src/data_loader.py::download_market_data` omite post-processing (merges), manifest y propagacion de `temporal_meta`. Ajeno al alcance de DT2 (`data_loader.py` no se toco).
- **Clasificacion:** ciclo de refactor arquitectonico. CERRADO 2026-09-17.
- **Deuda generada:** `K-DATA-LOADER-01` (abierta por dictamen del auditor).
- **Notas:**
  - El refactor no altero la salida del motor. Golden test lo verifica con datos congelados.
  - El paquete `indicators.mte` mantiene API publica: `from indicators.mte import compute_mte` sigue funcionando (via `from .engine import compute_mte`).

## K-DATA-LOADER-01 - Cache-hit path bypasses post-processing (ABIERTA)

- **Origen:** hallazgo colateral durante E2E de DT2 (2026-09-17).
- **Descripcion:** `src/data_loader.py::download_market_data` tiene un `return _df` en el cache path (dentro del bloque `if datetime.now() - mtime < timedelta(hours=CACHE_HOURS)`) que salta sin ejecutar:
  - `merge_commodities_into_market(data)` (FU-021-3C-bis).
  - `merge_cboe_into_market(data)` (FU-021-3D).
  - `write_artifact_with_manifest(data, ...)` (FU-002).
  - Bloque `[FU-021-5]` que setea `data.attrs["temporal_meta"]`.
- **Consecuencia observable:**
  - **CI (cache cold):** descarga real -> todos los bloques corren -> `temporal_meta` poblado -> `mte_state.json` con `effective_date="2026-09-16"`, `coverage=1.0`.
  - **Local (cache warm):** return temprano -> df.attrs vacio -> `compute_mte` escribe `effective_date=None`, `coverage=None`.
- **Evidencia empirica:**
  - Log CI `38092ed`: contiene `[commodities_merge]`, `[cboe_merge]`, `[MANIFEST]`, `[FU-021-5]`. State con `effective_date=2026-09-16`, `coverage=1.0`.
  - Log local E2E DT2: NO contiene ninguno de esos bloques. State con `effective_date=None`, `coverage=None`.
  - Diagnostico directo: `download_market_data` retorna df con `attrs={}` tras cache hit.
- **Impacto:**
  - Produccion (CI): sin impacto (cache cold siempre).
  - Local con cache warm: `temporal_meta` no se propaga; `mte_state.json` incompleto en `effective_date`/`coverage`. El resto del pipeline opera normalmente (Gate 10/10).
- **Origen (historia):** commits `e1f487b` (FU-021-5 Fase 3) y `a608605` (FU-021-3D Fase 6). Anteriores a DT2. No es regresion.
- **Prioridad:** MEDIA (no bloquea Gate; afecta trazabilidad temporal en runs locales).
- **Alcance inicial:**
  - **P0 confirmado:** `download_market_data` cache path.
  - **P1 investigacion:** `download_stock_prices` cache path. Evidencia parcial: `[FU-020] effective=2026-09-15 lag=1d` local vs `effective=2026-09-16 lag=0d` CI. No confirmado.
- **Decision arquitectonica pendiente:**
  - Opcion A: el cache contiene artefacto contractual completo -> `return _df` es correcto, pero el parquet debe persistir `temporal_meta` en el manifest.
  - Opcion B: el cache contiene raw/intermediate -> debe pasar por post-processing nuevamente.
  - Dictamen previo obligatorio antes del patch.
- **Clasificacion:** ciclo de investigacion + fix en `src/data_loader.py`. ABIERTA 2026-09-17.

## K-DATA-LOADER-01-cierre - Cierre del ciclo (RESUELTO)

- **Origen:** cierre de la investigacion abierta durante DT2.
- **Dictamen:** opcion B aprobada. El cache es raw/intermediate; debe pasar por post-processing en cada lectura. Idempotencia de `clean_oil_prices`, `_filter_non_eod_equity`, `_trim_market_data_to_equity_eod` y `merge_*` verificada antes de escribir patch.
- **Fix aplicado:** extraccion de `_postprocess_market_data(data, reference_date, run_id, *, write_manifest)` en `src/data_loader.py`. Cache-hit lo invoca con `write_manifest=False` (no reescribe parquet, si actualiza `temporal_meta`). Cache-miss con `write_manifest=True`.
- **Commits:** `9d77099`.
- **Tests:** `tests/test_data_loader_cache_postprocess.py` (4 nuevos). Total 551 -> 555 passed + 2 skipped.
- **Verificacion local (E2E):** cache-hit (CACHE_HOURS=23, parquet fresco) ejecuta post-procesado. Log contiene `[FU-021-3A] EQUITY_EOD effective=2026-09-16 ... coverage=99.81% (538/539)` + `[FU-021-5] 10 contratos resueltos: ...` sin `[CACHE] ... Forzando descarga`. `outputs/state/mte_state.json` con `effective_date='2026-09-16'`, `coverage=0.998`.
- **Verificacion CI (run 35167520594):** exit 0. Path cache-miss sigue funcionando. Gate 10/10.
- **Nota tecnica:** los tests nuevos usan `importlib.import_module` porque `src/temporal_contracts/__init__.py` sombrea el nombre del submodulo `consolidate` con un atributo funcion. Documentar patron.
- **Clasificacion:** ciclo de fix en `src/data_loader.py` + tests. RESUELTO 2026-09-17.
- **Deudas generadas:** ninguna nueva. Hallazgo colateral abierto como `K-CI-CRON-01`.

## K-DT2-GOLDEN-EOL - Hash mismatch del golden MTE local vs CI (RESUELTO)

- **Origen:** run CI `35166754806` sobre `9d77099`. Falla `tests/test_mte_engine.py::TestGoldenIntegrity::test_golden_hashes_valid` con `hash mismatch mte_financial_score.csv: b374d4b5a3fa6213 vs 694d189771dea08b`.
- **Contexto:** el run CI previo (`35157099407`, sobre `7035a1e`) tenia `528 collected`, pre-DT2. DT2 introdujo el test sin cobertura efectiva en CI.
- **Causa raiz:** working tree local tenia doble CRLF (`\r\r\n`) en los fixtures. El blob committed tenia `\r\n`. El golden se calculo sobre el working tree via `_sha256_normalized` (que colapsa `\r\n -> \n`), lo cual sobre `\r\r\n` da el hash del blob CRLF (`694d1897`), no el hash LF (`b374d4b5`). Local: `\r\r\n` -> `\r\n` -> hash `694d1897` OK. CI: `\r\n` -> `\n` -> hash `b374d4b5` != `694d1897` FAIL.
- **Diagnostico:** `len(working) - len(blob) = 2606` = exactamente el numero de secuencias `\r\r\n`. Confirmado con contadores byte-exactos sobre working tree, blob committed y variantes.
- **Fix aplicado:**
  - Renormalizar `tests/fixtures/*.csv` y `*.json` a LF puro (elimina `\r\r\n` -> `\n` y `\r\n` -> `\n`). Parquet intacto (binario).
  - Regenerar `sha256` y `size` del golden desde el working tree LF actual.
  - `git add --renormalize tests/fixtures/` para absorber blobs committed a LF.
- **Commits:** `ee34305`.
- **Ficheros afectados:** `mte_financial_score.csv`, `mte_all_signals.csv`, `mte_golden_2026-09-16.json` (3 ficheros; los otros 4 ya estaban en LF).
- **Tests:** `tests/test_mte_engine.py` (3 tests, todos verdes). Suite completa: 555 passed + 2 skipped.
- **Verificacion CI (run 35167520594):** `test_golden_hashes_valid PASSED`, `test_compute_mte_matches_golden PASSED`, `VALIDATION GATE: Sin errores (10 comprobaciones OK)`. Exit 0.
- **Leccion:** los hashes de fixtures en golden deben calcularse sobre el **blob committed** o sobre el working tree **LF normalizado**, nunca sobre el working tree con CRLF/mixto. Considerar migrar `_sha256_normalized` a leer siempre el blob via `git cat-file`.
- **Clasificacion:** ciclo de fix en fixtures + golden. RESUELTO 2026-09-17.
- **Deudas generadas:** ninguna nueva.

## K-CI-CRON-01 - Contratos STALE/INSUFFICIENT en run fuera de cron (ABIERTA, en observacion)

- **Origen:** run CI `35167520594` (workflow_dispatch, 2026-09-17 00:40 UTC) sobre `ee34305`.
- **Observacion:** los 10 contratos temporales resolvieron como `INSUFFICIENT`/`STALE`. Log: `[FU-021-3A] EQUITY_EOD effective=2026-09-15 requested=2026-09-17 function_lag=2d reference_lag=2d coverage=100.00% (539/539)` + `[FU-021-5] 10 contratos resueltos: EQUITY_EOD=INSUFFICIENT INDEX_EOD_USA=STALE ... FUTURE_SETTLEMENT=INSUFFICIENT SPOT_COMMODITY=INSUFFICIENT FX_DAILY_CUT=INSUFFICIENT`.
- **Diagnostico preliminar:** `EQUITY_EOD` con `max_lag=0` -> `INSUFFICIENT` por diseno cuando `function_lag > 0`. El run se disparo a las 00:40 UTC (antes del cron habitual 04:00 UTC). A esa hora, los proveedores no habian propagado el cierre del 16/09. La FSM reporta fielmente la antiguedad real.
- **Verificacion pendiente:** observar el proximo cron `0 4 * * *` (2026-09-17 04:00 UTC) sin intervencion.
  - Si el cron resuelve contratos como `OK`/`STALE` con lags <= max_lag -> CIERRE PASIVO. La FSM funciona. Run fuera de ventana no es bug.
  - Si el cron resuelve `INSUFFICIENT` -> ABRIR como bug formal: algo impide que `EQUITY_EOD` cierre con `max_lag=0` en runs normales.
- **Prioridad:** MEDIA (no bloquea Gate; el sistema opera con 10/10 comprobaciones OK).
- **Alcance:** observacion. Sin cambios de codigo hasta tener evidencia del cron.
- **Clasificacion:** hallazgo colateral post-K-DATA-LOADER-01. ABIERTA 2026-09-17, en observacion.

## DT1 - Refactor regimes/sector_regime.py (CERRADO)

- **Origen:** deuda tecnica ALTA. Fichero de 827 LOC.
- **Dictamen auditor (revisado):** C-GO con ajustes. El diagnostico previo ("refactor arquitectonico") se descarto tras Gate 0: 827 lineas brutas = 156 lineas de codigo real, 663 vacias, 8 comentarios. La "deuda ALTA por LOC" era un artefacto de medicion.
- **Hallazgo real (bug latente):** en `compute_sector_scores`, el bloque `components` (lineas 443-518) construia el dict FUERA del bucle principal, copiando los valores de `comp_rs20`, `comp_rs50`, etc. del ULTIMO sector iterado (XLC) a los 11 sectores. Reproducido con datos sinteticos: los 11 `components[sector]` eran identicos.
- **Evidencia de bug:** `tests/fixtures/_diag_components.py` (desechado) confirmo `*** BUG CONFIRMADO: todos los components son identicos ***`.
- **Cero consumidores:** grep en todo el repo. `fls.py` y `mte_confirmation.py` usan `fls_data['components']` (contador int de componentes FLS, contexto distinto). Nadie lee `sector_results['components']`.
- **Decision:** extirpar, no reparar (evita mantener codigo sin consumidor).
- **Fases ejecutadas:**
  - Fase 0: golden de caracterizacion (`tests/fixtures/sector_regime_golden.json`) + 5 tests. Congelan `ranking`, `regime`, `top3` como contrato, `last_scores` como diagnostico. Generador reproducible: `tests/fixtures/_gen_sector_regime_golden.py`.
  - Fase 1: eliminado bloque `components` + quitado del return. Tests verdes.
  - Fase 2: colapso de whitespace (827 -> 272 lineas). AST identico antes/despues (verificado con `ast.dump(include_attributes=False)`). BOM y LF preservados.
  - Fase 3: 11 tests edge cases (`tests/test_sector_regime_edge_cases.py`).
- **Commits:** `a1406f4`.
- **Verificacion local:** 571 passed + 2 skipped. pyflakes limpio. compileall OK. E2E `py run.py`: Gate 10/10, `NARROW RALLY`, 10 contratos resueltos.
- **Verificacion CI (run 35170230926):** exit 0. Los 16 tests DT1 verde en CI.
- **No toca:** scoring, `ranking`, `regime`, `last_scores`, `top3`, `compute_price_flow_rankings`, contratos temporales, reporte.
- **Clasificacion:** ciclo de fix + limpieza + cobertura. CERRADO 2026-09-17.
- **Deudas generadas:** ninguna nueva.

## K-CI-CRON-01 - Contratos STALE/INSUFFICIENT en run fuera de cron (CERRADO)

- **Cierre pasivo verificado.** Evidencia del cron real `0 4 * * *`.
- **Run:** `35204004152` (event=`schedule`, 2026-09-17 09:14 UTC, headSha=`9b44845`). Conclusion: success.
- **Log verificado:**
  - `[FU-021-3A] EQUITY_EOD effective=2026-09-16 requested=2026-09-17 function_lag=1d reference_lag=1d coverage=100.00% (539/539)`.
  - `[FU-021-5] 10 contratos resueltos: EQUITY_EOD=OK INDEX_EOD_USA=OK INDEX_EOD_EUROPA=STALE INDEX_EOD_COMMODITY=OK INDEX_EOD_CURRENCY=OK VOLATILITY_INDEX=OK RATE_YIELD=OK FUTURE_SETTLEMENT=INSUFFICIENT SPOT_COMMODITY=STALE FX_DAILY_CUT=STALE`.
  - `VALIDATION GATE: Sin errores (10 comprobaciones OK)`.
- **Conclusion:** la FSM opera correctamente. `EQUITY_EOD.max_lag=0` con `function_lag=1d` -> OK. Los runs manuales de la madrugada (00:40-01:57 UTC) veian `function_lag=2d` porque los proveedores aun no habian propagado el cierre del dia anterior. Comportamiento correcto por diseno.
- **Residuales esperados:** `INDEX_EOD_EUROPA=STALE` (calendario no-NYSE), `FUTURE_SETTLEMENT=INSUFFICIENT`, `SPOT_COMMODITY=STALE`, `FX_DAILY_CUT=STALE` (providers externos con lag). No son bug.
- **Clasificacion:** falsa alarma por run fuera de ventana. CERRADO 2026-09-17.

## DT3 - Refactor indicators/darkpool.py (CERRADO)

- **Origen:** deuda tecnica BAJA. Fichero de 286 LOC, 252 codigo real, 8 funciones.
- **Dictamen auditor:** B-GO (modulos hermanos, no paquete). `datetime.now()` y bare except dentro de Fase 1.
- **Bug normativo corregido (Fase 1):** `'fecha': datetime.now().strftime('%Y-%m-%d')` violaba regla del prompt ("fecha de observacion se deriva del dataset"). Reemplazado por `'fecha': week_start` (fecha publicacion FINRA). Import `datetime` eliminado.
- **Bare except corregido:** `except:` en `pd.read_csv(hist)` sustituido por `except (FileNotFoundError, pd.errors.EmptyDataError)`.
- **Bug latente corregido (Fase 5a):** `_get_all_tickers` asumia `str` en filtro `not t.startswith('^')`, rompia con NaN de celdas vacias en `etf_holdings.csv`. `isinstance(t, str)` ahora precede al filtro.
- **Fases ejecutadas:**
  - Fase 0: golden de caracterizacion (`tests/fixtures/darkpool_golden.json`) + 12 tests. Congela contrato observable (media_dark_pool, n_tickers_ats/total, z_score, week, status) + diagnostico (state, momentum, percentile, z_windows). Setup reproducible con FINRA mockeada (`_darkpool_setup.py`). Generador: `_gen_darkpool_golden.py`.
  - Fase 1: fix fecha + bare except. `test_fecha_actual_es_now` -> `test_fecha_es_week_start`.
  - Fase 2: extraer `robust_zscore`, `rolling_percentile`, `classify_darkpool`, `_compute_z_for_window` a `indicators/darkpool_scoring.py`. Re-export + `__all__` en `darkpool.py`.
  - Fase 3: extraer `_get_all_tickers`, `_get_volume_from_df` a `indicators/darkpool_io.py`. Re-export + `__all__`.
  - Fase 4: extraer `_backfill_history` a `indicators/darkpool_history.py`. Re-export + `__all__`. Imports `yfinance` y `safe_mean` eliminados del orquestador.
  - Fase 5: 18 tests edge cases (`tests/test_darkpool_edge_cases.py`). Cubre robust_zscore (mad=0, outlier, vacio), rolling_percentile, classify_darkpool extremos, _get_all_tickers formato invalido, _get_volume_from_df, _compute_z_for_window, verificacion de identidad de re-exports.
- **Arquitectura final:**
  - `indicators/darkpool.py` (147 LOC): orquestador + API publica + re-exports.
  - `indicators/darkpool_scoring.py`: 4 funciones puras (z-score, percentil, classify, ventana).
  - `indicators/darkpool_io.py`: 2 funciones (tickers, volumenes).
  - `indicators/darkpool_history.py`: backfill FINRA+Yahoo.
- **Commits:** `0ccc603` (F0+F1+F2), `06f3b27` (F3), `5aca394` (F4), `5ec1e21` (F5 + fix robustez).
- **Verificacion local:** 601 passed + 2 skipped. pyflakes limpio. compileall OK.
- **Verificacion CI:** runs `35171844806`, `35172738442`, `35173591293`, `35231715178`. Todos exit 0.
- **No toca:** schema `outputs/history/darkpool_history.csv`, `indicators/mte/` (consume `z_score`), `src/pipeline/market_data.py` (consume dict), algoritmo de scoring.
- **Deudas generadas (separadas):**
  - `K-DT3-YF-DIRECTO`: `_backfill_history` usa `yf.download` directo, fuera del router de providers. Cuestion arquitectonica de procedencia de datos, no de refactor.
  - `K-DT3-SIDE-EFFECT`: escritura directa de `outputs/history/darkpool_history.csv` sin `append_dedup` ni manifest FU-002.
  - `K-DT3-RUNTIMEWARN`: `robust_zscore` sobre serie vacia emite `RuntimeWarning` de numpy. Deuda menor.
- **Clasificacion:** ciclo de refactor modular + 2 fixes (1 normativo, 1 robustez). CERRADO 2026-09-17.

## FU-021-3C-bis-05 - Transfer doc original (OBSOLETO)

- **Origen:** prompt maestro v6.24 seccion 13.
- **Descripcion:** transfer doc original de FU-021-3C-bis marcado como desactualizado.
- **Gate 0 (2026-09-17):** busqueda recursiva `*TRANSFER*` / `*transfer*` en el repo (excluyendo `.git/`) retorna 0 ficheros. El transfer doc nunca fue commiteado; fue un artefacto de chat de la sesion 2026-09-16.
- **Estado:** **OBSOLETO 2026-09-17**. No hay fichero que actualizar.

## FU-021-3C-bis-07 - Informe formal FU-021-3C-bis (WONT FIX razonado)

- **Origen:** prompt maestro v6.24 seccion 13.
- **Descripcion:** informe formal del ciclo FU-021-3C-bis en `docs/auditoria/`.
- **Gate 0 (2026-09-17):** `FU-021-3C_INFORME_FUTURE_FX.md` (16/09/2026) documenta la fase previa BLOCKED; no cubre la resolucion. No existe informe standalone posterior.
- **Decision:** WONT FIX razonado. El ciclo esta documentado en `PROMPT_MAESTRO.md` seccion 11.16 + 15.1 (commits + verificaciones + presupuesto API). Un informe estilo `E5_INFORME_VIX3M.md` (~1h) no aporta informacion diferencial.
- **Reabrir si:** (a) auditoria externa lo requiere, (b) H1 (mutabilidad OilPriceAPI) escala a decision arquitectonica y necesita vehiculo documental.
- **Estado:** **WONT FIX 2026-09-17**.

## E1 - USDJPY=X diff 0.35% vs Yahoo (OBSOLETO)

- **Origen:** prompt maestro v6.24 seccion 13, heredado sin definicion documental.
- **Gate 0 (2026-09-17):** busqueda recursiva en todo el repo (excluyendo .git/archive/tests/outputs) de `\bE1\b` vinculado a USDJPY: 0 hits. Sin definicion operativa.
- **Verificacion empirica:** `market_data.parquet` (MultiIndex `(Price, Ticker)`) contiene `USDJPY=X`. Comparativa vs yfinance fresco:
  - 2026-09-10 a 2026-09-15: diff = 0.000% (identicos).
  - 2026-09-16: parquet=156.1880, yahoo=155.2660, diff=+0.594%.
- **Causa:** mutabilidad del proveedor Yahoo. Mismo patron que H1 (OilPriceAPI reviso CL=F 09-16). No es bug de pipeline.
- **Estado:** **OBSOLETO 2026-09-17**. Registrado como mutabilidad de proveedor, no como deuda.

## E2 - Causa raiz H-3C-7 (futuros Yahoo no reproducibles) (OBSOLETO)

- **Origen:** prompt maestro v6.24 seccion 13. Etiqueta agregada del hallazgo H-3C-7.
- **Gate 0 (2026-09-17):** H-3C-7 cerrado por ROI negativo. `FOLLOWUPS.md` L322: "Cierre sin causa raiz exacta: H-3C-7 no tiene causa raiz reproducible. Investigacion detenida por ROI negativo". `FU-021-3C_INFORME_FUTURE_FX.md` L21 + L323 confirman el cierre.
- **Causa subrogada:** futuros de commodities migrados a OilPriceAPI (FU-021-3C-bis). La rama Yahoo futuros ya no es fuente del sistema.
- **Estado:** **OBSOLETO 2026-09-17** (subrogado por cierre de H-3C-7).

## E3 - Cutoff exacto FX (17:00 ET provisional) (OBSOLETO)

- **Origen:** prompt maestro v6.24 seccion 13.
- **Gate 0 (2026-09-17):** el cutoff FX esta codificado y NO marcado como provisional:
  - `src/temporal_contracts/fx_daily_cut.py:3`: `Cutoff 17:00 ET. Lag heterogeneo por par.`
  - `src/temporal_contracts/registry.py:109`: `per_pair_max_lag={"EURUSD=X":0, "USDJPY=X":0, "USDCNY=X":1}`.
  - `src/temporal_contracts/_common.py:75`: `Fecha esperada FX: reference_date (cutoff 17:00 ET).`
- **Comparacion:** LSE/Xetra SI llevan marca "PROVISIONAL" en `market_hours.py` para calendarios; FX no la lleva. Cutoff 17:00 ET es definitivo por diseno.
- **Estado:** **OBSOLETO 2026-09-17**. No hay cutoff "provisional" que revisar.

## E4 - Persistencia df.attrs tras Q-P.3 (OBSOLETO)

- **Origen:** prompt maestro v6.24 seccion 13.
- **Gate 0 (2026-09-17):** el ciclo `.attrs` esta vivo y es coherente con Q-P.3:
  - Escritor: `src/data_loader.py:208` -> `data.attrs['temporal_meta'] = _meta`.
  - Lector: `src/pipeline/data_load.py:57` -> `df_market.attrs.get('temporal_meta', {})`.
  - Dictamen Q-P.3: `temporal_meta` (dict paralelo) es autoridad; `.attrs` es espejo auxiliar.
- **Verificacion empirica:** parquet no persiste `.attrs` entre procesos (verificado con `pd.read_parquet`). El ciclo es intra-run, correcto por diseno.
- **Estado:** **OBSOLETO 2026-09-17**. No hay problema de persistencia; el comportamiento observado es el correcto.

## H1 - Mutabilidad del dataset historico (CERRADO 2026-09-17 - WONT FIX / POLITICA ACEPTADA)

- **Origen:** hallazgo detectado durante K-FS-CI-PARITY-01 (2026-09-17). Sin K-ID previo.
- **Descripcion:** el dataset historico puede variar entre runs por tres mecanismos independientes: (a) proveedor externo revisa valores (OilPriceAPI, Yahoo); (b) pipeline regenera historicos por cambio de logica; (c) `append_dedup(keep='last')` sustituye observaciones sin registrar la revision.
- **Informe:** `docs/auditoria/H1_INFORME_MUTABILIDAD_HISTORICA.md` (commit `1d5c6c2`, 295 lineas).
- **Dictamen:** `docs/auditoria/H1_DICTAMEN_AUDITOR.md` (2026-09-17).
- **Decision del auditor:** C + D como direccion arquitectonica. A (snapshot) y B (congelacion) NO-GO hasta que exista requisito adicional.
- **Politica adoptada:** el dataset operativo es mutable por diseno; los outputs publicados se versionan; las revisiones de datos no constituyen por si mismas un error.
- **No toca codigo de produccion.** No se abre ciclo de implementacion.
- **Reabrir si:** (a) requisito regulatorio/compliance; (b) reconstruccion exacta de inputs exigida; (c) auditoria externa necesita verificar dataset completo de fecha pasada; (d) necesidad de distinguir automaticamente revision de proveedor vs regeneracion de pipeline.
- **Estado:** **CERRADO 2026-09-17**.

## DT4 - Reorganizacion de validation/ y scripts/ (CERRADO 2026-09-17 - WONT FIX razonado)

- **Origen:** prompt maestro v6.26 seccion 13.
- **Descripcion:** reorganizar las carpetas validation/ y scripts/ por subcategorias.
- **Gate 0 (2026-09-17):**
  - validation/: 8 ficheros activos en raiz + archive/ con 120 entradas historicas.
  - scripts/: 16 ficheros activos en raiz + archive/ con 8 entradas historicas.
  - Referencias en CI: 6 en daily_run.yml para validation/, 9 en workflows para scripts/.
  - Referencias en tests: 1 (tests/test_update_futures_skip.py:14 -> from scripts.update_futures import _inspect_parquet).
  - Referencias en codigo productivo (src/, run.py): 0.
- **Diagnostico:** estructura actual funcional y ordenada. archive/ ya separa historico. 24 ficheros activos totales son manejables sin subcategorias. Beneficio funcional de reorganizar: cero. Coste: 2h + riesgo de romper CI por ruta olvidada.
- **Decision:** WONT FIX razonado. Coste > beneficio.
- **Reabrir si:** (a) >40 scripts activos en cualquiera de las dos carpetas; (b) nueva necesidad de subcategorizacion por dominio; (c) solicitud de auditoria externa por compliance.
- **Estado:** **CERRADO 2026-09-17**.

## U+FFFD - Encoding residual en codigo productivo (RESUELTO 2026-09-17)

- **Origen:** prompt maestro v6.27 seccion 13. Aislado por decision del auditor.
- **Gate 0 (2026-09-17):**
  - 26 U+FFFD + 1 CP1252 em-dash en 4 ficheros productivos: indicators/mte/scoring.py (20), indicators/mte/decision.py (5), indicators/mte/engine.py (1), config/settings.py (1).
  - 23 ocurrencias adicionales en data/cache/qqq_model_embedded.json (gitignored, fuera de alcance).
  - Sin BOM, todo LF puro.
- **Verificacion historica (A):** barrido de los 8 commits que tocaron indicators/mte.py. El commit mas antiguo del repo (533d580, v3.15, 25/07/2026) ya tenia los 28 U+FFFD. No existe version sana en el historial. Verificacion agotada.
- **Fix (B):** reconstruccion por contexto linguistico (palabras 100% inferibles: ultimo, Dispersion, esta, VALIDOS, Credito, algun, INDICES, PUNTUACION, condicion, rotacion, inflacion, recesion, estanflacion, maxima, invalido, estres, subito, transicion, Histeresis, FUNCION, indices). Em-dash CP1252 -> U+2014 (coherente con L10/L80/L102 de settings.py).
- **Precision del auditor:** sin impacto esperado sobre la semantica ejecutable (comentarios/docstrings/reasons). No usar "cero riesgo" absoluto.
- **Commit:** d5b1c25 - fix(encoding): 27 ocurrencias corregidas.
- **Tests:** 615 passed + 2 skipped, 0 warnings.
- **Estado:** **RESUELTO 2026-09-17**.

## K-HUERFANO - Deteccion de lote parcial en download_market_data (RESUELTO 2026-09-17)

- **Origen:** verificacion de fechas por fuente (Gate 0, 2026-09-17).
- **Descripcion:** `download_market_data` acepta un lote como OK si `data_batch is not None and not data_batch.empty`. No verifica cobertura por ticker individual. Un lote con 4/5 tickers completos pasa sin retry.
- **Caso medido:** run manual 17/09/2026 11:15 ET. 1 de 562 tickers (`KHC`) sin Close en `expected_session=2026-09-16`, aunque el resto del lote si fue aceptado. Yahoo fresco si devolvia la vela (Volume=13.87M).
- **Fix (`3ccae42`):** helper `_check_khuerfano` + resolucion `_expected_session` + print `[K-HUERFANO]` tras `all_data.append`. Sin retry. +6 tests.
- **Dictamen auditor:** B (warning) GO + C (documentacion) GO. A (retry parcial) NO-GO ahora. D (fix completo + retry) NO-GO ahora. E (cerrar sin accion) NO-GO.
- **No crear K-ID nuevo.** Monitorizacion activa.
- **Reabrir ciclo A/D si:** (a) mismo ticker repetidamente; (b) multiples tickers; (c) produccion 04:00 UTC; (d) cobertura materialmente inferior.
- **Distinto de:** `FUTURE_SETTLEMENT=INSUFFICIENT` (estado contractual valido del temporal). Aqui es cobertura del loader.
- **Estado:** **RESUELTO 2026-09-17** (deteccion anadida, retry no implementado, monitorizacion activa).

## Ciclo de auditoria del reporte 2026-09-17 (2026-09-18)

Referencia: prompt maestro v6.30, secciones 11.17 a 11.19 y 15.21. Nueve fixes aplicados + 3 NO BUG + 3 WONT FIX/MONITORED. Detalle completo en el prompt; este listado es indice resumido.

### O2 - append_dedup colapsaba tickers en rs_internal (RESUELTO 2026-09-18)

- **Commit:** `9ff8daf`.
- **Causa:** subset `["date","sector"]` colapsaba ~500 tickers a 11 filas/dia.
- **Fix:** subset `["date","sector","ticker"]`.
- **Historico:** `outputs/history/rs_internal.csv` (88 filas) borrado y regenerado desde cero.
- **Tests:** 3 (`tests/test_rs_internal_dedup.py`).

### K-RS-INTERNAL-COMMIT-01 - `git add -u` no captura untracked (RESUELTO 2026-09-18)

- **Commit:** `5fbe16e`.
- **Causa:** step `Commit and push hist/state` usaba `git add -u` + lista explicita de `data/*`. Tras borrar el CSV en `9ff8daf`, el CSV regenerado por el pipeline quedaba untracked y nunca se commiteaba.
- **Fix:** `git add outputs/history outputs/state` + safety net que falla el workflow si queda untracked en esas carpetas.
- **Tests:** los del ciclo (verificado en CI).

### I2 - wyckoff_structure_core degenera a RANGE con NaN (RESUELTO 2026-09-18)

- **Commit:** `a6f4666`.
- **Callers afectados:** `indicators/sector_wyckoff_distribution.py:36`, `indicators/index_leaders.py:57-75`.
- **Fix:** helper `build_ticker_df(df, ticker)` + sustitucion en ambos callers.
- **Tests:** 5 (`tests/test_build_ticker_df_i2.py`).

### I3 - as_of_date vs effectiveDate en Rendimiento QQQ (RESUELTO 2026-09-18)

- **Commit:** `1d7bacd`.
- **Causa:** `flows_international.py:149` mostraba `as_of_date` (timestamp del run, con hora).
- **Fix:** priorizar `effectiveDate` (fecha del dataset). Fallback a `as_of_date` truncado.
- **Tests:** 4 (`tests/test_i3_effective_date_render.py`).

### I1 + O1 - Notas aclaratorias + fecha efectiva SSGA (RESUELTO 2026-09-18)

- **Commit:** `b74790c`.
- **I1:** dos tablas usan `Retorno 20d` con significados distintos. Nota en `render_momentum_sectores` y `render_tactical_leaders`.
- **O1:** nota fuente SSGA anade `Ultima fecha: YYYY-MM-DD`. Fallback `N/D` funcional (df de SSGA no lleva `Date`). WONT FIX / MONITORED en §12 del prompt v6.30.
- **Tests:** 5 (`tests/test_i1_o1_notas.py`).

### I4 - Evidence Matrix lag=1 (NO BUG)

- **Verificado:** no reproducible en el run del 17/09.
- **Causa historica del lag 16/09:** `evidence_matrix.py:170` hereda fecha de `sector_breadth_df`. Comportamiento correcto.

### O3 - FEZ primary_flow=0.00 (NO BUG)

- **Verificado:** `shares_outstanding` invariante en `etf_primary_flow.csv`. Comportamiento correcto.

### K-INDEX-RANGE-01 - Mismo patron I2 en index_phase y sector_regime (RESUELTO 2026-09-18)

- **Commit:** `562dc41`.
- **Callers afectados:** `indicators/index_phase.py:19,33` (4/8 indices), `regimes/sector_regime.py:114` (7/11 ETFs).
- **Fix:** `build_ticker_df` en los 3 callers.
- **Efecto colateral:** **SLPM desbloqueado**. LIS=+0.20, Eff Breadth=0.53, 20 tickers en *Acciones Seleccionadas*.
- **Tests:** 4 (`tests/test_index_phase_range.py`).

### PCR Indices N/D - index_pcr ausente del return dict (RESUELTO 2026-09-18)

- **Commit:** `849f981`.
- **Causa:** `compute_pcr_signals()` usaba `data["index_pcr"]` internamente pero no lo incluia en el return.
- **Fix:** 1 linea.
- **Tests:** 2 (`tests/test_pcr_indices.py`).

### K-RENDER-LEADER-TABLE-01 - Separator Markdown desalineado (RESUELTO 2026-09-18)

- **Commit:** `44cd544` (rebase final `29f393a`).
- **Causa:** separator 8 cols vs header 11 cols en `indicators/stock_leader.py:180`.
- **Fix:** separator con 11 grupos.
- **Tests:** 1 (`tests/test_leader_table_markdown.py`).

### K-RUN-OUT-OF-WINDOW-01 - Run manual fuera de ventana EOD (MONITORED 2026-09-18)

- **Incidente:** run manual 18/09 00:00 UTC. `coverage_pct=0.16` (262/313 MISSING_CLOSE).
- **Comportamiento pipeline:** correcto (FU-020 retrocedio `effective_date` a `2026-09-16`).
- **Fallo:** step `Commit and push hist/state` no tiene guarda para `coverage_pct < umbral`.
- **Accion:** commit `fc8faed` (revert) + push.
- **Reabrir si:** el cron real produce `coverage_pct < 0.9`. Fix candidato (BAJA): guarda en el workflow.

### VIX3M/VIX nan 2026-09-14 (WONT FIX 2026-09-18)

- **Verificado:** `cboe_vix3m.parquet` tiene `19.28` sin NaN. El `nan` es un artefacto historico del CSV `volatility_structure.csv`, propagado por `append_dedup`. No reproducible en local.
- **Accion:** sin fix.

### O1 SPDR `Ultima fecha: N/D` (MONITORED 2026-09-18)

- **Verificado:** el df que llega desde `flows_primary` no lleva columna `Date`. Fallback `N/D` funcional.
- **Accion:** sin fix. La fecha real esta en `Calidad, frescura y cobertura de datos`.


## IAE FA-1 - Ingestion SEC 13F cerrada y pusheada (RESUELTO 2026-09-19)

- **Origen:** Fase A del modulo IAE. Autorizada por dictamen Gate 0 (2026-09-19) como GO condicionado dividido en FA-1 + FA-2.
- **Alcance FA-1:** ingestion + schema + lineage del dataset SEC 13F.
- **Artefactos:**
  - `src/institutional_accumulation/sec_13f/` (7 ficheros: schema, downloader, parser, storage, manifest, ingest + `__init__`).
  - 72 tests IAE (downloader 15, schema 13, parser 15, storage 10, manifest 11, ingest 8).
- **Commits pusheados (7, HEAD 626b39d):**
  - `0017011` FA-1.1 downloader + schema.
  - `0568992` FA-1.2 parser.
  - `ee8809f` FA-1.3 storage + manifest.
  - `0c587f3` FA-1.4 ingest orquestador.
  - `52d098b` fixes tecnicos (pyflakes + docstring + DATE_COLUMNS + removesuffix).
  - `387e993` addendum al contrato v1.1 (D1-D5).
  - `626b39d` informe Gate FA-1.
- **Probe real (FA-1.5):** `ingest_13f` contra 3,822,885 filas del ZIP Q1 2026 (SHA-256 `05f4da8f526cd471...`). 7/7 row counts coinciden con Gate 0. Manifest v1 con 3 niveles hash (ZIP + TSV + Parquet). 3 flags `validation` a `True`.
- **Desviaciones documentadas (addendum D1-D5):** 7 parquets en `processed/{quarter}/`, manifest en `data/sec_13f/manifests/`, cadena de 3 niveles hash, nombres de campo, size-check diferido a CRC.
- **Fuera de FA-1 (FA-2):** Q2 (framework 3 niveles), filtro `PERIODOFREPORT`, CUSIP->ticker, DFND, amendments. NIPC permanece BLOQUEADO hasta Gate FA-2.
- **Estado:** **RESUELTO 2026-09-19**.
- **Reabrir ciclo:** apertura de FA-2 con Gate FA-2 al cierre.


## IAE FA-2 - SEC 13F core cerrada (RESUELTO 2026-09-19)

- **Origen:** continuacion de FA-1 dentro de la Fase A del modulo IAE.
- **Alcance FA-2:** filtro temporal + CUSIP resolver + reporting relationships + amendments.
- **Commits del ciclo (local, sin push al cierre del informe):**
  - `1679639` + `98b7956` + `1971cf5`: FA-2.1 filtro temporal (13 tests).
  - `4560ae3`: FA-2.2 CUSIP resolver (17 tests).
  - `1e42aaa` + `f7882f5` + `c0ba083`: FA-2.3 reporting relationships (FK `OTHERMANAGER2.SEQUENCENUMBER`, 33 tests).
  - `ae43985`: FA-2.4 canonical snapshot composicional (30 tests).
  - `5162cb7` a `9543de5`: informes, dictamenes, especificacion, addendum.
- **Pipeline verificado end-to-end (Q1 2026):**
  - 10,776 filings filtrados; 3,321,967 filas INFOTABLE.
  - Snapshot canonico: 8,762 accessions aplicados; 3,239,273 source lines canonicas.
  - Estrategias (10,648 grupos): SINGLE_HR 8,618; SINGLE_NOTICE 1,906;
    HR_PLUS_RESTATEMENT 100; HR_PLUS_NEW_HOLDINGS 19; HR_CHAIN_RESTATEMENT 2;
    HR_COMPOSITE 2; NOTICE_AMENDED 1.
  - Edges 3,509,475; resolved 1,351,172; resolution_rate 0.9883.
  - `duplicate_canonical_edges = 0`; `invalid_source_line_edges = 0`.
  - Anomalias registradas: 2 (NT_AUGMENTED, HR_WITH_AMENDMENT_FLAGS).
- **Dictamenes emitidos:**
  - `INSTITUTIONAL_ACCUMULATION_FA23_DICTAMEN.md` (hallazgo FK, caracterizacion C).
  - `INSTITUTIONAL_ACCUMULATION_FA23_DICTAMEN_CIERRE.md` (cierre FA-2.3).
  - `INSTITUTIONAL_ACCUMULATION_FA24_DICTAMEN_GATE0.md` (amendments).
  - `INSTITUTIONAL_ACCUMULATION_GATE_FA2_DICTAMEN.md` (PASS, push autorizado).
- **Estado:** FA-2 CERRADA / PASS 2026-09-19. Push autorizado del ciclo (20 commits).
- **Deuda posterior (fuera de FA-2):**
  - Curacion manual del crosswalk CUSIP -> ticker (tabla vacia en FA-2).
  - NIPC: desbloqueado respecto de FA-2; pendiente su propio Gate.
  - Observaciones no bloqueantes de FA-2.3: 922 CIK-like, patron pico 25 edges.
- **Reabrir ciclo:** NIPC, Breadth, New/Exit, clasificacion (post Gate FA-2);
  curacion CUSIP como ciclo paralelo.


## IAE CUSIP curation - Q1 2026 (RESUELTO 2026-09-19)

- **Origen:** deuda posterior declarada por Gate FA-2 (INSTITUTIONAL_ACCUMULATION_GATE_FA2_DICTAMEN.md, P-GATE.2).
- **Alcance:** poblar data/mappings/cusip_ticker_exceptions.csv para el universo 13F Q1 2026 (PERIODOFREPORT=2026-03-31).
- **Gate 0 empirico:**
  - Probe Q1 2026 (D:\13f_probe\processed\2026Q1\INFOTABLE.parquet, 3,822,885 filas).
  - etf_holdings.csv (526 filas; columnas etf/ticker/identifier/weight).
  - Los 5 casos (DD, HON, XOM, FDXF, HONA) se dividen en 2 grupos: discrepancia temporal / corporate-action (DD, HON, XOM) y entidades post-Q1 (HONA, FDXF).
  - Correccion al informe Gate 0: la afirmacion "CUSIP local desactualizado vs CUSIP 13F vigente" no se sostiene para XOM (30233Q108 no observado en 13F Q1 2026; no se presume variante historica de 30231G102 sin evidencia adicional).
- **Dictamenes aplicados (Q-CUR-1 / Q-CUR-2):**
  - CALL/PUT (26614N902, 26614N952, 438516906, 438516956, 30231G902, 30231G952) excluidos del crosswalk de equity. Derivados preservados en el 13F; mapping subyacente en modelo separado (no implementado en este ciclo).
  - HONA / FDXF excluidos del CSV Q1 2026 (posteriores a PERIODOFREPORT). Documentar como FUTURE_CORPORATE_ACTION / POST_Q1_ENTITY.
- **Alcance minimo ejecutado: 3 filas COM.**
  - 26614N102 -> DD.
  - 438516106 -> HON (reverse split 2:1 efectivo 2026-06-29 documentado en reason; no se introduce la frontera en el rango).
  - 30231G102 -> XOM.
  - valid_from=valid_to=2026-03-31 (no se inventan fechas historicas no auditadas).
  - source=SEC-EDGAR, verified_by=manual, title_of_class=COM.
- **Commits del ciclo:**
  - e4a6576 informe INSTITUTIONAL_ACCUMULATION_CUSIP_CURATION_INFORME.md.
  - e8d7e53 feat populate CSV (3 COM).
- **Mini-gate verificado:** resolve_cusip devuelve DD/HON/XOM para 2026-03-31. 820 passed + 2 skipped. pyflakes limpio tras limpieza de _probe_*.py (13 ficheros untracked de ciclos anteriores).
- **Estado:** RESUELTO 2026-09-19.
- **Deuda posterior:**
  - Ampliar valid_from/valid_to cuando exista evidencia historica suficiente.
  - Poblar HONA / FDXF cuando aparezcan en un 13F posterior con evidencia primaria.
  - Modelo separado para mapping de derivados CALL/PUT -> subyacente (no implementado).
- **Reabrir ciclo:** ampliacion temporal del crosswalk; NIPC (desbloqueado respecto de FA-2, pendiente propio Gate).
