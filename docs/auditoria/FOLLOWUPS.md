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

## FU-003 — Cosmetico: signos +0.00 y flechas ->

- **Origen:** prompt maestro v6.8 seccion 15.2.
- **Descripcion:** formato de +0.00 en scores pequenos; flecha -> en algunos textos.
- **Clasificacion:** P3 cosmetico.
- **Bloqueante:** no.

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

## FU-016 — Desfase 1 dia entre writers cuando run < PUBLISH_HOUR (documental)

- **Origen:** informe de implementaciones pendientes, 2026-09-15.
- **Descripcion:** cuando un run se ejecuta antes de PUBLISH_HOUR=23 (hora Madrid), `last_expected_market_date()` retrocede al ultimo dia con cierre publicado. Los writers derivan su fecha de fuentes distintas: `macro_regime` de `df_macro_manual['date'].max()` (FRED, publicado antes de las 23); `sector_breadth` de `df_stocks.index[-1]` (Yahoo, sesion con cierre ya publicado). Resultado: las dos tablas quedan desfasadas entre si.
- **Evidencia:** run del 2026-09-14 23:00. `macro_regime.csv` -> max 2026-09-14 (40 filas). `sector_breadth.csv` -> max 2026-09-11 (979 filas). Log: `[YAHOO_RAW] expected_session=2026-09-11 raw_last_date=2026-09-14`.
- **Impacto:** observabilidad, no funcionalidad. Un usuario que compare CSVs puede pensar que hay un bug donde no lo hay.
- **Clasificacion:** P3 documental.
- **Accion:** ninguna. Documentado para evitar falsos positivos en futuras auditorias de integridad.
- **Bloqueante:** no.

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
