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


## IAE NIPC - Gate 0 (PASS condicionado 2026-09-19)

- **Origen:** deuda posterior de Gate FA-2 (P-GATE.5: NIPC desbloqueado respecto de FA-2).
- **Informe previo:** docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_INFORME.md.
- **Dictamen:** docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_DICTAMEN.md.
- **Resultado:** GATE-NIPC.0 = PASS condicionado. GO para Gate-NIPC.1 (especificacion, SIN codigo). Implementacion NO autorizada. Push NO.
- **Baseline:** Q4 2025 ingestado en este ciclo (D:/13f_probe/processed/2025Q4/, ZIP SHA-256 ff340fc5...). Q1 2026 pre-existente.
- **Volumetria (filtro SH + null):**
  - Q4 2025: 10,676 filings | 3,124,594 filas canonicas | 24,200 CUSIPs.
  - Q1 2026: 10,776 filings | 3,188,083 filas canonicas | 24,838 CUSIPs.
  - Interseccion CUSIP: 21,356 (88.2% de Q4).
- **Resolucion de las 7 preguntas:**
  - Q-NIPC-1 (Q4 baseline): GO.
  - Q-NIPC-2 (unidad): GO condicionado. Entidad nueva `reported_position_unit = (report_period, filing_manager_cik, canonical_security, discretion_type)`. `SSHPRNAMT` entra UNA VEZ por source line. Prohibido N x SSHPRNAMT. `canonical_reporting_relationship_key` = dimension de provenance/evidencia, no unidad de suma.
  - Q-NIPC-3 (discretion): SOLE + DFND + OTR. Conservar separacion: NIPC_total/SOLE/DFND/OTR. DFND no es posicion invalida.
  - Q-NIPC-4 (universo): operativo = radar_equities ^ 13F_eligible ^ mapped. Universo tecnico separado para diagnostico. 13F completo NO como output operativo. 3 CUSIPs NO como output operativo.
  - Q-NIPC-5 (coverage): estado actual INSUFFICIENT (3/24,838 = 0.012%). Fuente externa pendiente de especificacion en Gate-NIPC.1.
  - Q-NIPC-6 (scope): minimo (DeltaShares + NIPC + coverage + status). Sin Breadth/NewExit/clasificacion.
  - Q-NIPC-7 (arquitectura): src/institutional_accumulation/aggregation/ con delta_shares.py + nipc.py. Sin breadth.py ni new_exit.py todavia.
- **Regla canonica DeltaShares:** SSHPRNAMTTYPE == "SH" AND PUTCALL IS NULL. PRN fuera.
- **Precision estructural:** NIPC = variacion trimestral de acciones largas reportadas en 13F. NO es "institutional trading flow". Shorts no se netean.
- **Proximo paso autorizado:** Gate-NIPC.1 (especificacion SIN codigo). Punto critico a resolver primero: capa masiva CUSIP -> security/ticker del universo radar (con 3 excepciones el NIPC productivo es INSUFFICIENT).
- **Secuencia completa:** Gate-NIPC.0 (PASS) -> Gate-NIPC.1 (spec) -> Gate-NIPC.2 (implementacion) -> Gate-NIPC.3 (validacion E2E Q4 2025 -> Q1 2026) -> Gate-NIPC.4 (cierre).
- **Regla local-first IAE activa:** no push a main hasta Gate-NIPC.3 PASS.
- **Estado:** Gate-NIPC.0 PASS condicionado 2026-09-19. Pendiente Gate-NIPC.1.


## IAE NIPC - Gate 0 Mapping (PASS 2026-09-19)

- **Origen:** continuacion de Gate-NIPC.0 (PASS condicionado). El auditor pidio medir la capa de mapping antes de la especificacion.
- **Informe previo:** docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_INFORME.md.
- **Dictamen Gate-NIPC.0:** docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_DICTAMEN.md.
- **Dictamen Gate 0 Mapping:** docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_MAPPING_DICTAMEN.md.
- **Resultado:** GATE 0 MAPPING = PASS. GO para Gate-NIPC.1 (especificacion, SIN codigo). Codigo NIPC NO autorizado todavia.
- **Fuentes internas inventariadas (data/mappings/, data/etf_holdings.csv, data/index_holdings.csv, data/amundi_yahoo_mapping.csv):**
  - cusip_ticker_exceptions.csv: 3 filas curadas (DD, HON, XOM).
  - etf_holdings.csv: 526 filas, 519 CUSIPs unicos. Crosswalk real CUSIP->ticker derivado de holdings ETF sectoriales.
  - isin_ticker_map.csv: 49 filas ISIN (no aplica USA).
  - index_holdings.csv: 2669 filas sin CUSIP (descartado).
  - amundi_yahoo_mapping.csv: ISIN Europa (no aplica USA).
  - **NO EXISTE crosswalk masivo CUSIP -> ticker del universo radar.**
- **Crosswalk efectivo:** 522 CUSIPs unicos, 0 ambiguedades (ONE_TO_ONE puro).
- **Las 4 metricas del auditor sobre Q1 2026:**
  - A) Cobertura por securities: 502/24,838 = 2.021%.
  - B) Cobertura universo radar USA: 220/242 = 90.91% (dato observado, NO umbral contractual).
  - C) Cobertura ponderada por SHARES (mapped): 39.18% (41.01% sin filas no-equity).
  - C') Cobertura ponderada SHARES en universo radar: 28.66% (29.995% sin filas no-equity). **INSUFFICIENT.**
  - D) Cobertura ponderada por VALUE (mapped): 68.49%. Solo contraste, no cambia estado.
- **Hallazgo TITLEOFCLASS:**
  - 32,812 filas canonicas (1.03%) con TITLEOFCLASS no-equity; 35.18B SSHPRNAMT (4.45% del total).
  - Casos extremos: CUSIP 329882225 (NIPST CONVERTIBLE BOND) = 10.59B shares; CUSIP 329882250 (NIPST PUT CONVERTIBLE BOND) = 6.99B shares.
  - Ruido puro (CUSIPs 100% no-equity): 1,045 CUSIPs, 3.065% shares. En crosswalk: 0. En radar: 0.
  - Contaminacion parcial en crosswalk: 10 CUSIPs (BAC, BX, JNJ, KMB, LMT, MS, NFLX, PWR, UBER, WFC), 1 fila no-equity cada uno, impacto despreciable.
- **Resolucion del dictamen sobre hallazgos:**
  - Q-NIPC-8: allowlist TITLEOFCLASS RECHAZADA. GO condicionado a clasificacion de security (`security_type = EQUITY/NON_EQUITY`). `TITLEOFCLASS` pasa a validacion/anomalia (TITLE_CLASS_CONFLICT), no autoridad primaria.
  - Umbral 90% NO aprobado como contractual (valor observado, no umbral).
  - NIPC = INSUFFICIENT hasta que Gate-NIPC.1 defina y se implemente una capa de mapping suficiente.
  - `section13f_eligible` != `security_type=EQUITY` != `ticker_mapped` != `radar_member`. Cuatro dimensiones separadas.
  - 22 tickers radar sin mapping (BRK-B, MOG-A, ...): clasificados como UNMAPPED_RADAR_SECURITY. BRK-B y MOG-A son problemas de clase/security identity, no faltantes simples.
  - Fuente externa: NO elegir todavia (OpenFIGI ni equivalente). Gate-NIPC.1 debe comparar alternativas con criterios formales.
- **Regla arquitectonica congelada:** TITLEOFCLASS -> NO allowlist equity -> security identity -> security_type (EQUITY/NON_EQUITY) -> ticker mapping -> radar intersection.
- **Proximo paso autorizado:** Gate-NIPC.1 - Especificacion SIN codigo. Debe definir 12 conceptos: security_identity, section13f_eligibility, security_type, ticker_mapping, source_precedence, temporal_validity, one_to_one/ambiguous, mapping_coverage, weighted_share_coverage, unmapped_weight, operational_universe, status. Y comparar fuentes externas antes de elegir.
- **Estado:** Gate 0 Mapping PASS 2026-09-19. Pendiente Gate-NIPC.1.


## IAE NIPC - Gate 0 FIGI (PASS 2026-09-19)

- **Origen:** continuacion del Gate 0 Mapping (PASS). El auditor pidio evaluar FIGI como puente CUSIP -> security identity antes de decidir la fuente externa.
- **Informe previo:** docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_MAPPING_DICTAMEN.md.
- **Dictamen Gate 0 FIGI:** docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_FIGI_DICTAMEN.md.
- **Resultado:** GATE 0 FIGI = PASS. FIGI DESCARTADO como pivote primario. GO para Gate 0 OpenFIGI (muestra estratificada). Codigo productivo NO autorizado.
- **Cobertura FIGI (Q1 2026):**
  - Filas canonicas con FIGI valido: 382,953 / 3,188,083 = 12.01%.
  - CUSIPs con al menos 1 FIGI: 9,449 / 24,838 = 38.04%.
  - Cobertura ponderada por SHARES: 10.98% (~89% sin FIGI).
  - Crosswalk 522 CUSIPs con FIGI: 501 (95.98%), pero solo 10.48% del SSHPRNAMT.
- **Formato observado:** 100% len=12; 99.58% prefijo BBG. 1,267 FIGIs (0.42%) con prefijo no-BBG (US0/US2/CA/IE...) son CUSIPs/ISINs mal etiquetados.
- **Ambiguedad estructural:**
  - CUSIP -> 1 FIGI: 3,167. CUSIP -> >1 FIGI: 6,282 (66%).
  - FIGI -> 1 CUSIP: 20,998. FIGI -> >1 CUSIP: 335 (1.57%).
  - Caso anomalo severo: BBG000B9XRY4 asignado a Apple + Dell + Stryker + Lamb Weston.
  - FIGI -> >1 NAMEOFISSUER: 4,921 (23.1%).
- **Consistencia intra vs cross-filing:**
  - Intra-filing (ACCESSION, CUSIP): 99.911% con 1 FIGI.
  - Cross-filing: alta ambiguedad. No es ruido aleatorio; es disparidad de versiones/proveedores entre filers.
- **Sin cobertura sobre NIPST:** los CUSIPs 329882225 y 329882250 (CONVERTIBLE BOND) no tienen FIGI. FIGI no resuelve el hallazgo TITLEOFCLASS.
- **Resolucion del dictamen:**
  - Q-FIGI-1: SI. FIGI descartado como pivote primario. Conservar como auxiliar/diagnostico (FIGI_RAW, FIGI_FORMAT_STATUS, FIGI_ANOMALY).
  - Q-FIGI-2: GO para Gate 0 OpenFIGI.
  - Q-FIGI-3: conservar 1,267 FIGIs no-BBG como anomalia clasificada. NO resolver heuristicamente.
  - Q-FIGI-4: verificacion Q4 2025 opcional, no bloqueante.
  - Q-FIGI-5: probe OpenFIGI con muestra ESTRATIFICADA (5 estratos: A=radar USA mapped, B=22 radar unmapped, C=top SHARES 13F, D=mid/low SHARES, E=clases problematicas BRK-B/MOG-A/ADR/CL). ~200-300 CUSIPs. NO solo top-100 ETF.
- **Regla arquitectonica OpenFIGI (congelada):** aunque OpenFIGI consiga 99% coverage, no sera autoridad automatica. La capa debe conservar source, mapping_method, mapping_timestamp, mapping_version, input_identifier, resolved_security, confidence/status. Conflictos -> CONFLICT explicito, no "OpenFIGI gana".
- **Clasificacion de respuestas OpenFIGI a registrar:** EXACT | NOT_FOUND | MULTIPLE_CANDIDATES | CONFLICT | NON_EQUITY | ERROR | RATE_LIMIT.
- **Proximo paso autorizado:** Gate 0 OpenFIGI con muestra estratificada. Sin codigo productivo hasta completar Gate 0 SEC 13(f) + Gate-NIPC.1.
- **Estado:** Gate 0 FIGI PASS 2026-09-19. Pendiente Gate 0 OpenFIGI.


## IAE NIPC - Gate 0 OpenFIGI (PASS como fuente candidata 2026-09-19)

- **Origen:** continuacion del Gate 0 FIGI (PASS, FIGI descartado como pivote). El auditor pidio evaluar OpenFIGI con muestra estratificada.
- **Informe previo:** docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_FIGI_DICTAMEN.md.
- **Dictamen Gate 0 OpenFIGI:** docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_OPENFIGI_DICTAMEN.md.
- **Resultado:** GATE 0 OPENFIGI = PASS como fuente CANDIDATA (no fuente unica). GO para Gate 0 SEC 13(f). Codigo NIPC NO autorizado.
- **Metodologia:** muestra estratificada de 152 CUSIPs unicos (estratos A/B/C/D/E). Endpoint /v3/mapping con idType=ID_CUSIP, exchCode=US. Batches de 5, sleep 2.5s. Tiempo: 89.5s.
- **Resultados globales:**
  - 113/152 con respuesta (74.3% observado sobre muestra).
  - 39/152 sin respuesta (29 "No identifier found", 10 "Invalid idValue format").
  - Dentro de hits: 112 EXACT_1, 1 MULTIPLE_CANDIDATES.
- **Por estrato (sobre muestra, no estimacion poblacional):**
  - A (radar mapped, ground truth): 46/48 = 95.8%. 0 mismatches vs ticker interno. Exactitud validada.
  - C (top SSHPRNAMT unmapped): 39/48 = 81.2%. Resuelve ETFs (VEA, GOVT, IVV...), ADRs (VALE, ABEV, ITUB, NOK), canadienses (CNQ, BN, BAM, ENB, SU, TD, MFC, SHOP, CVE, B), MLP (ET), y small caps USA (AUR, RKT, PLUG).
  - D (mid/low caps, percentiles 40-60): 24/50 = 48.0%. Gap critico de cobertura.
  - E (problematicos conocidos): 4/6. NIPST CONVERTIBLE BOND rechazados (Invalid idValue format). BRK-A/BRK-B y GOOG/GOOGL resueltos pero con tickers tipo BRK/A, BRK/B (barra, no guion) -> requiere normalizacion.
- **Hallazgo importante:** CUSIP 30231G102 (XOM, Exxon Mobil) NO encontrado por OpenFIGI ("No identifier found"). Gap de fuente en large cap USA con CUSIP correcto.
- **Arquitectura aprobada (precedencia):**
    1. Internal verified mapping.
    2. OpenFIGI fallback.
    3. UNRESOLVED.
  Conflicto explicito -> CONFLICT (no sobrescribir).
- **Reglas del contrato OpenFIGI (congeladas):**
  - Trazabilidad: input_identifier, id_type, resolved_figi, resolved_security, ticker, security_type, market_sector, exch_code, share_class_figi, mapping_source, mapping_timestamp_utc, mapping_endpoint, mapping_result_status. + hash/manifest del lote.
  - Temporalidad: OpenFIGI actual != verdad historica Q1. Pendiente validacion.
  - MULTIPLE_CANDIDATES -> AMBIGUOUS (no auto-resolver).
  - Los 29 NOT_FOUND -> UNRESOLVED (no heuristicas). XOM internal prevalece (SOURCE_GAP_OPENFIGI).
  - API key: GO para escalado. NO commitear. Env var/secret local. Rate limit con key: 25/6s, 100 jobs/req. Escalar 24,838 CUSIPs: ~60s con key vs ~3h19min sin key.
- **22 tickers radar sin CUSIP:** permanecen UNMAPPED_RADAR_SECURITY. Reverse lookup ticker -> CUSIP permitido solo como generacion de candidatos, no mapping canonico.
- **Distincion congelada:** OpenFIGI.FIGI (resultado de servicio externo) != 13F.FIGI (dato declarado por filer). No son el mismo problema.
- **9 criterios del dictamen:** CUSIP support/us equities/rate limits/failure modes CONFIRMADOS; temporal validity PENDIENTE; reproducibility POSIBLE (requiere snapshot); licensing a revisar (FIGI dominio publico, restricciones sobre identificadores propietarios).
- **Proximo paso autorizado:** Gate 0 SEC 13(f) con Official List Q4 2025 + Q1 2026 (listas trimestrales historicas SEC).
- **Estado:** Gate 0 OpenFIGI PASS como fuente candidata 2026-09-19. Pendiente Gate 0 SEC 13(f).


## IAE NIPC - Gate 0 SEC 13(f) format clarification (2026-09-19)

- **Origen:** continuacion del Gate 0 OpenFIGI (PASS como fuente candidata). Descarga de Official Lists SEC 13(f) Q4 2025 + Q1 2026 para medir elegibilidad normativa.
- **Informe previo:** docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_OPENFIGI_DICTAMEN.md.
- **Dictamen SEC 13(f) format clarification:** docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_SEC13F_DICTAMEN.md.
- **Resultado:** GO condicionado. Correcciones de formato aplicadas. Gate 0 SEC 13(f) continua. Codigo NIPC NO autorizado.
- **Ficheros descargados:**
  - https://www.sec.gov/files/investment/13flist2025q4.txt -> D:/13f_probe/official_list_13f/13flist_2025Q4.txt (994,842 bytes, 12,282 lineas).
  - https://www.sec.gov/files/investment/13flist2026q1.txt -> D:/13f_probe/official_list_13f/13flist_2026Q1.txt (1,995,921 bytes, 24,641 lineas).
  - Ficheros .xls/.xlsx dan 404; .txt es el formato vigente desde Q4 2025.
- **Correcciones de formato (auditor):**
  - Option Indicator esta en posicion 10 (`*` o espacio), NO sobre el CUSIP.
  - Status esta en posiciones 68-70: pos 68 = flag `*` SEC (cambio formato); pos 69 = `A` (alta) / `D` (baja); pos 70 = reservado.
  - Ultimo caracter (pos 80) NO es Status. Campo no identificado; NO interpretar.
  - Los duplicados Q1 (994 CUSIPs con 2 lineas) son eventos A/D legitimos, no errores.
- **Layout canonico confirmado (80 chars, LF):**
  - Pos 1-9: CUSIP.
  - Pos 10: Option Indicator (`*` o espacio).
  - Pos 11-40: Issuer Name.
  - Pos 41-67: Issuer Description.
  - Pos 68: Format Change Flag (`*` o espacio).
  - Pos 69: Add/Del Indicator (`A`, `D` o espacio).
  - Pos 70: reservado.
  - Pos 71-79: no documentado.
  - Pos 80: campo distinto no identificado.
- **Cifras corregidas:**
  - Q4 2025: 12,282 lineas, 12,282 CUSIPs unicos, 0 duplicados. Option Indicator: 6,300 espacio + 5,982 asterisco. Status: 1,232 flag `*`, 787 `A`, 445 `D`.
  - Q1 2026: 24,641 lineas, 22,659 CUSIPs unicos, 994 CUSIPs con par A/D (1,982 lineas duplicadas). Option Indicator: 18,627 espacio + 6,014 asterisco. Status: 1,819 flag `*`, 1,022 `A`, 797 `D`.
- **Decisiones del auditor:**
  - NO interpretar pos 80 hasta que SEC documente.
  - NO fusionar Option Indicator (pos 10) con Status (pos 68).
  - NO colapsar duplicados A/D de Q1 sin politica de resolucion.
  - NO asumir que Q1 tiene "el doble de CUSIPs" que Q4 (son 22,659 vs 12,282 unicos; el resto son eventos A/D + opciones como filas separadas).
  - NO tratar section13f_eligible como sinonimo de security_type=EQUITY.
  - Politica propuesta para resolucion A/D: por orden en el fichero (ultimo estado gana). Pendiente decision final en Gate-NIPC.1.
- **Proximo paso autorizado:** continuar Gate 0 SEC 13(f): parsear con layout canonico, cruzar con CUSIPs del 13F (SH + null), medir % cobertura + % ponderado por SSHPRNAMT + control de opciones. Sin codigo productivo.
- **Estado:** Gate 0 SEC 13(f) en progreso 2026-09-19 (formato caracterizado, pendiente analisis de cruce).


## IAE NIPC - Gate 0 SEC 13(f) CLOSED / PASS (2026-09-19)

- **Origen:** cierre del ciclo SEC 13(f) tras mini-probe correctivo Q4 2025.
- **Dictamenes del ciclo SEC 13(f):**
  - docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_SEC13F_DICTAMEN.md (formato).
  - docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_SEC13F_MINIPROBE_DICTAMEN.md (cierre PASS).
- **Resultado:** GATE 0 SEC 13(f) = PASS / CLOSED. GO para Gate-NIPC.1 (especificacion, sin codigo).
- **Hallazgo principal:** el TXT Q4 2025 publicado por SEC es un subconjunto empirico del PDF Q4 2025. Los 12,282 CUSIPs del TXT son subset exacto del PDF (0 txt-only). Los 10,064 CUSIPs pdf-only son CALL/PUT (5,030+5,024 en Q1 como referencia).
- **Terminologia adoptada:** NO usar "formato reducido" (no es etiqueta SEC). Decir: "TXT Q4 2025 publicado por SEC es, empiricamente, un subconjunto de los registros del PDF Q4 2025".
- **Correccion al informe original:** NO afirmar "Q4 usa flag *, Q1 añade CALL/PUT". Correcto: "La Official List Q4 ya usaba modelo underlying + * + CALL/PUT; el TXT Q4 publicado es subconjunto que omite filas CALL/PUT; Q1 dispone de TXT completo".
- **Layout canonico (ratificado por SEC):**
  - Pos 01-09: CUSIP.
  - Pos 10: Option Indicator (`*` o espacio).
  - Pos 11-40: Issuer Name.
  - Pos 41-67: Issuer Description.
  - Pos 68-70: STATUS (3 chars: `*A*` additions, `*D*` deletions, blank no change).
  - Pos 71-79: Blank.
  - Pos 80: Misc / Unused. NO INTERPRETAR.
- **Semantica Option Indicator `*`:** "la security tiene listed option". NO significa "esta fila es una opcion".
- **Semantica STATUS:** blank -> eligible; ADDED -> eligible; DELETED -> not eligible. CUSIP con estados incompatibles -> CONFLICT / REVIEW_REQUIRED (no "last row wins").
- **Aprobacion Opcion A:** usar TXT Q4 (subconjunto) + TXT Q1 (completo) para `section13f_eligible`. PDF Q4 conservado como evidencia auxiliar, no input operativo NIPC.
- **PDF Q4:** conservado en D:/13f_probe/official_list_13f/13flist_2025Q4.pdf como evidencia.
- **Cobertura ponderada SEC 13(f) (Q4-Q1 2025-2026):** 96.7-98.7% (SSHPRNAMT fuera lista 1.282% Q4, 3.259% Q1). Los no-listados son no-equity (CONVERTIBLE BOND, MMF, cash, PFD, extranjero).
- **Los 3 CUSIPs curados (DD, HON, XOM):** active en ambas listas. Coherente con crosswalk curado.
- **Deuda documentada (no implementada):** FUTURE_FORMAT_VARIANT - si un trimestre futuro no publica TXT suficiente, se abrira diseno especifico de PDF parser. NO implementar ahora.
- **Dimensiones congeladas para Gate-NIPC.1:**
  - section13f_eligible = CUSIP en Official List del trimestre, con resolucion por STATUS.
  - security_type = clasificacion independiente (no derivada de la lista).
  - option_status = derivado de 13F PUTCALL / instrument identity.
  - option_indicator_star = metadata: underlying has listed option. NO participa en clasificacion equity/option NIPC.
- **Arquitectura multicapa para Gate-NIPC.1:** internal verified -> OpenFIGI fallback -> UNRESOLVED, con CONFLICT explicito. + capa section13f_eligible (SEC) + capa security_type explicita.
- **Proximo paso autorizado:** Gate-NIPC.1 - especificacion SIN codigo. 12 dimensiones a definir + comparativa fuentes + snapshot/manifest + coverage thresholds.
- **Estado:** Gate 0 SEC 13(f) PASS / CLOSED 2026-09-19. Pendiente Gate-NIPC.1.


## IAE NIPC - Gate-NIPC.1 dictamen (PASS condicionado 2026-09-19)

- **Origen:** dictamen del auditor sobre la especificacion NIPC v1.0.
- **Especificacion v1.0:** docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md (commit e6f550a, 37,013 bytes).
- **Dictamen Gate-NIPC.1:** docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION_DICTAMEN.md (commit 4a7d4d0).
- **Resultado:** GATE-NIPC.1 = PASS CONDICIONADO. Arquitectura aprobada. Gate-NIPC.2 NO autorizado todavia. Revision v1.1 requerida.
- **5 correcciones obligatorias a la v1.1:**
  1. `canonical_security` explicito: identidad estable de la security, separada del `observed_security_identifier` (CUSIP del 13F). Capa obligatoria (report_period, observed_CUSIP) -> canonical_security para que cambios de CUSIP entre periodos no se traten como Exit+New.
  2. Match key interperiodo fijada en contrato: `(filing_manager_cik, canonical_security, discretion_type)` SIN `report_period`. `compute_delta_shares()` no debe aceptar `match_keys=None`. Salida con `match_status` en BOTH/NEW/EXIT/UNRESOLVED_IDENTITY.
  3. Coverage pairwise: anadir `coverage_previous`, `coverage_current`, `paired_security_coverage`, `paired_weighted_share_coverage`, `unmapped_weight_previous`, `unmapped_weight_current`.
  4. `TEMPORAL_UNVERIFIED`: politica ya en el contrato (no diferir a Gate-NIPC.2). Una resolucion OpenFIGI actual NO equivale automaticamente a mapping historico. Si no se puede demostrar valida para el report_period -> status TEMPORAL_UNVERIFIED -> fuera del operational_universe.
  5. `security_type` con `security_type_source` + `security_type_status`: RESOLVED_EQUITY / RESOLVED_NON_EQUITY / UNRESOLVED / CONFLICT. Determinante para casos SH + PUTCALL NULL + CONVERTIBLE BOND.
- **Correcciones adicionales aprobadas:**
  - `section13f_eligible` expresada como 5 estados (NOT_IN_LIST | DELETED | ADDED | ACTIVE | CONFLICT), no booleano puro. Booleano derivable.
  - Tests: ampliar de 14 a 18 familias minimas. Anadir: test_cusip_change_same_canonical_security, test_multi_edge_does_not_duplicate_delta, test_unresolved_mapping_blocks_pair_match, test_new_and_exit_zero_baseline_current.
  - Parser SEC 13(f) ubicado en `src/institutional_accumulation/sec_13f/identity/sec13f_list.py`. Responsabilidad: parse fixed-width 80 y devolver section13f_eligible + status + option_indicator + raw_line. NO resuelve ticker ni identidad.
- **NO aprobado:**
  - Threshold >=90%: sigue rechazado como umbral contractual (valor observado). Thresholds = UNDEFINED hasta fijacion previa a Gate-NIPC.2.
  - Implementar codigo antes de aplicar las 5 correcciones.
- **Razon del condicionamiento (cita literal del dictamen):** sin `canonical_security` estable y sin politica de matching interperiodo explicita, un pipeline perfectamente programado podria producir un NIPC matematicamente correcto pero semanticamente falso ante un cambio de CUSIP o una mapping gap entre Q4 y Q1.
- **Proximo paso autorizado:** revision v1.1 de la especificacion con los 5 cambios. Despues, Gate-NIPC.2 podra recibir GO.
- **Estado:** Gate-NIPC.1 PASS condicionado 2026-09-19. Pendiente v1.1 y dictamen favorable.


## IAE NIPC - Gate-NIPC.1 v1.1 dictamen + spec v1.2 + coverage policy (2026-09-19)

- **Origen:** dictamen del auditor sobre la especificacion NIPC v1.1 (PASS CONDICIONADO con 2 correcciones C1-revisada + C4-revisada).
- **Documentos del ciclo:**
  - Dictamen v1.1: docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION_V11_DICTAMEN.md
  - Spec v1.1 preservada: docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION_v1.1.md
  - Spec v1.2 activa: docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md
  - Coverage policy: docs/auditoria/NIPC_COVERAGE_POLICY.md
- **Resultado:** GATE-NIPC.1 v1.1 = PASS CONDICIONADO. C1 + C4 aplicadas en v1.2. Coverage policy creada con thresholds UNDEFINED.
- **C1-revisada (Q-ESP-V11-1):**
  - `ticker:<ticker>` PROHIBIDO como canonical_security.
  - Introducidas 2 capas de estado: `security_resolution_status` (5: CANONICAL | OBSERVED_ONLY | UNRESOLVED | AMBIGUOUS | CONFLICT) y `canonical_security_kind` (4: CANONICAL_FIGI | CANONICAL_EQUIVALENCE | OBSERVED_CUSIP_ONLY | UNRESOLVED).
  - `observed_security_key = cusip:<CUSIP>` (tecnico por periodo).
  - `canonical_security = NULL` salvo status == CANONICAL.
  - `data/mappings/cusip_equivalence.csv` (nueva) con esquema: CUSIP_A, canonical_security, valid_from, valid_to, source, reason, verified_by, source_document.
- **C4-revisada (Q-ESP-V11-4):**
  - Separacion obligatoria: identity temporal validity != metadata temporal validity.
  - `shareClassFIGI` / `figi` es estable frente a corporate actions.
  - `TEMPORAL_UNVERIFIED` aplica SOLO a metadata (ticker), no a identidad FIGI.
  - Cita literal congelada: "La fecha de consulta OpenFIGI no invalida por si sola una identidad FIGI estable; si impide asumir automaticamente que los atributos temporales de la respuesta actual eran validos en el report_period."
- **Coverage policy (Q-ESP-V11-9):**
  - Documento NIPC_COVERAGE_POLICY.md creado.
  - THRESHOLD_1 = UNDEFINED, THRESHOLD_2 = UNDEFINED.
  - Criterios de fijacion: universo operativo, sesgo large-cap, descomposicion de no-mapeados, alcanzabilidad, pairwise mas exigente que single-period.
  - Prohibido fijar por analogia con 90.91% o 29.995%.
  - Valores pendientes de dictamen especifico del auditor ANTES de Gate-NIPC.2.
- **Estado:** precondiciones documentales del auditor cerradas. Gate-NIPC.2 autorizado tras coverage policy numerica (no bloqueante para empezar `aggregation/` + `identity/sec13f_list.py`).
- **Proximo paso autorizado:** implementacion de modulos de sistema (sec13f_list, security_identity, cusip_equivalence, delta_shares, nipc) + tests + probe Q4->Q1.


## IAE NIPC - Probe end-to-end Q4 2025 -> Q1 2026 (2026-09-19)

- **Origen:** probe del motor NIPC completo (S1-S5) sobre datasets reales, previo a Gate-NIPC.3.
- **Informe:** docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_PROBE_INFORME.md
- **Evidence:** docs/auditoria/iae/evidence/nipc_gate0_probe/ (probe_nipc_e2e.py + HASHES.txt + README.md)
- **Resultado:** motor funciona segun contrato. Hallazgo estructural no cubierto por el contrato.
- **Metricas del motor:**
  - identity: pct_canonical = 0.0206 (Q4) / 0.0204 (Q1). Coincide con Gate 0 Mapping.
  - elegibilidad SEC: pct_eligible = 0.4846 (Q4) / 0.5264 (Q1).
  - NIPC observable: +23.6M (scope ALL) / +52.6M (scope ELIGIBLE).
  - coverage pairwise: paired_security_coverage = 0.0181 (ALL) / 0.0368 (ELIGIBLE).
  - status: INSUFFICIENT (thresholds UNDEFINED).
- **Hallazgo estructural (caso Vanguard):**
  - Q4 2025: VANGUARD GROUP INC (CIK 0000102909) presenta 13F-HR con holdings.
  - Q1 2026: el mismo CIK presenta 13F-NT (notice). Los holdings migran a VANGUARD CAPITAL MANAGEMENT LLC (CIK 0002100119) + VANGUARD PORTFOLIO MANAGEMENT LLC (CIK 0002100121).
  - Los CUSIPs (NVDA, AAPL, AMZN, etc.) son identicos entre periodos. Cambia el filer, no la security.
  - El match key C2 produce EXIT del viejo + NEW de los nuevos.
  - nipc_sole = -41.16B y nipc_dfnd = +41.28B se cancelan (neto +52M sobre ~82B brutos).
  - NO es bug: el motor implementa exactamente C2 (spec 4.7).
- **Implicacion:** el gap estructural dominante NO es la cobertura de CUSIPs, es la continuidad de filer entre trimestres. NIPC seguiria INSUFFICIENT con crosswalk 100% si el sesgo de reorganizacion no se controla.
- **Recomendacion del ingeniero:** NO tocar codigo. Documentar como limitacion conocida. Ejecutar mini-probe (top 20 filers) antes de fijar politica.
- **Preguntas al auditor:** Q-PROBE-1 a Q-PROBE-5 (seccion 6 del informe).
- **Estado:** 31 commits locales ahead. Sin push (local-first IAE). Esperando dictamen del auditor.


## IAE NIPC - Mini-probe filer continuity Q-PROBE-5 (2026-09-19)

- **Origen:** dictamen probe end-to-end (Q-PROBE-5, GO mini-probe top 20).
- **Informe:** docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_MINIPROBE_INFORME.md
- **Evidence:** docs/auditoria/iae/evidence/nipc_gate0_probe/probe_filer_continuity.py
- **Resultado:** Vanguard es caso aislado en el top 20. Evidencia NT -> OTHERMANAGER -> HR confirmada.
- **Clasificacion 24 CIKs (union top 20 Q4+Q1):**
  - CONTINUOUS_FILER: 21 (87.5%).
  - FILER_DISCONTINUITY: 3 (12.5%) - los 3 son Vanguard (padre + 2 filiales).
  - NT_TO_HR_RELATION_OBSERVED: 0 (los 3 DISCONTINUITY tienen NT visible).
  - HR_TO_HR_WITHOUT_RELATION: 0.
  - UNRESOLVED: 0.
- **Evidencia NT (caso Vanguard, Q1 2026):**
  - ACC=0000102909-26-002707 CIK=0000102909 13F-NT.
  - OTHERMANAGER declara 10 managers, incluyendo los 2 CIKs grandes de Q1:
    * 0002100119 VANGUARD CAPITAL MANAGEMENT LLC (Q1 shares=70.24B).
    * 0002100121 VANGUARD PORTFOLIO MANAGEMENT LLC (Q1 shares=21.03B).
  - 6 de las 10 filiales ya eran 13F-NT en Q4 (delegando al padre); en Q1 presentan HR directo.
  - Modelo Q4: padre HR + 6 filiales NT. Modelo Q1: padre NT + 10 filiales HR.
- **Hallazgo colateral:** GHISALLO CAPITAL (CIK 0001825214) pasa de 2.39B a 20.69B (+765%). Presente en ambos periodos. NO es discontinuidad de filer. Documentado como observacion.
- **Recomendaciones al auditor:** Q-MINI-1 a Q-MINI-5 (seccion 7 del informe).
- **Estado:** 36 commits locales ahead. Sin push. Pendiente dictamen.


## IAE NIPC - Dictamen mini-probe + correccion cuantitativa Vanguard (2026-09-19)

- **Origen:** dictamen del auditor sobre el mini-probe filer continuity Q-PROBE-5.
- **Documentos:**
  - Dictamen mini-probe: docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_MINIPROBE_DICTAMEN.md
  - Addendum correccion: docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_MINIPROBE_ADDENDUM.md
  - Evidence reconciliacion: docs/auditoria/iae/evidence/nipc_gate0_probe/probe_reconcile_vanguard.py
- **Resultado:** GO CONDICIONADO. Hallazgo estructural NT -> HR confirmado. Reconciliacion economica NO autorizada. Gate-NIPC.2 y .3 bloqueados.
- **Decisiones del auditor:**
  - Q-MINI-1 GO/CERRADO: Vanguard = limitacion conocida, concentrada en top 20, sin generalizacion al universo.
  - Q-MINI-2 GO/CERRADO: no precedencia destructiva. Separar `filer_status` (CONTINUOUS_FILER | FILER_DISCONTINUITY | UNRESOLVED) de `nt_to_hr_relation_observed` (bool) + `nt_to_hr_relation_targets` (list).
  - Q-MINI-3 GO DIAGNOSTICO: GHISALLO (+765%) requiere probe especifico antes de interpretar como fenomeno economico.
  - Q-MINI-4 CERRADO: relacion NT -> HR demostrada documentalmente; reconciliacion economica sin autorizacion.
  - Q-MINI-5 GO TOP 50: ampliar muestra a top 50 filers. NO saltar a universo completo.
- **Hallazgo cuantitativo obligatorio:** inconsistencia 35.329B vs 70.236B de CIK 0002100119 (Vanguard Capital).
  - Causa: el informe original sumo SSHPRNAMT desde INFOTABLE.parquet filtrado por periodo (raw), no desde canonical_snapshot.
  - HR original (001306, 34.906B) fue SUPERSEDED por RESTATEMENT posterior (001311, AN=1).
  - Raw (3 acc): 70.236B. Canonical (2 applied): 35.329B. Ratio 1.988.
  - CIK 0002100121 no afectado (ratio 1.0).
- **Cifras corregidas:**
  - Agregado filiales Q1 2026: 56,359,660,206 (antes: 91,267,483,611).
  - Padre Q4 vs filiales Q1: -6.08B (-9.74%). Antes implicaba +29B. Contraccion real.
- **NIPC observable:** NO afectado. El motor siempre opero sobre canonical_snapshot. Cifras confirmadas: +23.65M (ALL) / +52.57M (ELIGIBLE).
- **Leccion metodologica congelada:** todo agregado de SSHPRNAMT debe calcularse sobre el canonical_snapshot producido por apply_amendments. PROHIBIDO sumar SSHPRNAMT desde INFOTABLE.parquet filtrado por periodo cuando existan amendments RESTATEMENT.
- **Siguiente accion autorizada:** TOP 50 filer continuity probe (misma metodologia que top 20).
- **Estado:** 38 commits locales ahead. Sin push. Pendiente top 50 probe + dictamen posterior.


## IAE NIPC - TOP 50 filer continuity probe Q-MINI-5 (2026-09-19)

- **Origen:** dictamen mini-probe Q-MINI-5, GO TOP 50.
- **Informe:** docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_TOP50_INFORME.md
- **Evidence:** docs/auditoria/iae/evidence/nipc_gate0_probe/probe_filer_continuity_top50.py + top50_output.txt
- **Resultado:** Vanguard es el unico grupo con discontinuidad en top 50. No hay casos nuevos.
- **Clasificacion 54 CIKs (union top 50 Q4+Q1):**
  - CONTINUOUS_FILER: 50 (92.6%).
  - FILER_DISCONTINUITY: 4 (7.4%) - todos Vanguard.
  - UNRESOLVED: 0.
- **Los 4 FILER_DISCONTINUITY (todos Vanguard):**
  - 0000102909 VANGUARD GROUP INC (padre): Q4 HR -> Q1 ausente.
  - 0000933478 VANGUARD FIDUCIARY TRUST CO: Q4 ausente -> Q1 HR (era NT Q4).
  - 0002100119 VANGUARD CAPITAL MGMT LLC: Q4 ausente -> Q1 HR.
  - 0002100121 VANGUARD PORTFOLIO MGMT LLC: Q4 ausente -> Q1 HR.
- **NT_TO_HR observados: 2.**
  - Padre Vanguard NT Q1 (2026-05-08) -> 10 targets en OTHERMANAGER.
  - Fiduciary Trust NT Q4 -> target padre.
- **Comparativa pct_discontinuity:**
  - top 20: 3/24 = 12.5%.
  - top 50: 4/54 = 7.4%.
  - La bajada es artefacto de denominador, no tendencia. Vanguard es la unica fuente.
- **Metodologia confirmada:** agregados SSHPRNAMT sobre canonical_snapshot (regla congelada del addendum anterior).
- **Pendiente GHISALLO (Q-MINI-3):** +765% (2.39B -> 20.69B). Continua presente en ambos periodos. No es discontinuidad. Probe especifico pendiente.
- **Recomendaciones al auditor:** Q-T50-1 a Q-T50-5 (seccion 9 del informe).
- **Estado:** 40 commits locales ahead. Sin push. Pendiente dictamen top 50.


## IAE NIPC - Dictamen TOP 50 + spec v1.4 (2026-09-19)

- **Origen:** dictamen del auditor sobre TOP 50 filer continuity y addendum cuantitativo.
- **Dictamen:** docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_TOP50_DICTAMEN.md
- **Resultado:** FILER CONTINUITY CERRADO como caracterizacion del periodo. Bloqueo de Gate-NIPC.2 vuelve a coverage/thresholds.
- **Decisiones del auditor:**
  - Q-T50-1 CERRADO: no ampliar a top 100/universo. 4 discontinuidades en top 50, todas Vanguard.
  - Q-T50-2 CERRADO: `filer_status` por presencia documental del filing, NO por `SSHPRNAMT > 0`. Nueva dimension `position_mass_status` (HAS_SHARES | ZERO_SHARES | NO_CANONICAL_HOLDINGS).
  - Q-T50-3 CERRADO: NT->HR queda como evidencia documental. NO reconciliacion economica. C2 sin cambios.
  - Q-T50-4 GO DIAGNOSTICO: GHISALLO probe especifico autorizado, no bloqueante.
  - Q-T50-5 GO: siguiente fase = coverage baseline + OpenFIGI.
- **Addendum cuantitativo:** PASS. Regla `canonical_snapshot` congelada.
- **Spec v1.4:**
  - Version 1.4 (commit ff4157e).
  - Seccion 3.15 reescrita: filer continuity = control de integridad, NO threshold.
  - Seccion 3.15 incluye `position_mass_status` (Q-T50-2).
  - Seccion 14.2 actualizada: Gate-NIPC.2 pendiente de coverage/thresholds.
- **Coverage policy ampliada:**
  - Seccion 10.bis reformulada: filer continuity = control de integridad.
  - Eliminados THRESHOLD_3 y THRESHOLD_4 como umbrales.
  - Solo THRESHOLD_1 y THRESHOLD_2, ambos UNDEFINED.
- **Estado de gates:**
  - Q-PROBE-5 TOP 20: PASS.
  - Q-T50 TOP 50: PASS.
  - Vanguard concentration: CONFIRMADA.
  - NT -> OTHERMANAGER -> HR: CONFIRMADO documentalmente.
  - Addendum cuantitativo: PASS.
  - Regla canonical_snapshot: CONGELADA.
  - C2: VIGENTE.
  - Reconciliacion economica NT-HR: NO AUTORIZADA.
  - THRESHOLD_1/2: UNDEFINED.
  - Gate-NIPC.2: BLOQUEADO por coverage.
  - Gate-NIPC.3: NO AUTORIZADO.
- **Siguiente fase autorizada:**
  - Coverage baseline + OpenFIGI over unresolved-only.
  - GHISALLO probe (Q-T50-4).
  - Propuesta THRESHOLD_1/2 tras evidencia de coverage.
- **Estado:** 44 commits locales ahead. Sin push. Bloqueo vuelve a coverage/thresholds.


## IAE NIPC - Coverage baseline Fase A (2026-09-19)

- **Origen:** Q-T50-5 del dictamen TOP 50 (coverage baseline + OpenFIGI).
- **Alcance ejecutado:** coverage baseline (OpenFIGI queda fuera de esta fase).
- **Informe:** docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_BASELINE_INFORME.md
- **Evidencia:** docs/auditoria/iae/evidence/nipc_gate0_baseline/ (README + probe + output + HASHES).
- **Resultado:** las 6 metricas pairwise calculadas sobre 4 universos anidados.
- **Metricas OPERATIONAL_EQUITY (Q4 2025 -> Q1 2026):**
  - paired_security_coverage = 0.9909
  - paired_weighted_share_coverage = 0.9864
  - NIPC observable = -109,460,838
  - STATUS = INSUFFICIENT (thresholds UNDEFINED)
- **Hallazgos:**
  - H1: TECHNICAL_RADAR == OPERATIONAL_EQUITY (get_instrument_class no-op sobre crosswalk actual).
  - H2: coverage_* = 1.0 en OPERATIONAL_EQUITY por construccion (tautologico).
  - H3: NIPC cambia signo entre RAW (+23.65M) y OP_EQUITY (-109.46M).
- **Fallout ELIGIBLE_SEC -> TECHNICAL_RADAR (Q1 2026):**
  - IN_RADAR: 20.47% units / 28.75% peso.
  - CANONICAL_TICKER_OUTSIDE_RADAR: 13.40% units / 10.67% peso.
  - OBSERVED_ONLY_NO_CANONICAL: 66.13% units / 60.58% peso (gap OpenFIGI-addressable).
- **Incidente resuelto:** CRLF->LF en `baseline_output.txt`, hash regenerado, `git commit --amend`.
- **Estado:** Fase A cerrada. Gate-NIPC.2 SIGUE BLOQUEADO por thresholds. Sin push.
- **Siguiente fase autorizada:** propuesta THRESHOLD_1/THRESHOLD_2 con evidencia, o ejecucion OpenFIGI over unresolved-only (a dictamen del auditor).


## IAE NIPC - Micro-gate Coverage Contract Normalization (2026-09-19)

- **Origen:** dictamen del auditor sobre coverage baseline (Issue ABIERTO: circularidad `operational_universe`).
- **Cadena de fases:** F2.1 -> F2.1-bis -> F2.2 -> F2.3 -> F2.4.
- **Estado:**
  - F2.1 PASS (eda67ef). Inventario 3 conceptos: TARGET/RESOLVED/PAIRED no existen explicitamente en v1.0.
  - F2.1-bis PASS (4659ce1). Ninguna fuente de identidad en disco cumple (CUSIP + radar USA + independencia). Camino B declarado: `TARGET_CUSIP_REGISTRY = NOT AVAILABLE`.
  - F2.2 NO GO CON CAMBIOS OBLIGATORIOS (d4a926e + dictamen). Propuesta v1.1 resolvia algebraicamente pero no ontologicamente. 8 cambios obligatorios para v1.2.
  - F2.3 pendiente.
  - F2.4 NO AUTORIZADA.
- **Hallazgo colateral:** `mapping_coverage 90.91% (220/242)` de la policy v1.0 L115 sale exactamente de `etf_holdings.csv`, parte del crosswalk evaluado. Confirmacion numerica de la circularidad.
- **Reordenacion propuesta del pipeline (a dictamen):** baseline -> RADAR_TARGET_REGISTRY (via OpenFIGI) -> policy v1.2 -> medicion -> thresholds.
- **Estado de gates:** THRESHOLD_1/2 UNDEFINED. OpenFIGI NO AUTORIZADO. Gate-NIPC.2 BLOQUEADO. Gate-NIPC.3 NO AUTORIZADO. Policy v1.0 intacta (hash 57f2d01f...).
- **Documentos:**
  - `docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_CONTRACT_INVENTARIO.md`
  - `docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_F21BIS_FUENTES_IDENTIDAD.md`
  - `docs/auditoria/NIPC_COVERAGE_POLICY_V11_PROPUESTA.md` (NO GO)
  - `docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_BASELINE_DICTAMEN.md`
- **Commits:** eda67ef, d4a926e, 4659ce1.
- **Sin push.** 55 commits locales ahead.

## IAE NIPC - F2.2-v2 propuesta v1.2 (2026-09-19)

- **Origen:** dictamen F2.2 NO GO CON CAMBIOS OBLIGATORIOS + F2.1-bis (Camino B).
- **Documento:** `docs/auditoria/NIPC_COVERAGE_POLICY_V12_PROPUESTA.md` (commit 0518b96).
- **Estado:** BORRADOR. NO vigente. NO sustituye a la policy v1.0.
- **Alcance:**
  - Aplica los 8 cambios obligatorios del dictamen F2.2.
  - Camino B declarado: `RADAR_TARGET_REGISTRY = NOT AVAILABLE`.
  - coverage = NOT_MEASURABLE mientras no exista registry.
  - NIPC status = INSUFFICIENT por ausencia de denominador, no por valores bajos.
- **No toca:** motor (`nipc.py`, `delta_shares.py`, `security_identity.py`), C2, thresholds, OpenFIGI, baseline evidencia, spec v1.4, policy v1.0.
- **Cadena F2:**
  - F2.1 PASS (eda67ef).
  - F2.1-bis PASS (4659ce1).
  - F2.2 NO GO CON CAMBIOS (d4a926e).
  - F2.2-v2 borrador redactado (0518b96).
- **Estado de gates:** F2.3 PENDIENTE EXTERNO. F2.4 NO AUTORIZADA. THRESHOLD_1/2 UNDEFINED. OpenFIGI NO AUTORIZADO. Gate-NIPC.2 BLOQUEADO. Gate-NIPC.3 NO AUTORIZADO.
- **Policy v1.0 intacta:** hash 57f2d01f...
- **Sin push.** 58 commits locales ahead.


---

## Ciclo 2026-09-20 - Limpieza documental + regla 1 concepto = 1 fichero

**HEAD al cierre:** 4fe2b62.
**Ahead origin/main:** 108 commits.
**Tests:** 979 passed + 2 skipped.

### Contexto

Tras la auditoria estructural (14 bloques) y el diagnostico de
divergencias contrato <-> codigo, se detecto que `docs/auditoria/`
habia acumulado 81 .md sin estructura, con 5 versiones de la spec NIPC,
4 versiones de la policy de cobertura, 23 dictamenes sueltos y 17
informes sin consolidar. La trazabilidad se habia convertido en ruido.

### Acciones

1. Limpieza fisica de `docs/auditoria/`:
   - 81 .md -> 27 .md organizados en iae/ (9), radar/ (8),
     auditorias/ (4), raiz (3: PROMPT, FOLLOWUPS, README) + evidence/.
   - Consolidacion de 17 informes IAE en `iae/INFORME.md`.
   - Consolidacion de 23 dictamenes IAE en `iae/DICTAMENES.md`.
   - Borrado de versiones previas (spec v1.0-v1.3, policy V11/V12).
   - Borrado de transfers de sesion (6).
   - Borrado de huerfanos (4).
   - Borrado de `archive/` completo (56 .md redundantes).
   - Borrado de `validation/archive/` (60), `scripts/archive/` (4),
     `docs/plan/` (2).
   - Borrado de `__pycache__` (28 dirs) + `.pyc` (504).

2. Regla nueva (PROMPT_MAESTRO v6.44 seccion 3.7):
   - 1 concepto = 1 fichero vivo.
   - Al evolucionar, se edita in-place.
   - Las versiones previas se borran si su contenido esta embebido
     en la version vigente.
   - Prohibido ficheros con sufijos `_v1.md`, `_V12_PROPUESTA.md`,
     `_DICTAMEN_C.md`.
   - Basura = borrar. Sin apelar a git.

3. Nuevo `docs/auditoria/README.md`: navegacion + convenciones + rutas
   por rol.

4. Nuevo `docs/auditoria/iae/RECONCILIACION_CONTRATO_CODIGO.md`:
   expediente de las 3 divergencias contrato <-> codigo (D1 P61 no
   conectado, D2 P38 con proxy, D3 P60 sin raise). Input para F2.4.

### Commits

    4fe2b62  chore(docs): limpieza total documental + regla 1 concepto = 1 fichero

### Estado

- Working tree limpio.
- Validacion verde (compileall + pyflakes + pytest).
- Bloqueos vigentes sin cambios: F2.4 pendiente, THRESHOLD_1/2
  UNDEFINED, Gate-NIPC.2 BLOQUEADO, OpenFIGI masivo NO AUTORIZADO.

### Proximo paso

F2.4 debe dictaminar sobre las 3 divergencias del expediente. La
reconciliacion de codigo queda bloqueada hasta entonces.

---


---

## Ciclo 2026-09-20 (bis) - Diagnostico del run scheduled fallido

**HEAD:** c38e127.
**Run fallido:** 35432363006 (sabado 19/09, 08:34 UTC, scheduled).

### Diagnostico

El run scheduled del sabado fallo con:

    [FAIL] data/stock_prices.parquet.manifest.json:
           coverage_pct_last=0.936102 < 0.95
    [OK]   data/market_data.parquet.manifest.json
    Guard coverage: 1 fallo(s). Abortando.

Causa: los 20 tickers `.L` (London Stock Exchange) del universo
`stock_prices.parquet` no tenian Close. Yahoo no los sirvio a
las 09:34 London del sabado. Coverage = (313-20)/313 = 0.9361.
Guard bloqueo. `origin/main` intacto en `9d4a81e`.

**No es bug.** Es comportamiento esperado del guard. Es una
limitacion conocida (§12 del prompt): los 20 tickers `.L` no
tienen provider dedicado.

### Deuda registrada

Creado `radar/DEUDA.md` con 3 entradas:

    D-RADAR-01  20 tickers LSE sin provider dedicado (MEDIA)
    D-RADAR-02  Cron `0 4 * * *` los fines de semana (BAJA)
    D-RADAR-03  Guard coverage: no distingue core de LSE (MEDIA)

Decision del usuario (2026-09-20): buscar fuente dedicada para
los 20 `.L` y cerrar el problema de raiz.

### Estado

- main intacto.
- Guard funcionando.
- Deuda registrada.
- Sin cambios de codigo en este ciclo.

---


---

## Ciclo 2026-09-20 (ter) - Tests contractuales + dictamen externo aplicado

**HEAD al cierre:** 04f3222.
**Ahead origin/main:** 114.

### Entregables

1. **Dictamen del auditor externo** sobre el expediente F2.4 recibido.
   Aplicados los 13 puntos + 5 correcciones obligatorias:

       - A.6.3 reformulado (Q12/A: shareClassFIGI como clave de pairing,
         esperado alternativo Q12/B documentado).
       - Nomenclatura unificada D1/D2/D3 -> P60/P61/P38.
       - A.6.5 cambiado: crear v2, NO modificar in-place.
       - A.6.4 suavizado: exigir demostracion de semantica, no cambio
         de cifra.
       - HEAD 4fe2b62 marcado como snapshot historico.
       - P60 D3 reformulado (CUSIP/ISIN -> NULL, no ValueError).
       - A.6.6 separado de Gate-NIPC.2 (solo inputs).
       - F2.4 / F2.4-CLOSE como hitos distintos.
       - Tabla de dependencias separa P38 orquestacion (autonomo) de
         P38 materializacion (OpenFIGI).
       - Test "pesos agregados antes de max(Q4,Q1)" anadido.

2. **15 tests contractuales** creados:

       tests/test_p60_contract.py   4 (3 pass + 1 xfail)
       tests/test_p61_contract.py   5 (3 pass + 2 xfail)
       tests/test_p38_contract.py   5 (2 pass + 3 xfail)

   Los 6 xfail documentan ejecutablemente las 3 divergencias:

       D1 P61 (2 tests): resolver no invoca resolve_source_status +
                         cusip_ticker_exceptions sale UNVERIFIED.
       D2 P38 (3 tests): nipc no importa radar_target_catalog +
                         firma no acepta target_q4/q1 +
                         pesos no agregados por security.
       D3 P60 (1 test):  CSV sin columna identity_type asume TICKER.

   Cuando F2.4 autorice los fixes, se retiran los marcadores xfail
   y los tests deben pasar.

3. **Deuda radar registrada** (radar/DEUDA.md):

       D-RADAR-01  20 tickers LSE sin provider dedicado (MEDIA)
       D-RADAR-02  Cron `0 4 * * *` los fines de semana (BAJA)
       D-RADAR-03  Guard coverage: no distingue core de LSE (MEDIA)

### Verificacion

    compileall:  OK
    pyflakes:    0 warnings
    suite:       988 passed + 6 xfailed + 2 skipped

### Estado

- Working tree limpio.
- Local-first IAE activo. Sin push.
- F2.4 PENDIENTE EXTERNO.
- Gate-NIPC.2 BLOQUEADO.

### Proximo paso

Enviar el paquete al auditor externo:

    iae/RECONCILIACION_CONTRATO_CODIGO.md
    iae/REESTRUCTURACION_MODULO.md
    iae/FASE_A6_PLAN.md
    tests/test_p60_contract.py
    tests/test_p61_contract.py
    tests/test_p38_contract.py

---


---

## Ciclo 2026-09-20 (cuater) - Dictamen externo v2 sobre expediente F2.4

**HEAD al cierre:** 3ad57b1.
**Ahead origin/main:** 120.

### Contexto

Segunda revision del auditor externo sobre el paquete F2.4. Senalo 4
correcciones obligatorias + 8 adicionales (13 puntos en total).

### Correcciones aplicadas

**Obligatorias (4):**

  1. Eliminar toda referencia a `ValueError` para CUSIP/ISIN en
     REESTRUCTURACION y A.6.2.
     - El contrato vigente mantiene CUSIP/ISIN -> NULL.
     - La divergencia real es el default TICKER silencioso.

  2. Eliminar el test P38 que exigia que `nipc.py` importase
     `radar_target_catalog`.
     - La arquitectura exige que TARGET se reciba como parametro
       externo (construido por `target_builder.py`).

  3. Reformular tests P38 para probar TARGET externo + shareClassFIGI,
     no detalles de imports internos.
     - API normativa unica: `compute_contractual_coverage(...)`.
     - `compute_coverage_pairwise` = legacy/proxy (deprecada).

  4. Eliminar la afirmacion de que A.6 desbloquea Gate-NIPC.2.
     - A.6 solo entrega inputs. El desbloqueo requiere propuesta de
       thresholds + dictamen especifico.

**Adicionales aplicadas:**

  - HEAD normalizado (SUBMISSION / AUDITED / POST-SNAPSHOT).
  - D2 opcion B reformulada: proxy observacional (NO CONTRACTUAL).
  - Seccion 9 en RECONCILIACION sobre agregacion shareClassFIGI.
  - Criterios globales: "N + M + K" en vez de "979 + 11".
  - Test P60 reformulado (verifica no-TICKER-asumido, no `[]`).
  - Test P61 acoplado a implementacion eliminado.
  - A.6.2-P60 sin `ValueError`.
  - Nota de arquitectura: `nipc.py` NO importa catalogo.

### Tests contractuales (version final)

    tests/test_p60_contract.py    4 tests (3 pass + 1 xfail)
    tests/test_p61_contract.py    4 tests (3 pass + 1 xfail)
    tests/test_p38_contract.py    5 tests (2 pass + 3 xfail)

    Total: 14 tests, 9 pass, 5 xfail.

Los 5 xfail documentan ejecutablemente las 3 divergencias:

    P60 (1): default TICKER silencioso sin identity_type declarado.
    P61 (1): cusip_ticker_exceptions con vigencia -> no VERIFIED.
    P38 (3): compute_contractual_coverage inexistente +
             firma sin target_q4/q1 +
             sin agregacion por shareClassFIGI.

### Verificacion

    compileall:  OK
    pyflakes:    0 warnings
    suite:       988 passed + 5 xfailed + 2 skipped

### Estado

- Working tree limpio.
- Local-first IAE activo. Sin push.
- F2.4 PENDIENTE EXTERNO (aplicadas ya las correcciones que el
  auditor sugirio antes de emitir dictamen).

### Paquete listo

    iae/RECONCILIACION_CONTRATO_CODIGO.md
    iae/REESTRUCTURACION_MODULO.md
    iae/FASE_A6_PLAN.md
    tests/test_p60_contract.py
    tests/test_p61_contract.py
    tests/test_p38_contract.py

---


---

## Ciclo 2026-09-20 (quinquies) - Dictamen externo v3 sobre expediente F2.4

**HEAD al cierre:** 33bd3f9.
**Ahead origin/main:** 122.

### Correcciones aplicadas (6 puntos)

  1. HEAD en documentos: `2eb4dcd` -> `3ad57b1` (SUBMISSION HEAD).
  2. Eliminada la semantica "P60 raise CUSIP/ISIN" en
     REESTRUCTURACION seccion 1, 4.4 y 7.
  3. D3 reformulada: "default TICKER implicito" (no ValueError).
  4. A.6.1 ampliado con las 7 decisiones requeridas de F2.4:
        D1 (P61), D2 (P38), D3 (P60), Q12 (pairing),
        AGREG. (agregacion shareClassFIGI),
        OpenFIGI, Policy v1.3.
  5. Resuelta la arquitectura de agregacion P38:
        Opcion 2 (propuesta): funcion separada
        aggregate_positions_by_shareclass_figi() produce pesos
        agregados, compute_contractual_coverage() los recibe.
        Decision final a F2.4.
  6. Separada evidencia legacy/proxy de evidencia contractual:
        test_p38_legacy_denominador_cero_unavailable
        (regresion de compute_coverage_pairwise, no de la API
        contractual).

### Estado del paquete

    iae/RECONCILIACION_CONTRATO_CODIGO.md   actualizado
    iae/REESTRUCTURACION_MODULO.md          actualizado
    iae/FASE_A6_PLAN.md                     actualizado
    tests/test_p60_contract.py              4 tests (3 pass + 1 xfail)
    tests/test_p61_contract.py              4 tests (3 pass + 1 xfail)
    tests/test_p38_contract.py              5 tests (2 pass + 3 xfail)

### Verificacion

    pyflakes:  0 warnings
    suite:     988 passed + 5 xfailed + 2 skipped

### Estado

PENDIENTE regenerar `_entrega_F24.md` con la version actualizada y
enviar al auditor.

---


---

## Ciclo 2026-09-20 (sextus) - Dictamen externo v4 sobre expediente F2.4

**HEAD al cierre:** fc8de6c.
**Ahead origin/main:** 124.

### Correcciones aplicadas (5 puntos)

  1. HEAD residual `2eb4dcd` -> `3ad57b1` en RECONCILIACION (linea
     del dictamen F2.4 como objeto documental).
  2. REESTRUCTURACION §5.1: eliminada semantica "CUSIP/ISIN -> ValueError".
     Reformulado: CUSIP -> None, ISIN -> None, identity_type
     ausente/invalido -> ValueError.
  3. REESTRUCTURACION §5.2: anadido "Mantener/cubrir CUSIP/ISIN -> None".
  4. RECONCILIACION §6 tabla tests: `compute_coverage_pairwise` ->
     `compute_contractual_coverage` + "por security" -> "por shareClassFIGI".
  5. FASE_A6_PLAN: A.6.3 titulo neutro (no presupone Modelo A);
     criterio "xfail = 0" condicional a las decisiones F2.4. Anadido
     concepto "xfail residual aceptado" para capacidades declaradas
     NO CONTRACTUALES o DIFERIDAS.

### Verificacion de residuos

    RECONCILIACION: 2eb4dcd=0, CUSIP-raise=0, cov_pairwise-recibe-target=0
    REESTRUCTURACION: idem
    FASE_A6_PLAN: idem

### Estado del paquete

Expediente F2.4 en su version v4. Listo para regenerar entrega
consolidada y enviar al auditor.

---


---

## Ciclo 2026-09-20 (septimus) - F2.4 GO condicionado - correcciones finales

**HEAD al cierre:** c5abbbc.
**Ahead origin/main:** 126.

### Contexto

El auditor externo ha dado GO condicionado a 5 correcciones finales.
Todas aplicadas en este ciclo.

### Correcciones aplicadas

  1. RECONCILIACION §5: la dependencia de A.6 se amplia de "D1/D2/D3"
     a "D1/D2/D3 + Q12, AGREG., OpenFIGI, Policy v1.3".
  2. FASE_A6_PLAN A.6.1: titulo ampliado a "divergencias y decisiones
     arquitectonicas asociadas".
  3. FASE_A6_PLAN A.6.2: nota explicita sobre commits adicionales
     segun Q12/AGREG. (no son exactamente 3 commits fijos).
  4. FASE_A6_PLAN A.6.4: rama condicional D2=A (TARGET real -> recalcular
     evidencia v2) vs D2=B (proxy -> conservar separado, no apto
     THRESHOLD_2, no exigir TARGET_PAIRWISE contractual).
  5. Tests: `@pytest.mark.xfail(strict=True, ...)` en los 5 xfail. Si
     una divergencia desaparece accidentalmente antes de retirar el
     marcador, XPASS falla la suite -> fuerza sincronizacion
     codigo/test/documentacion.
  6. Test P60: nuevo `test_p60_identity_type_invalido_raises` cubre
     `identity_type="INVALID"` -> ValueError.

### Tests contractuales (final)

    tests/test_p60_contract.py    5 tests (4 pass + 1 xfail strict)
    tests/test_p61_contract.py    4 tests (3 pass + 1 xfail strict)
    tests/test_p38_contract.py    5 tests (2 pass + 3 xfail strict)

    Total: 15 tests, 10 pass, 5 xfail strict.

### Verificacion

    compileall:  OK
    pyflakes:    0 warnings
    suite:       989 passed + 5 xfailed + 2 skipped

### Estado

Expediente F2.4 en su version final (v5). Listo para enviar.
F2.4 GO CONDICIONADO a este paquete.

---


---

## Ciclo 2026-09-20 (octavus) - F2.4 v6 paquete final

**HEAD al cierre:** 8e4d957.
**Ahead origin/main:** 128.

### Correcciones aplicadas (4 puntos)

  1. RECONCILIACION D3: titulo reformulado a "infiere TICKER cuando
     falta identity_type declarado".
  2. RECONCILIACION §6: eliminada la frase "Ninguno existe hoy".
     Reemplazada por explicacion real de la tabla.
  3. FASE_A6_PLAN A.6.4:
     - titulo ampliado a "condicional a las decisiones F2.4".
     - precondicion ampliada (D1/D2/D3/Q12/AGREG/Policy).
     - rama D2=B: NO renombrar historicos. Generar v2 separada.
  4. REESTRUCTURACION: diagrama nipc.py mas preciso.

### Nota sobre las iteraciones

6 rondas de revision con el auditor externo. En la ultima, el propio
auditor afirmo: "Despues de esas cuatro correcciones, la entrega si la
trataria como paquete F2.4 listo para auditor externo." Aplicadas las
4, este v6 es el paquete final. Enviar y esperar dictamen.

### Estado

Paquete F2.4 v6 listo para enviar. Sin mas iteraciones previstas.

---


---

## HITO: F2.4 GO (2026-09-20)

**HEAD:** 23311b2.
**Ahead origin/main:** 129.

El auditor externo ha emitido **GO sobre el expediente F2.4** tras la
aplicacion de las 4 correcciones finales (commit `8e4d957`).

### Significado

- Expediente F2.4 aprobado.
- Fase A.6 desbloqueada.
- Decisiones D1/D2/D3/Q12/AGREG/OpenFIGI/Policy v1.3 deben ser
  aplicadas segun el dictamen formal del auditor.

### Pendiente antes de A.6

Confirmar con el auditor el contenido exacto de las 7 decisiones:

    D1 (P61)   ¿Conectar resolve_source_status o retirar
               temporal_validity.py?

    D2 (P38)   ¿TARGET real (requiere OpenFIGI) o proxy
               observacional NO CONTRACTUAL?

    D3 (P60)   ¿Rechazar filas sin identity_type o mantener
               default TICKER?

    Q12        ¿Pairing por shareClassFIGI (A) o por
               canonical_security (B)?

    AGREG.     ¿Opcion 1 (agregacion en coverage.py) o
               Opcion 2 (funcion separada)?

    OpenFIGI   ¿Autorizado para materializacion completa TARGET?

    Policy v1.3  ¿Aprobada o mantener v1.0?

Sin estas respuestas, A.6 no puede ejecutarse: el plan tiene ramas
condicionales para cada una.

---


## CICLO F2.4 - P60/P61/P38/P63/P64/P65 (2026-09-20)

Ciclo completo de reconciliacion contrato <-> codigo tras dictamen F2.4.

### Contexto

F2.4 (dictamen F2.4 formal, ver `iae/DICTAMENES.md` #24) emitio GO
CONDICIONADO sobre la arquitectura IAE + 3 bloqueantes estructurales +
10 reglas adicionales. El ciclo aplica las decisiones D1/D2/D3/Q12/AGREG./
OpenFIGI/Policy hasta donde es posible sin OpenFIGI masivo.

### Fase documental (5 commits)

    0ea5586  DICTAMENES.md #24 (dictamen F2.4 formal)
    150d24d  INFORME.md #18 + cabecera v2
    2197f1f  FASE_A6_PLAN.md (3 bloqueantes + sub-fases A.6.0 y A.6.2-bis)
    26978ef  REESTRUCTURACION_MODULO.md (rediseno TARGET + PositionRecord)
    6bc8675  NIPC_CONTRATOS_SEMANTICOS_v1.md (P62-P65) + A.6.5 in-place

### Fase implementacion (6 commits)

    5d60d48  fix(iae): P60 fail-closed - sin default TICKER (D3 F2.4)
             - _find_active_equivalence rechaza filas sin identity_type
             - load_cusip_equivalence lanza ValueError si falta columna
             - 9 tests heredados actualizados con identity_type explicito
             - test_q8_identity_type_default_ticker -> test_p60_csv_sin_identity_type_error

    1773cc9  fix(iae): P61 conectar resolve_source_status (D1 F2.4)
             - crosswalk_internal separado en cusip_ticker_exceptions vs etf_holdings
             - evidence incluye operational_source + valid_from + valid_to
             - _SOURCE_TO_OP_STATUS eliminado; resolve_source_status invocado
             - 42 tests security_identity + 4 P61 contractuales OK

    a48717a  feat(iae): P38 coverage contractual (D2 quirurgico F2.4)
             - NUEVO aggregation/coverage.py: PositionRecord + aggregate_positions_by_shareclass_figi + compute_contractual_coverage
             - compute_nipc expone evidence_class PROXY; compute_nipc_contractual ruta CONTRACTUAL
             - 13 tests P38 (TARGET externo, max(150,140)=150, solo VERIFIED contribuye)
             - NO toca MATCH_KEY ni C2

    a841cd6  test(iae): P63 contractual
             - test_p63_amendment_evita_exit_falso (R6: amendments pre-delta)
             - test_p63_othermanager_produce_exit_mas_new (R7 documentado)
             - R8: "SOLD" no en ALL_MATCH_STATUSES

    05f0a84  docs(iae): P63 GO CONDICIONADO (8 reglas + arquitectura + #25)
             - NIPC_CONTRATOS_SEMANTICOS_v1.md seccion 12 reescrita
             - MISSING = ausencia sin causa demostrable
             - REPORTING_CONFLICT != REPORTING_OVERLAP_UNRESOLVED
             - DICTAMENES.md #25

    2f8140a  fix(iae): P64 gross observed delta (F2.4 GO CONDICIONADO)
             - DELTA_SEMANTICS_GROSS_OBSERVED = True
             - P64_EVENTS_DEFERRED = (split, reverse_split, spin_off, merger, share_class_conversion)
             - 4 tests P64 (semantica, columnas, split no economico, CUSIP identidad)
             - Audit consumidores: solo nipc.py::_sum_delta, lee como magnitud

### Fase consolidacion P64 + P65 (1 commit)

    47594a4  docs(iae): expediente P64+P65 consolidado + contrato 13-14 + #26
             - NUEVO iae/P64_P65_EXPEDIENTE.md (447 lineas)
             - NIPC_CONTRATOS_SEMANTICOS_v1.md secciones 13 (P64) y 14 (P65) reescritas
             - DICTAMENES.md #26 (P65 v3 GO CONDICIONADO)

### Dictamenes del ciclo (registro consolidado en DICTAMENES.md)

| # | Tema | Resultado |
|---|---|---|
| 24 | F2.4 formal | GO CONDICIONADO arquitectura + 3 bloqueantes + 10 reglas |
| 25 | P63 Missing != Sold | GO CONDICIONADO (8 reglas, arquitectura multi-dimensional) |
| 26 | P65 v3 Manager Duplication | GO CONDICIONADO (3 correcciones de cierre) |

### Reglas semanticas consolidadas

    REPORTING RELATIONSHIP  !=  REPORTING NETWORK  !=  DEDUP AUTHORIZATION  !=  ECONOMIC OWNERSHIP

    P64: delta_shares = GROSS_OBSERVED_DELTA (no economic)
    P63: MISSING = ausencia sin causa demostrable (no toda ausencia)
    P65: solo dedup con evidencia cruzada entre filings (L3)

### Estado al cierre del ciclo

    Tests locales        1008 passed + 2 skipped + 0 xfailed
    pyflakes             0 warnings
    compileall           OK
    Gate validacion      10/10
    HEAD                 47594a4
    Ahead                141 commits locales
    Push                 NO (local-first IAE activo)

### Pendientes del ciclo

    P65 implementacion       4 commits (plan 2.17 del expediente)
    P62 point-in-time        Requiere OpenFIGI masivo
    Bloqueante 1 (TARGET)    Requiere OpenFIGI masivo
    Bloqueante 2 (PIT)       Requiere OpenFIGI masivo

### Bloqueos vigentes

    THRESHOLD_1 / THRESHOLD_2      UNDEFINED
    Gate-NIPC.2                    BLOQUEADO
    Gate-NIPC.3                    NO AUTORIZADO
    OpenFIGI masivo                NO AUTORIZADO
    Policy v1.3 aplicacion         NO AUTORIZADA
    Push a origin/main             NO

### Referencias

    iae/P64_P65_EXPEDIENTE.md                 ciclo P64+P65 completo
    iae/NIPC_CONTRATOS_SEMANTICOS_v1.md       contrato secciones 12-14
    iae/DICTAMENES.md                          entradas #24 #25 #26
    iae/INFORME.md                             entrada #18
    iae/FASE_A6_PLAN.md                        plan A.6 actualizado
    iae/REESTRUCTURACION_MODULO.md             rediseno arquitectonico

---


## CICLO P65 v1 - Manager Duplication (2026-09-20)

Implementacion de la capa de deduplicacion de reporting segun dictamen
P65 v3 (ver iae/DICTAMENES.md #26). GO CONDICIONADO con 3 correcciones.

### Correcciones de cierre aplicadas

1. REPORTING_CONFLICT solo si contradiccion documental explicita.
2. Anadido REPORTING_OVERLAP_UNRESOLVED.
3. DROP_DUP diferido a v2: sin evidencia cuantitativa externa, toda
   coexistencia de dos filings sobre la misma security es OVERLAP por
   definicion y se trata como KEEP. Fail-closed.

### Fase implementacion (4 commits)

    2ff1751  feat(iae): P65 Commit 1 - modelos + constantes + L1/L2/L3
             - reporting_dedup.py: TRANSITION_*, DEDUP_REASON_*,
               DEDUP_DECISION_*, EVIDENCE_LEVEL_*, DEDUP_AUDIT_COLUMNS
             - ReportingEvidence dataclass (sin economic_owner_cik)
             - classify_evidence_level(L1/L2/L3/None)
             - 14 tests estructurales

    420d1d0  feat(iae): P65 Commit 2 - build_effective_reporting_snapshot
             - PRE-delta: R1 con 7 requisitos (contrato 14.4)
             - Fast-path si l3_index vacio (sin iterar 2.4M filas)
             - 6 tests funcionales R1

    7f683e8  feat(iae): P65 Commit 3 - classify_reporting_transition
             - POST-delta: HANDOFF con reporting_for estable + A!=B
             - Ambiguedad (dos representantes) -> NULL fail-closed
             - 7 tests funcionales HANDOFF

    8c2fcf0  fix(iae): P65 Commit 2-fix - DROP_DUP diferido v2
             - _apply_intra_period_dedup NO emite DROP con 13F puro
             - Coexistencia L3 unidireccional -> OVERLAP_UNRESOLVED + KEEP
             - 5 tests del matrix (4, 5, 12, 13, 15)

### Fix colateral destapado por el probe

    3c2140e  fix(iae): cusip_equivalence.csv cabecera identity_type
             - El CSV en produccion no tenia la columna que P60 exige
             - 0 filas de datos afectadas (fichero vacio por diseno)
             - Comentario Q8 obsoleto en security_identity.py limpiado

### Evidencia empirica (probe e2e)

    62314ac  feat(iae): P65 Commit 4 - probe + evidencia fail-closed
             - Probe reducido (200 CIKs, Q4 2025 -> Q1 2026): ~30s
             - Probe completo (~3.3M filas): ~10 min (referencia)
             - Resultado:
                 dedup_audit rows = 0
                 reporting_transition = {NULL: 252.341}
                 match_status: BOTH 25.734, EXIT 7.625, NEW 2.260,
                               UNRESOLVED_IDENTITY 216.722
             - Evidencia: evidence/nipc_p65_probe/ (README + HASHES + output)

### Reglas consolidadas

    REPORTING RELATIONSHIP != REPORTING NETWORK != DEDUP AUTHORIZATION != ECONOMIC OWNERSHIP

    Solo L3 autoriza DROP. Sin L3 -> KEEP.
    Coexistencia -> OVERLAP_UNRESOLVED (v1).
    HANDOFF requiere reporting_for_manager_cik estable + A != B.

### Estado al cierre

    Tests locales        1039 passed + 2 skipped + 0 xfailed
    pyflakes             0 warnings
    compileall           OK
    Gate validacion      10/10
    HEAD                 62314ac (o posterior)
    Ahead                149 commits locales
    Push                 NO (local-first IAE activo)

### Pendientes (capacidad diferida)

    DROP_DUP efectivo        v2: evidencia cuantitativa externa
    L3 cruzado real          fuente del campo "Other Managers Reporting"
    Probe completo           ~10 min, opcional
    P62 point-in-time        OpenFIGI masivo
    Bloqueante 1 (TARGET)    OpenFIGI masivo
    Fases B-E                N-PORT cross-validation, etc.

### Referencias

    iae/P64_P65_EXPEDIENTE.md                       ciclo P64+P65 completo
    iae/NIPC_CONTRATOS_SEMANTICOS_v1.md             seccion 14 (P65)
    iae/DICTAMENES.md                                entrada #26
    iae/evidence/nipc_p65_probe/                     evidencia empirica
    src/institutional_accumulation/aggregation/reporting_dedup.py

---


## P66 - L3 reformulacion via OTHERMANAGER (2026-09-20)

- **Origen:** ciclo P65 v2 (DROP_DUP efectivo). Gate 0 extendido revelo
  que la evidencia cruzada para L3 no se encontraba en OTHERMANAGER2 ni
  en ADDITIONALINFORMATION.
- **Informe inicial:** `iae/P66_INFORME_HALLAZGO.md` (commit 2edb462).
  Hipotesis: la relacion vive en XML crudo del 13F-NT.
- **Revision externa:** identifico 3 errores metodologicos:
  1. Confusion OTHERMANAGER (reporting for this manager) vs
     OTHERMANAGER2 (included in this report). Son relaciones distintas.
  2. Caso Yorktown con filer invertido.
  3. "XML firmado digitalmente" sobreafirmado.
- **Gate 0.1 (commit 6b610b3):** OTHERMANAGER cubre 100% de NT filings
  (2,008/2,008 Q4; 2,045/2,045 Q1). OTHERMANAGER2 cubre 0% NT.
- **Gate 0.2 (commit 6b610b3):** caso Vanguard Q4 2025 -> Q1 2026
  documentado bidireccionalmente en OTHERMANAGER sin XML.
- **Conclusion:** P66 como ciclo de XML crudo + parser propio + 4,000
  descargas queda CANCELADO. La evidencia ya esta en el parquet.
- **Reemplazo:** `iae/P66_L3_REFORMULACION_PROPUESTA.md`.
  Propone reformular el requisito 4 de 14.3 anclado a
  `OTHERMANAGER[ACCESSION=B, CIK=A]`. Pendiente dictamen del auditor.
- **Estado:** PROPUESTA. No modifica contrato. No activa DROP_DUP.
  No toca reporting_dedup.py.
- **Hallazgo colateral:** Column 7 de INFOTABLE matchea
  OTHERMANAGER2.SEQUENCENUMBER (90% aprox), no OTHERMANAGER_SK.
  Requisito 2 y requisito 4 de L3 usan dos tablas distintas.
- **Reabrir si:** el auditor rechaza la reformulacion, o si el contrato
  14.3 requiere unificacion bajo una unica fuente.


## P66 - Dictamen externo #27 y propuesta v2 (2026-09-20)

- **Origen:** seguimiento del ciclo P66 (L3 reformulacion).
- **Dictamen #27:** GO CONDICIONADO. Fuente OTHERMANAGER aprobada.
  8 correcciones obligatorias antes del GO contractual definitivo.
- **Propuesta v2 redactada:** `iae/P66_L3_REFORMULACION_PROPUESTA.md`
  con las 8 correcciones aplicadas.
- **Correcciones clave:**
  - Terminologia: "evidencia estructurada cruzada entre filings".
  - Amendments: formalizar con CIK + PERIODOFREPORT + SUBMISSIONTYPE
    + ISAMENDMENT + AMENDMENTNO + AMENDMENTTYPE.
  - Identidad: CIK primario; FormNum fallback resoluble; CONFLICT
    si ambos contradicen.
  - Anadir CONFLICT al enum de estados.
  - DROP_DUP: solo con L3 completo (R1..R5) + R5 limpio.
- **Pendiente antes del GO contractual:**
  - Gate 0.3: granularidad de identidad (CIK/FormNum/ambos/ninguno)
    sobre NT Q4 2025 + Q1 2026.
  - Dictamen final sobre el texto exacto de 14.3.
- **NO se ha hecho:**
  - No se ha tocado NIPC_CONTRATOS_SEMANTICOS_v1.md.
  - No se ha tocado reporting_dedup.py.
  - No se ha activado DROP_DUP.
  - No se ha descargado XML de EDGAR.
- **Reabrir si:** el auditor emite dictamen final o solicita cambios
  adicionales sobre el texto contractual.


## P66 - Gate 0.3 granularidad de identidad (2026-09-20)

- **Origen:** correccion obligatoria #8 del dictamen #27.
- **Resultados:**
  - Cobertura 100% de NT en OTHERMANAGER.
  - Cardinalidad CIK <-> FormNum 1:1 perfecta (716/716 Q4; 696/696 Q1).
  - Composicion: 64.7% ambos, 4.5-5.0% solo CIK, 26.8-27.5% solo FormNum,
    3.5-3.6% ninguno.
  - 6.6% FormNum con prefijo `28-` en vez de `028-`. Normalizable.
  - Cero CONFLICT.
- **Conclusion:** tabla canonica `Form13FFileNumber -> CIK` construible
  1:1. Cobertura efectiva MATCH sube de 69.7% a ~96.5%.
- **Evidencia:** `iae/evidence/p66_gate03_granularidad/`.
- **Pendiente:** dictamen final del auditor sobre el texto contractual
  exacto de 14.3. Solo despues: tocar `NIPC_CONTRATOS_SEMANTICOS_v1.md`.
- **NO se ha hecho:** sin XML, sin reporting_dedup.py, sin DROP_DUP,
  sin tocar contrato.


## P66 - Gate 0.4 mapping FormNum -> CIK (2026-09-20)

- **Origen:** dictamen P66 v2, Bloqueo A.
- **Metodologia:** tabla canonica FormNum -> CIK vía COVERPAGE +
  SUBMISSION, join por ACCESSION_NUMBER.
- **Resultados:**
  - Cobertura del mapping: 97.81% Q4 / 98.24% Q1.
  - Cardinalidad 1:1 perfecta en ambas direcciones.
  - Normalizacion `28-` -> `028-` validada empiricamente.
  - Cobertura efectiva MATCH: 96.01% Q4 / 96.18% Q1.
  - N/D total (fail-closed): 3.99% Q4 / 3.82% Q1.
- **Conclusion:** BLOQUEO A RESUELTO. La tabla canonica es construible.
- **Hallazgo colateral:** FormNum `028-2813114` (7 digitos) anomalo.
  Volumen despreciable (6 filas). Documentado en evidencia.
- **Evidencia:** `iae/evidence/p66_gate04_mapping/`.
- **Pendiente:** BLOQUEO B (amendment probe). Requiere medir
  comportamiento de OTHERMANAGER en NT base vs NT/A RESTATEMENT vs
  NT/A ADDS NEW HOLDINGS.
- **NO se ha hecho:** sin XML, sin reporting_dedup.py, sin DROP_DUP,
  sin tocar contrato 14.3.


## P66 - Gate 0.5 amendment probe (2026-09-21)

- **Origen:** dictamen P66 v2, Bloqueo B.
- **Hallazgo principal:** los directorios del Data Set SEC estan
  organizados por fecha de presentacion, no por periodo objetivo.
  Contienen filings con multiples PERIODOFREPORT (48 en Q4 2025,
  77 en Q1 2026). Filtrar por directorio es incorrecto.
- **R3 formalizada:**
  - Filtrar filings por PERIODOFREPORT igual al periodo de analisis.
  - Ordenar amendments por AMENDMENTNO.
  - Semantica SEC: RESTATEMENT sustituye; NEW HOLDINGS complementa.
  - R4 se evalua sobre el filing efectivo.
- **Caso material:** CIK 0002056909. RESTATEMENT sustituye el
  OTHERMANAGER declarado (Prospector -> Gator Capital). Analizar
  sobre el base produce falso positivo.
- **Estadistica:**
  - Q4 2025 real: 1,900 base + 38 amend (37 RESTATEMENT, 1 NEW HOLDINGS).
  - Q1 2026 real: 1,907 base + 1 amend (NEW HOLDINGS).
- **Conclusion:** BLOQUEO B RESUELTO. R3 formalizada con semantica
  SEC completa.
- **Evidencia:** `iae/evidence/p66_gate05_amendment_probe/`.
- **Pendiente:** elevar al auditor el paquete completo (Gate 0.4 +
  Gate 0.5 + propuesta v2-bis). Esperar dictamen contractual final.
- **NO se ha hecho:** sin XML, sin reporting_dedup.py, sin DROP_DUP,
  sin tocar contrato 14.3.


## P66 - Dictamen v2-bis #28 y bloqueos pendientes (2026-09-21)

- **Origen:** dictamen del auditor externo sobre P66 v2-bis.
- **Resultado:** NO-GO contractual. GO tecnico condicionado a 6 bloqueos.
- **Aprobado:**
  - Fuente OTHERMANAGER como evidencia R4.
  - Separacion OTHERMANAGER / OTHERMANAGER2.
  - Arquitectura probatoria completa.
- **6 bloqueos:**
  1. Combination (13F-HR con REPORTTYPE = 13F COMBINATION REPORT).
  2. NEW HOLDINGS sobre OTHERMANAGER no probado.
  3. Gate 0.4 sin filtro PERIODOFREPORT.
  4. Mapping unidireccional FormNum -> CIK.
  5. Normalizacion no trunca (028-2813114 -> N/D).
  6. Nomenclatura MATCH -> IDENTITY_RESOLVED.
- **Reclasificacion:** Gate 0.5A PASS / Gate 0.5B PENDING.
- **Pendiente:** Gate 0.6 (Combination probe) + Gate 0.7 (NEW HOLDINGS)
  + Gate 0.4-reissue + propuesta v3.
- **NO se ha hecho:** sin XML, sin reporting_dedup.py, sin DROP_DUP,
  sin tocar contrato 14.3.


## P66 - Gate 0.6 Combination probe (2026-09-21)

- **Origen:** dictamen #28, bloqueo #1 (Combination excluido).
- **Resultado:** REPORTTYPE distingue limpiamente.
  - 13F HOLDINGS REPORT (13F-HR): 0% OM.
  - 13F COMBINATION REPORT (13F-HR, REPORTTYPE=13F COMBINATION): 100% OM.
  - 13F NOTICE (13F-NT): 100% OM.
- **Universo R4 corregido:**
  - Q4 2025: 2,334 (1,938 NOTICE + 396 COMBINATION). Delta +396 (+17.0%).
  - Q1 2026: 2,319 (1,908 NOTICE + 411 COMBINATION). Delta +411 (+21.5%).
- **Cobertura OM: 100.00%.** Sin casos anomalos.
- **Conclusion:** BLOQUEO #1 RESUELTO.
- **Evidencia:** `iae/evidence/p66_gate06_combination_probe/`.
- **Pendiente:** bloqueo #2 (NEW HOLDINGS probe / Gate 0.7),
  bloqueo #3 (Gate 0.4-reissue), bloqueo #4 (mapping unidireccional),
  bloqueo #5 (sin truncar 028-2813114), bloqueo #6 (nomenclatura).


## P66 - Gate 0.7 NEW HOLDINGS probe (2026-09-21)

- **Origen:** dictamen #28, bloqueo #2 (regla union prematura).
- **Resultado:** 3/3 NEW HOLDINGS en universo R4 con OTHERMANAGER
  identico al base. Cero cambios.
- **Regla contractual (fail-closed):**
  - Base: punto de partida.
  - RESTATEMENT: sustituye OTHERMANAGER (Gate 0.5, caso 0002056909).
  - NEW HOLDINGS: tratar como identidad. Caso divergente -> N/D o
    CONFLICT. NO asumir union ni sustitucion.
- **Muestra:** 3 casos sobre 4,653 filings. Insuficiente para
  confianza estadistica. Alternativa ofrecida al auditor: extender
  a 4-6 trimestres.
- **Conclusion:** BLOQUEO #2 RESUELTO provisionalmente con regla
  fail-closed.
- **Evidencia:** `iae/evidence/p66_gate07_newholdings_probe/`.
- **Pendiente:** bloqueo #3 (Gate 0.4-reissue), #4 (mapping
  unidireccional), #5 (sin truncar), #6 (nomenclatura).


## P66 - Gate 0.4-reissue con filtro PERIODOFREPORT (2026-09-21)

- **Origen:** dictamen #28, bloqueo #3.
- **Correcciones aplicadas:**
  - Filtro PERIODOFREPORT antes del JOIN COVERPAGE + SUBMISSION.
  - Normalizacion estricta (5 digitos exactos, sin truncar).
  - Nomenclatura MATCH -> IDENTITY_RESOLVED.
- **Resultados:**
  - Q4: cobertura 94.92% (era 96.01% sin filtro). Delta -1.09 pp.
  - Q1: cobertura 95.34% (era 96.18% sin filtro). Delta -0.84 pp.
- **Cardinalidad:** FormNum -> CIK 1:1 estricta (0 casos N:1).
- **Conclusion:** delta ~1 pp cuantifica la contaminacion cross-periodo
  del Gate 0.4 original. Regla corregida, cobertura >94%.
- **Bloqueos resueltos:** #3, #5, #6.
- **Evidencia:** `iae/evidence/p66_gate04_reissue_period/`.
- **Pendiente:** bloqueo #4 (mapping unidireccional) en propuesta v3.


## P66 - Propuesta v3 final (2026-09-21)

- **Origen:** cierre del bloqueo #4 del dictamen #28.
- **Correcciones aplicadas:**
  - Terminologia: "evidencia estructurada cruzada entre filings"
    (no "bidireccional").
  - Invariante del mapping: FormNum -> CIK unidireccional.
  - 0 CIK = UNRESOLVED; 1 CIK = IDENTITY_RESOLVED; >1 CIK = CONFLICT.
  - Nomenclatura: MATCH -> IDENTITY_RESOLVED.
  - Nueva seccion 10 "Invariante del mapping".
- **Los 6 bloqueos del dictamen #28 quedan resueltos:**
  - #1 Combination: RESUELTO (Gate 0.6).
  - #2 NEW HOLDINGS: RESUELTO (Gate 0.7, regla fail-closed).
  - #3 Gate 0.4-reissue: RESUELTO.
  - #4 Mapping unidireccional: RESUELTO (seccion 10).
  - #5 Sin truncar: APLICADO.
  - #6 Nomenclatura: APLICADO.
- **Pendiente externo:** reenviar propuesta v3 + evidencia al auditor
  para dictamen contractual final.
- **NO se ha hecho:** sin XML, sin reporting_dedup.py, sin DROP_DUP,
  sin tocar contrato 14.3.


## P66 - Dictamen #29 y bloqueo Gate 0.8 (2026-09-21)

- **Origen:** dictamen del auditor externo sobre propuesta v3.
- **Resultado:** GO CONDICIONADO.
- **Aprobado:** R4 con Combination, amendments fail-closed,
  mapping unidireccional, normalizacion sin truncar.
- **Bloqueo material nuevo:** Gate 0.8.
  - Gate 0.4-reissue midio identidad sobre `OTHERMANAGER(NT)`.
  - El universo R4 vigente es NOTICE + COMBINATION + amendments.
  - Auditor exige medir cobertura de identidad sobre el universo
    completo, separado por tipo y clase de amendment.
- **Correcciones textuales aplicadas:**
  - §2 R4: adoptado texto exacto del dictamen #29 §11.
  - §5.2: formulacion afinada de la cadena de amendments.
  - §10.4: reflejada aprobacion del bloqueo #4.
- **Pendiente:** Gate 0.8.
- **NO se ha hecho:** sin XML, sin reporting_dedup.py, sin DROP_DUP,
  sin tocar contrato 14.3.


## P66 - Gate 0.8 cobertura identidad universo R4 (2026-09-21)

- **Origen:** dictamen #29, bloqueo material.
- **Resultado:** cobertura R4 completa 91.70% (Q4) / 91.31% (Q1).
- **Delta vs NOTICE-solo:** -3.21 pp (Q4), -4.03 pp (Q1).
- **Drill-down:** 82-85% del N/D_null en COMBINATION_BASE se concentra
  en 2 filings del mismo filer (0001580642-). Sub-managers declarados
  sin CIK ni FormNum (NAME 100%, CRDNUMBER 85-88%, SECFILENUMBER
  64-69%).
- **Conclusion:** Gate 0.8 PASS. Mismo N/D fail-closed ya definido.
  Sin nueva clase de ambiguedad.
- **Evidencia:** `iae/evidence/p66_gate08_full_r4_coverage/`.
- **Estado:** paquete v3 completo. Listo para dictamen contractual final.
- **NO se ha hecho:** sin XML, sin reporting_dedup.py, sin DROP_DUP,
  sin tocar contrato 14.3.


## P66 - Dictamen #30 y cierre de especificacion (2026-09-21)

- **Origen:** dictamen del auditor externo sobre v3 con Gate 0.8.
- **Resultado:** NO-GO contractual. GO condicionado a 3 correcciones.
- **3 bloqueos aplicados:**
  - A: CONFLICT prevalece sobre MATCH.
  - B: NO_MATCH requiere resolucion completa de TODAS las filas
    OTHERMANAGER. Si existe N/D -> N/D.
  - C: >1 filing base -> N/D o CONFLICT, no orden arbitrario.
- **Correcciones textuales aplicadas:**
  - §3: definiciones reformuladas.
  - §4: precedencia explicita.
  - §5.2: regla de filing base unico.
  - §11: diagrama de decision.
- **Gate 0.8:** PASS confirmado por el auditor.
- **Pendiente:** Gate 0.9 (verificar existencia de >1 base real).
- **NO se ha hecho:** sin XML, sin reporting_dedup.py, sin DROP_DUP,
  sin tocar contrato 14.3.


## P66 - Gate 0.9 verificacion de >1 filing base (2026-09-21)

- **Origen:** bloqueo C del dictamen #30.
- **Resultado:** 1 caso en 4,596 grupos (0.02%).
  - Q4 2025: CIK 0002016827 con 13F-NT (2026-01-02) + 13F-HR
    COMBINATION (2026-02-20). Patron MIXED.
  - Q1 2026: 0 casos.
  - 0 casos de duplicidad estricta (MULTIPLE_NOTICE /
    MULTIPLE_COMBINATION).
- **Interpretacion:** evolucion documental. Mismo OTHERMANAGER
  (Vident Advisory) en ambos filings. El contrato lo clasifica
  como N/D fail-closed.
- **Conclusion:** la regla del bloqueo C esta justificada.
- **Evidencia:** `iae/evidence/p66_gate09_multiple_base/`.
- **Pendiente:** reenviar paquete completo al auditor para
  dictamen contractual definitivo.
- **NO se ha hecho:** sin XML, sin reporting_dedup.py, sin DROP_DUP,
  sin tocar contrato 14.3.


## P66 - Dictamen #31 y cierre de evidencia (2026-09-21)

- **Origen:** dictamen del auditor externo tras Gates 0.4-0.9.
- **Resultado:** NO-GO contractual. PASS en evidencia tecnica.
- **3 bloqueos normativos finales:**
  - 1: R3 con SUBMISSIONTYPE + REPORTTYPE.
  - 2: CONFLICT scoped a la evidencia candidata de A.
  - 3: 0 bases + amendments -> N/D, no False.
- **Terminologia:** "filing efectivo" reconocido como regla
  contractual IAE, no atribuida a la SEC.
- **Aplicado:** los 3 bloqueos al texto contractual + nota
  terminologica + diagrama actualizado.
- **Estado:** evidencia tecnica CERRADA (PASS por los 6 gates).
  Paquete completo listo para dictamen contractual final.
- **NO se ha hecho:** sin XML, sin reporting_dedup.py, sin DROP_DUP,
  sin tocar contrato 14.3.


## P66 - Dictamen #32 GO CONDICIONADO FINAL (2026-09-21)

- **Origen:** dictamen del auditor externo tras dictamen #31.
- **Resultado:** GO CONDICIONADO FINAL. Evidencia PASS. Sin nuevos
  gates ni dependencias.
- **2 correcciones aplicadas:**
  - A: §7 generalizado a universo R4 (NOTICE + COMBINATION).
  - B: R3 con tri-state TRUE / FALSE / N/D.
- **Trazabilidad:** cabecera actualizada con secuencia #28-#32.
- **Evidencia:** `iae/DICTAMENES.md` #32.
- **Pendiente:** dictamen contractual final sobre 14.3.
- **NO se ha hecho:** sin XML, sin reporting_dedup.py, sin DROP_DUP,
  sin tocar contrato 14.3.


## P66 - Dictamen #33 GO CONDICIONADO FINAL (2026-09-21)

- **Origen:** dictamen del auditor externo tras dictamen #32.
- **Resultado:** GO CONDICIONADO FINAL. Evidencia PASS definitivo.
- **2 correcciones aplicadas:**
  - 1: CONFLICT cubre CUALQUIER fila relevante para A.
    Prohibicion de seleccion selectiva (fila consistente + fila
    contradictoria -> CONFLICT).
  - 2: Nota de alcance: "filing efectivo" solo aplica a B/R3/R4.
    NO redefine R1/R2.
- **Trazabilidad:** cabecera actualizada a #28-#33.
- **Evidencia:** `iae/DICTAMENES.md` #33.
- **Pendiente:** dictamen contractual definitivo sobre 14.3.
- **NO se ha hecho:** sin XML, sin reporting_dedup.py, sin DROP_DUP,
  sin tocar contrato 14.3.


## P66 - Dictamen #34 y propuesta v4 consolidada (2026-09-21)

- **Origen:** dictamen #34 tras propuesta v3 con #33 aplicado.
- **Resultado:** NO-GO contractual. Evidencia PASS.
- **4 bloqueos aplicados:**
  - 1: FormNum normalizacion flexible (padding variable).
  - 2: Amendments con ISAMENDMENT != Y como criterio de base.
  - 3: candidate_A formalizado.
  - 4: L3 combinacion booleana explicita.
- **2 recomendadas aplicadas:**
  - OTHERMANAGER vacio -> N/D.
  - Correcciones editoriales (eliminadas v2/v3/v2-bis).
- **Cambio de proceso:** reescritura completa como v4 consolidada
  para romper el ciclo de refinamiento incremental.
- **Gate 0.10 ejecutado:** FormNum representation probe.
- **Evidencia:** `iae/DICTAMENES.md` #34 +
  `iae/evidence/p66_gate10_formnum_representation/`.
- **Pendiente:** dictamen contractual definitivo del auditor.
- **NO se ha hecho:** sin XML, sin reporting_dedup.py, sin DROP_DUP,
  sin tocar contrato 14.3.


## P66 - Dictamen #35 y propuesta v5 consolidada (2026-09-21)

- **Origen:** dictamen #35 tras propuesta v4.
- **Resultado:** NO-GO contractual. Evidencia PASS.
- **5 bloqueos aplicados en v5:**
  - 1: BASE_R4 global + >1 base R4 -> N/D.
  - 2: Validacion completa de la cadena de amendments.
  - 3: Estado INCONSISTENT -> CONFLICT. Identidad resuelta coherente.
  - 4: Prefijo FormNum {28, 028} estricto.
  - 5: NEW HOLDINGS consistente = igualdad exacta de conjunto.
- **1 ajuste menor:**
  - 6: Terminologia "canonicalizacion IAE" en §4.1.
- **Condicion de cierre del auditor:** v5 con estas 6 correcciones
  sin heuristicas nuevas = GO CONTRACTUAL.
- **Propuesta v5:** 699 lineas, 22,642 bytes. Pass interno OK.
- **Evidencia:** `iae/DICTAMENES.md` #35.
- **Pendiente:** enviar v5 al auditor.
- **NO se ha hecho:** sin XML, sin reporting_dedup.py, sin DROP_DUP,
  sin tocar contrato 14.3.


## P66 - Dictamen #36 y propuesta v6 candidata final (2026-09-21)

- **Origen:** dictamen #36 tras propuesta v5.
- **Resultado:** NO-GO contractual. Evidencia PASS.
- **3 cierres materiales aplicados:**
  - 1: BASE/AMENDMENT por SUBMISSIONTYPE+REPORTTYPE.
  - 2: Familia cerrada en cadena de amendments.
  - 3: NEW HOLDINGS determinabilidad estricta.
- **2 ajustes aplicados:**
  - 4: "secuencia sin huecos" etiquetada como regla IAE.
  - 5: correccion "Fin de la propuesta" a v6.
- **Condicion de cierre del auditor:** v6 sin heuristicas nuevas =
  GO CONTRACTUAL FINAL.
- **Propuesta v6:** candidata final.
- **Evidencia:** `iae/DICTAMENES.md` #36.
- **Pendiente:** enviar v6 al auditor.
- **NO se ha hecho:** sin XML, sin reporting_dedup.py, sin DROP_DUP,
  sin tocar contrato 14.3.


## P66 - Dictamen #37 y propuesta v7 ronda final (2026-09-21)

- **Origen:** dictamen #37 tras propuesta v6.
- **Resultado:** NO-GO contractual. Evidencia PASS.
- **3 correcciones aplicadas:**
  - 1: Flujo unico de R3 (§3.2, 4 pasos). BASE=0 + AMENDMENT>0 -> N/D.
  - 2: BASE_R4 sin `AND ISAMENDMENT != Y`. ISAMENDMENT como control
    de coherencia, no selector.
  - 3: Inspeccion global de familia en §3.2 PASO 3.
- **§0.5 Clausula de cierre:**
  - BLOQUEO MATERIAL: nueva version + dictamen.
  - RECOMENDACION DIFERIBLE: no bloquea.
  - GO de v7 autoriza traslado a 14.3.
- **Propuesta v7:** ronda final.
- **Evidencia:** `iae/DICTAMENES.md` #37.
- **Pendiente:** enviar v7 al auditor (ronda final).
- **NO se ha hecho:** sin XML, sin reporting_dedup.py, sin DROP_DUP,
  sin tocar contrato 14.3.


## P66 - Dictamen #38 y propuesta v7-bis ronda final (2026-09-21)

- **Origen:** dictamen #38 tras propuesta v7.
- **Resultado:** NO-GO contractual. 1 bloqueo material + 2 editoriales.
- **1 bloqueo material aplicado:**
  - §3.4 contradicia §3.2. Convertida en nota. Regla absoluta
    BASE > 1 -> R3 = N/D. CONFLICT no en R3.
- **2 correcciones editoriales:**
  - §9 trazabilidad #28-#38.
  - Fin v6 -> v7-bis.
- **1 observacion no bloqueante:** nota §9.2 sobre evidencia
  historica. READMEs de Gates no modificados. HASHES preservados.
- **Declaracion del auditor:** con esta correccion, v7-bis puede
  recibir GO CONTRACTUAL FINAL.
- **Propuesta v7-bis:** ronda final.
- **Evidencia:** `iae/DICTAMENES.md` #38.
- **Pendiente:** enviar v7-bis al auditor.
- **NO se ha hecho:** sin XML, sin reporting_dedup.py, sin DROP_DUP,
  sin tocar contrato 14.3.
