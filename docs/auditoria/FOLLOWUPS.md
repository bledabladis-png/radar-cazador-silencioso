# Follow-ups tecnicos registrados

## FU-001 — ffill multi-calendario en merge global (L352)

- **Origen:** C1, sesion 2026-09-12. Dictamen auditor.
- **Descripcion:** `stock_data_loader.py:352` aplica `ffill(limit=3)` sobre el DataFrame consolidado (Yahoo USA + Euronext + Xetra + BME). Ese ffill puede reintroducir valores imputados en tickers fuera del calendario NYSE, pese a haber sido clasificados como DATA_ISSUE en su etapa.
- **Evidencia:** tras el run C1 (2026-09-12), 20 tickers .L marcados como `DATA_ISSUE / MISSING_CLOSE_EXPECTED_SESSION` vuelven a mostrar `close[-1]==close[-2]` en el parquet final por el ffill de L352.
- **Impacto:** no afecta al pipeline USA actual ni invalida C1. Afecta a la coherencia de la marca DATA_ISSUE para tickers no-USA.
- **Clasificacion:** P2/P3 integridad tecnica localizada (auditor).
- **Accion:** disenar capa de proteccion por calendario (NYSE/BME/Xetra/Euronext) antes de generalizar la arquitectura a Europa.
- **Bloqueante:** no para C1/C2/C3. Obligatorio antes de declarar cerrada la nueva arquitectura de validacion multi-calendario.

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
