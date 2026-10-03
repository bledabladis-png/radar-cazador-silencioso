# 05 - BITACORA

**Registro de sesiones recientes. NO normativo. NO se pega al arrancar.**
**Para sesiones anteriores, ver `git log`.**
**Se poda por antiguedad: mantener las ultimas 5 sesiones. La 6ª se elimina o se resume en 04_HISTORICO.md.**

---

## 1. PROPOSITO

Bitacora de las ultimas sesiones. Contiene:
- Que se hizo.
- Que quedo pendiente.
- Commits asociados.
- Proximo paso sugerido.

**No es 04_HISTORICO.md.** El historico es tematico y estable. La bitacora es cronologica y se poda.

---

## 2. FORMATO DE UNA ENTRADA

Cada sesion se registra asi:

    ### YYYY-MM-DD — Titulo corto

    **Objetivo.** Que se pretendia hacer.

    **Hecho.**
    - Bloque 1.
    - Bloque 2.

    **Commits.** <lista de SHAs> (o rango).

    **Pendiente.**
    - Item 1.
    - Item 2.

    **Proximo paso sugerido.** Que deberia hacerse a continuacion.

Al cerrar una sesion nueva, se anade arriba (las mas recientes primero). Si hay mas de 10 fuera del dia en curso, la mas antigua se elimina o se resume en el historico. Ver reglas completas en §5.

---

## 3. SESIONES

### 2026-10-03 (tarde/noche) - D-06: validador 5b.X invalidado vs protocolo v2

**Objetivo.** Tests contractuales del validador 5b.X (P7 del handoff).
Bloqueo detectado antes de escribirlos: el script v1 no implementa el
protocolo v2.

**Hecho.**

- **D-06 detectado.** `scripts/validate_wyckoff_sow_5bX.py`
  (sha256 `CC01A6AB...`, commit `d324346`) implementa v1: 20 conf,
  `L > cutoff`, `12m+50ep`, escenarios por `lift_point`. El protocolo
  `35_protocolo_5bX.md` (v2, commit `9ef7497`) exige 550 confirmed
  H20-complete, `t0 >= cutoff`, escenarios por IC95%. El commit
  `ccba3ab` congelo el hash del script v1 bajo mensaje de
  sincronizacion v2.
- **Desalineaciones.** D-06.1 (20 vs 550), D-06.2 (L vs t0),
  D-06.3 (12m+50ep vigentes), D-06.4 (lift vs IC), D-06.5
  (H20_complete no verificado).
- **Dictamen externo 2026-10-03.** 5b.X original INVALIDADA como
  paquete ejecutable. Requiere 5b.X-bis/v3: nuevo script conforme,
  tests contractuales, nuevo sha256, revision de conformidad antes
  del freeze. No se reabre candidata ni umbrales.
- **Registro.** `docs/auditoria/wyckoff/37_d06_5bX_invalidada.md`.
- **Bloque de corpus (misma sesion).** Sincronizado `00_ARRANQUE.md`
  (tests 3024->3118, skips 2->5, nota de deuda del corpus);
  corregida la autorreferencia del generador `ESTADO_SISTEMA.md`
  (3 defectos: `datetime.now`, wall-clock de pytest, `%ad` vs `%ct`);
  quitados campos derivados del HEAD del fichero de estado;
  corregidas referencias a HEAD en el corpus.

**Commits.** 2b539b6 (00_ARRANQUE), 5e295ac+78bccce+d02e93f (generador),
b02ec17 (corpus), pendiente (D-06).

**Ampliacion (misma sesion, tramo 5b.X v3).**

- **Protocolo v3 redactado.** `35_protocolo_5bX_v3.md` (commit
  a7879f3). Preserva v2 + §14 auditoria de equivalencia + §15
  manifiesto + §16 tests exigidos.
- **Script v3 conforme.** `validate_wyckoff_sow_5bX.py` (commit
  d0874f4). 6 cambios normativos D-06, resto funcionalmente
  equivalente a 5b.4-bis.
- **Tests contractuales §16.** 19 tests (commit 0fd53fa + 42728ed).
  14 del contrato + C1 bootstrap + C2 guardrails.
- **Manifiesto de freeze.** `35_protocolo_5bX_v3.freeze.json`
  (commit d69e8e3). Rompe autorreferencia como el generador.
- **Dictamen final FIRMADO.** `38_dictamen_final_5bX_v3.md`
  (commit 927d7c5). revision_auditor=FIRMADO, estado=FROZEN.
  C1 (bootstrap equivalencia) y C2 (guardrails no bloqueantes)
  cerradas. Auditor explicita que la firma es sobre el paquete
  contractual, no sobre inspeccion visual del diff.

**Pendiente.**
- 5b.X v3: ciego. Solo se ejecuta cuando
  `n_confirmed_H20_complete >= 550` post-2026-10-02 (~2027).
  Script y protocolo inmutables (T7/T7b lo verifican).
- S-01-deep (`utils.py`, `instrument_registry.py`,
  `stock_data_loader.py`): candidato para proxima sesion.
- H5.3: cron 13F 20-nov-2026.
- C2 (920): bloqueado por auditor externo.

**Proximo paso sugerido.** S-01-deep, o H5.3 si llega la fecha.

**Ampliacion (misma sesion, tramo S-01-deep).**

- **`src/utils.py` (886 LOC).** D-07 detectado y cerrado (commit 59004c9).
  `confidence_from_range` con DataFrame devolvia 1.0 cuando una fila
  tenia 1 solo componente valido (rng=0). Contradecia el docstring
  ("<2 validos -> 0.5") y la version escalar `_confidence_range_row`.
  Fix: forzar NaN por fila donde `n_valid < 2`. Impacto real en
  `regimes/financial_conditions.py:124` (confianza del regimen
  financiero). 4 tests nuevos. Descartados como no-deuda: D-08-alt
  (`get_effective_meta` NaT, inalcanzable), D-09-alt
  (`clean_oil_prices` todos <=0, nulo en produccion).
- **`src/instrument_registry.py` (886 LOC, solo 4 funciones).**
  D-08 documental (commit c024c53). `resolve_symbol` tenia 3 casos
  reales y el docstring describia 1. Aclarado sin cambio de codigo.
- **`src/stock_data_loader.py` (866 LOC).** D-09 detectado y cerrado
  (commit 4c852a0). `last_expected_market_date()` sin argumento en
  las rutas de cache de `stock_data_loader.py:554` y
  `data_loader.py:280`, cuando la misma funcion ya habia calculado
  `_expected_session = last_expected_market_date(reference_date)` en
  linea 529. Divergia en runs post-medianoche (PUBLISH_HOUR). Fix:
  pasar `reference_date` explicito. Test AST nuevo que prohibe
  llamadas sin argumento en ambos loaders.

**Commits.** 59004c9 (D-07), c024c53 (D-08), 4c852a0 (D-09).

**Ampliacion (misma sesion, tramo S-02-deep).**

- **`src/market_calendar.py` (203 LOC).** Limpio. Calendario NYSE
  algoritmico con reglas observed correctas (verificado New Year
  sabado -> viernes anterior, adyacencia de anos, Juneteenth >=2022,
  Good Friday). Descartado D-10-alt (PUBLISH_HOUR=23 fijo falla ~1
  semana/ano en la transicion CET/CEST; limitacion documentada, no
  bug). Descartado D-10-alt2 (`_is_nyse_holiday` comprueba `d.year-1`
  innecesariamente; coste de un lookup extra, sin efecto).
- **`src/market_hours.py` (228 LOC).** Limpio. Fuente unica
  `config/market_close_regular.csv` con fail-loud al import.
  `is_session_closed` maneja las 3 ramas (pasado/presente/futuro) sin
  bug. Descartada redundancia (doble `_validate_market` via
  `is_trading_session`); no es bug.

**Commits.** Sin commits funcionales (auditoria sin hallazgos
materiales).

- **`src/data_loader.py` (377 LOC, resto post D-09).** Limpio.
  `_ticker_list`, `_is_equity_ticker`, `_filter_non_eod_equity`,
  `_trim_market_data_to_equity_eod`, `_postprocess_market_data`,
  `_check_khuerfano`, `download_market_data` verificados sin bug
  material. D-10-alt (orden no determinista en `list(set(tickers))`)
  descartado: conjunto total identico, downstream no depende de orden.
  Fallback `datetime.now()` en L253 documentado.

**Commits.** Sin commits funcionales (auditoria sin hallazgos
materiales).

**Ampliacion (misma sesion, tramo verificacion de arranque + H5.3).**

- **Verificacion de arranque completa.** HEAD `29f43c1` (posterior
  al handoff `8661d1a`, avance docs). Suite 3142+5+0, pyflakes y
  compileall limpios. IAE: 36 ficheros, 8441 LOC; mappings con sha256
  intactos; evidencia empirica en 8 directorios.
- **Discrepancia arranque<->fuente, corregida (commit f0fad01).**
  `00_ARRANQUE.md:82` declaraba "Cobertura configurada: 313/313
  tickers" sin yaml/constante/test que lo respaldara. `git log -S
  '313/313'` solo da `073b2fe` (commit fundacional del corpus,
  2026-09-28). 313 es el universo USA de `stock_prices` observado al
  2026-09-22 (documentado en `04_HISTORICO.md:498`,
  `05_BITACORA.md:547`, `scripts/health_check.py:46`). Reformulado
  a "Universo USA (stock_prices): 313 tickers (2026-09-22).
  Ver `04_HISTORICO.md`."
- **Umbral 10 KB explicitado.** `01_METODO.md:264` decia "(<=10 KB)"
  ambiguo. Fijado a "(<=10 KB = 10240 B)". Arranque queda en 10037 B.
- **Lecciones 2026-10-03 persistidas** en `01_METODO.md` (commit
  8a640ce): seccion 6 +5 bullets PowerShell/parsers; seccion 7 +3
  reglas de auditoria (protocolo<->script, autorreferencia, tests
  centinela).
- **H5.3 verificado contra codigo.** Cadena
  workflow<->script<->runbook alineada 9/9:
  `update_sec_13f.yml` step `id: source` decide `cron`/`dispatch`
  por `github.event_name`; pasa `--source` al script;
  `_write_ingest_trace` escribe JSONL con
  `{ts, quarter, source, actor, outcome}`; outcome en
  {INGEST, SKIP}; workflow commitea `ingest_traces.jsonl`. Runbook
  seccion 2.4.1 describe exactamente ese flujo. Sin desalineaciones.
  Pendiente unico: ejecucion del cron real 20-nov-2026 06:17 UTC.
- **`ESTADO_SISTEMA.md` regenerado** (commit ba91508): tests
  3118 -> 3142, sin campos derivados del HEAD (coherente con la
  regla de autorreferencia).

**Commits.** ba91508, f0fad01, 8a640ce.

**Ampliacion (misma sesion, tramo S-03-deep - 6 ficheros).**

- **S-03-deep arrancado.** Alcance: `src/report/` (20 ficheros),
  `src/pipeline/` (18), `regimes/` (8). Fuera: `indicators/` (frente 7
  CERRADO) y `src/temporal_contracts/` (A1 CERRADO). El handoff
  original escribio `src/regimes/` y `src/indicators/` como rutas,
  pero viven en la raiz del repo, no bajo `src/`.
- **Primer bloque: ficheros con mtime 2026-10-03 (D-05/D-04).**
  5 ficheros, 738 LOC. Cero ALTA, cero MEDIA. 1 BAJA corregida
  (FS-1: mtime sin UTC en `flows_secondary.py`, commit `637d78a`).
- **Segundo bloque: `iae_section.py`.** 216 LOC, mtime 2026-10-02.
  Cero ALTA, cero MEDIA. 2 BAJA + 5 INFO, todos WONT FIX razonado.
- **Total S-03-deep:** 6 ficheros, 954 LOC, ~15% del alcance.
  Cero ALTA, cero MEDIA, 1 BAJA corregida, 14 INFO/BAJA WONT FIX.
- **Audit docs:** `docs/auditoria/s03_deep/{validation_gate.md,
  mte_confirmation.md, d05_d04_block.md, iae_section.md}`.

**Commits.** `dee7edf`, `637d78a`, `f368fa9`, `e869446`, `779a8f5`.

**Pendiente actualizado.**
- 5b.X v3: ciego hasta 2027 (sin cambios).
- H5.3: verificado contra codigo 2026-10-03. Pendiente ejecucion cron real 20-nov-2026.
- C2 (920): bloqueado por auditor externo.
- S-02-deep: cerrado sin hallazgos materiales en sus 3 ficheros.
- Cobertura total: `src/report/`, `src/pipeline/` pendientes si se
  quiere exhaustividad.
- S-03-deep: 6/38 ficheros auditados. Pendiente: src/report/ (20), src/pipeline/ (13 restantes), regimes/ (8).

**Proximo paso sugerido.** Cierre de sesion. S-01-deep + S-02-deep
completan la auditoria de los 5 ficheros grandes de src/.

### 2026-10-03 - Cierre Wyckoff (umbral 550) + auditoria S-01 (D-01, D-02)

**Objetivo.** Handoff desde asistente saliente. Cerrar el frente Wyckoff
hasta donde permiten los bloqueos externos + arrancar auditoria S-01
sobre `indicators/` no-Wyckoff.

**Hecho.**

- **Handoff procesado.** Dossier de 20 preguntas al saliente, con 6
  gaps detectados en el plan v2 y config. Commits 17773df (8 gaps),
  7195b96 (regenerar ESTADO_SISTEMA).
- **Cierres documentales del backlog saliente.** W-07 (renumerar salida
  5b.X 36->37), W-08 (rename `compare_wyckoff_legacy_v13.py` a
  `_v18.py`), S-04 (documentar H5.3 en `07_RUNBOOK.md`). Commits
  39c9c06, ddaa3c5, dbe72a0.
- **Dictamen externo 5b.X recibido.** El auditor audito 35 v1 + QA 36
  y dictamino reescritura completa a v2: `t0 >= cutoff` (no `L >=`),
  H20-complete (no landmark), clasificacion A/B/C por IC (no por
  lift_point), D como estado de no ejecucion, congelacion del
  validador con hash+commit antes de OOS.
- **Power analysis implementado (`scripts/power_analysis_5bX.py`).**
  Simulacion cluster-bootstrap real reutilizando el calibrador
  5b.4-bis (load_dataset, precompute_features, compute_sow_cache,
  build_episodes_landmark, episode_metrics, bootstrap_lift_H20).
  Grid lift_real {0.03,0.05,0.074,0.10,0.15} x n_conf {100..800}
  x N_SIM=500. Commit 344dae9.
- **Umbral 550 confirmed H20-complete congelado.** Decision ex-ante
  basada en el primer n donde el CI95 lower de la potencia cruza
  0.80. Para lift=0.074 (observado en desarrollo): n=450 da lower
  0.780, n=550 da lower 0.806. Ref: `wyckoff_5bX_power_summary.json`.
- **Protocolo 35 v2 escrito.** 433 lineas, +227/-65, changelog
  explicito v1->v2 en §13. Commit 9ef7497.
- **Congelacion del validador.** sha256 CC01A6AB..., commit d324346,
  python 3.14, seed 20261002, B=2000. Registrado en §2.5 de 35 v2 +
  sincronizacion de plan v2 (§4.4, §4.5, bitacora, riesgos). Commit
  ccba3ab.
- **Auditoria S-01 (indicators/ no-Wyckoff).** Dos barridos de patrones
  del catalogo 01b:
  - Barrido 1 (patrones P3, P6, P13): D-01 detectado y cerrado.
  - Barrido 2 (patrones P1, P2, P4, P8, P13 fino): 0 bugs.
- **D-01 cerrado.** `darkpool_scoring.robust_zscore` propagaba NaN
  via `np.median` sobre MAD. Mismo patron que el fix historico de
  index_leaders (2f956ed) y que los hermanos options.py y fls.py,
  ya arreglados. Fix + test de regresion (6 casos). Commit 50787d3.
- **D-02 cerrado.** `index_leaders.compute_stock_metrics_for_index`
  usaba inline `np.median(np.abs(x - np.median(x)))` sin filtrar NaN.
  Extraido a `_mad_filtrado` de modulo (mismo contrato que
  stock_leader). Test de regresion (5 casos). Commit 013618e.
- **S-02 (auditoria src/report + src/pipeline).** Tres barridos de
  patrones (P1-P13 del catalogo 01b):
  - **D-04 cerrado.** `flows_secondary.py:88` hacia `_ref = reference_date`
    sin `.replace(tzinfo=None)`. `run.py:43` construye tz-aware, el
    caller no propagaba, TypeError capturado silencioso -> skip de
    `qqq_performance_data`. Fix + propaga + test. Commit d608711.
  - **D-05 cerrado.** Mismo patron en `mte_confirmation._compute_mte:36`
    y `validation_gate:39`. `compute_mte_confirmation:152` no propagaba
    `reference_date` a `_compute_mte`. Fix: normalizar tz + propagar +
    run.py callers. 3 tests. Commit e0aacda.
  - **S-02b/c/d cerrados sin bug.** `helpers.py` except defensivos,
    `breadth_metrics.py:107` no tiene resta tz (robustez menor),
    `engines.py:25` es bug latente ya citado en el catalogo 01b.
    Ver detalle en el acta del barrido (no documento).

**Commits.** f701350..013618e (10 commits). Rango:
`17773df`, `7195b96`, `39c9c06`, `ddaa3c5`, `dbe72a0`, `344dae9`,
`9ef7497`, `ccba3ab`, `50787d3`, `013618e`.

**Pendiente.**
- 5b.X: esperando datos con t0 >= 2026-10-02 + 550 confirmed
  H20-complete + guardrail 12m. Sin accion hasta que lleguen datos.
- 5c.4 / 5d / 5e: bloqueados en cadena por 5b.X.
- H5.3: verificacion del cron 13F del 20-nov-2026 (`update_sec_13f.yml`).
  Comando documentado en `07_RUNBOOK.md` §2.4.1.
- C2 (920): discrepancia H1-B. Bloqueado por auditor externo.
- S-01: segundo barrido sin hallazgos. No se justifica tercer barrido
  con los patrones restantes (P5, P6, P9-P12) — ROI < 1.
- D-03 (robustez menor): `momentum.py:47` `mfv.rolling(w).sum() /
  volume.rolling(w).sum()` sin `+1e-9`. Sin caso real disparador.

- S-02c (robustez menor): `breadth_metrics.py:107` fallback a
  `datetime.now()` cuando `reference_date=None`. Caller productivo
  (`run.py:140`) lo pasa, no hay bug funcional.

**Proximo paso sugerido.** Esperar dictamen / datos. Frente Wyckoff
completamente bloqueado hasta que existan datos post-cutoff. Alternativa
si se quiere avanzar: S-02 (`src/report/` y `src/pipeline/` con el
mismo metodo del Frente 7).

---

### 2026-10-02 (tarde/noche) - Wyckoff 5b.3 -> 5b.4 -> 5b.4-bis + QA + contrato v1.9

**Objetivo.** Continuar el rediseno del modulo Wyckoff. El usuario
reporto media implementacion en `docs/auditoria/wyckoff/` (27 ficheros)
con codigo en `indicators/wyckoff_v1.py` (v1.7) sin integracion en
pipeline.

**Hecho.**

- **Diagnostico inicial.** Verificado que `wyckoff_v1.py` NO esta
  integrado en el pipeline. Consumidores productivos
  (sector_wyckoff_distribution, index_leaders, index_phase,
  sector_breadth, stock_leader) siguen con `indicators/wyckoff.py`
  legacy v4.2. Solo scripts de analisis importan wyckoff_v1.
  Implica: local-first, E2E relajado hasta migracion.

- **Expediente 21 + contrato v1.8 + fail-closed.** El contrato v1.7
  era cosmetico (cuerpo arrastraba v1.3 + SOW v1.6). Se redacto
  expediente 21 (A1-A8) + dictamen externo. Contrato v1.8 nuevo con
  seccion SOW corregida. `detect_sow` pasa a fail-closed (kwarg-only
  obligatorio, ValueError sin args). `classify_wyckoff_phase` recibe
  `sow_params` opcional; sin el, no emite DISTRIBUTION (I36).

- **5b.3 ejecutada (639fe92).** Grid 240 con protocolo v3. FAIL
  formal: D3=0/240. Pero lift positivo universal 240/240.

- **Dictamen 24 (ef710e0).** **Inmortal time bias detectado.**
  Confirmed anclado en t_sow, baseline en t0. Relojes distintos.
  El +0.192 de 5b.3 no es estimacion valida. Requiere landmark.

- **5b.4 ejecutada (04e0620).** Landmark L=t0+M corregido. D3
  inalcanzable (bloque A max 12 confirmed, D3 pide 4 bloques
  evaluables). Lift real +0.064 (no +0.192). Heterogeneidad A/B
  negativos, C/D positivos.

- **Dictamen 26 (709a3d3).** Core landmark aprobado. 10 correcciones
  obligatorias al protocolo. Bloqueo: no hay holdout ciego.

- **5b.4-bis ejecutada (848ee23).** 3 bloques P1/P2/P3 (~20 meses).
  17/240 pasan D1-D3 por primera vez. Candidata N=60, M=30,
  X_ATR=0.25, Y_VOL=1.10. Firma +++ en los 3 bloques.
  lift=+0.0740, lower_CI=+0.0234.

- **Dictamen 33 (dacc7b2).** Freeze de validacion autorizado, freeze
  productivo NO. La firma +++ es consistencia de signo, no
  homogeneidad. Antes del freeze definitivo: IC por bloque + QA.

- **QA 5b.4-bis (34, dacc7b2).** IC por bloque:
    P1: +0.093  IC [-0.099, +0.296]  (cruza 0)
    P2: +0.050  IC [-0.024, +0.122]  (cruza 0)
    P3: +0.106  IC [+0.038, +0.178]  (NO cruza 0)
  Solo P3 es estadisticamente informativo. El +0.074 del ALL esta
  arrastrado por P3. Concentracion por ticker sana (top5=4.4%).

- **Contrato v1.9 (6f4a988).** Candidata SOW FROZEN_FOR_VALIDATION /
  NOT_PRODUCTION. `config/settings.py` sigue con los 4 SOW a None.
  fail-closed intacto.

- **Script 5b.X (d324346).** validate_wyckoff_sow_5bX.py listo,
  estado BLOCKED (sin datos post-2026-10-01).

- **QA precondiciones 5b.X (d82beea).** Los umbrales del protocolo
  5b.X (12 meses + 50 ep + 20 conf) son insuficientes para poder
  estadistico. Con lift ~+0.074, 12 meses dan lower_CI ~ -0.04.
  Se necesitan ~200 confirmed (~24-36 meses). Variabilidad temporal
  enorme (0 a 85.8 starts/mes entre sub-bloques).

- **Plan migracion v2 (dc5da26).** Reescritura con estado real.
  Frentes A-F. Correccion posterior: 5c desbloqueada desde 5b.2.

- **Frente F (QA interno).** F-01 no-deuda (rolling min_periods),
  F-02 fix (except acotado en classify_wyckoff_phase_meta),
  F-03 cobertura (8 tests directos de funciones puras).

- **Limpieza pyflakes (1ba2d1b).** 5 scripts de analisis sin
  warnings.

- **5c general (824ef72).** Comparativa legacy vs v1.8 re-ejecutada.
  Resultados identicos a v1.3 (componentes no-SOW no cambiaron).

**Commits.** 936340c, aabc921, e66cf41, 639fe92, ef710e0, 709a3d3,
9fb6bcf, 04e0620, 4595676, 848ee23, dacc7b2, 6f4a988, d324346,
d82beea, dc5da26, a00ee2c, aaf7923, 1ba2d1b, 824ef72.

**Pendiente.**

- Dictamen externo sobre umbrales de 5b.X (expediente 36).
- 5b.X bloqueada hasta datos post-2026-10-01 (~12+ meses).
- 5c.4 / 5d / 5e bloqueadas.
- Cosmetico: rename script compare_wyckoff_legacy_v13.py.

**Proximo paso sugerido.** Pasar 35 + 36 al auditor externo. Despues,
si el dictamen ajusta umbrales, actualizar protocolo 5b.X a v2.
Frentes internos no bloqueados: revisar `05_BITACORA` (poda),
`01_METODO` (division 20 KB), cubrir otros indicadores.

### 2026-10-02 (madrugada) - Catalog PIT: 242 -> 255 + schema v5 + validator productivo

**Objetivo.** Tras el incidente del cron trimestral (1-oct), resolver el
siguiente frente abierto: el subsistema catalog PIT estaba roto por
alineamiento posicional entre snapshot y assignments.

**Hecho.**

- **Expediente + dictamen externo.** Redactado expediente completo
  (`docs/auditoria/iae/catalog_pit_snapshot_growth_expediente.md`) y
  recibido dictamen (`..._dictamen.md`). Decision clave del auditor:
  `catalog_key` identifica; `radar_ticker` describe. Biyeccion por
  version activa, no global. Ancla de reconciliacion: share_class_figi.
  Ausencia puntual != retiro. Membership acumulativo. Commit 6bdb411.

- **Contencion CI (a7c96e2, 018cf42/4e16edf).** Snapshot 255 revertido a
  242 en el manifest + pausa del step `Regenerar catalogo radar` en
  daily_run.yml mientras se implementaba el fix. Publicado (037c7ed..4e16edf).

- **Paso 1 - Verificacion cruzada.** Binding de los 242 keys reconstruido
  por dos caminos independientes (posicion + snapshot_row_uid via
  membership). PASS 242/242 sin mismatches.

- **Commit A (e61689e).** Schema v5: `radar_ticker` y `share_class_figi`
  en catalog_assignments.csv; `radar_ticker` en catalog_membership.csv.
  `build_catalog_csvs.py` reescrito con algoritmo no posicional
  (merge_assignments por figi). Migracion v1->v2 usa membership como
  fuente autoritativa. Tests reescritos como invariantes + tests nuevos
  (binding no-posicional, crecimiento preserva keys, rename misma key,
  cambio figi nueva key, membership acumulativo).

- **Commit B (00bf59e).** Crecimiento 242 -> 255 real. 13 altas:
  ANET EL EMR GRMN HON HOOD HST MRVL PPL RDDT TGTX TXG USB.
  Fix: `assigned_entity_id` pasa a contador global monotono
  (radar_entity_0243..0255). Sin esto, las altas reutilizaban
  radar_entity_0001..0013.

- **Commit C (fe7dc74).** 3 fixes:
  - `catalog_key._vid_norm`: normaliza version_id quitando guiones.
    Coexisten formatos 20260922_01 y 2026-10-01_01; el `<` string
    invertia el predecessor chain.
  - `build_catalog_csvs.backfill_missing_versions`: recorre el
    manifest por valid_from ASC y solo pobla prev_uid_by_key con
    versiones ya procesadas. Fallback ticker_to_key cuando el figi
    cambia (rename o requery OpenFIGI). Backfill de 20260921_01
    (242 filas) -> membership 739 filas.
  - `pipeline_contractual.version_id`: se toma del manifest
    (valid_to=null), no de mem['version_id'].iloc[0].

- **Commit D (288fd7a).** Validator productivo: `validate_membership`
  invocado como PASO 0 de pipeline_contractual antes de build_target.
  0 errores sobre datos reales.

- **Commit E (225c09b).** Vista radar_target_catalog.csv regenerada
  desde snapshot 2026-10-01_01 (255 filas).

- **Limpieza (586121f).** pyflakes limpio.

**Commits.** `6bdb411`, `a7c96e2`, `4e16edf`, `e61689e`, `00bf59e`,
`fe7dc74`, `288fd7a`, `225c09b`, `586121f`.

**Estado.** assignments 255, membership 739 (242+242+255).
validate_membership: 0 errores. Suite 3024 passed + 2 skipped.
pyflakes LIMPIO.

**Nota sobre nipc.** El nipc_total del par vigente Q1/Q2 pasa de
8256882557 a 8168484363 por las 11 altas de los 13 (TGTX/TXG ausentes)
que ahora SI contribuyen al calculo. Es el objetivo del fix. El
baseline C2 (920) del par Q4/Q1 no se reproduce con los datos actuales
(-4264449012 congelado vs -4286796474 real); es residual conocido,
bloqueado por auditor externo.

**Pendiente.**

- C2 (920): bloqueado externo.
- H5.3: cron nov 2026.
- Regenerar 01_METODO (24 KB, umbral 20 KB).

- **Verificacion CI (2026-10-02 00:11 CEST, run 36942867498).**
  Primer run real que llega a los steps finales desde el fix del catalog
  PIT. Resultado: Run tests OK, Regenerar catalogo radar OK, Verify
  leader selection OK. Fallo unico: `Guard coverage` con
  `coverage_pct_last=0.20` en stock_prices. NO es regresion del fix:
  es latencia Yahoo (dispatch a las 22:11 UTC, 1h post-cierre USA, Yahoo
  aun no habia publicado el cierre del 1-oct). El run verde previo
  (36904431677, 18:08 UTC) paso Guard porque el gate calculo
  expected_session=2026-09-30 y Yahoo ya tenia datos completos. Deuda
  separada: el guard no distingue "latencia Yahoo" de "fallo real" —
  candidato a expediente propio si se decide atacar.

- **Guard coverage sigue fallando (2026-10-02 02:45 CEST, run 36946066084).**
  El fix PENDING/INVALID (937c0b4, 925e477) es correcto pero no aplica:
  la causa real es que Yahoo no ha publicado NINGUN cierre del 1-oct
  a las 02:28 CEST del 2-oct (ni USA ni EU). El log muestra
  `[K-HUERFANO] CAT/XOM/GOOGL/PFE/^FTSE/^IBEX/^STOXX50E: sin Close en
  expected_session=2026-10-01`. cobertura 64/316 = 0.20.
  El guard hace lo correcto: bloquea el commit con datos incompletos.
  Es latencia anomala de Yahoo (5h post-cierre USA, 9h post-cierre EU),
  no un bug del sistema. El diseno ya preve esto: los 5 slots diarios
  cubren ventanas de latencia. Proximo slot automatico: 03:17 UTC.
  No seguir disparando manuales en bucle: daran lo mismo hasta que
  Yahoo publique. Si el slot automatico tambien falla, abrir expediente
  propio de latencia Yahoo (out of scope de esta sesion).

  Correccion sobre la nota anterior: la hipotesis "Guard vs latencia
  Yahoo (solo europeos)" era incorrecta. Hoy la latencia afecta a
  todos los mercados simultaneamente. El fix PENDING sigue siendo
  correcto para el caso europeos-only, pero la causa de hoy es mas
  amplia.

- **Evidencia empirica (2026-10-02 02:48 CEST).** Prueba directa con
  yfinance sobre 7 tickers (AAPL, MSFT, XOM, GOOGL, ^FTSE, ^IBEX,
  ^STOXX50E). Resultado: 6/7 tienen fila del 2026-10-01 pero con
  close=NaN. ^STOXX50E ni tiene fila. Yahoo reconoce el dia como
  sesion cerrada, pero no ha publicado el precio. Confirmado: no es
  bug del sistema, es latencia del proveedor.
  Correccion al fix PENDING (937c0b4): es correcto para el caso
  europeos-only, pero hoy afecta tambien a USA. Un guard robusto a
  latencia tendria que reconocer "sesion cerrada, precio pendiente"
  para cualquier mercado, no solo los de MARKETS_WITH_PUBLICATION_LAG.
  Decision de diseno pendiente, sin urgencia. Plan: esperar el slot
  automatico 03:17 UTC (05:17 CEST) del 2-oct.

- **P5 auditoria tests anclados a estado: CERRADO como no-deuda (2026-10-02).**
  Barrido completo de tests que abren ficheros del repo. Resultado: 7
  candidatos, 0 anclados a estado real.
    - test_iae_section.py:124,138,139 -> mocks del pipeline, no ficheros.
    - test_build_ticker_df_i2.py:41,52 -> df sintetico con seed.
    - test_c2_temporalidad.py:187 -> verifica que NO se modifica el CSV
      real (anti-anclaje).
    - test_european_coverage.py:173 -> tmp_path + providers mockeados.
    - test_provider_nport.py:65 -> regresion estatica sobre el codigo
      del provider.
    - conftest.py:14 -> fixture con pytest.skip si el parquet no existe
      (solucion al problema CI).
    - test_holdings_filter.py -> funcion pura con parametrize.
  El patron del sistema de tests es correcto. El caso test_idempotencia
  era unico, no un patron. Ya reescrito en e61689e.

- **Auditoria de ceros del reporte: CERRADA (2026-10-02).**
  Barrido de las 919 lineas del reporte (outputs/report/reporte_diario.md
  del 2026-10-01 01:41). Tres categorias:
    1. Ceros reales (FEZ, NH/NL=0, Matriz Evidencia columna 0, N-PORT
       cambio 0 con balance identico): correctos sin signo.
    2. Ceros de formato en 29 sitios con f:+.Xf o _fmt_num(v,{:+.Xf})
       crudo: violaban FU-003a. Corregidos en d4565ab (aplicado
       _fmt_signed en confirmation, etf_flows, flows_international,
       leaders, rankings, sector_context, slpm).
    3. N/D correctos donde el dato falta.
  Caso FEZ verificado con evidencia: SSGA publica NAV diario, shares
  constantes en 63,300,967, primary_flow_usd=0 calculado (no
  imputado). Cero real, no bug.
  Regla reforzada: cero != sin dato. Un cero sospechoso se investiga;
  aqui el unico caso con apariencia sospechosa (FEZ) resulto ser real.

- **P4 politica de bajas: CERRADO como no-deuda + defensa preventiva
  (2026-10-02).** El catalogo es monotono por diseno: regenerate_
  radar_catalog._build_updated_catalog hace concat([catalog_df,
  nuevas_filas]) sin eliminar filas. Historial: 242 -> 242 -> 255.
  Cero bajas desde el inicio. El contrato A1 v5 no necesita politica
  de bajas hoy: no hay mecanismo de baja.
  Defensa: merge_assignments ahora detecta figis de asg_prev ausentes
  en el snapshot vigente y emite [WARN] si ocurre (no deberia bajo
  monotonia). Falla visible en lugar de silencio. Test en
  tests/test_catalog_bajas_p4.py (2 casos).
  Si algun dia se decide permitir bajas (delisting, cambio de
  universo), se hara con flujo administrativo manual: cerrar key con
  valid_to, no reactivar, reaparicion -> nueva key (dictamen auditor
  seccion 4).

- **Auditoria bloques 4-5 del reporte: CERRADA (2026-10-02).**
  Lectura linea a linea de L800-1099 (Rendimiento QQQ, Flujo QQQ,
  MTE, Confirmation Data, Cross-Asset Ratios, Dark Pools FINRA,
  Indices Internacionales, Sintesis, Matriz Evidencia, Anti-Double-
  Counting, IAE/13F). 4 hallazgos, 2 bugs reales:

  BUG 1 (1edefb1): Cobertura catalogo decia 240/242 (99.17%). El
  snapshot vigente tiene 255 keys. Causa doble: run_contractual_nipc
  no devuelve catalog_keys_total, iae_section caia al fallback
  hardcoded 242. Fix: _get_catalog_total() lee 'rows' del snapshot
  vigente del manifest.

  BUG 2 (1edefb1): NH-NL en Confirmation Data con ':+d' crudo ->
  '+0' con signo. Mismo bug latente en ad_net. Fix: _fmt_signed.

  Descartados por diseno (no son bugs):
  - 535/535 tickers en Dark Pools: universo FINRA ATS != universo
    stock_prices (313 USA). Distintos por definicion.
  - CBOE PCR (09-29) vs Estructura volatilidad (09-30): dos fuentes
    del mismo dataset con lags de actualizacion distintos.

  Suite tras fix: 3042 passed + 2 skipped.

- **Revision adicional bloques 4-5: 1 fix, 1 no-deuda (2026-10-02).**
  Segunda pasada sobre el reporte fresco del CI (run 36954815655,
  2026-10-02 04:18).
  FIX (commit siguiente): _fmt_num normaliza ruido numerico -0.00 a
  0.00. Afectaba Concentracion del liderazgo y otras tablas con
  medianas (rs_median, momentum_median). Mismo patron que FU-003a pero
  para fmt sin signo.
  NO-DEUDA documentada: '+0' en columna A/D de Sector Breadth & Health
  cuando advances=declines>0. NO es violacion de FU-003a: politica B3
  (2026-09-12) distingue explicitamente '+0' (balance real con datos)
  de 'N/D' (0/0 sin informacion direccional). Tests C3 lo cubren.
  Se revirtio un cambio inicial en _fmt_ad_net que quitaba el '+'.
  Spring/SOS vacios en tablas de lideres por sector: no es bug. '[v]'
  marca cuando spring/sos==1; vacio cuando 0. Documentado en
  test_leader_table_markdown.py.

- **Auditoria completa del reporte (8 fases, 2026-10-02): CERRADA.**
  Revision linea a linea de las 954 lineas del reporte CI (run
  36954815655, 2026-10-02 04:18) cruzando cada seccion con el modulo
  que la genera. Resultado: 6 fixes en 2 commits + 3 no-deuda
  documentados + 2 cosmeticos pendientes.

  FIXES aplicados:
  1. _fmt_num normaliza ruido -0.00 (8e1837b). Aplicado a
     Concentracion del liderazgo, Dispersos y demas tablas con
     medianas.
  2. CFTC flow_z (-0.00 en VIX lev_money) con _fmt_signed
     (a21611f).
  3. FLS desglose (RRP -0.00) con _fmt_num (a21611f).
  4. SPDR 'Ultima fecha: N/D': el df de get_etf_primary_flow_data
     no incluia 'Date'; el render hacia KeyError silencioso a N/D
     (a21611f).

  NO-DEUDA verificados:
  - Breadth EMA20/EMA50 = 2/11: correcto (umbral interno >=40% en
    sectors_base.py:152-154).
  - Concentracion Lider vs primera fila: correcto (distinto criterio
    de ordenacion, documentado en nota al pie).
  - Spring/SOS vacios en tablas de lideres: [v] solo si flag==1,
    vacio si 0. Documentado en test_leader_table_markdown.py.
  - ISF.L 'Flujo Estimado (GBP)' lee estimated_flow_eur: nombre
    historico compartido en _blackrock_base.py con DAXEX. Etiqueta
    GBP correcta (ISF cotiza en GBP). Renombrar romperia DAXEX.
  - QQQ SEC flow, IWM flow, N-PORT cambios 0: cifras verificadas a
    mano, coherentes.
  - LIS +0.43 verificado con formula: media de los 5 individuales
    da 0.426.
  - Leader Health Composite 79% verificado: 0.30*0.60 + 0.25*0.80 +
    0.25*1.00 + 0.20*0.80 = 0.79.
  - Opportunity Map 2x2 con umbrales T=-0.36, S=-0.09: clasificacion
    de los 11 sectores correcta.
  - Persistencia coincide entre 'Persistencia sectorial' y columna
    'Persist' de Structural Ranking.
  - Retornos 20d coinciden entre Momentum / Tactical / Rankings /
    Alerta XLU.

  COSMETICOS pendientes (baja prioridad):
  - '1 dias' en Data Freshness (deberia ser '1 dia').
  - Columna 'Ultima fecha: N/D' resuelta en este fix. Verificar en
    el proximo run.

  Nota metodologica: la auditoria se hizo cruzando el reporte con
  el codigo de cada seccion, no solo lectura del texto. Cuando un
  numero no cuadraba, se aplico la regla "el codigo manda sobre la
  documentacion" y se verifico con los CSV/parquets. Cuando el
  propio test cubria el caso (p.ej. '+0' en A/D es politica B3),
  se documento como no-deuda en lugar de forzar un fix.

- **Universo del radar: catalogo extiende el universo de descarga (83f94b9).**
  Auditoria completa del reporte + investigacion del universo destapo
  un hueco funcional: el catalogo radar es monotono (255 keys) pero el
  universo de descarga se limitaba a top-20 por sector + top-20 por
  indice (~250 tickers). Los tickers que caian del top-20 dejaban de
  descargarse aunque siguieran en el catalogo.
  Casos hoy: BX, CASY, CMG, ES, FOX, GD, JXN, PNC, PRAX, WY. Todos en
  posiciones 21-23 de sus ETFs (por debajo de TOP_N_SECTOR_COMPONENTS=20).
  Consecuencia: la cobertura observable del reporte IAE caia a
  241/255 por falta de datos frescos, no por bajas reales del universo.
  Fix: get_stock_list() anade los tickers del catalogo radar al
  universo. El catalogo es la fuente autoritativa de "tickers que el
  sistema sigue". No se toca TOP_N_SECTOR_COMPONENTS (breadth sigue
  calculando sobre top-20 por sector; solo cambia el universo de
  descarga).
  Rename coordinado: "Indices Internacionales" -> "Indices (USA +
  Europa)" (la seccion ya incluia S&P 500, DJI, NDX, RUT ademas de
  STOXX50, IBEX, DAX, FTSE). El nombre anterior mentia.

  Verificacion pendiente: el proximo run de CI debe mostrar
  Cobertura observable ~251/255 (antes 241/255) y los 10 tickers
  en el universo descargado.

**Proximo paso sugerido.** Verificar el daily_run del 2-oct con el step
Regenerar reactivado. Monitorizar que CI procesa el crecimiento correcto
y no rompe Run tests.

---
### 2026-10-01 (cierre) - Completion Receipt v1 + fix WLS NaN + deuda

**Objetivo.** Cerrar el dia tras el incidente del cron trimestral
(entrada anterior). Implementar el Completion Receipt (contrato
aprobado, dictamen externo) y documentar la deuda abierta.

**Hecho.**

- **Diagnostico externo del gate.** Expediente completo en
  `docs/auditoria/daily_run_gate_discrepancia.md`. Bug confirmado:
  `_manifest_satisfies` exige sha256 del parquet (gitignored) -> en
  CI la rama `CURRENT` es inalcanzable. Los 4 slots del 01-oct
  ejecutaron pipeline completo por esto.

- **Dictamen externo** (`docs/auditoria/daily_run_gate_dictamen.md`).
  Veredicto: defecto de diseno. Recomienda (E): separar integridad
  del artefacto (manifest <-> parquet <-> sha256, sin cambios) de
  idempotencia del workflow (Completion Receipt inmutable en GitHub
  Actions Artifacts). 8 invariantes I1-I8.

- **Contrato v1** (`docs/auditoria/daily_run_gate_contrato_v1.md`).
  Aprobado 2026-10-01. Implementacion autorizada.

- **Implementacion (6 commits).**
  - `e3c82fb`: `find_completion_receipt` en `pipeline_gate.py`;
    `_manifest_satisfies` conservado como funcion independiente;
    `scripts/write_completion_receipt.py`; 10 tests I1-I8.
  - `e440fec`: diagnostico `wls/tickers/len` en el verifier de
    lideres.
  - `feee18d`: `RECEIPT_FILENAME` = basename del artifact
    (`completion_receipt.json`, no `receipt.json`).
  - `f5e817f`: `GH_TOKEN` en env del step `Run gate` (GitHub no
    hereda `secrets.GITHUB_TOKEN` automaticamente).

- **Fix WLS NaN (`2f956ed` + `e819952`).** Independiente del gate.
  `compute_wls_for_index::robust_intra` usaba `np.median(np.abs(...))`
  que NO ignora NaN. Con 1 ticker NaN (SPCX en QQQ), toda la
  columna `rws_z/stab_z` se volvia NaN -> los 5 lideres del bloque
  Nasdaq-100 con `wls=NaN` -> `verify_leader_selection` FAIL.
  Fix: `np.nanmedian` + guard `pd.isna(mad)`. 4 tests de regresion.

- **Fix gitignore (`1c0ab0c`).** `completion_receipt.json` y
  `validation_gate_result.json` deben estar en `.gitignore` para que
  el guard de untracked del step `Commit and push hist/state` no
  aborte.

- **Verificacion CI end-to-end.** Run `36904431677` (verde
  completo): pipeline corre, sube artifact `completion-receipt-
  2026-09-30` (462 bytes). Run `36909383320`: gate devuelve
  `state=CURRENT`, `reason="completion receipt for 2026-09-30
  (run 36904431677)"`, `run-system=skipped`. **Contrato v1
  confirmado en produccion.**

- **Corpus sincronizado.** `ecd3648` (00_ARRANQUE + 01_METODO),
  `31c298b` (bitacora + historico atomicidad + poda a 12 entradas),
  `e0553ec` (§2 vs §5), `215da23` (snapshot final).

- **Limpieza pyflakes.** `912ee3d`: 8 warnings introducidos en la
  sesion de madrugada + 2 del fix de alineacion.

**Commits.** `1bead8a`, `27248d3`, `912ee3d`, `d0b0869`, `ecd3648`,
`31c298b`, `e0553ec`, `6858533`, `03b4192`, `f098a2a`, `2cb77ba`,
`36e013b`, `e3c82fb`, `e440fec`, `2f956ed`, `e819952`, `1c0ab0c`,
`feee18d`, `f5e817f`, `215da23`.

**Pendiente.**

- **Test rojo `test_build_catalog_csvs::test_idempotencia`**
  (preexistente, no de hoy). Aparece tras `snapshot_2026-10-01_01`
  (255 tickers). `build_catalog_csvs.py:195` accede
  `assign_ordered.iloc[i]` iterando `prev_ordered` (255) pero
  `assignments` en algun momento es 242. Mismatch positional.
  Merece sesion propia: ver `main()`, firmas de `build_assignments`
  / `build_membership`, decidir join por `catalog_key` vs positional.
- **Deudas latentes documentadas:** `stock_leader.py:59` y
  `options.py:46` usan `np.median(np.abs(...))` en contextos donde
  hoy no se manifiesta pero podria (ver `04_HISTORICO §3.3`).
- C2 (920): bloqueado externo.
- H5.3: cron nov 2026.

**Proximo paso sugerido.** Sesion de `build_catalog_csvs` + tests
de los 2 `np.median` latentes. Despues, `01_METODO` a 24 KB
(umbral 20 KB): dividir.

### 2026-10-01 (tarde) - Incidente cron trimestral + fix sys.path + fix alineacion

**Objetivo.** Verificar el primer schedule real de los 3 workflows trimestrales
(1-oct-2026, CEST 04:47/06:17/07:17). Aplicar 07_RUNBOOK §3. Diagnostico y fix
de los fallos.

**Hecho.**

- **Los 3 workflows fallaron en el primer schedule real** con
  `ModuleNotFoundError: No module named 'src'`. Causa: invocacion
  `python scripts/X.py` pone `scripts/` en `sys.path[0]`, no la raiz.
  Los 3 importan `from src.holdings_filter import ...` (introducido en
  D11, 30-sep) sin preparar el path. Bug de nuestra parte, no de GitHub.
- **Fix 1 (`1bead8a`):** `sys.path.insert(ROOT)` en
  `update_sector_holdings.py`, `update_index_holdings.py`,
  `parse_ssga_fez.py`, `parse_blackrock_final.py`. Patron ya en 18
  scripts del repo. Ademas: los 2 ultimos ejecutaban el cuerpo al
  importar (A6.3-01 pendiente). Refactor a `main()` + guard `__main__`.
- **Fix 2 (`27248d3`):** en `get_state_street_holdings`, los append de
  `names` y `weights` estaban fuera del `if is_valid_holding_ticker`.
  Con entradas invalidas del Excel de SSGA (cash placeholder, codigos
  numericos), las 3 listas se desalinean y `pd.DataFrame({...})` revienta
  con `All arrays must be of the same length`. Visible en `update_index_holdings`
  (SPY 504 tickers -> 2 fallidos; DIA 30 -> 2 fallidos). Fix: indentar
  bajo el `if`, patron ya correcto en `update_sector_holdings.py`.
- **Test nuevo `tests/test_scripts_sys_path.py`:** invoca `runpy` desde
  `cwd=scripts` sobre los 4 scripts. Rojo sin fix, verde con fix.
- **Test nuevo `tests/test_update_index_holdings_ssga.py`:** construye
  un Excel sintetico con 2 tickers validos + 2 invalidos. Verifica que
  el df resultante tiene 2 filas. Rojo sin fix, verde con fix.
- **Recuperacion.** Los 3 workflows re-disparados manualmente. Los 3
  verdes. Commits del bot: `66c442a` (etf_holdings.csv), `58a4c9f`
  (index_holdings europeos), `5e850df` (index_holdings SPY+DIA).
- **Observacion secundaria (no fix):** los 4 daily_run del 01-oct
  ejecutaron pipeline completo (`run-system=success` en los 4). El gate
  `_manifest_satisfies` exige sha256 del parquet, que esta gitignored
  y no existe en CI. Resultado: gate nunca devuelve `CURRENT` en CI,
  siempre `READY` via probe. Bug de diseno. Documento para auditor
  externo pendiente.
- **Limpieza pyflakes (`912ee3d`):** 8 warnings introducidos en la
  sesion de madrugada (Fix G/K/M/N) + 2 introducidos por el fix de hoy.
  Todos imports unused o variable sin usar. `pyflakes .` limpio tras
  el commit.
- **Corpus sincronizado (`ecd3648`):** `00_ARRANQUE` (tests 2945 ->
  2997, pyflakes `.`, `08_AUDITORIA_SECTOR_REGIME` en tabla §7, tamanos
  reales) + `01_METODO` (idem + patrones 10 -> 13, corpus 6 -> 8
  ficheros, poda bitacora 5 -> 10). Snapshot regenerado (`d0b0869`).

**Commits.** `1bead8a`, `27248d3`, `912ee3d`, `d0b0869`, `ecd3648`.

**Pendiente.**

- C2 (920): bloqueado externo.
- H5.3: cron nov 2026.
- **Bug gate daily_run:** documento para auditor externo. El gate no
  cumple su contrato en CI (parquet gitignored).
- `01_METODO.md` a 24 KB, umbral autoimpuesto 20 KB (dividir).

**Proximo paso sugerido.** Redactar el documento de auditor externo del
gate del daily_run.

### 2026-10-01 (madrugada) - Auditoria reporte diario + Fixes G/H/K/L/M/N

**Objetivo.** Auditoria del reporte diario generado el 30-sep-2026
(951 lineas, 70 secciones). Busqueda de NaN, ceros, inconsistencias,
bugs silenciosos y errores en la presentacion. Derivacion de fixes.

**Hecho (6 fixes).**

- **Fix G (FINRA sin walk-back).** `data_quality.py:258` aplicaba
  `_last_market_session` a la columna `week` de FINRA. La fecha-semana
  2026-09-07 (Labor Day) se convertia en 2026-09-04. Mientras el
  reporte publicaba "Semana FINRA: 2026-09-07". Fix: flag `walk_back`
  por fuente. Commit `1ca2ab6`.
- **Fix H (QQQ returns truncado).** `qqq_returns_yahoo.py` usaba
  `prices.index[-1]` sin truncar. Yahoo devuelve barra parcial intradia.
  Commit `1ca2ab6`. Verificado en test unitario (en produccion depende
  del horario; a las 00:00 CEST el mercado ya cerro).
- **Fix K (cache freshness europea).** Euronext y BME usaban
  `(ref - last).days <= 1`. Con ref=30-sep y cache=29-sep,
  consideraban "fresco". 13 Euronext + 19 BME no refrescaban.
  `stock_prices.parquet` caia a 89.8% cobertura. Xetra ya usaba
  FU-018 (correcto). Fix: helper `_cache_freshness.py`. Commit `22cbcb5`.
- **Fix L (tests fragiles).** 3 tests de `test_temporal_contracts`
  hardcodeaban 562 y 539. El universo cambio a 561/538. Fix: derivar
  de `df_real` con rango plausible. Commit `93ffb20`.
- **Fix M (cache-hit valida cobertura).** `stock_data_loader.py:536`
  y `data_loader.py:277` decidian cache-hit solo por fecha
  (`_df_last >= _last_exp`). Un parquet del 30-sep con 89.8% de
  cobertura era aceptado. El cascade europeo nunca se ejecutaba.
  Fix: helper `_last_row_coverage_ok` en `src/utils.py`. Gate de
  90%. Commit `c994f01`.
- **Fix N (fund flow descarga siempre).** 5 writers (ssga, amundi,
  _blackrock_base, blackrock_iwm, cftc) usaban `mtime < 23h ->
  cache-hit`. Si la fuente publicaba despues del ultimo run, el
  siguiente run aceptaba cache de 22h. Coste real de descarga sin
  cache: 13s. Fix: descarga siempre, cache solo fallback. Commit
  `b30a918`.

**Verificacion end-to-end.** 3 runs E2E. Fix K+M verificado:
cache SSGA 28->29-sep, cobertura stock_prices 89.8%->93.6%,
Sector Breadth publica 30-sep, sin WARN "Sin actualizacion".
Fix N verificado: 12 descargas SSGA + DAX + ISF + IWM + Amundi +
CFTC. Cero cache-hits. Cache avanzada al 29-30-sep.

**Commits.** `1ca2ab6`, `22cbcb5`, `93ffb20`, `c994f01`, `b30a918`,
`eec3d66` (regeneracion).

**Pendiente.**

- Cron trimestral 1-oct (hoy, 04:47/06:17/07:17 CEST).
- C2 (920): bloqueado externo.
- H5.3: cron nov 2026.
- 20 tickers LSE sin observacion local (scraper externo
  `lse-close-scraper` tiene el 30-sep; local congelado en 25-sep;
  CI funciona).

**Proximo paso sugerido.** Verificar cron 1-oct con 07_RUNBOOK 3.
### 2026-09-30 (noche, sesion 8) - Auditoria sector_regime + fixes H8/R2/H1 + Fix D

**Objetivo.** Auditoria externa del subsistema `regimes/sector_regime.py`
en colaboracion con un auditor. Primero recolectar evidencia y redactar
informe. Despues, con el dictamen, fase de diseno/contrato. Finalmente,
implementar los fixes aprobados y verificar end-to-end.

**Hecho.**

- **Fase 1 - Recoleccion y expediente.** Volcado de los 11 sectores
  sobre data/market_data.parquet (2605 sesiones). Distribucion de
  n_valid, penalty, dispersion. Identificacion de los 10 hallazgos
  H1-H10 (H8 material: close=NaN -> trend/breadth=-1.0, 1454 filas).
  Tres iteraciones con el auditor externo. Version final del
  expediente consolidado.

- **Fase 2 - Decisiones D1-D4.** D1: propagar NaN desde la raiz.
  D2: umbral n_valid >= 4. D3: mantener formulas, documentar
  semanticas. D4: recalcular historico via E2E.

- **Fase 3 - 6 commits de fix con verificacion triple.**
  43a2fdb (calendario cierres excepcionales Bush/Carter),
  ee24afc (trend_position propaga NaN, H8),
  c98c22a (effective_date en sector_regime, R2),
  e92f61c (umbral n_valid>=4, H1),
  c798857 (append_dedup en 5 writers de flujo, Fix D),
  7b0044d (normalizar Date en ssga, Fix D2).

- **Fase 4 - Run E2E doble.** Primer run destapo Fix D (perdida de
  fila 2026-09-29 en 2 CSV). Segundo run tras Fix D2: sin perdida,
  sin duplicacion, Gate 10/10, ranking identico.

**Commits.** `43a2fdb`, `ee24afc`, `c98c22a`, `e92f61c`, `c798857`,
`7b0044d`, `8dc7437` (regeneracion outputs).

**Pendiente.**

- C2 (920): bloqueado externo.
- H5.3: cron nov 2026.
- Regeneracion de historico completo (sector_rank_history retiene 2
  meses) si se quiere verificar H1/H8 sobre warm-up 2016 y XLC
  pre-2018.

**Proximo paso sugerido.** Verificar cron 1-oct (3 workflows
trimestrales, primera ejecucion por schedule) con 07_RUNBOOK 3. Si
verde, documentar tiempos reales en 07_RUNBOOK 2.3.

### 2026-09-30 (noche, sesion 7) - D37-D40: cobertura final + fósiles + auditoria workflows

**Objetivo.** Cerrar el backlog formal tras D36. Orden por ROI:
cobertura de scripts con logica real, limpieza de fosiles, auditoria
tematica de workflows.

**Hecho.**

- **D37 (heredado del traspaso).** HEAD inicial `db5705d`. Cerrado
  sin cambios hoy. pipeline_gate, guard_coverage, issue_manager
  cubiertos.

- **D38 (3 scripts, 3 commits).** Sub-bloque 1
  (`download_official_list_13f`): 27% -> 99%. 34 tests. Bug F5.6-X
  (fallback 404) cubierto. Commit `508bbd0`. Sub-bloque 2
  (`regenerate_cusip_crosswalk`): 69% -> 99%. 12 tests nuevos + 1
  bug latente corregido: `_load_radar_index` protegia `scf != "nan"`
  pero no `tk`. `str(NaN or "")` = "nan" (NaN es truthy). Fix con
  helper `_clean_str`. Commit `82bef32`. Sub-bloque 3
  (`qqq_returns_yahoo`): 63% -> 98%. 6 tests nuevos. Refactor del
  import en el test para que coverage lo instrumente. Commit
  `8dabcfe`. `iae_pipeline` descartado: E2E-only por diseno
  (docstring explicito, wrappers sobre modulos auditados).

- **D39 (5 commits).** Dead config CI: `daily_run.yml` intentaba
  `git add` de 4 rutas de commodities que ya no existen (OilPriceAPI
  eliminada). `git add <inexistente>` no falla, workflow verde,
  git add no-op. Fix en `2e4f9f6`. Fosiles de `02_ARQUITECTURA 11`:
  3 entradas (OilPriceAPI, Futuros BLOCKED, Cache 13F v3) ya no
  aplican. WONT FIX `as_of_date` (timestamp, no fecha del dato;
  unico consumidor prefiere `effectiveDate`). Regla anti-falso-
  positivo del test con evidencia empirica de D38. Commit `08d3fbd`.
  Poda bitacora: regla 5 sesiones -> 10 + dia en curso (no se
  aplicaba desde 2026-09-28, habia 12 entradas). `00_ARRANQUE 4`
  congelado -> puntero a bitacora. Commit `81e8548`. Criterio fino
  E2E en `01_METODO 4`. Commit `37eb3c3`. Skips opt-in
  formalizados: `@pytest.mark.network` (2 tests) y
  `integration_real_data` (8 sitios). Fosiles de cifra
  (2383 -> 2945). Commit `05f08d1`.

- **D40 (1 commit).** Auditoria tematica de workflows como conjunto
  (nunca hecha; A6.2 audito 10, hoy 9). Fosiles: `02_ARQUITECTURA`
  decia 10 workflows (real 9), `03_IAE` decia 286 LOC de
  `update_sec_13f.py` (real 373). Riesgo operativo para manana
  documentado: `update_macro_manual` dispara diario a `17 6 UTC`,
  coincide con `european` (concurrency.group distintos, no se
  serializan, ambos pushean a main). Commit `ea60a6c`.

**Commits.** `508bbd0`, `82bef32`, `8dabcfe`, `2e4f9f6`, `08d3fbd`,
`81e8548`, `37eb3c3`, `05f08d1`, `ea60a6c`.

**Pendiente.**

- Cron 1-oct: 3 workflows (sector 04:47, index 06:17, european 07:17
  CEST). Primera ejecucion por schedule. Riesgo de colision con
  `update_macro_manual` a 06:17 CEST. Ver `07_RUNBOOK 2.5` + `3`.
- C2 (920) OPEN. Bloqueado externo.
- H5.3: verificar cron 20-nov (13F, `source=cron, outcome=INGEST`).
- Deudas abiertas: cobertura <80% en modulos E2E-only por diseno
  (pipeline_contractual 38%, data_loader 59%, update_sec_13f 62%,
  options.py 69%). No accionables sin mocks masivos.

**Proximo paso sugerido.** Verificacion del cron 1-oct con
`07_RUNBOOK 3`. Si verde, documentar tiempos reales en `07_RUNBOOK 2.3`.

### 2026-09-30 (noche, sesion 6) — D34-D36: cobertura + bug real en freshness

**Objetivo.** Cerrar la cobertura de los modulos con logica real
pendiente (excluyendo IO/orquestacion E2E-only por diseno).

**Hecho.**

- **D34 (options_metrics + index_leaders).** options_metrics 88% -> 93%
  (ramas intermedias de classify_pcr/classify_ihr).
  index_leaders 70% -> 87% (rama router + select_index_leaders).
  14 tests.

- **D35 (health_check).** 52% -> 84%. 31 tests. Cubre _run_gh,
  _expected_slots, check_cron_slots, check_workflows,
  check_13f_cache, check_yahoo_revision, check_eu_usa_pattern,
  check_iae_section, _render_markdown, _ensure_label,
  _find_open_issue. No cubre main() (argparse + issue creation).

- **D36 (freshness + bug real).** 65% -> alto. 12 tests. **Durante
  la escritura de tests se detecto un bug real:** la seccion padre
  render_data_freshness protegia pd.Timestamp(pcr_data['last_date'])
  con try/except, pero _generate_coverage_table (llamada al final)
  repetia el mismo calculo SIN proteccion. Con un last_date no
  parseable (Yahoo/CBOE devolviendo algo raro), el reporte entero
  caia con DateParseError. Fix: try/except en los 2 bloques.
  Verificacion triple reproduce el bug ("no-es-fecha" -> DateParseError
  en rojo).

**Detalle del bug D36 (relevante):** es un bug latente en produccion,
no reproducible con datos normales. Solo revienta si la fuente externa
devuelve una fecha malformada. Sin el barrido de cobertura, habria
quedado enterrado hasta que Yahoo/CBOE fallara en real y el reporte
no se generase sin razon aparente.

**Commits.** `9fd7403` (D34), `bda86b9` (D35), `5e75d05` (D36).

**Pendiente.**

- C2 (920) OPEN. Bloqueado por auditor externo.
- H5.3: verificacion cron nov 2026.
- 2 tests skipped por --run-network en test_freshness (opt-in).
- Cron trimestral 1-oct-2026: aplicar 07_RUNBOOK §3.
- Modulos con cobertura <80% tras D36: pipeline_contractual (38%,
  orquestador E2E-only), data_loader (59%, red), health_check (84%),
  options.py (69%, IO), update_sec_13f (62%, red),
  stock_data_loader (77%), guard_coverage (76%),
  pipeline_gate (73%), issue_manager (77%),
  regenerate_cusip_crosswalk (69%), iae_pipeline (63%),
  qqq_returns_yahoo (63%), download_official_list_13f (27%).
  Los candidatos a un D37 si se decide: pipeline_gate, guard_coverage,
  issue_manager (logica real sin red).

**Proximo paso sugerido.** Cron 1-oct. Aplicar 07_RUNBOOK §3 a los
3 workflows trimestrales (sector_holdings 04:47, index_holdings
06:17, european_holdings 07:17 CEST). Si verde, continuar con
pipeline_gate + guard_coverage + issue_manager (D37).

---

### 2026-09-30 (noche, sesion 5) — D29-D33: cobertura de render, regimes, SLPM, flow, options

**Objetivo.** Continuar el barrido de cobertura tras D28. 5 bloques
tematicos ordenados por criticidad para el output del sistema.

**Hecho.**

- **D29 (render modules).** 3 modulos con 63-67%. sector_context
  63% -> 100%, sectorial 67% -> 100%, synthesis 66% -> 96%. 19 tests.

- **D30 (regimes).** volatility_regime 33% -> 100%,
  tactical_engine 14% -> 97%, structural_engine 21% -> 100%.
  Nota: volatility_regime requirio mock de la funcion interna
  (los datos sinteticos daban NaN por el rolling interno).

- **D31 (SLPM).** state_machine 12% -> 100%, slpm_v12 38% -> 93%.
  38 tests.

- **D32 (flow/divergencia).** sector_leader_divergence 16% -> 79%,
  sector_flow_characteristics 35% -> 100%. 21 tests.

- **D33 (options/rs/vol).** options_metrics 65% -> 88%,
  rs_internal 28% -> 98%, vol_metrics 15% -> 100%,
  options.py 67% -> 69% (solo el helper puro). 33 tests.

**Falsos positivos detectados durante verificacion triple:**
- D26: assert all sobre dict vacio. Corregido.
- D27: fixture con filas ya ordenadas. Corregido.
- D23: reference_date = hoy. Corregido.

Ninguno en D29-D33 (los tests fueron escritos con cuidado tras los
fallos anteriores).

**Commits.** `2cfcaae` (D29), `6030792` (D30), `d3dbad5` (D31),
`1c0807e` (D32), `104b4e2` (D33).

**Pendiente.**

- C2 (920) OPEN. Bloqueado por auditor externo.
- H5.3: verificacion cron nov 2026.
- 2 tests skipped por --run-network en test_freshness (opt-in).
- Cron trimestral 1-oct-2026: aplicar 07_RUNBOOK §3.
- Modulos con cobertura <80% tras D33: options.py (69%),
  pipeline_contractual (38% por diseño), data_loader (59%),
  stock_data_loader (77%), index_leaders (70% por ramas defensivas),
  sector_leader_divergence (79%), health_check (52%),
  update_sec_13f (62%). Los de IO/orquestacion E2E-only por diseno.
  Los de logica podrian cubrirse en sesion futura si se decide.

**Proximo paso sugerido.** Cron 1-oct. Aplicar 07_RUNBOOK §3. Si
verde, revisar si queda deuda P3 accionable o cerrar cobertura.

---

### 2026-09-30 (noche) — D19-D28: verificacion produccion + cobertura indicators/utils

**Objetivo.** Tras cerrar sesion 3 con D16/D18, continuar con la
verificacion en produccion del check_yahoo_revision (D19) y el barrido
de cobertura de los modulos restantes.

**Hecho.**

- **D19 (check_yahoo_revision en produccion).** Verificado sin commit:
  los 3 estados del check dan el resultado esperado. Sin cambio:
  `[OK] sin revision`. Sha256 modificado, mismo last_date:
  `[WARN] revision Yahoo: sha256 cambio`. Restaurado: `[OK]`.
  Sin codigo nuevo.

- **D20 (utils.py).** 74% -> 88%. detect_cross_module_conflict,
  _confidence_range_row, _try_cleanup. Nota: RECESSION no esta en
  financial_stress_states (['CRISIS', 'HIGH_STRESS', 'ESTRECHA',
  'STRESS']). Decision del sistema respetada.

- **D21 (stock_data_loader).** Cerrado como NO-DEUDA. Las funciones
  puras ya tenian test; los 110 sin cubrir son download_stock_prices
  (red) + _apply_lse_close_override (scraper LSE). ROI < 1. Mismo
  criterio que D18 para pipeline_contractual.

- **D22 (indicators).** 3 modulos que alimentan macro_regime:
  credit (19% -> 100%), breadth (22% -> 100%), macro_fundamental
  (13% -> 97%).

- **D23 (barrido datetime.now).** El inventario completo del repo dio
  64 llamadas reales. De ellas, 4 calculaban age contra el reloj en
  el pipeline productivo (mte_confirmation, validation_gate x2,
  flows_secondary) + 1 cosmetico en fundamental_signals. Fix: 5
  funciones reciben reference_date=None, propagado desde
  compute_all_regimes y run.py. Nota importante: el primer test que
  escribi usaba reference_date=hoy (2026-09-30) y era CIEGO al
  cambio. Corregido a una fecha claramente distinta.

- **D24 (verificacion end-to-end).** Run manual. Validation Gate
  10/10, NIPC 8256882557. La fila 2026-09-29 en macro_regime.csv
  queda identica (-0.0459, conf 0.3076), prueba de que D23 no altero
  el calculo previo. La fila nueva 2026-09-30 tiene score -0.0120.

- **D25-D28 (indicators).** Cierre del barrido de modulos con <80%:
  fls (7% -> 100%), index_phase (11% -> 92%),
  commodity_market_correlation (10% -> 90%),
  index_leaders (12% -> 70%), darkpool_history (8% -> 88%),
  breadth_equity (76% -> 84%), evidence_matrix (78% -> 94%),
  mte/decision (78% -> 88%), mte/engine (79% -> 95%).

**Falsos positivos detectados durante verificacion triple:**
- D26: `assert all(v == "ERROR" for v in phases.values())` con
  `phases={}` es True. Falso positivo. Corregido con verificacion
  previa de `len(phases)`.
- D27: fixture con filas ya ordenadas hacia que el test de sort fuera
  ciego. Corregido con orden intercalado.
- D23: reference_date = hoy. Falso positivo. Corregido.

**Commits.** `8e92f9a` (D20), `182c181` (D21), `1cbf4a4` (D22),
`ec9e5b2` (D23), `6dae899` (run D24), `72a3f74` (D25), `3fa7b0f` (D26),
`d3b1c0b` (D27), `3835476` (D28).

**Pendiente.**

- C2 (920) OPEN. Bloqueado por auditor externo.
- H5.3: verificacion cron nov 2026.
- 2 tests skipped por --run-network en test_freshness (opt-in).
- Cron trimestral 1-oct-2026: aplicar 07_RUNBOOK §3.

**Proximo paso sugerido.** Cron 1-oct. Aplicar 07_RUNBOOK. Si todo
verde, o retomar deuda o nueva auditoria.

---

### 2026-09-30 (tarde) — D4 + D3 + D2 + Runbook + D1: cierre de P1

**Objetivo.** Recibir traspaso del asistente saliente, asimilar contexto,
atacar la deuda abierta por prioridad. P1 tenia tres frentes (D1, D2, D3)
mas D4 (puntual) y una R (runbook del cron trimestral que dispara el 1-oct).

**Hecho.**

- **D4 (lag 13F unificado).** El saliente lo reporto como "falla en dispatch
  manual fuera de {feb,may,ago,nov}". Verificado: los 3 runs X eran
  `cancelled`, no fallos. No habia incidente. Pero al investigar aparecio
  una deuda real: **tres politicas de lag distintas convivian** (doc ~45d,
  cron ~50d, script 60d). El comentario del YAML mentia sobre su propio
  codigo. Fix: eliminar el `case` del YAML, delegar a `--latest`, unificar
  `SEC_13F_QUARTER_LAG_DAYS` a 50d, 4 casos frontera en test para que la
  suite no sea ciega al cambio.

- **D3 (health_check roto por diseno en CI).** El workflow no restaura
  cache; el runner no tiene parquets (no versionados, confirmado con
  `git cat-file` -> `exists on disk, but not in HEAD`). Consecuencia: 2 FAIL
  estructurales en cada run + 9 WARN ruidosos. Fix: `IS_CI` detecta el
  entorno, `parquet` y `iae_section` pasan a SKIP en CI (FAIL/WARN en local),
  `cron_slots` relaja umbral a 0->OK/1-2->WARN/3+->FAIL (delays 2-8h de
  GitHub son ruido, no incidente), workflows trimestrales sin runs -> SKIP.

- **D2 (determinismo vs Yahoo).** El saliente mezclaba dos problemas: A/D Net
  depende de Close, no de Volume; el volumen es D7. `K-LSE-YAHOO-REVISION-01`
  no era huerfano: ya tenia ficha en 02_ARQUITECTURA y 04_HISTORICO como
  MONITORED. Decision de arquitectura (opcion A del usuario): aceptar y
  documentar. `00_ARRANQUE §2` pasa de "Determinista" a "Determinista **dado
  un snapshot del input**". Nuevo check `check_yahoo_revision` compara
  manifest actual vs HEAD por sha256: mismo `last_date` con sha distinto ->
  WARN (revision detectada). 5 tests del helper.

- **R (runbook del cron 1-oct).** Los 4 workflows trimestrales **nunca se
  han disparado por schedule** en historial visible. Todos los runs
  conocidos son `workflow_dispatch`. El cron del 1-oct sera la primera
  ejecucion automatica real. Creado `Consolidacion_Documentos/07_RUNBOOK.md`
  con inventario de crons, verificacion post-cron y contingencias
  (`startup_failure`, conflicto de rebase, cron no disparado).
  Detectado que el traspaso listaba `update_sec_nport` como dia 1; el YAML
  dice dia 20.

- **D1 (comp_breadth + penalizacion dispersion).** El saliente lo describia
  como "tanh_normalize colapsa cuando la serie es constante". Verificado: es
  correcto, 27-61% de filas por sector con `comp_breadth==0`. Pero al
  verificar aparecio un **segundo bug independiente**: `_dispersion =
  std / (|mean|+1e-9)` explotaba cuando la media se aproximaba a cero,
  aniquilando el score de XLU, XLI, XLB. Fix breadth (mapeo directo
  `(breadth-0.5)*2`), fix dispersion (std poblacional `ddof=0` sin dividir
  por media, garantia penalty en [0.5,1]), 5 tests del contrato,
  verificacion triple (verde con fix, rojo con `ddof=1`, verde restaurado),
  golden regenerado una sola vez.

**Commits.** `4d1735b`, `9faa8c6`, `8d6076f`, `d446614`, `e3621cc`, `9d34eba`.

**Pendiente.**

- D5 (corpus desincronizado): 00_ARRANQUE decia 2340 tests (real 2369
  tras esta sesion), `01_METODO` 2297, `02_ARQUITECTURA` "10 contratos"
  (real 9). Sesion documental completa.
- D6 (`datetime.now()` en header del reporte). Viola `00_ARRANQUE §2`.
- D7 (volumen Yahoo no consolidado). WONT FIX razonado, documentar en
  `02_ARQUITECTURA §11`.
- BOM en 7 workflows. Cosmetico. Detectado durante D4.
- `_write_ingest_trace` ensucia manifests productivos incluso en rama
  `[SKIP]`. `Path.write_text` sin newline final en manifests.
- `concurrency.group` desalineado: `update_index_holdings` y
  `update_european_holdings` escriben ambos `data/index_holdings.csv`
  con grupos distintos. Riesgo de conflicto de rebase si coinciden.
- Residuos en `etf_holdings.csv` (ticker `-` y `XASU6` con peso negativo).
- Duplicacion de titulo en `05_BITACORA.md:91` (Integridad parquets).
- `test_sector_regime_dispersion.py` cuenta 5, pero el delta total
  de tests entre D4 y cierre es +5; D2 tambien sumo +5. Cuadra
  2344 -> 2348 (D4) -> 2359 (D3) -> 2364 (D2) -> 2369 (D1).

**Proximo paso sugerido.** Cron trimestral 1-oct-2026 (04:47, 06:17, 07:17
CEST): verificar con el runbook. D5 (corpus) o D6 (datetime.now) para
sesion siguiente.

---

### 2026-09-30 (madrugada) — Baseline IAE vs par vigente: aclaracion

**Objetivo.** Investigar la diferencia entre el NIPC del run manual de
GitHub (8256882557) y el baseline del corpus (-4264449012).

**Hallazgo.** No es bug. El reporte diario calcula el NIPC del par de
trimestres vigente (`_list_available_quarters` toma los 2 ultimos).
Al entrar 2026Q2 (commit c5e3ee0), el par paso de Q4-2025 -> Q1-2026 a
Q1-2026 -> Q2-2026. Reproducido localmente con el par nuevo:
8256882557, identico al CI. El baseline -4264449012 corresponde al par
antiguo, congelado en `current.json`.

**No se toca codigo.** El auto-avance del par es correcto y autonomo.
El pipeline se mantiene independiente del usuario. No se hardcodea
ningun par.

**Corpus actualizado.** 00_ARRANQUE, 02_ARQUITECTURA, 03_IAE,
04_HISTORICO y `current.json` declaran ahora explicitamente que el
baseline es historico (par Q4-2025 -> Q1-2026) y que el run diario
publica el NIPC del par vigente, que cambia con cada trimestre.

**Pendiente.**
- Cron 01:17 CEST (madrugada del 30): valida PENDING BME, EU 5y,
  regularMarketTime de Yahoo.
- Cabo B: revisar si Yahoo actualiza cierre del 29 a las 04:00 CEST.
- C2 (920): sin material del auditor, referido al par Q4->Q1.

**Hallazgos del run manual (2026-09-30, run 36642077307):**
- `update_futures` exit 1. BZ=F/CL=F BLOCKED (403 OilPriceAPI).
  `commodities_futures.parquet` atascado en 21-sep. Señal intencional
  con `continue-on-error: true`.
- Warning 'Cache 13F no disponible' es falso positivo. La cache se
  restauro por restore-key (v2), pero `cache-hit != 'true'` dispara
  la alerta. La primary key v3 aun no existe (se guardara el 20-nov).
- 3 tests skipped en CI (`market_data`, `stock_prices`,
  `european_tickers_recent`) por parquets gitignored. Skip silencioso.
- NIPC del run manual: `8256882557` (par vigente Q1-2026 -> Q2-2026).
  Distinto del baseline `-4264449012` (par Q4-2025 -> Q1-2026, congelado).
  Documentado en `c087cb8`.

**Proximo paso sugerido.** Abordar 3 fixes: (a) condicion de la warning
cache 13F, (b) skipped tests en CI, (c) evaluar sustituto OilPriceAPI
o documentar futuros como BLOCKED de facto con parquet congelado.

## 4. REFERENCIA A SESIONES ANTERIORES

Para sesiones anteriores al 2026-09-24, ver `git log --oneline`. Resumen tematico en `04_HISTORICO.md`.

---

## 5. REGLAS DE PODA

- La bitacora mantiene las ultimas **10 sesiones**. Ademas, siempre
  conserva las del dia en curso, aunque superen 10 (una jornada larga
  puede generar 6-7 entradas, y perder la de la manana mientras se
  trabaja la de la noche no aporta).
- Al anadir una nueva, si hay >10 fuera del dia en curso, la mas antigua
  se elimina. Nota: la regla era 'ultimas 5' hasta 2026-09-30. Nunca se
  aplico (12 entradas acumuladas). Se cambia al umbral real observado
  en lugar de fingir que se poda.
- **Antes de eliminar**, verificar que sus decisiones clave estan en `04_HISTORICO.md`. Si no, resumir primero.

---

**Fin de la bitacora.**