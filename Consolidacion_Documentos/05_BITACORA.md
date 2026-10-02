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

### 2026-09-30 (tarde, sesion 3) — D16 + D18: cobertura y contrato que miente

**Objetivo.** Tras cerrar sesion 2, continuar con D16 (deuda detectada
durante D3) y los 4 modulos con cobertura baja documentados en
00_ARRANQUE §5.

**Hecho.**

- **D16 (check_manifest mentia).** check_manifest solo verificaba el
  .manifest.json. En CI el manifest esta versionado, el parquet no
  (gitignored). Resultado: `[OK] VALID` sobre un artefacto ausente,
  mientras parquet daba `[FAIL]`. Mismo patron que D6/D13/D14:
  contrato que miente. Fix: manifest OK + parquet ausente -> SKIP en
  CI, WARN en local. Manifest ausente o invalido -> FAIL sin cambio.
  4 tests existentes actualizados con parquet dummy, 3 nuevos.

- **D18 (cobertura baja en 4 modulos).** Documentado en 00_ARRANQUE §5
  como "sin bug detectado tras inspeccion". Los numeros del corpus
  estaban desfasados: real eran 12/25/27/51%, no 12/12/27/50%.

  - macro_manual_loader: 12% -> 92%. 7 tests de carga de CSVs.
  - european_coverage: 25% -> 98%. 12 tests de _ref_to_date,
    _collect, _render_markdown, _append_csv, integracion.
  - pipeline_contractual: 27% -> 38%. 4 tests de build_identities +
    load_canonical. Orquestador run_contractual_nipc queda E2E-only
    por diseno explicito del test existente (scripts/
    iae_contractual_nipc_e2e.py).
  - data_loader: 51% -> 59%. 38 tests de _is_equity_ticker,
    _ticker_list, _check_khuerfano. download_market_data (red)
    E2E-only.

  Criterio unificado: NO inflar cobertura con mocks del orquestador.
  El corpus 00_ARRANQUE §5 se actualizo con los valores reales.

- **Verificacion end-to-end** (durante sesion 2, antes de bitacora).
  Run manual de run.py tras D6+D6b+D13. Confirmado en produccion:
  header con reference_date, macro_conf = 0.3076 (antes 0.5 fijo),
  ages coherentes. Validation Gate 10/10. NIPC 8256882557. Commit
  `66b9d5d`.

**Commits.** `88cc3ad`, `66b9d5d`, `267ae22`, `99f02e2`, `702ce28`,
`a690562`, `d7ec9df`. El bot intercalo `f3d6c75` (macro_manual FRED).

**Pendiente.**

- C2 (920) OPEN. Bloqueado por auditor externo (sin comando + HEAD).
- H5.3: verificacion cron nov 2026.
- 2 tests skipped por --run-network en test_freshness (opt-in).
- Cron trimestral 1-oct-2026: aplicar 07_RUNBOOK §3 (04:47, 06:17,
  07:17 CEST).

**Proximo paso sugerido.** Cron 1-oct. Aplicar 07_RUNBOOK. Si todo
verde, sesion de auditoria nueva o retomar deudas P3 residuales.

---

### 2026-09-30 (tarde, sesion 2) — D5+D6+D6b+D7+D9+D10+D11+D12+D13+D14+D15

**Objetivo.** Continuacion de la sesion tarde. Atacar deudas menores
tras cerrar P1. Algunas eran cosmeticas (BOM, dead code), otras
reales (D10, D11, D13, D6b).

**Hecho.**

- **D5 (corpus desincronizado).** Cinco ficheros con numeros y
  afirmaciones desfasadas: 00_ARRANQUE decia 2340 tests (real 2383),
  corpus v2 (real v3 con 07_RUNBOOK), "10 contratos" (real 9),
  "OilPriceAPI commodities" (eliminada 2026-09-30), 03_IAE decia
  8280 LOC (real 7088), 04_HISTORICO afirmaba erroneamente "H1-A
  reproduce golden con tolerancia 0".

- **D6 + D6b (datetime.now en reporte).** El header del reporte
  usaba datetime.now() sin tz. Violaba 00_ARRANQUE §2. Fix:
  generate_daily_report recibe reference_date, con fallback a
  now(ZoneInfo('Europe/Madrid')). Ademas 6 usos de datetime.now() en
  calculos de "age" (darkpool, freshness, helpers, sentiment). Todos
  propagados a reference_date naive.

- **D9 (concurrency.group desalineado).** update_index_holdings.yml y
  update_european_holdings.yml escriben ambos data/index_holdings.csv
  con grupos distintos. Fix: unificar en update_index_holdings_csv.

- **D10 (traza 13F).** _write_ingest_trace escribia
  ingest_source/ingest_actor en el manifest del trimestre, incluso
  en rama SKIP. El manifest es artefacto de integridad (sha256).
  Fix: append a data/sec_13f/ingest_traces.jsonl (JSONL versionado),
  manifest intacto, outcome explicito (SKIP/INGEST).

- **D11 (tickers residuales en holdings).** Los 4 parsers volcaban
  al CSV todo ticker no vacio: cash ("-"), futuros (IXAU6, XARU6),
  CUSIPs (2602335D), placeholders SSGA (999USDZ92). El consumidor
  filtraba con INVALID_TICKERS fragil. Fix: src/holdings_filter.py
  con regla estructural, aplicada en 4 parsers, INVALID_TICKERS
  retirada, CSV limpiado (526->502 y 2669->2661 filas).

- **D12 (cosmetico).** D7 (volumen Yahoo no consolidado) documentado
  como WONT FIX en 02_ARQUITECTURA §11. BOM UTF-8 retirado de 7
  workflows. Duplicacion de titulo en 05_BITACORA:91 corregida.

- **D13 (macro_conf hardcode).** compute_macro_regime devolvia
  conf = 0.5 siempre. El aviso de header.py (macro_conf < 0.30)
  nunca se activaba. Los otros 3 regimenes SI calculan confianza
  real. Fix: confidence_from_range (misma politica C19).

- **D14 (tests skipped CI).** Verificado: el saliente reportaba
  deuda de cobertura. FALSO: daily_run.yml ya ejecuta los tests de
  frescura en post-run con datos reales. Solo era cosmetica: sin -ra
  no mostraba razones, reason strings mentian ("CI fresco" tambien
  es falso en local), sin marker. Fix: pytest.ini con -ra + marker
  integration_real_data, reasons corregidos, daily_run pre-run
  excluye el marker.

- **D15 (dead code declarado).** Eliminados:
  finra.get_archive_index, finra.get_available_weeks,
  evidence_matrix.save_evidence_matrix, import Path huerfano.

**Commits.** `cb217ac`, `ec1f2da` (snapshot), `5ecafcb` (rango
c193eef..5ecafcb, 8 commits adicionales: D5, D6, D6b, D9, D10, D11,
D12, D13, D14, D15).

**Pendiente.**

- H5.3 verificacion cron nov 2026.
- C2 (920) OPEN. Bloqueado por auditor externo.
- 2 tests skipped por --run-network en test_freshness (opt-in).
- Modulos con cobertura baja: macro_manual_loader (12%),
  european_coverage (12%), pipeline_contractual (27%),
  data_loader (50%).
- Cron trimestral 1-oct-2026: verificar con 07_RUNBOOK.

**Proximo paso sugerido.** Cron 1-oct (04:47, 06:17, 07:17 CEST).
Aplicar 07_RUNBOOK §3. Si algo falla, §4.

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

### 2026-09-29 (noche) — Integridad parquets + EU 5y + PENDING BME + P65

**Objetivo.** Cerrar el frente de integridad detectado al auditar Ibex 35.

**Hecho.**
- **Guard/gate integridad (2a640f2).** Los .parquet estan gitignored,
  los .manifest.json tracked. Tras cada git pull divergen. Ni guard_coverage
  ni pipeline_gate verificaban sha256 real. Detectado: stock_prices y
  market_data con sha256 divergentes. Fix: ambos calculan sha256 del
  parquet sibling, fail-closed.
- **G-01 paridad (08fe06b).** market_data no truncaba a expected_session
  (stock_prices si). Movido `truncate_to_expected_session` a utils y
  aplicado en data_loader._postprocess_market_data.
- **Regeneracion (1ba5df9).** stock_prices con los 19 .MC que BME no
  habia servido el 28/09 por fallo transitorio.
- **PENDING + EU 5y (4bb54f8).** BME publica T+1. `_compute_by_market`
  marca PENDING (no INVALID) cuando la sesion cerro pero no hay
  observacion. Guard tolera PENDING, no todo-PENDING. Ademas BME y Xetra
  pasan de 430d a 1830d (5y). Euronext topa en el endpoint (~510 sesiones).
- **P65 (f131ce3).** Verificado: `_apply_intra_period_dedup` v1 no emite
  DROP_DUP. Todas las decisiones son KEEP. Conectar P65 no alteraria
  nipc_total. Hipotesis C2 descartada por diseno de codigo. Doc corregida
  (03_IAE.md) + current.json actualizado.

**Commits.** 2a640f2, 08fe06b, 1ba5df9, 4bb54f8, f131ce3.

**Pendiente.**
- Cron 01:17 CEST valida 13F cache, PENDING BME, EU 5y,
  regularMarketTime de Yahoo.
- Cabo B: confirmar si Yahoo revisa cierre del 29 a las 04:00 CEST.
- LSE scraper: cadencia externa + historico en JSON infrautilizado.
- C2 (920): sin material del auditor.

**Proximo paso sugerido.** Verificar el cron de la madrugada.

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