# 05 - BITACORA

**Registro de sesiones recientes. NO normativo. NO se pega al arrancar.**
**Para sesiones anteriores, ver `git log`.**
**Se poda por antiguedad: mantener las ultimas 10 sesiones + las del dia en curso. Al anadir una nueva, la mas antigua fuera del dia se elimina o se resume en 04_HISTORICO.md. Ver seccion 5.**

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

    ### YYYY-MM-DD â€” Titulo corto

    **Objetivo.** Que se pretendia hacer.

    **Hecho.**
    - Bloque 1.
    - Bloque 2.

    **Commits.** <lista de SHAs> (o rango).

    **Pendiente.**
    - Item 1.
    - Item 2.

    **Proximo paso sugerido.** Que deberia hacerse a continuacion.

Al cerrar una sesion nueva, se anade arriba (las mas recientes primero). Si hay mas de 10 fuera del dia en curso, la mas antigua se elimina o se resume en el historico. Ver reglas completas en Â§5.

---

## 3. SESIONES

### 2026-10-07 - Auditoria del reporte CI + 3 fixes (M-03, A-03, A-01/F6-1b)

**Objetivo.** Auditoria puntual del reporte generado en CI (run 37561760673,
headSha 446b1b66) en busca de errores, inconsistencias, fallos de
formulacion, fuentes y volcado de datos. Revisado punto por punto.

**Hallazgos clasificados (5):**

- **M-03 (MEDIA).** `Europa Primary Flow` mostraba N/D pese a tener 2 de 3
  series con Z-score valido. Causa: `flows_secondary.py` verificaba la
  columna `flow_zscore`, no el valor. LYXI (MAD0_NAN) aportaba NaN que
  envenenaba `sum()` y la media completa caia a NaN. Fix 5514c111:
  filtrar con `pd.notna()`. Verificado local + CI: `0.00` (los 3 series
  reales dieron shares_change=0 el 10-07; LYXI se filtra).

- **A-03 (ALTA).** `Dispersion entre sectores` saltaba el dia 2026-10-05
  sin log. Causa: `compute_sector_dispersion` devolvia `pd.DataFrame()`
  vacio si el input era vacio/None. El caller `sectors_base.py` hacia
  `if not df.empty: write else: skip`. El dia desaparecia silenciosamente.
  Fix 8c26bc6e: normalizar `price_rank_list or []`. Cae en la rama
  `n_valid < 8` ya existente. Fila con N/D. No se reescribe historico.
  Verificado CI: fila 2026-10-07 con valores reales.

- **A-01 / F6-1b (ALTA).** FINRA Dark Pool aparecia `RECENT` en
  Data Freshness y `CURRENT` en Calidad, misma fuente (2026-09-14) y
  misma edad (23d). Causa: F6-02 (2026-09-28) ajusto los umbrales
  de FINRA de (30, 45, 60) a (20, 28, 45) en `settings.py` y
  `helpers.py`, pero dejo `indicators/data_quality.py` con los
  umbrales antiguos hardcoded. Dos contratos paralelos. Fix bdaa0326:
  `data_quality.py` importa `FRESHNESS_FINRA`. Tests adaptados.
  Verificado CI: FINRA `RECENT` en ambas secciones (24d).

- **A-02 (INFO).** Top1=100% con Top3/Top5=N/D en Concentracion. No es
  bug. Top N = fraccion del retorno positivo, no del universo completo.
  Con 1 solo ticker positivo (XLF 2026-10-06) Top1=100% es trivial y
  Top3 no se calcula. Fix documental 7b3fe927: nota al pie.

- **INFO.** Print del MTE `Dark Pool ARCHIVAL ({age}d). Excluido del MTE.`
  usa el termino ARCHIVAL (que en FINRA significa >45d) para referirse
  a su umbral propio >14d. WONT FIX razonado: el literal esta anclado
  por `test_compute_mte_tz_aware_no_silencia_archival` como proxy del
  chequeo tz-aware. Cambiarlo obliga a tocar el test o a fragilizar el
  proxy. ROI < 1.

**Falsos positivos retirados tras Gate 0:**

- M-01 (Cob_* identicas fila a fila): la columna es cobertura del
  universo del sector (`n_valid / n_total`), no cobertura por metrica.
  Etiqueta enganosa, no bug.
- M-02 (XLU D1d EMA20=+60): dato real. XLU paso de 10% a 70% sobre
  EMA20 en una sesion (20/20 avances). Confirmado con serie cruda.

**Fix D parte 2 (fuera de la auditoria).** Mismo dia: 8 writers en
`data/providers/` y `indicators/` escribian CRLF a outputs/history.
`.gitattributes` los normalizaba al commitear, pero la doc del handoff
afirmaba cobertura total ("todos los writers"). Fix 3871746f con
`lineterminator='\n'`. En Linux no se nota; en local asegura LF.

**Verificacion.** Suite 3185 passed + 5 skipped en cada commit. E2E
local completo (exit=0, 19.5 min, Gate 10/10). Verificacion CI en el
run scheduled 2026-10-08 02:41 UTC (id 37719126203): los 4 fixes
confirmados.

**Proximo paso sugerido.** Considerar M-04 (candidato): en los CSV
`blackrock_dax_primary_flow` y `blackrock_isf_primary_flow`,
`flow_5d`/`flow_20d` no cambian cuando `shares_change=0`. Puede ser
diseno o bug. Auditar cuando haya sesion propia.

**Commits del dia (7-oct).** 5514c111 (M-03), 8c26bc6e (A-03),
bdaa0326 (A-01/F6-1b), 7b3fe927 (A-02), 3871746f (fix D parte 2),
27a86ce7 (ESTADO_SISTEMA), a486b326 (README golden), 446b1b66
(bitacora cabecera).

### 2026-10-06 - Limpieza estructural + fix rs_mom NaN

**Objetivo.** Reducir disco y ruido del repo, eliminar codigo/documentacion
obsoleta, y arreglar un bug detectado en el reporte del run de CI: `RS Mom`
aparecia como `nan%` en todas las tablas de acciones, con `n_valid_momentum=0`
en `sector_concentration.csv`.

**Hecho.**

- **Limpieza ejecutada (~810 MB liberados).**
  - `outputs/audit/`: 29 snapshots de sesiones cerradas (137 MB).
  - `data/cache/`: cache HTTP regenerable (71 MB).
  - `data/sec_13f/raw/`: TSV crudos de 13F (600 MB), `processed/` intacto.
  - `docs/automatica/`: 22 `.md` auto-generados por `generate_docs.py`.
  - `docs/auditoria/iae/evidence/`: 60 ficheros de probes de auditorias cerradas.
  - `docs/auditoria/daily_run_gate_*`: frente cerrado.
  - `scripts/diagnostico_wyckoff_5c.py`, `verificar_5_casos_v16.py`,
    `verificar_distribution_v15.py`: sin referencias en el repo.
  - `_patch_d4.py`, `.bak` residual, `.coverage`, logs temporales.
- **Archivado en `_archivo/`:** 42 ficheros del expediente Wyckoff (contratos
  v1.0-v1.7, revisiones, protocolos intermedios 5b.2-5b.4-bis, QA). Se
  conservan por trazabilidad de auditoria externa (5b.X 2027).
- **Ramas archivadas como tags `_archivo/*`:** `backup-pre-rebase-20260912`,
  `gold-standard-v1/v2/v3`. Commits conservados, ramas locales eliminadas.
- **Fix rs_mom NaN (commit 34c8029).** Causa raiz: `rs = close / price_etf`
  puede tener NaN residuales por festivos USA presentes en el indice comun.
  `np.log(rs).diff(20).iloc[-1]` propaga el NaN al ultimo valor.
  - `stock_leader.py`: `dropna()` + guard `len >= 21`, `_fmt_num` en render.
  - `sector_breadth.py`: mismo guard (ya presente en `index_leaders.py`).
  - Test nuevo: `test_stock_leader_rs_mom_nan_guard.py` (4 tests).
  - Verificado E2E: `analisis_lideres.csv` pasa de 0/20 a 20/20 rs_mom
    validos; `sector_concentration.csv` pasa de n_valid_momentum=0 a 14-15;
    reporte pasa de 53 `nan%` a 0.
- **Fix cobertura europea (commits cc867fd, 5299e4c).** Dos bugs
  encadenados que causaban 3 SIN_DATOS perpetuos en Xetra (`DB1.DE`,
  `HEI.DE`, `MUV2.DE`).
  - **Bug 1 (cc867fd).** `get_stock_list()` ignoraba tickers del mapa
    del provider si no estaban en `etf_holdings.csv`, `index_holdings.csv`
    (fuera del top-20 DAX por peso) ni en `radar_target_catalog.csv`.
    Fix: anadir una 4a fuente = `supported_tickers()` de los 3
    providers europeos (xetra, euronext, bme).
  - **Bug 2 (5299e4c).** El cache-hit de `stock_prices.parquet`
    aceptaba el parquet si la ultima fila tenia cobertura >=90% sobre
    sus columnas, sin validar que estuvieran TODOS los tickers de
    `get_stock_list()`. Los 3 nuevos nunca se descargaban. Fix: mover
    `get_stock_list()` antes del cache-check y validar tickers
    faltantes; si falta alguno, forzar descarga.
  - Test nuevo: `test_stock_list_includes_provider_maps.py` (4 tests).
  - Test adaptado: `test_download_cache_parquet_fresco_devuelve_df`
    (mockea `get_stock_list` para medir frescura, no universo).
  - Verificado E2E: `european_coverage` 48 OK + 3 SIN_DATOS -> 51 OK,
    0 SIN_DATOS. Cache Xetra con `DB1_DE.csv`, `HEI_DE.csv`,
    `MUV2_DE.csv` (1277 filas cada uno, 2021-10-04 a 2026-10-06).

- **Recuperacion de expedientes SOW (commit e528da19).** Los ficheros
  46 (dictamen externo v5 grounded) y 47 (estado post-dictamen) vivian
  solo en la rama `sow-v4-validation`. Se traen a main para que el
  protocolo v6 pueda referenciarlos desde el corpus.
- **Borrador protocolo v6 (commit b5370612).** `48_sow_protocol_v6.md`
  redactado como BORRADOR pendiente de firma del auditor externo.
  Basado en el dictamen 46: nested validation estricta, AIPW, propensity
  con soporte ex-ante, dos estimandos (RD^AIPW + Delta Brier/LogLoss),
  N_eval por regimen, prohibiciones explicitas. Etiquetas [F46] y [PF]
  distinguen lo fijado por el auditor de lo pendiente. NO hay
  implementacion de codigo hasta firma.

**Commits.** 34c8029 (fix rs_mom), cc867fd + 5299e4c (fix europeo),
2d9b775f + a005fc75 (fix churn EOL), e528da19 (expedientes 46-47),
b5370612 (borrador 48), mas los Daily hist/state.

**Pendiente.**

- Ninguno del dia. Los 3 frentes abiertos al inicio (churn EOL, warnings
  pyflakes, holdings_filter) se cerraron: los 2 ultimos no existian en el
  repositorio (solo en el handoff de traspaso).

**Proximo paso sugerido.** Confirmar el run de CI con los fixes
desplegados. Pendiente firma externa del protocolo v6.

### 2026-10-05 - E2E main post-reorder D3 + fix frescura Evidence Matrix

**Objetivo.** Cerrar el Hueco 1 del handoff: E2E real en main
post-merge del reorder D3 (commits 0dd363d y 1503fe4). Verificar que
el reorder no rompe el reporte.

**Hecho.**

- **Arranque verificado.** HEAD 46291c9 == origin/main. Suite 3175
  passed + 5 skipped. Gate 10/10.
- **E2E main (Hueco 1).** Snapshot pre -> run.py -> snapshot post.
  exit=0. Gate 10/10 post. Secciones ## pre/post: 52 == 52. Lineas
  totales: 1021 == 1021. Diff textual: 6 lineas (3 timestamps + 3 de
  tabla de frescura). Reorder D3 validado.
- **Bug BAJA detectado en E2E.** Fila `Evidence Matrix` en
  `## Calidad, frescura y cobertura de datos` reportaba `2026-10-01`
  (run anterior) en lugar de `2026-10-02` (run actual). Causa:
  `compute_data_quality` leia `evidence_matrix.csv` antes de que
  `finalize` lo reescribiera. Preexistente, cosmetico, ajeno a D3.
- **Fix.** `compute_market_data` acepta `skip_data_quality=False`.
  run.py llama con `True` y ejecuta `_compute_data_quality` tras
  `compute_final_matrices`. Commit 538f2aa. Verificado con E2E.
- **Anomalias resueltas.** (A) `analisis_lideres*.csv` aparecen en
  POST, no en PRE: generados por el run, no perdidos. (C) 3 CSVs de
  history encogen: cambio de `coverage 0.998...` -> `1.0` (mejora de
  539/540 -> 540/540), no reescritura retroactiva.
- **Higiene.** `.orig` residual borrado (`flows_secondary.py.orig`,
  03-oct). Commit 7aa9e68 `Daily hist/state` (convencion existente
  confirmada por `git log`).

**Commits.** 538f2aa (fix frescura), 7aa9e68 (Daily hist/state).

**Pendiente.**

- Hueco 3: dictamen del auditor sobre merge de gold-standard-v4 a
  main. Sigue sin pedir.
- Deuda: churn EOL en outputs/history y outputs/state. Pandas to_csv
  escribe CRLF; .gitattributes declara eol=lf. Cada run ensucia el
  working tree con ~25 ficheros. Fix de raiz: `lineterminator='\n'`
  en writers. Scope separado.

**Proximo paso sugerido.** Abrir Hueco 3 (dictamen gold-standard-v4)
o retomar pendientes del handoff (v6 AIPW, H5.3, C2).

### 2026-10-04 - Walk-forward SOW (plan v3) + expediente 41

**Objetivo.** Resolver la pregunta del usuario: validar SOW sin esperar
12 meses a 5b.X. El sistema es experimental y el legacy no discrimina
fases (CSV 3-oct: 20/20 RANGE en todos los sectores).

**Hecho.**

- **Plan v3 (cdec27d).** Enmienda al plan v2. Walk-forward historico
  como metodo alternativo. Split predefinido: A 2021-2023, B 2024-2026.
  PASS si lower_ci > 0 en ambos.
- **Descarga historica extendida (no productiva).** Yahoo, 2015-01-02
  -> 2021-09-23. 10.6s. 306/316 tickers. Fusionada en
  `data/stock_prices_extended.parquet` (10.7 anos).
- **Test confirmatorio FAIL.** A: lift -0.037, IC95 [-0.147, +0.077]
  (cruza 0). B: +0.095, [+0.043, +0.148] (positivo). Criterio plan v3
  no se cumple.
- **Analisis exploratorio anual (post-hoc, declarado).** 12 bloques.
  3 anos positivos (2020 +0.30, 2025 +0.13, 2026 +0.15). 0 negativos.
  Heterogeneidad temporal marcada.
- **Dictamen auditor externo.** "Dependencia de regimen" etiquetada
  como HIPOTESIS, no conclusion. "Refuta overfitting reciente" NO
  demostrado. 12 bloques = multiple testing. Filtro VIX/drawdown
  NO autorizado. Migracion core v1.8 SI autorizada, proceso separado.
- **Expediente 41** (`41_walk_forward_sow.md`). Estructura HECHO /
  OBSERVACION / HIPOTESIS / NO DEMOSTRADO / PROHIBIDO. Sometido a
  dictamen externo.

**Commits.** cdec27d (plan v3), 1a08884 (FAIL + scripts + expediente).

**Pendiente.**

- Dictamen externo sobre expediente 41.
- Migracion core v1.8 (4 fases sin SOW), proceso independiente.

**Proximo paso sugerido.** Esperar dictamen. Arrancar migracion core
en paralelo.

**Ampliacion (misma sesion, tramo 2).**

- **Walk-forward de calibracion.** Grid 240 combinaciones calibrado
  SOLO con TRAIN 2015-2020, validado sobre TEST 2021-2026 (OOS
  limpio, 10.7 anos de historico via Yahoo extendido).
- **Hallazgo:** 39/240 combos pasan D2 en TEST (16.2%). La region
  M=5, X_ATR >= 0.50 concentra los mejores. La ganadora TRAIN cae en
  percentil 1.2 del grid en TEST; la congelada v1.9 en percentil 20.4.
- **C1 (N=40, M=5, X_ATR=1.00, Y_VOL=1.10).** Seleccionada solo con
  TRAIN. TEST lift +0.103, IC95 [+0.018, +0.192]. **Evidencia OOS
  historica limpia.** No activada.
- **C2 (N=60, M=5, X_ATR=0.50, Y_VOL=1.10).** Top-1 del grid sobre
  TEST (seleccion contaminada). TEST lift +0.107, IC95 [+0.043, +0.170].
  Registrada como **exploratoria congelada**, no validada.
- **Expediente 43.** Interpretacion consolidada tras dictamen externo:
  C1 con evidencia OOS limpia, C2 exploratoria, C0/v1.9 intacta.
  Config SOW sigue None / fail-closed. 5b.X v3 solo aplica a C0.
- **Reproducibilidad verificada.** Compare ejecutado 2 veces limpias:
  mismo hash SHA256 del CSV, ~63s por ejecucion. Sin cache.

**Commits.** acd3632 (expediente 42 + scripts).

**Proximo paso sugerido.** Pasar expedientes 41 y 42 al auditor.
Arrancar migracion core v1.8.

### 2026-10-03 (noche, tramo 2) - Auditoria linea a linea de wyckoff_v1.py v1.8

**Objetivo.** Verificar el modulo Wyckoff recien reconstruido:
buen funcionamiento (logica) + buena implementacion (integracion al
pipeline). Pregunta del usuario: el run automatico del 3-oct puede
haber ejecutado el modulo sin la logica nueva.

**Hecho.**

- **Fase 0 read-only.** Modulo real: `indicators/wyckoff_v1.py`
  (18749 B, mtime 02/10/2026 22:50, version v1.8). 497 LOC. 15
  funciones publicas/privadas. Sin integracion al pipeline.
- **Confirmacion operativa.** El run del 3-oct 15:08 UTC
  (37132181161, SHA `b7dd6a3`) ejecuto el modulo LEGACY
  (`indicators/wyckoff.py`), no v1.8. Los 6 consumidores directos
  del pipeline importan `from indicators.wyckoff import ...`:
  `index_leaders.py:4`, `index_phase.py:3`, `sector_breadth.py:11`,
  `sector_wyckoff_distribution.py:10`, `stock_leader.py:4`,
  `regimes/sector_regime.py:13`.
- **No es bug.** El plan de migracion (`03_plan_migracion.md`)
  tiene el Frente D (migracion de consumidores) BLOQUEADO hasta
  cierre 5b.X. Prohibicion explicita del dictamen final Â§3.
  La migracion requiere 5b.X exitoso, 5c cerrada y dictamen externo.
- **Auditoria linea a linea.** Cubre el hueco declarado por el
  propio auditor externo (dictamen 38 Â§5): "No he inspeccionado
  fisicamente el diff d324346..d0874f4; por tanto, no voy a afirmar
  que lo he revisado linea por linea." El dictamen se emitio sobre
  el paquete contractual, no sobre el codigo.
- **Metodo.** Barrido de los 13 patrones de `01b_PATRONES.md` +
  revision linea a linea + cotejo I1-I36 contra tests por nombre.
- **4 hallazgos BAJA.** 3 dead code (W-01 `C_NORM_DISTR_MIN`, W-02
  `PREC_STRUCT_NEG`, W-03 `ALL_FASES`), residuos del 2026-10-02.
  1 trazabilidad (W-04): I16/I17/I18 cubiertas funcionalmente pero
  no mencionadas por nombre. Cotejo pasa de 33/36 a 36/36.
- **Cero ALTA, cero MEDIA.** 3 INFO WONT FIX razonado (T_NORM_WEAK
  duplicado con contrato redundante, fillna de build_ticker_df
  documentado en contrato Â§8.3, wyckoff_stability sin consumidor
  pero API publica en contrato Â§8.2).
- **Suite.** 3175 passed + 5 skipped. Sin regresion.

**Commits.** 036c4ef (dead code W-01..W-03), de7d729 (trazabilidad
I16/I17/I18), 6717bf9 (audit doc `40_auditoria_v18_linea.md`).

**Pendiente.**

- El modulo no tiene trabajo pendiente hasta 5b.X.
- Pipeline sigue con legacy por diseno.

**Proximo paso sugerido.** Continuar frente Wyckoff si emerge algo,
o esperar dictamen externo / datos 5b.X.

### 2026-10-03 (noche) - Frente B: caracterizacion de macro_regime

**Objetivo.** Cubrir el hueco detectado en S-03-deep: 12 ramas del
regime y compute_macro_score sin test directo. Frente B, elegido
sobre C (temporal_contracts) y D (institutional_accumulation).

**Hecho.**

- **Fase 0 read-only.** El handoff del saliente citaba la ruta
  `src/regimes/macro_regime.py`; no existe. El modulo real esta en
  `regimes/macro_regime.py` (raiz del repo, paquete hermano de
  `indicators/`). Consumidor: `src/pipeline/regimes.py`.
- **Diseno del fixture.** Descartado df_market sintetico completo
  (20+ columnas mantenidas a mano; ya lo rechazo el autor de
  `test_macro_regime_confidence.py`). Descartado refactor-extract
  (cambiaria produccion, contradice dictamen S-03-deep "CERRADO SIN
  CAMBIOS"). Elegido monkeypatch de `compute_macro_signals` y
  `compute_macro_score` en el namespace de `regimes.macro_regime`.
- **Sin golden.** Los 12 casos parametrizados son el contrato; un
  JSON congelado anadiria fragilidad sin cubrir nada nuevo (no hay
  ranking ni umbrales acoplados, a diferencia de sector_regime).
- **27 tests nuevos** en `tests/test_macro_regime_characterization.py`:
  12 ramas (una por caso), 5 de precedencia (input que satisface 2+
  ramas, gana la de menor indice), 1 contrato de ramas inalcanzables
  sin fundamentales (4/5/10 dependen de `last_inflation`; sin
  `df_macro_manual`, `last_infl=0`), 5 de `compute_macro_score`, 4 de
  cadena de fallback de `volatility`.
- **MR-2 cerrado con evidencia.** El ultimo test (sin `^VIX` ni
  `vol_regime_score`) confirma que `all_signals` sale sin columna
  `volatility` y `compute_macro_regime` lanzaria KeyError. WONT FIX
  condicional del audit doc confirmado.
- **MR-8 detectado y cerrado.** El comentario de `compute_macro_score`
  afirmaba renormalizacion entre niveles; el codigo solo la hace
  entre componentes del mismo nivel. Alineado el comentario. Cero
  cambio de comportamiento.
- **Fallos intermedios.** Bug propio en `_signals_with` (lista de
  listas por `[v]*3` sobre una lista); corregido con helper
  `_signals_const`. Calculo esperado inicial asumia renormalizacion
  entre niveles que no ocurre; tests reescritos con factor
  `Wc/(Wc+Wi+Wx)` desde `LEVEL_WEIGHTS`.

**Commits.** 7f13e6c (tests), 6a60842 (audit doc), 3dc0c1a (comentario
MR-8). Pendiente: este cierre documental.

**Pendiente.**

- Push local-first (3 commits verificados, E2E no aplica).
- `00_ARRANQUE.md` en 10213 B; margen 27 B. Sin tocar hoy.

**Proximo paso sugerido.** Elegir entre C (temporal_contracts) y D
(institutional_accumulation). C tiene LOC comparable pero menor
urgencia (A1 ya cerrado). D es mayor coste y sin sintoma concreto.

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
  a7879f3). Preserva v2 + Â§14 auditoria de equivalencia + Â§15
  manifiesto + Â§16 tests exigidos.
- **Script v3 conforme.** `validate_wyckoff_sow_5bX.py` (commit
  d0874f4). 6 cambios normativos D-06, resto funcionalmente
  equivalente a 5b.4-bis.
- **Tests contractuales Â§16.** 19 tests (commit 0fd53fa + 42728ed).
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

**Ampliacion (misma sesion, tramo S-03-deep completo + barridos).**

- **S-03-deep cerrado al 100%.** 46/46 ficheros auditados en 3
  rutas: `src/pipeline/` (18), `regimes/` (8), `src/report/` (20).
  Cero ALTA, cero MEDIA. 11 BAJA corregidas. 6 BAJA WONT FIX.
  90+ INFO WONT FIX.
- **Barrido del patron "11 hardcoded".** 9 ficheros, 27 sitios
  sustituidos por `EXPECTED_SECTOR_COUNT`.
- **Barrido de docstrings desalineados.** 5 ficheros corregidos.
  El barrido AST detecto 2 casos que se escaparon en revision manual.
- **Contratos mecanicos.** `tests/test_audit_contracts.py` con 6
  tests (2 contratos + 4 auto-verificaciones).
- **Lecciones de metodo persistidas en `01_METODO.md`.** 4 reglas
  nuevas en Â§7 + 1 en Â§10. Fichero en 19936 B, cerca del limite
  de 20 KB.

**Commits (tramo).** `325c714`, `982f135`, `c124674`, `840c355`,
`7cafe22`, `76b66c5`, `c54d201`, `f340580`, `4e5bf2a`, `b441f36`,
`aeb7191`, `b3af02c`, `48ef366`, `edb251f`, `a8e3736`, `4757c4f`,
`f4a227f`, `00f41be`, `22e42b6`.

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
  explicito v1->v2 en Â§13. Commit 9ef7497.
- **Congelacion del validador.** sha256 CC01A6AB..., commit d324346,
  python 3.14, seed 20261002, B=2000. Registrado en Â§2.5 de 35 v2 +
  sincronizacion de plan v2 (Â§4.4, Â§4.5, bitacora, riesgos). Commit
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
  Comando documentado en `07_RUNBOOK.md` Â§2.4.1.
- C2 (920): discrepancia H1-B. Bloqueado por auditor externo.
- S-01: segundo barrido sin hallazgos. No se justifica tercer barrido
  con los patrones restantes (P5, P6, P9-P12) â€” ROI < 1.
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
  separada: el guard no distingue "latencia Yahoo" de "fallo real" â€”
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