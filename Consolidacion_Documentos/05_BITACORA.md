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

Al cerrar una sesion nueva, se anade arriba (las mas recientes primero). Si hay mas de 5, la mas antigua se elimina o se resume en el historico.

---

## 3. SESIONES

### 2026-09-29 — G-01 (truncado stock_prices) + P66-01 (extraccion 14)

**Objetivo.** Diagnosticar el run rojo 36466343234 (slot 17 11, retraso GitHub 7h19m).
El guard abortaba el push por `last_date=2026-09-28 > expected_session=2026-09-25`.
Causa raiz: `stock_prices.parquet` es multi-mercado (US + LSE + Euronext + Xetra),
pero `expected_session` es NYSE-only por construccion. En ventana UE-cerrada +
USA-abierta, el df contiene fila europea con date > expected_session.

**Hecho.**

*G-01 (src/stock_data_loader.py).*
- Truncado del df a `index <= expected_session` solo para el parquet.
- El df en memoria se devuelve intacto: `leaders.py` ya lo trunca por
  `resolve_effective_date` antes de propagar.
- Ningun consumidor aguas abajo usa la fila europea parcial.

*P66-01 (test_p66_contract.py + Consolidacion_Documentos/06_IAE_P65_P66.md).*
- Hallazgo colateral: `cdf47ad` (borrado del corpus antiguo) elimino
  `docs/auditoria/iae/NIPC_CONTRATOS_SEMANTICOS_v1.md`. Los 10 tests
  Capa A de P66 quedaron huerfanos (creados en commits posteriores
  sobre vista obsoleta del repo).
- Extraccion del §14 (P65 + L3 cruzada P66, 313 lineas) a
  `Consolidacion_Documentos/06_IAE_P65_P66.md`.
- Reapuntado CONTRATO en `tests/test_p66_contract.py`.

**Refactor G-01 + tests (ddb4d31).**
- El fix G-01 estaba inline en `download_stock_prices`, sin cobertura.
- Se extrae a `_truncate_to_expected_session(df, expected_session)`
  -> (df_truncado, n_excluidas). Contrato documentado.
- 8 tests unitarios en `test_stock_data_loader_helpers.py`: noop (None,
  vacio, index no-Datetime, index == expected, index < expected),
  caso G-01 (descarta fila post-expected), no mutacion, y test de
  integracion (df truncado al writer + df intacto al caller).
- Suite: 2267 -> 2275 passed + 2 skipped.
- Justificacion de no hacer run end-to-end: el bug solo se manifiesta
  en ventana UE-cerrada + USA-abierta (~16:30-20:00 UTC). Un run
  manual ahora (madrugada UTC) no lo reproduce. Los tests unitarios
  cubren la logica independientemente de la hora y del slot.

**A6.1 (nucleo compartido).** Gate 0 + auditoria de
`src/utils.py` (629 LOC), `src/instrument_registry.py` (863 LOC),
`src/dependency_tracker.py` (112 LOC).

- Gate 0: `outputs/audit/A6_1_1_20260929_011819.txt` (426 lineas).
- **A6.1-01 (MEDIA) CORREGIDO (2f63db7).** `utils.py::_latest_closed_session`
  tenia `except Exception: pass` en el bucle de 30 iteraciones. Si
  `is_trading_session` / `is_session_closed` fallaban sistematicamente
  para un mercado, retornaba None sin traza. `_compute_by_market` lo
  traducia a `status=INVALID` (fail-closed correcto), pero el
  diagnostico se perdia. Fix: contar y reportar el primer error.
  Contrato fail-closed preservado.
- **A6.1-02 (MEDIA) DEUDA DOCUMENTADA, sin patch.** El outer
  `except Exception` de `write_artifact_with_manifest` retorna `{}` y
  deja el parquet previo intacto ante fallo tecnico (dtype, sha, json,
  os.replace). El test `test_artifact_manifest.py:85-103` ancla este
  contrato como intencional. Los 4 callers productivos
  (`data_loader.py:186`, `stock_data_loader.py:865`,
  `cboe_index.py:135`, `futures.py:265/278`) NO comprueban el retorno
  `{}`. El pipeline continua como si la escritura hubiera ocurrido. Fix
  requiere coordinar writer + 4 callers + tests; ROI bajo sin evidencia
  empirica de dano real. Se deja como deuda para bloque C (tests) o
  posterior.
- **Falsos positivos descartados (contra 01_METODO §8.6):**
  - `_latest_closed_session` en utils.py: **no es** patron 6. Compone
    `is_trading_session` + `is_session_closed` de `market_hours.py`.
  - `instrument_registry.py` sin imports top-level: data-driven
    (INSTRUMENTS + 4 funciones puras). `get_instrument_class` llama
    `get_market` internamente (L882).
  - Re-exports `normalize_yahoo_ticker` en data_loader/stock_data_loader:
    validados por `test_registry_yahoo_map.py:27-28`.
  - `dependency_tracker.py:102` `except Exception`: defensivo
    deliberado (import + inspect.signature; fallback a "No disponible").
- **Deuda fuera de A6.1, documentada:**
  `indicators/darkpool_scoring.py:12 def robust_zscore(series)` es
  variante local de 1 arg con contrato distinto al canonico (3 args,
  window=60). A3.4 solo cerro `options.py` y `fls.py`. Reabrir en
  bloque de calculo.

**Commits.** 3f789f1 (G-01), 5a039e6 (P66-01, amend de 6550b13),
ddb4d31 (refactor G-01 + tests), a4095b1 (docs cierre), fabcdae (snapshot),
2f63db7 (A6.1-01), 73758ce (snapshot A6.1-01).

**A6.2 (workflows, en curso).** Gate 0: `outputs/audit/A6_2_1_20260929_012650.txt`
(557 lineas, 10 workflows). Clasificacion:

- **A6.2-02 (BAJA) CORREGIDO (4fbb135).** `_cron_probe.yml` residual
  (introducido en 03f3b42 el 2026-09-27 para diagnosticar el finding
  "slot '17 3' no dispara"). El probe confirmo lo contrario: el cron
  SI dispara, con retrasos variables 2-8h. Finding cerrado con causa
  raiz: no-disparo era falso, es retraso variable de GitHub Actions.
  Probe eliminado. Comentario de `daily_run.yml` actualizado. Tabla
  de `02_ARQUITECTURA` limpiada.
- **A6.2-01 (MEDIA) DEUDA DOCUMENTADA.** 4 workflows comparten cron
  `17 6 UTC` (`update_macro_manual`, `update_qqq_sec_flow`,
  `update_sec_nport`, `update_sec_13f`). Dias 15 ene/jul (2 concurrentes),
  dias 20 ene/abr/jul/oct (2-3 concurrentes). Todos con
  `git pull --rebase` + `git push` sobre `main`. Ventana de race real;
  sin evidencia empirica de dano en 5 meses. Sin patch por ROI bajo.
- **A6.2-03 (MEDIA) DEUDA DOCUMENTADA.** 7 workflows `update_*`
  comparten estructura identica (checkout + setup-python + pip install
  + compileall + script + git add + commit + pull --rebase + push).
  Sin `workflow_call` ni composite action. Refactor masivo no procede
  sin contrato.
- **A6.2-04 (BAJA) DEUDA DOCUMENTADA.** `git pull --rebase` sin retry
  en 7 workflows. Fallo por conflicto → workflow rojo. Combinado con
  A6.2-01, ventana de conflicto.
- **A6.2-05 (MEDIA) CERRADO por A6.2-02.** Los 5 slots de
  `daily_run.yml` disparan con retrasos variables (2-8h observados),
  no con timing fijo. Documentado en el comentario de `daily_run.yml`.
- **Falsos positivos descartados:** `pytest tests/ validation/` en CI
  colecta solo `tests/` (2275); `validation/` no tiene tests pytest,
  sus 7 scripts se invocan en steps dedicados.

**A6.3 (scripts).** Gate 0: `outputs/audit/A6_3_1_20260929_013536.txt`
(822 lineas, 37 scripts + 3 en audit/).

- **A6.3-01 (MEDIA) CORREGIDO (1049b2a).** `update_european_holdings.py`
  y `update_sector_holdings.py` no tenian `if __name__ == "__main__"`.
  Cualquier import disparaba red + escritura (11 GETs a SSGA en sector;
  3 subprocess en european). Footgun real. Refactor a `main()` +
  `if __name__ == "__main__"`.
- **A6.3-02 (BAJA) CORREGIDO (1049b2a).** `subprocess.run(cmd,
  shell=True)` en `update_european_holdings.py:9`. Comando fijo sin
  metacaracteres, sin input externo. Cambiado a lista de args.
- **A6.3-03 (BAJA) DESCARTADO.** Los 20 ORPHAN del Gate no son dead
  code: `parse_*`/`amundi_holdings` se consumen via
  `update_european_holdings.py`; los `iae_*` y `generate_*` son
  auditoria reproducible manual documentada en `03_IAE §5` y
  `02_ARQUITECTURA §8`. El detector del Gate solo mira workflows
  directos.
- **A6.3-04 (BAJA) DESCARTADO.** `scripts/audit/*.py` sin `sys.exit`:
  exploratorios, se invocan manualmente, el retorno no se usa.
- **A6.3-05 (BAJA) DESCARTADO.** `datetime.now()` en 8 scripts es
  log de ejecucion (backup ts, timestamp UTC, doc autogenerada). No
  como fecha de observacion.
- **CLI contract:** 0 issues en 16 invocaciones de workflow contra
  12 scripts con argparse.

**A6.4 (validation + data/validator.py).** Gate 0:
`outputs/audit/A6_4_1_20260929_014123.txt`.

- **A6.4-01 (MEDIA) CORREGIDO.** `validate_history_quality.py` y
  `verify_leader_selection.py` ejecutaban al importar. Refactor a
  `main()` + guard.
- **A6.4-04 (MEDIA) CERRADO por dead code.** `run_all_audits.py`
  (354 LOC) sin caller, ultima modificacion 2026-09-10, invoca
  `['py', ...]` (Windows-only). Eliminado.
- **A6.4-02 (BAJA) WONT FIX razonado.** Los 2 scripts de validacion
  cruzada (`cross_provider_validation`, `qqq_sec_nport_cross_validation`)
  son descriptivos por diseno: escriben CSV a outputs/audit/ como
  artifact. No son gates. El contrato se preserva.
- **A6.4-05 / A6.4-03 (BAJA) falsos positivos.** `verify_leader_selection`
  ya limpiaba el md anterior. `data/validator.py` es solo 2 funciones
  puras.
- **Bug introducido y corregido en el refactor:** dos `import os`
  internos en `verify_leader_selection` se volvieron locales a
  `main()` y rompian refs previas. Eliminados.

**B (IAE) - bloque abierto y cerrado.**

Gate 0: `outputs/audit/B_1_1_20260929_015604.txt` (510 lineas, 36
ficheros, 8202 LOC).

- **B-01 (MEDIA) CORREGIDO (7e7f429).** `TargetUniverse.__post_init__`
  usaba `assert` para 4 invariantes estructurales. Bajo `python -O`
  los asserts se eliminan y las invariantes desaparecen
  silenciosamente. Sustituidos por `raise BuildTargetError` con
  mensaje que incluye tamanos concretos.
- **B-01b (test) (082cc82).** 5 tests cubriendo las 4 invariantes
  + caso bien formado. Coverage target_builder 65% -> 70%.
- **B-02 (docs) (49b471a).** 3 docstrings del modulo IAE actualizados
  al corpus consolidado (`06_IAE_P65_P66.md`). El `.md` original
  fue borrado en `cdf47ad`.
- **B-03 (docs) (0b0bb26 + 9da966f).** Cifras de `03_IAE` corregidas
  tras verificacion: LOC 8178 -> 8202, `35 funciones AST` -> 19 en
  `reporting_dedup.py`, firma completa de `compute_nipc_contractual`,
  censo de tests 845 -> 763 funciones test_ AST (871 pytest).
- **Deudas documentadas (sin patch por ROI bajo):**
  `build_effective_reporting_snapshot` sin caller productivo
  (ya declarada en `03_IAE §10`). C2 (920) OPEN. H5.3 cron nov 2026.

**B-04/05/06/07 + C-07 (auditoria funcional de IAE).**

Tras cerrar B-01/B-02/B-03, se hizo auditoria **funcional** (no solo
estructural): probes sobre parquets 13F reales y sobre la cadena
`run.py -> iae_section -> pipeline_contractual -> security_identity`.

- **C-01 (MEDIA) CORREGIDO (a5405e5).** `_derive_status` con
  `paired_weighted_share_coverage=None` lanzaba `TypeError`
  (`None >= float`).
- **C-04 (BAJA) CORREGIDO (a5405e5).** `_filter_canonical` no
  descartaba SSHPRNAMT negativo.
- **C-05 (MEDIA) CORREGIDO (4ec29d3).** Audit trail con
  `reporting_for_manager_cik == filing_manager_cik` ("A reporta
  para A") en lugar del representado real.
- **B-05 (docs) CORREGIDO (33b8cca).** E2E reconciliaba contra golden
  FROZEN obsoleto (12_5_historic); default pasa a `current.json`,
  `--historic` para el golden. Corregida doc erronea sobre H1-A.
- **B-06 / S-03 (MEDIA) CORREGIDO (ea0c3cb).** `pipeline_contractual`
  pasaba el CSV de equivalence crudo a `resolve_batch_identities`;
  `valid_from` llegaba como str y `_find_active_equivalence` comparaba
  `str <= Timestamp` -> TypeError. Fix: `load_cusip_equivalence`
  (valida + normaliza dtypes). Endurecida invariante CANONICAL ->
  `canonical != None`. Encadenado.
- **B-07 (MEDIA) CORREGIDO (43b0537).** `catalog_manifest.json` con
  2 snapshots abiertos. `target_catalog_as_of` inoperativo
  (CatalogAmbiguous). Cerrado `20260921_01`.
- **C-07 (BAJA) CORREGIDO (3902503).** NBSP en TITLEOFCLASS
  (`CL<NBSP>A`) caia a UNRESOLVED. Normalizado whitespace Unicode.

**Falsos positivos descartados** (con datos reales, sin fix):
- `.upper()` inconsistente en `relationships.classify_token`: el
  dataset SEC usa `NONE` mayusculas de forma consistente (0 casos).
- `operational_universe._apply_5_3b` dead-code parcial: es diseño
  (target excluye CUSIPs fuera de Official List por §5.3).
- `validate_membership` sin caller productivo: deuda documentada.
- `cusip_resolver.resolve_cusip` sin caller productivo: sin efecto.
- Divergencia delta/target (553k vs 240): es diseño del contrato,
  no bug.

**B CERRADO.** Suite: 2275 -> 2294 passed + 2 skipped.

**C (auditoria de tests) CERRADO.**

- Gate 0: 179 ficheros, 28817 LOC, 2115 funciones `test_`, 0 ficheros
  sin asserts. Patrones sospechosos clasificados: todos falsos
  positivos (ramas defensivas, fixtures MagicMock, skips condicionales
  de entorno).
- **C-07 (BAJA) CORREGIDO (3902503).** NBSP en TITLEOFCLASS
  (`CL<NBSP>A`) caia a UNRESOLVED.
- **C-08 (MEDIA) CORREGIDO (09bfa13).** `volatility_regime` mapeaba
  z=NaN a STRESS por cascada de comparaciones. Evidencia empirica:
  1803/1884 STRESS historicos eran falsos positivos. Fix: NaN -> "N/D".
  Coherente con el principio N/D del sistema. Impacto: no afecta a
  ninguna decision del pipeline (macro_regime usa VIX directo); solo
  a la etiqueta del reporte y a `detect_cross_module_conflict`.
- **Cobertura real por modulo**: identificados modulos con cobertura
  baja (`macro_manual_loader` 12%, `european_coverage` 12%,
  `pipeline_contractual` 27%, `data_loader` 50%). Sin bug detectado
  tras inspeccion. Deuda de tests, no de codigo.
- **Tests anclados a numero de linea**: ya cerrado en A3.3-13.

**C CERRADO.** Suite: 2294 -> 2297 passed + 2 skipped.

**A6 CERRADO** (A6.1 + A6.2 + A6.3 + A6.4). Resumen del bloque:

- A6.1 (nucleo compartido): 1 fix (A6.1-01), 1 deuda documentada (A6.1-02).
- A6.2 (workflows): 1 fix (A6.2-02, eliminacion del probe residual),
  3 deudas documentadas (A6.2-01/03/04).
- A6.3 (scripts): 2 fixes (A6.3-01/02), 3 descartes razonados.
- A6.4 (validation): 2 fixes (A6.4-01/04), 1 WONT FIX, 2 falsos positivos.

Suite final: 2275 passed + 2 skipped + 0 failed (antes de A6: 2267,
+8 tests nuevos del refactor G-01).

**Commits.** ... 2f63db7 (A6.1-01), 73758ce (snapshot A6.1-01), 4a31d4f
(cierre A6.1), 4fbb135 (A6.2-02), d321802 (cierre A6.2), 1049b2a
(A6.3-01/02), 84bb1ee (cierre A6.3), +A6.4 (pendiente de SHA final).

**Pendiente.** A6.1 (nucleo compartido). B (IAE completo). C (tests).
H5.3 (cron nov 2026). C2 (920).

**Proximo paso sugerido.** Retomar A6.1 (nucleo compartido). Verificar en
CI, cuando corra un slot real, que el guard pasa con el parquet nuevo.

---

### 2026-09-28 (tarde/noche) — Auditoria interna A1-A5 del sistema + refactor documental

**Objetivo.** Auditoria linea a linea del sistema por bloques, del nucleo temporal al pipeline. Posteriormente: consolidacion documental (dado que el corpus antiguo era inviable de usar).

**Hecho.**

*Auditoria A1 (nucleo temporal).*
- Calendario NYSE algoritmico (Computus + nth-weekday + observed). Bug: fuera de 2026-2027 los festivos no se reconocian.
- `last_expected_market_date` con tz Madrid. Bug: `datetime.now()` naive en CI (UTC) clasificaba mal.
- Guards de 366 iteraciones en bucles sin red de seguridad.
- Aridad 4-tuple uniforme en `compute_liquidity_score` (antes 3 o 4 segun rama).
- 5 commits. HEAD bab45a4..2c1bb13.

*Auditoria A2 (providers).*
- BME 0/19: `today = last_expected_market_date(reference_date)`. Walk-back a sesion publicada.
- tz-naive en `european_coverage` y `data_quality` (helpers `_ref_to_date`).
- SSGA fund-flow migrado al contrato comun `_fund_flow_utils`.
- `retry_call` sin sleep en el ultimo intento.
- Commits: c1b56e9, 17845c1, 858099e, 64f88cf, 8d0601c, 804dbd8, e3eedc7.

*Auditoria A3 (calculo).*
- Regimenes: aridad 4-tuple, except acotados, alias dead code.
- Breadth: SECTOR_ETFS unificado a `MARKET_TICKERS['sectors']` (7 indicadores + rankings + cross_asset). Fix del generador sintetico del golden (drift por posicion -> drift por alfabetico).
- MTE: whitespace collapse (-565 lineas), except acotados.
- Darkpool: bug real (`robust_zscore` devolvia ndarray en rama `mad==0`).
- SLPM: docstring huerfano, except acotado.
- Commits: 0973fb9, 1fe3bdc, 6ed9c6b, 9b31a9e, a6e2bbd, b723ea8, d3a3002, 1a5730e, 1c2e8d7.

*Auditoria A4 (reporte).*
- Bug observado: `nan` literal en fila 2026-09-14 del reporte (VIX3M/VIX). Fix: guard NaN en 6 campos numericos.
- Barrido del patron: 26 sitios guardados en 7 renders (sentiment, darkpool, leaders, market_context, synthesis, flows_international, rankings).
- 12 except acotados en 7 renders.
- Commits: 64f3331, e4d37f2, 83c1fad, f15ec3a, 9add68b, bbfeb19.

*Auditoria A5 (pipeline).*
- Bug estructural en `leaders.py`: `NameError` silenciado en rama INSUFFICIENT_COVERAGE (`_n_close` fuera del `if`). Corregido indentacion.
- Check muerto en `mte_confirmation.py`: `'all_signals' in dir()` siempre True. Corregido a check real.
- Guards faltantes en `indices_intl.py` (`compute_index_phases` y `download_stock_prices` sin try/except).
- SECTOR_ETFS duplicado en `engines.py` y `diagnostics.py`. Migrado a config.
- ~50 except acotados en 11 ficheros.
- Commits: 59204cd, 28afada, 08d7a23, 6d16bdf, de7f8a3.

*Refactor documental.*
- Nuevo corpus en `Consolidacion_Documentos/`: 00_ARRANQUE, 01_METODO, 02_ARQUITECTURA, 03_IAE, 04_HISTORICO, 05_BITACORA.
- Corpus antiguo (~570 KB) pendiente de borrado.
- Commits: 073b2fe, 3bd00e0, fd6d2c6, c05a02b, e917767.

**Commits de la sesion.** ~20 commits locales + 2 pushes + 4 runs `py run.py` completos.

**Pendiente.**
- Borrar corpus antiguo de `docs/auditoria/`.
- A6 (utils, registry, tracker, workflows, scripts, validation).
- B (IAE completo).
- C (tests).

**Proximo paso sugerido.** Verificar los 6 documentos consolidados (referencias cruzadas, contenido). Borrar corpus antiguo. Retomar A6.

---

### 2026-09-27 (anterior) — Ciclos de cierre del backlog consolidado

**Objetivo.** Cerrar el backlog MEDIA/BAJA pendiente de la auditoria radar 2026-09-26.

**Hecho.** 12 ciclos: Cat 1 quirurgicos (A5-71, F3-18, F2.4-01/02/03), fix side-effect test, 19 warnings pyflakes, F2.4-10/11/12/13 (state MTE), F2.4-04 (dead code NFCI/OAS), FINRA cache (68s -> 0.04s), cron probe + 5º slot + health-check bloque H, A5-70 (refactor BlackRock), K-BLACKROCK-CSV-STALE-01, eliminar `state.py::save_scenario`, Cat 4 desmontado, F5.6-02b/F7-04/F7-01/F5.6-03/F5.7-14/F5.7-19/F5.7-05/F5.6-06.

Ademas: desbloqueo SEC Official List Q2 2026 (sufijo `-txt`, commit 08a6c6a). Auditoria externa del IAE entregada.

**Commits.** 12+ commits. Suite: 2205 -> 2226.

**Pendiente.** H1-B (bloqueante en ese momento), H5.3.

**Proximo paso sugerido.** Cerrar H1-B.

---

### 2026-09-26 — Auditoria radar consolidada + F-IAE-LSE-INTEGRATION

**Objetivo.** Cerrar auditoria radar externa (205 hallazgos). Integrar scraper LSE.

**Hecho.**
- Auditoria externa: 205 hallazgos (6 ALTA, 98 MEDIA, 101 BAJA). 6 ALTA corregidos.
- F-IAE-LSE-INTEGRATION: 20 tickers .L via scraper privado, override parcial de Close. Verificado en CI real (run 36208872855). +91 tests (1766 -> 1857).

**Pendiente.** Ciclos MEDIA/BAJA.

**Proximo paso sugerido.** Cerrar backlog.

---

### 2026-09-25 — F-IAE-CRON-02 / F-IAE-GATE-01

**Objetivo.** Resolver fallo del run 36080921484 (262/313 tickers sin Close por latencia Yahoo).

**Hecho.** Multi-slot (4 disparos) + gate pre-pipeline (`scripts/pipeline_gate.py`) + issue-manager (`scripts/issue_manager.py`). Idempotencia por cobertura, no por `last_date`. Verificado en CI real (run 36148143256).

**Pendiente.** Integracion con workflows.

**Proximo paso sugerido.** Continuar con mejoras de infraestructura.

---

## 4. REFERENCIA A SESIONES ANTERIORES

Para sesiones anteriores al 2026-09-24, ver `git log --oneline`. Resumen tematico en `04_HISTORICO.md`.

---

## 5. REGLAS DE PODA

- La bitacora mantiene las ultimas **5 sesiones**.
- Al anadir una nueva, si hay >5, la mas antigua se elimina.
- **Antes de eliminar**, verificar que sus decisiones clave estan en `04_HISTORICO.md`. Si no, resumir primero.

---

**Fin de la bitacora.**