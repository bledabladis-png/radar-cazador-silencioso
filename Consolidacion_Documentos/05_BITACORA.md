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

**Commits.** 3f789f1 (G-01), 5a039e6 (P66-01, amend de 6550b13).

**Pendiente.** A6.1 (nucleo compartido). B (IAE completo). C (tests).
H5.3 (cron nov 2026). C2 (920).

**Proximo paso sugerido.** Verificar en CI que el proximo run real del
slot post-cierre USA publica parquet sin fallo de guard. Despues retomar A6.1.

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