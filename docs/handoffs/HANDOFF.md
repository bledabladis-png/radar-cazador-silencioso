# HANDOFF - Radar de Rotacion Sectorial

Documento de traspaso. Autocontenido. Pegar al inicio del nuevo chat
junto con `Consolidacion_Documentos/00_ARRANQUE.md`.

**Este fichero se sobrescribe cada sesion.** Historico en git.

---

## 1. ESTADO AL CIERRE (2026-10-24, sesion 3)

Rama:           main
HEAD:           consultar `git log --oneline -1` (no literal; el corpus no fija SHA)
Sincronizacion: main == origin/main
Tests local:    ver `Consolidacion_Documentos/ESTADO_SISTEMA.md`.
                Al cierre: 3776 passed, 2 skipped (suite con -n auto,
                filtro not integration_real_data and not integration_slow).
CI HEAD:        verde (run del commit de cierre).
Validation Gate: 10/10 (no reproducible standalone, ver §4.2).
Working tree:   limpio
Gobernanza:     docs/gobernanza/REGISTRO.yaml
Writers:        24 (output_registry.py)
Sistema Wyckoff: holdings-based (6 CSVs en outputs/history/).

**Sesion 3 completa el plan de demolicion + construccion del sistema
Wyckoff por holdings.** 16 commits: plan 63 + C1 a C5 + 1 fix.
SLPM, lideres por sector, fase ETF del ranking y `classify_wyckoff_phase`
retirados. El sistema clasifica fase directamente sobre los holdings
(US sectores, US indices, EU indices) y produce 6 CSVs descargables
con receipt.

---

## 2. SESION 2026-10-24 (SESION 3)

### 2.1. Plan aprobado (commit `16f2dd8a`, doc 63)

`docs/auditoria/wyckoff/63_plan_sistema_holdings_2026-10-24.md`.
7 commits planificados. Ejecutados como:

- **C1** `83187f8c`: modulo `indicators/wyckoff_holdings_report.py`
  (compute_wyckoff_holdings_report). 6 tablas (usa_sector_acc/dist,
  usa_index_acc/dist, eu_index_acc/dist). Criterio (iii): ACC+FORMING+
  MARKUP y DIST+FORMING+MARKDOWN. Cobertura: >=200 barras para
  incluir, >=460 para score valido. IWM fuera (top-20 = 6.2% peso).
- **C2.1** `a3a63766`: 6 OutputSpec nuevos en `output_registry.py`
  (26 total). Wrapper `src/pipeline/wyckoff_holdings.py`. Cableado en
  `run.py` tras `compute_sector_metrics`.
- **C2.2** `9b5b34cd`: renderers en `src/report/wyckoff_holdings.py`.
  2 secciones nuevas en el reporte ("Tickers en acumulacion" y
  "Tickers en distribucion"). Cableado en `report_generator.py`.
- **C3.1** `0c5b7fb3`: retirada SLPM. Borrados `slpm_v12.py`,
  `state_machine.py`, `state_transition.py`, `src/pipeline/slpm.py`,
  `src/report/slpm.py`. `SLPM_WEIGHTS`, `SLPM_EXPECTED_LEADERS`,
  `_forzar_lideres_slpm`, `render_slpm_*` eliminados. `leader_df`/`top_sector_flow` fuera de `compute_engines`.
- **C3.2** `676ab956`: Validation Gate adaptado. Firma
  `slpm_v12_data` -> `wyckoff_holdings_data`. 3 checks sustituidos
  (SLPM v1.2 -> Wyckoff holdings CSVs; Opportunity Map -> Cobertura
  holdings; Anti-Double-Counting -> Sanity holdings). Gate sigue 10/10,
  receipt sin cambios.
- **C3.3** `cf7659fb`: `scripts/generate_docs.py` limpio. Sin
  `06_indicadores_slpm.md` ni referencias SLPM.
- **C4.1** `753b7f22`: `indicators/stock_metrics.py` extraido de
  `stock_leader.py`. `compute_stock_metrics`, `compute_wls`,
  `_validate_wls_inputs`, `WLS_COMPONENTS`, `compute_all_stock_metrics`.
- **C4.2** `2ede0215`: `src/pipeline/stocks_data.py` extraido de
  `leaders.py`. `load_stocks_data` (df_stocks + effective_meta +
  holdings_df + HOLIDAY_MODE).
- **C4.3** `4bc45fd3`: borrados `stock_leader.py`, `leaders.py`,
  `leader_representativeness.py`, `sector_leader_divergence.py`,
  `sector_annotations.py`, `render_acciones_seleccionadas`.
  `sector_metrics.py` devuelve 3 sub-outputs (wyckoff, rs_internal,
  concentration). Registry 26 -> 24 writers. `run.py` sustituye
  `compute_leaders` por `load_stocks_data` + `compute_all_stock_metrics`.
- **C5.1** `8d9b90a0`: fase ETF retirada del ranking. Tupla del
  ranking de 4 a 3 elementos. `sector_regime.py` sin
  `classify_wyckoff_phase` ni `wyckoff_score`/`build_ticker_df` sobre
  el ETF. `sector_regime_matrix.py` deriva `structure_positive` de
  `sector_wyckoff_distribution_df` (pct_accumulation + pct_markup
  >= 50%). `wyckoff_phase` fuera de `sector_rankings.csv`.
- **C5.2** `c47cb420`: `classify_wyckoff_phase`,
  `classify_wyckoff_phase_meta`, `_is_distribution_candidate`,
  `_compute_precedent`, `_has_sufficient_data` borradas de
  `wyckoff_v1.py`. Constantes huerfanas: `FASE_*` (6),
  `STRUCT_STRONG/WEAK/BASE/DETERIORO`, `T_NORM_WEAK`, `C_NORM_COMPRESSION`,
  `PREC_STRUCT_WEAK`. Conservadas: `PRECEDENT_WINDOW`,
  `PREC_STRUCT_STRONG`, `STRUCT_DETERIORO`, `T_NORM_STRONG` (aun usadas
  por `detect_sow_v2`). `wyckoff_v1.py` queda como modulo de features
  y detectores primitivos.
- **Fix** `96276d05`: import huerfano en
  `test_struct_score_sector_indices.py` que pyflakes en CI detecto
  (local no, porque solo corri pyflakes por ficheros tocados).

### 2.2. Sistema Wyckoff en produccion (post sesion 3)

**Modulos:**
- `indicators/wyckoff_event.py` - dataclasses comunes.
- `indicators/wyckoff_distribution.py` - cascada distribucion.
- `indicators/wyckoff_accumulation.py` - cascada acumulacion.
- `indicators/wyckoff_orchestrator.py` - orquestador (`classify_phase`).
- `indicators/wyckoff_v1.py` - features y detectores primitivos
  (`wyckoff_score`, `detect_spring`, `detect_sos`, `detect_sow`,
  `detect_sow_v2`, `build_ticker_df`, `_atr_normalized`,
  `_trend_component`, `_volume_z`, `_effort_vs_result`,
  `wyckoff_stability`).
- `indicators/sector_wyckoff_distribution.py` - distribucion por
  sector (columnas pct_*, count_*).
- `indicators/wyckoff_holdings_report.py` - 6 tablas descargables.
- `indicators/stock_metrics.py` - metricas per-ticker + WLS.
- `src/pipeline/stocks_data.py` - carga df_stocks + holdings.
- `src/pipeline/wyckoff_holdings.py` - wrapper persistencia 6 CSVs.
- `src/report/wyckoff_holdings.py` - render 2 secciones nuevas.

**Artifacts descargables (outputs/history/):**
- `wyckoff_usa_sector_accumulation.csv`
- `wyckoff_usa_sector_distribution.csv`
- `wyckoff_usa_index_accumulation.csv`
- `wyckoff_usa_index_distribution.csv`
- `wyckoff_eu_index_accumulation.csv`
- `wyckoff_eu_index_distribution.csv`

**Cobertura (2026-10-24):**
- USA sectores (11): top-20 por weight. Peso top-20: XLB 90.1%,
  XLC 97.7%, XLE 99.0%, XLF 72.8%, XLI 57.2%, XLK 79.6%, XLP 86.7%,
  XLRE 86.7%, XLU 83.1%, XLV 78.9%, XLY 84.5%.
- USA indices: SPY (49.9%), QQQ (66.8%), DIA (85.0%). IWM excluido
  (top-20 = 6.2% de 1977 holdings, no representativo).
- EU indices: FEZ 66.3%, DAXEX 78.4%, ISF.L 60.8%, LYXI 93.2%.

### 2.3. Lo que se retiro

- **SLPM** completo: `slpm_v12.py`, `state_machine.py`,
  `state_transition.py`, `src/pipeline/slpm.py`, `src/report/slpm.py`,
  `SLPM_WEIGHTS`, `SLPM_EXPECTED_LEADERS`, tests asociados.
- **Lideres por sector**: `stock_leader.py`, `leaders.py`,
  `leader_representativeness.py`, `sector_leader_divergence.py`,
  `sector_annotations.py`, `render_acciones_seleccionadas`,
  tests asociados. `analisis_lideres.csv` deja de escribirse
  (historico con receipt previo se conserva).
- **Fase ETF del ranking**: `sector_regime.py` sin
  `classify_wyckoff_phase` sobre el ETF. Tupla de 3 elementos.
  `struct_scores` fuera del return.
- **`classify_wyckoff_phase`** y familia: borradas de `wyckoff_v1.py`.
  La clasificacion vive en el orquestador.

---

## 3. FRENTES CERRADOS

- H1-H19, A1-A6, B, C, F, E: cerrados.
- 5e legacy Wyckoff: cerrado 2026-10-10.
- `SOW-V6`: WONT_FIX 2026-10-24.
- `REP-AUDIT-20261020-REPORT-REVIEW-EXTERNAL`: FIXED.
- `UTAD_PENDING`/`SPRING_PENDING`: DESCARTADOS por pre-registro 61.
- Regla de vigencia unidireccional: DESCARTADA por pre-registro 60.
- `REP-AUDIT-20261020-CRON-SLOTS`: FIXED.
- Sistema de lideres por sector (todo): retirado en sesion 3.
- SLPM v1.2 (todo): retirado en sesion 3.
- `classify_wyckoff_phase` (legacy): retirado en sesion 3.

## 4. PENDIENTES VIVOS

### 4.1. SOW-BEAR-CONFIRMATION (IN_PROGRESS)

- Requiere contrato v1.10 firmado por auditor externo antes de implementar.
- Siguiente accion: redactar `docs/auditoria/wyckoff/57_contrato_v1_10.md`.
- Sin codigo en produccion hasta firma.
- `detect_sow_v2` sigue en codigo y con tests: base del futuro contrato.
- `WYCKOFF_SOW_*` = None, protegido por test de invariante
  (`test_wyckoff_v1_contract.py`).

### 4.2. Deuda tecnica abierta (post sesion 3)

- **`current_phase_basis` no va al CSV**: los 6 CSVs de holdings
  tienen columna `basis` pero el reporte no la pinta. Decision de
  producto: añadir columna `basis` al reporte o dejarla solo en CSV.
- **`ACCUMULATION=0` en produccion**: estado transitorio por diseño
  (SOS confirmado + precio dentro del rango). Documentar en 62 o en
  HANDOFF. No es bug.
- **Validation Gate no reproducible standalone**: no hay script ni
  target pytest. El "10/10" es estado del ultimo run productivo. Item
  OPEN. Opciones: documentar (a), escribir wrapper (b, coste alto),
  aceptar (c).
- **Caso TWST en CONFLICT**: documentar en 62 como caso conocido
  (dist formandose + acc formandose simultaneamente). No es falso
  positivo, es ambiguedad real.
- **IWM fuera**: sub-fase de descarga (top-200/300 por peso). Protocolo
  de 6 pasos obligatorio: calendario bursatil, 0 filas H=L=C, receipt,
  test. Decision tomada: fuera hasta que se aborde.
- **Cobertura USA incompleta**: 231/504 holdings en parquet.
  Ampliacion futura con el mismo protocolo.
- **Historia corta**: 14 tickers entre 200-460 barras. Completar
  historico si procede.
- **Render de `wyckoff_holdings_data`**: las 2 secciones nuevas
  renderizan tabla plana. Evaluar si se necesita paginacion o
  agrupacion visual cuando el volumen crezca.
- **`detect_spring`/`detect_sos` legacy**: siguen en `index_leaders.py`,
  `stock_metrics.py`. Decision separada sobre si migrar al sistema
  nuevo (range-anchored) o mantener.

### 4.3. Items OPEN observacionales (fechas)

| Item | Fecha | Accion |
|---|---|---|
| H5.3 | 2026-11-20 06:17 UTC | Verificar cron trimestral 13F |
| REP-AUDIT-20261020-RECEIPT-AUDIT-LAG | 2026-11-17 | Cierre con 4 semanas obs |
| REP-AUDIT-20261020-CRON-SLOTS-OBS | 2026-11-10 | Revision runs 20-10 a 10-11 |
| REP-AUDIT-20261020-SCHEDULER-DELAYS | 2026-11-10 | p95 delays |

### 4.4. BLOCKED_EXTERNAL

- `C2-920`: discrepancia 920 en baseline IAE. Auditor externo.
- `WYCKOFF-5BX`: esperando `n_confirmed_H20_complete >= 550` (~2027).

### 4.5. Documentacion pendiente (C6 sesion 3, no completado)

- `00_ARRANQUE.md`: §3 (estado), §4.2 (frente Wyckoff completo),
  L235 "15 detectores" -> 16, L242/L264 `classify_wyckoff_phase`.
- `02_ARQUITECTURA.md`: arbol de directorios (§2), §6 (SLPM),
  §7 (fases 8a/8b), §10 (decisiones), §14 (alcance), §16 (universos).
- `62_sistema_completo_2026-10-24.md`: L10/L135/L177
  `classify_wyckoff_phase`, L156 `stock_leader.py`, L235 "15" -> 16,
  añadir TWST, `ACCUMULATION=0`, `detect_sow_v2`.
- `04_HISTORICO.md` + `05_BITACORA.md`: entrada de sesion 3.
- Abrir items nuevos en REGISTRO (los de §4.2).

---

## 5. COMO ARRANCAR

    Set-Location D:\Macro_Sectorial
    git log --oneline -5
    git status -sb
    py scripts\generate_estado_sistema.py
    py scripts\verifica_gobernanza.py
    py scripts\genera_indice_auditorias.py
    py scripts\poda_bitacora.py --check
    py -m pytest tests\ -q --tb=line -n auto -m "not integration_real_data and not integration_slow"
    py -m pyflakes .
    py -m compileall . -q

Esperado:
- HEAD = el que devuelva `git log --oneline -1`.
- ahead 0, behind 0.
- Working tree limpio.
- Tests: ver `ESTADO_SISTEMA.md`. Referencia al cierre: 3776 passed
  + 2 skipped (con `-n auto`).
- Gobernanza: `[OK] Gobernanza coherente`.
- Bitacora: `[OK] Bitacora dentro de la regla`.
- **pyflakes sobre `.` (todo el repo), no por fichero.** CI corre
  `python -m pyflakes .`. Si solo se corre por fichero tocado, se
  escapan imports huerfanos en otros ficheros (fallo C5.1 sesion 3).
- compileall OK.

---

## 6. NOTAS CRITICAS

- **Verificar en CI, no solo en local.** Tests con `tmp_path` y fixtures
  sinteticos.
- **Un cambio = una verificacion = un commit.**
- **Verificar EOL (LF/CRLF) y BOM antes de cada patch.**
- **Here-strings >20 lineas o >5 `$` -> fichero temporal.**
- **pyflakes sobre todo el repo (`.`), no por fichero.** Leccion sesion
  3: el import huerfano de `test_struct_score_sector_indices.py` paso
  local y fallo en CI porque `pyflakes .` corre sobre todo el repo.
- **Antes de cambiar firmas de tuplas o diccionarios, enumerar TODOS
  los mocks y fixtures que los construyen.** Leccion sesion 3: la
  tupla del ranking paso de 4 a 3, y el primer barrido dejo 6 tuplas
  sin adaptar en `test_report_generator_smoke.py`. Usar AST sobre
  los tests, no regex de texto.
- **`git diff --check` despues de cualquier borrado por regex**, por
  si queda blank line at EOF.
- **Receipt fail-closed (enmienda v1.1).** BLOCKED o UNAVAILABLE -> no
  se emite receipt.
- **Contrato EQUITY_EOD `critical=True`.** `coverage_at_session < 0.90`
  bloquea publicacion.
- **Gobernanza canonica.** Items a `docs/gobernanza/REGISTRO.yaml`.
  El HANDOFF es vista derivada.
- **Anclas temporales.** Fechas literales a `ANCLAS.yaml`.
- **Sistema Wyckoff vigente**: holdings-based. ETFs solo como precio
  de referencia para RS. `classify_wyckoff_phase` retirado;
  clasificacion via `classify_phase` (orquestador).
- **Regla simetrica de vigencia**: un rango esta vigente mientras el
  precio NO lo ha roto por NINGUN lado. Definicion de rango, no
  umbral.
- **FORMING vs CONFIRMED**: FORMING = rango activo sin SOW/SOS
  confirmado. Senal temprana.
- **CONFLICT es estado honesto**: dos estructuras compitiendo.
  Documentado en 62 (pendiente añadir TWST).
- **Pre-registros fallidos NO se ajustan.** Ver 60 y 61.
- **`sector_regime_matrix.py` deriva `structure_positive` de holdings**
  (`sector_wyckoff_distribution_df`). Pendiente validar que las
  lecturas (`Alineacion positiva`, `Constructivo`, etc.) siguen
  coherentes con la nueva derivacion.
- **Receipt del gate sigue siendo 10/10.** `pipeline_gate.py`,
  `write_completion_receipt.py` y `provenance.py` sin tocar. Los 3
  checks sustituidos mantienen el conteo.
- **Nombres de writers**: 24. `output_registry.py` es la fuente.

## 7. REFERENCIAS

Corpus normativo: `Consolidacion_Documentos/` (12 docs).
Gobernanza: `docs/gobernanza/` (README, REGISTRO.yaml, ANCLAS.yaml).
Contrato del gate: `docs/auditoria/daily_run_gate_contrato_v1.md` +
`daily_run_gate_enmienda_v1_1.md` + `daily_run_gate_enmienda_v1_2.md`.
Expediente Wyckoff: `docs/auditoria/wyckoff/` (51-63 del 2026-10-24).

**Documentos clave para el traspaso:**
- `63_plan_sistema_holdings_2026-10-24.md` - plan del sistema actual.
- `62_sistema_completo_2026-10-24.md` - mapa tecnico (parcialmente
  obsoleto; corregir en C6).
- `60_current_phase_rule_v1.md` - pre-registro v1 fallido.
- `61_utad_spring_pending_v1.md` - pre-registro UTAD_PENDING fallido.

---

**Fin del handoff.**