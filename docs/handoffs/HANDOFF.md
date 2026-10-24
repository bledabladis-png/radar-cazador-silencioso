# HANDOFF - Radar de Rotacion Sectorial

Documento de traspaso. Autocontenido. Pegar al inicio del nuevo chat
junto con `Consolidacion_Documentos/00_ARRANQUE.md`.

**Este fichero se sobrescribe cada sesion.** Historico en git.

---

## 1. ESTADO AL CIERRE (2026-10-24, sesion 3 extendida)

Rama:           main
HEAD:           consultar `git log --oneline -1` (no literal; al cierre f7fdbeba)
Sincronizacion: main == origin/main
Tests:          3705 passed, 2 skipped (filtro estandar).
                Sin filtro: 3719 passed, 2 skipped.
CI HEAD:        verde.
Validation Gate: 10/10 (no reproducible standalone; ver §4.3).
Working tree:   limpio.
Gobernanza:     docs/gobernanza/REGISTRO.yaml.
Writers:        24 (output_registry.py).
Sistema Wyckoff: holdings-based. 6 CSVs en outputs/history/.

**Sesion 3 con dos fases:**

**Fase A (primera parte)** — plan 63 ejecutado casi completo: C1, C2,
C3, C4.1-C4.3, C5, C6.1, C6.2. Retirados SLPM, lideres por sector,
fase ETF del ranking, classify_wyckoff_phase. Sistema holdings-based
construido y cableado.

**Fase B (continuacion, este documento)** — al verificar con pipeline
real se detectaron dos problemas que invalidaban el "cierre" de la
fase A: (1) C4.4 (retirada del sistema legacy de indices) se habia
saltado, dejando 2 secciones legacy en el reporte con mojibake; (2) la
rotacion sectorial llevaba sin actualizarse desde el run del 10:44
porque sector_rank_history.py aun hacia unpacking de la tupla de 4
(tras C5.1 la tupla paso a 3) y el except lo silenciaba. Ambos
arreglados. Ademas se ha detectado deuda residual pendiente (C4.5).

---

## 2. SESION 2026-10-24 (sesion 3)

### 2.1. Plan 63 — primera parte (fase A)

`docs/auditoria/wyckoff/63_plan_sistema_holdings_2026-10-24.md`.

- C1 `83187f8c`: `indicators/wyckoff_holdings_report.py`. 6 tablas.
- C2.1 `a3a63766`: 6 OutputSpec. Wrapper `wyckoff_holdings.py`.
- C2.2 `9b5b34cd`: 2 secciones nuevas en el reporte.
- C3.1 `0c5b7fb3`: retirada SLPM.
- C3.2 `676ab956`: Validation Gate adaptado (10/10, 3 checks sustituidos).
- C3.3 `cf7659fb`: `generate_docs.py` limpio.
- C4.1 `753b7f22`: `stock_metrics.py` extraido.
- C4.2 `2ede0215`: `stocks_data.py` extraido.
- C4.3 `4bc45fd3`: retirada lideres por sector.
- C5.1 `8d9b90a0`: fase ETF fuera del ranking. Tupla 4 -> 3.
- C5.2 `c47cb420`: `classify_wyckoff_phase` y familia retiradas.
- Fix `96276d05`: import huerfano en test.
- C6.1 (a-d): HANDOFF, 00_ARRANQUE, 02_ARQUITECTURA, 62 actualizados.
- C6.2: REGISTRO items, rama feature/wyckoff-cascade borrada.

### 2.2. C4.4 — Retirada sistema legacy de indices (`f7fdbeba`)

El sistema legacy (classify_phase sobre ^GSPC/^DJI/^NDX + top-5
lideres por indice) coexistia con el holdings-based en el reporte.
2 secciones redundantes, con mojibake en el titulo.

**Borrados**: `index_phase.py`, `index_leaders.py`, `indices_intl.py`,
`config/index_tickers.py`, `render_indices_internacionales` (en
synthesis.py), 12 ficheros de test.

**Adaptados**: `run.py` (sin compute_indices_intl), `report_generator.py`
(sin 3 kwargs + render), `synthesis.py` (sin INDEX_CONFIG, _fmt_num,
FLOW_ZSCORE_WINDOW), `test_output_contract_ast.py`,
`test_render_modules_low_coverage_d29.py`, `generate_docs.py`.

**REGISTRO**: 4 items FIXED (REP-AUDIT-20261018-STRUCT,
REPORT-STRESS-LABEL, FLOW-RECONCILIATION, STRUCT-COLUMN) referenciaban
`test_struct_score_sector_indices.py` (borrado). Quitado el test,
anotada la retirada en evidencia.

### 2.3. Bug rotacion sectorial (`8819ba70`)

`sector_rank_history.py:54` hacia `for (ticker, name, score, fase)`
pero C5.1 dejo la tupla del ranking en 3 elementos. El except de
`sectors_base.py:56` lo capturaba (`Rotacion sectorial omitida: not
enough values to unpack (expected 4, got 3)`) y skipeaba
`sector_rank_deltas` con NO_DATA. La seccion "Rotacion sectorial
reciente" del reporte llevaba desde el run del 10:44 sin actualizarse.

Fix: unpacking 4 -> 3. Tests adaptados. `sectors_base.py` con
`traceback.print_exc()` en el except para que futuros fallos no se
silencien. Verificado en run real: rotacion calculada, seccion
presente en el reporte con 11 sectores.

### 2.4. Verificacion con pipeline real

3 runs completos del pipeline (exit 0, gate 10/10, 24 writers OK,
cero mojibake). Los DATA ISSUE (NaNs >10%) de ~35 tickers son
preexistentes (IPOs recientes con historico parcial en Yahoo).

### 2.5. Sistema Wyckoff en produccion (estado actual)

Modulos:
- `wyckoff_event.py`, `wyckoff_distribution.py`,
  `wyckoff_accumulation.py`, `wyckoff_orchestrator.py` (classify_phase
  + regla simetrica + FORMING + CONFLICT).
- `wyckoff_v1.py`: features (`wyckoff_score`, `_atr_normalized`,
  `_trend_component`, `_volume_z`, `_effort_vs_result`,
  `wyckoff_stability`) + detectores (`detect_spring`, `detect_sos`,
  `detect_sow`, `detect_sow_v2`, `build_ticker_df`). Sin
  `classify_wyckoff_phase`.
- `wyckoff_holdings_report.py`: 6 tablas.
- `stock_metrics.py`: metricas per-ticker + WLS.
- `sector_wyckoff_distribution.py`: agregado por sector.

Pipeline:
- `stocks_data.py`: `load_stocks_data`.
- `wyckoff_holdings.py`: wrapper persistencia 6 CSVs.
- `sector_metrics.py`: 3 sub-outputs (wyckoff, rs_internal,
  concentration).
- `engines.py`: tactical/structural/persistence (sin leader_df).
- `validation_gate.py`: 10 checks (3 sustituidos por holdings).

Reporte:
- `src/report/wyckoff_holdings.py`: 2 secciones nuevas.
- Sin SLPM, sin lideres, sin indices legacy.

6 CSVs descargables (`outputs/history/`):
- `wyckoff_usa_sector_acc/dist`, `wyckoff_usa_index_acc/dist`,
  `wyckoff_eu_index_acc/dist`.

Cobertura: top-20 por weight (57.2%-99.0% del peso del grupo). USA
indices: SPY, QQQ, DIA (IWM excluido). EU indices: FEZ, DAXEX, ISF.L,
LYXI.

### 2.6. Lo que se retiro (acumulado sesion 3)

- SLPM completo.
- Lideres por sector (`stock_leader.py`, `leaders.py`, etc.).
- Fase ETF del ranking.
- `classify_wyckoff_phase` y familia.
- Sistema legacy de indices (`index_phase.py`, `index_leaders.py`,
  `indices_intl.py`, `config/index_tickers.py`).

---

## 3. FRENTES CERRADOS

- H1-H19, A1-A6, B, C, F, E: cerrados.
- 5e legacy Wyckoff: cerrado 2026-10-10.
- `SOW-V6`: WONT_FIX 2026-10-24.
- `REP-AUDIT-20261020-REPORT-REVIEW-EXTERNAL`: FIXED.
- `REP-AUDIT-20261020-CRON-SLOTS`: FIXED.
- `UTAD_PENDING`/`SPRING_PENDING`: DESCARTADOS pre-registro 61.
- Regla de vigencia unidireccional: DESCARTADA pre-registro 60.
- SLPM completo, lideres por sector, fase ETF del ranking,
  `classify_wyckoff_phase`, sistema legacy de indices: RETIRADOS en
  sesion 3.
- C4.4: cerrado esta sesion.
- `VERIFY-LEADER-SELECTION-ROTO`: FIXED 2026-10-25 (commit `37bca52d`).
- `C4.5-SLPM-STRINGS-RESIDUALES`: FIXED 2026-10-25 (commit `c1e4af2e`).
- `STRUCTURAL-ENGINE-LEADER-BREADTH-ZOMBIE`: FIXED 2026-10-25 (commit `7679d3fb`).
- `REPORT-ENCODING-MOJIBAKE-ADC`: FALSE_POSITIVE 2026-10-25 (reporte en UTF-8 correcto, mojibake era consola PowerShell).

## 4. PENDIENTES VIVOS

### 4.2. Deudas tecnicas OPEN (sesion 3)

- `CURRENT-PHASE-BASIS-CSV`: columna `basis` no va al reporte.
- `ACCUMULATION-ZERO`: estado transitorio por diseno.
- `VALIDATION-GATE-NO-STANDALONE`: gate no reproducible standalone.
- `IWM-HOLDINGS-EXPANSION`: sub-fase descarga, protocolo 6 pasos.
- `SECTOR-REGIME-MATRIX-STRUCTURE-POSITIVE`: validacion empirica de la
  nueva derivacion (holdings-based desde C5.1).
- `PODA-BITACORA-DIA-EN-CURSO`: script cuenta dia en curso contra
  limite. Bug + ambiguedad en §5.

### 4.3. SOW-BEAR-CONFIRMATION (IN_PROGRESS)

- Requiere contrato v1.10 firmado por auditor externo antes de
  implementar.
- Siguiente accion: redactar `docs/auditoria/wyckoff/57_contrato_v1_10.md`.
- Sin codigo en produccion hasta firma.
- `detect_sow_v2` en codigo con tests. `WYCKOFF_SOW_*` = None.

### 4.4. Deuda documental residual

- `04_HISTORICO` + `05_BITACORA`: sesion 3 documentada. C4.4 + bug
  rotacion anadidos como sub-seccion.
- `62_sistema_completo_2026-10-24.md`: parcialmente obsoleto (no
  menciona C4.4). Correccion diferida.
- `02_ARQUITECTURA` corregido (index_tickers fuera del arbol).

### 4.5. Items OPEN observacionales (fechas)

| Item | Fecha | Accion |
|---|---|---|
| H5.3 | 2026-11-20 06:17 UTC | Verificar cron trimestral 13F |
| RECEIPT-AUDIT-LAG | 2026-11-17 | Cierre con 4 semanas obs |
| CRON-SLOTS-OBS | 2026-11-10 | Revision runs 20-10 a 10-11 |
| SCHEDULER-DELAYS | 2026-11-10 | p95 delays |

### 4.6. BLOCKED_EXTERNAL

- `C2-920`: discrepancia 920 baseline IAE. Auditor externo.
- `WYCKOFF-5BX`: `n_confirmed_H20_complete >= 550` (~2027).

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
- HEAD = `git log --oneline -1`.
- ahead 0, behind 0.
- Working tree limpio.
- Tests: ver `ESTADO_SISTEMA.md`. Al cierre: 3705 passed, 2 skipped
  (con `-n auto`).
- Gobernanza: `[OK] Gobernanza coherente`.
- Bitacora: **FALLA con PODA-BITACORA-DIA-EN-CURSO abierto**. Es
  esperado mientras el item siga OPEN. No bloquea trabajo.
- pyflakes sobre `.` silencio absoluto.
- compileall OK.

---

## 6. NOTAS CRITICAS

- **Verificar en CI, no solo en local.**
- **Un cambio = una verificacion = un commit.**
- **Verificar EOL (LF/CRLF) y BOM antes de cada patch.**
- **Here-strings >20 lineas o >5 `$` -> fichero temporal.**
- **Here-strings con `@\"...\"@` de PowerShell: los backticks rompen el
  triple-quote de Python.** Usar `@'...'@` (single-quote) o escribir el
  script a fichero con `[System.IO.File]::WriteAllText`. Leccion
  sesion 3: 2 fallos por esto.
- **pyflakes sobre todo el repo (`.`), no por fichero.** CI corre
  `python -m pyflakes .`. Un import huerfano en un fichero que no se
  toca sigue fallando en CI aunque `pyflakes <fichero>` de local este
  limpio.
- **Antes de cambiar firmas de tuplas o diccionarios, enumerar TODOS
  los consumidores via AST, no regex de texto.**
- **Cuando un `except` captura y sigue, verificar si el problema esta
  silenciado.** Un `print` + `skip(NO_DATA)` oculta fallos reales. El
  bug de rotacion sectorial duro desde C5.1 hasta la sesion 3 por un
  `except` que solo imprimia.
- **Auditar el reporte en 2 dimensiones**: headers `^## ` Y menciones
  textuales (strings dentro de notas). La auditoria por headers dejo
  pasar 11 menciones a SLPM en el reporte post C4.4.
- **Scripts de patch: no abortar al primer fallo. Acumular errores.**
  Si un replace falla en el item 2 de 4, no se aplica nada pero los
  items 3 y 4 tampoco. Mejor procesar todos e informar.
- **`git diff --check` despues de cualquier borrado por regex.**
- **Receipt fail-closed (enmienda v1.1).** BLOCKED o UNAVAILABLE -> no
  se emite receipt.
- **Contrato EQUITY_EOD `critical=True`.** coverage < 0.90 bloquea.
- **Gobernanza canonica.** Items a `docs/gobernanza/REGISTRO.yaml`.
  El HANDOFF es vista derivada.
- **Sistema Wyckoff vigente**: holdings-based. ETFs solo como precio
  de referencia. `classify_wyckoff_phase` retirado. Indices legacy
  retirados.
- **`sector_regime_matrix.py`** deriva `structure_positive` de holdings
  (`sector_wyckoff_distribution_df`). Pendiente validacion empirica
  (item `SECTOR-REGIME-MATRIX-STRUCTURE-POSITIVE`).
- **`structural_engine.py::compute_structural_score`** tiene un
  parametro zombie `leader_breadth=0.5`. Item OPEN.

## 7. REFERENCIAS

Corpus normativo: `Consolidacion_Documentos/` (12 docs).
Gobernanza: `docs/gobernanza/` (README, REGISTRO.yaml, ANCLAS.yaml).
Contrato del gate: `docs/auditoria/daily_run_gate_contrato_v1.md` +
enmiendas v1.1 y v1.2.
Expediente Wyckoff: `docs/auditoria/wyckoff/` (51-63).

**Documentos clave para el traspaso:**
- `63_plan_sistema_holdings_2026-10-24.md`: plan del sistema actual.
- `62_sistema_completo_2026-10-24.md`: mapa tecnico (parcialmente
  obsoleto; no menciona C4.4).
- `60_current_phase_rule_v1.md`: pre-registro v1 fallido.
- `61_utad_spring_pending_v1.md`: pre-registro UTAD_PENDING fallido.

---

**Fin del handoff.**