# HANDOFF - Radar de Rotacion Sectorial

Documento de traspaso. Autocontenido. Pegar al inicio del nuevo chat
junto con `Consolidacion_Documentos/00_ARRANQUE.md`.

**Este fichero se sobrescribe cada sesion.** Historico en git.

---

## 1. ESTADO AL CIERRE (2026-10-25, sesion 10)

Rama:            main
HEAD:            consultar `git log --oneline -1` (no literal; al cierre f5cb8834)
Sincronizacion:  main == origin/main
Tests:           3755 passed, 2 skipped (filtro estandar, -n auto)
                 Fuente autoritativa: Consolidacion_Documentos/ESTADO_SISTEMA.md
CI HEAD:         verde.
Validation Gate: 11/11 (no reproducible standalone; ver §4.1)
Working tree:    limpio.
Gobernanza:      docs/gobernanza/REGISTRO.yaml.
Writers:         25 (output_registry.py; +holdings_snapshots)
Sistema Wyckoff: holdings-based + snapshot PIT append-only
                 (holdings_snapshots.csv)
structure_ratio: continuo (no binario) en sector_regime_matrix v2

**Sesion 10: respuesta al dictamen externo 2026-10-25.**

Tras un dictamen externo NO APROBABLE como sistema de señales y
APROBABLE CONDICIONADO como herramienta descriptiva, se aceptaron
los 9 hallazgos. 6 cerrados end-to-end con codigo + tests + run real,
3 abiertos como items DICT3-*.

Cerrados end-to-end:
- PIT holdings: snapshot append-only diario (holdings_snapshots.csv)
  escrito desde daily_run + 3 scripts de update.
- H-03 (FORMING invalida si rompe rango): corregido. Historico 1086
  fechas regenerado. 52.3% de filas cambian solo FORMING/RANGE;
  confirmadas 0 cambios.
- H-06 (WLS stability = tanh(median/MAD) confusion): corregido a
  wyckoff_stability (MAD-based). 43% signos opuestos -> misma metrica.
- structure_ratio continua + n_estructuras en sector_regime_matrix.
  structure_positive binaria (0/11946) retirada.
- sector_regime_matrix v2 (fresh start).
- Bug Xetra 'datetime.date' object has no attribute 'date' corregido.

Abiertos tras el dictamen (DICT3-*): ver §4.2.
20+ commits, CI verde en cada push.

---

## 2. SESION 2026-10-25 (sesion 10): dictamen + cierres

Detalle por hallazgo en `docs/auditoria/wyckoff/68_respuesta_dictamen_
2026-10-25.md`.

### 2.1. PIT holdings (cerrado)

`etf_holdings.csv` e `index_holdings.csv` son estaticos (sin fecha).
Regenerar historico 2022-2026 con la composicion 2026Q4 introduce
sesgo de supervivencia. Fix: snapshot append-only diario. Cada run,
si la composicion de algun `(source_file, etf)` cambia (hash SHA256
de tickers ordenados), se anaden filas con la sesion del dia.
Ficheros: `src/pipeline/holdings_snapshot.py` (function
`compute_and_persist_holdings_snapshot`), 3 scripts de update
(`update_sector_holdings.py`, `update_index_holdings.py`,
`update_european_holdings.py`), registry `holdings_snapshots`
(writers 24 -> 25). Commit 51812676.
Tests: `test_holdings_snapshot.py` (8), `test_update_scripts_snapshot_call.py` (6).

### 2.2. H-03 FORMING invalida si rompe rango (cerrado)

`wyckoff_orchestrator._dist_derived_phase` y `_acc_derived_phase`
devolvian FORMING sin comprobar si el precio seguia dentro del rango.
Contradecia la regla simetrica de vigencia. Fix: si price fuera de
`[support-tol, resistance+tol]`, RANGE.
Historico regenerado: `scripts/regenerate_wyckoff_history.py`
(12 workers, checkpoint cada 25 tickers, valida contra .csv activo
antes de escribir). 11946 filas, 1086 fechas. 6244/11946 (52.3%)
cambian solo en FORMING y RANGE. Confirmadas (ACC, MARKUP, DIST,
MARKDOWN): 0 cambios.
Commits 3046990c + a655fc95.
Tests: `test_wyckoff_audit_2026_10_25.py` (9: H-02 SOW, H-03 FORMING).

### 2.3. H-06 WLS stability (cerrado)

`stock_metrics.py::compute_stock_metrics` calculaba `stability` como
`tanh(score_median/MAD)` (conviccion direccional: arrastraba el signo
de la mediana, saturada 36% en +-1.0). Debe ser
`wyckoff_stability(combined) = 1 - 2*tanh(MAD(combined)/K)`.
Diagnostico: 43% signos opuestos, corr(stability, t_norm) +0.49 vs
+0.16 para la nueva.
Commit f75d57fd. Tests: `test_h06_stability_fix.py` (3).

### 2.4. structure_ratio + n_estructuras (Bloque 2, cerrado)

`structure_positive` binaria `(pct_acc + pct_markup) >= 50` da
0/11946 con el universo correcto (pct_range domina 47-92%). Retirada.
Publicadas:
  estructura_clara = acc + acc_forming + markup
                   + dist + dist_forming + markdown
  structure_ratio  = (acc + acc_forming + markup) / estructura_clara
  n_estructuras    = estructura_clara (int)
Null si estructura_clara == 0. NO es señal predictiva.
Ficheros: `indicators/sector_regime_matrix.py`,
`src/report/sector_context.py`, `tests/test_sector_regime_matrix.py`.
Commit 1589e079.

### 2.5. sector_regime_matrix schema v2 (cerrado)

Bump `schema_version=2` en `output_registry.py`. Migracion automatica:
`.csv` -> `.v1.csv` (91 lineas historico) + fresh start (12 lineas).
Elimina `structure_positive` del activo.
Commits 80bd5f2b + df563503.

### 2.6. Gate check 11: migracion WARN vs regresion BLOCKED (cerrado)

`src/pipeline/validation_gate.py`: si `spec.schema_version > CSV .schema`
-> WARN (migracion pendiente, se completa en el siguiente `persist()`
del output). Si `spec < CSV` -> BLOCKED (codigo espera schema antiguo).
Antes ambos casos eran BLOCKED.
Commit 1cd28cf2.
Tests: `test_output_schema.py`, `test_validation_gate.py`.

### 2.7. Fix Xetra (cerrado)

`data/providers/xetra_provider.py` fallaba con `'datetime.date' object
has no attribute 'date'` cuando `reference_date` llegaba como `date`
(no `Timestamp`). 19 tickers DE sin datos cada run (SIE.DE, SAP.DE,
ALV.DE, ...).
Fix: helper `_as_date(x)` que acepta `date`/`datetime`/`Timestamp`/
`string`/`None`. Reemplaza 4 usos directos de `reference_date.date()`.
Commit 06391801. Tests: `test_xetra_provider_regression.py` (10).

### 2.8. Doc 68 + items DICT3 (cerrado)

Doc 68 = respuesta formal al dictamen. 1 item FIXED global
(`DICT3-WYCKOFF-AUDIT-2026-10-25`) + 4 items OPEN
(`DICT3-H04-INTERVALS`, `DICT3-H05-FILLNA-FUTURE`,
`DICT3-TOP20-COBERTURA`, `DICT3-REACCUMULATION`).
`SECTOR-REGIME-MATRIX-STRUCTURE-POSITIVE` -> SUPERSEDED (Bloque 2 + doc 68).
Commit f5cb8834.

---

## 3. FRENTES CERRADOS

- H1-H19, A1-A6, B, C, F, E: cerrados.
- 5e legacy Wyckoff: cerrado 2026-10-10.
- SLPM completo, lideres por sector, fase ETF del ranking,
  classify_wyckoff_phase, sistema legacy de indices: RETIRADOS en
  sesion 3.
- `SOW-V6`: WONT_FIX 2026-10-24.
- `UTAD_PENDING`/`SPRING_PENDING`: DESCARTADOS pre-registro 61.
- Regla de vigencia unidireccional: DESCARTADA pre-registro 60.
- `VERIFY-LEADER-SELECTION-ROTO`: FIXED 2026-10-25 (37bca52d).
- `C4.5-SLPM-STRINGS-RESIDUALES`: FIXED 2026-10-25 (c1e4af2e).
- `STRUCTURAL-ENGINE-LEADER-BREADTH-ZOMBIE`: FIXED 2026-10-25 (7679d3fb).
- `REPORT-ENCODING-MOJIBAKE-ADC`: FALSE_POSITIVE 2026-10-25.
- `SCHEMA-VERSIONING-OUTPUTS` (Fase 1): FIXED 2026-10-25 (c9a848b5).
- `DATA-QUALITY-SCHEMA-V` (Fase 2): FIXED 2026-10-25.
- `WYCKOFF-HISTORICO-REGENERADO` (Fase 3): FIXED 2026-10-25.
- `WYCKOFF-ESTRUCTURA-UNIFICADA` (Fase 4): FIXED 2026-10-25 (9960cdac).
- `SECTOR-REGIME-RENAME-RANK` (Fase 4b): FIXED 2026-10-25 (34930b81).
- `DICT3-WYCKOFF-AUDIT-2026-10-25`: FIXED 2026-10-25 (c13cd3a5).
- `SECTOR-REGIME-MATRIX-STRUCTURE-POSITIVE`: SUPERSEDED 2026-10-25.

---

## 4. PENDIENTES VIVOS

### 4.1. Fases del refactor de outputs (2-6)

Fase 1 cerrada (schema versioning). Fases 2, 3, 4, 4b cerradas en
sesion 10. Quedan:

- **Fase 5:** `structure_score` continuo con datos coherentes. Ya
  hay `structure_ratio` continua en sector_regime_matrix; falta
  diseñar el score compuesto final y su validacion empirica.
- **Fase 6:** descarga unificada (`Router` con politica unica para
  fed_data + treasury_data; auditar `b30a9182`, `4bb54f80`).

### 4.2. Deudas tecnicas OPEN

Generales:
- `CURRENT-PHASE-BASIS-CSV`: columna `basis` no va al reporte
  (solo al CSV descargable).
- `ACCUMULATION-ZERO`: estado transitorio por diseño. No es bug.
- `VALIDATION-GATE-NO-STANDALONE`: gate no reproducible standalone.
  Solo se ve dentro de `py run.py`.
- `IWM-HOLDINGS-EXPANSION`: sub-fase descarga, protocolo 6 pasos.
  IWM excluido (top-20 = 6.2% de 1977 holdings).
- `PODA-BITACORA-DIA-EN-CURSO`: mensaje opaco del script. Pendiente
  fix + aclarar seccion 5 de bitacora.

Del dictamen externo 2026-10-25 (doc 68):
- `DICT3-H04-INTERVALS` (OPEN): ICs por cohorte anual, criterios de
  inclusion, tratamiento de solapamiento temporal y correlacion
  cross-sector. Sin esto `structure_ratio` es descriptivo, no
  inferencial.
- `DICT3-H05-FILLNA-FUTURE` (OPEN): añadir columna de calidad por
  fila en `build_ticker_df` (`bar_flags`) sin romper contrato v1.8.
- `DICT3-TOP20-COBERTURA` (OPEN): universo dinamico por grupo
  (objetivo 80-90% peso). XLI 57.2%, SPY 49.9% actuales.
- `DICT3-REACCUMULATION` (OPEN): detector de re-acumulacion de
  continuacion en tendencia alcista. Frente nuevo con pre-registro.

### 4.3. SOW-BEAR-CONFIRMATION (IN_PROGRESS)

- Requiere contrato v1.10 firmado por auditor externo antes de
  implementar.
- Siguiente accion: redactar `docs/auditoria/wyckoff/57_contrato_v1_10.md`.
- Sin codigo en produccion hasta firma.
- `detect_sow_v2` en codigo con tests. `WYCKOFF_SOW_*` = None.

### 4.4. Deuda documental residual

- `62_sistema_completo_2026-10-24.md`: parcialmente obsoleto.
  Cabecera dice "Tests: 3776 passed" (real 3755/2). Correccion
  diferida.
- `00_ARRANQUE.md` §3: bloque scheduled-verify dice "20 writers"
  (actual 25). Foto historica. Correccion diferida.

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
    py scriptserifica_gobernanza.py
    py scripts\genera_indice_auditorias.py
    py scripts\poda_bitacora.py --check
    py -m pytest tests\ -q --tb=line -n auto -m "not integration_real_data and not integration_slow"
    py -m pyflakes .
    py -m compileall . -q

Esperado:
- HEAD = git log --oneline -1.
- ahead 0, behind 0 (o behind N si el bot ha pusheado; git pull --ff-only).
- Working tree limpio.
- Tests: ver ESTADO_SISTEMA.md (fuente autoritativa). Al cierre
  sesion 10: 3755 passed, 2 skipped (con -n auto).
- Gobernanza: [OK] Gobernanza coherente.
- Bitacora: FALLA con PODA-BITACORA-DIA-EN-CURSO abierto. Esperado
  mientras el item siga OPEN. No bloquea trabajo.
- pyflakes sobre . silencio absoluto.
- compileall OK.

---

## 6. NOTAS CRITICAS

Verificacion:
- Verificar en CI, no solo en local.
- Un cambio = una verificacion = un commit.
- pyflakes sobre todo el repo (.), no por fichero.
- git diff --check despues de cualquier borrado por regex.
- Antes de cambiar firmas de tuplas/dicts, enumerar TODOS los
  consumidores via AST, no regex de texto.
- Cuando un except captura y sigue, verificar si el problema esta
  silenciado. Bug de rotacion sectorial duro desde C5.1 hasta sesion
  3 por un except que solo imprimia.
- Auditar en 2 dimensiones: headers Y menciones textuales. Aplica
  a codigo (grep expandido) y a reporte (headers + strings).
- Cuando se modifica schema_version de un output, el gate falla por
  mismatch entre el deploy y el primer persist. Fix: migracion
  pendiente = WARN, regresion = BLOCKED.

Herramientas:
- Here-strings de PowerShell:
  * @"..."@ (double-quote) interpola y rompe con backticks.
  * @'...'@ (single-quote) no interpola, seguro para Python con
    backticks. Pero si el contenido tiene '@ al inicio de linea,
    termina el string.
  * Metodo mas seguro para ficheros criticos (REGISTRO, HANDOFF,
    bitacora): [System.IO.File]::WriteAllText con
    [System.Text.Encoding]::UTF8, edicion por lineas en Python.
- Mojibake en consola PowerShell != mojibake en el fichero.
  Verificar por bytes antes de abrir item.
- Commit con mensaje largo y backticks: usar -F fichero, no -m.
- El test de gobernanza exige commit alcanzable + evidencia >= 20.
  Secuencia: commit de codigo -> commit documental con hash.
- pytest xdist: fallos muestran "gwN" (worker). Es normal.

Contratos:
- Receipt fail-closed (enmienda v1.1). BLOCKED o UNAVAILABLE -> no
  se emite receipt.
- Contrato EQUITY_EOD critical=True. coverage < 0.90 bloquea.
- Gobernanza canonica. Items a docs/gobernanza/REGISTRO.yaml.
- Sistema Wyckoff vigente: holdings-based. ETFs como precio de
  referencia. classify_wyckoff_phase retirado. Indices legacy
  retirados.
- sector_regime_matrix.py publica structure_ratio continua +
  n_estructuras. NO structure_positive (retirada).
- stable PIT: holdings_snapshots.csv append-only. En 6-12 meses
  habra historia PIT real. Historicos previos (2022-2026)
  regenerados con composicion 2026Q4 son susceptibles de sesgo de
  supervivencia.
- daily_run con gate.should_run=false (finde/festivo) muestra runs
  de 1 min en skipped. No es fallo del pipeline.
- NaNs >10% en ~35 tickers (ABNB, APP, BE, CARR, CASY, CEG, COIN,
  CRWD, CTVA, DASH, DDOG, DOW, EXE, FDXF, FOX, FOXA, GEHC, GEV,
  HONA, HOOD, KVUE, MRNA, OTIS, PLTR, PSKY, Q, RDDT, SNDK, SOLV,
  UBER, VICI, VLTO, VRT, VYLR, WBD, XLC) son preexistentes (IPOs
  recientes con historico parcial en Yahoo).

---

## 7. REFERENCIAS

Corpus normativo: Consolidacion_Documentos/ (12 docs).
Gobernanza: docs/gobernanza/ (README, REGISTRO.yaml, ANCLAS.yaml).
Contrato del gate: docs/auditoria/daily_run_gate_contrato_v1.md +
enmiendas v1.1 y v1.2.
Expediente Wyckoff: docs/auditoria/wyckoff/ (51-68).

Documentos clave:
- 68_respuesta_dictamen_2026-10-25.md: respuesta formal al auditor.
- 66_regeneracion_historico_2026-10-25.md: regeneracion 1086 fechas.
- 65_unificacion_estructura_2026-10-25.md: Bloque 4, snapshot unico.
- 64_schema_versions_outputs.md: Fase 1, schema versioning.
- 63_plan_sistema_holdings_2026-10-24.md: plan sistema holdings.
- 62_sistema_completo_2026-10-24.md: mapa tecnico (parcialmente
  obsoleto, cabecera "Tests: 3776" desactualizada).
- 60_current_phase_rule_v1.md: pre-registro v1 fallido.
- 61_utad_spring_pending_v1.md: pre-registro UTAD_PENDING fallido.

---

**Fin del handoff.**
