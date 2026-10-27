# HANDOFF - Radar de Rotacion Sectorial

Documento de traspaso. Autocontenido. Pegar al inicio del nuevo chat
junto con `Consolidacion_Documentos/00_ARRANQUE.md`.

**Este fichero se sobrescribe cada sesion.** Historico en git.

**OBSOLETO desde 2026-10-27 (sesion 13).** El HANDOFF canonico vive
en la raiz del repo: `HANDOFF - Radar de Rotacion Sectorial.txt`.
Este fichero se conserva como historico de sesion 11. No usar como
fuente de estado.

---

## 1. ESTADO AL CIERRE (2026-10-26, sesion 11)

Rama:            main
HEAD:            consultar `git log --oneline -1` (no literal)
Sincronizacion:  main == origin/main al cierre del ultimo commit.
Tests:           3823 passed, 2 skipped (filtro estandar, -n auto).
                 Fuente autoritativa: Consolidacion_Documentos/ESTADO_SISTEMA.md
CI HEAD:         verde.
Validation Gate: 11/11 (no reproducible standalone; ver §6).
Working tree:    limpio.
Gobernanza:      docs/gobernanza/REGISTRO.yaml.
Writers:         25 (output_registry.py).
Sistema Wyckoff: holdings-based + snapshot PIT append-only
                 (holdings_snapshots.csv).
structure_ratio: continuo en sector_regime_matrix v2.
Seleccion holdings sector USA: target_weight 0.85 (A.2c, aprobada
                 condicionada por auditor 2026-10-26).
Indices USA y EU: nlargest(20) hasta nuevo frente.

**Sesion 11 (2026-10-26): A.2c de TOP20-COBERTURA.**

Trabajo principal: cambio del selector de holdings por sector USA
de nlargest(20) a acumular por peso hasta 85%. Regeneracion completa
del historico (1086 fechas). Commits principales del lote:

- `404dc7c5` Fase 2.1: yaml + get_stock_list fuente 5 (28 XLI).
- `557048f8` selector `_select_target_holdings` + 15 tests.
- `0c14a8a3` selection_mode paramétrico en compute_wyckoff_snapshot + 4 tests.
- `5e4fe2fc` regenerate_wyckoff_history parametrizado.
- `efd53cfe` MIN_TICKERS_PER_GROUP = TOP_N_HOLDINGS (evita regresión).
- `f4528b49` fix persist replace_date + 2 tests (bug huérfanas).
- `9338dcd4` fix PCR skip NO_DATA + 4 tests (bug gate).
- `e7f27cd0` A.2c completo: run.py + outputs regenerados + pre_top20_target.
- `a14b463b` REGISTRO TOP20 con evidencia.
- `8a0a8b1d` doc 69 §13 + H-03/H-04 evidencia actualizada.

**Efecto en cobertura:**
- XLI 20→44 (57.2% → 85.8%).
- XLF 19→22, XLK 20→22.
- Resto sin cambio material (XLU/XLV/XLY seleccionan tickers nuevos
  pero no están en parquet → cobertura efectiva sin cambio; ver
  `TOP20-FALTANTES-DATOS`).
- 0 regresiones por fila (date, sector) en 11946 filas.

**Bugs colaterales corregidos en la sesión:**

1. `persist()` es append-only por merge_keys. Los 6 CSVs de holdings
   son subconjuntos filtrados por fase: un ticker que sale de la fase
   deja su fila vieja huérfana. Detectado con GE/PWR/VRT en XLI (14
   filas CSV vs 11 snapshot). Fix: `replace_date=True` en
   `wyckoff_holdings.py`.

2. `compute_pcr_signals` retornaba `None` sin `skip()` cuando CBOE
   no responde. El gate detectaba "sin receipt pcr_history" y
   fallaba el run. Fix: `skip('pcr_history', 'NO_DATA')` en los dos
   return None tempranos.

**Dictamen externo 2026-10-26 sobre doc 70-v2:**

APROBACIÓN CONDICIONADA de A.2c. NO aprobación del cierre global
de DICT3-TOP20-COBERTURA. Mantener OPEN. 6 acciones pendientes
(§4.2.1).

---

## 2. SESION 2026-10-26 (sesion 11): detalle

### 2.1. Fase 2.1: expansion declarativa de universo (`404dc7c5`)

`data/universe_expansions.yaml` (nuevo) declara tickers adicionales
por grupo. `get_stock_list()` (fuente 5) los añade al universo de
descarga sin cambiar `TOP_N_SECTOR_COMPONENTS=20`.

Bloque inicial: `xli_extra` (28 tickers, XLI 57.2% → 89.0%).
Tests: `test_universe_expansions.py` (8). El yaml documenta por
grupo: cobertura actual, objetivo, alcanzada, tickers.

### 2.2. Selector por cobertura de peso (`557048f8`, `efd53cfe`)

`_select_target_holdings(group, target_weight=0.85,
min_tickers=TOP_N_HOLDINGS, max_tickers=300)`:
- Busqueda explícita de posición sobre cumsum, NO idxmax().
- Denominador: peso total del grupo, no de la selección.
- NaN en weight: descartado, no imputado.
- Objetivo inalcanzable: devuelve hasta max_tickers con
  alcanzado=False. Nunca finge conformidad.
- MIN_TICKERS_PER_GROUP=TOP_N_HOLDINGS evita regresión de grupos
  que ya cumplían >85% con top-20 (XLE 99%, XLC 97.7%, XLB 90.1%,
  XLP 86.7%, XLRE 86.7%). Sin esto, XLE pasaba de 20 a 13 tickers.
Tests: 15 en `test_select_target_holdings.py`.

### 2.3. selection_mode en compute_wyckoff_snapshot (`0c14a8a3`)

`compute_wyckoff_snapshot(df, usa_sect, usa_idx, eu_idx,
selection_mode="nlargest"|"target_weight")`. Default nlargest para
compatibilidad. `run.py:161` pasa `selection_mode="target_weight"`.
Tests: 4 en `test_wyckoff_snapshot_selection_mode.py`.

### 2.4. Regeneracion historica (backfill)

Script `regenerate_wyckoff_history.py` parametrizado con
`--selection-mode` y `--validate {strict,informative}`.
Backfill completo: 1086 fechas × 11 sectores. Resultado:
- 265691 filas ticker×fecha → 11946 filas agregadas.
- Coste: ~50 min con 12 workers.
- 0 regresiones vs base top-20 (verificado por fila date+sector).
- Expansión solo en XLI, XLF, XLK.

Backfill produce `sector_wyckoff_distribution.v2_full.csv`. Swap
manual a `.csv` + backup pre-swap en `.pre_top20_target.csv`
(versionado, hash 7D988C841C06E567..., 1089011 bytes).

### 2.5. Fix persist huérfanas (`f4528b49`)

Bug detectado al verificar run post-swap: `wyckoff_usa_sector_distribution`
tenía 14 filas XLI (CSV) vs 11 (snapshot). GE/PWR/VRT eran DIST_FORMING
en run anterior, ahora RANGE. Fila vieja seguía en CSV.

Causa: `persist()` hace `append_dedup(hist, new, merge_keys)`, que
solo añade/reemplaza. No elimina filas del histórico que no están
en el df nuevo. Los 6 CSVs son subconjuntos filtrados por fase: un
ticker que sale de la fase deja su fila huérfana.

Fix: parámetro `replace_date=True` en `persist` (output_writer.py).
Borra del histórico todas las filas de la fecha a reescribir, antes
del append. Solo lo activan los 6 CSVs de holdings.
Tests: 2 en `test_persist_replace_date.py`.

### 2.6. Fix PCR receipt ausente (`9338dcd4`)

Bug detectado en run post-A.2c: `Verificacion de outputs FALLIDA:
pcr_history: sin receipt en el run`. `compute_pcr_signals` retornaba
`None` en dos puntos (provider no disponible, get_options_data vacío)
sin emitir `skip`. El gate detecta "sin receipt" y falla.

Fix: `skip('pcr_history', 'NO_DATA')` en ambos return None tempranos.
Tests: 4 en `test_pcr_skip_no_data.py`.

### 2.7. Re-ejecucion H-03 y H-04 sobre nueva base

Doc 69 §13 (anexo, no modifica §2-§10). Resultados:
- H-03: 30/30 (idéntico, opera sobre parquet, no depende de base).
- H-04: `rho_bar` 0.524→0.471, ALL annual IC [0.268,0.754]→
  [0.294,0.765], ancho multiway/naive se mantiene 2.4×, estrato
  [11-20] 129→278 celdas. Criterio §9 sigue sin discriminar.

`DICT3-H04-INTERVALS` sigue OPEN. Decisión del auditor pendiente.

### 2.8. Dictamen externo 2026-10-26 sobre doc 70-v2

Veredicto: APROBACIÓN CONDICIONADA de A.2c; NO cierre global de
`DICT3-TOP20-COBERTURA`. 6 acciones pendientes (§4.2.1).

---

## 3. FRENTES CERRADOS (acumulado)

Sesiones previas (resumen):
- H1-H19, A1-A6, B, C, F, E: cerrados.
- SLPM, líderes por sector, fase ETF del ranking, classify_wyckoff_phase,
  sistema legacy de índices: RETIRADOS en sesión 3 (2026-10-24).
- `SOW-V6`: WONT_FIX 2026-10-24.
- `UTAD_PENDING`/`SPRING_PENDING`: DESCARTADOS pre-registro 61.
- Regla de vigencia unidireccional: DESCARTADA pre-registro 60.

Sesión 10 (2026-10-25):
- `PIT holdings` (holdings_snapshots.csv append-only).
- `H-03 FORMING invalida si rompe rango`.
- `H-06 WLS stability`.
- `structure_ratio` continua + n_estructuras.
- `sector_regime_matrix` schema v2.
- Fix Xetra (date has no attribute date).
- Gate check 11 (migración WARN vs regresión BLOCKED).

Sesión 11 (2026-10-26):
- `PODA-BITACORA-DIA-EN-CURSO`: FIXED.
- `DICT3-H05-FILLNA-FUTURE`: FIXED (`bar_flags` en build_ticker_df).
- `FASE-6-ROUTER-UNIFICADO`: FIXED (regimes delega en DataRouter,
  `effr` en FredProvider, ancla SOFR_NACIMIENTO actualizada).
- `A.2c TOP20 sectores USA`: APROBADA CONDICIONADA por auditor.
- `PERSIST-REPLACE-DATE`: FIXED (bug huérfanas en subconjuntos).
- `PCR-SKIP-NO-DATA`: FIXED (bug receipt ausente).

---

## 4. PENDIENTES VIVOS

### 4.1. Fases del refactor de outputs

Fases 1-4b y 6 cerradas. Queda:

- **Fase 5:** `structure_score` continuo. Ya hay `structure_ratio`
  continua; falta diseñar el score compuesto y su validación empírica.

### 4.2. Deudas técnicas OPEN

Generales:
- `ACCUMULATION-ZERO`: estado transitorio por diseño. No es bug.
- `VALIDATION-GATE-NO-STANDALONE`: gate solo dentro de `py run.py`.
- `IWM-HOLDINGS-EXPANSION`: sub-fase descarga, protocolo 6 pasos.

Del dictamen externo 2026-10-25 (doc 68):
- `DICT3-H04-INTERVALS` (OPEN): pre-registro doc 69 + implementación +
  §12 + §13. Decisión FIXED/OPEN pendiente de auditor.
- `DICT3-TOP20-COBERTURA` (OPEN): A.2c aprobada condicionada. Ver §4.2.1.
- `DICT3-REACCUMULATION` (OPEN): pre-registro + implementación.

#### 4.2.1. Acciones pendientes del dictamen 2026-10-26 sobre A.2c

El auditor aprobó A.2c condicionada y pidió 6 acciones concretas
antes de reconsiderar el cierre global. En orden de dependencia:

1. **P1-H01** (selector, contrato): sustituir el retorno booleano
   `alcanzado` por estados `REACHED`/`UNREACHABLE`/`INVALID_UNIVERSE`.
   Grupo vacío, todos los pesos NaN o suma<=0 NO deben considerarse
   conformes. Añadir tests específicos.

2. **P1-H03** (backfill, invariantes): pasar de "0 regresiones
   porque n_total no baja" a verificar 6 invariantes por fila:
   (a) 1086 fechas × 11 sectores, sin duplicados;
   (b) fase y eventos idénticos para ticker común;
   (c) count_* reconcilia con n_total;
   (d) pct_* reproducible desde count y denominador;
   (e) CSV coincide con snapshot;
   (f) diferencias explicadas por expansión y disponibilidad.

3. **P1-H02** (cobertura efectiva): tabla ponderada por grupo
   (seleccionado / con datos / evaluable / publicado). Definir
   formalmente qué es "evaluable" vs "publicado". Mantener
   `TOP20-FALTANTES-DATOS` y `TOP20-EU-HOLDINGS-INCOMPLETOS` abiertos.

4. **P1-H04** (sesgo PIT visible): la limitación "composición 2026Q4
   aplicada retrospectivamente" debe ir en los artefactos consumidos
   (no solo en doc de auditoría). Nota en reporte o metadato.

5. **P1-H06** (gobernanza separada): tickets independientes en
   REGISTRO para:
   - persistencia replace_date (ya FIXED en código, falta registro).
   - PCR skip NO_DATA (ya FIXED en código, falta registro).
   - Índices USA (SPY 49.9%, QQQ 66.8%, DIA 85%).
   - Índices EU (FEZ 66.3%, DAXEX 78.4%, ISF.L 60.8%, LYXI 93.2%).
   - IWM (6.2%).

6. **P1-H05** (H-04 separado): no mezclar aprobación de A.2c con
   cierre de H-04. `DICT3-H04-INTERVALS` sigue OPEN.

**Corrección documental adicional:** la tabla original dice
"XLU 83.1%, XLV 78.9%, XLY 84.5%" pero el resumen de A.2c decía
"resto no cambia porque ya cumplía". Eso es falso: XLU/XLV/XLY no
cumplían nominalmente. No cambian materialmente **porque sus tickers
extra no están en parquet**. Corregir en doc 70-v2b.

### 4.3. SOW-BEAR-CONFIRMATION (IN_PROGRESS)

- Requiere contrato v1.10 firmado por auditor externo.
- Siguiente acción: redactar `docs/auditoria/wyckoff/57_contrato_v1_10.md`.
- Sin código en producción hasta firma.
- `detect_sow_v2` en código con tests. `WYCKOFF_SOW_*` = None.

### 4.4. Deuda documental residual

- `00_ARRANQUE.md` línea 196: dice "5 skipped" (real 2). Foto
  histórica del run de sesión 3. Corregir o marcar como histórica.
- `62_sistema_completo_2026-10-24.md`: parcialmente obsoleto.
- `00_ARRANQUE §3`: bloque scheduled-verify dice "20 writers"
  (actual 25). Foto histórica.
- Doc 70-v2b pendiente de escribir (respuesta al dictamen 2026-10-26).

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
- Tests: ver ESTADO_SISTEMA.md. Al cierre sesión 11: 3823 passed,
  2 skipped.
- Gobernanza: `[OK] Gobernanza coherente`.
- Bitácora: `[OK] Bitacora dentro de la regla` (29 sesiones, poda
  aplicada en el cierre de sesión 11).
- pyflakes silencio, compileall OK.

---

## 6. NOTAS CRITICAS

Verificación:
- Verificar en CI, no solo en local.
- Un cambio = una verificación = un commit.
- pyflakes sobre todo el repo (.), no por fichero.
- git diff --check después de cualquier borrado por regex.
- Antes de cambiar firmas de tuplas/dicts, enumerar TODOS los
  consumidores via AST, no regex de texto.
- Cuando un except captura y sigue, verificar si el problema está
  silenciado.
- Auditar en 2 dimensiones: headers Y menciones textuales.
- Schema bump: gate distingue migración pendiente (WARN) de
  regresión (BLOCKED).
- **`git checkout -- outputs/` puede borrar trabajo no commiteado**.
  Antes de revertir outputs, hacer backup `.bak` o commit temporal.
- **Worktree antes que lanzar procesos largos**: si el proceso va a
  correr 30+ min, asegurarse de estar en el sitio correcto.

Herramientas:
- `@'...'@` (single-quote) en PowerShell: no interpola, seguro para
  Python con backticks. Si el contenido tiene `'@` al inicio de
  línea, termina el string. Ficheros críticos: usar
  `[System.IO.File]::WriteAllText` con `[System.Text.Encoding]::UTF8`.
- Mojibake en consola != mojibake en el fichero. Verificar por
  bytes antes de abrir item.
- Commit con mensaje largo y backticks: usar `-F fichero`.
- El test de gobernanza exige commit alcanzable + evidencia >= 20.
- pytest xdist: fallos muestran "gwN" (worker). Normal.
- `git worktree add -b <branch> <path>` (con `-b` para crear la
  rama si no existe).

Contratos:
- Receipt fail-closed (enmienda v1.1). BLOCKED o UNAVAILABLE -> no
  se emite receipt.
- Contrato EQUITY_EOD critical=True. coverage < 0.90 bloquea.
- Gobernanza canónica. Items a docs/gobernanza/REGISTRO.yaml.
- `sector_regime_matrix.py` publica `structure_ratio` continua +
  `n_estructuras`. NO `structure_positive` (retirada).
- `holdings_snapshots.csv` append-only. En 6-12 meses habrá historia
  PIT real. Históricos previos (2022-2026) regenerados con
  composición 2026Q4 tienen sesgo de supervivencia (ya reconocido
  desde sesión 3).
- **`persist(..., replace_date=True)`**: necesario para writers cuyo
  df es subconjunto filtrado (wyckoff_holdings 6 CSVs). Sin esto,
  tickers que salen de la fase dejan filas huérfanas.
- **`compute_pcr_signals(persist_history=True)`**: emite skip
  `NO_DATA` cuando CBOE no responde. Sin esto, el gate falla por
  "sin receipt".
- **Selector `_select_target_holdings`**: usa
  `MIN_TICKERS_PER_GROUP=TOP_N_HOLDINGS=20` para no regresar grupos
  >85%. Objetivo `TARGET_WEIGHT_COVERAGE=0.85`. Denominador: peso
  total del grupo.
- `compute_wyckoff_snapshot(selection_mode=...)`: default `nlargest`.
  `run.py` pasa `target_weight` (sectores USA). Índices USA y EU
  siguen nlargest.
- `daily_run` con gate.should_run=false (finde/festivo): runs de
  1 min en skipped. No es fallo del pipeline.
- NaNs >10% en ~35 tickers (ABNB, APP, BE, CARR, CASY, CEG, COIN,
  CRWD, CTVA, DASH, DDOG, DOW, EXE, FDXF, FOX, FOXA, GEHC, GEV,
  HONA, HOOD, KVUE, MRNA, OTIS, PLTR, PSKY, Q, RDDT, SNDK, SOLV,
  UBER, VICI, VLTO, VRT, VYLR, WBD, XLC) son preexistentes (IPOs
  recientes con histórico parcial en Yahoo).

---

## 7. REFERENCIAS

Corpus normativo: Consolidacion_Documentos/ (12 docs).
Gobernanza: docs/gobernanza/ (README, REGISTRO.yaml, ANCLAS.yaml).
Contrato del gate: docs/auditoria/daily_run_gate_contrato_v1.md +
enmiendas v1.1 y v1.2.
Expediente Wyckoff: docs/auditoria/wyckoff/ (51-70).

Documentos clave:
- `70_top20_cobertura_design_v2_2026-10-26.md`: respuesta doc 70-v1
  + los 7 hallazgos del dictamen 2026-10-26. APROBADO CONDICIONADO.
- `70_top20_cobertura_design_2026-10-26.md`: v1, NO APROBADO.
- `69_h04_intervals_2026-10-26.md`: pre-registro congelado + §12
  resultados top-20 + §13 resultados target_weight.
- `68_respuesta_dictamen_2026-10-25.md`: respuesta formal al auditor.
- `66_regeneracion_historico_2026-10-25.md`: regeneración H-03.
- `65_unificacion_estructura_2026-10-25.md`: Bloque 4, snapshot único.
- `64_schema_versions_outputs.md`: Fase 1, schema versioning.
- `63_plan_sistema_holdings_2026-10-24.md`: plan sistema holdings.
- `62_sistema_completo_2026-10-24.md`: mapa técnico (parcialmente
  obsoleto).
- `60_current_phase_rule_v1.md`, `61_utad_spring_pending_v1.md`:
  pre-registros fallidos conservados.

---

**Fin del handoff.**
