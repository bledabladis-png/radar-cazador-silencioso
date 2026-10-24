# HANDOFF - Radar de Rotacion Sectorial

Documento de traspaso. Autocontenido. Pegar al inicio del nuevo chat
junto con `Consolidacion_Documentos/00_ARRANQUE.md`.

**Este fichero se sobrescribe cada sesion.** Historico en git.

---

## 1. ESTADO AL CIERRE (2026-10-24)

Rama:           main
HEAD:           consultar `git log --oneline -1` (no literal; el corpus no fija SHA)
Sincronizacion: main == origin/main tras push final
Tests local:    ver `Consolidacion_Documentos/ESTADO_SISTEMA.md`.
                Al cierre: 3858 passed, 5 skipped (suite completa).
                3849 passed, 5 skipped con -m "not integration_real_data".
Validation Gate: 10/10
Working tree:   limpio
Gobernanza:     docs/gobernanza/REGISTRO.yaml. 124 items.
CI de tests:    .github/workflows/ci_tests.yml (push a main, paths-ignore: data/, outputs/).

**Foco de la sesion 2026-10-24:** cierre del frente SOW universal tras
7 experimentos. `SOW-V6` -> WONT_FIX. Apertura de `SOW-BEAR-CONFIRMATION`
(IN_PROGRESS, pendiente contrato v1.10 + auditor externo).
`WYCKOFF_SOW_* = None` sin cambios.

---

## 2. SESIONES 2026-10-22 a 2026-10-24

### 2.1. 2026-10-22: 11 bugs + Wyckoff B5/B3 (cierre tardio)

Ver `05_BITACORA.md` L92. Commits `d405d539` a `1cc8387a`.

### 2.2. 2026-10-23: scheduled target 2026-10-22 + autonomia + cierre documental

Ver `05_BITACORA.md` L47. Commits:
- `49248879` — cierre documental 22-23 oct (6 ficheros).
- `f677acb6` — SHA real en `REP-AUDIT-20261023-SCHEDULED-VERIFY`.
- `2374b605` — referencias traspaso inexistente -> HANDOFF.
- `838826af` — 3 items auditoria reporte 2026-10-22.
- `7e1582a7` — 4 notas cruzadas en `src/report/` (slpm, sector_context, market_context, volatility_mte).
- `968bcdcc` — cierre `NOTAS-CRUZADAS` con SHA real.

### 2.3. 2026-10-24: cierre SOW universal + apertura bear-confirmation

**Commits clave:**
- `e7327463` — `purged_kfold_sow.py` con criterio pre-registrado.
- `520a8db7` — SOW v2 lift within-ticker.
- `2dcc3325` — C1-M5 con cohorte original.
- `75ba6945` — C1-M10 (M=10 rompe semantica).
- `548d2773` — C1-M5: 2 folds significativos (2018, 2020).
- `b35495c8` — `SOW-V6` IN_PROGRESS tras diagnostico.
- `be2f7670` — cohorte reservada S&P 500 (268 tickers).
- `a3987f1c` — placebo_shift adaptativo.
- `c5485337` — placebo date-shift global.
- `61556a74`/`0c2455ab` — cohorte R3000 + vectorizacion.
- `3d1a215d` — resultado R3000 (FUNCIONA por criterio, pero ventana no evento).
- `b542d381` — `detect_sow_v2` (gate distribution).
- `260569bc`/`281cb378` — event study.
- `81962b59` — `SOW-V6` WONT_FIX + `SOW-BEAR-CONFIRMATION` IN_PROGRESS.
- `1ffc299b` — script descarga R3000 reproducible.
- `ff8f5a6f` — fix H7 en detect_sow_v2 + anclas cohorte.

**Descubrimientos:**
- SOW universal (C1-M5, M=5) tiene senal real vs placebo en 3423 tickers (+0.123 vs -0.077, 21.5 sd).
- Pero NO es universal: 4/9 folds positivos, IC95 entre anos [-0.058, +0.303].
- Patron bear real: 3/3 anos con >10% dias bear funcionan, 0/5 con <10%.
- 2020 domina (473/671 SOW en bear son de 2020).
- Ningun gate probado (VIX>60, bear) cumple criterio.

**Evidencia archivada:**
- `docs/auditoria/wyckoff/51_purged_kfold_2026-10-23/` — C0-M30, C1-M5 v1.
- `docs/auditoria/wyckoff/52_purged_kfold_c1_2026-10-24/` — C1-M5 result.
- `docs/auditoria/wyckoff/53_dictamen_2026-10-24.md` — dictamen SOW-V6 (pre-SOW).
- `docs/auditoria/wyckoff/55_reserved_r3000_2026-10-24/` — R3000 result.
- `docs/auditoria/wyckoff/56_event_study_2026-10-24/` — event study v1 vs v2.

**Cambios en codigo:**
- `indicators/wyckoff_v1.py`: anadida `detect_sow_v2` (gate distribution). `detect_sow` intacto.
- `scripts/calibrate_wyckoff_sow_5b4bis.py`: anadida `compute_sow_cache_v2`.
- `config/settings.py`: `WYCKOFF_SOW_* = None` sin cambios.
- Tests: 3858 passed, 5 skipped (era 3849 antes). 9 tests nuevos (detect_sow_v2 + H7).

---

## 3. FRENTES CERRADOS

- **A1-A5, A6, B, C, 6, 7, 8, 9, F, H1-H18**: cerrados.
- **5e legacy Wyckoff**: cerrado 2026-10-10.
- **SOW universal (SOW-V6)**: cerrado 2026-10-24 como WONT_FIX.
  Razon: senal real vs placebo pero no universal. 4/9 folds positivos,
  IC95 entre anos cruza 0. Ningun gate probado cumple criterio pre-registrado.

---

## 4. PENDIENTES VIVOS

### 4.1. SOW-BEAR-CONFIRMATION (IN_PROGRESS)

- Requiere contrato v1.10 firmado por auditor externo antes de implementar.
- Semantica: etiqueta de confirmacion dentro de regimen bear ya identificado.
- No activa DISTRIBUTION. No modifica `classify_wyckoff_phase`.
- Siguiente accion: redactar `docs/auditoria/wyckoff/57_contrato_v1_10.md`.
- Sin codigo en produccion hasta firma.

### 4.2. Items OPEN observacionales (con fecha)

| Item | Fecha | Accion |
|---|---|---|
| H5.3 | 2026-11-20 06:17 UTC | Verificar cron |
| REP-AUDIT-20261020-RECEIPT-AUDIT-LAG | 2026-11-17 | Cierre con 4 semanas obs |
| REP-AUDIT-20261020-CRON-SLOTS-OBS | 2026-11-10 | Revision runs 20-10 a 10-11 |
| REP-AUDIT-20261020-SCHEDULER-DELAYS | 2026-11-10 | p95 delays |

### 4.3. BLOCKED_EXTERNAL

- `C2-920`: discrepancia 920 en baseline IAE. Auditor externo.
- `WYCKOFF-5BX`: esperando `n_confirmed_H20_complete >= 550` (~2027).
  Sujeto a re-evaluacion si SOW-BEAR-CONFIRMATION demuestra que C0 no
  aporta en bear.

### 4.4. Deudas estructurales

- HANDOFF se edita a mano (no hay generador automatico).
- Modulos no integrados: `target_universe`, `timestamps`, `catalog_pit`,
  `absence`, `validate_membership`, `cusip_resolver.resolve_cusip`.
- `reporting_dedup` (P65/P66): capa de diagnostico por diseno.
- 5 tests skipped: diseno, no deuda.

---

## 5. COMO ARRANCAR

    Set-Location D:\Macro_Sectorial
    git log --oneline -5
    git status -sb
    py scripts\generate_estado_sistema.py
    py scripts\verifica_gobernanza.py
    py scripts\genera_indice_auditorias.py
    py scripts\poda_bitacora.py --check
    py -m pytest tests\ -q --tb=line -m "not integration_real_data"
    py -m pyflakes . ; py -m compileall . -q

Esperado:
- HEAD = el que devuelva `git log --oneline -1`.
- ahead 0, behind 0 (o behind N si el bot ha pusheado; `git pull --ff-only`).
- Working tree limpio.
- Tests: ver `ESTADO_SISTEMA.md`. Referencia con filtro: 3849 passed + 5
  skipped + 9 deselected. Snapshot sin filtro: 3858 (publicado en
  ESTADO_SISTEMA.md).
- Gobernanza: `[OK] Gobernanza coherente`.
- Indice H1-H18: `[OK] indice actualizado`.
- Bitacora: `[OK] Bitacora dentro de la regla`.
- pyflakes silencio, compileall OK.

---

## 6. NOTAS CRITICAS

### 6.1. SOW universal cerrado (2026-10-24)

No reabrir SOW-V6 como "SOW universal". Los datos no lo soportan:

Senal real vs placebo en cohorte reservada R3000 (3423 tickers):
lift +0.123 vs placebo -0.077 (21.5 sd), pero 4/9 folds positivos.

IC95 entre anos: [-0.058, +0.303]. Cruza cero.

3 gates probados (distribution candidate, VIX>60, market bear):
ninguno cumple criterio pre-registrado (CI95_lower > 0, 7/9 folds).

Reabrir solo con nuevo contrato v1.10 (bear-confirmation) o nueva
variable Y o nuevo dataset.

### 6.2. SOW-BEAR-CONFIRMATION no es shadow mode

El usuario veta explicitamente shadow mode como "aislado en paralelo
sin tocar produccion". La semantica de bear-confirmation es distinta:
es una etiqueta de confirmacion que se integra en el reporte, no una
senal autonomia que se evalua en paralelo. Requiere contrato v1.10
y auditor externo antes de implementar.

### 6.3. Regla de los placebos con distribucion

Si en una proxima sesion se testea un detector:

Placebo A: permutacion intra-ticker (B=100).

Placebo B: date-shift global (B=100). No un solo shift.

Criterio: lift real > mean(placebo) + 2*sd(placebo).

Sanity check: pb != lift (raise RuntimeError).

Un solo shift (como en el primer experimento) no es placebo, es ruido.

### 6.4. bootstrap entre anos, no por fecha

block_bootstrap_by_date sesga el IC hacia arriba porque subrepresenta
anos con lift bajo. El IC oficial para "senal estable entre regimenes"
debe ser bootstrap_by_year (9 unidades) o t-interval entre anos.
Ambos producen ICs mas anchos que el bootstrap por ticker.

### 6.5. Anclas temporales obligatorias

Toda fecha literal en src/ o scripts/ (salvo calibrate_*,
walk_forward_*, validate_wyckoff_*, download_extended_history,
power_analysis_5bX, iae_*) debe estar en ANCLAS.yaml.
El test test_anclas_temporales falla si no. Anclas nuevas 2026-10-24:
RESERVED_COHORT_START, RESERVED_COHORT_END.

### 6.6. H7 patron

test_audit_contracts_rolling_nan inspecciona texto, no ejecucion.
rolling(N, min_periods=N) debe ir precedido de .dropna() en la misma
linea o falla aunque la serie ya venga de dropna. Patron documentado
en cabecera de wyckoff_v1.py.

### 6.7. Scripts de descarga reproducibles

scripts/fetch_reserved_cohort.py (S&P 500 -> 268 tickers) y
scripts/fetch_reserved_cohort_r3000.py (US-Stock-Symbols -> 3423).
Los parquets resultantes estan gitignored. Scripts versionados para
reproducibilidad.

---

## 7. REFERENCIAS

Consolidacion_Documentos/00_ARRANQUE.md — contexto de arranque.

Consolidacion_Documentos/01_METODO.md — metodo de patch.

Consolidacion_Documentos/01b_PATRONES.md — patrones de bug.

Consolidacion_Documentos/02_ARQUITECTURA.md — mapa del sistema.

Consolidacion_Documentos/03_IAE.md — subsistema IAE.

Consolidacion_Documentos/04_HISTORICO.md — cronologia tematica.

Consolidacion_Documentos/05_BITACORA.md — sesiones recientes.

Consolidacion_Documentos/ESTADO_SISTEMA.md — snapshot auto-generado.

docs/gobernanza/REGISTRO.yaml — items canonicos.

docs/gobernanza/ANCLAS.yaml — anclas temporales.

docs/auditoria/wyckoff/ — 56+ ficheros del expediente.

Fin del HANDOFF.