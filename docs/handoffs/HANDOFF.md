# HANDOFF - Radar de Rotacion Sectorial

Documento de traspaso. Autocontenido. Pegar al inicio del nuevo chat
junto con `Consolidacion_Documentos/00_ARRANQUE.md`.

**Este fichero se sobrescribe cada sesion.** Historico en git.

---

## 1. ESTADO AL CIERRE (2026-10-24, sesion 2)

Rama:           main
HEAD:           consultar `git log --oneline -1` (no literal; el corpus no fija SHA)
Sincronizacion: main == origin/main
Tests local:    ver `Consolidacion_Documentos/ESTADO_SISTEMA.md`.
                Al cierre: 3946 passed, 5 skipped (suite con -n auto).
Validation Gate: 10/10
Working tree:   limpio
Gobernanza:     docs/gobernanza/REGISTRO.yaml
CI de tests:    .github/workflows/ci_tests.yml (push a main, paths-ignore: data/, outputs/)

**Dos bloques de trabajo en esta sesion:**

**Bloque A (cierre SOW universal + apertura bear-confirmation):**
Cierre `SOW-V6` -> WONT_FIX tras 7 experimentos sin cumplir criterio
pre-registrado. Apertura `SOW-BEAR-CONFIRMATION` (IN_PROGRESS). Ver
HANDOFF §2.1 y docs/auditoria/wyckoff/51-56.

**Bloque B (reconstruccion completa del sistema Wyckoff):**
Decision arquitectonica: dejar de usar ETFs sectoriales e indices como
fuente de fases Wyckoff. Construir cascadas por stock con deteccion de
acumulacion Y distribucion, orquestadas, cableadas al pipeline.
Resultado: 15 detectores en codigo, orquestador con fase actual
derivada, cableado completo a sector_wyckoff_distribution.py + reporte.
Ver HANDOFF §2.2 y docs/auditoria/wyckoff/60-62.

---

## 2. SESIONES 2026-10-24

### 2.1. Bloque A — Cierre SOW universal + apertura bear-confirmation

**Commits clave (antes del HEAD actual del bloque B):**
- `e7327463` — purged_kfold_sow.py con criterio pre-registrado
- `520a8db7` — SOW v2 lift within-ticker
- `2dcc3325` — C1-M5 cohorte original
- `548d2773` — C1-M5: 2 folds significativos (2018, 2020)
- `be2f7670` — cohorte reservada S&P 500 (268 tickers)
- `c5485337` — placebo date-shift global
- `61556a74`/`0c2455ab` — cohorte R3000 + vectorizacion
- `3d1a215d` — resultado R3000
- `b542d381` — detect_sow_v2 (gate distribution)
- `260569bc`/`281cb378` — event study v1 vs v2
- `81962b59` — SOW-V6 WONT_FIX + SOW-BEAR-CONFIRMATION IN_PROGRESS
- `ff8f5a6f` — H7 en detect_sow_v2 + anclas

**Descubrimientos:**
- SOW universal (C1-M5, M=5): senal real vs placebo en R3000 (3423
  tickers, lift +0.123 vs -0.077, 21.5 sd) pero NO universal (4/9
  folds positivos, IC95 entre anos [-0.058, +0.303] cruza 0).
- Patron bear: 3/3 anos con >10% dias bear funcionan, 0/5 con <10%.
  2020 domina la muestra (473/671 SOW en bear).
- Ningun gate probado (VIX>60, market bear) cumple criterio
  pre-registrado.
- Pre-registro v1 (regla de vigencia unidireccional) fallo en holdout
  S&P 500 (C1=18.6% < 20%). No se ajusto. Documentado en 60.

### 2.2. Bloque B — Reconstruccion completa del sistema Wyckoff

**Commits clave (rama main, todos pusheados):**
- `156738cd` — refactor: wyckoff_range -> wyckoff_distribution +
  extraer dataclasses a wyckoff_event
- `6a1eff10` — wyckoff_accumulation: SC->AR->Range->SOS->ACCUMULATION
- `c45ac6ab` — orquestador de fases con CONFLICT explicito
- `3c0b13f3` — arbitraje por resolution_event_date + range_active_at
- `37398f54` — regla simetrica de vigencia + MARKUP/MARKDOWN
- `1d502cac` — ST (Secondary Test) en ambas cascadas
- `a8bd235a` — detect_false_breakout_up/down (UT/UTAD y Spring)
- `17ecbe1d` — descartar UTAD_PENDING/SPRING_PENDING tras fallo
  pre-registro 61
- `4edba814` — PSY/PS/mSOW/mSOS/LPSY/LPS como evidencia pasiva
- `415487ff` — cablear pipeline a classify_phase + FORMING
- `bd81abe4` — integrar mSOW/mSOS/LPSY/LPS como evidencia pasiva
- `bfe491bf` — ST integrado como evidencia pasiva
- `40ac2d07` — anotar titulos sectoriales con composicion interna
- `64432648` — limpieza scripts diagnostico temporales

**Arquitectura nueva:**

Modulos en `indicators/`:
- `wyckoff_event.py` — dataclasses comunes: WyckoffEvent, Range,
  ClassifiedBreak, EventStatus, RangeState, RangeType, Classification
- `wyckoff_distribution.py` — cascada distribucion:
  BC -> AR -> Range -> SOW + detectores auxiliares (ST_dist, PSY,
  mSOW, LPSY, false_breakout_up)
- `wyckoff_accumulation.py` — cascada acumulacion simetrica:
  SC -> AR -> Range -> SOS + auxiliares (ST_acc, PS, mSOS, LPS,
  false_breakout_down)
- `wyckoff_orchestrator.py` — orquestador: corre ambas cascadas,
  arbitra con regla simetrica de vigencia, emite phase + event_phase

**Estados finales del orquestador:**
- `phase` (fase ACTUAL, deriva de event_phase + posicion del precio):
  - `DISTRIBUTION` — dist confirmada, precio dentro del rango
  - `MARKDOWN` — dist confirmada, precio bajo support-tol (consumada)
  - `ACCUMULATION` — acc confirmada, precio dentro
  - `MARKUP` — acc confirmada, precio sobre resistance+tol (consumada)
  - `DISTRIBUTION_FORMING` — rango activo con contexto dist, sin SOW
  - `ACCUMULATION_FORMING` — rango activo con contexto acc, sin SOS
  - `RANGE` — dist fallida / acc fallida / sin cascada activa
  - `CONFLICT` — dos estructuras compitiendo
  - `INSUFFICIENT_DATA` — falta de datos real
- `event_phase` (fase de EVENTO, historica, sin vigencia por precio)
- `phase_since` — fecha de resolucion del episodio ganador
- `current_phase_basis` — justificacion textual

**Cableado al pipeline:**
- `sector_wyckoff_distribution.py` migrado de `classify_wyckoff_phase`
  (legacy) a `classify_phase` (nuevo orquestador).
- `VALID_PHASES` ampliado con FORMING.
- `sector_context.py` renderiza columnas: Sector | AccF | Acc | Markup
  | Range | DistF | Dist | Markdown | Conflict | N | Cob Wyckoff.
- `run.py` postprocesa `leader_lines` con
  `annotate_sector_titles_with_composition`: cada titulo
  `## Sector: XLK (MARKUP)` se anota con la composicion interna de
  holdings: `*Composicion interna (22 componentes): 3 Markup (14%),
  8 Range (36%), 9 DistF (41%), 1 Dist (5%), 1 Markdown (5%)*`.

**Estado en produccion (ultimo bar 2026-10-22, 327 stocks):**
- 219 RANGE, 53 DISTRIBUTION_FORMING, 17 DISTRIBUTION,
  17 MARKDOWN, 13 MARKUP, 7 ACCUMULATION_FORMING, 1 CONFLICT.
- Agregado holdings: 157 RANGE, 35 DISTRIBUTION_FORMING,
  12 DISTRIBUTION, 12 MARKDOWN, 8 MARKUP, 6 ACCUMULATION_FORMING.
- XLK: 41% DistF (presion interna fuerte aunque el ETF diga MARKUP).
- XLC: 40% DistF.

**Pre-registros fallidos (no ocultos):**
- `60_current_phase_rule_v1.md`: regla de vigencia unidireccional.
  Fallo C1 (18.6% < 20%) en holdout. No se ajusto. Se reemplazo por
  regla simetrica (definicion de rango, no umbral).
- `61_utad_spring_pending_v1.md`: UTAD_PENDING/SPRING_PENDING.
  Fallo A3 (26 vs 3, ratio 8.67x > 5x por asimetria natural del
  mercado). No se ajusto. Los detectores quedan en codigo, sin
  cambiar classification.

**Limitaciones documentadas:**
- PSY y PS no disparan con umbrales actuales (0/50 en smoke sobre
  stocks reales). Documentado como deuda de calibracion. NO se
  recalibro post-hoc (evita overfitting).
- `sector_regime.py` (rankings) sigue usando `classify_wyckoff_phase`
  legacy para el titulo del sector. La anotacion de composicion
  interna del leader_lines mitiga la incoherencia.
- `mSOW`/`mSOS`/`LPSY`/`LPS` estan en `evidence` de cada cascada
  pero no se renderizan en el reporte.

---

## 3. FRENTES CERRADOS

- H1-H19, A1-A6, B, C, F, E: cerrados.
- 5e legacy Wyckoff: cerrado 2026-10-10.
- `SOW-V6`: WONT_FIX 2026-10-24.
- `REP-AUDIT-20261020-REPORT-REVIEW-EXTERNAL`: FIXED.
- `UTAD_PENDING`/`SPRING_PENDING`: DESCARTADOS por pre-registro 61.
- Regla de vigencia unidireccional: DESCARTADA por pre-registro 60.

## 4. PENDIENTES VIVOS

### 4.1. SOW-BEAR-CONFIRMATION (IN_PROGRESS)

- Requiere contrato v1.10 firmado por auditor externo antes de implementar.
- Siguiente accion: redactar `docs/auditoria/wyckoff/57_contrato_v1_10.md`.
- Sin codigo en produccion hasta firma.

### 4.2. Deuda tecnica abierta (nueva, sesion 2026-10-24)

- **Migrar `sector_regime.py`** al nuevo orquestador (decision de producto:
  ¿fase del ETF legacy o fase dominante de holdings?).
- **Renderizar evidencia pasiva** (`msow`, `msos`, `lpsy`, `lps`,
  `st_dist`, `st_acc`) en el reporte. Actualmente estan en `evidence`
  pero no se muestran.
- **Calibrar PSY/PS** con pre-registro formal (actualmente 0/50 disparos).
- **Auditoria externa del sistema Wyckoff nuevo completo** (15
  detectores + orquestador + cableado).

### 4.3. Items OPEN observacionales (fechas)

| Item | Fecha | Accion |
|---|---|---|
| H5.3 | 2026-11-20 06:17 UTC | Verificar cron |
| REP-AUDIT-20261020-RECEIPT-AUDIT-LAG | 2026-11-17 | Cierre con 4 semanas obs |
| REP-AUDIT-20261020-CRON-SLOTS-OBS | 2026-11-10 | Revision runs 20-10 a 10-11 |
| REP-AUDIT-20261020-SCHEDULER-DELAYS | 2026-11-10 | p95 delays |

### 4.4. BLOCKED_EXTERNAL

- `C2-920`: discrepancia 920 en baseline IAE. Auditor externo.
- `WYCKOFF-5BX`: esperando `n_confirmed_H20_complete >= 550` (~2027).

---

## 5. COMO ARRANCAR

    Set-Location D:\Macro_Sectorial
    git log --oneline -5
    git status -sb
    py scripts\generate_estado_sistema.py
    py scripts\verifica_gobernanza.py
    py scripts\poda_bitacora.py --check
    py -m pytest tests\ -q --tb=line -m "not integration_real_data and not integration_slow"
    py -m pyflakes . ; py -m compileall . -q

Esperado:
- HEAD = el que devuelva `git log --oneline -1`.
- ahead 0, behind 0.
- Working tree limpio.
- Tests: ver `ESTADO_SISTEMA.md`. Referencia: 3946 passed + 5 skipped
  (con `-n auto`).
- Gobernanza: `[OK] Gobernanza coherente`.
- Bitacora: `[OK] Bitacora dentro de la regla`.
- pyflakes silencio, compileall OK.

---

## 6. NOTAS CRITICAS

- **Verificar en CI, no solo en local.** Tests con `tmp_path` y fixtures
  sinteticos.
- **Un cambio = una verificacion = un commit.**
- **Verificar EOL (LF/CRLF) y BOM antes de cada patch.**
- **Here-strings >20 lineas o >5 `$` -> fichero temporal.**
- **Receipt fail-closed (enmienda v1.1).** BLOCKED o UNAVAILABLE -> no
  se emite receipt.
- **Contrato EQUITY_EOD `critical=True`.** `coverage_at_session < 0.90`
  bloquea publicacion.
- **Gobernanza canonica.** Items a `docs/gobernanza/REGISTRO.yaml`.
  El HANDOFF es vista derivada.
- **Anclas temporales.** Fechas literales a `ANCLAS.yaml`.
- **Wyckoff sistema nuevo:** ver `docs/auditoria/wyckoff/62_sistema_completo_2026-10-24.md`.
  Cascadas de distribucion y acumulacion independientes, orquestador
  con regla simetrica de vigencia, FORMING para estados tempranos,
  MARKUP/MARKDOWN como estados derivados.
- **Regla simetrica de vigencia:** un rango esta vigente mientras el
  precio NO lo ha roto por NINGUN lado. Si rompio por arriba (dist
  fallida, RANGE) o por abajo (dist consumada, MARKDOWN). Definicion
  de rango, no umbral.
- **FORMING vs CONFIRMED:** FORMING = rango activo sin SOW/SOS
  confirmado. Es senal temprana, no confirmacion. El reporte lo
  distingue (columnas AccF/DistF separadas de Acc/Dist).
- **CONFLICT es estado honesto:** dos estructuras compitiendo, no se
  elige una sin mas evidencia.
- **Pre-registros fallidos NO se ajustan.** Ver 60 y 61. La leccion es
  que un pre-registro que falla documenta una hipotesis rechazada, no
  una regla a retocar.
- **`sector_regime.py` sigue usando `classify_wyckoff_phase` legacy**
  para el titulo del sector. La anotacion de composicion interna en
  `leader_lines` mitiga la incoherencia. Migrar es decision de
  producto pendiente.

## 7. REFERENCIAS

Corpus normativo: `Consolidacion_Documentos/` (12 docs).
Gobernanza: `docs/gobernanza/` (README, REGISTRO.yaml, ANCLAS.yaml).
Contrato del gate: `docs/auditoria/daily_run_gate_contrato_v1.md` +
`daily_run_gate_enmienda_v1_1.md` + `daily_run_gate_enmienda_v1_2.md`.
Expediente Wyckoff: `docs/auditoria/wyckoff/` (51-62 del 2026-10-24).

**Documentos clave para el traspaso:**
- `62_sistema_completo_2026-10-24.md` — mapa tecnico del sistema
  Wyckoff nuevo (modulos, detectores, orquestador, cableado).
- `60_current_phase_rule_v1.md` — pre-registro v1 fallido (regla
  vigencia unidireccional).
- `61_utad_spring_pending_v1.md` — pre-registro UTAD_PENDING fallido.

---

**Fin del handoff.**