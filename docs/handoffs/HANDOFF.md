# HANDOFF - Radar de Rotacion Sectorial

Documento de traspaso. Autocontenido. Pegar al inicio del nuevo chat
junto con `Consolidacion_Documentos/00_ARRANQUE.md`.

**Este fichero se sobrescribe cada sesion.** Historico en git.

---

## 1. ESTADO AL CIERRE (2026-10-20)

Rama:           main
HEAD:           consultar `git log --oneline -1` (no literal; el corpus no fija SHA para evitar deriva)
Sincronizacion: main == origin/main tras push final
Tests local:    ver `Consolidacion_Documentos/ESTADO_SISTEMA.md`. Al cierre: 3774 passed + 5 skipped con -m "not integration_real_data".
Validation Gate: 10/10
Working tree:   limpio (salvo data/ y outputs/ que el bot actualiza)
Gobernanza:     docs/gobernanza/REGISTRO.yaml. 86 items (6 nuevos del 2026-10-20).
CI de tests:    .github/workflows/ci_tests.yml (push a main, paths-ignore: data/, outputs/).
Certificacion prod: PRIMER CICLO COMPLETADO. Target 2026-10-19 cubierto por
                run 39355608824 (20-10 01:50 UTC, delay 2h33m vs slot 23:17
                del lunes, GitHub scheduler). run-system ejecuto pipeline
                completo: gate+run-system+issue-manager success, 5 artifacts
                (receipt, attempt, daily-report, european-coverage, pyflakes).
                Primer receipt_audit poblado: 18 OK + 1 NO_DATE_CHECK, 0 MISSING.
                Primer data_quality con date=sesion (2026-10-19, sin findes).

---

## 2. SESIONES 2026-10-18 a 2026-10-20

### 2.1. Sesion manana (7 commits)

| SHA | Contenido |
|---|---|
| `9d8d8cf9` | H16: contadores 0/5 explicitos en Cross-Module |
| `3d8c9689` | H7: ETFs fuera del denominador de cobertura breadth |
| `37e1f02e` | H5: CFTC pos_change = net_position.diff (identidad) |
| `e7b12feb` | H10/H19: "Data Conf" -> "Conf. Frescura" |
| `caa176d1` | H11/H12: N-PORT fecha_observacion + fecha_presentacion |
| `a226f556` | Struct: columnas en Rankings Sectoriales e Indices |
| `22392813` | Limpieza pyflakes |

### 2.2. Sesion tarde (13 commits)

Fase A - Deudas generador GEN-ESTADO-*:

| SHA | Contenido |
|---|---|
| `2ca11b5a` | Regenerar ESTADO_SISTEMA (3641 -> 3739) |
| `80ae528b` | Abrir GEN-ESTADO-* + nota doble conteo HANDOFF |
| `348360f2` | Entrada bitacora 2026-10-18 (rebasada sobre bot FRED) |
| `7acb327a` | GEN-ESTADO-* OPEN -> IN_PROGRESS |
| `72659a9f` | GEN-ESTADO-VALIDATION-VACIO: quitar validation/ de pytest |
| `cb3253d3` | GEN-ESTADO-TESTS-NOT-FAILCLOSED: propagar rc a main |
| `8dd86e54` | Cierre GEN-ESTADO-* (con placeholders, reparado despues) |
| `3504480f` | Fix import scripts/ + SHAs reales + verificacion FIXED |

Fase B - Auditoria externa del reporte (punto 1):

| SHA | Contenido |
|---|---|
| `8eaecb0a` | IPS/IPI como proxy de mercado, no IPC/PPI |

Fase C - Deuda validation/ en workflows:

| SHA | Contenido |
|---|---|
| `cafc77d8` | Quitar validation/ del pytest en ci_tests y daily_run |

Fase D - Investigacion frescura + deudas del sistema:

| SHA | Contenido |
|---|---|
| `4ade7384` | data_quality: date = sesion bursatil, no fecha del run |
| `5ef44486` | Migrar data_quality.csv (155 filas, 9 fechas, 593->464) |
| `400e6c9b` | feat(observability): auditoria post-receipt vs filas persistidas |
| `212011f4` | Quitar imports unused (pyflakes) |

### 2.4. Sesion 2026-10-20 (8 commits: contaminacion de fechas)

**Contexto.** Revisión del scheduled del lunes 19-10 (target 2026-10-19).
Health check semanal falló con FAIL `cron_slots` (2/5 slots en 24h) aunque
el pipeline estaba operativo. Además, auditoría de contaminación de fechas
en `outputs/history/*.csv` detectó 4 CSVs con filas en sábados/domingos o
con `date` futura.

| SHA | Contenido |
|---|---|
| `f3efa062` | cron_slots FAIL solo si 0/5 slots en 24h (regla 3+ -> 1-4 WARN) |
| `6d2278c3` | european_coverage `_ref_to_date` = last_expected_market_date() |
| `da093a05` | migrar european_coverage.csv: 1734 -> 1326 filas, 0 no bursátiles |
| `f02cebd8` | data_quality: age europeas contra sesión (no contra run) |
| `14838307` | monitor_double_counting: date=sesion + run_timestamp |
| `5f464729` | migrar double_counting_monitor.csv: 225 -> 123 filas |
| `5ebe1c62` | eliminar slpm_history.csv fósil (sin writer activo desde 2026-07-17) |
| `c98c55f4` | 6 items REP-AUDIT-20261020-* en REGISTRO |

**Hallazgos y falsos positivos.** La auditoría de fechas marcó 11 CSVs como
`NO_BURSATIL`. 4 son bug real (european_coverage, double_counting_monitor,
data_quality rama europea, slpm_history). 7 son falsos positivos: CSVs de
fondos con calendario no-NYSE (amundi_lyxi, blackrock_dax/isf, invesco_qqq_nav,
etf_primary_flow con cierres NYSE legítimos, darkpool_history week=FINRA).
Documentados como `REP-AUDIT-20261020-FALSOS-POSITIVOS-CALENDARIO`.

**Lecciones.**
- El fix del 18-10 (`4ade7384`) solo cubrió la rama general de
  `compute_data_quality`. La rama europea usa `_last.date()` (con guion
  bajo), no matcheada por el replace. Bug descubierto en auditoría del 20-10.
- El contrato del 00_ARRANQUE "ningún writer publica date no bursátil"
  aplica a writers del pipeline NYSE. Los CSVs de proveedores con
  calendario propio (fondos europeos, NAVs, FINRA week) están fuera.
  Excepción documentada en 00_ARRANQUE §2 (pendiente de commit).
- GitHub scheduler con delays 2-7h. La regla `cron_slots` anterior
  (FAIL con 3+ missing) marcaba como caído un pipeline operativo. Ahora
  FAIL solo si 0/5. Issue #6 cerrado.

### 2.3. Cambios observables en el reporte

- INFO Cross-Module con ambos bloques siempre (Financial Stress: N/5 +
  Inflation Pressure: M/5, y Expansion si hay).
- Columna "Conf. Frescura" (no "Data Conf").
- Bloque volatilidad: "VOLATILIDAD REALIZADA del VIX".
- Alerta credito: "neutral" si |HYG/LQD Z| < 0.75 (antes 0.5).
- Bloque FLS: si FLS >= 70, aclara que es estres de corto plazo.
- Divergencia precio-flujo: sufijo " (debil)" si |ret| < 1%.
- A/D Line: declara universo (N/N tickers).
- CFTC Pos Change: identidad net.diff().
- N-PORT: Fecha observacion + Fecha presentacion (lag ~59d).
- Rankings Sectoriales + Indices: columna Struct con magnitud.
- Bloque MTE: aclara que IPS/IPI son proxy de mercado (no IPC/PPI).
- CSV macro_regime: financial_regime + real_liq_regime, sin alias legacy.
- data_quality.csv: date = sesion bursatil (no fecha del run).

### 2.5. Sesion 2026-10-20 (tarde) - 14 commits: dos auditorias externas + FLS

**Contexto.** Dos auditores externos enviaron revision del reporte 2026-10-19.
El primero: 14 items (4 criticas + 10 sospechosas). El segundo: 7 items
enfocados en Flow Proxy, trazabilidad temporal y sintesis credito.

**Cierres (14 commits).**

Auditoria #1:
- `0aec4ce5` (B1/B3/B4): sintesis sin atribuir HIGH_STRESS al credito,
  nota Flow Proxy vs Flujo Primario, columna `Struct Score`.
- `41b1c8bf` (B2): veto por variables materiales en `evidence_matrix`.
  XLC (100% bearish) ya no sale "FAVORABLE" con price/breadth negativos.
- `47071a51` (A2): `health_check` no da WARN en trimestrales con dispatch OK.
- `7f329d07`: 10 items `REP-AUDIT-20261020-*` + HANDOFF + generador
  `(sin filtro; suite completa)`.

Auditoria #2:
- `eaed52a7` (P1): nota Flow Proxy tabla (composite ETF) vs matriz
  (mediana componentes). XLE -0.50 en tabla, +1 en matriz, explicado.
- `68e4e40c` (P2): nota temporal provenance, recuento votos P/N/0 en
  matriz, nota regla Wyckoff.

Transversal:
- `9f6afa26` (A1): `post_receipt_audit` status `LAGGED` cuando
  `receipt.date != target_session`.
- `7475108f` (C8): nota FLS saturacion por nivel. Item
  `FLS-COMPONENT-DESIGN` OPEN.
- `33a12a10` (C10): ratio |Z| extremo en Cross-Asset.
- `878ad4fc` (C6): nota materialidad Europa (media simple, no AUM).
- `38f17674` (FLS): SOFR como spread SOFR-IORB, no nivel. FLS 79 -> 63.
  stressed 3/5 -> 2/5. Item queda OPEN por WALCL/CP.

**Falsos positivos documentados** (en `REVIEW-EXTERNAL`):
C1 (Wyckoff +1), C2 (Flow Proxy 30% WLS), C3 (SLPM NO_LEADERS),
C4 (datos desfasados), C5 (NIPC 13F), C7 (spread 10Y-3M), C9 (IPI).

**Pendiente.** Cerrar `REVIEW-EXTERNAL` tras run 2026-10-20.
`RECEIPT-AUDIT-LAG`
(observacion 4 semanas). `CRON-SLOTS-OBS` y `SCHEDULER-DELAYS`
(revision 2026-11-10).


---

## 3. FRENTES CERRADOS

- H4 report, H5, H6, H7, H10, H11, H12, H14, H15, H16, H17, H19.
- Capas 1-6 del plan de sostenibilidad.
- Struct publication (opcion B aprobada por usuario 2026-10-18).
- Issue #5 health-check cerrado.
- Dictamen externo Gate 0.5 aprobado con correcciones aplicadas.
- GEN-ESTADO-VALIDATION-VACIO (`72659a9f`).
- GEN-ESTADO-TESTS-NOT-FAILCLOSED (`cb3253d3`).
- Auditoria externa 2026-10-18 punto 1 IPS/IPI proxy (`8eaecb0a`).
- validation/ fuera de pytest en workflows (`cafc77d8`).
- data_quality.csv date = sesion bursatil (`4ade7384` + `5ef44486`).
- Receipt audit no bloqueante (`400e6c9b` + `212011f4`).
- REP-AUDIT-20261020-WARN-TRIMESTRALES: FIXED (`47071a51`).
- REP-AUDIT-20261020-ETF-FLOW-LAG: FALSE_POSITIVE (lag SSGA D+1/D+2).
- REP-AUDIT-20261020-REPORT-STRESS-LABEL: FIXED (`0aec4ce5`).
- REP-AUDIT-20261020-REPORT-FLOW-RECONCILIATION: FIXED (`0aec4ce5`).
- REP-AUDIT-20261020-REPORT-STRUCT-COLUMN: FIXED (`0aec4ce5`).
- REP-AUDIT-20261020-REPORT-XLC-EVIDENCE: FIXED (`41b1c8bf`).
- REP-AUDIT-20261020-REPORT-REVIEW-EXTERNAL: 13 items atendidos (auditor externo #1). P1, P2 FIXED en `eaed52a7`, `68e4e40c`. C1-C10 verificados (varios FALSE_POSITIVE).
- REP-AUDIT-20261020-FLS-COMPONENT-DESIGN: FIXED (`5a1e0c37`). SOFR FIXED
(`38f17674`), WALCL/CP WONT_FIX (saturacion estructural; diagnostico 22-10).
- REP-AUDIT-20261020-REPORT-REVIEW-EXTERNAL: FIXED. Run 39528211520 verde, validation_gate 10/10, FLS 63/100, primer LAGGED real.

---

## 4. PENDIENTES VIVOS

- **REP-AUDIT-20261020-RECEIPT-AUDIT-LAG** (OPEN): status LAGGED en post_receipt_audit; observacion 4 semanas.
- **REP-AUDIT-20261020-CRON-SLOTS-OBS** (OPEN): revision 2026-11-10.
- **REP-AUDIT-20261020-SCHEDULER-DELAYS** (OPEN): revision 2026-11-10.

### 4.0. Verificacion scheduled target 2026-10-22

Cuando dispare el bot (slot 23:17 UTC + delays), ejecutar
`verify_run_20261022.ps1` (gitignored). Criterios:
- `Liquidez Real` = `NEUTRAL` (no `N/A`). Fix router liquidez.
- `cboe_vix3m.parquet` sin no-bursatiles. Fix parquet.
- `writer_receipts.json` con receipt `pcr_history`. Fix side effect.
- `sector_correlation_summary` con valores coherentes. Fix P7.

Es verificacion de entorno (no de codigo). Los fixes ya estan
validados en `py run.py` local (suite 3845 passed, 5 skipped).

### 4.1. Certificacion scheduled lunes 2026-10-19 23:17 UTC [COMPLETED 2026-10-20]

**Estado: COMPLETADA.** Target 2026-10-19 cubierto por run `39355608824`
(20-10 01:50 UTC, delay 2h33m vs slot 23:17 del lunes, GitHub scheduler).
`run-system` ejecuto pipeline completo. Receipt emitido. `receipt_audit.csv`
poblado: 18 OK + 1 NO_DATE_CHECK, 0 MISSING. Detalle en §1 y §2.4.

Primer run scheduled con `target_session=2026-10-19`. Ciclo completo
v1.1+v1.2 desde cero. Ademas: primer run con post_receipt_audit activo
y con data_quality.csv escribiendo date=sesion bursatil.

Si a las 23:35 UTC no ha disparado: `gh workflow run daily_run.yml`
(sin inputs; target_session se calcula internamente). Los 5 slots
cubren la ventana.

Observar en el run:
- `receipt_audit.csv` en el artifact de outputs/history.
- Status de cada writer en el audit (esperado: todo OK salvo
  date=None en sector_rank_deltas).
- Health check del lunes 26-oct con `check_receipt_audit` WARN si
  hay anomalias.

### 4.2. Otros frentes

- **H5.3** (OPEN): cron 20-nov-2026 06:17 UTC. Sin accion antes.
- **C2 (920)**: bloqueado por auditor externo.
- **Wyckoff 5b.X v3**: bloqueado por datos (~2027).
- **SOW v6**: pendiente firma externa.
- **Ubuntu 26**: workflows pinnen `ubuntu-24.04`.

---

### 4.3. Cierre sesion 2026-10-22

11 bugs reales cerrados con test + verificacion en produccion local:
router liquidez, state v1/v2, docstring stale, ref rota, verifier
reachability, effr.csv, -0.0, pcr side effects, sector_correlation
P7, cboe_vix3m no-bursatiles, Wyckoff B5/B3. Suite: 3845 passed,
5 skipped. HEAD al cierre: `1bb56473`. Detalle en `05_BITACORA.md`
entrada 2026-10-22 y `04_HISTORICO.md` "Sesion 2026-10-22".

## 5. COMO ARRANCAR

    Set-Location D:\Macro_Sectorial
    git log --oneline -5
    git status -sb
    py scripts\generate_estado_sistema.py
    py scripts\verifica_gobernanza.py
    py scripts\genera_indice_auditorias.py
    py scripts\poda_bitacora.py --check
    py -m pytest tests/ -q --tb=line -m "not integration_real_data"
    py -m pyflakes . ; py -m compileall . -q

Esperado:
- HEAD = el que devuelva `git log --oneline -1`.
- ahead 0, behind 0 (o behind N si el bot ha pusheado; `git pull --ff-only`).
- working tree limpio.
- Tests: ver `ESTADO_SISTEMA.md`. Referencia con filtro: 3774 passed + 5
  skipped + 9 deselected. Snapshot sin filtro: 3783 (publicado en
  ESTADO_SISTEMA.md).
- Gobernanza: `[OK] Gobernanza coherente`.
- Indice H1-H18: `[OK] indice actualizado`.
- Bitacora: `[OK] Bitacora dentro de la regla`.
- pyflakes silencio, compileall OK.

---

## 6. NOTAS CRITICAS

- **Verificar en CI, no solo en local.** Tests con `tmp_path` y fixtures
  sinteticos. No leer `data/` ni `outputs/`.
- **Un cambio = una verificacion = un commit.**
- **Verificar EOL (LF/CRLF) y BOM antes de cada patch.** `report_generator.py`
  tiene BOM. `utils.py` y `finalize.py` son CRLF. Resto LF. HANDOFF,
  REGISTRO, BITACORA, 00_ARRANQUE, 02_ARQUITECTURA: LF puro.
- **Here-strings >20 lineas o >5 `$` -> fichero temporal**
  (`[IO.File]::WriteAllText` + `py`).
- **Receipt fail-closed (enmienda v1.1).** BLOCKED o UNAVAILABLE ->
  no se emite receipt.
- **Contrato EQUITY_EOD `critical=True`.** `coverage_at_session < 0.90`
  bloquea publicacion.
- **Schedule vs Dispatch.** Un dispatch reciente NUNCA oculta ausencia
  de schedule.
- **Gobernanza canonica.** Items a `docs/gobernanza/REGISTRO.yaml`.
  El HANDOFF es vista derivada.
- **Anclas temporales.** Fechas literales a `ANCLAS.yaml`.
- **Wyckoff phase (MARKUP/RANGE) es tendencia de fondo, no retorno
  reciente.** struct > 0.30 (umbral) para MARKUP. Columna Struct
  publica la magnitud para evitar lectura binaria.
- **date = sesion bursatil.** `data_quality.csv` escribe `date` =
  `last_expected_market_date(reference_date)`. Evita filas en sabados,
  domingos y festivos NYSE. `age_calendar_days` se mide contra la
  sesion, no contra el run. Verificado 2026-10-18.
- **Receipt audit no bloqueante.** `scripts/post_receipt_audit.py`
  compara `writer_receipts.json` con los CSV persistidos. Status
  `MISSING` = receipt dijo escrito pero CSV no tiene la fila (bug
  09-oct). No bloquea receipts. `health_check` lo lee los lunes y
  WARN si hay anomalias. Se observa 4-8 semanas antes de decidir si
  bloquear.
- **writer_receipts.json gitignored.** Output de estado del pipeline.
  El step de audit lo consume antes del commit del bot.

---

- **Contrato de fecha: pipeline NYSE vs proveedores con calendario propio.**
  "Ningun writer publica `date` no bursatil" (00_ARRANQUE §2) aplica a
  writers del pipeline NYSE. Los CSVs de proveedores con calendario no-NYSE
  estan fuera: fondos europeos (`amundi_lyxi_primary_flow`,
  `blackrock_dax_primary_flow`, `blackrock_isf_primary_flow`), NAVs de fondo
  (`invesco_qqq_nav_*` con fines de semana), `etf_primary_flow` (cierres
  NYSE legitimos: Good Friday, fin de ano), `darkpool_history` (week=FINRA
  lunes con walk-back). Documentado en
  REP-AUDIT-20261020-FALSOS-POSITIVOS-CALENDARIO.

## 7. REFERENCIAS

Corpus normativo: `Consolidacion_Documentos/` (12 docs).
Gobernanza: `docs/gobernanza/` (README, REGISTRO.yaml, ANCLAS.yaml).
Contrato del gate: `docs/auditoria/daily_run_gate_contrato_v1.md` +
`docs/auditoria/daily_run_gate_enmienda_v1_1.md` +
`docs/auditoria/daily_run_gate_enmienda_v1_2.md`.
Indice auditoria externa: `docs/auditoria/radar/audits/INDEX_H1_H18.md`.
Expedientes: `docs/auditoria/wyckoff/`, `docs/auditoria/iae/golden/`,
`docs/auditoria/s03_deep/`, `docs/auditoria/regimenes/`.

---

**Fin del handoff.**
