# HANDOFF - Radar de Rotacion Sectorial

Documento de traspaso. Autocontenido. Pegar al inicio del nuevo chat
junto con `Consolidacion_Documentos/00_ARRANQUE.md`.

**Este fichero se sobrescribe cada sesion.** Historico en git.

---

## 1. ESTADO AL CIERRE (2026-10-18 noche)

Rama:           main
HEAD:           consultar `git log --oneline -1` (no literal; el corpus no fija SHA para evitar deriva)
Sincronizacion: main == origin/main tras push final
Tests local:    ver `Consolidacion_Documentos/ESTADO_SISTEMA.md`. Al cierre: 3761 passed + 5 skipped con -m "not integration_real_data". Sin filtro (raw ESTADO_SISTEMA): ver seccion 2 del snapshot.
Validation Gate: 10/10
Working tree:   limpio (salvo data/ y outputs/ que el bot actualiza)
Gobernanza:     docs/gobernanza/REGISTRO.yaml. 80 items.
CI de tests:    .github/workflows/ci_tests.yml (push a main, paths-ignore: data/, outputs/).
                Ultimo verde en sesion: 39154201318 (head 212011f4).
Certificacion prod: PENDIENTE. Lunes 2026-10-19 23:17 UTC primer scheduled
                con target_session=2026-10-19. Ciclo completo v1.1+v1.2
                desde cero: gate READY sin receipt -> pipeline -> guard ->
                receipt con contract_validation_status -> attempt.
                Ademas: primer run con post_receipt_audit activo y con
                data_quality.csv escribiendo date=sesion bursatil.

---

## 2. SESIONES 2026-10-18

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

---

## 4. PENDIENTES VIVOS

### 4.1. Certificacion scheduled lunes 2026-10-19 23:17 UTC

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
- Tests: ver `ESTADO_SISTEMA.md`. Referencia: 3761 passed + 5 skipped
  con filtro. Sin filtro: ver snapshot.
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