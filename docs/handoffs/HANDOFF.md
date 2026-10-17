# HANDOFF - Radar de Rotacion Sectorial

Documento de traspaso. Autocontenido. Pegar al inicio del nuevo chat
junto con `Consolidacion_Documentos/00_ARRANQUE.md`.

**Este fichero se sobrescribe cada sesion.** Historico en git.

---

## 1. ESTADO AL CIERRE (2026-10-17 tarde)

Rama:           main
HEAD:           consultar `git log --oneline -1` (no literal; el corpus no fija SHA para evitar deriva)
Sincronizacion: main == origin/main tras push final
Tests local:    ver `Consolidacion_Documentos/ESTADO_SISTEMA.md`. Al cierre: 3702 passed + 5 skipped.
Validation Gate: 10/10
Working tree:   limpio (salvo data/ y outputs/ que el bot actualiza)
Gobernanza:     docs/gobernanza/REGISTRO.yaml. 67 items (9 nuevos hoy, todos FIXED).
CI de tests:    .github/workflows/ci_tests.yml (push a main, paths-ignore: data/, outputs/).
                Ultimos verdes en sesion: 39059526362 (head aa82a549) y posteriores.
Certificacion prod: PENDIENTE. Lunes 2026-10-19 23:17 UTC primer scheduled
                con target_session=2026-10-19. Ciclo completo v1.1+v1.2
                desde cero: gate READY sin receipt -> pipeline -> guard ->
                receipt con contract_validation_status -> attempt.

---

## 2. SESION 2026-10-17 (tarde)

### 2.1. Contexto

Sesion de sostenibilidad del reporte. Auditoria externa del reporte del
2026-10-17 (17:13 CEST) detecto textos categoricos sin soporte,
literales desincronizados de config, clasificacion de regimenes por
substring (accidental), y CSV con naming enganoso. Se aplicaron 10
capas de fixes con tests contractuales, todas con CI verde.

### 2.2. Commits de la sesion (12)

| SHA | Contenido |
|---|---|
| `09f32770` | H4 report: flow proxy N/D para tickers sin volumen (FX, DX-Y, SPGSCI) |
| `0d58c5ae` | Capa 1: literales ventana/umbral desde config/settings.py |
| `781da474` | Capa 2a: H14 (banda HYG/LQD 0.5->0.75) + H15 (texto FLS modular) |
| `9b3d5b33` | Capa 2b: H17 (umbral materialidad 1%) + mojibake Confirmacion |
| `c2d6b604` | Capa 3: H1 (Referencia/Generado) + H2 (local_run_id determinista) |
| `85b79cd9` | Capa 5: H6 A/D Line declara recalculo + universo |
| `ba4fca3a` | Capa 4a: clasificacion exacta de regimenes (no substring) |
| `a92f52e9` | Capa 4b: CSV financial_regime + real_liq_regime |
| `5d21b45d` | Capa 4c: publicar Expansion: N/5 en INFO Cross-Module |
| `f2dec91f` | Capa 6a: eliminar alias liquidity_regime + script migracion |
| `aa82a549` | Capa 6b: corregir texto de regimen de volatilidad |
| `a6adeacd` | Commit F: renombrar parametros liquidity_regime -> financial_regime |

Gobernanza: `86963d56` cierra los 9 items correspondientes en REGISTRO.yaml.

### 2.3. Cambios observables en el reporte

Respecto al reporte del 2026-10-17 manana, el proximo run publicara:
- INFO Cross-Module puede incluir "Expansion: N/5" si hay modulos en expansion.
- Bloque volatilidad: "VOLATILIDAD REALIZADA del VIX", no "VELOCIDAD (z-score del retorno)".
- Alerta credito: "neutral" si |HYG/LQD Z| entre 0.5 y 0.75.
- Bloque FLS: si FLS >= 70, aclara que es estres de corto plazo y liquidez real (FRED) es distinta dimension.
- Divergencia precio-flujo: sufijo " (debil)" si |ret| < 1%.
- A/D Line: declara universo (N/N tickers) y recalculo por run.
- CSV macro_regime: columnas financial_regime + real_liq_regime. Sin alias legacy liquidity_regime. 62 filas historicas migradas, real_liq_regime=NaN.

---

## 3. FRENTES CERRADOS EN LA SESION

- H4 report: flow proxy N/D (`09f32770`).
- Capas 1, 2a, 2b, 3, 4a, 4b, 4c, 5, 6a, 6b del plan de sostenibilidad.
- Commit F: renombrar parametros `liquidity_regime` -> `financial_regime`.
- Issue #5 (health-check): cerrado. `update_sec_13f cancelled` = FALSE_POSITIVE (doble dispatch manual 24-sep, run 36014839961). `cron_slots 3/5` = WONT_FIX (delay GitHub, cubierto por los 5 slots del daily_run).
- Dictamen externo Gate 0.5 aprobado con correcciones (aplicadas).

---

## 4. PENDIENTES VIVOS

### 4.1. Certificacion scheduled lunes 2026-10-19 23:17 UTC

Primer run scheduled con `target_session=2026-10-19`. Ciclo completo
v1.1+v1.2 desde cero: gate READY sin receipt, pipeline, guard, receipt
con `contract_validation_status`, attempt.

Si a las 23:35 UTC no ha disparado: `gh workflow run daily_run.yml`.
Los 5 slots cubren la ventana (03:17, 05:17, 07:17, 11:17 del 19/20-oct).

### 4.2. Otros frentes

- **H5.3** (OPEN): cron 20-nov-2026 06:17 UTC. Sin accion antes.
- **C2 (920)**: bloqueado por auditor externo.
- **Wyckoff 5b.X v3**: bloqueado por datos (~2027).
- **SOW v6**: pendiente firma externa.
- **Ubuntu 26**: `ubuntu-latest` migra el 19-oct-2026. Los workflows
  ya pinnen `ubuntu-24.04`.

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
- Tests: ver `ESTADO_SISTEMA.md`. Referencia: 3702 passed + 5 skipped.
- Gobernanza: `[OK] Gobernanza coherente`.
- Indice H1-H18: `[OK] indice actualizado`.
- Bitacora: `[OK] Bitacora dentro de la regla`.
- pyflakes silencio, compileall OK.

---

## 6. NOTAS CRITICAS

- **Verificar en CI, no solo en local.** Tests con `tmp_path` y fixtures sinteticos. No leer `data/` ni `outputs/` (gitignored en CI).
- **Un cambio = una verificacion = un commit.**
- **Verificar EOL (LF/CRLF) y BOM antes de cada patch.** `01_METODO` seccion 2. `report_generator.py` tiene BOM. `utils.py` y `finalize.py` son CRLF. Resto del corpus LF.
- **Here-strings >20 lineas o >5 `$` -> fichero temporal** (`[IO.File]::WriteAllText` + `py`). `01_METODO` seccion 10.
- **Receipt fail-closed (enmienda v1.1).** BLOCKED o UNAVAILABLE -> no se emite receipt.
- **Contrato EQUITY_EOD `critical=True`.** `coverage_at_session < 0.90` bloquea publicacion.
- **Schedule vs Dispatch.** Un dispatch reciente NUNCA oculta ausencia de schedule.
- **Gobernanza canonica.** Items a `docs/gobernanza/REGISTRO.yaml`. El HANDOFF es vista derivada.
- **Anclas temporales.** Fechas literales a `ANCLAS.yaml`.

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
