# HANDOFF - Radar de Rotacion Sectorial

Documento de traspaso entre sesiones. Autocontenido. Pegar al inicio
del nuevo chat junto con `Consolidacion_Documentos/00_ARRANQUE.md`.

**Este fichero se sobrescribe cada sesion.** El historico queda en git.
No crear variantes con fecha.

---

## 1. ESTADO AL CIERRE (2026-10-14 ~12:30 CEST)

Rama activa:    main
HEAD:           5e58197a
Sincronizacion: main == origin/main
Tests:          3273 passed + 5 skipped + 0 failed
Validation Gate: 10/10
Working tree:   limpio
Corpus:         v3 (Consolidacion_Documentos/00-08 + ESTADO_SISTEMA)

---

## 2. SESIONES 13-14 OCT (auditoria externa + revision del reporte)

### 2.0. Auditoria externa (13-oct)

Informe tecnico del auditor sobre el reporte del 13-oct: 16 hallazgos
originales (P1.x, P2.x, P3.x) + dictamen de seguimiento con 6 nuevos
(N1-N6). Todos atendidos. Documento:
`docs/auditoria/radar/audits/ANEXO_VERIFICACION_2026-10-13.md`.

- **P1.1 (fix real):** Leader Breadth usaba `close/precio_etf` (ratio
  nominal, depende de splits). Cambiado a `rs_ratio_20d` (retorno
  relativo 20d). Commits 0413518e + 00b01f46 (N1 fail-closed).
- **P1.3:** pesos WLS centralizados en `config/weights.WLS_WEIGHTS`.
- **P1.5 / N4:** clasificacion de evidencia en 3 niveles (OBSERVABLE /
  DERIVADA-DIRECTA / DERIVADA-COMPUESTA). Nota explicita de que la
  agregacion es heuristica descriptiva, no estadistica.
- **N2:** LIS marcado provisional (`lis_validated=False`), sin validar
  como senal decisoria. Requiere recalibracion estadistica.
- **N3:** validacion del contrato WLS extraida a
  `_validate_wls_inputs` (funcion pura). Persistence fuera de [0,1]
  clipeada + status. 8 tests en `test_wls_validation_n3.py`.
- **N5:** decision explicita sobre asincronia precio-flujo: mantener
  comparacion asincrona pero etiquetar con fecha del flujo en
  encabezado + nota reforzada.
- **N6:** formulas de cobertura y ventanas PCR publicadas. Correccion
  de la nota P2.4 (decia 252 sesiones; real 20).
- **P2.1:** test de contrato `_wyckoff_sign` (6 casos).

### 2.0b. N7 - Bug propio detectado durante la auditoria

`evidence_matrix.csv` y `sector_regime_matrix.csv` tenian 1 fila por
fecha en lugar de 11. Causa: `merge_keys=('date',)` en el registro
declarativo, cuando deberia ser `('date','sector')`. Bug pre-existente
a la migracion del contrato (los writers originales ya sobreescribian
con 1 fila). Fix: commit f5db9f5f + test permanente
`test_merge_keys_incluye_sector_si_csv_lo_tiene`.

### 2.0c. Revision del run del 13-oct (14-oct)

Cron de la madrugada: `38498223287` (23m30s, pipeline completo).
`Verificacion de outputs: OK (19 checks)`. Target session 2026-10-13.
N7 confirmado: `evidence_matrix` y `sector_regime_matrix` con 11 filas
en 2026-10-13 (fechas antiguas siguen con 1 fila; irrecuperables).

### 2.0d. Hallazgo N8 - LIQUIDEZ REAL (pendiente, documentado)

**NO fixado, en espera de sesion de diseno.**

El run del 13-oct declaro `Liquidez Real: CRISIS` (antes `NEUTRAL`).
Cambio a raiz del fix B1 (`fred.py` no extiende indice hasta hoy).
Diagnostico:
- `fed_funds` es serie MENSUAL (ultimo dato 2026-09-01).
- Las otras 3 (`fed_balance`, `reverse_repo`, `sofr`) son diarias/semanales.
- `robust_zscore(window=60)` aplicado por igual a una serie mensual no
  es estadisticamente valido (~2 observaciones por ventana).

Auditor externo (14-oct) rectifico dos afirmaciones mias:
- FEDFUNDS y SOFR NO son la misma variable con distinta frecuencia.
  FEDFUNDS: tasa efectiva overnight no garantizada entre depositarias.
  SOFR: financiacion overnight garantizada con Treasuries.
  Segmentos distintos del mercado monetario. No justifican exclusion
  por duplicidad.
- El score mezcla magnitudes de balance (fed_balance, reverse_repo)
  con precios de financiacion (sofr, fed_funds). Decision de diseno,
  no detalle que se cambia de paso.

**Propuesta (NO ejecutada):** documento de diseno
`docs/auditoria/radar/design/liquidity_score_redesign.md` con 4 opciones
(B1: DFF diaria, B2: eliminar FEDFUNDS, B3: separar en dos scores,
B4: mantener + documentar). Requiere backtest historico antes de tocar
pesos o composicion. Sesion propia. Estimacion: 3-5 h.

**Lo que NO se ha tocado:** composicion del score, pesos, 4-tuple,
umbrales, descarga de FEDFUNDS.

### 2.0e. Deuda doc-codigo detectada

`Consolidacion_Documentos/02_ARQUITECTURA.md:181` dice "Liquidity Score
(5 componentes): SOFR, WALCL, RRP, Discount...". El codigo usa 4
(`fed_balance`, `reverse_repo`, `sofr`, `fed_funds`). Gap preexistente.

---

## 3. QUE SE HIZO 11-12 OCT (sesion larga, 17 commits)

### 2.1. Deuda limpia (11-oct)

- bd2a92b5: retirados 3 scripts one-shot (`regenerate_*`).
- `src/report/sectorial.py.orig` borrado (huerfano, ignorado por git).
- `outputs/audit` reducido: 131 -> 66 MB.
- d67c1548: `ESTADO_SISTEMA.md` regenerado (estaba stale).

### 2.2. Contrato de persistencia de outputs (11-12 oct)

**Objetivo.** Cerrar la clase entera de bugs "writer silencioso"
(el gap 09-oct fue el sintoma). No un parche por CSV.

**Solucion en 3 piezas + AST test:**

- `src/pipeline/output_registry.py`: registro declarativo de 19 outputs
  (18 temporales + 1 snapshot). `WriteReceipt` + `ReceiptCollector`.
- `src/pipeline/output_writer.py`: wrapper unico `persist(name, df, ...)`
  / `skip(name, reason)`. Toda escritura a `outputs/history/` pasa por
  aqui. Fail-closed: name no registrado -> KeyError; razon de skip no
  legitima -> ValueError; derivar `date_value` del propio df si no se
  pasa.
- `src/pipeline/output_verifier.py`: `verify(collector)` con 3 reglas.
- `run.py`: collector activo al inicio; verificacion antes de cerrar;
  `sys.exit(1)` si falla; persiste `outputs/state/writer_receipts.json`.
- `tests/test_output_contract_ast.py`: AST que prohibe `to_csv()` fuera
  del wrapper. Excepciones: `output_writer.py`, `save_sector_rankings`,
  `compute_indices_intl` (outputs/report/, efimeros).

**19 writers migrados** en 6 commits por fichero. Gate 10/10, receipt v1
y `pipeline_gate.py` intactos.

Commits: f0311cdb, 8596690c, c43a6b00, 2593ac91, ceecad70, b1237083,
ae752882, ab838dbf.

### 2.3. Verificacion extendida de outputs (12-oct)

`scripts/verify_outputs.py`: comprueba que las ultimas N sesiones NYSE
tienen fila en cada CSV temporal del registro. Default `--since
2026-10-11` (despliegue del contrato). Gap real detectado:
`sector_flow_characteristics` sin fila para 09-oct (ROI bajo, sin
accion). Commits f428bf51, 69506319.

### 2.4. Triaje auditoria externa 2026-09-26 (12-oct)

- **ALTA: 6/6 cerradas** (F3-05, F3-16, F3-17, A5-13, F6-10, F6-28).
- **MEDIA: 71/71 clasificadas, 0 vivas.** 55 cerradas por cambios
  posteriores + 7 WONT FIX razonado + 9 rescop doc (cerradas hoy).
- **BAJA: 53/61 triadas.** 8 F6-* sin descripcion (informe original no
  portado; commit bc656374 solo tiene 3 ficheros).

Documentos: `docs/auditoria/radar/audits/TRIAGE_2026-10-12.md`,
`TRIAGE_BAJAS_2026-10-12.md`.

### 2.5. Verificacion del run del domingo 11-oct

El cron domingo madrugada (38103357426, 03:53 CEST) **skipeo por dise?o
del gate**: receipt del sabado 38016582210 ya emitido para
target_session=2026-10-09. Sin anomalia.

Gap 09-oct cerrado por run manual `ceb21d8a` (23:24 CEST). Causa raiz
(writer silencioso + receipt falso) cerrada por el contrato nuevo.

### 2.6. Cierre de deudas de higiene (12-oct)

- 0035be38: politica `outputs/audit/` en `01_METODO.md` seccion 10.1;
  poda `00_ARRANQUE` 11670 -> 10240 B; snapshot a 3247 tests.
- 06dd6cd6: D-07 cerrado con notas cruzadas en `sectorial.py` (Ret Lider
  = absoluto) y `market_context.py` (vs mercado 20d = relativo).
  D-04 cerrado sin parche (ya dice "Robust Z-Score").
- F2-02, F2-03, F2-04, F2-07, A5-57: anotados en `02_ARQUITECTURA`.

---

## 3. PENDIENTES VIVOS

### 3.1. Bloqueados por terceros
- C2 (920): discrepancia H1-B. Bloqueado por auditor externo.
- SOW protocol v6: borrador `48_sow_protocol_v6.md` pendiente firma.
- gold-standard-v4: archivada como tag.

### 3.2. Bloqueados por datos
- Wyckoff 5b.X v3: bloqueado hasta n_confirmed_H20_complete >= 550
  (~2027). Script y protocolo inmutables.

### 3.3. Con fecha futura
- H5.3: cron trimestral 13F 20-nov-2026 06:17 UTC.
- Primer test del contrato en prod: cron 23:17 UTC de hoy (13-oct
  01:17 CEST) con target_session=2026-10-12. Verificar
  `gh run view <id> --log | grep "Verificacion de outputs"`.

### 3.4. Deuda sin urgencia
- 8 F6-* sin descripcion: CERRADO como WONT FIX razonado
  (investigacion 2026-10-12; informe FASE 6 nunca existio como
  fichero; ver `TRIAGE_BAJAS_2026-10-12.md` seccion 4).
- ~11 BAJA vivas de baja prioridad: CERRADAS 2026-10-13 (ver
  TRIAGE_BAJAS_2026-10-12.md). 0 vivas reales.
- Regenerar `sector_flow_characteristics` 09-oct (ROI bajo).
- Cosmeticos: `render_calidad_datos` ubicacion (volatility_mte.py),
  divergencia runtime/CSV sin test, `test_render_freshness_d36`
  fragilidad.

---

## 4. COMO ARRANCAR

    Set-Location D:\Macro_Sectorial
    git log --oneline -5
    git status -sb
    py scripts\generate_estado_sistema.py
    py -m pytest tests/ -q -n 12 --tb=line -m "not integration_real_data"
    py -m pyflakes . ; py -m compileall . -q

Esperado:
- HEAD = 06dd6cd6 o posterior.
- ahead 0, behind 0.
- working tree limpio.
- 3247 passed + 5 skipped.
- pyflakes silencio, compileall OK.

Si divergencia: investigar antes de tocar nada.

---

## 5. NOTAS CRITICAS

- **No usar run.py en horario de mercado** (13:30-20:00 UTC).
  Politica: snapshot pre/post + revertir, salvo publicacion
  deliberada. Aplicado en 01_METODO seccion 4.

- **EOL del corpus: LF** en todos los docs (`00_ARRANQUE`,
  `01_METODO`, `02_ARQUITECTURA`, `04_HISTORICO`, `05_BITACORA`).
  Corregido 2026-10-12. **Regla:** al escribir docs con Python, usar
  `Path.write_text(..., newline='\n')` o `[System.IO.File]::WriteAllText`
  para no convertirlos a CRLF. Ver `01_METODO` seccion 6.

- **BOM:** `Out-File -Encoding utf8` de PowerShell mete BOM (U+FEFF).
  Usar `-Encoding ascii` para ficheros nuevos o quitar BOM despues.
  El corpus y los `.py` no llevan BOM. `test_output_contract_ast.py`
  usa `utf-8-sig` por defensa.

- **Patrones de bug** en `01b_PATRONES.md`. **Trampas PowerShell** en
  `01_METODO` seccion 6bis.

- **No confundir diagnostico cerrado con hipotesis fuerte.**

- **Antes de tocar un writer/reader temporal**, verificar la
  cadena completa: productor, consumidor, render.

---

## 6. REFERENCIAS

Corpus normativo en `Consolidacion_Documentos/` (12 docs):
00_ARRANQUE (rol + reglas), 01_METODO (patch/verificacion),
01b_PATRONES, 01c_LECCIONES, 02_ARQUITECTURA, 03_IAE,
04_HISTORICO (temas), 05_BITACORA (sesiones), 06_IAE_P65_P66,
07_RUNBOOK, 08_AUDITORIA_SECTOR_REGIME, ESTADO_SISTEMA (auto).

Expedientes: docs/auditoria/wyckoff/, docs/auditoria/iae/golden/,
docs/auditoria/s03_deep/, docs/auditoria/radar/audits/ (incluye
TRIAGE_2026-10-12.md + TRIAGE_BAJAS_2026-10-12.md).

---

**Fin del handoff.**
