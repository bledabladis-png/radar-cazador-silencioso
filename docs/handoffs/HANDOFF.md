# HANDOFF - Radar de Rotacion Sectorial

Documento de traspaso. Autocontenido. Pegar al inicio del nuevo chat
junto con `Consolidacion_Documentos/00_ARRANQUE.md`.

**Este fichero se sobrescribe cada sesion.** Historico en git.

---

## 1. ESTADO AL CIERRE (2026-10-25, sesion 4)

Rama:           main
HEAD:           consultar `git log --oneline -1` (no literal; al cierre 007442b7)
Sincronizacion: main == origin/main
Tests:          3705 passed, 2 skipped (filtro estandar).
                Sin filtro: 3719 passed, 2 skipped.
CI HEAD:        verde.
Validation Gate: 10/10 (no reproducible standalone; ver 4.1).
Working tree:   limpio.
Gobernanza:     docs/gobernanza/REGISTRO.yaml.
Writers:        24 (output_registry.py).
Sistema Wyckoff: holdings-based. 6 CSVs en outputs/history/.

**Sesion 4: 3 deudas de codigo cerradas + 1 falso positivo.**

Se cerro la lista de deudas de codigo de sesion 3:
- `VERIFY-LEADER-SELECTION-ROTO` (script roto en pipeline)
- `C4.5-SLPM-STRINGS-RESIDUALES` (9 ficheros productivos con menciones)
- `STRUCTURAL-ENGINE-LEADER-BREADTH-ZOMBIE` (parametro muerto, cosmetico)

Y se abrio + cerro como falso positivo un item que resulto ser un
defecto de observacion (mojibake en consola PowerShell, no en el
fichero):
- `REPORT-ENCODING-MOJIBAKE-ADC` -> FALSE_POSITIVE
- `SCHEMA-VERSIONING-OUTPUTS`: FIXED 2026-10-25 (commit `c9a848b5`).
- `WYCKOFF-ESTRUCTURA-UNIFICADA`: FIXED 2026-10-25 (commit `9960cdac`).
- `DATA-QUALITY-SCHEMA-V`: FIXED 2026-10-25.
- `SECTOR-REGIME-RENAME-RANK`: FIXED 2026-10-25 (Fase 4b).
- `WYCKOFF-HISTORICO-REGENERADO`: FIXED 2026-10-25 (Fase 3).

7 commits, CI verde en cada push. 4 frentes de codigo cerrados.

---

## 2. SESION 2026-10-25 (sesion 4)

### 2.1. VERIFY-LEADER-SELECTION-ROTO (37bca52d + 6511f567)

`validation/verify_leader_selection.py` importaba `config.index_tickers`
(borrado en C4.4) y leia `analisis_lideres*.csv` (no generados desde
C4.3). El step "Verify leader selection" del daily_run habria fallado
en el primer run-system real tras C4.4. Los runs del sabado 25-10
estaban en skipped por gate (no por exito del step).

Retirado:
- `git rm validation/verify_leader_selection.py`
- Step "Verify leader selection" (lineas 202-203 del daily_run.yml)
- Artefactos `analisis_lideres*.csv` (226-227) y `slpm_state.json` (239)
- `git rm outputs/state/slpm_state.json` (fosil sin productor desde C3.1)
- Comentario `data_quality.py:233`
- Bloque `# SLPM COVERAGE` huerfano en `settings.py:87-90`
- Fallback muerto `generate_docs.py:214`

### 2.2. C4.5-SLPM-STRINGS-RESIDUALES (c1e4af2e + 1c14b738)

El grep inicial (sesion 3) habia detectado 3 notas. El grep expandido
post-Commit A detecto 9 ficheros productivos (6 mas de los previstos).
La auditoria en 2 dimensiones aplica tambien al codigo: grep expandido,
no solo por lineas conocidas.

Retirado (9 ficheros):
- `dependency_tracker.py` v1.2 -> v1.3: DIRECT_DEPENDENCIES sin
  slpm_v12.py/stock_leader.py/Leader Breadth, liquidity.py ->
  liquidity_v2.py, anadido WLS -> sector_concentration.py. INDIRECT
  sin SLPM ni LIS, anadido WLS. EVIDENCE_CLASSIFICATION sin Leader
  Breadth. Nota 3 niveles sin leader breadth.
- `sectorial.py:54` sin SLPM
- `rankings.py:90,145` sin SLPM
- `engines.py:2` docstring sin "forzar lideres SLPM"
- `rs_internal.py:6` docstring sin State Machine ni stock_leader.py
- `momentum.py:73` sin index_leaders/stock_leader
- `report_generator.py:169` bloque `# C3 - LIDERES POR SECTOR` huerfano
- `helpers.py` bloque "Acciones lideres" eliminado
- `run.py` bloque limpieza CSVs de lideres eliminado

Verificado en run real (14.4 min, exit 0, gate 10/10, 24 writers):
cero menciones SLPM/slpm_v12/stock_leader/Leader Breadth/
leader_breadth/analisis_lideres/index_phase/index_leaders/
classify_wyckoff_phase en reporte. Cero "Acciones lideres".
Anti-Double-Counting con WLS y liquidity_v2.py.

### 2.3. STRUCTURAL-ENGINE-LEADER-BREADTH-ZOMBIE (7679d3fb + e2dd3b90)

`compute_structural_score` aceptaba `leader_breadth=0.5` como
parametro con default.

Diagnostico: el cuerpo NO usa `leader_breadth`. `STRUCTURAL_WEIGHTS`
solo tiene 3 claves (rs_structural 0.50, flow_structure 0.30,
persistence 0.20, suma 1.0). AST de llamadores: 10 llamadas (9 tests
+ engines.py), todas args=2 o kwargs explicitos sin el kwarg.
Cosmetico puro, no sesga el calculo.

Retirado: `leader_breadth=0.5` de la firma. Docstring ya correcto.
28 focalizados + 3705/2 suite verde.

### 2.4. REPORT-ENCODING-MOJIBAKE-ADC -> FALSE_POSITIVE (007442b7)

Detectado durante A5 de la verificacion C4.5: parecia haber mojibake
(ClasificaciÃ³n, CrÃ©dito) en el bloque Clasificacion de Evidencia del
Anti-Double-Counting del reporte.

Verificacion por bytes:
- `outputs/report/reporte_diario.md`: `Clasificación` y `Crédito` en
  UTF-8 correcto (C3-B3 = ó en ambos). Cero mojibake en el fichero.
- El "mojibake" era la consola PowerShell interpretando mal al leer
  con `Select-String`. El fichero estaba bien.
- `src/report_generator.py:open(..., encoding="utf-8")` correcto.

Cerrado como FALSE_POSITIVE en el mismo dia.

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
- C4.4: cerrado en sesion 3.
- `VERIFY-LEADER-SELECTION-ROTO`: FIXED 2026-10-25 (commit 37bca52d).
- `C4.5-SLPM-STRINGS-RESIDUALES`: FIXED 2026-10-25 (commit c1e4af2e).
- `STRUCTURAL-ENGINE-LEADER-BREADTH-ZOMBIE`: FIXED 2026-10-25 (commit 7679d3fb).
- `REPORT-ENCODING-MOJIBAKE-ADC`: FALSE_POSITIVE 2026-10-25.
- `DICT3-WYCKOFF-AUDIT-2026-10-25`: FIXED 2026-10-25 (commit c13cd3a5).
  H-03, H-06, PIT holdings cerrados. H-01, H-02 documentales.
  H-04, H-05, top-20, re-acumulacion abiertos como DICT3-*.
- `SECTOR-REGIME-MATRIX-STRUCTURE-POSITIVE`: SUPERSEDED 2026-10-25
  por Bloque 2 (structure_ratio continuo + doc 68).

---

## 4. PENDIENTES VIVOS

### 4.1. Fases 2-6 del refactor de outputs (post Fase 1)

Fase 1 (`SCHEMA-VERSIONING-OUTPUTS`, commit `c9a848b5`) cierra el
incidente de schema mezclado. Quedan 5 fases estructurales antes de
volver a `structure_score` y a la descarga:

- ~~**Fase 2 (documental):** `data_quality.SOURCES` reporta schema_v~~
  HECHA 2026-10-25. Columna Schema vN en tabla 'Calidad, frescura y
  cobertura'. `_read_schema_version` lee `.schema` lateral.
- ~~**Fase 3 (diferida):** regeneracion historica completa~~
  HECHA 2026-10-25. 11946 filas (1086 fechas), validacion exacta
  contra .csv activo. Doc 66. **Hallazgo critico:** structure_positive
  = 0/11946 con el universo correcto (pct_range domina 47-92%). Fase 5
  requiere rediseño, no calibracion.
- ~~**Fase 4 (critica):** unificacion de los 4 caminos de estructura~~
  HECHA 2026-10-25 (`9960cdac`). Snapshot unico en
  `compute_wyckoff_snapshot`. Agregado v3.0 y 6 CSVs derivan del mismo
  snapshot. Test de coherencia en CI.
- **Fase 5:** `structure_score` continuo con datos coherentes.
- **Fase 6:** descarga unificada (`Router` con politica unica para
  fed_data + treasury_data; auditar `b30a9182`, `4bb54f80`).

### 4.2. Deudas tecnicas OPEN

- `CURRENT-PHASE-BASIS-CSV`: columna `basis` no va al reporte (solo al
  CSV descargable). Alcance pequeno.
- `ACCUMULATION-ZERO`: estado transitorio por diseno. No es bug.
- `VALIDATION-GATE-NO-STANDALONE`: gate no reproducible standalone.
  Solo se ve dentro de `py run.py` con `Select-String "VALIDATION GATE"`.
- `IWM-HOLDINGS-EXPANSION`: sub-fase de descarga, protocolo de 6 pasos.
  IWM excluido del holdings-based (top-20 = 6.2% de 1977 holdings).
- `PODA-BITACORA-DIA-EN-CURSO`: mensaje del script opaco. El script
  reporta "1 heading(s) a podar (de 11 sesiones unicas): 2026-10-13"
  sin mencionar "dia en curso". Bug + ambiguedad en seccion 5 de
  bitacora.

**Del dictamen externo 2026-10-25 (doc 68):**

- `DICT3-H04-INTERVALS` (OPEN): structure_ratio continuo publicado;
  falta ICs por cohorte anual, criterios de inclusion, tratamiento
  de solapamiento temporal y correlacion cross-sector. Sin esto es
  descriptivo, no inferencial.
- `DICT3-H05-FILLNA-FUTURE` (OPEN): anadir columna de calidad por
  fila en build_ticker_df (bar_flags) para distinguir barra-sin-
  negociacion de barra-incompleta, sin romper contrato v1.8.
- `DICT3-TOP20-COBERTURA` (OPEN): universo dinamico por grupo
  (objetivo 80-90% peso). XLI 57.2%, SPY 49.9% actuales.
- `DICT3-REACCUMULATION` (OPEN): detector de re-acumulacion de
  continuacion en tendencia alcista. Frente nuevo con pre-registro
  propio.

### 4.3. SOW-BEAR-CONFIRMATION (IN_PROGRESS)

- Requiere contrato v1.10 firmado por auditor externo antes de implementar.
- Siguiente accion: redactar `docs/auditoria/wyckoff/57_contrato_v1_10.md`.
- Sin codigo en produccion hasta firma.
- `detect_sow_v2` en codigo con tests. `WYCKOFF_SOW_*` = None.

### 4.4. Deuda documental residual

- `62_sistema_completo_2026-10-24.md`: parcialmente obsoleto (no menciona
  C4.4). Cabecera dice "Tests: 3776 passed" (real 3705/2). Correccion
  diferida.
- `00_ARRANQUE.md` seccion 3: bloque scheduled-verify dice "20 writers
  (hoy 24 tras C2.1+C4.3)". Foto historica del run 22-10; actual 24.
  Mezcla cifras historicas y actuales. Correccion diferida.
- HANDOFF seccion 4: renumerada en sesion 4 (era 4.2-4.6, ahora
  4.1-4.5). Cosmetico.

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
  (con -n auto).
- Gobernanza: `[OK] Gobernanza coherente`.
- Bitacora: FALLA con PODA-BITACORA-DIA-EN-CURSO abierto. Es esperado
  mientras el item siga OPEN. No bloquea trabajo.
- pyflakes sobre `.` silencio absoluto.
- compileall OK.

---

## 6. NOTAS CRITICAS

Verificacion:
- Verificar en CI, no solo en local.
- Un cambio = una verificacion = un commit.
- pyflakes sobre todo el repo (`.`), no por fichero.
- `git diff --check` despues de cualquier borrado por regex.
- Antes de cambiar firmas de tuplas/dicts, enumerar TODOS los
  consumidores via AST, no regex de texto.
- Cuando un `except` captura y sigue, verificar si el problema esta
  silenciado. El bug de rotacion sectorial duro desde C5.1 hasta
  sesion 3 por un `except` que solo imprimia.
- Auditar el reporte en 2 dimensiones: headers `^## ` Y menciones
  textuales. La auditoria por headers dejo pasar 11 menciones a SLPM
  en post-C4.4. Aplica tambien al codigo: el grep inicial de C4.5
  (sesion 3) dio 3 ficheros, el grep expandido (sesion 4) dio 9.

Herramientas:
- Here-strings de PowerShell son fragiles. `@"..."@` (double-quote)
  interpola y rompe con backticks. `@...@` (single-quote) es mas
  seguro pero sigue fallando con contenido Unicode denso (seccion,
  acentos, backticks combinados). Leccion sesion 3 + 4: 3 fallos.
  Metodo seguro: `[System.IO.File]::ReadAllBytes` / `WriteAllBytes`
  con `[System.Text.Encoding]::UTF8.GetString/GetBytes` y edicion por
  lineas en PowerShell o Python. Evitar here-strings para ficheros
  criticos (REGISTRO, HANDOFF, bitacora).
- Mojibake en consola PowerShell no es mojibake en el fichero.
  Verificar por bytes (`[System.IO.File]::ReadAllBytes`) antes de
  abrir item. `Select-String` puede mostrar mal el Unicode al filtrar.
- El test de gobernanza exige commit alcanzable + evidencia >= 20.
  Secuencia: commit de codigo -> commit documental con hash. No se
  puede marcar FIXED en el mismo commit que crea el cambio.
- Un parametro con default neutro que nadie pasa puede parecer sesgo
  y ser cosmetico. Verificar el cuerpo de la funcion, no solo la
  firma. `STRUCTURAL-ENGINE-LEADER-BREADTH-ZOMBIE` era prioridad 2
  en HANDOFF sesion 3; debio ser prioridad 3.

Contratos:
- Receipt fail-closed (enmienda v1.1). BLOCKED o UNAVAILABLE -> no se
  emite receipt.
- Contrato EQUITY_EOD `critical=True`. coverage < 0.90 bloquea.
- Gobernanza canonica. Items a `docs/gobernanza/REGISTRO.yaml`. El
  HANDOFF es vista derivada.
- Sistema Wyckoff vigente: holdings-based. ETFs solo como precio de
  referencia. `classify_wyckoff_phase` retirado. Indices legacy
  retirados.
- `sector_regime_matrix.py` deriva `structure_positive` de holdings
  (`sector_wyckoff_distribution_df`). Pendiente validacion empirica.
- Los 3 runs del pipeline verificados en sesion 4 (uno a fondo, 14.4
  min): NaNs >10% en ~35 tickers son preexistentes (IPOs recientes
  con historico parcial en Yahoo). Lista completa en bitacora.
- daily_run con gate.should_run=false (fin de semana/festivo) muestra
  runs de 1 minuto en skipped. No es fallo del pipeline; es el gate
  decidiendo no ejecutar. Verificar gate, no solo conclusion final.

---

## 7. REFERENCIAS

Corpus normativo: `Consolidacion_Documentos/` (12 docs).
Gobernanza: `docs/gobernanza/` (README, REGISTRO.yaml, ANCLAS.yaml).
Contrato del gate: `docs/auditoria/daily_run_gate_contrato_v1.md` +
enmiendas v1.1 y v1.2.
Expediente Wyckoff: `docs/auditoria/wyckoff/` (51-63).

Documentos clave para el traspaso:
- `63_plan_sistema_holdings_2026-10-24.md`: plan del sistema actual.
- `62_sistema_completo_2026-10-24.md`: mapa tecnico (parcialmente
  obsoleto; no menciona C4.4).
- `60_current_phase_rule_v1.md`: pre-registro v1 fallido.
- `61_utad_spring_pending_v1.md`: pre-registro UTAD_PENDING fallido.

---

**Fin del handoff.**
