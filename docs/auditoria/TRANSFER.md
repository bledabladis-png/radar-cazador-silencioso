# TRANSFER DE SESION - 2026-09-21 v6

Documento complementario al PROMPT_MAESTRO v6.50 (docs/auditoria/PROMPT_MAESTRO.md).
No normativo. Si hay conflicto, gana el prompt.

**Como usarlo:**
1. Pegar este documento como primer mensaje.
2. Si tienes acceso al repo, adjuntar tambien PROMPT_MAESTRO.md.
3. Esperar confirmacion de asimilacion antes de empezar.

---

## 1. Rol

Eres el Ingeniero Supervisor del Radar de Rotacion Sectorial. Sistema
determinista, descriptivo, auditable, sin ML predictivo. Entorno
Windows, PowerShell, Python via py. Repo: D:\Macro_Sectorial.

Reglas de personalidad y metodo: ver PROMPT_MAESTRO v6.50 secciones 1 y 3.

---

## 2. Objetivo real del proyecto (leer antes de nada)

**El objetivo es IMPLEMENTAR el modulo IAE (Institutional Accumulation
Evidence) - analisis de posiciones institucionales via SEC 13F.**

NO es redactar documentos de diseño. NO es iterar propuestas ->
dictamenes -> propuestas. NO es acumular evidencia de que el diseño
esta bien.

**Alerta historica (2026-09-21):** el ciclo A.6.2-bis ha consumido 14
dictamenes (#44-#56) y 4 versiones de B1 (v1-v4) sin que se haya
escrito una sola linea de codigo B1. Eso es un fallo metodologico
del asistente anterior, no del auditor.

**Regla nueva (aplicar siempre):** cuando el diseño de una subfase
haya cerrado >=3 rondas de dictamen con patron "cierro N, aparecen N
nuevos", PARAR. Congelar diseño en la version vigente e IMPLEMENTAR.
Las dudas se resuelven escribiendo tests, no documentos.

---

## 3. Estado actual verificado (2026-09-21)

| Metrica | Valor |
|---|---|
| HEAD local | fd082a0 |
| origin/main | 9d4a81e |
| Ahead | 218 commits locales |
| Behind | 3 (bot CI) |
| Working tree | limpio |
| Push | NO (local-first IAE activo) |
| Prompt vigente | v6.50 |
| Tests locales | 1083 passed + 2 skipped + 3 failed preexistentes |
| pyflakes / compileall | LIMPIO / OK |
| Gate validacion | 10/10 |
| Cobertura radar | 313/313 |
| Contratos temporales | 10 |

**Los 3 failed de test_freshness.py son preexistentes** (parquets
locales desactualizados a 2026-09-16). Demostrado empiricamente en
`iae/evidence/p66_baseline_pre/`. NO son regresion.

---

## 4. Estado del modulo IAE (SEC 13F -> NIPC)

| Bloque | Estado |
|---|---|
| FA-1 + FA-2 (ingestion + identity) | CERRADO / PUSHED |
| NIPC (5 modulos + 91 tests) | IMPLEMENTADO (usa proxy observacional) |
| P60/P61/P38/P63/P64/P65 | CERRADOS documentalmente |
| P66 (§14.3 + reporting_dedup) | CERRADO (codigo implementado) |
| A.6.0 Gate 0 de los 3 bloqueantes | CERRADO |
| A.6.2-bis-B2-PIT (infraestructura temporal) | CERRADO (dictamen #53) |
| A.6.2-bis-B1 (TARGET + adaptador P38) | OPEN / BLOQUEADO (#56) |
| A.6.2-bis-B3 (semantica temporal 13F) | APPROVED COND / NO IMPLEMENTADO |

### 4.1. Unico modulo IAE implementado del ciclo A.6.2-bis

    src/institutional_accumulation/catalog_pit.py   (6991 bytes)
    tests/test_catalog_pit.py                       (16 tests)

Cubre: snapshots + manifest + sha256 externo + target_catalog_as_of.
NO cubre: catalog_key, TARGET_PAIRWISE, adaptador P38, colisiones,
validacion de asignacion (todo eso pertenece a B1).
---

## 5. Inventario real de lo que falta implementar en IAE

### 5.1. Modulos B1 (cero codigo, diseño v4 disponible)

    src/institutional_accumulation/identity/catalog_key.py       NO EXISTE
      - formato catalog_key
      - sha256_full + serializacion length-prefixed
      - load_assignments / validate_assignment
      - load_membership / validate_membership
      - B1_REQUIRED_COLUMNS + check_schema

    src/institutional_accumulation/identity/target_builder.py    NO EXISTE
      - build_target(snapshot, membership, assignments, ...)
      - vinculacion por snapshot_row_uid

    src/institutional_accumulation/identity/period_state.py      NO EXISTE
      - state_q4[K], state_q1[K]
      - identity_status enum
      - weight_status enum

    src/institutional_accumulation/aggregation/catalog_p38_adapter.py  NO EXISTE
      - catalog_to_p38_targets (full targets, sin filtro)
      - CoverageFeasibility

    src/institutional_accumulation/aggregation/catalog_validator.py   NO EXISTE
      - check_continuity
      - check_economic_collision
      - check_full_resolution (identity_status == RESOLVED)

### 5.2. Datos B1 (no existen)

    data/mappings/catalog_assignments.csv              NO EXISTE
    data/mappings/catalog_membership.csv               NO EXISTE
    data/mappings/catalog_reassignments_attempted.csv  NO EXISTE

### 5.3. B3 - semantica temporal

    src/institutional_accumulation/absence.py     NO EXISTE (o stub)
    Extender PositionRecord con period_end, filing_date, knowledge_date
    Extender provenance con effective_filing_accession + knowledge_date_status

### 5.4. Integracion (piezas implementadas pero no invocadas)

    coverage.py::compute_contractual_coverage   NADIE lo invoca
    reporting_dedup.py                          NO invocado por build_effective_reporting_snapshot
    temporal_validity.py                        NO invocado por security_identity (div. D1)
    catalog_pit.py                              NADIE lo consume (B1 sera el consumidor)

### 5.5. Datos historicos (no existen)

    Snapshot Q4 2025                            NO EXISTE (OpenFIGI masivo bloqueado)
    Snapshot Q1 2026                            NO EXISTE
    Crosswalk CUSIP->ticker curado              3 filas (Q-CUR), deuda grande

### 5.6. Thresholds / gates

    THRESHOLD_1                                 UNDEFINED
    THRESHOLD_2                                 BLOQUEADO (depende de B1+B2)
    Gate-NIPC.2                                 BLOQUEADO
    Gate-NIPC.3                                 NO AUTORIZADO
    DROP_DUP operacional                        NO AUTORIZADO
    Certificacion "acumulacion"                 BLOQUEADA

---

## 6. Decision pendiente del usuario (2026-09-21)

El usuario ha identificado el bucle de diseño como problema y ha
solicitado este TRANSFER para que el siguiente asistente retome el
trabajo **con el foco en implementar**.

**Dos opciones viables:**

### Opcion A - Congelar B1 v4 e implementar

- Implementar los 5 modulos de B1 (3 commits definidos en
  `iae/A62BIS_B1_SUBFASE.md` v4 §10).
- Las dudas que surjan en implementacion se resuelven con tests,
  no con documentos nuevos.
- NO someter v5 al auditor. NO redactar mas propuestas B1.

### Opcion B - Aparcar B1, avanzar otras partes

Piezas NO bloqueadas por B1:

- **B3 semantica temporal** (parcialmente independiente):
  `absence.py`, extension `PositionRecord` con 3 timestamps,
  RESTATEMENT / NEW HOLDINGS semantica.
- **Integracion `temporal_validity` en `security_identity`**
  (divergencia D1 del expediente F2.4).
- **Integracion `reporting_dedup` en el pipeline NIPC**.
- **Probe GHISALLO** CIK 0001825214 (+765%).
- **Curacion crosswalk CUSIP->ticker**.

**Recomendacion del asistente anterior:** Opcion A, con regla firme
de no abrir nuevas rondas de dictamen sobre B1. El diseño v4 cubre
mas de lo necesario para escribir codigo.

---

## 7. Ficheros clave

### 7.1. Contratos y documentos normativos

    NIPC_CONTRATOS_SEMANTICOS_v1.md          contrato de identidad, P38, P62-P65
    NIPC_COVERAGE_POLICY.md                  policy v1.0 (intacta)
    INSTITUTIONAL_ACCUMULATION_CONTRATO.md   contrato v1.1 del modulo IAE
    INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md  spec NIPC v1.4

### 7.2. Diseño del ciclo A.6.2-bis

    A62BIS_B1_SUBFASE.md       v4 (diseño B1, sin codigo)  <- congelar y usar
    A62BIS_B2_PIT_SUBFASE.md   CERRADO (unico con codigo)
    A62BIS_PROPUESTA.md        v9 (base historica)
    FASE_A6_PLAN.md            plan maestro A.6
    A60_EXPEDIENTE_AUDITOR.md  inventario de los 3 bloqueantes

### 7.3. Dictamenes e informes

    DICTAMENES.md              #1 a #56 (indice consolidado)
    INFORME.md                 informes consolidados (§1-§22)
    B2_PIT_INFORME_CIERRE.md   cierre formal B2-PIT
    P66_DOSSIER_AUDITOR.md     dossier P66

### 7.4. Evidencia empirica

    iae/evidence/p66_gate0*/           gates P66
    iae/evidence/p66_e2e_probe/        probe P66
    iae/evidence/p66_baseline_pre/     baseline freshness (3 failed preexistentes)
    iae/evidence/b2_pit_cierre/        cierre B2-PIT
    iae/evidence/nipc_gate0_*/         probes NIPC
---

## 8. Reglas criticas para el siguiente asistente

### 8.1. Sobre implementacion

- **No redactar mas documentos de diseño sin autorizacion explicita
  del usuario.** Si crees que hay algo que mejorar en B1, proponlo y
  espera. No lo conviertas en un documento nuevo.
- **Ante dudas de diseño, escribir tests.** Los tests revelan
  problemas reales; los documentos revelan problemas imaginarios.
- **Un cambio = un commit = una verificacion.** No saltarse el ciclo.

### 8.2. Sobre el ciclo A.6.2-bis

- B2-PIT: CERRADO por #53. NO reabrir.
- B1: diseño v4 congelado. Implementar sin abrir v5.
- B3: APPROVED COND. Puede implementarse en paralelo a B1.
- Si el auditor (externo o simulado) abre un nuevo bloqueo sobre B1,
  aplicar la regla de §2: PARAR. No entrar en bucle.

### 8.3. Prohibiciones vigentes

- NO modificar P38 (`coverage.py` firma y semantica intactas).
- NO activar OpenFIGI masivo.
- NO activar `DROP_DUP`.
- NO modificar contratos normativos.
- NO reacoplar B1 a B2-PIT.
- NO reescribir snapshots publicados.
- NO push a `origin/main`.
- NO certificar "acumulacion".

### 8.4. Sobre el uso del auditor

- El auditor es util para decisiones irreversibles (contratos,
  arquitectura grande).
- NO es util para refinar detalles de implementacion (eso lo resuelve
  el codigo + los tests).
- Si vas a elevar algo al auditor, primero pregunta al usuario si vale
  la pena.

---

## 9. Comandos de arranque

    Set-Location D:\Macro_Sectorial
    git log --oneline -10
    git status -sb
    git rev-list --count origin/main..HEAD
    py -m compileall . -q
    py -m pyflakes . 2>&1
    py -m pytest tests/ validation/ -q --tb=line

Esperado:
- HEAD = fd082a0 o posterior
- ahead 218 o mas
- working tree limpio
- 1083 passed + 2 skipped + 3 failed (freshness, preexistentes)
- pyflakes silencio
- compileall OK

---

## 10. Confirmacion esperada

    "Confirmado, contexto asimilado."

    Estado del sistema que reconozco:
      - HEAD fd082a0, ahead 218
      - 1083 passed + 2 skipped + 3 failed preexistentes
      - P66 CERRADO (contrato + codigo)
      - A.6.0 CERRADO
      - A.6.2-bis-B2-PIT CERRADO (dictamen #53)
      - A.6.2-bis-B1 OPEN / BLOQUEADO (#56). Diseño v4 congelado.
      - A.6.2-bis-B3 APPROVED COND
      - Dictamenes hasta #56
      - Objetivo real: IMPLEMENTAR IAE, no redactar mas documentos
      - Regla nueva: si el ciclo de diseño lleva >=3 rondas sin
        converger, PARAR e implementar

    Pregunta final: "Que hacemos?"

No empieces a proponer tareas sin antes confirmar asimilacion.

---

FIN DEL TRANSFER
Version 6.0 (2026-09-21). HEAD fd082a0.