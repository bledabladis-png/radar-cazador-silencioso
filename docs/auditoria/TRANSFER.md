# TRANSFER - Guia de onboarding

**NO es fuente de estado.**
**Estado vivo:** `iae/ESTADO_SISTEMA.md` + `iae/ESTADO_DECLARADO.md`.
**Referencia unica del modulo IAE:** `iae/IAE_MAESTRO.md`.

---

## Rol

Ingeniero Supervisor del Radar de Rotacion Sectorial (IAE).
Reglas de personalidad y metodo: `PROMPT_MAESTRO.md` secciones 1 y 3.

---

## Estado al cierre de la sesion (2026-09-23)

    HEAD                 ver docs/auditoria/iae/ESTADO_SISTEMA.md
    Ahead                ver docs/auditoria/iae/ESTADO_SISTEMA.md
    Working tree         LIMPIO
    Tests IAE            759 passed (criterio AST, ver abajo)
    Suite global         1440 passed + 2 skipped + 3 failed
                         (los 3 failed son test_freshness, ambientales)
    Push                 NO (local-first IAE)
    Cobertura IAE        110/110 funciones con llamada real (92% lineas)

Criterio del conteo de tests IAE: ficheros `tests/test_*.py` que importan
`src.institutional_accumulation` (verificado por AST). Reproducible con
`scripts/iae_test_census.py`.

---

## Documentos clave

    docs/auditoria/PROMPT_MAESTRO.md              v7.2
    docs/auditoria/TRANSFER.md                    este documento
    docs/auditoria/README.md                      navegacion

    docs/auditoria/iae/IAE_MAESTRO.md             referencia unica del modulo IAE
    docs/auditoria/iae/ESTADO_DECLARADO.md        estado de fases y deuda
    docs/auditoria/iae/ESTADO_SISTEMA.md          hechos autogenerados
    docs/auditoria/iae/evidence/                  evidencia empirica

Scripts reproducibles del modulo IAE:

    scripts/iae_reconciliation_b1.py              cadena completa delta + NIPC
    scripts/iae_validate_crosswalk_openfigi.py    validacion externa del crosswalk
    scripts/iae_test_census.py                    censo riguroso de tests
    scripts/iae_contractual_coverage.py           cadena contractual
                                                  (target -> adapter -> coverage)
    scripts/iae_pipeline.py                       orquestador ingestion + identity

Documentos historicos (no normativos): NIPC_CONTRATOS_SEMANTICOS_v1.md,
NIPC_COVERAGE_POLICY.md, INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md.

---

## Estado del modulo IAE en una frase

Implementado, testeado y verificado end-to-end sobre datos reales.
Con el crosswalk CUSIP extendido (246 filas, 242 tickers unicos):

- Q4 2025: 240 tickers contribuyen al TARGET
- Q1 2026: 240 tickers contribuyen al TARGET
- Delta radar: 553.321 filas (n_unresolved_identity = 0)
- NIPC radar: -4.316.734.936

Aislado del pipeline productivo. Entrega los datos necesarios para
el analisis de acumulacion institucional.

Detalle completo en `iae/IAE_MAESTRO.md`.

---

## Sesiones recientes (2026-09-22 / 2026-09-23)

Fase A (bloqueantes del dictamen auditor externo, cerrada):

- B1 - Reconciliacion radar + complemento = full. Verificado con
  `scripts/iae_reconciliation_b1.py`. Split oficial: por `canonical_security`
  (formato `equity:TICKER`). El split por CUSIP no es viable: las filas
  con `observed_security_key=cusip:XXX` son UNRESOLVED_IDENTITY.
- B2 - Cadena forense congelada. sha256 de INFOTABLE Q4/Q1, pandas 2.3.3,
  timestamp UTC en el pie de §12.
- B3 - Validacion externa del crosswalk via OpenFIGI. 227 de 243 CUSIPs
  verificados con shareClassFIGI coincidente, 16 sin hit (prefijos no-USA).
- B4 - Riesgo PIT declarado (§13.9): identidad historica vs pertenencia
  historica al radar.

Fase B (13 criticos, cerrada):

- B.1 - Trazabilidad numerica y honestidad de alcance (C5, C6, C12, C13, C14).
- B.2.1 - Semantica NIPC, cadena contractual, coherencia numerica
  (C7, C11, C16, C18).
- B.2.2 - Rename `cusip_ticker_exceptions.csv` -> `cusip_radar_crosswalk.csv`
  (C15). El string `source` interno se conserva como provenance historica.

Fase C (3 cambios funcionales, cerrada):

- C8 - `reporting_dedup` declarado diagnostico separado, no etapa del
  flujo contractual.
- C9 - `aggregate_positions_by_shareclass_figi` con `return_stats=True`:
  contadores observables (recibidos, verificados, excluidos por status).
- C10 - `coverage_available` (medible) + `coverage_quality`
  (UNAVAILABLE/PARTIAL/COMPLETE con threshold 0.95 sobre las DOS
  dimensiones). `coverage_status` legacy intacto.

---

## Reglas de operacion

- No push a `origin/main` (local-first IAE).
- No OpenFIGI masivo. Solo consultas dirigidas.
- No modificar codigo sin ciclo previo.
- No integracion a produccion sin validacion funcional.
- Todo bloque que haga `git commit` debe condicionarse a tests verdes
  (`if ($LASTEXITCODE -eq 0)`). Leccion del commit 977249b.

---

## Comandos de arranque

    Set-Location D:\Macro_Sectorial
    git log --oneline -5
    git status -sb
    py scripts\generate_estado_sistema.py
    py -m pytest tests/ -q --tb=line
    py -m pyflakes src\ scripts\ ; py -m compileall . -q

Esperado:

- HEAD ver docs/auditoria/iae/ESTADO_SISTEMA.md
- ahead ver docs/auditoria/iae/ESTADO_SISTEMA.md
- working tree limpio
- 1440 passed + 2 skipped + 3 failed (test_freshness, preexistentes)
- pyflakes silencio, compileall OK

Censo del modulo IAE (comando aparte, tarda unos segundos):

    py scripts\iae_test_census.py

Esperado: 42 ficheros, 759 tests collected.

---

## Confirmacion esperada

"Confirmado, contexto asimilado."

Estado que reconozco:

- HEAD ver docs/auditoria/iae/ESTADO_SISTEMA.md
- IAE implementado, testeado, aislado
- Crosswalk extendido a 246 filas (242 tickers unicos)
- Verificado end-to-end sobre 13F real Q4/Q1 (NIPC radar -4.316.734.936)
- Bloqueantes A (B1-B4) cerrados
- Criticos B (C1-C18) cerrados, salvo C17 (registro, sin accion)
- Fase C (C8, C9, C10) cerrada
- Prohibiciones respetadas
- Siguiente: Fase D - reenviar IAE_MAESTRO.md al auditor externo

Pregunta: "Que hacemos?"

---

FIN DE TRANSFER. 2026-09-23.
