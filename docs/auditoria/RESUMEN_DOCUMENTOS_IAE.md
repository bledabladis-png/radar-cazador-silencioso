# RESUMEN DE DOCUMENTOS - RADAR SECTORIAL / IAE (post dictamen TOP 50)

Fecha: 2026-09-19
Referencia: prompt v6.36, HEAD local d44bba0 (o hash tras amend).
Naturaleza: mapa de documentos. NO normativo. Ante conflicto gana el
            Prompt Maestro.

=============================================================================
1. NORMATIVOS (contrato principal)
=============================================================================

| Fichero                           | Contenido                            | Rol |
|-----------------------------------|--------------------------------------|-----|
| docs/auditoria/PROMPT_MAESTRO.md  | v6.36. Rol, premisas, metodologia,   | Normativo. |
|                                   | arquitectura, decisiones, K-IDs.     |     |
| docs/auditoria/FOLLOWUPS.md       | Registro vivo de K-IDs. Entrada mas  | Trazabilidad. |
|                                   | reciente: dictamen TOP 50 + spec v1.4.|    |
| docs/auditoria/NIPC_COVERAGE_POLICY.md | Thresholds UNDEFINED. Filer      | Politica. |
|                                   | continuity como control de integridad.|    |
| docs/auditoria/TRANSFER_SESION_2026-09-19_POST_TOP50.md | Contexto | Transfer. |
|                                   | para asistente entrante. NO normativo.|   |

=============================================================================
2. ESPECIFICACION NIPC (activa + versiones preservadas)
=============================================================================

| Fichero                                                    | Version | Tamano |
|------------------------------------------------------------|---------|--------|
| docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md | v1.4 (ACTIVA) | 66.7 KB |
| docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION_v1.3.md | v1.3 (preservada) | 65.6 KB |
| docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION_v1.2.md | v1.2 (preservada) | 60.2 KB |
| docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION_v1.1.md | v1.1 (preservada) | 52.7 KB |
| docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION_v1.0.md | v1.0 (preservada) | 37.0 KB |

Contenido clave de la v1.4 activa:

  - Seccion 3.6  temporal_validity (C4-revisada: identity != metadata).
  - Seccion 3.13 canonical_security (C1-revisada: ticker prohibido,
                5 estados + 4 kinds, observed_security_key).
  - Seccion 3.15 filer_continuity (CERRADO como caracterizacion,
                position_mass_status, NT->HR evidencia documental).
  - Seccion 4.7  matching interperiodo sin report_period.
  - Seccion 10   6 metricas pairwise obligatorias.
  - Seccion 11.2 interfaces publicas.
  - Seccion 14.2 Gate-NIPC.2 BLOQUEADO por coverage/thresholds.

=============================================================================
3. DICTAMENES DEL AUDITOR (6 vinculantes)
=============================================================================

Ciclo FA-1/FA-2 (cerrado y pusheado, en origin/main):

  docs/auditoria/INSTITUTIONAL_ACCUMULATION_DICTAMEN_GATE0.md
  docs/auditoria/INSTITUTIONAL_ACCUMULATION_GATE_FA1_INFORME.md
  docs/auditoria/INSTITUTIONAL_ACCUMULATION_GATE_FA2_DICTAMEN.md
  docs/auditoria/INSTITUTIONAL_ACCUMULATION_GATE_FA2_INFORME.md

Ciclo NIPC (post origin/main, 46 commits locales):

  docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION_V11_DICTAMEN.md
    -> PASS CONDICIONADO. C1-revisada + C4-revisada (spec v1.2).

  docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_PROBE_DICTAMEN.md
    -> GO condicionado. Mini-probe filer continuity autorizado.

  docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_MINIPROBE_DICTAMEN.md
    -> GO condicionado. Hallazgo cuantitativo obligatorio.

  docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_TOP50_DICTAMEN.md
    -> FILER CONTINUITY CERRADO. Coverage/thresholds bloquean Gate-NIPC.2.

=============================================================================
4. INFORMES DEL INGENIERO (4)
=============================================================================

  docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_PROBE_INFORME.md
    -> Probe end-to-end Q4->Q1 + hallazgo Vanguard.

  docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_MINIPROBE_INFORME.md
    -> Mini-probe top 20 filer continuity.

  docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_MINIPROBE_ADDENDUM.md
    -> Correccion cuantitativa 35.329B vs 70.236B.

  docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_TOP50_INFORME.md
    -> Top 50 filer continuity (cierre).

=============================================================================
5. EVIDENCE (probes reproducibles)
=============================================================================

docs/auditoria/evidence/nipc_gate0_probe/
  README.md                        Documentacion de la evidencia.
  HASHES.txt                       SHA-256 de cada artefacto.
  probe_nipc_e2e.py                Probe end-to-end Q4->Q1.
  probe_filer_continuity.py        Top 20 filer continuity.
  probe_filer_continuity_top50.py  Top 50 filer continuity.
  probe_reconcile_vanguard.py      Reconciliacion cuantitativa Vanguard.
  top50_output.txt                 Salida del top 50.

=============================================================================
6. CODIGO DEL MOTOR NIPC (5 modulos + tests)
=============================================================================

  src/institutional_accumulation/sec_13f/identity/sec13f_list.py
    Parser SEC Official List 13(f). 24 tests.
  src/institutional_accumulation/sec_13f/identity/security_identity.py
    Resolucion canonical_security (C1-revisada). 30 tests.
  src/institutional_accumulation/aggregation/delta_shares.py
    reported_position_unit + delta_shares. 18 tests.
  src/institutional_accumulation/aggregation/nipc.py
    compute_nipc + coverage pairwise. 19 tests.
  data/mappings/cusip_equivalence.csv
    Schema (cabecera). 0 filas por diseno.

Tests IAE NIPC: 24+30+18+19 = 91 nuevos (total proyecto: 911 + 2 skipped).

=============================================================================
7. CICLO FA-1/FA-2 (IAE Fase A - cerrado y pusheado)
=============================================================================

En origin/main. 10 documentos preservados:

  INSTITUTIONAL_ACCUMULATION_PROPUESTA.md
  INSTITUTIONAL_ACCUMULATION_CONTRATO.md
  INSTITUTIONAL_ACCUMULATION_CONTRATO_ADDENDUM.md
  INSTITUTIONAL_ACCUMULATION_GATE0_INFORME.md
  INSTITUTIONAL_ACCUMULATION_DICTAMEN_GATE0.md
  INSTITUTIONAL_ACCUMULATION_FA21_INFORME.md
  INSTITUTIONAL_ACCUMULATION_FA23_DICTAMEN.md
  INSTITUTIONAL_ACCUMULATION_FA23_DICTAMEN_C.md
  INSTITUTIONAL_ACCUMULATION_FA23_DICTAMEN_CIERRE.md
  INSTITUTIONAL_ACCUMULATION_FA24_DICTAMEN_GATE0.md
  INSTITUTIONAL_ACCUMULATION_FA24_ESPECIFICACION.md
  INSTITUTIONAL_ACCUMULATION_FA24_INFORME.md
  INSTITUTIONAL_ACCUMULATION_FA2_GATE0_DICTAMEN.md
  INSTITUTIONAL_ACCUMULATION_FA2_GATE0_INFORME.md
  INSTITUTIONAL_ACCUMULATION_GATE_FA1_INFORME.md
  INSTITUTIONAL_ACCUMULATION_GATE_FA2_DICTAMEN.md
  INSTITUTIONAL_ACCUMULATION_GATE_FA2_INFORME.md
  INSTITUTIONAL_ACCUMULATION_CUSIP_CURATION_INFORME.md

  Docs Gate 0 Mapping (ciclo NIPC intermedio, ya pusheados):
  INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_INFORME.md
  INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_DICTAMEN.md
  INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_MAPPING_DICTAMEN.md
  INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_FIGI_DICTAMEN.md
  INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_OPENFIGI_DICTAMEN.md
  INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_SEC13F_DICTAMEN.md
  INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_SEC13F_MINIPROBE_DICTAMEN.md
  INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION_DICTAMEN.md

=============================================================================
8. DOCUMENTACION AUTOMATICA (docs/automatica/)
=============================================================================

21 ficheros .md regenerables con py scripts/generate_docs.py.
Cobertura: arquitectura, configuracion, fuentes, regimenes, motores,
momentum, breadth, wyckoff, SLPM, opciones, darkpool, MTE, lideres,
reporte, auditorias, flujo primario ETF, CFTC, N-PORT, backup providers,
instrument registry, QQQ SEC flow.

=============================================================================
9. DOCUMENTOS HISTORICOS RELEVANTES (docs/auditoria/)
=============================================================================

| Fichero                                                     | Tema                              | Estado |
|-------------------------------------------------------------|-----------------------------------|--------|
| FU-021-3A_INFORME_Y_DICTAMEN.md                             | Filtro EOD market_data            | Resuelto |
| FU-021-3B_ANEXO_VOLATILITY_INDEX.md                         | Anexo volatility index            | Resuelto |
| FU-021-3B_INFORME_INDEX_EOD_RATE_YIELD.md                   | Contratos INDEX + RATE_YIELD      | Resuelto |
| FU-021-3C_INFORME_FUTURE_FX.md                              | FUTURE + FX (previo OilPriceAPI)  | Resuelto |
| FU-021-5_ESPECIFICACION_CONTRATO_TEMPORAL.md                | Spec contratos temporales (A)     | Resuelto |
| FU-021-5_ESPECIFICACION_CONTRATO_TEMPORAL_PARTE_B.md        | Spec contratos temporales (B)     | Resuelto |
| FU-021-5_PLAN_IMPLEMENTACION.md                             | Plan implementacion FU-021-5      | Resuelto |
| H1_INFORME_MUTABILIDAD_HISTORICA.md                         | Mutabilidad dataset historico     | WONT FIX |
| H1_DICTAMEN_AUDITOR.md                                      | Dictamen H1                       | WONT FIX |
| E5_INFORME_VIX3M.md                                         | Anomalia yfinance ^VIX3M -> CBOE  | Resuelto |
| DECISION_LSE.md                                             | Decision tickers .L               | Aceptado |
| oos_flows_dictamen.md                                       | Dictamen flows OOS                | Cerrado |
| auditoria_arquitectura.md                                   | Auditoria arquitectura            | Cerrado |

=============================================================================
10. WORKFLOWS GITHUB ACTIONS
=============================================================================

7 workflows en .github/workflows/:

  daily_run.yml                       run diario 04:00 UTC + validacion
  update_macro_manual.yml             FRED 25 series
  update_european_holdings.yml        holdings europeos trimestral
  update_index_holdings.yml           SPY/DIA/QQQ/IWM
  update_qqq_sec_flow.yml             QQQ SEC flow
  update_sec_nport.yml                N-PORT
  update_sector_holdings.yml          holdings sectoriales

=============================================================================
11. DATOS EXTERNOS (fuera del repo)
=============================================================================

D:/13f_probe/processed/
  2025Q4/           (7 parquets, baseline)
  2026Q1/           (7 parquets, target)

D:/13f_probe/official_list_13f/
  13flist_2025Q4.txt  (994,842 bytes)
  13flist_2026q1.txt  (1,995,921 bytes)
  13flist_2025Q4.pdf  (6,011,122 bytes, evidencia auxiliar)

D:/13f_q1_2026/
  01mar2026-31may2026_form13f.zip  (~994 MB)
  extracted/
  infotable_q1_2026.parquet        (~119 MB)

=============================================================================
12. ESTADO DEL CICLO ACTUAL (post TOP 50)
=============================================================================

| Metrica              | Valor                                    |
|----------------------|------------------------------------------|
| HEAD local           | d44bba0 (o hash tras amend)              |
| origin/main          | 9d4a81e                                  |
| Ahead                | 46 commits locales                       |
| Working tree         | limpio                                   |
| Prompt vigente       | v6.36                                    |
| Tests locales        | 911 passed + 2 skipped                   |
| pyflakes             | 0 warnings                               |
| compileall           | OK                                       |
| Push                 | NO (local-first IAE)                     |
| Gate-NIPC.2          | BLOQUEADO por coverage/thresholds        |
| Gate-NIPC.3          | NO AUTORIZADO                            |
| THRESHOLD_1/2        | UNDEFINED                                |
| Filer continuity     | CERRADO como caracterizacion del periodo |
| Reconciliacion NT-HR | NO AUTORIZADA                            |
| C2 (match key)       | VIGENTE / SIN PATCH                      |

=============================================================================
13. SIGUIENTE FASE AUTORIZADA (Q-T50-5)
=============================================================================

  1. Coverage baseline (estado actual del crosswalk interno).
  2. OpenFIGI over unresolved-only (solo CUSIPs sin mapping).
  3. Comparacion A/B (before / after OpenFIGI).
  4. Propuesta THRESHOLD_1 / THRESHOLD_2 con evidencia empirica.
  5. Dictamen auditor sobre thresholds.
  6. Gate-NIPC.2 (implementacion completa + publicacion).

Pendientes adicionales:
  - GHISALLO probe especifico (Q-T50-4, no bloqueante).
  - Reconciliacion NT-HR como linea de investigacion separada.

=============================================================================
14. ORDEN DE LECTURA SUGERIDO
=============================================================================

Para empezar:
  1. PROMPT_MAESTRO.md (normativo).
  2. FOLLOWUPS.md (contexto de deuda).
  3. Este resumen.
  4. TRANSFER_SESION_2026-09-19_POST_TOP50.md.

Si el foco es NIPC (estado actual):
  1. INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_TOP50_DICTAMEN.md
     (dictamen mas reciente, cierre de filer continuity).
  2. INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md (v1.4, activa).
  3. NIPC_COVERAGE_POLICY.md.
  4. Evidencia en evidence/nipc_gate0_probe/.

Si el foco es IAE Fase A (cerrada):
  1. INSTITUTIONAL_ACCUMULATION_GATE_FA2_DICTAMEN.md.
  2. INSTITUTIONAL_ACCUMULATION_FA24_ESPECIFICACION.md.
  3. Resto de docs IAE en orden cronologico.

Si el foco es deuda pendiente:
  1. FOLLOWUPS.md (ultimas entradas).
  2. Seccion 13 del prompt maestro.

=============================================================================
FIN DEL RESUMEN
Version 1.0 (2026-09-19). Referencia: prompt v6.36, HEAD local d44bba0.
