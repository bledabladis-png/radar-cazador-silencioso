# DOCUMENTO DE TRANSMISION - SESION 2026-09-19 (post dictamen TOP 50)

Documento complementario al PROMPT_MAESTRO v6.36.
Proposito: transferir estado real, ciclos cerrados, patrones operativos
y lecciones no obvias al asistente entrante.

No es normativo. Si hay conflicto con el prompt maestro, gana el prompt.

Autor: Ingeniero Supervisor saliente.
Receptor: Asistente entrante (DeepSeek, otro chat).
Fecha: 2026-09-19.

=============================================================================
SECCION A - Estado exacto al cierre
=============================================================================

| Metrica              | Valor                                         |
|----------------------|-----------------------------------------------|
| HEAD local           | a903ab1                                       |
| origin/main          | 9d4a81e                                       |
| Ahead                | 45                                            |
| Working tree         | limpio (sin modificados ni untracked)         |
| Prompt vigente       | v6.35 (a actualizar a v6.36 en este cierre)   |
| Tests locales        | 911 passed + 2 skipped                        |
| pyflakes             | 0 warnings                                    |
| compileall           | OK                                            |
| Push                 | NO (regla local-first IAE)                    |
| Gate-NIPC.2          | BLOQUEADO (por coverage/thresholds)           |
| Gate-NIPC.3          | NO AUTORIZADO                                 |
| THRESHOLD_1          | UNDEFINED                                     |
| THRESHOLD_2          | UNDEFINED                                     |
| Filer continuity     | CERRADO como caracterizacion del periodo      |
| Reconciliacion NT-HR | NO AUTORIZADA                                 |
| C2 (match key)       | VIGENTE / SIN PATCH                           |

Los 45 commits locales viajan juntos al cierre del proximo micro-gate.

Ultimos 15 commits (mas recientes primero):

  a903ab1  docs(iae): FOLLOWUPS - dictamen TOP 50 + spec v1.4
  ff4157e  docs(iae): NIPC spec v1.4 + coverage policy - filer continuity CERRADO
  8b9b4b1  docs(iae): rename NIPC especificacion v1.3 (preparar v1.4)
  cb1724e  docs(iae): preserve TOP 50 auditor dictamen
  6e64a72  docs(iae): FOLLOWUPS - TOP 50 filer continuity probe
  e969b97  docs(iae): TOP 50 filer continuity probe Q-MINI-5 + informe
  37d2879  docs(iae): FOLLOWUPS - dictamen mini-probe + addendum correccion
  64d0ed1  docs(iae): dictamen mini-probe + addendum correccion cuantitativa Vanguard + evidence
  ba1e6d1  docs(iae): FOLLOWUPS - mini-probe filer continuity Q-PROBE-5
  c228f39  docs(iae): mini-probe filer continuity Q-PROBE-5 + evidence
  d75e84c  docs(iae): NIPC spec v1.3 + coverage policy - filer_continuity (seccion 3.15)
  9fb8fc2  docs(iae): rename NIPC especificacion v1.2 (preparar v1.3)
  0d4a425  docs(iae): preserve NIPC probe auditor dictamen (GO mini-probe filer continuity)
  3acfc1c  docs(iae): FOLLOWUPS - probe NIPC end-to-end + hallazgo Vanguard
  90fdec9  docs(iae): informe probe NIPC end-to-end + evidence + hallazgo Vanguard

=============================================================================
SECCION B - Ciclo IAE NIPC (estado detallado)
=============================================================================

## B.1. Modulos implementados (5 modulos + tests)

### identity/sec13f_list.py (S1)
  Parser SEC Official List of Section 13(f) Securities.
  Layout fixed-width 80 chars. Extrae CUSIP (pos 1-9), option_indicator
  (pos 10), issuer_name (11-40), issuer_description (41-67), STATUS
  (68-70). NO interpreta pos 80.
  Resuelve section13f_eligible con 5 estados: NOT_IN_LIST | DELETED |
  ADDED | ACTIVE | CONFLICT.
  24 tests.

### identity/security_identity.py (S2)
  Resolucion de identidad canonica (C1-revisada, dictamen v1.1).
  Dos capas de estado:
    security_resolution_status: CANONICAL | OBSERVED_ONLY | UNRESOLVED
                                | AMBIGUOUS | CONFLICT
    canonical_security_kind:    CANONICAL_FIGI | CANONICAL_EQUIVALENCE
                                | OBSERVED_CUSIP_ONLY | UNRESOLVED
  `ticker:<ticker>` PROHIBIDO como canonical_security.
  Prohibiciones: no heuristica por NAMEOFISSUER, no inferir equivalence
  sin entrada explicita en cusip_equivalence.csv.
  30 tests.

### aggregation/delta_shares.py (S4)
  compute_reported_position_units + compute_delta_shares.
  reported_position_unit = (report_period, filing_manager_cik,
                             canonical_security, discretion_type).
  Match key interperiodo: (filing_manager_cik, canonical_security,
                            discretion_type) SIN report_period.
  SSHPRNAMT entra UNA SOLA VEZ por source line.
  match_status: BOTH | NEW | EXIT | UNRESOLVED_IDENTITY.
  18 tests.

### aggregation/nipc.py (S5)
  compute_nipc + compute_coverage_pairwise + compute_nipc_and_coverage.
  NIPC = sum_i DeltaShares_i (solo observable: BOTH + NEW + EXIT).
  Breakdown por discretion (SOLE + DFND + OTR).
  6 metricas pairwise obligatorias: coverage_previous, coverage_current,
    paired_security_coverage, paired_weighted_share_coverage,
    unmapped_weight_previous, unmapped_weight_current.
  Status: READY | INSUFFICIENT | CONFLICT | AMBIGUOUS | UNRESOLVED |
          TEMPORAL_UNVERIFIED. Actual: INSUFFICIENT (thresholds UNDEFINED).
  19 tests.

### data/mappings/cusip_equivalence.csv (S3)
  Cabecera vacia. Esquema: CUSIP_A, canonical_security, valid_from,
  valid_to, source, reason, verified_by, source_document.
  Poblacion manual curada cuando exista corporate action documentada.

## B.2. Metricas empiricas del motor (probe end-to-end Q4 2025 -> Q1 2026)

Identidad:
  pct_canonical = 0.0206 (Q4) / 0.0204 (Q1).
  ~98% de CUSIPs caen en OBSERVED_ONLY. Gap estructural de crosswalk.

Elegibilidad SEC:
  pct_eligible = 0.4846 (Q4) / 0.5264 (Q1).

NIPC observable:
  scope ALL:      +23,651,586
  scope ELIGIBLE: +52,570,648
  Breakdown: nipc_sole ~ -41.16B, nipc_dfnd ~ +41.28B, neto ~ +52M.
  Cancelacion por reorganizacion Vanguard.

Coverage pairwise:
  ALL:      paired_security_coverage=0.0181, paired_weighted=0.2271
  ELIGIBLE: paired_security_coverage=0.0368, paired_weighted=0.2772

## B.3. Hallazgo estructural Vanguard (CERRADO)

Q4 2025:
  VANGUARD GROUP INC (CIK 0000102909): 13F-HR con holdings.
  VANGUARD FIDUCIARY TRUST CO (CIK 0000933478): 13F-NT (delega al padre).

Q1 2026:
  VANGUARD GROUP INC: 13F-NT con 10 managers declarados en OTHERMANAGER.
  VANGUARD CAPITAL MANAGEMENT LLC (0002100119): 13F-HR/A (35.329B canonical).
  VANGUARD PORTFOLIO MANAGEMENT LLC (0002100121): 13F-HR/A (21.030B canonical).
  VANGUARD FIDUCIARY TRUST CO: 13F-HR directo (4.124B).

Modelo Q4: padre HR + 6 filiales NT.
Modelo Q1: padre NT + 10 filiales HR.

Efecto en match key C2:
  Padre -> EXIT (presente Q4, ausente Q1).
  Filiales -> NEW (ausentes Q4, presentes Q1).
  nipc_sole ~ -41.16B, nipc_dfnd ~ +41.28B (cancelacion).

NO es bug. El motor implementa C2 correctamente.

## B.4. Mini-probes filer continuity (CERRADOS)

TOP 20:
  Union CIKs: 24. CONTINUOUS_FILER=21 (87.5%), FILER_DISCONTINUITY=3
  (12.5%), todos Vanguard.

TOP 50:
  Union CIKs: 54. CONTINUOUS_FILER=50 (92.6%), FILER_DISCONTINUITY=4
  (7.4%), todos Vanguard.

Los 4 FILER_DISCONTINUITY (todos Vanguard):
  0000102909 VANGUARD GROUP INC (padre)
  0000933478 VANGUARD FIDUCIARY TRUST CO
  0002100119 VANGUARD CAPITAL MGMT LLC
  0002100121 VANGUARD PORTFOLIO MGMT LLC

NT_TO_HR observados: 2.
  Padre Vanguard NT Q1 -> 10 targets en OTHERMANAGER.
  Fiduciary Trust NT Q4 -> target padre.

Conclusion congelada:
  Concentracion en una estructura corporativa (Vanguard) durante este
  periodo. NO se extrapola al universo. NO es threshold.

## B.5. Correccion cuantitativa (addendum)

Inconsistencia detectada por auditor: 35.329B vs 70.236B del CIK 0002100119.
Causa: informe original sumaba SSHPRNAMT desde INFOTABLE.parquet filtrado
por periodo (raw), no desde canonical_snapshot. HR original (001306) fue
SUPERSEDED por RESTATEMENT posterior (001311).
Ratio 1.988 (doble conteo).

Cifras corregidas:
  Vanguard Capital Q1 canonical: 35,329,105,562
  Vanguard Portfolio Q1 canonical: 21,030,554,644
  Filiales Q1 canonical: 56,359,660,206 (antes: 91,267,483,611 NO UTILIZABLE)

REGLA CONGELADA:
  Todo agregado SSHPRNAMT debe calcularse sobre canonical_snapshot
  producido por apply_amendments. PROHIBIDO sumar desde INFOTABLE.parquet
  filtrado por periodo cuando existan amendments RESTATEMENT.

=============================================================================
SECCION C - Decisiones del auditor (dictamenes vinculantes)
=============================================================================

## C.1. Dictamen Gate-NIPC.1 v1.1 (2026-09-19)

C1-revisada: canonical_security no es ticker.
C4-revisada: identity vs metadata temporal.
PASS CONDICIONADO.

## C.2. Dictamen probe end-to-end (2026-09-19)

GO condicionado. Mini-probe filer continuity autorizado.

## C.3. Dictamen mini-probe TOP 20 (2026-09-19)

GO CONDICIONADO.
  Q-MINI-1 GO/CERRADO: Vanguard = limitacion conocida.
  Q-MINI-2 GO/CERRADO: separar filer_status de nt_to_hr_relation_observed.
  Q-MINI-3 GO DIAGNOSTICO: GHISALLO pendiente.
  Q-MINI-4 CERRADO: NT->HR evidencia documental, no economica.
  Q-MINI-5 GO TOP 50.
  Hallazgo cuantitativo obligatorio: reconciliar 35.329B vs 70.236B.

## C.4. Dictamen TOP 50 (2026-09-19)

FILER CONTINUITY CERRADO como caracterizacion del periodo.
  Q-T50-1 CERRADO: no ampliar a top 100/universo.
  Q-T50-2 CERRADO: filer_status por presencia documental (NO SSHPRNAMT>0).
                   Nueva dimension position_mass_status.
  Q-T50-3 CERRADO: NT->HR evidencia documental, no economica.
  Q-T50-4 GO DIAGNOSTICO: GHISALLO.
  Q-T50-5 GO: siguiente fase = coverage baseline + OpenFIGI.
  Addendum cuantitativo: PASS.

=============================================================================
SECCION D - Estado de gates y bloqueos
=============================================================================

  Q-PROBE-5 TOP 20:                     PASS
  Q-T50 TOP 50:                         PASS
  Vanguard concentration:               CONFIRMADA EN TOP 50
  NT -> OTHERMANAGER -> HR:             CONFIRMADO
  Addendum cuantitativo Vanguard:       PASS
  Regla canonical_snapshot:             CONGELADA
  C2:                                   VIGENTE / SIN PATCH
  Reconciliacion economica NT-HR:       NO AUTORIZADA
  GHISALLO:                             PROBE DIAGNOSTICO autorizado
  THRESHOLD_1:                          UNDEFINED
  THRESHOLD_2:                          UNDEFINED
  Gate-NIPC.2:                          BLOQUEADO por coverage
  Gate-NIPC.3:                          NO AUTORIZADO
  Coverage A/B:                         SIGUIENTE FASE

=============================================================================
SECCION E - Proximos pasos autorizados
=============================================================================

Siguiente fase autorizada por Q-T50-5:

  1. Coverage baseline (estado actual del crosswalk interno).
  2. OpenFIGI over unresolved-only (solo los CUSIPs sin mapping).
  3. Comparacion A/B (before / after OpenFIGI).
  4. Propuesta THRESHOLD_1 / THRESHOLD_2 con evidencia empirica.
  5. Dictamen auditor sobre thresholds.
  6. Gate-NIPC.2 (implementacion completa + publicacion).

Pendientes adicionales:
  - GHISALLO probe especifico (Q-T50-4, no bloqueante).
  - Reconciliacion NT-HR como linea de investigacion separada
    (NO autorizada como productiva).

=============================================================================
SECCION F - Documentos clave del ciclo
=============================================================================

## F.1. Spec NIPC (activa)

  docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md (v1.4)

Versiones preservadas:
  _v1.0.md, _v1.1.md, _v1.2.md, _v1.3.md

## F.2. Coverage policy

  docs/auditoria/NIPC_COVERAGE_POLICY.md

## F.3. Dictamenes del auditor (5)

  INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION_V11_DICTAMEN.md
  INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_PROBE_DICTAMEN.md
  INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_MINIPROBE_DICTAMEN.md
  INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_TOP50_DICTAMEN.md

## F.4. Informes del ingeniero (3)

  INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_PROBE_INFORME.md
  INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_MINIPROBE_INFORME.md
  INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_MINIPROBE_ADDENDUM.md
  INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_TOP50_INFORME.md

## F.5. Evidence (probes reproducibles)

  docs/auditoria/evidence/nipc_gate0_probe/
    probe_nipc_e2e.py
    probe_filer_continuity.py
    probe_filer_continuity_top50.py
    probe_reconcile_vanguard.py
    top50_output.txt
    README.md
    HASHES.txt

=============================================================================
SECCION G - Trampas recurrentes (actualizadas)
=============================================================================

## G.1. Notepad escapa markdown y dobla newlines

Notepad en Windows:
  - Escapa caracteres markdown al pegar: `# -> \#`, `* -> \*`,
    `_ -> \_`, `+ -> \+`, etc.
  - Dobla cada newline (cada `\n` se convierte en `\n\n`).
  - Convierte espacios a `&#x20;` en algunos contextos.

Patron de normalizacion obligatorio:
  1. CRLF -> LF.
  2. Decode HTML entities `&#...;` y `&#x...;`.
  3. Colapsar `\n\n` -> `\n`.
  4. Strip markdown escapes `\([#\-*_...])`.
  5. Transliterar unicode decorativo a ASCII.
  6. Verificar no-ASCII residual.

## G.2. [System.IO.File]::ReadAllBytes/WriteAllText usan CWD del proceso .NET

NO el CWD del shell PowerShell. Usar SIEMPRE path absoluto.

## G.3. Here-strings PowerShell >20 lineas o >5 $ se corrompen

Escribir a script Python temporal con [System.IO.File]::WriteAllText,
luego ejecutar con py.

## G.4. Backticks en here-string PowerShell se corrompen

Al escribir fences Markdown (triple-backtick), usar placeholder y
reemplazar con chr(96) antes de escribir.

## G.5. Here-strings con docstrings triples anidados

NO meter triple-double-quote dentro de un here-string doble-quote.
Alternativa: listar PARTS en Python y concatenar.

## G.6. Prohibido sumar SSHPRNAMT desde INFOTABLE crudo

REGLA CONGELADA (addendum Vanguard):
  Todo agregado de SSHPRNAMT debe calcularse sobre canonical_snapshot
  producido por apply_amendments.
  PROHIBIDO sumar desde INFOTABLE.parquet filtrado por periodo cuando
  existan amendments RESTATEMENT.

## G.7. Match key C2 sin report_period

La clave de matching interperiodo es:
  (filing_manager_cik, canonical_security, discretion_type)
SIN report_period. Incluirlo produce producto cartesiano incorrecto.

## G.8. canonical_security no es ticker

NO usar `ticker:<ticker>` como canonical_security definitivo.
Usar figi:<shareClassFIGI> (CANONICAL_FIGI) o equity:<canonical_id>
(CANONICAL_EQUIVALENCE). Sin evidencia, canonical_security = NULL.

## G.9. HTML entities de Notepad

Al pegar contenido desde el chat al Notepad, aparecen HTML entities:
  `&#x20;` (espacio), `&#x2F;` (slash), etc.
  Decodificar siempre: `&#x...; -> chr(int(hex, 16))`.

## G.10. Regla usuario decide cuando parar

El asistente NO cierra sesion por fatiga. El usuario decide.

=============================================================================
SECCION H - Comandos de arranque (verificacion)
=============================================================================

cd D:\Macro_Sectorial
git status -sb
git log --oneline -5
py -m compileall . -q
py -m pyflakes . 2>&1
py -m pytest tests/ validation/ -q --tb=short

Esperado:
  - HEAD = a903ab1
  - ahead 45
  - working tree limpio
  - 911 passed + 2 skipped
  - pyflakes silencio

=============================================================================
SECCION I - Lo que NO debes hacer
=============================================================================

  NO pushear los 45 commits hasta cierre de Gate-NIPC.3.
  NO modificar C2 (match key) por reorg de filer.
  NO convertir filer_continuity_pct en threshold.
  NO inferir economic_owner por parent/child.
  NO reconciliar NT <-> HR productivamente.
  NO sumar SSHPRNAMT desde INFOTABLE.parquet crudo.
  NO usar ticker:<ticker> como canonical_security.
  NO fijar THRESHOLD_1 ni THRESHOLD_2 antes del dictamen.
  NO ampliar top 50 a top 100 o universo sin autorizacion.
  NO asumir validez historica de OpenFIGI actual.
  NO usar "ultimo registro gana" en STATUS de la Official List.
  NO interpretar pos 80 de la Official List.
  NO commitear API keys.
  NO usar py -c con comillas dobles anidadas (usar script temporal).
  NO usar [System.IO.File]::ReadAllBytes/WriteAllText con path relativo.
  NO usar Notepad para contenido largo; usar script Python con
    [System.IO.File]::WriteAllText.

=============================================================================
SECCION J - Confirmacion esperada
=============================================================================

Tras leer el prompt maestro v6.36 + este documento, el asistente
entrante debe:

  1. Confirmar asimilacion del contexto.
  2. Reconocer HEAD a903ab1, ahead 45, tests 911+2, ciclo IAE NIPC
     hasta dictamen TOP 50 (filer continuity CERRADO).
  3. Reconocer que la siguiente fase autorizada es coverage baseline
     + OpenFIGI over unresolved-only.
  4. Reconocer que Gate-NIPC.2 sigue BLOQUEADO por coverage/thresholds.
  5. No proponer tareas nuevas sin antes preguntar que hacemos.
  6. Respetar regla "usuario decide cuando parar".

=============================================================================
FIN DEL DOCUMENTO DE TRANSMISION
Version 1.0 (2026-09-19). Referencia: prompt v6.35, HEAD local a903ab1.
