# DOCUMENTO DE TRANSMISION - SESION 2026-09-19 (post dictamen F2.2 NO GO)

Documento complementario al PROMPT_MAESTRO v6.38.
Proposito: transferir estado real, ciclos abiertos, decisiones del auditor
y lecciones operativas al asistente entrante.

No es normativo. Si hay conflicto con el prompt maestro, gana el prompt.

Autor: Ingeniero Supervisor saliente.
Receptor: Asistente entrante (DeepSeek, otro chat).
Fecha: 2026-09-19.

**Nota semantica sobre HEAD/ahead:** los valores de la tabla de la
Seccion A reflejan el estado al redactar esta actualizacion. Un commit
posterior (incluido el propio commit que introduce este texto) rehashea
el HEAD y avanza `ahead`. Verificar siempre con `git log --oneline -5`
y `git status -sb`. El desfase es por diseno, no un error.

=============================================================================
SECCION A - Estado exacto al cierre
=============================================================================

| Metrica              | Valor                                         |
|----------------------|-----------------------------------------------|
| HEAD local           | da5254c (al redactar este documento)          |
| origin/main          | 9d4a81e                                       |
| Ahead                | 56                                            |
| Working tree         | limpio (sin modificados ni untracked)         |
| Prompt vigente       | v6.38                                         |
| Tests locales        | 911 passed + 2 skipped                        |
| pyflakes             | 0 warnings                                    |
| compileall           | OK                                            |
| Push                 | NO (regla local-first IAE)                    |
| Gate-NIPC.2          | BLOQUEADO (por coverage/thresholds)           |
| Gate-NIPC.3          | NO AUTORIZADO                                 |
| THRESHOLD_1          | UNDEFINED                                     |
| THRESHOLD_2          | UNDEFINED                                     |
| OpenFIGI masivo      | NO AUTORIZADO                                 |
| RADAR_TARGET_REGISTRY| NOT AVAILABLE (Camino B declarado)            |
| Filer continuity     | CERRADO como caracterizacion del periodo      |
| Reconciliacion NT-HR | NO AUTORIZADA                                 |
| C2 (match key)       | VIGENTE / SIN PATCH                           |
| Policy v1.0 intacta  | SI (hash 57f2d01f...)                         |

Los 56 commits locales viajan juntos al cierre del proximo micro-gate.

Ultimos 8 commits de la sesion (mas recientes primero):

  da5254c  docs(iae): prompt maestro v6.38 - micro-gate Coverage Contract Normalization
  4659ce1  docs(iae): F2.1-bis fuentes identidad - Camino B RADAR_TARGET_REGISTRY NOT AVAILABLE
  d4a926e  docs(iae): F2.2 propuesta v1.1 NIPC_COVERAGE_POLICY - circularidad + TARGET/RESOLVED/PAIRED
  eda67ef  docs(iae): Gate 0 documental contrato cobertura NIPC - inventario 3 conceptos
  02b077c  docs(iae): preservar dictamen auditor coverage baseline - ISSUE ABIERTO circularidad
  5d4fe53  docs(iae): restaurar top50_output.txt + actualizar HASHES + README del probe
  62db3eb  docs(iae): prompt maestro v6.37 - Q-CUR + coverage baseline + ciclos huerfanos
  e9fd830  docs(iae): coverage baseline Q4->Q1 + evidencia + informe

Hashes SHA-256 de los ficheros clave (al redactar):

  PROMPT_MAESTRO.md                              0c6115d244cfd1027a6d3aaca3485c2cc2697cdffb19d082df3943d242513a11
  FOLLOWUPS.md                                   9177e0a4aceeaa6a0fccfa4f9ccfe808892897b61688612638ce7f15960b67b7
  NIPC_COVERAGE_POLICY.md                        57f2d01feb68d8916dfa4fa0451224c2ee3dda7e4631a7863e671257b3e91f06
  NIPC_COVERAGE_POLICY_V11_PROPUESTA.md          0d345ee9399d357b9901ef0a41d0cb9868b0b42ceec2992643c26a182cf38dc3
  ..._GATE0_COVERAGE_BASELINE_INFORME.md         75dbb896707d8aaec4f2795fa9abd2b23677f5b247350c51a9443c273b86b9b6
  ..._GATE0_COVERAGE_BASELINE_DICTAMEN.md        2997c27313f80ce8c4ed1d5e5ef804da6374e8c6d412e39489396d6a5eb1eedc
  ..._GATE0_COVERAGE_CONTRACT_INVENTARIO.md      7b8aa345cbcb8e5cf710a60535650db7e0824b7e55dc88d78efa77616ac92f94
  ..._F21BIS_FUENTES_IDENTIDAD.md                08793b1737c48a07956177c451a03c21d1d454b29a505deeae3abd49eb267132

=============================================================================
SECCION B - Contexto de la sesion
=============================================================================

Al inicio de la sesion, el asistente recibio:

  1. PROMPT_MAESTRO v6.36 (via archivo adjunto).
  2. TRANSFER_SESION_2026-09-19_POST_TOP50.md (contexto previo).
  3. RESUMEN_DOCUMENTOS_IAE.md.

Punto de partida: 45-48 commits locales ahead de origin/main, ciclo
NIPC cerrado como caracterizacion, Gate-NIPC.2 bloqueado por thresholds
UNDEFINED.

Trabajo realizado en esta sesion (orden cronologico):

  a) Verificacion de estado del sistema (5 hallazgos iniciales).
  b) Ciclo "coverage baseline" (Fase A del dictamen TOP 50).
  c) Actualizacion prompt maestro v6.36 -> v6.37 (ciclos Q-CUR + coverage
     baseline).
  d) Preservacion top50_output.txt + HASHES + README del probe.
  e) Preservacion dictamen auditor coverage baseline.
  f) Micro-gate Coverage Contract Normalization (F2.1, F2.1-bis, F2.2
     NO GO).
  g) Actualizacion prompt maestro v6.37 -> v6.38.

=============================================================================
SECCION C - Ciclo "coverage baseline" CERRADO
=============================================================================

Origen: Q-T50-5 del dictamen TOP 50.

Ejecutado:

  - Probe deterministico `probe_coverage_baseline.py` con 4 universos
    anidados: RAW_13F_SH_NULL / ELIGIBLE_SEC / TECHNICAL_RADAR /
    OPERATIONAL_EQUITY.
  - 6 metricas pairwise obligatorias sobre cada uno.
  - Fallout desglosado ELIGIBLE_SEC -> TECHNICAL_RADAR.

Resultados (Q4 2025 -> Q1 2026):

  RAW_13F_SH_NULL:    coverage 2.06% / 2.04%,  paired 1.81% / 22.71%
  ELIGIBLE_SEC:       coverage 4.24% / 3.86%,  paired 3.68% / 27.72%
  TECHNICAL_RADAR:    coverage 100% / 100%,    paired 99.09% / 98.64%
  OPERATIONAL_EQUITY: coverage 100% / 100%,    paired 99.09% / 98.64%

Fallout (Q1 2026): 20.47% IN_RADAR, 13.40% CANONICAL_TICKER_OUTSIDE_RADAR,
66.13% OBSERVED_ONLY_NO_CANONICAL.

Artefactos (commit e9fd830):

  docs/auditoria/evidence/nipc_gate0_baseline/README.md
  docs/auditoria/evidence/nipc_gate0_baseline/probe_coverage_baseline.py
  docs/auditoria/evidence/nipc_gate0_baseline/baseline_output.txt
  docs/auditoria/evidence/nipc_gate0_baseline/HASHES.txt
  docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_BASELINE_INFORME.md

Incidente resuelto: `baseline_output.txt` se genero con CRLF. `.gitattributes`
normaliza a LF. Correccion: normalizar + recalcular hash + `git commit --amend`.

Hallazgo del auditor (dictamen 02b077c):

  "La actual definicion contractual de `operational_universe` incorpora
   `ticker_mapped` antes de evaluar la cobertura."

  Resultado: coverage = 1.0000 por construccion sobre TECHNICAL_RADAR.
  No es cobertura del universo objetivo; es estabilidad interperiodo del
  subconjunto ya mapeado.

=============================================================================
SECCION D - Ciclo "micro-gate Coverage Contract Normalization" EN CURSO
=============================================================================

Autorizado por el dictamen del auditor (02b077c, seccion 9).

Cadena de fases:

  F2.1     Gate 0 documental: inventario de los 3 conceptos en la policy v1.0.
  F2.1-bis Fuentes de identidad: verificar existencia de RADAR_TARGET_REGISTRY
           independiente del crosswalk evaluado.
  F2.2     Propuesta v1.1 (borrador).
  F2.3     Dictamen del auditor sobre la propuesta.
  F2.4     Aplicacion de v1.1 (solo si F2.3 aprueba).

Estado por fase:

  F2.1 PASS (commit eda67ef).

    Inventario linea-por-linea de TARGET / RESOLVED / PAIRED en
    `NIPC_COVERAGE_POLICY.md` v1.0.

    Resultado:
      TARGET: 0 ocurrencias literales.
      RESOLVED: 0 ocurrencias literales.
      PAIRED: 13 ocurrencias, definicion vaga sin universo base.

  F2.1-bis PASS (commit 4659ce1). Camino B declarado.

    Fuentes inventariadas y verificadas:
      - `data/mappings/isin_ticker_map.csv`: 49 tickers, 0 USA.
      - `data/etf_holdings.csv`: 220/242 USA, ES el crosswalk evaluado.
      - `data/amundi_yahoo_mapping.csv`: 0 USA.
      - `data/index_holdings.csv`: sin CUSIP.
      - `data/mappings/cusip_equivalence.csv`: 0 filas.
      - `data/mappings/cusip_ticker_exceptions.csv`: 3 filas, parte del crosswalk.

    Ninguna fuente cumple (a) CUSIP/FIGI + (b) radar USA + (c) independencia
    del crosswalk evaluado.

    Declaracion: `TARGET_CUSIP_REGISTRY = NOT AVAILABLE (2026-09-19)`.

    Hallazgo colateral confirmado: el 90.91% (220/242) que la policy v1.0
    L115 cita como `mapping_coverage` sale exactamente de `etf_holdings.csv`.

  F2.2 NO GO CON CAMBIOS OBLIGATORIOS (commit d4a926e + dictamen).

    La propuesta v1.1 separaba TARGET/RESOLVED/PAIRED conceptualmente bien
    pero definia:

      TARGET_UNIVERSE = radar_equities ^ section13f_eligible ^ EQUITY

    con claves incompatibles: {ticker} ^ {CUSIP} ^ {ticker via
    get_instrument_class}. Intersectar requiere ejecutar el resolver que
    se pretende medir. Reproduce la circularidad en forma ontologica.

    Dictamen: "La arquitectura conceptual es GO. La definicion
    matematica/operativa de TARGET es NO GO."

  F2.3 PENDIENTE EXTERNO. No la puede ejecutar el asistente.

  F2.4 NO AUTORIZADA.

8 cambios obligatorios para v1.2 (dictamen F2.2):

  1. RADAR_TARGET_REGISTRY con identidad estable independiente del resolver
     evaluado.
  2. TARGET_UNIVERSE sobre clave comun a CUSIP/13F.
  3. RESOLVED_UNIVERSE con enums v1.3: CANONICAL_FIGI / CANONICAL_EQUIVALENCE
     (no estado generico "CANONICAL").
  4. PAIRED_UNIVERSE: `canonical_security` comun, sin "or
     observed_security_key".
  5. PAIRED_WEIGHTED_SHARE_COVERAGE: convencion Q4/Q1 inequivoca.
  6. Resolver contradiccion filer continuity (seccion 10.bis de la policy:
     dice "NO bloquea READY" y a la vez lista como condicion (3) de READY).
  7. Corregir referencia "Prompt Maestro v6.35" en policy L225 (vigente v6.38).
  8. Trazabilidad explicita del cambio semantico v1.0 -> v1.1 -> v1.2.

Reordenacion propuesta del pipeline (a dictamen del auditor, todavia NO
autorizada):

  v1.0 (descartado):
    baseline -> OpenFIGI -> thresholds

  v1.1 (F2.2, NO GO):
    baseline -> policy v1.1 -> OpenFIGI -> thresholds

  v1.2 (propuesta):
    baseline
      -> RADAR_TARGET_REGISTRY (via OpenFIGI como fuente)
      -> policy v1.2
      -> medicion sobre TARGET independiente
      -> thresholds

  La diferencia clave: OpenFIGI pasa de ser mejora posterior de coverage a
  ser fuente primaria para construir el target registry, ANTES de fijar
  thresholds.

=============================================================================
SECCION E - Documentos clave del ciclo actual
=============================================================================

## E.1. Normativos vigentes

  docs/auditoria/PROMPT_MAESTRO.md       v6.38.
  docs/auditoria/FOLLOWUPS.md            Registro vivo (ultima entrada:
                                          micro-gate Coverage Contract
                                          Normalization).

## E.2. Ciclo coverage baseline (cerrado)

  docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_BASELINE_INFORME.md
  docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_BASELINE_DICTAMEN.md
  docs/auditoria/evidence/nipc_gate0_baseline/ (README + probe + output + HASHES)

## E.3. Micro-gate Coverage Contract Normalization (en curso)

  docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_CONTRACT_INVENTARIO.md
    Inventario F2.1 (PASS).

  docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_F21BIS_FUENTES_IDENTIDAD.md
    Inventario F2.1-bis (PASS, Camino B).

  docs/auditoria/NIPC_COVERAGE_POLICY_V11_PROPUESTA.md
    Propuesta F2.2 (NO GO CON CAMBIOS OBLIGATORIOS).

  (Pendiente) NIPC_COVERAGE_POLICY_V12_PROPUESTA.md
    A redactar con Camino B declarado + los 8 cambios obligatorios.

## E.4. Policy auditada (intacta)

  docs/auditoria/NIPC_COVERAGE_POLICY.md
    v1.0. Hash 57f2d01f... Sin modificar desde la sesion anterior.

## E.5. Evidencia top 50 (restaurada)

  docs/auditoria/evidence/nipc_gate0_probe/
    Probe + top50_output.txt + HASHES (7 ficheros tracked).

=============================================================================
SECCION F - Reglas operativas criticas
=============================================================================

  - Local-first IAE: NO push a main hasta cierre de Gate-NIPC.3 (que sigue
    BLOQUEADO). Los 56 commits quedan en local.

  - No tocar `NIPC_COVERAGE_POLICY.md` v1.0 hasta dictamen favorable (F2.3)
    sobre la propuesta v1.2.

  - No tocar el motor NIPC: `delta_shares.py`, `nipc.py`,
    `security_identity.py`, C2 (match key sin report_period).

  - No ejecutar OpenFIGI masivo. El dictamen F2.2 explicitamente:
    "NO ejecutar OpenFIGI masivo todavia". Puede entrar como fuente para
    construir el target registry (reordenacion propuesta), pero solo con
    autorizacion especifica del auditor.

  - No inventar RADAR_TARGET_REGISTRY. Declarado NOT AVAILABLE.

  - No fijar THRESHOLD_1 / THRESHOLD_2. UNDEFINED hasta evidencia real
    sobre el nuevo denominador.

  - Baseline evidencia (evidence/nipc_gate0_baseline/) NO se reescribe. El
    auditor: "Una futura v1.1 de la policy debe producir una nueva medicion,
    no reinterpretar retroactivamente el resultado."

  - Regla canonical_snapshot: agregados SSHPRNAMT desde
    `apply_amendments().canonical_snapshot`. PROHIBIDO sumar desde
    INFOTABLE.parquet crudo.

=============================================================================
SECCION G - Trampas recurrentes (verificadas esta sesion)
=============================================================================

  G.1. Here-strings PowerShell grandes se corrompen silenciosamente.

    Regla: si tiene >20 lineas o >5 `$`, escribir a fichero temporal con
    `[System.IO.File]::WriteAllText`, luego ejecutar con `py`.

  G.2. `[System.IO.File]::ReadAllBytes/WriteAllText` usan CWD del proceso
    .NET, no del shell. Usar SIEMPRE path absoluto.

  G.3. `[char]96 * 3` NO funciona en PowerShell (no multiplica chars por int).
    Para triple-backtick, usar `[string][char]96 + [string][char]96 +
    [string][char]96` o escribir con Python (`chr(96)*3`).

  G.4. `Measure-Object -Line` cuenta solo lineas NO vacias. Para contar
    lineas logicas, usar `$arr.Count` o Python `t.count(chr(10)) + 1`.
    Discrepancia tipica: 1732 (reales) vs 1176 (no vacias).

  G.5. Redirect `>` de PowerShell a veces no captura stdout de `py.exe`
    (buffering cruzado). Usar `Start-Process -Wait -RedirectStandardOutput`
    con `-u` (unbuffered) o `cmd /c "py ... > file 2>&1"`.

  G.6. `.gitattributes` fuerza `*.txt eol=lf` y `*.md eol=lf`. Un fichero
    escrito con CRLF en Windows se normaliza a LF al commitear. Si se
    registra el hash SHA-256 del contenido CRLF, el hash no coincidira con
    lo que git almacena. Normalizar a LF ANTES de hashear.

  G.7. `Select-String -SimpleMatch` desactiva regex: `|` se trata como
    literal. No usar con patrones que contengan `|`.

  G.8. `Get-Content $f | Measure-Object -Line` sobre un fichero con BOM
    incluye la linea BOM. Verificar EOL con `[System.IO.File]::ReadAllBytes`
    + regex `\r\n` y `(?<!\r)\n`.

  G.9. Contenido con `$` o backticks en here-strings `@"..."@` de
    PowerShell se expande o escapa silenciosamente. Usar `@'...'@`
    (single-quote) si no se necesita expansion.

  G.10. Regla del usuario: "usuario decide cuando parar". El asistente NO
    cierra sesion por fatiga.

=============================================================================
SECCION H - Comandos de arranque (verificacion al inicio de sesion)
=============================================================================

  Set-Location D:\Macro_Sectorial
  git status -sb
  git log --oneline -8
  git rev-list --count origin/main..HEAD

  py -m compileall . -q
  py -m pyflakes . 2>&1
  py -m pytest tests/ validation/ -q --tb=short

Esperado al arrancar la nueva sesion:

  - HEAD = da5254c (o hash del commit del transfer que se acaba de crear).
  - ahead 56 (o 57 tras el commit del transfer).
  - Working tree limpio.
  - 911 passed + 2 skipped.
  - pyflakes silencio.

Verificacion de la policy intacta:

  (Get-FileHash docs\auditoria\NIPC_COVERAGE_POLICY.md -Algorithm SHA256).Hash.ToLower()
  # Esperado: 57f2d01feb68d8916dfa4fa0451224c2ee3dda7e4631a7863e671257b3e91f06

=============================================================================
SECCION I - Lo que NO debes hacer (asistente entrante)
=============================================================================

  NO pushear los 56 commits hasta cierre de Gate-NIPC.3.

  NO modificar `NIPC_COVERAGE_POLICY.md` v1.0. Solo se sustituye por v1.2/v1.1
  tras dictamen favorable (F2.3).

  NO modificar C2 (match key sin report_period) por reorganizaciones de filer.

  NO convertir filer_continuity_pct en threshold.

  NO inferir economic_owner por parent/child.

  NO reconciliar NT <-> HR productivamente.

  NO sumar SSHPRNAMT desde INFOTABLE.parquet crudo. Usar canonical_snapshot.

  NO usar `ticker:<ticker>` como canonical_security.

  NO fijar THRESHOLD_1 ni THRESHOLD_2 antes del dictamen.

  NO ejecutar OpenFIGI masivo. Solo con autorizacion especifica del auditor.

  NO construir RADAR_TARGET_REGISTRY con el crosswalk evaluado. Reproduce
  la circularidad.

  NO reescribir la evidencia del baseline (evidence/nipc_gate0_baseline/).

  NO tocar `security_identity.py`, `delta_shares.py`, `nipc.py` en esta fase.

  NO usar Notepad para contenido largo. Usar script Python con
  `[System.IO.File]::WriteAllText`.

  NO usar `py -c` con comillas dobles anidadas (usar script temporal).

  NO usar `[System.IO.File]::ReadAllBytes/WriteAllText` con path relativo
  (usa CWD del proceso .NET).

=============================================================================
SECCION J - Confirmacion esperada del asistente entrante
=============================================================================

Tras leer el prompt maestro v6.38 + este documento, el asistente
entrante debe:

  1. Confirmar asimilacion del contexto.

  2. Reconocer:
     - HEAD da5254c, ahead 56, tests 911+2.
     - Prompt v6.38 vigente.
     - Policy v1.0 intacta (hash 57f2d01f...).
     - Ciclo coverage baseline CERRADO.
     - Micro-gate Coverage Contract Normalization EN CURSO (F2.1 PASS,
       F2.1-bis PASS, F2.2 NO GO, F2.3 pendiente, F2.4 bloqueada).
     - RADAR_TARGET_REGISTRY = NOT AVAILABLE (Camino B).

  3. Reconocer que la siguiente fase autorizada es F2.2-v2: nueva propuesta
     `NIPC_COVERAGE_POLICY_V12_PROPUESTA.md` con Camino B declarado + los 8
     cambios obligatorios del dictamen F2.2.

  4. Reconocer que Gate-NIPC.2 sigue BLOQUEADO y OpenFIGI NO AUTORIZADO.

  5. No proponer tareas nuevas sin antes preguntar "Que hacemos?".

  6. Respetar regla "usuario decide cuando parar".

  7. Aplicar regla "nada se interpreta, todo se verifica": antes de actuar,
     comprobar el estado real con `git log` + `git status` + tests.

=============================================================================
FIN DEL DOCUMENTO DE TRANSMISION
Version 1.0 (2026-09-19). Referencia: prompt v6.38, HEAD local da5254c.
=============================================================================