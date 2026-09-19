# DOCUMENTO DE TRANSMISION - SESION 2026-09-20 (post fixes mecanicos + F2.4 pendiente)

Documento complementario al PROMPT_MAESTRO v6.42.
Proposito: transferir estado real, ciclos abiertos, decisiones del
auditor y lecciones operativas al asistente entrante.

No es normativo. Si hay conflicto con el prompt maestro, gana el prompt.

Autor: Ingeniero Supervisor saliente.
Receptor: Asistente entrante.
Fecha: 2026-09-20.

Nota semantica sobre HEAD/ahead: los valores de la Seccion A reflejan
el estado al redactar esta actualizacion. Un commit posterior (incluido
el propio commit que introduce este texto) rehashea HEAD y avanza
ahead. Verificar siempre con git log --oneline -5 y git status -sb.

=============================================================================
SECCION A - Estado exacto al cierre
=============================================================================

| Metrica              | Valor                                         |
|----------------------|-----------------------------------------------|
| HEAD local           | 3c59d41 (prompt v6.42 al redactar)            |
| origin/main          | 9d4a81e                                       |
| Ahead                | 97 (mas 1 si se commitea este transfer)       |
| Working tree         | limpio                                        |
| Prompt vigente       | v6.42                                         |
| Tests locales        | 946 passed + 2 skipped                        |
| pyflakes             | 0 warnings                                    |
| compileall           | OK                                            |
| Push                 | NO (regla local-first IAE)                    |
| Gate-NIPC.2          | BLOQUEADO (THRESHOLD_1/2 UNDEFINED)           |
| Gate-NIPC.3          | NO AUTORIZADO                                 |
| THRESHOLD_1          | UNDEFINED                                     |
| THRESHOLD_2          | UNDEFINED                                     |
| OpenFIGI masivo      | NO AUTORIZADO (24.838 CUSIPs)                 |
| OpenFIGI piloto      | GO (top 500 + top 2000 ejecutados)            |
| F2.4                 | PENDIENTE EXTERNO (informe entregado)         |
| Policy v1.0 intacta  | SI (hash 57f2d01f...)                         |
| Policy v1.1          | INTACTA (historico, NO GO F2.2)               |
| Policy v1.2          | INTACTA (historico, NO GO F2.3)               |
| Policy v1.3          | BORRADOR (hash 933fcb2e...)                   |
| Contrato v1          | PROPUESTO (hash c02d202f...)                  |

Ultimos 8 commits de la sesion (mas recientes primero):

  3c59d41  docs(iae): reconciliacion final de hashes en cadena documental
  85f9401  docs(iae): correcciones post-revision cruzada en informe post-fixes
  d462f72  docs(iae): materializar dictamen revision estructural 2026-09-20
  501126b  docs(iae): Tanda 3+4-bis - hash contrato abreviado en informes
  cc6a402  docs(iae): Tanda 3+4 - hash contrato actualizado en informes
  373c5cd  docs(iae): Tanda 2 - policy v1.3 precedencia transitoria + hash contrato
  367c58c  docs(iae): Tanda 1 - contrato v1, estados PROPUESTO + P38 denominador TARGET_PAIRWISE
  ff81691  docs(iae): informe de estado post-fixes al auditor F2.4

Los 6 fixes mecanicos (commits ordenados):

  0f1e5a4  fix(sec13f_list): P51 - linea corta preservada como anomalia
  714457e  fix(delta_shares): P32 - contador de descartes silenciosos
  eb687c4  fix(delta_shares): P14-BIS - fillna condicionado al _merge del JOIN
  e31d82e  fix(nipc): P31 - exponer nipc_total_available + docstring _sum_delta
  54478c3  fix(manifest): P18/P26 - project_root por env var + fallback derivado
  1575d17  fix(amendments): P70 - guarda explicita invariante HR_PLUS_RESTATEMENT

=============================================================================
SECCION B - Contexto de la sesion
=============================================================================

Al inicio de la sesion, el asistente recibio:

  1. PROMPT_MAESTRO v6.41.
  2. TRANSFER_SESION_2026-09-19_POST_F2.3BIS.md.

Punto de partida: 75 commits locales ahead, contrato semantico v1
redactado, policy v1.3 propuesta, informe F2.4 entregado internamente.

Trabajo realizado en esta sesion (orden cronologico):

  a) Deteccion de 6 fixes mecanicos pendientes (ya autorizados por el
     dictamen formal del auditor pero sin aplicar).
  b) Aplicacion de los 6 fixes (P51, P32, P14-BIS, P31, P18/P26,
     P70 guarda), uno por commit, con tests especificos y verificacion
     completa (946 passed, sin regresion).
  c) Materializacion del dictamen del auditor sobre la revision
     estructural (NIPC_DICTAMEN_REVISION_ESTRUCTURAL_2026-09-20.md).
  d) Redaccion del informe de estado post-fixes al auditor (Q1-Q10).
  e) Revision cruzada del auditor (12 puntos) -> 4 tandas de
     correcciones documentales.
  f) Reconciliacion final de hashes en la cadena documental.
  g) Actualizacion del prompt (v6.41 -> v6.42) y creacion de este
     transfer.

=============================================================================
SECCION C - Decisiones del auditor pendientes
=============================================================================

Estado formal al cierre de la sesion:

  Dictamen revision estructural   EMITIDO / MATERIALIZADO
                                  (NIPC_DICTAMEN_REVISION_ESTRUCTURAL
                                   _2026-09-20.md, hash 3d004540...)

  F2.4                            PENDIENTE EXTERNO
                                  (informe entregado al auditor)

  Decisiones solicitadas al auditor (informe F2.4, seccion 6.0):

    D1  Adopcion contractual del contrato v1 (P60, P61, P38).
    D2  Aprobacion de NIPC_COVERAGE_POLICY_V13_PROPUESTA.md.
    D3  Confirmacion del cierre de P70.
    D4  Autorizacion del paso 4 (tests especificos).

  Preguntas adicionales (informe post-fixes, seccion 5.2-5.3):

    Q1  Validacion de los 6 fixes mecanicos.
    Q2  Reclasificacion de P70 tras aplicar la guarda.
    Q3  Documento de referencia para F2.4.
    Q4  Naturaleza de los tests del paso 4 (rojos/verdes/ambos).
    Q5  RETIRADA - ubicacion de temporal_validity.py (impl., no contrato).
    Q6  Nota de "no apto" en baseline historico.
    Q7  Materializacion del campo identity_type por fuente.
    Q8  Extension de la guarda P70 a HR_CHAIN_RESTATEMENT.
    Q9  Regeneracion de evidencia P70.
    Q10 Push parcial de los fixes mecanicos.

=============================================================================
SECCION D - Documentos entregados al auditor
=============================================================================

Documentos principales (11 ficheros):

  1. NIPC_INFORME_ENTREGA_F24.md                      0763f16b...
  2. NIPC_INFORME_ESTADO_POST_FIXES_F24.md            49c397b7...
  3. NIPC_CONTRATOS_SEMANTICOS_v1.md                  c02d202f...
  4. NIPC_COVERAGE_POLICY_V13_PROPUESTA.md            933fcb2e...
  5. NIPC_P70_DICTAMEN.md                             e8d4e4e8...
  6. NIPC_DICTAMEN_REVISION_ESTRUCTURAL_2026-09-20.md 3d004540...
  7. INSTITUTIONAL_ACCUMULATION_REVISION_ESTRUCTURAL_2026-09-20.md  318f02e8...
  8. evidence/nipc_p70_probe/_probe_p70.py            d1742a76...
  9. evidence/nipc_p70_probe/probe_p70_result.json    809c51af...
 10. evidence/nipc_p70_probe/probe_p70_summary.txt    9e8c42d6...
 11. evidence/nipc_p70_probe/README.md                 1370017c...

Documentos preservados intactos:

  - NIPC_COVERAGE_POLICY.md v1.0 (57f2d01f...)
  - NIPC_COVERAGE_POLICY_V11_PROPUESTA.md
  - NIPC_COVERAGE_POLICY_V12_PROPUESTA.md
=============================================================================
SECCION E - Reglas operativas criticas
=============================================================================

  - Local-first IAE: NO push a main hasta cierre de Gate-NIPC.3.
    97 commits quedan en local.

  - NO ejecutar OpenFIGI masivo (24.838 CUSIPs) sin dictamen especifico.

  - NO modificar NIPC_COVERAGE_POLICY.md v1.0 (hash 57f2d01f...).

  - NO tocar el motor NIPC (delta_shares.py, nipc.py,
    security_identity.py) sin contrato aprobado.

  - NO fijar THRESHOLD_1 ni THRESHOLD_2. UNDEFINED hasta dictamen F2.4.

  - NO modificar los contratos P38/P60/P61 sin aprobacion del auditor.

  - Regla canonical_snapshot CONGELADA (agregados SSHPRNAMT desde
    apply_amendments().canonical_snapshot, nunca desde INFOTABLE.parquet).

  - Convencion temporal vigente para el piloto:
    RADAR_SNAPSHOT_DATE = 2026-09-19
    13F_OBSERVATION_PERIOD = 2026-03-31
    RADAR_MEMBERSHIP_MODE = CURRENT_RETROSPECTIVE

  - Sustituciones condicionales SIEMPRE en Python puro:
    usar pathlib.Path.read_text + replace(..., 1) con assert previo de
    count==1. NUNCA usar -replace de PowerShell con argumento numerico
    (interpreta el numero como patron de borrado).

=============================================================================
SECCION F - Trampas recurrentes confirmadas esta sesion
=============================================================================

  - -replace de PowerShell con argumento numerico.
    $x -replace "A", "B" -replace 1
    NO significa "sustituye 1 vez". Significa "sustituye A por B, luego
    borra todos los caracteres '1' del string". Corrompio el informe
    NIPC_INFORME_ENTREGA_F24.md durante la sesion (recuperado
    regenerando). Regla nueva: sustituciones condicionales en Python puro.

  - Anchors con sufijo de firma completa.
    def test_x(): != def test_x(tmp_path):
    Los anchors de tests deben incluir la firma completa con parametros.
    Fallo detectado en P18/P26 y P70.

  - Hash completo vs abreviado.
    Cuando se sustituye un hash en un documento, hay que sustituir
    TANTO la forma completa (64 chars) COMO las formas abreviadas
    (prefijos de 8 chars + "..."). Fallo detectado en la reconciliacion
    final de hashes (2d5084b9 completo en algunas lineas, abreviado
    en otras).

  - Reutilizar el hash viejo del propio documento tras modificarlo.
    Modificar un documento cambia su hash. Si ese documento se cita a
    si mismo por su propio hash (o si otro documento lo cita), hay que
    propagar el nuevo hash en cascada. Ya ocurrio con
    INSTITUTIONAL_ACCUMULATION_REVISION_ESTRUCTURAL y con
    NIPC_INFORME_ENTREGA_F24.

  - Here-strings PowerShell >20 lineas o >5 $ se corrompen.
    Escribir a _patch_*.py con [System.IO.File]::WriteAllText y
    ejecutar con py.

  - pyflakes sobre fichero .md da error de parseo. Ruido, no bug.

  - Los scripts _*.py en raiz estan gitignored (regla /_*.py).
    Los scripts de patch son temporales y se borran tras ejecutar.

  - [System.IO.File]::WriteAllText con path relativo resuelve contra
    C:\WINDOWS\system32. Usar Join-Path $PWD para construir paths
    absolutos antes de invocar metodos .NET.

=============================================================================
SECCION G - Comandos de arranque (verificacion al inicio)
=============================================================================

  Set-Location D:\Macro_Sectorial
  git log --oneline -8
  git status -sb
  git rev-list --count origin/main..HEAD

  py -m compileall . -q
  py -m pyflakes . 2>&1
  py -m pytest tests/ validation/ -q --tb=short

Esperado al arrancar la nueva sesion:

  - HEAD = 3c59d41 o el hash del commit del transfer recien creado.
  - ahead 97 o 98 (si el transfer se commitea).
  - Working tree limpio.
  - 946 passed + 2 skipped.
  - pyflakes silencio.

Verificacion de la policy intacta:

  (Get-FileHash docs\auditoria\NIPC_COVERAGE_POLICY.md -Algorithm SHA256).Hash.ToLower()
  # Esperado: 57f2d01feb68d8916dfa4fa0451224c2ee3dda7e4631a7863e671257b3e91f06

Verificacion del contrato v1 (debe seguir como PROPUESTO):

  Select-String -Path docs\auditoria\NIPC_CONTRATOS_SEMANTICOS_v1.md -Pattern 'PROPUESTO / SOMETIDO A F2.4'
  # Esperado: 3 matches (P60, P61, P38)

=============================================================================
SECCION H - Lo que NO debes hacer (asistente entrante)
=============================================================================

  NO pushear los commits hasta cierre de Gate-NIPC.3.

  NO modificar NIPC_COVERAGE_POLICY.md v1.0.

  NO ejecutar OpenFIGI masivo sin autorizacion expresa.

  NO tocar el motor NIPC (delta_shares.py, nipc.py, security_identity.py)
  sin contrato aprobado.

  NO fijar THRESHOLD_1 ni THRESHOLD_2.

  NO modificar los contratos P38/P60/P61 sin dictamen F2.4.

  NO inventar SERIES_ID, CUSIP, CIK, FIGI. Solo valores verificados
  contra outputs reales.

  NO usar ticker:<ticker> como canonical_security.

  NO reconciliar NT <-> HR.

  NO convertir filer_continuity_pct en threshold.

  NO sumar SSHPRNAMT desde INFOTABLE.parquet crudo.

  NO usar -replace de PowerShell con argumento numerico.

  NO usar Notepad para contenido largo. Usar script Python con
  [System.IO.File]::WriteAllText.

  NO usar py -c con comillas dobles anidadas (usar script temporal).

  NO confundir shareClassFIGI como identidad inmutable: cambia tras
  reorganizacion corporativa (caso XOM).

  Regla del usuario: el usuario decide cuando parar. El asistente
  NO cierra sesion por fatiga.

=============================================================================
SECCION I - Proximos pasos posibles
=============================================================================

  1. Esperar dictamen F2.4 del auditor sobre el paquete entregado.

  2. Si F2.4 es GO: iniciar paso 4 (tests especificos para P38, P60,
     P61). Los 6 fixes mecanicos ya estan aplicados, no bloquean.

  3. Si F2.4 es GO CONDICIONADO: leer condiciones explicitas y ajustar
     el alcance del paso 4 en consecuencia.

  4. Si F2.4 es NO GO: revisar que punto(s) se rechazan y rediseñar.

  5. Cierre de sesion.

Ninguna de estas acciones se ejecuta sin confirmacion del usuario.

=============================================================================
SECCION J - Confirmacion esperada del asistente entrante
=============================================================================

Tras leer el prompt maestro v6.42 + este documento, el asistente
entrante debe:

  1. Confirmar asimilacion del contexto.

  2. Reconocer:
     - HEAD 3c59d41, ahead 97, tests 946+2.
     - Prompt v6.42 vigente.
     - Policy v1.0 intacta (57f2d01f...).
     - Contrato v1 PROPUESTO (c02d202f...).
     - Policy v1.3 BORRADOR (933fcb2e...).
     - 6 fixes mecanicos aplicados (P51, P32, P14-BIS, P31, P18/P26, P70).
     - Dictamen revision estructural materializado (3d004540...).
     - Paquete F2.4 entregado al auditor (11 ficheros).

  3. Reconocer que sigue BLOQUEADO sin dictamen externo:
     - F2.4 (aprobacion contratos P38/P60/P61).
     - OpenFIGI masivo (24.838 CUSIPs).
     - THRESHOLD_1/2 fijados.
     - Gate-NIPC.2 y Gate-NIPC.3.
     - Aplicacion de policy v1.3.

  4. No proponer tareas nuevas sin antes preguntar "Que hacemos?".

  5. Respetar regla "usuario decide cuando parar".

  6. Aplicar regla "nada se interpreta, todo se verifica".

  7. Aplicar regla "sustituciones condicionales en Python puro".

=============================================================================
FIN DEL DOCUMENTO DE TRANSMISION
Version 1.0 (2026-09-20). Referencia: prompt v6.42, HEAD 3c59d41.
=============================================================================