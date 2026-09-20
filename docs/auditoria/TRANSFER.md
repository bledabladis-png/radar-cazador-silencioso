\# TRANSFER DE SESION - 2026-09-20



Documento complementario al PROMPT\_MAESTRO v6.45 (docs/auditoria/PROMPT\_MAESTRO.md).

No normativo. Si hay conflicto, gana el prompt.



\*\*Como usarlo:\*\*

1\. Pegar este documento como primer mensaje.

2\. Si tienes acceso al repo, adjuntar tambien PROMPT\_MAESTRO.md.

3\. Esperar confirmacion de asimilacion antes de empezar.



\---



\## 1. Rol



Eres el Ingeniero Supervisor del Radar de Rotacion Sectorial. Sistema

determinista, descriptivo, auditable, sin ML predictivo. Entorno Windows,

PowerShell, Python via `py`. Repo: D:\\Macro\_Sectorial.



Reglas de personalidad y metodo: ver PROMPT\_MAESTRO v6.45 seccion 1 y 3.



\---



\## 2. Estado actual verificado (2026-09-20)



&#x20;   | Metrica               | Valor                                    |

&#x20;   |-----------------------|------------------------------------------|

&#x20;   | HEAD local            | 3ad57b1                                  |

&#x20;   | origin/main           | 9d4a81e                                  |

&#x20;   | Ahead                 | 120 commits locales                      |

&#x20;   | Working tree          | limpio                                   |

&#x20;   | Push                  | NO (local-first IAE activo)              |

&#x20;   | Prompt vigente        | v6.45                                    |

&#x20;   | Tests locales         | 988 passed + 5 xfailed + 2 skipped       |

&#x20;   | pyflakes              | 0 warnings                               |

&#x20;   | compileall            | OK                                       |

&#x20;   | Gate validacion       | 10/10                                    |

&#x20;   | Cobertura radar       | 313/313                                  |



Los 115 commits locales NO se han pusheado. Regla local-first IAE: el

modulo IAE no se pushea hasta Gate-NIPC.3. `origin/main` esta en

`9d4a81e`, sano, con el sistema pre-IAE.



\---



\## 3. Estructura de docs/auditoria/



Reorganizada en esta sesion (81 .md -> 27). Regla: 1 concepto = 1 fichero.



&#x20;   docs/auditoria/

&#x20;     PROMPT\_MAESTRO.md    (v6.45, normativo)

&#x20;     FOLLOWUPS.md         (cronologia de decisiones)

&#x20;     README.md            (navegacion + reglas)

&#x20;     iae/                 (12 .md - modulo IAE)

&#x20;     radar/               (9 .md - contratos temporales + deuda)

&#x20;     auditorias/          (4 .md - auditorias estructurales)

&#x20;     evidence/            (6 subdirs - probes)

&#x20;     archive/             (vacio)



\### iae/ (lo mas relevante)



&#x20;   PROPUESTA.md                          diseno fundacional

&#x20;   CONTRATO.md                           contrato IAE v1.1

&#x20;   CONTRATO\_ADDENDUM.md                  D1-D5

&#x20;   NIPC\_ESPECIFICACION.md                spec NIPC v1.4

&#x20;   NIPC\_CONTRATOS\_SEMANTICOS\_v1.md       contrato P38/P60/P61

&#x20;   NIPC\_COVERAGE\_POLICY.md               policy v1.0 vigente

&#x20;   NIPC\_COVERAGE\_POLICY\_V13\_PROPUESTA.md policy propuesta

&#x20;   INFORME.md                            consolidado de 17 informes

&#x20;   DICTAMENES.md                         registro de 23 dictamenes

&#x20;   RECONCILIACION\_CONTRATO\_CODIGO.md     expediente 3 divergencias

&#x20;   REESTRUCTURACION\_MODULO.md            plan arquitectonico

&#x20;   FASE\_A6\_PLAN.md                       plan de ejecucion A.6



\### radar/



&#x20;   DEUDA.md                              deuda activa del radar

&#x20;   FU-021-5\_\*.md                         contratos temporales

&#x20;   FU-021-3B\_\*.md, FU-021-3C\_\*.md        informes FU-021-3

&#x20;   H1\_INFORME\_MUTABILIDAD\_HISTORICA.md

&#x20;   E5\_INFORME\_VIX3M.md



\---



\## 4. Reglas nuevas (esta sesion)



\### 4.1. Basura = borrar



Sin apelar a git. Si un fichero no se usa, no da contexto y no se lee,

se borra. Git no es papelera. Snapshot de sesion: borrar al cerrar.

Informe consolidado: sustituye a sus originales. Dictamen consolidado:

sustituye a los originales.



Ver PROMPT\_MAESTRO v6.45 seccion 3.7.



\### 4.2. 1 concepto = 1 fichero vivo



Al evolucionar, se edita in-place. Prohibido sufijos `\_v1.md`,

`\_V12\_PROPUESTA.md`, `\_DICTAMEN\_C.md`. Excepcion: contratos cuyo sha256

esta en la cadena autoritativa (NIPC\_CONTRATOS\_SEMANTICOS\_v1.md,

NIPC\_COVERAGE\_POLICY.md v1.0) NO se modifican. Se emite vN+1 con seccion

"Deriva de vN".



\### 4.3. PowerShell seguro



\- NO `-replace` con argumento numerico. Sustituciones condicionales en

&#x20; Python puro con `.replace(..., 1)` + assert de conteo.

\- `\[System.IO.File]::WriteAllText` con `Join-Path $PWD`. Nunca path

&#x20; relativo (va a C:\\WINDOWS\\system32).

\- Here-strings >20 lineas o >5 `$`: escribir a `\_patch\_XXX.py` con

&#x20; WriteAllText y ejecutar con `py`.

\- EOL objetivo: LF puro. UTF-8 sin BOM.



\### 4.4. Test execution



Siempre: `py -m compileall . -q` + `py -m pyflakes . 2>\&1` +

`py -m pytest tests/ validation/ -q --tb=short`.



\---



\## 5. Divergencias contrato <-> codigo (identificadas)



Expediente completo en `iae/RECONCILIACION\_CONTRATO\_CODIGO.md`. 3

divergencias, con evidencia ejecutable en 15 tests contractuales:



&#x20;   | Div | Contrato | Descripcion                                | Tests |

&#x20;   |-----|----------|--------------------------------------------|-------|

&#x20;   | D1  | P61      | resolver no invoca resolve\_source\_status   | 2     |

&#x20;   | D2  | P38      | nipc no construye TARGET real (proxy)      | 3     |

&#x20;   | D3  | P60      | default TICKER silencioso en CSV           | 1     |



Los 6 tests marcados `@pytest.mark.xfail` documentan estas divergencias.

Cuando F2.4 autorice los fixes, se retiran los marcadores y los tests

deben pasar.



\---



\## 6. Bloqueos vigentes



&#x20;   F2.4 (dictamen externo)          PENDIENTE EXTERNO

&#x20;   THRESHOLD\_1 / THRESHOLD\_2        UNDEFINED

&#x20;   Gate-NIPC.2                      BLOQUEADO

&#x20;   Gate-NIPC.3                      NO AUTORIZADO

&#x20;   OpenFIGI masivo (24.838 CUSIPs)  NO AUTORIZADO

&#x20;   Policy v1.3 aplicacion           NO AUTORIZADA

&#x20;   Push a origin/main               NO (local-first IAE)



\*\*F2.4 es el unico input externo bloqueante.\*\* Todo lo materializable

sin dictamen esta hecho.



\---



\## 7. Ciclos abiertos



\### 7.1. IAE (principal)



&#x20;   FASE A     FA-1 + FA-2              CERRADO / PUSHED

&#x20;   FASE A.5   NIPC + contratos         IMPLEMENTADO (sin push)

&#x20;   FASE A.6   Reconciliacion           PAQUETE LISTO, espera F2.4

&#x20;   FASE B-E   N-PORT, cross-val, etc.  NO INICIADO



\### 7.2. Deuda radar (radar/DEUDA.md)



&#x20;   D-RADAR-01  20 tickers LSE sin provider dedicado    MEDIA

&#x20;   D-RADAR-02  Cron `0 4 \* \* \*` fines de semana        BAJA

&#x20;   D-RADAR-03  Guard no distingue core de LSE          MEDIA



Contexto de D-RADAR-01: el universo `stock\_prices.parquet` incluye 20

tickers `.L` (London) sin provider dedicado. Dependen de Yahoo. Cuando

Yahoo falla, la cobertura cae a 0.9361 y `guard\_coverage` bloquea el

commit. El sistema queda intacto, pero el run aparece X. Decision:

buscar fuente dedicada (LSEG, EOD Historical, etc.).



\### 7.3. Fallo reciente del run scheduled



Run 35432363006 (sabado 19/09, 08:34 UTC) fallo con `\[FAIL]

stock\_prices coverage\_pct\_last=0.936102 < 0.95`. Causa: los 20 `.L`

sin Close. Guard bloqueo. `origin/main` intacto. \*\*No es bug del

pipeline.\*\* Documentado en `radar/DEUDA.md` D-RADAR-01.



\---



\## 8. Tests contractuales (evidencia para F2.4)



&#x20;   tests/test\_p60\_contract.py     4 tests (3 pass + 1 xfail)

&#x20;   tests/test\_p61\_contract.py     4 tests (3 pass + 1 xfail)

&#x20;   tests/test\_p38\_contract.py     5 tests (2 pass + 3 xfail)



Total: 14 tests, 9 pass, 5 xfail. NO tocan codigo productivo. Son

evidencia ejecutable para el auditor.



\---



\## 9. Como retomar



\### Si vas a seguir auditando / implementando



1\. Ejecutar comandos de arranque (ver seccion 10).

2\. Leer `PROMPT\_MAESTRO.md` completo.

3\. Leer `iae/RECONCILIACION\_CONTRATO\_CODIGO.md` +

&#x20;  `iae/REESTRUCTURACION\_MODULO.md` + `iae/FASE\_A6\_PLAN.md`.

4\. Preguntar al usuario que frente atacar.



\### Si vas a esperar a F2.4



Cerrar sesion. Documentacion lista, working tree limpio, sin deuda

tecnica activa.



\### Si vas a atacar deuda radar (independiente de F2.4)



Leer `radar/DEUDA.md`. Prioridad: D-RADAR-01 (buscar provider LSE) o

D-RADAR-03 (refinar guard para distinguir core vs LSE).



\### Si vas a auditar calculo (regimenes, scores, SLPM, MTE, darkpool)



Independiente de F2.4. No hay trabajo previo. Empezar por inventario

en `regimes/` e `indicators/`.



\---



\## 10. Comandos de arranque



&#x20;   Set-Location D:\\Macro\_Sectorial

&#x20;   git log --oneline -10

&#x20;   git status -sb

&#x20;   git rev-list --count origin/main..HEAD

&#x20;   py -m compileall . -q

&#x20;   py -m pyflakes . 2>\&1

&#x20;   py -m pytest tests/ validation/ -q --tb=short



Esperado:

&#x20;   - HEAD = 3ad57b1 o posterior

&#x20;   - ahead 120 o mas

&#x20;   - working tree limpio

&#x20;   - 988 passed + 5 xfailed + 2 skipped

&#x20;   - pyflakes silencio



\---



\## 11. Lo que NO hacer



&#x20;   - NO push a origin/main. Local-first IAE activo.

&#x20;   - NO modificar NIPC\_CONTRATOS\_SEMANTICOS\_v1.md (hash en cadena).

&#x20;   - NO modificar NIPC\_COVERAGE\_POLICY.md v1.0 (hash 57f2d01f...).

&#x20;   - NO ejecutar OpenFIGI masivo sin dictamen especifico.

&#x20;   - NO fijar THRESHOLD\_1 / THRESHOLD\_2 sin propuesta sobre evidencia v2.

&#x20;   - NO reconciliar NT <-> HR.

&#x20;   - NO sumar SSHPRNAMT desde INFOTABLE.parquet crudo.

&#x20;   - NO usar `-replace` de PowerShell con argumento numerico.

&#x20;   - NO cerrar sesion por fatiga (el usuario decide).



\---



\## 12. Confirmacion esperada



Cuando recibas este transfer, responde:



&#x20;   "Confirmado, contexto asimilado."



&#x20;   Estado del sistema que reconozco:

&#x20;     - HEAD 9c2ff66, ahead 120

&#x20;     - 988 passed + 5 xfailed + 2 skipped

&#x20;     - F2.4 PENDIENTE EXTERNO

&#x20;     - 3 divergencias P60/P61/P38 con 6 tests xfail documentandolas

&#x20;     - Deuda radar registrada (D-RADAR-01/02/03)



&#x20;   Pregunta final: "¿Que hacemos?"



No empieces a proponer tareas sin antes confirmar asimilacion.



\---



FIN DEL TRANSFER

Version 1.0 (2026-09-20). HEAD 9c2ff66.

