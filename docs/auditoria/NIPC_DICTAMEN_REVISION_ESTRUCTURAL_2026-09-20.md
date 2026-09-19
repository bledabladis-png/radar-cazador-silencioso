# DICTAMEN FORMAL DEL AUDITOR

**Objeto:** revision estructural completa del modulo IAE
**HEAD al emitir:** 77d4785
**Alcance:** src/institutional_accumulation/ + tests + documentacion auditada
**Estado:** NO GO PARA FIJAR THRESHOLDS / GO PARA CICLO DE CORRECCIONES SELECTIVAS
**Materializado:** 2026-09-20 (transcripcion de la sesion)

---

## 1. Dictamen ejecutivo

La revision ha cumplido correctamente su objetivo: no ha encontrado un fallo general del motor NIPC, sino una combinacion de:

- A. problemas contractuales/semanticos
- B. problemas de robustez
- C. deuda documental
- D. problemas de portabilidad

El nucleo NIPC no debe declararse defectuoso en terminos generales.

Sin embargo, existen **tres cuestiones que deben resolverse antes de cualquier fijacion de thresholds**:

- P38: paired_weighted_share_coverage
- P60: normalizacion de canonical_security
- P61: vigencia temporal del crosswalk interno

Y existe una cuarta:

- P70: semantica de classify_strategy

que debe cerrarse mediante una prueba de invariante antes de aceptar definitivamente la logica de amendments.

La revision tambien descubre que **P18/P26, P32, P51 y P14-BIS son correcciones de robustez razonables, pero no deben mezclarse con la discusion semantica de coverage**.

---

## 2. P38 - paired_weighted_share_coverage

### Decision: NO GO / BLOQUEO CRITICO

Este es el hallazgo mas importante del informe.

La evidencia presentada demuestra una discrepancia entre:

- docstring -> posiciones "en ambos"

y:

- implementacion -> union_sec = sec_c | sec_p

La cuestion no puede resolverse por preferencia de implementacion. Debe resolverse por contrato.

### Decision del auditor

La denominacion correcta para la metrica de **cobertura pairwise** debe utilizar como denominador el **conjunto de securities objetivo realmente comparables entre ambos periodos**, es decir:

    TARGET_Q4 INTERSECT TARGET_Q1

y no:

    TARGET_Q4 UNION TARGET_Q1

La razon es conceptual:

    PAIRED = security observable/resuelta en Q4 + security observable/resuelta en Q1

Una security que solo existe en uno de los dos periodos es NEW o EXIT; no es una security que pueda ser paired.

Por tanto:

    denominador pairwise = target securities presentes en ambos periodos

y:

    numerador = paired securities resueltas en ambos

### Convencion ponderada

Debe mantenerse la convencion ya propuesta:

    w(s) = max( SSHPRNAMT_Q4(s), SSHPRNAMT_Q1(s) )

una vez por security.

Entonces:

    paired_weighted_share_coverage
      = sum( w(s) [paired] )
        / sum( w(s) [TARGET_Q4 INTERSECT TARGET_Q1] )

Esto debe congelarse de forma literal en la policy.

### Consecuencia

**THRESHOLD_2 no puede fijarse todavia.**

Y los valores anteriores calculados con union_sec deben considerarse:

    HISTORICOS / NO APTOS PARA THRESHOLD_2

hasta recalcularse bajo la definicion aprobada.

No significa que todo el baseline quede invalidado. Solo la metrica afectada.

### Estado

**P38 = CONFIRMADO / BLOQUEO DE THRESHOLD_2.**
---

## 3. P60 - _normalize_canonical

### Decision: NO GO CONTRACTUAL / FIX REQUERIDO

El problema esta bien identificado.

Esta regla:

    valor desconocido -> equity:<valor>

**infiere el tipo de identidad** sin evidencia del origen.

Eso contradice el principio ya congelado:

    NO INFERIR IDENTIDAD

El problema no se limita a FIGI.

Un valor desnudo puede ser:

    ticker / FIGI / otro identificador / valor corrupto

y el codigo no dispone de informacion suficiente para decidirlo.

### Decision

No recomiendo una solucion basada exclusivamente en regex:

    if regex FIGI: figi: else: equity:

Eso sigue siendo una inferencia.

La solucion contractual correcta es que el **origen declare explicitamente el tipo de identidad**.

Por ejemplo:

    source = etf_holdings
    identity_type = TICKER

    source = openfigi
    identity_type = FIGI

y el normalizador reciba ese tipo.

Entonces:

    normalize(value, identity_type)

puede producir:

    TICKER -> equity:<ticker>
    FIGI   -> figi:<FIGI>

Un valor sin `identity_type` explicito debe quedar:

    UNRESOLVED / INVALID

no convertirse automaticamente en `equity:`.

### Estado

**P60 = CONFIRMADO.**

No aplicar todavia el patch sin disenar primero el contrato del normalizador.

---

## 4. P61 - etf_holdings.csv sin vigencia temporal

### Decision: NO GO COMO RESOLUCION HISTORICA CANONICA

Este hallazgo es mas importante de lo que su clasificacion MEDIA podria sugerir.

El problema es:

    etf_holdings: valid_from = NULL, valid_to = NULL

para una relacion que posteriormente se usa para resolver observaciones de Q4 2025 / Q1 2026.

Eso implica actualmente:

    relacion observada alguna vez
        -> tratada como valida indefinidamente

y esa equivalencia no esta demostrada.

Esto entra directamente en el contrato v1.3:

    identity validity != metadata validity != historical operational mapping

### Decision del auditor

No recomiendo simplemente inventar:

    valid_from = -infinito / valid_to = +infinito

ni asignar las fechas actuales a posteriori.

Hasta disponer de evidencia temporal, una relacion de `etf_holdings` utilizada para un periodo historico debe poder quedar:

    TEMPORAL_UNVERIFIED

y no:

    CANONICAL

cuando la policy exija identidad historica verificable.

Esto puede reducir la cobertura aparente. Eso seria un resultado metodologico, no un problema que deba ocultarse mediante un fallback.

### Consecuencia

P61 afecta directamente a:

    RESOLVED_UNIVERSE / resolution_coverage / paired_security_coverage

Por ello:

**P61 debe resolverse antes de fijar thresholds.**

### Estado

**P61 = CONFIRMADO / BLOQUEO DE VALIDACION HISTORICA.**
---

## 5. P70 - classify_strategy

### Decision: NO PATCH TODAVIA / CERRAR INVARIANTE

El informe detecta:

    HR_PLUS_RESTATEMENT -> iloc[1]
    HR_CHAIN_RESTATEMENT -> iloc[-1]

Pero la mera diferencia de indices **no demuestra todavia un bug**.

Si existe la invariante:

    HR_PLUS_RESTATEMENT -> exactamente 2 filings

entonces:

    iloc[1] == iloc[-1]

y no existe diferencia semantica.

El problema real seria que esa invariante no estuviera garantizada.

### Decision

Antes del patch:

    assert / test len(filings_group) == 2

para demostrar la precondicion.

Si la clasificacion puede legitimamente recibir:

    len > 2

entonces si:

    iloc[-1]

debe pasar a ser la regla comun.

### Estado

**P70 = ABIERTO / VALIDACION DE INVARIANTE.**

No modificar classify_strategy todavia.

---

## 6. P31 - _sum_delta

### Decision: GO PARA CORRECCION DOCUMENTAL/SEMANTICA

El docstring:

    None si no hay filas validas

es falso.

La funcion devuelve:

    0.0

en todos los caminos.

Esto no demuestra que el NIPC este calculando mal, porque el estado utiliza n_total y otras variables.

Pero si viola el principio:

    UNAVAILABLE != VALID ZERO

### Solucion recomendada

No romper compatibilidad cambiando inmediatamente 0.0 -> None.

Una solucion mas segura seria:

    nipc_total
    nipc_total_available

o:

    n_observations

manteniendo:

    nipc_total = 0.0

para compatibilidad.

### Estado

**P31 = CONFIRMADO / NO BLOQUEA COVERAGE.**

---

## 7. P18 / P26 - Path absoluto

### Decision: GO PARA FIX MECANICO

Es un defecto real:

    D:\Macro_Sectorial

no debe formar parte del default estructural del modulo.

La solucion propuesta:

    IAE_PROJECT_ROOT

o resolucion relativa al paquete es valida.

Pero hay una decision importante: **no introducir una segunda fuente de verdad de paths.**

Preferencia:

    caller explicito -> project_root

y solo como fallback:

    IAE_PROJECT_ROOT -> Path(__file__).resolve()

### Estado

**P18/P26 = FIX AUTORIZABLE.**

No afecta a la semantica NIPC.

---

## 8. P14-BIS - fillna(0.0)

### Decision: GO PARA FIX PREVENTIVO

El analisis del informe es correcto:

Hoy el FULL OUTER JOIN permite interpretar:

    left_only  -> previous = 0
    right_only -> current = 0

sin inventar datos.

Por tanto: **P14 original = NO BUG.**

Pero P14-BIS es una mejora de robustez legitima.

La propuesta:

    _merge + validacion BOTH

es preferible a:

    fillna global

porque preserva la semantica de procedencia.

### Estado

**P14-BIS = FIX AUTORIZABLE.**
---

## 9. P32 - SSHPRNAMT no parseable

### Decision: GO PARA FIX DE TRAZABILIDAD

No recomiendo cambiar todavia la politica:

    invalid SSHPRNAMT -> excluded

porque eso podria alterar resultados.

Si recomiendo anadir:

    n_dropped_invalid_sshprnamt

y, preferiblemente:

    dropped_invalid_sshprnamt_weight

si la arquitectura lo permite sin alterar el calculo.

La regla debe ser:

    drop != silence

El dato descartado debe quedar cuantificado.

### Estado

**P32 = FIX AUTORIZABLE.**

---

## 10. P51 - lineas cortas del Official List

### Decision: GO PARA HARDENING

Aqui si existe un riesgo real:

    linea invalida -> ljust(80) -> status blank -> ACTIVE

Es exactamente el tipo de transformacion que puede convertir:

    dato invalido

en:

    dato valido

sin evidencia.

La propuesta:

    len(line) < POS_STATUS[1] -> anomaly

es correcta conceptualmente.

Pero debe preservarse el registro de la anomalia y no convertirlo simplemente en un drop.

### Estado

**P51 = FIX AUTORIZABLE.**

---

## 11. P19 - OpenFIGI network retry

### Decision: GO DIFERIDO

El riesgo esta bien identificado:

    URLError / timeout -> chunk completo -> ERROR

Pero no afecta a la semantica del contrato mientras el piloto conserve los errores.

Puede corregirse despues de cerrar la arquitectura de coverage.

No mezclar:

    retry policy

con:

    identity semantics

### Estado

**P19 = DEUDA OPERACIONAL, NO BLOQUEO DE GATE.**
---

## 12. P20 - derivacion del radar

### Decision: GO DOCUMENTAL

El informe confirma que:

    stock_prices.parquet

es la fuente utilizada.

El riesgo no es critico mientras la convencion permanezca documentada:

    242 / sin sufijo europeo

y el snapshot quede identificado.

No convertiria ahora el criterio "." not in ticker en un refactor general.

Puede documentarse como:

    RADAR_TARGET_CONVENTION_V1

### Estado

**P20 = NO BLOQUEANTE.**

---

## 13. P21 - _pick_row duplicada

### Decision: NO TOCAR AHORA

Es deuda de mantenimiento.

No existe evidencia de divergencia semantica actual porque las dos copias implementan la misma prioridad:

    exchCode == US -> first

Puede unificarse posteriormente.

### Estado

**P21 = DEUDA / NO BLOQUEO.**

---

## 14. P24 - exch_code="US"

### Decision: CORRECTO PARA EL SCOPE USA

No lo considero bug.

El micro-gate esta definido deliberadamente sobre:

    radar USA

y:

    CUSIP -> OpenFIGI exchCode="US"

es coherente con esa frontera.

El hecho de que produzca NO_ID para CUSIPs no-USA no demuestra defecto; demuestra una restriccion del scope.

Eso si: NO_ID por scope debe distinguirse de NO_ID por ausencia real en OpenFIGI para las metricas ponderadas.

### Estado

**P24 = NO BUG / DOCUMENTAR.**

---

## 15. P39 - conflictos solo desde Q1

### Decision: NO BLOQUEANTE; DEFINIR LATERALMENTE

No existe suficiente evidencia en este informe para concluir que sea incorrecto.

Puede ser intencional que:

    status

describa exclusivamente la calidad del periodo actual.

No recomiendo cambiarlo ahora.

Pero debe documentarse:

    status_current

frente a:

    cross-period integrity

si posteriormente se utiliza para publicacion.

### Estado

**P39 = DISENO PENDIENTE / NO BLOQUEA AHORA.**

---

## 16. P75 / P33 - unicidad

### Decision: GO DIFERIDO

El keep="first" silencioso no debe perpetuarse indefinidamente.

Pero antes debe comprobarse cual es el contrato esperado:

    ACCESSION_NUMBER -> exactamente un CIK

Si eso esta demostrado, entonces assert is_unique es el hardening correcto.

No merece bloquear el ciclo actual.
---

## 17. Prioridad real de correcciones

La revision permite construir ahora un orden mucho mas limpio:

    PRIORIDAD 0 - BLOQUEOS SEMANTICOS
      P38
      P60
      P61
      P70

Despues:

    PRIORIDAD 1 - ROBUSTEZ CON IMPACTO EN TRAZABILIDAD
      P51
      P32
      P14-BIS
      P31

Despues:

    PRIORIDAD 2 - PORTABILIDAD / MANTENIMIENTO
      P18/P26
      P75/P33
      P21
      P19
      P20

No recomiendo tocar todo en un unico commit.

---

## 18. Que NO debe hacerse

No recomiendo:

    - modificar C2
    - cambiar match_key
    - tocar compute_delta_shares por P38
    - introducir reconciliacion economica NT-HR
    - ampliar OpenFIGI masivo todavia
    - fijar THRESHOLD_1/2
    - reescribir el baseline historico
    - resolver P61 inventando fechas
    - resolver P60 mediante inferencia silenciosa por regex

Especialmente importante:

**P38 no se corrige cambiando simplemente union_sec por intersection y cerrando el asunto.**

Primero hay que congelar la formula en la policy; despues modificar codigo; despues recalcular evidencia.

---

## 19. Secuencia autorizada

La secuencia que autorizaria es:

    1. Dictamen P38/P60/P61/P70
            |
            v
    2. Modificar propuesta de policy
            |
            v
    3. Aprobar contrato
            |
            v
    4. Tests especificos
            |
            v
    5. Fixes quirurgicos
            |
            v
    6. Nueva medicion de coverage
            |
            v
    7. OpenFIGI ampliado
            |
            v
    8. Threshold proposal
            |
            v
    9. Gate-NIPC.2

No al reves.

---

## 20. Estado de Gate-NIPC

    NIPC core calculation             FUNCIONAL
    C2                                VIGENTE
    Filer continuity                  CERRADO
    Coverage baseline                 HISTORICO / PASS DIAGNOSTICO

    P38                               BLOQUEADO
    P60                               BLOQUEADO
    P61                               BLOQUEADO
    P70                               ABIERTO

    THRESHOLD_1                       UNDEFINED
    THRESHOLD_2                       UNDEFINED

    OpenFIGI full run                 NO AUTORIZADO
    Policy v1.2                       NO APLICADA
    Gate-NIPC.2                       BLOQUEADO
    Gate-NIPC.3                       NO AUTORIZADO

---

## DICTAMEN FINAL

### P38 - NO GO

Denominador contractual de paired_weighted_share_coverage debe ser la **interseccion de TARGET Q4/Q1**, con peso max(Q4,Q1) una vez por security. THRESHOLD_2 permanece bloqueado hasta recalcular.

### P60 - NO GO

canonical_security no puede inferir tipo de identidad. Debe existir identity_type explicito o el valor queda sin resolver.

### P61 - NO GO

etf_holdings.csv no puede considerarse automaticamente valido para Q4/Q1 sin evidencia temporal. Debe respetarse la separacion entre identidad y vigencia historica.

### P70 - GO CONDICIONADO

Cerrar primero la invariante de cardinalidad de HR_PLUS_RESTATEMENT. Si puede haber mas de dos filings, unificar sobre el ultimo filing aplicado.

### P31 - GO FIX

Corregir contrato de disponibilidad sin romper compatibilidad.

### P14-BIS - GO FIX

Condicionar fillna al origen del FULL OUTER JOIN.

### P32 - GO FIX

Anadir contador de SSHPRNAMT descartados.

### P51 - GO FIX

No permitir que una linea corta se convierta silenciosamente en ACTIVE.

### P18/P26 - GO FIX

Eliminar path absoluto del default.

---

## Estado definitivo de la revision

**La revision estructural es aprobada como auditoria diagnostica.**

Pero no la interpreto como autorizacion para "limpiar 95 observaciones". El objetivo ahora es mucho mas concreto:

    P38 -> contrato de coverage
    P60 -> contrato de identidad
    P61 -> contrato temporal
    P70 -> contrato de amendments

**Esos cuatro deben cerrarse antes de volver a medir coverage y antes de tocar thresholds.**

El resto puede entrar despues en un ciclo de hardening separado, manteniendo el principio del proyecto: **un cambio semantico = un gate; un fix mecanico = un commit reversible; una nueva metrica = una nueva evidencia.**

---

Fin del dictamen. Materializado el 2026-09-20.
Referencia: informe INSTITUTIONAL_ACCUMULATION_REVISION_ESTRUCTURAL_2026-09-20.md.