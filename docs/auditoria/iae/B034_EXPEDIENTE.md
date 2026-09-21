# B-03 + B-04 - Expediente (correccion de diagnostico al dictamen #76)

**Fecha:** 2026-09-22.
**Origen:** dictamen #76 secciones 5 (B-03) y 6 (B-04).
**Naturaleza:** expediente para dictamen. NO modifica codigo. Acompana a
`A67_CONSULTA.md` y `B02_EXPEDIENTE.md`.

---

# B-03 - TARGET completo: el universo contractual NO son los 35.649 CUSIPs 13F

## 1. Correccion material al dictamen #76 seccion 5

El dictamen #76 seccion 5 afirma:

> "TARGET_Q1 = 18 FIGI es un resultado sobre el subconjunto efectivamente
> materializado, no una demostracion del TARGET contractual completo si
> el universo completo aun requiere resolucion batch."

Cifra del contexto previo: 24.838 CUSIPs pendientes.

Medicion real sobre HEAD actual:

    Q1 INFOTABLE:        3.822.885 filas | 35.649 CUSIPs unicos
    Q4 INFOTABLE:        3.473.209 filas | 33.780 CUSIPs unicos
    crosswalk_internal:  550 filas | 542 CUSIPs distintos
    Q1 CUSIPs en crosswalk:  522 (1.46%)
    Q1 CUSIPs sin crosswalk: 35.127

## 2. Por que los 35.127 no son un hueco del sistema

El `radar_target_catalog.csv` define el **universo contractual** del
sistema: 242 claves (radar_ticker), cada una con su share_class_figi
(240 OK, 2 MISS). Ese es el TARGET del proyecto.

Los 35.127 CUSIPs sin resolver son securities observadas en 13F que NO
pertenecen al radar. El resolver las marca `OBSERVED_ONLY`, estado
legitimo del contrato P60 (identidad observada, no canonica).

**Conclusion: los 35.127 no son un deficit de cobertura.** Son
securities fuera del universo contractual por diseno. La afirmacion
"24.838 pendientes de OpenFIGI" del contexto previo confundia universo
observado 13F (35.649) con universo contractual (242).

## 3. El hueco real

El unico deficit del TARGET contractual son las 2 keys MISS:

    BRK-B  (status=MISS en radar_target_catalog.csv)
    MOG-A  (status=MISS en radar_target_catalog.csv)

Ambas tienen CUSIP observado en 13F Q1 (084670702 y 615394202), pero
OpenFIGI no les asigno share_class_figi en la ejecucion 2026-09-19.

## 4. Opciones tecnicas para B-03

**Opcion A - OpenFIGI re-query sobre las 2 MISS.**
Volver a consultar OpenFIGI unicamente para BRK-B y MOG-A. Coste
minimo (2 lookups). No requiere OpenFIGI masivo ni API key si se usa
el plan gratuito. Elimina el bloqueo estructural de B-03.

**Opcion B - Dejar las 2 MISS como limitacion declarada.**
Aceptar cobertura contractual de 240/242 keys (99.17%) y documentar
las 2 excepciones. El probe reporta 20/20 sobre el subconjunto con
FIGI; el universo completo es 240/242.

**Opcion C - OpenFIGI masivo.**
Resolver los 35.127 CUSIPs observados contra OpenFIGI. Descartada por
el supervisor: no aporta valor contractual (estan fuera del radar) y
consume presupuesto de API sin justificacion.

## 5. Recomendacion B-03

**Opcion A + B combinadas:**

1. Ejecutar OpenFIGI re-query sobre BRK-B y MOG-A. Si resuelven, pasa
   a 242/242. Si no resuelven, se documentan como excepcion.
2. Reclasificar el lenguaje de B-03: ya no es "universo completo sin
   materializar", es "2 keys MISS por resolver o documentar".
3. Descartar OpenFIGI masivo como requisito del proyecto.

# B-04 - Pairwise real: no es problema de datos, es vigencia temporal de fuentes

## 6. Correccion material al dictamen #76 seccion 6

El dictamen #76 seccion 6 afirma:

> "El resultado real es TARGET_Q4 = 0. Eso demuestra correctamente el
> comportamiento fail-closed, pero no demuestra la operacion pairwise
> real."

Es correcto respecto del resultado. La causa NO es ausencia de datos:

    Q4 SUBMISSION: 11.372 filings
      PERIODOFREPORT=2025-12-31 -> 10.676 filings (el dominante)
    Q1 SUBMISSION: 11.761 filings
      PERIODOFREPORT=2026-03-31 -> 10.776 filings (el dominante)

Ambos trimestres tienen datos reales completos. El cuello de botella
es otra cosa.

## 7. Por que Q4 = 0 operacionales

§5.5 exige `security_resolution_status = CANONICAL` **y**
`operational_mapping_status = VERIFIED`.

Estado del resolver sobre 33.780 CUSIPs Q4:

    OBSERVED_ONLY -> 33.282
    CANONICAL     -> 498

Los 498 CANONICAL vienen todos de `etf_holdings` (526 filas del
crosswalk interno, sin vigencia declarada). Por P61 §2.4, ausencia de
vigencia -> `operational_mapping_status = TEMPORAL_UNVERIFIED`.

Los 24 CUSIPs de `cusip_ticker_exceptions.csv` tienen vigencia
`valid_from=valid_to=2026-03-31`, que cubre Q1 2026 pero NO Q4 2025.

**Resultado:**

    Q4: 0 VERIFIED (los 498 CANONICAL son TEMPORAL_UNVERIFIED)
    Q1: 24 VERIFIED (los 24 de exceptions con vigencia Q1)

Coherente con el probe: Q4 fail-closed correcto, Q1 con 24 operacionales.

## 8. Diagnostico B-04 reformulado

B-04 se desdobla en dos sub-problemas independientes:

**B-04.1 - No hay VERIFIED para Q4.**
Fuentes disponibles (etf_holdings sin vigencia + 24 exceptions con
vigencia Q1-only) no cubren Q4 2025. Decision necesaria: ¿anadir
vigencia a etf_holdings para cubrir Q4? ¿Construir exceptions Q4?

**B-04.2 - El overlap Q4<->Q1 con VERIFIED es 0.**
Incluso con Q4 resuelto, para tener `TARGET_PAIRWISE > 0` hacen falta
keys con CANONICAL + VERIFIED en ambos trimestres. Hoy: 0.

## 9. Opciones tecnicas para B-04

**Opcion A - Declarar la limitacion.**
El sistema no puede emitir `paired_*` con datos actuales porque no
existe vigencias cruzadas Q4/Q1 en las fuentes curadas. `P38 =
GO CONDICIONADO` se mantiene sin evidencia pairwise. Se acepta como
resultado del ciclo actual.

**Opcion B - Ampliar vigencias de etf_holdings.**
Revisar si `etf_holdings.csv` puede llevar `valid_from`/`valid_to`
declarados. Hoy el crosswalk resultante no los propaga
(TEMPORAL_UNVERIFIED por P61 §2.4). Requiere:
  - modificar el loader del crosswalk para propagar vigencia;
  - declarar vigencia en la fuente (documental);
  - actualizar tests y contratos.
Alto coste. Toca contrato P61 (prohibido sin dictamen).

**Opcion C - Construir exceptions Q4.**
Ampliar `cusip_ticker_exceptions.csv` con vigencia Q4 2025 para 20-30
CUSIPs relevantes, generando TARGET_PAIRWISE > 0. Coste bajo, evidencia
demostrativa. No pretende cobertura historica completa, solo probar la
rama pairwise con datos reales.

## 10. Recomendacion B-04

**Opcion C - exceptions Q4 de demostracion.**

Anadir 20-30 CUSIPs con `valid_from=2025-12-31, valid_to=2026-03-31`
al CSV de exceptions. Con eso:

    Q4: N VERIFIED (N = nuevas entradas)
    Q1: N + 24 VERIFIED
    TARGET_PAIRWISE > 0
    paired_security_coverage y paired_weighted calculables

La evidencia es demostrativa (20-30 CUSIPs sobre universo contractual),
suficiente para ejercitar la rama pairwise. No pretende cobertura
historica plena.

Justificacion: es la via de menor coste, no toca contratos, no requiere
OpenFIGI masivo, y desbloquea B-04 empiricamente sin reclamar cierre
contractual total.

# Preguntas al auditor

## Sobre B-03

  1. Se acepta la correccion material (TARGET contractual = 242 keys,
     no 35.649)?
  2. Se autoriza OpenFIGI re-query sobre BRK-B y MOG-A (Opcion A)?
  3. Se descarta definitivamente OpenFIGI masivo como requisito del
     proyecto?

## Sobre B-04

  4. Se acepta la correccion material (Q4 tiene datos; el cuello de
     botella es vigencia temporal de fuentes)?
  5. Se autoriza la Opcion C (exceptions Q4 de demostracion, 20-30
     CUSIPs) para ejercitar la rama pairwise?
  6. Alternativa: si se prefiere Opcion A (declarar la limitacion sin
     evidencia pairwise), se acepta que P38 permanezca GO CONDICIONADO
     sin cierre empirico completo?

# Anexos

- Resolver: `src/institutional_accumulation/sec_13f/identity/security_identity.py`
- Crosswalk: `data/mappings/etf_holdings.csv` + `cusip_ticker_exceptions.csv`
- Catalogo: `data/mappings/radar_target_catalog.csv`
- Probe A.6.4: `evidence/a64_integration_b1_p61_p38/`
- Expediente B-02: `B02_EXPEDIENTE.md`
- Consulta principal: `A67_CONSULTA.md`

---

Fin del expediente B-03 + B-04. HEAD al redactar: `e07f449`. 2026-09-22.
