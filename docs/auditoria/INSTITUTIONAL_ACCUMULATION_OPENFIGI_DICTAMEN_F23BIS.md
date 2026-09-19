# Dictamen F2.3-bis - OpenFIGI + RADAR_TARGET_CATALOG + TARGET_UNIVERSE

**Objeto:** evaluacion del piloto OpenFIGI + RADAR_TARGET_CATALOG + TARGET_UNIVERSE
**HEAD revisado:** c6d3e0a
**Fase:** F2.3-bis
**Estado global:** GO CONDICIONADO
**H1 arquitectonico:** CERRADO
**OpenFIGI masivo:** NO AUTORIZADO TODAVIA
**F2.4:** NO AUTORIZADA
**Gate-NIPC.2:** BLOQUEADO

---

## 1. Conclusion ejecutiva

La propuesta resuelve correctamente el bloqueo H1 planteado en el dictamen anterior.

La arquitectura ya no es:

    OpenFIGI -> define TARGET -> OpenFIGI -> mide su propia cobertura

sino:

    RADAR 242 tickers -> OpenFIGI -> RADAR_TARGET_CATALOG
                                       |
                                       v
    13F CUSIP -> OpenFIGI ID_CUSIP -> shareClassFIGI
                                       |
                                       v
                              cruce con catalogo -> TARGET MEMBERSHIP

Esta separacion es metodologicamente valida porque la pertenencia al
universo radar nace de los 242 tickers del radar, mientras OpenFIGI
proporciona la identidad que permite conectar esos tickers con las
observaciones 13F.

OpenFIGI documenta el mapping mediante TICKER y ID_CUSIP, y devuelve
FIGI/share-class metadata, por lo que la direccion
CUSIP -> FIGI/shareClassFIGI -> ticker esta soportada por la API.

Ademas, OpenFIGI establece que FIGI permanece intacto frente a cambios
ordinarios de ticker, lo que respalda su utilizacion como identidad
estable; no obstante, el catalogo debe conservar la temporalidad de la
membresia radar, porque identidad estable y pertenencia historica al
radar son conceptos distintos.

### Dictamen

H1 = CERRADO.

Pero: cerrar H1 no equivale todavia a demostrar que el catalogo es
historicamente completo, ni que OpenFIGI sea suficientemente fiable
para establecer los thresholds de NIPC.

---
## 2. RADAR_TARGET_CATALOG - PASS arquitectonico

El resultado: 242 filas, 240 OK, 2 MISS, es consistente con la
arquitectura propuesta.

Los dos MISS: BRK-B -> BRK/B, MOG-A -> MOG/A, son un problema de
normalizacion de ticker, no una ausencia automatica de identidad.

El hecho importante es que el catalogo esta construido a partir de
stock_prices.parquet -> 242 radar tickers -> OpenFIGI, y no desde
etf_holdings / cusip_exceptions / cusip_equivalence.

### Decision

RADAR_TARGET_CATALOG = GO COMO ARQUITECTURA.

Los dos MISS deben permanecer explicitos hasta resolver si son
unicamente diferencias de sintaxis de ticker. No autorizo convertir
silenciosamente BRK-B -> BRK/B ni MOG-A -> MOG/A en equivalencias
permanentes sin evidencia registrada.

---

## 3. Punto critico nuevo - temporalidad del catalogo

Este es ahora el principal hueco metodologico.

El catalogo contiene source_date = 2026-09-19, pero el target que
queremos medir es Q4 2025 y Q1 2026.

OpenFIGI permite utilizar una identidad estable aunque cambie el
ticker. Pero eso no demuestra que una security que forma parte del
radar el 19-09-2026 formara parte del radar en Q4 2025 o Q1 2026.

Hay que separar IDENTITY VALIDITY de RADAR MEMBERSHIP VALIDITY.

Una misma security puede conservar shareClassFIGI mientras entrar o
salir del universo radar.

### Por tanto

La v1.2/v1.3 definitiva debe especificar:

    RADAR_TARGET_CATALOG
        - identity
        - current radar membership
        - historical membership / validity

Si el proposito contractual es utilizar el radar actual como universo
objetivo retrospectivo, eso tambien es valido, pero debe decirse
explicitamente. No debe presentarse como "radar Q4/Q1" si solamente
es el radar actual aplicado retrospectivamente.

### Estado

ISSUE ABIERTO - no bloquea H1, si bloquea thresholds.

---
## 4. TARGET_UNIVERSE - arquitectura correcta, pero falta validacion

El flujo:

    CUSIP_13F -> OpenFIGI ID_CUSIP -> shareClassFIGI
              -> RADAR_TARGET_CATALOG -> target_membership

es correcto.

La SEC confirma que los valores de la Official List son las securities
que pueden reportarse en Form 13F y que la lista se actualiza
trimestralmente.

Por tanto, la cadena completa queda conceptualmente:

    13F observation -> SH + PUTCALL null -> SEC eligible
                    -> CUSIP -> shareClassFIGI -> radar catalog -> target

Esta es una definicion de target mucho mas limpia que la de v1.0.

---

## 5. El piloto TOP 500 no permite todavia fijar coverage

El resultado 142/500 = 28.4% es interesante como diagnostico. No lo
aceptaria todavia como evidencia de que existe una cobertura del 28.4%
del target.

Es una muestra seleccionada por SSHPRNAMT.

La proporcion 242/24.838 ~ 1% no es un benchmark directamente
comparable porque el TOP 500 no es una muestra aleatoria de CUSIPs:
esta deliberadamente concentrada por tamano de posicion.

Por ello la frase "28.4% refleja sobrerrepresentacion natural de large
caps" es una hipotesis razonable, pero todavia no una medicion
independiente.

### Decision

Debe documentarse como OBSERVACION DESCRIPTIVA, no como validacion de
cobertura.

---
## 6. H2 - CUSIP 329882225

Este caso si requiere atencion: CUSIP 329882225, SSHPRNAMT ~ 10.59B,
OpenFIGI = ERROR.

No lo trataria como un detalle menor. El problema no es que haya un
error entre 500 observaciones. El problema es que un ERROR situado
entre los CUSIPs de mayor peso puede tener impacto desproporcionado
en paired_weighted_share_coverage.

Por tanto, antes de thresholds debe registrarse al menos:

    NO_ID / ERROR
        - count
        - SSHPRNAMT total
        - % del weight total
        - top individual failures

El mismo principio se aplica a los 47 NO_ID.

### Decision

Diagnostico prioritario. No bloquea el cierre arquitectonico, pero si
bloquea cualquier conclusion sobre cobertura ponderada hasta conocer
su peso.

---
## 7. shareClassFIGI como clave de enlace

La decision de utilizar shareClassFIGI para enlazar radar ticker <->
CUSIP 13F es tecnicamente coherente.

La documentacion de OpenFIGI define el shareClassFIGI como
identificador a nivel de clase que permite relacionar instrumentos /
composite FIGIs de la misma clase.

Esto es importante porque el ticker puede cambiar mientras FIGI
permanece.

### Pero debe anadirse una regla

Un shareClassFIGI debe ser unico dentro del RADAR_TARGET_CATALOG o,
cuando no lo sea, marcarse como conflict / review.

No puede utilizarse un indice shareClassFIGI -> radar_ticker si
existen multiples candidatos sin documentar como se resuelven.

El informe actual dice que la cobertura es 240/242, pero no presenta
todavia:

    - duplicate shareClassFIGI
    - multiple OpenFIGI hits
    - conflicting candidates

Estos tres contadores deben aparecer en la validacion final del
catalogo.

---
## 8. Independencia: H1 cerrado, pero distinguir dos conceptos

La arquitectura es independiente del crosswalk interno, y eso queda
demostrado.

Pero existe una segunda cuestion:

    CATALOG identity  -> OpenFIGI
    13F resolution    -> OpenFIGI

Eso significa que ambas relaciones dependen de la misma fuente
externa.

No es circularidad contractual, porque TARGET membership nace del
conjunto de tickers radar. Pero si significa que existe dependencia
de proveedor comun.

Ejemplo:

    OpenFIGI asigna incorrectamente shareClassFIGI A al ticker radar
        +
    OpenFIGI devuelve el mismo shareClassFIGI A para un CUSIP
        ->
    target_membership = True

La comparacion podria ser internamente consistente aunque ambas
resoluciones compartan el mismo error.

Por tanto: H1 queda cerrado como independencia respecto del crosswalk
interno, pero la validacion externa de calidad de identidad sigue
siendo un control necesario. Esto debe formar parte del siguiente
gate.

---
## 9. B1 - Fix N-PORT

Mantengo el dictamen anterior: GO CONDICIONADO.

El fix ISIN rows + TICKER rows -> merge HOLDING_ID corrige un bug real
de transformacion. El aumento ticker=0 -> 1104, pairs=0 -> 1063 es
evidencia directa.

El fix puede mantenerse, pero recomiendo cerrar con un test de
regresion especifico.

---

## 10. B5 - autocorreccion

CERRADO.

La eliminacion de los cuatro SERIES_ID inferidos y su sustitucion por
los 11 verificados es suficiente para cerrar el incidente. El propio
informe deja trazabilidad de la contaminacion y de la correccion.

No requiere nuevo probe.

---
## 11. Autorizar TOP 2000?

### Decision: SI, pero con condiciones

Ya no recomiendo otro estudio conceptual. H1 esta suficientemente
resuelto para aumentar la muestra.

Pero el siguiente escalado debe cambiar ligeramente de objetivo:

    TOP 500  -> caracterizacion
    TOP 2000 -> validacion cuantitativa

Debe medir, ademas de target_membership, el peso:

    target_true_shares, target_false_shares,
    no_id_shares, error_shares

y sus porcentajes respectivos. No solo numeros de CUSIPs.

El objetivo del TOP 2000 ya no es "ver mas ejemplos", sino comprobar
si la relacion observada en el TOP 500 es estable al ampliar el
universo ponderado.

---

## 12. No autorizar todavia el universo completo de ~24.838

No es necesario aun. La razon es operacional y metodologica:

    TOP 500 -> TOP 2000 -> evaluar estabilidad -> decidir full run

Si el TOP 2000 muestra cobertura estable, pocos errores, errores de
bajo peso, conflictos practicamente nulos, identidad estable y
comportamiento consistente, entonces el full run queda mucho mejor
fundamentado.

---

## 13. Thresholds

Siguen: THRESHOLD_1 = UNDEFINED, THRESHOLD_2 = UNDEFINED.

No conocemos todavia la magnitud real de TARGET, RESOLVED, UNRESOLVED,
ERROR, PAIRED sobre el universo objetivo completo. El TOP 500 no basta
para convertir una observacion en threshold contractual.

---
## 14. F2.4 - aplicacion de policy v1.2

### Decision: NO AUTORIZADA todavia

La arquitectura v1.2 ya esta practicamente definida, pero antes de
sustituir v1.0 quiero que se incorporen dos correcciones documentales:

    1. RADAR_TARGET_CATALOG:
       identity validity vs radar membership validity.

    2. Weighted identity diagnostics:
       count + SSHPRNAMT para NO_ID/ERROR/conflicts.

Y una precision en la policy:

    OpenFIGI = resolver / fuente de identidad
    (no autoridad de pertenencia al radar)

La autoridad de pertenencia sigue siendo radar_equities = 242 tickers.

---

## 15. Estado formal

    F2.1 Inventory                         PASS / CERRADO
    F2.1-bis Identity sources              PASS / CERRADO

    F2.2 v1.1                              NO GO / HISTORICO
    F2.3 v1.2                              RECHAZADA
    F2.3-bis H1                            CERRADO

    RADAR_TARGET_CATALOG                   GO CONDICIONADO
    Target CUSIP -> SCF                    GO
    OpenFIGI como resolver                 GO
    OpenFIGI como autoridad radar          NO

    N-PORT como fuente auxiliar            GO
    N-PORT como catalogo completo          NO

    TOP 500                                CARACTERIZACION PASS
    TOP 2000                               GO
    FULL OPENFIGI                          NO AUTORIZADO

    Temporalidad catalog                   ISSUE ABIERTO
    NO_ID/ERROR weighted impact            ISSUE ABIERTO

    THRESHOLD_1                            UNDEFINED
    THRESHOLD_2                            UNDEFINED
    F2.4                                   NO AUTORIZADA
    Gate-NIPC.2                            BLOQUEADO
    Gate-NIPC.3                            NO AUTORIZADO

---
## DICTAMEN FINAL

### H1 - CERRADO

La separacion arquitectonica propuesta elimina la circularidad
detectada en v1.0:

    RADAR TICKERS -> TARGET CATALOG
    13F CUSIPS -> OpenFIGI -> shareClassFIGI -> TARGET MEMBERSHIP

El TARGET_CATALOG ya no depende del crosswalk interno cuya cobertura
posteriormente se pretende medir.

### GO - siguiente escalado

Autorizar TOP 2000 CUSIPs para validacion cuantitativa.

### NO GO - full OpenFIGI

No ejecutar todavia las ~24.838 consultas.

### NO GO - thresholds

No fijar THRESHOLD_1/2.

### Correcciones obligatorias antes de F2.4

La policy debe incorporar:

    RADAR identity != radar membership temporal

    y

    NO_ID / ERROR / CONFLICT
      -> contar y ponderar por SSHPRNAMT

Ademas, el PAIRED_WEIGHTED_SHARE_COVERAGE solo debe considerarse listo
para thresholding cuando la muestra ampliada haya demostrado que peso
economico queda fuera por errores de identidad.

---

## Decision de ciclo

F2.3-bis = PASS CONDICIONADO.

El bloqueo arquitectonico queda resuelto. El proyecto entra ahora en
una fase mucho mas limpia:

    TOP 500             OK
         |
         v
    TOP 2000            <- SIGUIENTE
         |
         v
    validacion weighted
         |
         v
    validacion temporal
         |
         v
    full target resolution
         |
         v
    TARGET / RESOLVED / PAIRED
         |
         v
    propuesta de thresholds
         |
         v
    dictamen Gate-NIPC.2

No tocar todavia NIPC_COVERAGE_POLICY.md v1.0, nipc.py, delta_shares.py,
security_identity.py ni C2.

---

Fin del dictamen F2.3-bis. Preservado literalmente para trazabilidad.
HEAD revisado: c6d3e0a. Fecha dictamen: 2026-09-19.