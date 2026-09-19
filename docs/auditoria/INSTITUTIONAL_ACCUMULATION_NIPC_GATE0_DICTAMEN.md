# DICTAMEN DEL AUDITOR EXTERNO

## GATE 0 - NIPC (Net Institutional Position Change)

**Fecha:** 19-09-2026
**Referencia:** IAE / SEC 13F / NIPC
**Resultado:** **GO CONDICIONADO PARA GATE-NIPC.1 - ESPECIFICACION SIN CODIGO**
**Implementacion:** NO autorizada todavia
**Push:** NO
**NIPC:** desbloqueado para diseno propio, no para publicacion productiva

---

## 1. Dictamen ejecutivo

El Gate 0 empirico queda **APROBADO COMO BASE PARA DISENO**.

Los dos periodos necesarios estan disponibles:

    2025Q4 -> baseline
    2026Q1 -> target

y ambos contienen snapshots 13F suficientemente estructurados para construir un calculo de variacion trimestral.

La SEC establece que Form 13F contiene informacion de posiciones y que el Information Table identifica `CUSIP`, `SSHPRNAMT`, `SSHPRNAMTTYPE`, `PUTCALL`, `INVESTMENTDISCRETION` y `OTHERMANAGER`. Tambien establece que las posiciones cortas no se reportan ni deben restarse de las largas.

**Pero NIPC no debe presentarse como "compras institucionales".**

La metrica sera:

    variacion trimestral de las acciones largas reportadas en Form 13F

Es una diferencia entre dos snapshots regulatorios; no demuestra por si sola una secuencia concreta de compras/ventas durante el trimestre.

---

## 2. Q-NIPC-1 - Q4 2025 como baseline

### **GO**

Se autoriza:

    2025Q4
       v
    baseline
       v
    2026Q1
       v
    DeltaShares

El requisito de disponer del trimestre anterior queda satisfecho.

No es necesario retrasar NIPC.

El hecho de utilizar Q4 2025 no altera la naturaleza de la metrica: Q4 es el snapshot anterior y Q1 el posterior.

---

## 3. Q-NIPC-2 - Unidad de agregacion

### **DECISION: CORRECCION ARQUITECTONICA**

No apruebo ninguna de las dos alternativas tal como estan formuladas:

    canonical_reporting_relationship

como unidad directamente aditiva, ni:

    filing_manager_cik

como identidad economica absoluta.

### Razon

La SEC explica que Column 7 identifica otro manager con quien el filer comparte investment discretion y vincula las securities de esa linea con ese manager. Tambien indica que puede haber mas de un manager incluido.

Por tanto:

    1 source line
          v
    N relationship edges

pero:

    1 source line
          v
    1 cantidad SSHPRNAMT

No existe una cantidad separada de shares para cada edge.

Por consiguiente, si se hiciera:

    Sum shares por canonical_reporting_relationship_key

una linea con tres managers podria aportar las mismas acciones tres veces.

Eso seria incompatible con la regla ya fijada de FA-2.3:

> **multi-edge relacional != multiplicacion economica del holding.**

## Unidad aprobada para NIPC

Propongo formalmente introducir una nueva entidad logica:

    reported_position_unit
    =
    (report_period,
     filing_manager_cik,
     canonical_security,
     discretion_type)

La cantidad `SSHPRNAMT` entra **una sola vez por source line**.

El `canonical_reporting_relationship_key` queda como:

    evidence / provenance dimension

y no como unidad automatica de suma.

### Caso multi-manager

Cuando:

    OTHERMANAGER = "1,2,3"

la posicion conserva:

    SSHPRNAMT = X

una sola vez.

Se generan:

    edge 1
    edge 2
    edge 3

para el grafo, pero no:

    X + X + X

para NIPC.

### Consecuencia

No autorizo todavia atribuir esas `X` acciones individualmente a los tres managers.

Eso requeriria una regla de asignacion que la fuente no proporciona.

**Q-NIPC-2 = GO condicionado a esta separacion.**

---

## 4. Q-NIPC-3 - INVESTMENTDISCRETION

### **GO - incluir SOLE + DFND + OTR**

No recomiendo:

    solo SOLE

ni:

    SOLE + OTR

La SEC define `SOLE`, `DEFINED` (representado en vuestro dataset como `DFND`) y `OTHER` (`OTR`) como categorias diferentes de investment discretion. En particular, `DEFINED` cubre relaciones de control/entidad y `OTHER` cubre otras formas de shared investment discretion.

Ademas, la SEC indica expresamente que puede haber varias lineas para la misma clase de security precisamente porque cambia el tipo de investment discretion.

Por tanto:

    NIPC
     |-- SOLE
     |-- DFND
     +-- OTR

debe ser el universo base.

### No significa tratarlos como identicos

El modulo debe conservar:

    discretion_type

y permitir:

    NIPC_total
    NIPC_SOLE
    NIPC_DFND
    NIPC_OTR

sin perder la separacion.

**DFND no debe excluirse arbitrariamente.**

---

## 5. Q-NIPC-4 - Universo NIPC

### **DICTAMEN: opcion (a) como contrato operativo, pero NO con la cobertura actual**

El contrato mantiene:

    radar_equities
    INTERSECT
    13F_eligible
    INTERSECT
    mapped

Eso sigue siendo correcto para el **NIPC productivo del radar**.

No autorizo sustituirlo por:

    24,838 CUSIPs

como universo operativo, porque romperia el vinculo con el universo radar.

Tampoco considero adecuado publicar NIPC sobre solo:

    3 CUSIPs

porque seria unicamente una prueba tecnica.

### Por tanto

Se crean conceptualmente dos capas:

    TECHNICAL UNIVERSE
    13F eligible securities
            v
    diagnostico del motor

y:

    OPERATIONAL UNIVERSE
    radar equities
    INTERSECT
    13F eligible
    INTERSECT
    mapped
            v
    NIPC publicable

El primer universo sirve para pruebas internas.

El segundo es el unico que puede alimentar el radar.

---

## 6. Q-NIPC-5 - Coverage

### **DICTAMEN: INSUFFICIENT - no publicar NIPC operativo actualmente**

La cobertura actual:

    3 / 24,838 = 0.012%

es manifiestamente insuficiente para representar el universo 13F.

No apruebo:

    0.012%
       v
    NIPC productivo

ni:

    3 securities
       v
    indicador institucional

Eso seria una falsa apariencia de cobertura.

## Tampoco autorizo todavia una fuente externa concreta

No debe incorporarse OpenFIGI, CUSIP Global Services u otra fuente simplemente para subir artificialmente el porcentaje.

Primero debe existir en **Gate-NIPC.1** una especificacion de:

    security mapping source
    authority
    coverage
    temporal validity
    conflict resolution
    quality controls

### Regla de publicacion

NIPC debera tener:

    mapping_coverage
    +
    institutional_weight_coverage

y ambas deberan satisfacer los controles contractuales.

Mientras no lo hagan:

    NIPC status = INSUFFICIENT

La metrica puede calcularse en **modo diagnostico**, pero no debe publicarse como senal operativa.

**Q-NIPC-5 = GO condicionado a diseno de coverage; estado actual = INSUFFICIENT.**

---

## 7. Q-NIPC-6 - Scope del primer ciclo

### **GO para opcion (a)**

El primer ciclo debe limitarse a:

    DeltaShares
    +
    NIPC
    +
    coverage
    +
    status

No introducir todavia:

    Institutional Breadth
    New / Exit
    classification

Esas capas necesitan sus propios controles y no son necesarias para cerrar la deuda NIPC.

Por tanto:

    Gate-NIPC.2
        v
    delta_shares
        v
    nipc

y nada mas.

**Q-NIPC-6 = GO.**

---

## 8. Q-NIPC-7 - Ubicacion

### **GO - opcion A**

Apruebo:

    src/institutional_accumulation/aggregation/

con responsabilidad separada de:

    identity/

Arquitectura:

    institutional_accumulation/
    +-- sec_13f/
    +-- identity/
    +-- aggregation/
        +-- delta_shares.py
        +-- nipc.py

La separacion es limpia:

    identity
        v
    determina quien / que / cuando

    aggregation
        v
    calcula variaciones

No anadiria todavia `breadth.py` ni `new_exit.py`.

**Q-NIPC-7 = GO.**

---

## 9. Regla canonica de DeltaShares

Se mantiene:

    SSHPRNAMTTYPE == "SH"
    AND
    PUTCALL IS NULL

La exclusion de opciones es correcta.

La SEC explica que, en una opcion, el CUSIP de las columnas del instrumento puede corresponder al **underlying stock**, pero `PUT/CALL` identifica que la posicion es una opcion. Por tanto, utilizar unicamente el CUSIP sin respetar `PUTCALL` podria convertir una opcion en una falsa posicion accionarial.

Asimismo, las posiciones `PRN` quedan fuera del NIPC de acciones por la condicion contractual `SH`.

---

## 10. Una precision critica sobre `DFND`

No debe interpretarse:

    DFND
    =
    manager no controla nada

La SEC lo define como **shared-defined investment discretion**, incluida la relacion entre entidades controladoras/controladas.

Por tanto, el tratamiento correcto es:

    DFND
    =
    posicion reportada bajo shared-defined discretion

No:

    DFND
    =
    posicion invalida

ni:

    DFND
    =
    posicion no institucional

---

## 11. NIPC y la limitacion estructural de 13F

El modulo debe documentar explicitamente:

    NIPC != institutional trading flow

mas exactamente:

    NIPC
    =
    change in reported long shares
    between two 13F reporting periods

La SEC establece ademas que los shorts no se incluyen ni se netean contra los longs.

Por tanto:

    DeltaShares < 0

significa:

> disminucion de acciones largas reportadas

y no necesariamente:

> venta neta institucional total

porque el universo 13F no representa todas las posiciones economicas ni las posiciones cortas.

---

## 12. Gate-NIPC.0 - Estado final

| Decision                           | Dictamen                          |
| ---------------------------------- | --------------------------------- |
| Q-NIPC-1 Q4 baseline               | **GO**                            |
| Q-NIPC-2 aggregation               | **GO condicionado**               |
| Relationship key como suma         | **NO**                            |
| Reported position unit             | **GO**                            |
| Multi-edge economico               | **NO duplicar**                   |
| Q-NIPC-3 discretion                | **SOLE + DFND + OTR**             |
| Q-NIPC-4 universo                  | **radar ^ 13F eligible ^ mapped** |
| 13F completo como output operativo | **NO**                            |
| 3 CUSIPs como output operativo     | **NO**                            |
| Q-NIPC-5 coverage actual           | **INSUFFICIENT**                  |
| External mapping                   | **pendiente de especificacion**   |
| Q-NIPC-6 scope                     | **DeltaShares + NIPC + coverage** |
| Q-NIPC-7 arquitectura              | **aggregation/**                  |
| Codigo NIPC                        | **NO autorizado todavia**         |
| Gate-NIPC.1                        | **AUTORIZADO**                    |
| Push                               | **NO**                            |

---

# DICTAMEN FORMAL

## **GATE-NIPC.0 -> PASS CONDICIONADO / GO PARA GATE-NIPC.1**

La deuda NIPC queda correctamente desbloqueada y existe evidencia suficiente para pasar a **especificacion formal**, pero **todavia no a implementacion**.

El principal ajuste arquitectonico queda congelado asi:

    canonical_reporting_relationship_key
            =
    IDENTIDAD / PROVENANCE

    reported_position_unit
            =
    UNIDAD DE AGREGACION DE SHARES

y:

    1 source line
       v
    1 cantidad SSHPRNAMT
       v
    N reporting edges

**Nunca:**

    1 source line
       v
    N x SSHPRNAMT

La SEC confirma que Column 7 enlaza securities con otros managers con quienes se comparte investment discretion y que estos managers se identifican mediante la lista de Other Included Managers; esa relacion no proporciona una distribucion individual de las acciones de una linea entre los managers enlazados.

### Proximo paso autorizado

    Gate-NIPC.1
       v
    cerrar contrato interno:
      - reported_position_unit
      - reglas multi-edge
      - discretion
      - universe
      - mapping layer
      - coverage thresholds
      - status INSUFFICIENT
       v
    SIN CODIGO

**No iniciar `delta_shares.py` ni `nipc.py` hasta cerrar Gate-NIPC.1.**

El punto que debe resolverse primero en ese Gate es la **capa masiva CUSIP -> security/ticker del universo radar**, porque con el crosswalk actual de 3 excepciones el NIPC productivo no tendria cobertura suficiente.
