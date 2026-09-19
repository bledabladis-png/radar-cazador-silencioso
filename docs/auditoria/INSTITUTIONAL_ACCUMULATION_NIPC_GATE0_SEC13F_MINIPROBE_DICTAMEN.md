# DICTAMEN DEL AUDITOR EXTERNO - MINI-PROBE CORRECTIVO Q4 2025

**Referencia:** IAE / NIPC / SEC 13(f)
**Resultado:** **PASS**
**Estado:** **Gate 0 SEC 13(f) CERRADO**
**Siguiente paso:** **GO para Gate-NIPC.1 - especificacion, sin codigo productivo**

El mini-probe ha resuelto la unica inconsistencia que impedia cerrar el Gate 0: **el TXT Q4 2025 publicado por SEC es un artefacto reducido/subconjunto respecto del PDF**, mientras que el PDF contiene la lista completa de instrumentos, incluidas las opciones. La SEC confirma oficialmente que publica PDF y TXT para Q4 2025 y Q1 2026, y que el formato TXT utiliza CUSIP, option indicator, issuer name, issuer description y `STATUS`.

Hay, no obstante, **dos precisiones documentales que deben quedar congeladas**.

---

## Q-MINI-1 - Se confirma el hallazgo?

### **SI**

La caracterizacion queda aceptada:

    Q4 2025 TXT
        v
    12,282 registros base

mientras:

    Q4 2025 PDF
        v
    lista completa
        v
    incluye CALL / PUT / warrants / units / rights...

La propia lista PDF Q4 muestra, por ejemplo:

    B38564108 * CMB.TECH NV SHS
    B38564908   CMB.TECH NV CALL
    B38564958   CMB.TECH NV PUT

y Apple aparece bajo el mismo patron.

Ademas, la hoja de informacion oficial explica que un asterisco identifica una security que tiene una opcion listada y que cada opcion se lista individualmente con su propio CUSIP.

### Matiz terminologico

No afirmaria:

> "SEC denomina al TXT Q4 formato reducido".

Eso no aparece como etiqueta oficial de SEC.

Si debe decirse:

> **"El TXT Q4 2025 publicado por SEC es, empiricamente, un subconjunto de los registros del PDF Q4 2025, limitado al conjunto base observado en este probe."**

Eso es demostrable y suficientemente preciso.

**Q-MINI-1 = CONFIRMADO.**

---

# Q-MINI-2 - Se aprueba opcion A?

## **SI, para NIPC**

Se aprueba:

    Q4 2025 TXT
        +
    Q1 2026 TXT

para construir la dimension:

    section13f_eligible

**pero aplicando `STATUS`, no simplemente membership bruto.**

La SEC define:

* `ADDED` -> la security pasa a ser Section 13(f);
* `DELETED` -> deja de ser Section 13(f) desde la lista anterior.

Por tanto:

    STATUS = blank
        -> eligible

    STATUS = *A*
        -> eligible

    STATUS = *D*
        -> NOT eligible

Y:

    mismo CUSIP
    + estados incompatibles
    -> CONFLICT / REVIEW_REQUIRED

### PDF Q4

Queda como **evidencia auxiliar**, no como input operativo de NIPC.

Eso es una decision limpia:

    TXT
     v
    eligibility operacional

    PDF
     v
    auditoria / evidencia completa

Y, dado que NIPC ya excluye PUTCALL != null, la ausencia de filas CALL/PUT del TXT Q4 **no afecta al calculo de elegibilidad del universo canonico NIPC**.

**Q-MINI-2 = GO.**

---

# Q-MINI-3 - Layout 68-70 / STATUS

## **SI - RATIFICADO**

La especificacion oficial SEC dice:

    01-09  CUSIP
    10     Option Indicator
    11-40  Issuer Name
    41-67  Issuer Description
    68-70  Status
    71-79  Blank
    80     Misc / Unused

y define `STATUS` mediante:

    *A* = additions
    *D* = deletions
    blank = no change

Por tanto, desaparece definitivamente del diseno cualquier interpretacion separada de:

    pos 68 = format flag
    pos 69 = A/D

como campos independientes.

Es: status = raw[67:70]

y despues: *A* / *D* / blank.

### Posicion 80

Se mantiene: DO NOT INTERPRET.

La SEC la define como `Misc` / `Unused`.

**Q-MINI-3 = CONFIRMADO.**

---

# Q-MINI-4 - Cerrar Gate 0 SEC 13(f)?

## **SI - PASS**

Las dimensiones quedan suficientemente definidas:

    section13f_eligible
    security_type
    option_status
    option_indicator_star
    status

y el control de opciones del NIPC queda:

    SSHPRNAMTTYPE == "SH"
    AND
    PUTCALL IS NULL

No se utilizara Option Indicator = * para clasificar una linea como opcion.

La hoja SEC confirma precisamente que `*` significa que la security tiene una opcion listada, mientras que las opciones se identifican individualmente en la lista y en Form 13F mediante el tratamiento especifico correspondiente.

### Gate cerrado

**GATE 0 SEC 13(f) = PASS / CLOSED.**

---

# Q-MINI-5 - Investigar Q3/Q2?

## **NO necesario**

Lo difiero.

Ya tenemos evidencia oficial suficiente de que el modelo de opciones no nacio en Q1 2026: el PDF Q4 2025 contiene explicitamente underlying * / CALL / PUT, y la propia SEC describe esta semantica como caracteristica de la Official List.

Investigar Q3/Q2 seria redundante para la decision arquitectonica actual.

Puede quedar como **evidencia historica opcional**, no como condicion de Gate-NIPC.1.

**Q-MINI-5 = DIFERIDO / NO BLOQUEANTE.**

---

# Q-MINI-6 - Implementar soporte PDF ahora?

## **NO - DOCUMENTAR COMO DEUDA**

No recomiendo incorporar ahora un parser PDF.

La situacion actual permite:

    Q4 2025 -> TXT para eligibility -> PDF conservado como evidencia
    Q1 2026 -> TXT para eligibility

Eso es suficiente para NIPC.

Si debe documentarse una deuda arquitectonica:

    FUTURE_FORMAT_VARIANT:
    Official List may require PDF ingestion
    if SEC does not publish a sufficiently complete TXT artifact.

Cuando aparezca un trimestre futuro sin TXT utilizable, se abre entonces el diseno especifico.

No debemos introducir ahora PDF parser / OCR / layout inference para resolver un problema que actualmente no bloquea el producto.

**Q-MINI-6 = DOCUMENTAR, NO IMPLEMENTAR.**

---

# 1. Correccion final al informe original

Hay una frase que debe quedar rectificada:

> "Q4 2025 usa el flag `*` como 'opciones disponibles'; Q1 2026 emite filas CALL/PUT separadas."

### **NO debe permanecer asi.**

La conclusion correcta es:

> **La Official List Q4 2025 ya utiliza el modelo `underlying + * + CALL/PUT`; el TXT Q4 2025 publicado es simplemente un subconjunto reducido que no contiene las filas CALL/PUT presentes en el PDF. Q1 2026 dispone de un TXT que si contiene esas filas.**

Esto queda respaldado por el PDF oficial Q4 y por el TXT oficial Q4.

---

# 2. Arquitectura SEC 13(f) congelada

                     Official List
                          |
              +-----------+-----------+
              |                       |
            STATUS                OPTION INDICATOR
              |                       |
              v                       v
         eligibility                 metadata only
              |
              v
        security_type
              |
              v
    13F canonical line
     SH + PUTCALL NULL
              |
              v
          NIPC

Y las opciones:

    Official List
        v
    CALL / PUT entries

son un fenomeno independiente del `*`.

---

# 3. Estado final

| Punto                       | Dictamen                             |
| --------------------------- | ------------------------------------ |
| Q-MINI-1                    | **CONFIRMADO**                       |
| TXT Q4 reducido/subconjunto | **CONFIRMADO empiricamente**         |
| Q-MINI-2                    | **GO**                               |
| TXT Q4 para NIPC            | **APROBADO**                         |
| PDF Q4                      | **evidencia auxiliar**               |
| Q-MINI-3                    | **CONFIRMADO**                       |
| `STATUS` 68-70              | **campo unico**                      |
| Pos. 80                     | **UNUSED / no interpretar**          |
| Q-MINI-4                    | **PASS**                             |
| Gate 0 SEC 13(f)            | **CLOSED / PASS**                    |
| Q-MINI-5                    | **DIFERIDO**                         |
| Q-MINI-6                    | **documentar deuda, no implementar** |
| Gate-NIPC.1                 | **GO**                               |
| Codigo NIPC                 | **todavia NO**                       |

# DICTAMEN FORMAL

## **GATE 0 SEC 13(f) -> PASS / CLOSED**

Queda aprobado el uso conjunto:

    Q4 2025 TXT
    Q1 2026 TXT

para `section13f_eligible`, con resolucion por `STATUS` y conservando los PDF como evidencia cuando existan.

La SEC confirma ademas que la lista es trimestral, que Q4 2025 y Q1 2026 tienen publicaciones TXT/PDF y que `STATUS`, `Option Indicator` y la estructura fija de 80 posiciones estan formalmente definidos.

## **Siguiente paso: Gate-NIPC.1 - especificacion, sin codigo.**

Ahi ya podemos fijar definitivamente la **capa multicriterio de security identity**:

    CUSIP
    -> section13f_eligible
    -> security_type
    -> mapping
    -> temporal_validity
    -> radar intersection
    -> coverage
    -> NIPC status

**No se autoriza todavia `delta_shares.py` ni `nipc.py`.**
