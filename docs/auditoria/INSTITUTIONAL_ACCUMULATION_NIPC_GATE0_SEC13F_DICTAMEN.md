# DICTAMEN DEL AUDITOR EXTERNO - GATE 0 SEC 13(F)

**Fecha:** 19-09-2026
**Referencia:** IAE / NIPC / SEC 13(f) Official List
**HEAD revisado:** `35e411f` local
**Resultado:** GO CONDICIONADO - caracterizacion de formato corregida; Gate 0 SEC 13(f) continua
**Codigo NIPC:** NO autorizado

---

# 0. Resumen de correcciones

Este dictamen corrige dos lecturas erroneas del probe anterior y fija
la interpretacion canonica del fichero SEC 13(f) Official List:

1. Option indicator esta en posicion 10 (`*` o espacio), NO un `*` sobre el
   CUSIP.
2. El ultimo caracter (posicion 80) NO es el Status. El Status esta en
   posiciones 68-70.
3. Los duplicados en Q1 2026 son legitimos (mismo CUSIP con alta y baja
   marcadas).
4. El caracter final (posicion 80) es un campo distinto, no identificado
   en este dictamen.

---

# 1. Correccion 1 - Option Indicator

## Formato correcto

    Posicion 10 = Option Indicator
    Valores: `*` o espacio.

## Ejemplos reales

    B38564108*CMB.TECH NV...
    ^ posicion 10 = `*`

    B5950S113 MDXHEALTH SA...
    ^ posicion 10 = espacio

## Consecuencia

NO debe interpretarse el `*` como parte del CUSIP ni como un flag
adyacente al CUSIP. Es una columna independiente que indica si el
instrumento listado es una opcion.

## Implicacion para NIPC

    CUSIP (pos 1-9)
    Option Indicator (pos 10)

son campos separados. NIPC ya filtra por `PUTCALL IS NULL` en INFOTABLE,
lo cual excluye opciones. El Option Indicator de la lista SEC sirve como
control cruzado: si el CUSIP observado en 13F tiene Option Indicator en
la lista, es coherente con lo que se espera.

---

# 2. Correccion 2 - Status (posiciones 68-70)

## Formato correcto

    Posicion 68: `*` o espacio.
      `*` = SEC marca el CUSIP por cambio de formato en el fichero.
    Posicion 69: `A`, `D` o espacio.
      `A` = alta (CUSIP anadido en el trimestre).
      `D` = baja (CUSIP retirado en el trimestre).
    Posicion 70: sin contenido relevante observado.

## Ejemplo

    98986M103*ZYNEX INC ... COM  *D* E

    Posicion 68: `*`
    Posicion 69: `D`
    -> CUSIP marcado por cambio de formato y baja en el trimestre.

## Implicacion

El Status permite reconstruir la evolucion de la lista entre trimestres:
CUSIPs que entran, salen, o cambian de formato. Esto es dimension
distinta de `section13f_eligible` (que es un booleano sobre la lista).

---

# 3. Correccion 3 - El ultimo caracter (posicion 80)

## Observacion

En Q4 2025: distribucion de ultimo caracter = `E`, `P`, `N`, `S`, `U`,
`A`, `%`.

En Q1 2026: todos los registros tienen `E`.

## Interpretacion

El campo de posicion 80 NO es Status. Es un campo distinto.

Posibles interpretaciones sin confirmar:

  (a) Indicador de tipo de instrumento a nivel SEC.
  (b) Flag de mercado/exchange.
  (c) Indicador de seccion normativa (13(f) vs otra).
  (d) Check-digit extendido o checksum.

## Recomendacion

NO interpretar la posicion 80 en Gate-NIPC.1 hasta tener documentacion
oficial SEC que la defina. Si se requiere su semantica, solicitar
confirmacion al SEC directamente o via FAQ.

---

# 4. Correccion 4 - Duplicados en Q1 2026

## Observacion

Q1 2026 tiene 994 CUSIPs con >1 linea (1,982 lineas adicionales).

Los pares tienen forma:

    M8216Q909 LIFEWARD LTD CALL *D* E
    M8216Q909 LIFEWARD LTD CALL *A* E

Es decir: mismo CUSIP, una linea con `D` (baja) y otra con `A` (alta).

## Interpretacion

Es un registro de eventos entre trimestres. El CUSIP aparece dos veces
en Q1 porque fue dado de baja y luego de alta en el mismo trimestre
(cambio de formato, reclasificacion, emisor renombrado, etc.).

## Implicacion para Gate-NIPC.1

La logica de `section13f_eligible` debe decidir la interpretacion:

  (a) `eligible = any(A)` (union) -> CUSIP elegible si tiene al menos
      un registro `A` vigente.
  (b) `eligible = last(D vs A)` (por orden en el fichero) -> CUSIP
      elegible si el ultimo estado es A.
  (c) `eligible = A and not D` (interseccion estricta).

Recomendacion del ingeniero: (b) por orden, porque el fichero SEC lista
los eventos en orden cronologico dentro del trimestre.

---

# 5. Formato canonico del fichero SEC 13(f) Official List

## Layout confirmado

    Posicion  Longitud  Campo
    --------  --------  -------------------------------------
    1-9       9         CUSIP
    10        1         Option Indicator (`*` o espacio)
    11-40     30        Issuer Name
    41-67     27        Issuer Description
    68        1         Format Change Flag (`*` o espacio)
    69        1         Add/Del Indicator (`A`, `D` o espacio)
    70        1         (reservado / sin contenido observado)
    71-79     9         (no documentado)
    80        1         Campo distinto no identificado

## Validacion

- 100% de las lineas tienen 80 caracteres.
- 0 lineas vacias.
- Ningun registro rompe el patron.

---

# 6. Cifras corregidas

## Q4 2025 (13flist2025q4.txt)

- 12,282 lineas.
- 12,282 CUSIPs unicos.
- 0 duplicados.
- 6,300 con Option Indicator = espacio.
- 5,982 con Option Indicator = `*`.
- Status (pos 68-69): 1,232 con `*` en pos 68; 787 con `A` en pos 69;
  445 con `D` en pos 69.

## Q1 2026 (13flist2026q1.txt)

- 24,641 lineas.
- 22,659 CUSIPs unicos.
- 1,982 lineas duplicadas (994 CUSIPs con par A/D).
- 18,627 con Option Indicator = espacio.
- 6,014 con Option Indicator = `*`.
- Status (pos 68-69): 1,819 con `*`; 1,022 con `A`; 797 con `D`.

## Observacion

El incremento de tamano Q1 vs Q4 no es solo por evento A/D: tambien
porque Q1 lista opciones como filas separadas (CALL, PUT) mientras Q4
las agrupa bajo el CUSIP base con Option Indicator = `*`.

---

# 7. Lo que NO debe hacerse

- NO interpretar la posicion 80 hasta que la SEC la documente.
- NO fusionar Option Indicator (`*` pos 10) con Status (`*` pos 68).
- NO colapsar los duplicados A/D de Q1 sin decidir la politica de
  resolucion.
- NO asumir que Q1 tiene "el doble de CUSIPs" que Q4: son 22,659 unicos
  vs 12,282 unicos; el resto son eventos A/D + opciones listadas.
- NO tratar `section13f_eligible` como sinónimo de `security_type=EQUITY`.
  Son dimensiones independientes (Q-NIPC-8).
- NO mezclar esta correccion con `Gate-NIPC.1`. Gate-NIPC.1 sigue
  pendiente.

---

# 8. Proximo paso autorizado

Continuar Gate 0 SEC 13(f):

1. Parsear el fichero con el layout canonico corregido.
2. Calcular para Q4 2025 y Q1 2026:
   - `eligible` por CUSIP (resolucion de A/D).
   - Option Indicator por CUSIP.
   - Status por CUSIP.
3. Cruzar con CUSIPs del 13F (canonical SH + null):
   - % CUSIPs en Official List.
   - % ponderado por SSHPRNAMT.
   - Opciones: cuantos `PUTCALL` no-null corresponden a Option Indicator.
4. Informe a auditor.

Sin codigo productivo. Sin mapping masivo. Sin Gate-NIPC.1.

---

# 9. Decisiones consolidadas

| Punto                                | Dictamen                              |
| ------------------------------------ | ------------------------------------- |
| Layout del fichero SEC               | CONFIRMADO (80 chars, LF)             |
| Option Indicator                     | pos 10, `*` o espacio                 |
| Status                               | pos 68-70, `*`/A/D                    |
| Ultimo caracter (pos 80)             | DESCONOCIDO; no interpretar           |
| Duplicados Q1                        | A/D legitimos                         |
| `section13f_eligible`                | resolucion de A/D pendiente           |
| `section13f_eligible != EQUITY`      | CONFIRMADO                            |
| Gate 0 SEC 13(f) continua            | GO                                    |
| Gate-NIPC.1                          | NO iniciar todavia                    |
| Codigo NIPC                          | NO                                    |

---

Fin del dictamen. Version 1.0 (2026-09-19). HEAD local 35e411f.
