# HALLAZGO 5c - DISTRIBUTION v1.4 sigue inalcanzable

**Documento de auditoria. Requiere dictamen externo.**
**Fecha:** 2026-10-02.
**Precedente:** dictamen 5c (H1 matematicamente demostrado en v1.3).

---

## 1. Contexto

v1.4 corrigio la contradiccion algebraica de v1.3:

    c_norm > 0.30  ->  c_norm > -0.10

Verificacion algebraica: con t_norm=-0.30, c_norm=-0.10, struct=-0.22
< -0.10. Cumple. La conjuncion ya no es imposible.

**Pero en el universo real, DISTRIBUTION sigue dando 0/316.**

---

## 2. Diagnostico empirico

### 2.1. Embudo de condiciones

| Filtro | Tickers |
|---|---:|
| `struct_score_t < -0.10` | 40 |
| `+ t_norm en banda (>0 o |t|<0.30)` | 24 |
| `+ c_norm > -0.10` | **3** |
| `+ struct_max > 0.30` (las 4) | **0** |

**Dos cuellos de botella:**

1. **`c_norm > -0.10` filtra 21 de 24 tickers.** De los 24 con deterioro + t en banda, solo 3 tienen `c_norm > -0.10`.

2. **Precedente fuerte elimina los 3 restantes.** De esos 3, ninguno tiene `struct_max > 0.30`.

### 2.2. Distribucion de `c_norm` en deterioro

Sobre los 24 tickers con `struct < -0.10` y `t_norm` en banda:

| Metrica | Valor |
|---|---:|
| min | -1.000 |
| p25 | -0.598 |
| mediana | **-0.410** |
| p75 | -0.263 |
| max | -0.005 |
| `c_norm > -0.10` | 3 de 24 |

**Lectura:** cuando el precio esta en deterioro, `c_norm` es **casi siempre negativo** (mediana -0.41). Razon estructural: la volatilidad se expande cuando el precio gira. `c_norm > -0.10` implica compresion, incompatible con el momento de deterioro.

**Correlacion negativa entre "deterioro" y "compresion".** No son independientes.

---

## 3. Naturaleza del hallazgo

El §5.4 actual exige **cuatro condiciones conjuntas** con dos de ellas
correlacionadas negativamente en el universo real:

    struct_score_t < -0.10        (deterioro)
    AND c_norm_t > -0.10          (compresion)

En el momento de deterioro, la volatilidad ya esta expandiendo. La
segunda condicion elimina casi todos los casos.

Y la tercera condicion adicional (precedente fuerte):
    
    AND struct_max > 0.30

elimina los pocos que pasan el filtro de compresion, porque si el
precedente fue fuerte y ahora hay deterioro, el deterioro suele ser
**reciente y severo** (con volatilidad expansionada, c_norm < -0.10).

**Conclusion:** el contrato §5.4 esta mal disenado en su estructura,
no en su umbral. Pedir compresion durante el deterioro es
contradictorio con la microestructura observada.

---

## 4. Diferencias con el hallazgo anterior (H1)

**H1 v1.3:** las condiciones eran algebraicamente imposibles (contradiccion matematica interna). Corregido en v1.4.

**H1 v1.4:** las condiciones son algebraicamente posibles (el test I25 lo demuestra), pero **empiricamente casi nunca coexisten** porque hay correlacion negativa entre dos de ellas. Es un defecto de modelado, no de aritmetica.

---

## 5. Opciones de solucion (a dictamen del auditor)

### O1 - Eliminar `c_norm` de DISTRIBUTION

    struct_score_t < -0.10
    AND (t_norm_t > 0 OR |t_norm_t| < T_NORM_WEAK)
    AND struct_max > 0.30

Deja 3 condiciones. La "compresion en techo" ya no es requisito.

**Ventaja:** simple, coherente con la evidencia.

**Inconveniente:** pierde un discriminador.

### O2 - Relajar `c_norm` a "no expansion extrema"

    c_norm_t > -0.50

Cubre mediana (-0.41) y p75 (-0.26). Deja fuera colas extremas.

**Ventaja:** mantiene un limite inferior (evita clasificar como DISTRIBUTION tickers con expansion muy fuerte).

**Inconveniente:** umbral arbitrario.

### O3 - Reformular DISTRIBUTION con un criterio de "techo" no basado en compresion

En Wyckoff, DISTRIBUTION es un rango de techo tras una subida. El
criterio deberia ser:

    precedente fuerte (struct_max alto)
    AND struct_t ha caido respecto al maximo (deterioro)
    AND t_norm aun no es bajista (no MARKDOWN)

Sin condicion de `c_norm`.

**Ventaja:** mas alineado con Wyckoff (Phase D de distribucion).

**Inconveniente:** introduce nuevo criterio relativo (struct_t vs struct_max), no un umbral absoluto.

### O4 - Definir DISTRIBUTION como "MARKUP en perdida de fuerza"

    struct_max > 0.30                      (fue fuerte)
    AND struct_t < 0.30 - X                (perdio fuerza)
    AND t_norm_t >= -T_NORM_STRONG         (no es bajista aun)

Donde X es la caida minima desde el maximo.

**Ventaja:** captura el concepto de "transicion desde arriba".

**Inconveniente:** introduce parametro X.

---

## 6. Recomendacion tecnica (a validar)

**O1 + O3 combinados:**

    struct_score_t < -0.10
    AND (t_norm_t > 0 OR |t_norm_t| < T_NORM_WEAK)
    AND struct_max > 0.30

Es decir:

- Precedente fuerte (fue fuerte recientemente).
- Deterioro actual (ya no lo es).
- Aun no bajista (t_norm no cumple MARKDOWN).
- **Sin requisito de compresion.**

**Razon:** la compresion es incompatible con el momento de deterioro.
Eliminarla es coherente con la evidencia empirica.

Con esta definicion, los 24 tickers con struct < -0.10 y t en banda
podrian entrar a DISTRIBUTION si su struct_max > 0.30. Verificar cuantos
cumplen eso es la siguiente iteracion de diagnostico.

---

## 7. Preguntas al auditor

1. **¿Se aprueba O1** (eliminar `c_norm` de DISTRIBUTION)?
2. **¿O prefieres O2** (relajar `c_norm` a un valor tipo -0.50)?
3. **¿O O3/O4** (reformular con criterio de transicion desde arriba)?
4. **¿Debe el §5.2 ACCUMULATION revisarse** por la misma razon?
   Alli se exige `c_norm > 0.30` (compresion alta), que es coherente
   con "base silenciosa", pero conviene confirmar.
5. **¿Re-ejecutar 5c.2 tras corregir §5.4**, o cerrar 5c con la
   distribucion actual?

---

## 8. Estado

- v1.4: implementada (fix algebraico).
- DISTRIBUTION: 0/316 empirico (defecto estructural de §5.4).
- RANGE: 211/316 (genuino, confirmado en dictamen 5c).
- H3 (5 casos): verificados, v1.3 correcto.
- 5d: bloqueada hasta resolver DISTRIBUTION.

---

Fin del hallazgo.
