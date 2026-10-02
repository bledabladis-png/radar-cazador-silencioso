# PROPUESTA v1.6 - Separacion CANDIDATE vs CONFIRMED

**Documento de propuesta. Requiere dictamen externo antes de implementar.**
**Fecha:** 2026-10-02.
**Precedente:** dictamen verificacion 5c.2 (2026-10-02).

---

## 1. Contexto

El auditor rechazo aceptar los 5 casos frontera como DISTRIBUTION
confirmada. Motivo: el contrato v1.5 confunde dos historias estructurales
incompatibles:

    A) Distribution temprana
       subida previa -> techo/rango -> deterioro -> oferta domina

    B) Correccion dentro de tendencia alcista
       subida previa -> pullback -> continuacion esperada

Ambas cumplen las condiciones actuales de v1.5:

    struct_score_t < -0.10
    AND t_norm_t > -0.30
    AND struct_max > 0.30

Falta una condicion que distinga A de B. El auditor propone:

> **v1.6 debe separar `DISTRIBUTION_CANDIDATE` de `DISTRIBUTION`,
> anadiendo una confirmacion estructural explicita.**

Y advierte: **no inventar el criterio a partir de los 5 casos**.

---

## 2. Estados resultantes

Contrato v1.6 propuesto:

    MARKUP                    fase direccional alcista
    ACCUMULATION              base tras caida
    RANGE                     estado residual
    DISTRIBUTION_CANDIDATE    deterioro tras fortaleza (no confirmado)
    DISTRIBUTION              candidate + confirmacion estructural
    MARKDOWN                  tendencia bajista establecida
    INSUFFICIENT_DATA         estado de calidad de datos

**6 fases de mercado** (antes 5). DISTRIBUTION_CANDIDATE es una fase
completa en la clasificacion, no una sub-etiqueta.

Consecuencia: `classify_wyckoff_phase` puede devolver cualquiera de las 6.
Los consumidores existentes (`stock_leader`, `index_leaders`,
`sector_breadth`, `sector_wyckoff_distribution`, `index_phase`) deben
recibir esta semantica.

---

## 3. Criterios de confirmacion (opciones)

Cinco opciones. Cada una define `DISTRIBUTION = candidate AND X`.
Diseñadas sin mirar los 5 casos concretos.

### Opcion A - SOW (Sign of Weakness)

Evento simetrico a `detect_sos` pero bajista:

    SOW_t = (
        Close_t < min(Low[t-N : t-1])
        AND Volume_t > mean(Volume[t-N : t-1])
    )

`DISTRIBUTION = candidate AND cualquier SOW en las ultimas M sesiones`.

**Ventaja:** coherente con Wyckoff (Phase D/E de distribucion se
confirma con ruptura de soporte/rango + volumen). Reutiliza la estructura
de `detect_sos`.

**Inconveniente:** introduce dos parametros nuevos (N, M).

### Opcion B - Persistencia del deterioro

    persistencia_M = (
        struct_score < STRUCT_DETERIORO
        durante M de las ultimas N sesiones
    )

`DISTRIBUTION = candidate AND persistencia_M`.

**Ventaja:** no requiere nuevo evento. Reutiliza `struct_score`. Robusto.

**Inconveniente:** no captura la naturaleza del deterioro (precio/volumen).
Es solo una condicion temporal.

### Opcion C - Estructura de maximos decrecientes (Lower High)

    LH = max(High[t-N : t-M]) < max(High[t-M : t-1])

Es decir: el maximo reciente esta por debajo del maximo anterior.

`DISTRIBUTION = candidate AND LH`.

**Ventaja:** captura el techo clasico (secuencia de lower highs).
Especifico de distribucion.

**Inconveniente:** requiere 2 parametros (N, M). No usa volumen.

### Opcion D - Rechazo desde el maximo

    rejection = (Close_t < max(Close[t-N : t-1]) * (1 - X))

Es decir: el precio ha caido X% desde el maximo reciente.

`DISTRIBUTION = candidate AND rejection`.

**Ventaja:** simple (1 parametro X). Captura la distancia desde el pico.

**Inconveniente:** umbral X arbitrario. No captura estructura.

### Opcion E - Combinacion SOW + Persistencia

    DISTRIBUTION = candidate AND (SOW OR persistencia_M)

Combina evento y estado.

**Ventaja:** robusto. Cubre tanto la confirmacion por evento (SOW)
como por estado (persistencia).

**Inconveniente:** mas complejo. Dos parametros o mas.

---

## 4. Comparacion de opciones

| Criterio | A | B | C | D | E |
|---|---|---|---|---|---|
| Requiere nueva funcion | Si | No | Si | No | Si |
| Parametros nuevos | 2 | 2 | 2 | 1 | 2-4 |
| Captura estructura precio | Si | No | Si | Parcial | Si |
| Usa volumen | Si | No | No | No | Parcial |
| Coherente con Wyckoff | Alta | Media | Alta | Baja | Alta |
| Simple de testear | Media | Alta | Media | Alta | Baja |

---

## 5. Impacto en el contrato

### Si se aprueba cualquier opcion:

- Anadir §5.4bis `DISTRIBUTION_CANDIDATE` (la logica actual de §5.4).
- Anadir §5.4ter `DISTRIBUTION` (candidate + confirmacion X).
- Nuevos parametros en `config/settings.py` (N, M o X segun opcion).
- Nuevos parametros en `02_proveniencia_parametros.md` marcados
  PROPUESTO (no calibrados).
- Tests nuevos obligatorios:
  - `test_distribution_candidate_is_reachable`.
  - `test_distribution_confirmed_requires_candidate`.
  - `test_distribution_confirmed_requires_confirmation`.

### Ajuste a los consumidores

Los 5 consumidores deben reconocer la nueva fase:
- `stock_leader`: distingue candidatos de confirmados.
- `index_leaders`: idem.
- `sector_breadth`: cuenta CANDIDATE y DISTRIBUTION por separado.
- `sector_wyckoff_distribution`: idem.
- `index_phase`: idem.

**Pero el impacto real depende de la opcion elegida.** Sin implementar
v1.6 no se sabe cuantos CANDIDATE pasarian a DISTRIBUTION confirmada.
En los 5 casos verificados, ninguno tiene un evento SOW evidente en el
OHLCV (habria que comprobarlo con la opcion elegida).

---

## 6. Sobre los 10 casos DISTRIBUTION v1.5

Con v1.6, los 10 casos actuales se reclasificarian:

- **Todos como `DISTRIBUTION_CANDIDATE`** (ninguno cumple confirmacion hoy,
  porque la confirmacion no existe).
- Algunos pasarian a `DISTRIBUTION` si cumplen el criterio elegido.
- Los 5 "claros" (ACS.MC, ANA.MC, BA, PCG, TEF.MC) probablemente
  pasarian a DISTRIBUTION bajo opciones A/B/C.
- Los 5 "frontera" (BKNG, O, SLB, VMRK, WFC) probablemente se quedarian
  como CANDIDATE (esa es la intencion).

**Pero no se decide a partir de estos 10.** La decision es del auditor.

---

## 7. Impacto en 5d

Con v1.6, la migracion de consumidores cambia:

- Los consumidores deben distinguir CANDIDATE de DISTRIBUTION.
- El reporte debe presentar las dos fases por separado.
- Algunos tests existentes cambian.

Esto retrasa 5d hasta que v1.6 este implementado, testeado y verificado
sobre el universo completo.

---

## 8. Preguntas al auditor

1. **¿Apruebas la separacion `DISTRIBUTION_CANDIDATE` vs `DISTRIBUTION`?**
2. **¿Que opcion prefieres para el criterio de confirmacion (A/B/C/D/E)?**
3. **Si ninguna de las 5, ¿que criterio propondrias?** Entiendo del
   dictamen que no quieres inventarlo a partir de los 5 casos, y que el
   criterio debe definirse antes de implementar.
4. **¿Los parametros nuevos (N, M, X) se fijan ahora o quedan PROPUESTOS
   hasta calibracion (5b.3)?**
5. **¿La separacion se aplica tambien a ACCUMULATION?** El contrato
   actual tiene una sola fase ACCUMULATION. Por simetria podria
   proponerse `ACCUMULATION_CANDIDATE` vs `ACCUMULATION`. Pero dado que
   no ha aparecido el problema (los ACCUMULATION no muestran el mismo
   conflicto), ¿se mantiene como esta?
6. **¿Antes de implementar v1.6, quieres que verifique los 5 casos
   claros bajo la opcion elegida (con OHLCV crudo) para confirmar que
   pasarian a DISTRIBUTION?** O se implementa primero y se verifica
   despues.

---

## 9. Estado

- v1.5: implementada, DISTRIBUTION alcanzable.
- Auditor: no acepta los 5 frontera.
- v1.6: propuesta de separacion CANDIDATE/CONFIRMED.
- 5d: bloqueada hasta dictamen + implementacion v1.6.

---

Fin de la propuesta.
