# HALLAZGO v1.3 - K no calibrado produce saturacion

**Documento de auditoria. Requiere dictamen externo.**
**Fecha:** 2026-10-02.

---

## 1. Contexto

v1.3 implementa la correccion P1 (t_norm = tanh(trend / K)) con
`WYCKOFF_T_NORM_K = 0.15`, marcado como PROPUESTO en proveniencia
(dictamen P1, condicion 4).

## 2. Observacion

Prueba sobre los 4 tickers diagnosticos:

| Ticker | v1.2 | v1.3 | t_norm v1.3 |
|---|---|---|---|
| MSFT | MARKUP | MARKUP | +0.678 |
| PLTR | MARKUP | MARKUP | +0.616 |
| INTC | RANGE | **MARKUP** | +0.926 |
| AMD | RANGE | **MARKUP** | +0.986 |

Los 4 tickers dan MARKUP. t_norm se satura en INTC/AMD (0.93-0.99).

## 3. Analisis

Con K=0.15:

    t_norm = 0.30 -> trend = 0.15 * atanh(0.30) ≈ 4.64%
    t_norm = 0.90 -> trend = 0.15 * atanh(0.90) ≈ 22.0%
    t_norm = 0.99 -> trend = 0.15 * atanh(0.99) ≈ 39.4%

Es decir: un trend del 22% produce t_norm = 0.90 (casi saturado).
Un trend del 39% produce t_norm = 0.99.

En equity, MA50/MA200 > 22% no es extremo (bull markets lo alcanzan
regularmente). K=0.15 es **demasiado pequeño**: el rango util de
t_norm se agota para tendencias normales.

## 4. Lo que NO es este hallazgo

- **No es un defecto de la formula.** `t_norm = tanh(trend/K)` mide
  correctamente nivel de tendencia escalado.
- **No es una regresion de v1.3.** v1.2 no tenia este problema porque
  la metrica era otra (desviacion de regimen).
- **No es un defecto de los tests.** Los 7 tests v1.3 pasan
  correctamente.
- **No es calibrable con los 4 casos.** Prohibido por dictamen.

Es una consecuencia **esperada** de proponer K sin calibracion formal.

## 5. Naturaleza del hallazgo

> **K=0.15 esta mal dimensionado para equity. Requiere calibracion
> formal en Fase 5b.**

No es un defecto del contrato; es una calibracion pendiente que la
implementacion provisional expone.

## 6. Candidatos de K

| K | trend para t_norm=0.30 | trend para t_norm=0.90 | Comentario |
|---:|---:|---:|---|
| 0.15 | 4.6% | 22.0% | PROPUESTO actual, saturado |
| 0.25 | 7.7% | 36.6% | Menos saturado |
| 0.35 | 10.8% | 51.3% | Rango amplio |

## 7. Decision

**No se modifica K en este commit.** Se mantiene 0.15 como PROPUESTO.

**Razon:** el propio dictamen P1 prohibe calibrar con los 4 casos. La
calibracion de K es Fase 5b y requiere dataset controlado.

## 8. Preguntas al auditor

1. ¿Debe K permanecer en 0.15 como PROPUESTO hasta Fase 5b, aunque el
   efecto observado sea saturacion?
2. ¿O se aprueba un K inicial distinto como punto de partida (0.25,
   0.35) sin que eso constituya calibracion (solo eleccion de escala)?
3. ¿Como debe documentarse la prohibicion de calibrar K sobre los 4
   casos en el contrato, mas alla del §10 actual?

## 9. Estado

- P1 (t_norm desviacion de regimen) resuelto: v1.3 implementa
  `t_norm = tanh(trend/K)`.
- v1.3 pasa los 7 tests obligatorios.
- K sigue PROPUESTO.
- Fases 5c/5d siguen bloqueadas hasta cerrar K.

---

Fin del hallazgo v1.3 sobre K.
