# PLAN DE MIGRACION - Modulo Wyckoff v1

**Hoja de ruta del proyecto de rediseno.**
**Fecha:** 2026-10-02.
**Referencia:** dictamen auditor externo 2026-10-02, §22 (orden correcto de intervencion).

---

## 1. Principios

- **No tocar el legacy hasta cerrar v1.** `indicators/wyckoff.py` permanece
  operativo durante todo el proyecto.
- **Un cambio = una verificacion = un commit.** Fase por fase.
- **Contrato antes que codigo.** Ninguna linea de implementacion se
  escribe sin haber cerrado la fase documental correspondiente.
- **Golden tests antes que migracion.** Los consumidores solo migran
  cuando los tests del contrato pasan.
- **No mezclar fases.** Fase 5 no empieza sin Fase 4 cerrada.

---

## 2. Fases

### Fase 1 - Semantica CERRADA

**Entregable:** `01_contrato_semantico_v1.md` §1-§4.

**Estado:** COMPLETADO 2026-10-02.

**Criterio de cierre:** las 5 decisiones (naturaleza, stability, effort,
accumulation, markdown) estan escritas y firmadas en el contrato.

---

### Fase 2 - Contrato de estados CERRADA

**Entregable:** `01_contrato_semantico_v1.md` §5-§9.

**Estado:** COMPLETADO 2026-10-02.

**Contenido:** definicion de ACCUMULATION, MARKUP, DISTRIBUTION, MARKDOWN,
RANGE, INSUFFICIENT_DATA. Invariantes I1-I10. Transiciones informativas.

**Criterio de cierre:** cada estado tiene condicion de activacion
explicita; ningun estado es fallback silencioso.

---

### Fase 3 - Proveniencia CERRADA

**Entregable:** `02_proveniencia_parametros.md`.

**Estado:** COMPLETADO 2026-10-02.

**Contenido:** tabla de parametro/valor/rationale/fuente/fecha/validacion
para las ventanas, pesos, umbrales, eventos y constantes.

**Criterio de cierre:** ningun parametro queda como "ND" oculto. Lo
que no tiene justificacion se declara explicitamente.

**Deuda aceptada:** los parametros marcados `HEUR` o `LEG` sin
validacion formal quedan registrados como tales. Se priorizan para
calibracion en Fase 5+ (post-v1).

---

### Fase 4 - Golden tests PENDIENTE

**Entregable:** `tests/test_wyckoff_contract.py`.

**Contenido minimo:**

- **T1**: `struct = 0.60*t_norm + 0.40*c_norm` (bit a bit, 1e-12).
- **T2**: `tact = 0.50*v_norm + 0.50*e_norm`.
- **T3**: `combined = 0.70*struct + 0.30*tact`.
- **T4**: todos los `*_norm` en (-1, 1) sobre serie sintetica larga.
- **T5**: `combined`, `struct`, `tact` en (-1, 1).
- **T6**: `stability` en (-1, 1).
- **T7**: `ALL_NAN != RANGE`. Sin observacion valida -> `INSUFFICIENT_DATA`.
- **T8**: `INSUFFICIENT_DATA` si `len(Close) < 200`.
- **T9**: sin look-ahead. La clasificacion en `t` no depende de `t+1`.
- **T10**: umbrales. Se construye una serie que cruce cada umbral y se
  verifica que el estado cambia en el punto esperado.
- **T11**: determinismo. Mismo input -> mismo output.
- **T12**: los 4 pesos suman 1.00 en cada nivel.

**Fixture:** serie OHLCV sintetica de 300 filas, deterministica (seed
fijo). No usar precios reales como golden (el auditor lo prohibe
explicitamente, §13).

**Criterio de cierre:** los 12 tests pasan en verde. Test de mutacion:
si se altera cualquier peso o umbral en el codigo, algun test cae.

**Duracion estimada:** 1 sesion.

---

### Fase 5 - Implementacion y migracion PENDIENTE

**Entregable:** `indicators/wyckoff_v1.py` + migracion de 5 consumidores.

**Sub-fases:**

**5a. Modulo v1 en paralelo.**
- Nuevo fichero `indicators/wyckoff_v1.py`.
- Contrato v1 implementado segun 01.
- Golden tests de Fase 4 ejecutados sobre el.
- **Legacy intacto.**

**5b. Calibracion de parametros PROPUESTOS.**
- K de estabilidad.
- Umbral DISTRIBUTION.
- Ventana ATR para base/rango.
- N sesiones spring reciente.
- Actualizacion de `02_proveniencia_parametros.md` con resultados.

**5c. Comparativa legacy vs v1.**
- Run E2E con ambos modulos.
- Diferencias por sector e indice.
- Documento: `04_comparativa_legacy_v1.md`.
- **Criterio de no-regresion:** el sistema v1 no cambia de forma
  traumática la composicion del top-5 de ningun sector/indice sin
  justificacion estructural.

**5d. Migracion de consumidores (orden por criticidad):**

1. `indicators/index_phase.py` (menor alcance: fase de indice).
2. `indicators/sector_wyckoff_distribution.py` (contadores).
3. `indicators/sector_breadth.py` (contadores).
4. `indicators/stock_leader.py` (sectores SPDR).
5. `indicators/index_leaders.py` (indices USA + Europa).

Cada migracion: un commit, suite verde, verificacion E2E local.

**5e. Retirada del legacy.**
- Solo tras migracion completa + 1 run en CI estable.
- Renombrar `wyckoff.py` a `wyckoff_legacy.py` (NO borrar; mantener
  por referencia).
- Ajustar imports en consumidores residuales.
- Actualizar `02_ARQUITECTURA.md`.

**Criterio de cierre:** 5/5 consumidores migrados, legacy renombrado,
suite verde, CI verde con el nuevo modulo en produccion.

**Duracion estimada:** 2-3 sesiones.

---

## 3. Orden estricto

    Fase 1 (CERRADA) -> Fase 2 (CERRADA) -> Fase 3 (CERRADA)
                                                      │
                                                      ▼
                                                  Fase 4
                                                      │
                                                      ▼
                                    Fase 5a -> 5b -> 5c -> 5d -> 5e

**No hay atajos.** El auditor lo ha exigido explicitamente (§22).

---

## 4. Riesgos identificados

| Riesgo | Mitigacion |
|---|---|
| La nueva clasificacion cambia el top-5 de sectores | Fase 5c (comparativa legacy vs v1). Documentar cada cambio. |
| Los pesos heurísticos no estan calibrados | Fase 5b. Calibrar con dataset controlado. |
| Retirada prematura del legacy rompe el pipeline | Fase 5e solo tras 1 run CI estable con v1. |
| Consumidores con comportamientos implicitos no documentados | Tests de contrato en cada consumidor antes de migrar. |
| Performance: v1 mas costoso que legacy | Benchmark en Fase 5a. Si supera 2x, revisar. |

---

## 5. Fuera de alcance

Este proyecto NO incluye:

- Rediseño del WLS.
- Rediseño del SLPM.
- Rediseño del pipeline de sectores.
- Cambios en `robust_zscore`.
- Cambios en `get_col`.

Si emergen durante Fase 5, se abren como sub-proyectos separados.

---

## 6. Criterio de exito global

El proyecto se considera cerrado cuando:

1. `indicators/wyckoff_v1.py` implementa el contrato v1 completo.
2. Los 12 golden tests de Fase 4 pasan en verde.
3. Los 5 consumidores migrados, suite verde.
4. Un run CI verde con el nuevo modulo en produccion.
5. Documento `04_comparativa_legacy_v1.md` publicado.
6. `02_ARQUITECTURA.md` y `03_IAE.md` actualizados.

---

## 7. Bitacora del proyecto

| Fecha | Hito | Estado |
|---|---|---|
| 2026-10-02 | Fase 1 cerrada (semantica) | OK |
| 2026-10-02 | Fase 2 cerrada (estados) | OK |
| 2026-10-02 | Fase 3 cerrada (proveniencia) | OK |
| PENDIENTE | Fase 4 (golden tests) | — |
| PENDIENTE | Fase 5a (modulo v1 en paralelo) | — |
| PENDIENTE | Fase 5b (calibracion) | — |
| PENDIENTE | Fase 5c (comparativa) | — |
| PENDIENTE | Fase 5d (migracion consumidores) | — |
| PENDIENTE | Fase 5e (retirada legacy) | — |

---

Fin del plan de migracion.

---

## 8. Actualizacion 2026-10-02 (dictamen P1 + revision c_norm/e_norm)

Tras el dictamen externo P1 (t_norm) y la revision de c_norm/e_norm,
el estado del proyecto queda asi:

    Fases 1-3   CERRADAS
    Fase 4      CERRADA
    Fase 5a     IMPLEMENTADA (v1.3)
    P1          RESUELTO arquitectonicamente con O2
    K           PROVISIONAL (= 0.15, sin calibrar)
    Fase 5b     ABIERTA - protocolo formal en 09_protocolo_fase_5b.md
    Fase 5c     BLOQUEADA oficialmente
    Fase 5d     BLOQUEADA
    Legacy      INTACTO

### Aclaraciones del dictamen

1. **K=0.15 se mantiene PROPUESTO.** No se cambia a 0.25/0.35 sin
   criterio predefinido (evita calibrar sobre los 4 casos).
2. **5c oficial bloqueada.** Solo se permite sensibilidad exploratoria
   de K explicitamente etiquetada como calibracion (no como validacion).
3. **5d sigue bloqueada.** Los 5 consumidores no se migran hasta que K
   este congelado.
4. **c_norm aprobado** (mantener como esta). El z-score sobre
   compression tiene sentido porque el contrato declara posicion
   relativa al regimen historico.
5. **e_norm aprobado** (mantener como esta). `tanh(effort_z - result_z)`
   es coherente con Effort vs Result clasico.
6. **No abrir P1 para c_norm/e_norm.**
7. **D-Pendiente registrada:** se asume que effort_z y result_z son
   suficientemente comparables. Es una hipotesis de escala que debe
   quedar documentada.

### Siguiente paso aprobado

Redactar el protocolo formal de Fase 5b con:
- dataset y periodo fijados ex-ante,
- grid predefinida,
- metricas definidas antes de observar resultados,
- criterios de aceptacion ex-ante,
- validacion walk-forward sobre periodo independiente,
- congelacion de K solo tras cumplir criterios.

Razon del auditor: "K debe calibrarse sobre trend, no sobre el
resultado de las fases".

### Sensibilidad exploratoria

Permitida si se etiqueta como analisis exploratorio (no como 5b ni 5c).
Prohibido usarla para elegir K.

---

## 9. Actualizacion 2026-10-02 (5b.2 exitosa)

**K = 0.25 congelado.** Fase 5b cerrada. Fase 5c desbloqueada.

Estado:

    Fases 1-3   CERRADAS
    Fase 4      CERRADA
    Fase 5a     IMPLEMENTADA (v1.3)
    Fase 5b     CERRADA (K = 0.25, calibrado segun protocolo v3)
    Fase 5c     DESBLOQUEADA
    Fase 5d     BLOQUEADA (hasta que 5c cierre)

### 5c - Comparativa legacy v1

Objetivo: comparar salidas de `indicators/wyckoff.py` (legacy) vs
`indicators/wyckoff_v1.py` (v1.3 con K=0.25) sobre el mismo universo y
periodo. Documentar diferencias.

- Distribuciones de fases por sector e indice.
- Diferencias en top-5 de indices.
- Diferencias en `rws_z` (que alimenta WLS).
- Sin migrar consumidores: solo diagnostico comparativo.

Criterio de cierre de 5c: documento `13_comparativa_legacy_v1.md` con
las discrepancias cuantificadas y su justificacion estructural.

### 5d - Migracion (bloqueada)

Sigue bloqueada hasta que 5c cierre y el auditor lo apruebe.
