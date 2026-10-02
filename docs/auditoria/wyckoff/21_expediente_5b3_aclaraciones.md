# EXPEDIENTE 5b.3 - ACLARACIONES PRE-GRID

**Documento de consulta. NO normativo. Requiere dictamen externo.**
**Fecha:** 2026-10-02.
**Precedente:** protocolo 5b.3 (`20_protocolo_fase_5b3.md`) + commit `b9149d6` (v1.7 implementado).
**Estado:** BLOQUEO de la fase 5b.3 hasta dictamen.

---

## 0. Proposito

5b.3 esta bloqueada. Antes de escribir el script de calibracion (grid
240 + holdout) se han detectado 8 discrepancias entre el protocolo,
el contrato, el codigo implementado y la configuracion productiva.

Ninguna es trivial. Todas afectan al outcome primario del experimento.
Sin resolverlas, cualquier script de calibracion produciria resultados
no auditables (implementacion distinta segun quien lo lea).

Se solicita dictamen externo sobre las preguntas del §4.

---

## 1. Contexto minimo

5b.3 calibra N, M, X_ATR, Y_VOL del SOW (Sign of Weakness) que
confirma DISTRIBUTION. Formula v1.7 (protocolo 5b.3 §1):

    support_t          = rolling_min(Low, N).shift(1)
    ATR_baseline_t     = ATR(window_atr).shift(1)
    volume_baseline_t  = rolling_mean(Volume, N).shift(1)
    break_depth_t      = (support_t - Close_t) / ATR_baseline_t
    volume_ratio_t     = Volume_t / volume_baseline_t
    SOW_t = (Close_t < support_t)
            AND (break_depth_t >= X_ATR)
            AND (volume_ratio_t >= Y_VOL)

Grid: N in {20,30,40,60} x M in {5,10,15,20,30} x X_ATR in {0.25,0.50,
0.75,1.00} x Y_VOL in {1.10,1.20,1.50} = 240 combinaciones.

Outcome primario (protocolo 0bis-C2): struct_deterioration_H20.

Criterio de exito (protocolo 10): al menos una combinacion cumple
D1-D4 en calibracion y holdout.

---

## 2. Estado verificado del sistema

**Commit:** b9149d6 (HEAD, origin/main).
**Suite:** 3087 passed + 5 skipped.

**Implementado:**
- indicators/wyckoff_v1.py v1.7: detect_sow con ATR ex-ante +
  umbrales X_ATR/Y_VOL.
- config/settings.py: WYCKOFF_SOW_X_ATR=0.50,
  WYCKOFF_SOW_Y_VOL=1.20 (defaults en produccion).
- tests/test_wyckoff_v1_contract.py: I30-I33 (4 tests v1.7).
- docs/auditoria/wyckoff/20_protocolo_fase_5b3.md actualizado con
  C1-C4 + D1-D4.
- docs/auditoria/wyckoff/01_contrato_semantico_v1_7.md creado.

**NO implementado:**
- Script de calibracion (grid 240).
- Script de holdout.
- Resultados 5b.3.

**NO integrado en produccion:**
- indicators/wyckoff_v1.py NO esta importado por run.py ni por
  ningun modulo del pipeline. Todos los consumidores productivos
  (sector_wyckoff_distribution.py, index_leaders.py,
  index_phase.py, sector_breadth.py, stock_leader.py) siguen
  importando indicators/wyckoff.py (legacy v4.2). La migracion no
  ha empezado.

---
## 3. Hallazgos

### A1 (ALTO) - Contrato v1.7 stale

**Evidencia.** docs/auditoria/wyckoff/01_contrato_semantico_v1_7.md:

- Titulo: "CONTRATO WYCKOFF v1.7".
- Seccion 12: "01_contrato_semantico_v1_3.md (este) -> activo".
- Firma: "Fin del contrato v1.3."
- Seccion 3.1 dice "K (WYCKOFF_T_NORM_K): ... 0.15" (real: 0.25
  calibrado en 5b.2).
- Seccion 5.4 muestra la formula SOW v1.6 sin ATR, sin X_ATR, sin
  Y_VOL.

**Causa.** El patch del commit anterior intento reemplazar el bloque
SOW en 01_contrato_semantico_v1_6.md con un "if old_sow in src:".
El anchor no matcheo; el "if" silencioso no escribio nada. El rename
del titulo si paso.

**Consecuencia.** El contrato vigente dice una formula; el codigo
implementa otra; el protocolo pide una tercera. La implementacion no
es auditable contra el contrato.

### A2 (ALTO) - Protocolo 5b.3 contradictorio sobre "deterioro continuo"

**Evidencia.** Mismo documento, dos definiciones incompatibles:

- Seccion 4.3 (texto original, no actualizado):
  pct_deterioro_continuo = % donde struct_t+H < struct_t
  O BIEN precio_t+H < precio_t * (1-0.02).
- Seccion 0bis-C2 (correccion del dictamen externo, posterior):
  "No mezclar ambas en un OR. Reportar por separado."
  Y define 4 metricas: struct_deterioration, price_weakness,
  below_support, lower_low.

**Consecuencia.** El outcome primario struct_deterioration_H20 que
pide C2 no coincide con el pct_deterioro_continuo que pide la 4.3.
El script no puede calcular las dos sin ambiguedad.

### A3 (MEDIO) - Desempate X_ATR invertido en 5.2

**Evidencia.** Seccion 5.2: "Desempate: menor X_ATR (mas conservador
respecto a magnitud)."

**Problema semantico.** X_ATR es el umbral minimo de penetracion.
Menor X_ATR = menos exigente = mas falsos positivos = menos
conservador. El texto dice lo contrario de lo que implica la formula.

**Pregunta:** se quiere preferir el umbral mas bajo (mas eventos,
menos exigente) o el mas alto (menos eventos, mas exigente)?

### A4 (MEDIO) - Metricas sin definicion operativa

**Evidencia.** Seccion 4 del protocolo:

- margen_pct_confirmed - "sensibilidad a variaciones de M". Con que
  otro parametro fijo? Se recalcula el grid entero variando solo M?
- n_candidate_events - por sesion? por ticker? agrupado por
  ticker-and-window?
- tasa_candidate_confirmed - formula exacta.
- Split 70/30 - por tiempo global, o por ticker, o ambos?
- Horizonte holdout - solo H=20 o tambien H=10/H=40?

**Consecuencia.** Cada implementador produciria un script distinto.

### A5 (ALTO, NUEVO) - W de struct_max sin especificar

**Evidencia.**

- Protocolo 5b.3 seccion 1, contrato v1.7 seccion 5.4: struct_max
  (historico [t-W+1, t-1]). W no se define en ningun sitio.
- Codigo indicators/wyckoff_v1.py::_compute_precedent usa
  PRECEDENT_WINDOW = 60.
- PRECEDENT_WINDOW esta marcado como "PROPUESTO, calibrar Fase 5b".
- Fase 5b se cerro con 5b.2 sin calibrar PRECEDENT_WINDOW.

**Consecuencia.** El codigo actual usa 60; el protocolo no lo declara.
Si 5b.3 calibra N sin especificar W, la calibracion se hace sobre una
W no declarada.

**Pregunta:** se fija W=60 en 5b.3 o se anade W al grid?

### A6 (MEDIO, NUEVO) - Defaults en config productivo

**Evidencia.** config/settings.py:

    WYCKOFF_SOW_X_ATR = 0.50  # PROPUESTO (5b.3)
    WYCKOFF_SOW_Y_VOL = 1.20  # PROPUESTO (5b.3)

Cualquier llamada a detect_sow() sin argumentos usara estos valores.
Hoy no hay consumidores productivos (v1 no esta integrado), pero es
deuda latente.

**Pregunta:** se sustituyen por None con validacion explicita
mientras 5b.3 no cierre? O se acepta que sean placeholders?

### A7 (BAJO, NUEVO) - Docstring stale

**Evidencia.** indicators/wyckoff_v1.py linea 4:

    Contrato: docs/auditoria/wyckoff/01_contrato_semantico_v1_1.md

Contrato vigente: v1.7 (titulo) / v1.3 (cuerpo, por A1).

### A8 (BAJO, NUEVO) - Seccion 11 protocolo desactualizada

**Evidencia.** Seccion 11 dice "v1.7 PROPUESTA". Ya esta implementada
(commit b9149d6). Cosmetico, se corrige al cerrar 5b.3.

---

## 4. Preguntas al auditor externo

**P1.** Sobre A1 (contrato v1.7 stale): se reescribe
01_contrato_semantico_v1_7.md para que el cuerpo refleje la formula
v1.7 del protocolo, o se abandona esa numeracion y se abre contrato
v1.8?

**P2.** Sobre A2 (contradiccion deterioro): el outcome primario es
struct_deterioration_H20 (C2), o el OR de la seccion 4.3? Si C2, se
reescribe la seccion 4.3 con las 4 metricas de C2?

**P3.** Sobre A3 (desempate): desempate por menor X_ATR (mas eventos)
o por mayor X_ATR (mas exigente)? Justificacion solicitada.

**P4.** Sobre A4 (metricas): definiciones operativas exactas de
margen_pct_confirmed, n_candidate_events, tasa_candidate_confirmed,
split, horizonte holdout?

**P5.** Sobre A5 (W de struct_max): se fija W=60 (valor actual del
codigo), se anade W al grid, o se resuelve de otra forma?

**P6.** Sobre A6 (defaults en config): se neutralizan hasta que 5b.3
cierre, o se aceptan como placeholders productivos?

**P7.** Sobre A7/A8 (cosmeticos): se corrigen en el mismo commit que
resuelva A1-A6, o se dejan para cierre de fase?

---

## 5. Que NO se ha tocado

- Cero cambios de codigo en wyckoff_v1.py ni en config/settings.py
  ni en tests.
- Cero cambios en el protocolo ni en el contrato.
- Ningun script de calibracion escrito.
- Ninguna ejecucion del grid.

El sistema esta exactamente en el estado del commit b9149d6.

---

## 6. Propuesta de siguiente paso

1. Recibir dictamen sobre P1-P7.
2. Aplicar correcciones documentales (contrato, protocolo) en un
   solo commit.
3. Aplicar cambios de codigo si el dictamen lo requiere (defaults,
   W, etc.).
4. Escribir el script de calibracion con las definiciones
   operativas cerradas.
5. Ejecutar 5b.3 (grid 240 + holdout).
6. Redactar informe de resultados y enviarlo a segunda ronda de
   auditoria antes de congelar parametros.

El criterio de la premisa "auto-mantenido, GitHub diario" no aplica
a 5b.3 (script de analisis, no pipeline productivo). Aplicara cuando
se aborde la migracion legacy -> v1.

---

**Fin del expediente.**