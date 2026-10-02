# PROTOCOLO FASE 5b.4-bis - v3

**Documento normativo. Sustituye al protocolo 5b.4 v2.**
**Fecha:** 2026-10-02.
**Precedente:** 29_dictamen_5b4_v2.md.
**Estado:** APROBADO por dictamen 29. Ejecutable.

---

## 0. Contexto

5b.4 v2 ejecuto correctamente pero no selecciono parametros (D3
inalcanzable). El dictamen 29 confirmo FAIL en seleccion y dio 10
directrices para el nuevo protocolo (seccion 19).

Este protocolo implementa esas directrices. No modifica D3 ni las
fronteras de bloques mirando resultados previos. Rediseña el esquema
temporal desde cero, con criterios fijados ex-ante.

Objetivo: determinar si el SOW anade informacion sobre deterioro
futuro mas alla de candidate, tratando la heterogeneidad temporal como
hallazgo a probar (no como anomalia a esconder), y con un criterio de
seleccion que no confunda suficiencia muestral con direccion del
efecto.

---

## 1. Lo que cambia respecto a v2

    v2 (5b.4):
      - 4 bloques: A, B, C, D (fronteras por calendario)
      - D1: n_confirmed >= 20 AND n_baseline >= 20
      - D2: lower_CI(lift_H20) > 0
      - D3: 4 bloques evaluables + >= 3 con lift > 0
      - Ranking: mediana_lift + MAD_lift + lexicografico

    v3 (5b.4-bis):
      - 3 bloques predefinidos por calendario, sin mirar resultados previos
      - Suficiencia muestral SEPARADA de evaluacion de estabilidad
      - ALL como descriptivo, no como criterio
      - Medida formal de heterogeneidad entre bloques
      - M neutral
      - D2 = filtro de desarrollo (no significacion confirmatoria)
      - UNA configuracion congelada al final
      - Validacion final = 5b.X (futura, datos no inspeccionados)

---

## 2. Definiciones (heredadas de v2, sin cambios)

### 2.1. Convencion temporal

Sesiones de negociacion, no dias naturales. t0 + M = M sesiones
posteriores al indice t0. Idem L + H.

### 2.2. Unidad de observacion

    candidate episode = (candidate_t=True) AND (candidate_{t-1}=False)
    por ticker

candidate segun contrato v1.8 seccion 5.4.

### 2.3. Landmark

    L = t0 + M

### 2.4. Grupo confirmed

    confirmed = existe SOW_t = 1 en [t0+1, L]

Semantica: SOW puede aparecer aunque candidate haya terminado antes de L.

### 2.5. Grupo baseline

    baseline = no existe SOW_t = 1 en [t0+1, L]

### 2.6. Elegibilidad (censura)

Elegible para horizonte H solo si existe informacion completa hasta
L + H. Misma regla para ambos grupos.

### 2.7. Outcome primario

    struct_deterioration_H20 = struct[L+20] < struct[L]

Comparacion punto-a-punto. No implica deterioro continuo.

### 2.8. Outcomes secundarios

    price_weakness_H{h} = close[L+h] < close[L] * 0.98
    below_support_H{h}  = close[L+h] < support[L]
    lower_low_H{h}      = min(close[L+1:L+h]) < min(close[L-N:L])

support[L] = rolling_min(Low, N).shift(1) evaluado en L.
Horizontes H10, H20, H40.

### 2.9. Lift

    Lift = P(struct_deterioration_H20 | confirmed)
         - P(struct_deterioration_H20 | baseline)

Estimando principal: episode-weighted. Publicar n_tickers, n_episodes,
n_confirmed, n_baseline.

---
---

## 3. Bloques temporales (rediseno ex-ante)

### 3.1. Regla de diseno

Las fronteras se fijan mirando solo el calendario y la distribucion
de candidate episodes disponible en el dataset, NO los resultados de
lift de 5b.3 o 5b.4. No se busca "donde el SOW funciona".

Objetivo: 3 bloques temporalmente comparables con capacidad muestral
suficiente.

### 3.2. Fronteras propuestas

Tres bloques de duracion similar, cubriendo el dataset
2021-10-01 -> 2026-10-01:

    Bloque P1 - periodo 1: 2021-10-01 -> 2023-06-30
    Bloque P2 - periodo 2: 2023-07-01 -> 2025-03-31
    Bloque P3 - periodo 3: 2025-04-01 -> 2026-10-01

Duraciones: ~21 meses / ~21 meses / ~18 meses.

Justificacion (no mira lift): tres bloques de ~20 meses cada uno
minimizan el desbalance muestral que hizo fracasar el bloque A de v2.
La frontera P2/P3 (2025-03-31) coincide con la fecha de split de 5b.3,
dato conocido de antemano.

Nota: los bloques no representan regimenes de mercado. Son estratos
temporales.

### 3.3. Asignacion de episodios

Un episodio se asigna al bloque que contiene su landmark L.

---

## 4. Suficiencia muestral SEPARADA de estabilidad

Cambio clave respecto a v2 (directriz del dictamen 29 seccion 9).

### 4.1. Gate de suficiencia (por bloque)

Un bloque es evaluable si:

    n_confirmed >= 20 AND n_baseline >= 20

Un bloque no evaluable se etiqueta INSUFFICIENT_SAMPLE, no FAIL. La
diferencia es metodologica: no sabemos si el SOW falla, solo que no
hay muestra para evaluarlo.

### 4.2. Evaluacion de estabilidad (por bloque evaluable)

Para cada bloque evaluable:

    lift_H20_bloque + IC95% (bootstrap ticker-cluster, B=2000)

Reportar: lift_point, lower_CI, upper_CI, signo.

### 4.3. Agregado ALL (descriptivo, no criterio)

    lift_H20_ALL = P(deterioro | confirmed) - P(deterioro | baseline)
    con IC bootstrap, B=2000

ALL es descriptivo. No gobierna la seleccion cuando hay
heterogeneidad material entre bloques (directriz del dictamen 29
seccion 4).

### 4.4. Medida formal de heterogeneidad

Definicion ex-ante:

    delta_hetero = max(lift_bloque) - min(lift_bloque)
    sobre bloques EVALUABLES

Y version bootstrap: distribucion de delta_hetero bajo remuestreo
ticker-cluster, B=2000. Reportar percentil del delta observado.

Regla: si n_bloques_evaluables >= 2 y los lifts tienen signos
opuestos, la heterogeneidad es material por observacion directa,
independientemente del test.

---

## 5. Criterios de seleccion ex-ante

### 5.1. Criterios duros

    D1: n_confirmed >= 20 AND n_baseline >= 20 (sobre ALL)
    D2: lower_CI(lift_H20_ALL) > 0 (filtro de desarrollo, B=2000)
    D3: de los bloques evaluables, al menos ceil(2/3) tienen lift > 0

Nota: D3 ya no exige que TODOS los bloques sean evaluables. Evalua
solo los que si lo son.

### 5.2. Metricas de estabilidad (informe obligatorio)

    delta_hetero
    n_bloques_evaluables
    n_bloques_con_lift_positivo
    firma_signos (e.g., "+ - +" o "+ + +")

### 5.3. D3_diagnostic (no puerta)

    pct_conf_struct_H20_ALL
    pct_base_struct_H20_ALL
    lift_pp_ALL

Solo informativo.

### 5.4. Ranking entre combinaciones que pasan D1-D3

    Paso 1: mayor n_bloques_con_lift_positivo (mas estabilidad).
    Paso 2: mayor mediana_lift (sobre bloques evaluables).
    Paso 3: menor delta_hetero (mas homogeneidad).
    Paso 4: tiebreak lexicografico (N, M, X_ATR, Y_VOL) ascendente.

M neutral: no se premia ni castiga por tamano.

---
---

## 6. Bootstrap

### 6.1. Capa 1: ticker-cluster

Remuestreo de tickers completos, conservando todos sus episodios.
B=2000, seed=20261002. Percentile bootstrap IC95% [2.5, 97.5].

Criterio D2: lower_CI(lift_H20_ALL) > 0.

Nota de semantica (dictamen 29 seccion 11): esto es un IC bilateral
del 95% usando su limite inferior. NO es un test unilateral al 5%. Es
mas conservador. Se congela expresamente como puerta de desarrollo,
no como prueba confirmatoria unilateral.

### 6.2. Capa 2: sensibilidad temporal/common-shock

Los tickers comparten shocks. El bootstrap ticker-cluster NO elimina
esa dependencia.

El informe declara: la dependencia intraticker se conserva mediante
bootstrap por cluster; permanece una sensibilidad potencial a
dependencia temporal/common-shock.

Sensibilidad concreta: re-ejecutar el calculo del lift por bloque
temporal y reportar dispersion. Especificado en este protocolo antes
de ver resultados.

---

## 7. Analisis prohibidos

- Bajar el umbral de D3 para hacer pasar alguna configuracion.
- Pasar de 4 a 3 bloques usando el resultado observado de 5b.4.
- Eliminar bloques porque perjudican el resultado.
- Congelar la primera de las 17 candidatas de 5b.4.
- Elegir M=30 por IC estrecho o M=5 por lift alto.
- Convertir P2+P3 en "regimen valido" sin prueba adicional.
- Aplicar Bonferroni o FDR para forzar el cierre de 5b.4-bis.
- Usar el bloque D de 5b.3/5b.4 como validacion historica ciega.
- Cualquier modificacion cuyo objetivo implicito sea conseguir que
  alguna configuracion pase.

---

## 8. Salida esperada

- outputs/audit/wyckoff_5b4bis_grid.csv (240 filas).
- outputs/audit/wyckoff_5b4bis_bootstrap.csv.
- outputs/audit/wyckoff_5b4bis_hetero.csv (delta_hetero por combinacion).
- outputs/audit/wyckoff_5b4bis_summary.json.
- outputs/audit/wyckoff_5b4bis_run.log.
- docs/auditoria/wyckoff/31_calibracion_5b4bis_resultados.md.

---

## 9. Criterios de exito

5b.4-bis tiene exito si:

- Al menos una combinacion pasa D1-D3.
- La combinacion elegida pasa D1-D3.
- La combinacion elegida tiene:
  - n_bloques_con_lift_positivo >= 2 de 3 (o 1 de 2 si solo 2 son
    evaluables);
  - mediana_lift > 0 sobre bloques evaluables.

NO se exige generalizacion en bloque historico. Eso pertenece a 5b.X.

Si ninguna combinacion pasa: 5b.4-bis FAIL. Se abre 5b.5 con otra
hipotesis o se abandona SOW como confirmador.

### 9.1. Estado al cierre de 5b.4-bis

Si tiene exito:

    N, M, X_ATR, Y_VOL -> CONGELADOS
    codigo             -> CONGELADO en un commit
    contrato           -> actualizado a v1.9 con parametros fijados

Pero NO se declaran "validados". Quedan a la espera de 5b.X.

---

## 10. Validacion final (5b.X)

5b.X sera una fase posterior, sobre datos temporalmente posteriores al
freeze.

Requisitos:

- Misma implementacion (mismo commit del modulo y del script).
- Parametros congelados.
- Sin recalibracion.
- Ventana temporal no inspeccionada.
- Criterio: lower_CI(lift_H20) > 0 sobre datos nuevos.

El bloque D actual (2025-09-01 -> 2026-10-01) NO se reutiliza como
holdout en ningun documento futuro. Ya esta contaminado por
inspeccion en 5b.3.

---

## 11. Estado

    v1.8              CONTRATO VIGENTE
    5b.3              FAIL (sesgo temporal)
    5b.4              FAIL en seleccion, PASS en diagnostico
    5b.4-bis (este)   PROTOCOLO v3, ejecutable
    5b.X              PENDIENTE (validacion futura, requiere ventana
                      no inspeccionada)
    Parametros SOW    None en config hasta cierre 5b.4-bis
    5c.4, 5d          BLOQUEADAS
    Legacy            INTACTO

---

## 12. Trazabilidad

- Dictamen 24: inmortal time bias detectado. Landmark requerido.
- Dictamen 26: core landmark aprobado. 10 correcciones normativas.
- Dictamen 29: 5b.4 FAIL en seleccion. 10 directrices para v3.
- Este protocolo implementa las directrices del dictamen 29.

---

**Fin del protocolo 5b.4-bis (v3).**
