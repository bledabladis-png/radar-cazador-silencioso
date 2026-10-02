# PROTOCOLO FASE 5b.X - Validacion out-of-sample de la candidata SOW

**Documento normativo. Define la validacion futura de la candidata.**
**Fecha:** 2026-10-02.
**Precedente:** dictamen `33_dictamen_5b4bis_v2.md` + contrato v1.9.
**Estado:** PENDIENTE DE EJECUCION. Bloqueado hasta disponibilidad de
datos posteriores al cutoff 2026-10-01.

---

## 0. Proposito

Validar out-of-sample la candidata SOW congelada en el contrato v1.9:

    N=60, M=30, X_ATR=0.25, Y_VOL=1.10

sobre datos temporalmente posteriores al 2026-10-01, con la misma
implementacion y sin recalibracion.

**Objetivo:** determinar si el lift +0.074 y la firma +++ observados
en 5b.4-bis sobreviven fuera de la muestra de seleccion, o si son
artefacto de una busqueda sobre 240 configuraciones.

---

## 1. Estado de partida

    v1.8              CONTRATO PRODUCTIVO VIGENTE
    v1.9              CANDIDATA FROZEN_FOR_VALIDATION
    config SOW        None (fail-closed)
    5b.4-bis          PASS desarrollo / NO confirmado
    5b.X              ESTE DOCUMENTO (pendiente de ejecucion)
    5c.4, 5d          BLOQUEADAS
    Legacy            INTACTO

**Cutoff temporal:** 2026-10-01 (ultima fecha del historico de
desarrollo). Todo dato posterior es candidato a validacion.

---

## 2. Condiciones previas a la ejecucion

5b.X NO puede ejecutarse hasta que existan **suficientes sesiones
posteriores al cutoff**. "Suficiente" se define en terminos de
capacidad de medir el outcome H20 sobre episodios nuevos, no en
terminos de tiempo calendario arbitrario.

### 2.1. Criterio minimo propuesto

Existencia de:

- Al menos 12 meses de sesiones bursatiles posteriores a 2026-10-01.
- Al menos 50 candidate episodes nuevos con landmark L >= 2026-10-02.
- Al menos 20 candidate episodes confirmados (SOW en ventana M).

**Nota:** el umbral concreto queda como propuesta sujeta a dictamen.
Si el auditor prefiere otro criterio (numero de sesiones, de
episodios, o mixto), se ajustara antes de ejecutar.

### 2.2. NO recalibrar

Los parametros N/M/X/Y estan congelados en v1.9. La ejecucion de
5b.X **no explora grid**: aplica la candidata una sola vez.

### 2.3. NO modificar el codigo

Se usa el mismo commit del script que la version congelada. Cualquier
cambio en `scripts/` invalida la validacion. El script de 5b.X sera
`scripts/validate_wyckoff_sow_5bX.py`, derivado de
`calibrate_wyckoff_sow_5b4bis.py` **sin tocar la logica del landmark,
de los bloques, ni del bootstrap**.

---

## 3. Definiciones

Heredadas del protocolo v3 (5b.4-bis) y del contrato v1.9:

- **Unidad:** candidate episode (t0 = entrada, candidate_t=True AND
  candidate_{t-1}=False).
- **Landmark:** L = t0 + M, con M=30.
- **Confirmed:** SOW en [t0+1, L].
- **Baseline:** sin SOW en [t0+1, L].
- **Outcome primario:** struct_deterioration_H20 = struct[L+20] < struct[L].
- **Outcome secundarios:** price_weakness, below_support, lower_low,
  H10 y H40.
- **Censura:** elegible solo si existe informacion hasta L+H.
- **Lift:** P(deterioro | confirmed) - P(deterioro | baseline),
  episode-weighted.

**Sin cambios respecto al protocolo v3. La validacion no introduce
variaciones.**

---

## 4. Bloques de validacion

Los bloques P1/P2/P3 del protocolo v3 son historicos y ya estan
"quemados" (seleccionados para el desarrollo). En 5b.X:

### 4.1. Bloque unico de validacion

Todo dato posterior a 2026-10-01 forma **un unico bloque de
validacion**, sin subdivisiones.

Razon: la subdivisión en 3 bloques fue decision de diseno para el
desarrollo. En validacion no se subdivide porque no hay masa critica
para 3 bloques con suficiente muestra.

### 4.2. Reporte por sub-bloques (informativo)

Si el bloque unico de validacion tiene suficiente muestra
(>100 episodios confirmados), se puede reportar adicionalmente por
sub-periodos de ~6 meses, **solo con caracter informativo**, no como
criterio de seleccion ni de decision.

---

## 5. Criterio de exito

### 5.1. Criterio primario

    IC95% lower del lift_H20 > 0

calculado con bootstrap ticker-cluster, B=2000, seed fija
(20261002).

**Se reporta:**
- lift_point
- lower_CI
- upper_CI
- n_confirmed, n_baseline, n_tickers

### 5.2. Comparacion con el desarrollo

Para contexto, se reporta lado a lado:

    Desarrollo (5b.4-bis):
        lift = +0.0740  IC [+0.0234, +0.1242]

    Validacion (5b.X):
        lift = ?         IC [?, ?]

**NO se exige que el lift de validacion sea >= el de desarrollo.**
Se exige que el IC no cruce 0.

### 5.3. Criterios secundarios (informativos)

- Firma de signos por sub-bloques (si aplica, seccion 4.2).
- Comportamiento de los outcomes secundarios.
- Concentracion por ticker (top5/top10 share).

---

## 6. Interpretacion de resultados

### 6.1. Escenarios posibles (declarados ex-ante)

**Escenario A - Validacion confirma.**
IC95% lower del lift > 0. La candidata pasa a contrato productivo
(v2.0). Se autoriza activacion en config y migracion de consumidores.

**Escenario B - Validacion neutra.**
IC95% cruza 0, pero lift_point > 0. La candidata no se confirma pero
tampoco se refuta. Se documenta como "no concluyente" y se decide si
ampliar la ventana de validacion o abandonar.

**Escenario C - Validacion refuta.**
IC95% cruza 0 y lift_point <= 0. La candidata se abandona. Se abre
5b.5 con otra hipotesis o se cierra la linea SOW como confirmador.

**Los tres escenarios se aceptan. El protocolo no esta disenado para
buscar confirmacion, sino para medir sin sesgo.**

---

## 7. Analisis prohibidos en 5b.X

- Recalibrar N/M/X/Y (estan congelados en v1.9).
- Explorar grid (no hay grid; una sola configuracion).
- Volver a seleccionar entre las 17 candidatas de 5b.4-bis.
- Subdividir el bloque de validacion en funcion de los resultados.
- Ampliar el grid si el lift cae.
- Usar los sub-bloques informativos como criterio de decision.
- Modificar el codigo entre freeze y validacion.
- Cambiar el cutoff temporal.
- Cambiar el criterio de exito (5.1) tras ver resultados.
- Reportar solo el resultado favorable.

**Cualquier cambio en estas reglas invalida la validacion. Si hace
falta cambiar algo, se abre 5b.X-bis con nuevo protocolo y se declara
que la validacion original ha sido abortada.**

---

## 8. Salida esperada

- `outputs/audit/wyckoff_5bX_grid.csv` (una fila: la candidata).
- `outputs/audit/wyckoff_5bX_bootstrap.csv` (una fila).
- `outputs/audit/wyckoff_5bX_summary.json`.
- `outputs/audit/wyckoff_5bX_run.log`.
- `docs/auditoria/wyckoff/36_validacion_5bX_resultados.md`.

---

## 9. Que NO se hace en 5b.X

- NO activar parametros en config.
- NO retirar fail-closed.
- NO migrar consumidores.
- NO abrir 5c.4 ni 5d.
- NO declarar exitosamente "SOW es confirmador" hasta que el informe
  de 5b.X lo confirme con IC95% lower > 0.

---

## 10. Estado

    v1.8              CONTRATO PRODUCTIVO VIGENTE
    v1.9              CANDIDATA FROZEN_FOR_VALIDATION
    5b.3              FAIL
    5b.4              FAIL seleccion / diagnostico valido
    5b.4-bis          PASS desarrollo / NO confirmado
    5b.X              ESTE DOCUMENTO, bloqueado hasta datos
    5c.4, 5d          BLOQUEADAS
    Legacy            INTACTO
    Config SOW        None (fail-closed)

---

## 11. Trazabilidad

- 5b.3 (`639fe92`): primera ejecucion con sesgo de anclaje.
- Dictamen 24: inmortal time bias detectado.
- 5b.4 (`04e0620`): landmark corregido. FAIL en seleccion (D3
  inalcanzable con 4 bloques).
- Dictamen 26: core landmark aprobado. 10 correcciones al protocolo.
- 5b.4-bis (`848ee23`): rediseno a 3 bloques. PASS desarrollo
  (17/240).
- Dictamen 33: freeze de validacion autorizado. Freeze productivo NO.
- QA 5b.4-bis (`dacc7b2`): IC por bloque. Solo P3 no cruza 0.
- Contrato v1.9: candidata congelada.
- 5b.X (este documento): validacion out-of-sample pendiente.

---

## 12. Condiciones de activacion (post 5b.X exitoso)

Si 5b.X resulta en Escenario A (IC95% lower > 0):

1. Emitir contrato v2.0 con STATUS=PRODUCTION.
2. Actualizar `config/settings.py`:
       WYCKOFF_SOW_WINDOW_N = 60
       WYCKOFF_SOW_MAX_AGE_M = 30
       WYCKOFF_SOW_X_ATR = 0.25
       WYCKOFF_SOW_Y_VOL = 1.10
3. Evaluar retirada del fail-closed (o mantenerlo como argumento
   explicito obligatorio).
4. Migrar consumidores de `wyckoff.py` legacy a `wyckoff_v1.py`
   (con E2E obligatorio).
5. Abrir 5c.4 y 5d.
6. Documentar en `07_RUNBOOK.md`.

**Ninguno de estos pasos se ejecuta hasta que 5b.X cierre con
Escenario A.**

---

**Fin del protocolo 5b.X.**
