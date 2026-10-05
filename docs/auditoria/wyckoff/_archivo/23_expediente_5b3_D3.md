# EXPEDIENTE 5b.3 - D3 y criterio de congelacion

**Documento de consulta. NO normativo. Requiere dictamen externo.**
**Fecha:** 2026-10-02.
**Precedente:** 22_calibracion_5b3_resultados.md + protocolo 5b.3 v3
+ contrato v1.8 + commit e66cf41.
**Estado:** BLOQUEO de la congelacion de parametros hasta dictamen.

---

## 0. Proposito

La calibracion 5b.3 se ejecuto completa (240 combinaciones, 315
tickers, 2415 episodios candidatos). Los resultados estan en
22_calibracion_5b3_resultados.md.

El cumplimiento de D1-D4 es:

    D1 (n_sow_raw >= 100):             240/240
    D2 (cal_n_confirmed >= 20):        240/240
    D3 (struct_deterioration_H20 >= 0.50):  0/240
    D4 (vecindario <= 30pp):           240/240

Ninguna combinacion pasa D1-D4. Todas pasan exactamente 3 de 4.

El propio dictamen (seccion 13) marco D3 como provisional:

> "Mantengo el umbral >= 50% como criterio ex-ante provisional. No voy
> a inventar ahora 55%, 60% o 65% sin evidencia. Pero D3 por si solo
> no basta. Debe coexistir con la comparacion confirmed candidates vs
> candidate baseline without confirmation. La metrica
> incremental_lift_H20 debe ser obligatoria en el informe, aunque no
> le asignaria todavia un umbral duro."

Se solicita dictamen sobre las preguntas del apartado 3.

---

## 1. Evidencia empirica

### 1.1. Lift positivo universal

incremental_lift_H20 > 0 en **240/240** combinaciones en calibracion
y **240/240** en holdout. El signo del lift no depende de la
configuracion. Rango en calibracion:

    min=+0.0408  mediana=+0.1035  max=+0.1920

### 1.2. Nivel absoluto de struct_deterioration_H20

    cal_pct_conf:  min=0.3404  mediana=0.3898  max=0.4500
    cal_pct_base:  min=0.2411  mediana=0.2881  max=0.3131

El maximo de H20 confirmado es 45.0%. D3 exige 50%. **Ninguna
combinacion alcanza el umbral.**

### 1.3. Verificacion H1/H3 (SLB, PCG)

Positiva. El filtro X_ATR + Y_VOL elimina los falsos positivos
marginales del 5c.3. Detalle en 22_calibracion_5b3_resultados.md
seccion 8.

### 1.4. Generalizacion cal -> hold

Para la candidata #1 (N=60, M=30, X=0.50, Y=1.20):

    cal:   conf=0.4458  base=0.2538  lift=+0.1920
    hold:  conf=0.5689  base=0.3427  lift=+0.2261

El lift crece en holdout. El nivel absoluto tambien. La combinacion
generaliza.

### 1.5. Divergencia cal vs hold en X_ATR

- Top-calibracion por lift: **X=0.50** (top-3: tres veces X=0.50).
- Top-holdout por lift: **X=1.00** (top-10: ocho veces X=1.00, dos
  veces X=0.75; ninguna X=0.50).

Es una divergencia real. Dos lecturas:

a) X=0.50 captura mas eventos (mayor muestra) -> mejor estimacion
   en calibracion (1254 sesiones, mas historia). El holdout tiene
   solo 389 sesiones y menos episodios -> ruido mas alto.
b) X=1.00 captura SOW mas severos -> mejor senal en el bloque
   temporal mas reciente (regimen distinto).

Sin evidencia adicional, no se puede decidir cual es la correcta.

---

## 2. Candidatas pre-dictamen (no elegidas)

Se listan a titulo informativo. **No se selecciona.**

**Candidata A** (mejor lift en calibracion):
    N=60, M=30, X_ATR=0.50, Y_VOL=1.20
    cal:  conf=0.4458 base=0.2538 lift=+0.1920 n_conf=406
    hold: conf=0.5689 base=0.3427 lift=+0.2261 n_conf=180

**Candidata B** (mejor lift en holdout):
    N=40, M=30, X_ATR=1.00, Y_VOL=1.10
    hold_lift_H20=+0.3381  hold_n_conf=104
    (cal: consultar grid CSV)

**Candidata C** (mayor nivel absoluto de H20 conf en calibracion):
    N=60, M=5, X_ATR=1.00, Y_VOL=1.20
    cal_pct_conf_struct_H20=0.4500  cal_n_confirmed=126
    cal_lift_H20=+0.1383
    (menor lift, menor muestra)

La candidata C se incluye para transparencia: es la unica que roza
el 50% de D3, pero a costa de reducir la muestra a 126 confirmados
en calibracion y de bajar el lift de +0.1920 a +0.1383. En holdout
su lift no esta en el top.

---

## 3. Preguntas al auditor externo

**P1.** Sobre D3. El umbral D3=0.50 no se cumple en ninguna
combinacion (maximo 45.0%). Opciones:

    a) Mantener D3=0.50 -> ninguna combinacion se congela. 5b.3
       cierra sin parametros elegidos. Se abre 5b.4 con otra
       definicion de outcome.
    b) Relajar D3 a un valor empiricamente alcanzable (p.ej. 0.40,
       que si cumplen varias combinaciones). Requiere justificacion
       ex-post que el protocolo prohibe.
    c) Sustituir D3 por un criterio de lift (p.ej. lift_H20 >= 0.15
       en calibracion y holdout, con un minimo de muestra). Alineado
       con el propio dictamen seccion 13.
    d) Otra opcion que el auditor considere.

**P2.** Sobre la divergencia cal vs hold en X_ATR (X=0.50 en cal,
X=1.00 en hold). Opciones:

    a) Priorizar calibracion (mas muestra). Congelar X=0.50.
    b) Priorizar holdout (mas reciente). Congelar X=1.00.
    c) Analisis adicional antes de decidir. Se solicitan indicaciones
       sobre que analisis.
    d) Otra opcion.

**P3.** Si se congela una combinacion, la propuesta del equipo es la
Candidata A (N=60, M=30, X=0.50, Y=1.20). El auditor puede validarla,
sustituirla por B o C, o pedir otro criterio de seleccion.

---

## 4. Anexo A - Datos agregados por N

Media de las 60 combinaciones de cada N (calibracion):

    N=20   H20_conf=0.3825  H20_base=0.2830  lift_H20=0.0995  n_sow_raw=4263  n_confirmed=359
    N=30   H20_conf=0.3746  H20_base=0.2871  lift_H20=0.0875  n_sow_raw=3294  n_confirmed=316
    N=40   H20_conf=0.3906  H20_base=0.2882  lift_H20=0.1025  n_sow_raw=2730  n_confirmed=282
    N=60   H20_conf=0.4182  H20_base=0.2864  lift_H20=0.1318  n_sow_raw=2033  n_confirmed=256

## 5. Anexo B - Datos agregados por X_ATR

Media de las 60 combinaciones de cada X_ATR (calibracion):

    X=0.25  lift_H20=0.1017  n_sow_raw=5039  n_confirmed=411
    X=0.50  lift_H20=0.1129  n_sow_raw=3439  n_confirmed=334
    X=0.75  lift_H20=0.1015  n_sow_raw=2305  n_confirmed=264
    X=1.00  lift_H20=0.1051  n_sow_raw=1536  n_confirmed=204

## 6. Anexo C - Datos agregados por Y_VOL

    Y=1.10  lift_H20=0.1088  n_sow_raw=3636  n_confirmed=331
    Y=1.20  lift_H20=0.1129  n_sow_raw=3288  n_confirmed=316
    Y=1.50  lift_H20=0.0942  n_sow_raw=2316  n_confirmed=263

## 7. Anexo D - Datos agregados por M

    M=5    lift_H20=0.0934  n_confirmed=217
    M=10   lift_H20=0.0994  n_confirmed=255
    M=15   lift_H20=0.0955  n_confirmed=291
    M=20   lift_H20=0.0997  n_confirmed=335
    M=30   lift_H20=0.1386  n_confirmed=418

## 8. Anexo E - Top-10 por hold_lift_H20

    N  M  X_ATR Y_VOL  hold_lift_H20  hold_n_conf
    40 30 1.00  1.10   0.3381         104
    40 30 1.00  1.20   0.3371         103
    60 30 1.00  1.10   0.3272          90
    60 30 1.00  1.20   0.3218          88
    40 10 1.00  1.20   0.3215          73
    40 10 1.00  1.10   0.3215          73
    60 10 1.00  1.10   0.3190          61
    40 15 1.00  1.20   0.3183          78
    40 15 1.00  1.10   0.3183          78
    40 30 1.00  1.50   0.3153          88

Ocho de diez con X=1.00. Dos con X=0.75. Ninguno con X=0.50.

## 9. Anexo F - Top-10 por cal_lift_H20

    N  M  X_ATR Y_VOL  cal_lift_H20  cal_n_conf  neighbour_delta_pp
    60 30 0.50  1.20   0.1920        406         5.01
    60 30 0.50  1.10   0.1874        424         5.25
    40 30 0.50  1.20   0.1730        450         5.62
    60 30 0.25  1.10   0.1704        505         5.01
    60 30 0.25  1.20   0.1702        477         4.39
    60 30 0.75  1.20   0.1661        325         5.01
    40 30 0.50  1.10   0.1653        468         6.06
    60 30 0.25  1.50   0.1617        389         3.09
    60 30 1.00  1.20   0.1610        252         4.51
    60 20 0.50  1.10   0.1563        342         5.44

Los primeros cuatro tienen neighbour_delta_pp bajo (< 6pp). La
estabilidad no esta reñida con lift alto.

## 10. Anexo G - Artefactos disponibles

    outputs/audit/wyckoff_5b3_grid.csv        240 filas x 101 columnas
    outputs/audit/wyckoff_5b3_summary.json    resumen JSON
    outputs/audit/wyckoff_5b3_run.log         log de ejecucion

Columnas del CSV: N, M, X_ATR, Y_VOL, n_sow_raw, n_sow_unique,
D1, D2, D3, D4, neighbour_confirmation_delta_pp, y para cada
scope (cal, hold) y horizonte (H10, H20, H40):
n_candidate_events, n_confirmed, tasa_candidate_confirmed,
n_out_conf_H{h}, n_out_base_H{h}, y por cada metrica
(struct/price/below/lower): pct_conf, pct_base, lift.

## 11. Que NO se ha tocado

- Cero cambios en config/settings.py desde e66cf41.
- Cero cambios en el contrato v1.8 ni en el protocolo 5b.3.
- Ninguna combinacion congelada.
- Ningun parametro SOW activado en produccion.
- El sistema sigue con parametros SOW a None (fail-closed).

## 12. Propuesta de siguiente paso

1. Recibir dictamen sobre P1-P3.
2. Si se congelan parametros: commit con config/settings.py actualizado
   + contrato v1.8 (quitar PROPUESTO de WYCKOFF_SOW_*) + tests que
   verifiquen los valores congelados.
3. Si no se congelan: se abre 5b.4 con la nueva definicion de outcome
   que indique el auditor.
4. En cualquier caso: 5c.4 y 5d siguen bloqueadas hasta cierre de 5b.3.

---

**Fin del expediente 23.**