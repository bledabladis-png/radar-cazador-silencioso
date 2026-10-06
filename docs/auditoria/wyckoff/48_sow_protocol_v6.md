# 48 - Protocolo SOW v6 (BORRADOR)

**Estado:** BORRADOR - NO NORMATIVO. Pendiente de firma del auditor externo.
**Fecha de redaccion:** 2026-10-06.
**Autoriza:** 46_dictamen_sow_v5_grounded.md (2026-10-04).
**Objeto:** especificar el protocolo de validacion v6 del detector SOW.


**Etiquetas usadas en este documento:**
- `[F46]` — fijado explicitamente por el dictamen 46. No modificable sin nuevo dictamen.
- `[PF]` — pendiente de firma del auditor externo. El redactor propone un default; el auditor fija el valor final.

---

## 1. Contexto y motivacion

El dictamen 46 (2026-10-04) rechaza el estimando "grounded v5" por un problema estructural: el modelo se ajustaba y se evaluaba sobre la misma ventana OUTER TEST. Eso invalida la interpretacion como validacion externa (nested validation).

El resultado de la ejecucion v5 (`decision.estado = MUESTRA INSUFICIENTE`, `N_eval = 2/8`) se archiva como **"no estimable bajo el estimando grounded v5"**, NO como "SOW tiene muestra insuficiente" ni como "SOW no discrimina".

El dictamen 46 especifica un estimando alternativo (v6) que:
- Vuelve a nested validation estricta.
- Usa un estimador de un paso / AIPW con nuisance models ajustados en OUTER TRAIN.
- Anade propensity/overlap con soporte ex-ante.
- Separa dos estimandos: primario (contraste de riesgo ajustado) y secundario (robustez predictiva via Brier/LogLoss).
- Define reglas de N_eval por regimen, no solo global.
- Obliga a corregir S-01 (manifest) y S-02 (asimetria placebo/paradigma) antes de la siguiente ejecucion.
---

## 2. Hipotesis pre-registrada

`[F46]` v6 evalua si el detector SOW aporta informacion sobre la variable Y = "deterioro a H20" que no este ya contenida en:
- R (regimen de mercado),
- X0 (controles por sector/ticker).

El contraste es por regimen: `RD_stress` y `RD_normal` separados. No se admite una conclusion "universal" que oculte un regimen con muy pocas observaciones.

`[PF]` Enunciado formal exacto de la hipotesis nula y alternativa. Propuesta del redactor:
- H0: `RD_stress = 0` Y `RD_normal = 0`.
- H1a (universal): `RD_stress != 0` Y `RD_normal != 0`, misma direccion.
- H1b (condicional): `RD_stress != 0`, `RD_normal ~ 0`.
- H1c (no): ambos ~ 0.
---

## 3. Diseno experimental

### 3.1. Estructura nested

`[F46]` La estructura de validacion es:

```
INNER TRAIN --> seleccion de 240 combinaciones --> 1 candidato fijo
                                                     |
                                                     v
OUTER TRAIN --> ajuste de nuisance models -------> m(Y|.), e(SOW|.)
                                                     |
                                                     v
OUTER TEST  --> Y observado + SOW observado -----> RD^AIPW
                                                     |
                                                     v
                                              MBB B20/B40
                                                     |
                                                     v
                                        placebo A / placebo B
                                                     |
                                                     v
                                             Brier / LogLoss
                                                     |
                                                     v
                                       decision universal / condicional / no
```

`[F46]` OUTER TEST **nunca** se usa para seleccionar candidato ni para ajustar modelo. Si un OUTER TEST tiene `n_normal = 0` (o muy pocos), el modelo NO se re-estima dentro de ese TEST; se evalua el modelo ya ajustado en OUTER TRAIN, y `RD_normal` puede resultar "no estimable" sin invalidar `RD_stress`.

### 3.2. Candidata evaluada

`[PF]` Candidata concreta. Opciones:
- (a) C0/v1.9 congelada: `N=60 M=30 X_ATR=0.25 Y_VOL=1.10` (contractual hoy).
- (b) C1: `N=40 M=5 X_ATR=1.00 Y_VOL=1.10` (evidencia temporal historica favorable, pendiente de purga).
- (c) Region M=5 X_ATR >= 0.50 (hipotesis de region).

Propuesta del redactor: (a) C0/v1.9. Motivo: es la unica con contrato congelado; C1/C2 necesitarian nuevo contrato candidato (segun expediente 43 seccion 10).

### 3.3. Datos y ventana temporal

`[PF]` Ventana TRAIN y TEST:
- TRAIN extendido: 2015-01-02 a 2020-12-31 (`data/stock_prices_extended.parquet`, no productivo).
- TEST: 2021-01-01 a 2026-10-01.

`[PF]` Embargo entre TRAIN y TEST: 50 sesiones (M_max + H20 en el peor caso).

### 3.4. Universo

`[PF]` Universo de tickers. Propuesta del redactor: mismo que expedientes 41/42 (315 tickers, top-20 por sector + catalogo radar).

---

## 4. Estimandos

### 4.1. Primario - contraste de riesgo ajustado (RD AIPW)

`[F46]` Para cada episodio i:
- `m_a(X,R) = P(Y=1 | SOW=a, X, R)`, ajustado en OUTER TRAIN.
- `e(X,R) = P(SOW=1 | X, R)`, ajustado en OUTER TRAIN.

Estimador de un paso / AIPW sobre OUTER TEST:

```
RD_hat = (1/n) * sum_i [
    m_1(X_i, R_i) - m_0(X_i, R_i)
    + (SOW_i / e_i) * (Y_i - m_1(X_i, R_i))
    - ((1 - SOW_i) / (1 - e_i)) * (Y_i - m_0(X_i, R_i))
]
```

Y lo mismo estratificado por `stress` y `normal`.

`[F46]` El contraste se mantiene en **puntos porcentuales**. Se conserva el umbral `delta_min = 5 pp`.

`[F46]` No se denomina "efecto causal". Se denomina **"contraste de riesgo ajustado asociado a SOW"** (SOW no es aleatorizado).

### 4.2. Secundario - robustez predictiva (Delta Brier / Delta LogLoss)

`[F46]` Dos modelos ajustados exclusivamente en OUTER TRAIN:
- Modelo completo: `SOW + R + X0`.
- Modelo baseline: `R + X0`.

En OUTER TEST, con `Y` realmente observado:
- `Delta Brier = Brier_baseline - Brier_SOW`.
- `Delta LogLoss = LogLoss_baseline - LogLoss_SOW`.

`[F46]` El objetivo es responder: "SOW anade capacidad predictiva fuera de muestra?". Un valor positivo de Delta Brier/Delta LogLoss indica mejora.

---

## 5. Propensity / overlap

`[F46]` El estimando AIPW introduce `e(X,R)`. Debe existir un soporte ex-ante:

```
epsilon <= e(X,R) <= 1 - epsilon
```

con `epsilon` preespecificado. Si demasiadas observaciones caen fuera del soporte, el regimen/fold se declara **no estimable**.

`[F46]` **No se decide `epsilon` despues de ver los resultados.**

`[PF]` Valor concreto de `epsilon`. Propuesta del redactor: `epsilon = 0.05`. Motivo: frecuencia SOW ~6%, evita pesos extremos.

---

## 6. Interaccion SOW * R

`[F46]` La interaccion `SOW * R`:
- Se estima en **OUTER TRAIN**.
- NO fold-por-fold.
- Si un OUTER TEST tiene pocos `normal`, eso afecta la **precision** de `RD_normal`, no obliga a cambiar el modelo.

`[F46]` NO se elimina la interaccion para evitar `LinAlgError` en folds concretos. Eso seria "model surgery despues de observar que folds fallan".

---

## 7. Regla de N_eval

`[F46]` Se separan:
- `N_eval_stress`: folds con soporte suficiente en regimen stress.
- `N_eval_normal`: folds con soporte suficiente en regimen normal.
- `N_eval_total`: total evaluables.

`[F46]` No se admite que `N_eval_total = 7` oculte `N_eval_normal = 2` cuando se declara conclusion universal.

`[PF]` Umbrales minimos. Propuesta del redactor:
- `N_eval_total >= 6` (mantener el minimo actual).
- `N_eval_normal >= 4` para declarar "universal".
- Si `N_eval_normal < 4` pero `N_eval_stress >= 6`, solo se admite conclusion "condicional a stress".

---

## 8. Bootstrap y placebos

`[F46]` Bootstrap MBB B20/B40 para IC.

`[F46]` REAL, PLACEBO A y PLACEBO B pasan por el **mismo estimador final**, cambiando solamente la asignacion placebo de SOW. Sin asimetria paradigmatica.

`[PF]` Definicion exacta de PLACEBO A y PLACEBO B. Propuesta del redactor:
- PLACEBO A: permutacion aleatoria de SOW dentro de cada fold.
- PLACEBO B: SOW retardado (lag N) para romper coincidencia temporal con Y.

---

## 9. Prohibiciones

`[F46]` No se hara ninguna de estas cuatro cosas:
- Fold 4 - quitar interaccion porque no converge.
- Fold 5 - quitar `normal`.
- Fold 6 - quitar `D2_prev`.
- Fold 7 - quitar `D3_prev`.

Tampoco:
- Combinar 2020+2021+2022 "porque asi hay mas normal".
- Usar `Y_TEST` para recalibrar el modelo y luego declarar ese mismo TEST como evidencia exterior.

---

## 10. S-01 y S-02

### 10.1. S-01 - Manifest de trazabilidad

`[F46]` CORREGIR antes de cualquier ejecucion posterior. El manifest debe contener:
- `protocol_version`
- `protocol_sha256`
- `code_base_commit`
- `prereg_commit`
- `data_sha256`

`[F46]` Tener `protocol_version = "v4.1"` cuando se ejecuta v5 es fallo de trazabilidad no aceptable para un experimento confirmatorio.

### 10.2. S-02 - Simetria placebo / paradigma

`[F46]` REAL, PLACEBO A, PLACEBO B deben pasar por el mismo estimador final. Actualmente placebo usa "paradigma predictivo" y bootstrap usa "paradigma grounded" - objetos estadisticos distintos.

---

## 11. Criterios de decision

`[F46]` La decision final es una de:
- **Universal:** SOW anade informacion en stress Y en normal.
- **Condicional:** SOW anade informacion solo en un regimen (tipicamente stress).
- **No:** SOW no anade informacion en ninguno.

`[PF]` Criterios numericos concretos. Propuesta del redactor:

Universal:
- `RD_stress` con IC95 lower > 0, y `RD_normal` con IC95 lower > 0.
- `N_eval_stress >= 6` y `N_eval_normal >= 4`.
- `Delta Brier > 0` y `Delta LogLoss > 0` en el modelo completo vs baseline.
- Placebos A y B con `RD` dentro del IC95 del efecto nulo (o al menos sin cruzar `delta_min`).

Condicional:
- `RD_stress` con IC95 lower > 0.
- `N_eval_stress >= 6`.
- `N_eval_normal < 4` (por falta de soporte, no por efecto nulo).
- `Delta Brier > 0` en stress.

No:
- `RD_stress` cruza 0 o `N_eval_stress < 6`.
- O placebos indistinguibles del real.

---

## 12. Trazabilidad y pre-registro

`[F46]` El protocolo se pre-registra antes de la ejecucion:
- Fichero con hash congelado.
- Commit especifico.
- Manifiesto de ejecucion con los 5 campos de S-01.

`[PF]` Forma concreta del pre-registro (fichero, ubicacion, proceso de congelacion).

---

## 13. Estado de este documento

BORRADOR. Los campos `[PF]` requieren firma del auditor externo. Los campos `[F46]` provienen del dictamen 46 y no son modificables sin nuevo dictamen.

La implementacion de codigo NO debe comenzar hasta que:
1. Este documento sea firmado.
2. Se corrijan S-01 y S-02.
3. Se cree el pre-registro con hash.

---

**Referencias:**
- `46_dictamen_sow_v5_grounded.md` (dictamen externo, 2026-10-04).
- `47_estado_sow_tras_v5_grounded.md` (estado, 2026-10-05).
- `41_walk_forward_sow.md`, `42_candidata_c2.md`, `43_estado_sow_tras_walkforward.md`.
- `01_contrato_semantico_v1_9.md` (candidata congelada).
- `35_protocolo_5bX_v3.md` (protocolo 5b.X, referencia cruzada para 2027).
