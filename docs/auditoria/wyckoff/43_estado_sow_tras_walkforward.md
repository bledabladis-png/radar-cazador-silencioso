# 43 - Estado consolidado del analisis SOW

- **Fecha:** 2026-10-04.
- **Motivo:** consolidar la interpretacion de los expedientes 41 y 42
  tras dictamen externo. El expediente 42 contenia una interpretacion
  incorrecta de la candidata C2 ("generaliza OOS", "v2.0").
- **Estado:** documento de referencia. Sustituye la interpretacion del 42.

## 1. Resumen ejecutivo

Se ejecutaron dos experimentos sobre la candidata SOW congelada v1.9
(expediente 41) y un grid de 240 combinaciones (expediente 42). El
resultado obliga a separar radicalmente tres candidatas con niveles de
evidencia distintos:

| Candidata | Parametros | Evidencia | Estado |
|---|---|---|---|
| **C0** (v1.9 congelada) | N=60 M=30 X_ATR=0.25 Y_VOL=1.10 | Walk-forward A/B FAIL | Congelada, referencia |
| **C1** (ganadora TRAIN) | N=40 M=5 X_ATR=1.00 Y_VOL=1.10 | TRAIN seleccion -> TEST D2 PASS (sin purga) | Evidencia temporal historica favorable; pendiente de purga (M+H20) y correccion por multiple testing |
| **C2** (top-1 TEST) | N=60 M=5 X_ATR=0.50 Y_VOL=1.10 | Seleccionada sobre TEST | Exploratoria congelada |

**C1 es la unica candidata con evidencia temporal historica favorable. Pendiente de purga (M+H20) y correccion por multiple testing (240 combos).**
**C2 tiene el mejor numero en TEST, pero fue elegida mirando TEST.**
**Ninguna se activa en produccion.**

## 2. Diseno experimental

    TRAIN  2015-01-02 -> 2020-12-31   (calibracion)
    TEST   2021-01-01 -> 2026-10-01   (evaluacion historica, no OOS virgen)

Historico extendido descargado de Yahoo (`data/stock_prices_extended.parquet`,
10.7 anos, no productivo). Grid de 240 combinaciones:
N in {20,30,40,60}, M in {5,10,15,20,30}, X_ATR in {0.25,0.50,0.75,1.00},
Y_VOL in {1.10,1.20,1.50}. Bootstrap B=2000, seed 20261002, cluster
por ticker.

## 3. Resultados

### 3.1. C0 (v1.9 congelada) - FAIL en walk-forward A/B

Split predefinido plan v3 (A 2021-2023, B 2024-2026):

    A: lift -0.037, IC95 [-0.147, +0.077]  (cruza 0)
    B: lift +0.095, IC95 [+0.043, +0.148]  (positivo)

Criterio PASS (lower_ci > 0 en A y B) no se cumple. FAIL.

### 3.2. Grid completo sobre TRAIN

De las 240 combinaciones, 237 pasan D1+D2+D3 sobre TRAIN. La ganadora
por ranking (lower_ci desc, n_conf desc) es:

    C1 = N=40 M=5 X_ATR=1.00 Y_VOL=1.10

### 3.3. C1 sobre TEST (resultado post-seleccion TRAIN)

C1 se eligio sin ver TEST. Evaluacion sobre TEST:

    TRAIN: lift +0.466, IC95 [+0.406, +0.522]
    TEST:  lift +0.103, IC95 [+0.018, +0.192]  (D2 = PASS)

Cadena valida: seleccion en TRAIN -> evaluacion en TEST independiente.
**C1 tiene evidencia temporal historica favorable. Pendiente de purga y correccion por multiple testing (ver 11bis-11quinquies).**

### 3.4. Grid completo sobre TEST (exploratorio)

240 combinaciones evaluadas sobre TEST. 39/240 (16.2%) pasan D2
(lower_ci > 0). La #1 por lift es:

    C2 = N=60 M=5 X_ATR=0.50 Y_VOL=1.10
    TEST: lift +0.107, IC95 [+0.043, +0.170]

Pero C2 fue elegida DESPUES de ver el resultado de TEST. Su IC no es
un intervalo OOS confirmatorio: es el maximo de 240 evaluaciones sobre
el mismo TEST. Selection bias / winner's curse.

## 4. Correccion de la interpretacion del expediente 42

El expediente 42 afirmaba:

- "C2 generaliza out-of-sample" -> **NO DEMOSTRADO.** La formulacion
  correcta es "C2 presenta el mejor resultado entre 240 configuraciones
  evaluadas en TEST".
- "C2 supera a C0/v1.9 por 4x" -> **NO CONFIRMATORIO.** La comparacion
  esta contaminada por seleccion de C2 sobre TEST.
- "Se propone C2 como candidata v2.0" -> **NO PROCEDE.** C2 se registra
  como candidata exploratoria congelada, no como v2.0 validada.
- "La region M=5 X_ATR>=0.50 era top en TRAIN" -> correcto, pero no
  convierte a C2 en candidata preespecificada antes de TEST. La region
  es una hipotesis; C2 es un punto dentro de esa region elegido con
  informacion de TEST.

## 5. Estado formal de las candidatas

    C0 / v1.9
        Contrato:      v1.9 (FROZEN_FOR_VALIDATION).
        Evidencia:     walk-forward A/B FAIL (expediente 41).
        Estado:        CONGELADA. 5b.X v3 aplicable.
        Produccion:    NO.

    C1
        Parametros:    N=40 M=5 X_ATR=1.00 Y_VOL=1.10.
        Contrato:      ninguno especifico (no congelada).
        Evidencia:     TRAIN -> TEST D2 PASS (sin purga).
                       IC95 TEST [+0.018, +0.192].
        Estado:        REGISTRADA. No activada.
        Produccion:    NO.
        Requisito p/ avanzar:  nuevo protocolo + validacion OOS
                       independiente (post-2026-10-01).

    C2
        Parametros:    N=60 M=5 X_ATR=0.50 Y_VOL=1.10.
        Contrato:      ninguno especifico.
        Evidencia:     top-1 de 240 sobre TEST (seleccion contaminada).
        Estado:        EXPLORATORIA CONGELADA.
        Produccion:    NO.
        Requisito p/ avanzar:  nuevo protocolo + validacion OOS
                       independiente (post-2026-10-01).

## 6. Contratos y umbrales

**5b.X v3** esta disenado especificamente para C0/v1.9:

    N=60 M=30 X_ATR=0.25 Y_VOL=1.10
    Umbral 550 confirmed H20-complete
    Cutoff: 2026-10-01

**No se transfiere automaticamente a C1 ni a C2.** Requisitos para
llevar C1 o C2 a validacion:

1. Nuevo contrato candidato (documento normativo).
2. Nuevo power analysis (el 550 depende de la estructura de episodios
   de C0; C2 cambia M=30->5, puede cambiar frecuencia de SOW,
   ratio confirmed/baseline, distribucion de clusters).
3. Nuevo script de validacion + hash congelado.
4. Validacion OOS independiente (post-2026-10-01).

**No se modifica retrospectivamente el 5b.X v3.**

## 7. Hallazgo sobre la region M=5, X_ATR alto

Aunque C2 como punto individual no esta validada, el grid completo
sobre TEST muestra una concentracion de resultados positivos en la
region M=5 con X_ATR >= 0.50. Esto es una **hipotesis de region**
(exploratoria, sujeta a multiplicidad), no una conclusion.

Comparativa cualitativa entre regiones (exploratorio, no confirmatorio):

- M=5, X_ATR alto: captura rupturas profundas rapidas. Funciona bien
  en 2018, 2022, 2026. Falla en 2025.
- M=30, X_ATR bajo (C0): captura correcciones amplias y lentas.
  Funciona en 2020, 2025, 2026. Falla en 2022.

La semantica mecanistica ("rupturas rapidas" vs "correcciones lentas")
es una hipotesis, no una propiedad demostrada.

## 8. Multiple testing

240 combinaciones evaluadas sobre TEST implican multiplicidad. Bajo
hipotesis nula completa (independencia), el numero esperado de falsos
positivos nominales al 5% seria 240 * 0.05 = 12. Los 39 observados
superan esa cifra, lo cual es interesante pero **no constituye prueba
formal** de una estructura de region, debido a la dependencia entre
candidatas y a la ausencia de un diseno preespecificado para ese test.

## 9. Que queda vigente y que no

    VIGENTE:
        Contrato v1.8 (produccion).
        Contrato v1.9 (candidata congelada).
        5b.X v3 (para C0/v1.9).
        config.WYCKOFF_SOW_* = None / fail-closed.
        Regla: no migrar consumidores por SOW hasta 5b.X o nuevo
        protocolo OOS.

    NO VIGENTE / CORREGIDO:
        "C2 es v2.0".
        "C2 generaliza OOS".
        "C2 supera a v1.9 por 4x" como afirmacion confirmatoria.

    ABIERTO (hipotesis para investigacion futura):
        Region M=5 / X_ATR alto merece un analisis especifico con
        diseno ex-ante (definicion de region, criterio, muestra).
        C1 merece validacion OOS oficial con nuevo protocolo.
        C2 merece solo si se congela ahora y se valida con OOS futuro.

## 10. Proximos pasos

**Inmediato (sin nueva decision):**
- Migracion core v1.8 (4 fases sin SOW). Autorizada. Independiente.

**Requiere dictamen externo:**
- Redactar nuevo contrato candidato C1 (si se decide llevarlo a
  validacion oficial).
- Redactar nuevo contrato candidato C2 (solo si se decide).
- Definir el diseno de la validacion OOS futura para C1/C2.

**Sin accion:**
- 5b.X (2027). Juez final para C0/v1.9.
- config SOW: None. fail-closed.

## 11. Artefactos

- Expediente 41: walk-forward A/B de C0/v1.9 (FAIL).
- Expediente 42: grid 240 + propuesta C2 (interpretacion corregida
  por este documento).
- Expediente 43: este documento (referencia consolidada).
- Scripts:
  - `scripts/walk_forward_sow.py`
  - `scripts/walk_forward_sow_anual.py`
  - `scripts/calibrate_sow_wf.py`
  - `scripts/calibrate_sow_wf_phase3.py`
  - `scripts/compare_sow_candidates.py`
- Datos: `data/stock_prices_extended.parquet` (no productivo, gitignored).

---

## 11bis. Independencia temporal TRAIN/TEST

TRAIN (2015-01-02 -> 2020-12-31) y TEST (2021-01-01 -> 2026-10-01)
se construyeron con gap = 0 sesiones. Un episodio con t0 en los
ultimos M+H20 dias de TRAIN usa precios de TEST para calcular su
label (outcome = t0 + M + H20). Con M en {5, 30} y H20 = 20, el
solape es de 25 a 50 sesiones por episodio en la frontera. TRAIN y
TEST no son independientes como conjuntos de informacion.

Mitigacion (no aplicada a los numeros actuales): embargo de 50
sesiones sobre TRAIN, drop de episodios con t0_idx + M + 20 >
last_train_idx. Con M_max = 30 y H20 = 20, el embargo cubre el
peor caso.

## 11ter. Reutilizacion previa del TEST

El rango 2021-01-01 -> 2026-10-01 no es un holdout virgen a nivel
programa. Antes de la evaluacion walk-forward de C1, ese rango ya
fue observado en exp 41 (walk-forward A/B sobre C0/v1.9) y en
analisis anuales previos. "OOS limpio" en sentido fuerte exige que
ningun procedimiento de seleccion o diagnostico haya visto el TEST.
No es el caso.

## 11quater. Seleccion intra-TRAIN de C1

C1 (N=40, M=5, X_ATR=1.00, Y_VOL=1.10) fue top-1 del grid sobre
TRAIN por criterio `lower_ci desc, n_conf desc` (`_select_winner`
en `calibrate_sow_wf.py`). De los 240 combos, 237 pasaron D1-D3 en
TRAIN (seccion 7), asi que la seleccion opero sobre 237 candidatas, no
sobre 1. El IC95 de C1 en TEST no incorpora la incertidumbre de
esa seleccion intra-TRAIN.

## 11quinquies. Ausencia de correccion por multiple testing

Se evaluaron 240 combinaciones. Bajo H0 (lift = 0) y alpha = 0.05
se esperan 12 falsos positivos por azar (seccion 8: declaro esto como
caveat). No se aplico correccion (Bonferroni, Benjamini-Hochberg
ni equivalente). Los IC reportados en las secciones 3 y 5 son sin corregir.
Con BH sobre 240 tests, es plausible que el limite inferior de C1
en TEST cruce cero.

---

**Fin del expediente 43.**
