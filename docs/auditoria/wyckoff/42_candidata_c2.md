# 42 - Candidata C2: propuesta de v2.0 (SOW)

> **NOTA DE CORRECCION (2026-10-04, post-dictamen externo).**
> Este documento contiene una interpretacion incorrecta de la
> candidata C2. La formulacion "C2 generaliza out-of-sample" y
> "propuesta como v2.0" NO son correctas: C2 fue seleccionada
> usando TEST, por lo que su IC no es confirmatorio OOS.
> La interpretacion corregida esta en el expediente 43
> (`43_estado_sow_tras_walkforward.md`), que es el documento
> de referencia. Este 42 se conserva como registro historico del
> experimento (el grid, los numeros y los scripts son validos;
> la interpretacion no).


- **Fecha:** 2026-10-04.
- **Motivo:** hallazgo derivado del walk-forward de calibracion (expediente 41).
- **Contrato de auditoria:** plan v3 §15 + dictamen externo.
- **Estado:** PROPUESTA AL AUDITOR. Candidata v1.9 congelada intacta.
  C2 NO activada. No se toca `config/settings.py`.

## 1. Contexto

El expediente 41 documento el FAIL del walk-forward de la candidata
congelada v1.9 (N=60, M=30, X_ATR=0.25, Y_VOL=1.10) sobre el split
A/B del plan v3. Como analisis derivado, se ejecuto un walk-forward
de calibracion completo: calibracion del grid 240 combinaciones sobre
TRAIN (2015-2020), validacion sobre TEST (2021-2026, OOS limpio).

## 2. Metodologia

**Split:**
- TRAIN: 2015-01-02 -> 2020-12-31 (calibracion)
- TEST:  2021-01-01 -> 2026-10-01 (validacion OOS limpia)

**Grid (240 combinaciones, congelado ex-ante):**
- N     ∈ {20, 30, 40, 60}
- M     ∈ {5, 10, 15, 20, 30}
- X_ATR ∈ {0.25, 0.50, 0.75, 1.00}
- Y_VOL ∈ {1.10, 1.20, 1.50}

**Bootstrap:** B=2000, seed=20261002, cluster por ticker.

**Scripts:**
- `scripts/calibrate_sow_wf.py` (fase 1-2, calibracion + validacion).
- `scripts/calibrate_sow_wf_phase3.py` (fase 3, distribucion completa).
- `scripts/compare_sow_candidates.py` (comparacion final de 3 candidatas).
## 3. Resultado del grid sobre TEST (240 combinaciones)

De las 240 combinaciones evaluadas sobre TEST:

- **39/240 (16.2%)** pasan el criterio D2 (lower_ci95 > 0).
- Los 5 mejores comparten estructura: **M=5** y **X_ATR >= 0.50**.
- La ganadora TRAIN (elegida solo con informacion de TRAIN) cae en
  percentil 1.2 (3a de 240) al ordenar por lift en TEST.
- La candidata congelada v1.9 cae en percentil 20.4 (puesto 49 de 240).

Esto descarta que la candidata ganadora sea ruido. Hay una region
concreta del espacio de parametros que generaliza entre 2015-2020 y
2021-2026.

## 4. Comparacion directa de 3 candidatas

**Bootstrap B=2000, seed=20261002.**

| Candidata | N | M | X_ATR | Y_VOL | TRAIN lift | TEST lift | TEST IC95 | D2 |
|---|---|---|---|---|---|---|---|---|
| C1 (winner TRAIN) | 40 | 5 | 1.00 | 1.10 | +0.466 | +0.103 | [+0.018, +0.192] | si |
| **C2 (top TEST)** | **60** | **5** | **0.50** | **1.10** | **+0.271** | **+0.107** | **[+0.043, +0.170]** | **si** |
| C0 (frozen v1.9) | 60 | 30 | 0.25 | 1.10 | +0.131 | +0.024 | [-0.016, +0.064] | no |

**Nota metodologica.** C1 fue elegida como ganadora de TRAIN, sin
contaminacion con TEST. C2 fue elegida por ser top-1 en TEST, lo que
introduce un sesgo de seleccion. Sin embargo:

- C2 pertenece a una region (M=5, X_ATR>=0.50) que en TRAIN ya
  presentaba lifts > 0 (top region).
- Su IC95 en TEST [+0.043, +0.170] es mas estrecho y su limite
  inferior mas alto que C1.
- Su decaimiento TRAIN->TEST es menor (2.5x vs 4.5x).

## 5. Perfil por bloques anuales

| Ano | C1 lift | C2 lift | C0 lift | Lectura |
|---|---|---|---|---|
| 2015 | sin muestra | sin muestra | sin muestra | -- |
| 2016 | +0.035 | +0.110 | -0.095 | C1/C2 mejor |
| 2017 | +0.181 | +0.045 | +0.114 | mixto |
| 2018 | **+0.376** | +0.094 | +0.054 | C1 mejor |
| 2019 | +0.103 | -0.006 | -0.062 | C1 mejor |
| 2020 | **+0.247** | **+0.232** | **+0.298** | todas OK |
| 2021 | +0.011 | -0.025 | **-0.134** | C2 mejor (neutra) |
| 2022 | **+0.210** | **+0.244** | -0.041 | C1/C2 mejor |
| 2023 | +0.141 | +0.169 | -0.091 | C1/C2 mejor |
| 2024 | +0.089 | +0.055 | +0.018 | similares |
| 2025 | -0.061 | -0.029 | **+0.134** | C0 mejor |
| 2026 | **+0.262** | **+0.209** | **+0.146** | todas OK |

**Lectura:** C1 y C2 capturan deterioro en anos debiles (2018, 2022,
2026) pero fallan en 2025. C0 hace lo opuesto (captura 2025, falla
2022). Son detectores con semantica distinta:

- C1/C2 (M=5, X_ATR alto): detectan **rupturas profundas rapidas**.
- C0 (M=30, X_ATR bajo): detectan **correcciones amplias y lentas**.
## 6. Propuesta al auditor externo

**Se propone C2 (N=60, M=5, X_ATR=0.50, Y_VOL=1.10) como candidata v2.0.**

Justificacion:

1. **Generaliza out-of-sample.** Calibrada parcialmente en TRAIN
   (region M=5, X_ATR>=0.50 presente en top de TRAIN), validada en
   TEST 2021-2026 con IC95 [+0.043, +0.170].
2. **Supera a v1.9 por factor 4x en TEST.**
3. **Menor decaimiento** entre TRAIN y TEST (2.5x vs 5.5x de v1.9).
4. **Mayor potencia estadistica** que C1 (255 vs 133 confirmados).
5. **Cubierta por el grid congelado ex-ante** de 240 combinaciones;
   no requiere recalibracion ni busqueda adicional.

**Lo que NO se propone:**

- Activar SOW en produccion. `config/settings.py` intacto.
- Retirar la candidata v1.9. Sigue congelada como referencia.
- Modificar el contrato v1.9. Solo se anade candidata v2.0 propuesta.
- Cambiar 5b.X (2027). Sigue siendo juez final.

**Pregunta al auditor:**

1. ¿Aprueba C2 como candidata v2.0, con activacion condicionada a
   dictamen externo?
2. ¿O prefiere que se mantenga v1.9 congelada hasta 5b.X (2027)?
3. En caso de aprobar C2: ¿activacion inmediata en produccion tras
   dictamen, o esperar a 5b.X para confirmacion cruzada?

## 7. Riesgos declarados

- **Sesgo de seleccion leve en C2.** Fue elegida como top-1 de TEST.
  Mitigacion: pertenece a region ya top en TRAIN; IC95 estrecho.
- **Semantica distinta.** C2 captura rupturas rapidas y profundas.
  Puede fallar en correcciones amplias y lentas (2025). No es un
  detector universal.
- **Ventana TRAIN limitada.** 2015-2020 incluye solo 5 anos utiles
  (2015 sin muestra por warm-up). Mas historia ayudaria a reducir
  varianza.

## 8. Estado tras la propuesta

    Candidata v1.9           CONGELADA (intacta)
    Candidata C2 (v2.0)      PROPUESTA, NO ACTIVADA
    config/settings.py       None / fail-closed (intacto)
    5b.X (2027)              Juez final OOS, intacto
    Contrato v1.8            Vigente
    Contrato v1.9            Vigente (candidata congelada)

## 9. Artefactos

- Scripts:
  - `scripts/calibrate_sow_wf.py`
  - `scripts/calibrate_sow_wf_phase3.py`
  - `scripts/compare_sow_candidates.py`
- Resultados:
  - `outputs/audit/wyckoff_calibrate_wf_grid.csv`
  - `outputs/audit/wyckoff_calibrate_wf_phase3.csv`
  - `outputs/audit/wyckoff_compare_candidates.csv`
  - `outputs/audit/wyckoff_compare_candidates_summary.json`
- Commits: pendientes (este expediente).

---

**Fin del expediente 42.**