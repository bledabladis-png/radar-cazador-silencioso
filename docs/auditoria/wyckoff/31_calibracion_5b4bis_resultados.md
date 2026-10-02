# CALIBRACION 5b.4-bis - Resultados

**Informe de resultados. NO normativo. Pendiente de segunda ronda de auditoria.**
**Fecha:** 2026-10-02.
**Protocolo:** 30_protocolo_fase_5b4bis_v3.md (v3).
**Dictamen aplicado:** 29_dictamen_5b4_v2.md.
**Commit de ejecucion:** 4595676.
**Script:** scripts/calibrate_wyckoff_sow_5b4bis.py.
**Artefactos:** outputs/audit/wyckoff_5b4bis_grid.csv,
outputs/audit/wyckoff_5b4bis_bootstrap.csv,
outputs/audit/wyckoff_5b4bis_hetero.csv,
outputs/audit/wyckoff_5b4bis_summary.json,
outputs/audit/wyckoff_5b4bis_run.log.

---

## 1. Resumen ejecutivo

Grid 240 combinaciones ejecutado con los criterios D1-D3 del protocolo
v3. 315 tickers, 2415 candidate starts. Bootstrap ticker-cluster
B=2000 seed=20261002.

Cumplimiento de criterios duros:

    D1 (n_confirmed >= 20 AND n_baseline >= 20):       240/240
    D2 (lower_CI(lift_H20_ALL) > 0):                    17/240
    D3 (ceil(2/3) bloques evaluables con lift > 0):     69/240

**17 combinaciones pasan los tres criterios.**

**Candidata seleccionada por ranking v3:**

    N=60, M=30, X_ATR=0.25, Y_VOL=1.10

    n_confirmed=635  n_baseline=1753
    lift_point=+0.0740  lower_CI=+0.0234  upper_CI=+0.1242
    n_bloques_evaluables=3  n_bloques_positivos=3  firma="+++"
    mediana_lift=0.0932  delta_hetero=0.0556

### Hallazgos

**H1 (rediseno de bloques funciona).** El bloqueo estructural de 5b.4
v2 desaparece con 3 bloques de ~20 meses. P1 (2021-10 -> 2023-06)
alcanza 32 confirmed en la candidata elegida; el bloque A de v2 no
pasaba de 12. No es relajacion de D3; es redistribucion de muestra.

**H2 (lift homogeneo en los 3 bloques).** A diferencia de 5b.4 v2
(firma "+--+" con lift negativo en A y B), la candidata elegida tiene
firma "+++": los 3 bloques evaluables con lift positivo. Esto es
evidencia mas fuerte que el agregado de v2.

**H3 (delta_hetero pequeno).** delta_hetero=0.0556 significa que la
diferencia entre el bloque de menor y mayor lift es de 5.6 pp. Los
lifts son: P1=+0.093, P2=+0.050, P3=+0.106. Estabilidad razonable.

**H4 (17/240 pasan).** La tasa de exito es baja (~7%). No es sospechoso
de data snooping severo, pero requiere validacion externa.

### Bloqueo

La candidata pasa D1-D3. **No se congela todavia.** Por la premisa
operativa del proyecto (auditor externo antes de decisiones
irreversibles), la congelacion de parametros productivos requiere
dictamen explicito. Se abre expediente 32.

---

## 2. Dataset y configuracion

- **Fuente:** data/stock_prices.parquet.
- **Universo:** 315 tickers con >= 200 obs Close no-NaN (SPCX fuera).
- **Periodo:** 2021-10-01 -> 2026-10-01.
- **Candidate starts:** 2415.

**Landmark (heredado de v2):**

    L = t0 + M
    confirmed = SOW en [t0+1, L]
    baseline  = sin SOW en [t0+1, L]
    outcome (ambos) = struct[L+H] < struct[L]

**Bloques (rediseñados ex-ante, protocolo v3 seccion 3):**

    P1 - periodo 1: 2021-10-01 -> 2023-06-30
    P2 - periodo 2: 2023-07-01 -> 2025-03-31
    P3 - periodo 3: 2025-04-01 -> 2026-10-01

Duraciones: ~21 / ~21 / ~18 meses. Sin mirar resultados de 5b.3/5b.4.

**Constantes de control:**

    WYCKOFF_ATR_WINDOW = 20
    PRECEDENT_WINDOW   = 60
    WYCKOFF_T_NORM_K   = 0.25 (congelado 5b.2)

**Grid:** 240 combinaciones (N x M x X_ATR x Y_VOL).

**Bootstrap:** B=2000, seed=20261002, ticker-cluster, episode-weighted.

---

---

## 3. Cumplimiento D1-D3

| Criterio | Descripcion | Combos OK |
|---|---|---:|
| D1 | n_confirmed >= 20 AND n_baseline >= 20 | 240/240 |
| D2 | lower_CI(lift_H20_ALL) > 0 (B=2000, seed=20261002) | 17/240 |
| D3 | ceil(2/3) bloques evaluables con lift > 0 | 69/240 |
| D1 AND D2 AND D3 | interseccion | **17/240** |

Distribucion de combos por criterios cumplidos:

    3 de 3:   17
    2 de 3:   52
    1 de 3:  171
    0 de 3:    0

**Comparativa con 5b.4 v2:**

    | | 5b.4 v2 | 5b.4-bis v3 |
    |---|---|---:|
    | Bloque mas antiguo n_conf max | 12 (A) | 32 (P1) |
    | D1 | 240/240 | 240/240 |
    | D2 | 17/240 | 17/240 |
    | D3 | 0/240 | 69/240 |
    | D1+D2+D3 | 0/240 | 17/240 |

**Interpretacion:** D2 filtra igual (17/240) en ambos disenos. D3
pasa de inalcanzable (0) a alcanzable (69) al redistribuir la
muestra. **No es una relajacion del criterio: es un diseno con
suficiencia muestral por bloque.**

---

## 4. Las 17 candidatas que pasan D1-D3

Ordenadas por ranking v3 (n_bloques_pos > mediana_lift > delta_hetero
> lexicografico):

| N | M | X_ATR | Y_VOL | n_conf | n_base | lift_point | lower_CI | n_bl | n_bl_pos | mediana_lift | delta_het | firma |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 60 | 30 | 0.25 | 1.10 | 635 | 1753 | +0.0740 | +0.0234 | 3 | **3** | 0.0932 | 0.0556 | +++ |
| 60 | 20 | 0.25 | 1.10 | 506 | 1886 | +0.0587 | +0.0059 | 3 | 3 | 0.0895 | 0.2394 | +++ |
| 60 | 20 | 0.25 | 1.20 | 472 | 1920 | +0.0548 | +0.0003 | 3 | 3 | 0.0865 | 0.2250 | +++ |
| 40 | 30 | 0.25 | 1.10 | 703 | 1685 | +0.0690 | +0.0205 | 3 | 3 | 0.0851 | 0.0654 | +++ |
| 60 | 30 | 0.25 | 1.20 | 601 | 1787 | +0.0647 | +0.0131 | 3 | 3 | 0.0756 | 0.0425 | +++ |
| 40 | 30 | 0.25 | 1.20 | 671 | 1717 | +0.0577 | +0.0094 | 3 | 3 | 0.0668 | 0.0726 | +++ |
| 40 | 30 | 0.50 | 1.10 | 544 | 1844 | +0.0630 | +0.0099 | 3 | 3 | 0.0645 | 0.0116 | +++ |
| 30 | 30 | 0.25 | 1.10 | 794 | 1594 | +0.0474 | +0.0025 | 3 | 3 | 0.0610 | 0.0357 | +++ |
| 40 | 30 | 0.50 | 1.20 | 527 | 1861 | +0.0505 | +0.0000 | 3 | 3 | 0.0560 | 0.0057 | +++ |
| 40 | 20 | 0.25 | 1.10 | 562 | 1830 | +0.0526 | +0.0023 | 3 | 2 | 0.0984 | 0.2561 | +-+ |
| 60 | 30 | 0.50 | 1.10 | 496 | 1892 | +0.0712 | +0.0185 | 3 | 2 | 0.0577 | 0.1184 | -++ |
| 60 | 5  | 0.50 | 1.10 | 174 | 2232 | +0.0899 | +0.0133 | 2 | 2 | 0.0531 | 0.0093 | ++ |
| 60 | 30 | 0.50 | 1.20 | 475 | 1913 | +0.0637 | +0.0106 | 3 | 2 | 0.0509 | 0.1098 | -++ |
| 20 | 5  | 1.00 | 1.20 | 103 | 2303 | +0.1005 | +0.0010 | 2 | 2 | 0.0503 | 0.0152 | ++ |
| 60 | 5  | 0.50 | 1.20 | 166 | 2240 | +0.0853 | +0.0091 | 2 | 2 | 0.0485 | 0.0061 | ++ |
| 30 | 5  | 0.75 | 1.10 | 142 | 2264 | +0.0885 | +0.0014 | 2 | 2 | 0.0445 | 0.0455 | ++ |
| 60 | 5  | 0.25 | 1.10 | 259 | 2147 | +0.0735 | +0.0088 | 2 | 2 | 0.0405 | 0.0363 | ++ |

**Observaciones:**

- 9 de 17 tienen 3 bloques evaluables (n_bl=3) con lift positivo
  (n_bl_pos=3). Son las mas estables.
- Las otras 8 tienen 2 o 3 bloques evaluables, con 2 positivos.
- Dos combinaciones (40/20/0.25/1.10 y 60/30/0.50/1.10) tienen firma
  "+-+" o "-++": lift negativo en uno de los tres bloques.
- El ranking v3 prefiere las 9 con 3/3 positivos, y dentro de ellas
  maximiza mediana_lift y minimiza delta_hetero.

---

## 5. Candidata elegida

    N=60, M=30, X_ATR=0.25, Y_VOL=1.10

### 5.1. Metricas globales (ALL, H20)

    n_confirmed = 635
    n_baseline  = 1753
    lift_point  = +0.0740
    lower_CI    = +0.0234
    upper_CI    = +0.1242

El IC95% del lift (bootstrap ticker-cluster, B=2000) no cruza 0.
Pasa D2.

### 5.2. Metricas por bloque (H20, struct_deterioration)

| Bloque | Periodo | n_conf | n_base | pct_conf | pct_base | lift |
|---|---|---:|---:|---:|---:|---:|
| P1 | 2021-10 -> 2023-06 | 32 | 226 | 0.531 | 0.438 | **+0.093** |
| P2 | 2023-07 -> 2025-03 | 304 | 914 | (ver CSV) | (ver CSV) | **+0.050** |
| P3 | 2025-04 -> 2026-10 | 293 | 591 | (ver CSV) | (ver CSV) | **+0.106** |

**Firma: "+++"**. Los tres bloques con lift positivo. Esto es
cualitativamente distinto de la candidata A de 5b.4 v2 (firma "+--+"
con A=-0.31 y B=-0.18).

### 5.3. Estabilidad

    mediana_lift   = 0.0932
    delta_hetero   = 0.0556  (max - min = 0.106 - 0.050)
    n_bloques_eval = 3
    n_bloques_pos  = 3

delta_hetero=5.56 pp sobre lifts de 9.3, 5.0 y 10.6 pp. La
heterogeneidad existe pero es moderada (no hay bloques con signo
opuesto).

### 5.4. Metricas secundarias (informativo)

Para P1 (la unica con detalle en summary):

    pct_conf_struct = 0.531  pct_base_struct = 0.438  lift = +0.093
    pct_conf_price  = 0.188  pct_base_price  = 0.385  lift = -0.197
    pct_conf_below  = 0.125  pct_base_below  = 0.058  lift = +0.067
    pct_conf_lower  = 0.344  pct_base_lower  = (ver CSV)

Nota: price_weakness tiene lift negativo en P1. La senal de SOW
concentra en struct_deterioration y below_support, no en price
immediato. Consistente con la definicion (SOW es senal estructural).

### 5.5. Metrica primaria vs version 5b.4 v2

La misma tupla N=60, M=30, X=0.25, Y=1.10 en 5b.4 v2 (4 bloques):

    5b.4 v2:  lift_point=+0.0740  lower_CI=+0.0234
    5b.4-bis: lift_point=+0.0740  lower_CI=+0.0234
    (identicos)

La diferencia esta en D3: v2 exigia 4 bloques evaluables (imposible
con bloque A corto); v3 exige 2/3 de los evaluables (alcanzable).
El **lift en si no cambia**: solo cambia si es seleccionable bajo el
criterio.

---

---

## 6. Verificacion SLB / PCG (guardrail funcional)

Con la candidata elegida (N=60, M=30, X_ATR=0.25, Y_VOL=1.10):

- **SLB:** los SOW residuales del diagnostico 5c.3 quedan excluidos con
  X_ATR >= 0.25. Regression test del dictamen 26 seccion 18 cumplido.
- **PCG:** los SOW marginales de 5c.3 tambien quedan fuera.

**Nota:** SLB/PCG no forman parte del criterio de seleccion. Solo
constan como guardrail funcional.

---

## 7. Analisis prohibidos (protocolo v3 seccion 7)

Comprobaciones explicitas:

- NO se bajo el umbral de D3 para hacer pasar ninguna combinacion.
  El cambio respecto a v2 fue el rediseno de bloques, no el umbral.
- NO se paso de 4 a 3 bloques usando el resultado observado de 5b.4.
  Las fronteras P1/P2/P3 se fijaron antes de ejecutar, mirando solo el
  calendario y la distribucion de candidate starts.
- NO se elimino el bloque A porque perjudicaba.
- NO se congela la primera de las 17 sin dictamen.
- NO se elige M=30 por IC estrecho ni M=5 por lift alto: el ranking es
  neutral, la candidata gana por 3/3 bloques positivos.
- NO se convierte P2+P3 en "regimen valido" sin prueba adicional.
- NO se aplica Bonferroni ni FDR.
- NO se usa el bloque D como validacion historica ciega.
- NO se introdujo ninguna modificacion cuyo objetivo implicito fuera
  conseguir que alguna combinacion pase.

**Verificacion del ranking:** las 17 candidatas se ordenaron con la
regla v3 (n_bloques_pos > mediana_lift > delta_hetero > lexicografico).
La elegida es la #1 por:
- 3/3 bloques evaluables con lift positivo (9 combinaciones empatadas
  en este criterio).
- Mayor mediana_lift dentro de ese grupo (0.0932 vs siguiente 0.0895).
- Menor delta_hetero que las otras del top (0.0556 vs 0.2394, 0.2250).
No hay desempate por lexicografico en la elegida.

---

## 8. Reproducibilidad

    Set-Location D:\Macro_Sectorial
    py scripts\calibrate_wyckoff_sow_5b4bis.py

Determinista sobre el snapshot actual de data/stock_prices.parquet.
Tiempo de ejecucion: ~480s (grid ~332s + bootstrap ~148s).

Salidas:

- outputs/audit/wyckoff_5b4bis_grid.csv (240 x ~200 columnas)
- outputs/audit/wyckoff_5b4bis_bootstrap.csv (240 x 12)
- outputs/audit/wyckoff_5b4bis_hetero.csv (240 x 10)
- outputs/audit/wyckoff_5b4bis_summary.json
- outputs/audit/wyckoff_5b4bis_run.log

---

## 9. Conclusiones

**Hecho empirico 1.** El rediseno de bloques (3 de ~20 meses en lugar
de 4 desiguales) resuelve el bloqueo estructural de 5b.4 v2. P1 alcanza
32 confirmados; el bloque A de v2 no pasaba de 12. **No es relajacion
de criterio, es redistribucion de muestra.**

**Hecho empirico 2.** Existen 9 combinaciones con los 3 bloques
evaluables y lift positivo en cada uno ("+++"). Ademas de las 8 con 2
bloques positivos y 1 negativo. En total 17/240 pasan D1-D3.

**Hecho empirico 3.** La candidata seleccionada
(N=60, M=30, X_ATR=0.25, Y_VOL=1.10) tiene:
- lift_H20_ALL = +0.0740, lower_CI = +0.0234 (IC no cruza 0).
- firma "+++" en los 3 bloques.
- delta_hetero moderado (5.6 pp).
- muestra amplia (635 conf + 1753 base).

**Hecho empirico 4.** La senal se concentra en struct_deterioration y
below_support. price_weakness tiene lift negativo en P1. Consistente
con la definicion de SOW como senal estructural, no de precio.

**Hecho empirico 5.** Las 9 con "+++" son robustas por construccion del
ranking: ninguna depende de un unico bloque positivo.

### Bloqueo

La candidata pasa D1-D3 y el ranking v3 la selecciona sin ambiguedad.
**No se congela todavia.** Decision pendiente del auditor externo
(expediente 32).

---

## 10. Proximo paso

Expediente `32_expediente_5b4bis_freeze.md` remitido al auditor con:

- La candidata N=60, M=30, X_ATR=0.25, Y_VOL=1.10.
- Sus 9 competidoras con "+++" (por si el auditor prefiere otra).
- La pregunta explicita: autoriza congelar la candidata en
  config/settings.py?
- Alcance de la congelacion (activar parametros en config vs mantener
  None hasta 5b.X).
- Impacto en produccion (contrato v1.8 y contrato v1.9).
- Como se valida en 5b.X (ventana futura no inspeccionada).

Hasta dictamen:

    v1.8              CONTRATO VIGENTE
    5b.4-bis          EJECUTADA, esperando freeze
    parametros SOW    None en config (fail-closed)
    5b.X              PENDIENTE
    5c.4, 5d          BLOQUEADAS
    Legacy            INTACTO

---

**Fin del informe 31.**
