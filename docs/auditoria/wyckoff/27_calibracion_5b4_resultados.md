# CALIBRACION 5b.4 - Resultados

**Informe de resultados. NO normativo. Pendiente de segunda ronda de auditoria.**
**Fecha:** 2026-10-02.
**Protocolo:** 25_protocolo_fase_5b4.md (v2).
**Dictamen aplicado:** 26_dictamen_5b4.md.
**Commit de ejecucion:** 9fb6bcf (script en el mismo commit).
**Script:** scripts/calibrate_wyckoff_sow_5b4.py.
**Artefactos:** outputs/audit/wyckoff_5b4_grid.csv (240 filas),
outputs/audit/wyckoff_5b4_bootstrap.csv, outputs/audit/wyckoff_5b4_summary.json,
outputs/audit/wyckoff_5b4_run.log.

---

## 1. Resumen ejecutivo

Grid completo (240 combinaciones) ejecutado con landmark L = t0 + M y
ventana de exposicion [t0+1, L], conforme al protocolo v2. 315 tickers,
2415 candidate starts. Bootstrap ticker-cluster B=2000 seed=20261002.

Cumplimiento de criterios duros:

    D1 (n_confirmed >= 20 AND n_baseline >= 20):       240/240
    D2 (lower_CI(lift_H20) > 0):                        17/240
    D3 (4 bloques evaluables + >= 3 con lift > 0):       0/240

**Ninguna combinacion pasa D1-D3.**

### Hallazgos clave

**H1 (confirma el sesgo).** El lift_H20 se ha reducido a ~1/3 del valor
de 5b3. Para la Candidata A (N=60,M=30,X=0.50,Y=1.20):
- 5b3: lift_point = +0.1920
- 5b4: lift_point = +0.0637

El landmark funciona. El +0.192 obtenido bajo el diseno de 5b3
no es una estimacion valida bajo el diseno temporal corregido; al
aplicar el landmark, el lift de la misma configuracion se reduce a
+0.0637.

**H2 (hallazgo nuevo).** El lift no es homogeneo entre bloques. Para
la misma candidata A:

    Bloque A: lift = -0.31  (n_conf=2, n_base=108)
    Bloque B: lift = -0.18  (n_conf=90, n_base=387)
    Bloque C: lift = +0.11  (n_conf=240, n_base=871)
    Bloque D: lift = +0.11  (n_conf=138, n_base=524)

El SOW tiene lift **negativo** en los bloques temporales 1-2 y
**positivo** en los bloques 3-4. El "lift positivo universal" de 5b3
ya no se sostiene.

**H3 (problema de diseno del protocolo).** D3 exige "4 bloques
evaluables + >= 3 con lift > 0". El bloque A **nunca alcanza** 20
confirmed (maximo observado en la grid: 12). Por tanto **ninguna
combinacion puede tener 4 bloques evaluables**, y D3 es
**inalcanzable por diseno**, independientemente del SOW.

**Consecuencia:** ninguna combinacion pasa D1-D3. 5b.4 no selecciona
parametros. Se abre expediente 28 para dictamen.

---

## 2. Dataset y configuracion

- **Fuente:** data/stock_prices.parquet.
- **Universo:** 315 tickers con >= 200 obs Close no-NaN (SPCX fuera).
- **Periodo:** 2021-10-01 -> 2026-10-01 (1254 sesiones).
- **Candidate starts totales:** 2415.

**Landmark y ventana (protocolo v2):**

    L = t0 + M
    confirmed = SOW en [t0+1, L]
    baseline  = sin SOW en [t0+1, L]
    outcome (ambos) = struct[L+H] < struct[L]

**Constantes de control:**

    WYCKOFF_ATR_WINDOW = 20
    PRECEDENT_WINDOW   = 60
    WYCKOFF_T_NORM_K   = 0.25 (congelado 5b.2)

**Bloques temporales (por fecha del landmark L):**

    A  2021-10-01 -> 2022-12-31
    B  2023-01-01 -> 2024-03-31
    C  2024-04-01 -> 2025-08-31
    D  2025-09-01 -> 2026-10-01

**Grid:** 240 combinaciones (N x M x X_ATR x Y_VOL).

**Bootstrap:** B=2000, seed=20261002, ticker-cluster, episode-weighted.

---

## 3. Cumplimiento D1-D3

| Criterio | Descripcion | Combos OK |
|---|---|---:|
| D1 | n_confirmed >= 20 AND n_baseline >= 20 | 240/240 |
| D2 | lower_CI(lift_H20) > 0 (B=2000, seed=20261002) | 17/240 |
| D3 | 4 bloques evaluables + >= 3 con lift > 0 | **0/240** |

Distribucion de combos por numero de criterios cumplidos:

    3 de 3:    0
    2 de 3:   17   (D1+D2, sin D3)
    1 de 3:  223   (solo D1)
    0 de 3:    0

**Bloqueo doble:** D2 elimina 223/240 y D3 elimina las 17 restantes.

---

## 4. El lift con landmark: comparativa con 5b3

Para la Candidata A (N=60, M=30, X=0.50, Y=1.20), identica tupla en
ambos experimentos:

| | 5b3 (anclaje asimetrico) | 5b4 (landmark) | Ratio |
|---|---:|---:|---:|
| lift_point H20 | +0.1920 | +0.0637 | 0.33 |
| cal_n_confirmed | 406 | 475 | — |
| cal_n_base | 1212 | 1913 | — |

**Interpretacion:** el +0.192 de 5b3 no es una estimacion valida
bajo el diseno temporal corregido. Al aplicar el landmark, el lift de
la misma configuracion se reduce a +0.0637 (ratio ~0.33).

**Nota de lenguaje (dictamen 29 seccion 0):** no debe escribirse
`0.1920 - 0.0637 = 0.1283` como "el sesgo". Es la diferencia entre dos
estimaciones obtenidas bajo disenos diferentes. Otros factores cambian
entre 5b3 y 5b4 (composicion de muestra, episodios elegibles,
n_confirmed/n_baseline), por lo que esos 12.83 pp no son atribuibles
causalmente al sesgo de anclaje.

---

## 5. El lift no es homogeneo entre bloques

Para la Candidata A (misma tupla), H20 struct_deterioration:

| Bloque | Periodo | n_conf | n_base | lift_H20 |
|---|---|---:|---:|---:|
| A | 2021-10 -> 2022-12 | 2 | 108 | **-0.31** |
| B | 2023-01 -> 2024-03 | 90 | 387 | **-0.18** |
| C | 2024-04 -> 2025-08 | 240 | 871 | +0.11 |
| D | 2025-09 -> 2026-10 | 138 | 524 | +0.11 |
| ALL | — | 470 | 1890 | +0.0637 |

**El signo del lift depende del bloque temporal.** Negativo en A-B,
positivo en C-D. El "lift positivo universal" de 5b3 no sobrevive la
correccion.

Otros top-5 por lift_point muestran el mismo patron (lift positivo en
ALL, negativo en A y B).

**Hipotesis (no confirmada):** el SOW discrimina deterioro futuro en
regimenes post-2024 (recuperacion/tendencia) pero no en regimenes
2021-2023 (bajista y rebote). Requiere analisis especifico. No se
concluye.

---

## 6. Bloque A: D3 inalcanzable por diseno

Distribucion del numero de bloques evaluables (con n_conf >= 20 AND
n_base >= 20) sobre las 240 combinaciones:

    n_bloques_evaluables = 1:   3 combos
    n_bloques_evaluables = 2:   3 combos
    n_bloques_evaluables = 3: 234 combos
    n_bloques_evaluables = 4:   0 combos

**Ninguna combinacion tiene los 4 bloques evaluables.** Bloque A
maximo: 12 confirmed en toda la grid (minimo: 1, mediana: 1).

**Causa:** el bloque A (2021-10 -> 2022-12) es el mas corto y tiene
pocos candidate episodes que cumplan las condiciones v1.8. Con muestras
tan bajas de confirmed, D3 en su forma actual no es alcanzable por
ninguna configuracion del SOW.

**Diagnostico:** D3 no es un criterio que mida la calidad del SOW. Es
un criterio que mide si los bloques tienen suficiente muestra
disponible. En el bloque A, **no la tienen**.

---

## 7. Top candidatas (informativo, no elegidas)

### 7.1. Top-10 por lower_CI(lift_H20)

| N | M | X_ATR | Y_VOL | n_conf | n_base | lift_point | lower_CI | D1 | D2 | D3 | n_bloques_eval | n_bloques_pos |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|
| 60 | 30 | 0.25 | 1.10 | 635 | 1753 | +0.074 | **+0.0234** | OK | OK | FALLA | 3 | 2 |
| 40 | 30 | 0.25 | 1.10 | 703 | 1685 | +0.069 | +0.0205 | OK | OK | FALLA | 3 | 2 |
| 60 | 30 | 0.50 | 1.10 | 496 | 1892 | +0.071 | +0.0185 | OK | OK | FALLA | 3 | 2 |
| 60 | 5  | 0.50 | 1.10 | 174 | 2232 | +0.090 | +0.0133 | OK | OK | FALLA | 3 | 2 |
| 60 | 30 | 0.25 | 1.20 | 601 | 1787 | +0.065 | +0.0131 | OK | OK | FALLA | 3 | 2 |
| 60 | 30 | 0.50 | 1.20 | 475 | 1913 | +0.064 | +0.0106 | OK | OK | FALLA | 3 | 2 |
| 40 | 30 | 0.50 | 1.10 | 544 | 1844 | +0.063 | +0.0099 | OK | OK | FALLA | 3 | 2 |
| 40 | 30 | 0.25 | 1.20 | 671 | 1717 | +0.058 | +0.0094 | OK | OK | FALLA | 3 | 2 |
| 60 | 5  | 0.50 | 1.20 | 166 | 2240 | +0.085 | +0.0091 | OK | OK | FALLA | 3 | 2 |
| 60 | 5  | 0.25 | 1.10 | 259 | 2147 | +0.074 | +0.0088 | OK | OK | FALLA | 3 | 2 |

Las 17 que pasan D2 tienen en comun: **n_bloques_lift_pos = 2** (nunca
3+). Nunca cumplen D3.

### 7.2. Top-10 por lift_point

| N | M | X_ATR | Y_VOL | n_conf | n_base | lift_point | lower_CI | D2 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 60 | 5  | 1.00 | 1.20 | 74 | 2332 | **+0.1065** | -0.0160 | NO |
| 60 | 5  | 1.00 | 1.10 | 77 | 2329 | +0.1023 | -0.0148 | NO |
| 20 | 5  | 1.00 | 1.20 | 103 | 2303 | +0.1005 | +0.0010 | **SI** |
| 20 | 5  | 1.00 | 1.50 | 96 | 2310 | +0.1004 | -0.0031 | NO |
| 40 | 5  | 1.00 | 1.20 | 83 | 2323 | +0.0949 | -0.0160 | NO |
| 30 | 5  | 1.00 | 1.50 | 88 | 2318 | +0.0935 | -0.0144 | NO |
| 60 | 5  | 0.75 | 1.10 | 119 | 2287 | +0.0932 | -0.0008 | NO |
| 20 | 5  | 1.00 | 1.10 | 105 | 2301 | +0.0919 | -0.0057 | NO |
| 60 | 5  | 0.50 | 1.10 | 174 | 2232 | +0.0899 | +0.0133 | **SI** |
| 40 | 5  | 1.00 | 1.10 | 84 | 2322 | +0.0897 | -0.0205 | NO |

Las combinaciones con **M=5** dominan el top-lift_point. Pero la
mayoria no pasa D2 (lower_CI < 0): muestra pequeña de confirmed + lift
grande = IC ancho.

**Contraste:** el top-lower_CI prefiere **M=30** (mas muestra, IC mas
estrecho). El top-lift_point prefiere **M=5** (pico puntual, IC ancho).

Esto es exactamente lo que el dictamen 26 seccion 9 anticipo: `lift > 0`
es demasiado debil; **lower_CI > 0** filtra los picos ruidosos.

### 7.3. Ninguna combinacion con n_bloques_lift_pos = 3 y D2

9 combinaciones tienen los 3 bloques evaluables con lift > 0. De esas,
solo 1 pasa D2 (N=20, M=5, X=1.00, Y=1.20, lower_CI=+0.0010). Y esa
tiene n_confirmed=103 (muestra moderada) y M=5 (ventana muy corta).

---

## 8. Bloques: muestra y estabilidad

Resumen por bloque (sobre las 240 combinaciones):

    Bloque A: n_conf min/med/max = 1/1/12      n_base min/med/max = 98/119/120
    Bloque B: n_conf min/med/max = 17/55/192   n_base min/med/max = 285/445/493
    Bloque C: n_conf min/med/max = 34/133/453  n_base min/med/max = 658/979/1087
    Bloque D: n_conf min/med/max = 18/72/292   n_base min/med/max = 370/579/627

**Bloque A:** nunca alcanza 20 confirmed. Inelegible por diseno.
**Bloque B:** a veces alcanza 20, a veces no. Fragil.
**Bloque C:** siempre evaluable (min 34 conf).
**Bloque D:** a veces evaluable (min 18).

La distribucion temporal de candidate episodes (por fecha de L) hace
que A y D sean los bloques con menos muestra.

---

## 9. Verificacion SLB / PCG (guardrail funcional)

Con el landmark correcto y la candidata top-lower_CI
(N=60, M=30, X=0.25, Y=1.10):

- **SLB:** SOW residuales en 90d con v1.6 = 3 -> con v1.8 (X>=0.25) = 0.
  Sigue funcionando como regression test.
- **PCG:** 4 SOW v1.6 -> 1 con v1.8 (el unico legitimo: bd=4.03, vr=6.82).

**No forman parte del criterio de seleccion.** Solo constan como guardrail.

---

## 10. Analisis prohibidos (protocolo v2 seccion 9)

Comprobaciones explicitas:

- NO se optimizo por n_confirmed ni por numero de DISTRIBUTION.
- NO se selecciono mirando el top de bloques sin criterio ex-ante.
- NO se reutilizo el holdout 2025-03-31/2026-10-01 como validacion ciega.
- NO se relajo D3 retroactivamente.
- NO se modifico el protocolo tras ver resultados.
- NO se anclaron confirmed y baseline en fechas distintas: ambos usan L.
- NO se calculo outcome sin usar el mismo L para ambos grupos.
- NO se permitio SOW en t0 como confirmacion: ventana [t0+1, L].
- NO se trataron episodios como observaciones independientes: bootstrap
  por ticker-cluster.
- NO se introdujo preferencia artificial por X_ATR/Y_VOL en tiebreak.

**Nota:** las 17 combinaciones que pasan D2 no se seleccionan porque
D3 falla estructuralmente. No se propone ranking entre ellas.

---

## 11. Reproducibilidad

    Set-Location D:\Macro_Sectorial
    py scripts\calibrate_wyckoff_sow_5b4.py

Determinista sobre el snapshot actual de data/stock_prices.parquet.
Tiempo de ejecucion en el equipo de referencia:

    Grid:       ~339s
    Bootstrap:  ~144s
    Total:      ~483s (~8 min)

Salidas:

- outputs/audit/wyckoff_5b4_grid.csv (240 filas x 224 columnas)
- outputs/audit/wyckoff_5b4_bootstrap.csv (240 filas x 12 columnas)
- outputs/audit/wyckoff_5b4_summary.json
- outputs/audit/wyckoff_5b4_run.log

---

## 12. Conclusiones

**Hecho empirico 1.** El landmark funciona. El lift de la Candidata A
pasa de +0.192 (5b3) a +0.064 (5b4). El +0.192 no es una estimacion
valida bajo el diseno temporal corregido: el resultado de 5b3 era
muy sensible al anclaje temporal.

**Hecho empirico 2.** El lift no es homogeneo entre bloques. Para la
Candidata A, lift negativo en bloques A (-0.31) y B (-0.18), positivo
en C (+0.11) y D (+0.11). El "lift positivo universal" de 5b3 no se
sostiene con landmark.

**Hecho empirico 3.** D3 es estructuralmente inalcanzable: el bloque A
nunca alcanza 20 confirmed. Ninguna combinacion puede tener 4 bloques
evaluables. El criterio no mide la calidad del SOW, mide la
disponibilidad de muestra por bloque.

**Hecho empirico 4.** Con M=5, el lift_point es mas alto pero el IC
suele cruzar 0. Con M=30, el lift es mas bajo pero el IC es estrecho.
lower_CI > 0 filtra picos ruidosos, como anticipo el dictamen 26
seccion 9.

**Hecho empirico 5.** 17/240 combinaciones pasan D2 (lower_CI > 0).
Todas ellas tienen exactamente 2 de 3 bloques con lift > 0. Ninguna
tiene 3.

**Bloqueo.** 5b.4 no selecciona parametros. Se abre expediente 28 con
las preguntas del auditor. Los parametros SOW siguen en None.

---

## 13. Proximo paso

Expediente `28_expediente_5b4_heterogeneidad.md` remitido al auditor.
Contiene:

- Pregunta sobre D3 (criterio estructuralmente inalcanzable).
- Pregunta sobre la heterogeneidad temporal (lift de signo opuesto
  entre bloques A-B y C-D).
- Pregunta sobre la interpretacion del lift global con landmark.
- Datos completos.

Hasta dictamen, el sistema queda:

    v1.8              CONTRATO VIGENTE
    5b.4 (protocolo)  v2 congelado en 9fb6bcf
    5b.4 (ejecucion)  COMPLETADA, bloqueada en seleccion
    parametros SOW    None en config
    5b.X              PENDIENTE (validacion futura)
    5c.4, 5d          BLOQUEADAS
    Legacy            INTACTO

---

**Fin del informe 27.**