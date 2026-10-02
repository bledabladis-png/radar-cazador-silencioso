# EXPEDIENTE 5b.4 - D3 inalcanzable y heterogeneidad temporal

**Documento de consulta. NO normativo. Requiere dictamen externo.**
**Fecha:** 2026-10-02.
**Precedente:** 27_calibracion_5b4_resultados.md + 25_protocolo_fase_5b4.md
(v2) + 26_dictamen_5b4.md + commit 9fb6bcf.
**Estado:** BLOQUEO de la seleccion hasta dictamen.

---

## 0. Proposito

La calibracion 5b.4 se ejecuto completa con landmark temporal
corregido. Ninguna combinacion pasa D1-D3. Los resultados estan en
27_calibracion_5b4_resultados.md.

Resumen:

    D1 (n_confirmed >= 20 AND n_baseline >= 20):  240/240
    D2 (lower_CI(lift_H20) > 0):                   17/240
    D3 (4 bloques evaluables + >= 3 con lift > 0):   0/240

El bloqueo tiene dos causas distintas:

1. **Causa A - D3 es estructuralmente inalcanzable.** El bloque A
   (2021-10 -> 2022-12) nunca alcanza 20 confirmed. Ninguna
   combinacion puede tener 4 bloques evaluables.

2. **Causa B - El lift no es homogeneo entre bloques.** El signo del
   lift depende del bloque temporal. Negativo en A-B, positivo en C-D.

Se solicita dictamen sobre las preguntas del apartado 3.

---

## 1. Evidencia empirica

### 1.1. D3 inalcanzable por diseno

Distribucion del numero de bloques evaluables (>=20 confirmed AND
>=20 baseline) sobre las 240 combinaciones:

    1 bloque evaluable:   3 combos
    2 bloques evaluables: 3 combos
    3 bloques evaluables: 234 combos
    4 bloques evaluables: 0 combos

**Bloque A maximo: 12 confirmed (mediana: 1, minimo: 1).** Nunca
alcanza el umbral. La causa es que el bloque A (15 meses) tiene pocos
candidate episodes que cumplan las condiciones del contrato v1.8.

**El D3 actual no mide la calidad del SOW.** Mide la disponibilidad de
muestra por bloque. En el bloque A, esa muestra no existe, y el
criterio falla estructuralmente.

### 1.2. Comparativa 5b3 vs 5b4 (confirma el sesgo del landmark)

Para la Candidata A (N=60, M=30, X=0.50, Y=1.20):

    | Experimento | lift_point_H20 | n_conf | n_base |
    | 5b3 (sesgado) |    +0.1920    |  406   |  1212  |
    | 5b4 (landmark) |    +0.0637    |  475   |  1913  |

El lift se reduce a ~1/3. El +0.192 era artefacto del anclaje
asimetrico, como anticipo el dictamen 24.

### 1.3. Heterogeneidad temporal del lift

Para la Candidata A, H20 struct_deterioration:

    Bloque A (2021-10 -> 2022-12): n_conf=2   n_base=108  lift=-0.31
    Bloque B (2023-01 -> 2024-03): n_conf=90  n_base=387  lift=-0.18
    Bloque C (2024-04 -> 2025-08): n_conf=240 n_base=871  lift=+0.11
    Bloque D (2025-09 -> 2026-10): n_conf=138 n_base=524  lift=+0.11
    ALL                          : n_conf=470 n_base=1890 lift=+0.0637

**El signo del lift depende del bloque temporal.** Negativo en A-B,
positivo en C-D. Patron reproducible en el top-5 por lift_point.

**Nota:** los bloques A y B corresponden a 2021-10 -> 2024-03. Bloques
C y D a 2024-04 -> 2026-10. La frontera temporal coincide con un
cambio de regimen de mercado (post-recuperacion 2024).

### 1.4. M=5 vs M=30: dos picos diferentes

El top-lift_point prefiere M=5. El top-lower_CI prefiere M=30.

    Top-lift_point: lift ~ 0.10 pero lower_CI suele cruzar 0
    Top-lower_CI:   lift ~ 0.06-0.07, lower_CI > 0

M=5 captura mas SOW cerca del candidate pero la muestra es pequeña
y el IC se ensancha. M=30 requiere SOW mas tardio, con menos pico
pero IC mas estrecho.

### 1.5. D2 (lower_CI > 0): 17/240

Las 17 combinaciones que pasan D2 tienen en comun:
- n_bloques_lift_pos = 2 (nunca 3).
- Pasan D1 tambien.
- Fallan D3 por el punto 1.1.

Ninguna combinacion del top-lift_point pasa D2. El IC filtra los
picos ruidosos, como anticipo el dictamen 26 seccion 9.

---

## 2. Interpretacion tentativa (NO es conclusion)

La evidencia sugiere, sin confirmar, que:

**Hipotesis alternativa A - El SOW no es confirmador universal.**

El SOW podria discriminar deterioro futuro solo en algunos regimenes
de mercado (post-2024, regimen de recuperacion/tendencia). En regimen
2021-2023 (bajista y rebote), el SOW no añade informacion sobre
deterioro futuro o incluso la invierte.

Bajo esta hipotesis:
- El "confirmador" no es el SOW, es el **regimen**.
- El lift agregado ALL es un promedio de dos regimenes opuestos.
- 5b.4 deberia reportar el resultado por bloque, no agregado.

**Hipotesis alternativa B - El SOW es confirmador debil.**

El SOW añade informacion marginal (+0.06 a +0.10 en lift_point) que
puede ser real pero no alcanza significancia estadistica robusta
cuando se clusteriza por ticker.

Bajo esta hipotesis:
- El hallazgo es "existe senal pero es fragil".
- El criterio lower_CI > 0 es el adecuado (17/240).
- El siguiente paso es validar esas 17 en 5b.X.

**Hipotesis alternativa C - El diseno de bloques es inadecuado.**

El bloque A tiene muy poca muestra porque 2021-2022 tuvo menos
episodios candidate elegibles. Los bloques no son comparables en
tamaño. La heterogeneidad puede ser artefacto de la muestra
desbalanceada, no del regimen.

Bajo esta hipotesis:
- El rediseno de bloques es la prioridad.
- Antes de concluir sobre regimen, hay que balancear la muestra.

**El equipo no puede decidir entre A, B y C con la evidencia actual.**
Requiere dictamen.

---

## 3. Preguntas al auditor externo

### P1. D3 inalcanzable por diseno

D3 exige 4 bloques evaluables, pero el bloque A nunca alcanza el
umbral. La condicion es estructuralmente inalcanzable.

**Pregunta:** como proceder?

    a) Reformular D3 para que los bloques con muestra insuficiente
       se excluyan del criterio (p.ej. "de los bloques evaluables,
       al menos la mitad con lift > 0").
    b) Rediseñar los bloques temporales para garantizar muestra
       suficiente en cada uno (fronteras diferentes, bloques mas
       largos, o solo 3 bloques).
    c) Eliminar D3 y sustituirlo por otro criterio que no dependa
       de bloques con muestra fija.
    d) Aceptar que 5b.4 no puede cerrar con este grid y abrir 5b.5
       con rediseno.

### P2. Heterogeneidad temporal del lift

El lift cambia de signo entre bloques. El "lift positivo universal"
de 5b3 no se sostiene con landmark. Tres hipotesis (A, B, C) en
seccion 2.

**Pregunta:** cual de las tres hipotesis priorizar?

    a) Hipotesis A: el SOW solo discrimina en algunos regimenes.
       Reportar por bloque. No agregar.
    b) Hipotesis B: el SOW es confirmador debil. Validar las 17 con
       lower_CI > 0 en 5b.X.
    c) Hipotesis C: los bloques no son comparables (muestra
       desbalanceada). Rediseñar bloques.
    d) Otra.

### P3. Criterio de agregacion con landmark

El lift ALL agregado mezcla bloques con signos opuestos. La media
de +0.0637 no representa ni a A-B ni a C-D.

**Pregunta:** se debe seguir reportando ALL, o solo por bloque?

    a) Solo por bloque. ALL queda como referencia, no como criterio.
    b) ALL pero con test de heterogeneidad (chi-cuadrado o
       equivalente) que detecte si los bloques son homogeneos.
    c) Pesado por muestra: bloques grandes pesan mas en ALL.
    d) Otra.

### P4. M=5 vs M=30

M=5 maximiza lift_point pero suele fallar D2. M=30 tiene lift mas
modesto pero IC estrecho.

**Pregunta:** el diseno prefiere M pequeño (mas pico) o M grande
(mas robusto)? El protocolo v2 no declara preferencia ex-ante.

    a) Preferir M grande (robustez).
    b) Preferir M pequeño (pico mas alto).
    c) Neutral: el ranking decide.
    d) Otra.

### P5. 17 candidatas que pasan D2

17 combinaciones pasan D2. Todas con 2 de 3 bloques evaluables con
lift > 0. Ninguna tiene 3.

**Pregunta:** las 17 son suficientes para congelar, o requieren
mas evidencia antes de 5b.X?

    a) Congelar una (la mejor) y pasar a 5b.X.
    b) 5b.X con las 17 en lugar de una sola (test multiple).
    c) Esperar a resolver P1 y P2 antes de decidir.
    d) Otra.

### P6. Sesgo de seleccion

Las 17 que pasan D2 son el resultado de 240 pruebas. Con 240
combinaciones, algunas pasaran por azar.

**Pregunta:** se aplica correccion por multiple testing?

    a) Si, Bonferroni (alpha/240).
    b) Si, FDR (Benjamini-Hochberg).
    c) No, se acepta el multiple testing y se valida en 5b.X.
    d) Otra.

### P7. Rediseno del bloque A

El bloque A es el mas problematico. 15 meses, pocos episodes.

**Pregunta:** eliminar el bloque A? Fusionar A+B?

    a) Eliminar A: analisis con B+C+D.
    b) Fusionar A+B en un bloque unico.
    c) Mantener A: es informativo aunque no sea evaluable.
    d) Otra.

---

## 4. Anexo A - Distribucion por bloque

Sobre las 240 combinaciones:

    Bloque A: n_conf min/med/max = 1/1/12      n_base min/med/max = 98/119/120
    Bloque B: n_conf min/med/max = 17/55/192   n_base min/med/max = 285/445/493
    Bloque C: n_conf min/med/max = 34/133/453  n_base min/med/max = 658/979/1087
    Bloque D: n_conf min/med/max = 18/72/292   n_base min/med/max = 370/579/627

Bloque A: nunca evaluable (necesita 20 conf, max 12).
Bloque B: a veces evaluable (17-192).
Bloque C: siempre evaluable (min 34).
Bloque D: a veces evaluable (min 18).

---

## 5. Anexo B - Top-10 por lower_CI (las 17 que pasan D2)

| N | M | X_ATR | Y_VOL | n_conf | n_base | lift_point | lower_CI | n_bloques_pos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 60 | 30 | 0.25 | 1.10 | 635 | 1753 | +0.0740 | +0.0234 | 2 |
| 40 | 30 | 0.25 | 1.10 | 703 | 1685 | +0.0690 | +0.0205 | 2 |
| 60 | 30 | 0.50 | 1.10 | 496 | 1892 | +0.0712 | +0.0185 | 2 |
| 60 |  5 | 0.50 | 1.10 | 174 | 2232 | +0.0899 | +0.0133 | 2 |
| 60 | 30 | 0.25 | 1.20 | 601 | 1787 | +0.0647 | +0.0131 | 2 |
| 60 | 30 | 0.50 | 1.20 | 475 | 1913 | +0.0637 | +0.0106 | 2 |
| 40 | 30 | 0.50 | 1.10 | 544 | 1844 | +0.0630 | +0.0099 | 2 |
| 40 | 30 | 0.25 | 1.20 | 671 | 1717 | +0.0577 | +0.0094 | 2 |
| 60 |  5 | 0.50 | 1.20 | 166 | 2240 | +0.0853 | +0.0091 | 2 |
| 60 |  5 | 0.25 | 1.10 | 259 | 2147 | +0.0735 | +0.0088 | 2 |

Patron: el top-lower_CI prefiere M=30 (excepto dos filas con M=5).
Todos tienen n_bloques_pos=2.

---

## 6. Anexo C - Top-10 por lift_point

| N | M | X_ATR | Y_VOL | n_conf | n_base | lift_point | lower_CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| 60 | 5 | 1.00 | 1.20 |  74 | 2332 | +0.1065 | -0.0160 |
| 60 | 5 | 1.00 | 1.10 |  77 | 2329 | +0.1023 | -0.0148 |
| 20 | 5 | 1.00 | 1.20 | 103 | 2303 | +0.1005 | +0.0010 |
| 20 | 5 | 1.00 | 1.50 |  96 | 2310 | +0.1004 | -0.0031 |
| 40 | 5 | 1.00 | 1.20 |  83 | 2323 | +0.0949 | -0.0160 |
| 30 | 5 | 1.00 | 1.50 |  88 | 2318 | +0.0935 | -0.0144 |
| 60 | 5 | 0.75 | 1.10 | 119 | 2287 | +0.0932 | -0.0008 |
| 20 | 5 | 1.00 | 1.10 | 105 | 2301 | +0.0919 | -0.0057 |
| 60 | 5 | 0.50 | 1.10 | 174 | 2232 | +0.0899 | +0.0133 |
| 40 | 5 | 1.00 | 1.10 |  84 | 2322 | +0.0897 | -0.0205 |

Patron: M=5 en todos. La mayoria no pasa D2 (IC cruza 0).

---

## 7. Anexo D - Candidata A en 5b4

N=60, M=30, X=0.50, Y=1.20:

    n_conf=475 n_base=1913
    lift_point=+0.0637  lower_CI=+0.0106  upper_CI=+0.1157
    D1=True  D2=True  D3=False

    Bloque A: n_conf=2   n_base=108  lift=-0.306
    Bloque B: n_conf=90  n_base=387  lift=-0.184
    Bloque C: n_conf=240 n_base=871  lift=+0.114
    Bloque D: n_conf=138 n_base=524  lift=+0.113
    ALL     : n_conf=470 n_base=1890 lift=+0.0637

D2 OK en el agregado. D3 falla por el bloque A.

---

## 8. Anexo E - Artefactos

    outputs/audit/wyckoff_5b4_grid.csv         240 filas x 224 columnas
    outputs/audit/wyckoff_5b4_bootstrap.csv    240 filas x 12 columnas
    outputs/audit/wyckoff_5b4_summary.json     resumen JSON
    outputs/audit/wyckoff_5b4_run.log          log de ejecucion

---

## 9. Que NO se ha tocado

- Cero cambios en config/settings.py. Parametros SOW siguen a None.
- Cero cambios en contrato v1.8 ni en protocolo 5b.4 v2.
- Ninguna combinacion congelada.
- Legacy intacto. Consumidores sin migrar.
- 5c.4 y 5d bloqueadas.

---

## 10. Propuesta de siguiente paso

1. Recibir dictamen sobre P1-P7.
2. Si se reformula D3 o se rediseñan bloques: actualizar protocolo
   a v3 y re-ejecutar grid (5b.4-bis) sin modificar el script (solo
   parametros del grid o del criterio).
3. Si se acepta la interpretacion por bloque y se congela una de las
   17: pasar a 5b.X con la combinacion congelada.
4. Si se rediseña la hipotesis: abrir 5b.5.

Hasta dictamen, 5b.4 queda formalmente **FAIL en seleccion** pero
**ejecutada correctamente**: el diseno del landmark funciona, el
sesgo se ha eliminado, y la evidencia nueva (heterogeneidad temporal)
es informativa.

---

**Fin del expediente 28.**