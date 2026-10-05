# QA 5b.4-bis - IC por bloque y concentracion por ticker

**Informe de QA. NO normativo. Pendiente de dictamen.**
**Fecha:** 2026-10-02.
**Precedente:** dictamen 33 seccion 19 (QA obligatorio antes del
freeze documental definitivo).
**Script:** scripts/qa_wyckoff_5b4bis.py (read-only).
**Artefactos:** outputs/audit/wyckoff_5b4bis_qa_blocks.csv,
outputs/audit/wyckoff_5b4bis_qa_tickers.csv,
outputs/audit/wyckoff_5b4bis_qa.json,
outputs/audit/wyckoff_5b4bis_qa_run.log.

---

## 0. Proposito

El dictamen 33 exigio antes del freeze documental definitivo:

- IC por bloque (P1/P2/P3 lower/upper).
- Distribucion de episodios por ticker.

Este informe cubre ambas.

---

## 1. Candidata

    N=60, M=30, X_ATR=0.25, Y_VOL=1.10

    n_episodes=2388 (conf=635, base=1753)
    n_tickers=300
    bootstrap B=2000, seed=20261002

---

## 2. IC por bloque

| Bloque | Periodo | lift | IC95% | n_conf | n_base | n_tickers |
|---|---|---:|---|---:|---:|---:|
| P1 | 2021-10 -> 2023-06 | +0.093 | **[-0.099, +0.296]** | 32 | 226 | 132 |
| P2 | 2023-07 -> 2025-03 | +0.050 | **[-0.024, +0.122]** | 304 | 914 | 278 |
| P3 | 2025-04 -> 2026-10 | +0.106 | [+0.038, +0.178] | 293 | 591 | 255 |
| ALL | - | +0.074 | [+0.023, +0.124] | 629 | 1731 | 300 |

### Lectura

**P1:** n_conf=32, IC muy ancho. Cruza 0. **El signo positivo no es
estadisticamente distinguible de 0 con la muestra disponible.**

**P2:** n_conf=304, IC mas estrecho pero **cruza 0** (lower=-0.024).
El lift podria ser nulo o negativo con esta muestra.

**P3:** n_conf=293, IC [+0.038, +0.178]. **No cruza 0.** Es el unico
bloque con evidencia de lift positivo bajo el IC bilateral 95%.

**ALL:** IC [+0.023, +0.124]. No cruza 0, pero **el intervalo se
sostiene por el peso de P3**. Con P3 en +0.106 (el mas alto) y P1/P2
cruzando 0 individualmente, la significancia del agregado depende
de un bloque.

### La firma "+++" NO es homogeneidad estadistica

El informe 31 documento "+++" como consistencia de signo. El QA
confirma que era exactamente eso: **signos positivos en los tres
bloques puntuales**, pero solo uno de los tres (P3) tiene evidencia
de lift positivo con IC no cruzando 0.

**El dictamen 33 seccion 5 lo predijo exactamente:**

> "El P1 puede tener un intervalo muy amplio por n_confirmed=32.
> Por ejemplo, conceptualmente podria ocurrir: P1: +9.3 pp IC
> [-20,+39], P2: +5.0 pp IC [+1,+9], P3: +10.6 pp IC [+6,+15] y
> decir 'los tres son estables' seria incorrecto."

La realidad observada es cercana a esa advertencia: P1 con IC ancho,
P2 con IC cruzando 0, P3 con IC positivo. **Solo P3 es
estadisticamente informativo.**

---

## 3. Concentracion por ticker

    n_tickers=300
    n_episodes=2360
    max_episodios_por_ticker=23
    mediana_episodios_por_ticker=8.0
    top5_share=0.044  (4.4%)
    top10_share=0.080 (8.0%)

### Lectura

Distribucion razonablemente uniforme. Los 5 tickers con mas episodios
concentran solo 4.4% del total; los 10 primeros, 8.0%. **No hay
concentracion anomala.**

El maximo de 23 episodios por ticker es coherente con tickers de
alta liquidez con muchos cruces de candidate. No invalida el analisis.

**Conclusion:** el lift agregado no esta dominado por unos pocos
tickers. La dependencia intraticker se gestiona con el bootstrap
cluster; la concentracion por cluster no es un problema adicional.

---

## 4. Implicaciones para el freeze de validacion

El dictamen 33 seccion 15 autorizo "freeze de validacion" (congelar la
candidata como especificacion normativa para 5b.X) y **no** "freeze
productivo" (activar en config).

**Ese freeze de validacion sigue siendo correcto.** Precisamente
porque los IC por bloque muestran que la evidencia es fragil, la
respuesta adecuada es validar en datos futuros, no activar.

### Lo que el QA cambia

- **No invalida** el "PASS desarrollo" del dictamen 33.
- **Debilita** cualquier lenguaje que presente "+++" como prueba de
  homogeneidad. El informe 31 ya fue corregido para esto.
- **Refuerza** la decision del dictamen 33 de no congelar en
  produccion.

### Lo que el QA NO cambia

- La candidata #1 sigue siendo la ganadora determinista del ranking
  v3, con las 9 competidoras cercanas.
- El cutoff 2026-10-01 para validacion futura sigue vigente.
- La condicion: "una sola configuracion, sin recalibrar en 5b.X"
  sigue vigente.

---

## 5. Consecuencias para 5b.X

Si el +0.074 esta arrastrado por P3 (2025-04 -> 2026-10), la validacion
en datos futuros respondera la pregunta mas util:

> El lift positivo es una propiedad estable del SOW, o es especifico
> de un regimen de mercado concreto?

**Hipotesis a testar en 5b.X (no resuelta ahora):**

- Si el lift en datos nuevos es comparable a P3 (+0.106), la hipotesis
  "SOW confirma en regimenes tendenciales" gana fuerza.
- Si el lift es cercano a 0 (como P2), la hipotesis "SOW aporta poco"
  gana.
- Si el lift es negativo (como P1 sugirio aunque sin significancia),
  la hipotesis "SOW empeora la discriminacion en algunos regimenes"
  gana.

**Ninguna de las tres se decide ahora.** El protocolo 5b.X debe fijar
ex-ante que estas tres salidas son posibles y como se reportan.

---

## 6. Integridad de artefactos

Verificado en esta ejecucion:

    grid CSV (240 filas)
    bootstrap CSV (240 filas)
    hetero CSV (240 filas)
    summary JSON
    run log
    qa blocks CSV
    qa tickers CSV
    qa JSON
    qa run log

Todos generados bajo el commit 848ee23. Reproducibles con:

    py scripts/calibrate_wyckoff_sow_5b4bis.py
    py scripts/qa_wyckoff_5b4bis.py

Determinista sobre el snapshot actual de data/stock_prices.parquet.

---

## 7. Conclusion

**Hecho empirico 1.** De los 3 bloques, solo P3 (2025-04 -> 2026-10)
tiene IC95% que no cruza 0. P1 (IC ancho por n_conf=32) y P2 (IC que
cruza 0 con n_conf=304) no tienen evidencia de lift positivo a nivel
individual.

**Hecho empirico 2.** El lift ALL [+0.023, +0.124] esta sostenido
principalmente por P3. Sin P3, la significancia del agregado seria
dudosa.

**Hecho empirico 3.** La concentracion por ticker es razonable
(top5=4.4%, top10=8.0%). El lift no esta dominado por unos pocos
tickers.

**Hecho empirico 4.** La firma "+++" es consistencia de signo
puntual, no homogeneidad estadistica. El dictamen 33 seccion 5 lo
anticipo.

### Bloqueo

- NO freeze productivo.
- SI freeze documental (v1.9 FROZEN_FOR_VALIDATION).
- NO activar config.
- Mantener fail-closed.
- Reservar datos posteriores a 2026-10-01 para 5b.X.

### Recomendacion

Redactar `35_protocolo_5bX.md` con el diseno de la validacion futura:
una sola configuracion, sin recalibrar, con las tres hipotesis de
salida declaradas ex-ante.

---

## 8. Estado

    v1.8                CONTRATO PRODUCTIVO VIGENTE
    5b.3                FAIL
    5b.4                FAIL seleccion / diagnostico valido
    5b.4-bis            PASS desarrollo / NO confirmado
    QA 5b.4-bis         COMPLETADO (este informe)
    v1.9                PENDIENTE (FROZEN_FOR_VALIDATION)
    5b.X                PENDIENTE (requiere datos posteriores a
                        2026-10-01)
    5c.4, 5d            BLOQUEADAS
    Legacy              INTACTO
    Consumidores        SIN MIGRAR
    Config SOW          None (fail-closed)

---

**Fin del informe 34.**
