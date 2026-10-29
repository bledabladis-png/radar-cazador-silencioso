# RESULTADO DICT3-REACCUMULATION (v4 ejecutada sobre cohort)

**Estado:** resultado experimental. NO confirmatorio. La evidencia
sobre cohort fue consumida previamente por SOW-V6 (ver P0-H01 §1 de
v4) y por tanto es EXPLORATORIA.

**Fecha:** 2026-10-29.
**Pre-registro:** `59_pre_registro_reaccumulation.md` v4 (commit 375f06bf).
**Detector:** `scripts/audit/reaccumulation_detector.py` (commit 0539813d).
**Tests contractuales:** `tests/test_reaccumulation_detector.py` (18 tests,
T1-T7).
**Datos:** `data/stock_prices_reserved_cohort.parquet` (268 tickers,
2015-01-02 .. 2026-10-23).
**Salida:** `outputs/audit/reaccumulation_events_cohort.csv`
(gitignored, 1185 episodios).

**Regla de congelacion:** los parametros de v4 (§2.8) NO se han
recalibrado tras ver los resultados. El fallo se documenta tal cual.

---

## 1. Ejecucion

- Tickers totales: 268.
- Tickers evaluados (>=500 sesiones): 268.
- Episodios detectados: 1185.
- Tickers con >=1 episodio: 255 (95.1%).
- Rango temporal de episodios: 2016 .. 2026.

Distribucion por año (date_start):

| Año | Episodios | Tickers con episodio |
|---|---:|---:|
| 2016 | 40 | 31 |
| 2017 | 290 | 138 |
| 2018 | 125 | 78 |
| 2019 | 106 | 73 |
| 2020 | 95 | 71 |
| 2021 | 163 | 98 |
| 2022 | 14 | 13 |
| 2023 | 79 | 56 |
| 2024 | 147 | 97 |
| 2025 | 52 | 38 |
| 2026 | 74 | 55 |

---

## 2. Criterio C1 (frecuencia): FALLA

Definicion v4 §5.1: ratio = (tickers con >=1 episodio) / (tickers
evaluables), por chunk de 252 sesiones. Exige que **todos los chunks
esten en [1%, 30%]**.

Proxy por año natural (200+ sesiones por ticker evaluado):

| Año | Evaluables | Con episodio | Ratio | C1 |
|---|---:|---:|---:|---|
| 2016 | 263 | 31 | 0.1179 | OK |
| 2017 | 266 | 138 | 0.5188 | **ALTO (>30%)** |
| 2018 | 267 | 78 | 0.2921 | OK |
| 2019 | 268 | 73 | 0.2724 | OK |
| 2020 | 268 | 71 | 0.2649 | OK |
| 2021 | 268 | 98 | 0.3657 | **ALTO (>30%)** |
| 2022 | 268 | 13 | 0.0485 | OK |
| 2023 | 268 | 56 | 0.2090 | OK |
| 2024 | 268 | 97 | 0.3619 | **ALTO (>30%)** |
| 2025 | 268 | 38 | 0.1418 | OK |
| 2026 | 268 | 55 | 0.2052 | OK |

Años con C1 OK: 8/11. **C1 = FALLA** (3 años por encima del 30%).

Interpretacion empirica: los años de C1 alto (2017, 2021, 2024) son
años alcistas fuertes (S&P 500 con retornos > 20%). El detector con
los umbrales congelados captura "reacumulacion generalizada", que
en esos años coincide con el mercado entero. No distingue
reacumulacion de bull market.

---

## 3. Criterio C2 (tasa de acierto): FALLA

Definicion v4 §5.2:
- Exito: retorno Close-to-Close a 63 sesiones >= +5%.
- Exige: p_hat >= 0.60, IC95_lower > 0.50, N >= 63.

Resultado (sin IC, solo punto):

- Episodios evaluables (63 sesiones futuras disponibles): 1171.
- Tasa de exito (>=5% en 63 sesiones): **0.3843 (38.43%)**.
- Retorno medio: **+1.32%**.
- Retorno mediana: **+2.31%**.

**C2 = FALLA**. 38.43% esta muy por debajo del 60% exigido. El
retorno medio y la mediana son positivos pero pequeños; el detector
captura episodios con drift ligeramente positivo pero no con
probabilidad de exito > 60%.

Como C2 falla el umbral puntual (0.3843 < 0.60), el IC95_lower
seria previsiblemente < 0.50 tambien. No se calcula bootstrap porque
la condicion de exito ya no se cumple en la tasa puntual.

---

## 4. Criterio C3 (solapamiento): no evaluado

No se calcula porque C1 y C2 ya fallan. Si C1 o C2 fallan, el
veredicto global es NO_FUNCIONA y no procede evaluar C3.

---

## 5. Veredicto global

**NO_FUNCIONA.** Los dos criterios evaluables (C1, C2) fallan con
los parametros congelados en v4. El detector experimental no produce
una validacion que satisfaga el protocolo pre-registrado sobre el
holdout disponible.

No se recalibran los parametros. Cambiar umbrales tras ver
resultados invalidaria la prueba. Si se quiere explorar umbrales
mas estrictos (trend_up > 0.10, pullback < -0.5, breakout > +1.0),
debe hacerse en una v5 con hipotesis distinta y muestra
confirmatoria nueva (periodo futuro, §6 de v4).

---

## 6. Lo que esto significa para el sistema

- El detector funciona tecnicamente (1185 episodios detectados, 0
  errores de ejecucion).
- El diseño conceptual (reacumulacion como familia separada) no se
  refuta por este resultado: se refuta la parametrizacion concreta.
- La evidencia NO es confirmatoria: cohort fue consumido por SOW-V6
  (P0-H01 de v4). Incluso si C1 y C2 hubieran pasado, seria
  evidencia exploratoria.
- El sistema productivo NO se ve afectado. El detector es aislado,
  no toca `classify_wyckoff_phase`, ni el orquestador, ni ningun CSV
  productivo.

---

## 7. Opciones de seguimiento (decision del auditor)

1. **Cerrar DICT3-REACCUMULATION como WONT_FIX** con la razon
   "el detector no supera C1/C2 con la parametrizacion congelada".
   El sistema no gana nada con reacumulacion que no tenga ya con el
   detector de reversion.
2. **Abrir v5** con parametros mas estrictos y protocolo congelado,
   ejecutando sobre la misma cohort (resultado exploratorio) y
   abriendo la ventana futura de §6 de v4 para confirmacion. Coste:
   otra iteracion de contrato + implementacion + 504 sesiones de
   espera.
3. **Congelar como esta**: documentar el hallazgo, no cerrar el
   frente, no abrir v5. Bajo ROI.

Los tres son legitimos. El resultado empirico esta documentado.

---

## 8. Reproducibilidad

    py scripts/audit/reaccumulation_detector.py  # T1-T7
    py -m pytest tests/test_reaccumulation_detector.py -q  # 18 passed
    py <script ejecucion cohort>  # 1185 episodios

El CSV de eventos se regenera deterministicamente. Los umbrales
estan congelados en `scripts/audit/reaccumulation_detector.py`
constantes `W`, `W_V`, `L_PREV`, `H_R1`, `TREND_UP_THRESHOLD`,
`CONSOLIDATION_MAX_WIDTH`, `PULLBACK_V_NORM`, `BREAKOUT_V_NORM`.

Fin del informe de resultado.