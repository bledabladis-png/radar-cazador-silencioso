# 41 - Walk-forward historico de SOW: evidencia exploratoria de dependencia de regimen

- **Fecha:** 2026-10-04.
- **Fichero auditado:** `indicators/wyckoff_v1.py` v1.8 (candidata SOW v1.9 congelada).
- **Candidata evaluada:** N=60, M=30, X_ATR=0.25, Y_VOL=1.10.
- **Contrato de auditoria:** plan de migracion v3, seccion 15.
- **Dictamen:** **FAIL** en el test confirmatorio predefinido. Hallazgo exploratorio de heterogeneidad temporal marcado como HIPOTESIS, no como conclusion.

## 1. Motivacion

El plan v3 introdujo el walk-forward historico como metodo de validacion
alternativo al 5b.X oficial (bloqueado hasta ~2027 por falta de datos
post-cutoff). Objetivo: decidir si la candidata SOW se activa o se
descarta sin esperar 12 meses.

Se aplico primero con el parquet productivo (2021-09-27 -> 2026-10-01,
5 anos). Ante la ausencia de OOS historico limpio en ese rango, se
descargo historico adicional via Yahoo hasta 2015-01-02
(`data/stock_prices_extended.parquet`, 10.7 anos, 315 tickers) para
ampliar el analisis. La descarga y el dataset extendido NO entran en
produccion.

## 2. Test confirmatorio (predefinido, plan v3 §15)

**Split congelado antes de ejecutar:**

    Periodo A: 2021-10-01 -> 2023-12-31
    Periodo B: 2024-01-01 -> 2026-10-01
    Criterio PASS: lower_CI95(lift_H20) > 0 en A y en B.

**Script:** `scripts/walk_forward_sow.py`.

**Bootstrap:** B=2000, seed=20261002, cluster por ticker.

**Resultado:**

| Periodo | n_conf | n_base | lift_H20 | IC95% | Veredicto |
|---|---|---|---|---|---|
| A (2021-2023) | 115 | 396 | -0.037 | [-0.147, +0.077] | cruza 0 |
| B (2024-2026) | 514 | 1335 | +0.095 | [+0.043, +0.148] | positivo |
| TOTAL | 629 | 1731 | +0.074 | [+0.023, +0.124] | mixto |

**VEREDICTO: FAIL.** La candidata no pasa el criterio predefinido.
El periodo A carece de evidencia de efecto positivo.

**Nota metodologica:** el split original del plan v3 se diseno cuando
solo existian 5 anos de datos. La ampliacion del parquet a 10 anos no
se uso para redefinir el split. El criterio se mantuvo exactamente
como estaba congelado antes de ver resultados.
## 3. Analisis exploratorio (post-hoc, no confirmatorio)

**Declarado EXPLORATORIO.** No forma parte del criterio PASS/FAIL.
Su funcion es formular hipotesis, no decidir.

**Script:** `scripts/walk_forward_sow_anual.py`.
**Dataset:** `data/stock_prices_extended.parquet` (10.7 anos).
**Bootstrap:** B=500, seed=20261002, cluster por ticker.

**Resultado por ano calendario:**

| Ano | n_conf | n_base | lift | IC95% | Significativo |
|---|---|---|---|---|---|
| 2015 | 0 | 0 | -- | -- | sin muestra |
| 2016 | 139 | 227 | -0.095 | [-0.190, +0.008] | no |
| 2017 | 100 | 377 | +0.114 | [-0.008, +0.245] | no |
| 2018 | 227 | 453 | +0.054 | [-0.014, +0.129] | no |
| 2019 | 134 | 435 | -0.062 | [-0.167, +0.032] | no |
| **2020** | **279** | **102** | **+0.298** | **[+0.209, +0.405]** | **si** |
| 2021 | 73 | 229 | -0.134 | [-0.244, +0.003] | no |
| 2022 | 192 | 409 | -0.041 | [-0.137, +0.062] | no |
| 2023 | 108 | 286 | -0.091 | [-0.220, +0.026] | no |
| 2024 | 146 | 581 | +0.018 | [-0.078, +0.121] | no |
| **2025** | **265** | **416** | **+0.134** | **[+0.050, +0.212]** | **si** |
| **2026** | **108** | **357** | **+0.146** | **[+0.050, +0.248]** | **si** |

**Lectura descriptiva:**

- 3 anos con IC95% que no cruza 0: 2020, 2025, 2026.
- 0 anos con IC95% totalmente negativo.
- 9 anos sin evidencia (IC cruza 0).

## 4. Heterogeneidad temporal: HECHO / OBSERVACION / HIPOTESIS

**HECHO (demostrado por los datos):**

- El efecto SOW no es homogeneo temporalmente.
- 2020 presenta un lift positivo grande (+0.298) con IC95% estrecho.
- 2025 y 2026 presentan lift positivo moderado (+0.13, +0.15).
- No hay ningun ano con evidencia estadistica de efecto negativo.

**OBSERVACION (lectura de los datos, sin inferencia):**

- Los anos con efecto significativo coinciden con periodos de estres
  sistemico (COVID, correcciones fin de ciclo con tipos altos).
- Los anos sin efecto incluyen tanto bull markets pacificos como
  correcciones ordenadas.

**HIPOTESIS (plausible, NO confirmada):**

- La eficacia del SOW puede ser condicional al regimen.
- Es posible que el SOW responda a un subconjunto concreto de
  episodios de deterioro sistemico (velocidad, amplitud, correlacion
  transversal, transicion de regimen), y no al "estres" generico.

**NO DEMOSTRADO:**

- Que exista un filtro operacional (VIX, drawdown, u otro) que mejore
  la validez del SOW. Introducirlo ahora seria calibracion post-hoc.
- Que el resultado de 2020 generalize a futuros episodios de estres.
- Que la candidata funcione fuera de la muestra analizada.

**PROHIBIDO (por dictamen externo previo y por el propio plan v3):**

- Introducir un filtro VIX/drawdown sin validacion OOS explicita.
- Recalibrar la candidata sobre los resultados del analisis anual.
- Activar SOW en produccion con base en este analisis.
## 5. Contraejemplos relevantes: 2018 y 2022

Dos anos objetivamente dificiles para el mercado no muestran efecto:

- **2018** (correccion Q4, S&P -20% pico a valle):
  lift +0.054, IC95% [-0.014, +0.129]. Cruza 0.
- **2022** (bear market ordenado, S&P -25% anual):
  lift -0.041, IC95% [-0.137, +0.062]. Cruza 0.

Estos dos anos refinan la hipotesis: el SOW no responde a "estres
generico" sino, posiblemente, a un subtipo concreto. Esto es mas
interesante tecnicamente pero tambien mas fragil: no se debe intentar
identificar el subtipo mirando estos resultados.

## 6. Multiple testing: declaracion explicita

El analisis incluye 12 comparaciones (un bloque por ano) mas los 2
bloques del test confirmatorio. Con ~14 contrastes al 95%, es
esperable que 1-2 aparezcan "significativos" por azar.

Esta declaracion debe acompanar cualquier lectura del analisis anual.
El test confirmatorio (seccion 2) es el unico que tiene caracter
decisorio. El analisis anual no lo tiene.

## 7. Nota: 2026 es ano incompleto

El bloque 2026 cubre 2026-01-01 -> 2026-10-01 (9 meses, no 12).
No afecta a la comparacion por episodios, pero afecta a cualquier
lectura "por ano" que asuma exposicion temporal homogenea.

## 8. Estado de la candidata tras este analisis

    Test confirmatorio          FAIL
    Analisis exploratorio       heterogeneidad
    Candidata                   CONGELADA / NO PRODUCTION
    config.WYCKOFF_SOW_*        None (intacto)
    fail-closed                 INTACTO
    5b.X (2027)                 INTACTO, juez final

Ninguna de las conclusiones de este expediente modifica:
- el contrato v1.9 (candidata congelada);
- el protocolo 5b.X v3 (firmado);
- la configuracion productiva;
- el estado fail-closed del modulo.
## 9. Propuesta al auditor externo

Se somete a dictamen externo:

1. **Aceptar el FAIL del test confirmatorio** como resultado del plan
   v3 §15. La candidata SOW no se activa.
2. **Registrar el hallazgo exploratorio** ("heterogeneidad temporal,
   2020 destacado, sin evidencia negativa") como hipotesis
   documentada, sin caracter normativo.
3. **Mantener 5b.X (2027) como juez final.** Si en 2027 la candidata
   pasa el OOS real, la decision se revisa. Si falla, se descarta
   definitivamente.
4. **No introducir filtro de regimen** ahora. Si el auditor lo
   considera oportuno en el futuro, requerira un diseno ex-ante
   especifico (definicion de regimen, criterio de exito, muestra).
5. **Migrar v1.8-core** (4 fases sin SOW) al pipeline como proceso
   independiente. No depende del resultado de SOW.

## 10. Artefactos

- Scripts:
  - `scripts/walk_forward_sow.py` (test confirmatorio).
  - `scripts/walk_forward_sow_anual.py` (analisis exploratorio).
  - `scripts/download_extended_history.py` (descarga historico).
  - `scripts/build_extended_dataset.py` (fusion de parquets).
- Datos:
  - `data/stock_prices_extended.parquet` (10.7 anos, NO productivo).
- Resultados:
  - `outputs/audit/wyckoff_walk_forward.csv` + `_summary.json`.
  - `outputs/audit/wyckoff_walk_forward_anual_ext.csv` + `_summary.json`.
- Commits: pendientes (este expediente cierra la sesion).

---

**Fin del expediente 41.**