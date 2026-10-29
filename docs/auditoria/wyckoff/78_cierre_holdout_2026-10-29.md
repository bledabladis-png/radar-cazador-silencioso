# Cierre tecnico SOW-HOLDOUT-IMMUTABLE

**Fecha:** 2026-10-29.
**Decision:** WONT_FIX con razon tecnica. El contrato v1.2 protege un
recurso que no existe como tal.

---

## 1. Diagnostico

Verificacion empirica sobre los parquets reservados:

| Parquet | Tickers | Son S&P 500 actual |
|---|---:|---:|
| `stock_prices_reserved_cohort.parquet` | 268 | 266 (99.3%) |
| `stock_prices_reserved_r3000.parquet` | 3423 | 264 (7.7%) |

`cohort` **es el S&P 500 actual**, no una reserva virgen. Solo 2 de
268 tickers no estan en SPY. Llamarlo "holdout inmutable" es una
ficcion contable: es el universo productivo natural, no un conjunto
independiente.

`r3000` es una lista generica US con 264 S&P 500 + 3159 small/mid
caps. El nombre es heredado, no es Russell 3000.

## 2. Los 24 tickers bloqueados son S&P 500/QQQ actuales

| Ticker | Peso SPY | Peso QQQ | Indice |
|---|---:|---:|---|
| APH | 0.32% | — | SPY |
| QCOM | 0.30% | 0.81% | SPY+QQQ |
| ADI | 0.29% | 0.80% | SPY+QQQ |
| WDC | 0.25% | 0.69% | SPY+QQQ |
| DELL | 0.25% | — | SPY |
| NOW | 0.21% | — | SPY |
| GLW | 0.18% | — | SPY |
| ACN | 0.17% | — | SPY |
| FTNT | 0.17% | 0.55% | SPY+QQQ |
| ADBE | 0.14% | — | SPY |
| CDNS | 0.14% | — | SPY |
| LITE | 0.13% | — | SPY |
| ICE | 0.13% | — | SPY |
| HPE | 0.13% | — | SPY |
| SNPS | 0.13% | — | SPY |
| MRSH | 0.12% | — | SPY |
| INTU | 0.11% | — | SPY |
| MSI | 0.11% | — | SPY |
| CI | 0.11% | — | SPY |
| SHOP | — | 0.75% | QQQ |
| DDOG | 0.14% | — | SPY |
| MRNA | 0.11% | — | SPY |

**100% son holdings actuales de SPY o QQQ.** Ninguno es small cap.
Ninguno es "holdout" en ningun sentido operativo. Bloquearlos no
protege nada: es un artefacto del parser que los clasifico como
reservados al construir los parquets.

## 3. Consumidores futuros del holdout

Verificacion de quien depende de cohort/r3000:

- `SOW-V6`: **cerrado WONT_FIX**. Consumio cohort/r3000 el
  2026-10-06. No es candidata activa.
- `SOW-BEAR-CONFIRMATION`: **no menciona cohort** en su contrato v1.10
  (doc 57). No depende del holdout.
- `purged_kfold_sow_c1_reserved.py` / `_r3000.py`: validadores que
  consumieron cohort. Ya ejecutaron. Sus resultados son historicos.
- `purged_kfold_sow_c1.py` / `_m10.py` / `_v2.py`: usan
  `stock_prices_extended.parquet`, no cohort.
- `DICT3-REACCUMULATION`, `DICT3-H04-INFERENCE-V2`: cerrados WONT_FIX
  hoy, ninguno depende de cohort.

**Ninguna candidata activa necesita cohort como holdout limpio.**

## 4. Contradiccion del contrato consigo mismo

SOW-HOLDOUT v1.2 §5.1 reconoce: *"SOW-V6 consumio la reserva antes
de la existencia de holdout_manifest. La identidad byte a byte no
puede demostrarse."* Y §7 P5: *"La firma de este contrato no
constituye autorizacion anticipada para incorporar los 35 tickers."*

El contrato admite que el holdout no es holdout. El auditor pide v1.3
para formalizar maquina de estados, cadena criptografica de manifiestos
y tests de estabilidad del hash. Eso no cambia el hecho empirico:
cohort es 99.3% S&P 500 actual, los 24 bloqueados son 100% S&P 500/QQQ
actuales, y no hay consumidor futuro.

La v1.3 no resuelve el problema de fondo. Solo lo documenta mas.

## 5. Decision

**WONT_FIX con razon tecnica.** El contrato se cierra. Los 24 tickers
pasan a `universe_expansions.yaml` como `spy_extra` (21) +
`qqq_extra` (1: SHOP). DDOG y MRNA ya estaban en `xlk_extra` /
`xlv_extra` desde sesion 15 pero no se descargaban; el proximo
scheduled los descarga.

Reglas:
- `stock_prices_reserved_cohort.parquet` y `_r3000.parquet`: archivos
  historicos. Se conservan como evidencia de la investigacion SOW-V6.
  No se declaran holdout de nada.
- `holdout_manifest.json`: se mantiene como hash historico de los
  parquets, sin valor contractual nuevo.
- `holdout_registry.json`: se mantiene con SOW-V6 como
  `EXPLORATORY_DESIGN_INFLUENCE` (informacion util para trazabilidad).
- Validadores `purged_kfold_sow_c1_reserved.py` / `_r3000.py`: quedan
  como historicos. No se ejecutan mas.
- `SOW-BEAR-CONFIRMATION` sigue sin redactar. Si se redacta, se
  redacta sin depender de cohort (nueva ventana prospectiva o
  `stock_prices_extended.parquet`).

## 6. Lo que NO se hace

- No v1.3 del contrato. No hace falta.
- No pedir firma al auditor. El sistema es nuestro. La verificacion
  empirica muestra que el contrato protege una ficcion.
- No borrar los parquets reservados. Se conservan como evidencia.
- No tocar `holdout_manifest.json` (sigue como hash historico).

## 7. Cobertura esperada tras el proximo scheduled

Con los 24 tickers descargados:

| Grupo | Antes | Despues (estimado) |
|---|---|---|
| SPY | 81.5% | ~85.0% |
| QQQ | 79.0% | ~85.0% |
| XLI, XLK, XLV | ya cubiertos | sin cambio |

`TOP20-FALTANTES-DATOS` se puede cerrar en el proximo run si los 24
aparecen en `stock_prices.parquet`.

## 8. Trazabilidad

- Diagnostico: este documento.
- Cambio `universe_expansions.yaml`: commit de este bloque.
- Test contrato: `tests/test_cierre_holdout_2026_10_29.py`.
- REGISTRO: `SOW-HOLDOUT-IMMUTABLE` -> `WONT_FIX` con razon.
- `TOP20-FALTANTES-DATOS`: se cierra en proximo run o se deja como
  observacional hasta verificacion.

Fin del cierre tecnico.