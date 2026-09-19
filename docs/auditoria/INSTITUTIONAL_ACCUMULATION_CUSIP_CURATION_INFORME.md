# INFORME DE CURACION CUSIP -> TICKER

**Modulo:** IAE (Institutional Accumulation Evidence) - SEC 13F
**Referencia:** prompt v6.35, HEAD 10a5cc2
**Fecha:** 2026-09-19
**Estado:** Analisis cerrado. Dictamen Q-CUR-1 / Q-CUR-2 aplicado. Alcance minimo aprobado.

---

## 0. Resumen ejecutivo

Gate 0 ejecutado sobre el estado real del crosswalk CUSIP -> ticker. Hallazgos:

- Tabla operativa vacia (solo cabecera, 74 bytes).
- Resolver con 17 tests verdes; cumple los 4 controles del dictamen FA-2.2.
- Sin consumidores en produccion (solo re-exports en `identity/__init__.py`).
- Poblar la tabla no altera output actual (riesgo = 0 en el pipeline).

Los 5 casos (DD, HON, XOM, FDXF, HONA) NO son homogeneos. Se dividen en dos grupos con semantica distinta. El alcance minimo aprobado por el auditor es poblar solo 3 filas COM (equity Q1 2026), excluyendo CALL/PUT y entidades post-Q1.

## 1. Contexto y alcance

- Alcance: `data/mappings/cusip_ticker_exceptions.csv` (tabla de excepciones con vigencia temporal).
- Fuera de alcance: modificacion del resolver, modificacion del contrato v1.1, tratamiento de derivados.
- Fuente primaria 13F: `D:\13f_probe\processed\2026Q1\INFOTABLE.parquet` (3,822,885 filas).
- Fuente local de contraste: `data/etf_holdings.csv` (526 filas; columnas etf/ticker/identifier/weight).

## 2. Estado real del crosswalk

### 2.1. Tabla operativa

- Ruta: `data/mappings/cusip_ticker_exceptions.csv`.
- Tamano: 74 bytes. Solo cabecera.
- Columnas: `CUSIP,ticker,valid_from,valid_to,source,reason,title_of_class,verified_by`.

### 2.2. Resolver

- Fichero: `src/institutional_accumulation/sec_13f/identity/cusip_resolver.py`.
- Firma: `resolve_cusip(cusip, report_period, exceptions_df)`.
- Controles activos: columnas obligatorias, `valid_from <= valid_to`, `source` no prohibido, `verified_by` en `{SEC, OpenFIGI, manual}`, sin solapes por `(CUSIP, ticker)`.
- Tests: 17 verdes en `tests/test_sec_13f_cusip_resolver.py`.

### 2.3. Consumidores

- `resolve_cusip`, `resolve_batch`, `load_exceptions`: sin consumidores en produccion.
- Solo re-exports en `src/institutional_accumulation/sec_13f/identity/__init__.py`.
- `market_hours.py::_load_exceptions` es homonimo, no consumidor.

## 3. Evidencia empirica

### 3.1. CUSIPs locales en `etf_holdings.csv`

- DD -> `26614N201` (no observado en 13F Q1 2026).
- HON -> `438516205` (no observado en 13F Q1 2026).
- XOM -> `30233Q108` (no observado en 13F Q1 2026).
- HONA -> `43849R105` (XLI, weight 0.899%).
- FDXF -> `314352105` (XLI, weight 0.256%).

### 3.2. CUSIPs observados en 13F Q1 2026 (`INFOTABLE.parquet`)

- DuPont: `26614N102` (COM), `26614N902` (CALL), `26614N952` (PUT).
- Honeywell: `438516106` (COM), `438516906` (CALL), `438516956` (PUT).
- Exxon Mobil: `30231G102` (COM), `30231G902` (CALL), `30231G952` (PUT).
- HONA `43849R105`: 0 filas.
- FDXF `314352105`: 0 filas.

### 3.3. Tabla comparativa

| Ticker | CUSIP local | CUSIP en 13F Q1 2026 | Grupo |
|---|---|---|---|
| DD | `26614N201` | `26614N102` | A |
| HON | `438516205` | `438516106` | A |
| XOM | `30233Q108` | `30231G102` | A |
| HONA | `43849R105` | (ausente) | B |
| FDXF | `314352105` | (ausente) | B |

## 4. Reconciliacion temporal entre CUSIP local y CUSIP observado en 13F Q1 2026

Terminologia aprobada. NO se usa "CUSIP local desactualizado" ni "CUSIP obsoleto". Se usa "discrepancia temporal" o "reconciliacion temporal".

### 4.1. Grupo A - Discrepancia temporal / corporate-action

- DD `26614N201` vs `26614N102`: variante temporal. El CUSIP local no se observa en 13F Q1 2026; el CUSIP 13F es el observado.
- HON `438516205` vs `438516106`: reverse split 2:1 efectivo el 29-06-2026 (posterior a Q1 2026). `438516106` es el identificador pertinente en Q1 2026; `438516205` es el identificador posterior a la corporate action.
- XOM `30233Q108` vs `30231G102`: `30233Q108` no se observa en el universo 13F Q1 2026. NO se presume que sea variante historica de `30231G102` sin evidencia adicional.

### 4.2. Grupo B - Entidades post-Q1 (spin-offs)

- HONA: Honeywell Aerospace inicio cotizacion independiente el 29-06-2026. Posterior a `PERIODOFREPORT=2026-03-31`.
- FDXF: FedEx Freight inicio cotizacion independiente el 01-06-2026. Posterior a `PERIODOFREPORT=2026-03-31`.
- Ambas entidades: FUERA del universo temporal Q1 2026.

## 5. Dictamen del auditor

### 5.1. Q-CUR-1 - CALL/PUT

EXCLUIR del crosswalk de equity.

- Los CUSIPs `26614N902`, `26614N952`, `438516906`, `438516956`, `30231G902`, `30231G952` no participan en este crosswalk.
- Los derivados se preservan en el 13F (no se eliminan de `INFOTABLE`).
- Mapping de derivados a subyacente: modelo separado, no implementado en este ciclo.

### 5.2. Q-CUR-2 - HONA / FDXF

EXCLUIR del CSV operativo Q1 2026.

- No poblar con `valid_from = fecha de spin-off` sin observacion 13F que lo requiera.
- Documentar en informe / FOLLOWUPS como `FUTURE_CORPORATE_ACTION / POST_Q1_ENTITY`.
- Poblar cuando aparezcan en un periodo 13F posterior con evidencia primaria.

### 5.3. Alcance minimo aprobado

Poblar UNICAMENTE las 3 filas COM del Grupo A:

    26614N102 -> DD
    438516106 -> HON
    30231G102 -> XOM

Sin CALL/PUT. Sin HONA. Sin FDXF. Sin modificacion del resolver. Sin modificacion del contrato v1.1.

## 6. Correcciones documentales incorporadas

1. Grupo A etiquetado como "discrepancia temporal / corporate-action" (no "CUSIP local obsoleto").
2. HON: `438516205` es el CUSIP posterior al reverse split 2:1 efectivo el 29-06-2026, no un CUSIP obsoleto.
3. XOM: `30233Q108` no se presume variante historica de `30231G102`. Se afirma solo "no observado en 13F Q1 2026".
4. Seccion 4 renombrada a "Reconciliacion temporal entre CUSIP local y CUSIP observado en 13F Q1 2026".

## 7. Secuencia de ejecucion

    Gate 0 CUSIP
          [OK]
            |
    informe documental (este fichero)
            |
    dictamen Q-CUR-1 / Q-CUR-2 (aplicado)
            |
    poblar solo 3 COM Q1
            |
    tests + probe real
            |
    mini-gate

## 8. Fuentes primarias

- SEC HON reverse split (29-06-2026): https://www.sec.gov/Archives/edgar/data/773840/000077384026000084/hon-20260625.htm
- Honeywell Aerospace spin-off (29-06-2026): https://www.honeywell.com/us/en/news/press-releases/2026/06/honeywell-aerospace-completes-spin-off-from-honeywell-technologies-and-begins-trading-on-nasdaq
- FedEx Freight spin-off (01-06-2026): https://newsroom.fedex.com/newsroom/global-english/fedex-completes-spin-off-of-fedex-freight
- 13F Q1 2026 (local): D:\13f_probe\processed\2026Q1\INFOTABLE.parquet
- etf_holdings.csv (local): data/etf_holdings.csv

## 9. Proximos pasos

1. Commit documental (este informe).
2. Poblar `data/mappings/cusip_ticker_exceptions.csv` con 3 filas COM (Seccion 5.3).
3. Validar con `load_exceptions()` + tests + probe real contra 13F Q1 2026.
4. Mini-gate: verificar que `resolve_cusip("26614N102", "2026-03-31", exc)` devuelve `DD`, y analogos para HON y XOM.
5. Actualizar FOLLOWUPS.md con cierre del ciclo.

---

Fin del informe. Version 1.0 (2026-09-19). Referencia: prompt v6.35, HEAD 10a5cc2.
