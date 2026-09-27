# AUDITORIA EXTERNA DEL IAE - Registro consolidado

**HEAD auditado:** `de8b8ffa34102a19a6005fce584a34a8268dae87`
**Fecha de emision del dictamen:** 2026-09-27
**Auditor:** LLM externo independiente (no es el Ingeniero Supervisor)
**Estado final:** **APROBADO CON CONDICIONES**

---

## 1. Alcance

Auditoria completa del modulo IAE (Institutional Accumulation Engine):

- Diseno arquitectonico.
- Implementacion y superficie publica.
- Verificacion (tests, cobertura, integracion).
- Integracion en produccion (`run.py`, workflows GitHub Actions).
- Documentacion normativa.

**Fuera de alcance:** resto del sistema radar (regimenes, scores, breadth, otros modulos).

---

## 2. Dictamen global

> El modulo presenta una arquitectura adecuada, separacion de responsabilidades, contratos observables explicitos, integracion productiva verificable, aislamiento correcto y una infraestructura de pruebas y evidencia considerablemente desarrollada. No obstante, el control E2E contractual presenta un FAIL reproducible respecto del baseline historico. La hipotesis de que el drift procede exclusivamente de la regeneracion del crosswalk es plausible y coherente con los datos aportados, pero no queda demostrada de forma forense mediante la trazabilidad de los registros afectados. Asimismo, el golden utilizado por el control no esta formalizado como artefacto independiente y la semantica temporal del workflow trimestral requiere aclaracion.

**Estado:** `APROBADO CON CONDICIONES`
**Bloqueantes para APROBADO definitivo:** H1-B, H4, H5.3.
**No bloqueantes:** H2, H3, H5.1, H5.2, H5.4, H5.5, H5.7.

---

## 3. Hallazgos

### 3.1. H1-A - Drift historico del E2E - CERRADO

**Problema:** `scripts/iae_contractual_nipc_e2e.py` falla contra el golden seccion 12.5 (`553.319` vs `553.321`).

**Causa demostrada:** regeneracion del crosswalk por Fase G. Los CUSIPs `023135906` (AMZN) y `595112903` (MU) fueron retirados; `84615Q103` (SPCX) anadido. Delta rows = -2; delta NIPC = -943.371.

**Prueba forense:** al restaurar el crosswalk `7caa86b`, el E2E reproduce el golden seccion 12.5 con tolerancia 0.

**Estado:** cerrado por el auditor. No reabrir.

### 3.2. H1-B - Clasificacion CALL/PUT incompleta - ABIERTO / BLOQUEANTE

**Problema:** el filtro `_filter_canonical` acepta filas con `PUTCALL=NULL` y `TITLEOFCLASS` con semantica de opcion.

**Alcance:** 97 CUSIPs afectados en Q4 2025, 28 en Q1 2026.

**Dano medido:** +53.228.375 sobre el NIPC actual (`-4.317.678.307` -> `-4.264.449.932` con filtro reforzado).

**Instruccion del auditor:**

- Fuente primaria de tipo: Official List SEC (Option Indicator en posicion 10).
- Resolucion temporal: lista Q4 2025 para periodo Q4 2025, etc.
- Aplicacion en capa de identidad, no en `_filter_canonical`.
- Regex `TITLEOFCLASS` como defensa secundaria.
- Auditoria explicita de exclusiones (contadores).
- No congelar NIPC baseline hasta cerrar H1-B.

### 3.3. H4 - Golden no versionado - ABIERTO, pendiente H1-B

**Problema:** `EXPECTED` hardcoded en `iae_contractual_nipc_e2e.py`. No hay artefacto independiente con hashes.

**Estructura propuesta por el auditor:**

```
docs/auditoria/iae/golden/
  12_5_historic.json       # Snapshot seccion 12.5 (7caa86b) - INMUTABLE
  current.json             # Snapshot post-fix H1-B (a crear)
  README.md
```

Cada golden debe incluir: `periods`, `crosswalk_sha256`, `catalog_sha256`, `parquet_hashes`, `git_head`, `expected`, `generated_at`, **`security_type_source`**, **`security_type_source_hash`**.

### 3.4. H5 - Workflow trimestral - AUDITADO, HALLAZGOS ABIERTOS

| ID | Hallazgo | Severidad |
|---|---|---|
| **H5.1** | Criterio de seleccion duplicado (workflow 51d vs script 60d) | MEDIO |
| **H5.2** | Fallo de ingesta sin alerta (no retry, no Issue) | MEDIO |
| **H5.3** | Q2 2026 ingestado manualmente (commit `c5e3ee0`, autor humano) | **BLOQUEANTE** |
| **H5.4** | Workflow no valida formato de `inputs.quarter` | BAJO |
| **H5.5** | Cache no invalida ante republicaciones SEC | BAJO |
| **H5.6** | Official List Q2 fallo (404) resuelto con sufijo `-txt` | HISTORICO / CERRADO |
| **H5.7** | Commit mixto `c5e3ee0` (13F + outputs pipeline) | BAJO / HISTORICO |

### 3.5. H2 / H3 - Hallazgos menores no bloqueantes

- **H2:** documentacion desactualizada ("9 contratos" vs 10 reales; "42 ficheros" vs 45).
- **H3:** scripts auxiliares sin clasificacion explicita (normativo / auditoria / experimental).

### 3.6. O1 - Observacion sobre "no imputar"

El auditor pide evidencia especifica de ausencia de imputacion (`fillna`, `ffill`, `bfill`, `interpolate`) en todos los caminos de codigo del IAE. Cobertura de lineas != prueba semantica.

---

## 4. Estado de auditoria tras ultima respuesta

| Finding | Estado |
|---|---|
| H1-A | CERRADO |
| H1-B | ABIERTO / BLOQUEANTE |
| H4 | ABIERTO, pendiente H1-B |
| H5.1 | ABIERTO |
| H5.2 | ABIERTO |
| H5.3 | ABIERTO / BLOQUEANTE |
| H5.4 | ABIERTO |
| H5.5 | ABIERTO |
| H5.6 | CERRADO / HISTORICO |
| H5.7 | HISTORICO |
| O1 | OBSERVACION |

**Estado global:** **APROBADO CON CONDICIONES**.

---

## 5. Instruccion operativa del auditor

1. **No congelar ningun NIPC baseline todavia.** `-4.317.678.307` = `PRE_H1B_OBSERVATION`; `-4.264.449.932` = "resultado de mitigacion".
2. **Implementar fix definitivo H1-B** con Official List como fuente primaria.
3. **Mantener regex `TITLEOFCLASS`** como defensa secundaria.
4. **Rehacer E2E + reconciliacion + cobertura + determinismo** tras el fix.
5. **Solo entonces crear `current.json`.**
6. **H5 se mantiene como ciclo abierto** (H5.1, H5.2, H5.3, H5.4, H5.5, H5.7 son deuda operacional).

---

## 6. Correcciones documentales pendientes

- **Frase "sin cambios de codigo productivo en 3 ciclos"** debe reformularse a "sin cambios en la logica de calculo productiva del IAE" (el commit `08a6c6a` si toco codigo de workflow/script).
- **H5 no debe declararse "CERRADO"** mientras H5.3 sea bloqueante. Usar **"AUDITADO / HALLAZGOS ABIERTOS"**.

---

## 7. Referencias externas aportadas por el auditor

- SEC - Official List of Section 13(f) Securities: `https://www.sec.gov/rules-regulations/staff-guidance/official-list-section-13f-securities`
- SEC - Frequently Asked Questions About Form 13F: `https://www.sec.gov/rules-regulations/staff-guidance/division-investment-management-frequently-asked-questions/frequently-asked-questions-about-form-13f`
- SEC - Form 13F Data Sets: `https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets`
- CUSIP Global Services - About CGS Identifiers: `https://www.cusip.com/identifiers.html`

---

**Fin del registro.**
