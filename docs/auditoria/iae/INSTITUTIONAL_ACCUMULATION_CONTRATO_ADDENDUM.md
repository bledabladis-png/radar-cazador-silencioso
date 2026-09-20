# ADDENDUM AL CONTRATO IAE v1.1

Version: 1.0 (2026-09-19)
Contrato base: docs/auditoria/INSTITUTIONAL_ACCUMULATION_CONTRATO.md (v1.1)
Dictamen Gate 0: docs/auditoria/INSTITUTIONAL_ACCUMULATION_DICTAMEN_GATE0.md
Prompt de referencia: v6.33
Naturaleza: reconciliacion documental. NO modifica el CONTRATO v1.1.

---

## 1. Proposito

El CONTRATO v1.1 se redacto antes del dictamen de FA-1.3. Durante
FA-1.1..FA-1.4 el auditor aprobo desviaciones tecnicas sobre Q6
(almacenamiento) y Q2 (framework de identidad). Estas desviaciones
son las implementadas y no estan recogidas en el CONTRATO v1.1.

Este addendum captura las desviaciones sin reabrir el contrato, en
virtud de la regla "contratos IAE inmutables dentro de una version".

## 2. Desviaciones respecto al CONTRATO v1.1

### D1 - Multiples parquets por trimestre

CONTRATO seccion 4/Q6: `data/sec_13f/2026Q1.parquet` (1 fichero).
Implementado: `data/sec_13f/processed/{quarter}/{TSV_NAME}.parquet` (7 ficheros).

Razon: preservacion del modelo relacional SEC. Las 7 tablas tienen
semanticas y cardinalidades distintas; colapsarlas pierde la
correspondencia directa con el dataset fuente. Aprobado en FA-1.3.

### D2 - Ubicacion del manifest

CONTRATO seccion 4/Q6: `data/manifests/sec_13f_2026Q1.json`.
Implementado: `data/sec_13f/manifests/sec_13f_2026Q1.json`.

Razon: coherencia estructural. Todo el ciclo SEC 13F (raw, processed,
manifests) queda bajo `data/sec_13f/`, reduciendo dispersion de rutas
en el arbol de datos. Aprobado en FA-1.3.

### D3 - Cadena de hash a 3 niveles

CONTRATO seccion 4/Q6: `source_sha256` + `parquet_sha256` (2 hashes).
Implementado: `source.sha256` + `tsv_files[name].sha256` +
`parquet_files[name].sha256` (3 niveles).

Razon: sin el nivel intermedio (TSV) no se puede demostrar que
artefacto intermedio produjo cada parquet. Cadena auditable completa:
SEC ZIP -> TSV -> Parquet. Aprobado en FA-1.3.

### D4 - Nombres de campo del manifest

| CONTRATO v1.1 | Implementado |
|---|---|
| `source_dataset` | `dataset` |
| `downloaded_at` | `generated_at_utc` |
| `parquet_sha256` (unico) | `parquet_files[name].sha256` (por artefacto) |

`generated_at_utc` refleja el momento de construccion del manifest,
no el momento de descarga. La diferencia es semantica, no cosmetica.

### D5 - Verificacion de tamano en cache-hit del downloader

Dictamen FA-1.1 requeria verificacion de tamano contra Content-Length
en cache-hit. Implementado: solo cache-hit por existencia.

Razon de la desviacion: `extract_13f_zip` ya verifica integridad
mediante CRC (`zipfile.testzip()`). Un ZIP truncado falla en esa capa.
Anadir un HEAD request en cache-hit supondria latencia adicional y
otro punto de fallo (rate limiting SEC) con beneficio marginal.

Docstring de `download_13f_zip` actualizado para reflejar el
comportamiento real: no hay verificacion de tamano en cache-hit.
Verificado en commit 52d098b.

## 3. Campos excluidos del manifest en FA-1

CONTRATO seccion 4/Q6 lista `filings_count`, `accession_min`, `accession_max`.
No incluidos en FA-1. Razon: FA-1 es ingestion pasiva; no calcula
metricas de negocio sobre filings ni rangos de accession.

Estos campos se anadiran a partir de FA-2, cuando exista logica de
resolucion de identidad. El informe Gate 0 ya contiene los valores
empiricos para Q1 2026.

## 4. Alcance

Este addendum NO modifica el CONTRATO v1.1. Documenta desviaciones
aprobadas por el auditor externo durante FA-1. Si el dictamen de
Gate FA-1 requiere ajustes adicionales, se emitira un addendum v1.1
o un contrato v1.2 segun corresponda.

---

Fin del addendum. Version 1.0 (2026-09-19).
Referencia: HEAD local previo a Gate FA-1.
