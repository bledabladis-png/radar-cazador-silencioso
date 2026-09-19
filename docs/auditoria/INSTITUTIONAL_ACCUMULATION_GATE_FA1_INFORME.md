# INFORME GATE FA-1 - IAE SEC 13F

Version: 1.0 (2026-09-19)
Contrato: docs/auditoria/INSTITUTIONAL_ACCUMULATION_CONTRATO.md (v1.1)
Addendum: docs/auditoria/INSTITUTIONAL_ACCUMULATION_CONTRATO_ADDENDUM.md
Prompt de referencia: v6.33
Estado: Gate FA-1 superado. Pendiente push.
Naturaleza: no normativo. Si hay conflicto, gana el Prompt Maestro.

---

## 1. Proposito

Cerrar el Gate FA-1 autorizado por el dictamen Gate 0 sobre la Fase A
del modulo IAE (Institutional Accumulation Evidence). Demostrar que la
capa de ingestion + schema + lineage del dataset SEC 13F es funcional,
auditable y consistente contra evidencia empirica real.

Alcance FA-1: ingestion, normalizacion, escritura atomica de parquets,
manifest de lineage a 3 niveles, validacion estructural.

Fuera de alcance FA-1: resolucion de identidad (Q2), CUSIP->ticker,
filtro PERIODOFREPORT, DeltaShares, NIPC, breadth, clasificacion.

## 2. Dataset verificado

| Campo | Valor |
|---|---|
| Dataset | SEC Form 13F Data Set |
| Trimestre | 2026 Q1 (quarter=2026Q1) |
| Periodo SEC | 01mar2026-31may2026 (source_period) |
| ZIP | 01mar2026-31may2026_form13f.zip |
| Tamano ZIP | 99,411,274 bytes (99.4 MB) |
| SHA-256 ZIP | 05f4da8f526cd471a387af45661a8cb3bbfaf345d3ec9c2cefd31349aa257e63 |
| Fuente | sec.gov / structureddata / form-13f-data-sets |

El SHA-256 del ZIP coincide entre el probe en memoria y la lectura
posterior del manifest desde disco (roundtrip JSON verificado).

## 3. Row counts vs Gate 0

El parser de FA-1 NO filtra por PERIODOFREPORT. Procesa el dataset
completo del ZIP, que incluye filings tardios de periodos historicos
agrupados por FILING_DATE (01-mar-2026 a 31-may-2026). Esto es
correcto por diseno: la ingestion es pasiva; el filtrado temporal
corresponde a FA-2 (identity resolution).

| TSV | Filas reales | Gate 0 empirico | Estado |
|---|---|---|---|
| SUBMISSION | 11,761 | 11,761 | OK |
| COVERPAGE | 11,761 | 11,761 | OK |
| SUMMARYPAGE | 9,846 | 9,846 | OK |
| OTHERMANAGER | 4,751 | 4,751 | OK |
| OTHERMANAGER2 | 4,242 | 4,242 | OK |
| SIGNATURE | 11,761 | 11,761 | OK |
| INFOTABLE | 3,822,885 | 3,822,885 | OK |

Resultado: 7/7 coincidencias exactas. El parser reproduce los conteos
del analisis manual del Gate 0 sin perdida ni duplicacion.

## 4. Artefactos generados

Layout en disco del probe (D:/13f_probe, fuera del repo):

    raw/01mar2026-31may2026/
      01mar2026-31may2026_form13f.zip
      extracted/*.tsv (7 TSVs)
    processed/2026Q1/
      *.parquet (7 ficheros)
    manifests/
      sec_13f_2026Q1.json

| Parquet | Tamano (bytes) |
|---|---|
| SUBMISSION.parquet | 209,118 |
| COVERPAGE.parquet | 823,480 |
| SUMMARYPAGE.parquet | 196,622 |
| OTHERMANAGER.parquet | 140,481 |
| OTHERMANAGER2.parquet | 127,768 |
| SIGNATURE.parquet | 590,029 |
| INFOTABLE.parquet | 104,363,361 |
| Manifest JSON | 3,993 |

Los 7 parquets se generan con compresion snappy. La escritura es
atomica (.tmp + os.replace). El manifest se escribe con la misma
disciplina.

## 5. Cadena de hash a 3 niveles

El manifest registra lineage completo:

  Nivel 1: source.sha256  (SHA-256 del ZIP descargado)
  Nivel 2: tsv_files[name].sha256 + size_bytes + rows
  Nivel 3: parquet_files[name].sha256 + size_bytes + rows

Los 3 niveles se verificaron no vacios (7 + 7 entradas). Sin el nivel
intermedio (TSV) no se puede demostrar que artefacto intermedio
produjo cada parquet. La cadena SEC ZIP -> TSV -> Parquet es
auditable completa.

## 6. Validation flags

| Flag | Valor |
|---|---|
| expected_files_present | True |
| zip_crc_valid | True |
| schema_valid | True |

expected_files_present: los 7 TSVs estaban en el ZIP.
zip_crc_valid: zipfile.testzip() no detecto corrupcion.
schema_valid: las columnas de los 7 TSVs coinciden con EXPECTED_COLUMNS.

## 7. Verificacion agregada

| Verificacion | Resultado |
|---|---|
| compileall | OK (silencio) |
| pyflakes | 0 warnings |
| pytest IAE | 72 passed |
| pytest global | 727 passed + 2 skipped |
| git estado | [ahead 6] pre-informe, sin untracked |

## 8. Observaciones

O1. INFOTABLE.parquet pesa 99.53 MB, por encima de la horquilla
30-60 MB barajada en el transfer. Causa: el parser de FA-1 procesa
el dataset COMPLETO (3,822,885 filas), no el filtrado a Q1
(3,321,967). Correcto por diseno. El filtro va en FA-2.

O2. Los extras del ZIP (FORM13F_readme.htm, FORM13F_metadata.json)
NO se extraen. extract_13f_zip solo extrae los 7 TSVs esperados.
Verificado por test test_extract_extras_ignorados.

O3. El manifest se lee desde disco con read_manifest y coincide campo
a campo con el dict devuelto por ingest_13f en memoria. Roundtrip JSON
verificado.

O4. Desviaciones respecto al CONTRATO v1.1 (7 parquets, ubicacion del
manifest, 3 niveles hash, nombres de campo, size-check diferido) estan
recogidas en el addendum al contrato v1.1. Ver docs/auditoria/
INSTITUTIONAL_ACCUMULATION_CONTRATO_ADDENDUM.md.

O5. El ZIP descargado a D:/13f_probe NO contamina el repo. git status
limpio. Las reglas .gitignore para data/sec_13f/ cubren produccion;
el probe usa una ruta externa al repo.

## 9. Recomendacion

GO para push a main.

Criterios satisfechos:
  - Los 6 modulos (downloader, schema, parser, storage, manifest,
    ingest) con tests verdes (72 IAE).
  - pyflakes 0 warnings.
  - 727 tests globales passing.
  - Probe integracion real OK contra 3.3M filas.
  - Manifest v1 con 3 niveles de hash verificable.
  - Addendum al contrato presente y commiteado.
  - Repo sin contaminacion.

Se autoriza push unico de los commits locales acumulados. Tras el
push, NIPC permanece BLOQUEADO hasta Gate FA-2.

## 10. Estado del repo

HEAD local pre-informe: 387e993 ([ahead 6] de origin/main 1f66b74).
Tras commitear este informe: [ahead 7].

Cadena de commits FA-1:

  0017011  FA-1.1 downloader + schema
  0568992  FA-1.2 parser
  ee8809f  FA-1.3 storage + manifest
  0c587f3  FA-1.4 ingest orquestador
  52d098b  fixes tecnicos (pyflakes + docstring + DATE_COLUMNS + removesuffix)
  387e993  addendum al contrato v1.1
  <este>  informe Gate FA-1

---

Fin del informe. Version 1.0 (2026-09-19).
Referencia: prompt v6.33, HEAD previo al push.
