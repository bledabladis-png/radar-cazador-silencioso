# INFORME GATE 0 FA-2 - IAE SEC 13F

Version: 1.0 (2026-09-19)
Contrato: docs/auditoria/INSTITUTIONAL_ACCUMULATION_CONTRATO.md (v1.1)
Addendum: docs/auditoria/INSTITUTIONAL_ACCUMULATION_CONTRATO_ADDENDUM.md
Dictamen previo: docs/auditoria/INSTITUTIONAL_ACCUMULATION_DICTAMEN_GATE0.md
Informe Gate FA-1: docs/auditoria/INSTITUTIONAL_ACCUMULATION_GATE_FA1_INFORME.md
Prompt de referencia: v6.34
Estado: inventario empirico + propuesta de diseno. Rev 2 (post dictamen auditor 2026-09-19, ver seccion 10).
Naturaleza: no normativo. Si hay conflicto, gana el Prompt Maestro.

---

## 1. Proposito

Cerrar el Gate 0 de FA-2 con evidencia empirica directa sobre los 7
parquets generados por FA-1 (3,822,885 filas INFOTABLE). Producir el
inventario real de fuentes disponibles para el resolver de identidad,
detectar bloqueantes no anticipados, y proponer las decisiones de
diseno que el auditor debe dictaminar antes de escribir codigo.

NO produce codigo. NO modifica el CONTRATO v1.1. NO toca el pipeline.
Solo evidencia empirica + propuesta de diseno.

## 2. Metodologia

Fuente: parquets generados por FA-1 en D:/13f_probe/processed/2026Q1/.
Script de reconocimiento: lee los 7 parquets + 2 CSVs del Radar
(data/etf_holdings.csv, data/mappings/isin_ticker_map.csv). Sin red.
Sin side effects. Fuera del repo.

Alcance del probe:
  Bloque 1 - confirmacion row counts y numero de columnas.
  Bloque 2 - SUBMISSION: SUBMISSIONTYPE + PERIODOFREPORT.
  Bloque 3 - COVERPAGE: REPORTTYPE + ISAMENDMENT.
  Bloque 4 - INFOTABLE: Q4 + Q2 empirico.
  Bloque 5 - esquema OTHERMANAGER + OTHERMANAGER2.
  Bloque 6 - etf_holdings.csv (fuente CUSIP->ticker).
  Bloque 7 - isin_ticker_map.csv.
  Bloque 8 - cobertura CUSIP INFOTABLE <-> etf_holdings.

## 3. Confirmaciones

| Parquet | Filas | Columnas |
|---|---|---|
| SUBMISSION | 11,761 | 5 |
| COVERPAGE | 11,761 | 21 |
| SUMMARYPAGE | 9,846 | 5 |
| OTHERMANAGER | 4,751 | 7 |
| OTHERMANAGER2 | 4,242 | 7 |
| SIGNATURE | 11,761 | 8 |
| INFOTABLE | 3,822,885 | 15 |

Los row counts coinciden con los medidos en FA-1.5 (probe real) y
con el dataset completo (no filtrado) del Gate 0 empirico original.
CUSIP presente en el 100% de las filas de INFOTABLE (0 null).

## 4. Discrepancias con el informe Gate 0

Todas las discrepancias tienen una unica causa: el informe Gate 0
midio sobre el dataset FILTRADO a PERIODOFREPORT=31-MAR-2026, mientras
que los parquets de FA-1 contienen el dataset COMPLETO del ZIP.
El parser de FA-1 no filtra (por diseno, ver addendum D5 y FA-1.5
observacion O1).

| Metrica | Informe (filtrado) | Parquets FA-1 (completo) | Delta |
|---|---|---|---|
| SUBMISSIONTYPE=13F-HR/A | 127 | 404 | +277 |
| REPORTTYPE=COMBINATION | 411 | 441 | +30 |
| ISAMENDMENT=Y | 441 | 441 | 0 |
| INFOTABLE filas | 3,321,967 | 3,822,885 | +500,918 |
| INFOTABLE CUSIPs unicos | 33,449 | 35,649 | +2,200 |
| OTHERMANAGER no vacio | 44.12% | 48.15% | +4.03% |

Ninguna es bug. Todas se alinean cuando FA-2 aplique el filtro
temporal sobre los parquets de FA-1.

### 4.1. Hallazgo reforzado Q8 (ISAMENDMENT)

ISAMENDMENT=Y aparece en 441 filas, exactamente el mismo numero que
REPORTTYPE=13F COMBINATION REPORT (441). La coincidencia es byte-exacta
sobre el dataset completo. Esto confirma empiricamente el dictamen Q8:
ISAMENDMENT=Y incluye COMBINATION, no solo enmiendas reales. La
clasificacion operativa debe ser por SUBMISSIONTYPE, no por ISAMENDMENT.

## 5. Hallazgos nuevos

### N1 - etf_holdings.csv: 526 filas, 519 CUSIPs unicos, 516 tickers

El informe Gate 0 mencionaba "501 CUSIPs". El CSV real tiene 526 filas,
519 identifiers unicos, 516 tickers unicos. La columna del CUSIP se
llama `identifier`, no `cusip`. Estructura: etf, ticker, identifier,
weight. Todos los identifiers tienen 9 caracteres (formato CUSIP
estandar).

Los 18 CUSIPs de diferencia respecto al informe requieren explicacion
antes de FA-2.2: ver Seccion 8 (pregunta al auditor).

### N2 - Interseccion INFOTABLE <-> etf_holdings = 499 CUSIPs

Sobre el dataset completo (sin filtrar): 499 CUSIPs comunes. El informe
Gate 0 reporto 496 tras filtrar a Q1 2026. Coherente: el filtro retira
~3 CUSIPs que no estaban presentes en el periodo canonico.

### N3 - isin_ticker_map.csv NO sirve para CUSIP->ticker

El fichero mapea ISIN -> ticker (49 filas, columnas isin + ticker).
Los CUSIPs del 13F no son ISINs. El unico mapping CUSIP disponible hoy
es data/etf_holdings.csv (columna identifier).

### N4 - OTHERMANAGER y OTHERMANAGER2 tienen semantica distinta

Esquemas:
  OTHERMANAGER : ACCESSION_NUMBER, OTHERMANAGER_SK, CIK, NAME, ...
  OTHERMANAGER2: ACCESSION_NUMBER, SEQUENCENUMBER,    CIK, NAME, ...

Semantica (a validar por el auditor):
  OTHERMANAGER  = sub-managers INCLUIDOS en este filing.
                  FK desde INFOTABLE.OTHERMANAGER -> OTHERMANAGER_SK.
  OTHERMANAGER2 = managers SUPERIORES para los que este filing reporta.
                  Relacion ascendente (inversa).

Ejemplo empirico: Eaton Vance Advisers International Ltd aparece en
OTHERMANAGER2 del accession 0000895421-26-000187 con SEQUENCENUMBER=22.

Grafo implicito para el nivel 3 del framework Q2:

  COVERPAGE.FILINGMANAGER (primario)
       |
       +-- incluye --> OTHERMANAGER (sub-managers)
       |
       +-- reporta para --> OTHERMANAGER2 (super-managers)
       |
       +-- discretion por fila --> INFOTABLE.INVESTMENTDISCRETION

### N5 - Sin tabla de excepciones CUSIP

Busqueda recursiva de *cusip* en el repo (excluyendo .git): 0 resultados.
La tabla con vigencia temporal propuesta en CONTRATO seccion 7 y
dictamen Q3 no existe. FA-2.2 debe crearla desde cero.

## 6. Decisiones de diseno

### Q-A - Filtro PERIODOFREPORT (aprobado por supervisor)

Opcion elegida: (1) filtro DENTRO del modulo FA-2, no en ingest_13f.
Razon: FA-1 es ingestion pasiva ya aprobada; el filtro es decision de
negocio, no de ingestion. Firma propuesta para FA-2.1:

    filter_by_period(dfs, period="2026-03-31") -> dict[str, DataFrame]

No modifica ingest_13f ni el parser de FA-1.

### Q-B - Clave canonica de manager (3 opciones al auditor)

Opcion B.1 (voto del supervisor):
  Clave: (primary_cik, submanager_cik|null, discretion_type, period).
  OTHERMANAGER2 = solo provenance, no entra en la clave.

Opcion B.2:
  Clave: (primary_cik, submanager_cik, supermanager_cik,
          discretion_type, period). Grafo completo.

Opcion B.3:
  Clave: (economic_owner_cik, discretion_type, period) + tabla de
  relaciones. OTHERMANAGER2 determina el economic_owner.

Razon del voto B.1: OTHERMANAGER2 describe una relacion INVERSA (quien
reporta PARA mi), no una atribucion de propiedad. La deduplicacion
real ocurre en OTHERMANAGER (quien esta DENTRO del filing).

Decision pendiente del auditor.

### Q-C - Tabla de excepciones CUSIP

Columnas propuestas (dictamen seccion 7 + 2 anadidos):

| Columna | Origen |
|---|---|
| CUSIP | dictamen seccion 7 |
| ticker | dictamen seccion 7 |
| valid_from | dictamen seccion 7 |
| valid_to | dictamen seccion 7 |
| source | dictamen seccion 7 |
| reason | dictamen seccion 7 |
| title_of_class | anadido (13F distingue clases) |
| verified_by | anadido (manual | OpenFIGI | SEC) |

Ubicacion: data/mappings/cusip_ticker_exceptions.csv.

### Q-D - Multiples variantes CUSIP para el mismo ticker

Modelo multi-fila con vigencia temporal. Ejemplo DuPont:

    CUSIP       ticker  valid_from  valid_to    source  reason
    26614N201   DD      2015-07-01  2017-08-31  SEC     DuPont pre-Corteva
    26614N902   DD      2017-09-01  2019-05-31  SEC     DuPont post-Dow
    26614N102   DD      2019-06-01  2023-12-31  SEC     DuPont post-Corteva
    26614N952   DD      2024-01-01  NULL        SEC     DuPont current

La vigencia temporal permite resolver el CUSIP correcto segun el
report_period del filing. Sin esto, un 13F historico con CUSIP
obsoleto quedaria UNMAPPED.

## 7. Sub-fases propuestas

    FA-2.0  Gate 0 (este informe) -> dictamen auditor
    FA-2.1  Filtro temporal + universo 503 -> commit local + mini-gate
    FA-2.2  CUSIP resolver + tabla excepciones -> commit local + mini-gate
    FA-2.3  Reporting relationships (grafo) -> commit local + mini-gate
    FA-2.3.bis  (condicional) Si el grafo descubre que amendments
                afectan a la atribucion -> reabrir diseno antes de FA-2.4
    FA-2.4  Amendments -> commit local + mini-gate
    FA-2.5  Probe real + informe Gate FA-2 -> dictamen auditor -> push

Cada mini-gate: tests + pyflakes + compileall + commit local.
Sin push hasta Gate FA-2. Regla local-first especifica IAE.

## 8. Preguntas al auditor

P1. Q-B: B.1, B.2 o B.3? (voto supervisor: B.1).

P2. Q-C: aprueba las 2 columnas anadidas (title_of_class, verified_by)
    y la ubicacion data/mappings/cusip_ticker_exceptions.csv?

P3. Q-D: aprueba el modelo multi-fila con vigencia temporal?

P4. Los 18 CUSIPs de diferencia entre etf_holdings.csv (519) y el
    universo IAE (501) requieren explicacion antes de FA-2.2. Origen
    probable: ETFs sectoriales excluidos, tickers fuera de radar_equities,
    o holdings sin ticker en market_data.parquet. Se propone abordarlo
    como primera tarea de FA-2.2.

P5. La semantica OTHERMANAGER vs OTHERMANAGER2 (N4) requiere validacion.
    En particular: la interpretacion "relacion inversa" y su no
    inclusion en la clave canonica (si se aprueba B.1).

## 9. Limpieza y estado

Probe: _probe_fa20.py borrado tras el reconocimiento. Sin side effects
en el repo. Parquets FA-1 intactos en D:/13f_probe/processed/2026Q1/.

Estado del repo al cierre de FA-2.0:

  HEAD local = origin/main = e2c3f02 (FOLLOWUPS IAE FA-1).
  Working tree limpio, sin untracked.
  727 tests + 2 skipped, 0 warnings.
  Sin cambios al CONTRATO v1.1, al pipeline ni a config/.

Este informe es documento puro. No hay cambios de codigo asociados.

---

## 10. Rev 2 - Correcciones del dictamen auditor (2026-09-19)

El dictamen externo (INSTITUTIONAL_ACCUMULATION_FA2_GATE0_DICTAMEN.md)
aprueba este informe con GO CONDICIONADO y las siguientes correcciones:

1. Renombrado semantico de la clave canonica de manager:
   - economic_owner_cik -> canonical_reporting_relationship_key.
   - Nombres PROHIBIDOS en codigo y documentacion hasta FA-2.4:
     economic_owner_cik, owner_cik, beneficial_owner_cik.
   - La clave representa una relacion de reporting/discretion,
     NO propiedad economica.

2. Orden arquitectonico FA-2.3 / FA-2.4 / FA-2.3.bis:
   - FA-2.3 atribucion PROVISIONAL.
   - FA-2.4 canonical snapshot del trimestre.
   - FA-2.3.bis atribucion DEFINITIVA (requisito, no condicional).
   FA-2.3 no puede declarar definitiva la atribucion hasta que FA-2.4
   haya determinado el snapshot canonico.

3. Q-A: campo contractual canonico = SUBMISSION.PERIODOFREPORT.
   REPORTCALENDARORQUARTER es control de coherencia, no sustituto.

4. Q-C controles anadidos: vigencia no solapada, source verificable,
   no inferencia silenciosa, title_of_class no decorativo.

5. Q-D control anadido: valid_from <= valid_to (cuando valid_to != NULL).

6. P4 (519 vs 501) pasa a requisito de FA-2.2. No bloquea FA-2.1.

7. Nota de cifras:
   - Dataset completo INFOTABLE: 3,822,885 filas.
   - Dataset filtrado Q1 2026:    3,321,967 filas.
   - Diferencia: 500,918 filas (filings fuera del periodo canonico).

---

Fin del informe. Version 1.0 (2026-09-19).
Referencia: prompt v6.34, HEAD e2c3f02.
Proximo paso autorizado: dictamen del auditor sobre Q-B, Q-C, Q-D y las
preguntas P1-P5. FA-2.1 no arranca sin dictamen.
