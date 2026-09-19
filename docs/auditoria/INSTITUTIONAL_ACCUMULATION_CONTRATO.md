# CONTRATO - Institutional Accumulation Evidence (IAE)

Version: 1.1 (2026-09-19)
Prompt de referencia: v6.33
Propuesta: docs/auditoria/INSTITUTIONAL_ACCUMULATION_PROPUESTA.md
Dictamen vinculante: docs/auditoria/INSTITUTIONAL_ACCUMULATION_DICTAMEN_GATE0.md
Informe Gate 0: docs/auditoria/INSTITUTIONAL_ACCUMULATION_GATE0_INFORME.md
Estado: GO condicionado Fase A. NIPC BLOQUEADO hasta Gate FA-2.
Naturaleza: no normativo. Si hay conflicto, gana el Prompt Maestro.

---

## 1. Proposito

Contrato de datos del modulo IAE. Define framework de identidad,
fuentes, temporalidad, clasificacion, criterios de Gate 0, roadmap
por fases. Nada se implementa sin superar el Gate correspondiente.

Cambio v1.1: incorpora dictamen del auditor sobre Gate 0 empirico
(2026-09-19). Reemplaza Q2 (framework 3 niveles), Q5 (503 + eligibility),
Q6 (storage A modificada), Q14 (UNMAPPED), Q15 (FA-1+FA-2).

## 2. Fuentes y jerarquia

| Nivel | Fuente | Uso |
|---|---|---|
| 1 | SEC 13F Data Set | Evidencia institucional directa |
| 1 | SEC N-PORT | Corroboracion (no suma) |
| 2 | ETF sponsor (SSGA/BlackRock/Amundi) | Flujo primario |
| 3 | CFTC TFF | Contexto derivados |
| 4 | FINRA ATS | Contexto microestructura |

Fuentes auxiliares (no autoritarias):
- OpenFIGI: mapping CUSIP->ticker, requiere validacion cruzada.
- Tabla de excepciones curada con vigencia temporal.

## 3. Framework de identidad (Q2 reformulado)

DECISION del auditor: NO aprobada la clave plana
(ACCESSION_NUMBER, CUSIP, OTHERMANAGER, INVESTMENTDISCRETION) como
unidad canonica final. Util como normalizacion intermedia, no como
identidad analitica.

Razon: OTHERMANAGER es referencia LOCAL al filing. La pregunta no es
"como deduplico filas" sino "a que manager / relacion de discrecion
debo atribuir economicamente cada fila".

### 3.1. Nivel 1 - Identidad fisica de fila

    ACCESSION_NUMBER + INFO_TABLE_ROW_ID

o el equivalente exacto que proporcione el dataset SEC.
Preserva procedencia de cada observacion original.

### 3.2. Nivel 2 - Identidad de security

    CUSIP + security_class + report_period

No solo CUSIP aislado cuando la clase tenga relevancia (Title of
Class es parte formal de la identificacion SEC).

### 3.3. Nivel 3 - Relacion institucional

    CANONICAL_MANAGER_RELATIONSHIP

Resuelve el grafo:
    reporting_manager
    <-> other_included_manager
    <-> other_manager_reporting_for_this_manager
    <-> investment_discretion

Preserva discretion_type = SOLE | DFND | OTR sin colapsarlos.

### 3.4. Clave analitica final (solo tras FA-2)

    (report_period,
     canonical_security,
     canonical_manager_relationship)

Habilita: DeltaShares, buyers, sellers, breadth, NIPC.

NIPC permanece BLOQUEADO hasta que el resolver de relaciones haya
pasado su Gate.

## 4. Decisiones vinculantes (Q1-Q8)

### Q1 - Registro
K-INSTITUTIONAL-ACCUMULATION-01 -> MONITORED/BAJA. No deuda activa.

### Q2 - Unidad canonica
Ver seccion 3. Framework 3 niveles. NIPC bloqueado hasta Gate FA-2.

### Q3 - CUSIP -> ticker
GO. Cobertura 99.60% del universo actual (501/503).
Quedan 5 excepciones (DD, HON, XOM, FDXF, HONA) con tratamiento
de identidad historica/corporativa.

DOS controles obligatorios en cada medicion:
  % securities mapeadas
  + % peso institucional no mapeado

99% en numero de valores seria insuficiente si el 1% concentra
parte material de NIPC.

### Q4 - DeltaShares canonico
GO. sshPrnamtType == "SH" AND putCall is null.
Excluye PRN, Put, Call. Market_value NO se usa para acumulacion.

### Q5 - Universo IAE
GO a 503 instrumentos, con condicion adicional.

Razon: 539 EQUITY - 36 ETFs (instrumentos sectoriales/globales) = 503.

Contrato: "503 instrumentos actualmente clasificados como equity
individual dentro del universo del Radar, sujetos posteriormente al
filtro de elegibilidad 13F (Section 13(f)) y resolucion de security
identity."

Universo definitivo condicionado a la clasificacion de security.

### Q6 - Almacenamiento
GO con estructura A modificada.

    data/sec_13f/
      2026Q1.parquet        <- gitignored
    data/manifests/
      sec_13f_2026Q1.json   <- tracked

Manifest contiene: source_dataset, quarter, download_url,
downloaded_at, source_sha256, parquet_sha256, row_count,
filings_count, accession_min/max, schema_version.

accession_number es LINEAGE SEC, no metadata decorativa.

Excepcion arquitectonica explicita a FU-002 por tamano de dataset.
Integridad via SEC source + source hash + parquet hash + accession.

### Q7 - publication_date
GO. filing_date real (SEC acceptance date). No deadline + 1.

### Q8 - Enmiendas
GO. Latest accepted amendment = snapshot canonico.
Lineage completo conservado.
Clasificacion por SUBMISSIONTYPE (13F-HR, 13F-HR/A).
NO por ISAMENDMENT (incluye COMBINATION, enganoso).

## 5. Decisiones vinculantes (Q9-Q16)

### Q9 - 13F vs N-PORT
Pendiente, no bloqueante Fase A.
Estados NOT_COMPARABLE / COMPARABLE antes de cualquier discrepancy.
Sin umbral numerico.

### Q10 - ETF representativo
Fuera de CONFIRMED_INSTITUTIONAL. No bloquea Fase A.
Reservado para CONFIRMED_FLOW con mapping explicito.

### Q11 - Seis meses
NO esperar. Backfill permitido.
Condicion: 2 periodos 13F consecutivos validos.

### Q12 - Doble clasificacion
GO. CONFIRMED_INSTITUTIONAL != CONFIRMED_FLOW.

### Q13 - Evidence quality HIGH
Eliminada. No se implementa sin definicion formal.

### Q14 - Tabla de transicion
Anadir estado tecnico UNMAPPED:

| Estado | Condicion |
|---|---|
| UNMAPPED | CUSIP no resuelto. Estado TECNICO, no clasificacion |
| NO_EVIDENCE | Cobertura suficiente + sin senal positiva |
| INSUFFICIENT | Cobertura o temporalidad insuficiente |
| COMPATIBLE | Evidencia positiva, no cumple confirmacion |
| CONFIRMED_INSTITUTIONAL | 13F + breadth + coverage + 2 periodos + N-PORT sin contradiccion |
| CONFIRMED_FLOW | CONFIRMED_INSTITUTIONAL + ETF Primary + mapping |

UNMAPPED NO es NO_EVIDENCE. Una accion puede tener evidencia
institucional desconocida porque el CUSIP no pudo resolverse.

### Q15 - Fase A autorizada
GO condicionado. Dividida en dos Gates:

FA-1: ingestion + schema + lineage
      -> Gate FA-1

FA-2: CUSIP + reporting relationships + amendments
      -> Gate FA-2

Solo tras FA-2: NIPC, Breadth, New/Exit, clasificacion.

### Q16 - Programa completo
NO-GO como bloque unico. Cadena:
13F -> Gate -> N-PORT -> Gate -> cross-validation -> Gate ->
sector aggregation -> Gate -> report.

## 6. INVESTMENTDISCRETION - tratamiento

SOLE, DFND, OTR son tres modalidades de discrecion, no tres
categorias de calidad.

DFND (defined): shared-defined investment discretion
(parent/subsidiary, adviser/investment company, insurer/separate).
La SEC exige segregar estas posiciones de determinadas relaciones.

DECISION: INCLUIR en el calculo pero conservar la naturaleza:
  discretion_type = SOLE | DFND | OTR

Nunca colapsar DFND = otro manager independiente.
Atribucion via resolver de relaciones (Nivel 3 del framework Q2).

Fase A (FA-2) debe resolver esta semantica junto con Q2.

## 7. Curacion de CUSIPs - vigencia temporal

Para los 5 CUSIPs problematicos (DD, HON, XOM, FDXF, HONA) y
futuros: tabla de excepciones CON VIGENCIA TEMPORAL.

    CUSIP | ticker | valid_from | valid_to | source | reason

NO un simple CUSIP -> ticker estatico. Corporate actions cambian
CUSIP mientras 13F usa el vigente.

Fuente primaria: SEC (CUSIP + Title of Class + Issuer + Report
Period). Mapping a ticker es capa secundaria del Radar.

OpenFIGI: fuente AUXILIAR, nunca autoridad unica (probo resolucion
incorrecta para XOM).

Monitorizacion trimestral obligatoria tras cada publicacion SEC.

## 8. Roadmap por fases

FASE A: 13F Core (GO condicionado)
  FA-1: ingestion + schema + lineage
        Entregables: ingestor, schema, manifest, integridad.
        Gate: verificar lineage SEC + integridad hash + schema.
  FA-2: CUSIP + reporting relationships + amendments
        Entregables: resolver de relaciones (grafo), tabla curada
        CUSIP, mapeo SH/PRN/DFND, tratamiento de enmiendas.
        Gate: demostrar 3.3M filas -> identidad canonica sin doble
        conteo. Metricas: coverage, duplicate elimination, manager
        attribution, DFND treatment, amendment resolution.

FASE B: N-PORT -> Gate
FASE C: cross-validation -> Gate
FASE D: sector aggregation -> Gate
FASE E: reporte -> Gate

CFTC y FINRA fuera del camino critico inicial.

## 9. Horizonte historico

Decision de producto. NO conservar solo "5 anos" por accidente.

Fase A: Q1 2026 + trimestre anterior suficientes para validar
DeltaShares. NO descargar 5+ anos en primer ciclo.

## 10. Condiciones de cierre

K-INSTITUTIONAL-ACCUMULATION-01 permanece MONITORED/BAJA.
NIPC bloqueado hasta Gate FA-2 superado.
Si FA-2 falla -> WONT FIX documentado.
Sin push a main de codigo hasta validar funcionalidad + beneficio.

## 11. Dataset del Gate 0 empirico

Trimestre: 2026 Q1 (PERIODOFREPORT=31-MAR-2026).
ZIP: 01mar2026-31may2026_form13f.zip (99.4 MB).
URL: sec.gov/files/structureddata/data/form-13f-data-sets/.
User-Agent: "Radar Sectorial Research <bledabladis@gmail.com>".
Extraido: 383 MB (7 TSVs + readme + metadata).
Almacenamiento temporal: D:\13f_q1_2026\ (fuera del repo).

---

Fin del contrato. Version 1.1 (2026-09-19).
Referencia: Prompt Maestro v6.33.
Proximo paso autorizado: Fase A - FA-1 (ingestion + schema + lineage).