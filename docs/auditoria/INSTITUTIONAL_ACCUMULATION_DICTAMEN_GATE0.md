# DICTAMEN AUDITOR - GATE 0 IAE 13F

Version: 1.0 (2026-09-19)
Informe evaluado: INSTITUTIONAL_ACCUMULATION_GATE0_INFORME.md
Estado: vinculante

Nota tecnica: flechas unicode simplificadas a ASCII (->, <->) por
limitacion de transporte.

---

## Resultado global

GO condicionado para Fase A. NO-GO para publicar NIPC o
CONFIRMED_INSTITUTIONAL todavia.

Correccion principal: separar identidad de fila SEC de unidad
analitica de atribucion institucional.

---

## Q1 - K-INSTITUTIONAL-ACCUMULATION-01

GO -> MONITORED / BAJA. No se convierte en deuda activa mientras no
exista implementacion autorizada.

## Q2 - Clave canonica 13F

DECISION: NO aprueba la clave (ACCESSION_NUMBER, CUSIP, OTHERMANAGER,
INVESTMENTDISCRETION) como unidad canonica final.

La clave es util para normalizar e inspeccionar el dataset, pero no
debe convertirse en identidad analitica de una posicion.

Razon: OTHERMANAGER es una referencia LOCAL al filing. La SEC indica
que la columna 7 enlaza posiciones con otros managers mediante
identificador asignado en el propio filing, definidos en la lista
correspondiente. Ademas, la SEC permite combinar holdings propios y
de otros en COMBINATION REPORT, y un 13F NOTICE puede indicar que
todas las posiciones se reportan en otro filing.

La pregunta no es "como deduplico filas" sino "a que manager /
relacion de discrecion debo atribuir economicamente cada fila".

### Framework de identidad aprobado (3 niveles)

Nivel 1 - identidad fisica de fila:
  ACCESSION_NUMBER + INFO_TABLE_ROW_ID (o equivalente SEC exacto)
  Preserva procedencia de cada observacion original.

Nivel 2 - identidad de security:
  CUSIP + security class + report period
  No solo CUSIP aislado cuando la clase tenga relevancia.

Nivel 3 - relacion institucional:
  CANONICAL_MANAGER_RELATIONSHIP
  Resuelve:
    reporting manager
    <-> other included manager
    <-> other manager reporting for this manager
    <-> investment discretion

### Clave analitica final (solo despues)

(report_period, canonical_security, canonical_manager_relationship)
-> habilita DeltaShares, buyers, sellers, breadth, NIPC

### Dictamen Q2

NIPC permanece BLOQUEADO hasta que el resolver de relaciones haya
pasado su Gate.

## Q3 - CUSIP a ticker

Resultado: DESBLOQUEADO, con precision.

El 99.60% supera el umbral. GO.

Frase correcta: "Cobertura de mapping del universo actual 99.60%;
quedan excepciones que requieren tratamiento de identidad
historica/corporativa."

Los 5 casos (DD, HON, XOM, FDXF, HONA) confirman que una
correspondencia estatica ticker <-> CUSIP no es suficiente tras
corporate actions. La SEC define CUSIP y Title of Class como parte
formal de identificacion.

DECISION: no exigir 100% antes de Fase A. Mantener coverage >= 95%
con DOS controles:
  % securities mapeadas
  + % peso institucional no mapeado

99% en numero de valores seria insuficiente si el 1% concentra
parte material de NIPC.

## Q4 - SH + putCall=null

GO confirmado contractualmente.

La SEC distingue explicitamente SH (shares), PRN (principal),
Put, Call. Universo base de ownership long observable = SH + putCall
vacio. PRN, Put, Call -> fuera de DeltaShares.

## INVESTMENTDISCRETION = DFND

DECISION: INCLUIR, no excluir.

La SEC define DEFINED/DFND como shared-defined investment discretion
(parent/subsidiary, adviser/investment company, insurer/separate
account). La SEC exige segregar estas posiciones de determinadas
relaciones compartidas.

SOLE, DFND, OTR son tres modalidades de discrecion, no tres
categorias de calidad.

DFND entra en el calculo pero conserva su naturaleza:
  discretion_type = SOLE | DFND | OTR

Nunca debe convertirse DFND = otro manager independiente. La
atribucion dependera del resolver de relaciones.

Fase A debe resolver esta semantica junto con Q2.

## Q5 - Universo IAE

GO a 503, con condicion adicional.

Aprobado excluir los 36 ETFs del universo de equity accumulation:
  539 EQUITY - 36 ETF = 503

Razon: el modulo mide cambios de posiciones en empresas; el flujo
del ETF se analiza en el bloque ETF Primary Flow.

Contrato debe decir: "503 instrumentos actualmente clasificados como
equity individual dentro del universo del Radar, sujetos
posteriormente al filtro de elegibilidad 13F y resolucion de
security identity."

Section 13(f) es una lista regulatoria de valores elegibles; no todo
instrumento clasificado generico como EQUITY tiene por que
constituir common stock adecuado. Universo definitivo condicionado
a la clasificacion de security.

## Q6 - Almacenamiento

DECISION: A modificada -> GO.

No meter 3.3M filas/trimestre en Git.

Estructura aprobada:
  data/sec_13f/
    2026Q1.parquet        <- gitignored
  data/manifests/
    sec_13f_2026Q1.json   <- tracked

Manifest incluye: source_dataset, quarter, download_url,
downloaded_at, source_sha256, parquet_sha256, row_count,
filings_count, accession_min/max, schema_version.

accession_number es LINEAGE de SEC, no metadata decorativa.

FU-002: excepcion arquitectonica explicita por tamano de dataset.
Integridad preservada via SEC source + source hash + parquet hash +
accession lineage.

## Q7 - publication_date

Resuelto: filing_date real. Deadline regulatorio NO es fecha de
publicacion efectiva. SEC expone Filing Date y Accepted en EDGAR.

## Q8 - enmiendas

GO. Latest accepted amendment como snapshot canonico, lineage
completo conservado.

Gate 0 demostro que ISAMENDMENT=Y no puede usarse como sinonimo de
13F-HR/A. Clasificacion operativa por SUBMISSIONTYPE.

## Q9 - 13F vs N-PORT

Pendiente, no bloqueante para Fase A. Estados NOT_COMPARABLE /
COMPARABLE antes de cualquier discrepancy. Sin umbral todavia.

## Q10 - ETF representative

Fuera de CONFIRMED_INSTITUTIONAL. No debe bloquear Fase A.
Reservado para CONFIRMED_FLOW con mapping explicito.

## Q11 - seis meses

No. Backfill permitido. Condicion: 2 periodos 13F consecutivos
validos. No go-live + 6 meses.

## Q12 - doble clasificacion

GO. CONFIRMED_INSTITUTIONAL != CONFIRMED_FLOW.

## Q13 - Evidence quality HIGH

GO eliminado. No implementar sin definicion formal.

## Q14 - Tabla de estados

GO tabla obligatoria. ANADIR estado UNMAPPED como estado TECNICO de
cobertura (no clasificacion de acumulacion).

NO confundir UNMAPPED con NO_EVIDENCE. Una accion puede tener
evidencia institucional desconocida porque el CUSIP no pudo
resolverse; eso no significa que no exista evidencia.

## Q15 - Siguiente paso

GO a Fase A, con condicion nueva.

NO autorizar 13F core como descarga -> DeltaShares -> NIPC ->
CONFIRMED (prematuro).

Fase A autorizada = SEC 13F ingestion + identity/relationship
resolution, dividida en dos Gates:

FA-1: ingestion + schema + lineage
      -> Gate
FA-2: CUSIP + reporting relationships + amendments
      -> Gate

Solo despues: NIPC, Breadth, New/Exit y clasificacion.

Resuelve la contradiccion central: Q2 es bloqueante para el calculo,
pero no debe bloquear el trabajo necesario para resolver Q2.

## Q16 - Programa completo

NO-GO como programa de 6-10 semanas. Cadena de Gates:

13F -> Gate -> N-PORT -> Gate -> cross-validation -> Gate ->
sector aggregation -> Gate -> report

## 5 CUSIPs problematicos

DECISION: curacion manual inicial + monitorizacion trimestral.
No introducir proveedor comercial externo ahora.

Dato primario de identidad viene de SEC: CUSIP + Title of Class +
Issuer + Report Period. Mapping a ticker es capa secundaria.

Para DD, HON, XOM, FDXF, HONA: tabla de excepciones CON VIGENCIA
TEMPORAL:
  CUSIP | ticker | valid_from | valid_to | source | reason
No un simple CUSIP -> ticker.

OpenFIGI: fuente AUXILIAR, nunca autoridad unica (probo resolucion
incorrecta para XOM).

## Horizonte historico

NO conservar unicamente "5 anos" por accidente del tamano del primer
parquet. Decision de producto.

Para Fase A basta con Q1 2026 + trimestre anterior para validar
DeltaShares. NO autorizar 5+ anos en primer ciclo.

## Dictamen final consolidado

| Precondicion | Dictamen |
|---|---|
| Q2 relacion de managers | BLOQUEANTE para NIPC; resolver en Fase A |
| Q3 mapping | GO - 99.60% supera umbral |
| Q4 SH + null | GO |
| Q5 excluir ETFs | GO - universo provisional 503 |
| Q7 filing date | GO |
| Q8 amendments | GO |
| DFND | INCLUIR, resolver atribucion |
| almacenamiento | GO con excepcion FU-002 documentada |
| 5 CUSIPs | curacion manual + vigencia temporal |
| Q9 N-PORT | pendiente |
| Q10 ETF mapping | fuera de camino critico |

### Encargo autorizado

Fase A = GO condicionado.

13F Core = ingestion, normalizacion, lineage, clasificacion
SH/PRN/Put/Call, resolucion de amendments y construccion del grafo
de reporting relationships.

NO autoriza publicar NIPC, Institutional Breadth ni
CONFIRMED_INSTITUTIONAL.

Proximo Gate debe demostrar:
  3.3M filas brutas -> identity + relationships -> posicion
  institucional canonica -> sin doble conteo

Metricas requeridas: coverage, duplicate elimination, manager
attribution, DFND treatment, amendment resolution.

---

Fin del dictamen. Version 1.0 (2026-09-19).