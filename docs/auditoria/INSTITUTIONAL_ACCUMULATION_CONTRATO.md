# CONTRATO - Institutional Accumulation Evidence (IAE)

Version: 1.0 (2026-09-19)
Prompt de referencia: v6.32 (HEAD 7dea18c)
Documento de propuesta: docs/auditoria/INSTITUTIONAL_ACCUMULATION_PROPUESTA.md
Dictamen auditor: vinculante (16 decisiones Q1-Q16)
Estado: contrato provisional, previo al Gate 0 empirico
Naturaleza: no normativo. Si hay conflicto, gana el Prompt Maestro.

---

## 1. Proposito

Cerrar el contrato de datos del modulo IAE antes de escribir codigo.
Define: identidades, fuentes, temporalidad, clasificacion, criterios
de Gate 0, roadmap por fases. Nada se implementa sin superar el Gate 0.

## 2. Fuentes y jerarquia

| Nivel | Fuente | Endpoint oficial | Uso |
|---|---|---|---|
| 1 | SEC 13F Data Set | www.sec.gov/files/structureddata/data/form-13f-data-sets/ | Evidencia institucional directa |
| 1 | SEC N-PORT | www.sec.gov/data-research/sec-markets-data/form-n-port-data-sets | Corroboracion (no suma) |
| 2 | ETF sponsor (SSGA/BlackRock/Amundi) | ya implementado | Flujo primario |
| 3 | CFTC TFF | publicreporting.cftc.gov | Contexto derivados |
| 4 | FINRA ATS | www.finra.org/filing-reporting/otc-transparency | Contexto microestructura |

## 3. Dataset del Gate 0

Trimestre: 2026 Q1 (periodo reportado 2026-03-31).
Ventana de publicacion: 01-mar-2026 a 31-may-2026.
ZIP: 01mar2026-31may2026_form13f.zip (99.4 MB).
URL: https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01mar2026-31may2026_form13f.zip
User-Agent obligatorio: "Radar Sectorial Research <bledabladis@gmail.com>".
Contenido: SUBMISSION.tsv, COVERPAGE.tsv, OTHERMANAGER.tsv,
OTHERMANAGER2.tsv, SIGNATURE.tsv, SUMMARYPAGE.tsv, INFOTABLE.tsv.

## 4. Decisiones vinculantes del auditor (Q1-Q8)

### Q1 - Registro del K-ID
K-INSTITUTIONAL-ACCUMULATION-01 -> MONITORED/BAJA. Deuda activa 0/0/0.

### Q2 - Deduplicacion de reporting relationships
PRECONDICION BLOQUEANTE.
Unidad canonica = relacion de reporte (filing -> reporting manager ->
other included managers -> other managers reporting for this manager).
NO aprobada "deduplicacion por nombre de grupo" (BlackRock/Vanguard/SSGA).
NO acepta "correccion estimada" de doble contabilizacion.
Sin resolucion -> NIPC no se publica.

### Q3 - CUSIP -> ticker
GATE 0 OBLIGATORIO. Umbral >=95% de cobertura de mapping del universo
elegible. Medir tambien: unmapped_count, unmapped_weight,
unmapped_by_sector, unmapped_top20.

### Q4 - DeltaShares canonico
sshPrnamtType == "SH" AND putCall is null.
Excluye: PRN, Put, Call. Market_value NO se usa para acumulacion.

### Q5 - Universo canonico
radar_equities AND section_13f_eligible AND mapped_security.
NO simple "539 equity USA".

### Q6 - Almacenamiento
data/sec_13f/ gitignored.
+ manifest por trimestre con accession_number (lineage primario),
  CIK, report_period, filing_date, sha256, source=EDGAR.
Excepcion explicita a FU-002 (artefacto grande no en Git).
NO: cache sin lineage. NO: guardar todo bruto en Git.

### Q7 - publication_date
= filing_date real (SEC acceptance date).
NO deadline + 1 dia.

### Q8 - Politica de enmiendas
Ultima 13F-HR/A aceptada = snapshot canonico.
Chain original_filing -> amendment_chain -> canonical_filing preservado.
NO tratar enmiendas como observaciones adicionales.

## 5. Decisiones vinculantes del auditor (Q9-Q16)

### Q9 - Contradiccion 13F <-> N-PORT
Sin umbral global. Primero comparability gate:
same manager + same fund + same security + same economic scope.
Estados: COMPARABLE | NOT_COMPARABLE.
Discrepancia solo computable si COMPARABLE.

### Q10 - ETF representativo
NO requisito de CONFIRMED_INSTITUTIONAL.
Mapping stock->ETF explicito y curado, NO inferido dinamicamente.

### Q11 - Primeros 6 meses
NO espera artificial. Backfill historico permitido.
CONFIRMED_INSTITUTIONAL requiere 2 periodos 13F consecutivos validos
en dataset, no 2 trimestres desde go-live.

### Q12 - Separacion de confirmaciones
CONFIRMED_INSTITUTIONAL = 13F + breadth + coverage + persistencia.
CONFIRMED_FLOW = CONFIRMED_INSTITUTIONAL + ETF Primary valido.
N-PORT y CFTC = corroboracion cuando comparables. No requisitos.

### Q13 - Evidence quality HIGH
ELIMINADA. Solo la matriz de evidencia, sin etiquetas narrativas.

### Q14 - Tabla de transicion
Ver seccion 6.

### Q15 - Siguiente paso
Opcion C: contrato + Gate 0 sobre 100 filings reales. GO.
Gate 0 debe producir numeros reales:
% mapeado, % duplicado, % SH, % PRN, % options,
# combination reports, # amendments, # unresolved securities, storage size.

### Q16 - Implementacion
Cadena de ciclos independientes: A (13F core) -> B (N-PORT) ->
C (cross-validation) -> D (sector aggregation) -> E (reporte).
CFTC y FINRA fuera del camino critico inicial.

## 6. Tabla de transicion de estados

| Estado | Condicion |
|---|---|
| NO_EVIDENCE | cobertura suficiente + sin senal positiva observable |
| INSUFFICIENT | cobertura o temporalidad insuficiente para concluir |
| COMPATIBLE | evidencia positiva pero no cumple confirmacion |
| CONFIRMED_INSTITUTIONAL | 13F + breadth>50% + coverage>=60% + 2 periodos consecutivos + N-PORT sin contradiccion material |
| CONFIRMED_FLOW | CONFIRMED_INSTITUTIONAL + ETF Primary positivo + mapping valido |

Frase literal para NO_EVIDENCE:
"El dataset no muestra evidencia positiva suficiente en las variables
observadas." NO significa "instituciones no compran".

## 7. Criterios de seleccion de los 100 filings (C.0)

Estratificacion determinista con semilla fija (semilla = hash del
report_period, para reproducibilidad).

| Estrato | Criterio | N | Motivo |
|---|---|---|---|
| A - Top AUM | Top 50 por value total (SUMMARYPAGE) | 50 | Concentracion de mercado |
| B - Medio | Siguientes 200, aleatorio | 30 | Mayoría de managers |
| C - Pequenos | AUM < $1B, aleatorio | 10 | Casos limite |
| D - Especiales | 13F-HR/A + Combination Report | 10 | Deduplicacion |

Total = 100 filings.

Ademas, analisis agregado sobre el universo completo (no muestra):
  - Total de filings en el trimestre
  - Distribucion por tipo (HR, HR/A, Combination, Notice)
  - Distribucion SH/PRN/Put/Call
  - # combination reports, # amendments
  - Storage size esperado

## 8. Protocolo del Gate 0 empirico

### C.2 - Descarga y extraccion
- Descargar ZIP a $env:TEMP\13f_q1_2026\ (fuera del repo).
- Extraer.
- Verificar integridad (7 TSVs presentes, count por fichero).

### C.3 - Analisis empirico
Responder con numeros reales:
  1. Volumen: filas por TSV, storage descomprimido.
  2. Tipos: distribucion HR/HR-A/Combination.
  3. Jerarquias: cuantos filings con OTHERMANAGER != vacio.
  4. Duplicados: managers con >1 CIK activos en el trimestre.
  5. SH/PRN: % de filas por tipo de sshPrnamt.
  6. Put/Call: % de filas con putCall != null.
  7. CUSIP: cobertura de mapping contra universo radar_equities.
  8. Enmiendas: cuantos filings tienen chain >=2.
  9. Errores: filings con XML invalido o campos nulos.

### C.4 - Informe Gate 0
Documento con resultados + recomendacion GO/NO-GO para implementacion
de la Fase A del roadmap (13F core).

## 9. Condiciones de cierre

K-INSTITUTIONAL-ACCUMULATION-01 permanece MONITORED/BAJA.
Gate 0 exitoso -> ciclo posterior de implementacion por fases (Q16).
Gate 0 fallido -> WONT FIX documentado.

Sin push a main durante Gate 0. Solo docs/auditoria/ y outputs/audit/.

---

Fin del contrato. Version 1.0 (2026-09-19).
Referencia: Prompt Maestro v6.32 (HEAD 7dea18c).
Proximo paso: Fase C.2 (descarga + extraccion).