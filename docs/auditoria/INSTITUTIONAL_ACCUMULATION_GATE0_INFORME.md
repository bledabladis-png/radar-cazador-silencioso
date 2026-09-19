# INFORME GATE 0 - Institutional Accumulation Evidence (IAE)

Version: 1.0 (2026-09-19)
Prompt de referencia: v6.32 (HEAD 7dea18c)
Contrato previo: docs/auditoria/INSTITUTIONAL_ACCUMULATION_CONTRATO.md
Propuesta: docs/auditoria/INSTITUTIONAL_ACCUMULATION_PROPUESTA.md
Dataset analizado: 13F Q1 2026 (SEC Data Set, publicado 2026-06-04)
Estado: ciclo C cerrado. Pendiente dictamen auditor sobre reformulaciones Q2 y Q5.

---

## 1. Proposito

Cerrar el Gate 0 empirico autorizado por el auditor (Q15-Q16) sobre
100+ filings 13F reales, antes de escribir codigo del modulo IAE.

Producir numeros reales que validen o refuten la viabilidad del modulo,
y detectar bloqueantes no anticipados en el diseno conceptual.

NO produce codigo. NO modifica el pipeline. Solo evidencia empirica.

## 2. Metodologia

Dataset: SEC Form 13F Data Set 2026 Q1.
ZIP: 01mar2026-31may2026_form13f.zip (99.4 MB).
Ventana filing: 01-mar-2026 a 31-may-2026.
Periodo reportado: 31-MAR-2026.
Extraccion: 383 MB (7 TSVs + readme + metadata).
User-Agent: Radar Sectorial Research <bledabladis@gmail.com>.
Almacenamiento temporal: D:\13f_q1_2026\ (fuera del repo).

Analisis en 3 pasos:
  Paso 1 - Perfil de los 5 TSVs pequenos (estructura de managers).
  Paso 2 - Analisis INFOTABLE con filtro PERIODOFREPORT=31-MAR-2026.
  Paso 3 - Mapping CUSIP->ticker contra universo radar.

## 3. Paso 1 - Perfil de filings y managers

### 3.1. Volumen del dataset
| TSV | Filas |
|---|---|
| SUBMISSION | 11,761 |
| COVERPAGE | 11,761 |
| SUMMARYPAGE | 9,846 |
| OTHERMANAGER | 4,751 |
| OTHERMANAGER2 | 4,242 |
| SIGNATURE | 11,761 |

### 3.2. Filtro temporal obligatorio

PERIODOFREPORT del ZIP contiene valores desde 2008 hasta 2026.
El ZIP agrupa filings por FILING_DATE, no por periodo reportado.
La mayoria de filings tienen PERIODOFREPORT=31-MAR-2026 pero hay
tardios de periodos historicos.

Filtrado a Q1 2026 real:
  SUBMISSION Q1 2026: 10,776 filings (vs 11,761 sin filtro, -8.4%)
  CIKs unicos: 10,648

### 3.3. Distribucion por tipo de filing (Q1 2026)
| SUBMISSIONTYPE | Q1 2026 |
|---|---|
| 13F-HR | 8,741 |
| 13F-NT (notice, sin holdings) | 1,907 |
| 13F-HR/A (enmienda) | 127 |
| 13F-NT/A | 1 |

### 3.4. REPORTTYPE (Q1 2026)
| REPORTTYPE | Q1 2026 |
|---|---|
| 13F HOLDINGS REPORT | 8,457 |
| 13F NOTICE | 1,908 |
| 13F COMBINATION REPORT | 411 |

### 3.5. Hallazgo: ISAMENDMENT es enganoso

El campo COVERPAGE.ISAMENDMENT=Y marca 441 filings (sin filtro),
pero las enmiendas reales (SUBMISSIONTYPE=13F-HR/A) son 127 en
Q1 2026. La coincidencia 441 con 441 combination reports del
dataset completo indica que ISAMENDMENT=Y incluye COMBINATION,
no solo restatements. Q8 se resuelve por SUBMISSIONTYPE.

### 3.6. Deduplicacion: hallazgo sorprendente

Union de CIKs (SUBMISSION + OTHERMANAGER + OTHERMANAGER2):
  CIKs en >1 filing historico: 3,110
  CIKs en >1 filing Q1 2026:   124

El "problema de deduplicacion" no es CIK duplication (124 casos).
El problema real es OTHERMANAGER, que afecta a 44% de las filas
(ver Paso 2).

## 4. Paso 2 - Analisis INFOTABLE

### 4.1. Volumen
INFOTABLE.tsv total: 3,822,885 filas.
Retenidas (Q1 2026): 3,321,967 filas (86.9%).
CUSIPs unicos: 33,449.
Filings con holdings: 8,856.

### 4.2. Q4 RESUELTO - filtro canonico DeltaShares

| Categoria | Filas | % |
|---|---|---|
| SH (shares) | 3,306,269 | 99.53% |
| PRN (principal) | 15,698 | 0.47% |
| Put | 57,443 | 1.73% |
| Call | 61,400 | 1.85% |
| Canonico (SH + putCall=null) | 3,188,083 | 95.97% |

El filtro Q4 (SH + putCall=null) preserva 95.97% del INFOTABLE.
Pierde <4.1%. Validacion empirica de la decision del auditor.

### 4.3. Q2 REFORMULADO - el problema es OTHERMANAGER

| Metrica | Valor | % |
|---|---|---|
| Filas con OTHERMANAGER != vacio | 1,465,620 | 44.12% |
| (ACCESSION, CUSIP) con >1 fila | 344,344 | 10.36% |
| 13F COMBINATION REPORT (Q1 2026) | 411 | - |
| CIKs con >1 filing (Q1 2026) | 124 | - |

CONCLUSION: el problema de deduplicacion NO es CIK duplicado.
Es que el 44% de las filas reportan posiciones para "other managers"
dentro del mismo filing. Sumar filas sin resolver esta dimension
duplica posiciones dentro del mismo accession.

La unidad canonica debe ser (ACCESSION_NUMBER, CUSIP, OTHERMANAGER,
INVESTMENTDISCRETION) o similar. Sin resolver Q2, NIPC no se publica.

Ejemplo: el Top 1 CUSIP del INFOTABLE (NVDA 67066G104) acumula
$3.14T en agregado simple, imposible economicamente. Confirmacion
del double counting.

### 4.4. INVESTMENTDISCRETION
| Valor | Filas | % |
|---|---|---|
| SOLE | 2,123,708 | 63.93% |
| DFND (defined) | 964,983 | 29.05% |
| OTR (other) | 233,276 | 7.02% |

Las posiciones DFND (definidas por mandato) probablemente requieren
tratamiento diferenciado. Pendiente de decision en Fase A.

### 4.5. Q3 REFORMULADO - FIGI NO es la via

CUSIP presente: 100.00% (siempre).
FIGI presente: 12.20% (405,352 de 3,321,967).

La via FIGI como mapping primario esta descartada. El mapping
depende de CUSIP->ticker externo (ver Paso 3).

## 5. Paso 3 - Mapping CUSIP->ticker contra universo radar

### 5.1. Universo radar real

market_data.parquet tiene 562 tickers:
  EQUITY: 539
  FUTURE: 5
  FX: 3
  INDEX: 10
  RATE_YIELD: 2
  VOLATILITY_INDEX: 3

De los 539 EQUITY, 36 son ETFs (instrumentos sectoriales XL*,
indices SPY/QQQ/IWM, etc.). Los ETFs no son holdings individuales
y su inclusion en IAE crea confusion semantica con ETF Primary Flow.

Universo IAE ajustado: 539 - 36 = 503 holdings individuales.

### 5.2. Cobertura CUSIP local (etf_holdings.csv)

etf_holdings.csv tiene 519 CUSIPs (11 ETFs XL*).
  Tickers con CUSIP local: 501/503 = 99.60%
  Sin CUSIP local: 2 (BF-B, BRK-B, clases B)

Veredicto Q3: cobertura 99.60% >= 95.00% -> DESBLOQUEADO.

### 5.3. Interseccion con 13F Q1 2026

CUSIPs locales: 501.
En 13F Q1 2026: 496 (99.0%).
NO en 13F: 5 (DD, FDXF, HON, HONA, XOM).

Analisis de los 5:
  DD (26614N201): variantes en 13F (26614N902, 26614N102, 26614N952).
    CUSIP local desactualizado. Reestructuracion DuPont.
  HON (438516205): variantes (438516906, ...). Posible spin-off Honeywell.
  XOM (30233Q108): ninguna variante similar. CUSIP local
    desactualizado vs CUSIP 13F vigente.
  FDXF, HONA: sin variantes. Tickers recientes o clases especiales.

Causa raiz: etf_holdings.csv proviene de SSGA (trimestral), y tras
eventos corporativos el CUSIP cambia mientras el 13F usa el vigente.

### 5.4. OpenFIGI - utilidad con gaps

Test con 10 CUSIPs (5 paso 1, 5 paso 3):
  9/10 devuelven ticker correcto con exchCode=US.
  1/10 (XOM 30231G102) devuelve EXMOC (Commercial Paper) en
    lugar de XOM. Ni con marketSector=Equity ni securityType=Common
    Stock se resuelve.

Conclusion: OpenFIGI es util para bulk (90%+), pero requiere
validacion cruzada. NO es fuente autoritativa unica.

## 6. Reformulaciones propuestas al auditor

### 6.1. Q2 - unidad canonica de reporte

Vigente: "unidad = relacion de reporte (filing -> reporting manager
-> other included managers -> other managers reporting)".

REFORMULACION: la clave practica es (ACCESSION_NUMBER, CUSIP,
OTHERMANAGER, INVESTMENTDISCRETION). No es "1 filing = 1 posicion
agregable". El 44% de filas con OTHERMANAGER confirma que hay
multiples atribuciones dentro del mismo filing.

Solicitud: validar esta clave.

### 6.2. Q5 - universo IAE elegible

Vigente: "radar_equities AND section_13f_eligible AND mapped".

REFORMULACION: excluir ETFs (instrumentos) del universo IAE.
Denominador pasa de 539 a 503.

Razon: los ETFs no son holdings individuales; su tratamiento
pertenece a ETF Primary Flow, no a IAE.

Solicitud: aprobar la exclusion.

## 7. Recomendacion GO/NO-GO

### 7.1. Precondiciones bloqueantes - estado

| Precondicion | Estado |
|---|---|
| Q2 deduplicacion reporting relationships | PARCIAL. Reformulacion propuesta. |
| Q3 CUSIP mapping >=95% | CUMPLIDO (99.60%). |
| Q7 publication_date = filing_date | CUMPLIDO (SUBMISSION.FILING_DATE). |
| Q8 enmiendas | CUMPLIDO (127 reales, no 441). |
| Q9 comparability gate 13F<->N-PORT | PENDIENTE (no bloqueante Fase A). |
| Q10 mapping stock->ETF | PENDIENTE (no bloqueante CONFIRMED_INSTITUTIONAL). |

### 7.2. Recomendacion

GO CONDICIONADO para Fase A (13F core), condicionado a:
  1. Validacion del auditor de la reformulacion Q2 (unidad canonica).
  2. Validacion del auditor de la reformulacion Q5 (excluir ETFs).
  3. Resolucion iterativa de Q2 dentro de Fase A (no antes).

Q3 desbloqueado. Q4 validado. Q7, Q8 cumplidos.

### 7.3. Riesgos residuales

- 5 CUSIPs locales desactualizados (DD, HON, XOM, FDXF, HONA).
  Coste: curacion manual + monitorizacion trimestral.
- OpenFIGI tiene gaps (10% XOM). Requiere validacion cruzada.
- INVESTMENTDISCRETION DFND (29%) requiere decision de tratamiento.
- Storage: 3.3M filas parquet por trimestre. Excepcion a FU-002.

### 7.4. Beneficio confirmado

- Dataset disponible, descargable, forma parte del dominio publico SEC.
- Cobertura del universo IAE: 99.6% con datos locales.
- Filtro canonico Q4 preserva 95.97%.
- Backfill historico posible (ZIPs publicos desde 2013).

## 8. Proximos pasos

1. Enviar informe al auditor externo (dictamen vinculante).
2. Esperar validacion de reformulaciones Q2 y Q5.
3. Si GO: iniciar Fase A del roadmap (13F core) con Q2 resuelto
   iterativamente.
4. Si NO-GO por Q2: registrar WONT FIX documentado y cerrar
   K-INSTITUTIONAL-ACCUMULATION-01.

---

Fin del informe. Version 1.0 (2026-09-19).
Referencia: Prompt Maestro v6.32 (HEAD 7dea18c).
Dataset: 13F Q1 2026 (SEC).