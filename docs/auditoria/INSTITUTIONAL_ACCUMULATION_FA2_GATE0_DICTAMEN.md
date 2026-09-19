# DICTAMEN AUDITOR - FA-2.0 GATE 0 IAE

Version: 1.0 (2026-09-19)
Informe evaluado: docs/auditoria/INSTITUTIONAL_ACCUMULATION_FA2_GATE0_INFORME.md (commit 5162cb7 local)
Origin: e2c3f02
Estado: vinculante. GO CONDICIONADO para FA-2.1.
Naturaleza: no normativo. Si hay conflicto, gana el Prompt Maestro.

---

## 0. Veredicto general

FA-1 APROBADA Y CERRADA. No se reabre.

FA-2.0 GO CONDICIONADO. La arquitectura no necesita rehacerse. El
siguiente cambio de ingenieria autorizado es FA-2.1. Ajustes de
diseno obligatorios antes de FA-2.2/FA-2.3.

Base documental: SEC Form 13F Data Sets son as-filed, incluyen
enmiendas, pueden contener redundancias. SUBMISSIONTYPE, PERIODOFREPORT,
OTHERMANAGER, OTHERMANAGER2 e INFOTABLE tienen las funciones descritas
en el informe FA-2.0.

## 1. FA-1 - Aprobada

Evidencia suficiente para cerrar FA-1:
  - 7/7 tablas con row counts identicos al probe de referencia.
  - Procesamiento del dataset completo, sin logica de negocio prematura.
  - Lineage ZIP -> TSV -> Parquet.
  - Hashes en los tres niveles.
  - Tests especificos + suite global.
  - Parser robusto con errors="coerce".
  - Separacion correcta entre ingestion pasiva y decisiones analiticas.

Claves oficiales confirmadas: ACCESSION_NUMBER;
(ACCESSION_NUMBER, OTHERMANAGER_SK) para OTHERMANAGER;
(ACCESSION_NUMBER, INFOTABLE_SK) para INFOTABLE.

## 2. P1 - Q-B: clave canonica de manager

DICTAMEN: B.1 GO, con correccion semantica obligatoria.

Estructura aprobada:

    (primary_cik, submanager_cik|null, discretion_type, period)

PERO no debe llamarse economic_owner ni tratarse como prueba de
propiedad economica.

Nombres OBLIGATORIOS: canonical_reporting_relationship_key.

Nombres PROHIBIDOS en codigo y documentacion hasta FA-2.4:
  - economic_owner_cik
  - owner_cik
  - beneficial_owner_cik

Razon: la 13F identifica al manager que presenta el informe y las
relaciones de managers para los que se reportan posiciones. Eso NO
convierte al CIK resultante en propietario economico definitivo.

## 3. P5 - OTHERMANAGER vs OTHERMANAGER2

DICTAMEN: VALIDADO, con precision terminologica.

SEC:
  - OTHERMANAGER  = otros managers reporting for this manager.
  - OTHERMANAGER2 = otros managers included in this report.

INFOTABLE.OTHERMANAGER = secuencia del other manager incluido con
quien se comparte discrecion.

Por tanto:
  OTHERMANAGER  -> relacion que afecta a la atribucion de la linea.
  OTHERMANAGER2 -> provenance / relacion inversa.

OTHERMANAGER2 es relevante para reconstruir el grafo, pero NO debe
entrar automaticamente en la clave de atribucion de una linea.

P5 = GO.

## 4. P2 - Q-C: tabla de excepciones CUSIP

DICTAMEN: GO CONDICIONADO.

Columnas aprobadas:
  CUSIP, ticker, valid_from, valid_to, source, reason,
  title_of_class, verified_by.

Ubicacion aprobada: data/mappings/cusip_ticker_exceptions.csv.

Cuatro controles obligatorios:

A. Vigencia temporal: imposible tener dos filas activas simultaneamente
   para CUSIP + ticker si los intervalos se solapan.

B. Fuente verificable: source no puede ser "manual", "internet",
   "known". Debe identificar la evidencia utilizada.

C. No inferencia silenciosa: no crear fila porque el ticker "parezca
   obvio". Debe existir evidencia documental suficiente para
   CUSIP -> issuer/security class -> ticker.

D. title_of_class NO es decorativo. La SEC define CUSIP junto con
   TITLEOFCLASS. El resolver no debe tratar CUSIP como identidad
   economica aislada del tipo/clase.

## 5. P3 - Q-D: multiples variantes CUSIP

DICTAMEN: GO.

Modelo multi-fila con vigencia temporal es el correcto. Mejor que un
CUSIP -> ticker estatico.

Firma obligatoria del resolver: resolver(cusip, report_period),
NO resolver(cusip).

Control obligatorio: valid_from <= valid_to cuando valid_to no es
NULL. Sin solapes incompatibles.

El ejemplo DuPont del informe es ejemplo de diseno, no evidencia de
que las fechas concretas sean correctas. No autorizado incorporar
esos valores como verdad sin evidencia.

## 6. P4 - 519 vs 501 CUSIPs

DICTAMEN: NO CERRADO.

No se sabe todavia que 519 != 501 sea un error. Los dos numeros
pueden pertenecer a universos semanticamente distintos.

Reconciliacion explicita obligatoria antes de construir excepciones:

| Categoria                                    |  N |
|----------------------------------------------|----|
| Identifiers en etf_holdings                  | 519 |
| Que son equity individual elegible           |   X |
| Excluidos por ser ETF                        |   X |
| Excluidos por clase/instrumento              |   X |
| CUSIPs IAE elegibles                         | 501 |
| Diferencia residual                          |   0 |

P4 pasa a ser requisito de FA-2.2. NO bloquea FA-2.1.

## 7. Q-A - Filtro temporal

DICTAMEN: GO.

Campo contractual canonico: SUBMISSION.PERIODOFREPORT.

REPORTCALENDARORQUARTER (COVERPAGE) es control de coherencia,
NO sustituto silencioso.

FA-2.1: no introducir CUSIP resolver, manager graph, NIPC,
economic owner, institutional confirmation.

## 8. Q8 - ISAMENDMENT vs SUBMISSIONTYPE

DICTAMEN: CONFIRMADO.

SUBMISSIONTYPE (13F-HR, 13F-HR/A, 13F-NT, 13F-NT/A) es la dimension
canonica para clasificar enmiendas.

COVERPAGE.ISAMENDMENT es campo documental / control auxiliar.

La coincidencia empirica (ISAMENDMENT=Y == 441 COMBINATION REPORT)
es diagnostico del dataset concreto, NO regla general del sistema.

## 9. Amendments - correccion arquitectonica

La SEC establece que una enmienda puede ser restatement o anadir
posiciones nuevas. Una restatement sustituye al filing anterior.

Consecuencia de orden:

    FA-2.3 atribucion provisional
       |
       v
    FA-2.4 canonical snapshot del trimestre
       |
       v
    FA-2.3.bis atribucion definitiva

FA-2.3 NO puede declarar definitiva la atribucion de managers hasta
que FA-2.4 haya determinado el snapshot canonico.

FA-2.3.bis deja de ser "posible descubrimiento" y pasa a ser
requisito arquitectonico.

## 10. P6 - Mini-gates

DICTAMEN: GO.

Cada subfase: pytest + pyflakes=0 + compileall=OK + working tree limpio
+ commit local.

Sin push de FA-2 hasta Gate FA-2 final.

## 11. Correccion documental obligatoria

El informe FA-2.0 usaba "3.3M filas". Debe sustituirse por el valor
exacto 3,822,885 (dataset completo). Distinguir siempre:

    dataset completo = 3,822,885
    dataset filtrado Q1 = 3,321,967

Tambien aplicar el renombrado semantico: economic_owner ->
canonical_reporting_relationship_key.

## 12. Autorizacion FA-2.1

AUTORIZADA sin modificar el diseno previamente aprobado.

Alcance:

    filter_by_period(dfs, period="2026-03-31") -> dict[str, DataFrame]
       |
       v
    SUBMISSION.PERIODOFREPORT
       |
       v
    periodo canonico 2026-03-31

Controles exigidos: filings_in_period, filings_outside_period,
13F-HR, 13F-HR/A, 13F-NT, 13F-NT/A.

NO introducir todavia: CUSIP resolver, manager graph, NIPC,
economic owner, institutional confirmation.

## 13. Tabla de dictamenes

| Punto | Dictamen |
|---|---|
| FA-1 | CERRADA / APROBADA |
| Q-A temporal | GO |
| P1 - B.1 | GO condicionado a renombrado semantico |
| P2 - excepciones CUSIP | GO condicionado |
| P3 - CUSIP temporal | GO |
| P4 - 519 vs 501 | ABIERTO; obligatorio resolver en FA-2.2 |
| P5 - OTHERMANAGER/2 | GO |
| P6 - mini-gates | GO |
| Amendments por SUBMISSIONTYPE | GO / confirmado |
| NIPC | BLOQUEADO |
| Push FA-2 | NO |
| FA-2.1 | AUTORIZADA |
| FA-2.2 | Pendiente de reconciliacion P4 |
| FA-2.3 | Pendiente de modelo definitivo + dependencia de canonicalizacion |

---

Fin del dictamen. Version 1.0 (2026-09-19).
