# INVENTARIO F2.1-bis - Fuentes de identidad disponibles para TARGET_UNIVERSE

**Modulo:** IAE (Institutional Accumulation Evidence) - SEC 13F / NIPC
**Referencia:** prompt maestro v6.37
**Fecha:** 2026-09-19
**Fase:** F2.1-bis (respuesta al dictamen F2.2 NO GO del auditor)
**Dictamen habilitante:** revision F2.2 (NO GO CON CAMBIOS OBLIGATORIOS, 2026-09-19)
**Fichero auditado:** no audita policy. Inventaria fuentes en disco.

---

## 0. Proposito

El dictamen F2.2 exige que TARGET_UNIVERSE tenga identidad
independiente del proceso de resolucion que se pretende medir
(hallazgo H-TARGET-1). El dictamen abre dos caminos excluyentes:

  Camino A: existe un RADAR_TARGET_REGISTRY independiente.
  Camino B: no existe y se declara NOT AVAILABLE.

Este informe determina cual aplica, verificando las fuentes de
identidad en disco. NO modifica la policy. NO escribe v1.2.

---

## 1. Fuentes inventariadas

Se han inspeccionado todas las fuentes de mapeo y holdings del repo:

  data/mappings/isin_ticker_map.csv
  data/mappings/cusip_ticker_exceptions.csv
  data/mappings/cusip_equivalence.csv
  data/etf_holdings.csv
  data/index_holdings.csv
  data/amundi_yahoo_mapping.csv
  config/index_tickers.py

Y busqueda exhaustiva de ficheros con nombres
RADAR_TARGET_REGISTRY / radar_registry / identity_registry /
cusip_registry: 0 resultados.

---

## 2. Estado de cada fuente

### 2.1. isin_ticker_map.csv

  Path:     data/mappings/isin_ticker_map.csv
  Cabecera: isin, ticker
  Filas:    50 (49 tickers unicos)
  Clave:    ISIN -> ticker

  Tickers sin sufijo EU (universo radar USA): 0
  Todos los tickers son europeos (ABI.BR, DBK.DE, BMW.DE, DHL.DE, ...).

  Independiente del crosswalk evaluado: SI.
  Aporta radar USA: NO (0 tickers USA).

### 2.2. cusip_ticker_exceptions.csv

  Path:     data/mappings/cusip_ticker_exceptions.csv
  Cabecera: CUSIP, ticker, valid_from, valid_to, source, reason,
            title_of_class, verified_by
  Filas:    3 operativas + cabecera (4 lineas totales)

  Independiente del crosswalk evaluado: NO. Es parte del crosswalk
  cargado por security_identity.load_crosswalk_internal().

### 2.3. cusip_equivalence.csv

  Path:     data/mappings/cusip_equivalence.csv
  Filas:    0 por diseno (solo cabecera).

  Independiente del crosswalk evaluado: NO. Es parte del crosswalk
  cargado por security_identity.load_cusip_equivalence().

### 2.4. etf_holdings.csv

  Path:     data/etf_holdings.csv
  Cabecera: etf, ticker, identifier, weight
  Filas:    519 CUSIPs unicos (identifier) / 516 tickers unicos.

  Cobertura radar USA:
    tickers en radar USA:  220
    radar USA total:       242
    ratio:                 220/242 = 90.91 %

  Independiente del crosswalk evaluado: NO. Es parte del crosswalk
  cargado por security_identity.load_crosswalk_internal().

  NOTA CRITICA: este 90.91 % es exactamente el valor citado en
  NIPC_COVERAGE_POLICY.md v1.0 L115 ("mapping_coverage (radar USA):
  90.91% (220/242)"). Es decir, la policy cita como "cobertura de
  mapping" el ratio del propio insumo del resolver.

### 2.5. amundi_yahoo_mapping.csv

  Path:     data/amundi_yahoo_mapping.csv
  Cabecera: isin, yahoo_ticker
  Filas:    >= 3 verificadas (ES0113900J37,SAN.MC; ES0113211835,BBVA.MC;
            ES0144580Y14,IBE.MC).

  Independiente del crosswalk evaluado: SI.
  Aporta radar USA: NO (todos tickers europeos).

### 2.6. index_holdings.csv

  Path:     data/index_holdings.csv
  Cabecera: etf, ticker, name, weight
  Filas:    significativas (93712 bytes).

  Independiente del crosswalk evaluado: SI (no se carga en el resolver).
  Aporta CUSIP: NO (no tiene columna identifier/CUSIP).

### 2.7. config/index_tickers.py

  Contiene mapping simbolico de indices a ETFs. No es un registro
  de identidades CUSIP -> ticker. No sirve para TARGET_UNIVERSE.

---

## 3. Evaluacion contra el dictamen F2.2

El dictamen exige que TARGET_UNIVERSE pueda construirse sin ejecutar
el proceso de resolucion que se pretende medir. Es decir, se requiere
una fuente con:

  (a) clave CUSIP o FIGI (comparable con 13F),
  (b) ticker del radar USA,
  (c) independiente del crosswalk evaluado.

Resultado por fuente:

  Fuente                       (a) CUSIP  (b) radar USA  (c) Indep.
  ---------------------------  ---------  -------------  ----------
  isin_ticker_map.csv          NO (ISIN)  NO (0 USA)     SI
  cusip_ticker_exceptions.csv  SI         SI (3)         NO
  cusip_equivalence.csv        SI         0 filas        NO
  etf_holdings.csv             SI         220 (90.91%)   NO
  amundi_yahoo_mapping.csv     NO (ISIN)  NO (0 USA)     SI
  index_holdings.csv           NO         --             SI

Ninguna fuente cumple las tres condiciones (a)+(b)+(c)
simultaneamente.

---

## 4. Conclusion: Camino B

De los dos caminos abiertos por el dictamen F2.2:

  Camino A: construir/congelar RADAR_TARGET_REGISTRY independiente.
  Camino B: declarar TARGET_CUSIP_REGISTRY = NOT AVAILABLE.

Aplica Camino B. No existe en disco una fuente suficientemente
completa e independiente del crosswalk evaluado.

La consecuencia, en palabras del propio dictamen:

  "no debemos fabricar un TARGET_CUSIP de 242 elementos utilizando
   el mismo crosswalk cuya cobertura queremos medir."

Este informe confirma que ese crosswalk (etf_holdings.csv +
cusip_ticker_exceptions.csv) es hoy la unica fuente con CUSIPs
radar USA, y es precisamente la capa cuya cobertura se quiere
evaluar. Construir TARGET con ella reproduce la circularidad.

### Declaracion formal

  TARGET_CUSIP_REGISTRY = NOT AVAILABLE (a fecha 2026-09-19).

---

## 5. Implicacion para el siguiente micro-gate

El dictamen F2.2 abre la puerta a que OpenFIGI entre ANTES como
fuente para construir el registro objetivo, en lugar de como mejora
posterior de coverage. Cita literal:

  "OpenFIGI/otra fuente de identidad puede formar parte de una fase
   de construccion y validacion del target registry, antes de
   establecer thresholds."

Es decir, el orden cambia:

  v1.0 (descartado):
    baseline -> OpenFIGI -> thresholds

  v1.1 (F2.2, NO GO):
    baseline -> policy v1.1 -> OpenFIGI -> thresholds

  Propuesta (F2.1-bis):
    baseline -> RADAR_TARGET_REGISTRY (via OpenFIGI como fuente)
             -> policy v1.2 con TARGET independiente
             -> medicion real sobre TARGET independiente
             -> thresholds

Esta reordenacion NO se ejecuta ahora. Es propuesta a dictamen del
auditor.

---

## 6. Que NO hace este informe

  - NO modifica NIPC_COVERAGE_POLICY.md (v1.0 intacta).
  - NO modifica NIPC_COVERAGE_POLICY_V11_PROPUESTA.md.
  - NO ejecuta OpenFIGI.
  - NO construye el RADAR_TARGET_REGISTRY.
  - NO fija thresholds.
  - NO toca el motor.

---

## 7. Referencias

  - Dictamen F2.2 (NO GO CON CAMBIOS):
      docs/auditoria/NIPC_COVERAGE_POLICY_V11_PROPUESTA.md (revisado)
      (el dictamen fue entregado como texto; commit correspondiente
       en la cadena local)
  - Dictamen baseline:
      docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_BASELINE_DICTAMEN.md
  - Inventario F2.1:
      docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_CONTRACT_INVENTARIO.md
  - Policy v1.0 (intacta):
      docs/auditoria/NIPC_COVERAGE_POLICY.md

---

Fin del inventario F2.1-bis. Version 1.0 (2026-09-19).
Conclusion: Camino B. RADAR_TARGET_REGISTRY NOT AVAILABLE.
Siguiente fase: propuesta de construccion del registry via OpenFIGI
(a dictamen del auditor).