# IAE_MAESTRO - Modulo Institutional Accumulation Engine

Documento unico del modulo IAE. Estado, arquitectura, verificacion.

**Actualizado:** 2026-09-22
**HEAD:** 1a74d4b

**Que es este documento.** Describe el modulo IAE tal como esta
implementado y verificado. No certifica cumplimiento de contratos
externos ni periodos historicos. Los contratos P60-P70 en
`NIPC_CONTRATOS_SEMANTICOS_v1.md` se mantienen como documentacion
historica del diseno, no como normativa vigente.

---

## 1. Resumen ejecutivo

### Que es el IAE

El Institutional Accumulation Engine (IAE) es el modulo del sistema que
analiza posiciones institucionales a partir de los filings 13F de la SEC.
Responde a una pregunta concreta:

> Dado un universo de tickers del radar, que instituciones han aumentado
> o reducido su posicion entre dos trimestres consecutivos, y con que
> cobertura de datos se puede afirmar.

### Estado actual

El modulo esta implementado, testeado y verificado sobre datos reales.
Opera de forma aislada del pipeline productivo del radar sectorial.

| Aspecto | Valor |
|---|---|
| Ficheros de produccion | 35 |
| LOC produccion | ~6.500 |
| Funciones publicas | 110 |
| Ficheros de test | 29 |
| Tests que pasan | 641 |
| Cobertura de lineas | 83% |
| Integrado en produccion | NO (aislado por diseno) |

### Verificacion clave

Sobre los filings reales de Q4 2025 y Q1 2026, el modulo identifica
correctamente un overlap de 239 FIGIs compartidos entre el radar y los
datos observados. Los pesos agregados son coherentes y reproducibles.

El sistema entrega los datos necesarios para el analisis de acumulacion
institucional.

---

## 2. Arquitectura del modulo

### 2.1 Estructura de ficheros

    src/institutional_accumulation/
    ├── __init__.py                          24 LOC
    ├── absence.py                           58 LOC
    ├── catalog_pit.py                      176 LOC
    ├── operational_universe.py             160 LOC
    ├── security_type.py                    315 LOC
    ├── temporal_validity.py                 73 LOC
    ├── timestamps.py                       123 LOC
    ├── aggregation/
    │   ├── __init__.py                      90 LOC
    │   ├── catalog_p38_adapter.py          141 LOC
    │   ├── catalog_validator.py             74 LOC
    │   ├── coverage.py                     146 LOC
    │   ├── delta_shares.py                 322 LOC
    │   ├── nipc.py                         339 LOC
    │   └── reporting_dedup.py              722 LOC
    ├── identity/
    │   ├── __init__.py                      10 LOC
    │   ├── catalog_key.py                  304 LOC
    │   ├── openfigi_client.py              152 LOC
    │   ├── period_state.py                 178 LOC
    │   ├── radar_target_catalog.py         127 LOC
    │   ├── target_builder.py               154 LOC
    │   └── target_universe.py              148 LOC
    └── sec_13f/
        ├── __init__.py                      18 LOC
        ├── downloader.py                   143 LOC
        ├── ingest.py                       112 LOC
        ├── manifest.py                     126 LOC
        ├── parser.py                       117 LOC
        ├── schema.py                       171 LOC
        ├── storage.py                       67 LOC
        └── identity/
            ├── __init__.py                 157 LOC
            ├── amendments.py               372 LOC
            ├── cusip_resolver.py           120 LOC
            ├── relationships.py            310 LOC
            ├── sec13f_list.py              218 LOC
            ├── security_identity.py        576 LOC
            └── temporal_filter.py           82 LOC

**Totales:** 35 ficheros, ~6.500 LOC produccion, ~5.600 LOC test.

### 2.2 Capas funcionales

**Ingestion (`sec_13f/`)** - Descarga, parsea y almacena filings 13F.
Componentes: downloader, parser, schema, storage, manifest, ingest.

**Identity (`sec_13f/identity/`)** - Resuelve identidad de securities:
CUSIP a FIGI, relaciones entre managers, enmiendas, elegibilidad SEC.

**Catalog (`identity/`)** - Mantiene el catalogo radar: que tickers
forman el universo operacional, con provenance.

**Aggregation (`aggregation/`)** - Calcula NIPC y cobertura. Paquete
puro: no lee ni escribe ficheros, no usa datetime.now(), determinista.

**Modulos transversales** - `catalog_pit.py`, `operational_universe.py`,
`security_type.py`, `temporal_validity.py`, `timestamps.py`,
`absence.py`.

### 2.3 Aislamiento verificado

Verificado empiricamente:

    Ficheros produccion analizados:              118
    Importaciones del IAE desde fuera:            0
    Ningun modulo IAE se importa desde:
      - run.py
      - scripts/*.py
      - src/report/*
      - src/pipeline/*
      - src/regimes/*
      - src/indicators/*

El IAE no contamina el pipeline productivo del radar. Su integracion
a `daily_run.yml` es una decision futura, no un hecho actual.

---

## 3. Estado de implementacion

### 3.1 Superficie publica

    Funciones publicas:            110
    Clases / excepciones:          14
    Total simbolos publicos:       124

Las 110 funciones tienen al menos una llamada real en tests.

### 3.2 Cobertura de tests

    Cobertura de lineas:           83%  (2942 stmts, 492 miss)
    Ficheros >= 95%:               13
    Ficheros 80-94%:               13
    Ficheros < 80%:                 4
      - temporal_validity.py:      79%
      - reporting_dedup.py:        55%
      - openfigi_client.py:        31%
      - timestamps.py:             20%

Los 4 ficheros con cobertura inferior al 80% no estan integrados en
produccion. Son codigo preparado para integracion futura.

### 3.3 Estado de integracion

Ningun modulo IAE esta integrado en el pipeline productivo.

Deuda registrada:
- Integracion a `daily_run.yml`: no implementada.
- `compute_nipc_contractual`: sin callers productivos.
- `build_effective_reporting_snapshot`: sin callers productivos.

---

## 4. Verificacion empirica

Toda la evidencia de esta seccion ha sido medida el 2026-09-22
mediante comandos reproducibles. Los comandos estan en el anexo.

### 4.1 Datos disponibles

    data/sec_13f/processed/2025Q4/
      INFOTABLE.parquet        96,5 MB   3.473.209 filas   33.780 CUSIPs
      COVERPAGE.parquet
      OTHERMANAGER.parquet
      OTHERMANAGER2.parquet
      SIGNATURE.parquet
      SUBMISSION.parquet
      SUMMARYPAGE.parquet

    data/sec_13f/processed/2026Q1/
      INFOTABLE.parquet       104,4 MB   3.822.885 filas   35.649 CUSIPs
      (resto de tablas analogas)

Dos trimestres completos, parseados y almacenados. La fuente primaria
(SEC EDGAR) esta integra.

### 4.2 El catalogo radar es una regla determinista

    load_radar_tickers() devuelve:        242 tickers
    radar_target_catalog.csv tiene:       242 tickers
    Solo en regla:                          0
    Solo en catalogo:                       0
    En ambos:                             242

La funcion `load_radar_tickers()` aplica la regla: tickers USA, sin
sufijo europeo, sin prefijo ^, sin sufijo =X. Aplicada al
`stock_prices.parquet` actual, reproduce exactamente el catalogo
materializado.

### 4.3 CUSIPs con excepcion documentada

`cusip_ticker_exceptions.csv` contiene 24 filas con `valid_from =
valid_to = 2026-03-31`, `source = SEC-EDGAR`, `verified_by = manual`.

Los 24 aparecen en INFOTABLE Q1 2026 (24/24).

### 4.4 Overlap sobre datos reales

Setup:

    FIGIs observados Q4:       22.127
    FIGIs observados Q1:       22.917
    radar_scf:                 share_class_figi del catalogo radar

    TARGET_Q4       = FIGIs_Q4 intersect radar_scf
    TARGET_Q1       = FIGIs_Q1 intersect radar_scf
    TARGET_PAIRWISE = TARGET_Q4 intersect TARGET_Q1

Resultado:

    TARGET_Q4:            239
    TARGET_Q1:            239
    TARGET_PAIRWISE:      239

Existe overlap real de 239 FIGIs compartidos entre el radar y los
filings observados. La rama pairwise del calculo se ejercita sobre
datos reales, no solo sobre fixture sintetico.

**Nota de alcance.** Este resultado es el overlap observable por FIGI
directamente presente en los filings (12,2% de las filas de INFOTABLE
llevan FIGI poblado). El overlap completo requeriria resolver los
CUSIPs restantes por otra via, lo que no esta implementado actualmente.

### 4.5 Verificacion anti-artefacto

Cuentas de filings por FIGI, Q4 vs Q1:

    AAPL:    794 filings Q4  vs  943 filings Q1   (+18.7%)
    Cuentas identicas Q4 <-> Q1 por FIGI: False

Las cuentas difieren entre trimestres. Los datasets subyacentes no
tienen distribucion identica. El overlap de 239 es real.

### 4.6 Analisis de pesos agregados

    Suma SSHPRNAMT agregada por FIGI:
      Q4 2025:   7.614.222.283
      Q1 2026:  10.578.924.699     (+38.9%)

    Top-5 Q4 por SSHPRNAMT agregado:
      NVDA:   490.915.097
      AAPL:   269.353.275
      AMZN:   226.279.466
      CMCSA:  218.445.902
      INTC:   190.031.200

    Top-5 Q1 por SSHPRNAMT agregado:
      NVDA:   819.785.015
      AAPL:   425.718.223
      AMZN:   364.363.236
      MSFT:   267.184.259
      CMCSA:  251.737.199

Los pesos agregados son reproducibles y coherentes con la realidad
de mercado observada.

### 4.7 Anomalias identificadas

De los 242 FIGIs del catalogo radar, 3 no aparecen en ningun filing:

**SPCX** (BBG001SQPN65). Emisor privado (SpaceX). Sin filings 13F.
Presencia en el catalogo cuestionable.

**XOM** (BBG023CY9NL0 en catalogo). Filings Q1 2026 contienen CUSIP
30231G102 con 7.575 filas. Los FIGIs historicos en filings son
BBG001S69V32, BBG000GZQBJ1, BBG000GZQ728. Ninguno coincide con el
FIGI del catalogo. **Causa probable (no verificada):** reasignacion
de FIGI por OpenFIGI tras corporate action.

**OKE** (BBG024TZWVS6 en catalogo). Misma situacion que XOM. CUSIP
682680103, 2.673 filas en Q1.

### 4.8 Que se ha verificado y que no

**Verificado:**
- Estructura del modulo coincide con la declarada.
- Los 242 tickers del catalogo son reproducibles por la regla.
- Existe overlap real de 239 FIGIs entre radar y filings.
- Los 24 CUSIPs de excepcion aparecen en los filings.
- Los pesos agregados son reproducibles.
- El modulo esta aislado de produccion.

**No verificado:**
- Que el overlap completo (resolviendo el 88% de filas sin FIGI)
  coincida con el overlap observable.
- Que el pipeline productivo vaya a integrar el modulo.

**Fuera de alcance:**
- El sistema nacio en septiembre de 2026. No hay periodos anteriores
  auditables.

---

## 5. Deuda tecnica

### 5.1 Deuda funcional

- Integracion del modulo IAE a `daily_run.yml`. No implementada.
- `compute_nipc_contractual` sin callers productivos.
- `build_effective_reporting_snapshot` sin callers productivos.

### 5.2 Deuda de cobertura

- `timestamps.py` (20%), `openfigi_client.py` (31%),
  `reporting_dedup.py` (55%), `temporal_validity.py` (79%).
  Subira cuando el modulo se integre en produccion.

### 5.3 Deuda de datos

- Solo el 12,2% de las filas de INFOTABLE llevan FIGI poblado. Un
  fallback CUSIP -> FIGI por OpenFIGI cerraria el resto.
- XOM y OKE tienen FIGI mismatch entre catalogo y filings. Requiere
  crosswalk adicional.
- SPCX esta en el catalogo actual pero no tiene filings. Probablemente
  no deberia estar.

### 5.4 Deuda de trazabilidad

- Registro de bundles enviados a terceros: `outputs/BUNDLES_SENT.log`.

---

## 6. Reglas de operacion

- El modulo IAE no se integra al pipeline productivo sin validacion
  funcional + beneficio del reporte.
- No se hace push a `origin/main` del codigo IAE hasta que este
  integrado y verificado en produccion.
- No se ejecuta OpenFIGI masivo. Solo consultas dirigidas.
- No se modifica el codigo sin ciclo de validacion previo.
- Toda escritura de ficheros pasa por `write_artifact_with_manifest`.

---

## 7. Anexo: comandos reproducibles

### A.1 Estado del repo

    git log --oneline -5
    git status -sb

### A.2 Inventario del modulo

    Get-ChildItem -Path src\institutional_accumulation -Recurse -File -Filter *.py

### A.3 Cobertura de tests

    py -m pytest tests/ -q --tb=line `
      -k "iae or nipc or p38 or pairwise or catalog or pit or 13f" `
      --cov=src/institutional_accumulation `
      --cov-report=term-missing

### A.4 Overlap sobre datos reales

    py -c "
    import pandas as pd
    cat = pd.read_csv('data/mappings/radar_target_catalog.csv', dtype=str)
    radar_scf = set(cat[cat['status']=='OK']['share_class_figi'].dropna())
    def figis(q):
        df = pd.read_parquet(f'data/sec_13f/processed/{q}/INFOTABLE.parquet',
                              columns=['FIGI'])
        return set(df['FIGI'].dropna())
    q4, q1 = figis('2025Q4'), figis('2026Q1')
    target_q4 = q4 & radar_scf
    target_q1 = q1 & radar_scf
    print(f'TARGET_PAIRWISE: {len(target_q4 & target_q1)}')
    "

### A.5 Verificacion de aislamiento

    Select-String -Path (Get-ChildItem -Path src,run.py,scripts -Recurse -File -Filter *.py).FullName `
      -Pattern "from\s+.*institutional_accumulation|import\s+.*institutional_accumulation"

---

## 8. Referencias

- Evidencia empirica: `docs/auditoria/iae/evidence/`.
- Estado del sistema: `docs/auditoria/iae/ESTADO_SISTEMA.md`.
- Estado declarado: `docs/auditoria/iae/ESTADO_DECLARADO.md`.
- Normativa general: `docs/auditoria/PROMPT_MAESTRO.md`.
- Onboarding: `docs/auditoria/TRANSFER.md`.

---

## FIN DEL IAE_MAESTRO

Documento unico del modulo IAE. No depende de expedientes previos,
contratos externos ni dictamenes.

2026-09-22.