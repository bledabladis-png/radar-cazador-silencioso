# 54 - Estado Gold Standard v2 y plan de ejecucion

Fecha: 2026-10-05
Rama: gold-standard-v2
Base: gold-standard-v1 @ ccb30b3 (congelada)
Autorizacion auditor: CONDICIONAL
  - P1 (3 anotadores): pendiente decision usuario
  - P2 (estructura ADMIN/ANNOTATOR): VERIFICADO

---

## 1. Estado del repositorio

    Rama activa:    gold-standard-v2
    Commits desde v1: 14
    Tests:          219 passed
    pyflakes:       limpio
    compileall:     exit 0
    Working tree:   limpio

No se ha tocado: wyckoff_v1.py, legacy, config, protocolo v3,
consumidores, main.

---

## 2. Cobertura funcional

Modulos en scripts/gold_standard/:

    constants.py           constantes congeladas
    seeds.py               SeedSequence por replica
    sampling_frame.py      frame sobre build_ticker_df
    sector_map.py          mapping con UNKNOWN
    sampling_a.py          Muestra A (200+200, min_gap por clase)
    sampling_b.py          Muestra B (colapso solo por periodo)
    power.py               dimensionamiento n_B (B=500 -> B=2000)
    render.py              PNG ciego 1600x900
    annotation.py          blind_id + ADMIN/ANNOTATOR_N
    adjudication.py        mayoria binaria + technical_ineligible
    kappa.py               Fleiss + Cohen
    metrics_simple.py      Wilson + confusion + F1
    metrics_weighted.py    Hajek ponderado + F1 directo
    bootstrap_rwy.py       Rao-Wu-Yue con SeedSequence
    bootstrap_cluster.py   cluster global + invalid_rate
    classify_as_of.py      wrapper temporal/as_of
    preflight.py           integracion real con datos
    run_capa1.py           orquestador con ADMIN/ANNOTATOR

---

## 3. Bloqueos auditor

### P0 (criticos) - todos cerrados

    P0.1  Filtro sectorial del universo        CERRADO
    P0.2  Sector UNKNOWN en vez de descartar   CERRADO
    P0.3  A y B disjuntas                      CERRADO
    P0.4  min_gap en sesiones, no dias         CERRADO
    P0.5  blind_ids sin materializar rango     CERRADO
    P0.6  bootstrap invalid_rate + K>=20       CERRADO

### P1 (importantes) - todos cerrados

    P1.1  F1 ponderado sin redondeo            CERRADO
    P1.2  B=2000 confirmacion                  CERRADO
    P1.3  SeedSequence determinista            CERRADO
    P1.4  ADMIN/ANNOTATOR_N separados          CERRADO
    P1.5  Colapso estratos solo por periodo    CERRADO
    P1.6  Preflight integration real           CERRADO

### Prerrequisitos auditor

    P1 (3 anotadores humanos)    PENDIENTE decision usuario
    P2 (estructura ADMIN/ANNOT)  VERIFICADO

---

## 4. Hallazgos tecnicos del dia

### 4.1 Contrato implicito del detector

detect_sow exige DataFrame sin NaN. Si recibe parquet crudo
(~43 NaN por ticker), rolling(60, min_periods=60) descarta toda
ventana con un NaN. Senal pierde 10x: 408 -> 2559 positivos.

Los 6 consumidores reales ya cumplian el contrato via
build_ticker_df. La implementacion Gold Standard no lo documentaba.

Fix: build_ticker_df antes de detect_sow en sampling_frame y
run_capa1.
Test: tests/test_gold_standard_contrato_detector.py.

### 4.2 min_gap con dos bugs encadenados

Bug A: min_gap aplicado al conjunto entero. Negativos dominan y
desplazan positivos en greedy aleatorio (2559 -> 12 positivos).

Bug B: pos medida dentro del subconjunto filtrado, no del
calendario completo del ticker.

Fix: min_gap por clase + pos_sesion real via build_calendar_map.
Test: tests/test_gold_standard_min_gap_por_clase.py.

### 4.3 run_capa1 no usaba ADMIN/ANNOTATOR_N

Usaba write_all_csvs (todo mezclado). Auditor P2 exigia
separacion.

Fix: write_paquetes_anotadores(cases_dir_source=...).
Test: tests/test_gold_standard_run_capa1_p14.py.

---

## 5. Preflight real

Artefacto: outputs/audit/preflight_gold_standard_20261005_071015.json

    Dataset:              316 tickers, 1297 sesiones
    Sampling frame:       243.435 episodios, 301 tickers
    Sector conocido:      220
    UNKNOWN:               96 tickers
    Detector:            2559 positivos (1,05%)
    Muestra A:            400 (200 pos + 200 neg), 234 tickers
    Muestra B:            200, 147 tickers, pesos 1092-1432
    Estratos:             24 (12 sectores x 2 periodos)
    Disjuncion A inter B: 0
    Blind IDs:            600 unicos
    Duracion:             26s

Todos los checks_ok en true.

---

## 6. Autorizacion vigente

AUTORIZADO CONDICIONALMENTE por auditor.

Permite:
    Generar 600 PNGs
    Generar CSVs de anotacion
    Generar mapping
    Asignacion ciega a anotadores
    Inicio de anotacion humana

No permite:
    PASS de Capa 1
    Validacion de detect_sow
    Modificar wyckoff_v1.py, config, protocolo v3, legacy,
      consumidores, main
    Merge a main

Condiciones:
    1. 3 anotadores humanos independientes
    2. Estructura ADMIN/ANNOTATOR_N confirmada (verificado)

---

## 7. Pasos a seguir

### 7.1 Decision inmediata

    ¿Hay 3 anotadores humanos disponibles?

    Si  -> run_capa1.py --go, entregar paquetes, iniciar anotacion.
    No  -> Opcion C: generar paquetes sin iniciar anotacion.
           Entregar cuando haya 3 anotadores.

### 7.2 Si se ejecuta --go

    Paso 1  py scripts\gold_standard\run_capa1.py --go
            Tiempo estimado: 5-10 min

    Paso 2  py scripts\gold_standard\verify_paquetes.py
            Verifica: 600 PNGs, 600 blind_ids unicos, 400 A + 200 B,
            A inter B = 0, mapping completo, PNGs sin corrupcion,
            PNGs sin texto identificativo (OCR muestra), estructura
            ADMIN/ANNOTATOR correcta, SHA-256 en manifest.

    Paso 3  Entrega a anotadores.
            Contrato firmado. Instrucciones. Fichas canonicas.
            Sin acceso a ADMIN/.

### 7.3 Anotacion

    Duracion estimada: 2-3 semanas por anotador.
    Cada anotador anota los 600 casos en orden aleatorio propio.
    technical_ineligible solo por defecto objetivo.
    Sin comunicacion entre anotadores.
    QA de supervision continua.
    Si abandono: sustitucion con protocolo identico.
    Prohibido cambiar rubrica a mitad.

### 7.4 Analisis post-anotacion

    Paso 4  adjudication.py
    Paso 5  kappa.py (Fleiss principal, solo 3 votos validos)
    Paso 6  metrics_weighted.py + bootstrap_rwy.py (Muestra B)
            metrics_simple.py (Muestra A descriptiva)
    Paso 7  Gate: LCI95(Se) >= 0.70, LCI95(Sp) >= 0.70
            Kappa >= 0.60

### 7.5 Resultado

    PASS        Capa 1 cerrada. Autorizar Capa 2 (Estudio B).
    FAIL        Diagnosticar causa antes de SOW v2.
    INCONCLUSO  Rediseñar referencia o ampliar muestra.

### 7.6 Despues de Capa 1

    Capa 2 - Estudio B (comparativo Legacy vs v1.8).
             Protocolo propio. Macro-F1 primaria.
             Comparacion pareada. Independiente de Capa 1.

    Estudio C - Pivot diagnostic. Solo lectura.
                HH/HL/LH/LL vs legacy t_norm vs v1.8 t_norm.

    5b.X OOS - Post 2027. Protocolo ya firmado.
               SOW v1.9 congelado.

### 7.7 Riesgos

    Anotadores no disponibles   Opcion C: paquetes preparados
    Abandono de anotador        Sustitucion con protocolo identico
    Kappa insuficiente          Rediseñar referencia
    PNGs con corrupcion         QA previo + verify_paquetes
    Seleccion retrospectiva     Protocolo congelado
    Merge accidental a main     Rama aislada

### 7.8 Prohibiciones vigentes

    NO tocar wyckoff_v1.py
    NO tocar legacy
    NO tocar config
    NO tocar protocolo v3 firmado
    NO migrar consumidores
    NO merge a main
    NO reabrir grid 2021-2026
    NO buscar parametros alternativos

---

## 8. Artefactos de auditoria

    outputs/audit/preflight_gold_standard_20261005_071015.json
    outputs/audit/wyckoff_legacy_vs_v18_2026-10-05.csv
    outputs/audit/wyckoff_sow_vs_drawdown_2026-10-05.csv
    outputs/audit/wyckoff_componentes_por_origen_2026-10-05.csv
    outputs/audit/wyckoff_sow_dropna_2026-10-05.csv

No versionados. Solo lectura.

---

Fin del documento.