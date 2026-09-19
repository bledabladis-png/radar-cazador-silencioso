# INFORME - Coverage baseline NIPC (Gate 0 baseline)

**Modulo:** IAE (Institutional Accumulation Evidence) - SEC 13F / NIPC
**Referencia:** prompt maestro v6.36
**Fecha:** 2026-09-19
**Fase:** coverage baseline (Q-T50-5, post dictamen TOP 50)
**Dictamen habilitante:** INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_TOP50_DICTAMEN.md
**Evidencia:** docs/auditoria/evidence/nipc_gate0_baseline/

---

## 0. Resumen ejecutivo

Medicion del estado de cobertura NIPC sobre 4 universos anidados
Q4 2025 -> Q1 2026, sin OpenFIGI, sin thresholds, sin modificacion
del motor.

Los cuatro universos:

    RAW_13F_SH_NULL      SH + PUTCALL NULL
       v
    ELIGIBLE_SEC         + SEC Official List {ACTIVE, ADDED}
       v
    TECHNICAL_RADAR      + ticker in radar_equities  (cierre del tecnico)
       v
    OPERATIONAL_EQUITY   + get_instrument_class(ticker) == EQUITY

Resultado: las 6 metricas pairwise obligatorias (NIPC_COVERAGE_POLICY.md
seccion 2) quedan calculadas sobre los 4 universos. Los valores
numericos se presentan en secciones 3-5. Los checks mecanicos de
integridad pasan (seccion 6). Este informe NO fija thresholds y NO
propone interpretacion economica.

---

## 1. Metodologia

Congelada en docs/auditoria/evidence/nipc_gate0_baseline/README.md antes
de producir resultados. Orden metodologico respetado:

    README congelado
       v
    probe escrito
       v
    pyflakes
       v
    ejecucion
       v
    resultados
       v
    este informe + HASHES.txt

Funciones productivas utilizadas (sin duplicar logica):

    temporal_filter::filter_by_period
    amendments::apply_amendments
    sec13f_list::load_official_list
    sec13f_list::resolve_eligibility
    security_identity::load_crosswalk_internal
    security_identity::load_cusip_equivalence
    security_identity::resolve_batch_identities
    delta_shares::compute_reported_position_units
    delta_shares::compute_delta_shares
    nipc::compute_nipc_and_coverage
    instrument_registry::get_instrument_class

Reglas congeladas aplicadas:

    - Agregados SSHPRNAMT desde canonical_snapshot (apply_amendments).
    - Match key C2 sin report_period.
    - canonical_security != ticker. Solo 'equity:<X>' produce ticker.
    - SSHPRNAMT entra UNA VEZ por source line.

---

## 2. Definiciones operativas

    RAW_13F_SH_NULL
      = SH + PUTCALL NULL del canonical_snapshot de cada periodo.

    ELIGIBLE_SEC
      = RAW_13F_SH_NULL ^ section13f_eligible in {ACTIVE, ADDED}
        segun SEC Official List of Section 13(f) Securities.

    TECHNICAL_RADAR
      = ELIGIBLE_SEC ^ ticker in radar_equities.
        Cierre del UNIVERSO_TECNICO aprobado (Vía 3).

    OPERATIONAL_EQUITY
      = TECHNICAL_RADAR ^ get_instrument_class(ticker) == EQUITY.
        Universo operativo de NIPC_COVERAGE_POLICY seccion 10.

radar_equities se deriva de data/stock_prices.parquet:

    stock_prices.parquet
      -> columnas MultiIndex, nivel 1 = ticker
      -> excluir sufijos: .L .DE .MC .PA .AS .MI .BR .ST .HE .CO .OL .VI .LS .IR

Fuentes:

    D:/13f_probe/processed/2025Q4/*.parquet    (7 TSVs)
    D:/13f_probe/processed/2026Q1/*.parquet    (7 TSVs)
    D:/13f_probe/official_list_13f/13flist_2025Q4.txt
    D:/13f_probe/official_list_13f/13flist_2026Q1.txt
    data/stock_prices.parquet
    data/etf_holdings.csv
    data/mappings/cusip_ticker_exceptions.csv
    data/mappings/cusip_equivalence.csv

---

## 3. Conteos por periodo

| Metrica | Q4 2025 | Q1 2026 |
|---|---|---|
| period (literal) | 2025-12-31 | 2026-03-31 |
| CUSIPs SH+null | 24,129 | 24,580 |
| CUSIPs elegibles SEC | 11,693 | 12,940 |
| units RAW_13F_SH_NULL | 2,360,246 | 2,420,326 |
| units ELIGIBLE_SEC | 2,335,031 | 2,397,186 |
| units TECHNICAL_RADAR | 474,395 | 490,732 |
| units OPERATIONAL_EQUITY | 474,395 | 490,732 |

Contexto:

    radar_equities = 242
    excluidos sufijo EU = 71
    crosswalk_internal rows = 529
    cusip_equivalence rows = 0

---

## 4. Fallout ELIGIBLE_SEC -> TECHNICAL_RADAR

### 4.1. Q4 2025 (2025-12-31)

| categoria | units | %units | peso_ssh | %peso |
|---|---:|---:|---:|---:|
| IN_RADAR | 474,395 | 20.32% | 207,935,058,532 | 29.18% |
| CANONICAL_TICKER_OUTSIDE_RADAR | 316,812 | 13.57% | 76,949,114,953 | 10.80% |
| FIGI_CANONICAL_NO_TICKER | 0 | 0.00% | 0 | 0.00% |
| OBSERVED_ONLY_NO_CANONICAL | 1,543,824 | 66.12% | 427,617,487,973 | 60.02% |
| UNRESOLVED_NO_CANONICAL | 0 | 0.00% | 0 | 0.00% |
| AMBIGUOUS_NO_CANONICAL | 0 | 0.00% | 0 | 0.00% |
| CONFLICT_NO_CANONICAL | 0 | 0.00% | 0 | 0.00% |
| OTHER | 0 | 0.00% | 0 | 0.00% |

Detalle CANONICAL_TICKER_OUTSIDE_RADAR: 278 tickers unicos.
Top 10 por peso SSHPRNAMT (unidades, peso):

    HBAN  1,143   1,406,097,880
    BSX   1,909   1,356,232,363
    USB   2,149   1,254,156,343
    APH   2,180   1,239,472,237
    HPE   1,170   1,202,389,429
    KHC   1,182   1,011,992,645
    NKE   2,211   1,007,241,054
    VTRS  1,072     974,684,392
    KEY     916     970,958,801
    FAST  1,477     962,499,029

Detalle NO_CANONICAL: 11,197 CUSIPs unicos sin canonical_security.
Top 10 por peso SSHPRNAMT (unidades, peso, status):

    G4124C109    581   3,080,218,553  OBSERVED_ONLY
    G6683N103    917   2,867,293,204  OBSERVED_ONLY
    30231G102  4,740   2,748,494,764  OBSERVED_ONLY
    921943858  2,894   2,354,601,744  OBSERVED_ONLY
    051774107    401   1,741,652,395  OBSERVED_ONLY
    G0250X107    692   1,539,837,884  OBSERVED_ONLY
    46434G103  2,272   1,536,037,921  OBSERVED_ONLY
    46432F842  2,401   1,531,134,788  OBSERVED_ONLY
    11271J107    978   1,529,992,122  OBSERVED_ONLY
    113004105    664   1,489,869,510  OBSERVED_ONLY

### 4.2. Q1 2026 (2026-03-31)

| categoria | units | %units | peso_ssh | %peso |
|---|---:|---:|---:|---:|
| IN_RADAR | 490,732 | 20.47% | 207,825,597,694 | 28.75% |
| CANONICAL_TICKER_OUTSIDE_RADAR | 321,171 | 13.40% | 77,111,146,439 | 10.67% |
| FIGI_CANONICAL_NO_TICKER | 0 | 0.00% | 0 | 0.00% |
| OBSERVED_ONLY_NO_CANONICAL | 1,585,283 | 66.13% | 437,832,560,640 | 60.58% |
| UNRESOLVED_NO_CANONICAL | 0 | 0.00% | 0 | 0.00% |
| AMBIGUOUS_NO_CANONICAL | 0 | 0.00% | 0 | 0.00% |
| CONFLICT_NO_CANONICAL | 0 | 0.00% | 0 | 0.00% |
| OTHER | 0 | 0.00% | 0 | 0.00% |

Detalle CANONICAL_TICKER_OUTSIDE_RADAR: 280 tickers unicos.
Top 10 por peso SSHPRNAMT (unidades, peso):

    HBAN  1,216   1,727,516,122
    BSX   1,811   1,340,523,447
    USB   2,126   1,246,026,818
    HPE   1,171   1,198,913,629
    APH   2,278   1,152,034,526
    KHC   1,173     997,967,863
    VTRS  1,085     990,157,565
    FAST  1,529     945,964,897
    NKE   2,069     945,698,661
    KEY     939     932,990,636

Detalle NO_CANONICAL: 12,440 CUSIPs unicos sin canonical_security.
Top 10 por peso SSHPRNAMT (unidades, peso, status):

    G4124C109    525   3,057,901,738  OBSERVED_ONLY
    G6683N103    952   2,863,406,734  OBSERVED_ONLY
    921943858  3,020   2,393,691,928  OBSERVED_ONLY
    46429B267    963   1,770,991,227  OBSERVED_ONLY
    46434G103  2,385   1,690,755,537  OBSERVED_ONLY
    46432F842  2,502   1,638,879,161  OBSERVED_ONLY
    051774107    430   1,631,567,073  OBSERVED_ONLY
    136385101    793   1,573,223,006  OBSERVED_ONLY
    113004105    672   1,481,590,084  OBSERVED_ONLY
    11271J107    971   1,475,718,392  OBSERVED_ONLY

---

## 5. Pairwise coverage - 4 universos anidados

### 5.1. RAW_13F_SH_NULL

    Q4 units: 2,360,246   Q1 units: 2,420,326
    delta filas: 4,082,327
    match_status: UNRESOLVED_IDENTITY=3,177,329  BOTH=698,245  NEW=113,710  EXIT=93,043
    NIPC total (observable): +23,651,586
    coverage_previous=0.0206  coverage_current=0.0204
    paired_security_coverage=0.0181  paired_weighted_share_coverage=0.2271
    unmapped_weight_previous=0.9794  unmapped_weight_current=0.9796
    STATUS: INSUFFICIENT

### 5.2. ELIGIBLE_SEC

    Q4 units: 2,335,031   Q1 units: 2,397,186
    delta filas: 4,034,035
    match_status: UNRESOLVED_IDENTITY=3,129,107  BOTH=698,182  NEW=113,721  EXIT=93,025
    NIPC total (observable): +52,570,648
    coverage_previous=0.0424  coverage_current=0.0386
    paired_security_coverage=0.0368  paired_weighted_share_coverage=0.2772
    unmapped_weight_previous=0.9576  unmapped_weight_current=0.9614
    STATUS: INSUFFICIENT

### 5.3. TECHNICAL_RADAR

    Q4 units: 474,395   Q1 units: 490,732
    delta filas: 537,453
    match_status: BOTH=427,674  NEW=63,058  EXIT=46,721
    NIPC total (observable): -109,460,838
    coverage_previous=1.0000  coverage_current=1.0000
    paired_security_coverage=0.9909  paired_weighted_share_coverage=0.9864
    unmapped_weight_previous=0.0000  unmapped_weight_current=0.0000
    STATUS: INSUFFICIENT

### 5.4. OPERATIONAL_EQUITY

    Q4 units: 474,395   Q1 units: 490,732
    delta filas: 537,453
    match_status: BOTH=427,674  NEW=63,058  EXIT=46,721
    NIPC total (observable): -109,460,838
    coverage_previous=1.0000  coverage_current=1.0000
    paired_security_coverage=0.9909  paired_weighted_share_coverage=0.9864
    unmapped_weight_previous=0.0000  unmapped_weight_current=0.0000
    STATUS: INSUFFICIENT

### 5.5. Comparativa (6 metricas + status)

| metrica | RAW_13F | ELIGIBLE_SEC | TECH_RADAR | OP_EQUITY |
|---|---:|---:|---:|---:|
| coverage_previous | 0.0206 | 0.0424 | 1.0000 | 1.0000 |
| coverage_current | 0.0204 | 0.0386 | 1.0000 | 1.0000 |
| paired_security_coverage | 0.0181 | 0.0368 | 0.9909 | 0.9909 |
| paired_weighted_share_coverage | 0.2271 | 0.2772 | 0.9864 | 0.9864 |
| unmapped_weight_previous | 0.9794 | 0.9576 | 0.0000 | 0.0000 |
| unmapped_weight_current | 0.9796 | 0.9614 | 0.0000 | 0.0000 |
| NIPC total (observable) | +23,651,586 | +52,570,648 | -109,460,838 | -109,460,838 |
| STATUS | INSUFFICIENT | INSUFFICIENT | INSUFFICIENT | INSUFFICIENT |

---

## 6. Checks mecanicos de integridad

| Check | Descripcion | Resultado |
|---|---|---|
| 1 | Suma %units fallout = 100.00% (±0.01) | Q4=100.01% (redondeo), Q1=100.00% PASS |
| 2 | Suma %peso fallout = 100.00% (±0.01) | Q4=100.00%, Q1=100.00% PASS |
| 3 | IN_RADAR units == units TECHNICAL_RADAR | Q4: 474,395 == 474,395; Q1: 490,732 == 490,732 PASS |
| 4 | OTHER units == 0 | Q4: 0; Q1: 0 PASS |

Marcadores de seccion (esperado 1 cada uno):

    PERIODO Q4_2025           1
    PERIODO Q1_2026           1
    fallout Q4_2025           1
    fallout Q1_2026           1
    PAIRWISE COVERAGE         1
    FIN                       1

Marcadores de universo (esperado >=2 cada uno):

    RAW_13F_SH_NULL           3
    ELIGIBLE_SEC              5
    TECHNICAL_RADAR           5
    OPERATIONAL_EQUITY        3

Spot-check numerico (linea y contenido):

    L5    radar_equities = 242
    L126  NIPC total (observable): -109460838
    L128  paired_security_coverage=0.9909  paired_weighted_share_coverage=0.9864
    L136  NIPC total (observable): -109460838
    L138  paired_security_coverage=0.9909  paired_weighted_share_coverage=0.9864

---

## 7. Evidencia reproducible

Directorio: docs/auditoria/evidence/nipc_gate0_baseline/

    README.md                     Metodologia (congelada antes del run).
    probe_coverage_baseline.py    Script del probe.
    baseline_output.txt           Salida cruda de la ejecucion (exit 0).
    HASHES.txt                    SHA-256 de los 3 ficheros.

SHA-256:

    2d84712458e1a393483d26810a971803faa98953e4381663bbb5d9b679f09002  README.md
    b2ede861bc4a49fc27932d09cd8aaac533e31dd71eed17beff182fc8dc8ca337  probe_coverage_baseline.py
    d5a69ad4588af02c218388cbfd111cc150781ded687f025bb31a19ef8845f1e0  baseline_output.txt

Reproducibilidad:

    py docs/auditoria/evidence/nipc_gate0_baseline/probe_coverage_baseline.py

Requiere acceso a D:/13f_probe/ (processed + official_list_13f) y
a data/stock_prices.parquet. Deterministico, sin datetime.now().

---

## 8. Lo que este informe NO hace

    - NO fija THRESHOLD_1 / THRESHOLD_2.
    - NO llama a OpenFIGI.
    - NO modifica el motor NIPC.
    - NO interpreta economicamente los numeros.
    - NO propone promocion de status.
    - NO abre Gate-NIPC.2.

---

## 9. Estado del gate

    Gate-NIPC.2                SIGUE BLOQUEADO por coverage/thresholds
    Gate-NIPC.3                NO AUTORIZADO
    THRESHOLD_1                UNDEFINED
    THRESHOLD_2                UNDEFINED
    OpenFIGI                   FUERA DE ESTA FASE
    Modificacion motor NIPC    NO
    Reconciliacion NT-HR       NO AUTORIZADA
    C2 (match key)             VIGENTE / SIN PATCH

---

Fin del informe. Version 1.0 (2026-09-19).
Evidencia congelada en docs/auditoria/evidence/nipc_gate0_baseline/.