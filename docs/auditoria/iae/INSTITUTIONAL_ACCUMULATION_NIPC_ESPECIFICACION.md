# ESPECIFICACION NIPC - Net Institutional Position Change

Modulo: IAE (Institutional Accumulation Evidence) - SEC 13F
Version: 1.4 (2026-09-19)
Versiones anteriores: 1.3 (commit 6e64a72; preservada como
                      INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION_v1.3.md)
                      1.2 (commit 5eeb022; preservada como
                      INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION_v1.2.md)
                      1.1 (commit 52fc53c; preservada como
                      INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION_v1.1.md)
                      1.0 (commit e6f550a; preservada como
                      INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION_v1.0.md)
Estado: ESPECIFICACION. Filer continuity CERRADO como caracterizacion
        del periodo (dictamen TOP 50 2026-09-19). Bloqueo de Gate-NIPC.2
        vuelve a coverage/thresholds (UNDEFINED).
Naturaleza: no normativo. Si hay conflicto con el Prompt Maestro, gana
            el Prompt Maestro.
Referencia: prompt v6.35, HEAD local cb1724e.

---

## Cambios respecto a v1.3

El dictamen TOP 50 (INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_TOP50_DICTAMEN.md,
2026-09-19) cerro el ciclo de caracterizacion de filer continuity y fijo
las siguientes decisiones.

### H2 - Filer continuity CERRADO como caracterizacion (Q-T50-1)

  El mini-probe top 50 confirma que las 4 discontinuidades detectadas
  son todas Vanguard (padre + Fiduciary Trust + 2 filiales nuevas). No
  se observan patrones adicionales. Ciclo cerrado.

  Consecuencia: seccion 3.15 reformulada. Filer continuity pasa a ser
  CONTROL DE INTEGRIDAD / DIAGNOSTICO, NO threshold cuantitativo.

### H3 - filer_status por presencia documental (Q-T50-2)

  `filer_status` se determina por la presencia del filing relevante en
  el periodo, NO por `SSHPRNAMT > 0`.

  Dimension separada anadida: `position_mass_status` con 3 estados:
    HAS_SHARES | ZERO_SHARES | NO_CANONICAL_HOLDINGS.

### H4 - NT -> HR como evidencia documental, no economica (Q-T50-3)

  La relacion NT -> OTHERMANAGER -> HR queda como evidencia documental.
  NO se autoriza inferencia economica. C2 sin cambios.

### H5 - GHISALLO como control de calidad (Q-T50-4)

  GHISALLO (CIK 0001825214, +765%) se mantiene como observacion. Probe
  diagnostico autorizado, no bloqueante.

### H6 - Siguiente fase: coverage baseline + OpenFIGI (Q-T50-5)

  El bloqueo de Gate-NIPC.2 vuelve a coverage/thresholds. Seccion 14
  actualizada con el nuevo orden.

---

## Cambios respecto a v1.2

El dictamen del auditor sobre el probe end-to-end (INSTITUTIONAL_
ACCUMULATION_NIPC_GATE0_PROBE_DICTAMEN.md, 2026-09-19) identifico un
hallazgo estructural no cubierto por el contrato: la discontinuidad de
filer CIK entre periodos. Este hallazgo se documenta aqui como
limitacion conocida + dimension independiente.

### H1 - Hallazgo filer continuity (Q-PROBE-1 a Q-PROBE-5)

  Anadida seccion 3.15.

  El match key C2 mide continuidad del FILER CIK, no continuidad
  economica del conjunto reportado. La SEC contempla estructuras donde
  un manager presenta 13F-NT (aviso de que otro manager reporta sus
  posiciones). Caso Vanguard Q4 2025 -> Q1 2026: 13F-HR unico pasa a
  13F-NT padre + 2 filiales 13F-HR. Los CUSIPs no cambian. El match key
  C2 produce EXIT + NEW legitimamente segun contrato.

  Consecuencias:
  - C2 NO se modifica. Permanece vigente.
  - Se anade seccion 3.15 como dimension independiente.
  - NIPC_COVERAGE_POLICY.md recibe seccion de filer continuity.
  - Mini-probe Q-PROBE-5 ejecutado antes de fijar thresholds.
  - Reconciliacion NT <-> HR: NO AUTORIZADA todavia.
  - Cualquier inferencia economica por parent/child: PROHIBIDA
    (economic_owner_cik sigue en FORBIDDEN_TERMS).

### Estado del ciclo

  Gate-NIPC.2: BLOQUEADO.
  Gate-NIPC.3: NO AUTORIZADO.
  Thresholds: UNDEFINED.
  C2: VIGENTE.
  Reconciliacion NT-HR: NO AUTORIZADA.

---

## Cambios respecto a v1.1

El dictamen del auditor sobre la v1.1 (INSTITUTIONAL_ACCUMULATION_NIPC_
ESPECIFICACION_V11_DICTAMEN.md, 2026-09-19) aprobo la arquitectura como
PASS CONDICIONADO y exigio 2 correcciones semanticas antes de autorizar
Gate-NIPC.2.

### C1-revisada (Q-ESP-V11-1) - canonical_security NO es ticker

  Reescrita seccion 3.13.

  Prohibido usar `ticker:<ticker>` como canonical_security definitivo:
  el ticker cambia en corporate actions, el FIGI no.

  Introducidas dos capas de estado independientes:

    security_resolution_status  (5 estados: CANONICAL, OBSERVED_ONLY,
                                 UNRESOLVED, AMBIGUOUS, CONFLICT)

    canonical_security_kind     (4 estados: CANONICAL_FIGI,
                                 CANONICAL_EQUIVALENCE,
                                 OBSERVED_CUSIP_ONLY, UNRESOLVED)

  `observed_security_key = cusip:<CUSIP>` es la clave tecnica por periodo.

  `canonical_security` es NULL salvo status == CANONICAL.

  Sin equivalence entry en data/mappings/cusip_equivalence.csv ->
  NO inferir equivalencia.

  Bug conceptual evitado:

    Sin equivalence entry:
      Q4: CUSIP_A -> observed_security_key = cusip:A
                     canonical_security    = NULL
      Q1: CUSIP_B -> observed_security_key = cusip:B
                     canonical_security    = NULL
      match_status = UNRESOLVED_IDENTITY para ambos
      -> NIPC no fabrica NEW + EXIT

    Con equivalence entry (CUSIP_A == CUSIP_B == canonical_X):
      Q4: CUSIP_A -> canonical_security = equity:X, shares=100
      Q1: CUSIP_B -> canonical_security = equity:X, shares=120
      match_status = BOTH
      -> NIPC = +20

### C4-revisada (Q-ESP-V11-4) - identidad vs metadata temporal

  Reescritas secciones 3.6, 7.2 y 7.6.

  Separacion obligatoria:

    identity temporal validity  !=  metadata temporal validity

  - `shareClassFIGI` / `figi` es ESTABLE. Una resolucion OpenFIGI actual
    puede identificar correctamente una security que ya existia en el
    report_period.
  - `ticker` actual NO se asume igual al ticker del report_period.

  Regla congelada (cita literal del dictamen v1.1):

    "La fecha de consulta OpenFIGI no invalida por si sola una
     identidad FIGI estable; si impide asumir automaticamente que los
     atributos temporales de la respuesta actual eran validos en el
     report_period."

  `TEMPORAL_UNVERIFIED` se mantiene, pero solo para METADATA (ticker),
  no para identidad FIGI.

### Pendiente antes de Gate-NIPC.2

  Politica de thresholds: ver NIPC_COVERAGE_POLICY.md.

---

## Cambios respecto a v1.0

El dictamen del auditor sobre la v1.0 (INSTITUTIONAL_ACCUMULATION_NIPC_
ESPECIFICACION_DICTAMEN.md, 2026-09-19) aprobo la arquitectura como
PASS CONDICIONADO y exigio 5 correcciones obligatorias antes de
autorizar Gate-NIPC.2. Esta v1.1 las incorpora.

### C1 - canonical_security explicito (dictamen Q-ESP-2)

  Anadida seccion 3.13.
  Distincion obligatoria entre:

      observed_security_identifier  (CUSIP del 13F)
      canonical_security            (identidad estable)

  Capa obligatoria: (report_period, observed_CUSIP) -> canonical_security.

  Sin esta capa, un cambio de CUSIP entre Q4 y Q1 se interpretaria
  erroneamente como Exit + New.

### C2 - Match key interperiodo fijada (dictamen Q-ESP-5)

  Actualizada seccion 11.2.

  compute_delta_shares() ya no acepta `match_keys=None`.

  Clave de matching fijada en contrato:

      (filing_manager_cik, canonical_security, discretion_type)

  SIN report_period. El report_period pertenece a cada snapshot, no a la
  union entre snapshots.

  Salida incluye match_status con al menos: BOTH | NEW | EXIT |
  UNRESOLVED_IDENTITY.

### C3 - Coverage pairwise (dictamen Q-ESP-7)

  Anadida seccion 3.14.
  Actualizada seccion 10.

  El coverage de un periodo no es suficiente para un calculo entre dos
  periodos. Se anaden:

      coverage_previous
      coverage_current
      paired_security_coverage
      paired_weighted_share_coverage
      unmapped_weight_previous
      unmapped_weight_current

### C4 - TEMPORAL_UNVERIFIED como politica (dictamen Q-ESP-8)

  Actualizadas secciones 3.6 y 7.2.

  Politica congelada: OpenFIGI actual NO equivale automaticamente a
  mapping historico. Si no se demuestra valida para el report_period:

      mapping_status = TEMPORAL_UNVERIFIED
      -> fuera del operational_universe
      -> diagnostico unicamente

### C5 - security_type con source + status (dictamen seccion 13)

  Actualizada seccion 3.3.

  Anadidos:

      security_type_source
      security_type_status (RESOLVED_EQUITY | RESOLVED_NON_EQUITY |
                            UNRESOLVED | CONFLICT)

### Correcciones adicionales aprobadas

  - section13f_eligible como 5 estados (NOT_IN_LIST | DELETED | ADDED |
    ACTIVE | CONFLICT), no booleano puro (dictamen seccion 14).
  - Tests: ampliados de 14 a 18 familias minimas (dictamen Q-ESP-6).
  - Parser SEC 13(f) ubicado en identity/sec13f_list.py (dictamen
    Q-ESP-9).

### NO aprobado en v1.1

  - Threshold >=90% como umbral contractual. Thresholds siguen
    UNDEFINED; se fijan antes de Gate-NIPC.2.
  - Implementacion de codigo antes del dictamen favorable a v1.1.

---

## 0. Resumen ejecutivo

NIPC (Net Institutional Position Change) se define contractualmente como:

    NIPC = sum_i DeltaShares_i

    DeltaShares_i,t = Shares_i,t - Shares_i,t-1

es decir, la variacion trimestral de acciones largas reportadas en
Form 13F para un universo de securities y managers resuelto por
identidad canonica.

Esta especificacion fija:

  - las 12 dimensiones del contrato interno de identidad y elegibilidad.
  - la entidad `canonical_security` (identidad estable) y su derivacion
    desde `observed_security_identifier`.
  - la entidad `reported_position_unit` (unidad de agregacion).
  - los filtros canonicos (SH + PUTCALL null).
  - el universo operativo (radar ^ 13F eligible ^ mapped ^ EQUITY) vs
    tecnico (13F SH + null).
  - la capa de mapping multicapa (internal -> OpenFIGI -> unresolved).
  - la politica de temporalidad (TEMPORAL_UNVERIFIED).
  - la integracion de la Official List SEC 13(f) (eligibility normativa
    con 5 estados).
  - el coverage pairwise (previous + current + paired).
  - la trazabilidad (snapshot + manifest).
  - la metodologia de coverage thresholds (sin prefijar umbrales).
  - las interfaces publicas previstas.
  - las prohibiciones explicitas.
  - la secuencia Gate-NIPC.2 -> .3 -> .4.

Estado actual de NIPC: INSUFFICIENT. El crosswalk interno actual
(etf_holdings + 3 excepciones curadas) cubre 90.91% de CUSIPs del radar
USA pero solo ~30% ponderado por SSHPRNAMT. No se publica NIPC productivo
hasta que la cobertura ponderada supere los controles contractuales.

---

## 1. Alcance y naturaleza

### 1.1. Que SI define esta especificacion

  - El contrato de identidad y clasificacion de securities del universo 13F.
  - El contrato de canonical_security (identidad estable).
  - El contrato de manager canonicalization (via identity/relationships).
  - El contrato de agregacion (reported_position_unit).
  - El contrato de matching interperiodo.
  - El contrato de mapping multicapa y su trazabilidad.
  - El contrato de integracion con la Official List SEC 13(f).
  - El contrato de temporalidad (TEMPORAL_UNVERIFIED).
  - Los criterios de status (READY | INSUFFICIENT | CONFLICT | AMBIGUOUS |
    UNRESOLVED | TEMPORAL_UNVERIFIED).
  - Los tests contractuales a escribir en Gate-NIPC.2.

### 1.2. Que NO define

  - Umbrales numericos de coverage. Solo la metodologia.
  - Implementacion (nombres definitivos de funciones, modulo aggregation/).
  - Politica de publicacion en el reporte. Esa decision es posterior.
  - Tratamiento de periodos futuros (Q2 2026+). Se hace por extension.
  - Como obtener evidencia historica de OpenFIGI (politica si, diseno no).

### 1.3. Alcance temporal del primer ciclo

  Baseline: 2025Q4 (PERIODOFREPORT=2025-12-31).
  Target:   2026Q1 (PERIODOFREPORT=2026-03-31).

  DeltaShares se calcula entre ambos snapshots canonicos.

---

## 2. Referencias normativas

### 2.1. Dictamenes vinculantes aplicados

| Fecha       | Documento                                                       | Aporta                                                     |
|-------------|-----------------------------------------------------------------|------------------------------------------------------------|
| 2026-09-19  | INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_DICTAMEN.md               | reported_position_unit; SOLE+DFND+OTR; universo vs tecnico |
| 2026-09-19  | INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_MAPPING_DICTAMEN.md       | 12 dimensiones; 4 propiedades separadas; INSUFFICIENT      |
| 2026-09-19  | INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_FIGI_DICTAMEN.md          | FIGI descartado como pivote; intra vs cross-filing         |
| 2026-09-19  | INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_OPENFIGI_DICTAMEN.md      | Precedencia multicapa; trazabilidad                        |
| 2026-09-19  | INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_SEC13F_DICTAMEN.md        | Layout 80-char; Option Indicator; STATUS                   |
| 2026-09-19  | INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_SEC13F_MINIPROBE_DICTAMEN.md | TXT Q4 subset de PDF; Opcion A aprobada                 |
| 2026-09-19  | INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION_DICTAMEN.md      | 5 correcciones obligatorias (C1-C5)                        |

### 2.2. Documentos previos en el repo

  - INSTITUTIONAL_ACCUMULATION_CONTRATO.md (v1.1).
  - INSTITUTIONAL_ACCUMULATION_PROPUESTA.md.
  - INSTITUTIONAL_ACCUMULATION_DICTAMEN_GATE0.md.
  - INSTITUTIONAL_ACCUMULATION_GATE_FA2_DICTAMEN.md.
  - INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION_v1.0.md (preservada).

### 2.3. Codigo existente reutilizable

  - src/institutional_accumulation/sec_13f/identity/temporal_filter.py
  - src/institutional_accumulation/sec_13f/identity/cusip_resolver.py
  - src/institutional_accumulation/sec_13f/identity/relationships.py
  - src/institutional_accumulation/sec_13f/identity/amendments.py

---

## 3. Las 12 dimensiones del contrato interno

El dictamen de Gate 0 Mapping exigio definir 12 conceptos para el contrato
interno. Se definen aqui uno por uno. Las secciones 3.13 y 3.14 anaden
dos dimensiones derivadas (canonical_security y coverage pairwise),
exigidas por el dictamen Gate-NIPC.1.

### 3.1. security_identity

Descripcion: identificador canonico de una security, independiente del
CUSIP declarado por cada filer.

Fuentes de identidad, por orden de autoridad:

  1. SEC 13F (CUSIP + TITLEOFCLASS + NAMEOFISSUER declarados en INFOTABLE).
  2. Crosswalk interno verificado (data/mappings/cusip_ticker_exceptions.csv
     + data/etf_holdings.csv).
  3. OpenFIGI via /v3/mapping (resuelve CUSIP -> FIGI, ticker, name,
     securityType, marketSector, exchCode, shareClassFIGI).
  4. UNRESOLVED si ninguna resuelve.

Representacion: string `canonical_security_id` con formato:

    cusip:<9-char-CUSIP>           -- si el CUSIP es la identidad observada
    cusip:<CUSIP>|figi:<FIGI>      -- cuando OpenFIGI confirma la FIGI
    unresolved:<CUSIP>             -- si no se pudo resolver

Nota: el FIGI del 13F (declarado por el filer) NO se usa como pivote.
Solo la FIGI devuelta por OpenFIGI al resolver el CUSIP.

Prohibiciones:
  - NO construir canonical_security_id por matching de NAMEOFISSUER
    (heuristica textual prohibida).
  - NO usar el FIGI del INFOTABLE como canonical_security_id.

### 3.2. section13f_eligibility

Descripcion: estado de un CUSIP respecto de la Official List of Section
13(f) Securities publicada por SEC para el trimestre.

Fuente: fichero .txt oficial del trimestre. Para el primer ciclo:

  - Q4 2025: D:/13f_probe/official_list_13f/13flist_2025Q4.txt
  - Q1 2026: D:/13f_probe/official_list_13f/13flist_2026q1.txt

Layout del fichero (80 chars fijos, LF):

  Pos 01-09: CUSIP Number.
  Pos 10:    Option Indicator (`*` o espacio).
  Pos 11-40: Issuer Name.
  Pos 41-67: Issuer Description.
  Pos 68-70: STATUS (3 chars: `*A*` additions, `*D*` deletions, blank no change).
  Pos 71-79: Blank.
  Pos 80:    Misc / Unused. NO INTERPRETAR.

Estados internos (5, no booleano puro; dictamen Gate-NIPC.1 seccion 14):

    NOT_IN_LIST   -- el CUSIP no aparece en la Official List del periodo.
    DELETED       -- aparece con STATUS = *D*.
    ADDED         -- aparece con STATUS = *A*.
    ACTIVE        -- aparece con STATUS = blank (sin cambio).
    CONFLICT      -- mismo CUSIP con estados incompatibles.

Booleano derivable:

    section13f_eligible = (estado in {ADDED, ACTIVE})

El CUSIP de opciones (pos 10 = espacio y issuer_description = CALL/PUT) NO
participa en el calculo de eligibility del universo SH + null.

Prohibiciones:
  - NO interpretar pos 80.
  - NO usar "ultimo registro gana" en lugar de STATUS.
  - NO confundir con `security_type=EQUITY` (dimensiones independientes).
  - NO colapsar a booleano antes de resolver CONFLICT.

### 3.3. security_type

Descripcion: clasificacion economica del instrumento. Valores permitidos:

    EQUITY          -- accion ordinaria, incluidas clases (CL A, CL B...).
    NON_EQUITY      -- preferentes, bonos, convertibles, units, warrants,
                       rights, depositary receipts no-equity.
    ETF             -- exchange-traded fund (dimension tecnica, distinta
                       de equity individual).
    UNKNOWN         -- clasificacion no determinable con evidencia actual.

#### Autoridad y status (dictamen Gate-NIPC.1 seccion 13)

Cada clasificacion se anota con dos campos adicionales:

    security_type_source   -- de donde viene la clasificacion.
    security_type_status   -- estado de la resolucion.

security_type_source valores:

    TITLEOFCLASS_13F       -- inferida desde TITLEOFCLASS del 13F.
    OPENFIGI_METADATA      -- inferida desde marketSector + securityType
                              de OpenFIGI.
    INTERNAL_CURATED       -- resuelta por curacion manual.
    COMBINED               -- mas de una fuente coincide.

security_type_status valores (4):

    RESOLVED_EQUITY        -- clasificacion = EQUITY con evidencia.
    RESOLVED_NON_EQUITY    -- clasificacion = NON_EQUITY con evidencia.
    UNRESOLVED             -- sin evidencia suficiente.
    CONFLICT               -- fuentes discrepan.

Casos especiales:

  - SSHPRNAMTTYPE=SH + PUTCALL=null + TITLEOFCLASS=CONVERTIBLE BOND
    -> security_type = NON_EQUITY, status = RESOLVED_NON_EQUITY.
  - Fuentes discrepan (TITLEOFCLASS dice COM, OpenFIGI dice bond)
    -> status = CONFLICT, no auto-resuelve.

Regla congelada: TITLEOFCLASS NO es una allowlist de equity (dictamen
Q-NIPC-8). Se usa como uno de varios inputs, no como autoridad unica.

Regla congelada: OpenFIGI NO es autoridad automatica. Su metadata
refuerza pero no decide por si sola.

### 3.4. ticker_mapping

Descripcion: mapping de CUSIP -> ticker del universo radar cuando exista.
Se resuelve exclusivamente por capas (ver seccion 7).

  internal_verified: data/mappings/cusip_ticker_exceptions.csv (curado) +
                     data/etf_holdings.csv (526 filas; 519 CUSIPs unicos).
  openfigi_fallback: respuesta de /v3/mapping con campo ticker + exchCode.
  unresolved:        no hay mapping.

Normalizacion de ticker devuelto por OpenFIGI:

  - OpenFIGI usa ticker tipo Bloomberg: `BRK/A`, `BRK/B`.
  - El radar usa ticker tipo Yahoo: `BRK-A`, `BRK-B`.
  - Conversion: replace("/", "-") en el ticker.

Regla congelada: OpenFIGI NO es autoridad automatica. Cuando discrepa con
internal, se genera CONFLICT explicito y se preservan ambas evidencias.

Regla congelada (C4 del dictamen): toda resolucion OpenFIGI cuyo
`mapping_timestamp_utc` sea posterior al `report_period` se marca como
TEMPORAL_UNVERIFIED salvo que exista evidencia documental de validez
historica. Un ticker resuelto hoy NO se asume valido para 2025-12-31 ni
2026-03-31 sin verificacion.

Prohibicion: no asignar ticker por proximidad de NAMEOFISSUER sin fuente
documental.

### 3.5. source_precedence

Descripcion: orden de autoridad entre fuentes cuando dos o mas resuelven
el mismo CUSIP.

Regla de precedencia congelada (dictamen Gate 0 OpenFIGI):

  1. Internal verified mapping (curado + etf_holdings).
  2. OpenFIGI.
  3. UNRESOLVED.

Cuando internal y OpenFIGI discrepan:

    CONFLICT

y se preservan AMBAS evidencias. No hay "gana el ultimo". El conflicto
queda registrado para revision.

Fuentes secundarias (no participan en precedencia de ticker):

  - SEC Official List: solo aporta `section13f_eligible`. No aporta ticker.
  - TITLEOFCLASS del 13F: solo aporta clasificacion de instrumento.
  - marketSector / securityType de OpenFIGI: refuerzan security_type.

### 3.6. temporal_validity

Descripcion: cada resolucion de mapping esta atada a una fecha de
vigencia. La fecha de consulta OpenFIGI no invalida por si sola una
identidad FIGI estable; si impide asumir automaticamente que los
atributos temporales de la respuesta actual eran validos en el
report_period.

#### Distincion obligatoria (C4-revisada, Q-ESP-V11-4)

    identity temporal validity
            !=
    metadata temporal validity

  - Identidad (FIGI/shareClassFIGI): ESTABLE. OpenFIGI documenta que
    un FIGI asignado no cambia ante corporate actions. Una resolucion
    actual puede identificar correctamente una security que ya
    existia en 2025-12-31 o 2026-03-31.

  - Metadata (ticker, name, exchCode): NO estable. El ticker actual
    NO se puede asumir como ticker del report_period sin evidencia
    adicional.

#### Politica congelada

    OpenFIGI actual != mapping historico automatico

Si una resolucion no puede demostrarse valida para el report_period
en su METADATA (ticker):

    mapping_metadata_status = TEMPORAL_UNVERIFIED

La identidad FIGI NO se invalida por la fecha de consulta. Solo la
metadata temporal queda sujeta a verificacion.

#### Efecto en el operational_universe

    metadata TEMPORAL_UNVERIFIED + FIGI presente
        -> canonical_security = figi:<shareClassFIGI>
        -> entra en operational_universe con metadata marcada

    metadata TEMPORAL_UNVERIFIED + sin FIGI
        -> status = UNRESOLVED o OBSERVED_ONLY
        -> fuera de operational_universe
        -> diagnostico

#### Registro

`mapping_timestamp_utc` se registra en cada resolucion. Si es
posterior al report_period, la METADATA (ticker) cae por defecto en
TEMPORAL_UNVERIFIED hasta validacion. La identidad FIGI se preserva.

#### Pendiente de diseno en Gate-NIPC.2

  - Como obtener evidencia historica del ticker (si fuera posible).
  - Reglas de conversion de candidato temporal a mapping historico.
  - Tratamiento de corporate actions (cambio de CUSIP, spin-off).

Prohibicion: no asumir que un ticker resuelto hoy era el mismo en
Q1 2026. La identidad FIGI si puede reutilizarse.

### 3.7. one_to_one / ambiguous

Descripcion: clasificacion de la cardinalidad del mapping.

Estados:

  ONE_TO_ONE      -- un CUSIP -> un ticker, sin ambiguedad.
  ONE_TO_MANY     -- un CUSIP -> varios tickers (clases, mercados, cambios).
  CONFLICTING     -- internal vs OpenFIGI devuelven tickers distintos.
  UNRESOLVED      -- ninguna fuente resuelve.

Origen de la ambiguedad:

  - Clases multiples (BRK-A/BRK-B, GOOG/GOOGL, MOG-A/MOG-B).
  - ADR / ordinaria en distintos mercados.
  - Cambios de ticker historicos.
  - Filings con CUSIP mal formado.

Regla congelada:

  - ONE_TO_MANY y CONFLICTING NO se resuelven automaticamente.
  - Se registran como AMBIGUOUS o CONFLICT segun corresponda.
  - Solo ONE_TO_ONE participa en el operational_universe.

### 3.8. mapping_coverage

Descripcion: metrica de cobertura por numero de securities.

Definicion:

    mapping_coverage = CUSIPs con ticker resuelto / CUSIPs universo

Segmentacion obligatoria (no una sola cifra):

  A. Universo 13F completo (SH + null).
  B. Universo radar USA (radar ^ 13F).
  C. Interseccion tecnica (radar ^ 13F ^ section13f_eligible).

Valores de referencia (Q1 2026, crosswalk actual):

  A: 502/24,838 = 2.021%
  B: 220/242   = 90.91%
  C: pendiente (requiere integracion SEC 13(f)).

Prohibicion: NO publicar una sola cifra de coverage como "cobertura NIPC".
Se reportan siempre las tres, con nota de que la operativa es C.

### 3.9. weighted_share_coverage

Descripcion: cobertura ponderada por masa accionarial reportada, no por
numero de securities.

Definicion:

    weighted_share_coverage = sum(SSHPRNAMT de securities mapeadas)
                            / sum(SSHPRNAMT del universo)

Segmentacion obligatoria (paralela a 3.8):

  A. Universo 13F completo.
  B. Universo radar USA.
  C. Interseccion tecnica.

Valores de referencia (Q1 2026):

  A: 39.179% (bruto) / 41.005% (sin no-equity)
  B: 28.660% (bruto) / 29.995% (sin no-equity)
  C: pendiente integracion SEC 13(f).

Distincion frente a mapping_coverage: puede haber alta cobertura por
securities y baja por shares (caso actual: 90.91% vs 28.66%). Ambas se
reportan siempre juntas.

VALUE (market value) NO se usa como ponderador para NIPC. Solo SSHPRNAMT.
VALUE queda como control secundario informativo.

### 3.10. unmapped_weight

Descripcion: la masa accionarial NO cubierta por el mapping, expresada
como porcentaje del total.

Definicion:

    unmapped_weight = 1 - weighted_share_coverage

Segmentacion paralela a 3.8 y 3.9.

Valores de referencia (Q1 2026):

  A: 60.821% (bruto) / 58.995% (sin no-equity)
  B: 71.340% (bruto) / 70.005% (sin no-equity)

Lectura critica: el 70% del SSHPRNAMT del universo radar USA esta
actualmente sin mapping. Es el gap estructural principal.

Umbral: el porcentaje maximo de unmapped_weight aceptable para publicar
NIPC productivo sera fijado por el auditor en funcion de los controles
contractuales. La especificacion NO fija numeros; define que la metrica
se reporta siempre.

### 3.11. operational_universe

Descripcion: universo efectivo sobre el que se calcula NIPC productivo.

Definicion (contrato Q5 + dictamen Gate-NIPC.0):

    operational_universe
    = radar_equities
    ^ section13f_eligible
    ^ ticker_mapped
    ^ security_type == EQUITY

Es decir:

  1. CUSIP esta en el universo radar (radar_equities).
  2. CUSIP esta en la Official List SEC del trimestre.
  3. CUSIP tiene ticker resuelto.
  4. CUSIP es equity segun security_type (no convertible bond, no unit,
     no PFD, no ETF).

Distincion explicita:

  - mapped != equity != section13f_eligible != radar_member.
  - Son cuatro propiedades independientes.

Universo tecnico (para diagnostico): todos los CUSIPs con SH + null,
sin filtro de radar ni de eligibility. Se usa para medir el motor, no
se publica.

### 3.12. status

Descripcion: estado final del calculo NIPC para un universo/periodo dado.

Valores permitidos (6 estados):

    READY                -- cobertura suficiente, ambos controles
                            contractuales (mapping_coverage +
                            weighted_share_coverage) superan umbrales.
                            Publicable.
    INSUFFICIENT         -- cobertura por debajo de umbral. Calculo de
                            diagnostico disponible, no publicable como
                            senal.
    CONFLICT             -- existen discrepancias no resueltas entre
                            fuentes (internal vs OpenFIGI). Requiere
                            intervencion.
    AMBIGUOUS            -- CUSIP con multiples candidatos one_to_many no
                            resueltos. Casos concretos.
    UNRESOLVED           -- CUSIP sin ninguna fuente que lo resuelva.
    TEMPORAL_UNVERIFIED  -- resolucion no demostrablemente valida para el
                            report_period. Fuera del operational_universe.

Regla por defecto: STATUS = INSUFFICIENT mientras los controles de
cobertura no se satisfagan.

Estado actual (Q1 2026): INSUFFICIENT.

Cambio de estado: solo el auditor puede promover a READY tras validacion
formal en Gate-NIPC.3.

Prohibicion: NUNCA publicar un valor NIPC como senal operativa si STATUS
!= READY.

### 3.13. canonical_security

Descripcion: identidad ESTABLE de una security a lo largo del tiempo,
independiente del CUSIP observado en cada filing trimestral.

Introducida por dictamen Gate-NIPC.1 v1.0 (Q-ESP-2) y reformulada por
dictamen Gate-NIPC.1 v1.1 (Q-ESP-V11-1). Correccion obligatoria
C1-revisada.

#### Distincion obligatoria

    observed_security_identifier
        = CUSIP declarado por el filer en el INFOTABLE
        = clave por periodo

    observed_security_key
        = "cusip:<CUSIP>"
        = clave tecnica derivada del observed_security_identifier

    canonical_security
        = identidad ESTABLE de la security
        = clave interperiodo

Son objetos distintos. El primero es lo que dice el filing. El segundo
es su representacion tecnica. El tercero es la identidad canonica
resuelta por evidencia.

#### Prohibicion explicita (C1-revisada)

NO se permite `ticker:<ticker>` como canonical_security definitivo.
OpenFIGI documenta que un FIGI no cambia ante corporate actions,
mientras el ticker asociado puede cambiar. Un ticker no es identidad
estable.

Representaciones permitidas de canonical_security:

    figi:<shareClassFIGI>   -- identidad resuelta via OpenFIGI
    equity:<canonical_id>   -- identidad resuelta via equivalence entry
    (NULL)                  -- sin identidad canonica resuelta

El CUSIP observado NUNCA se presenta como canonical_security estable
sin evidencia. Permanece como observed_security_key (tecnico).

#### Dos capas de estado independientes

**Capa 1 - security_resolution_status** (5 estados):

    CANONICAL     -- identidad resuelta con evidencia suficiente
                     (FIGI o equivalence entry)
    OBSERVED_ONLY -- solo CUSIP observado, sin evidencia de identidad
                     estable. NO participa en matching interperiodo.
    UNRESOLVED    -- sin resolucion
    AMBIGUOUS     -- multiples candidatos sin criterio automatico
    CONFLICT      -- fuentes discrepan

**Capa 2 - canonical_security_kind** (4 estados):

    CANONICAL_FIGI          -- canonical_security = figi:<shareClassFIGI>
    CANONICAL_EQUIVALENCE   -- canonical_security = equity:<canonical_id>
    OBSERVED_CUSIP_ONLY     -- canonical_security = NULL
    UNRESOLVED              -- canonical_security = NULL

#### Regla de matching interperiodo

    if security_resolution_status == CANONICAL:
        canonical_security puede usarse en la match key interperiodo
    else:
        canonical_security = NULL
        match_status = UNRESOLVED_IDENTITY
        -> no genera delta artificial

#### Fuentes para construir canonical_security

Por orden de autoridad:

  1. Internal verified + ticker estable -- canonical_security =
     equity:<canonical_id> (kind=CANONICAL_EQUIVALENCE).
  2. OpenFIGI shareClassFIGI -- canonical_security =
     figi:<shareClassFIGI> (kind=CANONICAL_FIGI).
  3. Sin evidencia -- canonical_security = NULL
     (kind=OBSERVED_CUSIP_ONLY o UNRESOLVED).

Nota: `ticker:<ticker>` NO es opcion valida en ninguno de los tres
casos.

#### Tabla de equivalencias temporales

Cuando el CUSIP cambia entre periodos por corporate action:

    -> se requiere tabla de equivalencia temporal.
    -> la tabla vive en data/mappings/cusip_equivalence.csv.
    -> poblacion manual curada, evidencia documental obligatoria.
    -> Sin entrada en la tabla: los CUSIPs son securities distintas
       (conservador).

Esquema obligatorio de la tabla:

    CUSIP_A, canonical_security, valid_from, valid_to,
    source, reason, verified_by, source_document (opcional)

Ejemplo conceptual del bug evitado:

    Sin equivalence entry:
      Q4: CUSIP_A -> observed_security_key = cusip:A
                    canonical_security    = NULL
                    security_resolution_status = OBSERVED_ONLY
      Q1: CUSIP_B -> observed_security_key = cusip:B
                    canonical_security    = NULL
                    security_resolution_status = OBSERVED_ONLY
      match_status = UNRESOLVED_IDENTITY para ambos
      -> NIPC no fabrica NEW + EXIT

    Con equivalence entry (CUSIP_A == CUSIP_B == canonical_X):
      Q4: CUSIP_A -> canonical_security = equity:X, shares=100
      Q1: CUSIP_B -> canonical_security = equity:X, shares=120
      match_status = BOTH
      -> NIPC = +20

#### Prohibiciones

  - NO asumir equivalencia por NAMEOFISSUER + TITLEOFCLASS.
  - NO emparejar por proximidad de CUSIP (prefijos 6-7 caracteres).
  - NO emparejar entre periodos sin evidencia documental.
  - NO usar ticker como canonical_security.
  - NO inferir equivalence sin entrada explicita en la tabla.

### 3.14. coverage_pairwise

Descripcion: cobertura de comparabilidad entre dos periodos consecutivos,
no solo de un periodo aislado.

Introducida por dictamen Gate-NIPC.1 (Q-ESP-7), correccion obligatoria
C3.

#### Razon

El coverage de un periodo por separado no es suficiente. Un CUSIP puede
estar mapeado en Q1 y no en Q4. Su DeltaShares no es observable aunque
exista un mapping en Q1. Por tanto, la cobertura relevante para NIPC es
la cobertura de PARES comparables.

#### Metricas obligatorias

    coverage_previous              -- Q4 2025: mapped / total
    coverage_current               -- Q1 2026: mapped / total
    paired_security_coverage       -- securities en ambos periodos con
                                      ticker resuelto en ambos
    paired_weighted_share_coverage -- ponderado por SSHPRNAMT de las
                                      securities emparejadas
    unmapped_weight_previous       -- 1 - coverage ponderado Q4
    unmapped_weight_current        -- 1 - coverage ponderado Q1

#### Definicion formal

    paired_security_coverage
      = |{ S : S tiene mapping en Q4 Y en Q1 }|
        / |{ S : S existe en Q4 O en Q1 }|

    paired_weighted_share_coverage
      = sum(SSHPRNAMT de S con mapping en ambos)
        / sum(SSHPRNAMT de S con posicion en ambos)

#### Umbral contractual

Los umbrales se fijan antes de Gate-NIPC.2. La especificacion solo define
las metricas. Thresholds = UNDEFINED por ahora.

#### Regla de publicacion

Solo `paired_*` determina si un NIPC es publicable. Si la cobertura
pairwise es baja, NIPC sera INSUFFICIENT aunque coverage_current sea alta.

---

### 3.15. filer_continuity

Descripcion: continuidad del FILER CIK entre periodos. Dimension de
CARACTERIZACION / CONTROL DE INTEGRIDAD, no de threshold.

Ciclo CERRADO por dictamen TOP 50 (Q-T50-1). Se mantiene como dimension
observable y diagnostica.

#### Estado del ciclo (dictamen TOP 50, 2026-09-19)

  TOP 20: 3/24 discontinuidades (12.5%). Todas Vanguard.
  TOP 50: 4/54 discontinuidades (7.4%). Todas Vanguard.
  Union TOP 50: 54 CIKs, 50 continuos, 4 discontinuos, 0 unresolved.

  Interpretacion congelada: la discontinuidad observada esta concentrada
  en el grupo Vanguard durante este periodo concreto. NO se extrapola al
  universo 13F. NO se convierte en threshold.

#### Dimensiones separadas (Q-T50-2)

  filer_status (por presencia documental del filing, NO por shares):
    CONTINUOUS_FILER       filing presente en ambos periodos
    FILER_DISCONTINUITY    filing presente en Q4 y ausente en Q1 (o viceversa)
    UNRESOLVED             no determinable

  position_mass_status (dependiente de SSHPRNAMT, ortogonal a filer_status):
    HAS_SHARES             shares > 0 en el periodo
    ZERO_SHARES            filing presente pero shares = 0
    NO_CANONICAL_HOLDINGS  filing NT (13F-NT / 13F-NT/A), sin holdings

  Caso NORGES BANK (CIK 0001374170): filer_status = CONTINUOUS_FILER,
  position_mass_status = ZERO_SHARES. Ambas dimensiones correctas.

  nt_to_hr_relation_observed (bool):
    True si existe NT con OTHERMANAGER target resoluble.

  nt_to_hr_relation_targets (list):
    Lista de CIKs declarados como targets NT.

#### Evidencia NT -> HR documentada

  Caso Vanguard Q4 2025 -> Q1 2026:

    CIK 0000102909 (VANGUARD GROUP INC):
      Q4 2025: 13F-HR con holdings.
      Q1 2026: 13F-NT con 10 managers declarados en OTHERMANAGER:
        0001550100, 0002100121, 0002100119, 0000933478,
        0000217448, 0001767306, 0001680208, 0001984256,
        0001811242, 0000947529.

  Confirmacion: la SEC contempla este mecanismo explicitamente. La
  relacion NT -> OTHERMANAGER -> HR queda como EVIDENCIA DOCUMENTAL.

#### Uso en NIPC

  Filer continuity NO es threshold. Es:
    - Control de integridad del NIPC observable.
    - Diagnostico del impacto de reorganizaciones de filer.
    - Alerta cuando la discontinuidad bruta supera un margen razonable.

  No se autoriza inferencia economica (mismo "propietario economico")
  por parent/child. FORBIDDEN_TERMS en relationships.py sigue vigente.

  C2 (match key) sin cambios.

#### Metricas observables (a reportar como diagnostico)

  filer_discontinuity_count          numero de CIKs con discontinuidad
  filer_discontinuity_pct            count / total CIKs observados
  filer_discontinuity_gross_shares   sum(abs(delta_shares)) de NEW+EXIT
                                     atribuibles a discontinuidad
  filer_discontinuity_concentration  top-N CIKs / total discontinuidad

  Estas metricas NO tienen threshold. Se reportan como observacion.

#### Prohibiciones explicitas

  - NO convertir filer_continuity_pct en threshold.
  - NO inferir economic_owner por parent/child.
  - NO modificar C2 por reorg de filer.
  - NO reconciliar NT <-> HR productivamente.
  - NO usar filer_status por SSHPRNAMT > 0: usar presencia documental
    del filing; usar position_mass_status para la dimension de masa.

## 4. reported_position_unit

### 4.1. Definicion

Entidad logica introducida por dictamen Gate-NIPC.0 (Q-NIPC-2). Es la
unidad de agregacion de shares para NIPC.

    reported_position_unit = (
        report_period,
        filing_manager_cik,
        canonical_security,
        discretion_type
    )

Nota: `canonical_security` es el objeto de 3.13, no el CUSIP observado.

### 4.2. Regla central

La cantidad `SSHPRNAMT` entra UNA SOLA VEZ por source line.

    source_line
        v
    SSHPRNAMT (una vez)

### 4.3. Prohibicion explicita

PROHIBIDO: multiplicar SSHPRNAMT por el numero de edges que produce
la relacion con OTHERMANAGER.

    source_line
        v
    N x SSHPRNAMT      <- PROHIBIDO

Razon: la SEC no distribuye individualmente las acciones de una linea
entre los managers enlazados. Multiplicar seria doble conteo economico.

### 4.4. Relacion con canonical_reporting_relationship_key

El key producido por `build_canonical_relationship` (filing_manager_cik +
included_manager_cik + discretion_type + report_period) es dimension de
PROVENANCE / EVIDENCIA, no unidad de agregacion directa.

Distincion congelada:

    canonical_reporting_relationship_key = IDENTIDAD / PROVENANCE
    reported_position_unit               = UNIDAD DE AGREGACION DE SHARES

Ambas coexisten. La primera enriquece el grafo de relaciones. La segunda
agrega shares.

### 4.5. Caso multi-manager

Cuando `OTHERMANAGER = "1,2,3"`:

  - La posicion conserva SSHPRNAMT = X una sola vez.
  - Se generan 3 edges (edge1, edge2, edge3) para el grafo.
  - NO se generan X + X + X para NIPC.

Consecuencia operativa: para calcular NIPC, se agrupa por
`reported_position_unit` (identidad de filing manager + canonical_security
+ discretion), NO por `canonical_reporting_relationship_key`.

### 4.6. Ejemplo

    Filing A (manager CIK=111):
      source_line_1: CUSIP=037833100 (AAPL), SSHPRNAMT=1000,
                     OTHERMANAGER="1,2", discretion=SOLE
      source_line_2: CUSIP=037833100 (AAPL), SSHPRNAMT=500,
                     OTHERMANAGER="", discretion=DFND

    reported_position_unit_1 = (2026Q1, 111, AAPL, SOLE) -> 1000 shares
    reported_position_unit_2 = (2026Q1, 111, AAPL, DFND) ->  500 shares

    NIPC_SOLE = DeltaShares(reported_position_unit_1)
    NIPC_DFND = DeltaShares(reported_position_unit_2)
    NIPC_total = NIPC_SOLE + NIPC_DFND + NIPC_OTR

    NUNCA:
      reported_position_unit_1 -> 2000 shares (x2 por los 2 managers)

Regla de agregacion final:

    NIPC = sum_i DeltaShares_i
    i recorre reported_position_unit

### 4.7. Matching interperiodo (C2, dictamen Q-ESP-5)

Para cruzar dos periodos consecutivos, la clave de matching queda FIJADA
en contrato:

    match_key = (
        filing_manager_cik,
        canonical_security,
        discretion_type
    )

SIN `report_period`.

Razon: `report_period` pertenece a cada snapshot, no a la union entre
snapshots. Incluirlo produciria un producto cartesiano incorrecto.

El cruce se hace con FULL OUTER JOIN:

    units_2025Q4
        FULL OUTER JOIN
    units_2026Q1
        ON match_key

Esto genera `match_status` con valores:

    BOTH                 -- posicion presente en ambos periodos.
    NEW                  -- solo en current (previous = 0 shares).
    EXIT                 -- solo en previous (current = 0 shares).
    UNRESOLVED_IDENTITY  -- no se puede emparejar por falta de
                            canonical_security estable.

Regla: NEW asume previous = 0; EXIT asume current = 0. NO imputar.

Prohibicion: `compute_delta_shares()` NO debe aceptar parametro
`match_keys=None`. La clave esta fijada; un consumidor no la modifica.

---

## 5. Filtros canonicos

### 5.1. Filtro base (contrato Q4)

    SSHPRNAMTTYPE == "SH"
    AND
    PUTCALL IS NULL

Excluye:

  - PRN (principal, no shares).
  - Call (opciones call).
  - Put (opciones put).

### 5.2. Discretion (dictamen Gate-NIPC.0 Q-NIPC-3)

INCLUIR SOLE + DFND + OTR. No excluir DFND.

    NIPC_total = NIPC_SOLE + NIPC_DFND + NIPC_OTR

Mantener separacion en todo momento. DFND no significa "manager no
controla nada": significa shared-defined investment discretion. Es una
posicion valida.

### 5.3. Section 13(f) eligibility

Aplicar `section13f_eligible` como filtro adicional sobre el universo
base. Un CUSIP puede estar en el 13F (SH + null) y no estar en la
Official List del trimestre. En ese caso:

    section13f_eligible = False (estado NOT_IN_LIST o DELETED)
    -> no entra en operational_universe

### 5.4. Security type

Aplicar `security_type_status == RESOLVED_EQUITY` como filtro sobre el
universo elegible. Excluye:

  - Convertible bonds (aunque pasen el filtro SH + null).
  - Units, preferreds, warrants, rights.
  - ETFs (universo tecnico, no operativo radar equity).

Estados UNRESOLVED y CONFLICT quedan fuera del operational_universe
hasta resolucion.

### 5.5. Orden de aplicacion (canonico)

    13F INFOTABLE
        v
    SH + null
        v
    section13f_eligible (estado in {ADDED, ACTIVE})
        v
    security_type_status == RESOLVED_EQUITY
        v
    ticker_mapped (mapping_status != TEMPORAL_UNVERIFIED)
        v
    operational_universe

---

## 6. Universo operativo vs universo tecnico

### 6.1. Universo tecnico

    technical_universe = 13F SH + null

Sin filtro de radar, sin filtro de eligibility. Se usa para diagnostico
del motor NIPC y para medir la capacidad de resolucion de identidad.

Tamano (Q1 2026): 24,838 CUSIPs unicos / 3,188,083 filas canonicas.

NO se publica.

### 6.2. Universo operativo

    operational_universe
    = radar_equities
    ^ section13f_eligible
    ^ ticker_mapped
    ^ security_type == EQUITY

Se usa para calcular NIPC productivo.

Tamano (Q1 2026): pendiente integracion SEC 13(f) sobre radar USA.

NO se publica hasta que cobertura ponderada supere controles.

### 6.3. Cuatro propiedades independientes

Explicitamente separadas (dictamen Gate 0 Mapping):

    mapped               -- tiene ticker resuelto
    equity               -- security_type == EQUITY
    section13f_eligible  -- esta en la Official List SEC del trimestre
    radar_member         -- ticker esta en stock_prices.parquet

Puede haber cualquier combinacion:

    mapped=True,  equity=True,  section13f_eligible=False, radar_member=True
    mapped=True,  equity=False, section13f_eligible=True,  radar_member=False
    mapped=False, equity=True,  section13f_eligible=True,  radar_member=True
    ...

Ninguna implica a las demas.

### 6.4. Regla de publicacion

    if status != READY:
        -> solo diagnostico interno
        -> NO senal operativa en el reporte

---

## 7. Capa de mapping multicapa

### 7.1. Internal verified

Fuentes:

  - data/mappings/cusip_ticker_exceptions.csv (curado, 3 filas Q1 2026:
    DD, HON, XOM).
  - data/etf_holdings.csv (526 filas, 519 CUSIPs unicos, crosswalk
    derivado de holdings ETF sectoriales).

Autoridad: primaria. Precedencia 1.

### 7.2. OpenFIGI fallback

Endpoint: https://api.openfigi.com/v3/mapping
Payload: [{"idType":"ID_CUSIP","idValue":"<CUSIP>","exchCode":"US"}]
Auth: opcional (env var OPENFIGI_API_KEY; nunca en repo).

Rate limits:

  Sin API key: 25 req/min, 5 jobs/req.
  Con API key: 25 req/6s, 100 jobs/req.

Autoridad: secundaria. Precedencia 2. Solo se consulta si internal no
resuelve.

Campos registrados por respuesta:

    input_identifier
    id_type
    resolved_figi
    resolved_security
    ticker
    security_type
    market_sector
    exch_code
    share_class_figi
    mapping_source
    mapping_timestamp_utc
    mapping_endpoint
    mapping_result_status

Estados de la respuesta:

    EXACT                  -- un unico candidato claro
    NOT_FOUND              -- sin resultado
    MULTIPLE_CANDIDATES    -- varios candidatos
    CONFLICT               -- contradice a internal
    NON_EQUITY             -- instrumento no equity confirmado
    ERROR                  -- error de formato o transporte
    RATE_LIMIT             -- 429
    TEMPORAL_UNVERIFIED    -- ver 3.6, 7.6 y C4-revisada

Matiz C4-revisada (Q-ESP-V11-4): `TEMPORAL_UNVERIFIED` aplica a la
METADATA temporal (ticker, name, exchCode), NO a la identidad FIGI.
El shareClassFIGI devuelto por OpenFIGI es estable frente a corporate
actions. Un mapping con identidad FIGI valida pero ticker no verificado
historicamente NO se descarta como identidad: se usa para
canonical_security (kind=CANONICAL_FIGI) y se marca su metadata como
TEMPORAL_UNVERIFIED.

Normalizacion de ticker: replace("/", "-") para alinear con Yahoo.
Ejemplo: BRK/A -> BRK-A.

### 7.3. UNRESOLVED

Cuando ninguna fuente resuelve el CUSIP:

    ticker_mapping = None
    status = UNRESOLVED
    -> no entra en operational_universe
    -> aparece en metricas de coverage como no mapeado

### 7.4. CONFLICT

Cuando internal y OpenFIGI devuelven tickers distintos:

    status = CONFLICT
    ambas evidencias preservadas
    no se resuelve automaticamente

Requiere intervencion (auditor o curacion manual adicional) para
promoverlo a READY.

### 7.5. AMBIGUOUS

Cuando un CUSIP tiene multiples candidatos (one_to_many) y no hay
criterio automatico para decidir:

    status = AMBIGUOUS
    -> no entra en operational_universe
    -> se registra para revision

Ejemplo tipico: clases multiples donde el CUSIP base no basta para
distinguir BRK-A de BRK-B sin contexto adicional.

### 7.6. TEMPORAL_UNVERIFIED (C4-revisada)

Politica congelada por dictamen Q-ESP-V11-4:

    OpenFIGI actual
        !=
    mapping historico automatico

Pero la regla se aplica SOLO a la METADATA temporal, no a la
identidad FIGI. Razon: OpenFIGI documenta que un FIGI no cambia ante
corporate actions; el ticker puede cambiar.

#### Distincion operativa

    identidad FIGI  -> ESTABLE, no invalidable por mapping_timestamp
    ticker actual   -> NO asumible como ticker historico

#### Efecto

Cuando la resolucion no puede demostrarse valida para el report_period
en su METADATA (ticker):

    mapping_metadata_status = TEMPORAL_UNVERIFIED
    canonical_security_kind = CANONICAL_FIGI (si FIGI presente)
    -> entra en operational_universe con metadata marcada

Si la resolucion carece tambien de FIGI estable:

    security_resolution_status = UNRESOLVED o OBSERVED_ONLY
    -> no entra en operational_universe
    -> solo diagnostico

#### Regla literal del dictamen v1.1

    "La fecha de consulta OpenFIGI no invalida por si sola una
     identidad FIGI estable; si impide asumir automaticamente que
     los atributos temporales de la respuesta actual eran validos
     en el report_period."

### 7.7. Flujo de resolucion

    CUSIP
      v
    internal verified?
      v (si)                        v (no)
    return ticker               OpenFIGI
                                    v
                              resultado?
                              v (EXACT)      v (NOT_FOUND)
                              return          UNRESOLVED
                              ticker
                              v
                              internal coincide?
                              v (si)         v (no)
                              ONE_TO_ONE     CONFLICT
                              v
                              temporal valida?
                              v (si)         v (no)
                              radar           TEMPORAL_UNVERIFIED
                              intersection

---

## 8. Integracion SEC 13(f) Official List

### 8.1. Fuente

Fichero .txt trimestral oficial SEC. Layout fixed-width 80 chars (ver
seccion 3.2).

Ubicaciones actuales:

    D:/13f_probe/official_list_13f/13flist_2025Q4.txt (Q4 2025)
    D:/13f_probe/official_list_13f/13flist_2026q1.txt (Q1 2026)
    D:/13f_probe/official_list_13f/13flist_2025Q4.pdf (evidencia auxiliar)

### 8.2. Resolucion de eligibility

Estados internos (5):

    NOT_IN_LIST   -- el CUSIP no aparece en la lista del periodo.
    DELETED       -- aparece con STATUS = *D*.
    ADDED         -- aparece con STATUS = *A*.
    ACTIVE        -- aparece con STATUS = blank.
    CONFLICT      -- mismo CUSIP con estados incompatibles.

Booleano derivable:

    section13f_eligible = (estado in {ADDED, ACTIVE})

### 8.3. Limitaciones conocidas

  - Q4 2025 .txt es subconjunto empirico del PDF Q4. Los CUSIPs base
    estan presentes; las filas CALL/PUT no.
  - NIPC no necesita CALL/PUT (ya filtra por PUTCALL IS NULL).
  - Deuda documentada: FUTURE_FORMAT_VARIANT. Si un trimestre futuro no
    publica .txt suficiente, se abrira diseno especifico de PDF parser.
    NO implementar ahora.

### 8.4. Distincion de dimensiones

    section13f_eligible  !=  security_type
    section13f_eligible  !=  ticker_mapped
    section13f_eligible  !=  radar_member

Cuatro propiedades independientes (ver seccion 6.3).

### 8.5. Parser propuesto (dictamen Q-ESP-9)

Ubicacion: src/institutional_accumulation/sec_13f/identity/sec13f_list.py.

Responsabilidad limitada:

  - Parsear fixed-width 80.
  - Extraer: CUSIP, option_indicator, issuer_name, issuer_description,
    status (3 chars).
  - Devolver: section13f_eligible, status, option_indicator, raw_line.

NO resuelve ticker ni security identity. Esa es responsabilidad de
cusip_resolver.py + OpenFIGI fallback.

---

## 9. Trazabilidad y snapshot

### 9.1. Manifest obligatorio

Cada lote de resolucion externa (OpenFIGI) debe producir un manifest
auditable con:

    batch_id
    timestamp_utc
    endpoint
    n_cusips
    n_hits
    n_errors
    sha256 del input (cusips.txt)
    sha256 del output (result.json)
    parametros de consulta (batch_size, sleep)
    api_key_used (bool, NUNCA el valor)

### 9.2. Preservacion

Formato: JSON + HASHES.txt en directorio de evidencia del ciclo.

Estructura actual (referencia):

    docs/auditoria/iae/evidence/nipc_gate0_openfigi/
        README.md
        build_sample.py
        run_openfigi.py
        sample.json
        cusips.txt
        result.json
        HASHES.txt

### 9.3. Snapshot vs estado vivo

Distincion obligatoria:

    snapshot OpenFIGI  = inmutable una vez cerrado el lote
    estado vivo        = respuestas actuales (pueden cambiar)

NIPC se calcula sobre snapshot, no sobre estado vivo.

### 9.4. Temporalidad

Cada resolucion se registra con `mapping_timestamp_utc`. El status
`TEMPORAL_UNVERIFIED` se anota mientras no exista validacion temporal.

Prohibicion: no asumir validez historica del ticker resuelto.

### 9.5. Prohibiciones de trazabilidad

  - NO commitear la API key en ningun artefacto.
  - NO omitir el sha256 en el manifest.
  - NO sobrescribir snapshots previos (cada lote uno nuevo).

---

## 10. Coverage thresholds - metodologia

### 10.1. Principio

Los umbrales numericos NO se fijan en esta especificacion. Se define
la metodologia y el auditor los fija antes de Gate-NIPC.2.

Razon: el dictamen Gate 0 Mapping rechazo convertir 90.91% (valor
observado) en umbral contractual. Los umbrales deben fijarse tras
disenar la capa de mapping y sus controles, no antes.

### 10.2. Controles simultaneos obligatorios

Contrato Q3 exige DOS controles simultaneos. Actualizado por dictamen
Gate-NIPC.1 (C3, Q-ESP-7) para operar sobre pares:

    (1) paired_security_coverage        >= THRESHOLD_1
    (2) paired_weighted_share_coverage  >= THRESHOLD_2

Ambos sobre operational_universe (no sobre universo tecnico).

Razon del cambio: la cobertura de un solo periodo no captura el caso de
una security mapeada en Q1 pero no en Q4 (delta no observable aunque
exista mapping en Q1).

### 10.3. Metricas de cobertura obligatorias (C3)

    coverage_previous              -- Q4 2025: mapped / total
    coverage_current               -- Q1 2026: mapped / total
    paired_security_coverage       -- securities con mapping en ambos
    paired_weighted_share_coverage -- ponderado por SSHPRNAMT
    unmapped_weight_previous       -- 1 - coverage ponderado Q4
    unmapped_weight_current        -- 1 - coverage ponderado Q1

Las seis metricas se reportan juntas. Nunca una sola.

### 10.4. Criterios de fijacion (recomendacion tecnica)

Los umbrales deben fijarse considerando:

  - Que el universo operativo del radar es el que importa (no 13F completo).
  - Que el sesgo large-cap es el riesgo principal del crosswalk actual.
  - Que los no mapeados pueden ser no-equity (excluidos legitimamente)
    o equity genuinamente faltante.
  - Que un umbral alcanzable al primer intento es preferible a uno
    aspiracional no alcanzable.
  - Que la cobertura pairwise es mas exigente que la cobertura de un
    solo periodo, y por tanto un umbral adecuado puede ser distinto.

Propuesta tecnica orientativa (NO aprobada, solo para discusion):

    THRESHOLD_1 (paired_security_coverage):        a determinar
    THRESHOLD_2 (paired_weighted_share_coverage):  a determinar

Nota: el dictamen Q-ESP-7 rechazo explicitamente >=90% como umbral
contractual. Los umbrales se fijan ANTES de Gate-NIPC.2.

### 10.5. Estado actual

    mapping_coverage (radar USA):         90.91% (crosswalk actual)
    weighted_share_coverage (radar USA):  29.995% (crosswalk actual)
    section13f_eligible (radar USA):      pendiente integracion
    paired_*:                             pendiente calculo

Por tanto: STATUS = INSUFFICIENT. No se publica NIPC productivo.

### 10.6. Regla de promocion

Solo el auditor promueve a READY tras validacion formal en Gate-NIPC.3.
Cambio de estado registrado en FOLLOWUPS.

---

## 11. Interfaces propuestas (conceptuales)

Esta seccion describe las interfaces PUBLICAS previstas para Gate-NIPC.2.
No incluye codigo. Los nombres definitivos se fijan al implementar.

### 11.1. Modulo aggregation/ (nuevo, aprobado por Q-NIPC-7)

    src/institutional_accumulation/aggregation/
        __init__.py
        delta_shares.py
        nipc.py

### 11.2. Interfaces publicas previstas

#### compute_reported_position_units

    compute_reported_position_units(infotable_df, submission_df,
                                    coverpage_df, *, report_period)
        -> DataFrame con columnas:
             report_period,
             filing_manager_cik,
             observed_security_key,
             security_resolution_status,
             canonical_security_kind,
             canonical_security,
             discretion_type,
             sshprnamt_total,
             n_source_lines

Nota (C1-revisada): `canonical_security` puede ser NULL si
security_resolution_status != CANONICAL. En ese caso,
`observed_security_key` (cusip:<CUSIP>) es la unica identidad tecnica.
Ver seccion 3.13.

#### compute_delta_shares

    compute_delta_shares(units_current, units_previous)
        -> DataFrame con columnas:
             filing_manager_cik,
             canonical_security,
             discretion_type,
             sshprnamt_current,
             sshprnamt_previous,
             delta_shares,
             match_status

Cambio respecto a v1.0 (C2, dictamen Q-ESP-5): la funcion ya NO acepta
parametro `match_keys=None`. La clave de matching esta FIJADA:

    (filing_manager_cik, canonical_security, discretion_type)

Y NO incluye `report_period`.

`match_status` toma valores: BOTH | NEW | EXIT | UNRESOLVED_IDENTITY.

#### compute_nipc

    compute_nipc(delta_df, *, discretion_breakdown=True)
        -> dict con:
             nipc_total,
             nipc_sole,
             nipc_dfnd,
             nipc_otr,
             coverage_security_previous,
             coverage_security_current,
             coverage_weighted_share_previous,
             coverage_weighted_share_current,
             paired_security_coverage,
             paired_weighted_share_coverage,
             unmapped_weight_previous,
             unmapped_weight_current,
             status

### 11.3. Contrato de pureza

Las funciones de aggregation/:

  - NO leen ficheros (reciben DataFrames).
  - NO escriben ficheros.
  - NO usan datetime.now() como fecha de observacion.
  - Son deterministas y testeables.

La orquestacion (leer parquets, escribir resultado) vive en
scripts/ o en un orchestrator separado.

### 11.4. Reutilizacion de identity/

aggregation/ consume la salida de identity/ (relationships + amendments)
sin modificarla. NO importa de amendments directamente si puede operar
sobre el snapshot canonico ya producido.

---

## 12. Tests previstos (contrato observable)

Gate-NIPC.2 debe escribir tests que fijen el comportamiento. Minimo 18
familias (14 de v1.0 + 4 nuevas por Q-ESP-6).

### 12.1. Tests de reported_position_unit (3)

  - test_multi_manager_no_multiplica_shares
      OTHERMANAGER="1,2,3" con SSHPRNAMT=X -> X, no 3X.
  - test_discretion_separada
      SOLE + DFND en el mismo filing -> 2 units, no 1.
  - test_reported_position_unit_key
      Clave estable entre periodos.

### 12.2. Tests de filtros canonicos (3)

  - test_filtro_sh_null
      PRN y PUTCALL no-null excluidos.
  - test_discretion_no_excluye_dfnd
      DFND entra en NIPC_total.
  - test_section13f_eligible_separado_de_equity
      CUSIP eligible pero security_type=NON_EQUITY -> fuera de operativo.

### 12.3. Tests de mapping multicapa (4)

  - test_internal_precedence
      internal verificado gana sobre OpenFIGI.
  - test_conflict_no_auto_resuelve
      Discrepancia -> CONFLICT, no sobrescribe.
  - test_ticker_normalizado
      BRK/A -> BRK-A.
  - test_ambiguous_no_entra_operativo
      MULTIPLE_CANDIDATES -> fuera.

### 12.4. Tests de status (2)

  - test_status_insufficient_sin_coverage
      Sin cobertura -> INSUFFICIENT.
  - test_status_ready_requiere_ambos_controles
      Ambos thresholds -> READY.

### 12.5. Tests de determinismo (2)

  - test_sin_datetime_now
      aggregation/ no usa datetime.now().
  - test_reproducible_mismo_input_mismo_output
      Ejecucion repetida produce resultado identico.

### 12.6. Tests adicionales exigidos por Q-ESP-6 (4)

  - test_cusip_change_same_canonical_security
      Q4: CUSIP_A -> Security_X (100 shares)
      Q1: CUSIP_B -> Security_X (120 shares)
      -> delta = +20. NO Exit(A) + New(B).

  - test_multi_edge_does_not_duplicate_delta
      Una source line con varios managers -> SSHPRNAMT una vez.

  - test_unresolved_mapping_blocks_pair_match
      Security mapeada en Q1 pero no en Q4:
      -> match_status = UNRESOLVED_IDENTITY, no NEW.
      -> no genera delta artificial.

  - test_new_and_exit_zero_baseline_current
      NEW -> previous = 0.
      EXIT -> current = 0.
      Nunca imputar valores.

---

## 13. Prohibiciones explicitas

### 13.1. Sobre datos

  - NO inventar fechas historicas de CUSIP.
  - NO usar datetime.now() como fecha de observacion en aggregation/.
  - NO imputar valores faltantes; usar N/D o excluir.
  - NO usar el FIGI del INFOTABLE como pivote canonico (filer-declarado,
    inconsistente cross-filing).
  - NO emparejar CUSIPs distintos sin evidencia documental.

### 13.2. Sobre agregacion

  - NO multiplicar SSHPRNAMT por numero de edges OTHERMANAGER.
  - NO sumar por canonical_reporting_relationship_key como si fuera
    unidad economica.
  - NO dividir economicamente un holding.
  - NO colapsar dimensiones (SOLE/DFND/OTR) al agregar.
  - NO incluir report_period en la match key interperiodo.

### 13.3. Sobre mapping

  - NO resolver heuristicamente (matching por NAMEOFISSUER).
  - NO usar OpenFIGI como autoridad automatica.
  - NO sobrescribir internal con OpenFIGI silenciosamente.
  - NO auto-resolver CONFLICT.
  - NO auto-resolver AMBIGUOUS.
  - NO asumir validez historica de OpenFIGI actual.

### 13.4. Sobre clasificacion

  - NO usar TITLEOFCLASS como allowlist de equity (Q-NIPC-8).
  - NO interpretar la posicion 80 de la Official List.
  - NO confundir section13f_eligible con security_type=EQUITY.
  - NO interpretar el Option Indicator `*` como "esta fila es opcion".
    Significa "underlying has listed option".
  - NO colapsar section13f_eligible a booleano antes de resolver CONFLICT.

### 13.5. Sobre publicacion

  - NO publicar NIPC productivo si status != READY.
  - NO reportar una unica cifra de coverage (siempre 6: prev/current/
    paired_sec/paired_w/prev_unmapped/current_unmapped).
  - NO usar VALUE como ponderador de cobertura NIPC.
  - NO escribir delta_shares.py ni nipc.py sin autorizacion previa del
    auditor (regla local-first IAE).

### 13.6. Sobre trazabilidad

  - NO commitear API key.
  - NO sobrescribir snapshots.
  - NO omitir sha256 en manifests.

---

## 14. Secuencia de implementacion

### 14.1. Estado actual

    Gate-NIPC.0        PASS condicionado
    Gate 0 Mapping     PASS
    Gate 0 FIGI        PASS (descartado como pivote)
    Gate 0 OpenFIGI    PASS como fuente candidata
    Gate 0 SEC 13(f)   PASS / CLOSED
    Gate-NIPC.1        Especificacion v1.0 -> PASS CONDICIONADO
    Gate-NIPC.1.bis    Especificacion v1.1 (esta; 5 correcciones C1-C5)

### 14.2. Gates pendientes

    Gate-NIPC.1.bis  (este documento)
        |
        v
    Dictamen del auditor sobre la v1.1.
        |
        v
    Gate-NIPC.2  Implementacion
        BLOQUEADO por coverage / thresholds.
        Modulos ya implementados en spec v1.2-v1.4:
          - identity/sec13f_list.py (parser SEC 13(f))
          - identity/security_identity.py (C1-revisada)
          - aggregation/delta_shares.py
          - aggregation/nipc.py
          - data/mappings/cusip_equivalence.csv
        Pendiente autorizacion formal de Gate-NIPC.2.
        |
        v
    Gate-NIPC.3  Validacion end-to-end
        - Q4 2025 (baseline) -> Q1 2026 (target)
        - NIPC calculado para universo tecnico completo
        - NIPC calculado para operational_universe
        - Coverage reportada (6 metricas)
        - Status derivado (INSUFFICIENT en primera iteracion)
        - Informe + dictamen
        |
        v
    Gate-NIPC.4  Cierre de ciclo
        - Prompt maestro actualizado
        - FOLLOWUPS actualizado
        - Push unico

### 14.3. Regla local-first IAE

NO push a main de codigo hasta Gate-NIPC.3 PASS. Los commits
documentales pueden pushearse juntos al cierre del ciclo.

Actualmente: 21 commits locales pendientes de push (dictamenes + docs
del ciclo NIPC).

---

## 15. Deuda documentada

### 15.1. Deuda del ciclo NIPC

  - Deuda: temporal_validity de OpenFIGI. PARCIALMENTE CERRADA.
    Politica fijada en 3.6 + 7.6 (C4-revisada).
    Pendiente: diseno de validacion de metadata temporal (ticker) en
    Gate-NIPC.2. La identidad FIGI es estable por contrato.
  - Deuda: crosswalk interno actual cubre 30% ponderado del radar USA.
    Estado: INSUFFICIENT.
    Abordaje: expansion del crosswalk via OpenFIGI (con API key).
  - Deuda: 22 tickers del radar sin CUSIP conocido.
    Estado: UNMAPPED_RADAR_SECURITY.
    Abordaje: reverse lookup OpenFIGI con idType=TICKER en Gate-NIPC.2.
  - Deuda: 9,351,957,157 SSHPRNAMT Q4 / 25,746,947,582 SSHPRNAMT Q1
    fuera de Official List SEC.
    Estado: Aceptado como no-equity segun la clasificacion contractual
    de la spec.
    Tratamiento normativo: la seccion 5.4 define su exclusion del
    operational_universe.
    Estado de implementacion: RESUELTO 2026-09-21. La cadena
    §5.1 -> §5.3 -> §5.4 -> §5.5 esta implementada y cerrada
    (dictamenes #61-#67):
      - security_type.py: clasificador Nivel A+B con tokenizacion.
      - operational_universe.py: build_operational_universe.
      - Contrato ticker_mapped = CANONICAL AND VERIFIED (#65).
      - PIT obligatorio (identity_period_iso, #66 H-66.1 / #67).
    Medicion post-§5.3 (Q1 2026, filas): 69,37% RESOLVED_EQUITY,
    9,04% RESOLVED_NON_EQUITY, 21,59% UNRESOLVED, 0% CONFLICT.
    Handoff §5.4 -> §5.5: PASS (identidad exacta entre
    operational_universe y RESOLVED_EQUITY).
    GHISALLO (2026-09-21): los CUSIP 329882225 y 329882250 forman parte
    de esta poblacion documental fuera de Official List para Q4 2025 y
    Q1 2026. Estar fuera de Official List no implica error de filing ni
    no-reportabilidad (dictamenes #57-#60).
    Residual no bloqueante: SPONSORED ADR/ADS, ADR, SH BEN INT,
    FUND, ACT, ETFs producto especifico pendientes de evidencia PIT.
    Marcadores residuales registrados bajo sub-deuda de
    K-INSTITUTIONAL-ACCUMULATION-01.
  - Deuda: tabla data/mappings/cusip_equivalence.csv (nueva, para
    corporate actions).
    Estado: PENDIENTE creacion.
    Abordaje: Gate-NIPC.2 (esquema + poblacion manual curada).
  - Deuda: calculo paired_* en la fase de validacion.
    Estado: PENDIENTE implementacion.
    Abordaje: Gate-NIPC.2 (compute_nipc amplia con 6 metricas).

### 15.2. Deuda fuera del ciclo NIPC

  - FUTURE_FORMAT_VARIANT: si un trimestre SEC futura no publica TXT
    suficientemente completo, se abrira diseno de PDF parser.
    Estado: DIFERIDO (no bloqueante).
  - Curacion manual de la tabla cusip_ticker_exceptions.csv ampliada.
    Estado: parcial (3 filas curadas).
    Abordaje: iterativo.

### 15.3. Avisos

  - Los avisos de `evidence/nipc_gate0_openfigi/__pycache__` deben
    limpiarse periodicamente antes de commitear.
  - El fichero .gitignore bloquea `_*.py` pero no `_*.json` ni `_*.txt`;
    los originales se borraron manualmente tras migracion a evidence/.

---

## 16. Referencias

### 16.1. Documentos del ciclo NIPC

  docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_INFORME.md
  docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_DICTAMEN.md
  docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_MAPPING_DICTAMEN.md
  docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_FIGI_DICTAMEN.md
  docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_OPENFIGI_DICTAMEN.md
  docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_SEC13F_DICTAMEN.md
  docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_SEC13F_MINIPROBE_DICTAMEN.md
  docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION_v1.0.md
  docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION_DICTAMEN.md

### 16.2. Evidencia empirica

  docs/auditoria/iae/evidence/nipc_gate0_openfigi/
  D:/13f_probe/processed/2025Q4/
  D:/13f_probe/processed/2026Q1/
  D:/13f_probe/official_list_13f/13flist_2025Q4.txt
  D:/13f_probe/official_list_13f/13flist_2026q1.txt
  D:/13f_probe/official_list_13f/13flist_2025Q4.pdf (evidencia auxiliar)

### 16.3. Contratos previos

  docs/auditoria/INSTITUTIONAL_ACCUMULATION_CONTRATO.md (v1.1)
  docs/auditoria/INSTITUTIONAL_ACCUMULATION_PROPUESTA.md
  docs/auditoria/INSTITUTIONAL_ACCUMULATION_DICTAMEN_GATE0.md

### 16.4. Prompt maestro

  docs/auditoria/PROMPT_MAESTRO.md (v6.35)

---

Fin de la especificacion v1.4. Fecha: 2026-09-19. HEAD local al redactar:
cb1724e. Estado: filer continuity CERRADO como caracterizacion. Bloqueo
de Gate-NIPC.2 vuelve a coverage/thresholds (UNDEFINED). Las versiones
v1.0 a v1.3 se preservan como INSTITUTIONAL_ACCUMULATION_NIPC_
ESPECIFICACION_v1.0.md, _v1.1.md, _v1.2.md y _v1.3.md respectivamente.
