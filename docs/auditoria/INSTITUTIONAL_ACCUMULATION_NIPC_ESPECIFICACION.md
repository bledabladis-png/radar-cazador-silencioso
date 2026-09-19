# ESPECIFICACION NIPC - Net Institutional Position Change

Modulo: IAE (Institutional Accumulation Evidence) - SEC 13F
Version: 1.0 (2026-09-19)
Estado: ESPECIFICACION. Sin codigo. Pendiente de dictamen final del auditor
        para autorizar Gate-NIPC.2 (implementacion).
Naturaleza: no normativo. Si hay conflicto con el Prompt Maestro, gana el
            Prompt Maestro.
Referencia: prompt v6.35, HEAD local 7c80af6.

---

## 0. Resumen ejecutivo

NIPC (Net Institutional Position Change) se define contractualmente como:

    NIPC = sum_i DeltaShares_i

    DeltaShares_i,t = Shares_i,t - Shares_i,t-1

es decir, la variacion trimestral de acciones largas reportadas en Form 13F
para un universo de securities y managers resuelto por identidad canonica.

Esta especificacion fija:

  - las 12 dimensiones del contrato interno de identidad y elegibilidad.
  - la entidad `reported_position_unit` como unidad de agregacion.
  - los filtros canonicos (SH + PUTCALL null).
  - el universo operativo (radar ^ 13F eligible ^ mapped) vs tecnico.
  - la capa de mapping multicapa (internal -> OpenFIGI -> unresolved).
  - la integracion de la Official List SEC 13(f) (eligibility normativa).
  - la trazabilidad de cada resolucion (snapshot + manifest).
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
  - El contrato de manager canonicalization (via identity/relationships).
  - El contrato de agregacion (reported_position_unit).
  - El contrato de mapping multicapa y su trazabilidad.
  - El contrato de integracion con la Official List SEC 13(f).
  - Los criterios de status (READY / INSUFFICIENT / CONFLICT / AMBIGUOUS /
    UNRESOLVED).
  - Los tests contractuales a escribir en Gate-NIPC.2.

### 1.2. Que NO define

  - Umbrales numericos de coverage. Solo la metodologia.
  - Implementacion (nombres definitivos de funciones, modulo aggregation/).
  - Politica de publicacion en el reporte. Esa decision es posterior.
  - Tratamiento de periodos futuros (Q2 2026+). Se hace por extension.

### 1.3. Alcance temporal del primer ciclo

  Baseline: 2025Q4 (PERIODOFREPORT=2025-12-31).
  Target:   2026Q1 (PERIODOFREPORT=2026-03-31).

  DeltaShares se calcula entre ambos snapshots canonicos.

---

## 2. Referencias normativas

### 2.1. Dictamenes vinculantes aplicados

| Fecha       | Documento                                                 | Aporta                                                       |
|-------------|-----------------------------------------------------------|--------------------------------------------------------------|
| 2026-09-19  | INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_DICTAMEN.md         | reported_position_unit; SOLE+DFND+OTR; universo vs tecnico   |
| 2026-09-19  | INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_MAPPING_DICTAMEN.md | 12 dimensiones; 4 propiedades separadas; INSUFFICIENT        |
| 2026-09-19  | INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_FIGI_DICTAMEN.md    | FIGI descartado como pivote; intra vs cross-filing           |
| 2026-09-19  | INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_OPENFIGI_DICTAMEN.md | Precedencia multicapa; trazabilidad; temporalidad pendiente |
| 2026-09-19  | INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_SEC13F_DICTAMEN.md  | Layout 80-char; Option Indicator; STATUS                     |
| 2026-09-19  | INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_SEC13F_MINIPROBE_DICTAMEN.md | TXT Q4 subset de PDF; Opcion A aprobada              |

### 2.2. Documentos previos en el repo

  - INSTITUTIONAL_ACCUMULATION_CONTRATO.md (v1.1).
  - INSTITUTIONAL_ACCUMULATION_PROPUESTA.md.
  - INSTITUTIONAL_ACCUMULATION_DICTAMEN_GATE0.md.
  - INSTITUTIONAL_ACCUMULATION_GATE_FA2_DICTAMEN.md.

### 2.3. Codigo existente reutilizable

  - src/institutional_accumulation/sec_13f/identity/temporal_filter.py
  - src/institutional_accumulation/sec_13f/identity/cusip_resolver.py
  - src/institutional_accumulation/sec_13f/identity/relationships.py
  - src/institutional_accumulation/sec_13f/identity/amendments.py

---

## 3. Las 12 dimensiones del contrato interno

El dictamen de Gate 0 Mapping exigio definir 12 conceptos para el contrato
interno. Se definen aqui uno por uno.

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

Descripcion: booleano que indica si un CUSIP figura en la Official List of
Section 13(f) Securities publicada por SEC para el trimestre.

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

Resolucion por STATUS:

  STATUS == blank    -> eligible
  STATUS == *A*      -> eligible
  STATUS == *D*      -> not eligible
  mismo CUSIP con estados incompatibles -> CONFLICT / REVIEW_REQUIRED

El CUSIP de opciones (pos 10 = espacio y issuer_description = CALL/PUT) NO
participa en el calculo de eligibility del universo SH + null.

Prohibiciones:
  - NO interpretar pos 80.
  - NO usar "ultimo registro gana" en lugar de STATUS.
  - NO confundir con `security_type=EQUITY` (dimensiones independientes).

### 3.3. security_type

Descripcion: clasificacion economica del instrumento. Valores permitidos:

    EQUITY          -- accion ordinaria, incluidas clases (CL A, CL B...).
    NON_EQUITY      -- preferentes, bonos, convertibles, units, warrants,
                       rights, depositary receipts no-equity.
    ETF             -- exchange-traded fund (dimension tecnica, distinta
                       de equity individual).
    UNKNOWN         -- clasificacion no determinable con evidencia actual.

Fuentes de clasificacion (por orden):

  1. TITLEOFCLASS del 13F + heuristicas semanticas (permitido aqui,
     porque es dimension distinta de la identidad).
  2. marketSector + securityType de OpenFIGI cuando resuelve.
  3. Sin evidencia -> UNKNOWN.

Regla congelada: TITLEOFCLASS NO es una allowlist de equity (dictamen
Q-NIPC-8). Se usa como uno de varios inputs, no como autoridad unica.

Ejemplo de conflicto legitimo:

    SSHPRNAMTTYPE = SH
    PUTCALL       = null
    TITLEOFCLASS  = CONVERTIBLE BOND
    -> security_type = NON_EQUITY
    -> el CUSIP se preserva en el dataset tecnico, pero NO entra al
       universo operativo NIPC.

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

Descripcion: cada resolucion de mapping esta atada a una fecha de vigencia.

Estado del arte:

  - OpenFIGI devuelve el ticker ACTUAL del instrumento, no
    necesariamente el vigente en el periodo (2025-12-31 o 2026-03-31).
  - Este desfase temporal NO esta resuelto en la especificacion v1.0.

Regla provisional:

  - Toda resolucion OpenFIGI se registra con `mapping_timestamp_utc`.
  - El status `TEMPORAL_VALIDITY_UNVERIFIED` se anota explicitamente
    en cada fila.
  - Las resoluciones sin verificacion temporal NO se publican como
    estado productivo (solo diagnostico).

Trabajo pendiente para Gate-NIPC.2:

  - Disenar validacion temporal (p. ej. comparar con el CUSIP del periodo
    y rechazar si OpenFIGI devuelve un CUSIP diferente).
  - Documentar en cada dictamen.

Prohibicion: no asumir que un ticker resuelto hoy era el mismo en Q1 2026.

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
  - Solo ONE_TO_ONE participa en el universo operativo.

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

Valores permitidos (5 estados):

    READY           -- cobertura suficiente, ambos controles contractuales
                       (mapping_coverage + weighted_share_coverage)
                       superan umbrales. Publicable.
    INSUFFICIENT    -- cobertura por debajo de umbral. Calculo de
                       diagnostico disponible, no publicable como senal.
    CONFLICT        -- existen discrepancias no resueltas entre fuentes
                       (internal vs OpenFIGI). Requiere intervencion.
    AMBIGUOUS       -- CUSIP con multiples candidatos one_to_many no
                       resueltos. Casos concretos.
    UNRESOLVED      -- CUSIP sin ninguna fuente que lo resuelva.

Regla por defecto: STATUS = INSUFFICIENT mientras los controles de
cobertura no se satisfagan.

Estado actual (Q1 2026): INSUFFICIENT.

Cambio de estado: solo el auditor puede promover a READY tras validacion
formal en Gate-NIPC.3.

Prohibicion: NUNCA publicar un valor NIPC como senal operativa si STATUS
!= READY.

---

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
`reported_position_unit` (identidad de filing manager + security +
discretion), NO por `canonical_reporting_relationship_key`.

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

    section13f_eligible = False
    -> no entra en operational_universe

### 5.4. Security type

Aplicar `security_type == EQUITY` como filtro sobre el universo
elegible. Excluye:

  - Convertible bonds (aunque pasen el filtro SH + null).
  - Units, preferreds, warrants, rights.
  - ETFs (universo tecnico, no operativo radar equity).

### 5.5. Orden de aplicacion (canonico)

    13F INFOTABLE
        v
    SH + null
        v
    section13f_eligible
        v
    security_type == EQUITY
        v
    ticker_mapped
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

    mapped=True, equity=True,  section13f_eligible=False, radar_member=True
    mapped=True, equity=False, section13f_eligible=True,  radar_member=False
    mapped=False, equity=True, section13f_eligible=True,  radar_member=True
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

### 7.6. Flujo de resolucion

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
                              radar intersection

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

    section13f_eligible = (CUSIP in Official List) AND (STATUS != DELETED)

Resolucion de STATUS:

    STATUS == blank    -> eligible
    STATUS == *A*      -> eligible
    STATUS == *D*      -> not eligible
    CUSIP con estados incompatibles -> CONFLICT / REVIEW_REQUIRED

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

    docs/auditoria/evidence/nipc_gate0_openfigi/
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
`TEMPORAL_VALIDITY_UNVERIFIED` se anota mientras no exista validacion
temporal.

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

Contrato Q3 exige DOS controles simultaneos:

    (1) mapping_coverage >= THRESHOLD_1
    (2) weighted_share_coverage >= THRESHOLD_2

Ambos sobre operational_universe (no sobre universo tecnico).

### 10.3. Criterios de fijacion (recomendacion tecnica)

Los umbrales deben fijarse considerando:

  - Que el universo operativo del radar es el que importa (no 13F completo).
  - Que el sesgo large-cap es el riesgo principal del crosswalk actual.
  - Que los no mapeados pueden ser no-equity (excluidos legitimamente)
    o equity genuinamente faltante.
  - Que un umbral alcanzable al primer intento es preferible a uno
    aspiracional no alcanzable.

Propuesta orientativa (NO aprobada, solo para discusion):

    THRESHOLD_1 (securities): >= 90% del radar USA
    THRESHOLD_2 (shares):     a determinar tras integracion SEC 13(f)

### 10.4. Estado actual

    mapping_coverage:         90.91% (radar USA, crosswalk actual)
    weighted_share_coverage:  29.995% (radar USA, crosswalk actual)
    section13f_eligible:      pendiente integracion sobre radar

Por tanto: STATUS = INSUFFICIENT. No se publica NIPC productivo.

### 10.5. Regla de promocion

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

    compute_reported_position_units(infotable_df, submission_df,
                                    coverpage_df, *, report_period)
        -> DataFrame con columnas:
             report_period,
             filing_manager_cik,
             canonical_security,
             discretion_type,
             sshprnamt_total,
             n_source_lines

    compute_delta_shares(units_current, units_previous, *,
                         match_keys=None)
        -> DataFrame con columnas:
             reported_position_unit_key,
             sshprnamt_current,
             sshprnamt_previous,
             delta_shares,
             match_status

    compute_nipc(delta_df, *, discretion_breakdown=True)
        -> dict con:
             nipc_total,
             nipc_sole,
             nipc_dfnd,
             nipc_otr,
             coverage_securities,
             coverage_weighted_shares,
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

Gate-NIPC.2 debe escribir tests que fijen el comportamiento. Lista minima:

### 12.1. Tests de reported_position_unit

  - test_multi_manager_no_multiplica_shares
      OTHERMANAGER="1,2,3" con SSHPRNAMT=X -> X, no 3X.
  - test_discretion_separada
      SOLE + DFND en el mismo filing -> 2 units, no 1.
  - test_reported_position_unit_key
      Clave estable entre periodos.

### 12.2. Tests de filtros canonicos

  - test_filtro_sh_null
      PRN y PUTCALL no-null excluidos.
  - test_discretion_no_excluye_dfnd
      DFND entra en NIPC_total.
  - test_section13f_eligible_separado_de_equity
      CUSIP eligible pero security_type=NON_EQUITY -> fuera de operativo.

### 12.3. Tests de mapping multicapa

  - test_internal_precedence
      internal verificado gana sobre OpenFIGI.
  - test_conflict_no_auto_resuelve
      Discrepancia -> CONFLICT, no sobrescribe.
  - test_ticker_normalizado
      BRK/A -> BRK-A.
  - test_ambiguous_no_entra_operativo
      MULTIPLE_CANDIDATES -> fuera.

### 12.4. Tests de status

  - test_status_insufficient_sin_coverage
      Sin cobertura -> INSUFFICIENT.
  - test_status_ready_requiere_ambos_controles
      Ambos thresholds -> READY.

### 12.5. Tests de determinismo

  - test_sin_datetime_now
      aggregation/ no usa datetime.now().
  - test_reproducible_mismo_input_mismo_output
      Ejecucion repetida produce resultado identico.

---

## 13. Prohibiciones explicitas

### 13.1. Sobre datos

  - NO inventar fechas historicas de CUSIP.
  - NO usar datetime.now() como fecha de observacion en aggregation/.
  - NO imputar valores faltantes; usar N/D o excluir.
  - NO usar el FIGI del INFOTABLE como pivote canonico (filer-declarado,
    inconsistente cross-filing).

### 13.2. Sobre agregacion

  - NO multiplicar SSHPRNAMT por numero de edges OTHERMANAGER.
  - NO sumar por canonical_reporting_relationship_key como si fuera
    unidad economica.
  - NO dividir economicamente un holding.
  - NO colapsar dimensiones (SOLE/DFND/OTR) al agregar.

### 13.3. Sobre mapping

  - NO resolver heuristicamente (matching por NAMEOFISSUER).
  - NO usar OpenFIGI como autoridad automatica.
  - NO sobrescribir internal con OpenFIGI silenciosamente.
  - NO auto-resolver CONFLICT.
  - NO auto-resolver AMBIGUOUS.

### 13.4. Sobre clasificacion

  - NO usar TITLEOFCLASS como allowlist de equity (Q-NIPC-8).
  - NO interpretar la posicion 80 de la Official List.
  - NO confundir section13f_eligible con security_type=EQUITY.
  - NO interpretar el Option Indicator `*` como "esta fila es opcion".
    Significa "underlying has listed option".

### 13.5. Sobre publicacion

  - NO publicar NIPC productivo si status != READY.
  - NO reportar una unica cifra de coverage (siempre 3: A/B/C).
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
    Gate-NIPC.1        ESTA especificacion

### 14.2. Gates pendientes

    Gate-NIPC.1  (este documento)
        |
        v
    Dictamen del auditor sobre la especificacion.
        |
        v
    Gate-NIPC.2  Implementacion
        - Modulo src/institutional_accumulation/aggregation/
        - delta_shares.py, nipc.py
        - Tests 12.1 a 12.5
        - Parser SEC 13(f) en identity/sec13f_list.py (o similar)
        - Cache OpenFIGI
        |
        v
    Gate-NIPC.3  Validacion end-to-end
        - Q4 2025 (baseline) -> Q1 2026 (target)
        - Mini-gate: NIPC calculado para universo tecnico completo
        - Mini-gate: NIPC calculado para operational_universe
        - Coverage reportada (A/B/C)
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

Actualmente: 15 commits locales pendientes de push (dictamenes + docs
del ciclo NIPC).

---

## 15. Deuda documentada

### 15.1. Deuda del ciclo NIPC

  - Deuda: temporal_validity de OpenFIGI no verificada.
    Estado: PENDIENTE.
    Abordaje: Gate-NIPC.2 (diseno de validacion temporal).
  - Deuda: crosswalk interno actual cubre 30% ponderado del radar USA.
    Estado: INSUFFICIENT.
    Abordaje: expansion del crosswalk via OpenFIGI (con API key).
  - Deuda: 22 tickers del radar sin CUSIP conocido.
    Estado: UNMAPPED_RADAR_SECURITY.
    Abordaje: reverse lookup OpenFIGI con idType=TICKER en Gate-NIPC.2.
  - Deuda: 9,351,957,157 SSHPRNAMT Q4 / 25,746,947,582 SSHPRNAMT Q1
    fuera de Official List SEC.
    Estado: Aceptado como no-equity (mayoria convertibles, MMF, cash).
    Abordaje: ninguno; documentar.

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

### 16.2. Evidencia empirica

  docs/auditoria/evidence/nipc_gate0_openfigi/
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

Fin de la especificacion. Version 1.0 (2026-09-19). HEAD local 7c80af6.
Estado: pendiente de dictamen del auditor para autorizar Gate-NIPC.2.
