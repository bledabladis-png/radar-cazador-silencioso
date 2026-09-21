# IAE - Reestructuracion del modulo

**Objeto:** plan arquitectonico de la reestructuracion del modulo IAE.

**Generado:** 2026-09-20.
**Estado:** PLAN vigente. F2.4 EMITIDO (2026-09-20, GO CONDICIONADO).
**Input para:** Fase A.6 (ejecucion).
**Referencia dictamen:** DICTAMENES.md #24 + INFORME.md #18.

---

## 1. Objetivo

Alinear el codigo del modulo IAE con el contrato semantico vigente
(NIPC_CONTRATOS_SEMANTICOS_v1.md) y con la policy de cobertura v1.3
(cuando se apruebe).

Tres divergencias detectadas (ver RECONCILIACION_CONTRATO_CODIGO.md):

    D1  P61 no conectado al resolver (GRAVE)
    D2  P38 no construye TARGET real (GRAVE)
    D3  P60 infiere TICKER cuando falta identity_type declarado (MENOR)

La reestructuracion resuelve las tres en una sola operacion coherente.

**Dictamen F2.4 (2026-09-20) - decisiones recibidas:**

    D1 P61       GO              Conectar resolve_source_status + evidence extendida.
    D2 P38       GO CONDICIONADO  TARGET real + rediseno independencia del mapping.
    D3 P60       GO              Fail-closed, sin default TICKER.
    Q12          Modelo A        shareClassFIGI (unidad = share class).
    AGREG.       Opcion 2        Funcion separada + solo VERIFIED al peso contractual.
    OpenFIGI     GO CONDICIONADO Snapshot + hash, no dependencia live.
    Policy v1.3  NO              v1.0 sigue normativa.

**3 bloqueantes estructurales introducidos por F2.4:**

    1. TARGET no puede depender del exito del mapping (sesgo de seleccion).
       Separacion CATALOGO (externo, versionado) vs TARGET_OBSERVED.
    2. Semantica point-in-time: catalog_version + valid_from/valid_to.
    3. 13F != flujo en tiempo real. Etiqueta: cambio trimestral observado.

---

## 2. Estado actual del modulo

    src/institutional_accumulation/
      __init__.py
      temporal_validity.py            <- raiz (ubicacion actual)
      aggregation/
        __init__.py
        delta_shares.py               <- intocable (C2)
        nipc.py                       <- cobertura pairwise sobre observed_security_key
                                          (proxy observacional; NO construye TARGET contractual)
      identity/
        __init__.py
        openfigi_client.py            <- cliente OpenFIGI
        radar_target_catalog.py       <- catalogo 242 filas
        target_universe.py            <- resolver CUSIP->membership
      sec_13f/
        downloader.py, parser.py, schema.py, storage.py, manifest.py, ingest.py
        identity/
          temporal_filter.py, cusip_resolver.py, relationships.py,
          amendments.py, sec13f_list.py, security_identity.py

---

## 3. Estructura objetivo

    src/institutional_accumulation/
      __init__.py
      aggregation/
        __init__.py
        delta_shares.py               (sin cambios - C2 intocable)
        nipc.py                       (FIRMA NUEVA)
        temporal_validity.py          (MOVIDO desde raiz)
        coverage.py                   (NUEVO - orquestador TARGET/RESOLVED/PAIRED)
      identity/
        __init__.py
        openfigi_client.py            (sin cambios)
        radar_target_catalog.py       (sin cambios)
        target_universe.py            (sin cambios - resolver individual)
        target_builder.py             (NUEVO - CATALOGO externo -> TARGET.
                                       Sin dependencia del resolver evaluado)
        position_record.py            (NUEVO - dataclass tipado para coverage)
      sec_13f/
        (sin cambios estructurales)
        identity/
          security_identity.py        (CAMBIO - D1 + D3)

---

## 4. Cambios concretos

### 4.1. Nuevo: `aggregation/coverage.py`

Orquestador contractual. Encapsula la cadena TARGET -> RESOLVED -> PAIRED.

`compute_contractual_coverage()` recibe pesos YA AGREGADOS por
`shareClassFIGI` (Opcion 2, ver RECONCILIACION seccion 9.1). La
operacion de agregacion vive en `aggregate_positions_by_shareclass_figi()`
que puede ubicarse en el mismo modulo o en un helper separado.

Interfaz:

    def compute_contractual_coverage(
        target_q4: set,
        target_q1: set,
        resolved_q4: set,
        resolved_q1: set,
        canonical_q4: dict,
        canonical_q1: dict,
        weights_q4: dict,
        weights_q1: dict,
    ) -> dict:
        # Esta es la FUNCION CONTRACTUAL P38 (una unica API normativa).
        """
        Devuelve:
          coverage_previous: float
          coverage_current: float
          paired_security_coverage: float
          paired_weighted_share_coverage: float | None
          coverage_status: "VALID" | "UNAVAILABLE"
          unmapped_count_previous: float
          unmapped_count_current: float
        """

Documentado con las definiciones literales del contrato P38 seccion
3.2 y 3.3. No contiene logica de construccion de los sets: recibe
los 4 universos como parametros.

### 4.2. Nuevo: `identity/target_builder.py` (P38 - materializacion)

**Reformulado por F2.4 (bloqueante 1):** el TARGET no puede depender del
exito del mapping. Si el TARGET se construye solo sobre observaciones
que OpenFIGI resolvio, el denominador depende del proceso que se mide
(sesgo de seleccion).

Arquitectura obligatoria:

    CATALOGO (externo, versionado)
       |
       v
    TARGET FIGIs  <- entidad propia, NO derivada del resolver evaluado
       |
       |   13F Q4 ---- mapping ----+
       |   13F Q1 ---- mapping ----+
       |
       v
    coverage / pairing

Constructor de TARGET. Consume SOLO:

- `radar_target_catalog.csv` (242 filas, sha256 11eabce8...).
- `catalog_version` + `catalog_valid_from` + `catalog_valid_to`
  (bloqueante 2, P62). Alternativa: `target_catalog_as_of(period_end)`.

Prohibido el uso de `security_identity`, `cusip_resolver` o el
resultado del mapping para construir TARGET (H1 F2.1-bis - circularidad).

Interfaz (a dictamen A.6):

    def build_target_universe(
        period_end: str,
        catalog_df,          # catalogo versionado
        catalog_as_of: str,  # PIT: fecha del catalogo a usar
    ) -> dict:
        """
        Devuelve:
          target: set[shareClassFIGI]
          provenance: list[dict]  # share_class_figi, radar_ticker, source
          diagnostics: dict       # n_catalog, n_filtered_by_as_of
        """

El mapping de las observaciones 13F se hace DESPUES, en una capa
separada. El TARGET se pasa a `coverage.py` como parametro externo.

### 4.3. Reforma: `aggregation/nipc.py` (P38 - orquestacion)

Firma nueva:

    def compute_nipc(
        units_current,
        units_previous,
        target_q4,           # NO opcional en ruta contractual
        target_q1,
        *,
        evidence_class: str, # "CONTRACTUAL" | "PROXY" (obligatorio)
    ) -> dict

- Prohibida la degradacion silenciosa a proxy (F2.4 regla 3).
- Si `target_q4 is None` o `target_q1 is None` y `evidence_class ==
  "CONTRACTUAL"`, lanzar error duro.
- Si `evidence_class == "PROXY"`, el resultado lleva
  `evidence_class: PROXY` explicito y NO es apto para THRESHOLD_2.
- Ruta contractual: delega a `coverage.compute_contractual_coverage`.

Relacion entre las tres funciones (api normativa):

    compute_contractual_coverage   FUNCION CONTRACTUAL P38 (nueva)
    compute_nipc                   orquestador (nueva firma)
    compute_coverage_pairwise      legacy / proxy (deprecada tras A.6)

`nipc.py` NO importa `radar_target_catalog` ni `target_universe` ni
`target_builder`. El TARGET se construye en `identity/target_builder.py`
y se pasa como parametro. Esa es la arquitectura contractual.
- El comentario de la linea 209 deja de decir "TARGET_PAIRWISE"
  cuando opera sobre `observed_security_key`.
- `unmapped_count` es int. `unmapped_weight` es float. No mezclar.

`compute_delta_shares` y `match_key` no se tocan.

### 4.4. Reforma: `sec_13f/identity/security_identity.py` (P61 conexion + P60 sin default TICKER)

P61 - Conectar `temporal_validity`:

- `evidence` incluye `source` + `valid_from` + `valid_to` de la fuente.
- `resolve_security_identity` invoca `resolve_source_status(...)`.
- El dict `_SOURCE_TO_OP_STATUS` se retira o se conserva solo como
  fallback documentado.
- `crosswalk_internal` se separa en sub-fuentes reales:
  `cusip_ticker_exceptions` (con vigencia) vs `etf_holdings` (sin
  vigencia). Propagado en `evidence`.

P60 - No inferencia de identity_type:

- `_normalize_canonical(value, identity_type)` mantiene la semantica
  contractual vigente:
      CUSIP -> None
      ISIN  -> None
  No introduce `ValueError` para CUSIP/ISIN.
- `_find_active_equivalence` no puede asumir `TICKER` cuando falta
  `identity_type` declarado por la fuente. Si la columna falta, la
  fila se rechaza con log explicito. El comportamiento observable
  exacto (rechazo, lista vacia, UNRESOLVED) queda a dictamen F2.4.

### 4.5. Movimiento: `temporal_validity.py`

De `src/institutional_accumulation/temporal_validity.py` a
`src/institutional_accumulation/aggregation/temporal_validity.py`.

Razon: es input directo de `nipc.py`, no un componente de la raiz.

Actualizar imports en:
- `security_identity.py`
- tests (`test_temporal_validity.py`)

---

## 5. Impacto en tests

### 5.1. Tests nuevos requeridos

    tests/test_p60_contract.py
      - identity_type CUSIP -> None
      - identity_type ISIN -> None
      - identity_type=None/invalido -> ValueError
      - CSV sin columna identity_type -> no inferir TICKER
      - Prefijo explicito (equity:/figi:) gana

    tests/test_p61_contract.py
      - resolve_security_identity invoca resolve_source_status
      - cusip_ticker_exceptions con vigencia -> VERIFIED
      - etf_holdings sin vigencia -> TEMPORAL_UNVERIFIED
      - aggregate_status 6 combinaciones Q7

    tests/test_p38_contract.py
      - compute_contractual_coverage recibe sets externos
      - mismo shareClassFIGI + CUSIPs distintos -> PAIRED
      - denominador cero -> UNAVAILABLE
      - interseccion TARGET_Q4 ^ TARGET_Q1

    tests/test_target_builder.py
      - build_target_universe con batch_fn inyectable (sin OpenFIGI real)
      - provenance completa
      - independencia del resolver evaluado

### 5.2. Tests existentes a actualizar

    tests/test_sec_13f_security_identity.py
      - Retirar default TICKER
      - Mantener/cubrir CUSIP/ISIN -> None
      - Anadir casos identity_type ausente/invalido -> ValueError

    tests/test_sec_13f_nipc.py
      - Firma nueva (recibe target_q4/target_q1)
      - Casos proxy (target=None) explicitos

    tests/test_temporal_validity.py
      - Ajustar import path si se mueve

---

## 6. Dependencias externas

    F2.4                  -> EMITIDO 2026-09-20 (GO CONDICIONADO).
    A.6.0 (Gate 0)        -> PREVIO. Sin dependencias externas.

    D1 (P61) + D3 (P60)   -> Autonomas. Solo requieren F2.4 (emitido).
    D2 (P38) quirurgico   -> Firma nueva nipc + coverage.py. Sin OpenFIGI.
    A.6.2-bis (rediseno)  -> OpenFIGI masivo NO AUTORIZADO + A.6.0 previo.
    Policy v1.3           -> NO aprobada. v1.0 sigue normativa.

**Consecuencia:** D1, D3 y la parte quirurgica de D2 se pueden
implementar tras A.6.0. El rediseno arquitectonico de TARGET
(A.6.2-bis) requiere ademas autorizacion especifica de OpenFIGI.

---

## 7. Orden de implementacion sugerido

    Paso 1  D3 (eliminar default TICKER)          - sin deps
    Paso 2  D1 (conectar P61 + sub-fuentes)       - sin deps
    Paso 3  Mover temporal_validity.py            - sin deps
    Paso 4  coverage.py + firma nueva nipc.py     - sin deps
    Paso 5  D2 (target_builder + OpenFIGI)        - requiere autorizacion
    Paso 6  Recalcular baseline + TOP 2000
    Paso 7  Emitir F2.4 definitivo

Los pasos 1-4 se pueden ejecutar en cuanto F2.4 sea GO.

---

## 8. Lo que NO cambia

- `aggregation/delta_shares.py` ni `compute_delta_shares`.
- `match_key` (spec 4.7).
- C2 (matching interperiodo).
- `NIPC_COVERAGE_POLICY.md` v1.0.
- Los 4 modulos `sec_13f/` no-identity.
- `identity/openfigi_client.py`, `radar_target_catalog.py`,
  `target_universe.py`.

---

## 9. Riesgos

    R1  Retirar default TICKER puede romper tests heredados que
        dependen del comportamiento actual.
    R2  Separar crosswalk_internal en sub-fuentes puede alterar
        los estados devueltos por `resolve_security_identity` en
        algunos CUSIPs.
    R3  `coverage.py` como modulo nuevo puede no integrarse bien con
        los consumidores actuales de `nipc.compute_coverage_pairwise`.
        Requiere inventario de callers antes.
    R4  Si TARGET se construye sobre observaciones ya mapeadas, el
        denominador depende del proceso medido (sesgo de seleccion,
        bloqueante 1 del F2.4). Mitigacion: separacion CATALOGO vs
        TARGET_OBSERVED en A.6.2-bis.
    R5  Catalogo actual sin valid_from/valid_to introduce survivorship
        bias al aplicarlo retroactivamente (bloqueante 2). Mitigacion:
        P62 point-in-time.
    R6  Distinguir "no aparece" de "vendio" requiere contrato explicito
        (P63). Sin el, un delta negativo puede no ser una venta.

Mitigacion: implementar paso a paso con snapshot de resultados antes
y despues de cada paso. Test de no-regresion por cada cambio.

---

## 10. Referencias

    | Documento                                       | Rol            |
    |-------------------------------------------------|----------------|
    | iae/RECONCILIACION_CONTRATO_CODIGO.md           | Divergencias   |
    | iae/NIPC_CONTRATOS_SEMANTICOS_v1.md             | Contrato       |
    | iae/NIPC_ESPECIFICACION.md                      | Spec NIPC v1.4 |

---

Fin del plan de reestructuracion. F2.4 emitido 2026-09-20.
Pendiente A.6.0 (Gate 0 de los 3 bloqueantes, sin tocar codigo).
