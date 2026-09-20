# IAE - Reestructuracion del modulo

**Objeto:** plan arquitectonico de la reestructuracion del modulo IAE.

**Generado:** 2026-09-20.
**Estado:** PLAN. No ejecutable hasta dictamen F2.4.
**Input para:** F2.4 (contexto) + Fase A.6 (ejecucion).

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
        target_builder.py             (NUEVO - construye TARGET_Q4/Q1)
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

Constructor de universo TARGET. Consume:

- `radar_target_catalog.csv` (242 filas, sha256 11eabce8...).
- Observacion 13F del trimestre (INFOTABLE filtrada por SUBMISSION).
- OpenFIGI batch (autorizacion requerida).

Interfaz:

    def build_target_universe(
        quarter: str,
        infotable_df,
        catalog_df,
        openfigi_batch_fn,   # inyectable para tests
    ) -> dict:
        """
        Devuelve:
          target: set[shareClassFIGI]
          provenance: list[dict]  # cusip, share_class_figi, radar_ticker
          diagnostics: dict       # n_resolved, n_no_id, n_error
        """

Prohibido el uso de `security_identity` o `cusip_resolver` para
construir TARGET (H1 F2.1-bis - circularidad).

### 4.3. Reforma: `aggregation/nipc.py` (P38 - orquestacion)

Firma nueva:

    def compute_nipc(
        units_current,
        units_previous,
        target_q4=None,
        target_q1=None,
    ) -> dict

- Si `target_q4`/`target_q1` son `None`, el calculo de coverage
  devuelve `coverage_status: "UNAVAILABLE"` (comportamiento explicito).
- Si estan presentes, delega a `coverage.compute_contractual_coverage`.

Relacion entre las tres funciones (api normativa):

    compute_contractual_coverage   FUNCION CONTRACTUAL P38 (nueva)
    compute_nipc                   orquestador (nueva firma)
    compute_coverage_pairwise      legacy / proxy (deprecada tras A.6)

`nipc.py` NO importa `radar_target_catalog` ni `target_universe`. El
TARGET se construye en `identity/target_builder.py` y se pasa como
parametro. Esa es la arquitectura contractual.
- El comentario de la linea 209 deja de decir "TARGET_PAIRWISE"
  cuando opera sobre `observed_security_key`.

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

    D2 (TARGET real)      -> OpenFIGI masivo (24.838 CUSIPs) - NO AUTORIZADO
    D2 (build)            -> Autorizacion especifica del auditor

    D1, D3                -> Autonomas. Solo requieren dictamen F2.4.

**Consecuencia:** D1 y D3 se pueden implementar sin OpenFIGI. D2
requiere autorizacion adicional.

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

Mitigacion: implementar paso a paso con snapshot de resultados antes
y despues de cada paso. Test de no-regresion por cada cambio.

---

## 10. Referencias

    | Documento                                       | Rol            |
    |-------------------------------------------------|----------------|
    | iae/RECONCILIACION_CONTRATO_CODIGO.md           | Divergencias   |
    | iae/NIPC_CONTRATOS_SEMANTICOS_v1.md             | Contrato       |
    | iae/NIPC_COVERAGE_POLICY_V13_PROPUESTA.md       | Policy         |
    | iae/NIPC_ESPECIFICACION.md                      | Spec NIPC v1.4 |

---

Fin del plan de reestructuracion. Pendiente dictamen F2.4.
