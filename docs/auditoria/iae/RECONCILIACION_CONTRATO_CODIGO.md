# IAE - Reconciliacion contrato <-> codigo

**Objeto:** expediente de reconciliacion entre el contrato semantico
vigente (NIPC_CONTRATOS_SEMANTICOS_v1.md) y el codigo implementado.

**SUBMISSION HEAD:** `2eb4dcd` (HEAD al entregar este expediente).
**AUDITED SNAPSHOT:** `4fe2b62` (commit sobre el que se realizo el analisis).
**POST-SNAPSHOT INTERMEDIATE:** `d3c9a87`, `79291ea`, `9c2ff66`, `04f3222`
(cambios documentales posteriores al snapshot, no afectan al analisis).

El dictamen F2.4 debe tomar como objeto documental la entrega identificada
por HEAD `2eb4dcd`; los demas hashes son snapshots historicos de trazabilidad.

**Generado:** 2026-09-20.
**Input para:** dictamen F2.4 del auditor externo.
**Estado:** DOCUMENTO DE TRABAJO. No normativo. No autoriza cambios de
codigo por si solo.

---

## 1. Contexto

El 2026-09-20, tras el cierre del ciclo de revision estructural
(95 hallazgos, 6 fixes mecanicos), se autorizo la implementacion
directa en codigo de los contratos P38/P60/P61 sin esperar F2.4.

Los commits correspondientes:

    | Hash    | Mensaje                                              |
    |---------|------------------------------------------------------|
    | 030f239 | P60: identity_type obligatorio en _normalize_canonical |
    | 6198d60 | P61: modulo temporal_validity                        |
    | fc12402 | P38: denominador TARGET_PAIRWISE + Q5 coverage_status |
    | 2a13a6f | P61: integrado en resolver                            |
    | 0f3ef8d | P61: end-to-end hasta _mapped_mask                    |
    | 83e60c3 | Q6 unmapped_count_* + Q8 identity_type en cusip_equivalence |

La autorizacion esta en PROMPT_MAESTRO v6.43 seccion 15.33. Es una
decision del supervisor, no un dictamen F2.4. Los contratos siguen
formalmente marcados como PROPUESTOS / SOMETIDOS A F2.4.

**Implicacion:** la implementacion se adelanto al contrato. Eso no es
necesariamente un problema, pero exige reconciliacion explicita antes
de que F2.4 emita dictamen.

---

## 2. Estado declarado vs estado real

    | Contrato | Estado declarado       | Estado real                |
    |----------|------------------------|----------------------------|
    | P60      | PROPUESTO / SOMETIDO   | Implementado parcialmente  |
    |          | A F2.4                 | (divergencia D3)           |
    | P61      | PROPUESTO / SOMETIDO   | Implementado nominalmente  |
    |          | A F2.4                 | (divergencia D1, grave)    |
    | P38      | PROPUESTO / SOMETIDO   | Implementado con proxy     |
    |          | A F2.4                 | (divergencia D2, grave)    |

Los tests pasan (979 + 2 skipped). Los tests validan la implementacion,
no el contrato. Por eso "todos los tests verdes" NO es evidencia de
cumplimiento contractual.

---

## 3. Divergencias detectadas

### D1 - P61 no conectado al resolver (GRAVE)

**Contrato P61 seccion 2.7:**

    operational_mapping_status(source, valid_from, valid_to, period) -> str

**Implementacion:**

- `temporal_validity.py` implementa `resolve_source_status(...)` (correcta).
- `resolve_security_identity` (security_identity.py L476-478) NO la invoca.
- Usa un dict hardcoded `_SOURCE_TO_OP_STATUS` (L443-447):

      "equivalence"         -> VERIFIED
      "crosswalk_internal"  -> TEMPORAL_UNVERIFIED
      "openfigi"            -> TEMPORAL_UNVERIFIED

**Consecuencias:**

- `cusip_ticker_exceptions.csv` (con vigencia `valid_from`/`valid_to`)
  siempre sale TEMPORAL_UNVERIFIED via `crosswalk_internal`, cuando el
  contrato exige VERIFIED si cubre el periodo.
- `cusip_equivalence.csv` (con vigencia) siempre sale VERIFIED sin
  verificar que la fila cubra el periodo.
- `aggregate_status` (regla Q7 completa) esta implementada pero NO se
  invoca en produccion. Solo se prueba en tests aislados.

### D2 - P38 no construye TARGET real (GRAVE)

**Contrato P38 seccion 3.2 + seccion 4.3:**

    TARGET_P = securities con observacion 13F en P
               AND presentes en RADAR_TARGET_CATALOG por shareClassFIGI

**Implementacion:**

- `nipc.py` L196-211:

      sec_c = _securities(curr)  # set(observed_security_key)
      sec_p = _securities(prev)
      target_pairwise = sec_c & sec_p

- `sec_c` y `sec_p` son conjuntos de CUSIPs observados, no de
  `shareClassFIGI` del catalogo radar.
- `nipc.py` no importa `radar_target_catalog` ni `target_universe`.
- El nombre `target_pairwise` es nominal: la implementacion real es
  `observed_pairwise`.

**Consecuencias:**

- El denominador ponderado usa CUSIPs observados, no TARGET.
- Los valores de `paired_weighted_share_coverage` publicados en baseline
  (e9fd830) y TOP 2000 (32baf9d) son historicos, no aptos para
  THRESHOLD_2 bajo la definicion contractual.

### D3 - P60 no lanza para CUSIP/ISIN (MENOR)

**Contrato P60 seccion 1.3 (formulacion vigente):**

    TICKER -> equity:<ticker>     (CANONICAL_EQUIVALENCE)
    FIGI   -> figi:<FIGI>         (CANONICAL_FIGI)
    CUSIP  -> NULL                (OBSERVED_CUSIP_ONLY)
    ISIN   -> NULL                (UNRESOLVED)
    identity_type ausente o invalido -> ValueError

Nota: la distincion es importante. El contrato NO exige ValueError
para CUSIP/ISIN: CUSIP/ISIN son entradas validas que producen NULL
como canonical. El ValueError se reserva para identity_type ausente
o fuera del enum.

**Implementacion:**

- `security_identity.py` L151-152:

      if identity_type in ("CUSIP", "ISIN"):
          return None

  Correcto para CUSIP/ISIN.

- `security_identity.py` L312 (`_find_active_equivalence`):

      itype = str(row["identity_type"]).strip() if has_type else "TICKER"

  Asume default TICKER si falta la columna. El contrato exige que el
  tipo venga declarado por la fuente, no asumido.

- `_find_active_equivalence` L312:

      itype = str(row["identity_type"]).strip() if has_type else "TICKER"

  Default TICKER si falta la columna. El contrato P60 seccion 1.6 exige
  que el tipo venga declarado por la fuente, no asumido.

**Consecuencias:**

- Un caller que espera excepcion para CUSIP obtiene None.
- Una fila del CSV sin columna `identity_type` produce canonical como
  TICKER, violando la prohibicion de inferir tipo.

---

## 4. Decisiones requeridas del auditor (F2.4)

Para cada divergencia, dos opciones validas:

**D1 (P61):**

  A. Implementar la conexion real: `resolve_security_identity` invoca
     `resolve_source_status` propagando valid_from/valid_to de cada
     fuente. Requiere (a) que `evidence` incluya valid_from/valid_to,
     (b) que `crosswalk_internal` se separe en sub-fuentes.

  B. Retirar `resolve_source_status` y `aggregate_status` del codigo:
     son contratos declarados que no se cumplen. Documentar la
     simplificacion (`_SOURCE_TO_OP_STATUS` como politica conservadora).

**D2 (P38):**

  A. Construir TARGET real:
     - Crear `identity/target_builder.py` que materialice TARGET_P
       desde RADAR_TARGET_CATALOG + OpenFIGI batch.
     - `nipc.py` recibe `target_q4` y `target_q1` como parametros.
     - Requiere OpenFIGI masivo (24.838 CUSIPs) NO AUTORIZADO hoy.

  B. Mantener `observed_security_key` como PROXY OBSERVACIONAL temporal,
     claramente etiquetado como NO CONTRACTUAL y no apto para evidencia
     de THRESHOLD_2 hasta disponer de TARGET real. Renombrar el
     comentario de `nipc.py` para no confundir.

**P60 (D3):**

  A. Eliminar default TICKER en `_find_active_equivalence`. Si la
     columna `identity_type` falta en `cusip_equivalence.csv`, la fila
     se rechaza con log explicito (no se asume TICKER).

  B. Mantener el default TICKER documentado como politica explicita
     de retrocompatibilidad. NO recomendado: contradice la regla
     "identidad declarada por la fuente".

  Nota: `_normalize_canonical` YA cumple el contrato para CUSIP/ISIN
  (devuelve NULL, no lanza). El punto de divergencia real es el
  default TICKER silencioso en `_find_active_equivalence`, no el
  raise.

---

## 5. Plan A.6 - Reconciliacion (post F2.4)

Una vez F2.4 emita dictamen sobre D1/D2/D3:

    A.6.1   Dictamen F2.4 sobre las 3 divergencias
    A.6.2   Fixes quirurgicos segun decision (1 commit por divergencia)
    A.6.3   Test de contrato por divergencia (nuevos)
    A.6.4   Recalcular baseline + TOP 2000 con nueva semantica
    A.6.5   Actualizar NIPC_CONTRATOS_SEMANTICOS_v1 (si aplica)
    A.6.6   Emitir F2.4 definitivo con evidencia actualizada

**Restricciones vigentes durante A.6:**

- NO tocar `delta_shares.py`, `compute_delta_shares`, `match_key`, C2.
- NO modificar `NIPC_COVERAGE_POLICY.md` v1.0 (hash 57f2d01f...).
- NO fijar THRESHOLD_1 / THRESHOLD_2.
- NO ejecutar OpenFIGI masivo sin autorizacion especifica.
- Local-first IAE: NO push hasta Gate-NIPC.3.

---

## 6. Tests contractuales minimos requeridos

Independiente de las decisiones, faltan estos tests. Ninguno existe hoy.

    | Contrato | Test                                                     |
    |----------|----------------------------------------------------------|
    | P60      | CUSIP -> NULL (ya pasa)                                  |
    | P60      | ISIN  -> NULL (ya pasa)                                  |
    | P60      | identity_type=None -> ValueError (ya pasa)               |
    | P60      | CSV sin columna identity_type -> NO asumir TICKER        |
    | P61      | resolve_security_identity invoca resolve_source_status   |
    | P61      | cusip_ticker_exceptions con vigencia -> VERIFIED         |
    | P61      | etf_holdings sin vigencia -> TEMPORAL_UNVERIFIED         |
    | P61      | aggregate_status sobre 6 combinaciones Q7                |
    | P38      | compute_coverage_pairwise recibe target_q4/target_q1     |
    | P38      | mismo shareClassFIGI, CUSIP distinto -> PAIRED           |
    | P38      | denominador cero -> UNAVAILABLE (ya existe)              |
    | P38      | TARGET_Q4 INTERSECT TARGET_Q1, no union                  |
    | P38      | pesos agregados por security ANTES de max(Q4,Q1)         |

Los tests actuales validan la implementacion proxy. Hay que anadir
validacion contractual explicita.

---

## 7. Lo que NO se toca en este ciclo

- `delta_shares.py`, `compute_delta_shares`, `match_key`.
- `NIPC_COVERAGE_POLICY.md` v1.0.
- Política de thresholds.
- OpenFIGI masivo.
- Match key C2.

---

## 8. Referencias

    | Documento                                      | Rol              |
    |------------------------------------------------|------------------|
    | iae/NIPC_CONTRATOS_SEMANTICOS_v1.md            | Contrato         |
    | iae/NIPC_COVERAGE_POLICY.md                    | Policy vigente   |
    | iae/NIPC_COVERAGE_POLICY_V13_PROPUESTA.md      | Policy propuesta |
    | iae/INFORME.md                                 | Evidencia IAE    |
    | iae/DICTAMENES.md                              | Registro         |

---

## 9. Asunto explicito para F2.4 (agregacion shareClassFIGI)

El auditor debe decidir la regla contractual cuando multiples CUSIPs
comparten `shareClassFIGI` en un mismo periodo. Escenarios:

    CUSIP_A -> FIGI_X -> VERIFIED
    CUSIP_B -> FIGI_X -> TEMPORAL_UNVERIFIED

    CUSIP_A -> FIGI_X -> CANONICAL
    CUSIP_B -> FIGI_X -> CONFLICT

Regla de agregacion propuesta (a dictamen):

    1. Resolver cada observacion a shareClassFIGI.
    2. Agrupar observaciones del periodo por shareClassFIGI.
    3. Sumar sshprnamt_total por grupo.
    4. Solo despues aplicar max(Q4_total, Q1_total) por shareClassFIGI.

La decision pendiente es si una observacion con estado distinto de
VERIFIED contribuye al peso agregado del shareClassFIGI o no.

Cuatro dimensiones a separar explicitamente:

    TARGET membership      (pertenece al universo contractual)
    RESOLVED status        (identidad resuelta)
    PAIRED status          (operacionalmente emparejado)
    weight contribution    (peso agregado para la metrica ponderada)

No deben quedar implicitamente acopladas.

---

Fin del expediente. Pendiente dictamen F2.4.
