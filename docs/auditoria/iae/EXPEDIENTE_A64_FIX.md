# EXPEDIENTE - Fix de cobertura contractual P38 (A.6.4)

**Fecha:** 2026-09-21
**Estado:** PENDIENTE de dictamen externo
**Origen:** auditoria externa del bundle 2026-09-21 (hallazgos H-05/H-06/H-07)
+ hallazgo adicional H-10.1 detectado por el supervisor.

**Regla respetada:** NO se ha tocado codigo productivo. `coverage.py`,
`catalog_p38_adapter.py`, `period_state.py` permanecen intactos. El fix
requiere dictamen previo (regla 4 del prompt).

---

## 1. Contexto

La evidencia A.6.4 (`evidence/a64_integration_b1_p61_p38/`) publicaba
`coverage = 1.0` sobre un subconjunto de 18 keys Q1 2026. La auditoria
externa detecto que el resultado podia ser artefacto (H-05/H-06/H-07).

El supervisor ejecuto un probe diagnostico (`probe_diagnose_h101.py`)
que confirma el bug con numeros.

## 2. Resultados del diagnostico (datos reales Q4 2025 / Q1 2026)

| Metrica | Valor real |
|---|---:|
| TARGET_Q4 (P61 seccion 5.5, keys reales) | **0** |
| TARGET_Q1 (P61 seccion 5.5, keys reales) | **20** |
| TARGET_PAIRWISE (Q4 interseccion Q1) | **0** |
| Universo B2-PIT completo | 242 keys (240 con FIGI) |
| Cobertura Q1 real / universo B2-PIT | **18 / 242 = 0.0744** |

**Adapter rechaza Q4 vacio**: `AdapterError: TARGET_PAIRWISE vacio`.

## 3. Diagnostico tecnico

### 3.1. Por que el probe original reportaba 1.0

El probe original llamaba:

    catalog_to_p38_targets(universe_sub, universe_sub,
                           state_q4=st_q, state_q1=st_q, ...)

Es decir, **Q4 = Q1 (mock)**. Al ser el mismo objeto, `target_pairwise`
no es vacio, y todas las metricas salen `1.0` trivialmente.

### 3.2. Bug estructural adicional (H-10.1)

Independientemente del mock del probe, hay 3 problemas en codigo productivo:

1. **`catalog_p38_adapter._records`** (L~150): asigna
   `operational_mapping_status="VERIFIED"` **incondicionalmente**. No
   verifica el estado operacional real del caller.

2. **`coverage.py::compute_contractual_coverage`** (L~130):
   `coverage_previous = len(res_q4) / len(all_q4)`, donde `all_q4` es
   "todos los records con FIGI" (observed), NO `TARGET_Q4`. Como el
   adapter marca VERIFIED, `res_q4 == all_q4` siempre -> `coverage = 1.0`.

3. **`PositionRecord.weight = 1.0` hardcoded** en `_records`. La masa
   real (SSHPRNAMT) no se propaga desde 13F. `paired_weighted_share_coverage`
   es identico a `paired_security_coverage`.

**Consecuencia:** incluso con datos reales, `compute_contractual_coverage`
no puede producir un valor distinto de 1.0 con el adapter actual.

## 4. Consecuencia

A.6.4 queda reclasificado como **smoke test de integracion del adapter**,
NO como evidencia cuantitativa de cobertura contractual. Ya aplicado en
`README.md` + `ESTADO_DECLARADO.md` seccion 8.

## 5. Propuesta de fix

### 5.1. Fix minimo (A1)

Alcance: cerrar H-05/H-06/H-10.1. H-07 queda declarado no resuelto.

1. **`catalog_p38_adapter.py`**: derivar `operational_mapping_status`
   del estado real propagado (no marcar VERIFIED incondicionalmente).
   Requiere que `period_state` transporte la evidencia operacional.

2. **`coverage.py`**: cambiar denominador de `coverage_previous` /
   `coverage_current` a `TARGET_Q4` / `TARGET_Q1` (contractual), no a
   observed-with-FIGI.

3. **`probe_integration_b1_p61_p38.py`**: eliminar mock Q4=Q1. Reportar
   fail-closed cuando Q4 este vacio. Publicar cardinalidades reales.

Resultado esperado: `coverage_previous = UNAVAILABLE` (Q4 vacio) o
`0.0744` si se usa Q1 vs universo; `paired_security_coverage = UNAVAILABLE`
(pairwise vacio). Honestidad > cifras bonitas.

### 5.2. Fix completo (A2)

Adicional a A1:

4. **Propagar SSHPRNAMT** desde `canonical_snapshot.INFOTABLE` hasta
   `PositionRecord.weight`. Afecta: `period_state.py`, `target_builder.py`,
   `catalog_p38_adapter.py`. **Requiere ciclo propio** por tocar 3 modulos
   productivos.

### 5.3. Orden recomendado

1. Dictamen externo sobre A1/A2.
2. Si A1: 3 commits pequenos (adapter, coverage, probe).
3. Si A2: ciclo completo con sub-dictamenes.
4. Re-ejecutar probe. Verificar numeros reales.

## 6. Preguntas al auditor

1. **Alcance del fix:** A1 (minimo, deja H-07 sin resolver) o A2 (completo,
   con propagacion SSHPRNAMT)?
2. **Denominador contractual:** `coverage_previous` debe ser
   `resolved / TARGET_Q4` o `resolved / observed_with_figi`? El contrato
   P38 (`NIPC_CONTRATOS_SEMANTICOS_v1.md` seccion 3) define cobertura
   sobre TARGET. Confirmar.
3. **Adapter VERIFIED:** debe derivar el estado del `period_state`, o es
   responsabilidad del caller (seccion 5.5) garantizar VERIFIED? Si es
   del caller, el adapter podria dejar de marcarlo (evita el bug), pero
   hay que documentar la precondicion.
4. **Peso real:** si A2, ¿`SSHPRNAMT` del filing base, o el
   `canonical_snapshot` post-amendments? El contrato P38 seccion 3.3 dice
   `w(s) = max(Q4, Q1)`, pero no especifica la fuente del shares.
5. **Q4 vacio + Q1 real:** ¿el sistema debe reportar `UNAVAILABLE` para
   `coverage_previous` y `paired_*`, o hay algun valor por defecto
   contractual que desconozco?

## 7. Evidencia adjunta

- `probe_diagnose_h101.py` - probe deterministico del diagnostico.
- `diagnose_output.txt` - salida cruda.
- `diagnose_result.json` - resultado estructurado.
- `README.md` - A.6.4 reclasificado.
- `ESTADO_DECLARADO.md` seccion 8 - tabla de hallazgos pendientes.

## 8. Prohibiciones respetadas

- NO se ha modificado `coverage.py`, `catalog_p38_adapter.py`,
  `period_state.py`.
- NO se ha tocado `nipc.py`, `delta_shares.py`, `security_identity.py`,
  `temporal_validity.py`.
- El probe diagnostico NO altera comportamiento productivo (solo lectura
  + analisis).
