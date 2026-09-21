# IAE - ESTADO DECLARADO

Declaraciones subjetivas del proyecto IAE. Los hechos verificables
(HEAD, tests, LOC, hashes) viven en `ESTADO_SISTEMA.md` y se regeneran
por script. Este fichero se edita cuando cambia el estado de una fase,
no cuando cambia el codigo.

**Actualizado:** 2026-09-21
**Responsable de actualizar:** Ingeniero Supervisor.
**Origen de verdad:** este fichero + `ESTADO_SISTEMA.md`. Los demas
documentos describen su tema; NO declaran el estado del sistema.

---

## 1. Fases del modulo IAE

| Fase | Estado | Referencia |
|---|---|---|
| FA-1 (ingestion SEC 13F) | CERRADO / PUSHED | - |
| FA-2 (identity: temporal_filter, cusip_resolver, relationships, amendments) | CERRADO / PUSHED | - |
| B2-PIT (infraestructura temporal catalogo) | CERRADO | #53 |
| B1 (TARGET + adaptador P38, 5 modulos) | CERRADO | #68 |
| B3 (semantica temporal 13F: timestamps + absence) | IMPLEMENTADO SPEC-SIDE (sin integracion) | - |
| Gap spec->codigo seccion 5.1-5.5 | CERRADO | #61-#67 |
| A.6.0 (Gate 0 bloqueantes F2.4) | CERRADO | #43 |
| A.6.1 (dictamen F2.4) | CERRADO | 2026-09-20 |
| A.6.2 (fixes quirurgicos P60/P61/P38) | CERRADO | - |
| A.6.2-bis-B2-PIT | CERRADO | #53 |
| A.6.2-bis-B1 | CERRADO | #68 |
| A.6.2-bis-B3 | IMPLEMENTADO SPEC-SIDE | - |
| A.6.3 (test Q12 Modelo A) | CERRADO | #72 |
| A.6.4-v2 (P38 aislado sobre TOP 2000) | CERRADO | #73 |
| A.6.4 (integracion B1+P61+P38 end-to-end) | CERRADO | #75 |
| A.6.5 (actualizar contratos in-place) | CERRADA | dccf70d + 911e95b + 0f0583d + a21b0db |
| A.6.6 (F2.4-CLOSE) | PENDIENTE - requiere dictamen externo | - |

## 2. Hallazgos cerrados

| ID | Descripcion | Dictamen |
|---|---|---|
| H-69.1 | source id exceptions -> cusip_ticker_exceptions | #70 |
| H-69.2 | colision BRK-B vs BRK.B; regla R-69.2 | #72 |
| H-73.1 | adapter B1<->P38 (weight_status mapeado a operational_mapping_status) | #75 |
| GHISALLO | +765% caracterizado como artefacto del probe (no economico) | #60 |

## 3. Contratos y policies vigentes

| Documento | Estado |
|---|---|
| `NIPC_CONTRATOS_SEMANTICOS_v1.md` | VIGENTE. P60/P61 GO, P38 GO CONDICIONADO |
| `NIPC_COVERAGE_POLICY.md` v1.0 | VIGENTE (normativa) |
| `NIPC_COVERAGE_POLICY_V11/V12/V13` | NO APROBADAS (borradores) |
| `INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md` v1.4 | Referencia, no normativa |

## 4. Prohibiciones vigentes

- NO push a main de codigo (local-first IAE)
- NO activar `compute_nipc_contractual` sin thresholds
- NO ejecutar OpenFIGI masivo
- NO activar `DROP_DUP`
- NO modificar contratos normativos sin dictamen
- NO reescribir snapshots publicados
- NO modificar `coverage.py`, `nipc.py`, `delta_shares.py`
- NO modificar `security_identity.py` ni `temporal_validity.py`

## 5. Deuda activa (no bloqueante)

- Integracion de modulos IAE al pipeline productivo (`daily_run.yml` NO los invoca)
- `compute_nipc_contractual` SIN CALLERS PRODUCTIVOS (verificado: solo tests)
- `build_effective_reporting_snapshot` SIN CALLERS PRODUCTIVOS (verificado: solo tests)
- `DROP_DUP` NO ACTIVADO (capacidad diferida v2)
- Marcadores residuales seccion 5.4 (SPONSORED ADR/ADS, ADR, SH BEN INT, FUND, ACT)
- Snapshots historicos Q4 2025 / Q1 2026 requieren OpenFIGI masivo
- Curacion crosswalk residual: ONB, SPCX, PTGX sin match 13F Q1 2026
- Gate-NIPC.2 BLOQUEADO por THRESHOLD_1/2 UNDEFINED
- A.6.6 PENDIENTE (F2.4-CLOSE)

## 6. Proxima accion autorizada

A.6.6 - emitir dictamen F2.4-CLOSE. Requiere dictamen externo.
Sin autorizacion externa, no procede.

## 7. Saneamiento documental (sub-deuda en curso)

Sesion 2026-09-21: detectadas 82 referencias fantasma en `docs/auditoria/`.

Cerrados:
- `DICTAMENES.md` (23 fantasmas, commit `10f65bb`)
- `INFORME.md` (17 fantasmas, commit `8e04d22`)
- Contradiccion A.6.5 en contrato NIPC (commits `0f0583d` + `a21b0db`)

Pendientes (no bloqueantes para A.6.6):
- `FOLLOWUPS.md` (~20 fantasmas)
- `PROMPT_MAESTRO.md` (8 fantasmas materiales; se limpian al escribir v6.56)
- `NIPC_COVERAGE_POLICY_V13_PROPUESTA.md` (2 fantasmas + 1 hash stale)
- Casos especificos de categoria C (~8)
