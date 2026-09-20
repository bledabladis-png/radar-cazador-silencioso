# IAE - Fase A.6 Plan de ejecucion

**Objeto:** plan detallado de la fase A.6 (reconciliacion contrato<->codigo).

**Generado:** 2026-09-20.
**Precondicion bloqueante:** dictamen F2.4 (GO o GO CONDICIONADO).
**Estado:** NO EJECUTABLE hasta F2.4.
**Referencia:** iae/RECONCILIACION_CONTRATO_CODIGO.md + iae/REESTRUCTURACION_MODULO.md.

---

## 1. Objetivo

Cerrar las 3 divergencias contrato<->codigo y recomputar la evidencia
empirica (baseline + TOP 2000) bajo la semantica contractual correcta.

Resultado esperado de A.6:

- Codigo alineado con contrato P38/P60/P61.
- Tests contractuales nuevos.
- Baseline + TOP 2000 recalculados.
- F2.4 definitivo con evidencia actualizada.
- Gate-NIPC.2 desbloqueado.

---

## 2. Sub-fases

### A.6.1 - Dictamen F2.4 (EXTERNO)

**Entrada:**
- iae/RECONCILIACION_CONTRATO_CODIGO.md
- iae/REESTRUCTURACION_MODULO.md
- iae/NIPC_CONTRATOS_SEMANTICOS_v1.md
- iae/NIPC_COVERAGE_POLICY_V13_PROPUESTA.md
- iae/INFORME.md
- iae/DICTAMENES.md

**Salida:**
- Dictamen F2.4 formal, con decisiones por D1/D2/D3.

**Criterios de aceptacion:**
- Decision explicita por divergencia (A o B).
- Postura sobre OpenFIGI masivo (autorizado / no).
- Postura sobre policy v1.3 (aprobar / mantener v1.0).

**Estado:** PENDIENTE EXTERNO.

---

### A.6.2 - Fixes quirurgicos (post F2.4 = GO)

3 commits minimos, uno por divergencia.

**Commit A.6.2-D3** - P60 raise para CUSIP/ISIN
- Archivo: `sec_13f/identity/security_identity.py`.
- Cambio: `_normalize_canonical` lanza `ValueError` para CUSIP/ISIN.
- Cambio: `_find_active_equivalence` no asume default TICKER.
- Test: `tests/test_p60_contract.py` (nuevo).

**Commit A.6.2-D1** - P61 conectado al resolver
- Archivo: `sec_13f/identity/security_identity.py`.
- Cambio: `evidence` incluye `valid_from`/`valid_to`.
- Cambio: `resolve_security_identity` invoca `resolve_source_status`.
- Cambio: `crosswalk_internal` separado en `cusip_ticker_exceptions`
  vs `etf_holdings`.
- Test: `tests/test_p61_contract.py` (nuevo).

**Commit A.6.2-coverage** - Firma nueva nipc
- Archivo: `aggregation/nipc.py` + `aggregation/coverage.py` (nuevo).
- Cambio: `compute_nipc` recibe `target_q4`/`target_q1` opcionales.
- Cambio: delegacion a `compute_contractual_coverage`.
- Cambio: mover `temporal_validity.py` a `aggregation/`.
- Test: `tests/test_p38_contract.py` (nuevo).

**Criterios de aceptacion:**
- Los 3 commits compilan.
- Pyflakes limpio.
- Tests contractuales nuevos pasan.
- Tests existentes no-regresionan.

---

### A.6.3 - Test de contrato P38 (shareClassFIGI compartido)

**Objetivo:** verificar que PAIRED empareja por `canonical_security`
comun cuando el `shareClassFIGI` coincide, aunque los CUSIPs observados
sean distintos.

Escenario:

    Q4: CUSIP_A -> shareClassFIGI_X -> equity:TICKER
    Q1: CUSIP_B -> shareClassFIGI_X -> figi:BBG...

Esperado:

    TARGET_PAIRWISE = {X}
    PAIRED          = {X}

Commit: `tests/test_p38_contract.py` extendido.

**Criterio de aceptacion:**
- Test pasa.
- El comportamiento es el esperado aunque `canonical_security(Q4) !=
  canonical_security(Q1)`.

---

### A.6.4 - Recalculo baseline + TOP 2000

**Precondicion:** A.6.2 y A.6.3 cerrados.

**Accion:**
- Reejecutar `probe_coverage_baseline.py` con firma nueva.
- Reejecutar `run_pilot_13f_top2000.py`.
- NO sobrescribir los ficheros historicos (`evidence/`).
- Escribir nuevos con sufijo `_v2`.

**Entregable:**
- `evidence/nipc_gate0_baseline_v2/`
- `evidence/nipc_gate0_target_identity_top2000_v2/`

**Criterio de aceptacion:**
- Ambas ejecuciones terminan exit 0.
- Los valores `paired_weighted_share_coverage` cambian respecto a la
  version historica.
- El resto de metricas se mantiene o cambia con justificacion.

---

### A.6.5 - Actualizar NIPC_CONTRATOS_SEMANTICOS_v1 -> v2 (si aplica)

Solo si A.6.1 autoriza cambios de contrato. Si no, el contrato v1
permanece intacto.

**Criterio de aceptacion:**
- Contrato actualizado in-place (regla 1 concepto = 1 fichero).
- Seccion "Cambios respecto a v1" anadida.
- Git conserva la version previa.

---

### A.6.6 - Emitir F2.4 definitivo

**Entrada:**
- Codigo alineado con contrato (A.6.2).
- Tests contractuales (A.6.3).
- Nueva evidencia (A.6.4).
- Contrato actualizado si aplica (A.6.5).

**Salida:**
- Dictamen F2.4 definitivo.
- Propagacion a FOLLOWUPS.md y PROMPT_MAESTRO.

**Criterio de aceptacion:**
- Gate-NIPC.2 desbloqueado.
- Baseline v2 + TOP 2000 v2 disponibles.
- Fase A.7 (Breadth, New/Exit, clasificacion) autorizable.

---

## 3. Orden de dependencia

    A.6.1 (F2.4)
        |
        v
    A.6.2 (3 commits quirurgicos) --+-- A.6.3 (test shareClassFIGI)
        |                            |
        v                            v
    A.6.4 (recalculo evidencia)
        |
        v
    A.6.5 (actualizar contrato si aplica)
        |
        v
    A.6.6 (F2.4 definitivo)

Los pasos A.6.2-D3, A.6.2-D1, A.6.2-coverage pueden ejecutarse en
paralelo si se hace un backup por cada uno.

---

## 4. Bloqueos y precondiciones

    | Bloqueo                       | Afecta a    | Desbloquea con      |
    |-------------------------------|-------------|---------------------|
    | F2.4 pendiente                | Todo A.6    | Dictamen externo    |
    | OpenFIGI masivo NO AUTORIZADO | A.6.2-D2    | Autorizacion F2.4   |
    | THRESHOLD_1/2 UNDEFINED       | A.6.6 final | A.6.4 + auditor     |
    | Gate-NIPC.2 BLOQUEADO         | A.7         | A.6.6               |

Los fixes D1/D3 y el fix de cobertura (firma nueva nipc) NO dependen
de OpenFIGI. Solo el D2 completo (TARGET real) lo requiere.

**Opcion estrategica:** implementar D1+D3+coverage en A.6.2 y dejar D2
para A.6.3 o A.6.4 si F2.4 no autoriza OpenFIGI todavia.

---

## 5. Reglas de ejecucion

- **Local-first IAE.** NO push hasta Gate-NIPC.3.
- **Un cambio = una verificacion = un commit.**
- **NO tocar `delta_shares.py`, `match_key`, C2.**
- **NO modificar policy v1.0** (hash 57f2d01f...).
- **Backup + snapshot pre/post** en cada cambio estructural.
- **Commit por sub-fase.** No mezclar D1 con D3.
- **Rollback quirurgico** si algun paso falla.

---

## 6. Criterios de aceptacion globales de A.6

    | Criterio                                       | Umbral          |
    |------------------------------------------------|-----------------|
    | Codigo alineado con contrato P38/P60/P61       | 100%            |
    | Tests contractuales nuevos                     | >= 11           |
    | Tests globales                                 | >= 979 + 11     |
    | Pyflakes                                       | 0 warnings      |
    | compileall                                     | OK              |
    | Baseline v2 + TOP 2000 v2                      | escritos        |
    | F2.4 definitivo emitido                        | si              |
    | Gate-NIPC.2 desbloqueado                       | si              |

---

## 7. Estimacion de esfuerzo (post F2.4=GO)

    A.6.2-D3          1 commit, 1 test nuevo              ~30 min
    A.6.2-D1          1 commit, 1 test nuevo              ~60 min
    A.6.2-coverage    1 commit, 1 test nuevo + move       ~90 min
    A.6.3             extension de test P38               ~30 min
    A.6.4             reejecucion baseline + TOP 2000     ~60 min
    A.6.5             actualizar contrato si aplica       ~30 min
    A.6.6             F2.4 definitivo (externo)           -

    Total estimado (supervisor): ~5 horas de trabajo tecnico
    + dictamen externo F2.4 (variable).

---

## 8. Referencias

    | Documento                                    | Rol                |
    |----------------------------------------------|--------------------|
    | iae/RECONCILIACION_CONTRATO_CODIGO.md        | Divergencias       |
    | iae/REESTRUCTURACION_MODULO.md               | Plan arquitectonico|
    | iae/NIPC_CONTRATOS_SEMANTICOS_v1.md          | Contrato           |
    | iae/NIPC_COVERAGE_POLICY_V13_PROPUESTA.md    | Policy propuesta   |

---

Fin del plan A.6. Bloqueado hasta F2.4.
