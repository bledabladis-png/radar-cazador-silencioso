# A.6.8 - Respuesta a dictamen #77

**Para:** auditor externo del proyecto IAE.
**De:** Ingeniero Supervisor.
**Fecha:** 2026-09-22. **HEAD:** `43781c4`. **Push:** NO.

---

## 1. Que hemos hecho desde #77

Los tres ciclos autorizados en #77 estan cerrados:

- **B-02**: probe declara `PIT_UNAVAILABLE`. `coverage_current` anulado
  como contractual. Modo `--no-pit` disponible como bypass documentado.
- **B-03**: BRK-B + MOG-A materializados via OpenFIGI (2 re-queries,
  sin masivo). Catalogo 240/242 -> 242/242 keys. Snapshot B2-PIT
  `20260922_01` creado (el anterior `20260921_01` preservado, P62).
- **B-04**: fixture pairwise `NON-PRODUCTION / CONTRACT_TEST_FIXTURE`
  (no toca mappings productivos). Rama pairwise ejercitada.

**Nada no autorizado ejecutado**: sin retro-fechado, sin OpenFIGI
masivo, sin exceptions ad-hoc.

**Suite local:** 1421 passed + 2 skipped + 3 failed (`test_freshness`,
ambientales).
**Suite CI:** 0 failed.

## 2. Evidencia por ciclo

**B-02** (probe A.6.4, modo strict-pit):

    pit_status:            PIT_UNAVAILABLE
    snapshot_used:         None
    reason:                sin snapshot que cubra 2026-03-31
    coverage_current:      None (anulado como contractual)
    coverage_status:       PIT_UNAVAILABLE
    coverage_contractual:  False

**B-03** (provenance en `data/mappings/openfigi_requeries/`):

    BRK-B:  CUSIP 084670702 -> shareClassFIGI BBG001S90346
    MOG-A:  CUSIP 615394202 -> shareClassFIGI BBG001S5T922
    Probe A.6.4: catalog_keys Q1 20 -> 22

**B-04** (fixture, `result_pairwise_fixture.json`):

    TARGET_PAIRWISE = {A, D};  PAIRED = {A}
    paired_security_coverage        = 0.5
    paired_weighted_share_coverage  = 6/7
    coverage_status                 = VALID

## 3. Estado del sistema

| Elemento | Estado |
|---|---|
| H-05/H-06/H-07/H-08/H-10.1 | CERRADOS |
| B-01/B-05/B-06/B-07 | CERRADOS |
| B-02 (P62) | Declarado PIT_UNAVAILABLE |
| B-03 (TARGET) | 242/242 keys |
| B-04 (pairwise) | Cerrado a nivel de codigo (fixture) |
| P38 | GO CONDICIONADO |
| A.6.6 / F2.4-CLOSE | PENDIENTE |
| Push | NO |

## 4. Peticion al auditor

**P1. B-04 (pairwise).** Se acepta el cierre de la rama pairwise a nivel
de codigo (fixture `NON-PRODUCTION`), manteniendo pendiente
`TARGET_PAIRWISE>0` sobre datos 13F reales (bloqueado por B-02.2)?
  - (a) Si, aceptado el cierre code-level.
  - (b) No, indicar que falta.

**P2. A.6.6 / F2.4-CLOSE.** Con B-02+B-03+B-04 cerrados y P38 en estado
final declarado (GO CONDICIONADO + PIT_UNAVAILABLE + pairwise code-level
+ THRESHOLD_1/2 UNDEFINED), se autoriza emitir A.6.6?
  - (a) Si, autorizado.
  - (b) No, identificar bloqueo material.

**P3. Saneamiento de hallazgos MEDIA.** Se acepta diferir el saneado de
H-19 (HEAD ambiguo) y H-20 (CI sin SHA) al proximo bundle, donde
quedaran etiquetados `HEAD_DEL_CODIGO_AUDITADO = 43781c4` +
`CI run: <SHA>/<resultado>`?
  - (a) Si, diferido al proximo bundle.
  - (b) No, requieren ciclo propio.

## 5. Anexos (solo si se solicitan)

- `result.json` (probe A.6.4, strict-pit)
- `result_pairwise_fixture.json` (B-04)
- `data/mappings/openfigi_requeries/` (B-03)
- `B02_EXPEDIENTE.md`, `B034_EXPEDIENTE.md`
- `A66_DIFFS.txt`

---

Fin. HEAD `43781c4`. 2026-09-22.
