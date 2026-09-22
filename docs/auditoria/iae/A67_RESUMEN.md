# A.6.7 - Respuesta a dictamen #76

**Para:** auditor externo del proyecto IAE.
**De:** Ingeniero Supervisor.
**Fecha:** 2026-09-22. **HEAD:** `c40896d`. **Push:** NO.

---

## 1. Que hemos hecho desde #76

Todo lo autorizado en #76 esta cerrado:

- Fix A2: H-05/H-06/H-07/H-10.1 corregidos (5 commits).
- Saneamiento: B-01/B-05/B-06/B-07 + H-08 cerrados.
- Cobertura: 110/110 funciones publicas IAE con test directo.
- Curacion: ONB + PTGX anadidos al crosswalk (18 -> 20 keys).
- Determinismo: verificado bit a bit en probes A.6.4 y P70.
- CI-clean: suite limpia en clone fresco (los 3 failed locales son ambientales por parquets stale).

**Suite local:** 3 failed (preexistentes, `test_freshness`) + 1415 passed + 2 skipped.
**Suite CI:** 0 failed.

## 2. Verificacion tecnica: 3 correcciones al dictamen #76

Antes de pedirte nuevas autorizaciones, verificamos codigo y datos. El
diagnostico de #76 era incorrecto en 3 puntos:

### B-02 (P62/PIT) - P62 SI esta implementado

- `src/institutional_accumulation/catalog_pit.py` (232 lineas, 16 tests).
- Incluye `target_catalog_as_of(period_end)` con reglas explicitas:
  0 snapshots cubren -> `CatalogNotAvailable`; 1 -> exito; >1 -> `CatalogAmbiguous`.
- Tests existentes: `test_as_of_backdating_prohibido_q1_2026`, `test_as_of_0_snapshots_que_cubren_raise`.
- **El sistema ya rechaza aplicar el snapshot actual a Q1 2026.**

**El bloqueo real no es codigo PIT. Es ausencia de snapshot historico**
(el unico registrado tiene `valid_from=2026-09-21`, posterior a `period_end=2026-03-31`).

### B-03 (TARGET) - universo contractual = 242 keys, no 35.649

- El universo contractual es `radar_target_catalog.csv`: 242 keys (240 OK + 2 MISS).
- Los ~35.000 CUSIPs del 13F son securities observadas que NO pertenecen al radar. Estado correcto: `OBSERVED_ONLY` (contrato P60).
- **No hace falta OpenFIGI masivo.** El hueco real son 2 keys MISS: BRK-B y MOG-A (2 re-queries OpenFIGI, no 24.838).

### B-04 (pairwise) - Q4 SI tiene datos; el cuello es vigencia temporal

- Q4 SUBMISSION: 10.676 filings con `PERIODOFREPORT=2025-12-31`.
- 498 CUSIPs Q4 resuelven `CANONICAL` via `etf_holdings` (sin vigencia declarada).
- §5.5 exige `CANONICAL + VERIFIED`. Sin vigencia -> `TEMPORAL_UNVERIFIED`. **Q4 = 0 operacionales, correcto por diseño.**
- Las 24 exceptions vigentes cubren solo Q1 (`valid_from=2026-03-31`), no Q4.
- **El bloqueo real es vigencia temporal de fuentes curadas, no ausencia de datos.**

## 3. Estado del sistema

| Elemento | Estado |
|---|---|
| H-05/H-06/H-07/H-10.1 | CERRADOS |
| H-08 | CERRADO (test) |
| B-01/B-05/B-06/B-07 | CERRADOS |
| B-02 (P62) | Implementado; bloqueo = snapshot historico |
| B-03 (TARGET) | 240/242 keys; 2 MISS (BRK-B, MOG-A) |
| B-04 (pairwise) | Bloqueado por vigencia temporal de fuentes |
| P38 | GO CONDICIONADO |
| A.6.6 | NO-GO #76 (pendiente nuevo dictamen) |
| Push | NO (local-first IAE) |

## 4. Peticion al auditor

**4 preguntas concretas. Respuesta binaria o breve en cada una.**

**P1. B-02 (P62/PIT).** Se acepta la correccion material (P62 implementado; bloqueo = ausencia de snapshot historico) y se autoriza:
  - (a) declarar la limitacion sin buscar snapshot retroactivo, o
  - (b) construir snapshot retroactivo con justificacion documental?

**P2. B-03 (TARGET).** Se acepta que el universo contractual = 242 keys (no 35.649) y se autoriza:
  - (a) 2 re-queries OpenFIGI sobre BRK-B/MOG-A, o
  - (b) declarar 240/242 (99.17%) como cobertura suficiente y cerrar B-03?

**P3. B-04 (pairwise).** Se acepta la correccion material (Q4 tiene datos; cuello = vigencia de fuentes) y se autoriza:
  - (a) 20-30 exceptions Q4 con vigencia cruzada para demostrar rama pairwise, o
  - (b) diferir pairwise hasta que exista snapshot real post-Q1?

**P4. A.6.6.** Con P1+P2+P3 resueltos (opciones a/a/a o b/b/b), se autoriza emitir A.6.6 / F2.4-CLOSE con P38 en su estado final declarado?

## 5. Anexos (solo si se solicitan)

Este documento es autocontenido. Si quieres verificar algun punto:

- Codigo PIT: `src/institutional_accumulation/catalog_pit.py`.
- Tests PIT: `tests/test_catalog_pit.py`.
- Probe A.6.4: `docs/auditoria/iae/evidence/a64_integration_b1_p61_p38/`.
- Expedientes detallados: `B02_EXPEDIENTE.md`, `B034_EXPEDIENTE.md`.
- Diffs completos: `A66_DIFFS.txt`.
- Estado: `ESTADO_DECLARADO.md`, `ESTADO_SISTEMA.md`.

---

Fin. HEAD `c40896d`. 2026-09-22.
