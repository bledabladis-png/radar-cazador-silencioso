# Golden versionado del IAE

Directorio de snapshots inmutables para el control E2E contractual del
modulo IAE. Estructura propuesta por el auditor externo (hallazgo H4,
ver `AUDITORIA_EXTERNA_2026-09-27.md` seccion 3.3).

## Ficheros

| Fichero | Estado | Proposito |
|---|---|---|
| `12_5_historic.json` | FROZEN | Snapshot del par Q4 2025 -> Q1 2026 en el commit 7caa86b. Inmutable. NO reproducible con HEAD actual (requiere tambien restaurar el codigo de 7caa86b). |
| `current.json` | ACTIVE_WITH_OPEN_DISCREPANCY | Baseline local reproducible del par Q4 2025 -> Q1 2026 con H1-B v2.3. Convive con la referencia externa no reproducida (-4.264.449.932). Reconciliation delta = 920 (OPEN hasta que el auditor publique comando + HEAD). |

## Reglas

1. **`12_5_historic.json` es inmutable.** No se reescribe. Reproduce
   el estado del pipeline en el commit 7caa86b, con el crosswalk
   anterior a la regeneracion por Fase G. **Correccion 2026-09-29:**
   restaurar solo el crosswalk NO reproduce el golden con HEAD. Los
   cambios posteriores H1-B v2.3 (ce32c77) y O1 (72fa824) alteran el
   calculo. Para reproducir 12_5_historic hay que restaurar tambien
   `src/institutional_accumulation/` al commit 7caa86b.
2. **`current.json` esta completo y activo desde 2026-09-28** (status ACTIVE_WITH_OPEN_DISCREPANCY). Contiene el baseline local reproducible del par Q4 2025 -> Q1 2026 con H1-B v2.3. Se mantiene con discrepancia abierta (C2, delta 920) hasta que el auditor publique comando + HEAD del valor externo.
   No se congela ningun NIPC baseline mientras H1-B sea bloqueante.
3. **Cada golden incluye:** hashes de crosswalk + catalog + equivalence
   + parquets de INFOTABLE, campos de periodo, y `expected` con el
   resultado del E2E contractual.
4. **`security_type_source` + `security_type_source_hash`** quedan
   declarados como contrato. Son `null` en `12_5_historic.json`
   (pre-fix) y se rellenan en `current.json` (post-fix) apuntando a
   la Official List SEC usada como fuente de tipo.

## Reproduccion

- Current (default): `py scripts/iae_contractual_nipc_e2e.py`
  reconcilia contra `current.json` (baseline local reproducible con
  HEAD). Gate de regresion activo.
- Historic: `py scripts/iae_contractual_nipc_e2e.py --historic`
  reconcilia contra `12_5_historic.json`. Falla con HEAD actual por
  cambios de codigo post-7caa86b. Para reproducir 12_5_historic con
  tolerancia 0 hay que restaurar codigo + crosswalk al commit
  7caa86b.

## Referencias

- `AUDITORIA_EXTERNA_2026-09-27.md` - dictamen del auditor (H4).
- `DISENO_H1B.md` - plan de cierre de H1-B (seccion 8).
- `IAE_MAESTRO.md` seccion 12.5 - fuente historica del snapshot
  `12_5_historic`.