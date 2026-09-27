# Golden versionado del IAE

Directorio de snapshots inmutables para el control E2E contractual del
modulo IAE. Estructura propuesta por el auditor externo (hallazgo H4,
ver `AUDITORIA_EXTERNA_2026-09-27.md` seccion 3.3).

## Ficheros

| Fichero | Estado | Proposito |
|---|---|---|
| `12_5_historic.json` | FROZEN | Snapshot del par Q4 2025 -> Q1 2026 con crosswalk de 7caa86b. Inmutable. Reproducible restaurando el crosswalk. |
| `current.json` | BLOCKED | Snapshot del par vigente post-H1-B. Se completa tras cierre del fix. Actualmente todos los campos son null. |

## Reglas

1. **`12_5_historic.json` es inmutable.** No se reescribe. Reproduce
   el estado del pipeline en el commit 7caa86b, con el crosswalk
   anterior a la regeneracion por Fase G.
2. **`current.json` no se completa hasta que H1-B este cerrado.**
   No se congela ningun NIPC baseline mientras H1-B sea bloqueante.
3. **Cada golden incluye:** hashes de crosswalk + catalog + equivalence
   + parquets de INFOTABLE, campos de periodo, y `expected` con el
   resultado del E2E contractual.
4. **`security_type_source` + `security_type_source_hash`** quedan
   declarados como contrato. Son `null` en `12_5_historic.json`
   (pre-fix) y se rellenan en `current.json` (post-fix) apuntando a
   la Official List SEC usada como fuente de tipo.

## Reproduccion

- Historic: `py scripts/iae_contractual_nipc_e2e.py` con crosswalk
  restaurado al de `7caa86b`. Esperado: reproduce el golden con
  tolerancia 0.
- Current: tras cierre de H1-B, ejecutar el E2E sobre el estado del
  pipeline y volcar el resultado.

## Referencias

- `AUDITORIA_EXTERNA_2026-09-27.md` - dictamen del auditor (H4).
- `DISENO_H1B.md` - plan de cierre de H1-B (seccion 8).
- `IAE_MAESTRO.md` seccion 12.5 - fuente historica del snapshot
  `12_5_historic`.