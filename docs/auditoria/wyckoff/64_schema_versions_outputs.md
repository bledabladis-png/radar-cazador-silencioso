# 64 - Schema versioning en outputs (Fase 1)

**Fecha:** 2026-10-25
**Commits:** `c9a848b5` (codigo) + `a88e85da` (CSV regenerado)
**Estado:** CERRADA. Fase 1 de 6.

## 1. Motivacion

El commit `415487ff` (sesion 3, 24-oct 11:16) anadio
`ACCUMULATION_FORMING`, `DISTRIBUTION_FORMING` y `CONFLICT` al schema de
`sector_wyckoff_distribution.csv`. El CSV existente tenia 330 filas v1
(sin esas columnas) y el pipeline anadio 11 filas v2 con `append_dedup`
ciego. Resultado: historico mezclado sin senal. Ni `persist()` ni el
gate ni `data_quality` lo detectaron.

Detectado 2026-10-25 durante auditoria post-sesion-4. Ver HANDOFF
sesion 4 seccion 4 (diagnostico).

## 2. Contrato nuevo

`OutputSpec` gana `schema_version: int = 1`. Los 24 outputs actuales
quedan en v1 por defecto. Solo `sector_wyckoff_distribution` bumpea a
v2 (por el cambio de `415487ff`).

`persist()` lee `<path>.schema` (fichero lateral con un entero). Reglas:
- Sin `.schema` ni CSV: escribe normal, anota `.schema`.
- `.schema` == spec.schema_version: append normal.
- `.schema` != spec.schema_version: COPIAR el CSV a `<path>.v{old}.csv`
  (backup), NO borrarlo, escribir CSV nuevo SOLO con las filas del run.
  No appendear (evita mezclar v1 con v2). Preserva la garantia de
  "si to_csv falla, el CSV viejo sigue intacto" (escritura atomica).
- `.schema` no existe pero CSV sí: asumir v1 (legacy).

`persist()` escribe `.schema` con `spec.schema_version` tras el
`tmp.replace` exitoso. Si falla la escritura del `.schema`, se ignora
(es metadata local).

## 3. Gate check 11

`run_validation_gate` verifica que todos los outputs `temporal` con
`.schema` coinciden con `spec.schema_version`. Reporta `MIGRATED` si
hay `<path>.v{old}.csv` al lado (transicion legitima). Falla si hay
mismatch no migrado. Total checks: 10 -> 11.

## 4. Politica de archivos

- `<path>.csv.schema`: metadata local, gitignored (`outputs/**/*.schema`).
  Fresh checkout = sin `.schema` = asume v1 = migra en primer run si spec
  bumpea. Determinista, agnostico al entorno.
- `<path>.v{old}.csv`: backup de migracion, gitignored
  (`outputs/history/*.v[0-9]*.csv`). Se queda en disco para analisis
  manual. No se commitea.

## 5. Migracion aplicada (2026-10-25)

- `sector_wyckoff_distribution.csv` (330 filas v1+v2 mezcladas,
  36500 bytes) -> `sector_wyckoff_distribution.v1.csv` (backup, untracked).
- `sector_wyckoff_distribution.csv` nuevo: 11 filas v2 del run
  2026-10-23 (1749 bytes). Columnas `count_accumulation_forming`,
  `count_distribution_forming`, `count_conflict` presentes.
- `.schema`: 2 (correcto).

## 6. Tests

`tests/test_output_schema.py`: 6 tests.
- `test_persist_primera_vez_escribe_schema`
- `test_persist_append_mismo_schema`
- `test_persist_migra_schema_distinto`
- `test_persist_legacy_sin_schema_file`
- `test_sector_wyckoff_distribution_schema_2`
- `test_resto_outputs_schema_1`

`tests/test_validation_gate.py`: `test_gate_caso_feliz` 10 -> 11.
`tests/conftest.py`: fixture autouse `_cleanup_schema_files` limpia
`.schema` tras cada test (los tests de `save_regime_history` escriben
al path real por diseno).

## 7. Verificacion

- Suite completa: 3711 passed, 2 skipped.
- Run real 15 min, gate 11/11.
- `[SCHEMA]` en el log: `sector_wyckoff_distribution: backup v1 ->
  v2 (v2 activo)`.
- CSV regenerado: 12 lineas (header + 11), cabecera con FORMING +
  CONFLICT.
- `.v1.csv` backup: 342 lineas (untracked, gitignored).
- `.schema`: 2.

## 8. Pendiente (Fase 2-6)

- **Fase 2:** `data_quality.SOURCES` reporta `schema_v` por fuente en
  tabla de frescura. Documento 64 se amplia.
- **Fase 3:** regeneracion historica completa (proyecto separado, no
  bloqueante). Protocolo en `docs/auditoria/wyckoff/65_regeneracion_historico.md`
  (pendiente).
- **Fase 4:** unificacion de los 4 caminos de "estructura sectorial"
  (`sector_regime.compute_sector_scores`, `sector_wyckoff_distribution`,
  `wyckoff_holdings_report`, `sector_regime_matrix.structure_positive`).
- **Fase 5:** `structure_score` continuo con datos coherentes.
- **Fase 6:** descarga unificada (`Router` con politica unica).

Detalle en HANDOFF seccion 4.
