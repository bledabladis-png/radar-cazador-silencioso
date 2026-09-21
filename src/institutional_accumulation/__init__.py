"""Modulo Institutional Accumulation Evidence (IAE).

Analisis de posiciones institucionales via SEC 13F. Deterministico,
descriptivo, auditable.

Estado (2026-09-21, prompt v6.53):
  FA-1 + FA-2 (ingestion + identity)  CERRADO / PUSHED
  B2-PIT                              CERRADO (#53)
  B1                                  CERRADO (#68)
  B3                                  IMPLEMENTADO SPEC-SIDE
  Gap spec->codigo §5.1-§5.5          CERRADO (#61-#67)

Regla: local-first. Sin push a main hasta validar funcionalidad
y beneficio del reporte.

Estructura:
  sec_13f/        ingestion + schema + parser + identity
  identity/       catalog_key + target_builder + period_state +
                  radar_target_catalog + target_universe + openfigi_client
  aggregation/    delta_shares + nipc + coverage + catalog_validator +
                  catalog_p38_adapter + reporting_dedup
  temporal_validity.py    P61 operational_mapping_status
  security_type.py        §3.3 + §5.4 Nivel A+B
  operational_universe.py §5.5
  timestamps.py           B3 derive + enrich
  absence.py              P63 stub
  catalog_pit.py          B2-PIT
"""