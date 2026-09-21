# A.6.4-v2 - Evidencia `compute_contractual_coverage` sobre TOP 2000

**Origen:** dictamen #72 (A.6.3 CERRADO, A.6.4 autorizado).
**Objetivo:** reejecutar la evidencia del piloto TOP 2000 bajo la ruta
CONTRACTUAL P38 (`compute_contractual_coverage`), en lugar del proxy
legacy.

## Alcance

**TARGET:** 210 CUSIPs del piloto `nipc_gate0_target_identity_top2000`
con `target_membership=True`. FIGIs resueltos por OpenFIGI durante el
piloto (evidencia cerrada F2.3-bis). El TARGET se hereda, no se
recalcula.

**Records:** construidos desde el canonical Q4 2025 / Q1 2026
(`filter_by_period` + `apply_amendments`), filtrando a los 210 CUSIPs
del piloto. Cada PositionRecord hereda el FIGI resuelto por el piloto
y declara `resolution_status=CANONICAL` + `operational_mapping_status=VERIFIED`.

**NO ejerce P61.** El probe aísla la ruta contractual P38. P61 solo
resuelve ~520 CUSIPs del corpus completo; cruzarla con el piloto
requeriría decidir tratamiento de los 210 CUSIPs que P61 no clasifica
como CANONICAL. Eso se deja como trabajo posterior si el auditor lo
requiere.

**NO ejerce B1.** No usa `catalog_p38_adapter`, `target_builder` ni
`operational_universe`. Es un test mínimo de la firma P38 contractual.

## Resultado

| Metrica | Valor |
|---|---|
| target_q4, target_q1 | 210 FIGIs |
| records_q4 | 210 |
| records_q1 | 210 |
| coverage_previous | 1.0 |
| coverage_current | 1.0 |
| paired_security_coverage | 1.0 |
| paired_weighted_share_coverage | 1.0 |
| coverage_status | **VALID** |
| unmapped_count_previous | 0 |
| unmapped_count_current | 0 |

## Lectura

`compute_contractual_coverage` ejecuta correctamente sobre el TARGET
real del piloto. Producido el output VALID esperado cuando el
universo está completamente resuelto (210/210 FIGIs).

Esto demuestra que:
1. La firma contractual de `compute_contractual_coverage` es consumible.
2. El calculo de las 6 metricas funciona sin proxy legacy.
3. La ruta CONTRACTUAL está disponible cuando hay TARGET + records.

**NO demuestra:**
- Que P61 esté integrado en la ruta contractual.
- Que B1 (adaptador) esté integrado.
- Que el universo completo sea procesable.
- Certificación de cobertura histórica.

## Limitaciones

- Solo 210 CUSIPs (los target_membership=True del piloto).
- Records con `resolution_status` y `operational_mapping_status`
  hardcoded. El probe aísla la firma P38, no la cadena completa.
- No se ha ejercitado el flujo normativo B1 (PASO 0-10).

## Uso

    py probe_top2000_contractual.py

Deterministico. Sin datetime.now(). Sin red. Sin OpenFIGI.

## Referencias

- Piloto original: `evidence/nipc_gate0_target_identity_top2000/`.
- Spec: `INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md` §3.14.
- Contrato: `NIPC_CONTRATOS_SEMANTICOS_v1.md` §3 y §4.
- Dictamenes: F2.4, #72.