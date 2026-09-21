# A.6.4 / A.6.6 - Integracion B1 + P61 + P38

**Origen:** dictamen #73 + #74 (H-73.1) + auditoria externa 2026-09-21
(H-05/H-06/H-07/H-10.1) + fix A2 (5 commits, 2026-09-21).
**Estado:** evidencia empirica POST fix A2. El probe ya no usa mock
Q4=Q1; el adapter propaga operational_mapping_status y weight desde
el state; coverage.py aplica Q5 literal (Q4 vacio -> UNAVAILABLE).

---

## Alcance de esta evidencia (post fix A2)

**Este probe demuestra:**

1. La cadena P61 seccion 5.5 -> B1 build_target -> period_state ->
   catalog_p38_adapter -> compute_contractual_coverage se ejecuta
   sobre datos reales 13F Q4 2025 y Q1 2026.
2. El adapter PROPAGA `operational_mapping_status` y `weight` desde
   el state (fix A2 c3, cierra H-10.1 y H-07).
3. SSHPRNAMT real se transporta desde el canonical_snapshot a
   `PositionRecord.weight` (fix A2 c2).
4. `coverage.py` usa denominador TARGET (no observed) y aplica Q5
   literal (fix A2 c4).
5. Q4 vacio produce `coverage_previous=None` (no 1.0). Q1 produce
   `coverage_current=1.0` calculado sobre el TARGET_Q1 MATERIALIZADO en
   esta ejecucion (20 FIGIs, tras curacion ONB/PTGX del 2026-09-22).
   **NO es certificacion historica plena de
   cobertura contractual Q1**: P62/PIT sigue pendiente (el snapshot
   tiene valid_from=2026-09-21, posterior al period_end Q1=2026-03-31)
   y el universo B2-PIT completo (242 keys) no esta materializado.

**Este probe NO demuestra (limitaciones vigentes):**

- Flujo PIT completo. `target_catalog_as_of` no se invoca: el snapshot
  B2-PIT vigente tiene `valid_from=2026-09-21` y no cubre Q4 2025 / Q1
  2026 por PIT. La cadena se ejecuta con build_target directo.
- Materializacion completa del universo 13F. El subconjunto con FIGI
  comun Q4<->Q1 es 20 keys tras curacion ONB/PTGX. El universo B2-PIT
  completo es 242 keys.
- Thresholds NIPC. Siguen UNDEFINED.

## Resultado empirico (result.json)

    coverage_previous              = None    (Q4 vacio, Q5 literal)
    coverage_current               = 1.0     (20/20 VERIFIED sobre TARGET_Q1)
    paired_security_coverage       = None    (TARGET_PAIRWISE = 0)
    paired_weighted_share_coverage = None
    coverage_status                = UNAVAILABLE

Cardinalidades (H-06):

    target_q4_figi          = 0
    target_q1_figi          = 18
    target_pairwise_figi    = 0
    records_q4_verified     = 0
    records_q1_verified     = 20
    paired_figi             = 0

Contraste pre/post fix A2 (mismo dataset, mismo subconjunto):

    Pre  fix: coverage_previous=1.0 (mock Q4=Q1), coverage_current=1.0,
              paired_*=1.0, weight=1.0 en todos los records.
              ARTEFACTO estructural (H-10.1 + H-07).
    Post fix: coverage_previous=None, coverage_current=1.0 sobre
              TARGET_Q1 real, paired_*=None, weight=SSHPRNAMT real.
              Resultado contractual con Q5 literal.

## Reproducibilidad

    py docs\auditoria\iae\evidence\a64_integration_b1_p61_p38\probe_integration_b1_p61_p38.py

Requiere los parquets en `data/sec_13f/processed/` + official list en
`D:\13f_probe\official_list_13f\` + snapshot B2-PIT + mappings.
Hashes de todos los inputs y outputs en `HASHES.txt`.

## Referencias

- Contrato P38: `docs/auditoria/iae/NIPC_CONTRATOS_SEMANTICOS_v1.md`
- Expediente del fix: `docs/auditoria/iae/EXPEDIENTE_A64_FIX.md`
- Estado de fases: `docs/auditoria/iae/ESTADO_DECLARADO.md`
- Bundle A.6.6: `docs/auditoria/iae/A66_BUNDLE.md`
