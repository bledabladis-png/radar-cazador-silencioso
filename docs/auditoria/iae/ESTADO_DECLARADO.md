# ESTADO_DECLARADO - Modulo IAE

Estado declarado de fases, prohibiciones y hallazgos del modulo IAE.
Documento vivo. Se actualiza cuando cambia el estado.

**Actualizado:** 2026-09-23

---

## 1. Fases

| Fase | Estado |
|---|---|
| FA-1 (ingestion + schema) | CERRADA |
| FA-2 (identity + relationships) | CERRADA |
| NIPC (implementacion) | CERRADA |
| Fase A - bloqueantes dictamen v1 (B1, B2, B3, B4) | CERRADA |
| Fase B - criticos dictamen v1 (C1-C18) | CERRADA |
| Fase C - funcionales (C8, C9, C10) | CERRADA |
| Fase D - reenviar al auditor externo | PENDIENTE |
| Fase E - integracion a `run.py` | CERRADA 2026-09-23 |

Dictamenes externos aplicados al modulo IAE (todos cerrados):
v2 (8 puntos), v3 (6), v4 (11), v5 (GATE 1 nomenclatura + GATE 2
identidad dual), v6 (auditoria empirica A.1), v7 (cierre final
IAE_MAESTRO). Detalle en §2.

---

## 2. Hallazgos cerrados

- H-05 a H-12, B-01 a B-07: ver historial en git.
- Fase A: B1 reconciliacion radar/complemento/full; B2 cadena forense
  (HEAD + sha256 + pandas + timestamp); B3 validacion externa OpenFIGI
  (227/227 match sobre 243 CUSIPs resolubles por OpenFIGI, 16 sin hit
  no-USA); B4 riesgo PIT declarado (§13.9).
- Fase B: C5 tabla universos, C6 aviso §4, C7 cadena contractual,
  C11 semantica NIPC, C12 frase eliminada, C13 invocadas != verificadas,
  C14 27 listados + 15 ausentes = 42 ficheros (criterio AST), C15 rename crosswalk, C16 nota 14.3.x,
  C18 coherencia numerica.
- Fase C: C8 reporting_dedup diagnostico separado, C9 stats observables,
  C10 coverage_available + coverage_quality.
- Dictamen v2: HEAD unificado via ESTADO_SISTEMA; 83% -> 81%;
  4 -> 6 ficheros; censo 27+15=42; C9/C10 en E2E; OKE via etf_holdings
  TEMPORAL_UNVERIFIED.
- Dictamen v3 (auditor externo, 2026-09-23): 6 puntos cerrados.
  Cobertura real sobre los 759 tests: 92% (antes declarado 81% sobre
  los 557 seleccionados por -k). Contractual vs PROXY separado en
  IAE_MAESTRO §10.7. Umbral 0.95 etiquetado como operativo (gobernanza)
  en §10.5. Fail-closed denom=0 en coverage.py: coverage_quality
  UNAVAILABLE, no PARTIAL (fix + test de regresion). Nota C9 en §12.3
  (E2E no ejercita exclusion; validado por tests con fixture). §5.2
  reformulado sin prediccion. Censo IAE 758 -> 759 tests (42 ficheros).
  Commits: 89b98e9, 97dd12d, 373c75d.
- Dictamen v4 (auditor externo, 2026-09-23): 11 puntos.
  3 bloqueantes cerrados: cobertura 92% reproducible via
  scripts/iae_coverage.py; separadas cobertura catalogo (99,17%) y
  cobertura TARGET construido (100%), con endogeneidad declarada en
  §13.5; §13.7 retitulado a "Alcance de posiciones del NIPC" con
  filtro §10.1 explicito.
  7 importantes cerrados: ref cruzada §10.7 (§10.3 -> §10.5); §13.3
  error historico (valores 20/31/31/55/64/79 vs actuales 87/89/93/91/98/96);
  frase OpenFIGI reformulada (evidencia sobre la muestra, no sobre el
  universo); invariante C9 (n_received = suma disjunta ordenada);
  fail-closed Σw(TARGET_PAIRWISE)=0 explicito en §10.5; reporting_dedup
  35 funciones AST / 64 casos pytest; cabecera con HEAD operativo +
  snapshot de mediciones (§2.1, §12) en 7caa86b.
  1 importante diferido: evidencia externa OpenFIGI no archivada con
  hash/timestamp; registrado como deuda activa en §3.
  Commits: 4e6b012, 7a4c38a, a2c64b5.
- Dictamen v4, punto 7 (evidencia externa OpenFIGI): CERRADO 2026-09-23.
  Script `iae_validate_crosswalk_openfigi.py` versiona los artefactos
  (_input.json, _raw.json, _summary.txt, HASHES_*.txt) con timestamp UTC,
  parametros, hashes y script_version. Fix str/bytes latente en el
  bloque de escritura de HASHES. 5 tests con fixture mock cubren la
  generacion de artefactos (verificado: script_version en input/summary/
  hashes). Commits: 0d5b3ab, 47cba12.
- Dictamen v5 (auditor fresco, 2026-09-23): 15 puntos. GATE 1 cerrado
  (nomenclatura catalog/TARGET, IAE_MAESTRO §1, §13.5). GATE 2
  verificado empiricamente: 0 fragmentacion por shareClassFIGI en la
  configuracion E2E auditada. §13.10 reformulada: dualidad
  `equity:` / `figi:` = capacidad teorica del resolver, no observada
  en E2E con `figi_lookup=None`. Commits: 66557fb, 6869a85.
- Dictamen v6 (auditor externo, 2026-09-23): GATE 2 cerrado
  empiricamente sobre los 4.780.572 units del E2E. Script nuevo
  `scripts/iae_identity_uniqueness_audit.py`. 0 shareClassFIGI
  fragmentadas, 0 uso de rama `figi:*`, delta NIPC = 0 al normalizar.
  §12.5/§12.6 corregidos para reconocer `canonical_security in
  {equity:*, figi:*}`. 4 puntos menores aplicados (§13.1 OpenFIGI
  suavizado, §13.9 titulado ASSUMPTION, §4.3 source_raw + derivation,
  §1 marca de snapshot). Commits: d3e5fa2, bbbd7d2, 0eeb6a2.
- Dictamen v7 (auditor externo, 2026-09-23): cierre final.
  §12.4 corregido (cobertura interna TARGET 1,0 / catalogo declarado
  99,17%). §13.10 retitulado. §12.6 reformulado con conjunto canonico
  explicito. Commits: 0772272, 8b00303.

---

## 3. Deuda activa

- Integracion a `run.py` completada el 2026-09-23 (fase adicional del
  pipeline productivo, + seccion del reporte diario). La integracion a
  `daily_run.yml` queda cubierta porque `daily_run.yml` invoca `run.py`.
- `compute_nipc_contractual` sin callers productivos.
- `build_effective_reporting_snapshot` sin callers productivos.
- `scripts/iae_contractual_coverage.py` reproduce §12.3; pendiente
  integrarlo al flujo continuo de validacion.
- SPCX en catalogo radar sin fuente de identidad (emisor privado).
- Cobertura FIGI del 12,2% en INFOTABLE. Un fallback CUSIP -> FIGI
  via OpenFIGI podria ampliar la cobertura, pero no se ha demostrado
  que resuelva exhaustivamente los CUSIPs restantes (ver IAE_MAESTRO
  §13.1 y §12.4).
- Deuda semantica del nombre `NIPC` si se expone en reporte (§5.4).
- Identidad canonica dual (`equity:<TICKER>` / `figi:<FIGI>` en
  `canonical_security`): capacidad del resolver no observada en la
  configuracion E2E auditada. Auditoria empirica A.1 sobre 4.780.572
  units: 0 shareClassFIGI bajo dos canonical_security, 0 uso de la
  rama `figi:*`, delta NIPC = 0 al normalizar por shareClassFIGI.
  Ver IAE_MAESTRO §13.10. Reabrir si se activa `figi_lookup` en algun
  punto del pipeline, si aparece un caso `figi:*` en units o delta,
  o si se modifica el modelo de resolucion de identidad.

Detalle en `IAE_MAESTRO.md` seccion 5.

---

## 4. Reglas de operacion

- No integracion a produccion sin validacion funcional previa.
- No push a `origin/main` hasta integracion verificada.
- No OpenFIGI masivo. Solo consultas dirigidas.
- No modificar codigo sin ciclo previo.
- Toda escritura pasa por `write_artifact_with_manifest`.
- Todo bloque que haga `git commit` debe condicionarse a tests verdes
  (`if ($LASTEXITCODE -eq 0)`).

---

## 5. Documentos vigentes

- `IAE_MAESTRO.md` - referencia unica del modulo.
- `ESTADO_SISTEMA.md` - hechos autogenerados (fuente unica de HEAD).
- `ESTADO_DECLARADO.md` - este documento.

### Scripts IAE publicados

- `scripts/iae_test_census.py` - censo AST de tests del modulo.
- `scripts/iae_coverage.py` - cobertura reproducible sobre los 759 tests.
- `scripts/iae_contractual_coverage.py` - cadena contractual completa.
- `scripts/iae_reconciliation_b1.py` - reconciliacion radar + complemento = full.
- `scripts/iae_validate_crosswalk_openfigi.py` - validacion externa OpenFIGI.
- `scripts/iae_identity_uniqueness_audit.py` - auditoria unicidad por shareClassFIGI.
- `scripts/iae_pipeline.py` - orquestador ingestion + identity.
- `scripts/build_catalog_csvs.py` - construccion de catalogos CSV.

Documentos historicos (no normativos): `NIPC_CONTRATOS_SEMANTICOS_v1.md`,
`NIPC_COVERAGE_POLICY.md`, `INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md`.
Se conservan como documentacion de diseno; no son contratos vigentes.

---

**Nota:** los contratos P60-P70 en `NIPC_CONTRATOS_SEMANTICOS_v1.md`
se mantienen como documentacion historica de como se diseno el modulo.
No son normativos vigentes.
