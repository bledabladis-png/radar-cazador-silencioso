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
| Fase E - integracion a `run.py` | BLOQUEADA hasta Fase D |

Dictamen del auditor v2: 8 puntos cerrados (BLOQUEANTE HEAD, 3
CRITICOS de coherencia, 2 IMPORTANTES C9/C10, incoherencia OKE,
cobertura con cautela).

---

## 2. Hallazgos cerrados

- H-05 a H-12, B-01 a B-07: ver historial en git.
- Fase A: B1 reconciliacion radar/complemento/full; B2 cadena forense
  (HEAD + sha256 + pandas + timestamp); B3 validacion externa OpenFIGI
  (227/227 match, 16 sin hit no-USA); B4 riesgo PIT declarado (§13.9).
- Fase B: C5 tabla universos, C6 aviso §4, C7 cadena contractual,
  C11 semantica NIPC, C12 frase eliminada, C13 invocadas != verificadas,
  C14 30+5 ficheros, C15 rename crosswalk, C16 nota 14.3.x,
  C18 coherencia numerica.
- Fase C: C8 reporting_dedup diagnostico separado, C9 stats observables,
  C10 coverage_available + coverage_quality.
- Dictamen v2: HEAD unificado via ESTADO_SISTEMA; 83% -> 81%;
  4 -> 6 ficheros; censo 27+15=42; C9/C10 en E2E; OKE via etf_holdings
  TEMPORAL_UNVERIFIED.

---

## 3. Deuda activa

- Integracion a `daily_run.yml` no implementada.
- `compute_nipc_contractual` sin callers productivos.
- `build_effective_reporting_snapshot` sin callers productivos.
- `scripts/iae_contractual_coverage.py` reproduce §12.3; pendiente
  integrarlo al flujo continuo de validacion.
- SPCX en catalogo radar sin fuente de identidad (emisor privado).
- Cobertura FIGI del 12,2% en INFOTABLE.
- Deuda semantica del nombre `NIPC` si se expone en reporte (§5.4).

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

Documentos historicos (no normativos): `NIPC_CONTRATOS_SEMANTICOS_v1.md`,
`NIPC_COVERAGE_POLICY.md`, `INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md`.
Se conservan como documentacion de diseno; no son contratos vigentes.

---

**Nota:** los contratos P60-P70 en `NIPC_CONTRATOS_SEMANTICOS_v1.md`
se mantienen como documentacion historica de como se diseno el modulo.
No son normativos vigentes.
