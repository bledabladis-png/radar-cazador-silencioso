# ESTADO_DECLARADO - Modulo IAE

Estado declarado de fases, prohibiciones y hallazgos del modulo IAE.
Documento vivo. Se actualiza cuando cambia el estado.

**Actualizado:** 2026-09-22

---

## 1. Fases

| Fase | Estado |
|---|---|
| FA-1 (ingestion + schema) | CERRADA |
| FA-2 (identity + relationships) | CERRADA |
| NIPC (implementacion) | CERRADA |
| Integracion a produccion | NO IMPLEMENTADA |
| Certificacion externa | NO REQUERIDA |

---

## 2. Hallazgos cerrados

- H-05: probe sin mock Q4=Q1
- H-06: probe publica cardinalidades
- H-07: adapter propaga weight
- H-08: test ortogonal identity
- H-10.1: adapter propaga operational_mapping_status
- H-11: hashes de provenance
- H-12: hash de catalogo etiquetado
- B-01: saneamiento documental
- B-05: precision coverage
- B-06: cadena de pesos sin doble conteo
- B-07: suite CI limpia
- B-02: implementacion PIT
- B-03: materializacion catalogo 242/242
- B-04: fixture pairwise

---

## 3. Deuda activa

- Integracion a `daily_run.yml` no implementada.
- `compute_nipc_contractual` sin callers productivos.
- `build_effective_reporting_snapshot` sin callers productivos.
- Crosswalk FIGI historico (XOM, OKE).
- SPCX en catalogo actual, sin filings.
- Cobertura FIGI del 12,2% en INFOTABLE.

Detalle en `IAE_MAESTRO.md` seccion 5.

---

## 4. Reglas de operacion

- No integracion a produccion sin validacion funcional previa.
- No push a `origin/main` hasta integracion verificada.
- No OpenFIGI masivo.
- No modificar codigo sin ciclo previo.
- Toda escritura pasa por `write_artifact_with_manifest`.

---

## 5. Documentos vigentes

- `IAE_MAESTRO.md` - referencia unica del modulo.
- `ESTADO_SISTEMA.md` - hechos autogenerados.
- `ESTADO_DECLARADO.md` - este documento.
- `NIPC_CONTRATOS_SEMANTICOS_v1.md` - contrato de referencia.
- `NIPC_COVERAGE_POLICY.md` - policy de referencia.
- `INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md` - especificacion.

---

**Nota:** los contratos P60-P70 en `NIPC_CONTRATOS_SEMANTICOS_v1.md`
se mantienen como documentacion historica de como se diseno el modulo.
No son normativos vigentes.