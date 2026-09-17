# Archive - Informes de ciclos cerrados

**Creado:** 2026-09-17 (commit e941367)
**Motivo:** mantener `docs/auditoria/` en raiz como carpeta limpia y funcional, con solo los documentos normativos y vigentes.

## Criterio de archivado

Un fichero se mueve a `archive/` si cumple **todas** estas condiciones:

1. Su ciclo esta cerrado (dictamen emitido + implementacion verificada, o informe final superado por otro posterior).
2. No esta referenciado por `PROMPT_MAESTRO.md` ni por `FOLLOWUPS.md`.
3. No es decision vigente ni base normativa activa.

Los ficheros archivados siguen accesibles con `git log --follow <path>` para consultar el historial completo.


---

## Contenido

### Auditorias iniciales (2026-08-12, superadas)

- `AUDITORIA_DICTAMEN_FINAL_v4.2.md` - Aprobado 12/08. Superado por v4.3.
- `AUDITORIA_DICTAMEN_FINAL_v4.3.md` - Aprobado 12/08. Superado por informe final.
- `INFORME_AUDITORIA_FINAL.md` - Informe interno Fases 0-5, 12/08.
- `double_counting_v1.md` - Auditoria double counting 26/08.

### Planes de refactor (cerrados)

- `PLAN_REFACTOR_C1.md` - Refactor `src/report_generator.py` (C1). Completado.
- `PLAN_REFACTOR_C2.md` - Refactor `run.py` (C2). Completado.

### Briefings pre-dictamen (ya dictaminados)

- `BRIEFING_AUDITOR_CONTRATO_TEMPORAL.md` - Briefing previo FU-021-5 Parte A.
- `BRIEFING_AUDITOR_PARTE_B.md` - Briefing previo FU-021-5 Parte B.
- `CICLO2_CIERRE_AUDITORIA.md` - Cierre Ciclo 2, pre-dictamen 15/09.


### Dictamenes de ciclos cerrados

- `DICTAMEN_PARTE_B.md` - Dictamen FU-021-5 Parte B (GO CONDICIONADO 16/09). Ciclo cerrado.
- `DICTAMEN_PLAN_IMPLEMENTACION.md` - Dictamen plan FU-021-5 (GO 16/09). Ciclo cerrado.

### Informes tecnicos de ciclos cerrados

- `FU-018-1_PROVIDER_TEMPORAL_SEMANTICS.md` - Inventario semantica temporal providers (cerrado GO 15/09).
- `FU-018-2_DISENO_MODELO_TEMPORAL.md` - Contrato modelo temporal minimo (implementado en FU-018-3).
- `FU-021-3A_ADENDUM.md` - Adendum FU-021-3A. Ciclo cerrado.
- `FU-021-3A_ADENDUM_2.md` - Adendum 2 FU-021-3A. Ciclo cerrado.

---

## Reapertura

Ningun fichero de `archive/` es decision vigente. Para restaurar uno a raiz:

    git mv docs/auditoria/archive/NOMBRE.md docs/auditoria/NOMBRE.md

Si un fichero archivado necesita citarse en `PROMPT_MAESTRO.md` o `FOLLOWUPS.md`, primero restaurarlo a raiz.
