# IAE - P66 Gate 0.9 Multiples filings base por (CIK, PERIODOFREPORT)

**Objeto:** verificar existencia real de >1 filing base independiente
para mismo CIK + PERIODOFREPORT en el universo R4. Bloqueo C del
dictamen #30.

**Origen:** dictamen #30 (2026-09-21), bloqueo C: la regla
">1 base -> N/D o CONFLICT" requiere justificacion empirica.

**Fecha:** 2026-09-21.
**Autor:** Ingeniero Supervisor.
**Naturaleza:** evidencia directa, no normativa.

---

## 1. Metodologia

Script deterministico. Filtrado por `PERIODOFREPORT` dominante.
Universo R4 base: NOTICE (13F-NT) + COMBINATION (13F-HR, REPORTTYPE
13F COMBINATION REPORT), excluyendo amendments (ISAMENDMENT != Y).

Agrupacion por `(CIK, PERIODOFREPORT)`. Conteo de filings base por
grupo.

---

## 2. Resultados

### 2025Q4 (PERIOD=2025-12-31)

| Metrica | Valor |
|---------|------:|
| Grupos (CIK, PERIOD) | 2,287 |
| Con 1 filing base | 2,286 |
| Con >1 filing base | **1** |
| MULTIPLE_NOTICE | 0 |
| MULTIPLE_COMBINATION | 0 |
| MIXED (NT + HR) | **1** |

### 2026Q1 (PERIOD=2026-03-31)

| Metrica | Valor |
|---------|------:|
| Grupos (CIK, PERIOD) | 2,309 |
| Con 1 filing base | 2,309 |
| Con >1 filing base | **0** |

---

## 3. Caso unico: CIK 0002016827 (2025Q4)

| Filing | Tipo | Fecha | OTHERMANAGER | INFOTABLE |
|--------|------|-------|--------------|----------:|
| `0002016827-26-000001` | 13F-NT | 2026-01-02 | Vident Advisory (0001744347) | 0 filas |
| `0002016827-26-000005` | 13F-HR COMBINATION | 2026-02-20 | Vident Advisory (0001744347) | 78 filas |

**Observaciones:**

- Mismo `CIK`, mismo `PERIODOFREPORT`, mismo `OTHERMANAGER`
  declarado.
- El NT tiene 0 holdings (delegacion total).
- El HR/Combination tiene 78 filas en INFOTABLE (holdings propios
  + delegacion).
- Ninguno es amendment (`ISAMENDMENT != Y`).
- 7 semanas entre presentaciones.

**Interpretacion:** es una evolucion documental del filer. Presento
primero un NT (delegacion total), luego un HR/Combination
(reporte propio + delegacion). La evidencia OTHERMANAGER es identica
en ambos.

---

## 4. Conclusion

**La regla del bloqueo C esta empiricamente justificada.**

- 4,596 grupos en el universo Q4+Q1.
- 1 caso con >1 filing base (0.02%).
- 0 casos de duplicidad estricta (MULTIPLE_NOTICE o MULTIPLE_COMBINATION).
- El unico caso es MIXED, que el contrato ya clasifica como N/D.

**Aplicacion de la regla:** el caso CIK 0002016827 produce `R3 = N/D`
porque el filing efectivo no es determinable univocamente sin
heuristica.

**Recomendacion al auditor:** considerar si el patron "NT -> HR/Combination
mismo periodo" merece una regla especifica (prevalecer el posterior)
o queda como N/D estricto. La evidencia actual apoya mantener N/D
fail-closed.

---

## 5. Refs

- `iae/DICTAMENES.md` #30.
- SEC Form 13F (reglas de filing base vs amendment).

---

Fin del informe Gate 0.9.
