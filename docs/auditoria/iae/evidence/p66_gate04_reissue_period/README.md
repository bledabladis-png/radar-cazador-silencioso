# IAE - P66 Gate 0.4-reissue (filtro PERIODOFREPORT)

**Objeto:** re-ejecucion del mapping `FormNum -> CIK` con filtro
explicito por `PERIODOFREPORT`. Bloqueo #3 del dictamen #28.

**Origen:** dictamen #28, bloqueo #3. Gate 0.4 original construia
el mapping sin filtrar por periodo, mezclando filings cross-periodo
(Gate 0.5A).

**Fecha:** 2026-09-21.
**Autor:** Ingeniero Supervisor.
**Naturaleza:** evidencia directa, no normativa.

---

## 1. Metodologia

Script deterministico. Re-ejecucion de Gate 0.4 con dos correcciones:

1. **Filtro por `PERIODOFREPORT`** antes del JOIN COVERPAGE + SUBMISSION.
2. **Normalizacion estricta** de FormNum: solo `28-XXXXX` o `028-XXXXX`
   con 5 digitos exactos. Prohibido truncar (bloqueo #5 aplicado).
3. **Nomenclatura corregida**: MATCH -> IDENTITY_RESOLVED.

---

## 2. Resultados comparados

### 2.1. Mapping FormNum -> CIK

| Metrica | Q4 original | Q4 reissue | Q1 original | Q1 reissue |
|---------|------------:|-----------:|------------:|-----------:|
| FormNum unicos en OTHERMANAGER (NT) | 914 | 906 | 908 | 901 |
| Resueltos | 894 (97.81%) | 888 (98.01%) | 892 (98.24%) | 887 (98.45%) |
| FormNum con >1 CIK | 0 | 0 | 0 | 0 |

**Delta a nivel FormNum:** −8 (Q4), −7 (Q1). Volumen despreciable.

### 2.2. Filas OTHERMANAGER(NT) por ruta de resolucion

| Categoria | Q4 original | Q4 reissue | Q1 original | Q1 reissue |
|-----------|------------:|-----------:|------------:|-----------:|
| Total filas | 2,804 | 2,734 | 2,827 | 2,683 |
| A. IDENTITY_RESOLVED por CIK | 1,955 (69.72%) | 1,889 (69.09%) | 1,948 (68.91%) | 1,820 (67.83%) |
| B. IDENTITY_RESOLVED por FormNum | 737 (26.28%) | 706 (25.82%) | 771 (27.27%) | 738 (27.51%) |
| C. N/D FormNum no resuelto | 15 (0.53%) | 10 (0.37%) | 5 (0.18%) | 2 (0.07%) |
| D. N/D ni CIK ni FormNum | 97 (3.46%) | 129 (4.72%) | 103 (3.64%) | 123 (4.58%) |
| **Cobertura de resolucion de identidad** | **96.01%** | **94.92%** | **96.18%** | **95.34%** |
| **Delta** | | **−1.09 pp** | | **−0.84 pp** |

---

## 3. Mecanismo del delta

El Gate 0.4 original construia el mapping desde TODOS los COVERPAGE
del directorio, incluyendo los de periodos anteriores. Eso permitia
resolver FormNums que solo aparecian en filings cross-periodo.

El reissue construye el mapping solo desde COVERPAGE del periodo
correcto. Algunas referencias cross-period que antes resolvian
ahora no resuelven.

El delta de −1 pp es la cuantificacion de la contaminacion.

---

## 4. Validacion del dictamen #28

El auditor #28 dijo:

> "Puede que el resultado numerico siga siendo practicamente
> identico. Pero el contrato necesita que la regla sea correcta,
> no que accidentalmente funcione con el dataset actual."

El reissue confirma:
- La regla era incorrecta (contaminacion cross-periodo).
- El efecto numerico es pequeno pero real (~1 pp).
- La correccion no invalida la arquitectura: la cobertura sigue
  siendo >94% con la regla correcta.

---

## 5. Bloqueos resueltos por este Gate

- **#3 Gate 0.4-reissue con filtro PERIODOFREPORT: RESUELTO.**
- **#5 Normalizacion sin truncar: APLICADO** (el caso 028-2813114
  queda automaticamente en N/D).
- **#6 Nomenclatura IDENTITY_RESOLVED: APLICADO.**

---

## 6. Refs

- `iae/DICTAMENES.md` #28, bloqueo #3.
- Gate 0.4 original (`p66_gate04_mapping/`).
- Gate 0.5A (cross-period contamination).
- Gate 0.6 (universo R4 con Combination).

---

Fin del informe Gate 0.4-reissue.
