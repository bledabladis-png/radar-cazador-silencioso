# IAE - P66 Gate 0.10 FormNum representation

**Objeto:** medir longitudes reales del sufijo del
`FORM13FFILENUMBER` en el universo R4 para fundamentar la regla de
canonicalizacion. Bloqueo #1 del dictamen #34.

**Origen:** dictamen #34. Auditor senala que la regla "5 digitos
exactos" rechaza identidades SEC validas (filings reales con
`028-6538`, `028-2013`, `028-398`, `028-5788`).

**Fecha:** 2026-09-21.
**Autor:** Ingeniero Supervisor.
**Naturaleza:** evidencia directa, no normativa.

---

## 1. Metodologia

Medicion de `COVERPAGE.FORM13FFILENUMBER` y
`OTHERMANAGER.FORM13FFILENUMBER` filtrando por `PERIODOFREPORT`
dominante. Conteo de longitudes de sufijo (digitos tras el guion).

---

## 2. Resultados

### 2.1. COVERPAGE (estructurado por SEC)

| Trimestre | non-null | Prefijo | Sufijo |
|-----------|---------:|--------:|-------:|
| 2025Q4    | 10,676   | 100% `28` | 100% 5 digitos |
| 2026Q1    | 10,776   | 100% `28` | 100% 5 digitos |

**La SEC normaliza a 5 digitos en la tabla estructurada.**

### 2.2. OTHERMANAGER (declarado por filer)

**2025Q4 (4,077 filas con FormNum):**

| Longitud sufijo | Filas | % |
|----------------:|------:|--:|
| 3 digitos       | 3     | 0.07% |
| 4 digitos       | 25    | 0.61% |
| 5 digitos       | 4,044 | 99.19% |
| 6 digitos       | 1     | 0.02% |
| 7 digitos       | 4     | 0.10% |

**2026Q1 (3,954 filas con FormNum):**

| Longitud sufijo | Filas | % |
|----------------:|------:|--:|
| 3 digitos       | 4     | 0.10% |
| 4 digitos       | 29    | 0.73% |
| 5 digitos       | 3,918 | 99.09% |
| 6 digitos       | 1     | 0.03% |
| 7 digitos       | 2     | 0.05% |

### 2.3. Ejemplos por longitud

    L=3: 028-694
    L=4: 28-4545
    L=5: 028-22660
    L=6: 028-064460
    L=7: 028-2813114 (anomalia)

### 2.4. Cruce literal OTHERMANAGER vs COVERPAGE

    Q4 2025:
        FormNum unicos en OTHERMANAGER(R4): 1,497
        FormNum unicos en COVERPAGE:        10,524
        Interseccion literal:               1,360
        Solo en OTHERMANAGER:                 137
    Q1 2026:
        FormNum unicos en OTHERMANAGER(R4): 1,495
        FormNum unicos en COVERPAGE:        10,648
        Interseccion literal:               1,351
        Solo en OTHERMANAGER:                 144

Los "solo en OTHERMANAGER" son managers sin filing propio en el
periodo. N/D fail-closed los cubre.

---

## 3. Conclusion

1. **La regla "5 digitos exactos" es incorrecta** como condicion de
   validez. Rechaza representaciones SEC validas (`028-694`,
   `28-4545`).

2. **La regla correcta es canonicalizacion con padding flexible:**
   prefijo a 3 digitos, sufijo a 5 digitos tras strip de leading
   zeros.

3. **Sufijos con >5 digitos tras strip son UNRESOLVED** (ej.
   `028-2813114`). No truncar, no heuristica.

4. **COVERPAGE es la fuente de verdad para el mapping FormNum ->
   CIK.** OTHERMANAGER puede tener variantes; COVERPAGE normaliza
   a 5 digitos.

5. **Regla definitiva incorporada en la propuesta v4 seccion 4.2.**

---

## 4. Refs

- `iae/DICTAMENES.md` #34.
- Propuesta v4 seccion 4 (Normalizacion FormNum).
- SEC Form 13F (documentacion oficial).

---

Fin del informe Gate 0.10.
