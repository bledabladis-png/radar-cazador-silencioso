# IAE - P66 Gate 0.3 Granularidad de identidad en OTHERMANAGER

**Objeto:** medicion de granularidad de identidad (CIK / FormNum) en
la tabla OTHERMANAGER del Data Set SEC, sobre NT filings Q4 2025 y
Q1 2026.

**Origen:** dictamen externo #27 (P66 GO CONDICIONADO), correccion
obligatoria #8.

**Fecha:** 2026-09-20.
**Autor:** Ingeniero Supervisor.
**Naturaleza:** evidencia directa, no normativa.

---

## 1. Metodologia

Script deterministico, sin `datetime.now()`. Lee los parquets del
Data Set SEC en `D:\13f_probe\processed\`.

Filtro: `SUBMISSION.SUBMISSIONTYPE` contiene `NT`.
Tabla evaluada: `OTHERMANAGER` (no `OTHERMANAGER2`).
Cruce por `ACCESSION_NUMBER`.

Mediciones:
- Total NT filings.
- Filas OTHERMANAGER en NT.
- Accesiones con al menos 1 fila.
- Granularidad: CIK poblado / FormNum poblado / ambos / solo uno /
  ninguno.
- Consistencia CIK <-> FormNum (cardinalidad 1:1 o N:1).
- Estructura de FormNum (prefijo 028- vs otros).

---

## 2. Resultados

### 2.1. Cobertura por trimestre

| Trimestre | NT totales | Filas OTHERMANAGER | Acc. con >=1 fila | Acc. sin filas |
|-----------|-----------:|-------------------:|------------------:|---------------:|
| 2025Q4    | 2,008      | 2,804              | 2,008             | 0              |
| 2026Q1    | 2,045      | 2,827              | 2,045             | 0              |

**Cobertura 100% en ambos trimestres.**

### 2.2. Granularidad de identidad

| Categoria | Q4 filas | Q4 % | Q1 filas | Q1 % |
|-----------|---------:|-----:|---------:|-----:|
| con CIK poblado      | 1,955 | 69.72% | 1,948 | 68.91% |
| con FormNum poblado  | 2,567 | 91.55% | 2,596 | 91.83% |
| con ambos poblados   | 1,815 | 64.73% | 1,820 | 64.38% |
| solo CIK             |   140 |  4.99% |   128 |  4.53% |
| solo FormNum         |   752 | 26.82% |   776 | 27.45% |
| ninguno              |    97 |  3.46% |   103 |  3.64% |

### 2.3. Consistencia CIK <-> FormNum

Sobre filas con ambos poblados:

| Trimestre | CIK unicos | CIK con 1 FormNum | CIK con >1 | FormNum unicos | FormNum con 1 CIK | FormNum con >1 |
|-----------|-----------:|------------------:|-----------:|---------------:|------------------:|---------------:|
| 2025Q4    | 716        | 716               | 0          | 716            | 716               | 0              |
| 2026Q1    | 696        | 696               | 0          | 696            | 696               | 0              |

**Cardinalidad 1:1 perfecta. Cero casos N:1 en ninguna direccion.**

### 2.4. Estructura de FormNum

| Trimestre | FormNum poblado | Prefijo `028-` | Otros |
|-----------|----------------:|---------------:|------:|
| 2025Q4    | 2,567           | 2,398 (93.42%) | 169 (6.58%) |
| 2026Q1    | 2,596           | 2,423 (93.34%) | 173 (6.66%) |

**Otros** = prefijo `28-` sin cero a la izquierda. Todas las filas
afectadas tienen CIK=`<NA>`.

Casos representativos: grupo Bain Capital Life Sciences y entidades
afines. Mismo FormNum repetido en multiples filas (patron esperado:
varias filiales reportan por el mismo manager padre).

---

## 3. Conclusiones

1. **La tabla canónica CIK <-> FormNum es 1:1**, sin ambiguedad.
   Construible del propio Data Set.

2. **El 26.82% de filas "solo FormNum" es 100% resoluble** via tabla
   canonica, tras normalizacion trivial `28- -> 028-`.

3. **Cobertura efectiva MATCH sin tabla canonica:** 69.72% (solo CIK).
   **Con tabla canonica:** ~96.5% (CIK + FormNum normalizado).

4. **El 3.46% sin CIK ni FormNum** queda en N/D fail-closed, segun
   dictamen #27.

5. **Cero casos CONFLICT** (CIK presente y FormNum presente apuntando
   a entidades distintas). El escenario CONFLICT definido por el
   auditor no se materializa en los datos.

---

## 4. Implicacion para el contrato 14.3

La tabla canonica `Form13FFileNumber -> CIK` debe existir para que
el fallback funcione. Sin ella:

- 26.82% de filas quedaria en N/D artificialmente.
- NIPC seguiria INSUFFICIENT por sesgo de universo, no por
  ausencia real de evidencia.

La tabla puede construirse:
- Del propio Data Set SEC (interseccion de CIK + FormNum poblados).
- Versionada por trimestre.
- Persistida como artefacto de IAE.

---

## 5. Refs

- `iae/DICTAMENES.md` #27 (correccion #8).
- `iae/P66_L3_REFORMULACION_PROPUESTA.md` v2 seccion 9.
- `granularidad_output.txt` (salida cruda).

---

Fin del informe Gate 0.3.
