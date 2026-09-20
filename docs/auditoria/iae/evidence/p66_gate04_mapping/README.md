> **Nota de nomenclatura (correccion del auditor #28, 2026-09-21):**
>
> En esta evidencia, "MATCH" debe leerse como "IDENTITY_RESOLVED".
> Los porcentajes 96.01% / 96.18% son **cobertura de resolucion de
> identidad de las filas OTHERMANAGER**, no "cobertura L3" ni
> "match rate". L3 requiere ademas R1 + R2 + R3 + R5 y la comparacion
> `CIK_resuelto == CIK_A`.
>
> El analisis de este Gate es previo a la correccion #3 del dictamen
> #28 (falta filtro explicito por PERIODOFREPORT). Ver Gate 0.4-reissue
> en el siguiente ciclo.

---

# IAE - P66 Gate 0.4 Mapping Form13FFileNumber -> CIK

**Objeto:** demostracion de que la tabla canonica FormNum -> CIK es
construible vía COVERPAGE + SUBMISSION del Data Set SEC.

**Origen:** dictamen P66 v2, Bloqueo A. Exigencia de demostrar
cobertura del mapping, no solo consistencia interna.

**Fecha:** 2026-09-20.
**Autor:** Ingeniero Supervisor.
**Naturaleza:** evidencia directa, no normativa.

---

## 1. Metodologia

Script deterministico, sin `datetime.now()`.

### 1.1. Construccion del mapping

    COVERPAGE  (ACCESSION_NUMBER, FORM13FFILENUMBER, ...)
        JOIN
    SUBMISSION (ACCESSION_NUMBER, CIK, PERIODOFREPORT, SUBMISSIONTYPE)
        ON ACCESSION_NUMBER

    Filtro: FORM13FFILENUMBER no-null AND CIK no-null.
    Agrupacion: FORM13FFILENUMBER (normalizado) -> CIK.

### 1.2. Normalizacion aplicada

    normalize("28-XXXXX")   -> "028-XXXXX"
    normalize("028-XXXXX")  -> "028-XXXXX"

El prefijo se normaliza a 3 digitos (028) y el sufijo a 5 digitos.
Regla aplicada en Gate 0.4 y 0.4b.

### 1.3. Universo evaluado

    FormNum unicos en OTHERMANAGER restringido a NT filings.
    Q4 2025: 914 FormNum unicos.
    Q1 2026: 908 FormNum unicos.

### 1.4. Evaluacion

    1. Lookup de cada FormNum en el mapping COVERPAGE.
    2. Cardinalidad FormNum -> CIK.
    3. Cardinalidad CIK -> FormNum (desde COVERPAGE).
    4. Cuantificacion de filas OTHERMANAGER por ruta de resolucion.

---

## 2. Resultados

### 2.1. Cobertura del mapping

| Trimestre | FormNum unicos | Resueltos | No resueltos | Cobertura |
|-----------|---------------:|----------:|-------------:|----------:|
| 2025Q4    | 914            | 894       | 20           | 97.81%    |
| 2026Q1    | 908            | 892       | 16           | 98.24%    |

### 2.2. Cardinalidad

| Trimestre | FormNum con 1 CIK | FormNum con >1 CIK | CIK con 1 FormNum | CIK con >1 FormNum |
|-----------|------------------:|-------------------:|------------------:|-------------------:|
| 2025Q4    | 894               | 0                  | 10,535            | 0                  |
| 2026Q1    | 892               | 0                  | 10,672            | 0                  |

**Cardinalidad 1:1 perfecta en ambas direcciones.** Sin ambiguedad.

### 2.3. Impacto en filas (Gate 0.4b)

| Categoria | Q4 filas | Q4 % | Q1 filas | Q1 % |
|-----------|---------:|-----:|---------:|-----:|
| Total filas OTHERMANAGER en NT | 2,804 | — | 2,827 | — |
| A. MATCH por CIK directo | 1,955 | 69.72% | 1,948 | 68.91% |
| B. MATCH por FormNum fallback | 737 | 26.28% | 771 | 27.27% |
| C. N/D FormNum no resuelto | 15 | 0.53% | 5 | 0.18% |
| D. N/D ni CIK ni FormNum | 97 | 3.46% | 103 | 3.64% |
| **Cobertura efectiva MATCH** | **96.01%** | | **96.18%** | |
| **N/D total** | **3.99%** | | **3.82%** | |

**El fallback FormNum amplia la cobertura MATCH en 26.28-27.27 puntos
porcentuales.** Sin tabla canonica, casi un tercio de las filas
quedaria en N/D artificialmente.

### 2.4. N/D por FormNum no resuelto — detalle

Q4 2025 (15 filas, 5 FormNum unicos):
- 028-03140 (8 filas)
- 028-2813114 (4 filas) <- ANOMALIA, ver 2.5
- 028-10955 (1 fila)
- 028-64460 (1 fila)
- 028-10245 (1 fila)

Q1 2026 (5 filas, 4 FormNum unicos):
- 028-2813114 (2 filas) <- ANOMALIA, ver 2.5
- 028-64460 (1 fila)
- 028-14014 (1 fila)
- 028-04685 (1 fila)

Volumen despreciable (0.53% / 0.18% de filas). Todos caen a N/D
fail-closed segun dictamen.

### 2.5. Anomalia detectada: 028-2813114

Formato observado: `028-2813114` (7 digitos tras el prefijo).
Formato estandar esperado: `028-XXXXX` (5 digitos).

Hipotesis:
- Error en la fuente SEC al publicar el FormNum.
- Concatenacion accidental de dos valores.
- Formato no estandar de un filer concreto.

NO bloquea Gate 0.4 (volumen despreciable). Se documenta como
hallazgo colateral para investigacion futura si reaparece.

---

## 3. Conclusiones

1. **Gate 0.4 PASS.** La tabla canonica FormNum -> CIK es construible
   vía COVERPAGE + SUBMISSION con cardinalidad 1:1 verificada.

2. **Cobertura efectiva MATCH: 96.01% Q4 / 96.18% Q1.**

3. **N/D total: 3.99% Q4 / 3.82% Q1.** Fail-closed segun dictamen.

4. **Normalizacion `28-` -> `028-` validada** por el Gate 0.4: los
   valores de OTHERMANAGER con prefijo `28-` resuelven a FormNum con
   prefijo `028-` en COVERPAGE sin ambiguedad.

5. **Cero CONFLICT** en cardinalidad bidireccional.

6. **Bloqueo A del dictamen P66 v2: RESUELTO.**

---

## 4. Refs

- `iae/DICTAMENES.md` #27.
- `iae/P66_L3_REFORMULACION_PROPUESTA.md` v2 seccion 9.
- `granularidad_output.txt` (Gate 0.3).
- `gate04_output.txt` (Gate 0.4).
- `gate04b_output.txt` (Gate 0.4b).

---

Fin del informe Gate 0.4.
