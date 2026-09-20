> **Reclasificacion (dictamen #28, 2026-09-21):**
>
> Este Gate se divide en dos partes:
>
> - **Gate 0.5A - Contaminacion cross-period: PASS.**
>   Demostrado: los directorios SEC son cross-periodo. Filtrar por
>   `PERIODOFREPORT` es obligatorio.
>
> - **Gate 0.5B - Semantica completa de amendments: PENDING.**
>   No demostrado aun: comportamiento de `OTHERMANAGER` en
>   `NEW HOLDINGS` cuando el amendment declara contenido distinto
>   del base. La regla "union" del v2-bis era prematura.
>
> Ademas, el universo del probe debe incluir `13F COMBINATION REPORT`
> (13F-HR con REPORTTYPE = 13F COMBINATION REPORT), no solo NT.
> Gate 0.6 cubrira esta extension.
>
> El caso CIK 0002056909 (RESTATEMENT que sustituye OTHERMANAGER)
> permanece valido como evidencia de que RESTATEMENT sustituye.
> Lo pendiente es NEW HOLDINGS.

---

# IAE - P66 Gate 0.5 Amendment probe NT

**Objeto:** medicion del comportamiento de amendments en NT filings,
para formalizar contractualmente R3 (filing efectivo).

**Origen:** dictamen P66 v2, Bloqueo B.

**Fecha:** 2026-09-20 (ejecucion 2026-09-21).
**Autor:** Ingeniero Supervisor.
**Naturaleza:** evidencia directa, no normativa.

---

## 1. Hallazgo principal

**Los directorios del Data Set SEC estan organizados por FECHA DE
PRESENTACION, no por periodo objetivo.**

Cada directorio contiene filings con multiples `PERIODOFREPORT`:

| Directorio | Periodos distintos | Periodo dominante |
|-----------|-------------------:|------------------:|
| 2025Q4    | 48                 | 2025-12-31 (10,676 filings) |
| 2026Q1    | 77                 | 2026-03-31 (10,776 filings) |

Los NT de periodos anteriores presentes en cada directorio son
late amendments o filings procesados tardes por el SEC.

**Regla arquitectonica:** filtrar SIEMPRE por `PERIODOFREPORT`,
nunca por directorio.

---

## 2. Comportamiento de amendments en NT (periodo correcto)

Filtrando por periodo correcto:

### 2025Q4 (PERIOD=2025-12-31)

| Categoria | Filings |
|-----------|--------:|
| NT base | 1,900 |
| NT/A RESTATEMENT | 37 |
| NT/A NEW HOLDINGS | 1 |

Casos inspeccionados: 38 pares base+amend.
- 37/38 con `OTHERMANAGER` **identico**.
- 1/38 con `OTHERMANAGER` **distinto** (ver seccion 3).

### 2026Q1 (PERIOD=2026-03-31)

| Categoria | Filings |
|-----------|--------:|
| NT base | 1,907 |
| NT/A RESTATEMENT | 0 |
| NT/A NEW HOLDINGS | 1 |

Un unico amend, tipo NEW HOLDINGS, con base existente.

---

## 3. Caso RESTATEMENT que cambia OTHERMANAGER

CIK `0002056909`:

| Filing | Tipo | OTHERMANAGER |
|--------|------|--------------|
| `0001376474-26-000019` (base) | 13F-NT | CIK=`<NA>` FormNum=`028-04685` (Prospector Partners) |
| `0001376474-26-000046` (amend) | 13F-NT/A RESTATEMENT AMENDNO=1 | CIK=`0001570284` FormNum=`028-16376` (Gator Capital Management) |

**El RESTATEMENT sustituye el contenido de OTHERMANAGER.**

Consecuencia: R4 debe evaluarse sobre el **filing efectivo**, no
sobre el filing base. Si un RESTATEMENT cambia el manager declarado,
R4 se recalcula con el nuevo contenido.

---

## 4. `AMENDMENTNO` — cardinalidad observada

| Periodo | AMENDMENTNO=1 | AMENDMENTNO=2 |
|---------|--------------:|--------------:|
| 2025-12-31 (Q4) | 38 | 0 |
| 2026-03-31 (Q1) | 1 | 0 |
| 2025-06-30 (late) | - | 4 (todos NT/A con AMENDNO=2) |

Los casos AMENDMENTNO=2 corresponden a late amendments de periodos
muy antiguos, presentados en Q1 2026. No afectan al analisis
Q4 2025 -> Q1 2026.

---

## 5. R3 formalizada

    R3(B, period) := existe filing efectivo de B
                     para ese PERIODOFREPORT

    "Filing efectivo":
      1. Filtrar filings de CIK=B con PERIODOFREPORT=period
         y SUBMISSIONTYPE in {13F-NT, 13F-NT/A}.

      2. Filtrar el directorio por PERIODOFREPORT.
         Prohibido filtrar por directorio fisico.

      3. Ordenar filings por AMENDMENTNO ascendente
         (base AMENDMENTNO=null tratado como orden 0).

      4. Aplicar semantica SEC:
         - Base:          punto de partida.
         - RESTATEMENT:   sustituye el snapshot (incluye OTHERMANAGER).
         - NEW HOLDINGS:  union con el snapshot actual.
         - Resultado:     snapshot efectivo.

      5. R4 se evalua sobre el snapshot efectivo, NO sobre el base.

    Salvaguardas:
      - Si no existe filing en el periodo: R3 = False.
      - Trazabilidad: registrar cadena de amendments aplicados.

---

## 6. Conclusiones

1. **Filtrar por PERIODOFREPORT es obligatorio.** Los directorios
   contienen filings cross-periodo.

2. **RESTATEMENT puede cambiar OTHERMANAGER.** R4 debe evaluarse
   sobre el filing efectivo.

3. **NEW HOLDINGS complementa sin invalidar el base.** El unico
   caso Q1 2026 tiene base existente.

4. **`AMENDMENTNO` es monotono en el periodo activo.** AMENDMENTNO=2
   solo aparece en late amendments a periodos muy antiguos.

5. **Gate 0.5 PASS.** El bloqueo B del dictamen P66 v2 queda
   resuelto con semantica SEC estandar.

---

## 7. Refs

- `iae/DICTAMENES.md` #27.
- `iae/P66_L3_REFORMULACION_PROPUESTA.md` v2 seccion 5 (amendments).
- Gate 0.4 (mapping FormNum -> CIK).
- SEC Form 13F y EDGAR Filer Manual.

---

Fin del informe Gate 0.5.
