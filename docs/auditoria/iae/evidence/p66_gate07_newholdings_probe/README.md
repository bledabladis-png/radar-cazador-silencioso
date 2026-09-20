# IAE - P66 Gate 0.7 NEW HOLDINGS probe

**Objeto:** medir comportamiento de `OTHERMANAGER` cuando un
amendment es de tipo `NEW HOLDINGS`. Bloqueo #2 del dictamen #28.

**Origen:** dictamen #28, bloqueo #2 (regla union era prematura).

**Fecha:** 2026-09-21.
**Autor:** Ingeniero Supervisor.
**Naturaleza:** evidencia directa, no normativa.

---

## 1. Metodologia

Script deterministico. Filtrado por `PERIODOFREPORT` = periodo
dominante del directorio (leccion Gate 0.5A).

Universo evaluado: R4 (NOTICE + COMBINATION) segun REPORTTYPE
(leccion Gate 0.6).

Para cada `NEW HOLDINGS` en R4: comparar `OTHERMANAGER` del filing
base vs el amendment.

---

## 2. Resultados

### 2025Q4 (PERIOD=2025-12-31, universo R4=2,334)

| Categoria | Casos |
|-----------|------:|
| NEW HOLDINGS en R4 | 2 |
| Identicos (sin cambio) | 2 |
| Union (base < amend) | 0 |
| Sustitucion (amend < base) | 0 |
| Otros (diferencia mixta) | 0 |
| Sin base en el periodo | 0 |

Desglose por tipo:
- 13F-HR/A COMBINATION REPORT: 1
- 13F-NT/A NOTICE: 1

### 2026Q1 (PERIOD=2026-03-31, universo R4=2,319)

| Categoria | Casos |
|-----------|------:|
| NEW HOLDINGS en R4 | 1 |
| Identicos (sin cambio) | 1 |
| Union | 0 |
| Sustitucion | 0 |
| Otros | 0 |
| Sin base | 0 |

Desglose: 13F-NT/A NOTICE.

---

## 3. Conclusion

**Empiricamente:** 3/3 NEW HOLDINGS conservan `OTHERMANAGER` identico
al base. Cero casos de cambio.

**Estadisticamente:** muestra insuficiente. 3 casos sobre 4,653
filings en el universo R4 no permiten afirmar que la semantica
"union" nunca ocurra.

**Regla contractual propuesta (fail-closed):**

    Para determinar OTHERMANAGER efectivo del filing B:

    - Base: punto de partida.
    - RESTATEMENT: sustituye el snapshot (evidencia Gate 0.5:
      caso CIK 0002056909).
    - NEW HOLDINGS: observado identico en 3/3 casos (Gate 0.7).
      Regla provisional: tratar como identidad. Si en el futuro
      se observa un NEW HOLDINGS con OTHERMANAGER distinto:
      estado = N/D o CONFLICT segun corresponda. NO asumir union
      ni sustitucion sin evidencia.

Esta regla:
- Es fail-closed.
- No fabrica identidad (no union automatica).
- Es coherente con SEC: NEW HOLDINGS suplementa holdings, no
  necesariamente reescribe la lista de reporting managers.
- Se reabre si aparecen casos que la refuten.

---

## 4. Recomendacion de extension (opcional)

La muestra de 3 casos es insuficiente para afirmar la regla
con alta confianza estadistica. Alternativas:

(a) Aceptar la regla fail-closed con la evidencia actual.
(b) Extender el probe a 4-6 trimestres historicos (Q1 2024 - Q1 2026)
    antes del GO contractual.

El auditor decide.

---

## 5. Refs

- `iae/DICTAMENES.md` #28, bloqueo #2.
- Gate 0.5 (RESTATEMENT sustituye: caso 0002056909).
- Gate 0.6 (universo R4).
- SEC Form 13F (semantica NEW HOLDINGS).

---

Fin del informe Gate 0.7.
