# 80 - WYCKOFF-EVENTS-ARTIFACT v1

**Pre-registro. 2026-10-31, sesion 22.**
**Contrato firmado antes de tocar codigo.**

## 1. Motivacion

Doc 62 (sesion 3, 2026-10-24) dejo dos deudas documentadas:

> "Calibrar PSY/PS con pre-registro formal (actualmente 0/50 disparos)."
> "Renderizar evidencia pasiva (msow, msos, lpsy, lps, st_dist, ...)."

El "0/50" era falso: probe 2026-10-31 encontro PSY 35 disparos y PS 8
disparos en 20 holdings (2022-2026). El detector funciona; simplemente
no se invoca en produccion ni se expone al CSV.

Estado actual (auditoria read-only 2026-10-31):

| Detector | Invocado en produccion | En dist/acc_evidence | En CSV |
|---|---|---|---|
| mSOW | SI (en _enrich_dist_evidence) | SI | NO |
| mSOS | SI (en _enrich_acc_evidence) | SI | NO |
| LPSY | SI (condicional a SOW) | SI | NO |
| LPS | SI (condicional a SOS) | SI | NO |
| ST_dist | SI (condicional a BC) | SI | NO |
| ST_acc | SI (condicional a SC) | SI | NO |
| PSY | NO (dead code) | NO | NO |
| PS | NO (dead code) | NO | NO |

## 2. Alcance

Anadir 8 detectores de evidencia pasiva a los 6 CSVs Wyckoff
(usa_sector_acc, usa_sector_dist, usa_index_acc, usa_index_dist,
eu_index_acc, eu_index_dist).

**Fuera de alcance:**
- Detectar eventos BC, AR, SC, SOW, SOS, UT, Spring: son parte de la
  cascada de clasificacion de fase, ya cubiertos por phase/event_phase.
- Escaneo historico de eventos (buscar en N ultimas sesiones). Se
  mantiene el diseno actual: solo ultima barra. El consumidor puede
  escanear por ventana si lo necesita (documentado en doc 62).
- Renderizado en el reporte .md. Solo CSV.
- Calibracion de umbrales PSY/PS. Se usan los actuales.

## 3. Schema v2

**14 columnas actuales + 16 nuevas = 30 columnas.**

Columnas nuevas (8 eventos x 2 campos):

    psy_status, psy_date,
    ps_status, ps_date,
    msow_status, msow_date,
    msos_status, msos_date,
    lpsy_status, lpsy_date,
    lps_status, lps_date,
    st_dist_status, st_dist_date,
    st_acc_status, st_acc_date,

**Valores de `*_status`:**
- `CONFIRMED`: evento detectado en la ultima barra.
- `NOT_OBSERVED`: detector ejecutado, condiciones no cumplidas.
- `NOT_EVALUABLE`: faltan datos (historico insuficiente, features NaN).
- `""` (vacio): detector no aplica (p.ej. ST_dist sin BC confirmado).

**Valores de `*_date`:**
- `YYYY-MM-DD` si `status=CONFIRMED`.
- `""` en el resto de casos.

**Nota:** los campos `*_date` reflejan la fecha del ultimo evento
detectado (la ultima barra del df, no escaneo historico). Coherente
con la semantica de `phase` y `event_phase` actuales.

**Schema version:** 1 -> 2 para los 6 CSVs.

## 4. Cambios de codigo

### 4.1. Wirear PSY y PS (2 lineas nuevas)

`indicators/wyckoff_distribution.py::_enrich_dist_evidence`: anadir

    try:
        psy = detect_psy(sub, ticker)
        evidence["psy"] = {"status": psy.status,
                          "event_date": str(psy.event_date) if psy.event_date else None}
    except Exception as exc:
        evidence["psy"] = {"status": "ERROR", "error": str(exc)}

`indicators/wyckoff_accumulation.py::_enrich_acc_evidence`: idem con
`detect_ps`.

Coste medido (bench 2026-10-31): +35ms por ticker para ambos. Sobre
601 tickers: +21 segundos al run. Run actual ~15 min. Impacto < 0.3%.

### 4.2. Exponer los 8 eventos al CSV

`indicators/wyckoff_holdings_report.py`:

1. `OUTPUT_COLUMNS`: anadir las 16 columnas al final.
2. `_evaluate_holding`: leer `fp.dist_evidence` y `fp.acc_evidence`,
   extraer `{status, event_date}` de cada uno. Rellenar las 16
   columnas con defaults `""` si el evento no esta.

### 4.3. Schema bump en output_registry

`schema_version: 1 -> 2` para los 6 outputs `wyckoff_usa_*` y
`wyckoff_eu_*`.

Los CSVs se regeneran en el proximo run. Los `.schema` en
`outputs/history/` se actualizan a `2` por el writer.

### 4.4. Validation gate

El check 11 (schema versioning) ya valida `spec.schema_version` vs
`.schema`. Sin cambios. Verifica que los 6 CSVs se escriban con
schema_version=2.

## 5. Tests

`tests/test_wyckoff_holdings_report.py` (+10-12):
- Cada evento con CONFIRMED produce `*_status=CONFIRMED` y `*_date` no vacio.
- Cada evento con NOT_OBSERVED produce `*_status=NOT_OBSERVED` y `*_date=""`.
- Cada evento con NOT_EVALUABLE produce `*_status=NOT_EVALUABLE`.
- Detector no aplicable produce `*_status=""` y `*_date=""`.
- `OUTPUT_COLUMNS` tiene exactamente 30 columnas.

`tests/test_wyckoff_events_contract.py` (nuevo, +6-8):
- Contrato: los 8 detectores estan wireados en dist/acc_evidence.
- Contrato: `_evaluate_holding` lee los 8 campos de evidence.
- Contrato: los 6 CSVs tienen schema_version=2.
- Heuristico: todo detector en `_DETECTORS_EVIDENCE` debe estar en
  `OUTPUT_COLUMNS` (falla si alguien anade un detector y olvida la
  columna).

## 6. E2E obligatorio (METODO seccion 4)

Cambio en `indicators/` -> E2E obligatorio.

Snapshot pre de `outputs/history/wyckoff_usa_*` y `outputs/history/wyckoff_eu_*`.
`py run.py` real (~15 min).
Snapshot post.
Comparacion:
1. Los 6 CSVs ahora tienen 30 columnas.
2. `.schema` de los 6 es `2`.
3. `phase`, `event_phase`, `phase_since`, `weight`, `close` identicos
   a pre en filas comunes (no regresion).
4. `psy_status`, `ps_status` presentes con valores validos
   (CONFIRMED/NOT_OBSERVED/NOT_EVALUABLE/"").
5. Los 6 eventos que ya se calculaban ahora aparecen en el CSV con
   sus valores correspondientes.
6. Exit=0. Gate 11/11.
7. Resto de outputs sin cambios.

Rollback de `outputs/history` y `outputs/state` al cerrar.

## 7. Rollback si E2E falla

- `git checkout -- indicators/ src/pipeline/ tests/`.
- `git checkout -- outputs/history outputs/state`.
- Anotar en este documento que fase fallo y por que.
- Sesion 23 empieza con el diagnostico hecho.

## 8. Documentacion

- Doc 62: anotar que la deuda "renderizar evidencia pasiva" esta
  cerrada.
- ARRANQUE seccion 4.2: rectificar el "0/50" con los resultados del
  probe 2026-10-31.
- HANDOFF seccion 1: mencionar el schema v2 y los 8 eventos.
- REGISTRO: item `WYCKOFF-EVENTS-ARTIFACT` FIXED con el commit.
- Bitacora: entrada en el cierre.

## 9. Estimacion

~4.5-5h. Pre-registro (30 min) + wireado PSY/PS (30 min) + schema bump
en report (1h) + tests (1.5h) + E2E (30 min) + docs (30 min).

## 10. Decisiones

- **B sobre B1 (columnas vs columna unica con eventos)**: pandas
  filtra directamente, sin parsear strings. Mas limpio para analisis.
- **8 detectores todos, no solo PSY/PS**: los otros 6 ya se calculan
  en cada run. Exponerlos es leer del dict y escribir. Coste marginal
  +1.5h; valor duplicado.
- **Sin escaneo historico**: se mantiene diseno actual (ultima barra).
  Escanear N sesiones es feature aparte, pre-registro propio.
- **Sin render en .md**: solo CSV. El reporte se queda como esta.
