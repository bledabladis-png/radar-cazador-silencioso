# Radar - Deuda activa

**Objeto:** registro de deuda activa del radar. Un solo fichero vivo.
Cuando una deuda se cierra, se retira de aqui. No se archiva: se borra.

**Actualizado:** 2026-09-20.

---

## D-RADAR-01 - 20 tickers LSE sin provider dedicado

**Severidad:** MEDIA.
**Detectado:** 2026-09-20, diagnostico del run scheduled fallido
35432363006 (sabado 19, 08:34 UTC).

### Situacion

El universo de `stock_prices.parquet` incluye 20 tickers de London
Stock Exchange (`.L`). No tienen provider europeo dedicado (Euronext,
Xetra, BME son los que existen). Dependen de Yahoo como unico origen.

Lista completa (universo actual):

    AAL.L  AZN.L  BA.L  BARC.L  BATS.L  BP.L  DGE.L  GLEN.L  GSK.L
    HSBA.L  LLOY.L  LSEG.L  NG.L  NWG.L  REL.L  RIO.L  RR.L
    SHEL.L  STAN.L  ULVR.L

### Comportamiento observado

- Sesion bursatil normal: Yahoo sirve los 20. Coverage 1.0.
- Fuera de horario o con lag de Yahoo: los 20 caen. Coverage
  (313-20)/313 = 0.9361.

El sabado 19/09 a las 08:34 UTC (09:34 London, LSE abierta, pero
sesion esperada = viernes 18), Yahoo no devolvio el cierre del
viernes para los 20. El guard bloqueo el commit. Main intacto.

### Impacto

- Runs diarios scheduled fallan si Yahoo no sirve los 20.
- `guard_coverage` no distingue missing core (293 tickers con
  provider) de missing LSE (20 sin provider).

### Fix propuesto

Dos partes:

  **Parte 1** - Encontrar fuente dedicada para los 20 .L.
    Candidatos a evaluar:
      - LSEG (London Stock Exchange Group) API.
      - Financial Times / marketwatch scraping (NO, fragil).
      - EOD Historical Data (plan gratuito limitado).
      - Tiingo (verificar cobertura LSE).
      - Alpha Vantage (verificar cobertura LSE).
      - Stooq (descartado por WAF segun prompt).
    Criterios: API estable, ~20 tickers, actualizacion diaria,
    coste aceptable.

  **Parte 2** - Refinar `guard_coverage.py` para distinguir
    missing core de missing known-no-provider:
      - Anadir al manifest `n_tickers_missing_close_last_core` y
        `coverage_pct_last_core` (excluye los 20 .L).
      - Guard usa `coverage_pct_last_core` si existe; retrocompatible
        con manifests viejos.
      - WARN si los 20 .L caen; FAIL solo si cae el core.

### Estado

ABIERTA. Prioridad MEDIA. No bloquea el pipeline (main queda
intacto cuando el guard bloquea), pero genera runs X recurrentes.

Reabrir ciclo cuando:
  - Se elija la fuente dedicada (Parte 1), o
  - Se aplique el refinamiento del guard (Parte 2).

---

## D-RADAR-02 - Cron `0 4 * * *` los fines de semana

**Severidad:** BAJA.
**Detectado:** 2026-09-20.

### Situacion

El cron de `daily_run.yml` es `0 4 * * *` (7 dias). Corre sabado
y domingo aunque no haya sesion bursatil. Los dos dias toman
datos del viernes.

Diseño original declarado en FU-001 (comentario del YAML): la
ventana 04:00 UTC es limpia (sin mercados USA/EU abiertos, sin
pre-market).

### Impacto

- Sabados y domingos consumen recursos sin aportar informacion
  nueva respecto al viernes.
- Si Yahoo falla ese dia, el run X aparece aunque la sesion
  esperada (viernes) ya estaba OK el viernes.
- Duplica el coste de API en fine de semana.

### Fix propuesto

  Opcion A: cambiar a `0 4 * * 1-5` (lunes a viernes).
    Ventaja: elimina 2 runs/semana sin informacion nueva.
    Riesgo: si el viernes falla, no hay re-intento el sabado.

  Opcion B: mantener 7 dias. Coherente con "backfill continuo".
    Riesgo: runs X por ruido (ver D-RADAR-01).

Decision: pendiente. Relacionada con D-RADAR-01.

### Estado

ABIERTA. Prioridad BAJA. No bloquea.

---

## D-RADAR-03 - Guard coverage: no distingue core de LSE

**Severidad:** MEDIA.
**Detectado:** 2026-09-20.

Descrito como Parte 2 del fix de D-RADAR-01. Se separa como deuda
independiente porque se puede aplicar sin esperar a la fuente
dedicada.

### Fix propuesto

  - `src/utils.py::write_artifact_with_manifest`:
    anadir al bloque `quality`:
      - `n_tickers_missing_close_last_core` (int)
      - `coverage_pct_last_core` (float)
    Calculados excluyendo los 20 tickers .L declarados en
    `config/tickers_sin_provider.txt`.
  - `scripts/guard_coverage.py`: usar `coverage_pct_last_core`
    si existe en el manifest. Retrocompatible (si no existe,
    usa `coverage_pct_last`).
  - Test nuevo: `tests/test_guard_coverage_core.py`.

### Estado

ABIERTA. Prioridad MEDIA. Puede aplicarse sin esperar D-RADAR-01.

---

## Deuda cerrada recientemente

(nada que reportar)
