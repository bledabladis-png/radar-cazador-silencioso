# Auditoría FASE 5 — Providers (2026-09-26)

**Alcance:** capa de providers externos (`data/providers/`).
**Método:** read-only. Sin modificar código.
**Providers auditados:** 13 de 27 (Yahoo, OilPriceAPI, Xetra, BME, Euronext, BackupProvider, Blackrock DAX/ISF/IWM estructural, SSGA, Amundi, Invesco estructural, Router).
**Providers no auditados en profundidad:** macro (`fred`, `cboe`, `cboe_index`, `polygon`), flows trimestrales (7), algunos fund data (IWM completo).

## Resumen

| Severidad | Count |
|---|---:|
| ALTA | 1 |
| MEDIA | 23 |
| BAJA | 26 |
| **Total** | **50** |

**Hallazgo ALTA único:**

- **A5-13** — `YahooProvider.get_prices` devuelve el parquet completo en fallo. Bug de duplicación de columnas en `pd.concat`. Latente.

**Patrón dominante:** los providers tienen una cultura defensiva uniforme:

> **Nunca fallar hacia arriba. Devolver algo plausible (caché, vacío, subset). Loguear el error a consola. El pipeline sigue.**

Fortaleza: el pipeline raramente se cae.
Debilidad: los errores se vuelven invisibles. OilPriceAPI 403 + best-effort + `|| echo` (F3-05/F3-16/F3-17) es el caso extremo.

---
## Hallazgos por provider

### Router (`router.py`)

- **A5-09 (MEDIA)** — `get_market_data` no pasa ventana temporal a Yahoo. Descarga 10 años completos en cada fetch cuando la caché ha caducado (>23h).
- **A5-10 (MEDIA)** — `preferred_order = ["yahoo", "fred", "polygon"]`. FRED no es equity. Deprioriza Polygon (que sí lo es).
- **A5-11 (MEDIA)** — `_load_cache` devuelve subset silencioso. Si faltan 100 tickers de 562, devuelve 462 sin marcar incompletitud.
- **A5-12 (BAJA)** — 2 `except:` desnudos en `get_treasury_data` y `get_options_data`.
- **A5-15 (MEDIA)** — No distingue "lote vacío" de "provider devolvió caché completa".
- **A5-18 (MEDIA)** — `_check_khuerfano` solo imprime. Detección pasiva sin acción.
- **A5-14 (BAJA)** — `Yahoo.is_available()` redundante con `_download_with_retries` que ya lanza excepción.

### Yahoo (`yahoo.py`)

- **A5-13 (ALTA)** — Fallback `_load_cache()` devuelve parquet completo. En fallo de lote (5 tickers), `data_loader.py` recibe 562 tickers, añade a `all_data`, y `pd.concat` produce columnas duplicadas. El manifest no detecta esto (`pct_dup_last` mide filas, no columnas). Latente: Yahoo lleva semanas funcionando bien en el cron.

### BackupProvider (`backup_providers.py`)

- **A5-30 (MEDIA)** — Orden de providers: tiingo (1/min) primero, finnhub (60/min) último. El más rápido se prueba en 4º lugar.
- **A5-31 (MEDIA)** — `_validate_with_cache` devuelve `None` tanto en "sin referencia" como en "error de validación". Los tres casos colapsan.
- **A5-32 (MEDIA)** — `_load_reference_cache` deduplica silenciosamente `market_data > stock_prices` (keep=first). Pierde los datos de stock_prices sin aviso.
- **A5-33 (MEDIA)** — Fallback lento: con 15 tickers en backup, tiingo tarda ~15 minutos. Puede exceder timeout del workflow.
- **A5-34 (BAJA)** — `_validate_ohlcv` asume MultiIndex. Sin guard.
- **A5-35 (BAJA)** — CircuitBreaker HALF_OPEN sin transición explícita.
- **A5-36 (MEDIA)** — `datetime.now()` en los 5 métodos `_*_daily`. No respeta `reference_date`.
- **A5-37 (BAJA)** — Inconsistencia `datetime.now()` vs `pd.Timestamp.now()`.

### Xetra (`xetra_provider.py`)

- **A5-40 (MEDIA)** — Fallback a caché obsoleta sin marcar staleness. El pipeline recibe datos posiblemente de semanas atrás.
- **A5-41 (MEDIA)** — Error de fetch → ticker omitido silenciosamente. Sin contador `failed_tickers`.
- **A5-42 (BAJA)** — `TOKEN_TTL_SAFETY = 120` definido y nunca usado.
- **A5-43 (BAJA)** — `WS_START` evaluado en import time con `datetime.now()`.
- **A5-44 (BAJA)** — `end_iso` mezcla hora local con sufijo UTC (`Z`).
- **A5-46 (BAJA)** — `keep="last"` aquí vs. `keep="first"` en backup.
- **A5-47 (BAJA)** — `_to_multiindex` sin try/except en cache-hit path.
- **A5-48 (BAJA)** — `pd.Timestamp.now()` en `get_prices` para `today_norm`.
- **A5-49 (BAJA)** — `_query_one` descarta datos parciales si falla la 2ª iteración.

### Euronext (`euronext_provider.py`)

- **A5-50 (MEDIA)** — Sin walk-back FU-014. Usa estimación heurística `int(days_gap * 1.4) + 5`. Suficiente para gaps normales, no para Semana Santa.
- **A5-51 (MEDIA)** — `_cache_is_fresh` sin `reference_date`. Inconsistente con Xetra/BME (FU-018).
- **A5-52 (BAJA)** — `enddate = datetime.now()`. No respeta reference_date.
- **A5-53 (BAJA)** — `("adjusted", "Y")` duplicado en el POST.
- **A5-54 (MEDIA)** — `_to_multiindex` sin try/except en cache-hit path. Mismo patrón que Xetra A5-47.
- **A5-55 (BAJA)** — `_save_cache` con `to_csv` (no atómico).
- **A5-56 (MEDIA)** — Error de fetch → drop silencioso.

### BME (`bme_provider.py`)

- **A5-57 (MEDIA)** — `Open <- reference` (precio de referencia, no apertura real). Contamina cualquier indicador que consuma Open.
- **A5-58 (BAJA)** — `MAX_SESSIONS = 1250` definido y nunca usado.
- **A5-59 (BAJA)** — `datetime.now()` sin reference_date.
- **A5-60 (BAJA)** — `INTER_REQUEST_DELAY = 0.3` aplicado solo tras éxito.
- **A5-61 (BAJA)** — `_to_multiindex` sin try/except en cache-hit path.
- **A5-62 (MEDIA)** — Error de fetch → drop silencioso.

### Fund data (Amundi, Blackrock DAX/ISF/IWM, SSGA, Invesco)

- **A5-70 (MEDIA)** — `blackrock_fund_data.py` y `blackrock_isf_fund_data.py` son **copy-paste ~98%**. Deberían ser un módulo parametrizado.
- **A5-71 (MEDIA)** — `except:` desnudo en `parse_fecha_es` de ambos.
- **A5-72 (MEDIA)** — Patrón "caché pese al error" en Amundi, Blackrock DAX, Blackrock ISF. El caller no distingue caché fresca de caché obsoleta.
- **A5-73 (MEDIA)** — `datetime.now()` en los 6 ficheros. Ninguno recibe `reference_date`.
- **A5-74 (BAJA)** — `robust_z` duplicado entre Blackrock DAX e ISF.
- **A5-76 (MEDIA)** — `robust_z` de Blackrock DAX/ISF usa **media/std**, no mediana/MAD. El nombre miente.
- **A5-77 (MEDIA)** — 3 implementaciones distintas de "robust_z": canónica (utils), media/std (Blackrock), mediana/MAD (SSGA inline).
- **A5-78 (BAJA)** — Dead code: `if not download_fund_file(): return pd.DataFrame()` nunca se activa (retorna True incluso con caché obsoleta).
- **A5-79 (MEDIA)** — Escritura no atómica de caché en SSGA.
- **A5-80 (BAJA)** — `download_fund_file` retorna True con caché obsoleta.

**Nota positiva:** `invesco_client.py` es el único con calidad de librería (logging, ClientError, retry, JSONDecodeError específico). Contraste con los otros 5.

---
## Patrón definitivo — "cultura del fallback silencioso"

Tras auditar 13 providers, el sistema tiene **un contrato informal uniforme**:

> **Fallar por ticker es aceptable. Fallar por lote no. Nunca propagar excepción al pipeline. Loguear el fallo y continuar con el resto.**

**Manifestaciones:**

| Provider | Forma del fallback | Gravedad |
|---|---|---|
| Yahoo | Devuelve parquet completo | **ALTA** (A5-13) |
| OilPriceAPI | 403 → best-effort → exit 0 | **ALTA** (F3-05/F3-16/F3-17) |
| Xetra | Caché obsoleta per-ticker | MEDIA (A5-40) |
| Euronext | Estimación de gap | MEDIA (A5-50) |
| BME | Caché obsoleta per-ticker | MEDIA |
| BackupProvider | Devuelve vacío + log | MEDIA |
| Blackrock/Amundi | "Caché pese al error" | MEDIA (A5-72) |
| SSGA | `errors.append(ticker); continue` | MEDIA |

**Mitigación actual:**

- El manifest calcula cobertura por mercado (FU-002-bymarket).
- El guard bloquea commits si cobertura < 0.95 en mercados activos.
- El health_check vigila las últimas 5 filas.
- El cron multi-slot F-IAE-CRON-02 reintenta si los 4 slots fallan.

**Lo que NO está cubierto:**

- El provider no declara **qué tickers fallaron**. El pipeline recibe "algo" y el manifest lo detecta por coverage, pero el provider no lo reporta.
- **No hay alerta cuando el fallback es caché obsoleta**. El manifest detecta coverage bajo pero no distingue "dato fresco" de "dato de hace 3 semanas".
- El sistema no distingue "descarga OK" de "fallback aplicado". El campo `source` del manifest es estático.

---

## Recomendaciones priorizadas

### Prioridad 1 — ALTA

1. **A5-13** — Fix Yahoo: propagar excepción o filtrar la caché por `tickers` antes de devolverla.
2. **F3-05 + F3-16 + F3-17** — Decidir política OilPriceAPI: upgrade plan / declarar FUTURE_SETTLEMENT BLOCKED / buscar proveedor alternativo. Añadir detección 403 con alerta.

### Prioridad 2 — MEDIA (impacto funcional)

3. **A5-76 + A5-77** — Unificar `robust_z` en `src/utils.py::robust_zscore` y eliminar las dos variantes de Blackrock/SSGA.
4. **A5-40, A5-41, A5-56, A5-62** — Los 4 providers de equity (Yahoo, Xetra, Euronext, BME) deben reportar explícitamente tickers que fallan. Propuesta: `provider.get_last_failed_tickers()` o campo `failed` en el DataFrame.attrs.
5. **A5-57** — Documentar `Open=reference` de BME en §6.3 y ajustar cualquier indicador que consuma Open.
6. **A5-70** — Parametrizar `blackrock_fund_data.py` (fusionar DAX/ISF).
7. **A5-51, A5-59** — Aplicar FU-018 a Euronext y BME (consistencia con Xetra).
8. **A5-72, A5-79** — Escribir caché de fund data de forma atómica.

### Prioridad 3 — MEDIA (arquitectura)

9. **A5-09** — Router: pasar `since_date` a Yahoo. Bug análogo a FU-014 pero para el mercado USA.
10. **A5-10** — Reordenar `preferred_order`. Polygon antes que FRED.
11. **A5-11** — Router `_load_cache`: propagar `RuntimeError` si faltan tickers (no subset silencioso).
12. **A5-30 + A5-33** — Reordenar providers backup por velocidad (finnhub primero) o paralelizar.
13. **A5-31 + A5-32** — Distinguir estados en `_validate_with_cache`; no deduplicar silenciosamente en `_load_reference_cache`.
14. **A5-70, A5-73** — Propagar `reference_date` a fund data.

### Prioridad 4 — BAJA

15. Resto de hallazgos (except desnudos, dead constants, inconsistencias de estilo).
16. Auditoría de los providers no cubiertos: macro (`fred`, `cboe`, `cboe_index`, `polygon`) y flows trimestrales (7 ficheros).

---
## Lo que NO cubre FASE 5

- **Providers macro:** `fred.py`, `cboe.py`, `cboe_index.py`, `polygon.py` — solo verificado que están vivos y quién los consume.
- **Flows trimestrales:** `cftc_data.py`, `finra.py`, `qqq_nport_flow.py`, `qqq_sec_primary_flow.py`, `sec_fund_flow.py`, `sec_nport_international_leader_flows.py`, `sec_nport_quarters_position_change.py`. 7 ficheros pequeños (~20-142 LOC), probablemente comparten patrón defensivo.
- **Blackrock IWM** (`blackrock_iwm_fund_data.py`, 492 LOC): verificado estructuralmente que es reescritura independiente de DAX/ISF. No auditado línea a línea.
- **Amundi e Invesco:** estructura visible, no auditados `compute_primary_flow` ni `fetch_all` en detalle.

**Recomendación de cierre:** considerar FASE 5 suficientemente cubierta. Los providers no auditados son análogos a los ya vistos o de bajo impacto.

## Artefactos generados

- Scripts temporales de auditoría (borrados tras uso): `_probe_confidence_edge.py`, `_probe_unknown_ticker.py`, `audit_inventory.py`, `audit_orphans.py`, `audit_artifacts.py`.

## Próxima fase recomendada

**FASE 6 (reporte).** Motivo: las discrepancias doc-vs-código detectadas en FASE 2 y FASE 5 (F2-02, F2-03, F2-04, F2-07, F3-05, A5-76) pueden propagarse al reporte. Si el reporte comunica una métrica como si fuera X cuando en realidad es Y, el usuario final lee información incorrecta. Impacto directo en la utilidad del sistema.

Alternativamente, cerrar la auditoría y sintetizar en un informe final consolidado.

---

**Fin del informe FASE 5.**

Generado: 2026-09-26
HEAD: `b7193c4`
Providers auditados: 13 de 27
Total hallazgos: 50 (1 ALTA, 23 MEDIA, 26 BAJA)