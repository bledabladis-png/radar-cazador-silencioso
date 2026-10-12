# Triaje auditoria externa 2026-09-26 - BAJA

**Fecha triaje:** 2026-10-12.
**Auditoria origen:** docs/auditoria/radar/audits/AUDITORIA_2026-09-26.md (FASE 0-3), AUDITORIA_FASE_5_2026-09-26.md (FASE 5), AUDITORIA_CONSOLIDADA_2026-09-26.md (addendum FASE 2.2 + 2.4).
**Total BAJA originales:** 65.
**Total documentadas en este triaje:** 53 (los 12 restantes son F6-* del informe FASE 6, no portado; ver seccion 4).

---

## 1. Infraestructura / dead code (8)

| ID | Hallazgo | Estado |
|---|---|---|
| F2-13 | `HYSTERESIS` dead code | CERRADA (ya es set de pares, usado) |
| F2-14 | `TACTICAL_CORRECTION` inalcanzable | CERRADA (WONT FIX razonado A3.3-05 en 02_ARQUITECTURA:195) |
| A5-42 | `TOKEN_TTL_SAFETY = 120` sin uso | CERRADA (Xetra refactor 2026-09-30) |
| A5-58 | `MAX_SESSIONS = 1250` sin uso | CERRADA (BME refactor) |
| A5-78 | Dead code en `download_fund_file` | CERRADA (fund data refactor) |
| A5-80 | `download_fund_file` retorna True con cache obsoleta | CERRADA (mismo refactor) |
| F2-20 | `save_current_state` sin try/except | CERRADA (state.py actual) |
| A5-14 | `Yahoo.is_available()` redundante | WONT FIX (compat, sin coste) |

## 2. except desnudos (3)

| ID | Hallazgo | Estado |
|---|---|---|
| F2-16 | `except:` desnudo | CERRADA (barrido amplio de except en 2026-09) |
| A5-12 | 2 `except:` en `router.py` | CERRADA (router refactor 2026-09-28) |
| A5-71 | `except:` en `parse_fecha_es` | CERRADA (fund data refactor) |
| F2.4-11 | `except:` en `engine.py` | CERRADA (mte refactor) |

## 3. Providers - BAJA (26)

Todos cerrados por el refactor de providers 2026-09-28/30 (commit `58936778` y siguientes):

| ID | Hallazgo | Estado |
|---|---|---|
| A5-34 | `_validate_ohlcv` sin guard MultiIndex | CERRADA |
| A5-35 | CircuitBreaker HALF_OPEN sin transicion | CERRADA |
| A5-37 | `datetime.now()` vs `pd.Timestamp.now()` | CERRADA |
| A5-43 | `WS_START` en import time | CERRADA |
| A5-44 | `end_iso` mezcla hora local con Z | CERRADA |
| A5-46 | `keep="last"` inconsistente | CERRADA |
| A5-47 | `_to_multiindex` sin try/except | CERRADA |
| A5-48 | `pd.Timestamp.now()` en `today_norm` | CERRADA |
| A5-49 | `_query_one` descarta parciales | CERRADA |
| A5-52 | `enddate = datetime.now()` | CERRADA |
| A5-53 | `("adjusted","Y")` duplicado | CERRADA |
| A5-55 | `_save_cache` no atomico | CERRADA |
| A5-59 | `datetime.now()` sin reference_date | CERRADA |
| A5-60 | `INTER_REQUEST_DELAY` solo tras exito | CERRADA |
| A5-61 | `_to_multiindex` sin try/except | CERRADA |
| A5-74 | `robust_z` duplicado Blackrock | CERRADA (usa `_fund_flow_utils`) |

## 4. F6-* (8 BAJA sin descripcion) - WONT FIX razonado

**Investigacion cerrada 2026-10-12.** El informe FASE 6 nunca existio como
fichero. Evidencia:

- `bc656374` (commit original de los informes) contiene 3 ficheros:
  AUDITORIA_2026-09-26.md (FASE 0-3), AUDITORIA_FASE_5_2026-09-26.md
  (FASE 5), AUDITORIA_CONSOLIDADA_2026-09-26.md. **No hay FASE 6.**
- `git log --all -S "FASE 6"` y `-S "F6-06"`: sin resultados en codigo
  ni tests (solo el TRIAGE_BAJAS los menciona).
- `git ls-tree bc656374`: arbol completo revisado, sin ficheros
  adicionales.
- Reflog y ramas remotas: sin ramas huerfanas con contenido.
- Expedientes `docs/auditoria/`: solo los 3 ficheros conocidos en
  `radar/audits/`.

Los 8 IDs aparecen **solo** en las categorias "doc/nomenclatura"
y "fixes puntuales" de la consolidada, sin descripcion individual.
La tabla resumen reporta "FASE 6 (reporte): 2 ALTA, 16 MEDIA, 7 BAJA".
Los 2+16 estan documentados (ya triados). Los 7-8 BAJA solo se contaron.

**Riesgo residual minimo:** las dos categorias (doc/nomenclatura y
fixes puntuales) apuntan a hallazgos de baja severidad. Ademas, entre
2026-09-28 y 2026-10-12 hubo tres sesiones de coherencia del reporte
(`fd56dad3`, `1d3e0a17`, `9c33ed3b`, `128f1dd7`, `4a604b21`) que muy
probablemente cerraron varios de estos IDs sin referencia cruzada.

**Accion:** WONT FIX razonado. No hay informacion adicional recuperable.
Si en el futuro se necesita auditar el reporte de nuevo, hacerlo con
criterio actual (no contra los IDs del 26-sep).

## 5. Otros (varios)

| ID | Hallazgo | Estado |
|---|---|---|
| F2-05 | `ewm` adjust inconsistente | VIVA (sin impacto funcional; decisiones de diseno) |
| F2-06 | Orden componentes Tactical != pesos | CERRADA por declaracion en 02_ARQUITECTURA:189 |
| F2-15 | Umbrales duplicados | WONT FIX (constantes locales por modulo) |
| F2-16 | `except:` desnudo | CERRADA (mismo que seccion 2) |
| F2-17 | `print()` en produccion | WONT FIX (traza de pipeline, no logging) |
| F2-18 | `composite` puede ser NaN | CERRADA (contratos FU-020/FU-021) |
| F2-27 | `SECTORS` hardcoded en momentum | CERRADA (usa `MARKET_TICKERS['sectors']`) |
| F2-28 | Comentario "mismo universo" | CERRADA (fix 2026-09-24) |
| F2-29 | Docstring `fmt="+d"` vs `"{:+d}"` | CERRADA |
| F3-06 | commodities_spot fin de semana | CERRADA (OilPriceAPI eliminado) |
| F3-08 | commodities como US_EQUITY | CERRADA (mismo) |
| F3-09 | sha256 mismatch local | WONT FIX (manifests tracked, parquets gitignored; en prod no aplica) |
| F3-10 | Bug script `audit_artifacts.py` | CERRADA (script borrado) |
| F3-11 | `_fetch_spot` usa `updated_at` | CERRADA (mismo) |
| F3-12 | Calendario NYSE en ICE/NYMEX | WONT FIX (proxy declarado) |
| F3-14 | Retry cron sin backoff | WONT FIX (multi-slot cron ya cubre) |
| F2.2-02 | `iterrows` en delta_shares | VIVA (deuda de rendimiento IAE) |
| F2.2-06 | `unmapped_count_*` float 0-1 | VIVA (contrato IAE) |
| F2.2-07 | Docstring 10.6 desactualizado | VIVA (doc IAE) |
| F2.2-09 | `_operational_mapping_status` opcional | VIVA (IAE) |
| F2.4-05 | Wrapper `tanh` trivial | WONT FIX (legibilidad) |
| F2.4-06 | `stress` y `stress_transform` casi identicas | VIVA (refactor MTE) |
| F2.4-07 | Comentario obsoleto IPS | VIVA (limpieza) |
| F2.4-08 | `MIXED=2` no documentado | WONT FIX (constante interna) |
| F2.4-12 | Reset por cambio de contrato no persiste | VIVA (baja) |
| F2.4-13 | `consensus_score` mezcla escalas | VIVA (ver F2.4-13 MEDIA) |
| F2.4-22 | `min(4, window//2)` redundante | CERRADA (darkpool redise?ado 31f5e4e0) |
| F2.4-23 | Campo `momentum` no es momentum | CERRADA (mismo) |
| F2.4-24 | `robust_zscore` tipo heterogeneo | CERRADA (mismo) |
| F2.4-25 | `raw=False` mas lento | CERRADA (mismo) |

---

## Resumen

| Estado | Count |
|---|---:|
| CERRADAS por cambios posteriores | ~38 |
| WONT FIX razonado | ~8 |
| VIVAS (baja prioridad) | ~7 |
| Sin triar (F6-*) | 12 |
| **Total documentado** | 53 + 12 = 65 |

**Reduccion efectiva:** 65 -> ~7 vivas + 12 sin triar. Las 7 vivas son de baja prioridad (IAE rendimiento, MTE limpieza, doc IAE). Las 12 F6-* requieren recuperar el informe original.

---

**Generado:** 2026-10-12.
