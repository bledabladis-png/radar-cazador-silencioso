# Auditoría del sistema Radar - 2026-09-26

**Alcance:** FASE 0 (inventario) + FASE 2.1 (regímenes/scores) + FASE 2.3 (breadth) + FASE 3 (artefactos).
**HEAD de referencia:** `b7193c4` (verificado al inicio de la sesión).
**Método:** read-only. Sin modificar código productivo ni datos. Hallazgos con evidencia reproducible.
**Estado:** 3 fases cerradas. Fases 4, 5, 6 pendientes.

## Resumen ejecutivo

| Severidad | Count | Áreas |
|---|---:|---|
| ALTA | 3 | OilPriceAPI (futuros/plan/errores silenciosos) |
| MEDIA | 11 | doc-vs-código, dead code, registry, guard latente |
| BAJA | 12 | estilo, higiene, deuda menor |
| **Total** | **26** | — |

**Hallazgos ALTA (todos OilPriceAPI-adyacentes):**

1. **F3-05** — OilPriceAPI no sirve futuros con el plan actual (403 Feature access required). El sistema cree cubrir 5 commodities; cubre 3. `FUTURE_SETTLEMENT` (BZ=F, CL=F) está BLOCKED de facto, no declarado.
2. **F3-16** — El writer de `commodities_spot.parquet` puede publicar artefactos con `status=INVALID` (pct_dup_last=1.0) sin bloquear el write.
3. **F3-17** — `update_futures.py` + workflow con `best-effort` + `|| echo` enmascaran errores graves (403 + artefacto corrupto → exit 0 → commit).

**Patrón dominante:** la deuda es **documental y de observabilidad**, no de cálculo. El código produce los números que se propone producir. Lo que falla es que la documentación no refleja la realidad (PROMPT §6.1, §7, §11.16) y el sistema no alerta cuando algo falla silenciosamente.

---
## FASE 0 — Inventario (Gate 0)

**Herramientas creadas:** `scripts/audit_inventory.py`, `scripts/audit_orphans.py` (read-only).

**Resultados:**

- **0 ciclos de import.** El grafo es un DAG.
- **0 huérfanos reales.** Los 24 candidatos iniciales eran falsos positivos (imports relativos, `__init__.py` con re-exports, `conftest`).
- **Inventario consistente con lo declarado.** IAE 36 ficheros / 8017 LOC exacto; `src/pipeline` 18 = 17+`__init__`; `src/report` 20 = 19+`__init__`.
- **Basenames duplicados: legítimos.** `base.py`, `breadth.py`, `darkpool.py`, `leaders.py`, `slpm.py` en capas distintas; `conftest` ×2 (markers vs. fixtures).

**Hallazgo:**

- **F0-01 (MEDIA)** — `generate_estado_sistema.py` usa lista hardcodeada `EVIDENCE_DIRS`. 3 directorios de evidencia existen y no se reportan (`nipc_gate0_openfigi`, `nipc_gate0_probe`, `nipc_gate0_target_identity`). 1 directorio declarado no existe (`nipc_p70_probe`). Fix de 5 líneas pendiente.

---

## FASE 2.1 — Regímenes y Scores

**Fórmulas verificadas contra PROMPT §7:**

- `robust_zscore`: coincidencia exacta con §7.1.
- Pesos Sector Regime: coincidencia con §7 (suma 1.0, validado por `validate_weights`).
- Pesos Tactical y Structural: coincidencia con §7.
- Ventanas Structural (63/126/252): coincide con `config/settings.py`.

**Hallazgos:**

- **F2-01 (MEDIA, latente)** — `confidence_from_range` ruta DataFrame no respeta la regla `<2 válidos → 0.5` del propio docstring. En datos actuales: 0 filas afectadas. Latente.
- **F2-02 (MEDIA)** — Sector Regime aplica `SECTOR_DISPERSION_PENALTY=0.5` NO documentado en §7. La fórmula declarada nunca coincide con la real.
- **F2-03 (MEDIA)** — Sector Regime renormaliza por pesos válidos (`/ valid_weight_sum`). §7 describe suma ponderada fija.
- **F2-04 (MEDIA)** — Tactical "Breadth20" es proxy temporal (`close > EMA20`), NO el breadth sectorial por constituyentes. Homónimos entre secciones.
- **F2-05 (BAJA)** — `ewm` con `adjust` implícito distinto entre `sector_regime.py` (`adjust=False`) y `tactical_engine.py` (default `adjust=True`).
- **F2-06 (BAJA)** — §7 lista componentes Tactical en orden distinto a sus pesos.
- **F2-07 (MEDIA)** — `effective_composite` en SLPM tiene discontinuidad ×2.04 en `coverage=0.5`.
- **F2-10 (MEDIA)** — `leader_metrics` acepta `rs_momentum` y `rs_mom_20` como alias. Contrato débil.
- **F2-12 (MEDIA)** — `state_transition.py` usa `datetime.now()` en `indicators/`. Zona gris (valor solo para log, no para observación).
- **F2-13 (MEDIA)** — `HYSTERESIS` en `state_transition.py`: los valores de las tuplas son dead code (nunca se leen; solo `bool(rule)`).
- **F2-14 (MEDIA)** — Estado `TACTICAL_CORRECTION` inalcanzable. Declarado en 3 sitios (docstring, mapping, histéresis), nunca producido por el clasificador.
- **F2-15/16/17/18 (BAJA)** — Umbrales duplicados; `except:` desnudo; `print()` en producción; `composite` puede ser NaN.
- **F2-19 (MEDIA)** — `slpm_state.json` sin sector. Histéresis cross-sector si el líder rota.
- **F2-20 (BAJA)** — `save_current_state` sin try/except.
- **F2-21 (MEDIA)** — `consecutive_count` mide el estado previo, no el candidato. La histéresis no protege contra candidatos aislados como sugiere el nombre `HOLDING_..._CANDIDATE_...`.

---
## FASE 2.3 — Breadth

**Fixes recientes verificados:**

- **A1 fix (8c6330f)** — `TOP_N_SECTOR_COMPONENTS` correctamente aplicado en `compute_sector_breadth`.
- **Fix A/D (fc21671)** — Continuidad temporal vía `previous_market_day` correcta. Coincide con la verificación empírica declarada (107 / 206).
- **C1 fix `_delta` (2fe1c45)** — `max_calendar_days = max(days+5, int(days*1.6)+3)`.
- **`_fmt_ad_net` (C3)** — Las 3 reglas declaradas se cumplen.

**Hallazgos:**

- **F2-25 (MEDIA, latente)** — `_delta` de `sector_breadth_momentum.py` usa tolerancia por días naturales, NO continuidad vía `previous_market_day`. Espejo del bug fc21671, no propagado.
- **F2-26 (MEDIA, latente)** — Guard `sector_price is not None` es siempre True (porque `_get_series` devuelve Series vacía, nunca None). Riesgo `IndexError` si el sector no está en `df_market`.
- **F2-27 (BAJA)** — `SECTORS` hardcoded en `sector_breadth_momentum.py` (duplica `config/tickers.py`).
- **F2-28/29 (BAJA)** — Comentario engañoso "mismo universo"; docstring `fmt="+d"` vs. código `"{:+d}"`.

---

## FASE 3 — Artefactos y Datos

**Auditoría de los 5 parquets + 5 manifests + 1 provenance.** Script: `scripts/audit_artifacts.py` (read-only).

**Datos limpios:**

- `stock_prices`: coverage 100%, todos los mercados VALID, 0 close_nan.
- `cboe_vix3m`: 4281 filas, VALID.
- `market_data`: coverage 99.47%, US_EQUITY VALID.
- `lse_close_provenance.json`: status OK, 20 tickers LSE.

**Hallazgos ALTA:**

- **F3-05 (ALTA)** — `commodities_futures.parquet` desactualizado desde 21/09. Ejecución manual del script confirma: **OilPriceAPI devuelve 403 "Feature access required" para `/v1/futures/*`**. El endpoint requiere plan Professional. `FUTURE_SETTLEMENT` (BZ=F, CL=F) está BLOCKED de facto. **PROMPT §6.1 y §11.16 declaran 5 commodities cubiertos; realidad: 3.** Este hallazgo invalida la premisa documentada.
- **F3-16 (ALTA)** — El writer de `commodities_spot.parquet` puede escribir con `status=INVALID` (pct_dup_last=1.0) sin bloquear. Verificado empíricamente: `update_futures.py` reescribió el parquet con status INVALID (revertido).
- **F3-17 (ALTA)** — `best-effort` + `|| echo "..."` enmascara errores graves. `update_futures.py` retorna 0 con 403 + artefacto INVALID. GH Actions verde.

**Hallazgos MEDIA:**

- **F3-13 (MEDIA)** — `temporal_contract=None` en commodities (deuda declarada, FU-021-5-futuro).
- **F3-15 (MEDIA)** — `health_check.py` no verifica commodities. Un parquet obsoleto no genera alerta.
- **F3-18 (MEDIA)** — `merge_commodities_into_market` rompe contrato "in-place" cuando añade columna nueva (`sort_index` reasigna referencia local).
- **F3-19 (MEDIA, latente)** — `get_market('DX-Y.NYB') == UNKNOWN` (índice ICE). Interacción con `guard_coverage`: en escenario de coverage global < 0.95, `_all_markets_valid` falla por el UNKNOWN INVALID → guard bloquea el commit por un falso positivo.

**Hallazgos BAJA:**

- **F3-06 (BAJA)** — `commodities_spot` acepta filas con `updated_at` de fin de semana. Mitigado por intersection en `merge_commodities_into_market`.
- **F3-08 (BAJA)** — Commodities clasificados como `US_EQUITY` en `by_market`. Mala etiqueta.
- **F3-09 (BAJA)** — sha256 mismatch local en `market_data`/`stock_prices`. Causa: manifests tracked en git + parquets gitignored. En producción no aplica. Higiene local.
- **F3-10 (BAJA)** — Bug del script `audit_artifacts.py` (sys.path).
- **F3-11 (BAJA)** — `_fetch_spot` usa `updated_at` como fecha (fuente del bug F3-06).
- **F3-12 (BAJA)** — Calendario NYSE aplicado a ICE/NYMEX (proxy declarado).
- **F3-14 (BAJA)** — Retry del cron sin backoff ni notificación.

---
## Patrones detectados

**1. Doc vs. código (11 hallazgos).**
El PROMPT describe fórmulas "ideales" (suma ponderada simple, breadth única, 5 commodities cubiertos) y la implementación aplica transformaciones no documentadas (penalty de dispersión, renormalización por pesos válidos, breadth temporal distinta, 3 commodities). El código hace lo que quiere hacer, pero la doc miente.

**2. Infraestructura muerta (2 hallazgos).**
`TACTICAL_CORRECTION` (F2-14) y los valores de `HYSTERESIS` (F2-13) son infraestructura mantenida sin ejecución.

**3. Fixes no propagados (2 hallazgos).**
Los fixes de septiembre (A/D continuity, cobertura top-20) están bien implementados en su sitio original. `A/D continuity` (fc21671) no se propagó a `_delta` de momentum (F2-25).

**4. Errores silenciosos en capa externa (3 hallazgos).**
OilPriceAPI 403 en futuros → sin alerta. Best-effort + `|| echo` → sin Issue. health_check no cubre commodities → sin vigilancia. El sistema puede degradar silenciosamente.

**5. Registro incompleto (1 hallazgo).**
`get_market('DX-Y.NYB') == UNKNOWN` es cola residual de A2.3. `get_instrument_class` sí lo clasifica bien, pero `get_market` no.

---

## Recomendaciones por orden de prioridad

### Prioridad 1 — ALTA (impacto operativo)

1. **F3-05 + F3-17:** decidir política de OilPriceAPI. Opciones:
   - (a) Upgrade a plan Professional → recupera futuros.
   - (b) Declarar FUTURE_SETTLEMENT como BLOCKED permanentemente. Actualizar §6.1, §11.16, §12.
   - (c) Buscar proveedor alternativo para BZ/CL.
   
   En paralelo: añadir detección de `403` en `update_futures.py` que abra Issue automático (no silenciar).

2. **F3-16:** bloquear `write_artifact_with_manifest` cuando `status==INVALID` (o al menos, no commitear artefactos INVALID desde el workflow). Alternativa: `git add` condicionado a `manifest.quality.status != INVALID`.

3. **F3-17:** cambiar `best-effort` por `best-effort + alerta`. Al menos: `update_futures.py` retorna exit ≠ 0 cuando hay 403/401 (fallo de credenciales no es "best-effort").

### Prioridad 2 — MEDIA (deuda relevante)

4. **F2-19 + F2-21:** fix del state file SLPM. Guardar `sector_etf` + rediseñar histéresis con candidato explícito.

5. **F3-15:** extender `health_check.py` para verificar commodities (date_max, status).

6. **F3-19 + F3-07:** añadir `DX-Y.NYB` (y otros índices ICE) a `get_market`. Cierra dos hallazgos.

7. **F2-02, F2-03, F2-04:** actualizar PROMPT §7 con las fórmulas reales. No tocar código.

8. **F2-07, F2-10, F2-13, F2-14:** limpieza de SLPM.

9. **F3-18:** fix `sort_index` del merge.

### Prioridad 3 — BAJA (deuda menor)

10. **F0-01:** enumeración dinámica de `EVIDENCE_DIRS`.
11. **F2-25, F2-26:** propagar continuidad temporal a `_delta`; fix guard `sector_price`.
12. **F3-06, F3-11:** filtrar filas no-bursátiles en `_fetch_spot`.
13. **F3-09:** decisión sobre versionar o no manifests (`.gitignore` o `repair_manifest.py`).
14. Resto de BAJA: estilo y comentarios.

### No urgente

- F2-01 (latente, no activado).
- F2-05, F2-06, F2-11, F2-12, F2-15, F2-16, F2-17, F2-18, F2-20, F2-27, F2-28, F2-29.
- F3-08, F3-10, F3-12, F3-14.

---
## Lo que NO cubre esta auditoría

- **FASE 1 (arquitectura profunda).** Parcialmente cubierta por FASE 0 (0 ciclos, 0 huérfanos). Falta verificación de fuentes únicas (`market_calendar`, `instrument_registry`, `utils.write_artifact_with_manifest`) contra duplicación real.
- **FASE 2.2 (IAE §10.x).** No iniciada. 845 tests pasan, pero la coherencia fórmula-vs-implementación no se ha verificado línea por línea.
- **FASE 2.4 (MTE + Dark Pool).** No iniciada.
- **FASE 4 (tests).** No iniciada. Suite verde, pero cobertura real por módulo del radar (IAE tiene 90%, radar no medido).
- **FASE 5 (providers).** No iniciada. Solo OilPriceAPI auditado (por accidente).
- **FASE 6 (reporte).** No iniciada. `render_*` no verificado contra datos reales.
- **FASE 7 (dead code sistemático).** Parcialmente cubierta por F2-13, F2-14.

## Artefactos generados en esta auditoría

- `scripts/audit_inventory.py` (con `.orig` de backup)
- `scripts/audit_orphans.py`
- `scripts/audit_artifacts.py`
- `scripts/_probe_confidence_edge.py` (temporal)
- `scripts/_probe_unknown_ticker.py` (temporal)
- `_backup_futures_20260926_224851/` (temporal, revertido)

Decisión pendiente al cierre de la auditoría completa: versionar los scripts de auditoría en `scripts/audit/` o borrarlos.

## Próxima fase recomendada

**FASE 5 (providers).** Motivo: OilPriceAPI fue el primer provider roto silenciosamente. Los 27 restantes pueden tener patrones similares (fallo sin alerta, rate limit no respetado, fallback mal documentado). ROI alto: un provider roto implica cobertura declarada falsa.

Después: FASE 2.2 (IAE) → FASE 6 (reporte).

---

**Fin del informe.**

Generado: 2026-09-26
HEAD: `b7193c4`
Total hallazgos: 26 (3 ALTA, 11 MEDIA, 12 BAJA)