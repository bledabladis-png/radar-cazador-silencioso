# AUDITORIA ESTRUCTURAL DEL SISTEMA - 2026-09-20



**Objeto:** auditoria tecnica estructural del sistema Radar de Rotacion

Sectorial, con foco en contratos temporales (FU-021-5 + FU-021-3C-bis),

modulo IAE (NIPC + contratos semanticos P38/P60/P61) y manifests FU-002.



**Solicitada por:** el usuario, tras detectar que los 3 documentos de

contexto (PROMPT_MAESTRO v6.43, transfer 2026-09-20, informe tecnico IAE)

estaban desactualizados entre si.



**Ejecutada por:** asistente, en 14 bloques de Gate 0 con evidencia directa.



**Alcance:** estructural. NO se audito calculo (regimenes, scores, SLPM,

MTE, darkpool, reporte) ni ejecucion del pipeline en CI.



---



## 1. Estado del sistema verificado



| Metrica | Declarado en prompt | Verificado |

|---|---|---|

| HEAD | 83e60c3 | **d3c9a87** |

| origin/main | - | 9d4a81e |

| Ahead | 104 | **107** |

| Working tree | limpio | limpio |

| Tests | 979 passed + 2 skipped | **979 passed + 2 skipped** |

| compileall | OK | OK (silencioso) |

| pyflakes | 0 warnings | **0 (sin output)** |

| Gate 10/10 | OK | no verificado en este ciclo |



Desviacion: +3 commits respecto al prompt v6.43 (los 3 son documentales:

9c32c9b, e8c3bf2, d3c9a87). No afectan a tests.



---



## 2. Alcance cubierto



- Repo, commits, working tree, coherencia HEAD/ahead.

- Validacion declarada (compileall, pyflakes, pytest).

- Manifests FU-002 en data/: 5 ficheros. Hash manifest vs hash disco.

- Reader de manifests: BackupProvider + _load_reference_cache +

&#x20; _validate_with_cache.

- Los 10 contratos temporales (base.py, registry.py, _common.py,

&#x20; equity_eod, index_eod, volatility_index, rate_yield,

&#x20; future_settlement, spot_commodity, fx_daily_cut, consolidate).

- P60 (_normalize_canonical), P61 (temporal_validity + integracion en

&#x20; resolve_security_identity), P38 (compute_coverage_pairwise + _mapped_mask).

- Propagacion de temporal_meta (data_loader -> data_load ->

&#x20; get_effective_meta -> writer_observation_date).

- Inventario de tests por fichero (pytest --collect-only).

- Callers de write_artifact_with_manifest y de resolve_all_contracts.



---



## 3. Hallazgos por severidad



### 3.1. ALTA



Ninguno.



### 3.2. MEDIA (5)



Los 5 comparten racional: **semantica declarativa no operativa** o

**fallo silencioso sin distincion UNAVAILABLE/INVALID**.



#### H-B10-A - per_pair_max_lag no se aplica en la FSM



`src/temporal_contracts/fx_daily_cut.py:37` declara:



&#x20;   max_lag_days = {"EURUSD=X": 0, "USDJPY=X": 0, "USDCNY=X": 1}

&#x20;   per_pair_max_lag = {"EURUSD=X": 0, "USDJPY=X": 0, "USDCNY=X": 1}



`compute_status` (base.py L58-61) colapsa el dict a su maximo:



&#x20;   if isinstance(max_lag_days, dict):

&#x20;       ml = max(max_lag_days.values()) if max_lag_days else 0



`per_pair_max_lag` se declara pero nunca se lee. Grep: 15 declaraciones,

0 lecturas. Consecuencia: un lag=1 sobre EURUSD=X (declarado max=0) se

marca STALE en lugar de INSUFFICIENT.



#### H-B10-B - per_ticker_lag no se aplica en la FSM



`per_ticker_lag` se declara en 7 contratos (INDEX_EOD_USA,

INDEX_EOD_EUROPA, INDEX_EOD_COMMODITY, INDEX_EOD_CURRENCY,

VOLATILITY_INDEX, RATE_YIELD, FUTURE_SETTLEMENT, SPOT_COMMODITY).

`compute_status` solo lee `max_lag_days`. Consecuencia: un lag=5 sobre

^FTSE (declarado max=1) se marca STALE en lugar de INSUFFICIENT.



#### H-B11-P - fx_expected no aplica cutoff 17:00 ET



`_common.py:67-69`:



&#x20;   def fx_expected(reference_date):

&#x20;       """Fecha esperada FX: fecha de reference_date (cutoff 17:00 ET)."""

&#x20;       return to_date(reference_date)



El docstring promete cutoff, el codigo no lo evalua. Run a 00:00 ET:

fx_expected = hoy, ultimo dato FX disponible probablemente ayer, lag=1.

Con max_lag colapsado a 1 (H-B10-A), EURUSD=X sale STALE en lugar de

INSUFFICIENT.



#### H-B12-A - try/except silencioso en resolucion de contratos



`src/data_loader.py:203-214`:



&#x20;   try:

&#x20;       from src.temporal_contracts import resolve_all_contracts

&#x20;       from src.temporal_contracts.consolidate import build_temporal_meta

&#x20;       _resolutions = resolve_all_contracts(data, reference_date)

&#x20;       _meta = build_temporal_meta(_resolutions, reference_date, run_id)

&#x20;       data.attrs['temporal_meta'] = _meta

&#x20;       ...

&#x20;   except Exception as e:

&#x20;       print(f"  [FU-021-5][WARN] Fallo resolviendo contratos: {e}")



Si falla la resolucion de CUALQUIER contrato, los 10 pierden

temporal_meta. `data.attrs['temporal_meta']` nunca se asigna.

`data_load.py:57` devuelve `{}` (default). Los 14 consumidores reciben

`temporal_meta={}`. `writer_observation_date` cae a fallback silencioso.



Unico sintoma: un print a stdout. En CI nadie lo lee si el workflow

pasa. El manifest ya se escribio antes (linea 194-201).



No hay evidencia de ocurrencia en historico local (outputs/audit y

outputs/report estan gitignored). MEDIA latente no materializada.



#### H-B13-B - default {} enmascara la ausencia



`src/pipeline/data_load.py:57`:



&#x20;   'temporal_meta': df_market.attrs.get('temporal_meta', {}),



`{}` es indistinguible de "resuelto y vacio". Los consumidores no pueden

distinguir fallo de "no aplica". Contradice §3.1 del prompt: "NO

confundir VALID con UNAVAILABLE".



### 3.3. BAJA (10)



| ID | Descripcion |

|---|---|

| H-B8-A | docstring `identity_type opcional` (security_identity.py:85) contradice P60 |

| H-B8-C | registry.py docstring "9 contratos", son 10 |

| H-B10-C | consolidate.py docstring "9 contratos", son 10 |

| H-B11-A | __init__.py docstring "9 contratos", son 10 |

| H-B9-C | nipc.py docstring `unmapped_weight_*` stale (renombrado a `unmapped_count_*`) |

| H-B9-E | `unmapped_count=0.0` en caso empty, coherente seria 1.0 |

| H-B10-D | IndexEODCurrency declara session_calendar="ICE", usa nyse_expected |

| H-B11-K | volatility_index.py docstring "Hereda de INDEX_EOD_USA" (falso, hereda de _IndexEODBase) |

| H-B11-M | _IndexEODBase._expected sin @abstractmethod |

| H-B13-A | try/except todo-o-nada en resolucion de contratos (amplifica H-B12-A) |

| H-B13-C | data_load.load_all_data docstring omite `temporal_meta` del dict retornado |

| H-B7-A | informe IAE dice "~280 tests", real 324 |

| H-B7-B | 5 ficheros con conteo documental ±1 a ±3 |



### 3.4. INFO (7)



| ID | Descripcion |

|---|---|

| H1 | prompt v6.42 no recibido (commits c1a82bc) |

| H2 | Tandas 1-4 no documentadas (commits 367c58c, 373c5cd, cc6a402, 501126b) |

| H3 | transfer #2 (commit d3c9a87) no recibido |

| H4 | §15.33 dice "8 commits P60/P61/P38", son 6 |

| H-B10-E | weekday_expected no conoce festivos EU (docstring lo declara PROVISIONAL) |

| H-B10-F | EquityEOD stateful (eligible_universe en __init__) |

| H-B11-Q | to_date no acepta strings |

| `_weighted` | codigo muerto en nipc.py:218-224 |



---



## 4. Autocritica: hipotesis retiradas durante la auditoria



- **H-B5-A (ALTA, retirada)**: "nadie lee los manifests FU-002". Falso.

&#x20; El reader real es `BackupProvider._load_reference_cache`

&#x20; (data/providers/backup_providers.py:105). Mi busqueda no incluyo

&#x20; `data/providers/`.

- **H-B6-A (MEDIA, retirada)**: "_load_reference_cache no existe".

&#x20; Falso. Existe con ese nombre exacto en BackupProvider.

- **H-B6-B (MEDIA, retirada)**: "callers CBOE/commodities no existen".

&#x20; Falso. Estan en scripts/update_cboe.py:29 y scripts/update_futures.py:33-34.

- **H-B12-B (MEDIA, refutada)**: "riesgo de perdida de attrs". No se

&#x20; materializa: asignacion en data_loader despues de merges, lectura

&#x20; inmediata en data_load. Solo 2 usos de .attrs en todo el repo.



Leccion operativa: extender el scope de busqueda (`data/`, `scripts/`)

antes de declarar "no existe".



---



## 5. Confirmaciones sin hallazgo



- FSM `compute_status` (base.py L34-78): orden de precedencia correcto.

- Catalogo de contratos: 10 entradas en CONTRACTS_REGISTRY.

- P60 `_normalize_canonical(value, identity_type)`: ValueError si None.

- P61 `aggregate_status`: 6 reglas Q7 correctas.

- P38 `target_pairwise = sec_c \& sec_p` (interseccion, no union).

- `_mapped_mask`: filtra CANONICAL \& operational_mapping_status == VERIFIED.

- Manifests FU-002: 5/5 coherentes. Logica de status en src/utils.py:467-478

&#x20; cumple §11.8 al 100%.

- Reader BackupProvider: FSM VALID / INVALID / UNAVAILABLE correcta.

&#x20; Los 5 returns True silenciosos eliminados.

- MATCH_KEY sin report_period (spec 4.5).

- P14-BIS aplicado (fillna condicionado al _merge del JOIN).

- Propagacion temporal_meta: 14 consumidores via get_effective_meta.

- Q-P.3 respetado: df.attrs solo se usa en data_loader (asignacion) y

&#x20; data_load (lectura inmediata).



---



## 6. Lo que NO cubrio esta auditoria



- Regimenes: regimes/*.py. Ninguno leido.

- Scores: Tactical, Structural, Sector Regime, Macro Regime.

- SLPM v1.2: indicators/slpm_v12.py.

- MTE: indicators/mte/.

- Darkpool: indicators/darkpool/.

- Calculo matematico: robust_zscore, flow_proxy, agregaciones.

- Reporte: src/report/*.py.

- Cascada europea: src/stock_data_loader.py.

- CI real: sin ejecucion de workflow, solo local.

- NIPC con datos Q1 2026 reales: motor no ejecutado.

- Gate 10/10 en produccion: no verificado en este ciclo.



---



## 7. Recomendacion al auditor externo



**Un unico dictamen arquitectonico cierra los 5 MEDIA:**



**D1.** `per_ticker_lag` y `per_pair_max_lag`: ¿implementar en

`compute_status` / `resolve` (aplicacion por ticker y por par), o

retirarlos del registry y de los contratos? Si se implementa, extender

`fx_expected` con cutoff 17:00 ET (H-B11-P).



**D2.** try/except en `data_loader.py:203-214`: ¿elevar a excepcion

fatal, resolver por-contrato con agregacion parcial, o propagar flag

`temporal_meta_available: bool` a los consumidores?



**D3.** `data_load.py:57` default `{}`: ¿sustituir por `None` explicito

+ flag de disponibilidad, o por dict con campo `available: False`?



**Los BAJA/INFO** se agrupan en una sola pasada cosmetica (docstrings +

`@abstractmethod` + eliminar `_weighted`). No requieren dictamen uno a

uno.



---



## 8. Material de soporte



- Commits auditados: d3c9a87 (HEAD), origin/main 9d4a81e.

- 14 bloques de Gate 0 ejecutados por el usuario, con evidencia directa.

- Ficheros leidos: base.py, registry.py, _common.py, equity_eod.py,

&#x20; index_eod.py, volatility_index.py, rate_yield.py, future_settlement.py,

&#x20; spot_commodity.py, fx_daily_cut.py, consolidate.py, __init__.py,

&#x20; security_identity.py, temporal_validity.py, nipc.py, delta_shares.py,

&#x20; data_loader.py, data_load.py, utils.py (parcial), backup_providers.py

&#x20; (parcial), manifests data/*.manifest.json, tests

&#x20; test_artifact_manifest.py + test_backup_provider_reference.py.



---



FIN DEL INFORME

HEAD: d3c9a87

Sin commit. Sin push. Materializado en local para revision.

