# INFORME GATE 0 - NIPC (Net Institutional Position Change)

**Modulo:** IAE (Institutional Accumulation Evidence) - SEC 13F
**Referencia:** prompt v6.35, HEAD 9d4a81e
**Fecha:** 2026-09-19
**Estado:** Gate 0 completado. Pendiente dictamen del auditor (7 preguntas bloqueantes).

---

## 0. Resumen ejecutivo

Gate 0 empirico completado para NIPC. Estado:

- FA-2 CERRADO / PASS; NIPC desbloqueado respecto de FA-2 (dictamen P-GATE.5).
- Ciclo de curacion CUSIP cerrado y pusheado (informe + CSV + FOLLOWUPS, HEAD 9d4a81e).
- **Q4 2025 ingestado localmente** (`D:\13f_probe\processed\2025Q4\`, SHA-256 ZIP `ff340fc5...`).
- Los dos periodos consecutivos requeridos por el contrato ya existen: Q4 2025 (baseline) + Q1 2026 (target).
- Volumetria viable: 21,356 CUSIPs comunes entre periodos (88.2% de Q4).
- Gap: NIPC no existe como modulo; no hay `delta_shares.py`, `nipc.py`, `breadth.py`, `new_exit.py`.

**Se solicita dictamen sobre 7 preguntas** antes de iniciar diseno/implementacion.

## 1. Contexto del ciclo

- **Origen:** deuda posterior de Gate FA-2 (INSTITUTIONAL_ACCUMULATION_GATE_FA2_DICTAMEN.md, P-GATE.5).
- **Definicion contractual (PROPUESTA §6.3):**
  - `NIPC = sum_i DeltaShares_i`.
  - `DeltaShares_i,t = Shares_i,t - Shares_i,t-1`.
  - Solo se publica tras resolver identidad de managers.
- **DeltaShares canonico (PROPUESTA §6.1 + Q4):** `sshPrnamtType == "SH" AND putCall is null`. Excluye PRN, Put, Call. Market_value NO se usa.
- **Clave analitica autorizada (DICTAMEN_GATE0 §65):** `(report_period, canonical_security, canonical_manager_relationship)`.
- **Regla local-first IAE activa:** no push de codigo a main hasta validar funcionalidad + beneficio.

## 2. Estado de artefactos

### 2.1. Ingesta local disponible

| Quarter | Ruta | ZIP SHA-256 | Fecha ingesta |
|---|---|---|---|
| 2025Q4 | `D:\13f_probe\processed\2025Q4\` | `ff340fc5dc0bc60539b03c3a97fffa3fe4ab0d728d6507a4aadabb4a888fbe01` | 2026-09-19 |
| 2026Q1 | `D:\13f_probe\processed\2026Q1\` | (registrado en manifest) | (pre-existente) |

Manifests: `D:\13f_probe\manifests\sec_13f_2025Q4.json` y `sec_13f_2026Q1.json`.

### 2.2. Modulos existentes en `src/institutional_accumulation/sec_13f/`

- `downloader.py`: `download_13f_zip(period, dest_dir, ...)`, `extract_13f_zip(...)`.
- `parser.py`: `parse_13f(...)`.
- `storage.py`: `write_parquets(...)`, `get_manifest_path(...)`.
- `manifest.py`: `build_manifest(...)`, `write_manifest(...)`.
- `ingest.py`: `ingest_13f(quarter, source_period, base_dir=..., ...)`.
- `identity/temporal_filter.py`: `filter_by_period(...)`.
- `identity/cusip_resolver.py`: `load_exceptions`, `resolve_cusip`, `resolve_batch`.
- `identity/relationships.py`: `build_canonical_relationship(infotable_df, om2_df, submission_df, *, report_period, coverpage_df=None)`.
- `identity/amendments.py`: `apply_amendments(dfs, *, period)`.

### 2.3. Lo que NO existe

- No hay modulo de agregacion (`NIPC`, `DeltaShares`, `breadth`, `new/exit`, `clasificacion`).
- No hay columna `canonical_security` explicita; se opera via CUSIP + crosswalk.

## 3. Volumetria empirica

Filtrado por `PERIODOFREPORT`, con filtro canonico `SH + putCall=null`:

| Metrica | 2025Q4 | 2026Q1 |
|---|---|---|
| Filings (PERIODOFREPORT exacto) | 10,676 | 10,776 |
| Managers unicos (CIK) | 10,524 | 10,648 |
| Filas INFOTABLE filtradas | 3,257,817 | 3,321,967 |
| Filas canonicas (SH + null) | 3,124,594 | 3,188,083 |
| CUSIPs unicos | 24,200 | 24,838 |

**Interseccion CUSIP Q4 2025 vs Q1 2026: 21,356 (88.2% de Q4).**

**Cobertura crosswalk curado:** 3/3 CUSIPs curados (`26614N102`, `438516106`, `30231G102`) presentes en ambos periodos.

## 4. Gap analysis para NIPC

### 4.1. Lo que ya esta disponible

- Dos periodos consecutivos ingestados.
- `canonical_reporting_relationship_key` via `build_canonical_relationship`.
- Snapshot canonico composicional via `apply_amendments` (con lineage).
- Crosswalk CUSIP -> ticker (3 filas curadas Q1 2026).
- Filtro canonico `SH + null` establecido por contrato.

### 4.2. Lo que falta disenar/implementar

1. **Filtro de identidad canonica de security:** ¿CUSIP directo o via crosswalk?
2. **Unidad de agregacion:** ¿sobre que clave suma el NIPC?
3. **Filtro de discretion:** ¿`INVESTMENTDISCRETION` entra o no? DFND (29% Q1) es el caso critico.
4. **Tratamiento del crosswalk incompleto:** 3/24,838 CUSIPs curados en Q1.
5. **Coverage minimo:** umbral y comportamiento si insuficiente.
6. **Publicacion:** ¿`NIPC` publico con `INSUFFICIENT` o se omite?
7. **Ubicacion del modulo:** subpaquete nuevo o extension de `identity/`.

## 5. Preguntas bloqueantes

### Q-NIPC-1 - Uso de Q4 2025

Autorizado el ingreso de Q4 2025 como baseline para DeltaShares. ¿Se ratifica? (Q4 2025 ya esta en local; el SHA-256 queda registrado). ¿O se prefiere diferir NIPC a un ciclo posterior?

### Q-NIPC-2 - Unidad de agregacion

`sum_i DeltaShares_i`: ¿el indice `i` recorre `canonical_manager_relationship` (identidad filing + included manager + discretion) o `filing_manager_cik` puro? Diferencia: un filing manager puede reportar la misma security bajo distintos `included_manager_cik` + `discretion_type`; agregar por filing_manager colapsaria estas dimensiones.

### Q-NIPC-3 - INVESTMENTDISCRETION

Contrato Q4 dice `SH + putCall=null`. ¿Se anade filtro sobre `INVESTMENTDISCRETION ∈ {SOLE, DFND, OTR}`? DFND (~29% en Q1) es el caso delicado: el 13F documenta que parte del holding es "discrecion no total", pero sigue siendo 13F reportable.

### Q-NIPC-4 - Universo NIPC

¿NIPC se calcula sobre los 24,838 CUSIPs del 13F Q1, o solo sobre los que aparecen en el crosswalk curado + universo del radar? El contrato dice universo = `radar_equities AND section_13f_eligible AND mapped`, pero hoy el crosswalk solo tiene 3 CUSIPs.

### Q-NIPC-5 - Coverage y publicacion

¿NIPC con coverage bajo se publica como `INSUFFICIENT`, o se omite del todo? ¿Que umbral minimo? Contrato Q3 exige "% securities mapeadas" + "% peso institucional no mapeado" como DOS controles simultaneos.

### Q-NIPC-6 - Scope del primer ciclo NIPC

Opciones:

- **Minimo:** DeltaShares + NIPC con filtro canonico + coverage reportada.
- **Intermedio:** lo anterior + Institutional Breadth (buyers/sellers) segun PROPUESTA §6.2.
- **Completo:** lo anterior + New/Exits (§6.4) + clasificacion (§7, tabla NO_EVIDENCE / INSUFFICIENT / COMPATIBLE / CONFIRMED_INSTITUTIONAL / CONFIRMED_FLOW).

### Q-NIPC-7 - Ubicacion arquitectonica

Opciones:

- **A:** Nuevo subpaquete `src/institutional_accumulation/aggregation/` con `delta_shares.py`, `nipc.py`, `breadth.py`.
- **B:** Extension de `identity/` con un nuevo `aggregation.py`.
- **C:** Nuevo modulo hermano `src/institutional_accumulation/nipc/`.

## 6. Propuesta de secuencia

    Gate-NIPC.0 (esta) - Gate 0 empirico -> informe -> dictamen
           |
    Gate-NIPC.1 - Especificacion NIPC (contrato interno)
           |
    Gate-NIPC.2 - Implementacion (delta_shares + nipc + tests)
           |
    Gate-NIPC.3 - Validacion end-to-end (Q4 2025 -> Q1 2026)
           |
    Gate-NIPC.4 - Informe de cierre + dictamen final

Regla local-first IAE: no push hasta Gate-NIPC.3 PASS.

## 7. Fuentes

- Contrato: `docs/auditoria/INSTITUTIONAL_ACCUMULATION_CONTRATO.md` (§Q4, §Q11).
- Propuesta: `docs/auditoria/INSTITUTIONAL_ACCUMULATION_PROPUESTA.md` (§6.1 a §6.6, §7).
- Dictamen Gate 0: `docs/auditoria/INSTITUTIONAL_ACCUMULATION_DICTAMEN_GATE0.md` (§Q2, §Q3, §Q4, §Q15).
- Dictamen Gate FA-2: `docs/auditoria/INSTITUTIONAL_ACCUMULATION_GATE_FA2_DICTAMEN.md` (§P-GATE.5).
- Informe curacion CUSIP: `docs/auditoria/INSTITUTIONAL_ACCUMULATION_CUSIP_CURATION_INFORME.md`.
- Probe Q4 2025: `D:\13f_probe\processed\2025Q4\` + manifest `sec_13f_2025Q4.json`.
- Probe Q1 2026: `D:\13f_probe\processed\2026Q1\`.

---

Fin del informe. Version 1.0 (2026-09-19). Referencia: prompt v6.35, HEAD 9d4a81e.
