# EVIDENCIA FORENSE H1-A - Drift del E2E contractual

**Fecha:** 2026-09-27
**Estado:** CERRADO por el auditor externo
**Referencia:** hallazgo H1-A del dictamen de auditoria externa (ver `AUDITORIA_EXTERNA_2026-09-27.md`)

---

## 1. Problema

El script `scripts/iae_contractual_nipc_e2e.py` falla contra el golden de la seccion 12.5 del `IAE_MAESTRO.md`:

| Metrica | HEAD actual | Golden 12.5 | Delta |
|---|---:|---:|---:|
| delta_radar_rows | 553.319 | 553.321 | -2 |
| nipc_total | -4.317.678.307 | -4.316.734.936 | -943.371 |
| nipc_sole | -32.484.688.301 | -32.484.713.330 | +25.029 |
| nipc_dfnd | 28.316.684.008 | 28.317.652.408 | -968.400 |
| nipc_otr | -149.674.014 | -149.674.014 | 0 (PASS) |
| evidence_class | CONTRACTUAL | CONTRACTUAL | PASS |
| determinismo | PASS | PASS | PASS |

El desglose es internamente consistente:
```
+25.029 (sole) - 968.400 (dfnd) = -943.371 (total)
-1 (both) - 1 (new) = -2 (radar rows)
```

---

## 2. Hipotesis inicial

El drift procede de la regeneracion del crosswalk `data/mappings/cusip_radar_crosswalk.csv` por la Fase G.

Pista inicial: diff `7caa86b` vs HEAD mostro 222 filas removed / 221 filas added, pero solo 1-2 CUSIPs con cambio real de presencia.

---

## 3. Cadena forense — paso a paso

### 3.1. Diff estructural del crosswalk

```powershell
git show 7caa86b:data/mappings/cusip_radar_crosswalk.csv
git diff 7caa86b HEAD -- data/mappings/cusip_radar_crosswalk.csv
```

**Resultado:**

| Metrica | 7caa86b | HEAD | Delta |
|---|---:|---:|---:|
| Filas totales | 246 | 245 | -1 |
| CUSIPs retirados | - | 2 | -2 |
| CUSIPs anadidos | - | 1 | +1 |

**CUSIPs afectados:**

| CUSIP | Ticker | Accion | Naturaleza (SEC) |
|---|---|---|---|
| 023135906 | AMZN | Retirado | CALL |
| 595112903 | MU | Retirado | CALL |
| 84615Q103 | SPCX | Anadido | Ausente en INFOTABLE Q4/Q1 |

### 3.2. Diff por campo (mismo CUSIP, distintos valores)

Los 222/221 filas del diff bruto corresponden a cambios de contenido en CUSIPs comunes, no a cambios de presencia:

| Campo | Filas con diff |
|---|---:|
| reason | 220 |
| title_of_class | 220 |
| valid_to | 219 |
| valid_from | 1 |

El cambio de `valid_to: '2026-03-31' -> ''` es cosmetico (afecta solo al formato de la vigencia, no a la resolucion de identidad para los periodos Q4/Q1).

### 3.3. Barrido CUSIP-por-CUSIP

Se ejecuta el pipeline contractual 5 veces sobre los mismos parquets, cambiando solo el crosswalk:

```python
# Script simplificado (real: _h1_fase6.py, ver mas abajo)
head_content = Crosswalk HEAD
crosswalk_variants = [
    ("HEAD (245 filas, base)", head_content),
    ("HEAD + 023135906 (AMZN)", head_content + fila_AMZN),
    ("HEAD + 595112903 (MU)", head_content + fila_MU),
    ("HEAD + 023135906 + 595112903", head_content + fila_AMZN + fila_MU),
    ("HEAD - SPCX + AMZN + MU (sin SPCX)", sin_spcx + fila_AMZN + fila_MU),
]
for label, cw in crosswalk_variants:
    write_crosswalk(cw)
    reload_pipeline()
    r = run_contractual_nipc()
    print(label, r["delta_radar_rows"], r["nipc_total"])
```

**Resultado:**

| Crosswalk | delta_radar_rows | nipc_total |
|---|---:|---:|
| HEAD (245 filas, base) | 553.319 | -4.317.678.307 |
| HEAD + 023135906 (AMZN) | 553.320 | -4.316.735.138 |
| HEAD + 595112903 (MU) | 553.320 | -4.317.678.105 |
| HEAD + AMZN + MU | **553.321** | **-4.316.734.936** |
| 7caa86b completo | 553.321 | -4.316.734.936 |

**Coincidencia exacta con el golden 12.5 al restaurar el crosswalk historico, con tolerancia 0.**

### 3.4. Descomposicion aritmetica

```
AMZN:  rows +1  NIPC +943.169
MU:    rows +1  NIPC +202
                    -------
TOTAL: rows +2  NIPC +943.371
```

Verificacion con el delta del E2E:
```
25.029 (sole) - 968.400 (dfnd) = -943.371 ✓
-1 both - 1 new = -2 rows ✓
```

### 3.5. SPCX descartado como causante

El crosswalk HEAD contiene `84615Q103` (SPCX), ausente en `7caa86b`. SPCX no aparece en INFOTABLE Q4 2025 ni Q1 2026 (solo en Q2 2026). Su presencia/ausencia no afecta al E2E Q4->Q1.

Prueba directa: `HEAD - SPCX + AMZN + MU` da el mismo resultado (553.321 / -4.316.734.936) que `HEAD + AMZN + MU`. **SPCX es irrelevante para el drift.**

---

## 4. Conclusion

**El drift es atribuible integramente a la regeneracion del crosswalk por Fase G entre `7caa86b` y HEAD.**

- No hay regresion de codigo. Los 2 commits IAE entre `b7193c4` y HEAD (`1805766`, `e0064ee`) tocan dead code, no logica de calculo.
- El golden historico seccion 12.5 sigue siendo valido para su snapshot.
- El script `iae_contractual_nipc_e2e.py` compara contra cifras hardcoded que dejaron de ser invariantes. **H4 (golden versionado) es la solucion estructural.**

---

## 5. Reproduccion

### 5.1. Prerequisitos

- Parquets 13F en `data/sec_13f/processed/` (Q4 2025, Q1 2026).
- Crosswalk en `data/mappings/cusip_radar_crosswalk.csv`.
- `7caa86b` accesible en git.

### 5.2. Script reproducible

```powershell
# Backup del crosswalk actual
Copy-Item data/mappings/cusip_radar_crosswalk.csv _backup.csv

# Restaurar el de 7caa86b
git show 7caa86b:data/mappings/cusip_radar_crosswalk.csv |
    Out-File -Encoding utf8 data/mappings/cusip_radar_crosswalk.csv

# Ejecutar E2E (debe dar PASS exacto)
py scripts/iae_contractual_nipc_e2e.py

# Restaurar HEAD
Copy-Item _backup.csv data/mappings/cusip_radar_crosswalk.csv
Remove-Item _backup.csv
```

**Salida esperada:**
```
RECONCILIACION 12.5
  [PASS] delta_radar_rows     actual=553321  esperado=553321
  [PASS] nipc_total           actual=-4316734936.0  esperado=-4316734936.0
  [PASS] nipc_sole            actual=-32484713330.0  esperado=-32484713330.0
  [PASS] nipc_dfnd            actual=28317652408.0  esperado=28317652408.0
  [PASS] nipc_otr             actual=-149674014.0  esperado=-149674014.0
  [PASS] evidence_class       actual=CONTRACTUAL  esperado=CONTRACTUAL
  [PASS] determinismo_nipc_total
  [PASS] determinismo_delta_radar_rows
RESULTADO: PASS
```

---

## 6. Implicaciones para H4 (golden versionado)

El drift demuestra que el golden seccion 12.5 no es invariante respecto al crosswalk. Cualquier regeneracion del crosswalk (Fase G en adelante) lo invalida.

**Estructura propuesta (ver DISENO_H1B.md seccion 7):**

```
docs/auditoria/iae/golden/
  12_5_historic.json    # Snapshot seccion 12.5 con crosswalk_sha256 = <hash de 7caa86b>
  current.json          # Snapshot post-fix H1-B (a crear tras cierre H1-B)
  README.md
```

Cada golden debe incluir:
- `periods` (Q4 2025, Q1 2026)
- `crosswalk_sha256`
- `catalog_sha256`
- `parquet_hashes` (INFOTABLE por trimestre)
- `git_head`
- `expected` (todas las metricas)
- `generated_at`
- `security_type_source` (anadido por H1-B)
- `security_type_source_hash`

---

## 7. Evidencia persistida

JSONs generados durante la investigacion:

| Fichero | Contenido |
|---|---|
| `outputs/audit/iae_e1_contractual_nipc/20260927T175406Z_iae_e1.json` | E2E con crosswalk HEAD (FAIL) |
| `outputs/audit/iae_e1_contractual_nipc/20260927T183749Z_iae_e1.json` | E2E con crosswalk 7caa86b (PASS) |
| `outputs/audit/b1_reconciliation/20260927T181051Z_b1_final.json` | Reconciliacion B1 (PASS) |

Los JSONs no estan versionados en git (viven en `outputs/audit/`). Si se necesita reproducir la evidencia, ejecutar los scripts de la seccion 5.

---

## 8. Estado

**CERRADO por el auditor externo.**

No reabrir sin causa material (cambio de parquets, cambio de crosswalk, o nuevo drift en el E2E). Si aparece un drift nuevo, reabrir como H1-A-bis con evidencia forense actualizada.

---

**Fin del documento.**
