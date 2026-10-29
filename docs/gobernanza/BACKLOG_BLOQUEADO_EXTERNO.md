# Backlog bloqueado externo

**Documento de consolidacion. Vista unica de lo que no depende de
nosotros.**

Fecha: 2026-10-29 (sesion 16).
Regla: este documento es **indice**. La fuente canonica es
REGISTRO.yaml. Si discrepa, manda el REGISTRO.

---

## Resumen

| Categoria | N | Quien bloquea |
|---|---:|---|
| Esperando auditor externo | 3 | Auditor |
| Esperando datos futuros | 1 | Proveedor / tiempo |
| Esperando decision de producto | 1 | Usuario |

**Total items bloqueados: 5.**

**Ninguno bloquea el pipeline diario.** Los bloqueos afectan a
mejoras opcionales (validacion estadistica SOW, activacion SOW)
y a una decision de producto (IWM).

---

## 1. Esperando auditor externo (3 items)

### 1.1. Documentos listos para enviar

| Item | Documento | Version | Commit | Que pide |
|---|---|---|---|---|
| DICT3-TOP20-COBERTURA | docs/auditoria/wyckoff/70_top20_cobertura_design_v2d_2026-10-26.md | v2d | fc0b695b | Cierre A.2c |
| DICT3-H04-INTERVALS | docs/auditoria/wyckoff/73_respuesta_dictamen_h04_2026-10-26.md | v1 | 5b855ce5 | Cierre registro rechazo |

### 1.2. Bloqueados por otra cosa

| Item | Bloqueo | Notas |
|---|---|---|
| C2-920 | Auditor debe publicar comando + HEAD del valor externo (-4.264.449.932). Sin eso, delta 920 no es reproducible. | Dependencia: docs/auditoria/iae/golden/README.md |

### 1.3. SOW-BEAR-CONFIRMATION

v1.10.1 sin redactar. 3 P0 del dictamen 2026-10-28: universo ETF vs
constituyentes, parametros SOW explicitos, market proxy fijo. **Ya no
depende de cohort** (a diferencia de v1.10, que si lo requeria). Puede
usar stock_prices_extended.parquet o una ventana prospectiva nueva.

No urgente: no bloquea pipeline ni promocion actual.

---

## 2. Esperando datos futuros (1 item)

| Item | Dependencia | Fecha estimada |
|---|---|---|
| WYCKOFF-5BX | n_confirmed_H20_complete >= 550 post-2026-10-02 | ~2027 |

No hay accion. Solo esperar a que los datos se acumulen.

---

## 3. Esperando decision de producto (1 item)

| Item | Decision pendiente |
|---|---|
| IWM-HOLDINGS-EXPANSION | IWM (Russell 2000) tiene 1977 holdings, top-20 cubre ~6.8%. Para 50% harian falta ~700 tickers. Sistema holdings-based no gana nada con microcaps. |

Decision del usuario. No depende del auditor. Si la respuesta es
"no", se cierra WONT_FIX con razon. Si es "si", trabajo de descarga
+ validacion de 6 pasos.

---

## 4. Observacionales con fecha (NO son bloqueo externo)

| Item | Fecha | Accion |
|---|---|---|
| H5.3 | 2026-11-20 | Verificar cron trimestral 13F |
| RECEIPT-AUDIT-LAG | 2026-11-17 | Cierre con 4 semanas obs |
| CRON-SLOTS-OBS | 2026-11-10 | Revision runs 20-10 a 10-11 |
| SCHEDULER-DELAYS | 2026-11-10 | p95 delays |

Estos no son "bloqueados externos". Solo esperan fecha. Se listan
aqui por conveniencia.

---

## 5. Verificacion proximo scheduled

20 tickers nuevos en universe_expansions.yaml (19 SPY + 1 QQQ),
pendientes de descarga en scheduled 23:17 UTC 29-10 o 03:17 UTC 30-10.

Verificar con:

    gh run list --workflow=daily_run.yml --limit=3
    gh run download <rid> -n p1-h02-coverage -D tmp
    # inspeccionar tmp/p1_h02_cobertura_efectiva.csv

Esperado: SPY cov_vs_grupo ~85%, QQQ ~85%.

---

## 6. Como se actualiza este documento

1. Actualizar REGISTRO.yaml (fuente canonica).
2. Actualizar la fila correspondiente aqui.
3. Si el item se cierra, retirarlo de la tabla.

Si este documento y REGISTRO discrepan, manda REGISTRO.
