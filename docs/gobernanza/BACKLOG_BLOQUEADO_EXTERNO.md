# Backlog bloqueado externo

**Documento de consolidacion. Una sola vista de todo lo que no
depende de nosotros.**

Fecha: 2026-10-28 (sesion 15).
Regla: este documento es **indice**. La fuente canonica de cada
item es `REGISTRO.yaml`. Si hay discrepancia, manda el REGISTRO.

**Proposito.** Evitar que cada bloqueo externo genere su propio
documento preparatorio. Aqui se ve, de un vistazo, que depende de
que. No se preparan mas versiones de nada hasta que el bloqueo se
resuelva.

---

## Resumen

| Categoria | N | Quien bloquea |
|---|---:|---|
| Esperando auditor externo | 8 | Auditor |
| Esperando datos futuros | 1 | Proveedor / tiempo |
| Esperando decision de producto | 1 | Usuario |
| Esperando item interno | 2 | Nosotros (via otro item) |

**Total items bloqueados: 12.**

**Ninguno bloquea el pipeline diario.** El pipeline (`run.py`,
caches, retry, P7, except, gobernanza) funciona sin ninguno de
estos. Los bloqueos afectan a mejoras opcionales (validacion
estadistica SOW, extension de cobertura, protocolo inferencial).

---

## 1. Esperando auditor externo (8 items)

### 1.1. Documentos listos para enviar

Ya tienen version lista. Solo falta que el usuario los envie al
canal del auditor (no definido en el repo).

| Item | Documento | Version | Commit | Que pide |
|---|---|---|---|---|
| SOW-HOLDOUT-IMMUTABLE | `docs/auditoria/wyckoff/58_contrato_holdout_inmutable.md` | **v1.2** | d5de47f6 | Firma contrato |
| DICT3-REACCUMULATION | `docs/auditoria/wyckoff/59_pre_registro_reaccumulation.md` | v1 | 0bff883f | Firma pre-registro |
| DICT3-TOP20-COBERTURA | `docs/auditoria/wyckoff/70_top20_cobertura_design_v2d_2026-10-26.md` | v2d | fc0b695b | Cierre A.2c |
| TOP20-USA-INDICES | `docs/auditoria/wyckoff/70_top20_cobertura_design_v2b_2026-10-26.md` | v2b | b6b75e55 | Division de alcance |
| DICT3-H04-INFERENCE-V2 | `docs/auditoria/wyckoff/73_respuesta_dictamen_h04_2026-10-26.md` | v1 | 5b855ce5 | 4 decisiones diseno |

### 1.2. Bloqueados por otra cosa

| Item | Bloqueo | Notas |
|---|---|---|
| DICT3-H04-INTERVALS | Auditor ya rechazo IC95 (dictamen H-04). Convive con DICT3-H04-INFERENCE-V2. No hay accion propia. | Sigue OPEN como registro del rechazo. |
| C2-920 | Auditor debe publicar comando + HEAD del valor externo (-4.264.449.932). Sin eso, delta 920 no es reproducible. | Dependencia: `docs/auditoria/iae/golden/README.md`. |

### 1.3. Ya respondidos en sesion 15

| Item | Respuesta | Doc |
|---|---|---|
| SOW-BEAR-CONFIRMATION | APROBADO CON CAMBIOS → requiere v1.10.1. **v1.10.1 NO redactada.** | — |
| SOW-HOLDOUT-IMMUTABLE | APROBADO CON CAMBIOS → requiere v1.1. **v1.1 SI redactada** (a5a26bdb). Decisiones H-04/H-07 en doc 74 (93bc595d). | doc 74 |

**Accion inmediata unica:** enviar v1.1 de SOW-HOLDOUT + doc 74 al
auditor. Nada mas.

**No se prepara v1.10.1 de SOW-BEAR** hasta que el usuario decida
si quiere abordar ese dictamen en otra sesion. Requiere 3 P0
nuevos (universo ETF, parametros SOW explicitos, market proxy
fijo) y redactar un documento nuevo. No es urgente: SOW-BEAR no
bloquea pipeline.

---

## 2. Esperando datos futuros (1 item)

| Item | Dependencia | Fecha estimada |
|---|---|---|
| WYCKOFF-5BX | `n_confirmed_H20_complete >= 550` post-2026-10-02 | ~2027 |

No hay accion. Solo esperar a que los datos se acumulen. El
sistema sigue produciendo runs diarios y contando.

---

## 3. Esperando decision de producto (1 item)

| Item | Decision pendiente |
|---|---|
| IWM-HOLDINGS-EXPANSION | Microcaps: ¿entran al sistema holdings-based o no? El top-20 cubre 6.8% del peso. Para 50% harian falta ~700 tickers. El sistema holdings-based no gana nada con microcaps. |

Decision del usuario. No depende del auditor. Si la respuesta es
"no", se cierra WONT_FIX con razon. Si es "si", es un trabajo de
descarga + validacion de 6 pasos (protocolo en REGISTRO).

---

## 4. Esperando item interno (2 items)

| Item | Depende de | Notas |
|---|---|---|
| TOP20-EU-INDICES | EU-HOLDINGS-DOWNLOAD-CAP (IN_PROGRESS) | Cuando el pipeline descargue los 112 tickers EU, se puede disenar selector por peso para indices EU. |
| TOP20-FALTANTES-DATOS | SOW-HOLDOUT-IMMUTABLE (parcial) + DDOG/MRNA (autonomia) | 35 tickers esperan firma SOW-HOLDOUT. DDOG/MRNA en proximo run. 3 IPO recientes sin accion. |

---

## 5. Lo que NO se hace hasta que algo cambie

- No se redactan mas versiones de contratos por si acaso.
- No se preparan respuestas hipoteticas a dictamenes no recibidos.
- No se implementan tests T6-T10 de SOW-HOLDOUT hasta firma v1.1
  (definidos en §9.2 del contrato, el auditor no los exige escritos
  para firmar).
- No se toca SOW-BEAR v1.10.1 hasta que el usuario lo pida.
- No se generan mas documentos de gobernanza para items que ya
  tienen su version lista.

---

## 6. Lo que SI se puede hacer (operativo, sin bloqueo)

| Trabajo | Tipo |
|---|---|
| Verificar proximo run scheduled (~04-06Z) | Operativo |
| Bugs que aparezcan en produccion | Operativo |
| Resource leaks, race conditions (ROI bajo) | Deuda menor |
| Revisar HANDOFF / bitacora al cierre de sesion | Documental |
| Enviar v1.1 + doc 74 al auditor | Accion del usuario |

---

## 7. Como se actualiza este documento

Cuando un item cambie de estado:

1. Actualizar `REGISTRO.yaml` (fuente canonica).
2. Actualizar la fila correspondiente aqui.
3. Si el item se cierra, retirarlo de la tabla.

Si este documento y REGISTRO discrepan, manda REGISTRO.
