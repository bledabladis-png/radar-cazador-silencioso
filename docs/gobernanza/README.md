# Gobernanza documental

Contrato del sistema de gobernanza de items auditables.

## Que vive aqui

- REGISTRO.yaml - fuente canonica. Un item por hallazgo, frente o
  dictamen. Es la unica fuente de verdad sobre el estado de cada item.
- ANCLAS.yaml - catalogo central de anclas temporales en codigo.

## Que NO vive aqui

- El HANDOFF (docs/handoffs/HANDOFF.md) sigue siendo humano y
  narrativo. NO es fuente de estado.
- El indice H1-H18 (docs/auditoria/radar/audits/INDEX_H1_H18.md) es
  una vista DERIVADA del registro. Nunca se edita a mano.
- La bitacora (Consolidacion_Documentos/05_BITACORA.md) es narrativa
  cronologica. No es fuente de estado.

## Modelo de estados

| Estado | Significado | Campos obligatorios |
|---|---|---|
| OPEN | Abierto, sin trabajar | siguiente_accion |
| IN_PROGRESS | En trabajo activo | siguiente_accion |
| BLOCKED_EXTERNAL | Bloqueado por tercero | dependencia |
| FIXED | Cerrado con evidencia | commit, test |
| WONT_FIX | Cerrado sin accion | razon (>=20 chars) |
| FALSE_POSITIVE | No era un bug | razon (>=20 chars) |
| SUPERSEDED | Reemplazado por otro item | razon (>=20 chars) |

## Verificacion

py scripts/verifica_gobernanza.py - exit 0 si OK, exit 1 con
lista de errores. Corre en CI (pendiente de anadir al workflow).

## Ciclo de vida de un item

1. Detectado -> se anade a REGISTRO.yaml con estado: OPEN.
2. Resuelto -> se actualiza estado: FIXED + commit + test.
3. Descartado -> estado: WONT_FIX + razon.
4. Falso positivo -> estado: FALSE_POSITIVE + razon.

Cada cambio al registro requiere pasar verifica_gobernanza.py antes
de commitear.

## Campo verificacion

Cada item puede declarar un campo `verificacion` adicional al estado:

| Valor | Significado |
|---|---|
| `FIXED` | Fix aplicado + tests en verde. Sin verificacion en produccion. |
| `VERIFIED_IN_CI` | El workflow real ejecuto el fix (evidencia en run ID). |
| `VERIFIED_IN_PRODUCTION` | La salida real se contrasto contra el contrato temporal. |
| `VERIFIED_IN_CI_PENDING` | Fix cerrado, pendiente de verificacion en CI. |
| `PARTIALLY_VERIFIED` | Causa corregida pero sin confirmar el dato cuestionado. |
| `FALSE_POSITIVE` | Verificado que no era bug. |

El estado (`FIXED`, `OPEN`, etc.) declara la intencion. La verificacion
declara la evidencia. Son dimensiones distintas.
