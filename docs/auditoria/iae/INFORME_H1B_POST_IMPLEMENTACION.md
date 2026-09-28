# Informe H1-B post-implementacion

**Fecha:** 2026-09-28
**Referencia:** diseno aprobado `DISENO_H1B.md` v2 (commit 321c385)
**Implementacion:** commits 54d8435 + ce32c77 (local, no pusheado)
**Estado:** pendiente de validacion del auditor

---

## 1. Resultado empirico

Cadena E2E contractual ejecutada sobre el par Q4 2025 -> Q1 2026
(`scripts/iae_contractual_nipc_e2e.py`):

| Metrica | Pre-H1B (HEAD befa95b) | Post-H1B v2.3 (HEAD ce32c77) | Delta |
|---|---:|---:|---:|
| `nipc_total` | -4.317.678.307 | **-4.264.449.012** | +53.229.295 |
| `nipc_sole` | -32.484.713.330 | -32.431.459.286 | +53.254.044 |
| `nipc_dfnd` | +28.317.652.408 | +28.316.684.288 | -968.120 |
| `nipc_otr` | -149.674.014 | -149.674.014 | 0 |
| `delta_radar_rows` | 553.321 | 553.314 | -7 |
| `evidence_class` | CONTRACTUAL | CONTRACTUAL | - |
| determinismo | - | PASS | - |

Comparacion con el dictamen del auditor:

| Valor | Origen | Diferencia vs nuestro |
|---|---:|---:|
| -4.317.678.307 | PRE_H1B_OBSERVATION (medido por nosotros) | 0 |
| -4.264.449.932 | **Declarado** por el auditor, sin comando ni HEAD | 920 |
| -4.264.449.012 | **Nuestro H1-B v2.3** (medido y reproducible) | - |

**Nota metodologica (post-dictamen 2026-09-28).** El valor
`-4.264.449.932` **no es un golden medido por este repositorio**. Es un
valor declarado por el auditor en su dictamen inicial, sin publicar el
comando ni el HEAD. No ha sido posible reproducirlo con ninguna
configuracion. Ver `INFORME_H1B_RECONCILIACION_FINAL.md` seccion C2
para el detalle completo.

La afirmacion original "coincide dentro del margen de redondeo" se
retira: presuponia una metodologia del auditor no documentada. Lo que
si se puede afirmar: nuestro sistema produce `-4.264.449.012`,
determinista y reproducible.

---

## 2. Hallazgo estructural: el filtro primario es defensa, no mejora

Durante la implementacion medimos el efecto de cada componente por
separado. El resultado es contundente:

| Version | Filtro aplicado | nipc_total | delta_radar_rows |
|---|---|---:|---:|
| v2.0 | solo regex TITLEOFCLASS | -4.264.449.012 | 553.314 |
| v2.2 | regex + excluye OPTION+UNRESOLVED | -4.222.124.799 | 553.271 |
| v2.3 | regex + excluye SOLO OPTION | **-4.264.449.012** | **553.314** |

**v2.3 = v2.0.** El filtro primario por Official List NO anade nada
sobre la regex en el dataset actual.

**Razon empirica:** de los 23 CUSIPs marcados OPTION por la Official
List Q4 2025, **los 23 tambien tienen TITLEOFCLASS con CALL/PUT/OPT**
en los filings (regex=True). En Q1 2026 hay ~10 CUSIPs OPTION en la
lista que la regex no matchea, pero **no aparecen en INFOTABLE** (no
hay filings que los reporten), luego no contribuian al delta.

**Consecuencia arquitectonica:** el filtro primario por Official List
es **defensa en profundidad** para el caso adversarial futuro
(PUTCALL=NULL + TITLEOFCLASS=COM + CUSIP de opcion confirmada por la
lista). No es una mejora del NIPC observable actual. Esto justifica
mantenerlo pero degrada su prioridad: no bloquea el cierre de H1-B.

---

## 3. Hallazgo critico: la regla literal del auditor produce falsos positivos

La regla del diseno v2 era: "no encontrado en Official List -> UNRESOLVED
-> excluir". La implementacion literal (v2.2) produjo un falso positivo
material:

**CUSIPs afectados en Q4 2025:**
- 11.550 CUSIPs con `instrument_type=UNKNOWN` excluidos.
- 11 OPTION confirmadas excluidas.

**CUSIPs afectados en Q1 2026:**
- 11.487 CUSIPs UNKNOWN excluidos.
- 33 OPTION confirmadas excluidas.

**Tickers de equity legítima afectados:** AMCR (-36,1M), LRCX (-6,2M).
Ambos son equity real (AMCR es emisor irlandes con CUSIP `G0250X149`,
LRCX con CUSIPs `512807108`/`512807306`) que **no aparece en la Official
List** por razones de formato/registro, no por ser opciones.

**Atribucion del NIPC eliminado por v2.2:**
- EQUITY (238 tickers): 0
- UNRESOLVED (3 tickers): -42.324.213

**Diagnostico:** la regla literal "no encontrado -> excluir" aplica
"no imputar" solo en una direccion (no imputar equity), pero introduce
imputacion en la otra: convierte "sin evidencia" en "evidencia negativa",
lo cual es falso positivo.

**Correccion v2.3:** solo se excluye OPTION confirmada. UNRESOLVED se
mantiene. Razonamiento:
- OPTION confirmada: evidencia positiva -> excluir.
- EQUITY confirmada: evidencia positiva -> incluir.
- Sin evidencia: no decidir. No imputar ni equity ni opcion.

Esta interpretacion es mas conservadora que la del diseno v2 y
preserva el principio "no imputar" en ambas direcciones.

---

## 4. Verificacion empirica del cambio v2.3

**Tests:** 94 passed en los 3 ficheros afectados. Suite completa 2226
passed + 2 skipped + 0 failed. pyflakes limpio, compileall OK.

**Regresion confirmada:** el unico cambio observable vs v2.0 es el
delta_radar_rows (553.314 vs 553.314, identico) y el nipc_total
(-4.264.449.012, identico). Es decir, v2.3 es funcionalmente equivalente
a v2.0 y preserva la correccion de v2.3 sobre los falsos positivos de
v2.2.

**Alcance del determinismo (nota post-dictamen 2026-09-28).** El
determinismo probado aplica al dominio verificado: mismo proceso, mismo
HEAD, mismos inputs, misma ejecucion. No se interpreta como determinismo
absoluto en cualquier entorno. Pendiente de fijar para reproducibilidad
completa: hash/version de inputs, version de Python, versiones de
dependencias, plataforma.

**Caso adversarial pendiente de verificar con datos reales:**
`PUTCALL=NULL` + `TITLEOFCLASS='COM'` + CUSIP de opcion que la regex
no captura. Buscamos candidatos en INFOTABLE Q4/Q1 y no encontramos
ninguno en el dataset actual. El filtro primario por Official List
queda implementado como defensa para cuando aparezca.

---

## 5. Condiciones del auditor

| Condicion | Estado |
|---|---|
| Suite completa sin fallos | PASS (2226 + 2 skipped) |
| E2E contractual ejecutado | PASS (resultado arriba) |
| Reconciliacion NIPC documentada | PASS (seccion 1) |
| Casos EQUITY / OPTION / UNRESOLVED | PASS (tests unitarios + integracion) |
| Caso adversarial PUTCALL=NULL+TOC=COM | No encontrado en datos reales; cubierto por tests |
| Temporalidad (fuente por periodo) | PASS (Q4 usa lista Q4, Q1 usa lista Q1) |
| Compatibilidad eligible_for_nipc | PASS (campo nuevo, eligible intacto) |

**Condicion parcialmente cumplida:** la condicion 5 (caso adversarial
real) no se ha podido verificar con datos reales porque no hay ningun
caso en el dataset actual. Se declara como limitacion de la
verificacion empirica. El codigo esta preparado.

---

## 6. Estado del fix

**H1-B: implementado en local. Pendiente de validacion del auditor.**

No se ha pusheado. Los commits locales son:
- `54d8435` - implementacion inicial v2 (con bug de falsos positivos).
- `ce32c77` - correccion v2.3 (filtro solo OPTION + UNRESOLVED mantenido).

**No se ha congelado `current.json`.** El golden sigue con
`current.json` bloqueado.

**Valor de referencia:** `-4.264.449.012` es POST_H1B_OBSERVATION, no
baseline contractual.

---

## 7. Preguntas al auditor

1. **Acepta v2.3** como interpretacion correcta de "no imputar"
   (no imputar equity NI imputar opcion), o prefiere mantener la regla
   literal (UNRESOLVED -> excluir) aun con los falsos positivos sobre
   AMCR/LRCX documentados?

2. **Acepta la conclusion de que el filtro primario por Official List
   es defensa en profundidad** (no mejora el NIPC observable actual)
   y por tanto H1-B se puede cerrar en su estado actual?

3. **Caso adversarial:** confirma que la imposibilidad de verificar
   empiricamente el caso PUTCALL=NULL + TOC=COM + CUSIP opcion es
   aceptable como limitacion declarada (cubierta por tests unitarios),
   o exige busqueda adicional en fuentes externas?

4. **Si acepta v2.3**, autoriza la congelacion de `current.json` con
   `nipc_total = -4.264.449.012` como baseline post-H1-B?

---

**Fin del informe.**
