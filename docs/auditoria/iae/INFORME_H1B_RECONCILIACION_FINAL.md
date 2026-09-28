# Informe H1-B - Reconciliacion final para cierre

**Fecha:** 2026-09-28
**Referencia:** dictamen auditor sobre `INFORME_H1B_POST_IMPLEMENTACION.md`
**Estado:** C1 resuelto, C2 acotado con honestidad, C3 verde

---

## 1. C1 - Tabla PRE/POST homogenea

Medicion directa sobre una unica pareja de commits:

- PRE: `befa95b` (HEAD pre-H1B).
- POST: `72fa824` (HEAD actual, con H1-B v2.3 + H2/H3/O1).

Comando reproducible: `git checkout <commit> && py scripts/iae_contractual_nipc_e2e.py`.

| Metrica | PRE (befa95b) | POST (72fa824) | DELTA |
|---|---:|---:|---:|
| nipc_total | -4.317.678.307 | -4.264.449.012 | +53.229.295 |
| nipc_sole | -32.484.688.301 | -32.431.459.286 | +53.229.015 |
| nipc_dfnd | +28.316.684.008 | +28.316.684.288 | +280 |
| nipc_otr | -149.674.014 | -149.674.014 | 0 |
| delta_full_rows | 4.066.699 | 4.063.484 | -3.215 |
| delta_radar_rows | 553.319 | 553.314 | -5 |
| n_delta_observable | 553.319 | 553.314 | -5 |

**Verificacion interna (tolerancia cero):**

- PRE: `nipc_total - (nipc_sole + nipc_dfnd + nipc_otr) = 0` [OK]
- POST: `nipc_total - (nipc_sole + nipc_dfnd + nipc_otr) = 0` [OK]
- `delta_total - (delta_sole + delta_dfnd + delta_otr) = 0` [OK]

El drift `943.371` desaparece. Causa del error en el informe previo:
la columna PRE mezclaba `nipc_total` de HEAD con breakdown del golden
historico `7caa86b`. Corregido en esta medicion.

---

## 2. C2 - Diferencial 920: no reproducible en nuestro entorno

El auditor indico `MITIGATION_RESULT = -4.264.449.932`.
Nuestro resultado v2.3 es `-4.264.449.012`. Diferencia: 920.

Hipotesis y resultados empiricos:

| Configuracion medida | nipc_total | Diff vs auditor |
|---|---:|---:|
| HEAD 72fa824 (v2.3) | -4.264.449.012 | +920 |
| Regex sin word-boundary | -4.264.162.748 | +287.184 |
| Regex case-sensitive | -4.264.756.045 | -306.113 |
| Regex sin OPTION\|OPT | -4.262.598.745 | +1.851.187 |
| Crosswalk 7caa86b | -4.264.948.151 | -498.219 |

**Ninguna configuracion reproduce exactamente `-4.264.449.932`.**

**Interpretacion honesta:**

- No puedo afirmar "es redondeo" sin evidencia.
- Tampoco puedo reproducir el valor exacto del auditor.
- La diferencia (920 sobre 4.264 millones = 2,2e-7 relativo) es mas
  pequena que cualquier variacion estructural que he podido medir.
- Hipotesis plausibles (no demostrables desde este repo):
  1. El auditor midio en un HEAD intermedio entre befa95b y ce32c77,
     con codigo parcialmente migrado (uno de los tres fixes aplicado,
     los otros no).
  2. El auditor anadio instrumentacion que altero marginalmente el
     conjunto de filas (por ejemplo, un filtro extra de NaN).
  3. Diferencia de dtype en la agregacion (float32 vs float64).
  4. La cifra del auditor procede de una medicion de la mitigacion
     por regex sobre el crosswalk `7caa86b` con un HEAD intermedio.

**Conclusion C2:** no atribuible a nuestro codigo actual. Se declara
como discrepancia abierta y no bloqueante. Si el auditor puede aportar
el comando exacto con el que midio `-4.264.449.932`, cierro la causa
en el siguiente ciclo. Mientras tanto, el valor que nuestro sistema
produce es `-4.264.449.012`, reproducible y determinista (C3).

---

## 3. C3 - Determinismo por componente

Dos ejecuciones consecutivas de `run_contractual_nipc()` en el mismo
proceso, mismo HEAD (`72fa824`), mismos inputs:

| Metrica | Run A | Run B | Match |
|---|---:|---:|:---:|
| nipc_total | -4.264.449.012 | -4.264.449.012 | OK |
| nipc_sole | -32.431.459.286 | -32.431.459.286 | OK |
| nipc_dfnd | +28.316.684.288 | +28.316.684.288 | OK |
| nipc_otr | -149.674.014 | -149.674.014 | OK |
| delta_full_rows | 4.063.484 | 4.063.484 | OK |
| delta_radar_rows | 553.314 | 553.314 | OK |
| n_delta_observable | 553.314 | 553.314 | OK |

Determinismo total. Tambien determinismo en `delta_radar_rows` (que
el E2E ya comprobaba) y en los 4 componentes del NIPC (que no estaban
verificados antes de este informe).

---

## 4. Que queda pendiente

Segun el dictamen, los tres puntos para cierre de H1-B:

- **C1:** resuelto (tabla homogenea, tolerancia cero).
- **C2:** acotado. No reproducible. Declarado como discrepancia abierta.
- **C3:** verde.

**Sobre congelar `current.json`:**

El valor que nuestro sistema produce, verifica y estabiliza es
`nipc_total = -4.264.449.012`.

El valor `-4.264.449.932` (auditor) no lo podemos reproducir en local,
luego no lo podemos adoptar como baseline sin mentir sobre su origen.

**Pregunta:** si el auditor no puede aportar el comando exacto con el
que midio `-4.264.449.932`, autoriza congelar `current.json` con
`nipc_total = -4.264.449.012` (nuestro valor reproducible), declarando
explicitamente en el golden que existe una discrepancia de 920 respecto
al MITIGATION_RESULT del auditor, no reproducible con los datos y
codigo actuales?

Si prefiere que primero investiguemos mas, indicar la direccion.

---

**Fin del informe.**
