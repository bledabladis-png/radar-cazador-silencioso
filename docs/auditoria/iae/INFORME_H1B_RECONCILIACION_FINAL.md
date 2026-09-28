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

## 2. C2 - Diferencial 920: valor del auditor no reproducible

**Aclaracion previa (critica).** El valor `-4.264.449.932` **no es un
golden medido por este repositorio**. Es un valor **declarado** por el
auditor externo en su dictamen inicial (`AUDITORIA_EXTERNA_2026-09-27.md`,
seccion 3.2). El auditor no publico ni el comando ni el HEAD con el que
lo midio. Todas las referencias a `-4.264.449.932` en los documentos
del proyecto derivan de esa unica declaracion.

**Estado real del numero:**

- Declarado por el auditor: `-4.264.449.932`.
- No reproducido por este repositorio con ninguna configuracion conocida.
- No se dispone del comando ni del HEAD originales.

**Mediciones efectivas realizadas** (todas con codigo y datos de este
repo, HEAD `72fa824`, salvo la indicada):

| Configuracion | nipc_total | Diff vs declaracion (932) |
|---|---:|---:|
| HEAD 72fa824 (v2.3) | -4.264.449.012 | +920 |
| Regex sin word-boundary | -4.264.162.748 | +287.184 |
| Regex case-sensitive | -4.264.756.045 | -306.113 |
| Regex sin OPTION\|OPT | -4.262.598.745 | +1.851.187 |
| Crosswalk 7caa86b (git show) | -4.264.948.151 | -498.219 |

**Lectura correcta:** el +920 que aparece en el informe previo **no es
un diferencial entre dos mediciones reproducibles**. Es la diferencia
entre:
- Nuestra unica medicion reproducible: `-4.264.449.012`.
- Una cifra declarada por el auditor sin metodologia publicada.

Tratar el +920 como una discrepancia tecnica a resolver es un error
metodologico: presupone que el valor del auditor es un golden
verificable. No lo es.

**Lo que se puede afirmar con evidencia:**

- Nuestro sistema produce `-4.264.449.012` (determinista, C3 verde).
- Ese valor es reproducible por cualquier tercero con el comando
  documentado.
- El valor `-4.264.449.932` no es reproducible localmente.
- Cualquier comparacion contra el 932 presupone una metodologia que
  desconocemos.

**Peticion al auditor:** si aporta el comando exacto y el HEAD con el
que midio `-4.264.449.932`, la diferencia se puede atribuir. Sin esa
informacion, la discrepancia se declara como **no atribuible a este
repositorio** y no bloqueante para el cierre de H1-B.

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
