# PROVENIENCIA DE PARAMETROS - Modulo Wyckoff v1

**Documento normativo complementario al contrato v1.**
**Fecha:** 2026-10-02.
**Referencia:** dictamen auditor externo 2026-10-02, §12 (gobernanza).

---

## 1. Proposito

Registrar, para cada parametro del modulo Wyckoff v1:

    parametro | valor | unidad | rationale | fuente | fecha | validacion

**Regla dura:** si un parametro no tiene fuente verificable, se declara
`ND` (no documentado) y **no se declara como validado**. La honestidad
del registro es mas importante que la apariencia de rigor.

---

## 2. Categorias de fuente

- **LIT**: literatura (Wyckoff original, StockCharts, textos cuantitativos).
- **CONV**: convencion de la industria (ej. MA200 como "largo plazo").
- **HEUR**: heuristica propia sin calibracion formal.
- **CAL**: calibrado sobre datos propios con metodologia descrita.
- **LEG**: herencia del modulo legacy sin justificacion conocida.
- **ND**: no documentado. Se requiere decision futura.

---

## 3. Tabla de proveniencia v1

### 3.1. Ventanas temporales

| Parametro | Valor | Unidad | Rationale | Fuente | Fecha | Validacion |
|---|---:|---|---|---|---|---|
| MA rapida (trend) | 50 | sesiones | Convencion de analisis tecnico | CONV | ND | ND |
| MA lenta (trend) | 200 | sesiones | Definicion clasica de tendencia largo plazo | CONV | ND | ND |
| ATR (compresion) | 20 | sesiones | Ventana estandar para ATR | CONV | ND | ND |
| Volumen (z-score) | 60 | sesiones | Ventana trimestral bursatil aprox. | HEUR | ND | ND |
| Volumen (min_periods) | 20 | sesiones | Cubre 1 mes bursatil de huecos | HEUR | ND | ND |
| Precio (movimiento) | 20 | sesiones | Ventana mensual estandar | CONV | ND | ND |
| Estabilidad (MAD) | 10 | sesiones | Alineado con persistencia_10d ya usada | LEG | ND | ND |

**Nota sobre MA200:** el uso de 200 sesiones como frontera entre
corto y largo plazo es convencion ampliamente adoptada en analisis
tecnico. No requiere calibracion propia, pero si registro.

**Nota sobre Volumen 60/20:** el legacy usaba 60/20. Se conserva para
no romper comparabilidad. No hay estudio que valide estos valores.

### 3.2. Pesos de composicion

| Parametro | Valor | Rationale | Fuente | Fecha | Validacion |
|---|---:|---|---|---|---|
| struct: trend | 0.60 | Tendencia es primaria en Wyckoff | LIT | ND | ND |
| struct: compression | 0.40 | Compresion es secundaria | LIT | ND | ND |
| tact: volume | 0.50 | Simetria entre los dos ejes tacticos | HEUR | ND | ND |
| tact: effort | 0.50 | Simetria entre los dos ejes tacticos | HEUR | ND | ND |
| combined: struct | 0.70 | Fase es propiedad estructural | HEUR | ND | ND |
| combined: tact | 0.30 | Tactico es confirmatorio | HEUR | ND | ND |

**Observacion de auditoria:** los pesos actuales (0.60/0.40, 0.50/0.50,
0.70/0.30) son heuristica heredada del legacy. No hay estudio de
sensibilidad. Se conservan por continuidad, NO porque esten validados.

**Consecuencia:** un cambio futuro de pesos debe tratarse como cambio
de comportamiento del sistema, no como refactor. Cualquier ajuste debe
documentarse con estudio de impacto.

### 3.3. Umbrales de clasificacion

| Parametro | Valor | Rationale | Fuente | Fecha | Validacion |
|---|---:|---|---|---|---|
| t_norm fuerte | 0.30 | Frontera empirica en tanh; mas alla = sesgo claro | HEUR | ND | ND |
| t_norm debil | 0.30 (simetrico) | Simetria alcista/bajista | HEUR | ND | ND |
| c_norm compresion | 0.30 | Compresion notable | HEUR | ND | ND |
| combined MARKUP | 0.30 | Alineado con t_norm fuerte | HEUR | ND | ND |
| combined DISTRIBUTION | -0.10 | Umbral asimetrico (a validar) | HEUR | ND | ND |
| combined MARKDOWN | -0.30 | Simetrico a MARKUP | HEUR | ND | ND |

**Observacion:** el umbral `combined < -0.10` para DISTRIBUTION es
asimetrico respecto a MARKUP (0.30). Esta eleccion debe validarse en
Fase 5 o reemplazarse por un criterio estructural puro.

### 3.4. Deteccion de eventos

| Parametro | Valor | Rationale | Fuente | Fecha | Validacion |
|---|---:|---|---|---|---|
| Spring: volumen | 1.5x MA20 | Sobreesfuerzo respecto a media | HEUR | ND | ND |
| SOS: volumen | 1.0x MA20 | Confirmacion basica | HEUR | ND | ND |
| Ventana de max/min | 20 | Ventana mensual | CONV | ND | ND |

**Observacion:** la asimetria 1.5x vs 1.0x entre spring y SOS no tiene
justificacion documentada. Se conserva por continuidad con legacy.

### 3.5. Constantes numericas internas

| Constante | Valor | Origen | Fuente | Validacion |
|---|---:|---|---|---|
| MAD a sigma | 1.4826 | Factor de consistencia MAD->sigma bajo normalidad | LIT | Verificado |
| Epsilon (1e-9) | 1e-9 | Evita division por cero | HEUR | Aceptado |
| Clip z-score | +/-5 | Rango razonable para series financieras | HEUR | Aceptado |
| ffill limit | 3 | Cubre puentes de festivos sin enmascarar huecos largos | HEUR | ND |

**Nota:** `1.4826` esta matematicamente justificado (ver formula en
docs externos de estadistica robusta). Las demas constantes son
elecciones operativas sin calibracion.

### 3.6. Parametros nuevos (a definir en Fase 5)

| Parametro | Valor propuesto | Rationale | Estado |
|---|---:|---|---|
| Ventana estabilidad MAD | 10 | Alineado con persistencia_10d existente | PROPUESTO |
| K de estabilidad | por calibrar | Escala que hace util el rango (-1, 1) | PENDIENTE |
| ATR base/rango | 60 | Mediana trimestral para detectar base | PROPUESTO |
| N sesiones para confirmar spring reciente | 5 | Proximo a la semana bursatil | PROPUESTO |
| Umbral DISTRIBUTION asimetrico | -0.10 | Justificar o reemplazar por criterio estructural | PENDIENTE |


---

## 3.7. Parametros v1.1 (precedente estructural)

Introducidos en el contrato v1.1 (ver 04_revision_contrato_v1_1.md).

| Parametro | Valor | Rationale | Fuente | Fecha | Validacion |
|---|---:|---|---|---|---|
| PRECEDENT_WINDOW (N) | 60 | Trimestre bursatil aprox. Alineado con volumen z-score | HEUR | 2026-10-02 | PROPUESTO |
| PREC_STRUCT_WEAK | -0.20 | Umbral de "debilidad reciente" | HEUR | 2026-10-02 | PROPUESTO |
| PREC_STRUCT_STRONG | 0.30 | Umbral de "fortaleza reciente" (simetrico a STRUCT_STRONG) | HEUR | 2026-10-02 | PROPUESTO |
| PREC_STRUCT_NEG | -0.30 | Umbral de "bajista reciente" (simetrico a STRUCT_WEAK) | HEUR | 2026-10-02 | PROPUESTO |
| STRUCT_BASE_LOW | -0.20 | Banda de base: limite inferior | HEUR | 2026-10-02 | PROPUESTO |
| STRUCT_BASE_HIGH | 0.20 | Banda de base: limite superior | HEUR | 2026-10-02 | PROPUESTO |
| STRUCT_DETERIORO | -0.10 | Umbral para DISTRIBUTION | HEUR | 2026-10-02 | PROPUESTO |

**Observacion:** todos los umbrales de precedente son PROPUESTOS. Se
marcan explicitamente como no validados. Razon: calibracion con dataset
controlado es Fase 5b posterior. La validacion de contrato (I1-I14) no
depende de los valores concretos; la calibracion empirica si.

---

## 3.8. Parametros eliminados de v1.0

| Parametro | Valor v1.0 | Motivo de eliminacion |
|---|---:|---|
| COMBINED_MARKUP | 0.30 | Veto tactico indebido sobre estructura (D2) |
| (MARKUP) c_norm < 0.30 | — | Incompatible con bull market ordenado (D1) |
| (ACCUMULATION) tact > 0 | — | Silencioso por diseno; no debe vetar acumulacion (D3) |


---

## 4. Proceso de registro

Todo cambio de parametro debe:

1. Actualizar esta tabla.
2. Declarar la nueva fuente (LIT/CONV/HEUR/CAL/LEG).
3. Si es CAL, adjuntar metodologia (dataset, metrica, validacion out-of-sample).
4. Registrar fecha del cambio.
5. Actualizar `validation_period`.

**Prohibido:**
- Modificar valores sin actualizar esta tabla.
- Declarar "validado" un parametro que no paso por metodologia CAL.

---

## 5. Prioridad de investigacion futura (post-v1)

Orden de ROI estimado para calibrar con datos:

1. **K de estabilidad.** Sin calibrar, `stability` no discrimina.
2. **Umbral DISTRIBUTION.** Asimetria actual sin justificacion.
3. **Pesos tacticos (0.50/0.50).** Simetria por convencion, no por evidencia.
4. **Spring 1.5x vs SOS 1.0x.** Aceptable pero sin origen.
5. **Ventana de volumen 60 vs 20.** El legacy usa 60; el nombre
   `relative_volume` implicaba 20. Decidir cual es el "correcto".

---

## 6. Vigencia

Esta tabla se regenera cuando:

- Cambia cualquier parametro del modulo.
- Se cierra una calibracion (pasa de PROPUESTO a CAL).
- El auditor externo lo solicita.

**Fecha de creacion:** 2026-10-02.
**Proxima revision:** al cierre de Fase 5.

---

Fin del documento de proveniencia.
