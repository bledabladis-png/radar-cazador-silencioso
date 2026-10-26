# Doc 73 - Respuesta al dictamen H-04 (structure_ratio)

**Fichero:** docs/auditoria/wyckoff/73_respuesta_dictamen_h04_2026-10-26.md
**Estado:** para revision externa.
**Dictamen respondido:** "Dictamen externo H-04 - structure_ratio",
recibido 2026-10-26 tras envio de doc 70-v2c.
**Predecesores:** doc 69 (pre-registro + §12 + §13), doc 70-v2c
(solicitud de reconsideracion).

---

## 0. Resumen ejecutivo

**Veredicto del auditor aceptado sin reservas:** NO APROBADO para
publicar intervalos de confianza como respaldo inferencial validado.
`DICT3-H04-INTERVALS` permanece **OPEN**. `structure_ratio` sigue
como metrica **descriptiva**, no inferencial ni predictiva.

Los 5 hallazgos (P0-H04-01..03, P1-H04-04..05) son aceptados. Este
documento verifica que el sistema **no incumple hoy** ninguna de las
restricciones del dictamen (no publica IC95 en el reporte, no usa
`structure_ratio` para clasificar sectores individualmente), y
registra la deuda nueva: `DICT3-H04-INFERENCE-V2` (nuevo protocolo
inferencial con validacion independiente).

No se reinterpreta §9, no se modifica el pre-registro congelado, no
se re-ejecuta el procedimiento buscando otro resultado.

---

## 1. Aceptacion de los hallazgos

### P0-H04-01 - Bootstrap mal especificado

**Aceptado.** El pre-registro §2-§10 no especifica con precision
suficiente:

- Unidad remuestreada en cada dimension (dias vs sectores vs medias
  anuales por sector).
- Identidad del sector a traves de anos remuestreados.
- Dependencia temporal dentro de cada ano.
- Tratamiento de anos parciales, sectores ausentes, celdas no validas.

El pseudocodigo `ratio[y, s]` es ambiguo entre "media anual por sector"
y "observacion diaria". El pigeonhole bootstrap clasico (Owen) no se
reduce a dos etapas secuenciales sin justificar que preserva las
dependencias correctas.

**No afirmo que el codigo real este mal implementado.** Afirmo, con el
auditor, que el protocolo congelado no aporta informacion suficiente
para validar la implementacion. La validacion es trabajo pendiente,
no un fallo demostrado.

### P0-H04-02 - Cinco anos no son base solida

**Aceptado.** Cinco cohortes anuales, tres completas y dos parciales.
11 sectores correlacionados con cobertura desigual. El propio doc 69
reconoce el problema de pocos clusters. 2000 repeticiones bootstrap
reducen error numerico, no crean anos independientes.

La serie anual muestra variacion estructural (0.784 -> 0.212 sobre base
target_weight). Tratar esos anos como intercambiables requiere justificar
que su poblacion temporal tiene interpretacion comun. No se justifica
en el pre-registro.

### P0-H04-03 - Criterio §9 no discrimina

**Aceptado.** `structure_ratio` en [0,1]. Media no negativa. Un IC que
excluye 0 es el resultado esperable y no distingue metrica informativa
de metrica con poca utilidad analitica. §12.4 lo reconoce. El analisis
post hoc con 0.5 como referencia (que el dictamen cita como
"intuitivamente interpretable") tampoco se acepta: no es pre-registrado,
no esta validado como punto de neutralidad del detector, y observar que
un IC contiene 0.5 no demuestra equivalencia.

**Declaracion formal:** el criterio §9 de doc 69 se declara
**metodologicamente no concluyente**. No se reinterpreta como superado.
No se reemplaza retroactivamente por el criterio 0.5 para cerrar el
frente.

### P1-H04-04 - ALL annual no reconciliado con medias anuales

**Aceptado.** La media aritmetica de las 5 cifras anuales (0.784 +
0.824 + 0.572 + 0.234 + 0.212) / 5 = 0.5252 no coincide con `ALL
annual mean = 0.510`. Mismo patron en base top-20 (0.5270 vs 0.498).
El estimador global probablemente pondera por observaciones diarias
validas, no por anos; el pre-registro no lo declara ni demuestra que
el bootstrap reproduzca la misma ponderacion.

**Accion requerida:** documentar explicitamente si el promedio global
pondera por ano o por dia. Hasta entonces, `ALL annual` no es auditable
de forma independiente.

### P1-H04-05 - Anchura 2.4x no valida multiway

**Aceptado.** Que dos metodos (naive iid y multiway) produzcan
intervalos distintos no demuestra que el mas ancho sea correcto ni que
alcance cobertura nominal 95%. La justificacion de un metodo para datos
dependientes debe apoyarse en sus propiedades y en la coherencia con la
estructura de dependencia, no en el tamano del intervalo.

La conclusion de que el bootstrap naive es insuficiente se mantiene.
La conclusion de que multiway queda validado por ser mas ancho se
retira.

---

## 2. Verificacion de no incumplimiento actual

El dictamen prohibe:

| Prohibicion | Verificacion |
|---|---|
| Publicar IC95 como evidencia inferencial validada | Reporte diario: 0 hits `ci_lo|ci_hi|intervalo|IC95|bootstrap` (grep sobre `src/report/`). |
| Interpretar `structure_ratio` como senal predictiva | No hay uso predictivo documentado. Solo descriptivo en `sector_context.py:24`. |
| Seleccionar/clasificar sectores con solo esta metrica | `structure_positive` retirada en sesion 10. `sector_regime_matrix` v2 publica `structure_ratio` continua + `n_estructuras`, no booleana. No se usa para ranking. |

**Conclusion:** el sistema actual cumple las restricciones del dictamen
sin necesidad de cambios de codigo. Los IC se generan solo en
`outputs/audit/h04_intervals.csv` (gitignored, no consumido por el
reporte ni por ningun writer registrado).

---

## 3. Estado de gobernanza resultante

- `DICT3-H04-INTERVALS`: **OPEN** (sin cambio de estado).
- `structure_ratio`: **descriptiva**, no inferencial ni predictiva.
- Publicacion de IC95 en reporte: **no autorizada**.
- Doc 69 §2-§10: **congelado**, con nota explicita de que §9 no
  discrimino por construccion.
- Nuevo item: `DICT3-H04-INFERENCE-V2` (OPEN, deuda de diseno).

---

## 4. Plan para nueva investigacion inferencial

El dictamen §5 exige nuevo protocolo con 4 decisiones. Se registra como
`DICT3-H04-INFERENCE-V2`, sin codigo hasta aprobacion del diseno.

**Decision 1 - Estimando.**
Tres objetivos distintos, no excluyentes pero si separables:
(a) describir el promedio de la ventana 2022-2026 observada;
(b) estimar el promedio de una poblacion de anos (si se define cual);
(c) inferir sobre regimenes futuros.
El pre-registro actual mezcla los tres. El nuevo debe elegir uno
explicitamente y justificar por que.

**Decision 2 - Remuestreo.**
Correspondencia entre unidad de inferencia y procedimiento. Debe
resolver:
- Unidad remuestreada en cada dimension (dia, sector, ano, media anual
  por sector).
- Identidad preservada del sector.
- Dependencia temporal dentro de cada ano.
- Tratamiento de anos parciales y sectores ausentes.
- Ponderacion (por dia vs por ano) coherente con el estimando.
Validacion: reproducir el pigeonhole bootstrap de Owen con todas las
decisiones explicitas, o justificar una alternativa.

**Decision 3 - Hipotesis y criterio.**
Un criterio que discrimine. Posibilidades:
- Contraste de un valor de referencia (0.5) con justificacion previa de
  por que es neutral y que diferencia es material.
- Test de equivalencia (TOST) o intervalo de equivalencia pre-registrado.
- Criterio sobre subgrupos con n_estructuras >= umbral.
No basta con "IC contiene/excluye X". El criterio debe distinguir
evidencia insuficiente de evidencia de equivalencia.

**Decision 4 - Validacion independiente.**
Los resultados de §12-§13 ya estan observados. Cualquier protocolo nuevo
basado en ellos es **exploratorio**, no confirmatorio. La validacion
confirmatoria requiere datos futuros o muestra independiente definida
antes de mirar los resultados. Propuesta concreta: usar 2027 como
holdout prospectivo, con el protocolo congelado antes de 2027-01-01.

---

## 5. Lo que este documento NO hace

- No cierra `DICT3-H04-INTERVALS`.
- No modifica el pre-registro §2-§10 de doc 69.
- No reinterpreta §9 como superado.
- No reemplaza §9 por 0.5 como criterio post hoc.
- No re-ejecuta `h04_intervals.py` con otra semilla o metodo buscando
  resultado mas favorable.
- No retira `structure_ratio` descriptiva del reporte (autorizada).
- No implementa el nuevo protocolo (`DICT3-H04-INFERENCE-V2`, OPEN).

---

## 6. Impacto en otros frentes

`DICT3-TOP20-COBERTURA`: sin cambio. El dictamen H-04 separa
explicitamente la aprobacion del selector A.2c de la validez
inferencial de `structure_ratio`. Las 4 acciones ejecutables del
dictamen 2026-10-26 estan cerradas (b3793d7c, 2d84ae82, adfbb69b,
f967985d); P1-H05 queda documentada aqui y en doc 70-v2d.

---

**Preparado 2026-10-26. Pendiente de envio al auditor externo.**