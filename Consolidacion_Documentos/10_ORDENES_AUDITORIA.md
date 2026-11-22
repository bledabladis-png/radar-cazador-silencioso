\# 10 - ÓRDENES DE TRABAJO PARA AUDITORÍA

\*\*Proyecto:\*\* Radar de Rotación Sectorial  

\*\*Herramienta:\*\* AnythingLLM + Qwen 3.5 4B  

\*\*Documento complementario:\*\* 00\_ARRANQUE.md + 09\_AUDITORIA\_TECNICA.md



\## INSTRUCCIÓN GENERAL



Las siguientes órdenes se ejecutan individualmente y en orden.



No avances automáticamente a la siguiente fase.



Cada fase debe finalizar con un informe de resultados, evidencias, limitaciones y estado de cobertura.



La auditoría inicial es de solo lectura.



Se aplican todas las reglas de `00\_ARRANQUE.md` y `09\_AUDITORIA\_TECNICA.md`.



Reglas adicionales:



\- No inventar resultados de comandos.

\- No asumir que un archivo se ha leído porque se haya encontrado su nombre.

\- No analizar código sin consultar sus dependencias relevantes.

\- No repetir investigaciones cerradas sin evidencia nueva.

\- No modificar archivos.

\- No corregir defectos durante el diagnóstico.

\- No ejecutar procesos externos o destructivos.

\- No hacer commits ni push.

\- No ampliar el alcance sin justificarlo.

\- Registrar los archivos y funciones realmente examinados.

\- Diferenciar los hallazgos nuevos de los ya documentados.



Si el contexto de Qwen resulta insuficiente, dividir el trabajo en iteraciones pequeñas y mantener un registro de lo pendiente.



\---



\# FASE 1 — RECONOCIMIENTO, INVENTARIO Y ARQUITECTURA



\## Orden de trabajo



Quiero que realices la primera fase de auditoría técnica del Radar de Rotación Sectorial.



\*\*Objetivo:\*\* comprender la estructura real del repositorio y determinar qué componentes existen, cómo están organizados y cuáles son sus relaciones.



No audites todavía las fórmulas ni propongas modificaciones.



\### 1. Comprobación inicial



Comprueba, si tienes acceso real:



\- Ruta del proyecto.

\- HEAD actual.

\- Rama activa.

\- Estado del working tree.

\- Últimos commits.

\- Documentación de arranque.

\- Estado del sistema.

\- Registro de gobernanza.



Utiliza comandos de consulta que no modifiquen el repositorio.



No regeneres documentos ni ejecutes procesos de escritura durante esta fase.



\### 2. Inventario del proyecto



Identifica:



\- Directorios principales.

\- Módulos Python.

\- Scripts.

\- Puntos de entrada.

\- Configuraciones.

\- Dependencias.

\- Tests.

\- Sistemas de almacenamiento.

\- Generadores de informes.

\- Sistemas de automatización.

\- Documentación técnica.

\- Artefactos relevantes.



No te limites a enumerar archivos. Explica la responsabilidad de cada grupo funcional.



\### 3. Arquitectura



Reconstruye la arquitectura real del proyecto.



Identifica:



\- Módulos principales.

\- Responsabilidades.

\- Dependencias.

\- Relaciones entre módulos.

\- Productores y consumidores de datos.

\- Componentes centrales.

\- Componentes auxiliares.

\- Sistemas transversales.

\- Dependencias externas.



Investiga posibles:



\- Monolitos.

\- Acoplamientos excesivos.

\- Dependencias circulares.

\- Código duplicado.

\- Responsabilidades mezcladas.

\- Configuraciones dispersas.

\- Componentes obsoletos.

\- Código aparentemente desconectado.



No declares un componente desconectado sin consultar quién lo invoca y si su exclusión está documentada.



\### 4. Mapa general



Construye un mapa conceptual del funcionamiento:



FUENTES EXTERNAS

↓

DESCARGA

↓

PROCESAMIENTO

↓

VALIDACIÓN

↓

CÁLCULOS E INDICADORES

↓

AGREGACIÓN

↓

PERSISTENCIA

↓

REPORTE



Sitúa en el mapa los módulos reales que hayas podido identificar.



\### 5. Documentación



Localiza los contratos, decisiones y auditorías anteriores que regulan la arquitectura.



Identifica qué elementos están:



\- Correctamente documentados.

\- Parcialmente documentados.

\- Sin documentación suficiente.

\- Documentados pero aparentemente divergentes del código.



\### 6. Entregable



Genera un informe con:



1\. Estado inicial del repositorio.

2\. Inventario de estructura.

3\. Mapa de arquitectura.

4\. Puntos de entrada.

5\. Mapa preliminar de dependencias.

6\. Componentes centrales y auxiliares.

7\. Posibles monolitos y acoplamientos.

8\. Componentes que requieren investigación adicional.

9\. Falsos positivos descartados.

10\. Limitaciones de acceso.

11\. Archivos examinados.

12\. Cobertura de esta fase.



No propongas refactorizaciones todavía.



\*\*Condición de cierre:\*\* debe existir una comprensión suficiente de la arquitectura para seleccionar los siguientes bloques de auditoría.



\---



\# FASE 2 — AUDITORÍA DE FUENTES Y DESCARGA DE DATOS



\## Orden de trabajo



Realiza una auditoría de la obtención de datos del Radar de Rotación Sectorial.



\*\*Objetivo:\*\* determinar si las fuentes externas están correctamente integradas y si los datos descargados son fiables, completos, coherentes y temporalmente válidos.



La prioridad es detectar datos incorrectos que puedan propagarse silenciosamente al análisis.



\### 1. Inventario de fuentes



Identifica los proveedores y sistemas de obtención de datos.



Para cada fuente, determina:



\- Qué información proporciona.

\- Dónde se configura.

\- Qué función realiza la descarga.

\- Qué endpoint o mecanismo utiliza.

\- Qué parámetros transmite.

\- Qué formato recibe.

\- Qué frecuencia de actualización tiene.

\- Qué módulos consumen sus resultados.

\- Dónde persiste la información.



No supongas que una fuente documentada está operativa.



\### 2. Auditoría de descargas



Investiga:



\- Endpoints obsoletos.

\- Parámetros incorrectos.

\- Errores de autenticación.

\- Límites de API.

\- Rate limiting.

\- Reintentos incorrectos.

\- Timeouts.

\- Circuit breaker.

\- Errores de conexión.

\- Respuestas incompletas.

\- Respuestas aparentemente válidas con contenido incorrecto.

\- Cambios de estructura del proveedor.

\- Parsing defectuoso.

\- Descargas duplicadas.

\- Cachés inconsistentes.



Comprueba el tratamiento de los errores.



Identifica cualquier caso donde un fallo de descarga pueda transformarse en un valor aparentemente válido.



\### 3. Integridad de los datos



Investiga:



\- Valores nulos.

\- Duplicados.

\- Fechas ausentes.

\- Fechas desplazadas.

\- Datos antiguos.

\- Series incompletas.

\- Valores extremos.

\- Unidades incorrectas.

\- Monedas incompatibles.

\- Porcentajes expresados en escalas distintas.

\- Valores convertidos erróneamente.

\- Datos rellenados sin justificación.

\- Datos históricos revisados por el proveedor.



Comprueba especialmente si los valores ausentes se convierten en cero.



\### 4. Integridad temporal



Comprueba:



\- Fecha de observación.

\- Fecha de publicación.

\- Fecha de descarga.

\- Fecha de ejecución.

\- Zona horaria.

\- Calendario de mercado.

\- Festivos.

\- Frecuencia de publicación.

\- Retrasos de proveedores.

\- Revisión histórica de datos.



Contrasta las reglas temporales R1-R6 y las excepciones documentadas en `00\_ARRANQUE.md`.



Comprueba que no se introduce información futura respecto a la fecha histórica analizada.



\### 5. Instrumentos y universos



Revisa:



\- Normalización de tickers mediante `src/instrument\_registry.py`.

\- Coherencia entre instrumentos descargados y analizados.

\- Tratamiento de tickers no válidos.

\- Ticker health.

\- Instrumentos deslistados.

\- IPOs recientes.

\- Holdings.

\- Cobertura del universo elegible.



Comprueba que las exclusiones están justificadas y son coherentes entre módulos.



\### 6. Contratos y persistencia



Contrasta:



\- TemporalContracts.

\- Validation Gate.

\- Output registry.

\- Manifests de integridad.

\- Writers.

\- Receipts.

\- Cobertura.

\- Lag.

\- Estado de los artefactos.



No declares que un dato está validado únicamente porque exista un fichero.



\### 7. Entregable



Para cada proveedor o grupo de fuentes:



1\. Fuente y finalidad.

2\. Archivos implicados.

3\. Funciones de descarga.

4\. Procesamiento.

5\. Validaciones.

6\. Problemas confirmados.

7\. Riesgos.

8\. Tests disponibles.

9\. Hallazgos P0-P3.

10\. Evidencias.

11\. Cobertura revisada.



Finaliza con una tabla:



| Fuente | Descarga | Integridad | Temporalidad | Persistencia | Dictamen |

|---|---|---|---|---|---|



Utiliza CONFIRMADO, PROBABLE, HIPÓTESIS o NO VERIFICADO según la evidencia.



\*\*Condición de cierre:\*\* las fuentes y sus mecanismos de obtención deben quedar documentados y sus riesgos identificados.



\---



\# FASE 3 — AUDITORÍA MATEMÁTICA, ESTADÍSTICA E INDICADORES



\## Orden de trabajo



Realiza una auditoría técnica y matemática de los cálculos del Radar.



\*\*Objetivo:\*\* comprobar que las fórmulas, transformaciones, ventanas temporales y resultados implementados son correctos respecto a las definiciones y contratos del sistema.



Esta fase tiene prioridad máxima junto con la integridad de datos.



\### 1. Inventario de cálculos



Identifica los principales módulos de cálculo e indicadores.



Localiza:



\- Fórmulas.

\- Funciones.

\- Parámetros.

\- Ventanas.

\- Umbrales.

\- Normalizaciones.

\- Pesos.

\- Agregaciones.

\- Variables intermedias.

\- Resultados publicados.



Consulta la definición contractual de cada métrica antes de evaluarla.



\### 2. Auditoría matemática



Para cada cálculo seleccionado:



1\. Identifica los inputs.

2\. Comprueba sus unidades.

3\. Localiza la fórmula teórica o contractual.

4\. Examina la implementación.

5\. Comprueba denominadores.

6\. Comprueba las escalas.

7\. Comprueba ventanas temporales.

8\. Revisa valores ausentes.

9\. Analiza precisión y redondeos.

10\. Contrasta el resultado con una implementación independiente.



Investiga errores en:



\- Retornos.

\- Medias.

\- Medias ponderadas.

\- Ratios.

\- Variaciones porcentuales.

\- Z-scores.

\- Normalizaciones.

\- Correlaciones.

\- Agregaciones sectoriales.

\- Indicadores de amplitud.

\- Métricas de flujo.

\- Métricas de volatilidad.

\- Clasificaciones y puntuaciones.



No presupongas que todas estas métricas utilizan el mismo método. Respeta sus definiciones particulares.



\### 3. Auditoría temporal



Comprueba:



\- Alineación entre series.

\- Número de observaciones.

\- Ventanas bursátiles frente a naturales.

\- Fechas efectivas.

\- Look-ahead bias.

\- Leakage temporal.

\- Datos revisados.

\- Fechas de publicación.

\- Coherencia entre universos.



Contrasta las reglas R1-R6 y la utilización de `resolve\_effective\_date()`.



Investiga si diferentes métricas derivadas de un mismo universo utilizan observaciones temporales incompatibles.



\### 4. Consistencia entre módulos



Busca métricas repetidas en distintos componentes.



Comprueba si utilizan:



\- La misma definición.

\- La misma fórmula.

\- Las mismas unidades.

\- Las mismas ventanas.

\- El mismo universo.

\- Los mismos filtros.

\- Los mismos criterios de validez.



Identifica discrepancias semánticas entre métricas que comparten nombre.



No propongas unificarlas automáticamente si existen contratos que justifican diferencias.



\### 5. Reproducibilidad



Selecciona ejemplos concretos y reconstruye los cálculos.



Compara:



\- Input original.

\- Input transformado.

\- Resultado intermedio.

\- Resultado final.

\- Valor publicado.



Utiliza cálculos manuales o implementaciones independientes cuando sea posible.



No modifiques datos reales.



Si necesitas crear fixtures, hazlo solamente en un entorno temporal y sin alterar los artefactos del proyecto.



\### 6. Sesgos estadísticos



Investiga:



\- Sesgo de supervivencia.

\- Sesgo de selección.

\- Look-ahead bias.

\- Leakage.

\- Muestras incompletas.

\- Ventanas elegidas incorrectamente.

\- Cambios de metodología.

\- Comparaciones incompatibles.

\- Interpretaciones estadísticamente injustificadas.



No propongas optimizar parámetros ni modificar umbrales.



El objetivo es validar el método existente, no buscar mejores resultados retrospectivos.



\### 7. Entregable



Por cada indicador auditado:



| Elemento | Resultado |

|---|---|

| Indicador | Nombre |

| Módulo | Archivo y función |

| Definición | Contrato aplicable |

| Inputs | Variables utilizadas |

| Fórmula | Implementación |

| Ventana | Temporalidad |

| Unidades | Escala |

| Contraste | Resultado independiente |

| Dictamen | Estado |

| Evidencia | Código y pruebas |



Clasifica los hallazgos P0-P3.



Diferencia entre:



\- Error matemático demostrado.

\- Error de implementación.

\- Inconsistencia de definición.

\- Riesgo estadístico.

\- Mejora metodológica.

\- Hipótesis pendiente.



\*\*Condición de cierre:\*\* cada cálculo crítico incluido en el alcance debe contar con un dictamen y evidencias suficientes.



\---



\# FASE 4 — AUDITORÍA DE CABLEADO, INTEGRACIÓN Y REPORTE FINAL



\## Orden de trabajo



Realiza una auditoría de extremo a extremo del flujo de información.



\*\*Objetivo:\*\* demostrar que la información descargada atraviesa correctamente los módulos y que los resultados publicados en el informe diario coinciden con los resultados realmente calculados.



Esta fase busca especialmente errores de integración y fallos silenciosos.



\### 1. Reconstrucción del flujo



Selecciona un conjunto de indicadores o bloques funcionales y reconstruye su recorrido:



FUENTE

↓

DESCARGADOR

↓

PARSER

↓

VALIDACIÓN

↓

TRANSFORMACIÓN

↓

CÁLCULO

↓

AGREGACIÓN

↓

WRITER

↓

RENDER

↓

INFORME FINAL



Identifica las funciones concretas de cada etapa.



\### 2. Auditoría de productores y consumidores



Comprueba:



\- Quién produce cada variable.

\- Quién la consume.

\- Qué claves utiliza.

\- Qué tipos transmite.

\- Qué valores puede recibir.

\- Qué campos son opcionales.

\- Qué valores por defecto existen.

\- Qué contratos protegen la comunicación.



Investiga especialmente los diccionarios y estructuras compartidas.



Contrasta `DICT-CONTRACT-FRAMEWORK` y sus pares productor-consumidor.



\### 3. Errores de cableado



Busca:



\- Funciones que no reciben parámetros esperados.

\- Resultados calculados pero no consumidos.

\- Funciones que nunca se invocan.

\- Módulos conectados a versiones antiguas.

\- Importaciones incorrectas.

\- Campos con nombres incompatibles.

\- Datos sobrescritos.

\- Datos descartados.

\- Variables intercambiadas.

\- Errores de orden de ejecución.

\- Configuraciones que no llegan al consumidor.

\- Resultados que se transmiten parcialmente.



Distingue desconexiones intencionadas y documentadas de errores reales.



\### 4. Persistencia



Contrasta:



\- Resultado interno.

\- Writer.

\- Fichero generado.

\- Manifest.

\- Receipt.

\- Histórico.

\- Lector posterior.



Investiga si el estado comunicado por el writer coincide con el contenido persistido.



\### 5. Generación del informe



Comprueba:



\- Correspondencia entre resultados y campos publicados.

\- Valores intercambiados.

\- Campos ausentes.

\- Duplicaciones.

\- Valores desactualizados.

\- Fechas distintas.

\- Coberturas incompatibles.

\- Estados mal interpretados.

\- Formatos incorrectos.

\- Datos parciales presentados como completos.



Verifica que los estados diagnósticos se representan correctamente.



\### 6. Trazabilidad



Para cada indicador seleccionado, intenta responder:



\- ¿Qué fuente lo originó?

\- ¿Qué fecha tiene?

\- ¿Qué transformaciones sufrió?

\- ¿Qué función lo calculó?

\- ¿Qué validación superó?

\- ¿Dónde se almacenó?

\- ¿Qué función lo publicó?

\- ¿Qué valor terminó viendo el usuario?



Identifica cualquier punto donde se pierda la trazabilidad.



\### 7. Entregable



Elabora un mapa real de los recorridos examinados.



Incluye:



\- Archivo.

\- Función.

\- Productor.

\- Consumidor.

\- Contrato.

\- Datos transmitidos.

\- Resultado.

\- Problemas.

\- Tests.

\- Evidencias.



Clasifica los defectos P0-P3.



No afirmes que todo el pipeline está validado si solo has comprobado determinados indicadores.



\*\*Condición de cierre:\*\* los flujos críticos examinados deben poder seguirse desde su origen hasta su publicación.



\---



\# FASE 5 — AUDITORÍA DE ROBUSTEZ, SEGURIDAD, RENDIMIENTO Y TESTS



\## Orden de trabajo



Realiza una auditoría de fiabilidad operativa y capacidad de recuperación del sistema.



\*\*Objetivo:\*\* detectar escenarios en los que el Radar pueda fallar, quedarse incompleto, consumir recursos excesivos o generar información incorrecta sin comunicarlo adecuadamente.



\### 1. Gestión de errores



Investiga:



\- Excepciones ocultas.

\- `except Exception`.

\- `except: pass`.

\- Retornos vacíos.

\- Valores por defecto.

\- Estados inconsistentes.

\- Errores absorbidos.

\- Mensajes de error incompletos.

\- Fallos sin logging.

\- Fallos que no se reflejan en health\_check.



Determina qué ocurre cuando falla una fuente o un módulo intermedio.



\### 2. Ejecución y recuperación



Audita:



\- Secuencia de tareas.

\- Reintentos.

\- Timeouts.

\- Procesos simultáneos.

\- Condiciones de carrera.

\- Idempotencia.

\- Interrupciones.

\- Recuperación.

\- Ejecuciones duplicadas.

\- Persistencia parcial.

\- Receipts.

\- Scheduler.

\- Daily run gate.



Comprueba si una ejecución incompleta puede parecer una ejecución correcta.



\### 3. Seguridad



Busca:



\- API keys expuestas.

\- Contraseñas.

\- Secretos en logs.

\- Dependencias vulnerables.

\- Ejecución insegura de comandos.

\- Gestión incorrecta de archivos.

\- Permisos innecesarios.

\- Deserialización insegura.

\- Información sensible almacenada incorrectamente.



No reproduzcas secretos en el informe.



\### 4. Rendimiento



Investiga:



\- Descargas redundantes.

\- Operaciones repetidas.

\- Cálculos innecesarios.

\- Bucles ineficientes.

\- Problemas de memoria.

\- Procesamientos de DataFrames.

\- Escrituras excesivas.

\- Consultas redundantes.

\- Cuellos de botella.



No optimices nada durante esta fase.



Toda mejora sugerida debe preservar los resultados del sistema para el mismo snapshot.



\### 5. Auditoría de tests



Examina:



\- Cobertura unitaria.

\- Integración.

\- Contratos.

\- Temporalidad.

\- Persistencia.

\- Cálculos matemáticos.

\- Regresiones.

\- Tests centinela.

\- Tests de generación del informe.



Busca funciones críticas sin tests o con tests insuficientes.



Comprueba si los tests verifican la lógica o solamente la existencia de resultados.



Antes de ejecutar una prueba, comprueba sus efectos secundarios.



No ejecutes tests de red opt-in, procesos reales ni operaciones externas sin autorización.



Respeta el protocolo de tests de `00\_ARRANQUE.md`.



\### 6. Observabilidad



Determina si existen mecanismos para detectar:



\- Datos ausentes.

\- Datos obsoletos.

\- Fuentes degradadas.

\- Cobertura insuficiente.

\- Resultados parciales.

\- Fallos de writers.

\- Inconsistencias entre módulos.

\- Problemas del scheduler.

\- Ejecuciones incompletas.



Comprueba si health\_check refleja correctamente los problemas reales.



\### 7. Entregable



Presenta:



1\. Fallos de robustez.

2\. Riesgos operativos.

3\. Problemas de seguridad.

4\. Cuellos de botella.

5\. Tests insuficientes.

6\. Problemas de observabilidad.

7\. Escenarios de fallo.

8\. Evidencias.

9\. Prioridad P0-P3.

10\. Mejoras recomendadas.



\*\*Condición de cierre:\*\* deben estar identificados los principales escenarios de fallo y las limitaciones de detección y recuperación.



\---



\# FASE 6 — CONSOLIDACIÓN, DICTAMEN Y PLAN DE CORRECCIÓN



\## Orden de trabajo



Consolida los resultados de las cinco fases anteriores.



\*\*Objetivo:\*\* emitir un dictamen técnico integral, diferenciando los defectos confirmados de los riesgos, las decisiones de diseño y los problemas pendientes de investigar.



No repitas todas las auditorías desde cero.



Utiliza las evidencias y los identificadores de hallazgos ya obtenidos.



\### 1. Consolidación



Revisa los resultados de las fases:



\- Inventario y arquitectura.

\- Fuentes y descarga.

\- Matemáticas e indicadores.

\- Cableado e integración.

\- Robustez, seguridad y tests.



Comprueba que los diagnósticos no se contradigan.



\### 2. Eliminación de duplicados



Identifica:



\- Hallazgos repetidos.

\- Diferentes síntomas de una misma causa raíz.

\- Problemas ya solucionados.

\- Problemas ya documentados.

\- Falsos positivos.

\- Decisiones WONT\_FIX.

\- Defectos que han reaparecido.



No contabilices varias veces un mismo defecto.



Agrupa síntomas cuando compartan una causa raíz, pero conserva las evidencias de cada manifestación.



\### 3. Verificación del estado



Contrasta los hallazgos con:



\- HEAD actual.

\- Working tree.

\- REGISTRO.yaml.

\- ESTADO\_SISTEMA.md.

\- Tests.

\- Contratos.

\- Evidencias de producción, cuando existan.



No declares una solución desplegada si solo existe un commit local.



\### 4. Clasificación final



Consolida el número de hallazgos por:



\- P0.

\- P1.

\- P2.

\- P3.



Separa:



\- Confirmados.

\- Reproducidos.

\- Probables.

\- Hipótesis.

\- No verificados.

\- Descartados.



No utilices el número de defectos como medida de calidad de la auditoría.



\### 5. Plan de corrección



Ordena las correcciones atendiendo a:



1\. Riesgo sobre la integridad de los datos.

2\. Riesgo de resultados matemáticos incorrectos.

3\. Incumplimiento de contratos.

4\. Fallos de integración.

5\. Fallos de persistencia y publicación.

6\. Robustez.

7\. Seguridad.

8\. Rendimiento.

9\. Mantenibilidad.



Para cada corrección indica:



\- Hallazgo asociado.

\- Causa raíz.

\- Archivos afectados.

\- Cambio mínimo necesario.

\- Dependencias.

\- Riesgos de regresión.

\- Tests necesarios.

\- Verificación de aceptación.



No implementes ninguna corrección.



\### 6. Gobernanza



Comprueba si deben crearse o actualizarse items en `REGISTRO.yaml`.



Antes de proponer un alta, busca duplicidades.



No cambies los estados de gobernanza sin autorización.



\### 7. Dictamen final



Clasifica el estado de cada bloque:



\- VERIFICADO.

\- VERIFICADO CON LIMITACIONES.

\- DEFECTOS CONFIRMADOS.

\- RIESGO PENDIENTE.

\- BLOQUEADO.

\- NO AUDITADO.



Entrega un resumen global que responda:



1\. ¿La arquitectura es coherente?

2\. ¿Las fuentes están correctamente integradas?

3\. ¿La integridad de los datos está protegida?

4\. ¿Los cálculos están contrastados?

5\. ¿El flujo de información es correcto?

6\. ¿El informe refleja los resultados internos?

7\. ¿El sistema detecta los fallos?

8\. ¿Existen riesgos que puedan invalidar los informes diarios?

9\. ¿Qué debe corregirse primero?

10\. ¿Qué no se ha podido demostrar?



No declares el proyecto completamente auditado si quedan componentes críticos sin revisar.



\### 8. Entregable final



Genera un informe técnico consolidado con:



\- Resumen ejecutivo.

\- Estado del sistema.

\- Alcance real.

\- Cobertura por bloques.

\- Hallazgos consolidados.

\- Evidencias.

\- Riesgos.

\- Falsos positivos.

\- Deuda técnica.

\- Plan priorizado.

\- Limitaciones.

\- Dictamen.



\*\*Condición de cierre:\*\* el diagnóstico debe permitir decidir qué corregir, qué mantener y qué riesgos aceptar, sin alterar todavía el proyecto.



\---



\# FIN DEL DOCUMENTO

