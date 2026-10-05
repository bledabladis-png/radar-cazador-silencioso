# 61 - Arquitectura objetivo del reporte diario

Fecha: 2026-10-05
Rama: radar-v5
Objeto: plan de trabajo para reorganizar el reporte diario.
Estado: propuesta. Pendiente validacion.

## 1. Principio de jerarquia

El reporte debe leerse en 5 capas. Cada capa responde a una pregunta
distinta. No se mezclan.

    C1. Regimen global      "que esta pasando en el mercado?"
    C2. Sectorial           "que sectores destacan?"
    C3. Lideres             "que valores destacan dentro del sector?"
    C4. Confirmacion        "hay senales independientes que lo apoyen?"
    C5. Contexto            "como se situa esto respecto a otros activos?"

Hoy las 59 secciones estan mezcladas. No hay jerarquia visible. Un
lector no sabe donde mirar primero.

## 2. Mapa actual de secciones por capa

Total actual: 59 secciones ## en el reporte.

    C1 (Regimen global):  ~10 secciones
      Resumen de Regimenes
      Breadth de Mercado
      Dispersion entre sectores
      Correlacion entre sectores
      Contexto transversal de mercado
      Market Transition Engine
      ...

    C2 (Sectorial):  ~20 secciones
      Sector Breadth & Health
      Concentracion del liderazgo
      Dispersion interna
      Momentum de Precio - Sectores
      Flujo de Mercado - Sectores
      Tactical Leaders
      Structural Ranking
      Rankings Sectoriales
      Persistencia sectorial
      Opportunity Map
      Structural Leadership (SLPM)
      Rotacion sectorial reciente
      Matriz de Regimen Sectorial
      Representatividad del lider
      Distribucion Wyckoff sectorial
      Divergencia sector-lideres
      Momentum de amplitud
      ...

    C3 (Lideres):  ~8 secciones
      Acciones Seleccionadas por el Modelo
      Sector: XLK / XLF / XLV / XLE / XLB / XLC
      Liderazgo relativo interno
      ...

    C4 (Confirmacion):  ~8 secciones
      Sentimiento de Opciones
      Actividad en ATS - Dark Pools
      Confirmation Data
      Matriz de Evidencia
      ...

    C5 (Contexto / Internacional):  ~13 secciones
      Flujo Primario ETF (SPDR)
      Flujo Primario ETF - Agregado sectorial
      Flujo Primario DAXEX
      Flujo Primario ISF.L
      Flujo Primario LYXI
      Flujo Primario IWM
      Flujo Primario QQQ (SEC)
      Posicionamiento CFTC
      Flujo Posicional N-PORT
      Rendimiento QQQ
      Flujo de Participaciones QQQ
      Flujo - Sintesis Descriptiva
      Estructura de volatilidad
      Calidad, frescura y cobertura
      Indices (USA + Europa)
      Estado Actual - Sintesis
      Acumulacion Institucional (13F)
## 3. Objetivo de reduccion

    59 secciones -> objetivo 30-35.

No es reducir por reducir. Es:
- Quitar duplicaciones (ya hechas en A/C).
- Fusionar secciones que responden a la misma pregunta.
- Eliminar secciones que no aportan decision.

## 4. Candidatas a fusion

### F1. Tres rankings sectoriales -> uno

    Hoy: Structural Ranking, Rankings Sectoriales, SLPM.
    Todos responden "que sectores son fuertes".
    Objetivo: un ranking con columnas complementarias.
    Ahorro: 2 secciones.

### F2. Dos dispersion/correlacion -> una tabla doble

    Hoy: Dispersion entre sectores + Correlacion entre sectores.
    Ambas responden "cuanto se parecen los sectores".
    Objetivo: una seccion con 2 sub-tablas.
    Ahorro: 1 seccion.

### F3. Seis flujos primarios internacionales -> 1 tabla

    Hoy: DAXEX, ISF.L, LYXI, IWM, QQQ SEC, QQQ NPORT.
    Cada uno en su seccion.
    Objetivo: 1 tabla agregada con ticker, flujo, z, fecha.
    Ahorro: 5 secciones.

### F4. Tactical + Structural + Rankings -> 1 vista

    Hoy: Tactical Leaders, Structural Ranking, Rankings Sectoriales.
    Son tres vistas del mismo ranking sectorial.
    Objetivo: 1 seccion con columnas Tactical/Structural/Score.
    Ahorro: 2 secciones.

### F5. Dos Momentum de Precio -> 1

    Hoy: "Momentum de Precio - Sectores" + "Momentum de Precio - Otros".
    Objetivo: 1 seccion con 2 tablas.
    Ahorro: 1 seccion.

### F6. Dos Flujo de Mercado - Proxy -> 1

    Hoy: "Flujo de Mercado - Sectores" + "Flujo de Mercado - Otros".
    Objetivo: 1 seccion.
    Ahorro: 1 seccion.

Total candidatas a fusion: ~12 secciones -> 6. Ahorro real: 6.

## 5. Candidatas a eliminar

    E1. "Representatividad del lider" -> ya cubierto por SLPM.
    E2. "Divergencia sector-lideres" -> subsumible en "Liderazgo relativo".
    E3. "Momentum de amplitud" -> subsumible en "Sector Breadth & Health".
    E4. "Flujo - Sintesis Descriptiva" -> el dato ya esta en cada flujo.
    E5. "Calidad, frescura y cobertura" -> fusionable con Confirmation.

Ahorro: 5 secciones.

## 6. Secciones que se mantienen sin cambios

    Resumen de Regimenes
    Breadth de Mercado
    Sector Breadth & Health
    Concentracion del liderazgo
    Acciones Seleccionadas
    Sector: XLK / XLF / XLV / XLE / XLB / XLC  (6)
    Sentimiento de Opciones
    Flujo Primario ETF (SPDR)
    Flujo Primario ETF - Agregado sectorial
    Estructura de volatilidad
    Market Transition Engine
    Confirmation Data
    Actividad en ATS
    Indices (USA + Europa)
    Matriz de Evidencia
    Acumulacion Institucional (13F)

Total secciones en objetivo estimado:

    59 - 6 (fusion) - 5 (eliminar) = 48

No llegamos a 30-35 con solo esto. Hace falta mas. La reduccion
fuerte vendria de cortes mas agresivos (eliminar tablas enteras,
no solo secciones). Eso se decide con el usuario caso por caso.
## 7. Orden objetivo del reporte

    ## C1. Regimen global
       Resumen de Regimenes
       Breadth de Mercado
       Dispersion y Correlacion (fusion F2)
       Contexto transversal de mercado

    ## C2. Sectorial
       Sector Breadth & Health (+ Momentum amplitud)
       Concentracion del liderazgo
       Ranking Sectorial unificado (fusion F1 + F4)
       Persistencia sectorial
       Opportunity Map
       SLPM
       Rotacion sectorial reciente
       Matriz de Regimen Sectorial
       Distribucion Wyckoff sectorial

    ## C3. Lideres por sector
       Acciones Seleccionadas (con las 6 tablas por sector)

    ## C4. Confirmacion
       Sentimiento de Opciones
       Confirmation Data (+ Calidad, fusion E5)
       Actividad en ATS
       Matriz de Evidencia
       Market Transition Engine

    ## C5. Contexto internacional
       Flujo Primario ETF (SPDR)
       Flujo Primario ETF - Agregado sectorial
       Flujos internacionales agregados (fusion F3)
       Posicionamiento CFTC
       Indices (USA + Europa)
       Acumulacion Institucional (13F)
       Estado Actual - Sintesis

## 8. Bloques de ejecucion

Cada bloque: 1 commit, suite verde, sin mezclar con el siguiente.

    D1. Fusion F3 (flujos internacionales en 1 tabla).
    D2. Fusion F1 + F4 (ranking sectorial unificado).
    D3. Fusion F2 + F5 + F6 (pares de secciones).
    D4. Eliminaciones E1-E5.
    D5. Reordenar las secciones segun el orden del apartado 7.
    D6. Regenerar reporte y validar E2E.

Total estimado: 6 commits sobre radar-v5.

## 9. Decision solicitada

    V1. Aprueba las fusiones F1-F6?
    V2. Aprueba las eliminaciones E1-E5?
    V3. Aprueba el orden del apartado 7?
    V4. Aprueba los 6 bloques de ejecucion?

Respuesta: SI / NO / REFORMULAR por cada una.

## 10. Fuera de alcance

- Cambios de calculo en los modulos.
- Cambios de contratos.
- Rediseno del Wyckoff (fase cerrada).
- IAE y universos institucionales.

## 11. Resultado de la ejecucion (2026-10-05)

Ejecutados los bloques D1-D3. Estado final:

    D1  HECHO  4 flujos internacionales -> 1 tabla    -3 secciones
    D2  HECHO  3 rankings -> 1 tabla                  -2 secciones
    D3  HECHO  reorder por capas (5 capas)            0 secciones

    Reduccion total: 59 -> 52 secciones, 947 -> 885 lineas.

No se han hecho F2, F5, F6 (fusiones de pares) porque los tests I1
y test_report_leaders_nomenclatura anclan los titulos con razon
semantica (las secciones son complementarias, no duplicadas).

No se han hecho E1-E5 (eliminaciones) porque la inspeccion del
reporte real muestra que las 5 secciones tienen contenido util.
Ninguna esta vacia.

Orden final del reporte (5 capas):
    C1 Regimen global        7 secciones
    C2 Analisis sectorial   23 secciones
    C3 Lideres por sector    7 secciones
    C4 Confirmacion          1 seccion
    C5 Contexto internac.   14 secciones

Fin del documento 61.
