# 60 - Politica de universos del reporte diario

Fecha: 2026-10-05
Rama: main
Objeto: documentar los universos, funciones y criterios que coexisten
        en el reporte diario, y proponer politica de unificacion.
Estado: diagnostico. Sin cambios de codigo.

## 1. Contexto

El reporte diario tiene 947 lineas y 59 secciones. Usa 3 universos
distintos para clasificar "fases Wyckoff", 3 definiciones distintas
de "lider", y 2 funciones Wyckoff con salida identica pero semantica
distinta. Las contradicciones estan documentadas con ~50 notas al
pie, pero no resueltas. El reporte se disculpa consigo mismo en lugar
de declarar su semantica.

Este documento no toca codigo. Solo mapea el problema y propone
politica. La decision final es del usuario.

## 2. Inventario de universos

### 2.1. U1 - Top-20 sectorial (~220 tickers)

    Tamano:     20 tickers x 11 sectores = 220
    Definicion: config.TOP_N_SECTOR_COMPONENTS = 20
    Usado por:
      indicators/sector_breadth.py:63   (A/D por sector)
      indicators/stock_leader.py:174    (top-15 -> top-5 WLS)
    Funcion Wyckoff: classify_wyckoff_phase (legacy)

### 2.2. U2 - Holdings completo del ETF (variable)

    Tamano:     variable por ETF (hasta cientos de tickers)
    Definicion: group['ticker'].tolist() sin head()
    Usado por:
      indicators/sector_wyckoff_distribution.py:22
                  (conteo de fases por sector)
    Funcion Wyckoff: classify_wyckoff_phase (legacy)

### 2.3. U3 - ETF sectorial / indice (11 + 8)

    Tamano:     11 ETF sectoriales + 8 indices internacionales
    Definicion: MARKET_TICKERS['sectors'] + INDEX_CONFIG
    Usado por:
      regimes/sector_regime.py:168      (fase del ETF sectorial)
      indicators/index_phase.py:20      (fase de indices)
    Funcion Wyckoff: wyckoff_structure_core (legacy, distinta)

### 2.4. U4 - Universo completo USA (~326 tickers)

    Tamano:     ~326 tickers USA de stock_prices
    Definicion: sin filtro sectorial ni top-N
    Usado por:
      src/pipeline/mte_confirmation.py  (A/D net, NH/NL)
    Funcion Wyckoff: NO usa (solo A/D y NH/NL)

### 2.5. U5 - Institucional IAE (~255 CUSIPs)

    Tamano:     ~255 CUSIPs del catalogo contractual
    Definicion: src/institutional_accumulation/operational_universe.py
    Usado por:
      seccion IAE (13F, NIPC)
    Funcion Wyckoff: NO usa
    Nota: ortogonal a U1-U4. Solo comparte el termino "universo".
## 3. Contradicciones concretas

### C1 - Dos funciones Wyckoff con salida identica

    classify_wyckoff_phase:  conjunciones estructurales
    wyckoff_structure_core:  bandas de combined
    Salida de ambas: ACCUMULATION | MARKUP | RANGE | DISTRIBUTION |
                     MARKDOWN

    Usan classify:  sector_breadth, sector_wyckoff_distribution,
                    stock_leader, index_leaders
    Usan core:      sector_regime, index_phase

    Efecto observado 2026-10-05: XLI figura como DISTRIBUTION en
    sector_regime y como 0 DISTRIBUTION en sector_wyckoff_distribution.

### C2 - Dos universos para conteo de fases

    U1 (top-20)    -> conteos A/D por sector
    U2 (completo)  -> conteos ACC/MK/RANGE/DIST/MD

    El reporte muestra ambas tablas. Los numeros no cuadran.
    Notas al pie L70 y L72 lo admiten.

### C3 - Tres definiciones de "lider"

    D1: mayor retorno 20d
        Usado en: Concentracion del liderazgo, Liderazgo interno
    D2: mayor WLS
        Usado en: Representatividad, primera fila de secciones
    D3: top-15 por weight -> top-5 por WLS
        Usado en: Acciones Seleccionadas

    Notas al pie L89, L492, L623 lo admiten. La palabra "lider"
    aparece en al menos 3 secciones con significados distintos.

### C4 - Dos universos de flujo de mercado

    "Flujo de Mercado - Sectores (Proxy)": media sectorial de
        flow_proxy_z
    "Flujo Primario ETF - Caracteristicas": suma de flujos
        primarios por ETF (shares x NAV)

    Ambas usan la palabra "flujo". Semantica distinta. Un sector
    puede tener flow proxy positivo y flujo primario negativo.

### C5 - Notas al pie autocorrectivas

    El reporte contiene ~50 notas al pie. Muchas son de la forma
    "esta metrica NO coincide con aquella". Ejemplos:

      L70: "La columna A/D de esta tabla agrega solo top-20 por
            sector. No coincide con la metrica Advance/Decline Net
            de Confirmation Data."
      L89: "Criterio 'Lider': ticker con mayor retorno 20d...
            No coincide con Representatividad, que ordena por WLS."
      L124: "Retorno 20d mide la mediana... Distinta del retorno
             del ETF sectorial mostrado en otra seccion."
      L492: "La primera fila de cada sector... No coincide con la
             primera fila de Representatividad."

    Sintoma: el diseno no resuelve las contradicciones. Se defiende
    con notas al pie. Un lector nuevo no puede entender el reporte
    sin leer las 50 notas.

### C6 - Wyckoff uniforme en regimen homogeneo

    L1002: "Wyckoff puede aparecer uniforme cuando el mercado esta
    en un regimen homogeneo."

    Consecuencia: cuando Wyckoff sale uniforme (todos los sectores
    con la misma fase), el ranking sectorial por Wyckoff pierde
    poder discriminativo. No es contradiccion, es limitacion. Se
    documenta aqui para no olvidarla.
## 4. Politica propuesta

Para cada contradiccion, tres opciones:

    (a) Unificar         - misma definicion en todo el reporte
    (b) Declarar         - cabecera de seccion explicita la semantica
    (c) Eliminar         - retirar una de las versiones

### C1 - Funciones Wyckoff

    Recomendado: (b) declarar en cabecera de cada seccion
    Razon: unificar exige migrar wyckoff_structure_core a v1.8
    (prohibido por dictamen 57b seccion 6) o reescribir los 2
    consumidores. Coste alto, beneficio bajo. Declarar es barato.

    Texto propuesto en cabecera de "Matriz de Regimen Sectorial":
      "Fase calculada con wyckoff_structure_core (bandas de
       combined). Distinta de la fase en 'Distribucion Wyckoff
       sectorial', que usa classify_wyckoff_phase (conjunciones)."

### C2 - Conteos de fases

    Recomendado: (c) eliminar una de las dos tablas.
    Razon: dos tablas con conteos de fases del mismo sector es
    ruido puro. La tabla de sector_wyckoff_distribution (U2) es
    la mas completa y util. La columna de conteos en sector_breadth
    (U1) puede eliminarse.

    Alternativa: (a) unificar todo en U1. Requiere cambiar
    sector_wyckoff_distribution para usar top-20 en lugar de
    holdings completo. Decision de producto.

### C3 - Definiciones de lider

    Recomendado: (b) + reducir secciones.
    De 3 definiciones, mantener 2: D1 (retorno 20d) y D2 (WLS).
    Eliminar D3 (dos pasos) por ser composicion no declarada.
    Cada seccion con la palabra "lider" declara en cabecera cual usa.

    Texto tipo:
      "Lider en esta seccion: mayor retorno 20d entre top-20."

### C4 - Flujos de mercado

    Recomendado: (b) renombrar secciones para no solapar terminos.
    Renombrar "Flujo de Mercado - Sectores (Proxy)" a
    "Flow Proxy por sector". Renombrar "Flujo Primario ETF -
    Caracteristicas" a "Flujo primario ETF por sector".
    Distingue los dos conceptos sin eliminar ninguno.

### C5 - Notas al pie

    Recomendado: (a) reducir tras aplicar C1-C4.
    Objetivo: <= 20 notas al pie, todas de metodo o fuente.
    Ninguna del tipo "esto no es aquello" (porque ya estara
    declarado en cabecera).

    Estimacion: ~30 notas eliminables (redundantes, repetidas
    entre secciones, o ya cubiertas por cabecera).

### C6 - Wyckoff uniforme

    Recomendado: (b) declarar limitacion en la seccion.
    Cuando el mercado este en regimen homogeneo, indicar en la
    seccion de distribucion Wyckoff que la uniformidad limita la
    interpretacion por sector.
## 5. Decision solicitada

    D1. Acepta la politica propuesta para C1-C6?
    D2. Sobre C2, elige: (a) unificar en U1 (top-20) o
        (c) eliminar tabla de sector_breadth.
    D3. Sobre C3, acepta mantener 2 definiciones de lider (D1, D2)
        y eliminar D3 (top-15 -> top-5)?
    D4. Sobre C4, acepta renombrar las 2 secciones de flujo?
    D5. Sobre C5, acepta reducir notas al pie a <=20?

Respuesta: SI / NO / REFORMULAR por cada una.

## 6. Fuera de alcance

- Migracion de wyckoff_structure_core a v1.8 (prohibida por 57b).
- Cambio de umbrales TOP_N_* (decision de producto, no tecnica).
- Reorganizacion de modulos Python (fase posterior a este doc).
- IAE / operational_universe (ortogonal, sin relacion).
- SOW y Capa 1-3 del frente Wyckoff (cerradas en 57-59).

## Anexo. Ubicaciones de codigo

    config/settings.py:
      TOP_N_CANDIDATES = 15
      TOP_N_SECTOR_COMPONENTS = 20
      TOP_N_LEADERS = 5
      EXPECTED_SECTOR_COUNT = 11

    Universos:
      U1: indicators/sector_breadth.py:63
      U1: indicators/stock_leader.py:174
      U2: indicators/sector_wyckoff_distribution.py:22
      U3: regimes/sector_regime.py:168
      U3: indicators/index_phase.py:20
      U4: src/pipeline/mte_confirmation.py
      U5: src/institutional_accumulation/operational_universe.py

    Funciones Wyckoff:
      classify_wyckoff_phase:  indicators/wyckoff.py:383
      wyckoff_structure_core:  indicators/wyckoff.py:301

## Anexo. Notas al pie del reporte (2026-10-05)

Referencias por linea en outputs/report/reporte_diario.md:

    L70   universo top-20 vs completo (A/D)
    L72   cobertura 20 componentes por sector
    L89   definicion de lider por retorno 20d
    L124  retorno 20d componentes vs ETF
    L155  repeticion de L124
    L261  Structural Strength != SLPM
    L297  filtro por fase Wyckoff
    L299  criterio seleccion 2 pasos
    L395  flujo atomico vs agregado
    L414  repeticion de L395
    L492  definicion de lider por retorno 20d
    L623  definicion de lider por WLS
    L1002 Wyckoff uniforme en regimen homogeneo
    L1008 advertencia de acoplamiento de senales
    (resto: notas de metodo y fuente, mantenibles)

## 7. Decisiones del usuario (2026-10-05)

    P1. Unificar conteo de fases con universo completo del ETF (U2).
        Eliminar la tabla duplicada de sector_breadth.

    P2. Nueva definicion de lider:
        - Sector en ACC.
        - Tomar los 20 mayores por peso en el ETF.
        - Ordenar por WLS (criterio actual, no inventar).
        - Mostrar los 5 mejores, marcando cuales estan en ACC
          y cuales no.
        Sin filtro previo por fase (los 5 pueden estar fuera de ACC).

    P3. Renombrar las 2 secciones de flujo para distinguirlas.

    P4. Arreglar la causa de cada nota al pie, luego reducir a ~20.
        Orden de ejecucion: A -> C -> D -> B.

    P5. No anadir aviso sobre Wyckoff plano (redundante, ya se ve
        en la tabla).

## 8. Plan de ejecucion

    Rama: radar-v5 (desde main).
    Orden: A -> C -> D -> B.

    A. Renombrar 2 secciones de flujo (src/report/*.py).
    C. Unificar conteo de fases (eliminar tabla duplicada).
    D. Nuevo criterio de lider en stock_leader.py.
    B. Limpiar notas al pie residuales.

    Un commit por bloque. Suite verde + E2E si toca pipeline.

Fin del documento 60.