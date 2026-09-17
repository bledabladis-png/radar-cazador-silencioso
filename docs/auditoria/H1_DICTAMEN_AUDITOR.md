# Dictamen del auditor - H1 Mutabilidad historica

**Fecha:** 2026-09-17
**Informe base:** `H1_INFORME_MUTABILIDAD_HISTORICA.md` (commit `1d5c6c2`)
**HEAD de dictamen:** `1d5c6c2` (origin/main)
**Estado:** DICTAMEN EMITIDO. Decision: WONT FIX / POLITICA ACEPTADA.

---

## Diagnostico

**Aprobado. No hay bug que reparar. Si existe una cuestion arquitectonica real de gobernanza de datos.**

La evidencia demuestra tres mecanismos independientes:

    proveedor revisa valores
            +
    pipeline puede regenerar historicos
            +
    append_dedup sustituye observaciones
            |
    dataset vivo = mutable por diseno

No se aprueba convertir H1 automaticamente en un proyecto de inmutabilidad total.

## Decision principal

**No se exige, de momento, reproducibilidad byte-exacta de TODO el dataset historico por cada run.**

Requisito establecido:

> **Los artefactos publicados deben ser reproducibles a nivel de version y las revisiones posteriores de datos deben ser detectables.**

Distincion clave:

    reproducir el reporte publicado
                !=
    reconstruir byte-a-byte todos los inputs originales


---

## 1. Byte-exactitud de todo el dataset

**Respuesta: NO como requisito actual.**

No existe en la evidencia un requisito regulatorio, contractual u operativo que obligue a conservar copia byte-exacta de los 562 instrumentos para cada ejecucion historica. El coste seria desproporcionado para el uso actual.

Los `outputs/history/*.csv` versionados ya proporcionan congelacion historica del **resultado publicado**. El problema restante es que no siempre puede reproducirse el input que lo genero.

## 2. Granularidad

Como no se exige byte-exactitud completa, no se fija todavia una politica diaria/semanal/mensual de snapshots. La granularidad sera decision posterior si aparece requisito de reconstruccion de inputs.

Nivel minimo actual suficiente:

    publicacion historica -> version Git del artefacto

## 3. Commit `35af4ba` (1485 valores regenerados)

Tratar como:

> **nuevo baseline funcional del pipeline**, con trazabilidad del cambio.

No se intenta reconstruir ahora los 1485 valores originales. El commit `35af4ba` ya constituye la trazabilidad del motivo:

    logic change -> regeneracion -> nuevo historico

Debe quedar identificado como **regeneracion deliberada**, no como revision silenciosa de proveedor.

## 4. Ampliar FU-002

**Si, pero no dentro de FU-002 inmediatamente.**

FU-002 actualmente significa:

    hash del artefacto -> integridad en el momento de escritura

Eso es coherente. La deteccion cross-run seria otro concepto:

    FU-002 -> integridad
    H1-D   -> deteccion de revision historica

No se reutiliza FU-002 para una responsabilidad nueva sin diseno especifico.


---

## 5. `market_data.parquet` / `stock_prices.parquet`

**No recomiendo dejar de ignorarlos automaticamente.**

Estado actual valido operativamente:

    parquet -> gitignored
    manifest -> tracked

El manifest debe entenderse correctamente:

> **es una huella historica no verificable si el artefacto original ya no esta disponible.**

No es un bug del manifest. Resolverlo requiere decidir posteriormente entre almacenamiento externo o snapshots/versionado de artefactos. No se recomienda meter esos parquets completos en Git solo para cerrar H1.

## 6. Fuente historica oficial

Si hace falta una decision conceptual clara.

Para auditoria de outputs:

    outputs/history/*.csv -> historico publicado/versionado

Para datos de trabajo:

    data/*.parquet -> fuente de calculo/entrada del pipeline

No se declaran dos fuentes de verdad equivalentes. Jerarquia adoptada:

    DATA            -> inputs de calculo
    OUTPUTS/HISTORY -> outputs historicos publicados

## 7. Compliance / regulacion

Con la informacion disponible:

**No hay evidencia de obligacion regulatoria o contractual de inmutabilidad.**

No se impone la opcion B. Si aparece ese requisito, H1 debe reabrirse y la decision cambiaria sustancialmente.

---

## Decision sobre opciones A/B/C/D

Dictamen combinado:

| Opcion | Dictamen |
|---|---|
| A - Snapshot | **NO ahora; posible fase posterior** |
| B - Congelacion | **NO-GO** |
| C - Mutabilidad documentada | **GO** |
| D - Deteccion de revisiones | **GO, como evolucion futura** |

**Precision importante: C no significa ignorar H1.**

Politica adoptada:

> **El dataset operativo es mutable por diseno. Los outputs publicados se versionan. Las revisiones de datos no constituyen por si mismas un error.**

Cuando el sistema necesite mayor trazabilidad, `C + D` seria la evolucion natural. No se aprueba A hasta que exista necesidad real de reconstruir inputs historicos completos.


---

## Estado recomendado del proyecto

    H1
    |-- Diagnostico                                  OK
    |-- Bug funcional                                NO
    |-- Politica de mutabilidad                      ACEPTADA
    |-- Outputs historicos                           versionados por Git
    |-- Inputs parquet                               datos de trabajo
    |-- Revisiones de proveedor                      permitidas
    |-- Deteccion cross-run                          mejora futura

## Cierre H1

**H1 queda CERRADO - WONT FIX / POLITICA ACEPTADA.**

No se abre actualmente un ciclo de implementacion.

## Condiciones de reapertura

Reabrir H1 si:

- aparece requisito regulatorio/compliance;
- se exige reconstruccion exacta de inputs;
- auditoria externa necesita verificar el dataset completo de una fecha pasada;
- aparece necesidad de distinguir automaticamente revision de proveedor frente a regeneracion del pipeline.

## Matiz de auditoria

No escribir en el prompt:

> "No produce perdida de informacion en el momento del run."

Es cierto para las mediciones aportadas, pero demasiado absoluto para una politica historica. Formulacion correcta:

> **La mutabilidad observada no implica perdida de informacion dentro de la ejecucion que genera el artefacto; si puede impedir reconstruir versiones anteriores del input si estas no fueron conservadas.**

Esa formulacion refleja exactamente lo demostrado.

---

**Dictamen final: C + D como direccion arquitectonica, sin patch ahora. A y B descartadas hasta que exista un requisito adicional.**
