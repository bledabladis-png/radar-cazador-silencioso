## Proposito
Calcula el PCR (Put/Call Ratio) y el IHR (Institutional Hedge Ratio) a partir de datos de CBOE.

## Arquitectura

- compute_pcr_signals(): orquestador principal.
- options_metrics.py: funciones de calculo (IHR, PCR, Put/Call Share, etc.).
- classify_pcr(): clasifica el Z-Score del PCR en Panico, Miedo, Neutral, Optimismo, Euforia.
- classify_ihr(): clasifica el IHR en Especulacion, Equilibrado, Cobertura institucional.


## Formulas

- **IHR:** PCR Indices / PCR Acciones.
- **Volume PCR:** Put Volume / Call Volume.
- **OI PCR:** Put OI / Call OI.


## Salidas

- Seccion 'Sentimiento de Opciones (OMS v2.0)' en el reporte.
- PCR Total, PCR Indices, PCR Acciones, IHR, Volumen en Indices, Put/Call Share.


## Limitaciones Conocidas
El Z-Score del PCR requiere al menos 20 dias de historial. Con menos de 20 registros, no se calcula.
