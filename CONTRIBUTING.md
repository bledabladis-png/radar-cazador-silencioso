# Reglas de contribución

## Modificación de señales, pesos o umbrales

Ningún cambio que afecte a la lógica decisoria será aceptado sin:

1. Evidencia de redundancia o necesidad funcional.
2. Análisis de información incremental.
3. Validación OOS/multiperíodo.
4. Documentación trazable del cambio.

## Regla específica de double counting

- Dependencia entre señales no implica doble contabilidad.
- No se elimina una señal solo por correlación alta.
- Toda corrección debe demostrar que la señal era redundante en la decisión final.
- La prueba OOS es obligatoria antes de consolidar una eliminación.

## Arquitectura SLPM v1.2

La State Machine decisoria recibe exclusivamente:

- Structural
- Breadth
- Persistence

LIS es diagnóstico.
Tactical es informativo/Opportunity Map.
