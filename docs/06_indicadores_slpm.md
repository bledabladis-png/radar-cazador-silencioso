# Structural Leadership (SLPM v1.2)

## Proposito
Audita la calidad del liderazgo del sector #1 del ranking.

## Componentes
- **Leader Breadth v2:** amplitud del liderazgo (RS, momentum, flujo, Wyckoff).
- **Leader Integrity Score (LIS):** intensidad/calidad de los lideres individuales.
- **Flow Divergence 2.0:** divergencias entre flujo de lideres y sector.
- **State Machine:** clasifica el estado (CONFIRMED, EMERGING, TACTICAL_CORRECTION, STRUCTURAL_DECAY, LOST, UNRESOLVED).

## Umbrales de la State Machine
| Parametro | Valor |
|---|---|
| breadth_max_decay | 0.35 |
| lis_min_confirmed | 0.30 |
| persistence_max_emerging | 0.50 |
| persistence_min_confirmed | 0.50 |
| structural_min_confirmed | 0.20 |
| structural_min_emerging | 0.20 |

## Logica de Clasificacion
```
Clasifica el estado de liderazgo del sector #1 usando la State Machine.
    Estados: CONFIRMED, EMERGING, TACTICAL_CORRECTION, STRUCTURAL_DECAY,
    LOST, UNRESOLVED.
    Umbrales: T>0.20, S>0.20, LIS>0.30, Breadth>0.50, Persistence>0.50, etc.
```