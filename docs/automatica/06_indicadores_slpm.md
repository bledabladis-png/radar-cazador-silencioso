## Proposito
Audita la calidad del liderazgo del sector #1 del ranking. No es otro ranking: evalua si el lider es estructuralmente solido.

## Arquitectura

- compute_leader_breadth_v2(): amplitud del liderazgo (RS, momentum, flujo, Wyckoff).
- compute_leader_integrity(): LIS (intensidad/calidad de los lideres individuales).
- compute_flow_divergence_v2(): divergencias entre flujo de lideres y sector.
- classify_leadership_state(): State Machine con 6 estados + UNRESOLVED.
- confirm_transition(): histeresis temporal.


## Formulas
**State Machine:** Clasifica el estado de liderazgo del sector #1 usando la State Machine.
    Estados: CONFIRMED, EMERGING, TACTICAL_CORRECTION, STRUCTURAL_DECAY,
    LOST, UNRESOLVED.
    Umbrales: S>0.20, Breadth>0.35, Persistence>0.50. Tactical no es decisorio.

## Salidas

- Estado (CONFIRMED, EMERGING, TACTICAL_CORRECTION, STRUCTURAL_DECAY, LOST, UNRESOLVED).
- Leader Breadth, LIS, Flow Divergence 2.0, Effective Breadth.
- LQ Dimensions (P, C, S, Cf).
- Seccion completa en el reporte bajo 'Structural Leadership (SLPM v1.2)'.


## Umbrales de la State Machine
| Parametro | Valor |
|---|---|
| breadth_max_decay | 0.35 |
| persistence_max_emerging | 0.50 |
| persistence_min_confirmed | 0.50 |
| structural_min_confirmed | 0.20 |
| structural_min_emerging | 0.20 |
