## Proposito
Audita la calidad del liderazgo del sector #1 del ranking. No es otro ranking: evalua si el lider es estructuralmente solido.

## Arquitectura

- compute_leader_breadth_v2(): amplitud del liderazgo (RS, momentum, flujo, Wyckoff). Es la senhal decisoria de lideres en la State Machine.
- compute_leader_integrity(): LIS (intensidad/calidad de los lideres individuales). Es metrica de diagnostico, no participa en la clasificacion.
- compute_flow_divergence_v2(): divergencias entre flujo de lideres y sector.
- classify_leadership_state(): State Machine con 6 estados + UNRESOLVED. No recibe LIS.
- confirm_transition(): histeresis temporal.

## Formulas
**State Machine:** Clasifica el estado de liderazgo del sector #1 usando la State Machine.
    Estados: CONFIRMED, EMERGING, TACTICAL_CORRECTION, STRUCTURAL_DECAY,
    LOST, UNRESOLVED.
    Umbrales: T>0.20, S>0.20, Breadth>0.35, Persistence>0.50, etc.
    LIS esta excluido de la decision para evitar doble conteo con Breadth.

## Evidencia de redundancia Breadth/LIS

- Correlacion Spearman(Breadth, LIS) = 1.0 en la muestra evaluada.
- Tras confirmar redundancia perfecta, LIS se retiro de la State Machine.
- LIS permanece en el reporte y en input_scores como metrica de diagnostico.

## Sensibilidad de la State Machine (ablacion BASE contemporaneo)

| Componente | Sensibilidad | Interpretacion |
|---|---|---:|---|
| Structural | 51.2% | Principal discriminador |
| Breadth | 22.7% | Filtro de deterioro estructural |
| Tactical | 8.8% | Detector de correccion tactica |
| Persistence | N/M | No medible con los datos historicos actuales |
| LIS | eliminado | Diagnostico, no decisorio |

*Nota: La ablacion compara BASE vs variante con un componente neutralizado, no contra estados historicos antiguos.*

## Salidas

- Estado (CONFIRMED, EMERGING, TACTICAL_CORRECTION, STRUCTURAL_DECAY, LOST, UNRESOLVED).
- Leader Breadth, LIS (diagnostico), Flow Divergence 2.0, Effective Breadth.
- LQ Dimensions (P, C, S, Cf).
- Seccion completa en el reporte bajo 'Structural Leadership (SLPM v1.2)'.

## Pendientes de validacion

- Medir sensibilidad de Persistence con historico real.
- Medir sensibilidad de Coverage (0.25, 0.50, 0.75, 1.00).
- Ejecutar regresion BASE vs BASE+LIS para confirmar que LIS no cambia materialmente el estado.

## Umbrales de la State Machine

| Parametro | Valor |
|---|---|
| structural_min_confirmed | 0.20 |
| structural_min_emerging | 0.20 |
| structural_max_decay | -0.20 |
| structural_max_lost | -0.40 |
| tactical_max_correction | -0.20 |
| breadth_max_decay | 0.35 |
| persistence_min_confirmed | 0.50 |
| persistence_max_emerging | 0.50 |
