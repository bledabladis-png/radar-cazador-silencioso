## Proposito
Calculan el Tactical Score (corto plazo) y el Structural Score (largo plazo) para cada sector.

## Arquitectura

- 	actical_engine.py: compute_tactical_score().
- structural_engine.py: compute_structural_score().
Ambos se usan en el Opportunity Map y en el SLPM.


## Formulas

**Tactical Score:** Calcula el Tactical Score combinando 5 componentes de corto plazo.
    Pesos: RS20(30%), Momentum20(25%), Flow(20%), Breadth20(15%), Aceleracion(10%).
    Resultado acotado a [-1, +1].
**Structural Score:** Calcula el Structural Score de largo plazo.
    Pesos: RS multi-ventana 63/126/252d (50%), Flow Structure (30%),
    Persistence (20%). Resultado acotado a [-1, +1].


## Salidas

- Tactical Score: valor entre -1 y +1.
- Structural Score: valor entre -1 y +1.
Ambos aparecen en las tablas de rankings y en el Opportunity Map.

