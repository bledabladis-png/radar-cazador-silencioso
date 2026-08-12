# Motores Tactico y Estructural

## Tactical Engine
```
Calcula el Tactical Score combinando 5 componentes de corto plazo.
    Pesos: RS20(30%), Momentum20(25%), Flow(20%), Breadth20(15%), Aceleracion(10%).
    Resultado acotado a [-1, +1].
```

## Structural Engine
```
Calcula el Structural Score de largo plazo.
    Pesos: RS multi-ventana 63/126/252d (35%), Leader Breadth (25%),
    Flow Structure (20%), Persistence (20%). Resultado acotado a [-1, +1].
```
