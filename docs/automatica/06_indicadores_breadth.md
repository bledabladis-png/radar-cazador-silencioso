## Proposito
Mide la amplitud del mercado sectorial (porcentaje de sectores sobre sus EMAs) y detecta divergencias.

## Arquitectura

- compute_breadth(): porcentajes sobre EMA20, EMA50, EMA200.
- readth_equity.py: avances/descensos del mercado general.


## Formulas
**Breadth:** Calcula % de sectores sobre EMAs y nuevos maximos/minimos.

## Salidas

- % sobre EMA20/50/200 mostrado en la seccion Breadth de Mercado.
- Divergencias breadth en la seccion de Divergencias Detectadas.


## Constantes de Breadth
| Constante | Valor |
|---|---|
| BREADTH_EMA_FAST | 20 |
| BREADTH_EMA_MEDIUM | 50 |
| BREADTH_EMA_SLOW | 200 |
