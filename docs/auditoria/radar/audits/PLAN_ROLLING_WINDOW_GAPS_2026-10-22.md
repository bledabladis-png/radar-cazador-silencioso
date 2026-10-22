# Plan de regresion - Rolling window gaps

**Item:** REP-AUDIT-20261022-ROLLING-WINDOW-GAPS
**Origen:** Auditoria externa #4, hallazgo P7.
**Fecha:** 2026-10-22
**Estado:** OPEN.

## 1. Contexto y correccion de alcance

El item original dice: "Aplica a sector_dispersion, sector_correlation_summary y modulos con ventanas derivadas."

Verificacion 2026-10-22:
- `sector_dispersion.py` **no usa rolling()**. Calcula dispersion
  cross-sectional puntual de retornos 20d (una fila por dia). No es
  vulnerable al problema descrito.
- `sector_correlation_summary` **existe como writer y CSV**
  (`outputs/history/sector_correlation_summary.csv`, en
  `output_registry` desde 2026-10-11). El modulo Python que lo
  calcula es `indicators/sector_correlation.py`; pendiente verificar
  si usa rolling() y si es vulnerable al problema descrito.
- "Modulos con ventanas derivadas": 13 modulos usan `.rolling()` en
  codigo productivo (ver seccion 2).

**Alcance corregido:** modulos con `.rolling()` sin `min_periods`
explicito, cuya serie de entrada pueda tener discontinuidades. Los
huecos conocidos (2026-09-21, 22, 10-05) estan en
`sector_dispersion.csv`, pero ese modulo no usa rolling. El problema
puede afectar a otros modulos que consumen series con huecos.

## 2. Inventario de modulos con rolling()

| Modulo | Lineas con rolling | min_periods | Riesgo |
|---|---|---|---|
| indicators/breadth_core.py | 34-35 | si (window) | bajo |
| indicators/breadth_equity.py | 95 | si (10) | bajo |
| indicators/commodity_market_correlation.py | 28-29 | **no** | **alto** |
| indicators/index_leaders.py | 90-91 | **no** | **alto** |
| indicators/momentum.py | 26-27, 50 | si (var), **no** (50) | medio |
| indicators/options.py | 161 | si (20) | bajo |
| indicators/sector_breadth.py | 156-157 | si (252) | bajo |
| indicators/stock_leader.py | 114-116, 139, 149 | si (var), **no** (139, 149) | medio |
| indicators/volatility.py | 25-27, 60 | **no** (25, 60) | **alto** |
| indicators/volatility_structure.py | 36, 40 | si (15, 45) | bajo |
| indicators/vol_metrics.py | 21, 25 | si (10, 30) | bajo |
| indicators/wyckoff_v1.py | 97, 108, 110 | si (window) | bajo |
| src/utils.py | 3 usos | verificar | medio |

Los modulos marcados "alto" son los candidatos prioritarios del
test de regresion.

## 3. Huecos conocidos y pendientes

- `outputs/history/sector_dispersion.csv`: 3 huecos (2026-09-21, 22,
  10-05). Fix A-03 (2026-10-07) previene nuevos huecos, no rellena
  los anteriores.
- `outputs/history/sector_breadth.csv`: 1244 filas, 2026-05-06 a
  2026-10-20. Pendiente: detectar huecos por dia unico.
- `outputs/history/cross_asset_correlation.csv`: Pendiente: verificar
  estructura y huecos.
- Otras series: pendiente de inventario.

## 4. Test de regresion propuesto

Para cada modulo candidato (rolling sin min_periods):

1. Construir serie A: tal como esta (con huecos).
2. Construir serie B: reindexada al calendario explicito de sesiones
   (con NaN en huecos).
3. Calcular rolling(N) sobre A y sobre B.
4. Comparar resultados en fechas donde ambas tienen datos.
5. Si difieren > umbral (p.ej. 1e-6), documentar.

El calendario explicito se obtiene de `src/market_calendar.py`
(verificar API exacta antes de implementar).

## 5. Opciones de solucion

- (a) Reindexar al calendario explicito + `min_periods` para forzar
  NaN cuando faltan datos. Coste: cambio en cada modulo candidato.
  Beneficio: rolling(N) cubre exactamente N sesiones, no N filas.
- (b) Documentar como limitacion conocida + nota en reporte. Coste:
  bajo. Beneficio: transparencia, no cambia comportamiento.

Criterio de eleccion: si el impacto es material (diferencia > umbral
en fechas recientes), aplicar (a). Si es cosmetico, (b).

## 6. Criterios de cierre

- [ ] Test implementado para al menos 1 modulo candidato.
- [ ] Resultados documentados (que modulos, que fechas, que magnitud).
- [ ] Decision aplicada (a o b) o documentada en REGISTRO.
- [ ] Si (a): commit con tests de regresion.
- [ ] Si (b): nota en el reporte y en el item.