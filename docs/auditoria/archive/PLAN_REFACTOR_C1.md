# Plan de refactor: src/report_generator.py (C1)

**Fecha:** 2026-09-11
**Estado:** EN EJECUCION (local-first, push al final)
**Objetivo:** Dividir `generate_daily_report` (1271 lineas, 444 appends, 64 parametros)
en modulos por seccion.

## Politica de ejecucion

- Local-first: NO se hace push hasta que TODO este verificado.
- Backup obligatorio: `report_generator.py.orig` antes de tocar.
- Snapshot del reporte: `_snapshot_reporte_antes_c1.md` (41 KB).
- Verificacion entre fases: 42 tests + diff vs snapshot.
- Si falla algo: rollback al `.orig` y diagnostico.

## Arquitectura objetivo

    src/report/
    |-- __init__.py       expone generate_daily_report
    |-- helpers.py        _fmt_num, _classify_*_freshness, _generate_coverage_table
    |-- header.py         Header + Resumen Regimenes
    |-- freshness.py      Data Freshness + Cross-Module
    |-- breadth.py        Breadth & Health + Concentracion
    |-- rankings.py       Rankings + Persistencia
    |-- flows.py          Flow chars + Divergencia + Liderazgo + Momentum amplitud
    |-- synthesis.py      Sintesis + Volatilidad + Confirmation + Dark Pools
    +-- generator.py      Orquestador (~150 lineas)

## Orden de extraccion (menor a mayor riesgo)

1. Helpers (ya aislados, 37 tests los cubren).
2. Resumen Regimenes (piloto simple).
3. Data Freshness.
4. Dark Pools.
5. Breadth + Concentracion.
6. Rankings + Persistencia.
7. Flows (varias).
8. Sintesis + Volatilidad + Confirmation.
9. Cross-Module.

## Contrato de modulos

Cada modulo expone:

    def render_<seccion>(<inputs>) -> list[str]:
        '''Devuelve lineas markdown. Sin side effects.'''

`generator.py` encadena con `lines.extend(...)`.

## Decisiones

1. Sin side effects en modulos. Los `to_csv` finales (sector_rankings.csv,
   historial de regimenes) se mueven a `run.py`.
2. Firma de `generate_daily_report` sin cambios (compatibilidad con run.py).
3. Sin `**kwargs` en render_*: cada modulo declara solo lo que usa.
4. `lines.append` -> `lines.extend(list)`.

## Criterios de aceptacion

- [ ] 42 tests pasan (helpers + smoke).
- [ ] Reporte generado coincide byte-a-byte con snapshot (modulo timestamps).
- [ ] `py run.py` corre sin errores.
- [ ] `py -m compileall .` sin errores.
- [ ] `py -m pyflakes src/report/*.py` sin warnings.
- [ ] Validation Gate 10/10.

## Rollback

    Copy-Item src/report_generator.py.orig src/report_generator.py -Force
    Remove-Item src/report/ -Recurse -Force
