# Plan de refactor: run.py (C2)

**Fecha:** 2026-09-11
**Estado:** EN EJECUCION (local-first, push al final)
**Objetivo:** Reducir run.py de 1286 lineas (main = ~1237 lineas) a un
orquestador de ~150-200 lineas, extrayendo las 13 fases del pipeline a
modulos en src/pipeline/.

## Politica de ejecucion

- Local-first: NO push hasta verificar todo.
- Backup obligatorio: run.py.orig antes de tocar.
- Snapshot del reporte: _snapshot_reporte_antes_c2.md.
- Verificacion entre fases: pyflakes + tests + compileall.
- Smoke run tras fases criticas (C2-8, C2-10, C2-14).
- Si falla: rollback quirurgico.

## Arquitectura objetivo

    src/pipeline/
    |-- __init__.py
    |-- data_load.py           Fase 1: descarga + validacion + macro manual
    |-- regimes.py             Fase 2: Financial + Liquidity + Volatility + Macro
    |-- sectors_base.py        Fase 3: rankings sectoriales + price/flow + rotacion
    |-- flows_primary.py       Fase 4: SSGA + BlackRock + Amundi + QQQ SEC + CFTC
    |-- flows_secondary.py     Fase 5: N-PORT + QQQ Yahoo + NPORT-P + sintesis
    |-- leaders.py             Fase 6: leader selection + divergencia + wyckoff + RS
    |-- sector_context.py      Fase 7: concentracion + reps + breadth + momentum
    |-- engines.py             Fase 8: tactical + structural + persistence + SLPM
    |-- diagnostics.py         Fase 9: agreement + price-flow + shock + vol + calidad
    |-- mte_confirmation.py    Fase 10: MTE + cross-module + confirmation
    |-- indices_intl.py        Fase 11: Wyckoff + oportunidades internacionales
    |-- validation_gate.py     Fase 12: Gate 10/10
    +-- finalize.py            Fase 13: matrices + reporte + cobertura

main() queda como orquestador que llama a las fases en orden.

## Contrato de modulos

Cada funcion de fase:
- Recibe parametros explicitos (sin Context global).
- Devuelve tupla/dict con los outputs que la siguiente fase necesita.
- No tiene side effects ocultos (excepto los ya identificados).

## Orden de extraccion

1. data_load (bajo riesgo)
2. regimes (bajo)
3. sectors_base (medio)
4. flows_primary (medio)
5. flows_secondary (medio)
6. leaders (alto)
7. sector_context (alto)
8. engines (alto)
9. diagnostics (alto)
10. mte_confirmation (medio)
11. indices_intl (medio)
12. validation_gate + finalize (alto)

## Criterios de aceptacion

- [ ] 106 tests pasan.
- [ ] Reporte generado coincide con _snapshot_reporte_antes_c2.md (94%+).
- [ ] py run.py completo sin errores.
- [ ] py -m compileall . -q sin errores.
- [ ] py -m pyflakes . sin warnings.
- [ ] Validation Gate 10/10.

## Rollback

    Copy-Item run.py.orig run.py -Force
    Remove-Item src/pipeline/ -Recurse -Force
