## Proposito
Documenta la arquitectura QQQ SEC Primary Flow y la relacion con Invesco.

## Arquitectura

                    QQQ
                     |
          +----------+-----------+
          |                      |
         SEC                  Invesco
          |                      |
          |               +------+-------+
          |               |              |
    FUENTE PRIMARIA   NAV historico   Shares
    DEL FLUJO         Performance     snapshot
          |
          v
   primary_flow
          |
          v
     RADAR v4.3

- SEC EDGAR es la fuente primaria oficial del flujo primario QQQ.
- Invesco es fuente secundaria complementaria para NAV, performance y snapshot de shares.
- El CI de GitHub no depende de Invesco para completar el reporte.

## Fuentes y periodicidad

| Dato | Fuente | Formulario/Endpoint | Periodicidad |
|------|--------|---------------------|--------------|
| Flujo primario QQQ | SEC EDGAR | N-30B-2 (anual), N-CSRS (semestral) | Anual/semestral |
| Flujo de participaciones QQQ | SEC EDGAR | NPORT-P Item B.6 | Trimestral |
| NAV historico | Invesco DNG API | /navs | Diario (local) |
| Performance | Invesco DNG API | /performance/standard | Local |
| Shares outstanding | Invesco DNG API | /prices | Snapshot local |

## Formulas

- shares_flow = shares_sold + shares_repurchased
- shares_change = shares_end - shares_beginning
- primary_flow_usd = proceeds_shares_sold + value_shares_repurchased
- Validacion: |shares_flow - shares_change| ~ 0
- Validacion: |primary_flow_usd - (proceeds_shares_sold + value_shares_repurchased)| ~ 0

## Salidas

- outputs/history/qqq_sec_primary_flow.csv
- outputs/history/qqq_nport_flow.csv
- outputs/history/invesco_qqq_*.csv (local, no CI)

## Integracion en CI

- daily_run.yml NO ejecuta el proveedor Invesco.
- El flujo QQQ SEC se actualiza mediante workflow semestral update_qqq_sec_flow.yml.
- La validacion de integridad esta en validation/validate_history_quality.py.
- El reporte muestra '## Flujo Primario QQQ (SEC, Trimestral/Semestral)' y '## Flujo de Participaciones QQQ (NPORT-P)'.
- La seccion '## Rendimiento QQQ (Invesco)' solo aparece si hay datos frescos (TTL 7 dias).

## Limitaciones

- No existe historico diario publico de shares outstanding de QQQ en Invesco.
- Invesco bloquea GitHub Actions con HTTP 406; por eso no se usa en CI.
- El snapshot de shares outstanding de Invesco es solo para cross-check, no para flujo primario diario.
