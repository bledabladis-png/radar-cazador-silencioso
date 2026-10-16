# Ubuntu runner pinning (2026-10-16)

## Contexto

GitHub anuncio (publicado 17-sep-2026) que el despliegue gradual de
Ubuntu 26.04 sobre `ubuntu-latest` **comenzara el 19-oct-2026 y
finalizara el 19-nov-2026**. Hasta esa fecha, `ubuntu-latest` resuelve
a Ubuntu 24.04 LTS.

**Nota cronologica:** cuando se redacto este documento (16-oct-2026)
la migracion aun no habia comenzado. El pin a `ubuntu-24.04` es una
medida **preventiva** anterior al despliegue, no una reaccion a un
fallo del CI.

## Riesgo

Los workflows del repo usan wheels binarios de Python con version
pinned:

- numpy 2.4.2
- pandas 2.3.3
- pyarrow 23.0.1
- lxml 6.0.2
- curl_cffi 0.16.0
- scipy 1.17.1
- scikit-learn 1.8.0
- matplotlib 3.10.8

Si un wheel no tiene build para Ubuntu 26 + CPython 3.14, `pip install
-r requirements.txt` falla en el step inicial del job. Con 5 slots
diarios del `daily_run`, eso encadena 5 runs rojos antes de que el
health check semanal (lunes 06:47 UTC) lo detecte.

## Decision

Pinneado explicito de **`ubuntu-24.04` LTS** en los 9 workflows.
Ubuntu 24.04 tiene soporte hasta 2029. Elimina la sorpresa del
despliegue gradual 19-oct a 19-nov 2026.

El cambio lo verifica `tests/test_workflows_runs_on_pinned.py`:
- Ningun workflow puede usar `ubuntu-latest` ni alias flotantes.
- Todos deben usar version explicita `ubuntu-XX.YY`.

## Migracion a Ubuntu 26

No inmediata. La migracion correcta es:

1. **Esperar** a que GitHub publique imagenes estables de 26 LTS.
2. **Verificar** disponibilidad de wheels en PyPI para Ubuntu 26 +
   CPython 3.14 (o version activa).
3. **Cambiar** los workflows de `ubuntu-24.04` a `ubuntu-26.04`.
4. **Verificar** en un dispatch manual: `gh workflow run daily_run.yml`.
5. **Actualizar** este documento con la fecha de migracion.

Sin prisas: Ubuntu 24.04 sigue soportada. La migracion se hace cuando
sea segura, no cuando GitHub lo insinue.

## Referencia

- Runner images releases: https://github.com/actions/runner-images/releases
- Migracion Ubuntu 26: https://github.com/actions/runner-images/issues/14748