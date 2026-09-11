# Scripts archivados

Scripts históricos conservados por trazabilidad. **No se invocan desde workflows ni desde `run.py`.**

## Contenido

| Script | Motivo de archivo | Fecha archivado |
|--------|-------------------|:---------------:|
| `backfill_sector_breadth.py` | Backfill puntual de `sector_breadth.csv` | 07-sep-2026 |
| `backfill_sector_rank_history.py` | Backfill puntual de `sector_rank_history.csv` | 07-sep-2026 |
| `download_sec_nport.py` | Descarga legacy N-PORT, sustituido por `scripts/update_sec_nport_data.py` (cascada de trimestres) | 11-sep-2026 |

## Reactivación

Si en el futuro se necesitan:

1. Mover el script de vuelta a `scripts/`.
2. Verificar que sus dependencias siguen válidas (paths, formatos de CSV, APIs).
3. Actualizar referencias en workflows si aplica.

## Verificación previa al archivado

Antes de archivar se verificó (grep global) que:

- Ningún workflow los invoca.
- Ningún script los invoca vía `subprocess`.
- `run.py` no los menciona.
- Documentación no depende de ellos.
