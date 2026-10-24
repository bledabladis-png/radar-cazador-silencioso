"""Anotaciones sobre las lineas del reporte generadas en otros modulos.

Postprocesado: no modifica los datos que producen las secciones, solo
añade contexto textual que el productor original no podia conocer.
"""
from __future__ import annotations

import pandas as pd


PHASE_ORDER = [
    ("accumulation", "Acc"),
    ("accumulation_forming", "AccF"),
    ("markup", "Markup"),
    ("range", "Range"),
    ("distribution_forming", "DistF"),
    ("distribution", "Dist"),
    ("markdown", "Markdown"),
    ("conflict", "Conflict"),
]


def _composition_line(row) -> str | None:
    """Construye la linea de composicion interna de un sector a partir de
    una fila de sector_wyckoff_distribution_df. Devuelve None si no hay
    componentes con datos."""
    n = int(row.get("n_valid_wyckoff", 0) or 0)
    if n == 0:
        return None
    parts = []
    for key, label in PHASE_ORDER:
        cnt = row.get(f"count_{key}", 0)
        pct = row.get(f"pct_{key}", 0)
        if pd.notna(cnt) and cnt > 0 and pd.notna(pct):
            parts.append(f"{int(cnt)} {label} ({pct:.0f}%)")
    if not parts:
        return None
    return f"*Composicion interna ({n} componentes): {', '.join(parts)}*"


def annotate_sector_titles_with_composition(leader_lines, sector_wyckoff_distribution_df):
    """Anota cada '## Sector: XXX (FASE)' con una linea de composicion
    interna extraida de sector_wyckoff_distribution_df.

    La insercion va justo despues del titulo (antes de la tabla de lideres).
    Idempotente: si el titulo ya tiene una linea 'Composicion interna'
    inmediatamente debajo, no añade otra.

    Si el df es None o esta vacio, devuelve leader_lines sin modificar.
    Si un sector del titulo no aparece en el df, se deja sin anotar.
    """
    if leader_lines is None or sector_wyckoff_distribution_df is None:
        return leader_lines
    if sector_wyckoff_distribution_df is None or sector_wyckoff_distribution_df.empty:
        return leader_lines

    # Indexar por sector (última fila si hay duplicados historicos)
    df = sector_wyckoff_distribution_df
    if "date" in df.columns:
        df = df.sort_values("date").drop_duplicates("sector", keep="last")
    by_sector = df.set_index("sector", drop=False)

    result = []
    i = 0
    while i < len(leader_lines):
        line = leader_lines[i]
        result.append(line)
        # Detectar titulo "## Sector: XXX (FASE)"
        if isinstance(line, str) and line.startswith("## Sector:"):
            # Extraer sector
            sector = None
            try:
                rest = line[len("## Sector:"):].strip()
                sector = rest.split("(")[0].strip()
            except (ValueError, IndexError):
                sector = None
            # Idempotencia: no duplicar si ya hay composicion en las
            # siguientes lineas no vacias tras el titulo (hasta 3 saltos).
            already_annotated = False
            for offset in range(1, 4):
                j = i + offset
                if j >= len(leader_lines):
                    break
                cand = leader_lines[j]
                if not isinstance(cand, str):
                    break
                if "Composicion interna" in cand:
                    already_annotated = True
                    break
                if cand.strip() and not cand.startswith("## Sector:"):
                    # Linea con contenido que no es composicion -> no esta anotado
                    break
            if sector and sector in by_sector.index and not already_annotated:
                row = by_sector.loc[sector]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                comp = _composition_line(row)
                if comp:
                    result.append("\n")
                    result.append(comp)
                    result.append("\n")
        i += 1
    return result