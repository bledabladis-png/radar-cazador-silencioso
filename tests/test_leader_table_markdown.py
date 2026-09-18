# -*- coding: utf-8 -*-
"""Test de regresion K-RENDER-LEADER-TABLE-01 (2026-09-18).

Bug: el separator Markdown de la tabla de lideres sectoriales
(indicators/stock_leader.py:180) tenia 8 columnas mientras el header
(L179) tenia 11. En GFM, un separator mas corto que el header hace
que las columnas sobrantes (Pers 20d, Spring, SOS) se ignoren al
renderizar.

Fix: separator alineado a 11 columnas.
"""
import re
from pathlib import Path


def test_separator_alineado_con_header():
    """El header y el separator deben tener el mismo numero de columnas."""
    f = Path(__file__).resolve().parent.parent / 'indicators' / 'stock_leader.py'
    text = f.read_text(encoding='utf-8-sig')

    # Buscar las lineas consecutivas header + separator
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if '| Ticker | RS | RS Mom | Flujo (z) | WLS | Fase Wyckoff | Pers 5d |' in line:
            header_idx = i
            break

    assert header_idx is not None, 'Header de tabla de lideres no encontrado'

    header = lines[header_idx]
    separator = lines[header_idx + 1]

    # Contar columnas por numero de pipes
    def count_cols(line):
        # Quitar el prefijo 'lines.append(' y sufijo ')' si aplica
        s = line.strip()
        # Extraer el string entre comillas simples
        m = re.search(r"'(.*)'", s)
        if m:
            s = m.group(1)
        return s.count('|') - 1

    n_header = count_cols(header)
    n_separator = count_cols(separator)

    assert n_header == n_separator, (
        f'Desalineacion: header={n_header} cols, separator={n_separator} cols'
    )
    assert n_header == 11, f'Esperado 11 columnas, obtenido {n_header}'