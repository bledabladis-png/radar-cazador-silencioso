"""Test AST: prohibe rolling(N, min_periods=N) con N>=60 sobre
series que puedan tener NaN dispersos.

Regresion H7 (auditoria externa 2026-10-15): rolling(N, min_periods=N)
exige que las N filas consecutivas de la ventana sean todas no-NaN. En
series con huecos (festivos USA/EU), ninguna ventana cumple y el
resultado es NaN en toda la serie. Verificado: 289/329 tickers
producen rolling(252)=NaN sobre stock_prices.parquet.

Excepciones permitidas:
- rolling sobre serie ya dropna() en la misma linea o linea anterior.
- rolling con min_periods < window.
- rolling(10) con min_periods distinto o relajado.
"""
from pathlib import Path
import ast


ROOT = Path(__file__).resolve().parent.parent
EXCLUDE_PARTS = {".venv", "venv", ".git", "__pycache__", "node_modules",
                 "outputs", "data", "docs", "tests"}

# Umbral: window >= 60 exige 60 filas no-NaN consecutivas.
# Por debajo no aplica (ventanas cortas se recuperan).
UMBRAL_WINDOW = 60


def _iter_py(root):
    for py in root.rglob("*.py"):
        if any(part in EXCLUDE_PARTS for part in py.parts):
            continue
        yield py


def _extraer_window_minp(node):
    """Extrae window y min_periods de una llamada rolling.

    Devuelve (win_valor, win_nombre, minp_valor, minp_nombre).

    - win_valor/minp_valor: valor literal si es ast.Constant, si no None.
    - win_nombre/minp_nombre: nombre del identifier si es ast.Name, si no None.

    Esto permite detectar dos casos:
      1) rolling(252, min_periods=252)         -> literales iguales
      2) rolling(window, min_periods=window)   -> mismo identifier
    """
    win_valor = win_nombre = minp_valor = minp_nombre = None

    # Primer arg posicional
    if node.args:
        a = node.args[0]
        if isinstance(a, ast.Constant):
            win_valor = a.value
        elif isinstance(a, ast.Name):
            win_nombre = a.id

    for kw in node.keywords:
        if kw.arg == "window":
            if isinstance(kw.value, ast.Constant):
                win_valor = kw.value.value
            elif isinstance(kw.value, ast.Name):
                win_nombre = kw.value.id
        if kw.arg == "min_periods":
            if isinstance(kw.value, ast.Constant):
                minp_valor = kw.value.value
            elif isinstance(kw.value, ast.Name):
                minp_nombre = kw.value.id

    return win_valor, win_nombre, minp_valor, minp_nombre


# Whitelist: lineas donde el rolling opera sobre series ya limpias
# aguas arriba en OTRA funcion o con ffill, y por diseno el NaN no existe.
# Cada entrada debe venir con justificacion. Añadir aqui solo tras verificar
# empiricamente que el patron no produce todo-NaN.
WHITELIST = {
    # commodity_market_correlation: ret_sector/ret_bench/ret_comm ya son
    # .pct_change().dropna() en lineas 15-17. La interseccion con .dropna()
    # aguas arriba garantiza que no hay NaN.
    ("indicators/commodity_market_correlation.py", 28),
    ("indicators/commodity_market_correlation.py", 29),
    # momentum.compute_cmf: high/low/close/volume aplican .ffill() al inicio.
    # Por diseno, la serie entra sin NaN al rolling.
    ("indicators/momentum.py", 50),
    # scripts/calibrate_wyckoff_sow_5b3.py: script one-shot congelado.
    # No productivo. No tocar sin dictamen.
    ("scripts/calibrate_wyckoff_sow_5b3.py", 105),
}


def _esta_en_whitelist(path_rel, lineno):
    return (path_rel, lineno) in WHITELIST


def _linea_contiene_dropna(fuente_lineas, lineno, ventana=5):
    """Comprueba si la linea o las `ventana` anteriores contienen .dropna().

    La ventana de 5 lineas permite el patron comun:
        _clean = series.dropna()
        ...
        _clean.rolling(N, min_periods=N)...
    Sin obligar a escribir dropna inline.
    """
    idx = lineno - 1
    inicio = max(0, idx - ventana)
    for i in range(inicio, min(idx + 1, len(fuente_lineas))):
        if ".dropna()" in fuente_lineas[i]:
            return True
    return False


def test_no_rolling_window_igual_min_periods_sobre_series_con_nan():
    """Detecta el patron rolling(N, min_periods=N) con N>=UMBRAL_WINDOW
    sobre lineas que NO aplican dropna previo.

    Si este test falla, alguien ha reintroducido el bug H7. Ver
    indicadores/breadth_core.py, sector_breadth.py, stock_leader.py,
    breadth_equity.py, wyckoff_v1.py para referencia del fix.
    """
    violaciones = []
    for py in _iter_py(ROOT):
        try:
            fuente = py.read_text(encoding="utf-8", errors="ignore")
            fuente_lineas = fuente.splitlines()
            tree = ast.parse(fuente)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr != "rolling":
                continue

            win_valor, win_nombre, minp_valor, minp_nombre = _extraer_window_minp(node)

            # Determinar window (literal o variable)
            win_repr = None
            win_es_larga = False
            if isinstance(win_valor, int):
                win_repr = str(win_valor)
                win_es_larga = win_valor >= UMBRAL_WINDOW
            elif win_nombre is not None:
                win_repr = win_nombre
                # Conservador: si es variable, asumimos que puede ser >= 60
                # (no podemos saberlo estaticamente sin analisis de contexto).
                win_es_larga = True

            if not win_es_larga:
                continue

            # Determinar min_periods efectivo
            minp_repr = None
            minp_igual_a_window = False

            if minp_valor is None and minp_nombre is None:
                # Sin min_periods explicito: pandas default = window.
                # Mismo problema potencial.
                minp_igual_a_window = True
                minp_repr = "(default=window)"
            elif isinstance(minp_valor, int) and isinstance(win_valor, int):
                # Ambos literales
                if minp_valor == win_valor:
                    minp_igual_a_window = True
                    minp_repr = str(minp_valor)
            elif minp_nombre is not None and win_nombre is not None:
                # Ambos variables con el mismo nombre
                if minp_nombre == win_nombre:
                    minp_igual_a_window = True
                    minp_repr = minp_nombre
            elif minp_nombre is not None and win_valor is None:
                # min_periods=variable, window=variable (identicos por convencion)
                if minp_nombre == win_nombre:
                    minp_igual_a_window = True
                    minp_repr = minp_nombre

            if not minp_igual_a_window:
                continue

            if _esta_en_whitelist(str(py.relative_to(ROOT)).replace("\\", "/"), node.lineno):
                continue

            if _linea_contiene_dropna(fuente_lineas, node.lineno):
                continue

            violaciones.append(
                f"{py.relative_to(ROOT)}:{node.lineno} "
                f"rolling({win_repr}, min_periods={minp_repr}) sin dropna previo"
            )

    assert not violaciones, (
        "Patron H7 detectado (rolling N=N sin dropna previo sobre series "
        "con NaN dispersos):\n" + "\n".join(violaciones)
    )


def test_whitelist_breadth_core_usa_serie_por_ticker():
    """breadth_core.compute_new_highs_lows debe iterar por columna
    (serie nativa por ticker), no aplicar rolling al DataFrame completo."""
    bc = ROOT / "indicators" / "breadth_core.py"
    fuente = bc.read_text(encoding="utf-8")
    # Debe contener el bucle por columna
    assert "for col in prices.columns:" in fuente, (
        "breadth_core.compute_new_highs_lows perdio el bucle por ticker. "
        "Regresion H7."
    )
    assert ".dropna()" in fuente, (
        "breadth_core.compute_new_highs_lows debe dropna por ticker."
    )
