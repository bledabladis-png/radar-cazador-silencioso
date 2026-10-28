"""Guard AST anti-P7: cierra la puerta a los patrones de look-ahead.

Auditoria AST 2026-10-28 (P7-ROLLING-WINDOW-GAPS). Inventario de
todo el codigo de produccion (src/, indicators/, validation/,
config/, run.py).

Patrones bloqueados:
- `iloc[-N:]` con N > 1. Riesgo: ventana por posicion fisica, no por
  sesion, si la serie tiene huecos. N=1 es valor unico, permitido.
  Helper: `src.market_calendar.slice_last_n_sessions`.
- `shift(-N)`: look-ahead directo.
- `fillna(method='bfill'|'backfill')`: rellena con futuro.
- `datetime.now()` / `pd.Timestamp.now()` en modulos de observacion.
  Solo permitido con whitelist explicita (cada entrada con razon
  >= 20 chars).

Ficheros escaneados: src/, indicators/, validation/, config/,
run.py. No se escanean tests/, scripts/, docs/.

Si un patron nuevo aparece, el test falla con la ruta, linea y
funcion exacta. Para anadir a la whitelist, documentar la razon
(es la unica forma de que el guard no se degrade a "todo OK").
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TARGET_DIRS = [ROOT / "src", ROOT / "indicators", ROOT / "validation", ROOT / "config"]
TARGET_FILES = [ROOT / "run.py"]


# =============================================================================
# Whitelist (cada entrada verificada manualmente 2026-10-28)
# =============================================================================
# Formato: (ruta_relativa_posix, func_stack_concatenado, valor_esperado)
# func_stack_concatenado = "fn_outer::fn_inner" o "<module>" si esta a
# nivel top-level.
DATETIME_NOW_WHITELIST = {
    # === Raiz de "ahora" del sistema ===
    # Unica funcion que define "ahora". run.py la invoca y propaga.
    ("src/market_calendar.py", "last_expected_market_date"):
        "Raiz de la cadena de 'ahora': define que es 'now' si el caller no lo pasa.",
    ("run.py", "main"):
        "Raiz del run: reference_date = now() al arrancar. Justificado en HANDOFF (datetime.now solo para ejecucion, no observacion).",
    # === Helper centralizado ===
    # resolve_reference_date agrupa el fallback defensivo. Los 7 sitios
    # pre-refactor (flows_secondary, mte_confirmation, validation_gate,
    # darkpool, freshness, helpers, sentiment) ya no llevan datetime.now
    # directo. Si en Fase 4 se hace reference_date required, se toca 1
    # sitio (este) en vez de 7.
    ("src/utils.py", "resolve_reference_date"):
        "Helper central del fallback reference_date-or-now. Unico punto con datetime.now() ademas de la raiz.",
    # === Fallbacks defensivos que no se pueden eliminar sin cambio de contrato ===
    # data_loader: reference_date = now() si None. Caller productivo
    # (run.py) siempre pasa. Mismo patron que resolve_reference_date.
    ("src/data_loader.py", "download_market_data"):
        "Fallback defensivo: reference_date = now() si None. Caller productivo pasa reference_date.",
    # breadth_metrics: mismo patron (reference_date = now() si None).
    ("src/pipeline/breadth_metrics.py", "_compute_sector_breadth_health"):
        "Fallback defensivo: reference_date = now() si None. Caller productivo pasa reference_date.",
    # stock_data_loader: download_stock_prices tiene dos fallbacks
    # (reference_date = now() si None y cache freshness contra now()).
    ("src/stock_data_loader.py", "download_stock_prices"):
        "Fallback defensivo (reference_date = now si None) + cache freshness contra now().",
    # macro_fundamental: variante con tz-check posterior.
    ("indicators/macro_fundamental.py", "fundamental_signals"):
        "Fallback a now() si reference_date None + normalizacion tz posterior. Caller productivo pasa reference_date.",
    # report_generator: fallback a now(Europe/Madrid) tz-aware.
    ("src/report_generator.py", "generate_daily_report"):
        "Fallback a now(Europe/Madrid) si reference_date None. El caller productivo pasa reference_date.",
    # regimes: fallback documentado en commit 86ff433c.
    ("src/pipeline/regimes.py", "_fetch_liquidity_v2_data"):
        "Fallback a now() si reference_date es None; en produccion run.py siempre lo pasa.",
    # === Cache freshness (contra now, correcto) ===
    # data_loader.py:273 mide si el CSV cacheado en disco es fresco
    # comparando contra now(). Correcto: el cache se creo en "ahora
    # real", no en la fecha de observacion.
    # stock_data_loader.py:616 idem.
    # (Misma funcion download_market_data, misma entrada).
    # === Logs / manifests / audit files ===
    # manifest.py:113 generated_at_utc del 13F. Log.
    ("src/institutional_accumulation/sec_13f/manifest.py", "build_manifest"):
        "generated_at_utc del manifest 13F: timestamp de generacion, no de observacion.",
    # provenance.py:127 generated_at del reporte. Log.
    ("src/report/provenance.py", "build_provenance"):
        "generated_at del reporte: timestamp de generacion, no de observacion.",
    # options.py:242 timestamp del bloque PCR. Log.
    ("indicators/options.py", "compute_pcr_signals"):
        "timestamp del bloque PCR: metadato de generacion, no de observacion.",
    # monitor_double_counting.py:51 run_iso del log. Script de validacion.
    ("validation/monitor_double_counting.py", "main"):
        "Script de validacion: run_iso del log, no observacion.",
    # === Classify freshness / validate history (correctos contra now) ===
    # stock_data_loader.py:401 clasifica ticker como OK/STALE/FAILED
    # comparando antiguedad contra now(). Aceptado por diseno: el
    # resultado depende de cuando se ejecuta, se usa solo en freshness.
    # Ticket P7-CLASSIFY-TICKER-NOW si se decide migrar a reference_date.
    ("src/stock_data_loader.py", "_classify_ticker"):
        "Aceptado por diseno: classify freshness contra now(); no derivable del dataset.",
    # data_quality.py:88 _ref_to_date(None) = hoy. Fallback. Caller
    # productivo siempre pasa ref.
    ("indicators/data_quality.py", "_ref_to_date"):
        "Fallback a now() si ref es None; el caller productivo siempre pasa ref.",
    # validate_history_quality.py:154 mide antiguedad de la historia
    # contra now(). Correcto: es su proposito.
    ("validation/validate_history_quality.py", "main"):
        "Script de validacion de historia: medir antiguedad contra now() es su proposito.",
}


# =============================================================================
# Scanner
# =============================================================================

def _archivos():
    out = []
    for d in TARGET_DIRS:
        if d.exists():
            out.extend(sorted(d.rglob("*.py")))
    for f in TARGET_FILES:
        if f.exists() and f not in out:
            out.append(f)
    return out


class _Scanner(ast.NodeVisitor):
    """Recorre un .py y acumula hallazgos con su func_stack."""

    def __init__(self, path):
        self.path = path
        self.stack = []
        self.hallazgos = {
            "ILOC_NEGATIVO": [],
            "SHIFT_NEGATIVO": [],
            "FILLNA_BFILL": [],
            "DATETIME_NOW": [],
        }

    def _push(self, node):
        self.stack.append(node.name)

    def _pop(self):
        self.stack.pop()

    def _record(self, tipo, node, **extra):
        self.hallazgos[tipo].append({
            "lineno": node.lineno,
            "func_stack": tuple(self.stack),
            **extra,
        })

    def visit_FunctionDef(self, node):
        self._push(node)
        self.generic_visit(node)
        self._pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        try:
            nombre = ast.unparse(node.func)
        except Exception:
            nombre = ""
        if nombre in ("datetime.now", "datetime.utcnow",
                      "pd.Timestamp.now", "pd.Timestamp.utcnow",
                      "Timestamp.now", "Timestamp.utcnow"):
            self._record("DATETIME_NOW", node, nombre=nombre)
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "shift" and node.args:
                a0 = node.args[0]
                if isinstance(a0, ast.UnaryOp) and isinstance(a0.op, ast.USub):
                    self._record("SHIFT_NEGATIVO", node)
            if node.func.attr == "fillna":
                for kw in node.keywords:
                    if kw.arg == "method":
                        try:
                            v = ast.literal_eval(kw.value)
                            if isinstance(v, str) and v.lower() in ("bfill", "backfill"):
                                self._record("FILLNA_BFILL", node)
                        except Exception:
                            pass
        self.generic_visit(node)

    def visit_Subscript(self, node):
        sl = node.slice
        if isinstance(sl, ast.Slice) and sl.lower is not None:
            lo = sl.lower
            if isinstance(lo, ast.UnaryOp) and isinstance(lo.op, ast.USub):
                if isinstance(lo.operand, ast.Constant) and isinstance(lo.operand.value, int):
                    if lo.operand.value > 1:
                        self._record("ILOC_NEGATIVO", node, N=lo.operand.value)
        self.generic_visit(node)


def _escanear_todo():
    """Devuelve dict: tipo -> lista de (ruta, lineno, func_stack, extra)."""
    out = {k: [] for k in (
        "ILOC_NEGATIVO", "SHIFT_NEGATIVO", "FILLNA_BFILL", "DATETIME_NOW",
    )}
    for p in _archivos():
        try:
            src = p.read_text(encoding="utf-8-sig", errors="replace")
            tree = ast.parse(src)
        except (OSError, SyntaxError):
            continue
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        s = _Scanner(rel)
        s.visit(tree)
        for tipo, hits in s.hallazgos.items():
            for h in hits:
                out[tipo].append((rel, h["lineno"], h["func_stack"], h))
    return out


# =============================================================================
# Tests
# =============================================================================

def test_no_iloc_negativo_literal():
    """Ningun `iloc[-N:]` con N > 1. Usar slice_last_n_sessions."""
    todo = _escanear_todo()
    hits = todo["ILOC_NEGATIVO"]
    assert not hits, (
        "Regresion P7: iloc[-N:] con N>1 sobre series temporales. "
        "Usar src.market_calendar.slice_last_n_sessions. "
        + "\n".join(f"{r}:{ln} funcs={fs} N={h['N']}" for r, ln, fs, h in hits)
    )


def test_no_shift_negativo():
    """Ningun shift(-N): look-ahead directo."""
    todo = _escanear_todo()
    hits = todo["SHIFT_NEGATIVO"]
    assert not hits, (
        "Regresion P7: shift(-N) look-ahead. "
        + "\n".join(f"{r}:{ln} funcs={fs}" for r, ln, fs, h in hits)
    )


def test_no_fillna_bfill():
    """Ningun fillna(method='bfill'|'backfill')."""
    todo = _escanear_todo()
    hits = todo["FILLNA_BFILL"]
    assert not hits, (
        "Regresion P7: fillna con bfill/backfill rellena con futuro. "
        + "\n".join(f"{r}:{ln} funcs={fs}" for r, ln, fs, h in hits)
    )


def test_datetime_now_solo_whitelist():
    """datetime.now()/Timestamp.now() solo en funciones whitelisted."""
    todo = _escanear_todo()
    whitelist_keys = set(DATETIME_NOW_WHITELIST.keys())
    no_whitelisted = []
    for r, ln, fs, h in todo["DATETIME_NOW"]:
        key = (r, "::".join(fs) if fs else "<module>")
        if key not in whitelist_keys:
            no_whitelisted.append((r, ln, fs, h["nombre"]))
    assert not no_whitelisted, (
        "datetime.now()/Timestamp.now() en funcion no whitelisted. "
        "Anadir a DATETIME_NOW_WHITELIST en este fichero con razon "
        "explicita, o refactorizar para aceptar reference_date. "
        + "\n".join(f"{r}:{ln} funcs={fs} {n}" for r, ln, fs, n in no_whitelisted)
    )


def test_whitelist_tiene_razon_documentada():
    """Cada entrada whitelisted tiene razon >= 20 chars. Evita
    degradar el guard a 'todo OK' sin justificar."""
    for (ruta, func), razon in DATETIME_NOW_WHITELIST.items():
        assert isinstance(razon, str), f"{ruta}::{func} razon no es str"
        assert len(razon) >= 20, (
            f"{ruta}::{func} razon < 20 chars: {razon!r}. "
            f"Explicar por que el now() es legitimo."
        )


def test_whitelist_apunta_a_sitios_existentes():
    """Cada (ruta, func) whitelisted existe en el codigo actual. Evita
    entradas obsoletas que no cubren nada."""
    errores = []
    for (ruta, func), _razon in DATETIME_NOW_WHITELIST.items():
        p = ROOT / ruta
        if not p.exists():
            errores.append(f"{ruta} no existe")
            continue
        src = p.read_text(encoding="utf-8", errors="replace")
        if func and func != "<module>":
            # El nombre de la funcion debe aparecer como def en el fichero
            if f"def {func}(" not in src:
                errores.append(f"{ruta}::{func} no encontrada en el fichero")
    assert not errores, "Whitelist obsoleta: " + "; ".join(errores)