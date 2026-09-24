#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Health check del sistema Radar.

Verifica 7 bloques:
  A. Workflows: ultima ejecucion por schedule dentro de ventana.
  B. Cache 13F (IAE): trimestre actualizado.
  C. Manifests: quality.status de stock_prices y market_data.
  D. Cobertura ultimas 5 sesiones del parquet.
  E. Fechas no bursatiles en el indice del parquet.
  F. Patron contaminacion Europa-USA en ultima fila.
  G. Seccion IAE en el ultimo reporte generado.

Uso:
    py scripts/health_check.py              # ejecucion local, salida consola
    py scripts/health_check.py --json       # salida JSON
    py scripts/health_check.py --no-issues  # no abre/cierra issues (local)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

REPO = "bledabladis-png/radar-cazador-silencioso"

# --- Umbrales ---
COVERAGE_FAIL_THRESHOLD = 0.50
COVERAGE_WARN_THRESHOLD = 0.80
COVERAGE_OK_THRESHOLD = 0.95

# Fechas bursatiles con cobertura historicamente incompleta.
# Uso exclusivo en check_coverage_last_5 para no alertar de un
# defecto ya documentado. NO excluye la fecha de otros controles.
# 2026-09-22: fallo puntual de Yahoo. 206/313 tickers USA sin Close
# en esa sesion. Solo 36 llegaron (los top de cada sector).
CONFIRMED_INCOMPLETE_DATES = {"2026-09-22"}

WORKFLOW_EXPECTATIONS = {
    "daily_run.yml": {"max_days": 2},
    "update_macro_manual.yml": {"max_days": 2},
    "update_sector_holdings.yml": {"max_days": 100},
    "update_index_holdings.yml": {"max_days": 100},
    "update_european_holdings.yml": {"max_days": 100},
    "update_sec_nport.yml": {"max_days": 100},
    "update_sec_13f.yml": {"max_days": 100},
    "update_qqq_sec_flow.yml": {"max_days": 200},
}

OK, WARN, FAIL, SKIP = "OK", "WARN", "FAIL", "SKIP"


class Result:
    __slots__ = ("name", "status", "message")

    def __init__(self, name: str, status: str, message: str):
        self.name = name
        self.status = status
        self.message = message

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "message": self.message}


def _run_gh(args: list) -> str | None:
    try:
        r = subprocess.run(
            ["gh"] + args, capture_output=True, text=True, timeout=30
        )
        if r.returncode != 0:
            return None
        return r.stdout.strip()
    except Exception:
        return None

# =========================================================
# CHECK A - Workflows: ultima ejecucion por schedule
# =========================================================
def check_workflows() -> list:
    results = []
    for yml, meta in WORKFLOW_EXPECTATIONS.items():
        out = _run_gh([
            "run", "list", "--workflow", yml, "--event", "schedule",
            "--limit", "1", "--json", "createdAt,conclusion,status",
        ])
        if out is None:
            results.append(Result(f"workflow:{yml}", SKIP, "gh no disponible"))
            continue
        try:
            runs = json.loads(out)
        except Exception:
            results.append(Result(f"workflow:{yml}", WARN, "JSON invalido"))
            continue
        if not runs:
            results.append(Result(f"workflow:{yml}", WARN,
                "sin ejecuciones por schedule en el historial"))
            continue
        last = runs[0]
        try:
            created = datetime.fromisoformat(last["createdAt"].replace("Z", "+00:00"))
            now = datetime.now(created.tzinfo)
            delta_days = (now - created).total_seconds() / 86400
        except Exception:
            results.append(Result(f"workflow:{yml}", WARN, "timestamp ilegible"))
            continue
        conclusion = last.get("conclusion") or "unknown"
        if delta_days > meta["max_days"]:
            results.append(Result(f"workflow:{yml}", FAIL,
                f"ultima schedule hace {delta_days:.1f}d (>{meta['max_days']}d)"))
        elif conclusion == "failure":
            results.append(Result(f"workflow:{yml}", WARN,
                f"ultima schedule fue failure hace {delta_days:.1f}d"))
        else:
            results.append(Result(f"workflow:{yml}", OK,
                f"{delta_days*24:.0f}h atras ({conclusion})"))
    return results


# =========================================================
# CHECK B - Cache 13F (IAE)
# =========================================================
def _expected_quarters(today: date, n: int = 4) -> list:
    """Ultimos n trimestres cerrados (Q-1, Q-2, ...)."""
    out = []
    y, q = today.year, (today.month - 1) // 3 + 1
    for _ in range(n):
        q -= 1
        if q == 0:
            q = 4
            y -= 1
        out.append(f"{y}Q{q}")
    return out


def check_13f_cache() -> list:
    p = PROJECT_ROOT / "data" / "sec_13f" / "latest_quarter.txt"
    if not p.exists():
        return [Result("cache_13f", FAIL, "latest_quarter.txt no existe")]
    content = p.read_text(encoding="utf-8").strip()
    expected = _expected_quarters(date.today(), n=4)
    if content in expected:
        return [Result("cache_13f", OK,
            f"{content} (esperados: {expected[:3]})")]
    return [Result("cache_13f", WARN,
        f"{content} no en {expected[:3]}")]


# =========================================================
# CHECK C - Manifest quality
# =========================================================
def check_manifest(name: str) -> list:
    p = PROJECT_ROOT / "data" / f"{name}.parquet.manifest.json"
    if not p.exists():
        return [Result(f"manifest:{name}", FAIL, "manifest no existe")]
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return [Result(f"manifest:{name}", FAIL, f"JSON invalido: {e}")]
    q = m.get("quality", {})
    status = q.get("status", "UNKNOWN")
    last = q.get("last_date", "?")
    expected = q.get("expected_session", "?")
    nan_last = q.get("close_nan_last", 0)
    if status == "INVALID":
        return [Result(f"manifest:{name}", FAIL,
            f"INVALID (last={last} expected={expected})")]
    if status == "VALID_WITH_MISSING":
        return [Result(f"manifest:{name}", WARN,
            f"VALID_WITH_MISSING (close_nan_last={nan_last})")]
    if status == "VALID":
        return [Result(f"manifest:{name}", OK, f"VALID (last={last})")]
    return [Result(f"manifest:{name}", WARN, f"status={status}")]

# =========================================================
# CHECK D - Cobertura ultimas 5 sesiones
# =========================================================
def check_coverage_last_5(df: pd.DataFrame) -> list:
    close_cols = [c for c in df.columns if isinstance(c, tuple) and c[0] == "Close"]
    if not close_cols:
        return [Result("coverage:last", FAIL, "no hay columnas Close")]
    n_total = len(close_cols)
    tail = df.tail(5)
    coverages = []
    for idx, row in tail.iterrows():
        n_valid = int(row[close_cols].notna().sum())
        coverages.append((str(idx.date()), n_valid / n_total))
    results = []
    last_date, last_cov = coverages[-1]
    if last_cov < COVERAGE_FAIL_THRESHOLD:
        results.append(Result("coverage:last", FAIL,
            f"{last_date}: {last_cov:.1%} ({int(last_cov*n_total)}/{n_total})"))
    elif last_cov < COVERAGE_OK_THRESHOLD:
        results.append(Result("coverage:last", WARN,
            f"{last_date}: {last_cov:.1%}"))
    else:
        results.append(Result("coverage:last", OK,
            f"{last_date}: {last_cov:.1%}"))
    # Fechas marcadas como historicamente incompletas no cuentan como
    # defecto actual (fueron un fallo puntual documentado).
    bad_hist = [(d, c) for d, c in coverages[:-1]
                if c < COVERAGE_WARN_THRESHOLD
                and d not in CONFIRMED_INCOMPLETE_DATES]
    if bad_hist:
        detail = ", ".join(f"{d}={c:.0%}" for d, c in bad_hist)
        results.append(Result("coverage:hist", WARN,
            f"{len(bad_hist)}/4 filas < {COVERAGE_WARN_THRESHOLD:.0%}: {detail}"))
    else:
        n_checked = sum(1 for d, _ in coverages[:-1]
                        if d not in CONFIRMED_INCOMPLETE_DATES)
        excluded = len(coverages[:-1]) - n_checked
        if excluded:
            results.append(Result("coverage:hist", OK,
                f"{n_checked} filas OK ({excluded} excluidas por incompletez documentada)"))
        else:
            results.append(Result("coverage:hist", OK,
                f"{n_checked} filas historicas OK"))
    return results


# =========================================================
# CHECK E - Fechas no bursatiles en el indice
# =========================================================
def check_non_market_days(df: pd.DataFrame) -> list:
    from src.market_calendar import is_market_day
    non_market = []
    for idx in df.index:
        try:
            d = idx.date() if hasattr(idx, "date") else idx
            if not is_market_day(d):
                non_market.append(str(d))
        except Exception:
            continue
    if non_market:
        return [Result("fechas_no_bursatiles", WARN,
            f"{len(non_market)} fechas no bursatiles: {non_market[:5]}")]
    return [Result("fechas_no_bursatiles", OK,
        f"{len(df)} fechas todas bursatiles")]


# =========================================================
# CHECK F - Patron contaminacion Europa-USA en ultima fila
# =========================================================
def check_eu_usa_pattern(df: pd.DataFrame) -> list:
    from src.instrument_registry import get_market
    close_cols = [c for c in df.columns if isinstance(c, tuple) and c[0] == "Close"]
    if not close_cols or len(df) == 0:
        return [Result("eu_usa_pattern", SKIP, "sin datos")]
    last = df.iloc[-1]
    by_market = {}
    for col in close_cols:
        ticker = col[1]
        try:
            market = get_market(ticker)
        except Exception:
            market = "UNKNOWN"
        slot = by_market.setdefault(market, {"n": 0, "valid": 0})
        slot["n"] += 1
        if not pd.isna(last[col]):
            slot["valid"] += 1
    eu_markets = {"EURONEXT", "XETRA", "BME"}
    usa_markets = {"US_EQUITY"}
    def _cov(markets):
        n = sum(v["n"] for k, v in by_market.items() if k in markets)
        v = sum(v["valid"] for k, v in by_market.items() if k in markets)
        return (v / n) if n else None
    eu_cov = _cov(eu_markets)
    usa_cov = _cov(usa_markets)
    eu_s = f"{eu_cov:.0%}" if eu_cov is not None else "N/A"
    usa_s = f"{usa_cov:.0%}" if usa_cov is not None else "N/A"
    if eu_cov is not None and usa_cov is not None and eu_cov > 0.8 and usa_cov < 0.2:
        return [Result("eu_usa_pattern", WARN,
            f"K-STOCK-PRICES-EOD-01 (EU={eu_s} USA={usa_s})")]
    return [Result("eu_usa_pattern", OK, f"EU={eu_s} USA={usa_s}")]


# =========================================================
# CHECK G - Seccion IAE en el ultimo reporte
# =========================================================
def check_iae_section() -> list:
    p = PROJECT_ROOT / "outputs" / "report" / "reporte_diario.md"
    if not p.exists():
        return [Result("iae_section", WARN, "reporte_diario.md no existe")]
    text = p.read_text(encoding="utf-8", errors="replace")
    if "## Acumulacion Institucional (13F)" not in text:
        return [Result("iae_section", WARN, "seccion IAE no encontrada")]
    if "STALE" in text and "Official List" in text:
        return [Result("iae_section", OK, "STALE (official_list_pending) - esperado")]
    if "STALE" in text:
        return [Result("iae_section", WARN, "STALE con razon desconocida")]
    return [Result("iae_section", OK, "seccion presente")]

# =========================================================
# ORQUESTACION
# =========================================================
def run_all_checks() -> list:
    results = []
    # A
    results.extend(check_workflows())
    # B
    results.extend(check_13f_cache())
    # C
    results.extend(check_manifest("stock_prices"))
    results.extend(check_manifest("market_data"))
    # D, E, F - requieren parquet
    sp_path = PROJECT_ROOT / "data" / "stock_prices.parquet"
    if sp_path.exists():
        try:
            df = pd.read_parquet(sp_path)
            results.extend(check_coverage_last_5(df))
            results.extend(check_non_market_days(df))
            results.extend(check_eu_usa_pattern(df))
        except Exception as e:
            results.append(Result("parquet", FAIL, f"error leyendo parquet: {e}"))
    else:
        results.append(Result("parquet", FAIL, "stock_prices.parquet no existe"))
    # G
    results.extend(check_iae_section())
    return results


def _summary_counts(results: list) -> dict:
    counts = {OK: 0, WARN: 0, FAIL: 0, SKIP: 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    return counts


def _overall_status(results: list) -> str:
    counts = _summary_counts(results)
    if counts[FAIL] > 0:
        return FAIL
    if counts[WARN] > 0:
        return WARN
    return OK


def _render_markdown(results: list, now_utc: datetime) -> str:
    lines = [f"## Health Check - {now_utc.strftime('%Y-%m-%d %H:%M UTC')}", ""]
    lines.append("| Check | Estado | Detalle |")
    lines.append("|---|---|---|")
    icon = {OK: "[OK]", WARN: "[WARN]", FAIL: "[FAIL]", SKIP: "[SKIP]"}
    for r in results:
        lines.append(f"| {r.name} | {icon.get(r.status, r.status)} | {r.message} |")
    lines.append("")
    c = _summary_counts(results)
    lines.append(f"**Resumen:** {c[OK]} OK · {c[WARN]} WARN · {c[FAIL]} FAIL · {c[SKIP]} SKIP")
    return "\n".join(lines)


# =========================================================
# Issue management (GitHub Actions)
# =========================================================
ISSUE_LABEL = "health-check"
ISSUE_TITLE = "[health-check] Discrepancias detectadas"


def _ensure_label():
    subprocess.run(
        ["gh", "label", "create", ISSUE_LABEL, "--force",
         "--description", "Health check automatico del sistema",
         "--color", "0e8a16"],
        capture_output=True, text=True, timeout=15,
    )


def _find_open_issue() -> int | None:
    out = _run_gh([
        "issue", "list", "--label", ISSUE_LABEL, "--state", "open",
        "--json", "number", "--limit", "1",
    ])
    if not out:
        return None
    try:
        items = json.loads(out)
        return items[0]["number"] if items else None
    except Exception:
        return None


def _update_issue(overall: str, body: str):
    _ensure_label()
    existing = _find_open_issue()
    if overall == OK:
        if existing:
            _run_gh(["issue", "close", str(existing),
                     "--comment", "Health check OK - todo recuperado."])
            print(f"[health] issue #{existing} cerrada (recuperado)")
        return
    title_prefix = "[health-check]"
    if overall == FAIL:
        title = f"{title_prefix} FAIL - discrepancias criticas"
    else:
        title = f"{title_prefix} WARN - discrepancias no criticas"
    if existing:
        _run_gh(["issue", "comment", str(existing), "--body", body])
        _run_gh(["issue", "edit", str(existing), "--title", title])
        print(f"[health] issue #{existing} actualizada")
    else:
        r = subprocess.run(
            ["gh", "issue", "create", "--title", title, "--body", body,
             "--label", ISSUE_LABEL],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            print(f"[health] issue creada: {r.stdout.strip()}")


# =========================================================
# MAIN
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="salida JSON")
    ap.add_argument("--no-issues", action="store_true",
                    help="no abre/cierra issues (uso local)")
    args = ap.parse_args()

    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    results = run_all_checks()
    overall = _overall_status(results)
    counts = _summary_counts(results)

    md = _render_markdown(results, now_utc)
    print(md)
    print()

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as f:
            f.write(md + "\n")

    if args.json:
        print(json.dumps({
            "overall": overall,
            "counts": counts,
            "results": [r.to_dict() for r in results],
        }, indent=2))

    if not args.no_issues:
        try:
            _update_issue(overall, md)
        except Exception as e:
            print(f"[health] WARN: no se pudo actualizar issue: {e}")

    sys.exit(1 if overall == FAIL else 0)


if __name__ == "__main__":
    main()