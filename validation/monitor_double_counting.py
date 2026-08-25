
# -*- coding: utf-8 -*-
"""
Monitoreo continuo de dependencias entre señales.

Este script NO modifica pesos ni umbrales.
Solo calcula correlaciones clave y emite alertas informativas.

Salida:
    outputs/history/double_counting_monitor.csv
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_CSV = PROJECT_ROOT / "outputs" / "history" / "double_counting_monitor.csv"

UMBRALES = {
    "structural_vs_breadth": 0.85,
    "rs_mom_vs_flow": 0.80,
    "rs_mom_vs_wls": 0.85,
    "stability_vs_wyckoff": 0.85,
}

def _spearman(a, b):
    """Correlación Spearman segura entre dos series."""
    try:
        a = pd.to_numeric(a, errors="coerce")
        b = pd.to_numeric(b, errors="coerce")
        mask = a.notna() & b.notna()
        if mask.sum() < 10:
            return None
        return round(float(pd.Series(a[mask]).corr(pd.Series(b[mask]), method="spearman")), 4)
    except Exception:
        return None

def main() -> None:
    metrics = []
    timestamp = datetime.now().isoformat(timespec="seconds")
    files = {
        "leaders": PROJECT_ROOT / "outputs" / "report" / "analisis_lideres.csv",
        "slpm": PROJECT_ROOT / "outputs" / "history" / "slpm_history.csv",
    }

    # 1. Líderes sectoriales
    leaders_path = files["leaders"]
    if leaders_path.exists():
        try:
            df = pd.read_csv(leaders_path)
            checks = [
                ("rs_mom_vs_flow", "rs_mom", "flow_proxy_z"),
                ("rs_mom_vs_wls", "rs_mom", "wls"),
                ("stability_vs_wyckoff", "stability", "wyckoff_score"),
            ]
            for metric, col_a, col_b in checks:
                if col_a in df.columns and col_b in df.columns:
                    value = _spearman(df[col_a], df[col_b])
                    if value is not None:
                        umbral = UMBRALES.get(metric, 0.85)
                        status = "ALERTA" if abs(value) > umbral else "OK"
                        metrics.append({
                            "date": timestamp,
                            "metric": metric,
                            "value": value,
                            "status": status,
                        })
                        print(f"{metric}: {value:+.4f} [{status}]")
        except Exception as exc:
            print(f"Error leyendo líderes: {exc}")

    # 2. Structural vs Breadth en histórico SLPM
    slpm_path = files["slpm"]
    if slpm_path.exists():
        try:
            df = pd.read_csv(slpm_path)
            if "structural_score" in df.columns and "leader_breadth" in df.columns:
                value = _spearman(df["structural_score"], df["leader_breadth"])
                if value is not None:
                    umbral = UMBRALES["structural_vs_breadth"]
                    status = "ALERTA" if abs(value) > umbral else "OK"
                    metrics.append({
                        "date": timestamp,
                        "metric": "structural_vs_breadth",
                        "value": value,
                        "status": status,
                    })
                    print(f"structural_vs_breadth: {value:+.4f} [{status}]")
        except Exception as exc:
            print(f"Error leyendo slpm_history: {exc}")

    if not metrics:
        print("No se generaron métricas.")
        return

    # Guardar CSV acumulativo
    df_out = pd.DataFrame(metrics)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_CSV.exists():
        df_old = pd.read_csv(OUTPUT_CSV)
        df_out = pd.concat([df_old, df_out], ignore_index=True)

    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Monitoreo guardado en {OUTPUT_CSV}")

    # Si hay alertas, salir con código 1 para visibilidad en CI, pero no bloquear
    if (df_out["status"] == "ALERTA").any():
        print("⚠️ Se detectaron dependencias altas. No se modifica el sistema.")
        sys.exit(0)  # No bloquear

    sys.exit(0)

if __name__ == "__main__":
    main()
