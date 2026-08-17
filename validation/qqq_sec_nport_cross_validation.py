# -*- coding: utf-8 -*-
"""
Validación cruzada QQQ SEC vs NPORT-P B.6.

Compara totales trimestrales de NPORT-P con registros SEC
cuando coinciden las fechas de fin de período.

Salida:
    outputs/audit/qqq_sec_nport_cross_validation.csv

No emite señales. Es control descriptivo de consistencia.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SEC_CSV = PROJECT_ROOT / "outputs" / "history" / "qqq_sec_primary_flow.csv"
NPORT_CSV = PROJECT_ROOT / "outputs" / "history" / "qqq_nport_flow.csv"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "audit" / "qqq_sec_nport_cross_validation.csv"


def main() -> None:
    if not SEC_CSV.exists():
        print("No existe QQQ SEC CSV. Validación omitida.")
        return

    if not NPORT_CSV.exists():
        print("No existe QQQ NPORT-P CSV. Validación omitida.")
        return

    sec_df = pd.read_csv(SEC_CSV, encoding="utf-8-sig")
    nport_df = pd.read_csv(NPORT_CSV, encoding="utf-8-sig")

    if sec_df.empty or nport_df.empty:
        print("Datos vacíos. Validación omitida.")
        return

    # Agrupar NPORT-P mensual por report_date
    nport_agg = (
        nport_df.groupby("report_date", as_index=False)
        .agg(
            nport_sales=("sales", "sum"),
            nport_redemptions=("redemptions", "sum"),
            nport_net_flow=("net_flow", "sum"),
        )
    )

    rows = []

    for _, np_row in nport_agg.iterrows():
        date = str(np_row["report_date"])[:10]
        sec_match = sec_df[sec_df["period_end_date"].astype(str) == date]

        if sec_match.empty:
            rows.append({
                "report_date": date,
                "sec_period_end": None,
                "sec_period_type": None,
                "nport_sales": np_row["nport_sales"],
                "nport_redemptions": np_row["nport_redemptions"],
                "nport_net_flow": np_row["nport_net_flow"],
                "sec_proceeds": None,
                "sec_repurchased": None,
                "sec_primary_flow_usd": None,
                "sales_ratio": None,
                "redemptions_ratio": None,
                "status": "NO_SEC_MATCH",
            })
            continue

        sec_row = sec_match.iloc[0]

        sec_proceeds = sec_row.get("proceeds_shares_sold", None)
        sec_repurchased = sec_row.get("value_shares_repurchased", None)
        sec_primary_flow = sec_row.get("primary_flow_usd", None)

        sales_ratio = (
            np_row["nport_sales"] / sec_proceeds
            if pd.notna(sec_proceeds) and sec_proceeds > 0
            else None
        )
        redemptions_ratio = (
            np_row["nport_redemptions"] / abs(sec_repurchased)
            if pd.notna(sec_repurchased) and sec_repurchased != 0
            else None
        )

        # Regla de consistencia: NPORT trimestral debe ser fracción de SEC semestral
        if (
            sales_ratio is not None
            and 0 < sales_ratio <= 1
            and redemptions_ratio is not None
            and 0 < redemptions_ratio <= 1
        ):
            status = "OK"
        else:
            status = "CHECK"

        rows.append({
            "report_date": date,
            "sec_period_end": sec_row.get("period_end_date", None),
            "sec_period_type": sec_row.get("period_type", None),
            "nport_sales": np_row["nport_sales"],
            "nport_redemptions": np_row["nport_redemptions"],
            "nport_net_flow": np_row["nport_net_flow"],
            "sec_proceeds": sec_proceeds,
            "sec_repurchased": sec_repurchased,
            "sec_primary_flow_usd": sec_primary_flow,
            "sales_ratio": sales_ratio,
            "redemptions_ratio": redemptions_ratio,
            "status": status,
        })

    result_df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"Validación cruzada QQQ SEC vs NPORT-P guardada: {OUTPUT_CSV}")
    print(result_df.to_string(index=False))

    # Resumen de estado
    if "status" in result_df.columns:
        ok_count = (result_df["status"] == "OK").sum()
        check_count = (result_df["status"] == "CHECK").sum()
        print(f"\nResumen: {ok_count} OK, {check_count} CHECK")


if __name__ == "__main__":
    main()
