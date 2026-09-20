"""Build stratified sample for OpenFIGI probe (NIPC Gate 0 OpenFIGI).

Reproducible: random_state=42, deterministic strata.
Usage: py build_sample.py
Output: sample.json, cusips.txt (in current working directory).
"""
import json
from pathlib import Path

import pandas as pd

PROBE = Path("D:/13f_probe/processed/2026Q1")
ROOT = Path("D:/Macro_Sectorial")
OUT = Path(__file__).parent

# 1. Canonical universe Q1 2026
sub = pd.read_parquet(PROBE / "SUBMISSION.parquet")
inf = pd.read_parquet(PROBE / "INFOTABLE.parquet")
accs = set(sub[sub["PERIODOFREPORT"] == pd.Timestamp("2026-03-31")]["ACCESSION_NUMBER"].astype(str).str.strip())
inf_f = inf[inf["ACCESSION_NUMBER"].astype(str).str.strip().isin(accs)].copy()
inf_canon = inf_f[(inf_f["SSHPRNAMTTYPE"] == "SH") & (inf_f["PUTCALL"].isna())].copy()
inf_canon["CUSIP_str"] = inf_canon["CUSIP"].astype(str).str.strip()

# 2. Crosswalk effectivo
cw = {}
eh = pd.read_csv(ROOT / "data/etf_holdings.csv", dtype=str)
for _, r in eh.iterrows():
    cw[str(r["identifier"]).strip()] = str(r["ticker"]).strip()
cur = pd.read_csv(ROOT / "data/mappings/cusip_ticker_exceptions.csv", dtype=str)
for _, r in cur.iterrows():
    cw[str(r["CUSIP"]).strip()] = str(r["ticker"]).strip()

# 3. Radar USA
sp = pd.read_parquet(ROOT / "data/stock_prices.parquet")
radar = set(sp.columns.get_level_values(1).unique())
EU = (".L",".DE",".MC",".PA",".AS",".MI",".BR",".ST",".HE",".CO",".OL",".VI",".LS",".IR")
radar_usa = {t for t in radar if not any(t.endswith(s) for s in EU)}

# 4. Aggregate by CUSIP
by_cusip = inf_canon.groupby("CUSIP_str").agg(
    sshprnamt=("SSHPRNAMT", "sum"),
    n_filas=("SSHPRNAMT", "size"),
    name=("NAMEOFISSUER", lambda s: s.mode().iloc[0] if len(s.mode()) else None),
    title=("TITLEOFCLASS", lambda s: s.mode().iloc[0] if len(s.mode()) else None),
).reset_index()
by_cusip["ticker"] = by_cusip["CUSIP_str"].map(cw)
by_cusip["in_radar"] = by_cusip["ticker"].isin(radar_usa)
by_cusip["mapped"] = by_cusip["CUSIP_str"].isin(cw)

# 5. Stratified sample
muestra = {}

# A: radar USA mapped (top 50 by shares)
A = by_cusip[by_cusip["mapped"] & by_cusip["in_radar"]].nlargest(50, "sshprnamt")
muestra["A_radar_mapped"] = A[["CUSIP_str","ticker","name","title","sshprnamt"]].to_dict("records")

# B: radar USA unmapped tickers (no CUSIP known)
muestra["B_radar_unmapped_tickers"] = sorted(radar_usa - set(cw.values()))

# C: top 50 unmapped by shares
C = by_cusip[~by_cusip["mapped"]].nlargest(50, "sshprnamt")
muestra["C_top_unmapped"] = C[["CUSIP_str","name","title","sshprnamt"]].to_dict("records")

# D: percentiles 40-60, random 50, seed 42
p40 = by_cusip["sshprnamt"].quantile(0.40)
p60 = by_cusip["sshprnamt"].quantile(0.60)
D = by_cusip[(by_cusip["sshprnamt"] >= p40) & (by_cusip["sshprnamt"] <= p60)].sample(50, random_state=42)
muestra["D_mid_low"] = D[["CUSIP_str","name","title","sshprnamt"]].to_dict("records")

# E: problematics
E_cusips = ["329882225","329882250","084670108","084670702","02079K107","02079K305"]
E = by_cusip[by_cusip["CUSIP_str"].isin(E_cusips)]
muestra["E_problematicos"] = E[["CUSIP_str","name","title","sshprnamt"]].to_dict("records")

# 6. Serialize
out = {
    "generated_at": "2026-09-19",
    "period": "2026Q1",
    "estratos_counts": {k: len(v) if isinstance(v, list) else None for k, v in muestra.items()},
    "muestra": muestra,
    "radar_usa_total": len(radar_usa),
}
(OUT / "sample.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8", newline="\n")

# 7. Flat CUSIP list
cusips = set()
for v in muestra.values():
    if isinstance(v, list):
        for r in v:
            if isinstance(r, dict) and "CUSIP_str" in r:
                cusips.add(r["CUSIP_str"])
(OUT / "cusips.txt").write_text("\n".join(sorted(cusips)), encoding="utf-8", newline="\n")

print(f"sample.json: {len(muestra)} estratos, {len(cusips)} CUSIPs unicos")
