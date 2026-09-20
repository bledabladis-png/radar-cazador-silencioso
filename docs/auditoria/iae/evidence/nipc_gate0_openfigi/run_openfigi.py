"""Query OpenFIGI /v3/mapping for CUSIPs in cusips.txt.

No API key: batches of 5, sleep 2.5s (25 req/min).
With OPENFIGI_API_KEY env var: batches of 100, sleep 1s.

Usage: py run_openfigi.py
Output: result.json (in current working directory).
"""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://api.openfigi.com/v3/mapping"
UA = "Macro_Sectorial-research/1.0"
HERE = Path(__file__).parent
CUSIPS_FILE = HERE / "cusips.txt"
OUT_FILE = HERE / "result.json"

api_key = os.environ.get("OPENFIGI_API_KEY", "").strip()
BATCH = 100 if api_key else 5
SLEEP = 1.0 if api_key else 2.5
MAX_RETRY = 2

headers = {"Content-Type": "application/json", "User-Agent": UA, "Accept": "application/json"}
if api_key:
    headers["X-OPENFIGI-APIKEY"] = api_key

cusips = [c.strip() for c in CUSIPS_FILE.read_text(encoding="utf-8").splitlines() if c.strip()]
print(f"CUSIPs: {len(cusips)}  batch={BATCH}  sleep={SLEEP}  api_key={'yes' if api_key else 'no'}")

results = {}
errors = {}
t0 = time.time()

for i in range(0, len(cusips), BATCH):
    chunk = cusips[i:i+BATCH]
    payload = [{"idType": "ID_CUSIP", "idValue": c, "exchCode": "US"} for c in chunk]
    body = json.dumps(payload).encode("utf-8")

    attempt = 0
    ok = False
    while attempt <= MAX_RETRY and not ok:
        try:
            req = urllib.request.Request(URL, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode("utf-8")
            parsed = json.loads(raw)
            for c, resp in zip(chunk, parsed):
                if isinstance(resp, dict) and "data" in resp:
                    results[c] = resp["data"]
                elif isinstance(resp, dict) and "error" in resp:
                    errors[c] = resp["error"]
                else:
                    errors[c] = f"unknown shape: {resp}"
            ok = True
        except urllib.error.HTTPError as e:
            code = e.code
            msg = e.read().decode("utf-8")[:200]
            if code in (429, 500, 503):
                attempt += 1
                wait = 5 * attempt
                print(f"  [RETRY {attempt}/{MAX_RETRY}] batch {i//BATCH+1} HTTP {code}, sleep {wait}s")
                time.sleep(wait)
            else:
                for c in chunk:
                    errors[c] = f"HTTP {code}: {msg}"
                ok = True
        except Exception as e:
            for c in chunk:
                errors[c] = f"{type(e).__name__}: {e}"
            ok = True

    done = min(i+BATCH, len(cusips))
    print(f"  Batch {i//BATCH+1:3d}: {done}/{len(cusips)} ({time.time()-t0:.1f}s)")
    time.sleep(SLEEP)

out = {
    "url": URL,
    "batches_of": BATCH,
    "n_cusips": len(cusips),
    "results": results,
    "errors": errors,
}
OUT_FILE.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8", newline="\n")
print(f"result.json: {len(results)} hits, {len(errors)} errors ({time.time()-t0:.1f}s)")
