import re
import sys
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import requests
from src.utils import append_dedup


CACHE_DIR = Path("data/cache/sec/qqq")
OUTPUT_CSV = Path("outputs/history/qqq_nport_flow.csv")

# Fix 2026-09-29 (F5.7-20 completion): descarga automatica del XML
# NPORT-P. Antes, data/cache/sec/qqq/ estaba gitignored y ningun
# script lo poblaba en CI. El step del workflow fallaba con sys.exit(1).
# Ahora el script descubre el NPORT-P mas reciente via submissions API
# y descarga el XML renderizado a cache si esta obsoleto.
SEC_UA = "Radar Sectorial Research <bledabladis@gmail.com>"
SEC_HEADERS = {"User-Agent": SEC_UA}
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK0001067839.json"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data/1067839"
# NPORT-P publica ~60 dias post-cierre. 120 dias = 1 trimestre completo +
# margen para cubrir un ciclo entero sin descarga innecesaria.
STALENESS_DAYS = 120
DOWNLOAD_RETRIES = 3
DOWNLOAD_BACKOFF_S = 5


def _http_get(url, timeout=60):
    """GET con retry y backoff. Lanza RuntimeError si agota intentos."""
    last_exc = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            r = requests.get(url, headers=SEC_HEADERS, timeout=timeout)
            if r.status_code >= 400:
                last_exc = RuntimeError(f"HTTP {r.status_code} en {url}")
                if r.status_code in (400, 401, 403, 404):
                    raise last_exc
            else:
                return r
        except (requests.Timeout, requests.ConnectionError) as e:
            last_exc = e
        except RuntimeError:
            raise
        if attempt < DOWNLOAD_RETRIES:
            time.sleep(DOWNLOAD_BACKOFF_S * (2 ** (attempt - 1)))
    raise RuntimeError(f"descarga fallida tras {DOWNLOAD_RETRIES} intentos: {last_exc}")


def _discover_latest_nport_metadata():
    """Devuelve (report_date, accession_clean) del NPORT-P mas reciente."""
    r = _http_get(SUBMISSIONS_URL)
    data = r.json()
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accs = recent.get("accessionNumber", [])
    reports = recent.get("reportDate", [])
    candidates = []
    for f, a, rd in zip(forms, accs, reports):
        if f != "NPORT-P":
            continue
        if not rd:
            continue
        candidates.append((rd, a))
    if not candidates:
        raise RuntimeError("EDGAR no tiene filings NPORT-P para QQQ")
    candidates.sort(reverse=True)
    report_date, accession = candidates[0]
    return report_date, accession.replace("-", "")


def download_latest_nport_xml(cache_dir=CACHE_DIR, *, force=False):
    """Descarga el XML NPORT-P mas reciente a cache si esta obsoleto.

    Devuelve Path al XML en cache. Si el cache ya tiene el NPORT-P mas
    reciente con menos de STALENESS_DAYS, no descarga (idempotente).
    force=True fuerza la descarga.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_date, acc_clean = _discover_latest_nport_metadata()
    target = cache_dir / f"nport_{report_date}.xml"
    if target.exists() and not force:
        age_days = (datetime.now() - datetime.fromtimestamp(target.stat().st_mtime)).days
        if age_days <= STALENESS_DAYS:
            print(f"[NPORT-CACHE] hit: {target.name} (age={age_days}d)")
            return target
    url = f"{ARCHIVES_BASE}/{acc_clean}/xslFormNPORT-P_X01/primary_doc.xml"
    print(f"[NPORT-DOWNLOAD] {url}")
    r = _http_get(url)
    text = r.text
    if "Item B.6" not in text or "Item B.7" not in text:
        raise RuntimeError(
            f"XML descargado no contiene Item B.6/B.7 (accession={acc_clean})"
        )
    target.write_text(text, encoding="utf-8")
    print(f"[NPORT-DOWNLOAD] guardado {target} ({len(text)} chars)")
    return target


def find_latest_xml(cache_dir=Path("data/cache/sec/qqq")):
    """Localiza el XML NPORT-P mas reciente en cache.

    F5.7-19: antes la ruta del XML estaba hardcoded a un
    trimestre concreto. Cuando el XML se actualice, el script
    seguia leyendo el viejo silenciosamente.
    """
    if not cache_dir.exists():
        return None
    files = sorted(cache_dir.glob("nport_*.xml"))
    return files[-1] if files else None


def parse_report_date_from_xml_name(xml_path):
    """Extrae la fecha YYYY-MM-DD del nombre del XML. None si no cuadra."""
    m = re.search(r"nport_(\d{4}-\d{2}-\d{2})\.xml$", xml_path.name)
    return m.group(1) if m else None

def extract_b6_flow(xml_text):
    """Extrae Month 1-3 sales/redemptions de Item B.6."""
    b6_match = re.search(
        r'Item B\.6\. Flow information\.(.*?)(?=Item B\.7\.)',
        xml_text,
        flags=re.DOTALL | re.IGNORECASE
    )
    if not b6_match:
        raise ValueError("No se encontró la sección B.6")

    b6_html = b6_match.group(1)
    # Limpiar HTML
    text = re.sub(r'<[^>]+>', ' ', b6_html)
    text = re.sub(r'\s+', ' ', text)

    rows = []
    for month in range(1, 4):
        # Patrón flexible para sales y redemptions
        pattern = (
            rf'Month\s+{month}.*?'
            r'Total net asset value of shares sold.*?'
            r'([0-9]+\.[0-9]+).*?'
            r'Total net asset value of shares redeemed or repurchased, including exchanges\.'
            r'.*?([0-9]+\.[0-9]+)'
        )
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            continue
        sales = float(m.group(1))
        redemptions = float(m.group(2))
        rows.append({
            'month': month,
            'sales': sales,
            'redemptions': redemptions,
            'net_flow': sales - redemptions,
        })

    if not rows:
        raise ValueError("No se pudieron extraer los meses de B.6")
    return pd.DataFrame(rows)

def main():
    # Fix 2026-09-29: garantizar cache poblado antes de buscar.
    try:
        download_latest_nport_xml()
    except Exception as e:
        print(f"ERROR: no se pudo preparar cache NPORT-P: {type(e).__name__}: {e}")
        sys.exit(1)
    xml_path = find_latest_xml()
    if xml_path is None:
        print(f"ERROR: no hay XML NPORT-P en {CACHE_DIR}")
        sys.exit(1)
    report_date = parse_report_date_from_xml_name(xml_path)
    if report_date is None:
        print(f"ERROR: no se puede derivar report_date de {xml_path.name}")
        sys.exit(1)
    print(f"Leyendo {xml_path} (report_date={report_date}) ...")
    xml_text = xml_path.read_text(encoding='utf-8', errors='ignore')
    df = extract_b6_flow(xml_text)

    # Añadir metadatos
    df['report_date'] = report_date
    df['ticker'] = 'QQQ'
    df['source'] = 'SEC NPORT-P B.6'

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    # Fix D (2026-09-30): preservar historico con append_dedup.
    if OUTPUT_CSV.exists():
        try:
            _hist_existing = pd.read_csv(OUTPUT_CSV)
            df = append_dedup(_hist_existing, df, ["report_date", "month", "ticker"])
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as _e:
            print(f'  [WARN] qqq_nport_flow existente ilegible: {_e}')
    _tmp_out = OUTPUT_CSV.with_suffix(OUTPUT_CSV.suffix + '.tmp')
    df.to_csv(_tmp_out, index=False)
    _tmp_out.replace(OUTPUT_CSV)
    print(f"Guardado en {OUTPUT_CSV}")
    print(df[['month','sales','redemptions','net_flow']].to_string(index=False))

if __name__ == '__main__':
    main()
