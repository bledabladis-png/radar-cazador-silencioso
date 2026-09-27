import re
import sys
from pathlib import Path
import pandas as pd

CACHE_DIR = Path("data/cache/sec/qqq")
OUTPUT_CSV = Path("outputs/history/qqq_nport_flow.csv")


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
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Guardado en {OUTPUT_CSV}")
    print(df[['month','sales','redemptions','net_flow']].to_string(index=False))

if __name__ == '__main__':
    main()
