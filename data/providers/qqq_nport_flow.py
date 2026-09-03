import re
from pathlib import Path
import pandas as pd

XML_PATH = Path("data/cache/sec/qqq/nport_2026-03-31.xml")
OUTPUT_CSV = Path("outputs/history/qqq_nport_flow.csv")

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
    print(f"Leyendo {XML_PATH} ...")
    xml_text = XML_PATH.read_text(encoding='utf-8', errors='ignore')
    df = extract_b6_flow(xml_text)

    # Añadir metadatos
    df['report_date'] = '2026-03-31'
    df['ticker'] = 'QQQ'
    df['source'] = 'SEC NPORT-P B.6'

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Guardado en {OUTPUT_CSV}")
    print(df[['month','sales','redemptions','net_flow']].to_string(index=False))

if __name__ == '__main__':
    main()
