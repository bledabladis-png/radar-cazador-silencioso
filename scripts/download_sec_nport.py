import requests
import zipfile
from pathlib import Path
from io import BytesIO

BASE = "https://www.sec.gov/files/dera/data/form-n-port-data-sets"
QUARTER = "2026q2"
ZIP_URL = f"{BASE}/{QUARTER}_nport.zip"
OUTPUT_DIR = Path("data/nport")
NEEDED_FILES = [
    "SUBMISSION.tsv",
    "REGISTRANT.tsv",
    "FUND_REPORTED_INFO.tsv",
    "FUND_REPORTED_HOLDING.tsv",
    "IDENTIFIERS.tsv",
]

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {ZIP_URL} ...")
    headers = {"User-Agent": "Macro_Sectorial contacto@example.com"}
    r = requests.get(ZIP_URL, headers=headers, timeout=600)
    r.raise_for_status()
    print(f"Descargado: {len(r.content)/(1024*1024):.2f} MB")

    z = zipfile.ZipFile(BytesIO(r.content))
    for name in NEEDED_FILES:
        print(f"Extrayendo {name} ...")
        z.extract(name, OUTPUT_DIR)
    print("Archivos extraídos en", OUTPUT_DIR)

if __name__ == "__main__":
    main()
