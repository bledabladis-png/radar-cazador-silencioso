"""
integration_check.py -- Verifica que el reporte diario contiene todas las secciones obligatorias.
Se ejecuta tras run.py para validar la integridad del sistema.
"""
import sys, os

REPORT_PATH = "outputs/report/reporte_diario.md"

REQUIRED_SECTIONS = [
    "## Resumen de Regimenes",
    "## Breadth de Mercado",
    "## Rankings Sectoriales",
    "## Momentum de Precio - Sectores",
    "## Flujo Institucional - Sectores",
    "## Momentum de Precio - Otros Activos",
    "## Flujo Institucional - Otros Activos",
    "## Lideres Sectoriales",
    "## Sentimiento de Opciones (OMS",
    "## Market Transition Engine (MTE",
    "## Flujos Institucionales (Dark Pools",
]

def main():
    if not os.path.exists(REPORT_PATH):
        print(f"ERROR: Reporte no encontrado en {REPORT_PATH}")
        sys.exit(1)

    with open(REPORT_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    missing = []
    for section in REQUIRED_SECTIONS:
        if section not in content:
            missing.append(section)

    if missing:
        print(f"ERROR: Faltan {len(missing)} secciones en el reporte:")
        for s in missing:
            print(f"  - {s}")
        sys.exit(1)

    print(f"✓ Reporte completo: {len(REQUIRED_SECTIONS)}/{len(REQUIRED_SECTIONS)} secciones presentes.")
    print("✓ Integración validada.")

if __name__ == "__main__":
    main()
