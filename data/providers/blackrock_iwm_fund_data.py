"""
Proveedor BlackRock / iShares para IWM.

Fuente oficial:
    BlackRock iShares Fund Download
    portfolioId = 239710

Extrae de la hoja Historical:
    - NAV per Share
    - Shares Outstanding
    - As Of

Calcula:
    - shares_change
    - primary_flow_usd
    - estimated_net_assets_usd
    - primary_flow_pct
    - primary_flow_z
    - primary_flow_5d
    - primary_flow_20d

IMPORTANTE
----------
Se separan tres capas:

1. RAW
   Datos procedentes directamente del fund file de BlackRock.

2. CALCULATED
   Variables derivadas matemáticamente de RAW.

3. SIGNAL
   Transformaciones estadísticas utilizadas por el Radar.

El histórico RAW nunca se modifica ni se recorta por outliers.
Los outliers únicamente afectan a la señal estadística.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

ISIN_IWM = "US4642876555"
TICKER_IWM = "IWM"
PORTFOLIO_ID = "239710"

FUND_NAME = "iShares Russell 2000 ETF"

API_URL = (
    "https://www.blackrock.com/varnish-api/"
    "blk-one01-product-data/product-data/api/v1/"
    "get-fund-document"
)

CACHE_DIR = Path("data/cache/blackrock")
RAW_CACHE_FILE = CACHE_DIR / "iwm_fund_download_response.bin"

HISTORY_DIR = Path("outputs/history")
HISTORY_CSV = HISTORY_DIR / "blackrock_iwm_historical.csv"

TIMEOUT = 60


HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "*/*",
    "Referer": (
        "https://www.ishares.com/us/products/"
        "239710/ishares-russell-2000-etf"
    ),
}


# =============================================================================
# DESCARGA
# =============================================================================

def build_fund_url() -> str:
    """Construye la URL oficial del fund download de BlackRock."""

    params = {
        "appType": "PRODUCT_PAGE",
        "appSubType": "ISHARES",
        "targetSite": "us-ishares",
        "locale": "en_US",
        "portfolioId": PORTFOLIO_ID,
        "component": "fundDownload",
        "userType": "individual",
    }

    request = requests.Request(
        "GET",
        API_URL,
        params=params,
    ).prepare()

    return request.url


def download_fund_file(
    force_download: bool = False,
) -> bytes:
    """
    Descarga el fund file oficial de BlackRock.

    Utiliza caché local para evitar descargas innecesarias.
    """

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_CACHE_FILE.exists() and not force_download:
        mtime = datetime.fromtimestamp(RAW_CACHE_FILE.stat().st_mtime)
        age = datetime.now() - mtime
        if age <= timedelta(hours=23):
            print(f"  Usando caché: {RAW_CACHE_FILE} (antigüedad {age})")
            return RAW_CACHE_FILE.read_bytes()
        print(f"  Caché obsoleta ({age}). Descargando de nuevo...")

    url = build_fund_url()

    print("  Descargando fund file IWM desde BlackRock...")
    print(f"  Portfolio ID: {PORTFOLIO_ID}")

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=TIMEOUT,
    )

    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    content_length = len(response.content)

    print(f"  HTTP: {response.status_code}")
    print(f"  Content-Type: {content_type}")
    print(f"  Bytes: {content_length}")

    if content_length < 1000:
        raise RuntimeError(
            "La respuesta de BlackRock es demasiado pequeña; "
            "posible error del endpoint."
        )

    # El fund file de BlackRock es SpreadsheetML/XML aunque
    # el Content-Type sea application/vnd.ms-excel.
    if b"<ss:Workbook" not in response.content[:50000]:
        raise RuntimeError(
            "La respuesta no parece un fund file SpreadsheetML válido."
        )

    RAW_CACHE_FILE.write_bytes(response.content)

    print(f"  Caché guardada: {RAW_CACHE_FILE}")

    return response.content


# =============================================================================
# EXTRACCIÓN DEL WORKSHEET HISTORICAL
# =============================================================================

def extract_historical_sheet(raw: bytes) -> str:
    """
    Extrae el contenido de la hoja Historical.

    No utiliza ElementTree porque el SpreadsheetML descargado por BlackRock
    puede contener HTML/XML no perfectamente válido dentro de algunas celdas.
    """

    text = raw.decode("utf-8", errors="replace")

    match = re.search(
        r'<ss:Worksheet\s+ss:Name="Historical">(.*?)'
        r'</ss:Worksheet>',
        text,
        flags=re.DOTALL,
    )

    if not match:
        raise RuntimeError(
            "No se encontró la hoja Historical en el fund file de BlackRock."
        )

    return match.group(1)


def extract_cell_values(row_xml: str) -> list[str]:
    """
    Extrae valores de las celdas de una fila SpreadsheetML.
    """

    values = []

    for match in re.finditer(
        r"<ss:Data[^>]*>(.*?)</ss:Data>",
        row_xml,
        flags=re.DOTALL,
    ):
        value = match.group(1)

        # El fund file puede contener entidades/HTML.
        value = value.replace("&amp;", "&")
        value = value.replace("&lt;", "<")
        value = value.replace("&gt;", ">")
        value = value.replace("&quot;", '"')
        value = value.replace("&#39;", "'")

        # Eliminamos etiquetas HTML residuales.
        value = re.sub(r"<[^>]+>", "", value)

        values.append(value.strip())

    return values


def parse_historical_sheet(
    historical_xml: str,
) -> pd.DataFrame:
    """
    Convierte la hoja Historical en DataFrame.

    Estructura esperada:

        As Of | NAV per Share | Ex-Dividends | Shares Outstanding
    """

    rows = re.findall(
        r"<ss:Row[^>]*>(.*?)</ss:Row>",
        historical_xml,
        flags=re.DOTALL,
    )

    print(f"  Filas XML detectadas: {len(rows)}")

    if not rows:
        raise RuntimeError(
            "No se encontraron filas en la hoja Historical."
        )

    records = []

    header_found = False

    for row_xml in rows:

        values = extract_cell_values(row_xml)

        if not values:
            continue

        # Detectamos la cabecera real.
        normalized = [
            value.lower().strip()
            for value in values
        ]

        if (
            len(normalized) >= 4
            and normalized[0] == "as of"
            and normalized[1] == "nav per share"
            and normalized[3] == "shares outstanding"
        ):
            header_found = True
            continue

        if not header_found:
            continue

        if len(values) < 4:
            continue

        date_raw = values[0]
        nav_raw = values[1]
        shares_raw = values[3]

        try:
            date = datetime.strptime(
                date_raw,
                "%b %d, %Y",
            ).date()
        except ValueError:
            continue

        try:
            nav = float(
                nav_raw.replace(",", "")
            )
        except (ValueError, TypeError):
            continue

        try:
            shares = float(
                shares_raw.replace(",", "")
            )
        except (ValueError, TypeError):
            continue

        records.append(
            {
                "date": date,
                "nav": nav,
                "shares_outstanding": shares,
            }
        )

    if not records:
        raise RuntimeError(
            "No se pudieron extraer registros válidos "
            "de la hoja Historical."
        )

    df = pd.DataFrame(records)

    df["date"] = pd.to_datetime(df["date"])

    df = (
        df
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )

    print(f"  Registros extraídos: {len(df)}")

    return df


# =============================================================================
# CÁLCULO PRIMARY FLOW
# =============================================================================

def robust_zscore(
    series: pd.Series,
    window: int = 120,
    min_periods: int = 20,
) -> pd.Series:
    """
    Z-score robusto rolling mediante mediana + MAD.

    z = (x - median) / (1.4826 * MAD)

    No se recortan los valores originales.
    """

    def calculate(window_values: pd.Series) -> float:

        values = window_values.dropna()

        if len(values) < min_periods:
            return float("nan")

        median = values.median()

        mad = (
            values
            .sub(median)
            .abs()
            .median()
        )

        if pd.isna(mad) or mad <= 0:
            return 0.0

        latest = values.iloc[-1]

        return (
            latest - median
        ) / (
            1.4826 * mad + 1e-12
        )

    return series.rolling(
        window=window,
        min_periods=min_periods,
    ).apply(
        calculate,
        raw=False,
    )


def compute_primary_flow(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula la capa CALCULATED + SIGNAL.
    """

    df = df.copy()

    # -------------------------------------------------------------------------
    # RAW validation
    # -------------------------------------------------------------------------

    if (df["nav"] <= 0).any():
        raise ValueError(
            "Se detectaron NAV <= 0."
        )

    if (df["shares_outstanding"] <= 0).any():
        raise ValueError(
            "Se detectaron Shares Outstanding <= 0."
        )

    # -------------------------------------------------------------------------
    # CALCULATED
    # -------------------------------------------------------------------------

    df["shares_change"] = (
        df["shares_outstanding"].diff()
    )

    # Flujo primario estimado.
    #
    # IMPORTANTE:
    # No se denomina "actual fund flow".
    # Es un proxy basado en creación/redención de participaciones.
    df["primary_flow_usd"] = (
        df["shares_change"] * df["nav"]
    )

    # Patrimonio implícito:
    #
    # Shares × NAV
    #
    # Se mantiene explícitamente como "estimated/implied"
    # y no como patrimonio oficial BlackRock.
    df["estimated_net_assets_usd"] = (
        df["shares_outstanding"] * df["nav"]
    )

    # Porcentaje del patrimonio implícito.
    df["primary_flow_pct"] = (
        df["primary_flow_usd"]
        / df["estimated_net_assets_usd"]
        * 100.0
    )

    # -------------------------------------------------------------------------
    # SIGNAL
    # -------------------------------------------------------------------------

    df["primary_flow_z"] = robust_zscore(
        df["primary_flow_pct"],
        window=120,
        min_periods=20,
    )

    # Suavizado.
    df["primary_flow_5d"] = (
        df["primary_flow_usd"]
        .rolling(5, min_periods=5)
        .mean()
    )

    df["primary_flow_20d"] = (
        df["primary_flow_usd"]
        .rolling(20, min_periods=20)
        .mean()
    )

    # Porcentaje suavizado.
    df["primary_flow_pct_5d"] = (
        df["primary_flow_pct"]
        .rolling(5, min_periods=5)
        .mean()
    )

    df["primary_flow_pct_20d"] = (
        df["primary_flow_pct"]
        .rolling(20, min_periods=20)
        .mean()
    )

    return df


# =============================================================================
# HISTÓRICO
# =============================================================================

def update_history(
    df_new: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fusiona el nuevo histórico con el histórico existente.

    Las fechas nuevas sustituyen a versiones anteriores de la misma fecha.
    """

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if HISTORY_CSV.exists():

        print(
            f"  Histórico existente: {HISTORY_CSV}"
        )

        df_old = pd.read_csv(
            HISTORY_CSV,
            parse_dates=["date"],
        )

        df = pd.concat(
            [
                df_old,
                df_new,
            ],
            ignore_index=True,
        )

        # Nueva extracción gana si existe la misma fecha.
        df = (
            df
            .drop_duplicates(
                subset=["date"],
                keep="last",
            )
            .sort_values("date")
            .reset_index(drop=True)
        )

    else:

        df = df_new.copy()

    df.to_csv(
        HISTORY_CSV,
        index=False,
    )

    return df


# =============================================================================
# API PÚBLICA DEL PROVEEDOR
# =============================================================================

def get_blackrock_iwm_primary_flow(
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Descarga/actualiza IWM y devuelve la última observación.

    Returns
    -------
    pandas.DataFrame
        Última fila del histórico con todas las variables RAW,
        CALCULATED y SIGNAL.
    """

    raw = download_fund_file(
        force_download=force_download,
    )

    historical_xml = extract_historical_sheet(
        raw
    )

    df = parse_historical_sheet(
        historical_xml
    )

    df = compute_primary_flow(
        df
    )

    df = update_history(
        df
    )

    latest = df.tail(1).copy()

    print()
    print("=" * 80)
    print("IWM — PRIMARY FLOW")
    print("=" * 80)

    print(
        "Histórico:",
        df["date"].min().date(),
        "→",
        df["date"].max().date(),
    )

    print(
        "Registros:",
        len(df),
    )

    print()
    print(
        latest[
            [
                "date",
                "nav",
                "shares_outstanding",
                "shares_change",
                "primary_flow_usd",
                "estimated_net_assets_usd",
                "primary_flow_pct",
                "primary_flow_z",
                "primary_flow_5d",
                "primary_flow_20d",
            ]
        ].to_string(index=False)
    )

    return latest


# Alias corto para mantener compatibilidad.
def get_iwm_primary_flow(
    force_download: bool = False,
) -> pd.DataFrame:
    return get_blackrock_iwm_primary_flow(
        force_download=force_download
    )


# =============================================================================
# EJECUCIÓN DIRECTA
# =============================================================================

if __name__ == "__main__":

    df = get_blackrock_iwm_primary_flow(
        force_download=True
    )

    print()
    print("Archivo histórico:")
    print(HISTORY_CSV)

