"""
blackrock_scraper.py - Scraper mejorado para ETFs de iShares desde BlackRock.
Extrae NAV, acciones en circulación (shares) y AUM de la sección "Factores clave".
"""

import requests
import re
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BlackRockScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36',
            'Accept-Language': 'es-ES,es;q=0.9',
        })
        self.product_urls = {
            'IVV': 'https://www.blackrock.com/cl/productos/239726/ishares-core-sp-500-etf',
            # Añade aquí los demás tickers cuando tengas sus URLs
        }

    def _parse_spanish_number(self, text):
        """
        Convierte un número en formato español (ej. "5.369.122,00" o "730.387.303.330")
        a float. Maneja correctamente los miles y decimales.
        """
        # Eliminar puntos (separadores de miles)
        text = text.replace('.', '')
        # Reemplazar coma decimal por punto
        text = text.replace(',', '.')
        try:
            return float(text)
        except:
            return None

    def get_fund_data(self, ticker):
        if ticker not in self.product_urls:
            logger.error(f"Ticker {ticker} no soportado")
            return None

        url = self.product_urls[ticker]
        logger.info(f"Obteniendo datos de {ticker} desde {url}")

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Encontrar la sección de Factores Clave
            # Buscar el encabezado <h2> que contenga "Factores clave"
            key_facts_header = soup.find('h2', string=re.compile(r'Factores clave', re.IGNORECASE))
            if not key_facts_header:
                logger.error("No se encontró la sección 'Factores clave'")
                return None

            # Obtener el contenedor de la sección (puede ser un <div> hermano o el padre)
            # Asumimos que los datos están en una lista <ul> o en párrafos <p> después del encabezado
            section = key_facts_header.find_next('ul')
            if not section:
                section = key_facts_header.find_next('div', class_='key-facts')  # Intenta con una clase común

            if not section:
                logger.error("No se encontró el contenedor de datos después del encabezado")
                # Fallback: buscar todos los números en formato español en la página
                # (pero esto es menos fiable)
                return self._fallback_extraction(response.text, ticker)

            # Extraer todo el texto de la sección
            text = section.get_text(" ", strip=True)

            # Buscar números en formato español (con puntos y coma)
            # Patrón: dígitos, luego grupos opcionales de tres dígitos con punto, y coma decimal opcional
            patron = r'\d{1,3}(?:\.\d{3})*(?:,\d{2})?'
            numeros_texto = re.findall(patron, text)

            if len(numeros_texto) < 3:
                logger.warning(f"Solo se encontraron {len(numeros_texto)} números en la sección, se esperaban al menos 3")
                return self._fallback_extraction(response.text, ticker)

            # El primer número suele ser el AUM (el más grande)
            aum_text = numeros_texto[0]
            aum = self._parse_spanish_number(aum_text)

            # El segundo número es el NAV
            nav_text = numeros_texto[1]
            nav = self._parse_spanish_number(nav_text)

            # El tercer número son las shares
            shares_text = numeros_texto[2]
            shares = self._parse_spanish_number(shares_text)

            return {
                'fecha': datetime.now().date(),
                'ticker': ticker,
                'nav': nav,
                'shares': shares,
                'aum': aum,
            }

        except Exception as e:
            logger.error(f"Error obteniendo datos de {ticker}: {e}")
            return None

    def _fallback_extraction(self, html_text, ticker):
        """Método de respaldo si la extracción por sección falla."""
        logger.warning("Usando método de extracción de respaldo (búsqueda global de números)")
        patron = r'\d{1,3}(?:\.\d{3})*(?:,\d{2})?'
        numeros_texto = re.findall(patron, html_text)
        numeros = [self._parse_spanish_number(n) for n in numeros_texto if self._parse_spanish_number(n) is not None]
        numeros.sort(reverse=True)

        aum = numeros[0] if numeros and numeros[0] > 1e9 else None
        # Buscar NAV (alrededor de 600) y shares (alrededor de 5 millones)
        nav = None
        shares = None
        for n in numeros:
            if 500 <= n <= 1000:
                nav = n
            elif 1e6 <= n <= 1e7:  # Entre 1 y 10 millones
                shares = n

        return {
            'fecha': datetime.now().date(),
            'ticker': ticker,
            'nav': nav,
            'shares': shares,
            'aum': aum,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = BlackRockScraper()
    data = scraper.get_fund_data('IVV')
    print(data)