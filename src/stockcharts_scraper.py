"""
stockcharts_scraper.py - Scraper de indicadores de breadth desde StockCharts
Obtiene valores diarios de:
- $NYAD (Advance-Decline Line)
- $NYHL (New Highs - New Lows)
- $NYMO (McClellan Oscillator)
- $NYA50R (% de acciones sobre MA50)
- $NYA200R (% de acciones sobre MA200)
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class StockChartsScraper:
    def __init__(self, data_dir='data/raw/stockcharts'):
        self.base_url = "https://stockcharts.com/freecharts"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.indicators = {
            'nyad': '$NYAD',
            'nyhl': '$NYHL',
            'nymo': '$NYMO',
            'nya50r': '$NYA50R',
            'nya200r': '$NYA200R'
        }
        
        # Archivo para guardar el histórico
        self.history_file = self.data_dir / 'stockcharts_history.parquet'
    
    def fetch_indicator(self, symbol):
        """Obtiene el valor actual de un indicador desde StockCharts."""
        url = f"{self.base_url}/{symbol}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Buscar el valor numérico - diferentes estrategias según la página
            # 1. Buscar por clase específica (ajustar según inspección)
            value_tag = soup.find('span', class_='last-value')
            if value_tag:
                text = value_tag.text.strip().replace(',', '')
                return float(text)
            
            # 2. Buscar en tablas
            table = soup.find('table', class_='data')
            if table:
                rows = table.find_all('tr')
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 2 and 'Last' in cols[0].text:
                        return float(cols[1].text.strip().replace(',', ''))
            
            # 3. Buscar cualquier número grande al inicio de la página (fallback)
            text = soup.get_text()
            import re
            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', text[:1000])
            if numbers:
                return float(numbers[0])
            
            logger.warning(f"No se pudo extraer valor para {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None
    
    def fetch_all(self):
        """Obtiene todos los indicadores y guarda en histórico."""
        today = datetime.now().date()
        logger.info(f"Obteniendo datos de StockCharts para {today}")
        
        row = {'date': today}
        for key, symbol in self.indicators.items():
            value = self.fetch_indicator(symbol)
            if value is not None:
                row[key] = value
            else:
                row[key] = None
            time.sleep(1)  # Respetar rate limiting
        
        # Cargar histórico existente
        if self.history_file.exists():
            df = pd.read_parquet(self.history_file)
        else:
            df = pd.DataFrame(columns=['date'] + list(self.indicators.keys()))
        
        # Añadir nueva fila
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df = df.drop_duplicates(subset=['date'], keep='last')
        df = df.sort_values('date')
        
        # Guardar
        df.to_parquet(self.history_file)
        logger.info(f"Datos guardados en {self.history_file}")
        
        return row
    
    def load_history(self):
        """Carga el histórico completo."""
        if self.history_file.exists():
            return pd.read_parquet(self.history_file)
        return pd.DataFrame()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = StockChartsScraper()
    today_data = scraper.fetch_all()
    print("Datos de hoy:", today_data)
    print("\nHistórico:")
    print(scraper.load_history().tail())