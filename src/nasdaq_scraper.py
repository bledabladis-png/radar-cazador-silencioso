"""
nasdaq_scraper.py - Scraper de datos de breadth desde NASDAQ
Obtiene:
- Advancing issues
- Declining issues
- New highs
- New lows
- Up volume
- Down volume
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import logging
from pathlib import Path
import re

logger = logging.getLogger(__name__)

class NasdaqScraper:
    def __init__(self, data_dir='data/raw/nasdaq'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # URLs de las páginas de NASDAQ
        self.urls = {
            'advance_decline': 'https://www.nasdaq.com/market-activity/advance-decline',
            'new_highs_lows': 'https://www.nasdaq.com/market-activity/new-highs-new-lows'
        }
        
        self.history_file = self.data_dir / 'nasdaq_breadth.parquet'
    
    def fetch_page(self, url):
        """Obtiene el HTML de una página."""
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def parse_advance_decline(self, html):
        """Extrae advancing y declining issues de la página."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Buscar la tabla de datos
        table = soup.find('table', class_='nasdaq-table')
        if not table:
            # Intentar con cualquier tabla
            table = soup.find('table')
        
        if not table:
            logger.warning("No se encontró tabla en la página de advance/decline")
            return {}
        
        # Extraer datos de la tabla
        rows = table.find_all('tr')
        data = {}
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 2:
                label = cols[0].text.strip().lower()
                value = cols[1].text.strip().replace(',', '')
                if 'advancing' in label:
                    data['advancing_issues'] = int(value) if value.isdigit() else None
                elif 'declining' in label:
                    data['declining_issues'] = int(value) if value.isdigit() else None
                elif 'unchanged' in label:
                    data['unchanged_issues'] = int(value) if value.isdigit() else None
                elif 'advancing volume' in label:
                    data['advancing_volume'] = int(value) if value.isdigit() else None
                elif 'declining volume' in label:
                    data['declining_volume'] = int(value) if value.isdigit() else None
        
        return data
    
    def parse_new_highs_lows(self, html):
        """Extrae new highs y new lows de la página."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Buscar la tabla
        table = soup.find('table', class_='nasdaq-table')
        if not table:
            table = soup.find('table')
        
        if not table:
            logger.warning("No se encontró tabla en la página de new highs/lows")
            return {}
        
        data = {}
        rows = table.find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 2:
                label = cols[0].text.strip().lower()
                value = cols[1].text.strip().replace(',', '')
                if 'new highs' in label:
                    data['new_highs'] = int(value) if value.isdigit() else None
                elif 'new lows' in label:
                    data['new_lows'] = int(value) if value.isdigit() else None
        
        return data
    
    def fetch_today(self):
        """Obtiene todos los datos del día."""
        today = datetime.now().date()
        logger.info(f"Obteniendo datos de NASDAQ para {today}")
        
        all_data = {'date': today}
        
        # Advance/Decline
        html = self.fetch_page(self.urls['advance_decline'])
        if html:
            ad_data = self.parse_advance_decline(html)
            all_data.update(ad_data)
        
        time.sleep(1)
        
        # New Highs/Lows
        html = self.fetch_page(self.urls['new_highs_lows'])
        if html:
            hl_data = self.parse_new_highs_lows(html)
            all_data.update(hl_data)
        
        # Calcular indicadores derivados
        if 'advancing_issues' in all_data and 'declining_issues' in all_data:
            adv = all_data['advancing_issues']
            dec = all_data['declining_issues']
            if adv is not None and dec is not None and (adv + dec) > 0:
                all_data['ad_ratio'] = adv / (adv + dec)
                all_data['ad_line_delta'] = adv - dec
        
        if 'new_highs' in all_data and 'new_lows' in all_data:
            nh = all_data['new_highs']
            nl = all_data['new_lows']
            if nh is not None and nl is not None:
                all_data['hl_ratio'] = nh / (nh + nl) if (nh + nl) > 0 else 0.5
                all_data['hl_net'] = nh - nl
        
        return all_data
    
    def update_history(self):
        """Obtiene datos del día y los añade al histórico."""
        today_data = self.fetch_today()
        
        # Cargar histórico existente
        if self.history_file.exists():
            df = pd.read_parquet(self.history_file)
        else:
            df = pd.DataFrame()
        
        # Convertir a DataFrame y añadir
        new_row = pd.DataFrame([today_data])
        df = pd.concat([df, new_row], ignore_index=True)
        df = df.drop_duplicates(subset=['date'], keep='last')
        df = df.sort_values('date')
        
        # Guardar
        df.to_parquet(self.history_file)
        logger.info(f"Datos guardados en {self.history_file}")
        
        return today_data
    
    def load_history(self):
        """Carga el histórico completo."""
        if self.history_file.exists():
            return pd.read_parquet(self.history_file)
        return pd.DataFrame()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = NasdaqScraper()
    today_data = scraper.update_history()
    print("Datos de hoy:", today_data)
    print("\nHistórico:")
    print(scraper.load_history().tail())