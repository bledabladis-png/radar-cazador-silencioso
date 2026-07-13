import requests
import re
url2 = 'https://otctransparency.finra.org/otctransparency/assets/download.html'
resp2 = requests.get(url2, headers={'User-Agent': 'Mozilla/5.0'})
print('URL2 Status:', resp2.status_code)
links = re.findall(r'https://[^\"\s]+\.(txt|csv)[^\"\s]*', resp2.text)
print('Enlaces encontrados:', links[:5])
text = resp2.text
print('\n--- Muestra del HTML ---')
print(text[:2000])
