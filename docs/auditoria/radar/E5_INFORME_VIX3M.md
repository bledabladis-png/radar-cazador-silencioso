# E5 — Anomalia ^VIX3M en Yahoo Finance

**Fase:** investigacion empirica
**Fecha:** 2026-09-17
**HEAD de referencia:** 18601ee (origin/main)
**Alcance:** ticker ^VIX3M y el contrato temporal VOLATILITY_INDEX
**Autorizado por:** sesion E5 (post-ciclo FU-021-3C-bis)
**Estado:** investigacion cerrada. Requiere decision arquitectonica.

---

## 0. Motivo del informe

El contrato temporal VOLATILITY_INDEX aparece como INSUFFICIENT en cada run real desde la publicacion del anexo FU-021-3B (2026-09-16). La causa declarada en el log del pipeline era "^VIX3M: DATA ISSUE (NaNs >10%)".

En el registro de deudas del proyecto, E5 quedo catalogada como "anomalia yfinance (individual vs batch)" — una hipotesis de causa raiz, no un hecho medido. El objetivo de este informe es cerrar la investigacion midiendo cada eslabon de la cadena y sustituir la hipotesis por evidencia reproducible. La etiqueta anterior es incorrecta.

Este informe no modifica codigo de produccion. Es precondicion para decidir si se modifica (provider CBOE, rediseno del contrato, o WONT FIX).

---

## 1. Hechos medidos

Todas las mediciones se han ejecutado sobre HEAD 18601ee, con el parquet data/market_data.parquet tal como lo dejo el ultimo run E2E. Los scripts de reproduccion estan en la Seccion 7.

### 1.1. Estado del artefacto data/market_data.parquet

Shape: (2604, 2810). Rango temporal: 2016-09-16 -> 2026-09-15.

| Ticker | Filas | NaN | Ratio | Primera obs. | Ultima obs. |
|---|---|---|---|---|---|
| ^VIX | 2604 | 90 | 0.0346 | 2016-09-16 | 2026-09-15 |
| ^VIX3M | 2604 | 2604 | 1.0000 | Nunca | Nunca |
| ^VXN | 2604 | 92 | 0.0353 | 2016-09-16 | 2026-09-15 |

^VIX3M entra al parquet como columna completamente vacia. No hay un solo valor no nulo en 10 anos.

### 1.2. Yahoo Finance — individual vs batch

Medicion directa con yfinance, period='5y', auto_adjust=True.

| Consulta | Filas devueltas | NaN | Ultima obs. |
|---|---|---|---|
| yf.download(['^VIX3M']) | 1 | 0 | 2026-09-16 |
| yf.download(['^VIX','^VIX3M','^VXN']) — ^VIX | 1257 | 0 | 2026-09-16 |
| yf.download(['^VIX','^VIX3M','^VXN']) — ^VIX3M | 1257 | 1256 | 2026-09-16 |
| yf.download(['^VIX','^VIX3M','^VXN']) — ^VXN | 1257 | 3 | 2026-09-16 |

Individual y batch devuelven exactamente 1 fila util para ^VIX3M. No hay diferencia observable entre modos.

La hipotesis H1 ("individual vs batch") queda descartada empiricamente.

### 1.3. Barrido de simbolos alternativos en Yahoo

| Simbolo | Resultado | Diagnostico |
|---|---|---|
| ^VIX3M | 1 fila (solo hoy) | Historico ausente |
| ^VIX9D | 1 fila (solo hoy) | Historico ausente |
| ^VXV (simbolo pre-2021 CBOE) | delisted, 0 filas | Retirado por Yahoo |
| VIX3M (sin circunflejo) | delisted, 0 filas | Retirado por Yahoo |

Patron consistente: Yahoo sirve con normalidad los indices spot de volatilidad (^VIX, ^VXN) pero no sirve los indices de term structure (^VIX3M, ^VIX9D). Devuelve exclusivamente la cotizacion del dia en curso.

### 1.4. Validador interno data/validator.py

Codigo relevante:

    if close.isna().sum() / len(close) > 0.1:
        issues[t] = 'DATA ISSUE (NaNs >10%)'

- Umbral: MAX_NAN_RATIO = 0.10 (config/settings.py:46).
- Ventana: toda la serie descargada (~5 anos con period='5y' en el loader). No es la ultima fila. No es ventana movil.
- Consecuencia: si un ticker supera el 10% de NaN en su serie completa, se excluye de valid_cols y no entra al parquet final.

Resultado sobre el batch de volatilidad:

    validos = ['^VIX', '^VXN']
    issues  = {'^VIX3M': 'DATA ISSUE (NaNs >10%)'}

### 1.5. CBOE — verificacion de fuente alternativa

Consultado cdn.cboe.com, endpoints publicos de historicos diarios.

| URL | HTTP | Lineas | Formato | Auth |
|---|---|---|---|---|
| .../VIX3M_History.csv | 200 | 4273 | DATE,OPEN,HIGH,LOW,CLOSE | ninguna |
| .../VIX_History.csv | 200 | 9273 | DATE,OPEN,HIGH,LOW,CLOSE | ninguna |

Sin Cloudflare, sin WAF, sin prueba de trabajo, sin bloqueo por IP. CBOE publica el historico oficial VIX3M con cobertura amplia (pre-2010).

---

## 2. Diagnostico tecnico

### 2.1. Hipotesis descartadas

- H1 — "Individual vs batch". Descartada. Individual y batch producen el mismo resultado (1 fila).
- H2 — "Cutoff de antiguedad silencioso en Yahoo". Descartada. ^VIX y ^VXN no experimentan cutoff.
- H3 — "Problema de nuestro query". Descartada. La query es identica para los tres tickers; solo ^VIX3M falla.
- H4 — "Error de peticion del sistema". Descartada. yf.download desde consola independiente reproduce el fallo.

### 2.2. Hecho confirmado

Yahoo Finance ha dejado de servir el historico de los indices de term structure de volatilidad. Los tickers ^VIX3M y ^VIX9D devuelven unicamente la cotizacion actual. Los indices spot ^VIX y ^VXN siguen operativos. El simbolo historico ^VXV esta retirado.

No es un problema del sistema. No es un problema de la query. No es un problema de configuracion. Es una restriccion de datos del proveedor Yahoo, fuera de nuestro control.

### 2.3. Consecuencia mecanica en el pipeline

data/validator.py clasifica ^VIX3M como DATA ISSUE con 100% de NaN sobre la serie completa. El ticker se excluye de valid_cols. Entra al parquet como columna vacia. El contrato temporal VOLATILITY_INDEX, cuyo universo elegible es {^VIX, ^VIX3M, ^VXN}, evalua cobertura de 2/3 -> por debajo del umbral -> INSUFFICIENT.

Ningun eslabon de la cadena es un bug. El sistema reporta con precision lo que Yahoo entrega.

---

## 3. Cadena causal

    Yahoo deja de servir historico de ^VIX3M
            |
            v
    src/stock_data_loader.py descarga batch Yahoo de 5 anos
       -> ^VIX3M llega como columna todo-NaN
            |
            v
    data/validator.py: ratio NaN = 1.0000 > MAX_NAN_RATIO (0.10)
       -> status = 'DATA ISSUE (NaNs >10%)'
       -> excluido de valid_cols
            |
            v
    data/market_data.parquet: columna ('Close','^VIX3M') presente, 100% NaN
            |
            v
    src/temporal_contracts/volatility_index.py::VOLATILITY_INDEX.resolve()
       -> universo elegible {^VIX, ^VIX3M, ^VXN}, cobertura observada 2/3
            |
            v
    VOLATILITY_INDEX = INSUFFICIENT (repetido en cada run desde 2026-09-16)

---

## 4. Impacto

### 4.1. Sobre la infraestructura

- Gate 10/10: intacto.
- Contratos temporales: 9 en OK/STALE, 1 en INSUFFICIENT.
- Runs reales: la bandera INSUFFICIENT aparece sistematicamente en cada ejecucion.

### 4.2. Sobre los consumidores directos de ^VIX3M

| Consumidor | Linea | Efecto actual |
|---|---|---|
| regimes/macro_regime.py | 52 | vix_term sobre serie vacia -> valor neutro |
| indicators/mte.py | 1747 | Term structure VIX3M-VIX -> N/D |
| indicators/volatility_structure.py | 46 | term_ratio, term_slope -> N/D |
| src/report/volatility_mte.py | 18 | Columna VIX3M/VIX en reporte -> N/D |

El resto del sistema funciona. La degradacion esta contenida al term structure.

### 4.3. Sobre la auditoria

- El sistema no imputa ni finge. Reporta INSUFFICIENT con precision.
- La bandera persistente degrada la senal-a-ruido del registro de contratos.
- Un contrato roto de forma permanente es, en si mismo, un hallazgo auditable.

---

## 5. Opciones

### Opcion A — WONT FIX

Aceptar VOLATILITY_INDEX = INSUFFICIENT como estado permanente. Coste 0. Riesgo: silenciar el sintoma sin resolver la causa.

### Opcion B — Provider CBOE para term structure

Introducir un CBOEProvider analogo a FuturesProvider (FU-021-3C-bis). Endpoint publico, sin auth. Alimenta ^VIX3M en df_market. Coste medio (~200-300 LOC). Impacto: VOLATILITY_INDEX recupera OK/STALE. Semantica intacta. Mismo patron arquitectonico que OilPriceAPI.

### Opcion C — Rediseno del contrato VOLATILITY_INDEX

Reducir universo a {^VIX, ^VXN}. Coste bajo. Riesgo: perdida semantica de term structure. Requiere dictamen nuevo porque redefine un contrato ratificado.

### Opcion D — Fuente Yahoo alterna

Usar Tiingo/Polygon/FMP. Coste alto. Impacto no verificado. Riesgo: dependencia de proveedor comercial para un ticker con fuente oficial publica.

---

## 6. Recomendacion tecnica

Opcion B (provider CBOE).

Razones:

1. Fuente verificada funcional en esta misma sesion (HTTP 200, CSV estandar, sin auth).
2. Mismo patron arquitectonico que FuturesProvider (FU-021-3C-bis), ya validado.
3. Semantica intacta: sigue siendo el indice oficial VIX3M.
4. Cobertura para ^VIX9D si en el futuro se incorpora.
5. Bajo riesgo operativo: si CBOE falla, VOLATILITY_INDEX cae a INSUFFICIENT. Estado conocido. Gate no se rompe.

La decision no corresponde al ingeniero supervisor. Este informe presenta la recomendacion tecnica; la eleccion A/B/C/D es arquitectonica y requiere dictamen del auditor externo.

---

## 7. Preguntas al auditor

1. Opcion A, B, C o D?
2. Si B: solo ^VIX3M o tambien ^VIX9D?
3. Si B: provider paralelo o cascada Yahoo?
4. Si B: registrar nueva regla R6 ("term structure via CBOE exclusivamente")?
5. Si C: perdida de term structure aceptada como decision de diseno?
6. Cierra E5 como deuda o genera nueva FU?

---

## 8. Evidencia reproducible

### 8.1. Medicion parquet

    import pandas as pd
    df = pd.read_parquet('data/market_data.parquet')
    for t in ['^VIX', '^VIX3M', '^VXN']:
        c = df[('Close', t)]
        print(f'{t}: len={len(c)} nans={c.isna().sum()} ratio={c.isna().sum()/len(c):.4f}')

### 8.2. Medicion Yahoo

    import yfinance as yf
    bat = yf.download(['^VIX', '^VIX3M', '^VXN'], period='5y', auto_adjust=True, progress=False)
    for t in ['^VIX', '^VIX3M', '^VXN']:
        c = bat[('Close', t)]
        print(f'{t}: nans={c.isna().sum()} ratio={c.isna().sum()/len(c):.4f}')

### 8.3. Medicion validador

    from data.validator import validate_market_data
    valid, issues = validate_market_data(bat)
    print(valid, issues)

### 8.4. Medicion CBOE

    import urllib.request
    u = 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv'
    req = urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=15) as r:
        print(r.status, len(r.read().splitlines()))

---

## 9. Anexo — Contrato VOLATILITY_INDEX

| Propiedad | Valor |
|---|---|
| Nombre | VOLATILITY_INDEX |
| Familia | INDEX |
| Herencia | INDEX_EOD_USA |
| Universo | ^VIX, ^VIX3M, ^VXN |
| per_ticker_lag | {'^VIX': 1, '^VIX3M': 1, '^VXN': 1} |
| max_lag_days | 1 |
| Definicion | src/temporal_contracts/volatility_index.py |
| Registro | src/temporal_contracts/registry.py |

### 9.1. Consumidores directos de ^VIX3M

- regimes/macro_regime.py:52 — vix_term = tanh_normalize(vix3m - vix)
- indicators/mte.py:1747 — VIX term structure
- indicators/volatility_structure.py:46 — term_ratio, term_slope
- src/report/volatility_mte.py:18 — visualizacion VIX3M/VIX

Todos operan hoy sobre serie vacia.

### 9.2. Fuentes consultadas en el registry

src/instrument_registry.py declara para ^VIX3M: Yahoo, Polygon, Tiingo, Finnhub, FMP, Twelve Data. Ninguna alternativa probada en esta sesion salvo Yahoo (roto) y CBOE (via externa).

### 9.3. CBOE — fuente verificada

- Endpoint: https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv
- Formato: CSV, cabecera DATE,OPEN,HIGH,LOW,CLOSE
- Cobertura verificada: 4273 lineas (~17 anos de sesiones)
- Autenticacion: ninguna
- Frecuencia: diaria (post-cierre CBOE)
- Retencion: historica completa, no limitada

---

**Fin del informe E5.**
**Fecha de redaccion:** 2026-09-17.
**HEAD de referencia:** 18601ee.
**Autor:** Ingeniero supervisor del Radar de Rotacion Sectorial.
**Destinatario:** Auditor externo.
**Accion esperada:** dictamen sobre Opcion A/B/C/D.
