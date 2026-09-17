# FU-018-1 - Inventario de semantica temporal por provider

**Fecha:** 2026-09-15
**Commit de referencia:** e59e4e8
**Estado:** cerrado - GO (dictamen auditor externo 2026-09-15)
**Objetivo:** determinar que representa el campo `Close` que devuelve cada provider antes de disenar cualquier solucion temporal.

---

## 1. Metodologia

Para cada provider se ha inspeccionado:

- El endpoint o API utilizada.
- El campo del que se extrae `Close`.
- Si el endpoint es "historico" (busca sesiones pasadas) o "vivo" (sigue la sesion actual).
- Si devuelve o no la vela del dia en curso cuando el mercado esta abierto.
- Los datos reales del run de las 14:47 UTC del 15/09/2026 (todos los mercados abiertos).

---

## 2. Tabla de inventario

| # | Universo | Tickers | Provider | Endpoint | Campo -> Close | Naturaleza | Devuelve dia en curso |
|---|---|---|---|---|---|---|---|
| 1 | USA | 262 | Yahoo Finance | `yf.download(period='5y')` | `Close` | Intradia + EOD | Si (vivo) |
| 2 | UK `.L` | 20 | Yahoo Finance | `yf.download(period='5y')` | `Close` | Intradia + EOD | Si (vivo) |
| 3 | Euronext | 13 | Euronext | `getHistoricalPricePopup` | `close` (no `last`) | EOD | No |
| 4 | Xetra | 19 | Deutsche Boerse | WS MDS `resolution=1D` DELAYED | `close` | Intradia + EOD | Si (parcial) |
| 5 | BME | 19 | Bolsa de Madrid | `HistoricalSharesPrices` | `close` | EOD | No |

---

## 3. Evidencia por provider

### 3.1. Yahoo (USA + UK)

**Endpoint:** `yf.download(batch, period='5y', auto_adjust=True)`

- Resolucion por defecto para `period='5y'` es `1d`.
- Yahoo devuelve la vela del dia en curso con el precio vivo cuando el mercado esta abierto.

**Evidencia** (log del run 14:47 UTC):
[YAHOO_RAW] batch=batch_1
expected_session=2026-09-14
raw_last_date=2026-09-15 <- fila del dia en curso
close_nan=0/10 (0.0%) <- con valor (precio vivo)

text

**Semantica de Close:** precio vivo cuando el mercado esta abierto; cierre cuando ha cerrado.

**Requiere filtrado:** SI.

### 3.2. Euronext

**Endpoint:** `POST https://live.euronext.com/en/ajax/getHistoricalPricePopup/{instrument}`

**Parametros:**
- `enddate`: fecha fin
- `nbSession`: numero de sesiones a devolver
- `adjusted=Y`: precios ajustados

**Parseo:** tabla HTML -> filas con `date` en formato `DD/MM/YYYY`.

**Columnas devueltas:** `date, open, high, low, last, close, num_shares, turnover, vwap`

**Dos campos relevantes:**
- `last` = ultimo precio negociado (intradia si mercado abierto).
- `close` = cierre oficial de la sesion.

El provider extrae **`close`**.

**Evidencia** (log del run 14:47 UTC):
[EURONEXT] AIR.PA gap=4d, recuperando 10 sesiones
[EURONEXT] AIR.PA OK (301 filas, 2025-07-11 -> 2026-09-14)

text

Ultima fecha devuelta: **2026-09-14**, no 15. **Con el mercado abierto el 15.**

**Semantica de Close:** cierre de sesion (EOD).

**Requiere filtrado:** NO.

### 3.3. Xetra

**Endpoint:** WebSocket MDS en `wss://api.live.deutsche-boerse.com/v1/mds/ws`

**Parametros de la query:**
- `resolution: "1D"`
- `quality: "DELAYED"`
- `start`, `end`

**Semantica del campo `close`:** cierre de la vela. Pero con `resolution=1D`, si el mercado esta abierto, la vela del dia en curso aun no ha cerrado -> `close` = **precio vivo**.

**Evidencia** (log del run 14:47 UTC):
[XETRA] SIE.DE OK (303 filas, 2025-07-08 -> 2026-09-15)

text

Ultima fecha devuelta: **2026-09-15**. **Con el mercado abierto.**

Comparar con Euronext: mismo run, mismo mercado abierto, pero Xetra si devuelve la vela del 15. **Es un caso de "vela diaria en construccion", no EOD.**

**Semantica de Close:** precio vivo cuando el mercado esta abierto; cierre cuando ha cerrado.

**Requiere filtrado:** SI.

### 3.4. BME

**Endpoint:** `GET https://apiweb.bolsasymercados.es/Market/v1/EQ/HistoricalSharesPrices`

**Parametros:**
- `from`, `to`: rango de fechas
- `pageSize=0` (todas las filas)

**Campos devueltos por fila:** `date (YYYYMMDD), reference, high, low, close, volume`

**Nota lateral:** `open` se mapea desde `reference` (precio de referencia de la sesion, no open de subasta).

**Evidencia** (log del run 14:47 UTC):
[BME] SAN.MC gap=4d, recuperando desde 20260914
[BME] SAN.MC OK (304 filas, 2025-07-08 -> 2026-09-14)

text

Ultima fecha devuelta: **2026-09-14**. **Con el mercado abierto el 15.**

**Semantica de Close:** cierre de sesion (EOD).

**Requiere filtrado:** NO.

---

## 4. Conclusiones

### 4.1. Dos grupos operativos

**Grupo A - Providers que entregan EOD por construccion del endpoint (32 tickers):**

- Euronext (13)
- BME (19)

No necesitan filtrado. Sus endpoints actuales son historicos y devuelven la ultima sesion cerrada.

**Grupo B - Providers que pueden devolver la vela del dia en curso (301 tickers):**

- Yahoo USA (262)
- Yahoo UK `.L` (20)
- Xetra (19)

**Si** necesitan filtrado cuando su mercado esta abierto.

### 4.2. Precision de alcance

En el universo analizado, tres vias de datos pueden introducir en el DataFrame una observacion correspondiente a la sesion en curso antes de su cierre EOD: Yahoo USA, Yahoo UK y Xetra. Euronext y BME, en sus endpoints actualmente utilizados, ya entregan la ultima sesion cerrada.

**Nota:** la clasificacion queda ligada al endpoint actual, no a una propiedad eterna del proveedor. Si en el futuro se cambia el endpoint de Euronext o BME, la clasificacion debe revisarse.

### 4.3. Hallazgo lateral: `Open` en BME

BME mapea `open <- reference`, que no es el open de subasta. Es el precio de referencia de la sesion. **Fuera de scope de FU-018**, anotado para futura revision.

---

## 5. Impacto en el diseno de FU-018

El modelo temporal **no necesita**:

- Un modulo global `market_hours.py` con horarios de todos los mercados.
- Aplicar mascara a los 313 tickers.
- Revisar half-days de Euronext/BME.

**Necesita:**

- Un helper minimo que determine, para cada provider del Grupo B, si la sesion correspondiente a la ultima fecha devuelta ya ha cerrado en `reference_date`.
- Eliminacion de la fila no-EOD en cada provider del Grupo B **antes del concat**.
- Tests comparativos en el run 14:47 UTC.

---

## 6. Evidencia cruzada

**Run del 2026-09-15 14:47 UTC** (todos los mercados abiertos):

| Provider | Ultima fecha devuelta | Coincide con semantica esperada |
|---|---|---|
| Yahoo USA | 2026-09-15 | Si (intradia) |
| Yahoo .L | 2026-09-15 | Si (intradia) |
| Euronext | 2026-09-14 | Si (EOD) |
| Xetra | 2026-09-15 | Si (intradia, vela parcial) |
| BME | 2026-09-14 | Si (EOD) |

---

**Fin del inventario FU-018-1. Dictamen auditor externo: CERRADO - GO.**
