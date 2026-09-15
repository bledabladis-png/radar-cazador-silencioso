# FU-018-2 - Contrato del modelo temporal minimo (EOD eligibility)

**Fecha:** 2026-09-15
**Commit de referencia:** e645da4
**Estado:** APROBADO CONDICIONADO (dictamen auditor externo 2026-09-15)
**Antecede a:** FU-018-3 (implementacion en 3 commits)

---

## 1. Contrato aprobado

### 1.1. `src/instrument_registry.py`

Anadir UNA funcion (metadato del instrumento):

```python
def get_market(ticker: str) -> str:
    """Devuelve el mercado del ticker segun sufijo.

    Sin sufijo  -> 'US_EQUITY'
    .L          -> 'LSE'
    .DE         -> 'XETRA'
    .MC         -> 'BME'
    .PA .AS .MI -> 'EURONEXT'
    Otro        -> 'UNKNOWN' (nunca adivinar)
    """
Nota: el mapping es valido para el universo de tickers soportado actualmente. No es una afirmacion universal de que todo ticker .L pertenece a LSE en cualquier universo futuro.

1.2. src/market_hours.py (nuevo)
TRES funciones con responsabilidades separadas:

python
def is_trading_session(market: str, session_date) -> bool:
    """True si `session_date` es dia de negociacion en `market`.

    - US_EQUITY: reutiliza is_market_day (calendario oficial NYSE).
    - LSE/XETRA: PROVISIONAL, lunes-viernes. Documentar limitacion.
    - BME/EURONEXT: no se usan en FU-018-3, pero disponibles.
    - market desconocido -> raise UnknownMarketError.
    """

def get_session_close(market: str, session_date) -> datetime:
    """Hora de cierre efectiva de `session_date` en `market`.

    - Devuelve datetime timezone-aware en la tz del mercado.
    - Aplica excepciones por fecha si existen.
    - Si session_date no es dia de negociacion -> raise ValueError.
      (Nunca devolver None o False silencioso.)
    """

def is_session_closed(market: str, session_date, reference_date) -> bool:
    """True si la sesion de `session_date` ya cerro en `reference_date`.

    Precondiciones:
    - reference_date timezone-aware, si no -> ValueError.
    - market desconocido -> UnknownMarketError.
    - session_date debe ser dia de negociacion.
      Si no -> raise ValueError.
      (No devolver False silencioso: "no negociable" != "aun abierto".)

    Logica:
    - session_date > reference_date.date() -> False + WARN
    - session_date < reference_date.date() -> True
    - session_date == reference_date.date() -> comparar reference_date
                                               con get_session_close()
    """
1.3. config/market_close_regular.csv (nuevo)
csv
market,timezone,regular_close
US_EQUITY,America/New_York,16:00
LSE,Europe/London,16:30
XETRA,Europe/Berlin,17:30
BME,Europe/Madrid,17:30
EURONEXT,Europe/Paris,17:30
Hora local del mercado, no UTC. zoneinfo aplica DST automaticamente.

1.4. config/market_close_exceptions.csv (nuevo, vacio)
csv
market,date,close,reason
Regla: no se anade ninguna fila sin documento oficial del mercado. Se llena en fase posterior con verificacion.

2. Correcciones integradas (dictamen final)
C1. No presentar lunes-viernes europeo como calendario oficial
is_trading_session para LSE/XETRA se implementa con la regla PROVISIONAL lunes-viernes. Se documenta explicitamente:

python
# LSE / XETRA: PROVISIONAL. No es calendario oficial completo.
# Festivos locales no contemplados. Solo valido para determinar
# "no es fin de semana". Se ampliara con calendarios oficiales.
Tests reconocen esta limitacion (no testean festivos europeos).

C2. Fecha no negociable != "sesion abierta"
is_session_closed no devuelve False silencioso para fechas no negociables. raise ValueError explicito. El caller debe comprobar is_trading_session primero.

C3. get_session_close() rechaza fechas no negociables
Si session_date no es dia de negociacion: raise ValueError. Nunca devolver un valor por defecto que se interprete como "cierre valido".

C4. Scope FU-018-3 = Yahoo USA/UK + Xetra
Sin tocar Euronext/BME.

Sin tocar FU-002, manifest, BackupProvider.

Sin tocar indices/futuros/FX.

Sin tocar cron.

3. Reglas del contrato
3.1. reference_date timezone-aware obligatorio
python
if reference_date.tzinfo is None:
    raise ValueError("reference_date must be timezone-aware")
3.2. Mercado desconocido -> excepcion
python
class UnknownMarketError(Exception):
    """Raised when market is not recognized by market_hours."""
Se propaga. El caller EOD excluye la observacion (no elegible). Nunca fail-open.

3.3. Filtro: solo si last_date == reference_date.date()
text
last_date
   |
get_market(ticker) -> market
   |
is_trading_session(market, last_date)?
   |-- False -> WARN (no deberia estar aqui esa fila)
   |-- True
        |
        is_session_closed(market, last_date, reference_date)?
           |-- True  -> conservar
           |-- False -> eliminar
Ejemplo: last_date=14/09, ref=15/09 -> is_session_closed -> True -> conservar. No se toca nada.

4. Alcance FU-018-3
SI entra:

src/market_hours.py (nuevo).

config/market_close_regular.csv (nuevo).

config/market_close_exceptions.csv (nuevo, vacio).

get_market() en instrument_registry.py.

Tests unitarios del modelo temporal.

Filtro en Xetra (commit 2).

Filtro en Yahoo (commit 3).

Propagacion de reference_date.

NO entra:

Fuentes internas de calendarios oficiales.

Half-days europeos (verificacion pendiente).

FU-002, manifest, BackupProvider.

Indices, futuros, FX.

Cron.

market_calendar.py existente (solo se reutiliza is_market_day para US_EQUITY).

Euronext / BME (sin cambios).

5. Tests unitarios (Commit 1)
#    Caso    Esperado
1    US_EQUITY, sesion ayer, ref hoy    True
2    US_EQUITY, sesion hoy, ref 15:00 Madrid    False (antes de 16:00 ET)
3    US_EQUITY, sesion hoy, ref 23:00 Madrid    True (despues de 16:00 ET)
4    XETRA, sesion hoy, ref 16:47 Madrid    False (antes de 17:30 Berlín)
5    XETRA, sesion hoy, ref 19:00 Madrid    True
6    LSE, sesion hoy, ref 15:00 Madrid    False (antes de 16:30 Londres)
7    LSE, sesion hoy, ref 18:00 Madrid    True
8    reference_date naive    ValueError
9    market desconocido    UnknownMarketError
10    US_EQUITY, festivo Labor Day 2026-09-07    is_trading_session -> False
11    get_session_close sobre fecha no negociable    ValueError
12    is_session_closed sobre fecha no negociable    ValueError
13    session_date > reference_date.date()    False + WARN
14    Cambio DST (2026-10-25 / 2026-11-01)    Hora local correcta
15    get_market sufijos (.L, .DE, .MC, .PA, .AS, .MI, sin sufijo)    Correcto
16    get_market sufijo desconocido (.ZZ)    'UNKNOWN'
6. Plan de implementacion
Commit 1 - Modelo temporal (FU-018-3a)
src/market_hours.py

config/market_close_regular.csv

config/market_close_exceptions.csv (vacio)

get_market() en instrument_registry.py

tests/test_fu018_market_hours.py (16 tests)

Commit 2 - Xetra (FU-018-3b)
Firma XetraProvider.get_prices(..., reference_date=None).

Filtro sobre ultima fila antes de devolver.

tests/test_fu018_xetra_eod.py.

Commit 3 - Yahoo (FU-018-3c)
Filtro en stock_data_loader.py sobre lotes Yahoo (USA + .L).

tests/test_fu018_yahoo_eod.py.

Verificacion CI
workflow_dispatch en horario intradia.

Confirmar max(index parquet) == 2026-09-14.

Confirmar que cada provider conserva el 14/09.

7. Dictamen final
APROBADO CONDICIONADO con las siguientes condiciones integradas:

Lunes-viernes europeo: PROVISIONAL, documentado, no presentado como oficial.

Fecha no negociable: rechazo explicito, no False silencioso.

get_session_close rechaza fechas no negociables.

Scope exclusivo Yahoo USA/UK + Xetra.

GO para FU-018-3 Commit 1.

Fin de FU-018-2 v2. Commit de referencia: e645da4. Fecha: 2026-09-15.
