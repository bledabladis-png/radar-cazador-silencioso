# Decisión: descartar LSEG widget para tickers .L

**Fecha:** 2026-09-11
**Contexto:** investigación para automatizar los 20 tickers de London Stock Exchange.
**Resultado:** descartado por complejidad operativa.

---

## 1. Resumen ejecutivo

Se evaluó el uso del widget de Refinitiv (`refinitiv-widgets.financial.com`)
para obtener históricos diarios de la LSE. Se reconstruyó el flujo completo
de autenticación a partir del JavaScript de la web pública. **Descartado por
cadena de autenticación demasiado compleja y frágil.**

Los 20 tickers `.L` siguen obteniéndose vía **Yahoo Finance**. Cobertura
actual: 313/313 tickers (100%), sin impacto operativo.

---

## 2. Flujo de autenticación descubierto

El endpoint real es:

    GET https://refinitiv-widgets.financial.com/rest/api/timeseries/historical

Requiere una cadena de 5 saltos:

    1. GET  api.londonstockexchange.com/api/gw/feedhandler/token/saml
       -> Devuelve encodedToken (JWT/SAML)

    2. Base64 decode
       -> SAMLResponse

    3. POST refinitiv-widgets.financial.com/auth/api/v1/sessions/samllogin
       -> Devuelve SID (session id)

    4. POST refinitiv-widgets.financial.com/auth/api/v1/tokens (header sid: <SID>)
       -> Devuelve JWT (valido 5 minutos)

    5. GET  refinitiv-widgets.financial.com/rest/api/timeseries/historical
       (header jwt: <JWT>)
       -> Devuelve datos historicos

---

## 3. Motivos del descarte

| Razon | Impacto |
|-------|---------|
| **5 saltos de autenticacion** | 5 puntos de fallo potencial |
| **JWT valido 5 minutos** | Hay que renovar cada ~4 min. Un daily_run (10-15 min) requeriria 3-4 renovaciones |
| **SID viene del servidor** | No se puede generar localmente; requiere peticion previa |
| **CORS estricto** | Requiere `Origin: https://www.londonstockexchange.com` |
| **Fragilidad** | financial.com puede cambiar el flujo sin aviso |
| **Alternativas viables** | Yahoo cubre los 20 tickers con calidad aceptable |

**Conclusion:** el ROI no justifica 4-6 horas de implementacion + mantenimiento
continuo ante un widget comercial que puede cambiar sin avisar.

---

## 4. Alternativas evaluadas

| Alternativa | Estado |
|-------------|:------:|
| Yahoo Finance | OK - Operativo (actual) |
| LSEG Data Platform oficial | NO - Requiere contrato comercial |
| Widget Refinitiv (investigado) | NO - Descartado |
| BME-style endpoint publico LSE | NO - No existe |
| Stooq | NO - Bloqueado por WAF + PoW |
| Investing.com | NO - Protegido por Cloudflare |

---

## 5. Estado actual

- **20 tickers .L** cubiertos por Yahoo Finance.
- **Cobertura global:** 313/313 (100%).
- **FAILED:** 0.
- **Frescura:** ~1 dia de retraso (equivalente a Yahoo US).

---

## 6. Reevaluacion futura

Reconsiderar si:

- LSEG ofrece un endpoint publico (poco probable).
- Yahoo degrada la calidad para .L (no observado).
- Se valora otro proveedor comercial con API simple.

---

*Documentado el 2026-09-11 por auditoria tecnica.*
