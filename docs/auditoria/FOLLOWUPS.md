# Follow-ups tecnicos registrados

## FU-001 — ffill multi-calendario en merge global (L352)

- **Origen:** C1, sesion 2026-09-12. Dictamen auditor.
- **Descripcion:** `stock_data_loader.py:352` aplica `ffill(limit=3)` sobre el DataFrame consolidado (Yahoo USA + Euronext + Xetra + BME). Ese ffill puede reintroducir valores imputados en tickers fuera del calendario NYSE, pese a haber sido clasificados como DATA_ISSUE en su etapa.
- **Evidencia:** tras el run C1 (2026-09-12), 20 tickers .L marcados como `DATA_ISSUE / MISSING_CLOSE_EXPECTED_SESSION` vuelven a mostrar `close[-1]==close[-2]` en el parquet final por el ffill de L352.
- **Impacto:** no afecta al pipeline USA actual ni invalida C1. Afecta a la coherencia de la marca DATA_ISSUE para tickers no-USA.
- **Clasificacion:** P2/P3 integridad tecnica localizada (auditor).
- **Accion:** disenar capa de proteccion por calendario (NYSE/BME/Xetra/Euronext) antes de generalizar la arquitectura a Europa.
- **Bloqueante:** no para C1/C2/C3. Obligatorio antes de declarar cerrada la nueva arquitectura de validacion multi-calendario.

## FU-002 — Validacion circular en BackupProvider

- **Origen:** informe v2, hallazgo B4. Dictamen auditor.
- **Descripcion:** `_validate_with_cache` compara el nuevo dato contra los mismos parquets que luego sobreescribe. Si el cache esta contaminado, la validacion pasa.
- **Evidencia:** T7 mostro 41/112 tickers contaminados en reference_cache.
- **Clasificacion:** P2 Technical Debt estructural.
- **Accion:** referencia independiente del artefacto validado.
- **Bloqueante:** no para C1/C2/C3. Criterio de cierre: antes de declarar cerrada la nueva arquitectura de validacion.

## FU-003 — Cosmetico: signos +0.00 y flechas ->

- **Origen:** prompt maestro v6.8 seccion 15.2.
- **Descripcion:** formato de +0.00 en scores pequenos; flecha -> en algunos textos.
- **Clasificacion:** P3 cosmetico.
- **Bloqueante:** no.

## FU-004 — Origen del colapso `macro_regime.csv` (315 -> 1) sin identificar

- **Origen:** sesion 2026-09-12, verificacion E2E-pre.
- **Descripcion:** `outputs/history/macro_regime.csv` paso de 314 filas (HEAD `829f9c6`) a 1 fila (working tree). El codigo actual de `save_regime_history` (C4-code) es coherente con el resultado final, pero no explica el colapso inicial.
- **Evidencia:**
  - Snapshot `outputs/audit/c4_code_verif/macro_regime_pre.csv` (06:40:45) ya mostraba 1 fila antes del run.
  - HEAD `829f9c6` tenia 314 filas con strings de fecha distintos (`2026-06-19 16:38:26.367013`, etc.), todas B2-contaminadas.
  - `save_regime_history` con `dtype=str` + `drop_duplicates(subset=['date'])` no habria colapsado 314 strings distintos.
- **Impacto:** ninguno funcional. El estado actual (1 fila, `2026-09-11`, formato `YYYY-MM-DD`) es el deseado.
- **Hipotesis:** algun paso de la bateria de verificacion C4-code (pre-06:40) escribio al path real en lugar de tmp, o hubo `git checkout` selectivo. No confirmado.
- **Clasificacion:** P3 documental.
- **Accion:** ninguna urgente. Si reaparece un colapso similar en otro writer, investigar el origen con mas instrumentacion.
- **Bloqueante:** no.

