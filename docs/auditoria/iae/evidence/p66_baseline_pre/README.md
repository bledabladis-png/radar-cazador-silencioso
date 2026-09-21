# IAE - P66 baseline pre-ciclo (freshness)

**Objeto:** demostrar empiricamente que los 3 fallos de `tests/test_freshness.py`
observados en la suite del ciclo P66 NO son regresion introducida por el ciclo,
sino estado local de los parquets (gitignored) en disco.

**Origen:** dictamen #41 (auditoria de salida P66), reserva de auditoria
sobre la suite global.

**Fecha:** 2026-09-21.
**Autor:** Ingeniero Supervisor.
**Naturaleza:** evidencia directa, no normativa.

---

## 1. Metodologia

`git worktree add --detach D:\_baseline_p66_tmp 64b0637`

- `64b0637` es el commit inmediatamente anterior al inicio del ciclo P66
  (padre de `bff0945`, primer commit del ciclo).
- En el worktree se copian los parquets actuales (gitignored) desde
  `D:\Macro_Sectorial\data\` para reproducir exactamente el estado local.
- Se ejecuta `tests/test_freshness.py`.

Restricciones: worktree detached; no afecta al working tree principal;
worktree eliminado tras la prueba. Sin IO de red. Sin `datetime.now()`.

---

## 2. Resultado

### Codigo pre-P66 (`64b0637`) + parquets actuales (2026-09-16)

    tests/test_freshness.py -q --tb=line

    3 failed, 34 passed, 2 skipped

Fallos:

- `test_market_data_fresh`     -> market_data stale: 5 dias (max 4)
- `test_stock_prices_fresh`    -> stock_prices stale: 5 dias (max 4)
- `test_european_tickers_recent` -> solo 19/51 (37%) europeos recientes

### Codigo actual (`b0f8850`) + mismos parquets

    tests/ validation/ -q

    3 failed, 1067 passed, 2 skipped

Fallos: **exactamente los mismos 3**, con los mismos mensajes.

---

## 3. Conclusion

**Los 3 fallos son preexistentes al ciclo P66.** No son regresion.
No hay ninguna modificacion del ciclo P66 que afecte a `test_freshness.py`
ni a los parquets:

    git log --oneline 64b0637..HEAD -- tests/test_freshness.py
    (sin salida)

    git log --oneline 64b0637..HEAD -- data/market_data.parquet data/stock_prices.parquet
    (sin salida)

Causa raiz: `data/market_data.parquet` y `data/stock_prices.parquet` en disco
tienen `index.max() = 2026-09-16`, 5 dias antes de la ejecucion (2026-09-21).
`MAX_DAILY_AGE = 4`. Los parquets son gitignored (`.gitignore` lineas 6 y 8),
por lo que cada clone/CI partira de estado limpio (sin parquets -> tests
skipped, no failed).

Los parquets se regeneran con el proximo `daily_run.yml` (cron `0 4 * * *`).
No se regeneran a mano en este ciclo: forzar actualizacion fuera de cron
activaria el patron K-RUN-OUT-OF-WINDOW-01.

---

## 4. Ficheros

    baseline_output.txt    salida cruda del pytest en el worktree pre-P66
    README.md              este documento
    HASHES.txt             SHA-256 de los dos anteriores

---

## 5. Referencias

- Dictamen #41 (auditoria de salida P66), seccion 5.
- `.gitignore` lineas 6 y 8.
- `tests/test_freshness.py` (no modificado en `64b0637..HEAD`).

---

Fin del README. Baseline ejecutado 2026-09-21 sobre worktree detached.