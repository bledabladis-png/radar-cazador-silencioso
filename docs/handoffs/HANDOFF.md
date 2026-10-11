# HANDOFF - Radar de Rotacion Sectorial

Documento de traspaso entre sesiones. Autocontenido. Pegar al inicio
del nuevo chat junto con `Consolidacion_Documentos/00_ARRANQUE.md`.

**Este fichero se sobrescribe cada sesion.** El historico queda en git.
No crear variantes con fecha.

---

## 1. ESTADO AL CIERRE (2026-10-11 23:00 CEST)

Rama activa:    main
HEAD:           ceb21d8a
Sincronizacion: main == origin/main
Tests:          3204 passed + 5 skipped + 0 failed
Validation Gate: 10/10
Working tree:   limpio

---

## 2. QUE SE HIZO 08-11 OCT (3 sesiones)

### 2.1. Verificacion del sistema (08-oct)

Revision bajo peticion del usuario: estado real del pipeline,
artefactos, frescura, providers.

Fixes:
- c09a268c: check_iae_section aislaba STALE al bloque correcto.
- d495dc1e: save_regime_history prioriza effective_date.
- 8e8826f5: check_workflows mira ultimo run + trata cancelled como WARN.
- 0017bf4f: vigilancia de etf_holdings.csv / index_holdings.csv.

### 2.2. Retirada modulo Wyckoff legacy (10-oct)

- indicators/wyckoff.py borrado (17747a98).
- 2 tests legacy retirados. Regresion I2 / K-INDEX-RANGE-01 portada
  a test_wyckoff_v1_contract.py.
- 5e cerrado en plan de migracion (f8d6dff9).
- Motivo: v1.8 en produccion desde 08-oct, CI estable 08/09/10-oct,
  consumidores residuales no productivos.

### 2.3. Auditoria del reporte 10-oct (15 hallazgos)

3 bugs reales corregidos:

- H-01 (9c33ed3b): candado `is_market_day(reference_date)` en
  breadth_metrics.py bloqueaba escritura en runs de fin de semana
  que producen la sesion del viernes. Fix: usar
  `last_expected_market_date(reference_date)`.
- H-03 (1d3e0a17): data_quality.py declaraba 15 fuentes.
  Anadidas 11. SLPM retirado (congelado desde julio sin writer).
- H-09 (4a604b21): heterogeneity_type retirado (constante, redundante).

Falsos positivos: H-02, H-04..H-08, H-11..H-15.

### 2.4. Duplicidades del reporte (11-oct)

- D-02 (fd56dad3): seccion "Persistencia sectorial" eliminada
  (duplicaba columna Persist en Rankings).
- D-03 (fd56dad3): seccion "Tactical Leaders" eliminada
  (duplicaba 5 de 6 columnas).
- D-01 (128f1dd7): Data Freshness vs Calidad. Ambas se mantienen.
  Nota cruzada. Miden por caminos distintos (runtime vs CSV).

### 2.5. Run manual 11-oct (gap 09-oct)

El bug H-01 bloqueo la escritura de la fila 09-oct en
sector_breadth.csv y sus 3 derivados durante el run del sabado
10-oct. El receipt del target se emitio igual, gate no reintenta.

Run manual domingo 11-oct (ceb21d8a). Recuperadas las 4 filas
09-oct. Verifico los 4 fixes anteriores en produccion.

**NO es parche recurrente.** Con 9c33ed3b el run del sabado 03:17 UTC
escribe siempre el cierre del viernes. El gap fue daño puntual.

---

## 3. PENDIENTES VIVOS

### 3.1. Bloqueados por terceros
- C2 (920): discrepancia H1-B. Local -4.264.449.012 vs auditor
  -4.264.449.932. Sin comando del auditor, no contrastable.
- SOW protocol v6: borrador 48_sow_protocol_v6.md pendiente firma.
- gold-standard-v4: archivada como tag.

### 3.2. Bloqueados por datos
- Wyckoff 5b.X v3: bloqueado hasta n_confirmed_H20_complete >= 550
  (~2027). Script y protocolo inmutables.

### 3.3. Con fecha futura
- H5.3: cron trimestral 13F 20-nov-2026 06:17 UTC.

### 3.4. Deuda sin urgencia
- Auto-backfill de CSVs derivados: feature nueva. Diseno minimo
  ~30 min. Pendiente decision. El fix 9c33ed3b ya previene el bug
  que lo justifico.
- 00_ARRANQUE a ~250 B del umbral 11 KB.
- Auditoria externa 2026-09-26 (142 hallazgos): informes en
  docs/auditoria/radar/audits/. 71 MEDIA + 65 BAJA sin tracking.
- Cosmeticos del reporte: D-04 (PCR z vs Robust Z-Score),
  D-07 (Lider retorno absoluto vs relativo).

---

## 4. COMO ARRANCAR

    Set-Location D:\Macro_Sectorial
    git log --oneline -5
    git status -sb
    py scripts\generate_estado_sistema.py
    py -m pytest tests/ -q -n 12 --tb=line
    py -m pyflakes . ; py -m compileall . -q

Esperado:
- HEAD = ceb21d8a o posterior.
- ahead 0, behind 0.
- working tree limpio.
- 3204 passed + 5 skipped.
- pyflakes silencio, compileall OK.

Si divergencia: investigar antes de tocar nada.

---

## 5. NOTAS CRITICAS

- **No usar run.py en horario de mercado** (13:30-20:00 UTC).
  Politica: snapshot pre/post + revertir, salvo publicacion
  deliberada. Aplicado en 01_METODO seccion 4.

- **EOL mixto en corpus.** 00_ARRANQUE y 01c_LECCIONES en CRLF.
  01_METODO, 02_ARQUITECTURA, 05_BITACORA en LF. 04_HISTORICO
  mixto. Verificar con repr() antes de patch multilinea.

- **Patrones de bug** en 01b_PATRONES.md. **Trampas PowerShell** en
  01_METODO seccion 6bis.

- **No confundir diagnostico cerrado con hipotesis fuerte.**

- **Antes de tocar un writer/reader temporal**, verificar la
  cadena completa: productor, consumidor, render.

---

## 6. REFERENCIAS

Corpus normativo en `Consolidacion_Documentos/` (12 docs):
00_ARRANQUE (rol + reglas), 01_METODO (patch/verificacion),
01b_PATRONES, 01c_LECCIONES, 02_ARQUITECTURA, 03_IAE,
04_HISTORICO (temas), 05_BITACORA (sesiones), 06_IAE_P65_P66,
07_RUNBOOK, 08_AUDITORIA_SECTOR_REGIME, ESTADO_SISTEMA (auto).

Expedientes: docs/auditoria/wyckoff/, docs/auditoria/iae/golden/,
docs/auditoria/s03_deep/, docs/auditoria/radar/audits/.

---

**Fin del handoff.**
