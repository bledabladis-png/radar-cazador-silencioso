# NIPC_INFORME_ESTADO_POST_FIXES_F24

**Objeto:** comunicacion al auditor del estado del sistema tras aplicar los 6 fixes mecanicos autorizados.
**HEAD al redactar:** d462f72.
**Destinatario:** auditor externo.
**Documento complementario:** NIPC_INFORME_ENTREGA_F24.md (hash 0763f16b..., inmutable).
**Estado:** BORRADOR pendiente de envio.
**Autor:** Ingeniero Supervisor.
**Fecha:** 2026-09-20.

---

## 0. Resumen ejecutivo

Desde la emision del dictamen sobre la revision estructural (2026-09-20,
materializado en NIPC_DICTAMEN_REVISION_ESTRUCTURAL_2026-09-20.md, hash
3d004540...) se han aplicado los 6 fixes mecanicos autorizados por ese
dictamen (GO FIX para P51, P32, P14-BIS, P31, P18/P26; GO CONDICIONADO
para P70):

**Nota sobre F2.4:** F2.4 sigue PENDIENTE. Este informe no cierra F2.4 ni
presupone su resultado. Se emite como comunicacion complementaria al
informe de entrega F2.4 (hash 0763f16b...), informando del estado del
sistema tras aplicar los fixes autorizados por el dictamen de revision
estructural.

    P51      linea corta preservada como anomalia
    P32      contador de descartes silenciosos
    P14-BIS  fillna condicionado al _merge del FULL OUTER JOIN
    P31      nipc_total_available + docstring de _sum_delta
    P18/P26  project_root por env var + fallback derivado
    P70      guarda explicita invariante HR_PLUS_RESTATEMENT

**Ninguno de estos fixes toca los contratos semanticos P38/P60/P61.** El motor
NIPC (identity resolution, coverage pairwise, TARGET/RESOLVED/PAIRED) sigue
sin cambios respecto del estado en que se redacto el informe F2.4.

**Consecuencias para F2.4:**

1. Las 4 decisiones solicitadas (D1-D4) **no cambian**. El informe F2.4 sigue
   siendo el documento de referencia para la solicitud.
2. El estado del sistema cambio en metricas: 95 commits ahead (era 82),
   946 tests locales (era 941), 6 ficheros de codigo productivo modificados.
3. La guarda de P70, propuesta en el dictamen P70 como "fix a aplicar en
   fase 5", **ya esta aplicada**. Esto refuerza D3 (cierre de P70) sin
   modificar la decision solicitada.

**Este informe anade 8 preguntas nuevas derivadas del estado post-fixes.**
Ninguna bloquea F2.4; todas son para preparar el paso 4 (tests especificos)
o para clarificar detalles del contrato antes de su implementacion.

---

## 1. Estado exacto del sistema

### 1.1. Metricas

| Metrica | Valor |
|---|---:|
| HEAD local | 1575d17 |
| origin/main | 9d4a81e |
| Ahead | 89 |
| Working tree | limpio |
| Tests locales | 946 passed + 2 skipped |
| pyflakes | 0 warnings |
| compileall | OK |
| Push | NO (regla local-first IAE) |

### 1.2. Documentos clave (hashes)

| Documento | Hash SHA-256 |
|---|---|
| NIPC_COVERAGE_POLICY.md (v1.0) | 57f2d01f... |
| NIPC_CONTRATOS_SEMANTICOS_v1.md | c02d202f... |
| NIPC_P70_DICTAMEN.md | e8d4e4e8... |
| NIPC_INFORME_ENTREGA_F24.md | 0763f16b... |
| NIPC_DICTAMEN_REVISION_ESTRUCTURAL_2026-09-20.md | 3d004540... |
| INSTITUTIONAL_ACCUMULATION_REVISION_ESTRUCTURAL_2026-09-20.md | 318f02e8... |

### 1.3. Bloqueos vigentes (sin cambios)

    P38 / P60 / P61              PENDIENTE F2.4
    THRESHOLD_1                  UNDEFINED
    THRESHOLD_2                  UNDEFINED
    Gate-NIPC.2                  BLOQUEADO
    Gate-NIPC.3                  NO AUTORIZADO
    OpenFIGI masivo              NO AUTORIZADO
    Policy v1.3 aplicacion       NO AUTORIZADA
---

## 2. Los 6 fixes mecanicos aplicados

Los 6 fixes fueron autorizados por el dictamen del 2026-09-20 sobre la
revision estructural (NIPC_DICTAMEN_REVISION_ESTRUCTURAL_2026-09-20.md,
hash 3d004540...). Ese dictamen asigno:

    GO FIX          P51, P32, P14-BIS, P31, P18/P26
    GO CONDICIONADO P70


Todos ellos corresponden a bloqueos con estado **GO FIX** en el dictamen
formal del auditor (2026-09-20) o **GO CONDICIONADO** en el dictamen P70.
Ninguno requería esperar F2.4.

### 2.1. P51 — linea corta preservada como anomalia

**Commit:** 0f1e5a4.
**Ficheros:** `src/institutional_accumulation/sec_13f/identity/sec13f_list.py`,
`tests/test_sec_13f_list.py`.

**Cambio:** `parse_line` con `len(line) < POS_STATUS[1]` (70 chars) devuelve
un dict con `status=None` en lugar de rellenar con espacios y clasificar como
`ACTIVE`.

**Evidencia:** 3 tests nuevos. Antes: linea corta -> `status=ACTIVE` (falso
positivo de elegibilidad 13F). Despues: `status=None` -> `_resolve_one_status`
la trata como `CONFLICT`.

### 2.2. P32 — contador de descartes silenciosos

**Commit:** 714457e.
**Ficheros:** `src/institutional_accumulation/aggregation/delta_shares.py`,
`tests/test_sec_13f_delta_shares.py`.

**Cambio:** `_filter_canonical` devuelve `(df, stats)`. El dict `stats`
cuantifica: `n_rows_input`, `n_after_sh_putcall_null`,
`n_dropped_not_sh_or_putcall`, `n_dropped_invalid_sshprnamt`, `n_rows_output`.
`compute_reported_position_units` acepta kwarg `return_stats=False` (por
defecto: mantiene contrato de DataFrame).

**Evidencia:** 3 tests nuevos. La politica de drop NO cambia; solo se
cuantifica (drop != silence).

### 2.3. P14-BIS — fillna condicionado al _merge del JOIN

**Commit:** eb687c4.
**Ficheros:** `src/institutional_accumulation/aggregation/delta_shares.py`,
`tests/test_sec_13f_delta_shares.py`.

**Cambio:** el `fillna(0.0)` global en `compute_delta_shares` se sustituye por
asignacion condicionada: `mask_left_only -> previous=0` (NEW);
`mask_right_only -> current=0` (EXIT). Un NaN en una fila `BOTH` dispara
`ValueError("corrupcion: N filas BOTH con NaN en sshprnamt")` en lugar de
imputarse silenciosamente.

**Evidencia:** 1 test nuevo (`test_delta_shares_both_con_nan_aborta`). La
semantica NEW/EXIT es identica; la validacion es mas estricta.

### 2.4. P31 — nipc_total_available + docstring _sum_delta

**Commit:** e31d82e.
**Ficheros:** `src/institutional_accumulation/aggregation/nipc.py`,
`tests/test_sec_13f_nipc.py`.

**Cambio:** `compute_nipc` anade el campo `nipc_total_available` (bool) en
ambas ramas. Docstring de `_sum_delta` corregido (devolvia siempre `0.0`,
nunca `None` como prometia). `nipc_total` sigue siendo `0.0` por
compatibilidad.

**Evidencia:** 2 tests nuevos (`available=True` con datos, `available=False`
sin datos). El campo se propaga a `compute_nipc_and_coverage` por
`dict(nipc)`.

### 2.5. P18/P26 — project_root por env var + fallback derivado

**Commit:** 54478c3.
**Ficheros:** `src/institutional_accumulation/sec_13f/manifest.py`,
`tests/test_sec_13f_manifest.py`.

**Cambio:** `DEFAULT_PROJECT_ROOT` deja de ser literal `Path(r"D:\\Macro_Sectorial")`.
Nueva resolucion: env var `IAE_PROJECT_ROOT` si esta definida; en su defecto,
`Path(__file__).resolve().parents[3]`.

**Evidencia:** 2 tests nuevos. El default apunta a la raiz del repo en
cualquier maquina; el codigo fuente no contiene ningun literal `D:\\Macro`.

### 2.6. P70 — guarda explicita invariante HR_PLUS_RESTATEMENT

**Commit:** 1575d17.
**Ficheros:** `src/institutional_accumulation/sec_13f/identity/amendments.py`,
`tests/test_sec_13f_amendments.py`.

**Cambio:** en la rama `if base == STRATEGY_HR_PLUS_RESTATEMENT:` de
`classify_strategy`, se anade:

    if len(filings_group) != 2:
        raise AssertionError(
            "invariante rota: HR_PLUS_RESTATEMENT con "
            f"{len(filings_group)} filings"
        )

NO se cambia `iloc[1]` por `iloc[-1]`. NO se modifica la logica de
clasificacion. La guarda convierte una precondicion implicita del
clasificador en contrato verificable.

**Evidencia:** 1 test nuevo
(`test_classify_strategy_guarda_invariante_hr_plus_restatement`) que
monkeypatchea `_classify_strategy_from_types` para simular un futuro cambio
que rompa la invariante; el test verifica que la guarda dispara.
---

## 3. Lo que NO ha cambiado

- **Contratos semanticos P38/P60/P61.** Los 6 fixes son ortogonales a las
  decisiones que F2.4 debe resolver.
- **Motor NIPC (nucleo).** `delta_shares.py::compute_delta_shares` cambio
  internamente para P14-BIS, pero la semantica NEW/EXIT/BOTH y el
  `MATCH_KEY` son identicos.
- **THRESHOLD_1 / THRESHOLD_2.** UNDEFINED.
- **Gate-NIPC.2.** BLOQUEADO.
- **Gate-NIPC.3.** NO AUTORIZADO.
- **OpenFIGI masivo.** NO AUTORIZADO.
- **Policy v1.3 aplicacion.** NO AUTORIZADA.
- **Informe F2.4 original.** Inmutable (hash 0763f16b...). Sigue siendo el
  documento de referencia para las decisiones D1-D4.

---

## 4. Cadena documental vigente

    NIPC_INFORME_ENTREGA_F24.md (0763f16b...)         [decision F2.4]
       |
       +-- cita -> NIPC_CONTRATOS_SEMANTICOS_v1.md (c02d202f...)
       |              |
       |              +-- cita -> NIPC_P70_DICTAMEN.md (e8d4e4e8...)
       |                            |
       |                            +-- cita -> _probe_p70.py (d1742a76...)
       |                                          |
       |                                          +-- produce -> result.json (809c51af...)
       |                                          +-- produce -> summary.txt (9e8c42d6...)
       |
       +-- cita -> NIPC_COVERAGE_POLICY_V13_PROPUESTA.md (933fcb2e...)
       |
       +-- cita -> INSTITUTIONAL_ACCUMULATION_REVISION_ESTRUCTURAL_2026-09-20.md (318f02e8...)

    NIPC_INFORME_ESTADO_POST_FIXES_F24.md (este documento)
       |
       +-- documenta -> 6 fixes mecanicos (commits 0f1e5a4..1575d17)
       +-- complementa -> NIPC_INFORME_ENTREGA_F24.md

---

## 5. Preguntas al auditor

### 5.1. Decisiones heredadas (D1-D4) — recordatorio

Sin cambios respecto del informe F2.4. Se reproducen para completitud:

- **D1.** Adopcion contractual del contrato semantico v1 (P60, P61, P38).
- **D2.** Aprobacion de NIPC_COVERAGE_POLICY_V13_PROPUESTA.md como base
  normativa.
- **D3.** Confirmacion del cierre de P70 como GO CONDICIONADO.
- **D4.** Autorizacion del paso 4 (tests especificos).

### 5.2. Preguntas nuevas derivadas del estado post-fixes

**Q1. Validacion de los 6 fixes mecanicos.**

Los 6 fixes fueron autorizados individualmente por el dictamen formal.
?Considera el auditor que su sola aplicacion y verificacion por tests
(946 passed) es suficiente, o desea revisar alguno de los fixes antes de
cerrar F2.4?

**Q2. Reclasificacion de P70 tras aplicar la guarda.**

La guarda de P70 estaba clasificada como "fix propuesto, a aplicar en fase 5".
Ahora esta aplicada. ?Reclasifica el auditor P70 de **GO CONDICIONADO** a
**GO definitivo**? ?O mantiene la clasificacion hasta verificar la
implementacion en un run real?

**Q3. Documento de referencia para F2.4.**

?El informe F2.4 original (0763f16b...) sigue siendo el documento de
referencia, o este informe post-fixes lo sustituye como input principal para
el dictamen?

**Q4. Naturaleza de los tests del paso 4.**

Los tests especificos de P38/P60/P61 previsiblemente fallaran contra el codigo
actual (que aun no implementa los contratos). ?El auditor quiere:

  (a) tests rojos que documentan el comportamiento actual como evidencia de
      la desviacion, para que pasen a verde cuando se apliquen los fixes, o
  (b) tests verdes escritos contra el contrato futuro, que fallaran hasta
      que se apliquen los fixes, o
  (c) ambos por separado (tests "before" y "after")?

**Q5. (Retirada como decision F2.4; trasladada a fase de implementacion).**

La ubicacion del modulo `temporal_validity.py` (contrato v1 seccion 8.2)
es una decision de implementacion, no de contrato. No forma parte de F2.4.
Se resolvera durante el paso 4 (tests especificos) o el paso 5 (fixes) de
la secuencia autorizada.

**Q6. Tratamiento del baseline historico en P38.**

La seccion 3.5 del contrato declara el baseline historico como
**HISTORICO / NO APTO PARA THRESHOLD_2**. ?Debe anadirse una nota explicita
de "no apto" a los documentos del baseline existentes
(NIPC_GATE0_COVERAGE_BASELINE_INFORME.md, OPENFIGI_TOP2000_INFORME.md), o
basta con la mencion en v1.3 y el contrato v1?

**Q7. Materializacion de la declaracion explicita de `identity_type` (P60).**

El contrato v1 seccion 1.6 exige que cada fuente **declare** el tipo de
identidad que aporta. La implementacion puede materializarse como:

  (a) un campo nuevo en cada CSV (`identity_type` por fila),
  (b) una tabla separada `data/mappings/identity_sources.csv`, o
  (c) otra opcion equivalente.

Nota: un mapping fijo en codigo (`etf_holdings.csv -> TICKER`) NO es
equivalente a una declaracion por la fuente. El programa no debe asignar
el tipo; la fuente debe declararlo. Esta opcion queda excluida por
contradiccion con el principio NO INFERIR IDENTIDAD de P60.

**Q8. Extension de la guarda P70 a `HR_CHAIN_RESTATEMENT`.**

La guarda actual aplica solo a `HR_PLUS_RESTATEMENT` (`len == 2`). La rama
`HR_CHAIN_RESTATEMENT` acepta `len >= 2`. ?Quiere el auditor una guarda
equivalente (`len > 2` cuando se asigna esa estrategia) o esa rama no tiene
invariante de cardinalidad que preservar?

### 5.3. Preguntas opcionales (no bloqueantes)

**Q9. Regeneracion de evidencia P70.**

El probe P70 se ejecuto antes de aplicar la guarda. ?Quiere el auditor que se
regenere la evidencia (mismo probe, mismos parquets) para confirmar que los
resultados no cambian con la guarda aplicada, o la ejecucion previa es
suficiente?

**Q10. Push a origin/main de los 6 fixes.**

Los 89 commits locales permanecen sin push (regla local-first IAE: no push
hasta cierre de Gate-NIPC.3). ?Confirma el auditor que la regla sigue
aplicando, o autoriza un push parcial (por ejemplo, solo los 6 fixes
mecanicos, sin los documentos del ciclo F2.4) para preservar el trabajo
contra perdida local?

---

## 5.bis. Notas de trazabilidad

### 5.bis.1. Discrepancia de hash del informe de revision estructural

La primera version del informe de revision estructural commiteada en
f107b27 tenia un doble problema de formato: (a) los caracteres especiales
de Markdown estaban escapados con backslash (`\#`, `\*\*`, `\_`); (b) las
lineas estaban en CRLF en lugar de LF. El fichero fue corregido en 9a0f66f
(fix escapes+EOL). Como consecuencia, el hash cambio:

    hash pre-fix  (f107b27):  d5385b1f0d9babee885fd08e9573051c67e318fd0734792d85e640be1a89a591
    hash post-fix (9a0f66f):  318f02e82ef179e28f9938c32491e1a827942bdb2dd53d5aa0d4aad8032b137a

El hash vigente en HEAD es 318f02e8.... La referencia a d5385b1f... en
documentos previos al 9a0f66f es historica.

### 5.bis.2. Etiqueta "max_group_len" en el probe P70

El probe P70 (probe_p70_summary.txt) etiqueta max_group_len como "global,
todos los grupos". Tecnicamente el probe solo actualiza max_len en 4 de
las 8 ramas de estrategia (HR_PLUS_RESTATEMENT, HR_CHAIN_RESTATEMENT,
HR_COMPOSITE, HR_PLUS_NEW_HOLDINGS). Las ramas SINGLE_HR, SINGLE_NOTICE,
NOTICE_AMENDED, REVIEW_REQUIRED no contribuyen a max_len. La etiqueta
"global" es por tanto imprecisa. No afecta a la conclusion del probe
(que se apoya en el argumento estructural), pero se documenta para evitar
lecturas erroneas.

### 5.bis.3. Test P70: implementado distinto del propuesto

El dictamen P70 (seccion 6) proponia como test verificar que un grupo con
len > 2 y primer filing 13F-HR clasifica como HR_CHAIN_RESTATEMENT, no
como HR_PLUS_RESTATEMENT.

El test finalmente implementado (commit 1575d17) es distinto: monkeypatcha
_classify_strategy_from_types para forzar que devuelva HR_PLUS_RESTATEMENT
y verifica que la guarda dispara AssertionError. Los dos tests son validos
pero verifican cosas distintas:

    test propuesto  -> verifica la rama real del clasificador
    test aplicado   -> verifica la guarda contractual introducida por P70

Ambos podrian coexistir. El implementado refuerza directamente la guarda,
que era el objetivo del fix.

---

## 6. Cierre

**Los 6 fixes mecanicos autorizados estan aplicados y verificados.**
**El informe F2.4 original sigue siendo el documento de referencia.**
**Los contratos semanticos P38/P60/P61 siguen pendientes de F2.4.**

Las 8 preguntas de la seccion 5.2 no bloquean F2.4: son preparatorias para el
paso 4 o para aclarar detalles del contrato antes de implementarlo. Las 2
preguntas opcionales de la seccion 5.3 son de proceso, no de semantica.

**Ningun cambio adicional se aplicara al modulo hasta recibir F2.4.**

---

Fin del informe de estado post-fixes.
Version 1.0 (2026-09-20). HEAD 1575d17.
Complementa NIPC_INFORME_ENTREGA_F24.md (0763f16b...).