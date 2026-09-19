# DICTAMEN P70 - Cierre de la incidencia de classify_strategy

**Objeto:** contrato de clasificacion de amendments SEC 13F (HR_PLUS_RESTATEMENT).
**HEAD al redactar:** 77d4785.
**Alcance:** src/institutional_accumulation/sec_13f/identity/amendments.py::classify_strategy.
**Evidencia:** docs/auditoria/evidence/nipc_p70_probe/ (probe reproducible Q4 2025 + Q1 2026).
**Estado:** GO CONDICIONADO (caso B del arbol de decision del dictamen del auditor).
**Autor:** Ingeniero Supervisor (revision interna).
**Fecha:** 2026-09-20.

---

## Resumen ejecutivo

El informe de revision estructural (2026-09-20) detecto una asimetria en
classify_strategy:

- Rama HR_PLUS_RESTATEMENT -> evalua filings_group.iloc[1].
- Rama HR_CHAIN_RESTATEMENT -> evalua filings_group.iloc[-1].

El auditor clasifico la incidencia como P70 = ABIERTO / VALIDACION DE
INVARIANTE, y autorizo un probe empirico con dos objetivos:

1. Contar grupos HR_PLUS_RESTATEMENT con len > 2 en Q4 2025 y Q1 2026.
2. Contar casos donde iloc[1] != iloc[-1] en AMENDMENTTYPE.

Resultado: 195 grupos auditados (100 en Q1 2026, 95 en Q4 2025). Cero
casos len > 2. Cero casos iloc[1] != iloc[-1]. Cero casos con filas
intermedias y mismo AMENDMENTTYPE.

Adicionalmente, se demuestra que la invariante no es accidental: el
propio clasificador garantiza len(HR_PLUS_RESTATEMENT) == 2 por
construccion (_classify_strategy_from_types).

Dictamen: iloc[1] == iloc[-1] es trivialmente cierto cuando len == 2.
La asimetria detectada no puede causar un fallo operativo mientras la
invariante se cumpla. P70 pasa a GO CONDICIONADO con el fix propuesto
(guarda explicita, sin cambio de logica).
---

## 1. Contexto

Durante la revision estructural del modulo IAE (informe REVISION_ESTRUCTURAL_2026-09-20),
se identifico la siguiente asimetria en amendments.py::classify_strategy:

    if base == STRATEGY_HR_PLUS_RESTATEMENT:
        at = _norm_str(filings_group.iloc[1].get("AMENDMENTTYPE"))
        if at == "NEW HOLDINGS":
            return STRATEGY_HR_PLUS_NEW_HOLDINGS
        return STRATEGY_HR_PLUS_RESTATEMENT

    if base == STRATEGY_HR_CHAIN_RESTATEMENT:
        last = _norm_str(filings_group.iloc[-1].get("AMENDMENTTYPE"))
        if last == "NEW HOLDINGS":
            return STRATEGY_HR_COMPOSITE
        return STRATEGY_HR_CHAIN_RESTATEMENT

Una rama mira el segundo filing; la otra, el ultimo. El informe
la clasifico como P70 (MEDIA) por simetria de intencion, sin poder
determinar si era un defecto o una consecuencia de la invariante.

El dictamen formal del auditor (2026-09-20, seccion 5) resolvio que la
mera diferencia de indices no demuestra un bug. Autorizo cerrar la
invariante empiricamente antes de tocar el codigo.

---

## 2. Metodologia del probe

Diseno autorizado por el auditor:

- Alcance temporal: Q4 2025 + Q1 2026 (no solo Q1). Justificacion: la
  funcion es comun a ambos periodos; el objetivo es verificar una
  invariante del contrato de amendments, no la ausencia accidental del
  caso en un trimestre.
- Modo: solo diagnostico. Sin modificar classify_strategy. Sin logica
  paralela al clasificador.
- Reutilizacion: el probe importa y llama directamente a
  filter_by_period, order_filings y classify_strategy de produccion.
  Los grupos que analiza son exactamente los mismos que recibe el
  clasificador en un run productivo.
- Preguntas: cuatro, definidas literalmente por el auditor:
    1. Cuantos grupos llegan a HR_PLUS_RESTATEMENT.
    2. Cuantos tienen len(group) > 2.
    3. En cuantos normalize(iloc[1]["AMENDMENTTYPE"]) != normalize(iloc[-1]["AMENDMENTTYPE"]).
    4. Existen casos iloc[1] == iloc[-1] con filas intermedias.
- Evidencia: docs/auditoria/evidence/nipc_p70_probe/probe_p70_result.json
  y probe_p70_summary.txt, reproducibles con py _probe_p70.py.
---

## 3. Evidencia empirica

### 3.1. Metricas por trimestre

| Metrica                                       | Q1 2026 | Q4 2025 |
|---|---:|---:|
| n_groups_total                                | 10.648  | 10.524  |
| n_groups_hr_plus_restatement                  | 100     | 95      |
| n_groups_hr_chain_restatement                 | 2       | 2       |
| n_groups_hr_composite                         | 2       | 2       |
| n_groups_hr_plus_new_holdings                 | 19      | 9       |
| n_groups_len_gt_2                             | 0       | 0       |
| n_groups_second_vs_last_diff                  | 0       | 0       |
| n_groups_second_eq_last_with_intermediate     | 0       | 0       |
| max_group_len (global, todos los grupos)      | 3       | 4       |

### 3.2. Lectura

- 195 grupos HR_PLUS_RESTATEMENT auditados (100 + 95). Cero con
  len > 2. Cero con iloc[1] != iloc[-1]. Cero con filas intermedias
  y AMENDMENTTYPE igual en segundo y ultimo filing.
- El campo max_group_len = 4 en Q4 2025 no contradice la invariante:
  ese valor corresponde al maximo de todos los grupos, no al maximo
  de los HR_PLUS_RESTATEMENT. Los grupos de tamano > 2 se clasifican
  en HR_CHAIN_RESTATEMENT o HR_COMPOSITE, por construccion del
  clasificador.
- n_examples_saved = 0 en ambos trimestres: el probe no encontro
  ningun caso que cumpliese los criterios de "ejemplo problematico"
  (len > 2 o iloc[1] != iloc[-1]).
---

## 4. Confirmacion estructural

La invariante len(HR_PLUS_RESTATEMENT) == 2 no es una observacion
accidental del dataset. Esta garantizada por el propio clasificador
en _classify_strategy_from_types:

    if types[0] == "13F-HR":
        rest = types[1:]
        if all(t == "13F-HR/A" for t in rest):
            if len(rest) == 1:
                return STRATEGY_HR_PLUS_RESTATEMENT   # len == 2 garantizado
            return STRATEGY_HR_CHAIN_RESTATEMENT      # len > 2 garantizado

Consecuencias:

1. HR_PLUS_RESTATEMENT solo se asigna cuando hay exactamente dos
   filings (uno original 13F-HR + uno 13F-HR/A).
2. Los grupos con mas de dos filings van a HR_CHAIN_RESTATEMENT por
   construccion, independientemente del contenido del dataset.
3. Dado len == 2, la expresion iloc[1] == iloc[-1] es trivialmente
   cierta. La asimetria entre iloc[1] (rama HR_PLUS_RESTATEMENT) e
   iloc[-1] (rama HR_CHAIN_RESTATEMENT) no puede divergir en el
   universo de entradas que recibe la primera rama.

### 4.1. Implicacion

El fix no consiste en unificar los indices (no hay divergencia que
resolver). El fix consiste en convertir la invariante implicita en
guarda explicita, de modo que si un cambio futuro del clasificador
(o del dataset) rompe la precondicion, el fallo sea ruidoso y no
silencioso.
---

## 5. Dictamen

P70 = GO CONDICIONADO (caso B del arbol de decision del dictamen
del auditor).

Justificacion:

- La evidencia empirica (Q4 2025 + Q1 2026) satisface los tres criterios
  del caso B: n_groups_len_gt_2 == 0, n_groups_second_vs_last_diff == 0,
  n_groups_second_eq_last_with_intermediate == 0.
- La invariante esta respaldada adicionalmente por el analisis
  estructural del clasificador: HR_PLUS_RESTATEMENT -> len == 2 es
  una propiedad del codigo, no del dataset.
- La asimetria iloc[1] vs iloc[-1] no puede causar un defecto operativo
  bajo la invariante vigente.

El cambio propuesto no modifica la logica de clasificacion. Solo
anade una guarda explicita. Un commit reversible, coherente con el
principio del proyecto "un fix mecanico = un commit reversible".
---

## 6. Fix propuesto

Ubicacion: src/institutional_accumulation/sec_13f/identity/amendments.py,
funcion classify_strategy.

Cambio:

    if base == STRATEGY_HR_PLUS_RESTATEMENT:
        if len(filings_group) != 2:
            raise AssertionError(
                "invariante rota: HR_PLUS_RESTATEMENT con "
                f"{len(filings_group)} filings"
            )
        if "AMENDMENTTYPE" in filings_group.columns:
            at = _norm_str(filings_group.iloc[1].get("AMENDMENTTYPE"))
            if at == "NEW HOLDINGS":
                return STRATEGY_HR_PLUS_NEW_HOLDINGS
        return STRATEGY_HR_PLUS_RESTATEMENT

Justificacion:

- No se cambia iloc[1] por iloc[-1]. Bajo la invariante, son
  equivalentes. Introducir el cambio no aportaria valor y romperia la
  trazabilidad con el dictamen FA-2.4 (que especifica iloc[1] para
  este caso).
- La guarda convierte una precondicion implicita del clasificador en
  contrato verificable. Si el clasificador se modifica en el futuro
  (por ejemplo, admitiendo cadenas HR + HR/A + HR/A en
  HR_PLUS_RESTATEMENT), la guarda fallara inmediatamente y forzara
  una revision contractual.

Test asociado: en tests/test_sec_13f_amendments.py, anadir un
caso que verifique que un grupo con len > 2 y primer filing 13F-HR
clasifica como HR_CHAIN_RESTATEMENT, no como HR_PLUS_RESTATEMENT.
El test no depende del dataset: usa un DataFrame sintetico con 3 filings.

Estado: fix autorizado pero no aplicado. Se ejecutara en fase 5
de la secuencia del dictamen del auditor, tras cerrar P38 + P60 + P61.
---

## 7. Limites de la evidencia

Se declaran explicitamente:

1. Cobertura temporal limitada. Los trimestres auditados son Q4 2025 y
   Q1 2026 (dos periodos). No se ha verificado la invariante sobre
   trimestres anteriores (Q1 2025, Q2 2025, Q3 2025) ni posteriores.
   Sin embargo, la garantia estructural del clasificador hace que el
   argumento no dependa del periodo: la invariante se cumple por codigo,
   no por coincidencia del dataset.

2. Dependencia del dataset SEC. Los numeros provienen del ZIP oficial
   01dec2025-28feb2026_form13f.zip (Q4 2025) y
   01mar2026-31may2026_form13f.zip (Q1 2026). Cualquier re-edicion del
   dataset por parte de SEC podria alterar los conteos brutos, pero no
   la conclusion estructural.

3. Ausencia de verificacion sobre HR_PLUS_NEW_HOLDINGS. El caso
   analogo (grupos HR_PLUS_NEW_HOLDINGS, que tambien surgen de
   base == HR_PLUS_RESTATEMENT) comparte la misma rama y la misma
   guarda. El probe no lo trata por separado.

4. Sin OpenFIGI ni identity. El probe no usa identity, no usa
   OpenFIGI, no toca NIPC. Solo lee SUBMISSION.tsv y COVERPAGE.tsv
   filtrados por periodo.

5. HEAD no commiteado. El dictamen, los scripts temporales
   (_probe_p70.py, _ingest_local_p70.py, _ingest_q4_2025.py) y la
   evidencia generada no estan commiteados al cierre de esta redaccion.
   HEAD se mantiene en 77d4785.
---

## 8. Anexos

### 8.1. Ficheros de evidencia

    docs/auditoria/evidence/nipc_p70_probe/
      probe_p70_result.json    (resultado estructurado)
      probe_p70_summary.txt    (resumen legible)

SHA-256:

    probe_p70_result.json  809c51af47ae154c6c01bba1cf3983a6c0143fd3b0711958fa8b9e6e18bae96f
    probe_p70_summary.txt  9e8c42d6a742d773be7e531782a7d8514a5c00aa64efc5080d81a718d6384fe5

### 8.2. Reproduccion

    # 1. Ingesta Q1 2026 (requiere D:\13f_q1_2026\extracted)
    py _ingest_local_p70.py

    # 2. Ingesta Q4 2025 (descarga SEC)
    py _ingest_q4_2025.py

    # 3. Ejecucion del probe
    py _probe_p70.py

    # 4. Inspeccion
    Get-Content docs\auditoria\evidence\nipc_p70_probe\probe_p70_summary.txt

### 8.3. Trazabilidad al dictamen del auditor

Este dictamen cierra la seccion 5 del dictamen formal del auditor
(2026-09-20), que clasifico P70 como ABIERTO / VALIDACION DE INVARIANTE
y autorizo el probe con el arbol de decision A/B/C.

Resultado aplicado: caso B.

---

## 9. Estado de bloqueos NIPC tras P70

    P38  paired_weighted_share_coverage  BLOQUEADO (dictamen pendiente)
    P60  canonical_security / prefijos   BLOQUEADO (dictamen pendiente)
    P61  etf_holdings sin vigencia       BLOQUEADO (dictamen pendiente)
    P70  classify_strategy               CERRADO / GO CONDICIONADO

    THRESHOLD_1                          UNDEFINED
    THRESHOLD_2                          UNDEFINED
    Gate-NIPC.2                          BLOQUEADO
    Gate-NIPC.3                          NO AUTORIZADO
    F2.4                                 NO AUTORIZADA

---

Fin del dictamen P70.
Version 1.0 (2026-09-20). HEAD 77d4785.