# 47 - Estado SOW tras v5 grounded

Fecha: 2026-10-05
Rama: sow-v4-validation (HEAD fd76f9a)
Referencia: 46_dictamen_sow_v5_grounded.md

## Contexto

Fase confirmatoria del detector SOW. v4.1 implementado en `47140d0`.
v5 (outer grounded) implementado en `80a0e08` como modo alternativo
(`--outer-mode grounded`) del mismo script.

Ejecucion v5: `py scripts\sow_v4\run_validation.py --outer-mode grounded`.
Resultado en `outputs/audit/sow_v4/validation_grounded.json` (no versionado,
gitignore de `outputs/audit`).

## Resultado

- `decision.estado = MUESTRA INSUFICIENTE`
- `decision.motivo = N_eval=2`
- `N_eval = 2/8` (minimo protocolo 6)

Folds evaluables: 1 (2017) y 2 (2018).
Folds con fallo de fit grounded: 3, 4, 5, 6, 7.

Causa inmediata: `fit_primary(test_eps)` no converge o da `LinAlgError`
cuando `R_stress` es constante (folds 4, 5: n_normal=0), `n_normal` es
pequeno (folds 3, 7: 8 y 75) o hay colinealidad entre D1/D2/D3 (fold 6).

## Diagnostico del auditor

El estimando grounded ajusta el modelo sobre OUTER_TEST y evalua sobre el
mismo OUTER_TEST. No es validacion out-of-sample. Reutiliza los datos de
evaluacion para ajuste, produce optimismo por reutilizacion y ademas tiene
problemas de identificabilidad cuando la ventana es homogenea.

`MUESTRA INSUFICIENTE (N_eval=2)` se acepta como diagnostico del estimando
v5 grounded, **no** como conclusion cientifica sobre SOW.

Dictamen: v5 grounded NO APROBADO como validacion confirmatoria.
v6 requerido.

## Hallazgos registrados

- **S-01.** `manifest.protocol_version` hardcodeado a `"v4.1"` en
  `manifest.py`. Trazabilidad no fiable por ese campo. Corregir antes
  de siguiente ejecucion.
- **S-02.** Placebos en paradigma predictivo (`fit_primary(train_eps)`),
  bootstrap primario en paradigma grounded. `p_rand` y IC primario
  pertenecen a objetos estadisticos distintos. Subsumido por S-03 en v5;
  debe resolverse en v6 (placebos con el mismo estimador final que el real).
- **S-03.** El estimando grounded no es estimable en 5/7 folds por
  inviabilidad numerica. Ademas, cuando es estimable (fold 1), anchura
  CI95 ~ 0.44 y `LowerCI95(stress) = -0.10`, incompatible con
  `LowerCI95 > DELTA_MIN = 0.05`.

## Direccion v6 (resumen del dictamen 46)

- Candidata seleccionada en INNER, ciega a OUTER_TEST.
- Modelos nuisance `m(Y|SOW,X,R)` y `e(SOW|X,R)` estimados exclusivamente
  en OUTER TRAIN.
- OUTER TEST aporta `Y` observado y `SOW` observado.
- Estimando primario: contraste de riesgo ajustado AIPW, estratificado
  por stress/normal.
- `delta_min = 5 pp` se mantiene.
- Soporte de propensity preespecificado: `eps <= e(X,R) <= 1-eps`.
  Fuera de soporte -> fold/regimen no estimable.
- Bootstrap MBB (B20/B40) recalcula el estimador AIPW en cada replica.
  No reajusta nuisance.
- Placebos A y B con el mismo estimador final que el real.
- Secundario: `Delta Brier` y `Delta LogLoss` entre modelo baseline
  `R + X0` y `R + X0 + SOW`, ambos ajustados en OUTER TRAIN, evaluados
  contra Y observado en OUTER TEST.
- Regla `N_eval` desglosada: `N_eval_stress`, `N_eval_normal`,
  `N_eval_total` por separado.

SOW v6 no se implementa hasta que exista spec firmada por el auditor.
No hay `SOW_PROTOCOL_V6.md` en disco.

## Estado de la rama

- `sow-v4-validation` HEAD `fd76f9a`. No mergeada a `main`.
- `main` sigue en `8c811c1`.
- La rama se conserva como registro historico de la fase exploratoria
  v4.1 + v5. No se borra.
- No se mergea a `main` sin dictamen favorable sobre v6 y su validacion.

## Pendientes

- `44_dictamen_sow_v4_1.md` y `45_dictamen_sow_v5.md`: dictamenes del
  auditor sobre v4.1 y v5. Aun en chat, no en disco. Pendiente materializar.
- S-01 y S-02: corregir antes de la siguiente ejecucion del pipeline.
- `SOW_PROTOCOL_V6.md`: redactar con el auditor antes de implementar v6.
- No tocar `scripts/sow_v4/` hasta que v6 este firmado.

## Lo que NO se ha hecho

- No se ha modificado `scripts/sow_v4/` tras el run v5.
- No se han recalculado umbrales.
- No se ha relanzado el run.
- No se ha mergeado `sow-v4-validation` a `main`.
- No se ha creado `validation_predictive.json`.
