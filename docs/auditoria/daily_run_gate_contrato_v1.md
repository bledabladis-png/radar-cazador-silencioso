
# Contrato v1: Completion Receipt para el gate del daily_run

**Expediente:** `docs/auditoria/daily_run_gate_discrepancia.md`
**Dictamen externo:** `docs/auditoria/daily_run_gate_dictamen.md` (recomendacion E, invariantes I1-I8)
**Fecha:** 2026-10-01
**Estado:** APROBADO 2026-10-01. Implementacion autorizada.

---

## 1. ALCANCE

### 1.1. Cambia

- `scripts/pipeline_gate.py::evaluate`: la fuente de `CURRENT` pasa a ser
  `find_completion_receipt(target_session)`, no `_manifest_satisfies`.
- `.github/workflows/daily_run.yml`:
  - Job `gate`: permiso `actions: read`; variable de entorno con
    repo owner/name si es necesaria para la API.
  - Job `run-system`: nuevo step `upload-artifact` al final (post
    `git push`), que publica el `completion receipt`.

### 1.2. NO cambia

- `_manifest_satisfies`: se conserva intacta, como contrato de
  integridad del artefacto. No interviene en la decision `CURRENT`.
- `scripts/guard_coverage.py`: sin tocar.
- `_probe_panel`, `_probe_panel_once`, `resolve_slot_flags`,
  `resolve_target_session`, `_write_github_output`: sin tocar.
- `08_AUDITORIA_SECTOR_REGIME.md`, `07_RUNBOOK.md`, `05_BITACORA.md`:
  sin tocar en esta fase.
- No hay migracion de datos historicos. No hay cambio en los
  `outputs/*.csv` ni en `data/*.parquet` ni en sus manifests.

### 1.3. Fuera de alcance

- Reducir el numero de slots (opcion d): rechazada por el auditor.
- Tocar el `concurrency.group` del daily_run: el grupo `daily-run` con
  `queue: max` se mantiene. El auditor lo declara ortogonal (seccion 16
  del dictamen).
- Refactor de `pipeline_gate.py` mas alla del cambio de fuente de
  `CURRENT`.

---

## 2. CONTRATO DE `CURRENT`

`evaluate(target_session)` cambia la primera rama:

```
def evaluate(target_session):
    receipt = find_completion_receipt(target_session)
    if receipt is not None:
        return {
            "state": "CURRENT",
            "should_run": False,
            "reason": "completion receipt for {0} (run {1})".format(
                target_session, receipt["run_id"]),
            "expected_session": target_session,
            "probe_coverage": None,
            "receipt": receipt,
        }
    probe_cov, probe_err = _probe_panel(target_session)
    ...
```

`find_completion_receipt` devuelve un dict con el receipt validado, o
`None`. Valida las dos capas descritas en la seccion 10 del dictamen:

- Capa 1: existe un artifact `completion-receipt-<target_session>` y su
  contenido cumple el schema §3.
- Capa 2: el workflow run que lo subio concluyo con
  `conclusion == "success"`.

Si Capa 1 pasa y Capa 2 falla: se trata como `None` (no falso-CURRENT).

Si la consulta a la API falla (red, token, 5xx): se trata como `None`
(invariante I6).

---

## 3. SCHEMA DEL RECEIPT

Fichero JSON, nombre del artifact: `completion-receipt-<target_session>`.
Un solo fichero dentro: `receipt.json`.

```
{
  "schema_version": 1,
  "status": "COMPLETED",
  "target_session": "2026-09-30",
  "workflow": "daily_run",
  "run_id": 36876380805,
  "run_attempt": 1,
  "completed_at": "2026-10-01T14:45:12Z",
  "commit_sha": "abc123...",
  "manifest_sha256": "def456...",
  "manifest_coverage_pct": 99.12,
  "guard_coverage": "OK",
  "validation_gate": "10/10",
  "pipeline_conclusion": "success"
}
```

Campos minimos para que el receipt sea valido:

- `schema_version` (int, == 1): permite evolucion futura.
- `status` (str, == "COMPLETED"): no se acepta otro estado hoy.
- `target_session` (str ISO date): debe coincidir EXACTAMENTE con el
  `target_session` que el gate esta evaluando (I7).
- `workflow` (str): debe ser "daily_run".
- `run_id` (int): identificador del run de GitHub.
- `completed_at` (ISO datetime UTC): trazabilidad.
- `manifest_sha256` (str): sha256 del parquet, informativo.
- `manifest_coverage_pct` (float): informativo.
- `validation_gate` (str): debe ser "10/10" para que el receipt sea
  valido como evidencia de pipeline OK.
- `pipeline_conclusion` (str): debe ser "success".

El gate rechaza el receipt si:

- `schema_version != 1`.
- `status != "COMPLETED"`.
- `target_session != target_session_actual`.
- `workflow != "daily_run"`.
- `validation_gate != "10/10"`.
- `pipeline_conclusion != "success"`.
- El run referenciado por `run_id` no tiene `conclusion == "success"`.

---

## 4. PERMISOS

- Job `gate`: anade `actions: read` a los permisos existentes
  (`contents: read`). Necesario para leer artifacts de runs anteriores
  con el GITHUB_TOKEN. No se requieren secrets adicionales.
- Job `run-system`: sin cambios. `contents: write` ya cubre el
  `upload-artifact`.
- Job `issue-manager`: sin cambios.

---

## 5. MOMENTO DE EMISION

Nuevo step en `run-system`, posicionado DESPUES de
`Commit and push hist/state` y antes del cierre del job:

```
      - name: Upload completion receipt
        if: success()
        uses: actions/upload-artifact@v4
        with:
          name: completion-receipt-${{ needs.gate.outputs.expected_session }}
          path: outputs/state/completion_receipt.json
          retention-days: 90
          if-no-files-found: error
```

`if: success()` implica que solo se ejecuta si TODOS los steps previos
de `run-system` terminaron con exito. En particular:

- `pytest tests/test_freshness.py` OK.
- `validate_history_quality` OK.
- `guard_coverage --threshold 0.95` OK.
- `Commit and push hist/state` OK.

Esto satisface la invariante I3 del dictamen.

`retention-days: 90`: alineado con el default de GitHub y con la
seccion 12 del dictamen (fail-safe, no falso-CURRENT si el artifact
expira).

El fichero `outputs/state/completion_receipt.json` se genera en un
step previo (por ejemplo tras `guard_coverage`, antes del commit),
para que su contenido refleje el estado real del pipeline. Se
**excluye** del commit (`git add` selectivo ya existente no lo toca).

`if-no-files-found: error`: si el fichero no existe, el step falla y
NO se sube artifact. Esto es fail-safe: sin receipt, el siguiente slot
hara `READY` y volvera a correr. Preferimos duplicar trabajo a declarar
CURRENT sin evidencia.

---

## 6. INVARIANTES I1-I8 COMO CRITERIOS DE ACEPTACION

Cada invariante lleva un test concreto. Los tests iran en
`tests/test_pipeline_gate_receipt.py` (nuevo).

### I1 — Nunca existe `CURRENT` por ausencia de parquet

Test: `test_i1_sin_receipt_sin_parquet_no_current`. Mock de la API
devuelve lista vacia de artifacts. `evaluate(target)` != CURRENT.

### I2 — Nunca existe `CURRENT` por un manifest aislado

Test: `test_i2_manifest_valido_sin_receipt_no_current`. Manifest con
sha256 valido en disco (o ausente, indiferente) + API sin artifacts.
`evaluate(target)` != CURRENT. Confirma que `_manifest_satisfies` ya
no participa en la decision de `CURRENT`.

### I3 — Receipt solo se genera tras ultimo check

Test: `test_i3_step_receipt_despues_de_guard`. Inspeccion estatica del
`daily_run.yml` (o ejecucion en un runner sintetico) verificando que
el step `Upload completion receipt` tiene `if: success()` y esta
despues de `guard_coverage` y `Commit and push`.

### I4 — Receipt inmutable e identifica ejecucion

Test: `test_i4_schema_receipt_campos_obligatorios`. El schema del
receipt (§3) es valido contra los campos obligatorios. Se valida
`run_id`, `commit_sha`, `manifest_sha256`, `completed_at`,
`schema_version`.

### I5 — Expiration es fail-open

Test: `test_i5_receipt_expirado_no_current`. Mock de API devuelve
`{"artifacts": []}` porque el receipt ha expirado. `evaluate` !=
CURRENT. El gate cae al probe.

### I6 — API failure != CURRENT

Test: `test_i6_api_error_no_current`. Mock de API lanza excepcion o
devuelve 5xx. `evaluate` != CURRENT. No se convierte el error en
CURRENT.

### I7 — Target session exacta

Test: `test_i7_receipt_de_otra_session_no_current`. Existe receipt
para `2026-09-29`, se evalua `2026-09-30`. `evaluate` != CURRENT.

### I8 — Receipt no sustituye manifest

Test: `test_i8_manifest_satisfies_sin_cambios`. `_manifest_satisfies`
sigue devolviendo `False` en CI por parquet ausente (comportamiento
actual, no modificado por este contrato).

---

## 7. PLAN DE VERIFICACION

### 7.1. Reproduccion del bug actual

`py docs\auditoria\daily_run_gate_repro.py` sigue dando `False`. El
script no cambia con este contrato (verifica `_manifest_satisfies`, que
no se toca).

### 7.2. Tests unitarios del nuevo contrato

`tests/test_pipeline_gate_receipt.py`: los 8 tests de §6. Todos verdes
antes del push.

### 7.3. Test local del gate con receipt simulado

Script `tests/manual_gate_with_receipt.py` (a decidir si versionado o
en `docs/auditoria/`): monkeypatch de la API para devolver un receipt
valido; `evaluate(target)` devuelve `CURRENT`. Permite verificar el
flujo end-to-end sin tocar GitHub.

### 7.4. E2E real

Una vez mergeado:

1. `workflow_dispatch` manual del `daily_run`.
2. Verificar que se ejecuta `Upload completion receipt` y el artifact
   `completion-receipt-2026-09-30` aparece en GitHub.
3. Esperar al slot siguiente (real o via dispatch). Verificar que el
   gate devuelve `CURRENT` y `run-system` se skipea.
4. Verificar que `issue-manager` no abre issue (no hay fallo).
5. Documentar en `07_RUNBOOK`.

### 7.5. Rollback

Si algo falla en produccion:

1. Revertir el commit que cambia `pipeline_gate.py` (vuelve a
   `_manifest_satisfies` como fuente de CURRENT). Comportamiento
   equivalente a la actualidad.
2. Revertir el step `Upload completion receipt` del yml. Limpieza.
3. Los artifacts ya subidos se pueden borrar con `gh api --method
   DELETE`.

El rollback es de un solo commit, reversible.

---

## 8. COSTE Y RIESGO

### 8.1. Coste de implementacion

- Cambio de `pipeline_gate.py`: ~80 lineas nuevas
  (`find_completion_receipt`, validador de schema, cliente API).
- Nuevo test: ~150 lineas.
- Cambio del yml: ~15 lineas.
- Documentacion: este fichero + ampliacion de `07_RUNBOOK`.

Estimacion: 1 sesion de implementacion + 1 sesion de E2E.

### 8.2. Coste operativo (con la correccion del repo publico)

- Minutos de runner: NO aplica (repo publico).
- Llamadas a la API por gate: ~2 (listar artifacts + verificar run).
  5 gates/dia = 10 llamadas/dia. Dentro del rate limit de la API de
  GitHub.
- Almacenamiento de artifacts: ~1 KB por receipt, con 90 dias de
  retencion. Despreciable.

### 8.3. Riesgo residual

- **Falso-CURRENT:** no hay, porque el receipt solo se genera tras
  `if: success()` con validacion `10/10` y `success`.
- **Falso-READY:** puede ocurrir si el receipt expira (90 dias). Es
  fail-safe. Duplica trabajo, no corrompe.
- **Bloqueo por rate limit de API:** improbable con 10 llamadas/dia.
  Si ocurre, I6 asegura que se trata como `None` (no falso-CURRENT).

---

## 9. OPCIONES DESCARTADAS (a)-(d)

Documentado para que el historial no vuelva a proponerlas:

- **(a) Cache de parquet+manifest:** funciona pero usa cache como
  almacenamiento de estado de negocio. GitHub recomienda cache solo
  para rendimiento, con eviction a los 7 dias de no acceso. El
  dictamen la clasifica "hotfix valido, no contrato definitivo".
- **(b) SHA best-effort en CI:** convierte `CURRENT` de "demostrado" a
  "confiado". No aprobada por el dictamen como solucion final.
- **(c) Marker commitado en la rama:** el marker depende del snapshot
  Git del workflow (GitHub fija el SHA al disparar el evento). Un slot
  puede no ver el marker que el anterior acaba de publicar. Buena idea,
  implementacion insuficiente.
- **(d) Reducir slots:** no corrige el defecto, solo su frecuencia.
  Requiere estudio historico de latencia de Yahoo que no esta hecho.

---

## 10. CRITERIOS DE ABANDONO

Si durante la implementacion:

- El `upload-artifact` de un receipt unitario supera los 30 segundos.
- La API de GitHub para listar artifacts por `target_session` no es
  fiable en repos publicos.
- Cualquier invariante I1-I8 no es verificable con un test limpio.

Entonces se detiene la implementacion y se reabre el expediente con
el auditor. No se implementa una version degradada silenciosamente.

---

## 11. FIRMA

**APROBADO 2026-10-01.**

Fundamento: dictamen externo de la misma fecha
(`docs/auditoria/daily_run_gate_dictamen.md`) recomienda (E) como
unica arquitectura aprobada. El contrato implementa (E) sin
desviaciones. Las 8 invariantes I1-I8 son transcripcion literal de
la seccion 15 del dictamen.

Criterios cumplidos:

1. Commit con mencion "APROBADO" (este).
2. I1-I8 verificadas contra dictamen: identicas.
3. Entrada en `05_BITACORA.md`: pendiente al cierre de la sesion de
   implementacion (se anade en el mismo commit o en el de cierre).

`pipeline_gate.py` y `daily_run.yml` se modifican en la sesion de
implementacion inmediata.
