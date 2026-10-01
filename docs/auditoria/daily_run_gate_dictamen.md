# Dictamen del auditor externo



**Expediente:** `docs/auditoria/daily_run_gate_discrepancia.md`

**Fecha:** 2026-10-01

**Veredicto:** diagnostico CONFIRMADO (defecto de diseno). Ninguna

de las opciones (a)-(d) aprobada como solucion final. Recomendacion:

(E) separar integridad del artefacto de idempotencia del workflow,

usando un Completion Receipt inmutable en GitHub Actions Artifacts.

`_manifest_satisfies` se conserva sin cambios.



**Invariantes propuestas:** I1-I8 (ver seccion 15 del dictamen).



---



He revisado el expediente como auditor externo de arquitectura, CI/CD e integridad de datos, y he contrastado además los puntos de comportamiento relevantes de GitHub Actions con su documentación actual.

# Dictamen del auditor

**La causa raíz está correctamente identificada.** El fallo no está en Yahoo ni en el cálculo de cobertura: está en el **límite de confianza entre el `manifest` y el filesystem del runner**.

En CI, `_manifest_satisfies()` exige:

1. manifest correcto;
2. `expected_session` correcta;
3. cobertura suficiente;
4. SHA declarado;
5. **parquet físico presente**;
6. SHA del parquet = SHA declarado.

Como el parquet está explícitamente fuera de Git, los puntos 5–6 son imposibles en el `gate`. Por tanto:

> **En el entorno CI actual, `CURRENT` no es simplemente improbable; es estructuralmente inalcanzable.**

El apéndice A demuestra correctamente la propiedad mecánica, aunque por sí solo no demostraría el estado real de GitHub; esa demostración queda completada por los cuatro runs observados y el código aportado.

---

# 1. El error conceptual que originó el problema

El cambio `2a640f2` resolvió un problema real:

> un `manifest` puede no corresponder al parquet local.

Por tanto, exigir SHA del parquet es correcto **para comprobar integridad del artefacto cuando ambos están presentes**.

El error fue usar esa comprobación como requisito para una función cuyo segundo entorno de ejecución es deliberadamente un runner donde el parquet no está presente.

Se mezclaron dos contratos diferentes:

### Contrato A — integridad del artefacto

```text
manifest ↔ parquet físico
```

Pregunta:

> «¿Este manifest describe exactamente este parquet?»

Aquí **SHA-256 sí tiene sentido** y debe seguir siendo fail-closed.

### Contrato B — idempotencia del workflow

```text
¿ya completamos target_session?
```

Pregunta:

> «¿Existe evidencia fiable de que un run anterior terminó correctamente para esta sesión?»

Aquí **no necesitas el parquet**.

Esta separación es, a mi juicio, la corrección arquitectónica fundamental.

---

# 2. Evaluación de las opciones §9

| Opción                                  | Valor técnico          | Problema principal                                         | Dictamen                        |
| --------------------------------------- | ---------------------- | ---------------------------------------------------------- | ------------------------------- |
| **A. Cache parquet + manifest**         | Bueno                  | Usa cache como almacenamiento de estado de negocio         | **Aceptable como parche**       |
| **B. SHA best-effort en CI**            | Medio                  | Introduce `CURRENT` sin verificar integridad física        | **No como solución definitiva** |
| **C. Marker commitado en rama**         | Bueno en concepto      | El marker depende del snapshot Git del workflow            | **No tal como está diseñada**   |
| **D. Reducir slots**                    | No corrige el contrato | Elimina redundancia, pero no resuelve idempotencia         | **No como fix**                 |
| **E. Completion receipt externo a Git** | Muy bueno              | Requiere consultar Actions/API y definir un contrato nuevo | **Recomendación**               |

La opción que aprobaría para producción es una variante de **E**.

---

# 3. Por qué NO aprobaría (b) como diseño final

La propuesta:

```python
if not p.exists():
    return True
```

convierte:

```text
manifest válido
+
parquet no verificable
=
CURRENT
```

Eso es una relajación significativa del contrato.

El problema no es solamente teórico. Un `manifest` puede ser:

* antiguo;
* incorrectamente generado;
* parcialmente publicado;
* alterado por un commit posterior;
* correspondiente a una ejecución distinta;
* sintácticamente correcto pero semánticamente incorrecto.

Con (b), CI no tiene forma de distinguir:

```text
manifest correcto y artefacto realmente producido
```

de:

```text
manifest correcto pero artefacto inexistente/no verificable
```

Por tanto, **sí introduce un riesgo real de falso-CURRENT**.

Ahora bien, hay una distinción importante:

### ¿Es riesgo “inaceptable”?

No necesariamente para un sistema donde un falso-CURRENT solamente provoca que se omita una ejecución redundante.

Pero el riesgo debe medirse contra la semántica del contrato.

Si `CURRENT` significa:

> «Tenemos evidencia de que `target_session` fue procesada correctamente»

entonces (b) **no proporciona esa evidencia**; simplemente confía en el manifest.

Yo lo clasificaría como:

**parche operativo aceptable, arquitectura no aprobada como estado final.**

---

# 4. (a) Cache: funciona, pero no usaría el parquet como mecanismo de estado

(a) sí consigue algo importante:

```text
run 1
 └─ genera parquet + manifest
 └─ guarda cache

run 2
 └─ restaura cache
 └─ SHA válido
 └─ CURRENT
```

Eso hace que el contrato original vuelva a funcionar.

Además, una pérdida de cache es **fallo seguro en términos de integridad**:

```text
cache miss
→ READY
→ rerun
```

No genera falso-CURRENT.

Eso es bueno.

Pero hay una objeción arquitectónica: **GitHub recomienda cache para reutilización y rendimiento, no como almacén de estado operativo**. Las caches tienen alcance por branch y mecanismos de eviction; actualmente GitHub indica que las entradas no accedidas durante más de 7 días pueden eliminarse y que la cache no debe tratarse como almacenamiento persistente. ([GitHub Docs][1])

Por tanto:

> **A es funcionalmente correcta, pero conceptualmente está utilizando una primitiva de performance como base de un contrato de control.**

Además, no veo necesidad de cachear ~10 MB de parquet exclusivamente para responder a:

> «¿ya terminó este target_session?»

La pregunta requiere un **receipt**, no el dataset.

---

# 5. El problema oculto de (c)

Aquí está el punto más importante que añadiría al expediente.

La idea de:

```text
outputs/state/daily_completed.json
```

es conceptualmente buena, pero **commitado en la rama no es una solución suficiente tal cual**.

GitHub documenta que un `schedule` utiliza el último commit del default branch **en el momento del evento programado**, y que cada workflow run tiene asociado un SHA/ref del evento. `actions/checkout` utiliza por defecto ese ref/SHA asociado al evento. ([GitHub Docs][2])

Esto genera una carrera:

```text
23:17  SLOT 1 EVENT
       └─ run A

03:17  SLOT 2 EVENT
       └─ GitHub fija SHA_B

04:00  run A termina
       └─ commit daily_completed.json
       └─ main = SHA_C

04:05  run B empieza
       └─ checkout puede seguir estando en SHA_B
       └─ SHA_B no contiene daily_completed.json
```

Resultado:

> **slot 2 puede no ver el marker que slot 1 acaba de publicar.**

Por eso no aprobaría (c) sin modificar una de estas dos cosas:

```text
checkout dinámico de origin/main
```

o

```text
lectura del estado fuera del snapshot Git del run
```

Y en cuanto haces eso, estamos prácticamente llegando a la opción E.

---

# 6. ¿Puede (c) producir otro “contrato que miente”?

**Sí.**

Y el riesgo es exactamente el patrón que señalas.

Imaginemos:

```json
{
  "target_session": "2026-09-30",
  "gate": "10/10",
  "manifest_sha": "abc..."
}
```

El nombre parece decir:

> «el sistema terminó correctamente».

Pero ¿qué demuestra exactamente?

Puede significar:

1. parquet generado;
2. parquet validado;
3. manifest generado;
4. tests pasados;
5. outputs generados;
6. commit realizado;

o solamente:

7. un script escribió el JSON.

Por eso el problema no se soluciona creando un marker. Hay que definir **una prueba de finalización**, no simplemente un fichero que diga `"completed": true`.

Ese es precisamente el patrón de contrato mentiroso que hay que evitar.

---

# 7. Arquitectura que recomiendo: Completion Receipt

Mi propuesta es introducir una nueva evidencia:

## `daily completion receipt`

No sería:

```text
estado de los datos
```

sino:

```text
evidencia de finalización de una ejecución
```

Y la almacenaría en **GitHub Actions Artifacts**, no en Git.

GitHub define los artifacts precisamente como almacenamiento de datos producidos por workflows que persisten después del job/run. Además, con `upload-artifact` v4 los artifacts son inmutables una vez cargados salvo eliminación/recreación, y están disponibles mediante la API. ([GitHub Docs][3])

Esto encaja mucho mejor con el problema.

---

# 8. Contrato propuesto

Por ejemplo:

```json
{
  "schema_version": 1,
  "status": "COMPLETED",
  "target_session": "2026-09-30",
  "workflow": "daily_run",
  "run_id": 36804481750,
  "completed_at": "2026-10-01T10:02:31Z",
  "manifest_sha256": "....",
  "manifest_coverage_pct": 99.12,
  "gate": "10/10",
  "commit_sha": "...."
}
```

Pero **el punto clave es el momento en que se genera**.

Debe generarse solamente después de:

```text
pipeline completo
      ↓
validaciones
      ↓
manifest correcto
      ↓
guard/gates correctos
      ↓
publicación/commit realizada
      ↓
workflow sigue SUCCESS
      ↓
UPLOAD completion receipt
```

Así:

> la existencia del receipt es evidencia de que la ejecución alcanzó el estado contractual de finalización.

---

# 9. Condición exacta para `CURRENT`

Yo cambiaría conceptualmente el gate a:

```python
completion = find_completion_receipt(target_session)

if completion is valid:
    return CURRENT
```

y dejaría `_manifest_satisfies()` como función independiente.

Es decir:

```text
                    ┌─────────────────────┐
                    │ target_session      │
                    └──────────┬──────────┘
                               │
                     ¿completion receipt?
                         /            \
                       YES             NO
                       │                │
                   CURRENT          probe Yahoo
                                      │
                             ┌────────┴────────┐
                           READY           NOT_READY
```

Mientras que:

```text
_manifest_satisfies()
```

sigue siendo responsable exclusivamente de:

```text
manifest ↔ parquet ↔ sha256
```

No de la idempotencia de CI.

Esto elimina la contaminación conceptual actual.

---

# 10. No confiaría solamente en el JSON

Para evitar el “contrato que miente”, el `gate` debería validar **dos capas**.

### Capa 1 — existencia del receipt

Debe existir artifact con:

```text
target_session = X
status = COMPLETED
schema_version = soportada
```

### Capa 2 — run que lo generó

El gate debería comprobar que ese receipt procede de un workflow run apropiado y exitoso.

La API de GitHub permite listar artifacts de workflow runs y requiere permiso `Actions: read` para repos privados mediante token con permisos adecuados. ([GitHub Docs][4])

Así evitamos:

```text
artifact existe
≠
run completó correctamente
```

y exigimos:

```text
artifact válido
+
target_session correcta
+
workflow correcto
+
run correcto
+
conclusion = success
```

---

# 11. El SHA no desaparece

Esto es importante.

**No recomiendo eliminar el SHA-256.**

Simplemente cambia de sitio dentro del contrato.

### Antes

```text
CURRENT
   ↓
manifest
   ↓
parquet local
   ↓
sha256
```

y eso es imposible en CI.

### Después

```text
CI idempotency
   ↓
completion receipt
   ↓
successful run
```

y, por separado:

```text
artifact integrity
   ↓
manifest
   ↓
parquet
   ↓
sha256
```

De esta forma se conservan las dos garantías sin obligar al gate a poseer el parquet.

---

# 12. ¿Y si el receipt desaparece?

Eso es una propiedad importante del sistema.

Los artifacts tienen retención configurable y, por defecto, GitHub indica 90 días para artifacts y logs, aunque puede modificarse. ([GitHub Docs][5])

Para vuestro caso eso significa que una ejecución antigua puede desaparecer del almacén de artifacts.

Pero eso **no genera falso-CURRENT**:

```text
receipt desaparecido
→ gate no encuentra evidencia
→ READY
→ rerun
```

El resultado es:

**duplicación de trabajo, no corrupción de estado.**

Eso es exactamente el comportamiento que queremos de un sistema fail-safe.

---

# 13. Evaluación formal de las cuatro opciones

### (a) Cache

**Segura contra falso-CURRENT si se valida SHA.**

Problema: cache es infraestructura de reutilización y tiene eviction. ([GitHub Docs][1])

La consideraría un **hotfix válido**, no el contrato definitivo.

### (b) Best-effort SHA

**Reduce inmediatamente el problema actual.**

Pero transforma:

```text
CURRENT = demostrado
```

en:

```text
CURRENT = confiado
```

Es la opción con mayor relajación semántica.

### (c) Marker en Git

**Buena idea de dominio, mala implementación concreta.**

Hay que corregir el problema del SHA/ref del evento y definir rigurosamente qué significa `COMPLETED`. ([GitHub Docs][2])

### (d) Menos slots

No corrige el defecto.

Solamente reduce la cantidad de veces que se manifiesta.

No aprobaría un cambio de arquitectura basado en esa opción sin un estudio histórico de latencia de Yahoo.

---

# 14. Mi decisión como auditor

## No aprobaría ninguna de (a)-(d) exactamente como está redactada.

Aprobaría:

> **(E) separar explícitamente “integridad del artefacto” de “finalización/idempotencia”, utilizando un completion receipt inmutable de GitHub Actions.**

La arquitectura quedaría:

```text
                    DAILY RUN
                        │
                  target_session
                        │
                 ┌──────┴──────┐
                 │             │
           COMPLETION       YAHOO PROBE
             RECEIPT             │
                 │          publicado?
                 │             │
                YES           YES
                 │             │
              CURRENT        READY
                 │
              no run
```

Y paralelamente:

```text
manifest ↔ parquet ↔ sha256
```

sigue siendo **fail-closed** y no se toca.

---

# 15. Condiciones de aprobación del nuevo diseño

Yo pondría estas invariantes en el contrato:

### I1 — Nunca existe `CURRENT` por ausencia de parquet

La existencia del parquet deja de ser requisito del gate.

### I2 — Nunca existe `CURRENT` por un manifest aislado

El manifest no basta para declarar finalización CI.

### I3 — El receipt solamente puede generarse después del último check contractual

Especialmente después de `guard_coverage`, `10/10` y publicación de resultados.

### I4 — Receipt inmutable

No usar un fichero mutable reutilizado.

Un receipt debe identificar una ejecución concreta:

```text
target_session
run_id
commit_sha
manifest_sha256
timestamp
schema_version
```

### I5 — Expiration es fail-open

Si GitHub ya no conserva la evidencia:

```text
NO RECEIPT → READY
```

nunca:

```text
NO RECEIPT → CURRENT
```

### I6 — API failure ≠ CURRENT

Si la consulta a Actions falla:

```text
ERROR
should_run = false
```

No se debe convertir un error de consulta en `CURRENT`.

### I7 — Target session exacta

Nunca reutilizar:

```text
receipt de 2026-09-29
```

para:

```text
target_session = 2026-09-30
```

aunque la cobertura sea suficiente.

### I8 — El receipt no sustituye al manifest

Son dos contratos distintos.

---

# 16. Una observación adicional importante sobre vuestra concurrencia

El `concurrency.group: daily-run` es útil para serializar ejecuciones, pero **no debe considerarse el mecanismo de idempotencia**.

Son problemas diferentes:

```text
concurrency
= evita ejecuciones simultáneas

idempotency gate
= evita ejecuciones repetidas para el mismo target_session
```

Que el primero exista no garantiza el segundo.

GitHub permite en la actualidad configurar cola de ejecuciones mediante `queue: max`, por lo que efectivamente podéis tener workflows distintos esperando secuencialmente en la misma agrupación. ([GitHub Docs][6])

Por eso el estado de finalización debe existir independientemente del snapshot del código con el que comenzó cada workflow.

---

# 17. Veredicto final

### Diagnóstico

**CONFIRMADO — defecto de diseño.**

El gate actual es:

```text
manifest-based locally
probe-based in CI
```

aunque conceptualmente se pretendía:

```text
manifest-based idempotency
+
probe only for publication readiness
```

### (b)

**No lo aprobaría como solución definitiva.** Es válido como hotfix de muy corto plazo si se acepta explícitamente la degradación de garantía.

### (a)

**Aceptable como solución temporal**, especialmente si queréis restaurar idempotencia con cambios mínimos. Pero usaría la cache para un receipt pequeño, no para convertir el parquet en almacén de estado.

### (c)

**La idea es correcta; la implementación propuesta no.** El hecho de que el marker esté commitado en Git introduce una dependencia innecesaria del SHA/ref del workflow y puede producir precisamente otro caso de estado aparentemente válido pero no observable. ([GitHub Docs][2])

### (d)

**No es un arreglo del defecto.**

### Recomendación de auditoría

**E — Completion Receipt en GitHub Actions Artifacts + `_manifest_satisfies()` conservado como comprobación independiente de integridad.**

Es la solución que mejor separa:

```text
"¿los datos están íntegros?"
```

de

```text
"¿ya completamos esta sesión?"
```

y evita que el gate tenga que reconstruir artificialmente en CI un artefacto que no debería necesitar.

La mejora más importante no es cambiar `if not p.exists()`: es **cambiar qué evidencia necesita el gate para poder declarar `CURRENT`**. ([GitHub Docs][3])

[1]: https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching?utm_source=chatgpt.com "Dependency caching reference - GitHub Docs"
[2]: https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows?via=aivyx&utm_source=chatgpt.com "Events that trigger workflows - GitHub Docs"
[3]: https://docs.github.com/en/actions/concepts/workflows-and-actions/workflow-artifacts?utm_source=chatgpt.com "Workflow artifacts - GitHub Docs"
[4]: https://docs.github.com/en/rest/actions/artifacts?utm_source=chatgpt.com "REST API endpoints for GitHub Actions artifacts - GitHub Docs"
[5]: https://docs.github.com/en/enterprise-cloud%40latest/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository?utm_source=chatgpt.com "Managing GitHub Actions settings for a repository - GitHub Enterprise Cloud Docs"
[6]: https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax?utm_source=chatgpt.com "Workflow syntax for GitHub Actions - GitHub Docs"

