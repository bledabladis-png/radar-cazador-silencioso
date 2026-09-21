# IAE - A.6.2-bis Pregunta estrategica al auditor

**Objeto:** elevar al auditor una decision de estrategia sobre el ciclo
A.6.2-bis, tras 9 iteraciones de propuesta -> dictamen sin cierre.

**Origen:** dictamen #51 (v8 NO-GO). Acumulacion de 9 versiones
(v1-v9) y 8 dictamenes NO-GO (#44-#51) sobre la misma subfase.

**Fecha:** 2026-09-21.
**HEAD al redactar:** fbfde7f.
**Naturaleza:** consulta estrategica. NO modifica contrato ni codigo.

---

## 0. Resumen ejecutivo

A.6.2-bis lleva **9 iteraciones** (v1 a v9) y **8 dictamenes NO-GO
consecutivos** (#44 a #51). Cada ronda cierra bloqueos reales, pero
el numero de bloqueos nuevos por ronda no esta decreciendo.

Se solicita al auditor:

> Una decision sobre la estrategia del ciclo antes de producir v10.

Tres opciones tecnicas (seccion 5). Recomendacion del redactor:
**Opcion 1** (partir A.6.2-bis en subfases autonomamente cerrables).

---

## 1. Estado del ciclo

| Version | Commit | Dictamen | Bloqueos | Nuevos |
|---|---|---|---|---|
| v1 | f383932 | #44 | 4 | 4 |
| v2 | 9aa0727 | #45 | 3 | 3 |
| v3 | fa03a97 | #46 | 2 | 2 |
| v4 | 30ed3df | #47 | 2 | 2 |
| v5 | 6c5e33c | #48 | 3 | 3 |
| v6 | 641ef38 | #49 | 3 | 3 |
| v7 | 2f1f323 | #50 | 3 | 3 |
| v8 | 3315259 | #51 | 2 | 2 |
| v9 | fbfde7f | pendiente | - | - |

**Total:** 9 versiones. 8 dictamenes NO-GO. 22 bloqueos materiales
cerrados a lo largo del ciclo.

---

## 2. Patron observado

Cada dictamen cierra los bloqueos declarados en el dictamen anterior,
pero descubre entre 2 y 4 bloqueos nuevos.

    v1 -> #44: 4 bloqueos  ->  v2 cierra
    v2 -> #45: 3 bloqueos  ->  v3 cierra
    v3 -> #46: 2 bloqueos  ->  v4 cierra
    v4 -> #47: 2 bloqueos  ->  v5 cierra
    v5 -> #48: 3 bloqueos  ->  v6 cierra
    v6 -> #49: 3 bloqueos  ->  v7 cierra
    v7 -> #50: 3 bloqueos  ->  v8 cierra
    v8 -> #51: 2 bloqueos  ->  v9 cierra (pendiente)

**Observacion:** el numero de bloqueos nuevos por ronda **no decrece
monotonamente**. Se ha estabilizado en 2-3 por ronda desde #46.
---

## 3. Analisis

Cada bloqueo cerrado ha sido material (no cosmetico). No hay ciclos
inutiles. Pero el patron sugiere que **el nivel de detalle exigido
para cerrar A.6.2-bis como una unica subfase completa es muy alto**:
la subfase agrupa 3 bloqueantes estructurales F2.4 (B1, B2, B3) que
tocan simultaneamente:

- identidad administrativa (catalog_key, asignacion).
- identidad economica (share_class_figi, unidad P38).
- disponibilidad temporal (snapshots, point-in-time).
- semantica temporal por posicion (knowledge_date, amendments).
- flujo normativo fail-closed (9 pasos).
- preservacion de interfaces contractuales (P38 intacta).

Es plausible que la convergencia requiera muchas mas rondas si se
mantiene el alcance actual. El ROI del redactor por ronda es
decreciente:

    ronda 1: 4 bloqueos -> 1 arquitectura completa
    ronda 4: 2 bloqueos -> 2 garantias operacionales
    ronda 8: 2 bloqueos -> 2 semantica fina (asignacion + dominio)

Sin garantia de que la ronda 9 (v10) o la ronda 10 (v11) cierren el
ciclo.

**Restriccion:** no se propone reducir la exigencia. Se propone
**reestructurar el alcance** para que cada subfase pueda cerrarse
sobre un conjunto mas pequeno de invariantes.

---

## 4. Subfases del alcance actual

A.6.2-bis agrupa tres bloqueantes F2.4:

**B1 - TARGET independiente del mapping**
- catalog_key (identidad administrativa).
- Adaptador P38 (identidad economica).
- TARGET_PAIRWISE formal.
- Flujo normativo 9 pasos.
- Validators (continuity, collision, assignment).

**B2 - Point-in-time**
- Snapshots inmutables.
- Manifest.
- Intervalos semiabiertos.
- Hash externo.
- No backdating.
- `target_catalog_as_of`.

**B3 - 13F != flujo en tiempo real**
- 3 timestamps (`period_end`, `filing_date`, `knowledge_date`).
- RESTATEMENT / NEW HOLDINGS semantica.
- N/D explicito.
- `absence.py` stub.

**Estado por bloqueante (consolidado de #51 seccion 11):**

    B1  BLOCKED (2 cierres v9: asignacion + dominio P38)
    B2  APPROVED (sin cambios desde #46, 6 rondas aprobado)
    B3  APPROVED CONDITIONAL (desde #48, sin regresiones)

**Dato relevante:** B2 lleva **6 rondas consecutivas aprobado sin
modificacion** (#46, #47, #48, #49, #50, #51). B3 lleva aprobado
condicionalmente 4 rondas sin regresion (#48, #49, #50, #51).
B1 sigue bloqueado, con bloqueos nuevos por ronda.
---

## 5. Opciones

### Opcion 1 - Partir A.6.2-bis en subfases autonomamente cerrables

    A.6.2-bis-B2   -> CERRADO (#46). Autorizado a implementar.
    A.6.2-bis-B1   -> Continua en diseno.
    A.6.2-bis-B3   -> Continua en diseno.

**Rationale:** B2 lleva 6 rondas aprobado sin cambios. Se puede
materializar sin arriesgar coherencia con B1/B3 porque B2 es sobre
**el catalogo** (snapshots + versionado) y B1/B3 son sobre **el
consumo del catalogo** (TARGET + semantica temporal). El acoplamiento
es debil: B1 y B3 *leen* snapshots, no los modifican.

**Ventajas:**
- Rompe el ciclo. B2 se implementa en 1-2 commits.
- A.6.3 y A.6.4 parcialmente desbloqueables (al menos la parte
  catalogo).
- Reduce el alcance de las proximas rondas de B1/B3.

**Riesgo:** el auditor podria argumentar que B1/B2 comparten
estructura (`catalog_key`, `snapshot`) y no son separables. En ese
caso el redactor aplicaria la Opcion 2 o 3.

### Opcion 2 - Clausula de cierre por agotamiento

Inspirada en §14.3.7 de `NIPC_CONTRATOS_SEMANTICOS_v1.md` (aplicada
en P66). Establecer:

    BLOQUEO MATERIAL      -> impide cierre
    RECOMENDACION DIFERIBLE -> no impide cierre, se anota como deuda

Con la v10, declarar A.6.2-bis CERRADO al nivel de arquitectura.
Refinamientos posteriores van a deuda tecnica (`A.6.x`) y **no
bloquean implementacion**.

**Ventajas:**
- Cierra el ciclo inmediatamente.
- Preserva la trazabilidad.

**Riesgo:** contradice la filosofia fail-closed del proyecto. Los
bloqueos declarados hasta ahora no son cosmeticos; anularlos por
agotamiento podria dejar pasar un problema real.

### Opcion 3 - Continuar iterando (v10, v11, v12...)

Sin cambio de estrategia. Cada ronda produce 2-3 bloqueos nuevos.

**Ventajas:** mantiene la exigencia actual.

**Riesgo:** sin garantia de convergencia. El ROI por ronda decrece.

---

## 6. Recomendacion del redactor

**Opcion 1** (partir A.6.2-bis).

Motivos:

1. **B2 esta listo.** 6 rondas aprobado. No hay razon tecnica para
   mantenerlo bloqueado por B1.
2. **B1 y B3 no dependen funcionalmente de B1.** El catalogo versionado
   (B2) es un input; B1 y B3 lo consumen. Se puede implementar el
   input y diferir el consumo.
3. **Desbloquea A.6.3 parcialmente.** A.6.3 es "test P38 de pairing
   segun decision Q12" - requiere catalogo (B2) + TARGET (B1). B2 solo
   cubre la parte catalogo. A.6.4 (recalculo) sigue bloqueado por
   B1 (TARGET real).
4. **Reduce el riesgo de las proximas rondas.** B1 (asignacion +
   dominio P38) y B3 (semantica temporal) pueden iterarse
   separadamente, con menos variables en juego.

Si el auditor prefiere otra opcion, el redactor la aplica sin
objecion. La pregunta es **sobre la estrategia**, no sobre los
bloqueos individuales.

---

## 7. Pregunta al auditor

**Se solicita una decision entre:**

    (1) Partir A.6.2-bis: B2 CERRADO y autorizado a implementar.
        B1 y B3 continuan en diseno separadamente.

    (2) Aplicar clausula de cierre por agotamiento: v10 cierra
        A.6.2-bis al nivel de arquitectura. Refinamientos -> deuda.

    (3) Continuar iterando v10, v11, v12... sin cambio de alcance.

**Si el auditor elige (1):** el redactor prepara un plan de
implementacion de B2 en 2 commits (modelo + integracion + tests) y
mantiene B1/B3 como subfases paralelas.

**Si el auditor elige (2):** el redactor propone un texto de
clausula para A.6.2-bis §X.Y con regla BLOQUEO MATERIAL vs
RECOMENDACION DIFERIBLE y lo somete a dictamen.

**Si el auditor elige (3):** el redactor produce v10 con los 2
cierres de #51 (asignacion + dominio P38) y continua.

---

Fin de la consulta estrategica. HEAD fbfde7f.