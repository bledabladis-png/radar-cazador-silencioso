El TOP 50 confirma el patron con suficiente evidencia para **cerrar el ciclo de mini-probes de continuidad de filer para este periodo**, sin convertirlo en una afirmacion poblacional. La correccion cuantitativa del addendum tambien queda satisfactoriamente cerrada: el doble conteo del HR `SUPERSEDED` quedo aislado y el motor NIPC no estaba afectado.

# DICTAMEN FORMAL DEL AUDITOR

**Objeto:** evaluacion del TOP 50 Filer Continuity Q4 2025 -> Q1 2026 y addendum cuantitativo Vanguard
**HEAD revisado:** `37d2879` / addendum `ba1e6d1`
**Estado:** **CERRADO COMO CARACTERIZACION DEL PERIODO**
**Gate-NIPC.2:** **SIGUE BLOQUEADO** por coverage/thresholds, no por filer continuity
**Gate-NIPC.3:** **NO AUTORIZADO**

---

## 1. Dictamen ejecutivo

El TOP 50 aporta evidencia suficiente para cerrar la investigacion inicial de continuidad de filer:

```text
54 CIKs observados
+---- 50 CONTINUOUS_FILER
+---- 4 FILER_DISCONTINUITY
+---- 0 UNRESOLVED
```

Los cuatro casos de discontinuidad corresponden a la estructura Vanguard.

Ademas, existe evidencia documental directa de la transicion:

```text
13F-NT
   v
OTHER MANAGERS
   v
13F-HR
```

La SEC contempla expresamente este mecanismo: un 13F-NT se utiliza cuando las posiciones del manager se reportan por otros managers, que aparecen identificados en la relacion de "Other Managers". ([SEC][1])

### Conclusion

**No hay evidencia, dentro del TOP 50 de Q4 2025 -> Q1 2026, de un segundo patron de reorganizacion de filer comparable al de Vanguard.**

Eso justifica **cerrar el mini-probe de continuidad**, pero no justifica extrapolar el 92,6 % de continuidad a todo el universo 13F.

---

# 2. Q-T50-1 - ?Top 50 suficiente?

### Decision: **SI, CIERRE DEL CICLO**

No recomiendo extender ahora a TOP 100 ni al universo completo.

La secuencia realizada:

```text
TOP 20
   v
Vanguard detectado
   v
TOP 50
   v
ningun nuevo patron
```

es suficiente para la finalidad original del mini-probe: determinar si el hallazgo Vanguard parecia acompanado por otros patrones visibles entre los grandes filers.

El resultado permite afirmar:

> En el TOP 50 analizado para Q4 2025 -> Q1 2026, las discontinuidades observadas estan concentradas en Vanguard.

No permite afirmar:

> Vanguard es el unico caso existente en todo el universo 13F.

### Estado

**Q-T50-1 = CERRADO.**

No mas expansion de muestra en esta fase.

---

# 3. Q-T50-2 - Presencia por filing frente a shares

### Decision: **PRESENCIA DOCUMENTAL DEL FILER PREVALECE**

Para `filer_status`, la variable correcta es la **existencia del filing relevante en el periodo**, no `SSHPRNAMT > 0`.

Esto es especialmente importante para 13F-NT.

Un `13F-NT` puede tener precisamente la funcion de declarar que las posiciones del manager se reportan por otros managers; por tanto, puede ser perfectamente valido que no existan holdings propios en el snapshot. ([SEC][1])

Por ello:

```text
filing exists
      !=
shares > 0
```

y no deben confundirse.

### Aplicacion a NORGES

El caso:

```text
Q4 shares > 0
Q1 filing presente
Q1 SH+NULL = 0
```

se clasifica correctamente como:

```text
CONTINUOUS_FILER
```

desde el punto de vista de continuidad documental.

Debe anadirse una dimension separada:

```text
position_mass_status
```

o equivalente, por ejemplo:

```text
HAS_SHARES
ZERO_SHARES
NO_CANONICAL_HOLDINGS
```

pero **no cambiar `filer_status` a DISCONTINUITY**.

### Estado

**Q-T50-2 = CERRADO.**

---

# 4. Q-T50-3 - NT <-> HR

### Decision: **PROHIBICION DE RECONCILIACION SE MANTIENE**

La evidencia ya es inequivoca respecto de la relacion documental.

En Vanguard:

```text
Q1 2026
VANGUARD GROUP INC
13F-NT
       v
OTHER MANAGERS
       v
VANGUARD CAPITAL MANAGEMENT LLC
VANGUARD PORTFOLIO MANAGEMENT LLC
       v
13F-HR
```

La SEC describe este mecanismo como reporting realizado por otros managers identificados en el NT. ([SEC][1])

Pero:

```text
REPORTING RELATION
        !=
ECONOMIC OWNER
```

Por tanto:

### No autorizado

```text
EXIT(CIK_A, security)
+
NEW(CIK_B, security)
->
same economic exposure
```

### Si autorizado

```text
NT target observed
NT target CIK
HR target exists
same period
filing lineage
```

como **evidencia documental**.

No se modifica C2.

### Estado

**Q-T50-3 = CERRADO.**

---

# 5. Q-T50-4 - GHISALLO

### Decision: **GO DIAGNOSTICO, PERO NO BLOQUEANTE**

El salto:

```text
2.392B -> 20.691B
Delta ~= +765 %
```

merece auditoria.

Pero no pertenece al problema de filer continuity.

El CIK esta presente en ambos periodos, por lo que el dato no constituye una discontinuidad de filer.

Tampoco debe etiquetarse todavia como:

```text
"crecimiento genuino"
```

El unico hecho demostrado es el cambio cuantitativo del snapshot canonico.

### Probe especifico autorizado

Debe comprobar:

```text
Q4 holdings
Q1 holdings
amendments
no securities
NEW / EXIT / BOTH
concentracion del incremento
top securities por SSHPRNAMT
OTHERMANAGER / filing structure
```

Objetivo:

```text
?crece de forma distribuida?
?entra una gran posicion?
?cambia el perimetro del reporting?
?hay alguna anomalia de canonicalizacion?
```

### Estado

**Q-T50-4 = GO DIAGNOSTICO.**

No debe retrasar la medicion de coverage salvo que aparezca una anomalia de ingestion/canonicalizacion.

---

# 6. Addendum Vanguard - correccion cuantitativa

### Decision: **PASS**

El addendum resuelve correctamente el problema senalado en el dictamen anterior.

La secuencia:

```text
13F-HR
    v
13F-HR/A RESTATEMENT
    v
SUPERSEDED
```

mas:

```text
13F-HR/A NEW HOLDINGS
```

demuestra por que sumar directamente los tres filings produciria doble conteo.

La SEC presenta efectivamente los `13F-HR/A` como enmiendas de los holdings reports; ejemplos actuales muestran que una enmienda puede estar asociada expresamente a un documento de `RESTATEMENT`. ([SEC][2])

La regla metodologica queda correctamente congelada:

> **Todo agregado de `SSHPRNAMT` utilizado por NIPC debe proceder del `canonical_snapshot`; el raw se utiliza para lineage y auditoria de amendments, nunca para agregacion economica.**

### Resultado corregido

```text
Vanguard Capital Q1
35,329,105,562

Vanguard Portfolio Q1
21,030,554,644

Filiales Q1
56,359,660,206
```

La cifra anterior:

```text
91,267,483,611
```

queda definitivamente:

```text
NO UTILIZABLE
```

### Estado

**CORRECCION CUANTITATIVA = CERRADA / PASS.**

---

# 7. Interpretacion correcta de la continuidad observada

Los numeros:

```text
TOP 20 -> 12.5 %
TOP 50 -> 7.4 %
```

no deben incorporarse a la policy como tasas de referencia.

Tampoco debe afirmarse que la discontinuidad "disminuye" estadisticamente al ampliar la muestra.

La lectura correcta es:

```text
Q4 -> Q1
Vanguard = 4 CIK discontinuous
resto top 50 = 0 nuevos casos
```

Es una **concentracion observada en una estructura corporativa de filing concreta** durante un periodo concreto.

---

# 8. Implicacion para el diseno de NIPC

Quedan separadas tres dimensiones:

```text
IDENTITY
?misma security?

REPORTING CONTINUITY
?mismo filing manager?

REPORTING RELATION
?existe evidencia NT / OTHERMANAGER?
```

Y una cuarta dimension que no esta autorizada:

```text
ECONOMIC RECONCILIATION
?misma exposicion economica?
```

La arquitectura actual puede conservar:

```text
filing_manager_cik
+
filer_status
+
nt_to_hr_relation_observed
+
nt_targets
```

sin modificar C2.

Esto evita introducir una atribucion economica que la evidencia disponible todavia no justifica.

---

# 9. ?Debe la continuidad de filer convertirse ahora en un threshold?

### Decision: **NO**

No recomiendo convertir:

```text
92.6 %
```

en:

```text
continuity_threshold = 92.6 %
```

ni ninguna cifra derivada del TOP 20/TOP 50.

La continuidad de filer debe incorporarse inicialmente como:

```text
CONTROL DE INTEGRIDAD / DIAGNOSTICO
```

y no como threshold cuantitativo hasta disponer de una definicion poblacional y longitudinal suficientemente justificada.

Esto mantiene la disciplina del proyecto: **medir primero y no convertir una observacion de una muestra en una regla estructural.**

---

# 10. Q-T50-5 - siguiente paso

### Decision: **GO -> Coverage baseline + OpenFIGI**

El ciclo de filer continuity queda suficientemente caracterizado para avanzar.

El siguiente trabajo principal debe ser:

```text
BASELINE INTERNAL
        v
coverage metrics
        v
OPENFIGI unresolved-only
        v
coverage metrics
        v
comparacion A/B
```

y posteriormente:

```text
evidencia coverage
+
evidencia filer continuity
        v
propuesta THRESHOLD_1 / THRESHOLD_2
```

No recomiendo disenar ahora la reconciliacion NT-HR productiva.

Puede conservarse como **linea de investigacion separada**, no como bloqueo del proximo micro-gate.

---

# 11. Estado final de gates

```text
Q-PROBE-5 TOP 20                     PASS
Q-T50 TOP 50                         PASS
Vanguard concentration               CONFIRMADA EN TOP 50
NT -> OTHERMANAGER -> HR               CONFIRMADO
Addendum cuantitativo Vanguard       PASS
Regla canonical_snapshot             CONGELADA
C2                                  VIGENTE / SIN PATCH
Reconciliacion economica NT-HR       NO AUTORIZADA
GHISALLO                             PROBE DIAGNOSTICO
THRESHOLD_1                          UNDEFINED
THRESHOLD_2                          UNDEFINED

Gate-NIPC.2                         BLOQUEADO
Gate-NIPC.3                         NO AUTORIZADO
Coverage A/B                         SIGUIENTE FASE
```

# DICTAMEN FINAL

**Q-T50-1 -> CERRADO / GO.**
No ampliar ahora a TOP 100/universo completo.

**Q-T50-2 -> CERRADO / GO.**
`filer_status` se determina por **presencia documental del filing**; `SSHPRNAMT > 0` es una dimension separada.

**Q-T50-3 -> CERRADO / NO PATCH.**
NT->HR queda como evidencia documental. No se autoriza inferencia economica.

**Q-T50-4 -> GO DIAGNOSTICO.**
GHISALLO merece probe especifico, pero no bloquea coverage.

**Q-T50-5 -> GO.**
El siguiente paso principal es **coverage baseline + OpenFIGI**, con thresholds todavia `UNDEFINED`.

### Estado de decision

**El problema de filer continuity deja de ser el bloqueo inmediato de Gate-NIPC.2.**

El bloqueo vuelve a ser el que ya estaba definido:

```text
COVERAGE EMPIRICA
        +
THRESHOLD_1
        +
THRESHOLD_2
```

La reconciliacion NT<->HR queda fuera del alcance productivo hasta que exista una especificacion separada y evidencia suficiente para justificar que significa exactamente "continuidad economica".

La observacion de GHISALLO la mantendria como **control de calidad del dato/NIPC**, no como excusa para retrasar indefinidamente el siguiente micro-gate. El proyecto ya tiene suficiente evidencia sobre filer continuity para pasar a la medicion de cobertura.

[1]: https://www.sec.gov/Archives/edgar/data/1983408/000198340826000010/xslForm13F_X02/primary_doc.xml?utm_source=chatgpt.com "SEC FORM 13F-NT"
[2]: https://www.sec.gov/Archives/edgar/data/1275218/000127521826000016/0001275218-26-000016-index.htm?utm_source=chatgpt.com "EDGAR Filing Documents for 0001275218-26-000016"

