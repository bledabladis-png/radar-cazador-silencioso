He revisado el mini-probe como auditor, incluyendo la mecanica 13F-NT/HR contra documentacion oficial de la SEC. La evidencia **confirma el hallazgo Vanguard**, pero hay una inconsistencia cuantitativa interna que debe corregirse antes de utilizar el resultado para fijar thresholds. La SEC confirma que un 13F-NT se utiliza cuando las posiciones del manager se reportan en el 13F de otros managers y que el Cover Page identifica esos "other managers". ([SEC][1])

# DICTAMEN FORMAL DEL AUDITOR

**Objeto:** Mini-probe Q-PROBE-5 sobre continuidad de filer
**HEAD revisado:** `d75e84c`
**Estado:** evidencia SEC de transicion NT -> HR confirmada
**Dictamen global:** **GO CONDICIONADO**
**Gate-NIPC.2:** **NO CERRADO**
**Gate-NIPC.3:** **NO AUTORIZADO**

---

## 1. Dictamen ejecutivo

El mini-probe demuestra tres hechos relevantes:

1. El caso Vanguard **no es un simple cambio de ticker, CUSIP o identidad de security**: el mismo universo de posiciones pasa de un filer CIK a otros filers y existe evidencia SEC explicita mediante 13F-NT.
2. Dentro del universo analizado de 24 CIKs del top-20 ampliado, la discontinuidad observada esta concentrada en los **tres CIKs de Vanguard**.
3. La mecanica `13F-NT -> OTHERMANAGER -> 13F-HR` esta documentada directamente por la SEC. La propia SEC explica que el 13F-NT identifica al/los manager(s) que reportan las posiciones del manager notificante. ([SEC][1])

Sin embargo, el informe contiene una **inconsistencia cuantitativa critica** en la cifra atribuida a Vanguard Capital Management:

```text
Q1 top-20:
0002100119 = 35.329B shares

Seccion 4:
0002100119 = 70.236B shares
```

y posteriormente:

```text
Q1 filiales:
70.236B + 21.031B = 91.267B
```

La cifra de `70.236B` no es compatible con los `35.329B` de la tabla del top 20.

Ademas, el filing SEC `0002100119-26-001306` es efectivamente un **13F-HR**, periodo 2026-03-31, y el filing `0002100121-26-000861` tambien es un **13F-HR** del mismo periodo. ([SEC][2])

Por ello, **no debe conservarse como cifra demostrada la conclusion "Q1 filiales = 91.27B" hasta reconciliar si 70.236B procede de doble conteo de filings/amendments o de otra agregacion.**

Esto no invalida el hallazgo estructural `NT -> HR`; si invalida temporalmente esa parte concreta del calculo cuantitativo.

---

# 2. Q-MINI-1 - ?Vanguard como limitacion conocida y caso aislado?

### Decision: **SI, con una precision de alcance**

Debe documentarse como:

> **limitacion estructural conocida del matching por `filing_manager_cik`, demostrada mediante un caso Vanguard; dentro del top-20 investigado, el fenomeno observado esta concentrado en Vanguard.**

No debe escribirse:

> "Vanguard es un fenomeno aislado del universo 13F".

El probe solo permite afirmar:

```text
top-20 / 24 CIK observados
    v
3 discontinuidades
    v
3 pertenecen a Vanguard
```

Eso **no permite generalizar al universo completo**.

### Estado

**Q-MINI-1 = CERRADO.**

No dispara todavia un diseno productivo de reconciliacion.

---

# 3. Q-MINI-2 - Precedencia taxonomica

### Decision: **NO recomiendo una precedencia destructiva**

La solucion mas robusta es separar:

```text
primary_filer_status
+
relationship_evidence
```

en lugar de obligar a elegir uno de estos estados excluyentes:

```text
FILER_DISCONTINUITY
NT_TO_HR_RELATION_OBSERVED
```

El caso Vanguard demuestra precisamente que ambos hechos pueden coexistir.

Propongo:

```text
filer_status
    CONTINUOUS_FILER
    FILER_DISCONTINUITY
    UNRESOLVED
```

y un indicador independiente:

```text
nt_to_hr_relation_observed = True / False
```

mas, opcionalmente:

```text
nt_to_hr_relation_targets = [...]
```

Entonces Vanguard queda correctamente representado como:

```text
filer_status = FILER_DISCONTINUITY
nt_to_hr_relation_observed = True
```

Esto es superior a hacer:

```text
NT_TO_HR_RELATION_OBSERVED > FILER_DISCONTINUITY
```

porque la precedencia propuesta por el ingeniero **ocultaria precisamente la discontinuidad** que queremos medir.

### Decision

**Q-MINI-2 = CERRADO.**

No hace falta cambiar el codigo actual: es una decision de taxonomia del probe/politica futura.

---

# 4. Q-MINI-3 - GHISALLO +765%

### Decision: **GO para probe especifico, NO bloqueo del ciclo**

No aceptaria todavia la expresion:

> "crecimiento genuino".

Lo unico demostrado por el informe es:

```text
Q4 = 2.392B
Q1 = 20.691B
Delta = +765%
```

y que el CIK esta presente en ambos periodos.

Eso excluye **esta explicacion concreta** de una simple ausencia/presencia del filer, pero no determina todavia la causa economica.

Puede deberse, entre otras posibilidades, a cambios reales de posiciones, cambios de reporting, cobertura distinta de entidades o composicion de filings.

Por tanto:

```text
GHISALLO
= OBSERVACION RELEVANTE
!= EVENTO ECONOMICO DEMOSTRADO
```

### Probe autorizado

No necesitamos un estudio masivo.

Debe comprobarse:

```text
filings Q4
filings Q1
amendments
numero de securities
top securities por SSHPRNAMT
NEW / EXIT / BOTH
discretion
OTHERMANAGER
```

El objetivo es determinar si el +765% aparece como:

```text
a) aumento distribuido entre muchas posiciones
b) concentracion en pocas posiciones
c) cambio de perimetro de reporting
d) efecto de amendment/canonicalizacion
```

### Decision

**Q-MINI-3 = GO DIAGNOSTICO.**

No modificar thresholds todavia por este dato.

---

# 5. Q-MINI-4 - Reconciliacion NT <-> HR

### Decision: **Si, continua pendiente**

La evidencia ya permite afirmar:

```text
13F-NT
   v
identifica otros managers
   v
13F-HR de esos managers
```

La SEC confirma que el 13F-NT se utiliza precisamente cuando las posiciones se reportan mediante otros managers y que la lista de "Other Managers reporting for this Manager" sirve para localizar esas posiciones. ([SEC][1])

El filing de Vanguard `0000102909-26-002707` esta registrado por la SEC como **13F-NT**, periodo 31-03-2026, mientras que `0002100119-26-001306` y `0002100121-26-000861` son **13F-HR** para el mismo periodo. ([SEC][3])

Pero esto demuestra:

```text
REPORTING RELATIONSHIP
```

no necesariamente:

```text
SAME ECONOMIC EXPOSURE
```

La propia documentacion de la SEC distingue las funciones de estos mecanismos de reporting y advierte que las relaciones de managers no equivalen automaticamente a atribucion economica. ([SEC][4])

### Decision

**Q-MINI-4 = CERRADO COMO DECISION DE ALCANCE.**

No se autoriza todavia reconciliacion productiva.

---

# 6. Q-MINI-5 - ?Top 50 / 100 / universo completo?

### Decision: **GO -> ampliar primero a TOP 50**

No recomiendo saltar directamente al universo completo.

El resultado actual es:

```text
24 CIK observados
21 continuos
3 discontinuos
3 = Vanguard
```

El dato `12,5%` es suficientemente importante como para justificar ampliar la muestra, pero no suficiente para inferir prevalencia poblacional.

La secuencia correcta es:

```text
TOP 20   -> realizado
TOP 50   -> siguiente
TOP 100  -> solo si aparecen nuevos patrones
UNIVERSO -> si el top-100 revela heterogeneidad relevante
```

El Top 50 debe utilizar exactamente la misma metodologia del mini-probe para que la comparacion sea valida.

### Decision

**Q-MINI-5 = GO TOP 50.**

No autorizaria todavia un census completo.

---

# 7. Hallazgo cuantitativo critico que debe corregirse

Este es ahora el punto mas importante del informe.

La tabla del top Q1 dice:

```text
0002100119 VANGUARD CAPITAL MGMT LLC = 35.329B
```

pero la seccion Vanguard dice:

```text
0002100119 = 70.236B
```

y despues utiliza:

```text
70.236B + 21.031B = 91.267B
```

Esto debe investigarse antes de utilizar el numero en cualquier documento normativo.

Una explicacion plausible es que la segunda cifra este agregando **multiples filings/amendments**, mientras que la primera procede del `canonical_snapshot`.

No debe asumirse: debe demostrarse.

La SEC confirma que `0002100119-26-001306` es un 13F-HR y que existen ademas otros filings/amendments para ese CIK y periodo; por tanto, la posibilidad de doble conteo es metodologicamente real y debe auditarse contra la canonicalizacion de FA-2.4. ([SEC][2])

### Accion obligatoria

Antes de cualquier uso cuantitativo:

```text
raw filings
      v
canonical_snapshot
      v
sum SSHPRNAMT
      v
comparar con top-20
```

La cifra correcta debe salir del **mismo snapshot canonico que utiliza NIPC**, nunca de una suma independiente de filings.

### Estado

**BLOQUEO QUANTITATIVO LOCAL DEL INFORME.**

No bloquea la conclusion estructural `NT -> HR`, pero si bloquea la reutilizacion de la cifra `91.27B`.

---

# 8. Que si queda demostrado

Con independencia de esa discrepancia:

```text
Q4 2025
Vanguard Group Inc
CIK 0000102909
13F-HR

        v

Q1 2026
Vanguard Group Inc
13F-NT

        v
OTHER MANAGERS

        v

Vanguard Capital Management LLC
CIK 0002100119
13F-HR

Vanguard Portfolio Management LLC
CIK 0002100121
13F-HR
```

La SEC identifica el filing de Vanguard Group como `13F-NT` para el periodo 2026-03-31 y los dos filings de las filiales como `13F-HR` para el mismo periodo. ([SEC][3])

Esto es evidencia suficientemente fuerte para mantener abierto el issue de **filer continuity**.

---

# 9. Consecuencia para el modelo NIPC

La interpretacion debe quedar separada en tres capas:

```text
CAPA 1 - IDENTIDAD
?Es la misma security?
        v
CUSIP / canonical_security

CAPA 2 - REPORTING
?Es el mismo filing manager?
        v
filing_manager_cik

CAPA 3 - RECONCILIACION
?Existe evidencia documental para relacionar
la transicion entre reporters?
        v
NT / OTHERMANAGER / HR
```

Actualmente el sistema cubre de forma solida las dos primeras dentro de su contrato.

La tercera esta **observada pero no incorporada al calculo**.

Por eso no debe modificarse C2 todavia.

---

# 10. Thresholds

El dictamen anterior sobre thresholds permanece vigente:

```text
THRESHOLD_1 = UNDEFINED
THRESHOLD_2 = UNDEFINED
```

Pero ahora la policy debe contemplar **dos controles independientes**:

```text
Coverage / comparabilidad
+
Filer continuity / reporting integrity
```

No propongo todavia una formula ni un umbral para continuidad.

Primero hay que medir el TOP 50.

---

# 11. Estado formal de gates

```text
Q-PROBE-5                         PASS
NT -> OTHERMANAGER -> HR            CONFIRMADO
Vanguard como limitacion           CONFIRMADO
Reconciliacion economica           NO AUTORIZADA
GHISALLO                           PROBE PENDIENTE
Top-50 filer continuity            GO
Coverage thresholds                UNDEFINED

Gate-NIPC.2                        BLOQUEADO
Gate-NIPC.3                        NO AUTORIZADO
```

# DICTAMEN FINAL

**Q-MINI-1 - GO / CERRADO**
Vanguard queda documentado como limitacion conocida; concentracion observada dentro del top 20, sin generalizacion al universo completo.

**Q-MINI-2 - GO / CERRADO**
No utilizar precedencia destructiva. Separar `filer_status` de `nt_to_hr_relation_observed`.

**Q-MINI-3 - GO DIAGNOSTICO**
Investigar GHISALLO antes de interpretar el salto como fenomeno economico.

**Q-MINI-4 - CERRADO**
La relacion NT->HR esta demostrada documentalmente; la reconciliacion economica sigue sin autorizacion.

**Q-MINI-5 - GO TOP 50**
Ampliar la muestra antes de fijar thresholds. No saltar aun al universo completo.

**HALLAZGO ADICIONAL OBLIGATORIO:** reconciliar la discrepancia `35.329B vs 70.236B` de Vanguard Capital usando el `canonical_snapshot`. Hasta resolverla, la cifra `91.267B` queda **NO UTILIZABLE COMO EVIDENCIA CUANTITATIVA**.

La siguiente accion autorizada es, por tanto, **dos probes pequenos y separados**: primero reconciliar la cifra Vanguard contra FA-2.4/canonical snapshot; despues ejecutar el **TOP 50 filer-continuity probe**. No modificar C2, NIPC ni fijar thresholds todavia.

Hay ademas una conclusion metodologica importante: **no usaria el 87,5% de continuidad del top-20 como argumento para establecer un threshold**. Es una observacion descriptiva de esa muestra y periodo, no una estimacion suficiente de la poblacion 13F.


