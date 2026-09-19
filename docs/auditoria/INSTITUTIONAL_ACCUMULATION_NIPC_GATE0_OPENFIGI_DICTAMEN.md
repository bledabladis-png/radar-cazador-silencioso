# DICTAMEN DEL AUDITOR EXTERNO

## GATE 0 - OPENFIGI COMO SERVICIO DE CROSSWALK CUSIP -> SECURITY/TICKER

**Fecha:** 19-09-2026
**Referencia:** IAE / NIPC / Gate 0 OpenFIGI
**HEAD revisado:** `8f0561f` local
**Resultado:** **PASS COMO FUENTE CANDIDATA - NO ADOPCION COMO FUENTE UNICA**
**Siguiente paso:** **GO para Gate 0 SEC 13(f)**
**Codigo NIPC:** NO autorizado todavia

---

## 1. Q-OF-1 - La muestra estratificada es suficiente?

### **SI, para una decision de candidatura; NO, para estimar cobertura global**

El diseno de la muestra cumple lo solicitado: radar con ground truth; top por `SSHPRNAMT`; tramo medio/bajo; casos problematicos; cardinalidad y clases; errores de servicio separados.

Por tanto, el Gate ha demostrado suficientemente que **OpenFIGI merece entrar en la arquitectura candidata**.

Pero queda corregida una interpretacion: 74.3% es la cobertura observada sobre los 152 CUSIPs de la muestra, no la cobertura estimada del universo 24.838.

Asimismo: A=95.8%, C=81.2%, D=48.0% son resultados de esos estratos concretos, no estimaciones estadisticas de todas las large caps, mid caps o small caps del universo.

### Dictamen

**Q-OF-1 = PASS.**

---

# 2. Q-OF-2 - Se adopta OpenFIGI?

## **SI, pero como fuente secundaria de resolucion, no como autoridad unica**

La evidencia permite adoptar conceptualmente:

    Fuente 1: internal verified mapping
            v
    Fuente 2: OpenFIGI
            v
    unresolved

No apruebo OpenFIGI como fuente unica.

### Razones

Evidencia positiva fuerte: ground truth estrato A 46/46 respuestas con ticker correcto; top CUSIPs 39/48 encontrados. OpenFIGI proporciona metadata util: ticker, name, securityType, marketSector, exchCode, shareClassFIGI y FIGI.

Pero existen gaps reales: 29 No identifier found; 10 Invalid idValue format; 1 Multiple candidates. Y un caso importante: 30231G102 = XOM, OpenFIGI -> No identifier found.

Por tanto, la fuente no puede ser unica.

### Arquitectura aprobada

    CUSIP 13F
        v
    internal verified mapping
        v
    si no existe
        v
    OpenFIGI
        v
    si no resuelve
        v
    UNRESOLVED

### Conflicto

Si existe internal verified CUSIP -> X y OpenFIGI -> Y, no se sobreescribe automaticamente. Debe producirse CONFLICT y conservar ambas evidencias.

### Dictamen

**Q-OF-2 = GO como fallback externo controlado.**

No queda aprobada todavia la produccion masiva.

---

# 3. Q-OF-3 - Que hacer con los 29 `No identifier found`?

## **No resolverlos por heuristica**

`No identifier found` debe convertirse en OPENFIGI_NOT_FOUND y posteriormente, si ninguna fuente legitima lo resuelve: UNRESOLVED. No adivinar por NAMEOFISSUER ni corregir por CUSIP parecido.

### Caso XOM

El caso 30231G102 es especialmente importante porque ya existe un mapping interno verificado: 30231G102 -> XOM. Por tanto internal=XOM + OpenFIGI=NOT_FOUND no constituye conflicto de identidad. Es SOURCE_GAP_OPENFIGI y el mapping interno prevalece.

### Los demas

Solo recomendaria investigacion individual cuando concurra operational radar + peso relevante, o cuando el caso provoque una caida material de coverage. No se justifica investigar los 29 uno por uno ahora.

### Dictamen

**Q-OF-3 = GO - conservar como gap de fuente; internal mapping tiene precedencia.**

---

# 4. Q-OF-4 - Pasamos a SEC 13(f)?

## **SI - AUTORIZADO**

Este es el siguiente paso correcto.

La SEC mantiene una **Official List of Section 13(f) Securities** y la actualiza trimestralmente; actualmente estan publicadas tanto la lista del Q1 2026 como la del Q4 2025.

Para NIPC debemos trabajar especificamente con Q4 2025 Official List (baseline) y Q1 2026 Official List (target). No con la lista actual Q2 2026 para decidir retroactivamente la elegibilidad de Q1.

### Gate 0 SEC 13(f) debe medir

    CUSIPs 13F Q4 -> presentes en Official List Q4
    CUSIPs 13F Q1 -> presentes en Official List Q1

y separar: eligible / not_listed / option_variant / changed_status.

La SEC describe ademas la estructura del fichero oficial con CUSIP, option indicator, issuer name, issuer description y status, por lo que esta fuente tiene exactamente el tipo de dimension normativa que nos falta: eligibility, no ticker mapping.

### Dictamen

**Q-OF-4 = GO.**

---

# 5. Q-OF-5 - Los 22 tickers sin CUSIP

## **No utilizarlos todavia como mapping canonico**

OpenFIGI soporta `TICKER` como `idType` y acepta `exchCode`, por lo que tecnicamente es posible hacer ticker -> OpenFIGI -> CUSIP/FIGI/metadata.

Pero para NIPC eso solo debe ser generacion de candidatos.

No BRK-B -> CUSIP actual -> asumir CUSIP Q1 2026, porque estamos trabajando con snapshots historicos.

### Politica aprobada

    RADAR TICKER
          v
    reverse lookup candidato
          v
    buscar CUSIP resultante en Q1 2026
          v
    validar NAMEOFISSUER / TITLEOFCLASS
          v
    si coherente -> candidate mapping
    si no -> unresolved

Y si devuelve varios instrumentos: MULTIPLE_CANDIDATES, no elegir uno arbitrariamente.

### Por ahora

Los 22 permanecen UNMAPPED_RADAR_SECURITY hasta ese analisis.

**Q-OF-5 = GO condicionado; no entra aun en la capa canonica.**

---

# 6. Q-OF-6 - API key

## **GO para solicitar/usar API key en la fase de escalado**

Correccion aritmetica: OpenFIGI documenta 25/min y 5 jobs/request sin clave; 25/6s y 100 jobs/request con clave.

Para 24.838 CUSIPs:

Sin API key: ceil(24838/5) = 4.968 requests a 25/min = ~198.7 min = ~3h19min.
Con API key: ceil(24838/100) = 249 requests a 25/6s = ~59.8 segundos antes de overhead/reintentos.

La diferencia operacional es enorme. OpenFIGI permite obtener API key creando cuenta; para cuentas institucionales hay soporte de emails institucionales compartidos.

### Condiciones

La clave: NO debe entrar en Git, NO debe aparecer en scripts, NO debe aparecer en manifests. Debe utilizarse mediante variable de entorno/secret local.

### Licencia

Los identificadores FIGI estan dedicados al dominio publico; existen restricciones de licencia sobre identificadores propietarios de terceros y la API no devuelve esos identificadores. Antes del uso productivo hay que documentar exactamente que campos de la respuesta se almacenaran y redistribuiran.

### Dictamen

**Q-OF-6 = GO para escalado posterior.**

No hace falta solicitarla para continuar inmediatamente con SEC 13(f), pero sera recomendable antes del mapping masivo.

---

# 7. Q-OF-7 - Contrato OpenFIGI

## **GO condicionado - se fijan ya las reglas fundamentales**

### Precedencia

    1. Internal verified mapping
    2. OpenFIGI
    3. Unresolved

Pero internal no es automaticamente correcto. Cuando haya discrepancia: CONFLICT sin sobrescribir.

### Trazabilidad minima

Cada resolucion externa debera poder reconstruirse mediante: input_identifier, id_type, resolved_figi, resolved_security, ticker, security_type, market_sector, exch_code, share_class_figi, mapping_source, mapping_timestamp_utc, mapping_endpoint, mapping_result_status. Y un hash/manifest del lote consultado.

### Temporalidad

Este es ahora el principal asunto pendiente.

OpenFIGI documenta que un FIGI asignado a un instrumento no cambia una vez emitido, lo que hace interesante el FIGI externo como posible identificador estable. Pero la metadata asociada, especialmente ticker, no debe tratarse automaticamente como historica.

Por tanto OpenFIGI actual != verdad historica Q1 hasta validarlo.

### Ambiguedad

MULTIPLE_CANDIDATES -> AMBIGUOUS -> NO resolver automaticamente. No convertirlo silenciosamente en EXACT.

### Dictamen

**Q-OF-7 = GO condicionado a temporalidad + snapshot de respuestas.**

---

# 8. Observacion importante: el FIGI externo y el FIGI del 13F no son el mismo problema

El Gate anterior descarto INFOTABLE.FIGI como pivote porque los filers lo proporcionan de forma inconsistente.

Eso no impide utilizar CUSIP -> OpenFIGI API -> FIGI externo como resultado de un servicio de resolucion.

Son dos fuentes diferentes:

    13F.FIGI       = dato declarado por filer
    OpenFIGI.FIGI  = resultado de resolver el identificador contra OpenFIGI

La documentacion de OpenFIGI define el FIGI como identificador unico de un instrumento y establece que el FIGI asignado no cambia una vez emitido.

Esto merece ser investigado explicitamente en Gate-NIPC.1 porque puede convertirse en la base de canonical_security, pero todavia no lo doy por decidido.

---

# 9. Estado de los 9 criterios solicitados

| Criterio          | Evidencia actual                                                         |
| ----------------- | ------------------------------------------------------------------------ |
| Coverage          | Variable; alta en A/C, baja en D; muestra, no estimacion poblacional     |
| Temporal validity | PENDIENTE                                                                |
| CUSIP support     | CONFIRMADO                                                               |
| US equities       | CONFIRMADO                                                               |
| Class handling    | PROMETEDOR; requiere validacion contractual                              |
| Rate limits       | CONFIRMADO                                                               |
| Reproducibility   | POSIBLE; exige snapshot/manifest                                         |
| Licensing/usage   | FIGI abierto; revisar campos/metadatos almacenados                       |
| Failure modes     | BIEN CARACTERIZADOS                                                      |

OpenFIGI documenta oficialmente `ID_CUSIP`, `TICKER`, `securityType`, `marketSector`, `shareClassFIGI`, limites de uso y codigos de error `429/500/503`.

---

# 10. Dictamen final

## **GATE 0 OPENFIGI -> PASS COMO FUENTE CANDIDATA**

No hay base para declararlo OpenFIGI = fuente unica, pero si hay base suficiente para OpenFIGI = candidato real para segunda capa de mapping, con internal verified -> OpenFIGI fallback -> unresolved y CONFLICT explicito.

### Siguiente paso

## **GATE 0 SEC 13(f) -> GO**

Debe medir sobre ambos periodos: 2025Q4 Official List y 2026Q1 Official List, porque la SEC publica la lista trimestralmente y mantiene especificamente los archivos historicos Q4 2025 y Q1 2026.

### Despues

    Gate 0 SEC 13(f)
            v
    Gate-NIPC.1

y sera en Gate-NIPC.1 donde se fijaran definitivamente: canonical_security, mapping precedence, temporal validity, OpenFIGI snapshot, coverage thresholds, CONFLICT / AMBIGUOUS / UNRESOLVED.

No autorizaria todavia el mapping masivo de los 24.838 CUSIPs ni la escritura de `delta_shares.py`.

**Estado formal: OPENFIGI PASS como capa candidata; SEC 13(f) GO; NIPC continua sin codigo productivo.**
