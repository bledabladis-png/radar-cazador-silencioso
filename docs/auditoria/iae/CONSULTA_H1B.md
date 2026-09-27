# Consulta al auditor externo - Gate 0 de H1-B

**Fecha:** 2026-09-27
**Origen:** Gate 0 read-only previo al rediseno del fix H1-B.
**Referencia:** hallazgo H1-B de AUDITORIA_EXTERNA_2026-09-27.md
**Estado:** dictamen del auditor recibido 2026-09-27. Diseno H1-B en
rediseno. No se implementa ningun fix hasta aprobacion explicita.

---

## 1. Motivo de la consulta

El diseno tecnico de H1-B (DISENO_TECNICO_FIX_H1-B.md, BORRADOR) instruye:

> Resolucion temporal: lista Q4 2025 para periodo Q4 2025; lista Q1 2026
> para Q1 2026; etc.

Antes de implementar, se ejecuto Gate 0 read-only para verificar que la
fuente primaria (Official List SEC) sirve la informacion de tipo CALL/PUT
con granularidad temporal. El resultado demuestra que la Official List
Q4 2025 existe en formato TXT, pero no contiene todos los option CUSIPs
observados en filings Q4. Por tanto, la Official List Q4 no es una fuente
exhaustiva para clasificar retrospectivamente todos los CUSIPs del
universo INFOTABLE.

---

## 2. Evidencia

### 2.1. Conteo de CALL/PUT por trimestre

| Trimestre | Lineas | CALL | PUT | CUSIP sufijo 9 | CUSIP sufijo 5 |
|---|---:|---:|---:|---:|---:|
| 2025Q4 | 12.282 | 11 | 7 | 1.283 | 1.238 |
| 2026Q1 | 24.641 | 6.029 | 6.021 | 2.590 | 2.465 |
| 2026Q2 | 25.333 | 6.128 | 6.116 | 2.680 | 2.532 |

- CALL/PUT = lineas con issuer_description igual a CALL/PUT (pos 41-67).
- CUSIP sufijo 9/5 = lineas cuyo noveno caracter es 9 (patron CALL) o 5 (patron PUT).

### 2.2. Verificacion de staleness

- sha256 local 13flist_2025Q4.txt: 6A70489393BC1D1977D00FABEDB58550E904FE9C0A4A98CF35361920B91847C9
- sha256 descarga fresca (misma URL canonica): identico.
- Bytes: 994.842 local == 994.842 fresh.
- EOL: 12.282 LF, 0 CRLF. Un LF por linea. Header/footer bien formados.
- Conclusion: el fichero local no esta truncado ni stale.
### 2.3. Fuente primaria Q4 2025: existe

- URL canonica https://www.sec.gov/files/investment/13flist2025q4.txt:
  HTTP 200. Es el fichero oficial que SEC enlaza desde su pagina de
  Official List.
- Contiene 12.282 lineas.
- El fichero local coincide byte a byte con la descarga fresca (2.2).

### 2.4. Variantes anuales y -txt: no existen

| URL probada | HTTP |
|---|---|
| https://www.sec.gov/files/investment/13flist2025.txt | 404 |
| https://www.sec.gov/files/investment/13flist2025q4-txt.txt | 404 |
| https://www.sec.gov/files/investment/13flist2024q4-txt.txt | 404 |
| https://www.sec.gov/files/investment/13flist2024.txt | 404 |

No existen ficheros anuales ni variantes -txt para Q4 2025 ni Q4 2024.
La unica fuente para Q4 2025 es la canonica de 2.3.

**Rectificacion registrada (2026-09-27, post-dictamen del auditor).**
El borrador previo de este documento titulaba esta seccion "Busqueda de
fuentes alternativas" y concluia "No existe fichero anual ni variante
-txt para Q4 2025 ni Q4 2024." Esa frase, fuera de contexto, podia
inducir a error. La formulacion correcta: la fuente Q4 2025 canonica
existe (2.3); las variantes alternativas probadas no existen (2.4).
Detalle completo en la seccion 7.

### 2.5. Muestra concreta: AMZN y MU

| CUSIP | Q4 2025 | Q1 2026 | Q2 2026 |
|---|---|---|---|
| 023135106 AMZN equity | COM | COM | COM |
| 023135906 AMZN CALL | NO ENCONTRADO | CALL | CALL |
| 023135956 AMZN PUT | NO ENCONTRADO | PUT | PUT |
| 595112103 MU equity | COM | COM | COM |
| 595112903 MU CALL | NO ENCONTRADO | CALL | CALL |
| 595112953 MU PUT | NO ENCONTRADO | PUT | PUT |

---
## 3. Lectura

Q4 2025 tiene ~12.000 lineas equity + 18 opciones descritas.
Q1 2026 tiene ~12.600 lineas equity + ~12.050 opciones descritas.
Excluyendo opciones, ambos trimestres son comparables en cobertura equity.

Lectura ratificada por el auditor externo:

- La Official List Q4 2025 existe, es oficial y es temporalmente aplicable.
- Contiene opciones (18 descritas), pero NO contiene los option CUSIPs
  problematicos 023135906 (AMZN CALL) ni 595112903 (MU CALL), pese a
  que esos CUSIPs aparecen en filings Q4 como CALL.
- Por tanto: la Official List Q4 no es una fuente exhaustiva de
  clasificacion para todo CUSIP observado en INFOTABLE.
- Adicionalmente, la SEC documenta que los 13F reportan opciones usando el
  CUSIP del subyacente + columna PUT/CALL, aunque en la practica algunos
  filers usan el option CUSIP. Esa asimetria entre fuente oficial y fuente
  de filings no se resuelve con una sola fuente.

Escenario C ("esperar nueva publicacion Q4") queda descartado: el fichero
ya esta publicado y archivado.

---
## 4. Preguntas al auditor y respuestas recibidas

Preguntas originales enviadas 2026-09-27:

1. La instruccion del diseno ("lista Q4 2025 para periodo Q4 2025") es
   inejecutable tal cual: la lista existe pero no contiene los option
   CUSIPs problematicos. Como se calcularon los "97 CUSIPs afectados en
   Q4 2025" del dictamen?

2. Escenarios contemplados:
   - (A) Fuente distinta para Q4 2025.
   - (B) Rediseno con asimetria declarada.
   - (C) Esperar nueva publicacion Q4.

3. Es aceptable un fix asimetrico para cerrar H1-B?

Respuestas del auditor (dictamen 2026-09-27):

1. Los 97 CUSIPs NO pueden atribuirse a la Official List Q4. Debe
   identificarse en el expediente la fuente exacta de esa cifra.

2. B es la direccion correcta. A solo si aparece una fuente alternativa
   realmente oficial. C queda descartado.

3. Si, con asimetria declarada: Q1+ clasificacion externa por Official
   List; Q4 tratamiento conservador con UNRESOLVED. No producir una
   falsa sensacion de cobertura Q4.

---

## 5. Que NO se ha hecho

- No se ha modificado codigo del IAE.
- No se ha implementado el fix H1-B.
- No se ha congelado ningun NIPC baseline.
- No se ha tocado el crosswalk ni los parquets.

---

## 6. Estado de H1-B tras dictamen

**H1-B: BLOQUEADO PARA IMPLEMENTACION.** Razon refinada: no existe una
fuente unica que permita clasificar retrospectivamente todos los CUSIPs
observados en Q4 2025 mediante la Official List Q4. La lista oficial
existe, pero no es exhaustiva para los option CUSIPs observados en
filings.

Proximo paso: redisenar el contrato H1-B en torno a:

    IDENTIDAD + TIPO + TEMPORALIDAD
            |
            v
    EQUITY / OPTION / OTHER / UNRESOLVED
            |
            v
    solo lo verificable entra en NIPC

Los hallazgos H4 (golden) y H5.3 (Q2 ingestado manualmente) siguen su
curso independiente.

---
## 7. Rectificacion registrada (2026-09-27, post-dictamen del auditor)

El auditor externo emitio dictamen sobre este documento el 2026-09-27.
Incluyo una correccion material a la seccion 2.3 de la version original.
Se registra aqui la rectificacion para que quede en el expediente:

**Afirmacion original (incorrecta):**
> "No existe fichero anual ni variante -txt para Q4 2025 ni Q4 2024."

**Formulacion correcta (per dictamen del auditor):**
> "La SEC si dispone de 13flist2025q4.txt. El error estuvo en la
> busqueda de variantes de URL, no en la ausencia del recurso. Lo que
> si es cierto es que dicho archivo no contiene 023135906 ni
> 595112903, pese a que esos CUSIPs aparecen en determinados filings
> Q4 como CALL."

**Causa raiz del error.** Se extrapolo de "no existen variantes -txt"
a "no existe fichero Q4", sin cotejar la afirmacion contra la propia
evidencia en mano: el fichero Q4 canonico ya habia devuelto HTTP 200
en el mismo Gate 0.

**Correccion aplicada.** Seccion 2.3 reformulada (fuente primaria
existe); seccion 2.4 lista las variantes que efectivamente no existen;
seccion 3 anade la lectura ratificada por el auditor; seccion 4
documenta las respuestas del auditor a las 3 preguntas.

**Fuentes del auditor:**
- SEC - Official List of Section 13(f) Securities
  https://www.sec.gov/rules-regulations/staff-guidance/official-list-section-13f-securities
- SEC - FAQ Form 13F
  https://www.sec.gov/rules-regulations/staff-guidance/division-investment-management-frequently-asked-questions/frequently-asked-questions-about-form-13f
- SEC - Form 13F Data Sets
  https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets

---
## 8. Contrato conceptual H1-B tras dictamen

El auditor recomienda el contrato:

    CUSIP observado
           |
           v
    Official List del periodo
           |
           +-- encontrado + opcion      -> OPTION  (excluir del NIPC)
           +-- encontrado + equity      -> EQUITY  (permitir)
           +-- no encontrado            -> UNRESOLVED (no imputar -> excluir)

Regla dura: no encontrado != equity. La equivalencia silenciosa
PUTCALL=NULL + TITLEOFCLASS=COM -> equity:XXX es precisamente lo
que contaminaba el NIPC.

Aplicacion asimetrica:
- Q1 2026 en adelante: clasificacion por Official List (exhaustiva).
- Q4 2025 y anteriores: tratamiento conservador. CUSIP sin tipo
  certificable -> UNRESOLVED. Documentado como limitacion declarada.

El valor -4.317.678.307 sigue siendo PRE_H1B_OBSERVATION, no
baseline contractual. No se congela hasta cierre de H1-B.

---

Fin de la consulta.