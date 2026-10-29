# Dictamen externo — DICT3-H04-INFERENCE-V2

Fecha: 2026-10-29 · Frente: DICT3-H04-INFERENCE-V2 · Estado
evaluado: propuesta (doc 76) pendiente de firma.

## 1. Dictamen ejecutivo

NO APROBABLE EN SU REDACCIÓN ACTUAL. Requiere revisión metodológica
antes de la firma.

La propuesta mejora el diseño rechazado en doc 69 (delimita estimando,
preserva dependencia transversal, distingue equivalencia de
insuficiencia, congela validación prospectiva). Sin embargo, contiene
problemas estadísticos que afectan a la validez del intervalo, la
clasificación de resultados y la coherencia entre tamaño muestral y
calendario.

Bloqueantes:
- B1: bootstrap no completamente especificado.
- B2: TOST y tres veredictos no implementados coherentemente.
- B3: holdout 2027 no garantiza 36 bloques efectivos (1 año = 12
  meses, no 36).
- B4: métrica y unidad de observación no suficientemente definidas.

## 2. Hallazgos críticos

H-01 CRÍTICA: incompatibilidad temporal holdout prospectivo (§2.6
  exige 36 meses; §4 usa 2027 = 12 meses).
H-02 CRÍTICA: estimando validación no coincide con estimando declarado
  (histórico 2022-2026 vs confirmatorio 2027).
H-03 CRÍTICA: regla TOST incompleta, DIFERENTE no ejecutable, veredictos
  no exhaustivos ni mutuamente excluyentes.
H-04 ALTA: justificación bootstrap mensual no demuestra validez.
H-05 ALTA: fecha cierre ventana en el futuro (29-oct vs 31-oct).
H-06 ALTA: unidad de observación y ponderación no definidas.
H-07 ALTA: margen equivalencia 0.05 sin validación.
H-08 ALTA: regla publicación IC95 confunde inferencia con significación.
H-09 ALTA: validación anual no cubre todas las afirmaciones.
H-10 a H-14: coherencia de datos y arquitectura (2026 incompleto,
  cambios de régimen, convergencia bootstrap, política ausentes,
  integridad entre capas).

## 3. Propuesta corrección mínima (v3)

1. Separar estimandos histórico (exploratorio) y confirmatorio (2027).
2. Resolver tamaño muestral (año justificado o ventana 36 meses).
3. Formalizar algoritmo inferencial completo.
4. Justificar margen 0.05 y bootstrap.
5. Separar resultados de autorización (INSUFICIENTE no oculta IC).
6. Blindar registro confirmatorio.

## 4. Criterios de aceptación v3

Checklist de 10 puntos (estimandos separados, compatibilidad ventana/
mínimo bloques, fórmula structure_ratio, algoritmo bootstrap, reglas
TOST/DIFERENTE/INSUFICIENTE inequívocas, margen 0.05 justificado sin
datos confirmatorios, política ausentes, publicación IC separada,
control multiplicidad, registro verificable).

## 5. Dictamen final

NO APROBABLE — REQUIERE V3.

Estado productivo: `structure_ratio` continúa descriptiva. No se
autoriza IC95 inferencial en reporte a partir de esta propuesta.

Siguiente paso: emitir DICT3-H04-INFERENCE-V3 con las correcciones y
someterla a nueva auditoría.

Fin del dictamen.