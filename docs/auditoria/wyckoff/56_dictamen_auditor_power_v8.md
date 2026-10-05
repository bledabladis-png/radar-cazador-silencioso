# 56 - Dictamen del auditor sobre POWER v8 SCREENING

Fecha: 2026-10-05
Rama: gold-standard-v3
Resultado: NO-GO / SUSPENDIDO POR INSUFICIENCIA DE POTENCIA PARA Se

---

## Decision

D - SUSPENDER v8 como diseno de validacion.

No se autoriza:
  - run_capa1.py --go
  - entrega a anotadores
  - modificacion ad hoc de los targets

## Resultado

  Sp    PASS
  PPV   PASS
  NPV   PASS
  Se    FAIL
  ──────────────
  GLOBAL = FAIL

## Causa

Con q_D = 1.0512% y pi_Y = 0.008, Se = 0.70:
  FN (Y=1, D=0) en poblacion = 584
  P(FN | D=0) = 0.242%
  Muestreando 400 de D=0: ~0.97 FN esperados
  Para e(Se) <= 0.07 con Se ~ 0.70: n_FN ~ 164
  n_D0 necesarios: ~67.500
  K necesario: ~33.750
  Factor sobre techo 800: 42x

## Opciones evaluadas

A. Relajar e(Se) a 0.20-0.30: RECHAZADA (retrospectiva)
B. Eliminar Se del gate: RECHAZADA (Se es fundamental)
C. Rediseñar ctx: NO AUTORIZADA dentro de v8; via para v9 con
   condiciones: proxy ex ante, independiente de Y, probabilidades
   conocidas, hipotesis de enriquecimiento cuantificada previamente
D. Suspender v8: APROBADA

## Prohibiciones post-cierre

NO bajar e(Se)
NO eliminar Se
NO cambiar pi_Y tras observar FAIL
NO cambiar q_D
NO tocar FROZEN_V19_PARAMS
NO redefinir Y
NO mover unidades entre celdas con efecto retrospectivo
NO cambiar B/R_MC para conseguir PASS
NO declarar PASS por Sp/PPV/NPV
NO entregar unidades a anotadores
NO usar panel futuro para retroajustar power model

## Recomendacion para v9

Ruta 1: aumentar drasticamente anotaciones. Mantener Se target = 0.07,
        dimensionar n_D0 necesario. Economicamente poco viable.
Ruta 2: disenar rare-event enrichment formalmente justificable.
        No basta cambiar ctx. Requiere demostrar
        P(FN | ctx=1, D=0) o factor de enriquecimiento suficiente
        ANTES de observar la nueva simulacion.

Si el factor de enriquecimiento requerido es irrealista,
v9 tambien debera suspenderse.

## Conclusion

v8 FAIL BY CAPACITY - Se
Congelado como artefacto POWER_V8_SCREENING_FAIL.
Se preserva como evidencia.
No se anota. No se ejecuta --go. No se modifica v8 para obtener PASS.