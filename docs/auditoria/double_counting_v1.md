# Auditoría Double Counting v1

**Fecha:** 2026-08-26
**Sistema:** Radar de Rotación Sectorial v4.3
**Repositorio:** https://github.com/bledabladis-png/radar-cazador-silencioso
**Rama:** main
**Último commit auditado:** ba91d6c

## Estado oficial

Double Counting — Correcciones estructurales v1 completadas; auditoría de dependencias residuales y OOS pendiente.

## Principios no negociables

1. Dependencia no es doble contabilidad.
2. La correlación por sí sola no justifica eliminar una señal.
3. Toda corrección exige análisis de información incremental.
4. Ninguna corrección se consolida sin validación OOS/multiperíodo.
5. No se tocan pesos ni umbrales sin evidencia OOS.
6. Toda modificación debe ser trazable.

## Canales auditados

### 1. LIS ↔ Leader Breadth

- Evidencia: Spearman = 1.0
- Decisión: LIS fuera de la State Machine
- Estado: corregido

### 2. Tactical ↔ RS/Momentum

- Evidencia: R² = 0.9285, residuo sin relación con Structural
- Decisión: Tactical fuera de la State Machine
- Estado: corregido

### 3. Leader Breadth dentro de Structural Score

- Evidencia: Spearman = 0.7983
- Decisión: Breadth eliminado de Structural
- Estado: corregido

### 4. Persistence → Structural/SLPM

- Evidencia: 21.7% de estados modificados al eliminarla; CONFIRMED→EMERGING
- Decisión: mantener
- Estado: legítimo

### 5. RS → Structural

- Evidencia: ΔAUC = +0.0527
- Decisión: mantener
- Estado: complementario

### 6. VIX/Credit → Financial Conditions → MTE

- Evidencia no circular: ΔAUC = +0.2236
- Decisión: mantener
- Estado: complementario, pendiente de robustez multiperíodo

### 7. Flow

- Evidencia: correlaciones bajas
- Decisión: mantener
- Estado: sin evidencia actual de double counting

## Validación OOS

### Intento fallido de eliminar Persistence

- Resultado: CONFIRMED 74 → 0; todo UNRESOLVED
- Acción: revertido
- Lección: una señal correlacionada puede ser decisiva

## Arquitectura resultante

State Machine decisoria:

- Structural
- Breadth
- Persistence

Fuera de decisión:

- LIS → diagnóstico
- Tactical → Opportunity Map

## Lecciones aprendidas

- Dependencia ≠ redundancia.
- La validación OOS es obligatoria.
- La información incremental es la prueba correcta.
- No tocar pesos sin evidencia.

## Próximos pasos

- Robustez OOS multiperíodo de las correcciones v1.
- Auditoría incremental definitiva de Persistence.
- Robustez de MTE con múltiples eventos externos.
- Monitoreo continuo de dependencias.
- Tests de regresión para bloquear reintroducción de canales redundantes.
