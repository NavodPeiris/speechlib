# Metodologia: ATDD + TDD + One-Piece-Flow

## Principios

1. **ATDD primero**: el acceptance test describe el comportamiento deseado
   desde afuera del sistema (como lo ve el usuario o el pipeline completo).
   Se escribe en RED antes de tocar produccion.

2. **TDD hacia adentro**: para implementar lo que el AT exige, se escriben
   unit tests en RED (test_context, test_pipeline, test_steps, test_cli)
   antes de escribir el codigo de produccion.

3. **Red-Green-Refactor estricto**: ningun cambio de produccion sin test en
   RED previo. El refactor solo ocurre cuando todos los tests son GREEN.

4. **One-piece-flow**: el trabajo se divide en slices verticales minimos.
   Cada slice atraviesa todas las capas necesarias (test + produccion) y
   termina en GREEN antes de abrir el siguiente.

5. **Commit + push por slice**: cada slice completo (GREEN + refactor) genera
   un commit atomico con mensaje descriptivo y se pushea antes de avanzar.

---

## Ciclo por slice

```
1. Identificar el comportamiento del slice (una sola cosa)
2. Escribir AT en test_acceptance.py -> RED
3. Escribir unit tests en la capa afectada -> RED
4. Escribir produccion minima para GREEN
5. Refactor si es necesario (tests siguen GREEN)
6. git commit + git push
7. Avanzar al siguiente slice
```

## Reglas

- Un slice = un commit = un push
- Nunca avanzar al slice N+1 si el slice N no esta en GREEN

### Cobertura en fronteras entre capas (wiring tests)

Unit tests aislados en cada capa no garantizan que los valores fluyan
correctamente entre capas. Este es el "integration gap": dos capas
pueden tener todos sus unit tests en GREEN y aun asi conectarse mal.

El patron GOOS (Growing Object-Oriented Software Guided by Tests) lo
resuelve con el doble loop:
- El AT (loop externo) falla si el "cableado" entre capas es incorrecto.
- Los unit tests (loop interno) validan la logica de cada capa por separado.

Si el AT cubre el comportamiento de extremo a extremo, los bugs de
"wiring" —valor construido en la capa A, consumido incorrectamente en
la capa B— quedan atrapados automaticamente en el loop externo.

Regla: por cada comportamiento relevante, el AT debe ejercer el camino
completo desde la entrada del sistema (CLI, API, evento) hasta el
efecto observable (archivo generado, llamada a funcion, estado del
contexto). Si no hay AT para ese comportamiento, agregar al menos un
test de integracion que cubra la frontera critica, incluyendo el caso
por defecto (ausencia del flag u opcion).


### Tests existentes y cambios de comportamiento

- Los tests de slices anteriores no deben regresar a RED
- Si un cambio de codigo rompe un test existente:

  a) **Comportamiento cambio intencionalmente** -> 
     CORREGIR el test en el mismo slice para reflejar el nuevo comportamiento (pasa a RED, y de ahi se corrige siguiendo TDD)
  
  b) **Comportamiento no deberia cambiar** -> 
     DIAGNOSTICAR y corregir el codigo de produccion (el test esta correcto, pero detecta RED y debe pasar a GREEN)

- Esta regla aplica a cualquier cambio de comportamiento


## Ejemplo: Tests existentes + Cambio de comportamiento

Cuando cambias el comportamiento de una funcionalidad (no solo UI):
- El test existente verifica el comportamiento anterior
- El nuevo comportamiento es diferente
- DEBES actualizar el test existente en el mismo slice
- El test debe verificar el nuevo comportamiento, no el anterior

El objetivo es que todos los tests esten siempre en GREEN.

## Seguimiento de planes basados en slices
**NUNCA** cambiar el orden de implementacion de los slices presentado en un plan
