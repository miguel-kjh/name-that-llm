---
aliases: ["ACTIVE"]
type: active-state
---

# Active State

## Current Objective

Montar la validación GroupKFold por `prompt_id` (5 folds) y medir el baseline de TF-IDF de
n-gramas de caracteres + regresión logística con balanced accuracy.

## Next Step

Fijar los 5 folds agrupados por `prompt_id`, formular la hipótesis del baseline y medirlo,
informando también el accuracy por tramo de longitud ([[F001]]: la longitud total no
identifica al modelo, así que los pasajes cortos tendrán menos señal).

## Blocker

None.

## Working Set

- [[H001]] → [[E001]] → [[F001]]: EDA de longitud (issue #1). La longitud no depende del
  modelo; la longitud de frase sí, pendiente de confirmar en el EDA de estilo.
- Código: `src/name_that_llm/eda_length.py`, `scripts/eda_length.py`, `tests/`.
- Primary skill de la fase: `$build-maintainable-software` (implementación del EDA).

## Links

- Mission: [[CHALLENGE]]
- Latest finding: [[F001]]
