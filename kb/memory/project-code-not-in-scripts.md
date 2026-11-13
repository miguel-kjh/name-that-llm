# El código del proyecto no va en `scripts/`

**Regla:** `scripts/` es del framework Limina (`kb_validate.py`, `kb_new_artifact.py`, hooks…).
El código del proyecto va en el paquete `src/name_that_llm/`: la lógica en módulos del
paquete y los puntos de entrada de experimentos en `src/name_that_llm/experiments/`, que se
ejecutan con `python -m name_that_llm.experiments.<nombre>`.

**Por qué:** el usuario lo corrigió el 2026-11-13, después de que `eda_length.py` y
`run_baseline.py` acabaran en `scripts/` y quedaran mezclados con el framework. Además, tener
todo en el paquete permite subirlo a Kaggle como dataset y reutilizarlo en el notebook.

**Cómo aplicarla:** al crear cualquier script de experimento, EDA o envío, ponerlo en
`src/name_that_llm/experiments/` (o en un módulo del paquete si es lógica reutilizable). No
añadir nada a `scripts/` salvo cambios del propio Limina.
