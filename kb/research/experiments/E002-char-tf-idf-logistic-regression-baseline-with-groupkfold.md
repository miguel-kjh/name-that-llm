---
id: "E002"
aliases: ["E002"]
type: experiment
status: COMPLETED
hypothesis: "H002"
created: "2026-11-13"
completed: "2026-11-13"
tags: [baseline, validation]
---

# E002 — Char TF-IDF + logistic regression baseline with GroupKFold

> **Status**: COMPLETED
> **Hypothesis**: [[H002]]
> **Created**: 2026-11-13
> **Completed**: 2026-11-13

## Objective

Fijar el protocolo de evaluación (GroupKFold por `prompt_id`, 5 folds) y la cifra de referencia
del baseline de la misión, y decidir si sus errores se concentran dentro de cada laboratorio
([[H002]]: proporción > 0,50 frente a 0,176 al azar). Issue de GitHub #3.

## Setup

- Environment: `.venv` del proyecto, Python 3.12, scikit-learn 1.9.1. Versiones exactas, hash
  de los datos y revisión del código en `kb/research/data/E002/manifest.json`.
- Data: `data/train.csv` completo (2.699 filas, 150 prompts, 18 modelos).
- Validation: `GroupKFold(n_splits=5, shuffle=True)` por `prompt_id`, semillas 0, 1 y 2.
- Model: TfidfVectorizer (`char_wb`, n-gramas 1–5, sublinear_tf, min_df=2, sin pasar a
  minúsculas) + LogisticRegression (C=10, max_iter=2000). Configuración fija, sin ajustar.
- Compute: CPU, unos 4 minutos en total.
- Code: `src/name_that_llm/{validation,metrics,baseline,labels}.py`, `src/name_that_llm/experiments/run_baseline.py`.

## Procedure

1. `source .venv/bin/activate && python -m pytest tests` (38 tests de folds, métricas y
   pipeline).
2. `python -m name_that_llm.experiments.run_baseline`: por cada semilla, predicciones out-of-fold con un modelo
   nuevo por fold; después, control con etiquetas barajadas (semilla de folds 0).

## Expected Outcome

- Confirm if: `python -m name_that_llm.experiments.run_baseline` -> proporción de errores dentro del
  laboratorio > 0,50
- Reject if: `python -m name_that_llm.experiments.run_baseline` -> proporción < 0,35
- Inconclusive if: entre 0,35 y 0,50, o menos de 100 errores.
- Guardarraíl: el control con etiquetas barajadas debe dar un balanced accuracy ≤ 8 %.

## Progress

- [x] (`2026-11-13`) Tests de folds, métricas y pipeline: 38 pasan.
- [x] (`2026-11-13`) Ejecución con 3 semillas y control con etiquetas barajadas.
- [x] (`2026-11-13`) El script se movió de `scripts/run_baseline.py` a
  `src/name_that_llm/experiments/run_baseline.py` (`scripts/` es de Limina). Se volvió a
  ejecutar desde la nueva ubicación: todos los CSV y el `summary.json` salen idénticos byte a
  byte. El `manifest.json` guardado es el de la ejecución original, así que su `command` aún
  muestra la ruta antigua y su `code_revision` lleva `-dirty` (se ejecutó antes del commit
  `27e800e`, con el mismo código).

## Results

Datos en bruto en `kb/research/data/E002/` (`manifest.json`, `summary.json`, `runs.csv`,
`folds.csv`, `oof.csv`, `per_class_recall.csv`, `confusion.csv`, `length_buckets.csv`).

| Métrica (OOF) | Semilla 0 | Semilla 1 | Semilla 2 | Media | Azar |
|---|---|---|---|---|---|
| Balanced accuracy (modelo) | 0,322 | 0,310 | 0,304 | **0,312** | 0,056 |
| Balanced accuracy (laboratorio) | 0,545 | 0,533 | 0,538 | 0,539 | — |
| Errores dentro del laboratorio | 0,525 | 0,516 | 0,513 | **0,518** | 0,176 |
| Número de errores | 1.831 | 1.863 | 1.877 | 1.857 | — |

- Por fold (15 folds): balanced accuracy 0,312 ± 0,015, de 0,283 a 0,331.
- Control con etiquetas barajadas: balanced accuracy 0,056 (azar 0,056); proporción dentro
  del laboratorio 0,171 (azar 0,176). Guardarraíl superado.
- Por longitud (acumulado de las 3 semillas): < 80 palabras 0,256; 80–149 0,313;
  150+ 0,375 de balanced accuracy.
- Recall por clase: de 0,16 (glm-5.3-flash, claude-fable-5) a 0,59 (gemini-3.1-pro-preview).
- Confusiones dentro del laboratorio que superan a los aciertos (acumulado de las 3 semillas):
  glm-5.3-flash se predice más como glm-5.3 (76) que como sí mismo (70); gemini-3.8-flash más
  como gemini-3.7-flash (141) que como sí mismo (137); gpt-5.6-sol más como gpt-5.6-luna (93)
  y casi igual como gpt-5.6-terra (82) que como sí mismo (79).

## Analysis

- La validación es fiable: el control con etiquetas barajadas cae exactamente al azar, tanto en
  balanced accuracy como en la proporción dentro del laboratorio. Esto también comprueba la
  fórmula de la proporción al azar (0,171 medido frente a 0,176 teórico).
- [[H002]] se confirma con la regla fijada: 0,518 > 0,50 en la media y en las 3 semillas por
  separado (0,513–0,525), con unos 1.860 errores. El margen es estrecho: casi la mitad de los
  errores son entre laboratorios, así que no basta con resolver los hermanos.
- El laboratorio se acierta mucho mejor que el modelo (0,539 frente a 0,312), pero 0,539 sobre
  6 laboratorios sigue lejos de resolverse.
- La longitud importa: los pasajes de 150+ palabras se aciertan 12 puntos más que los de menos
  de 80.
- Las familias más difíciles son ZAI (glm-5.3 frente a su flash, casi indistinguibles),
  gemini 3.6/3.7/3.8-flash y gpt-5.6 luna/sol/terra, como avisaba la misión.

## Decision

- Cifra de referencia del baseline: **0,312 de balanced accuracy** (GroupKFold por
  `prompt_id`, 5 folds, media de 3 semillas, ± 0,015 entre folds). Objetivo de la misión:
  ≥ 0,412 con este mismo protocolo y estos mismos folds (`folds.csv`).
- Un clasificador en dos etapas (laboratorio y luego modelo) se puede probar, pero no basta
  por sí solo: hay que mejorar también la separación entre laboratorios.
- Siguiente: notebook de envío a Kaggle para comprobar que la validación local y el
  leaderboard van en la misma dirección.

## Links

- Mission: [[CHALLENGE]]
- Active State: [[ACTIVE]]
- Parent Hypothesis: [[H002]]
- Finding: [[F002]]
