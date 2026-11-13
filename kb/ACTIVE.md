---
aliases: ["ACTIVE"]
type: active-state
---

# Active State

## Current Objective

Superar el baseline (0,312, [[F002]]) en al menos 10 puntos de balanced accuracy con un
encoder transformer fine-tuneado ([[H003]]), y saber si conviene un encoder MLM o uno de
embeddings ([[H004]]). El baseline no se sube a Kaggle (decisión del usuario, 2026-11-13); la
correlación con el leaderboard se comprobará con los primeros envíos de modelos candidatos.

## Next Step

[[E003]] está implementado y verificado (preflight superado, smoke test de todas las fases,
89 tests), en la rama `e003-encoders` (issue #5). Falta ejecutarlo, **solo cuando el usuario
lo diga**, en este orden y en segundo plano:

1. `python -m name_that_llm.experiments.run_encoder --phase search` (30 runs de 10 épocas,
   ~1,5–2 h). Revisar `search_curves.csv` y el (lr, k*) elegido antes de seguir.
2. `--phase eval`, `--phase control --arms A`, `--phase timing`, `--phase summarize`.
3. Análisis en E003, F003 (y F004 si H004 es concluyente) y PR de `e003-encoders` a `main`.

Objetivo de H003: ≥ 0,412 de balanced accuracy (media de las semillas 1 y 2).

## Blocker

None.

## Working Set

- [[H003]], [[H004]]: encoder fine-tuneado frente al baseline; MLM frente a embeddings con
  el mismo backbone. Base bibliográfica: [[L001]] (relevancia HIGH); [[L002]] (LOW).
- [[H002]] → [[E002]] → [[F002]]: baseline 0,312; 52 % de los errores dentro del laboratorio.
  Folds fijos en `kb/research/data/E002/folds.csv`.
- [[H001]] → [[E001]] → [[F001]]: la longitud no identifica al modelo; la longitud de frase
  quizá sí.
- Código: `src/name_that_llm/{validation,metrics,baseline,labels}.py`,
  `src/name_that_llm/{encoder,decisions}.py`,
  `src/name_that_llm/experiments/{run_baseline,run_encoder,provenance}.py`.
- Entorno GPU: `uv sync --group gpu` (torch 2.11+cu128, transformers 5.23). Los pesos se
  cargan siempre en float32: gte-modernbert-base viene en float16 y diverge a NaN.
- Primary skill de la fase: `$experiment-rigor` (ejecución y análisis de E003).

## Links

- Mission: [[CHALLENGE]]
- Latest finding: [[F002]]
- Running experiment: [[E003]]
- Open hypotheses: [[H003]], [[H004]]
