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

Implementar y ejecutar [[E003]] (compartido por [[H003]] y [[H004]]): fine-tuning completo de
`ModernBERT-base` y `gte-modernbert-base` con mean pooling + cabeza lineal, mismos folds de
E002, rejilla de lr con la semilla 0 y decisión con las semillas 1 y 2. Objetivo: ≥ 0,412.

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
  `src/name_that_llm/experiments/run_baseline.py`.
- Primary skill de la fase: `$experiment-rigor` (diseño de E003).

## Links

- Mission: [[CHALLENGE]]
- Latest finding: [[F002]]
- Open hypotheses: [[H003]], [[H004]]
