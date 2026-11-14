---
aliases: ["ACTIVE"]
type: active-state
---

# Active State

## Current Objective

Superar el baseline (0,307 en las semillas 1 y 2, [[F002]]) en al menos 10 puntos de balanced
accuracy (≥ 0,412). El mejor modelo individual es `gte-modernbert-base` fine-tuneado: 0,357
([[F003]]). El baseline no se sube a Kaggle (decisión del usuario, 2026-11-13); la correlación
con el leaderboard se comprobará con los primeros envíos de modelos candidatos.

## Next Step

[[E003]] terminado (2026-11-14, RTX 4090): [[H003]] rechazada (mejor brazo 0,357 < 0,362) y
[[H004]] rechazada (gte supera a ModernBERT MLM en 10/10 folds, [[F004]]). Decidir con el
usuario la siguiente hipótesis. Propuesta: ensemble TF-IDF + gte con las probabilidades OOF ya
guardadas (sin GPU), y después escalar a un encoder de embeddings más grande o atacar la
confusión dentro del laboratorio. PR #6 fusionado en `main` el 2026-11-14 (issue #5 cerrado).

## Blocker

None.

## Working Set

- [[H003]], [[H004]] → [[E003]] → [[F003]], [[F004]]: encoder fine-tuneado 0,357 (+0,05);
  contrastivo > MLM. OOF por run en `kb/research/data/E003/oof/` (semillas 1 y 2 para el
  ensemble). Base bibliográfica: [[L001]] (HIGH); [[L002]] (LOW).
- [[H002]] → [[E002]] → [[F002]]: baseline 0,312; 52 % de los errores dentro del laboratorio.
  Folds fijos en `kb/research/data/E002/folds.csv`.
- [[H001]] → [[E001]] → [[F001]]: la longitud no identifica al modelo; la longitud de frase
  quizá sí.
- Código: `src/name_that_llm/{validation,metrics,baseline,labels}.py`,
  `src/name_that_llm/{encoder,decisions}.py`,
  `src/name_that_llm/experiments/{run_baseline,run_encoder,provenance}.py`.
- Entorno GPU: `uv sync --group gpu` (torch 2.11+cu128, transformers 5.23), RTX 4090. Los
  pesos se cargan siempre en float32: gte-modernbert-base viene en float16 y diverge a NaN.
- Primary skill de la siguiente fase: `$experiment-rigor` (diseño de la siguiente hipótesis).

## Links

- Mission: [[CHALLENGE]]
- Latest findings: [[F003]], [[F004]]
- Last experiment: [[E003]]
