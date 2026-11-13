---
aliases: ["ACTIVE"]
type: active-state
---

# Active State

## Current Objective

Comprobar que la validación local (GroupKFold por `prompt_id`) y el leaderboard público de
Kaggle van en la misma dirección, empezando por un notebook de envío con el baseline.

## Next Step

Planificar con el usuario el notebook de envío del baseline (offline, < 9 h, escribe
`submission.parquet` con `id,model`). Referencia local: 0,312 de balanced accuracy
([[F002]]); objetivo de la misión: ≥ 0,412 con los mismos folds.

## Blocker

None.

## Working Set

- [[H002]] → [[E002]] → [[F002]]: baseline 0,312; 52 % de los errores dentro del laboratorio.
  Folds fijos en `kb/research/data/E002/folds.csv`.
- [[H001]] → [[E001]] → [[F001]]: la longitud no identifica al modelo; la longitud de frase
  quizá sí.
- Código: `src/name_that_llm/{validation,metrics,baseline,labels}.py`,
  `scripts/run_baseline.py`.
- Primary skill de la fase: `$experiment-rigor` (protocolo y baseline).

## Links

- Mission: [[CHALLENGE]]
- Latest finding: [[F002]]
