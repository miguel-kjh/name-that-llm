---
aliases: ["ACTIVE"]
type: active-state
---

# Active State

## Current Objective

Superar el baseline (0,307 en las semillas 1 y 2, [[F002]]) en al menos 10 puntos de balanced
accuracy (≥ 0,412). El mejor modelo individual es `gte-modernbert-base` fine-tuneado: 0,357
([[F003]]). Ahora se prueba si escalar encoders de embeddings con LoRA (Qwen3-Embedding 4B/8B)
llega al objetivo ([[H005]]).

## Next Step

[[E004]] diseñado (issue #7, rama `e004-lora`), con el protocolo reducido acordado con el
usuario: G y Q0 buscan lr, Q4 y Q8 heredan el lr de Q0, y Q8 se ejecuta siempre. Siguiente:
implementar según el plan del issue #7 (añadir `peft`, ampliar `encoder.py`, `runs.py`,
`run_lora.py` y tests), regresión de E003 y smoke test. Ejecutar preflight, piloto y
búsqueda **solo cuando el usuario lo diga**.

## Blocker

None.

## Working Set

- [[H005]] → [[E004]]: encoders de embeddings con LoRA (embeddinggemma-2, Qwen3-Embedding
  0,6B/4B/8B); umbrales 0,412 (confirma) y 0,377 (rechaza).
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
- Primary skill de la fase: `$build-maintainable-software` (implementación de E004).

## Links

- Mission: [[CHALLENGE]]
- Latest findings: [[F003]], [[F004]]
- Last experiment: [[E003]]
- Designed experiment: [[E004]]
- Open hypothesis: [[H005]]
