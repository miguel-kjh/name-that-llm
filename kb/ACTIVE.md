---
aliases: ["ACTIVE"]
type: active-state
---

# Active State

## Current Objective

Convertir el primer modelo que cumple el objetivo local en un envío válido de Kaggle.
Qwen3-Embedding-8B con LoRA llega a 0,443 de balanced accuracy (baseline 0,307 + 13,6
puntos, ≥ 0,412 en los 10 folds; [[F005]]). [[H005]] confirmada.

## Next Step

[[E004]] completado (rama `e004-lora`, issue #7); falta revisar el PR con el usuario.
Después: diseñar el notebook de envío con Q8 (adaptador entrenado en local con todo el
train, subido como dataset) y medir en Kaggle memoria y tiempo de inferencia: la T4 tiene
15 GB y no tiene bf16, y los pesos del 8B ocupan ~15 GB (fp16, 2×T4 o cuantización). Si no
cabe, parar y consultar. Después, [[H008]], [[H007]] y [[H006]] (en ese orden) y el ensemble.

## Blocker

None.

## Working Set

- Propuestas tras [[E004]] (sin experimento todavía, mismo umbral ≥ 0,458 y ≥ 7/10 folds frente
  a Q8): [[H006]] quitar el tema (capas intermedias, adversario de prompt), [[H007]] contrastiva
  con hermanos + regularización, [[H008]] enrutador por grupo de estilo + especialistas.
- [[H005]] → [[E004]] → [[F005]]: Qwen3-Embedding con LoRA escala 0,357 (0,6B) → 0,400 (4B)
  → 0,443 (8B); embeddinggemma-2 0,301 (descartado). OOF en `kb/research/data/E004/oof/`.
  La medición de tiempo en la 4090 es ruidosa (VRAM llena, WSL desborda a memoria del sistema).
- [[H003]], [[H004]] → [[E003]] → [[F003]], [[F004]]: encoder fine-tuneado 0,357 (+0,05);
  contrastivo > MLM. OOF por run en `kb/research/data/E003/oof/` (semillas 1 y 2 para el
  ensemble). Base bibliográfica: [[L001]] (HIGH); [[L002]] (LOW).
- [[H002]] → [[E002]] → [[F002]]: baseline 0,312; 52 % de los errores dentro del laboratorio.
  Folds fijos en `kb/research/data/E002/folds.csv`.
- [[H001]] → [[E001]] → [[F001]]: la longitud no identifica al modelo; la longitud de frase
  quizá sí.
- Código: `src/name_that_llm/{validation,metrics,baseline,labels}.py`,
  `src/name_that_llm/{encoder,decisions}.py`,
  `src/name_that_llm/experiments/{run_baseline,run_encoder,runs,run_lora,provenance}.py`.
- Entorno GPU: `uv sync --group gpu` (torch 2.11+cu128, transformers 5.23), RTX 4090. Los
  pesos se cargan siempre en float32: gte-modernbert-base viene en float16 y diverge a NaN.
- Primary skill de la fase: `$build-maintainable-software` (implementación de E004).

## Links

- Mission: [[CHALLENGE]]
- Latest findings: [[F005]], [[F004]]
- Last experiment: [[E004]]
