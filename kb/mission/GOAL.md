---
aliases: ["GOAL"]
type: codex-goal
---

# Codex Goal

> **Generated**: 2026-11-13

## Command

```text
/goal Advance the Limina research mission in kb/mission/CHALLENGE.md: Identificar cuál de los 18 modelos (de OpenAI, Google, Anthropic, Qwen, ZAI y Moonshot AI) escribió cada pasaje de la competición de Kaggle "Name That LLM", y entender qué rasgos de estilo distinguen a cada modelo y a cada laboratorio.. Use this context: `train.csv` tiene 2.699 pasajes generados a partir de 150 prompts, con las columnas `id`, `prompt_id`, `text` y `model`. El test oculto tiene unos 4.000 pasajes de prompts que **no** aparecen en train: el modelo tiene que generalizar a prompts nuevos. Los pasajes miden entre 40 y 250 palabras, y la longitud es aleatoria. Métrica: balanced accuracy (media del recall por clase). La mezcla de modelos cambia entre el leaderboard público y el privado. Hay modelos muy parecidos dentro de un mismo laboratorio (por ejemplo, gpt-5.6-luna, gpt-5.6-sol y gpt-5.6-terra). Baseline previsto: TF-IDF de n-gramas de caracteres + regresión logística.. Treat success as: Validación local con GroupKFold por `prompt_id` (5 folds) como única referencia de evaluación. Superar el baseline (TF-IDF de n-gramas de caracteres + regresión logística) en al menos 10 puntos de balanced accuracy en esa validación local. Que la validación local y el leaderboard público vayan en la misma dirección en al menos 3 envíos seguidos. Un notebook de envío que corra dentro de los límites de Kaggle y genere un `submission.parquet` válido.. Follow AGENTS.md and keep durable state in kb/. At each continuation, read kb/ACTIVE.md and the linked working set, choose the next highest-value unknown, preserve the H -> E -> F evidence chain, update kb/ACTIVE.md before stopping, and run concrete tools or document why none are possible. Verify completion with a passing `python3 scripts/kb_validate.py` run and a final audit that separates confirmed findings, proxy evidence, blocked claims, and remaining uncertainty. Use only these resources and boundaries: Envío solo mediante notebook de Kaggle: máximo 9 h en CPU o GPU y sin internet. Se permiten datos y modelos preentrenados públicos y gratuitos, subidos como dataset de Kaggle. Entorno local gestionado con `uv`. Los datos van en `data/` y no se suben a git. Plazos: aceptar las reglas y fusionar equipos antes del 2027-01-06; envío final antes del 2027-01-13 a las 23:59 UTC. Prohibido: datos privados y llamadas a APIs de los modelos en el notebook de envío. Nunca validar con particiones aleatorias; toda validación agrupa por `prompt_id`.. Respect these constraints: Ask when blocked on access, trust in the evaluation, or strategic decisions. Persist durable evidence in `kb/`. Keep active state in `kb/ACTIVE.md`.. If blocked or budget-limited, do not mark the Goal complete; Parar y preguntar si: la validación local deja de correlacionar con el leaderboard público; algo necesita más cómputo del que permite Kaggle (9 h, sin internet); hay dudas sobre si un dato o modelo externo está permitido..
```

## Completion Evidence

- `python3 scripts/kb_validate.py` passes.
- `kb/ACTIVE.md` reflects the final objective, next step, blocker, and working set.
- The final audit separates confirmed findings, proxy evidence, blocked claims, and remaining uncertainty.

## Iteration Policy

Use `kb/mission/CHALLENGE.md` as the mission brief, `kb/ACTIVE.md` as the current state, and the linked working set for detailed context. Preserve the `H -> E -> F` evidence chain.

## Blocked Stop Condition

If blocked or budget-limited, do not mark the Goal complete. Parar y preguntar si: la validación local deja de correlacionar con el leaderboard público; algo necesita más cómputo del que permite Kaggle (9 h, sin internet); hay dudas sobre si un dato o modelo externo está permitido..
