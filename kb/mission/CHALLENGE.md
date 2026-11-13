---
aliases: ["CHALLENGE"]
type: mission
---

# Research Mission

## Objective

Identificar cuál de los 18 modelos (de OpenAI, Google, Anthropic, Qwen, ZAI y Moonshot AI) escribió
cada pasaje de la competición de Kaggle "Name That LLM", y entender qué rasgos de estilo distinguen
a cada modelo y a cada laboratorio.

## Context

- `train.csv` tiene 2.699 pasajes generados a partir de 150 prompts, con las columnas `id`,
  `prompt_id`, `text` y `model`.
- El test oculto tiene unos 4.000 pasajes de prompts que **no** aparecen en train: el modelo tiene
  que generalizar a prompts nuevos.
- Los pasajes miden entre 40 y 250 palabras, y la longitud es aleatoria.
- Métrica: balanced accuracy (media del recall por clase). La mezcla de modelos cambia entre el
  leaderboard público y el privado.
- Hay modelos muy parecidos dentro de un mismo laboratorio (por ejemplo, gpt-5.6-luna, gpt-5.6-sol
  y gpt-5.6-terra).
- Baseline previsto: TF-IDF de n-gramas de caracteres + regresión logística.

## Success Criteria

- Validación local con GroupKFold por `prompt_id` (5 folds) como única referencia de evaluación.
- Superar el baseline (TF-IDF de n-gramas de caracteres + regresión logística) en al menos
  10 puntos de balanced accuracy en esa validación local.
- Que la validación local y el leaderboard público vayan en la misma dirección en al menos
  3 envíos seguidos.
- Un notebook de envío que corra dentro de los límites de Kaggle y genere un `submission.parquet`
  válido.

## Resources & Boundaries

- Envío solo mediante notebook de Kaggle: máximo 9 h en CPU o GPU y sin internet.
- Se permiten datos y modelos preentrenados públicos y gratuitos, subidos como dataset de Kaggle.
- Entorno local gestionado con `uv`. Los datos van en `data/` y no se suben a git.
- Plazos: aceptar las reglas y fusionar equipos antes del 2027-01-06; envío final antes del
  2027-01-13 a las 23:59 UTC.
- Prohibido: datos privados y llamadas a APIs de los modelos en el notebook de envío.
- Nunca validar con particiones aleatorias; toda validación agrupa por `prompt_id`.

## Constraints

- Ask when blocked on access, trust in the evaluation, or strategic decisions.
- Persist durable evidence in `kb/`.
- Keep active state in `kb/ACTIVE.md`.

## Blocked Stop Condition

Parar y preguntar si:

- la validación local deja de correlacionar con el leaderboard público;
- algo necesita más cómputo del que permite Kaggle (9 h, sin internet);
- hay dudas sobre si un dato o modelo externo está permitido.

## Links

- Active State: [[ACTIVE]]
- Dashboard: [[DASHBOARD]]
