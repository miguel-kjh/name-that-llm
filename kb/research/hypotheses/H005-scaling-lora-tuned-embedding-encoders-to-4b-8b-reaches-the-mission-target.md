---
id: "H005"
aliases: ["H005"]
type: hypothesis
status: PROPOSED
created: "2026-11-14"
last_updated: "2026-11-14"
tags: [encoder, embeddings, lora, scaling]
---

# H005 — Scaling LoRA-tuned embedding encoders to 4B-8B reaches the mission target

> **Status**: PROPOSED
> **Created**: 2026-11-14
> **Last updated**: 2026-11-14

## Statement

Si entrenamos con LoRA encoders de embeddings mucho más grandes que gte-modernbert-base
(149M), en concreto `Qwen3-Embedding-4B` y `Qwen3-Embedding-8B`, con la misma cabeza lineal y
el mismo protocolo de validación que [[E003]], al menos uno de los dos llegará a **≥ 0,412**
de balanced accuracy con GroupKFold por `prompt_id` (media de las semillas de folds 1 y 2),
frente a 0,357 del mejor encoder actual ([[F003]]) y 0,307 del baseline en las mismas
semillas ([[F002]]). El control con etiquetas barajadas no superará 0,08.

## Mechanism

[[F003]] y [[F004]] muestran que un encoder de embeddings fine-tuneado capta señal de estilo
que el TF-IDF no ve, pero con 149M parámetros se queda en 0,357 y memoriza el train (pérdida
~0,001). Un encoder de 4–8B, preentrenado como LLM y luego como modelo de embeddings, debería
traer representaciones más ricas de léxico, sintaxis y estructura del discurso, justo los
rasgos que distinguen a los modelos según [[L001]]. LoRA entrena pocos parámetros (unos
pocos millones), lo que limita la capacidad de memorizar los ~120 pasajes por clase y obliga
a apoyarse en lo que el modelo ya sabe.

## Why This Might Generalize

La señal que buscamos viene del entrenamiento de cada modelo generador, no del tema del
prompt, y GroupKFold por `prompt_id` mide la generalización a prompts nuevos que exige el test
oculto. En [[L001]], la accuracy sube con el tamaño del encoder (el mejor es un encoder
basado en LLM, LLM2vec).

## Shortcut Risks

- **Aprender el tema en vez del estilo:** lo controla GroupKFold; ningún prompt aparece en
  train y validación a la vez.
- **Ajuste sobre los folds en que se mide:** lr y época de parada se eligen con la semilla 0;
  la decisión se toma con las semillas 1 y 2.
- **lr transferido (protocolo reducido):** para 4B y 8B el lr no se busca, se toma el elegido
  para el 0,6B. Si el lr óptimo cambia con el tamaño, los modelos grandes quedarían por debajo
  de su capacidad real. Un rechazo sería de "esta receta con este lr", no de la escala en
  general.
- **Afinidad de familia:** Qwen3-Embedding viene de Qwen3, de un laboratorio que está entre
  las clases (`qwen3.8-max`). Podría reconocer mejor el estilo de su propia familia. Se
  revisa el recall por clase y por laboratorio para detectarlo.
- **Cuantificación:** si el 8B no cabe en bf16 y se entrena con QLoRA de 4 bits, la
  comparación 4B frente a 8B mezcla tamaño y precisión. Se anota si ocurre.
- **Coste en Kaggle:** un modelo que gana en local pero no hace la inferencia de ~4.000
  pasajes en el hardware de Kaggle (sin internet, < 9 h) no sirve para la misión.

## Test Plan

- Experiment: [[E004]].
- Datos: `data/train.csv` completo; folds fijos de `kb/research/data/E002/folds.csv`.
- Modelos (todos con LoRA, Apache 2.0, commits fijados al crear E004):
  - `google/embeddinggemma-2`, solo la parte de texto (270M; pooling por media). Es otra
    familia del mismo tamaño que gte, no un modelo más grande.
  - `Qwen/Qwen3-Embedding-0.6B`, `-4B`, `-8B` (pooling por último token). Es la curva de
    escala: misma familia y receta, solo cambia el tamaño.
- Receta común: LoRA r = 16, α = 32, dropout 0,05, en todas las capas lineales; embedding
  normalizado (L2, reescalado a norma √d; añadido tras el smoke test de [[E004]]); cabeza
  lineal entrenada entera; prompt nativo de clasificación de cada modelo; máximo 512 tokens;
  bf16 (8B en bf16 con gradient checkpointing; QLoRA de 4 bits solo si no cabe en 24 GB).
- Protocolo reducido (decidido con el usuario, 2026-11-14):
  - embeddinggemma-2 y Qwen 0,6B: búsqueda de lr {1e-4, 3e-4} con la semilla 0, calendario
    de 6 épocas evaluado tras cada época; se elige (lr, k*).
  - Qwen 4B y 8B: usan el lr elegido para el 0,6B; con la semilla 0 solo se elige k* (1–6).
    El 8B se ejecuta siempre, sin depender del resultado del 4B.
  - Decisión: todos los modelos con su (lr, k*) en las semillas 1 y 2.
- Métrica principal: balanced accuracy OOF por modelo (media de las semillas 1 y 2).
- Secundarias: curva de escala 0,6B → 4B → 8B; embeddinggemma-2 frente a gte ([[F004]]);
  balanced accuracy por laboratorio; proporción de errores dentro del laboratorio; recall
  por clase y por tramo de longitud.
- Guardarraíles: control con etiquetas barajadas ≤ 0,08; 0 pasajes truncados; runs estables
  (la pérdida baja al menos un 50 %); inferencia de 4.000 pasajes medida en la 4090 y
  estimada para el hardware de Kaggle.
- Confirm if: `python -m name_that_llm.experiments.run_lora --phase summarize` -> Qwen 4B u 8B logra balanced accuracy ≥ 0,412 (media de las semillas 1 y 2)
- Reject if: `python -m name_that_llm.experiments.run_lora --phase summarize` -> ni el 4B ni el 8B superan 0,377 (gte + 0,02), con entrenamiento estable
- Inconclusive if: el mejor de 4B/8B queda entre 0,377 y 0,412 (la escala ayuda pero no
  basta), o hay runs inestables sin explicar.

## Evidence

- A favor: [[F003]], el encoder de 149M ya supera al baseline en los 10 folds, pero memoriza
  el train; hay margen para un modelo con mejor representación de partida.
- A favor: [[F004]], los encoders de embeddings (contrastivos) funcionan mejor que los MLM
  con nuestros datos, así que escalar dentro de esa familia es la apuesta natural.
- A favor: [[L001]], la accuracy crece con el tamaño del encoder y el mejor es un encoder
  basado en LLM.
- En contra: [[F003]], el 53 % de los errores son entre modelos del mismo laboratorio, y un
  modelo más grande no tiene por qué separar mejor a los hermanos.
- En contra: [[L001]], entre modelos de la misma familia (Qwen) se queda en 59,8 %, y con
  pocos datos la accuracy cae mucho.

## Conclusion

_Pendiente de E004._

## Links

- Mission: [[CHALLENGE]]
- Active State: [[ACTIVE]]
- Literature: [[L001]]
- Prior findings: [[F002]], [[F003]], [[F004]]
- Related hypothesis: [[H003]], [[H004]]
- Experiment: [[E004]]
