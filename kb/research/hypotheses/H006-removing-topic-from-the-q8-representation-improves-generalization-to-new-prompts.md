---
id: "H006"
aliases: ["H006"]
type: hypothesis
status: PROPOSED
created: "2026-11-15"
last_updated: "2026-11-15"
tags: [encoder, embeddings, lora, topic, layers]
---

# H006 — Removing topic from the Q8 representation improves generalization to new prompts

> **Status**: PROPOSED
> **Created**: 2026-11-15
> **Last updated**: 2026-11-15

## Statement

Si a Qwen3-Embedding-8B con LoRA ([[F005]]) le quitamos información del tema del prompt, ya
sea con pooling de capas intermedias o con una cabeza adversaria que intenta adivinar el
prompt (inversión de gradiente), la balanced accuracy con GroupKFold por `prompt_id` (media
de las semillas 1 y 2) llegará a **≥ 0,458**, frente a 0,443 de Q8 en [[E004]], ganando en
al menos 7 de los 10 folds. Como comprobación del mecanismo, el AUC de agrupación por prompt
en validación bajará de 0,66 a ≤ 0,58.

## Mechanism

El diagnóstico de embeddings de [[E004]] muestra que el embedding base de la última capa
agrupa casi perfectamente por prompt (AUC 0,99) y casi nada por modelo (0,53). LoRA deshace
esos grupos solo en parte: en validación siguen agrupando por prompt (AUC 0,66). Si una parte
de la capacidad del clasificador se gasta en tema, que no se repite entre train y test, esa
parte no generaliza.

- **Capas intermedias:** en modelos de lenguaje, las capas intermedias suelen conservar más
  rasgos de forma (sintaxis, puntuación, estilo) y la última se especializa en semántica,
  sobre todo en un modelo entrenado para similitud. Concatenar o promediar con pesos
  aprendidos el pooling de varias capas da a la cabeza acceso directo a esos rasgos.
- **Adversario de prompt:** una segunda cabeza predice el prompt del pasaje y su gradiente
  llega invertido al encoder, que aprende a no codificar el tema. En train cada prompt tiene
  un pasaje por modelo, así que el prompt es independiente de la etiqueta y quitarlo no
  debería borrar señal de autor.

## Why This Might Generalize

El test oculto tiene prompts que no están en train. Una representación sin tema no puede
apoyarse en coincidencias temáticas entre train y validación; GroupKFold por `prompt_id`
mide exactamente esa transferencia.

## Shortcut Risks

- **Ajuste sobre los folds en que se mide:** la variante y sus hiperparámetros se eligen con
  la semilla de folds 0; la decisión se toma con las semillas 1 y 2.
- **Ruido de GPU:** ~±0,02 por fold entre ejecuciones idénticas ([[E004]]). Por eso la
  decisión usa la media de 10 folds y exige ganar en la mayoría de folds, no una diferencia
  puntual.
- **El adversario puede destruir señal útil:** si el estilo de un modelo depende del tipo de
  prompt (p. ej. listas en prompts técnicos), quitar el tema también quita estilo. Se vigila
  con la accuracy y el recall por clase, no solo con el AUC de prompt.
- **Inestabilidad del entrenamiento adversario:** la inversión de gradiente es sensible a su
  peso. Se fija una rejilla pequeña con la semilla 0 y se registra la pérdida de las dos
  cabezas.
- **Capas intermedias y coste:** guardar varias capas no cambia el coste de cómputo, pero sí
  la memoria de activaciones en Q8 (micro-batch 4). Se mide en el preflight.

## Test Plan

- Experiment: por crear.
- Datos y folds como [[E004]]; misma receta de Q8 (LoRA r 16, lr 3e-4, 6 épocas, bf16).
- Variantes, filtradas primero con Q0 (barato) en la semilla 0:
  - capas: pooling de un conjunto fijo de capas (p. ej. 1/2, 3/4 y la última), combinadas
    con pesos aprendidos;
  - adversario: cabeza de prompt con inversión de gradiente, peso λ en una rejilla pequeña;
  - ambas.
- La mejor variante con Q0 se ejecuta con Q8 en la semilla 0 (elige época) y se decide con
  las semillas 1 y 2, comparando por fold con las ejecuciones de Q8 de [[E004]].
- Métrica principal: balanced accuracy OOF (media de las semillas 1 y 2).
- Secundarias: AUC de agrupación por prompt y por modelo en validación (mismo diagnóstico de
  [[E004]]), recall por clase, balanced accuracy por laboratorio.
- Guardarraíles: los de [[E004]] (control con etiquetas barajadas ≤ 0,08, 0 truncados, runs
  estables).
- Confirm if: la variante elegida logra una balanced accuracy media (semillas 1 y 2) ≥ 0,458 (Q8 de [[E004]] + 0,015) y gana a Q8 en ≥ 7 de los 10 folds
- Reject if: la variante elegida no supera a Q8 de [[E004]] (diferencia media ≤ 0) con entrenamiento estable
- Inconclusive if: mejora entre 0 y 0,015, o gana en menos de 7 folds, o hay runs inestables
  sin explicar.

## Evidence

- A favor: diagnóstico de embeddings de [[E004]]: la última capa base es un mapa de temas, y
  tras LoRA queda tema residual en validación (AUC por prompt 0,66).
- A favor: [[F005]]: Q8 memoriza el train (pérdida 0,001); quitar un atajo disponible puede
  reducir esa memorización.
- En contra: LoRA ya reduce el tema de 0,99 a 0,66 sin ayuda; puede que lo que queda sea
  inseparable del estilo.

## Conclusion

_Pendiente._

## Links

- Mission: [[CHALLENGE]]
- Active State: [[ACTIVE]]
- Prior findings: [[F005]]
- Prior experiment: [[E004]]
- Related hypothesis: [[H005]], [[H007]], [[H008]]
