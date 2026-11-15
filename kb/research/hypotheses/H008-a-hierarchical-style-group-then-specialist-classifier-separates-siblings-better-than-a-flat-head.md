---
id: "H008"
aliases: ["H008"]
type: hypothesis
status: PROPOSED
created: "2026-11-15"
last_updated: "2026-11-15"
tags: [encoder, lora, hierarchical, siblings]
---

# H008 — A hierarchical style-group then specialist classifier separates siblings better than a flat head

> **Status**: PROPOSED
> **Created**: 2026-11-15
> **Last updated**: 2026-11-15

## Statement

Un clasificador en dos niveles, un enrutador que asigna cada pasaje a un grupo de estilo y un
especialista por grupo entrenado solo con los modelos de ese grupo, combinado como
p(modelo) = p(grupo) · p(modelo | grupo), llegará a una balanced accuracy con GroupKFold por
`prompt_id` (media de las semillas 1 y 2) de **≥ 0,458**, frente a 0,443 de Q8 plano en
[[E004]], ganando en al menos 7 de los 10 folds. Como comprobación del mecanismo, la
accuracy dentro de los grupos subirá en los pares más confundidos: ZAI frente a Anthropic,
Qwen y Moonshot frente a OpenAI, y gpt-5.6-luna/sol/terra.

## Mechanism

Las predicciones de Q8 tienen una estructura jerárquica ([[E004]], análisis por
laboratorio): tres regiones de estilo bien separadas (Google; Anthropic + ZAI; OpenAI + Qwen
+ Moonshot), fronteras difusas entre laboratorios dentro de una región (39 % de ZAI se
predice como Anthropic; 29 % de Qwen y 26 % de Moonshot como OpenAI) y aún más difusas entre
hermanos. Un clasificador plano reparte su capacidad entre 18 clases, la mayoría fáciles de
distinguir entre sí. Un especialista que solo ve los 4–7 modelos de su grupo dedica todo el
fine-tuning a las diferencias finas de ese grupo, y su LoRA no tiene que servir también para
separar a Google de OpenAI.

## Why This Might Generalize

Los grupos reflejan parecidos de estilo entre laboratorios, que vienen del entrenamiento de
los modelos generadores y no de los prompts. GroupKFold por `prompt_id` mide si el
especialista generaliza a prompts nuevos.

## Shortcut Risks

- **Ajuste sobre los folds en que se mide:** la variante y sus hiperparámetros se eligen con
  la semilla de folds 0; la decisión se toma con las semillas 1 y 2.
- **Ruido de GPU:** ~±0,02 por fold entre ejecuciones idénticas ([[E004]]). Por eso la
  decisión usa la media de 10 folds y exige ganar en la mayoría de folds, no una diferencia
  puntual.
- **Grupos elegidos con los datos de decisión:** los tres grupos salen del análisis de las
  semillas 1 y 2. Para no usar los folds de decisión en el diseño, los grupos se fijan con la
  matriz de confusión de la búsqueda de Q8 (semilla 0) antes de entrenar nada. Si salen
  distintos de los de arriba, mandan los de la semilla 0.
- **Errores del enrutador:** un pasaje mal enrutado ya no puede acertar con una combinación
  dura. Se usa la combinación suave (producto de probabilidades sobre todos los grupos) y se
  informa de la accuracy del enrutador por grupo.
- **Menos datos por especialista:** cada uno ve solo 4–7 de 18 modelos (~500–850 pasajes de
  train por fold). Puede memorizar aún más rápido; la época se elige con la semilla 0.
- **Coste en Kaggle:** el enrutador más tres especialistas son cuatro adaptadores sobre el
  mismo modelo base. En inferencia se carga la base una vez, pero el tiempo se multiplica por
  cuatro. Se mide y se compara con el límite de 9 h.

## Test Plan

- Experiment: por crear.
- Datos y folds como [[E004]]; misma receta de Q8.
- Grupos fijados con la confusión de la búsqueda de Q8 en la semilla 0 de [[E004]].
- Enrutador: Q8 plano de [[E004]] (p(grupo) = suma de sus probabilidades por grupo), sin
  reentrenar. Variante opcional: enrutador entrenado con 3 clases.
- Especialistas: un LoRA de Q8 por grupo de más de un modelo, entrenado solo con los
  pasajes de su grupo; época elegida con la semilla 0. Primero se filtra con Q0.
- Decisión con las semillas 1 y 2, comparando por fold con Q8 plano de [[E004]].
- Métrica principal: balanced accuracy OOF (media de las semillas 1 y 2) de la combinación
  suave.
- Secundarias: accuracy del enrutador por grupo; balanced accuracy de cada especialista
  dentro de su grupo frente a Q8 plano restringido a ese grupo; recall de los pares objetivo;
  proporción de errores dentro del laboratorio.
- Guardarraíles: los de [[E004]], más el tiempo de inferencia de los cuatro adaptadores.
- Confirm if: la variante elegida logra una balanced accuracy media (semillas 1 y 2) ≥ 0,458 (Q8 de [[E004]] + 0,015) y gana a Q8 en ≥ 7 de los 10 folds
- Reject if: la variante elegida no supera a Q8 de [[E004]] (diferencia media ≤ 0) con entrenamiento estable
- Inconclusive if: mejora entre 0 y 0,015, o gana en menos de 7 folds, o hay runs inestables
  sin explicar.

## Evidence

- A favor: [[F005]] y el análisis por laboratorio de [[E004]]: Google se acierta al 96 %, y
  los errores se concentran dentro de dos regiones de estilo.
- A favor: diagnóstico de embeddings de [[E004]]: el t-SNE de validación muestra las mismas
  tres regiones.
- En contra: un especialista con menos datos puede generalizar peor que el modelo plano, que
  aprende rasgos compartidos de todos los modelos.

## Conclusion

_Pendiente._

## Links

- Mission: [[CHALLENGE]]
- Active State: [[ACTIVE]]
- Prior findings: [[F005]]
- Prior experiment: [[E004]]
- Related hypothesis: [[H005]], [[H006]], [[H007]]
