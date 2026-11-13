---
id: "H004"
aliases: ["H004"]
type: hypothesis
status: PROPOSED
created: "2026-11-13"
last_updated: "2026-11-13"
tags: [encoder, embeddings, fine-tuning]
---

# H004 — MLM encoder beats its contrastive embedding version at equal backbone

> **Status**: PROPOSED
> **Created**: 2026-11-13
> **Last updated**: 2026-11-13

## Statement

Con el mismo backbone (ModernBERT-base, 149M), la misma receta de fine-tuning y los mismos
folds, el encoder preentrenado solo con MLM (`ModernBERT-base`, brazo A) logrará una balanced
accuracy igual o mayor que su versión con entrenamiento contrastivo de embeddings
(`gte-modernbert-base`, brazo B). Esperamos A ≥ B, y A > B por al menos 0,02 si el
entrenamiento contrastivo borra parte del estilo.

## Mechanism

El entrenamiento contrastivo de un modelo de embeddings acerca textos con el mismo significado
aunque estén escritos de forma distinta (paráfrasis). Es decir, le enseña a ignorar el estilo,
que es justo nuestra señal. El preentrenamiento MLM, en cambio, obliga a conservar la
información de superficie (léxico, puntuación, formato) para predecir los tokens
enmascarados.

Explicación alternativa a favor de B: su espacio ya está organizado por textos y con
solo unos 120 ejemplos por clase en cada fold podría converger mejor. Por eso el resultado
no es obvio.

## Why This Might Generalize

La comparación aísla un único factor (el entrenamiento contrastivo) con arquitectura, tamaño
y receta fijos. Si el efecto aparece, debería repetirse con otros pares MLM/embeddings del
mismo backbone, y orienta qué familia de modelos escalar después.

## Shortcut Risks

- **Ajuste desigual:** cada brazo elige su propio learning rate de la misma rejilla, para no
  favorecer a uno con un lr pensado para el otro.
- **Diferencias de tokenizador o longitud máxima:** comprobar que los dos usan el mismo
  tokenizador y el mismo truncado.
- **Ruido entre folds:** con 2.699 pasajes, diferencias de 1–2 puntos pueden ser ruido. Por
  eso se pide además ganar en la mayoría de folds.
- **Inferencia de embeddings:** fine-tuning completo en los dos brazos. Congelar B sería
  otro método y no sirve para esta comparación.

## Test Plan

- Experiment(s): [[E003]] (compartido con [[H003]]).
- Brazos, datos, folds, rejilla de lr, semillas, receta y guardarraíles: los de [[H003]].
- Métrica principal: diferencia de balanced accuracy A − B (media de las semillas 1 y 2) y
  número de folds (de 10) en los que gana cada brazo.
- Secundaria: diferencia en la proporción de errores dentro del laboratorio.
- Condición previa de validez: los dos brazos entrenan de forma estable y superan al baseline
  (0,312). Si alguno no, la comparación es inconclusa.
- Confirm if: `python -m name_that_llm.experiments.run_encoder` -> A − B ≥ 0,02 y A gana en ≥ 8 de los 10 folds
- Reject if: `python -m name_that_llm.experiments.run_encoder` -> B − A ≥ 0,02 y B gana en ≥ 8 de los 10 folds
- Inconclusive if: |A − B| < 0,02, o ninguno gana en ≥ 8 de los 10 folds. En ese caso se
  considera empate y se elige el más simple (A) para seguir.

## Evidence

- A favor: [[L001]], la señal está en léxico y formato (barajar letras la hunde, solo
  markdown da 73–78 %); un entrenamiento que hace invariantes las paráfrasis debería perder
  parte de ella.
- En contra: [[L001]], el parafraseo conserva más del 90 % de la accuracy (Tabla 10), así que
  parte de la señal es semántica y sobreviviría al entrenamiento contrastivo.
- En contra: el mejor modelo de [[L001]] (LLM2vec) es un encoder con entrenamiento
  contrastivo, aunque el paper no aísla ese factor del tamaño.

## Conclusion

Pendiente de E003.

## Links

- Mission: [[CHALLENGE]]
- Active State: [[ACTIVE]]
- Literature: [[L001]]
- Related hypothesis: [[H003]]
- Experiment: [[E003]]
