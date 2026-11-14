---
id: "F004"
aliases: ["F004"]
type: finding
hypothesis: "H003"
experiment: "E003"
impact: "MEDIUM"
created: "2026-11-14"
tags: [encoder, embeddings, fine-tuning]
---

# F004 — Contrastive gte-modernbert beats MLM ModernBERT in 10 of 10 folds

> **Created**: 2026-11-14
> **Hypothesis**: [[H003]] (experimento principal); decide [[H004]], la comparación secundaria de [[E003]]
> **Experiment**: [[E003]]
> **Impact**: MEDIUM

## Finding

Con el mismo backbone (ModernBERT-base, 149M, mismo tokenizador), la versión con entrenamiento
contrastivo (`gte-modernbert-base`) supera a la versión MLM, al revés de lo que proponía
[[H004]]. B − A = +0,039 de balanced accuracy y B gana en los 10 folds de las semillas 1 y 2.
[[H004]] queda **rechazada**: el entrenamiento contrastivo no borra la señal de estilo que
necesitamos; con nuestros datos, la ayuda.

## Evidence

- [[E003]], fase de decisión: A 0,318 (0,317 / 0,319), B 0,357 (0,358 / 0,355).
- Victorias por fold: B 10, A 0. Diferencia mínima por fold +0,003, máxima +0,067.
- B mejora el recall en 14 de 18 clases; las mayores subidas son claude-opus-5 (+0,14),
  gpt-5.5 (+0,13) y gpt-5.6-luna (+0,12). A gana en gemini-3.1-pro-preview (0,68 frente a
  0,60) y gpt-5.6-terra.
- La curva de búsqueda de B es más estable: meseta 0,35–0,36 con lr 5e-5 de la época 5 a la
  10; la de A tiene un pico en la época 6 (0,349) que no se mantiene (0,33 en semillas nuevas).

## What Improved For Real

- Sabemos qué familia escalar: encoders de embeddings (contrastivos), no MLM puros.

## Remaining Debt

- Cada brazo usa su mejor configuración de la búsqueda (A lr 3e-5 / 6 épocas, B lr 5e-5 /
  8 épocas), así que la comparación incluye la sensibilidad a la configuración. Aun así, ninguna
  configuración de A en la búsqueda pasó de 0,336 de forma sostenida.
- Un solo par de modelos; no sabemos si se repite con otros backbones o tamaños.
- La causa no está aislada: puede ser el entrenamiento contrastivo en sí, el mayor volumen de
  datos de gte o una inicialización que converge mejor con pocos datos.

## Next Move

- Si se escala el encoder, empezar por modelos de embeddings (p. ej. gte o similares más
  grandes) en lugar de MLM.

## Links

- Mission: [[CHALLENGE]]
- Active State: [[ACTIVE]]
- Parent Hypothesis: [[H003]]
- Decided hypothesis: [[H004]]
- Parent Experiment: [[E003]]
- Related finding: [[F003]]
