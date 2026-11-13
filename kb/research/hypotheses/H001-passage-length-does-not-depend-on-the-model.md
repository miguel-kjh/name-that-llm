---
id: "H001"
aliases: ["H001"]
type: hypothesis
status: CONFIRMED
created: "2026-11-13"
last_updated: "2026-11-13"
tags: [eda, length]
---

# H001 — Passage length does not depend on the model

> **Status**: CONFIRMED
> **Created**: 2026-11-13
> **Last updated**: 2026-11-13

## Statement

La longitud de los pasajes (en palabras) no depende del modelo que los generó: el modelo explica
menos del 1 % de la variación de longitud (ε² de Kruskal-Wallis < 0,01) y, dentro de cada prompt,
ningún modelo queda sistemáticamente arriba o abajo (W de Kendall del test de Friedman < 0,05).

## Mechanism

La misión dice que la longitud objetivo (40–250 palabras) se fija al azar por pasaje. Si los
modelos respetan esa longitud, la longitud es ruido del generador y no señal de estilo. Si algún
modelo se pasa o se queda corto de forma sistemática, la longitud lleva información del modelo.

## Why This Might Generalize

El test oculto usa prompts distintos, pero el mismo mecanismo de longitud aleatoria. Si la
longitud depende del modelo, esa dependencia viene del modelo y no del tema, así que debería
mantenerse en prompts nuevos. La comparación dentro de cada prompt (Friedman) quita el efecto
del tema.

## Shortcut Risks

- Con 150 pasajes por modelo, un p-valor bajo puede salir con efectos minúsculos. Por eso se
  decide por el tamaño del efecto (ε², W) y no por el p-valor.
- ε² no vale 0 aunque no haya efecto: si H₀ es cierta, E[H] = k − 1, así que
  E[ε²] = 17/2698 ≈ 0,0063. El umbral de 0,01 está poco por encima de ese nivel. El criterio de
  rechazo también exige p < 0,01, lo que con 17 grados de libertad equivale a ε² ≥ 0,0124. Si
  ε² está entre 0,01 y 0,0124, el resultado no es concluyente. Para W de Kendall, el valor
  esperado sin efecto es 1/149 ≈ 0,0067, muy por debajo de 0,05. (Lo descubrieron los tests
  sintéticos antes de mirar los datos reales.)
- Las palabras se cuentan separando por espacios, así que la puntuación o el markdown pegados
  pueden mover un poco el conteo según el modelo.
- Hay 41 pasajes fuera del rango 40–250 declarado (18 por debajo y 23 por encima, contando
  palabras por espacios). Si se concentran en un modelo, pueden mover el resultado aunque el
  resto de pasajes sea igual.

## Test Plan

- Experiment(s): [[E001]]
- Confirm if: `python -m name_that_llm.experiments.eda_length` -> ε²(words, model) < 0,01 y W(words) < 0,05
- Reject if: `python -m name_that_llm.experiments.eda_length` -> ε²(words, model) ≥ 0,01 o W(words) ≥ 0,05,
  con p < 0,01

## Evidence

- La misión dice que la longitud es aleatoria ([[CHALLENGE]]).
- Primer vistazo a `train.csv`: 2.699 pasajes, 150 prompts × 18 modelos (falta 1 de
  `claude-fable-5`), palabras de 36 a 274 con mediana 101.

## Conclusion

Confirmada para la longitud en palabras: ε² = 0,0087 < 0,01 y W = 0,009 < 0,05, con p ≈ 0,14
([[E001]]). Los caracteres dan lo mismo. Observación exploratoria fuera de este criterio: el
número de frases sí depende del modelo (ε² = 0,031). Ver [[F001]].

## Links

- Mission: [[CHALLENGE]]
- Active State: [[ACTIVE]]
- Experiment: [[E001]]
- Finding: [[F001]]
