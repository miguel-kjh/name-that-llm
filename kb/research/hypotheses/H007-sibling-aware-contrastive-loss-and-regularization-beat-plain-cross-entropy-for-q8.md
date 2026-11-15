---
id: "H007"
aliases: ["H007"]
type: hypothesis
status: PROPOSED
created: "2026-11-15"
last_updated: "2026-11-15"
tags: [encoder, lora, contrastive, regularization, siblings]
---

# H007 — Sibling-aware contrastive loss and regularization beat plain cross-entropy for Q8

> **Status**: PROPOSED
> **Created**: 2026-11-15
> **Last updated**: 2026-11-15

## Statement

Si entrenamos Qwen3-Embedding-8B con LoRA añadiendo a la entropía cruzada una pérdida
contrastiva supervisada con negativos difíciles del mismo laboratorio, y regularización
contra la memorización (label smoothing y R-drop), la balanced accuracy con GroupKFold por
`prompt_id` (media de las semillas 1 y 2) llegará a **≥ 0,458**, frente a 0,443 de Q8 en
[[E004]], ganando en al menos 7 de los 10 folds. Como comprobación del mecanismo, la
proporción de errores dentro del laboratorio bajará del 58 % y el recall medio de los
hermanos más confundidos (gpt-5.6-luna/sol/terra, glm-5.3/glm-5.3-flash) subirá.

## Mechanism

- **Los hermanos están casi juntos:** en validación, el coseno medio entre pasajes del mismo
  modelo (0,24) apenas supera al de modelos distintos del mismo laboratorio (0,20), frente a
  0,04 entre laboratorios ([[E004]], diagnóstico). La entropía cruzada no pide margen entre
  clases una vez el train está bien clasificado.
- **Memorización:** Q8 llega a pérdida de train 0,001 en 6 épocas. A partir de ahí el
  gradiente casi desaparece y el modelo deja de aprender rasgos que separen hermanos.
- Una pérdida contrastiva que acerca los pasajes del mismo modelo y aleja en especial a sus
  hermanos (negativos del mismo laboratorio con más peso) pide ese margen de forma explícita.
  Label smoothing y R-drop mantienen el gradiente vivo y penalizan la confianza extrema.

## Why This Might Generalize

La pérdida contrastiva actúa entre pasajes de prompts distintos dentro del lote, así que el
modelo tiene que juntar pasajes de un mismo autor que hablan de temas diferentes: es el
mismo tipo de generalización que exige el test.

## Shortcut Risks

- **Ajuste sobre los folds en que se mide:** la variante y sus hiperparámetros se eligen con
  la semilla de folds 0; la decisión se toma con las semillas 1 y 2.
- **Ruido de GPU:** ~±0,02 por fold entre ejecuciones idénticas ([[E004]]). Por eso la
  decisión usa la media de 10 folds y exige ganar en la mayoría de folds, no una diferencia
  puntual.
- **Lotes pequeños:** con batch efectivo 16 y micro-batch 4 en Q8, un lote rara vez trae dos
  pasajes del mismo modelo. Hace falta un muestreador que garantice pares positivos y
  hermanos en cada lote (p. ej. 2 pasajes × 8 modelos, con hermanos del mismo laboratorio),
  o un banco de memoria de embeddings. Se comprueba en el preflight que cada lote tiene
  positivos.
- **Dos cambios a la vez:** la contrastiva y la regularización pueden ayudar por separado o
  estorbarse. La búsqueda con la semilla 0 incluye cada una sola y las dos juntas; la
  decisión se toma con la combinación elegida, y se informa de qué parte aporta.
- **El muestreador cambia el número de pasos por época:** se fija el número de pasos para
  igualar el cómputo con [[E004]].

## Test Plan

- Experiment: por crear.
- Datos y folds como [[E004]]; misma receta de base de Q8.
- Variantes, filtradas primero con Q0 en la semilla 0:
  - CE + contrastiva supervisada (temperatura y peso en una rejilla pequeña; negativos del
    mismo laboratorio con peso extra);
  - CE + label smoothing 0,1 + R-drop;
  - las dos juntas.
- La mejor variante con Q0 se ejecuta con Q8 en la semilla 0 (elige época) y se decide con
  las semillas 1 y 2, comparando por fold con Q8 de [[E004]].
- Métrica principal: balanced accuracy OOF (media de las semillas 1 y 2).
- Secundarias: proporción de errores dentro del laboratorio; recall de gpt-5.6-luna/sol/terra
  (0,40/0,35/0,36 en [[E004]]) y glm-5.3/flash (0,30/0,33); coseno medio mismo modelo frente
  a hermanos en validación; pérdida final de train (memorización).
- Guardarraíles: los de [[E004]].
- Confirm if: la variante elegida logra una balanced accuracy media (semillas 1 y 2) ≥ 0,458 (Q8 de [[E004]] + 0,015) y gana a Q8 en ≥ 7 de los 10 folds
- Reject if: la variante elegida no supera a Q8 de [[E004]] (diferencia media ≤ 0) con entrenamiento estable
- Inconclusive if: mejora entre 0 y 0,015, o gana en menos de 7 folds, o hay runs inestables
  sin explicar.

## Evidence

- A favor: diagnóstico de [[E004]]: los hermanos están a 0,04 de coseno de los pasajes del
  mismo modelo; el margen es mínimo.
- A favor: [[F005]]: el 58 % de los errores de Q8 son dentro del laboratorio y el train está
  memorizado.
- En contra: con ~120 pasajes por modelo en train, puede que no haya señal suficiente para
  separar a los hermanos, sea cual sea la pérdida.

## Conclusion

_Pendiente._

## Links

- Mission: [[CHALLENGE]]
- Active State: [[ACTIVE]]
- Prior findings: [[F005]]
- Prior experiment: [[E004]]
- Related hypothesis: [[H005]], [[H006]], [[H008]]
