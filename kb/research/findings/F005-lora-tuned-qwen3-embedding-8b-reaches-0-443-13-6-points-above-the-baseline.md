---
id: "F005"
aliases: ["F005"]
type: finding
hypothesis: "H005"
experiment: "E004"
impact: "HIGH"
created: "2026-11-15"
tags: [encoder, embeddings, lora, scaling]
---

# F005 — LoRA-tuned Qwen3-Embedding-8B reaches 0.443, 13.6 points above the baseline

> **Created**: 2026-11-15
> **Hypothesis**: [[H005]]
> **Experiment**: [[E004]]
> **Impact**: HIGH

## Finding

`Qwen3-Embedding-8B` entrenado con LoRA (r 16, lr 3e-4, 6 épocas) y una cabeza lineal llega a
**0,443** de balanced accuracy con GroupKFold por `prompt_id` (media de las semillas de folds
1 y 2). Supera el objetivo de la misión (≥ 0,412, baseline + 10 puntos) en cada uno de los 10
folds y confirma [[H005]]. La familia Qwen3-Embedding escala de forma regular: 0,357 (0,6B),
0,400 (4B), 0,443 (8B), unos +0,043 por cada salto de tamaño.

## Evidence

- [[E004]], fase de decisión (semillas 1 y 2, 10 folds por brazo):

  | Brazo | Media | Semilla 1 / 2 | Peor / mejor fold | Frente al baseline (victorias) |
  |---|---|---|---|---|
  | G embeddinggemma-2 | 0,301 | 0,308 / 0,294 | 0,276 / 0,322 | −0,006 (4/10) |
  | Q0 Qwen 0,6B | 0,357 | 0,350 / 0,365 | 0,324 / 0,389 | +0,050 (10/10) |
  | Q4 Qwen 4B | 0,400 | 0,406 / 0,395 | 0,361 / 0,439 | +0,093 (10/10) |
  | Q8 Qwen 8B | **0,443** | 0,445 / 0,441 | **0,415** / 0,469 | **+0,136 (10/10)** |

- Escala por fold: Q4 − Q0 = +0,043 (10/10 folds); Q8 − Q4 = +0,043 (8/10); Q8 − gte =
  +0,086 (10/10).
- La búsqueda no infló el resultado de Q8: 0,436 con la semilla 0 y 0,443 con semillas nuevas.
  La de Q4 sí bajó algo (0,423 → 0,400).
- Guardarraíles: control con etiquetas barajadas 0,058 (azar 0,056); 0 pasajes truncados;
  todos los runs de decisión estables.
- Balanced accuracy por laboratorio: Q8 0,679, Q0 0,609. El 58 % de los errores de Q8
  siguen siendo entre modelos del mismo laboratorio (Q0 55 %): mejora en todo, pero los
  hermanos `gpt-5.6-*` y `glm-5.3*` siguen siendo lo más difícil (recall 0,30–0,40).
- Sin señal de afinidad de familia: el recall de `qwen3.8-max` con Q8 (0,54) es el cuarto
  de 18, y ya era alto con embeddinggemma (0,38, segundo).
- La longitud manda: Q8 obtiene 0,366 en pasajes < 80 palabras y 0,521 en 150+.

## What Improved For Real

- Hay un primer modelo que cumple el criterio de validación local de la misión.
- Sabemos que escalar el encoder rinde y que la curva aún no se ha aplanado a 8B.
- embeddinggemma-2 con LoRA no aporta: se queda al nivel del baseline (5,6 puntos por debajo
  de gte con fine-tuning completo, 0/10 folds).

## Remaining Debt

- **Kaggle sin verificar.** El guardarraíl solo extrapola el tiempo, y en la 4090 es ruidoso
  (Q8: 166 s y 615 s para 4.000 pasajes en dos mediciones, ×10 ≈ 0,5–1,7 h; la VRAM queda
  llena al predecir y WSL desborda a memoria del sistema). La memoria no está resuelta: los pesos del 8B en 16 bits
  ocupan ~15 GB, lo mismo que una T4, y la T4/P100 no tiene bf16. Hace falta probar fp16
  (riesgo de desbordamiento), repartir el modelo en 2×T4 o cuantizar para inferencia.
- **Memoriza.** La pérdida de train de Q8 acaba en 0,001–0,01: LoRA no evita la memorización,
  al contrario de lo que suponía el mecanismo de [[H005]].
- lr transferido del 0,6B y sin búsqueda propia: el 8B podría rendir más con otro lr o con
  otro número de épocas (eligió 6, el máximo del calendario).
- Las semillas de GPU dan ~±0,02 por fold; la ventaja de Q8 sobre el umbral (+0,031 de
  media, peor fold +0,003) es mayor que eso en media, no en cada fold.
- Falta comprobar que la validación local va en la misma dirección que el leaderboard público.

## Next Move

- Construir el notebook de envío con Q8: entrenar el adaptador en local con todo el train,
  subirlo como dataset y medir en Kaggle la inferencia (memoria y tiempo) en T4/P100.
- Usar las OOF de Q8, Q4 y gte para el ensemble pendiente.

## Links

- Mission: [[CHALLENGE]]
- Active State: [[ACTIVE]]
- Parent Hypothesis: [[H005]]
- Parent Experiment: [[E004]]
- Prior findings: [[F002]], [[F003]], [[F004]]
