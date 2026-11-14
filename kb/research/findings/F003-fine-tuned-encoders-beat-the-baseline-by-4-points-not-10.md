---
id: "F003"
aliases: ["F003"]
type: finding
hypothesis: "H003"
experiment: "E003"
impact: "HIGH"
created: "2026-11-14"
tags: [encoder, fine-tuning, baseline]
---

# F003 — Fine-tuned encoders beat the baseline by 4 points, not 10

> **Created**: 2026-11-14
> **Hypothesis**: [[H003]]
> **Experiment**: [[E003]]
> **Impact**: HIGH

## Finding

Fine-tunear un encoder de 149M con la receta de [[L001]] (mean pooling + cabeza lineal, todo
entrenado) **no** llega al objetivo de la misión. El mejor brazo, `gte-modernbert-base`, logra
0,357 de balanced accuracy con GroupKFold por `prompt_id` (media de las semillas 1 y 2), frente
a 0,307 del baseline en las mismas semillas: +0,05, lejos de los +0,10 necesarios (≥ 0,412) y
por debajo del umbral de rechazo (0,362). [[H003]] queda **rechazada** con un método válido.
La mejora es pequeña pero real: B gana al baseline en los 10 folds.

## Evidence

- [[E003]], fase de decisión: 2 brazos × 2 semillas × 5 folds, configuración elegida en la
  semilla 0 (A: lr 3e-5, 6 épocas; B: lr 5e-5, 8 épocas).
- B: 0,358 / 0,355 por semilla; folds 0,335–0,380; delta frente al baseline +0,030 a +0,072
  en los 10 folds.
- A (`ModernBERT-base`): 0,317 / 0,319; delta −0,020 a +0,037, positivo en 8 de 10 folds.
  Prácticamente igual que el baseline.
- Guardarraíles superados: control con etiquetas barajadas 0,051 (azar 0,056), 0 truncados,
  todos los runs estables, inferencia de 4.000 pasajes en 3,5–4,5 s en la RTX 4090.
- Errores dentro del laboratorio: 0,534 (B) y 0,527 (A), frente a 0,518 del baseline. El
  encoder no resuelve la confusión entre hermanos; mejora algo la separación por laboratorio
  (balanced accuracy de laboratorio 0,593 frente a 0,539).
- Por longitud (B): 0,291 (< 80 palabras), 0,372 (80–149), 0,413 (150+). La mejora sobre el
  baseline (0,256 / 0,313 / 0,375) es parecida en todos los tramos.
- Recall por clase (B): sigue habiendo clases casi perdidas, claude-fable-5 0,16, glm-5.3-flash
  0,19, glm-5.3 0,21, gpt-5.6-terra 0,22, gpt-5.6-sol 0,22.

## What Improved For Real

- Hay un segundo modelo, distinto del TF-IDF, que supera al baseline de forma consistente y
  cuyas probabilidades OOF están guardadas (`kb/research/data/E003/oof/`) para probar un
  ensemble sin reentrenar.
- Sabemos que con ~150 pasajes por clase y esta receta, un encoder base no basta: el cuello de
  botella no es solo el tipo de representación.

## Remaining Debt

- Solo se probó una receta (base 149M, mean pooling, 6–8 épocas). No se descartan modelos más
  grandes, más datos ni otras cabezas; el rechazo es de esta receta, no de los encoders en
  general.
- El lr de B está en el borde superior de la rejilla (5e-5); un lr mayor podría dar algo más,
  pero no es plausible que cierre 5,5 puntos.
- Con lr 3e-5–5e-5 los dos brazos memorizan el train (pérdida ~0,001): sobreajuste fuerte con
  tan pocos datos por clase.
- Sigue sin comprobarse la correlación entre la validación local y el leaderboard.

## Next Move

- Probar un ensemble TF-IDF + encoder con las probabilidades OOF ya guardadas: barato y
  responde si las dos señales son complementarias.
- Después, decidir entre escalar el encoder (modelo grande, más épocas con regularización) o
  atacar la confusión dentro del laboratorio con rasgos de estilo ([[F001]], [[F002]]).

## Links

- Mission: [[CHALLENGE]]
- Active State: [[ACTIVE]]
- Parent Hypothesis: [[H003]]
- Parent Experiment: [[E003]]
- Related finding: [[F004]]
- Comparator: [[F002]]
