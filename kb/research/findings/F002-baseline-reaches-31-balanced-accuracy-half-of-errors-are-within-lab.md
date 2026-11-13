---
id: "F002"
aliases: ["F002"]
type: finding
hypothesis: "H002"
experiment: "E002"
impact: "HIGH"
created: "2026-11-13"
tags: [baseline, validation]
---

# F002 — Baseline reaches 31% balanced accuracy, half of errors are within-lab

> **Created**: 2026-11-13
> **Hypothesis**: [[H002]]
> **Experiment**: [[E002]]
> **Impact**: HIGH

## Finding

El baseline de la misión (TF-IDF de caracteres + regresión logística) logra 0,312 de balanced
accuracy con GroupKFold por `prompt_id` (azar: 0,056). El 52 % de sus errores son entre
modelos del mismo laboratorio (azar: 18 %), y el otro 48 % entre laboratorios.

## Evidence

- [[E002]]: 5 folds × 3 semillas, 2.699 pasajes, unos 1.860 errores por semilla.
- Balanced accuracy 0,312 (semillas 0,304–0,322; folds 0,312 ± 0,015).
- Errores dentro del laboratorio 0,518 (semillas 0,513–0,525).
- Laboratorio: 0,539 de balanced accuracy.
- Control con etiquetas barajadas: 0,056, igual que el azar. No hay fuga de información.
- Por longitud: 0,256 (< 80 palabras), 0,313 (80–149), 0,375 (150+).

## What Improved For Real

- Hay un protocolo de evaluación comprobado (folds fijos en `folds.csv`, control de fuga) y una
  cifra de referencia: el objetivo de la misión es ≥ 0,412 con este protocolo.
- Sabemos dónde están los errores: la mitad entre hermanos (sobre todo glm-5.3 frente a su flash,
  gemini 3.6/3.7/3.8-flash y gpt-5.6 luna/sol/terra) y la otra mitad entre laboratorios.

## Remaining Debt

- No sabemos aún si esta validación local se corresponde con el leaderboard de Kaggle.
- La configuración del baseline es fija y sin ajustar; un baseline ajustado podría estar algo
  más alto, y conviene tenerlo en cuenta al medir la mejora de 10 puntos.
- La proporción dentro del laboratorio supera el umbral de 0,50 por poco.

## Next Move

- Notebook de envío a Kaggle con este baseline para comprobar la correlación con el leaderboard.
- Después, hipótesis de mejora que ataquen las dos fuentes de error: rasgos de estilo
  normalizados por longitud (incluida la longitud de frase, [[F001]]) y un enfoque en dos etapas.

## Links

- Mission: [[CHALLENGE]]
- Active State: [[ACTIVE]]
- Parent Hypothesis: [[H002]]
- Parent Experiment: [[E002]]
