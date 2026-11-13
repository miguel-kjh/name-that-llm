---
id: "H002"
aliases: ["H002"]
type: hypothesis
status: CONFIRMED
created: "2026-11-13"
last_updated: "2026-11-13"
tags: [baseline, validation]
---

# H002 — Baseline errors concentrate within each lab

> **Status**: CONFIRMED
> **Created**: 2026-11-13
> **Last updated**: 2026-11-13

## Statement

Con el baseline de la misión (TF-IDF de n-gramas de caracteres + regresión logística) evaluado
con GroupKFold por `prompt_id` (5 folds, 3 semillas de reparto), más del 50 % de los errores
son confusiones entre modelos del mismo laboratorio, frente al 17,6 % esperado si las
confusiones fueran al azar.

## Mechanism

Los modelos de un mismo laboratorio comparten datos de entrenamiento, ajuste fino y guías de
estilo, así que sus hábitos de superficie (puntuación, tipografía, muletillas) se parecen.
Los n-gramas de caracteres capturan sobre todo esos hábitos de superficie, por lo que deberían
separar bien laboratorios y mal modelos hermanos.

## Why This Might Generalize

El parecido entre hermanos viene del proceso de entrenamiento, no del tema de los prompts. Al
validar con prompts no vistos (GroupKFold), la estructura de errores medida debería ser la
misma en el test oculto.

## Shortcut Risks

- Si el baseline acierta casi siempre, quedan pocos errores y la proporción es ruidosa. Se
  informa el número de errores junto a la proporción.
- Una fuga de información (prompt repetido entre folds, rasgos que identifican el `id`) inflaría
  el accuracy. Lo controlan los tests de folds y el control con etiquetas barajadas.
- La proporción al azar (17,6 %) depende del tamaño de cada laboratorio: Anthropic 5, OpenAI 5,
  Google 4, ZAI 2, Moonshot AI 1, Qwen 1. Es 54/306 = Σ s(s−1) / (18·17).

## Test Plan

- Experiment(s): [[E002]]
- Baseline: configuración fija sin ajustar, documentada en el `manifest.json` de E002.
- Métrica principal: proporción de errores dentro del mismo laboratorio (OOF, media de 3
  semillas). Secundarias: balanced accuracy por modelo y por laboratorio, recall por clase,
  accuracy por tramo de longitud.
- Guardarraíl: el control con etiquetas barajadas debe dar un balanced accuracy ≤ 8 %
  (azar: 5,6 %). Si no, el experimento no es válido.
- Confirm if: `uv run python scripts/run_baseline.py` -> proporción dentro del laboratorio > 0,50
- Reject if: `uv run python scripts/run_baseline.py` -> proporción dentro del laboratorio < 0,35
- Inconclusive if: entre 0,35 y 0,50, o menos de 100 errores en total.

## Evidence

- La misión avisa de modelos muy parecidos dentro de un laboratorio (gpt-5.6-luna/sol/terra).
- [[F001]]: la longitud total no identifica al modelo, así que el baseline depende del estilo.

## Conclusion

Confirmada por la regla fijada: el 51,8 % de los errores son dentro del laboratorio (semillas
0,513–0,525, unos 1.860 errores), frente al 17,6 % al azar. El margen sobre el umbral de 0,50
es estrecho: casi la mitad de los errores son entre laboratorios. Ver [[E002]] y [[F002]].

## Links

- Mission: [[CHALLENGE]]
- Active State: [[ACTIVE]]
- Prior finding: [[F001]]
- Experiment: [[E002]]
- Finding: [[F002]]
