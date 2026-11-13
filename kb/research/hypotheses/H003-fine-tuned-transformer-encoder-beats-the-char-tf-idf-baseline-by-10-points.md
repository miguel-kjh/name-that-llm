---
id: "H003"
aliases: ["H003"]
type: hypothesis
status: PROPOSED
created: "2026-11-13"
last_updated: "2026-11-13"
tags: [encoder, fine-tuning]
---

# H003 — Fine-tuned transformer encoder beats the char TF-IDF baseline by 10 points

> **Status**: PROPOSED
> **Created**: 2026-11-13
> **Last updated**: 2026-11-13

## Statement

Si fine-tuneamos un encoder transformer con la receta de [[L001]] (encoder + mean pooling +
cabeza lineal, todo entrenado) sobre los 2.699 pasajes de train, la balanced accuracy con
GroupKFold por `prompt_id` (5 folds × 3 semillas, los mismos folds de [[E002]]) pasará de
0,312 (baseline de [[F002]]) a **≥ 0,412** en al menos uno de los dos brazos
(`ModernBERT-base` o `gte-modernbert-base`), sin que el control con etiquetas barajadas supere
0,08.

## Mechanism

Los n-gramas de caracteres solo ven hábitos locales de superficie (puntuación, fragmentos de
palabras). Un encoder preentrenado y fine-tuneado ve el pasaje entero en contexto: qué palabras
elige, cómo abre el texto, cómo lo estructura y qué contenido prioriza. Según [[L001]], ahí
está la mayor parte de la señal: barajar las palabras apenas baja la accuracy, barajar las
letras la hunde, y un fine-tuning supera a TF-IDF de palabras + LR en el mismo texto (95,1 %
frente a 85,5 %, §3.1).

## Why This Might Generalize

Los rasgos que busca el encoder (léxico, aperturas, estructura) vienen del entrenamiento de
cada modelo, no del tema del prompt. La validación por grupos de `prompt_id` mide justo la
generalización a prompts nuevos que exige el test oculto.

## Shortcut Risks

- **Aprender el tema en vez del estilo:** lo controla GroupKFold; ningún prompt aparece en
  train y validación a la vez.
- **Pocos datos:** unos 120 ejemplos por clase en cada fold de train. El fine-tuning puede
  sobreajustar o no converger. Se reporta la curva de pérdida y la varianza entre semillas.
- **Ajustar sobre los folds en los que se mide:** el learning rate se elige con la semilla 0 y
  la decisión se toma con las semillas 1 y 2.
- **Longitud:** [[F001]] indica que la longitud total no identifica al modelo, así que no
  debería servir de atajo.

## Test Plan

- Experiment(s): [[E003]] (compartido con [[H004]]).
- Datos: `data/train.csv` completo; folds fijos de `kb/research/data/E002/folds.csv`.
- Brazos: A = `answerdotai/ModernBERT-base`, B = `Alibaba-NLP/gte-modernbert-base`. Ambos con
  fine-tuning completo.
- Variables controladas: mean pooling + cabeza lineal, máximo 512 tokens, mismas épocas,
  scheduler, batch y semillas en los dos brazos.
- Ajuste: learning rate {1e-5, 3e-5, 5e-5} y época de parada k* (1–10, calendario de 10
  épocas) elegidos por brazo con la semilla de folds 0; decisión con las semillas 1 y 2 (la
  semilla 0 se informa aparte). El ajuste de épocas se añadió al diseñar [[E003]], antes de
  ejecutar.
- Métrica principal: balanced accuracy OOF por modelo (media de semillas).
- Secundarias: balanced accuracy por laboratorio, proporción de errores dentro del
  laboratorio (comparable con [[F002]]), recall por clase y por tramo de longitud.
- Guardarraíles: control con etiquetas barajadas ≤ 0,08; inferencia de unos 4.000 textos
  que quepa holgada en el límite de Kaggle (9 h, sin internet).
- Validez: si algún run colapsa (accuracy cerca del azar o la pérdida no baja), se clasifica
  como fallo de implementación o señal insuficiente, no como rechazo.
- Presupuesto: 2 brazos × 3 lr × 5 folds (10 épocas cada uno) para elegir la configuración, más 2 brazos × 2 semillas × 5 folds
  para decidir. GPU local (RTX 4070 SUPER, 12 GB).
- Confirm if: `python -m name_that_llm.experiments.run_encoder` -> el mejor brazo logra balanced accuracy ≥ 0,412 (media de las semillas 1 y 2)
- Reject if: `python -m name_that_llm.experiments.run_encoder` -> los dos brazos < 0,362 con entrenamiento estable y la rejilla de lr completa
- Inconclusive if: el mejor brazo queda entre 0,362 y 0,412, o hay runs inestables sin
  explicar.

## Evidence

- A favor: [[L001]], fine-tuning de encoders con 97,1 % en 5 chat APIs; supera a TF-IDF +
  LR en el mismo texto y la señal está en palabras y estructura, no en letras.
- A favor: [[F002]], el baseline de n-gramas de caracteres se queda en 0,312.
- En contra: [[L001]] usa 10K textos por modelo y textos de hasta 512 tokens con markdown;
  con pocos datos su accuracy cae mucho (Fig. 3). Nosotros tenemos unos 150 pasajes por
  modelo de 40–250 palabras.
- En contra: [[L001]], entre modelos de la misma familia (Qwen) solo logra 59,8 %, y la mitad
  de nuestros errores son de ese tipo ([[F002]]).

## Conclusion

Pendiente de E003.

## Links

- Mission: [[CHALLENGE]]
- Active State: [[ACTIVE]]
- Literature: [[L001]]
- Prior findings: [[F001]], [[F002]]
- Related hypothesis: [[H004]]
- Experiment: [[E003]]
