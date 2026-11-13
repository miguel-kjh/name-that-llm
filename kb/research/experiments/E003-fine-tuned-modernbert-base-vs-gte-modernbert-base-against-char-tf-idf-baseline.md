---
id: "E003"
aliases: ["E003"]
type: experiment
status: DESIGNED
hypothesis: "H003"
created: "2026-11-13"
completed: ""
tags: [encoder, embeddings, fine-tuning]
---

# E003 — Fine-tuned ModernBERT-base vs gte-modernbert-base against char TF-IDF baseline

> **Status**: DESIGNED
> **Hypothesis**: [[H003]] (principal), [[H004]] (comparación entre brazos)
> **Created**: 2026-11-13
> **Completed**:

## Objective

Responder dos decisiones con un único conjunto de runs:

1. **[[H003]]:** ¿un encoder fine-tuneado con la receta de [[L001]] llega a ≥ 0,412 de balanced
   accuracy (baseline 0,312, [[F002]]) con GroupKFold por `prompt_id`? Si sí, pasa a ser el
   modelo principal en lugar del TF-IDF.
2. **[[H004]]:** con el mismo backbone, ¿el encoder MLM (`ModernBERT-base`) supera a su versión
   con entrenamiento contrastivo (`gte-modernbert-base`) en ≥ 0,02 y en ≥ 8 de 10 folds? Decide
   qué familia de modelos escalar después.

## Setup

- **Environment:** `.venv` del proyecto (Python 3.12). Grupo de dependencias `gpu` en
  `pyproject.toml`: `torch` 2.11.0+cu128 y `transformers` 5.23.0. Sin `flash-attn`. Versiones
  exactas en `manifest.json`.
- **Code:** `src/name_that_llm/encoder.py` (modelo y bucle de entrenamiento),
  `src/name_that_llm/decisions.py` (reglas de decisión), `src/name_that_llm/experiments/run_encoder.py`
  (fases). Reutiliza `validation.py`, `metrics.py` y `labels.py` de E002. Issue #5, rama
  `e003-encoders`.
- **Data:** `data/train.csv` completo (2.699 pasajes, 150 prompts, 18 modelos con 149–150
  pasajes cada uno; 36–274 palabras, mediana 101). Hash SHA-256 igual al de E002. Ningún
  pasaje tiene markdown, así que la señal de formato de [[L001]] no aplica aquí.
- **Validation:** los folds fijos de `kb/research/data/E002/folds.csv` (semillas 0, 1 y 2;
  5 folds por semilla). Se leen del archivo, no se regeneran.
- **Brazos:**

  | Brazo | Modelo (Hugging Face) | Preentrenamiento |
  |---|---|---|
  | A | `answerdotai/ModernBERT-base` | MLM |
  | B | `Alibaba-NLP/gte-modernbert-base` | ModernBERT-base + entrenamiento contrastivo |

  Mismo tokenizador (vocabulario idéntico, comprobado) y misma arquitectura (149M). Commits
  fijados: A `8949b909ec900327062f0ebf497f51aef5e6f0c8`, B
  `e7f32e3c00f91d699e8c43b53106206bcc72bb22`.
  - Pesos siempre en float32 para los dos brazos; bf16 solo vía autocast. El checkpoint de B
    está guardado en float16 y transformers 5 lo cargaba así por defecto (ver Progress).
- **Receta común (fija para los dos brazos):**
  - Mean pooling sobre tokens no padding → dropout 0,1 → capa lineal 768 → 18. Fine-tuning
    completo, sin congelar capas.
  - Texto crudo, sin normalizar (sin minúsculas, sin limpiar puntuación). Máximo 512 tokens
    con truncado; se espera 0 % de pasajes truncados (≤ 274 palabras).
  - AdamW, weight decay 0,01, batch 16, warmup lineal 10 % + decaimiento lineal, gradient
    clip 1,0, bf16, cross-entropy sin pesos de clase (las clases están balanceadas).
  - Calendario de lr de 10 épocas (warmup lineal 10 % + decaimiento lineal) en todos los runs.
  - Early stopping sin mirar el fold que se mide: la época de parada k* se elige en la
    búsqueda (semilla 0) y en la fase de decisión se entrena con el mismo calendario de 10
    épocas y se para tras la época k*.
- **Búsqueda (por brazo):** lr {1e-5, 3e-5, 5e-5}; cada lr se entrena 10 épocas y el fold de
  validación se evalúa al final de cada una. Se elige el par (lr, k*) con mayor balanced
  accuracy OOF de la semilla 0. Sustituye a la rejilla de épocas {4, 8} del primer diseño
  (cambio decidido con el usuario antes de ejecutar; ver Progress).
- **Seeds:** la semilla de folds `s` se usa también como semilla de entrenamiento (`torch`,
  `numpy`, `random`, inicialización de la cabeza, orden de los batches).
- **Comparator:** E002 (TF-IDF de caracteres + LR), mismos folds. Valores por semilla en
  `kb/research/data/E002/summary.json`: 0,322 / 0,310 / 0,304 (media de las semillas 1 y 2:
  0,307).
- **Compute:** GPU local NVIDIA RTX 4070 SUPER (12 GB). Estimación: 1–3 min por run.

## Procedure

0. **Preparación.** Añadir el grupo `gpu` y descargar los dos modelos a la caché local.
1. **Tests y comprobaciones previas** (`python -m pytest tests`):
   - Los folds leídos de `folds.csv` coinciden con los de E002 (mismos ids por fold y semilla).
   - El mean pooling ignora el padding (mismo embedding con y sin padding extra).
   - Porcentaje de pasajes truncados a 512 tokens = 0 con el tokenizador de ModernBERT.
   - Prueba de sobreajuste: entrenar con 64 pasajes (lr 5e-5, 30 épocas) llega a ≥ 95 % de
     accuracy en esos mismos 64. Si no, hay un fallo de implementación y no se sigue.
   - Los pesos de los dos brazos se cargan en float32.
2. **Fase de búsqueda (semilla 0):**
   `python -m name_that_llm.experiments.run_encoder --phase search`
   - 2 brazos × 3 lr × 5 folds = 30 runs de 10 épocas, evaluados tras cada época.
   - Se elige por brazo el par (lr, k*) con mayor balanced accuracy OOF de la semilla 0 (los 5
     folds juntos). Empates (< 0,005): menos épocas y luego menor lr.
3. **Fase de decisión (semillas 1 y 2):**
   `python -m name_that_llm.experiments.run_encoder --phase eval`
   - (lr, k*) elegido por brazo, calendario de 10 épocas parado tras k*;
     2 brazos × 2 semillas × 5 folds = 20 runs.
   - Se guardan las probabilidades OOF (para ensembles futuros) y las métricas por fold.
4. **Control con etiquetas barajadas:**
   `python -m name_that_llm.experiments.run_encoder --phase control --arms A`
   - Brazo A, (lr, k*) elegido, semilla de folds 1, etiquetas barajadas dentro del train de
     cada fold.
5. **Guardarraíl de inferencia:** medir el tiempo de predecir 4.000 pasajes (los de train
   repetidos hasta 4.000) con el mejor brazo en la GPU local, incluida la carga del modelo.
6. **Resumen:** `python -m name_that_llm.experiments.run_encoder --phase summarize` escribe
   `summary.json` con las reglas de decisión aplicadas.

## Expected Outcome

Métrica principal: balanced accuracy OOF por modelo, media de las semillas 1 y 2.

**[[H003]] (umbrales fijados antes de ejecutar):**

- Confirm if: `python -m name_that_llm.experiments.run_encoder --phase summarize` -> el mejor brazo logra balanced accuracy ≥ 0,412
- Reject if: `python -m name_that_llm.experiments.run_encoder --phase summarize` -> los dos brazos < 0,362, con entrenamiento estable y la rejilla completa
- Inconclusive if: el mejor brazo queda entre 0,362 y 0,412, o hay runs inestables sin
  explicar.

**[[H004]] (A − B en los 10 folds de las semillas 1 y 2):**

- Confirm if: `python -m name_that_llm.experiments.run_encoder --phase summarize` -> A − B ≥ 0,02 y A gana en ≥ 8 de 10 folds
- Reject if: `python -m name_that_llm.experiments.run_encoder --phase summarize` -> B − A ≥ 0,02 y B gana en ≥ 8 de 10 folds
- Inconclusive if: |A − B| < 0,02 o ningún brazo gana en ≥ 8 folds; también si algún brazo
  no supera al baseline (0,312). En empate se sigue con A.

**Guardarraíles (si alguno falla, el experimento no es válido):**

- Control con etiquetas barajadas: balanced accuracy ≤ 0,08 (azar 0,056).
- Sin pasajes truncados.
- Estabilidad: en cada run de la fase de decisión la pérdida de train baja al menos un 50 %
  respecto a la inicial y la balanced accuracy no queda por debajo de 0,10. Un run que no
  cumple se investiga como fallo de implementación, no cuenta como resultado. En la fase de
  búsqueda la estabilidad se registra pero no invalida: una configuración que no entrena bien
  (p. ej. lr 1e-5 con 4 épocas) simplemente no se elige.
- Inferencia de 4.000 pasajes en la GPU local < 10 min, para que quepa holgada en Kaggle
  (GPU más lenta, límite de 9 h) incluso si se entrena dentro del notebook.

**Métricas secundarias** (no deciden, explican):

- Balanced accuracy por laboratorio.
- Proporción de errores dentro del laboratorio (comparable con 0,518 de [[F002]]).
- Recall por clase y matriz de confusión.
- Balanced accuracy por tramo de longitud (< 80, 80–149, 150+ palabras), como en E002.
- Delta por fold frente al baseline en los mismos folds.

**Reglas de parada:**

- Si tras la fase de búsqueda el mejor brazo no llega a 0,20 en la semilla 0, parar y revisar
  la implementación antes de gastar la fase de decisión.
- Presupuesto: 55 runs de un fold (30 de búsqueda de 10 épocas + 20 de decisión + 5 de
  control), como máximo 550 épocas de entrenamiento. Si el tiempo real supera 3× la
  estimación, parar y replantear la rejilla.

## Progress

- [x] (`2026-11-13`) Diseño del experimento. La rejilla de [[H003]] se amplía con épocas
  {4, 8} antes de ejecutar nada.
- [x] (`2026-11-13`) Dependencias `gpu` y descarga de modelos con commits fijados.
- [x] (`2026-11-13`) Implementación y tests: 82 pasan (44 nuevos), en CPU y sin descargas.
- [x] (`2026-11-13`) Precisiones antes de ejecutar la búsqueda: la estabilidad decide solo
  sobre los runs de la fase de decisión; las predicciones OOF se guardan por run en `oof/`;
  una matriz de confusión por brazo.
- [x] (`2026-11-13`) Primer preflight **fallido**: B daba pérdida NaN. Causa: transformers 5
  carga el dtype del checkpoint, B está en float16 y AdamW sobre pesos float16 diverge. A
  estaba en float32, así que además había una diferencia de precisión entre brazos. Arreglo:
  `dtype=torch.float32` al cargar, test unitario y comprobación de dtype en el preflight.
- [x] (`2026-11-13`) Preflight **superado** (`preflight.json`): folds idénticos a E002 en las 3
  semillas; 0 pasajes truncados (máximo 396 tokens); pesos float32; los dos brazos memorizan
  los 64 pasajes (100 %; pérdida A 3,22 → 0,0007, B 3,14 → 0,0014).
- [x] (`2026-11-13`) Cambio de diseño acordado con el usuario antes de ejecutar: en vez de
  la rejilla de épocas {4, 8}, cada lr se entrena 10 épocas con la curva por época y se
  elige (lr, k*); la fase de decisión para tras k* sobre el mismo calendario de 10 épocas.
  Un test comprueba que parar en k con el calendario de 10 reproduce exactamente la época k
  del run largo. 88 tests pasan.
- [x] (`2026-11-13`) Smoke test de todas las fases en GPU (30 prompts, fold 0, en el
  scratchpad, fuera de `kb/`): search → eval → control → timing → summarize corren de principio a
  fin; los runs de decisión paran en k* (historial de k*+1 pérdidas); las probabilidades
  suman 1; inferencia de 4.000 pasajes en 7–8 s por brazo. Sus métricas no significan nada
  (unos 430 pasajes de train) y no se usan. Arreglos tras el smoke: temporizador monotónico
  (un run marcó −0,6 s por un salto de reloj de WSL), log de decisión sin "best epoch", y la
  regla de parada ahora bloquea `--phase eval` en vez de solo avisar. 89 tests pasan.
- Estimación con los datos completos: unos 3–4 min por run de búsqueda de 10 épocas, unas
  2,5–3 h para todo el experimento.
- [ ] Fase de búsqueda (semilla 0).
- [ ] Fase de decisión (semillas 1 y 2).
- [ ] Control con etiquetas barajadas y guardarraíl de inferencia.
- [ ] Resumen, análisis y finding.

## Results

Datos en bruto previstos en `kb/research/data/E003/`:

- `manifest.json`: commit del código, hash de los datos, versiones, commits de los modelos en
  el hub, GPU, configuración elegida por brazo.
- `search.csv`: una fila por brazo × lr × fold (semilla 0), con la balanced accuracy de cada
  época.
- `search_curves.csv`: balanced accuracy OOF por brazo × lr × época (los 5 folds juntos).
- `runs.csv`: una fila por brazo × semilla × fold de la fase de decisión, con balanced accuracy,
  métricas secundarias, pérdida inicial y final, y tiempo.
- `preflight.json`, `timing.json`: comprobaciones previas y guardarraíl de inferencia.
- `oof/`: un parquet por run con las probabilidades OOF (18 columnas) y la predicción.
- `per_class_recall.csv`, `length_buckets.csv` (con columna de brazo) y
  `confusion_A.csv`, `confusion_B.csv`.
- `summary.json`: métricas principales, deltas frente al baseline, guardarraíles y estado de
  decisión de H003 y H004.

Los checkpoints de los modelos no se guardan en `kb/` (pesan ~600 MB cada uno).

## Analysis

_Pendiente._

## Decision

_Pendiente._

## Links

- Mission: [[CHALLENGE]]
- Active State: [[ACTIVE]]
- Parent Hypothesis: [[H003]]
- Related hypothesis: [[H004]]
- Comparator: [[E002]]
- Literature: [[L001]]
