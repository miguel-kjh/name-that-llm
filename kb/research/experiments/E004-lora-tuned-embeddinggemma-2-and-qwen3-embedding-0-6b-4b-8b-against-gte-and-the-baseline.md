---
id: "E004"
aliases: ["E004"]
type: experiment
status: DESIGNED
hypothesis: "H005"
created: "2026-11-14"
completed: ""
tags: [encoder, embeddings, lora, scaling]
---

# E004 — LoRA-tuned embeddinggemma-2 and Qwen3-Embedding 0.6B-4B-8B against gte and the baseline

> **Status**: DESIGNED
> **Hypothesis**: [[H005]]
> **Created**: 2026-11-14
> **Completed**:

## Objective

Responder con un único conjunto de runs:

1. **[[H005]]:** ¿un encoder de embeddings de 4B u 8B entrenado con LoRA llega a ≥ 0,412 de
   balanced accuracy con GroupKFold por `prompt_id` (media de las semillas 1 y 2)?
2. **Secundario:** ¿cómo es la curva de escala 0,6B → 4B → 8B dentro de la misma familia, y
   cómo queda embeddinggemma-2 (otra familia, tamaño parecido a gte) frente a gte ([[F004]])?

## Setup

- **Environment:** `.venv` del proyecto (Python 3.12), grupo `gpu` (torch 2.11.0+cu128,
  transformers 5.23.0) más `peft`; `bitsandbytes` solo si el 8B necesita QLoRA. Versiones
  exactas en `manifest.json`.
- **Code (por escribir):** ampliar `src/name_that_llm/encoder.py` con pooling por último
  token, padding a la izquierda, prompts, LoRA (vía `peft`) y gradient checkpointing; runner
  nuevo `src/name_that_llm/experiments/run_lora.py` con las mismas fases que `run_encoder.py`.
  Reutiliza `validation.py`, `metrics.py`, `labels.py` y `decisions.py`. Issue #7, rama
  `e004-lora`.
- **Data:** `data/train.csv` completo (2.699 pasajes, 150 prompts, 18 modelos). Hash SHA-256
  igual al de E002 y E003.
- **Validation:** folds fijos de `kb/research/data/E002/folds.csv` (semillas 0, 1 y 2; 5 folds
  por semilla). Se leen del archivo, no se regeneran.
- **Modelos** (todos Apache 2.0, no gated, commits fijados):

  | Brazo | Modelo (Hugging Face) | Parámetros | Pooling | Commit |
  |---|---|---|---|---|
  | G | `google/embeddinggemma-2` (solo texto) | 270M | media | `f9972ac22e696a105c482202646979e33e4b5a1e` |
  | Q0 | `Qwen/Qwen3-Embedding-0.6B` | 0,6B | último token | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
  | Q4 | `Qwen/Qwen3-Embedding-4B` | 4,0B | último token | `5cf2132abc99cad020ac570b19d031efec650f2b` |
  | Q8 | `Qwen/Qwen3-Embedding-8B` | 7,6B | último token | `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` |

  - G se carga sin los codificadores de visión y audio (`vision_config=None`,
    `audio_config=None`).
- **Receta común (fija para todos los brazos):**
  - Pooling nativo de cada modelo → normalización L2 reescalada a norma √d → dropout 0,1 →
    capa lineal → 18 clases. La normalización se añadió tras el smoke test (decisión del
    usuario, 2026-11-14; ver Progress).
  - LoRA r = 16, α = 32, dropout 0,05, en todas las capas lineales del backbone (atención y
    MLP); backbone congelado; cabeza lineal entrenada entera.
  - Prompt nativo de clasificación, igual en train y validación:
    - G: `task: classification | query: {texto}`.
    - Q*: `Instruct: Identify which language model wrote this passage\nQuery:{texto}`.
  - Texto crudo, sin normalizar. Máximo 512 tokens con truncado; se espera 0 % truncados.
  - Pesos base en bf16; parámetros LoRA y cabeza en float32; autocast bf16.
  - Q4 y Q8 en bf16 con gradient checkpointing. Q8 cabe en 24 GB (medido), así que no se
    usa QLoRA.
  - AdamW, weight decay 0,01, batch efectivo 16 (acumulación de gradiente si la memoria no
    da), warmup lineal 10 % + decaimiento lineal, gradient clip 1,0, cross-entropy sin pesos.
  - Calendario de 6 épocas en todos los runs; se evalúa el fold de validación al final de
    cada época. En la fase de decisión se entrena con el mismo calendario de 6 épocas y se
    para tras la época k*.
- **Protocolo reducido** (decidido con el usuario, 2026-11-14):
  - G y Q0: búsqueda de lr {1e-4, 3e-4} con la semilla 0; se elige (lr, k*) por brazo.
  - Q4 y Q8: usan el lr elegido para Q0; con la semilla 0 solo se elige k* (1–6). Q8 se
    ejecuta siempre, sin depender del resultado de Q4.
  - Selección: mayor balanced accuracy OOF de la semilla 0 (los 5 folds juntos). Empates
    (< 0,005): menos épocas y luego menor lr.
- **Seeds:** la semilla de folds `s` se usa también como semilla de entrenamiento.
- **Comparators:** [[E003]] brazo B (gte-modernbert-base, 0,357 en las semillas 1 y 2) y
  [[E002]] (TF-IDF, 0,307 en las semillas 1 y 2), mismos folds.
- **Compute:** GPU local NVIDIA RTX 4090 (24 GB). Estimación inicial por época, a confirmar
  en el piloto: G ~15 s, Q0 ~35 s, Q4 ~4 min, Q8 ~8 min.

## Procedure

0. **Preparación.** `uv add --group gpu peft`; descargar los cuatro modelos a la caché local
   con los commits fijados.
1. **Tests** (`python -m pytest tests`, CPU y sin descargas): pooling por último token con
   padding a la izquierda (mismo embedding con y sin padding extra); el prompt se añade igual
   en train y predicción; solo LoRA y cabeza son entrenables; parar en k con el calendario de
   6 épocas reproduce la época k del run largo.
2. **Preflight** (`--phase preflight`), por brazo:
   - Folds idénticos a E002 en las 3 semillas.
   - 0 pasajes truncados a 512 tokens (con el prompt incluido).
   - En Q*, el último token de cada secuencia es el token de fin que usa el modelo.
   - Dtype de los pesos base (bf16) y número de parámetros entrenables.
   - Memorización: 64 pasajes, lr 3e-4, 30 épocas → ≥ 95 % de accuracy en esos 64.
3. **Piloto** (`--phase pilot`): 1 fold, 1 época por brazo; mide tiempo por época y memoria
   máxima. Decide si Q8 cabe en bf16. Si la estimación total supera 3× la de este diseño,
   parar y replantear con el usuario.
4. **Búsqueda (semilla 0)** (`--phase search`):
   - G y Q0: 2 lr × 5 folds = 10 runs de 6 épocas por brazo.
   - Q4 y Q8: lr de Q0 × 5 folds = 5 runs de 6 épocas por brazo.
   - Total 30 runs. Se ejecuta en orden G, Q0, Q4, Q8 (Q4 y Q8 necesitan el lr de Q0).
5. **Decisión (semillas 1 y 2)** (`--phase eval`): 4 brazos × 2 semillas × 5 folds = 40 runs,
   cada uno con su (lr, k*). Se guardan las probabilidades OOF y las métricas por fold.
6. **Control con etiquetas barajadas** (`--phase control --arms Q0`): Q0 con su (lr, k*),
   semilla de folds 1, etiquetas barajadas dentro del train de cada fold. Q0 comparte todo el
   código con Q4 y Q8 y cuesta poco.
7. **Guardarraíl de inferencia** (`--phase timing`): tiempo de predecir 4.000 pasajes con cada
   brazo en la 4090, incluida la carga del modelo y del adaptador LoRA; se extrapola al
   hardware de Kaggle.
8. **Resumen** (`--phase summarize`): `summary.json` con las reglas de decisión aplicadas.

## Expected Outcome

Métrica principal: balanced accuracy OOF por brazo, media de las semillas 1 y 2.

**[[H005]] (umbrales fijados antes de ejecutar):**

- Confirm if: `python -m name_that_llm.experiments.run_lora --phase summarize` -> Q4 o Q8 logra balanced accuracy ≥ 0,412
- Reject if: `python -m name_that_llm.experiments.run_lora --phase summarize` -> ni Q4 ni Q8 superan 0,377 (gte + 0,02), con entrenamiento estable
- Inconclusive if: el mejor de Q4/Q8 queda entre 0,377 y 0,412, o hay runs inestables sin
  explicar.

**Guardarraíles (si alguno falla, el experimento no es válido):**

- Control con etiquetas barajadas: balanced accuracy ≤ 0,08 (azar 0,056).
- Sin pasajes truncados.
- Estabilidad: en cada run de la fase de decisión la pérdida de train baja al menos un 50 %
  respecto a la inicial y la balanced accuracy no queda por debajo de 0,10. En la búsqueda se
  registra pero no invalida.
- Inferencia de 4.000 pasajes: < 10 min en la 4090 por brazo; para Q8 se estima además el
  tiempo en el hardware de Kaggle y debe caber holgadamente en 9 h. Un brazo que no cumple
  puede confirmar o rechazar [[H005]], pero no pasa a candidato de envío.

**Métricas secundarias** (no deciden, explican):

- Curva de escala Q0 → Q4 → Q8 y diferencia por fold entre brazos consecutivos.
- G frente a gte (E003 brazo B), sabiendo que también cambia LoRA frente a fine-tuning
  completo.
- Balanced accuracy por laboratorio y proporción de errores dentro del laboratorio
  (comparables con [[F002]] y [[F003]]).
- Recall por clase (atención a `qwen3.8-max`, por la afinidad de familia), matriz de confusión
  y balanced accuracy por tramo de longitud (< 80, 80–149, 150+ palabras).
- Delta por fold frente al baseline y frente a gte en los mismos folds.

**Reglas de parada:**

- Si tras la búsqueda Q0 no llega a 0,20 en la semilla 0, parar y revisar la implementación.
- Presupuesto: 30 runs de búsqueda (6 épocas) + 40 de decisión + 5 de control. Estimación
  inicial ~12 h de GPU, dominada por Q4 (~3,5 h) y Q8 (~7 h). Si el piloto o el tiempo real
  superan 3× la estimación, parar y replantear con el usuario.

## Progress

- [x] (`2026-11-14`) Diseño del experimento con el protocolo reducido acordado con el
  usuario. Q8 en bf16; QLoRA solo si no cabe.
- [x] (`2026-11-14`) Dependencias: `peft` 0.21.3 (trae `accelerate` 1.15.0). Modelos
  descargados con los commits fijados. La descarga por xet (backend por defecto de
  huggingface_hub) se colgaba sin avanzar; con `HF_HUB_DISABLE_XET=1` funciona.
- [x] (`2026-11-14`) Implementación: `encoder.py` (pooling por último token, `TextFormat`,
  LoRA, gradient checkpointing, micro-batches, `output_size`), `decisions.py`
  (`h005_status`), `experiments/runs.py` (maquinaria de runs compartida con E003, movida sin
  cambiar la lógica) y `experiments/run_lora.py`. 134 tests pasan (89 de E003 sin tocar).
- Decisiones y hallazgos de implementación (antes de ejecutar nada de E004):
  - **Token final de Qwen:** el tokenizer de Qwen3-Embedding ya termina cada secuencia en
    `<|endoftext|>`, el token sobre el que el modelo hace pooling, pero su `eos_token` es
    `<|im_end|>`. Una primera versión añadía `<|im_end|>` detrás y habría hecho el pooling
    sobre el token equivocado. Ahora `TextFormat.end_token` nombra el token explícitamente
    y se garantiza exactamente uno al final; el preflight lo comprueba.
  - **Padding a la derecha** en todos los brazos, también en Qwen (el ejemplo oficial usa
    padding a la izquierda). Con atención causal y el último token real, es idéntico a
    procesar cada texto sin padding; el izquierdo desplaza las posiciones según el padding.
    Hay un test de invariancia.
  - **embeddinggemma-2** proyecta sus estados de 512 a 768 (`text_config.embedding_dim`)
    antes de devolverlos; la cabeza usa la anchura real de la salida. Cargado solo con
    texto: 271M parámetros, 6,1M entrenables con LoRA. `all-linear` incluye también las
    proyecciones por capa (per-layer embeddings) de su arquitectura.
  - Q0: 10,1M parámetros LoRA entrenables de 606M. Máximo 394 tokens (Q0) y 368 (G) con el
    prompt; 0 truncados.
  - **Regresión de E003:** el run B, semilla 1, fold 0 con el código refactorizado empieza
    con la misma pérdida exacta (2,92339) y termina en 0,359 frente a 0,356. Repetido con
    el mismo código da 0,339: el no determinismo de la GPU entre ejecuciones idénticas es
    de ~0,02 por fold, mayor que la diferencia con la referencia. La refactorización no
    cambia el comportamiento. Consecuencia para el análisis: diferencias por fold < 0,02
    entre brazos son ruido; las decisiones usan medias de 10 folds.
- [x] (`2026-11-14`) Smoke test de todas las fases con G y Q0 (30 prompts, fold 0, en el
  scratchpad; métricas sin significado). Todas corren. La fase de timing falló una vez con
  `CUDA error: device not ready` y al repetirla funcionó (transitorio, WSL); el lanzador
  reintenta cada fase hasta 3 veces (el runner es reanudable).
- [x] (`2026-11-14`) **Cambio de receta tras el smoke, antes de cualquier run real:** la
  pérdida inicial de G era 17–20 (azar ln 18 = 2,89) porque su embedding proyectado tiene una
  norma muy grande. El usuario eligió normalizar el embedding (L2), como hacen los pipelines
  nativos de los dos modelos. Con norma 1 los logits quedan acotados por la norma de la cabeza:
  memorizando 64 pasajes la pérdida se quedaba en 2,2–2,5 tras 30 épocas (aunque con 100 % de
  acierto) y habría fallado el guardarraíl de estabilidad. Se reescala a norma √d (RMS 1 por
  dimensión, la escala de una salida de LayerNorm): pérdida inicial 3,0 (Q0) y 3,2 (G), y con
  lr 1e-4 baja a 0,004 en 30 épocas. Se aplica a los cuatro brazos.
- [x] (`2026-11-14`) Memoria en el peor caso (los 16 pasajes más largos), bf16 + gradient
  checkpointing: Q4 con micro-batch 16, pico 11,5 GB; Q8 con micro-batch 4, pico 16,9 GB al
  entrenar y 17,5 GB al predecir (de 24 GB). **Q8 cabe en bf16; no hace falta QLoRA.**
  Micro-batches finales: G 16, Q0 16, Q4 16, Q8 4 (batch efectivo 16 en todos).
- [ ] Preflight y piloto de tiempo y memoria.
- [ ] Búsqueda (semilla 0).
- [ ] Decisión (semillas 1 y 2).
- [ ] Control, guardarraíl de inferencia y resumen.
- [ ] Análisis y finding.

## Results

Datos en bruto previstos en `kb/research/data/E004/` (mismo formato que E003):

- `manifest.json`, `preflight.json`, `pilot.json`, `timing.json`.
- `search.csv`, `search_curves.csv`, `runs.csv`.
- `oof/`: un parquet por run con las probabilidades OOF.
- `per_class_recall.csv`, `length_buckets.csv`, `confusion_{brazo}.csv`.
- `summary.json`: métricas, deltas, guardarraíles y estado de [[H005]].

Los adaptadores LoRA y los pesos base no se guardan en `kb/`.

## Analysis

_Pendiente._

## Decision

_Pendiente._

## Links

- Mission: [[CHALLENGE]]
- Active State: [[ACTIVE]]
- Parent Hypothesis: [[H005]]
- Comparators: [[E003]], [[E002]]
- Prior findings: [[F003]], [[F004]]
- Literature: [[L001]]
