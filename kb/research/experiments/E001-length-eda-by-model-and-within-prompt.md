---
id: "E001"
aliases: ["E001"]
type: experiment
status: COMPLETED
hypothesis: "H001"
created: "2026-11-13"
completed: "2026-11-13"
tags: [eda, length]
---

# E001 — Length EDA by model and within prompt

> **Status**: COMPLETED
> **Hypothesis**: [[H001]]
> **Created**: 2026-11-13
> **Completed**: 2026-11-13

## Objective

Comprobar si la longitud de los pasajes depende del modelo o del laboratorio, quitando el
efecto del prompt gracias al diseño cruzado (150 prompts × 18 modelos). Issue de GitHub #1.

## Setup

- Environment: `uv`, Python ≥ 3.12, scipy, pandas, matplotlib.
- Data: `data/train.csv` (2.699 filas). `data/test.csv` local solo tiene 3 filas de ejemplo.
- Compute: CPU, segundos.
- Dependencies: `src/name_that_llm/eda_length.py`, `src/name_that_llm/experiments/eda_length.py`,
  `tests/test_eda_length.py`.

## Procedure

1. Rasgos por pasaje: palabras (separadas por espacios), caracteres y frases (fin en `.!?`
   seguido de espacio o fin de texto).
2. Kruskal-Wallis por modelo y por laboratorio, con ε² = H/(n−1) y su valor esperado sin
   efecto, (k−1)/(n−1).
3. Friedman por modelo dentro de cada prompt (149 prompts completos), con W de Kendall.
4. Rasgos centrados por prompt y boxplots por modelo, en bruto y centrados.
5. Comparación train/test con Kolmogorov-Smirnov.

## Expected Outcome

- Confirm if: `python -m name_that_llm.experiments.eda_length` -> ε²(n_words, model) < 0,01 y
  W(n_words) < 0,05
- Reject if: `python -m name_that_llm.experiments.eda_length` -> ε²(n_words, model) ≥ 0,01 o
  W(n_words) ≥ 0,05, con p < 0,01

## Progress

- [x] (`2026-11-13`) Tests sintéticos con respuesta conocida: 22 pasan (`uv run pytest tests`).
  Revelaron que ε² no vale 0 sin efecto; se anotó en [[H001]] antes de ver los datos reales.
- [x] (`2026-11-13`) Cuadres con los datos reales: 2.699 filas, 150 prompts, 18 modelos,
  149 prompts completos; palabras min 36, mediana 101, max 274.
- [x] (`2026-11-13`) Ejecución del script y revisión visual de las figuras.

## Results

Salidas en `kb/research/data/eda_length/` (fuera de git por la regla `data/` de `.gitignore`;
se regeneran con el script).

| Rasgo | Test | Agrupación | p | Efecto | Valor | Sin efecto |
|---|---|---|---|---|---|---|
| n_words | Kruskal | modelo | 0,138 | ε² | 0,0087 | 0,0063 |
| n_words | Kruskal | laboratorio | 0,224 | ε² | 0,0026 | 0,0019 |
| n_words | Friedman | modelo | 0,154 | W | 0,0090 | 0,0067 |
| n_chars | Kruskal | modelo | 0,099 | ε² | 0,0092 | 0,0063 |
| n_chars | Friedman | modelo | 0,221 | W | 0,0083 | 0,0067 |
| n_sentences | Kruskal | modelo | < 10⁻⁶ | ε² | 0,0309 | 0,0063 |
| n_sentences | Kruskal | laboratorio | 0,056 | ε² | 0,0040 | 0,0019 |
| n_sentences | Friedman | modelo | < 10⁻⁶ | W | 0,0348 | 0,0067 |

- Mediana de palabras por modelo: de 84 (gemini-3.8-flash) a 117 (claude-opus-5), con rangos
  intercuartílicos muy solapados. gemini-3.8-flash queda el más bajo también centrado por prompt.
- Fuera del rango declarado 40–250: 41 pasajes (18 por debajo, 23 por encima), repartidos entre
  15 modelos; el que más, claude-opus-4-6 con 7 (todos por encima).
- Mediana de palabras por frase: de 19,8 (gemini-3.1-pro-preview) a 26,5 (claude-opus-4-6).
- Train frente a test: no informativo, el `test.csv` local tiene 3 filas.

## Analysis

- La longitud en palabras y caracteres no depende del modelo ni del laboratorio a un nivel
  útil: ε² y W están cerca de lo esperado sin efecto y los p-valores no son significativos.
  [[H001]] se confirma.
- El número de frases sí depende del modelo (ε² ≈ 5 veces el valor nulo). Como la longitud
  total no cambia, la señal es la longitud de frase, un rasgo de estilo. Es exploratorio: no
  estaba en el criterio de [[H001]], y el contador de frases puede tener sesgos por modelo
  (abreviaturas, listas o markdown sin punto final).
- Entre laboratorios la señal de frases es débil (p = 0,056): la diferencia es entre modelos
  concretos, no entre laboratorios.

## Decision

- No usar la longitud total como rasgo, pero sí normalizar los rasgos de conteo por longitud,
  porque varía mucho (36–274 palabras) sin informar del modelo.
- Llevar la longitud de frase (y su variabilidad) al EDA de estilo (punto 3), con un contador
  de frases revisado a mano.
- Medir el baseline por tramos de longitud: los pasajes cortos tendrán menos señal.

## Links

- Mission: [[CHALLENGE]]
- Active State: [[ACTIVE]]
- Parent Hypothesis: [[H001]]
- Finding: [[F001]]
