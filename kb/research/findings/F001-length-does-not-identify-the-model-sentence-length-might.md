---
id: "F001"
aliases: ["F001"]
type: finding
hypothesis: "H001"
experiment: "E001"
impact: "MEDIUM"
created: "2026-11-13"
tags: [eda, length]
---

# F001 — Length does not identify the model, sentence length might

> **Created**: 2026-11-13
> **Hypothesis**: [[H001]]
> **Experiment**: [[E001]]
> **Impact**: MEDIUM

## Finding

La longitud total de un pasaje (palabras o caracteres) no lleva información útil del modelo:
el modelo explica menos del 1 % de su variación, tanto en global como dentro de cada prompt.
El número de frases, en cambio, sí varía según el modelo, así que la longitud de frase es un
candidato a rasgo de estilo.

## Evidence

- n_words por modelo: Kruskal ε² = 0,0087 (sin efecto: 0,0063), p = 0,14; Friedman dentro
  de prompt W = 0,009 (sin efecto: 0,0067), p = 0,15. n_chars da lo mismo.
- n_words por laboratorio: ε² = 0,0026, p = 0,22.
- n_sentences por modelo: ε² = 0,031, W = 0,035, p < 10⁻⁶. Mediana de palabras por frase
  de 19,8 (gemini-3.1-pro-preview) a 26,5 (claude-opus-4-6).
- Detalle y tabla completa en [[E001]].

## What Improved For Real

- Sabemos que la longitud es ruido del generador y no un atajo: un clasificador que dependa de
  ella no generalizará. Hay que normalizar los conteos por longitud.
- El diseño cruzado permite comparar dentro de cada prompt, y el centrado por prompt ya está
  implementado y probado para los siguientes rasgos.

## Remaining Debt

- La señal de frases es exploratoria (fuera del criterio previo de [[H001]]) y el contador de
  frases es una regex simple que puede sesgarse por modelo (abreviaturas, listas, markdown).
- No se pudo comparar train con test: el `test.csv` local solo tiene 3 filas.
- No se midió cómo cae el accuracy en pasajes cortos; queda para el baseline.

## Next Move

- Medir el baseline (GroupKFold por `prompt_id`) e informar el accuracy por tramo de longitud.
- En el EDA de estilo, revisar a mano el contador de frases y estudiar longitud de frase y su
  variabilidad por modelo, centradas por prompt.

## Links

- Mission: [[CHALLENGE]]
- Active State: [[ACTIVE]]
- Parent Hypothesis: [[H001]]
- Parent Experiment: [[E001]]
