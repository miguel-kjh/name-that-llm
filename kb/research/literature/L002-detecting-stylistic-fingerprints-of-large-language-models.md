---
id: "L002"
aliases: ["L002"]
type: literature
source_type: "PAPER"
relevance: "LOW"
source_url: "https://arxiv.org/abs/2503.01659"
date_reviewed: "2026-11-13"
tags: []
---

# L002 — Detecting Stylistic Fingerprints of Large Language Models

## 0. Ficha

| Campo | Valor |
|---|---|
| Autores | Yehonatan Bitton, Elad Bitton, Shai Nisan (Copyleaks) |
| Venue y año | Preprint arXiv, marzo 2025 (sin venue revisado) |
| Enlace (arXiv / DOI) | https://arxiv.org/abs/2503.01659 |
| Código | No publicado |
| PDF local | `papers/2503.01659v1.pdf` |
| Fecha de lectura | 2026-11-13 |

---

## 1. El núcleo

**X — ¿Qué intentan hacer y por qué es relevante?**
Clasificar qué familia de LLM (Claude, Gemini, Llama, OpenAI) escribió un texto, con muy pocos falsos positivos, porque atribuir mal un texto tiene costes legales, reputacionales y de propiedad intelectual [§1.1, §1.4].

**Y — ¿Por qué es difícil?**
La tarea es sensible al coste: un falso positivo cuesta más que no atribuir. Un solo clasificador no da suficiente confianza [§1.4, §2.1]. No se explica por qué fallan enfoques anteriores en concreto.

**Z — ¿Cuál es su contribución?**
Un ensemble de tres clasificadores de arquitecturas y datos de entrenamiento distintos que solo emite predicción si los tres coinciden (voto unánime); si no, devuelve "no-agreement" [§2.1, §2.5].

**1a — ¿Cómo lo verifican experimentalmente?**
Test de 200.000 textos (50.000 por familia). El ensemble unánime logra precisión 0,9988 (macro F0.5 0,9988) y FPR 0,0004, absteniéndose en el 1,06 % de los textos [Tabla 5, Tabla 6]. Además lo aplican a cuatro modelos no vistos (phi-4, Grok-1, Mixtral-8x7b, DeepSeek-R1) [§3].

**1b — ¿Hay respaldo teórico?**
Solo un argumento clásico de ensembles: con $L$ clasificadores de error $p<0{,}5$ e independientes, el error del voto unánime es $p^L$ [§2.1]. No hay teoremas propios.

**2 — Contribuciones que reclaman los autores**
- Un método "novedoso" y sensible al coste para atribuir textos a familias de LLMs [§4].
- Precisión muy alta (0,9988) y FPR muy bajo (0,0004) [§4].
- Capacidad de distinguir textos de modelos vistos y no vistos, y relaciones estilísticas entre modelos (DeepSeek-R1 se parece a OpenAI) [§3.3, §4].

---

## 2. Trabajos relacionados (hermanos)

| Trabajo | Diferencia (supuestos / método) | ¿Comparado en experimentos? | Si no, ¿por qué? |
|---|---|---|---|
| Authorship attribution clásica (n-gramas, POS, KNN, Naive Bayes) | Rasgos manuales para autores humanos | No | No se justifica |
| Atribución neuronal con modelos preentrenados [5, 13, 14, 18] | Un único clasificador fine-tuneado | No | No se justifica |
| Detección IA vs. humano (perplexity, burstiness, BERT) | Binaria | No | Problema distinto [§1.2.2] |
| Watermarking (SynthID) | Requiere intervenir en la generación | No | Se presenta como enfoque con otras limitaciones [§1.2.2] |
| McGovern et al. 2024 | Fingerprints léxicos/sintácticos | No | No se justifica |

---

## 3. Antecedentes (ancestros)

**Conceptos previos necesarios**
- Authorship attribution y AI forensics [§1.2].
- Ensembles por voto mayoritario y unánime; clasificación sensible al coste [§2.1].
- Métricas: precisión, recall, F-beta con β=0,5, FPR por clase y macro [§2.3].

**Definición del problema**
Clasificación multiclase de un texto en inglés en $K=4$ familias de LLM, con opción de abstenerse ("no-agreement"). Las métricas del ensemble unánime se calculan excluyendo las abstenciones [§2.3, Tabla 5].

**Supuestos inusuales o restrictivos**
- Las clases son familias de modelos, no modelos concretos.
- Las métricas excluyen el 1,06 % de textos en los que el ensemble se abstiene [Tabla 5].

**¿El escenario del problema es una contribución nueva?**
No. Es authorship attribution entre LLMs con abstención.

---

## 4. Método

**Qué hacen**
1. Entrenan tres clasificadores (I, II, III) con "arquitecturas distintas" y muestras aleatorias distintas de textos de las cuatro familias, balanceadas por clase [§2.2].
2. Evalúan cada uno en el test de 200K [Tablas 1–3].
3. Combinan por voto mayoritario [Tabla 4] y por voto unánime con abstención [Tabla 5].
4. Aplican el ensemble unánime a 13.000 prompts respondidos por cuatro modelos no vistos y miden la distribución de predicciones [§3].

**Por qué lo hacen así**
- Diversidad de arquitecturas y datos para que los errores estén poco correlacionados [§2.2].
- Voto unánime para minimizar falsos positivos, aceptando menos recall [§2.1].

---

## 5. Configuración experimental

**Instancia del problema**
- Test: 50.000 textos por familia (Claude, Gemini, Llama, OpenAI), en inglés [§2.3].
- Modelos no vistos: phi-4, Grok-1 (locales), Mixtral-8x7b-instruct y DeepSeek-R1 (TogetherAI), con 13.000 prompts de cuatro repositorios de Hugging Face (ChatGPT-prompts, alpaca-gpt4, Magpie-Pro, awesome-chatgpt-prompts) [§3.2].

**Detalles de implementación**
- Arquitecturas de los tres clasificadores: No se aborda en el paper.
- Tamaño y origen del train, versiones concretas de los modelos de cada familia, prompts del train, longitud de los textos, hiperparámetros y cómputo: No se aborda en el paper.

---

## 6. Resultados y discusión

**Resultados principales frente a baselines**

| Tarea / dataset | Métrica | Método | Mejor baseline |
|---|---|---|---|
| 4 familias, test 200K | macro F0.5 | 0,9988 (unánime, sin abstenciones) | 0,9946 (mayoría); 0,9935 (clasificador II) [Tabla 6] |
| 4 familias, test 200K | macro FPR | 0,0004 | 0,0018 (mayoría); 0,0022 (clasificador II) [Tabla 6] |
| 4 familias, test 200K | % abstención | 1,06 % | — [Tabla 5] |
| phi-4 / Grok-1 (no vistos) | % no-agreement | 99,3 % / 100 % | — [Fig. 1] |
| Mixtral (no visto) | % no-agreement | 65 % (26 % OpenAI, 8,8 % Llama) | — [Fig. 1c] |
| DeepSeek-R1 (no visto) | % predicho OpenAI | 74,2 % (25,7 % no-agreement) | — [Fig. 1d] |

**Rigor estadístico**
No hay intervalos de confianza, desviación estándar ni semillas. Una sola partición.

**¿Comparación justa?**
Los únicos baselines son sus propios clasificadores. No se compara con ningún método publicado ni con un clasificador simple (p. ej. TF-IDF + LR).

**Ablaciones**
- Voto unánime frente a mayoritario frente a individual [Tabla 6]. El paso de mayoría a unanimidad reduce el FPR un 78 % relativo, lejos de la caída exponencial ideal, señal de que los errores están correlacionados [§2.6].
- Errores más frecuentes: OpenAI ↔ Llama en los tres clasificadores [Tablas 1–3].

**Limitaciones que reconocen los autores**
- No analizan qué rasgos lingüísticos explican las predicciones [§4].
- Pocas familias, pocos tipos de clasificador, solo inglés [§4].

---

## 7. Conclusión y trabajo futuro (descendientes)

**Trabajo futuro propuesto**
- Investigar los rasgos lingüísticos que explican las predicciones [§4].
- Más LLMs, más clasificadores, más idiomas y más tipos de ensemble [§4].
- Entender por qué el clasificador II captura matices que los otros no [§2.6].

**Preguntas abiertas**
- Causa de la similitud DeepSeek-R1 ↔ OpenAI [§3.3, §4].

---

## 8. Lectura crítica

> Esta sección es la valoración del lector, no lo que dice el paper.

**¿Contribuciones claramente separadas del trabajo previo?**
No. El voto unánime con abstención es una técnica estándar y el paper la presenta como "novel method" [Abstract, §4] sin compararla con nada externo.

**Afirmaciones sin respaldo o lenguaje grandilocuente**
- "This work contributes significantly to the area…" [§4] sin comparación con el estado del arte.
- Los resultados no son reproducibles: no se describen las arquitecturas, los datos de entrenamiento ni los hiperparámetros [§2.2].
- La similitud DeepSeek-R1 ↔ OpenAI se sugiere como "striking" sin control: el ensemble está obligado a elegir entre cuatro clases y OpenAI puede ser la clase "por defecto" (también recibe la mayoría de errores de Mixtral).
- Los resultados en modelos no vistos se leen como éxito ("no-agreement" alto), pero 74,2 % de DeepSeek-R1 atribuido con confianza a OpenAI es precisamente un falso positivo de alto coste según su propio marco.

**Limitaciones no mencionadas por los autores**
- No se dice si los prompts de test están separados de los de train; las cifras de ~99,9 % podrían estar infladas por prompts o temas compartidos.
- No se indica la longitud de los textos.
- Las métricas excluyen las abstenciones, lo que mejora las cifras.

**¿Es coherente la historia X → Y → Z → 1?**
La lógica (coste → unanimidad → menos FP) es coherente, pero el "Y" está ausente y el "1" no permite comprobar nada por falta de detalles.

**TL;DR propio**
Ensemble de tres clasificadores no descritos con voto unánime para atribuir textos a cuatro familias de LLM con muy pocos falsos positivos. Las cifras son altísimas, pero sin detalles del método ni de los datos no son verificables. No merece leerse a fondo; como mucho, el dato de que OpenAI ↔ Llama y DeepSeek ↔ OpenAI se confunden.

---

## 9. Relevancia para mi investigación

> Valoración respecto a la misión de `kb/mission/CHALLENGE.md`.

**Qué aporta a la misión**
- Poco. La idea de combinar clasificadores diversos es aplicable a nuestro problema, pero eso ya es práctica estándar.
- La abstención no sirve: la balanced accuracy exige predecir siempre una clase.

**Límites para nuestro caso**
- Clasifica 4 familias, no 18 modelos con variantes del mismo laboratorio.
- No describe el método, así que no se puede replicar ni adaptar.
- No informa de cómo separa prompts entre train y test, que es la clave de nuestra validación.

**Ideas que se podrían probar**
- Un ensemble de modelos diversos (p. ej. TF-IDF de caracteres, TF-IDF de palabras, encoder fine-tuneado) promediando probabilidades, evaluado en la misma GroupKFold. Es una idea genérica, no específica de este paper.

**Relación con la base de conocimiento**
- [[L001]]: trata el mismo problema con mucho más rigor y diagnóstico.
- [[F002]]: nuestra línea de base de referencia para cualquier ensemble.

**Relevancia:** LOW — método no descrito y escenario (4 familias, con abstención) muy alejado de nuestras 18 clases y la balanced accuracy.

## Links

- Misión: [[CHALLENGE]]
- Estado actual: [[ACTIVE]]
- Relacionados: [[L001]], [[F002]]
