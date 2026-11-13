---
id: "L001"
aliases: ["L001"]
type: literature
source_type: "PAPER"
relevance: "HIGH"
source_url: "https://arxiv.org/abs/2502.12150"
date_reviewed: "2026-11-13"
tags: []
---

# L001 — Idiosyncrasies in Large Language Models

## 0. Ficha

| Campo | Valor |
|---|---|
| Autores | Mingjie Sun*, Yida Yin*, Zhiqiu Xu, J. Zico Kolter, Zhuang Liu (CMU, UC Berkeley, UPenn, Princeton) |
| Venue y año | ICML 2025 (PMLR 267); arXiv v2, junio 2025 |
| Enlace (arXiv / DOI) | https://arxiv.org/abs/2502.12150 |
| Código | https://github.com/locuslab/llm-idiosyncrasies |
| PDF local | `papers/2502.12150v2.pdf` |
| Fecha de lectura | 2026-11-13 |

---

## 1. El núcleo

**X — ¿Qué intentan hacer y por qué es relevante?**
Medir y explicar las "idiosincrasias" de los LLMs: patrones propios en sus salidas que permiten saber qué modelo escribió un texto. Plantean la tarea como clasificación N-way (texto → LLM de origen). Sirve para entender procedencia de contenido, similitud entre modelos y efectos del entrenamiento con datos sintéticos [§1, §2].

**Y — ¿Por qué es difícil?**
Los LLMs comparten arquitectura (Transformer), objetivo (next-token) y gran parte de los datos de preentrenamiento, así que a priori podrían escribir de forma indistinguible. Casi todo el trabajo previo distingue humano vs. IA, no IA vs. IA [§1, §2, §5].

**Z — ¿Cuál es su contribución?**
Un marco sencillo: generar respuestas de varios LLMs a los mismos prompts, fine-tunear un modelo de embeddings (LLM2vec + LoRA + cabeza lineal) para predecir el origen y usar transformaciones del texto (barajar palabras/letras, quitar o aislar markdown, parafrasear, traducir, resumir) para aislar qué tipo de información lleva la señal [§2, §3].

**1a — ¿Cómo lo verifican experimentalmente?**
Tres grupos de modelos (5 chat APIs, 4 instruct de 7–9B, sus 4 versiones base), 10K textos de train y 1K de validación por modelo. Resultado titular: 97,1 % de accuracy en la clasificación 5-way de ChatGPT/Claude/Grok/Gemini/DeepSeek (azar 20 %) [Tabla 1a]. Dentro de la familia Qwen2.5 (7B–72B): 59,8 % en 4-way [Tabla 2].

**1b — ¿Hay respaldo teórico?**
No se aborda en el paper.

**2 — Contribuciones que reclaman los autores**
- Demostrar que los LLMs tienen idiosincrasias que un clasificador neuronal detecta con muy alta accuracy, de forma robusta entre combinaciones de modelos, datasets de prompts y controles de prompt [§2].
- Localizar las idiosincrasias en la distribución de palabras, el formato markdown y la semántica [§3].
- Caracterizarlas en lenguaje natural con un LLM como juez [§3.3].
- Implicaciones: las idiosincrasias se heredan vía SFT con datos sintéticos, el marco sirve para medir similitud entre modelos, y los leaderboards por votación son manipulables [§4].

---

## 2. Trabajos relacionados (hermanos)

| Trabajo | Diferencia (supuestos / método) | ¿Comparado en experimentos? | Si no, ¿por qué? |
|---|---|---|---|
| Dataset classification (Torralba & Efros 2011; Liu & He 2025; Zeng et al. 2024) | Clasifican de qué dataset viene una imagen para medir sesgo; aquí se clasifica qué LLM generó un texto | No | Es la inspiración conceptual, no un método competidor [§5] |
| Mansour & Heckel 2024 | Aplica dataset classification a datasets de preentrenamiento de LLMs; aquí se miran las salidas del modelo | No | No se justifica |
| Detección humano vs. IA (DetectGPT, Fast-DetectGPT, GLTR, etc.) | Problema binario humano/IA; aquí IA vs. IA | No | Problema distinto [§5] |
| Authorship attribution neuronal (Uchendu et al. 2020; Antoun et al. 2023; Huang et al. 2024) | Atribuyen texto a generadores concretos, también con modelos fine-tuneados | No | No se justifica; el paper se presenta como estudio de las idiosincrasias, no como mejor atribuidor |
| McGovern et al. 2024 | Fingerprints léxicos/sintácticos de LLMs frente a humanos | No | No se justifica |
| VibeCheck (Dunlap et al. 2025) | Rasgos alineados con el usuario (formal, amable); aquí patrones generalizables que expliquen la accuracy | No | Objetivo distinto [§5] |
| Suzuki et al. 2025 | Variaciones sutiles en el entrenamiento producen salidas distinguibles | No | Trabajo concurrente posterior al preprint [§5] |

---

## 3. Antecedentes (ancestros)

**Conceptos previos necesarios**
- Clasificación de secuencias con encoders preentrenados (BERT, T5, GPT-2, ELMo) y LLM2vec (LLM decoder convertido en encoder de texto).
- LoRA para fine-tuning eficiente.
- TF-IDF + regresión logística como clasificador lineal interpretable.
- Estrategias de decoding (greedy, temperatura, top-k, top-p).

**Definición del problema**
$N$ LLMs $f_1,\dots,f_N$; un conjunto de prompts $\mathcal{D}$; $\mathcal{O}_i$ son las salidas de $f_i$ sobre $\mathcal{D}$. Se entrena un clasificador N-way que, dado un texto $o$, predice $i$. Si todas las $\mathcal{O}_i$ siguieran la misma distribución, la accuracy sería la del azar; la accuracy por encima del azar se usa como medida de idiosincrasia [§2].

**Supuestos inusuales o restrictivos**
- Conjunto cerrado de modelos conocidos de antemano (lo reconocen como limitación) [§6].
- Respuestas largas (hasta 512 tokens) a prompts de chat, con markdown; muestreo por defecto de cada API [Apéndice A.1].
- Mismos prompts para todos los modelos; split train/val idéntico entre LLMs [§2.1].

**¿El escenario del problema es una contribución nueva?**
Parcialmente: la atribución a LLMs concretos ya existía (authorship attribution neuronal); lo nuevo es usarla como herramienta de medida, con el enfoque de "dataset classification", y el análisis sistemático de qué la explica.

---

## 4. Método

**Qué hacen**
1. Generan 11K respuestas por modelo (10K train / 1K val) con prompts de UltraChat (chat e instruct) o de FineWeb (base) [§2.1].
2. Fine-tunean LLM2vec con LoRA y una cabeza lineal sobre el embedding medio de los tokens, con los primeros 512 tokens de cada respuesta [§2, Apéndice A.2].
3. Miden accuracy en validación para cada combinación de modelos (pares y N-way) [Tablas 1, 2].
4. Repiten con controles: instrucciones de longitud/formato en el prompt, truncado a k tokens, distintos decodings, distintos encoders, tamaño de train, y generalización entre datasets de prompts [§2.2].
5. Aplican transformaciones al texto para aislar fuentes de señal: quitar caracteres especiales, barajar palabras, barajar letras, dejar solo markdown (resto → "xxx"), parafrasear, traducir al chino y resumir con GPT-4o-mini [§3].
6. Extraen frases características con TF-IDF (uni- y bigramas) + regresión logística, analizan la primera palabra y la distribución de elementos markdown [§3.1, §3.2].
7. Usan un LLM juez con identidades anonimizadas para describir en lenguaje natural las diferencias (35 pares → 5 viñetas por modelo) [§3.3, Apéndice A.3].

**Por qué lo hacen así**
- LLM2vec: es el mejor de los encoders probados [Tabla 6].
- Transformaciones: cada una destruye un tipo de información (orden, léxico, formato, forma superficial) y la caída de accuracy mide cuánto aportaba [§3].
- TF-IDF + LR: los coeficientes dan un ranking interpretable de frases [§3.1].
- Juez anonimizado: evitar que el juez use conocimiento previo de los modelos [Apéndice A.3].

---

## 5. Configuración experimental

**Instancia del problema**
- Chat APIs: GPT-4o-2024-08-06, Claude-3.5-Sonnet-20241022, Grok-Beta, Gemini-1.5-Pro-002, DeepSeek-Chat (V3), consultados entre nov. 2024 y feb. 2025 [Apéndice A.1].
- Instruct: Llama3.1-8b, Gemma2-9b, Qwen2.5-7b, Mistral-v3-7b (greedy). Base: las mismas versiones base (T=0,6, repetition penalty 1,1) [§2.1, Apéndice A.1].
- Familia Qwen2.5 instruct 7B/14B/32B/72B [Tabla 2].
- Prompts: UltraChat; para generalización también Cosmopedia, LmsysChat y WildChat; para base, FineWeb [§2.1, Tabla 3].

**Detalles de implementación**
- LLM2vec con LoRA (r=16, α=32, dropout 0,05, lr 5e-5); encoders pequeños con fine-tuning completo y búsqueda de lr en {3e-3 … 1e-6} [Apéndice A.2].
- AdamW, weight decay 0,001, 3 épocas, batch 8, cosine decay, warmup lineal 10 %, gradient clip 0,3 [Tabla 11].
- Input truncado a 512 tokens [§2].
- Cómputo: No se aborda en el paper.

---

## 6. Resultados y discusión

**Resultados principales frente a baselines**

| Tarea / dataset | Métrica | Método | Mejor baseline |
|---|---|---|---|
| 5 chat APIs, N-way | accuracy | 97,1 % (LLM2vec) | 92,1 % (GPT-2) [Tabla 6] |
| 4 instruct, N-way | accuracy | 96,3 % | 92,3 % (GPT-2) [Tabla 6] |
| 4 base, N-way | accuracy | 87,3 % | 80,2 % (GPT-2) [Tabla 6] |
| Qwen2.5 7B/14B/32B/72B instruct | accuracy | 59,8 % (azar 25 %) | — [Tabla 2] |
| Qwen2.5-7B base vs. instruct | accuracy | 96,8 % | — [§2.1] |
| Mismo modelo, 5 temperaturas | accuracy | 37,9 % (azar 20 %) | — [§2.2] |
| Chat APIs sin caracteres especiales, TF-IDF palabras uni+bigramas + LR | accuracy | 85,5 % (instruct 83,7 %) | frente a 95,1 % / 93,8 % del fine-tuning en el mismo texto [§3.1] |

Otros resultados clave:
- Pares de modelos: > 90 % en casi todas las combinaciones, muchas ~99 % [Tabla 1].
- Generalización entre datasets de prompts: 88,3–99,8 % [Tabla 3].
- Con control de longitud (≤ 100 palabras, un párrafo): 93,0 %; con control de formato (texto plano): 91,4 % (instruct, frente a 96,3 %) [Tabla 4].
- Truncado: ~50 % con un solo token en chat/instruct; satura hacia 256 tokens; a 64 tokens ronda el 85–90 % (lectura de la Figura 2) [Fig. 2].
- Curva de datos: 40,3 % con 10 muestras en chat; ~80 % con 100; converge hacia 10K [Fig. 3].
- Transformaciones (chat/instruct/base): barajar palabras 88,9/88,9/68,3; barajar letras 39,1/38,6/38,9; solo markdown 73,1/77,7/38,5; parafrasear 91,4/92,2/71,7; traducir 91,8/92,7/74,0; resumir 58,1/57,5/44,7 [Tablas 8, 9, 10].
- Distribución de letras casi idéntica entre modelos; distribución de palabras frecuentes ("the", "and"...) distinta [Fig. 5].
- Frases características: ChatGPT "certainly", "such as", "overall"; Claude "here", "according to", "based on" [Fig. 6, 7].
- SFT con datos sintéticos: fine-tunear dos base con los mismos datos de ChatGPT baja la separabilidad de 96,5 % a 59,8 %; con datos de Llama vs. Gemma, sube a 98,9 % [§4].

**Rigor estadístico**
No se reportan desviaciones estándar, intervalos de confianza ni número de semillas. Una sola partición 10K/1K.

**¿Comparación justa?**
Los encoders pequeños tienen búsqueda de lr; LLM2vec usa un lr fijo [Apéndice A.2]. No se reporta cómputo. No se compara con métodos de atribución previos.

**Ablaciones**
- El orden de palabras y los caracteres especiales aportan poco (−2 a −8 puntos); la distribución de letras no lleva casi señal [Tabla 8].
- El markdown solo ya da 73–78 % en chat/instruct [Tabla 9].
- La semántica sobrevive a parafrasear y traducir (> 90 %), y en parte a resumir (~58 %) [Tabla 10]; consistente con otro LLM reescritor [Tabla 12].
- El encoder importa: de ~90 % (ELMo/BERT/T5) a 97 % (LLM2vec) [Tabla 6].
- Más datos ayudan de forma monótona [Fig. 3].

**Limitaciones que reconocen los autores**
- Solo arquitecturas Transformer [§6].
- No explican cómo surgen las idiosincrasias en el entrenamiento [§6].
- Conjunto cerrado de modelos; un ensayo 10-way da 92,2 % [§6].

---

## 7. Conclusión y trabajo futuro (descendientes)

**Trabajo futuro propuesto**
- Generalizar a state space models y LLMs de difusión [§6].
- Entender qué parte del entrenamiento produce las idiosincrasias [§6].
- Escenarios con muchos modelos candidatos o desconocidos [§6].
- Relación con la destilación [§6].

**Preguntas abiertas**
- Por qué modelos de la misma familia son mucho menos separables (59,8 %).
- Robustez de los leaderboards por votación frente a ataques basados en atribución [§4].

---

## 8. Lectura crítica

> Esta sección es la valoración del lector, no lo que dice el paper.

**¿Contribuciones claramente separadas del trabajo previo?**
Bastante. Atribuyen la idea a dataset classification y citan la atribución neuronal. Pero "unveil" en el abstract exagera la novedad: la atribución entre LLMs ya existía [§5].

**Afirmaciones sin respaldo o lenguaje grandilocuente**
- Sin semillas ni varianza, diferencias de 1–2 puntos (p. ej. 97,1 frente a 97,7 en pares, Tabla 1) no son interpretables.
- "Los clasificadores aprenden patrones muy robustos y transferibles" [§2.1]: solo se prueba entre datasets de prompts en inglés con los mismos modelos.
- "La semántica importa más que la elección exacta de palabras" [§3.3]: el parafraseo conserva formato y estructura (lo admiten en el Apéndice C), así que no aísla la semántica.

**Limitaciones no mencionadas por los autores**
- Textos largos (hasta 512 tokens) con markdown. En textos cortos y en prosa la señal debe ser bastante menor; el truncado de la Fig. 2 lo sugiere pero no se separa del efecto del markdown.
- Los modelos de cada grupo son de laboratorios distintos; el caso difícil (variantes del mismo laboratorio) solo se toca con Qwen, con un resultado modesto.
- No se informa cómo se separan train y val respecto a los prompts: solo que el split es el mismo para todos los LLMs, lo que sugiere (sin confirmarlo) que es por prompt.
- Las versiones de los modelos son de finales de 2024; las idiosincrasias pueden cambiar entre versiones.

**¿Es coherente la historia X → Y → Z → 1?**
Sí. La pregunta es clara y los experimentos de transformación responden a la pregunta del "porqué" de forma ordenada. El salto débil está en las implicaciones (§4), que son más ilustrativas que concluyentes.

**TL;DR propio**
Un fine-tuning simple de un encoder distingue LLMs de laboratorios distintos casi perfectamente, y la señal está sobre todo en la elección de palabras, las aperturas, el markdown y el contenido, no en las letras ni en el orden. Los casos difíciles (mismo laboratorio, textos cortos, sin formato) están poco explorados y son justo los nuestros. Merece leerse a fondo por el diseño de las transformaciones de diagnóstico, no por el método de clasificación.

---

## 9. Relevancia para mi investigación

> Valoración respecto a la misión de `kb/mission/CHALLENGE.md`.

**Qué aporta a la misión**
- Dice dónde buscar la señal: palabras y frases (sobre todo la apertura), markdown y contenido; casi nada en la distribución de letras. Nuestro baseline usa n-gramas de caracteres, que cubren las palabras solo de forma indirecta.
- Da un orden de magnitud de lo que aporta pasar de un modelo lineal sobre TF-IDF de palabras (85,5 %) a un encoder fine-tuneado (95,1 %) en el mismo texto [§3.1].
- Confirma que separar modelos del mismo laboratorio es mucho más difícil (Qwen 59,8 %), igual que el 52 % de errores dentro del laboratorio de [[F002]].
- Las transformaciones (barajar palabras, quitar puntuación, solo markdown) son un protocolo de diagnóstico que podemos aplicar a nuestros 18 modelos.

**Límites para nuestro caso**
- Sus textos son respuestas largas (≤ 512 tokens) con markdown; los nuestros son pasajes de 40–250 palabras, y no sabemos aún cuánto markdown tienen.
- Ellos tienen 10K textos por modelo; nosotros unos 150 por modelo (2.699 / 18). Su Fig. 3 muestra que la accuracy cae mucho con pocos datos.
- Sus grupos tienen 4–5 modelos casi todos de laboratorios distintos; nosotros, 18 modelos con varias variantes por laboratorio (gpt-5.6 luna/sol/terra).
- No reportan varianza ni validación agrupada explícita por prompt; sus cifras no son comparables con nuestra GroupKFold.
- El resultado de temperaturas (37,9 % con 5 temperaturas del mismo modelo) avisa de que, si algunas variantes de un laboratorio difieren solo en el decoding, separarlas puede ser casi imposible.

**Ideas que se podrían probar**
- TF-IDF de palabras uni+bigramas (combinado con el de caracteres) en el mismo pipeline LR, como mejora barata del baseline.
- Rasgos explícitos de apertura (primera palabra / primeras k palabras) y de formato (conteo de negritas, cabeceras, listas, viñetas), normalizados por longitud.
- Fine-tunear un encoder pequeño (DeBERTa/ModernBERT o similar) que quepa en el límite de 9 h offline, comparándolo con TF-IDF en la misma GroupKFold y vigilando el poco dato por clase.
- Aplicar sus transformaciones como diagnóstico (barajar palabras, quitar puntuación, solo markdown) para saber qué señal domina en nuestros datos.

**Relación con la base de conocimiento**
- [[F002]]: baseline 0,312 y la mitad de los errores dentro del laboratorio; el paper explica por qué ese es el caso difícil.
- [[H002]]: la hipótesis de errores dentro del laboratorio coincide con el resultado de Qwen [Tabla 2].
- [[F001]]: la longitud no identifica al modelo; coherente con que el control de longitud apenas afecte en el paper [Tabla 4].

**Relevancia:** HIGH — es la referencia directa de nuestra tarea (atribución texto → LLM) y señala qué rasgos probar, aunque en un escenario más fácil que el nuestro.

## Links

- Misión: [[CHALLENGE]]
- Estado actual: [[ACTIVE]]
- Relacionados: [[F001]], [[F002]], [[H002]], [[L002]]
