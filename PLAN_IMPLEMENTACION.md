# Generador de exámenes médicos desde MedQuAD — plan de implementación

Proyecto de ingeniería de IA generativa. Reutiliza el corpus MedQuAD del proyecto
anterior de RAG, pero con otra tarea: **generar preguntas de opción múltiple**
con un modelo afinado por fine-tuning, servido en una app local.

## Artefactos y flujo

```
notebooks/01_limpieza.ipynb        →  medqa_clean.parquet
notebooks/02_dataset_mcq.ipynb     →  chunks.parquet   (la BD de la app)
                                      mcq_train/val/test.parquet  (para entrenar)
notebooks/03_finetuning.ipynb      →  adapter/  (33 min en la GPU local)
app/app.py                         ←  adapter/ + chunks.parquet
```

Los tres notebooks corren **una sola vez**. La app no entrena ni genera dataset:
carga el adaptador y la tabla de chunks, y responde.

---

# Notebook 01 — Limpieza

Es el notebook `01_data_preparation_and_cleaning.ipynb` del proyecto anterior,
que ya hace todo el trabajo. Se reutiliza con **tres cambios**.

## Lo que ya hace y se conserva

Once pasos sobre `HoangHa/MedQuaD` (47,441 filas):

| Paso | Qué hace |
|---|---|
| 1 | descarta filas sin `answer` y columnas UMLS |
| 2 | elimina respuestas no informativas (< 50 chars, "topics", "faqs") |
| 3 | quita preguntas incrustadas al inicio del campo `answer` |
| 4 | deduplica pares `question`+`answer` idénticos |
| 5 | deduplica preguntas repetidas, conservando la respuesta más larga |
| 6 | reasigna `question_id` único |
| 7 | normaliza espacios, tabs y saltos de línea |
| 8 | limita respuestas genéricas a 5 apariciones (`answer_is_generic`) |
| 9 | marca `answer_needs_chunking` si supera 1,500 chars |
| 10 | rellena `question_focus` nulo infiriéndolo de la URL |
| 11 | construye splits train/val/test estratificados por `question_type` |

## Cambio 1 — no descartar `category`

El paso 1 hace:

```python
df = df.drop(columns=["synonyms", "umls_cui", "umls_semantic_types",
                      "umls_semantic_group", "category"])
```

`category` conviene conservarla: es una dimensión extra para el selector de temas
de la app, además de `question_type` y `question_focus`. Las cuatro columnas UMLS
sí se descartan igual.

## Cambio 2 — los splits del paso 11 no sirven acá

Aquellos splits se hicieron a nivel de **pregunta** y para evaluar retrieval.
Este proyecto necesita splits a nivel de **chunk** y sin fuga entre train y test,
y eso se resuelve en el notebook 02. El paso 11 se puede dejar corriendo (no
molesta) o comentar; lo que importa es que la salida que usa el notebook 02 es
`medqa_clean.parquet`, **no** `medqa_train.parquet`.

## Cambio 3 — la salida es el corpus completo

El notebook 02 del proyecto anterior chunkeaba solo `train`, porque el resto se
reservaba para evaluar. Acá la app tiene que poder generar preguntas de
**cualquier** tema, así que se chunkea `medqa_clean.parquet` entero.

## Salida

`medqa_clean.parquet` con:

```
question_id · question · answer · question_focus · question_type · category
document_id · document_source · document_url
answer_is_generic · answer_needs_chunking · answer_len
```

---

# Notebook 02 — Dataset de preguntas de opción múltiple

El notebook más importante. Produce las dos tablas que consumen el resto del
proyecto.

## 2.1 Chunking

Se reutiliza `RecursiveCharacterTextSplitter` con la estrategia **B_512**
(`chunk_size=512`, `chunk_overlap=64`), que fue la ganadora en el proyecto
anterior. La razón vale también acá, aunque la tarea sea otra: 512 caracteres es
aproximadamente un hecho médico completo, que es la unidad correcta para formular
**una** pregunta. Con 1024 el modelo tiene que elegir arbitrariamente sobre qué
preguntar y las preguntas salen vagas.

Un cambio necesario respecto del código anterior: `chunk_id` allá era el índice
dentro del documento (0, 1, 2...), que no es único global. Acá hace falta una
clave primaria:

```python
df_chunks["chunk_uid"] = (
    df_chunks["question_id"] + "__" + df_chunks["chunk_id"].astype(str)
)
```

## 2.2 Filtro de aptitud del chunk

No todo fragmento sirve para formular una pregunta. Antes de gastar llamadas a la
API se descartan:

| Descarte | Criterio |
|---|---|
| Muy corto | menos de 200 caracteres: no alcanza para una pregunta con respaldo |
| Arranca cortado | empieza en minúscula o en medio de una oración (continuación de otro chunk) |
| Casi todo enumeración | más del 60% de las líneas empiezan con viñeta o guion |
| Sin contenido afirmativo | no contiene ningún verbo conjugado (heurística simple con regex) |
| Genérico | `answer_is_generic == True` |

Este filtro es barato y evita pagar por chunks que iban a producir basura.

## 2.3 Muestreo para la API

Aquí está la decisión de alcance más importante del proyecto:

> **La tabla de chunks completa es la BD de la app. El subconjunto que pasa por
> la API existe únicamente para entrenar al modelo.**

Eso significa que **no** hay que generar preguntas para los ~38,000 chunks. Se
toma una muestra estratificada por `question_type` de entre **4,000 y 6,000**
chunks, que es de sobra para LoRA. El resto de los chunks nunca ve la API: cuando
un usuario pida preguntas sobre ellos, las genera el modelo afinado. Eso es
precisamente lo que justifica el fine-tuning.

Costo estimado con `gpt-4o-mini` (unos 600 tokens de entrada y 200 de salida por
chunk): alrededor de **1 dólar por cada 5,000 chunks**.

## 2.4 Generación con el LLM

Un llamado por chunk. El punto crítico —el que resuelve el problema de las
respuestas larguísimas— es que **la respuesta correcta no es el chunk, sino una
afirmación corta extraída del chunk**.

```
Eres un docente de medicina que redacta preguntas de examen.

A partir del FRAGMENTO, escribe UNA pregunta de opción múltiple.

Reglas obligatorias:
- La respuesta correcta debe estar explícitamente respaldada por el fragmento.
- Las tres opciones incorrectas deben ser plausibles para alguien que no
  estudió el tema, pero inequívocamente falsas según el fragmento.
- Las CUATRO opciones deben tener longitud similar y NINGUNA puede pasar de
  15 palabras. Si el fragmento no permite respuestas cortas, devuelve
  {"descartar": true}.
- La pregunta debe entenderse sola. Prohibido referirse al texto
  ("según el fragmento", "el texto menciona", "de acuerdo al documento").
- No inventes datos que no estén en el fragmento.
- dificultad: "facil" si la respuesta es un dato literal, "media" si exige
  relacionar dos partes, "dificil" si exige distinguir entre conceptos
  parecidos.

Responde SOLO con este JSON:
{"pregunta": "...", "correcta": "...", "incorrectas": ["...", "...", "..."],
 "dificultad": "facil|media|dificil"}
```

La salida `{"descartar": true}` es deliberada: es mejor perder un chunk que meter
una pregunta mala en el set de entrenamiento, porque el modelo aprendería a
imitarla.

Detalles de implementación: `temperature=0.7` (algo de variedad para que el
dataset no sea monótono), concurrencia con `ThreadPoolExecutor` de 8 a 16 hilos,
reintentos con backoff, y guardado incremental a JSONL para poder reanudar si se
corta.

## 2.5 Filtros de calidad automáticos

Sobre cada JSON devuelto, en orden de costo creciente:

| # | Filtro | Rechaza si |
|---|---|---|
| 1 | JSON válido | no parsea, o le faltan campos |
| 2 | Cuatro opciones distintas | dos opciones coinciden tras normalizar (minúsculas, sin puntuación) |
| 3 | Longitud máxima | alguna opción supera 15 palabras |
| 4 | Longitud pareja | `len(más larga) / len(más corta) > 2.5` |
| 5 | Distractor contenido en el chunk | un distractor aparece textualmente en el fragmento |
| 6 | Correcta con respaldo | la respuesta correcta no tiene solapamiento léxico con el chunk |
| 7 | Pregunta autocontenida | contiene "según el texto", "el fragmento", "el documento", "mencionado" |
| 8 | Juez LLM (muestra) | sobre 200 casos al azar: ¿hay exactamente una respuesta correcta? |

El **filtro 4** merece explicación: si la opción correcta es siempre la más larga
y detallada, se puede acertar sin saber nada del tema. Es el sesgo más común en
exámenes generados automáticamente, y se corrige midiendo.

El **filtro 5** es el más importante para la seguridad de la pregunta: si un
distractor está literalmente en el fragmento, es probable que también sea cierto,
y entonces la pregunta tiene dos respuestas válidas.

El notebook debe **reportar la tasa de descarte por filtro**. Si un filtro
descarta el 40%, el prompt está mal y hay que corregirlo antes de escalar.

## 2.6 Splits sin fuga

Las preguntas aprobadas se dividen en train/val/test, con una restricción que no
es negociable:

> **Todos los chunks de un mismo `document_id` van al mismo split.**

Sin eso, dos chunks solapados del mismo documento —recordá que hay 64 caracteres
de overlap— caerían en train y en test, y la evaluación mediría memorización en
vez de generalización. El split se hace agrupando por documento
(`GroupShuffleSplit` de scikit-learn), estratificando por `question_type` en lo
posible: 80% train, 10% validación, 10% test.

## 2.7 Salidas

**`chunks.parquet`** — la base de datos que consulta la app. Todos los chunks
aptos, sin preguntas:

```
chunk_uid · chunk_text · question_focus · question_type · category
document_source · document_url · n_chars
```

Consultarla es un filtro exacto por columna con pandas. **No hace falta base
vectorial**: el usuario elige de una lista y `question_type`, `category` y
`question_focus` ya son columnas. Una búsqueda semántica solo haría falta si se
permitiera escribir un tema libre, y eso queda como mejora opcional.

**`mcq_train.parquet` / `mcq_val.parquet` / `mcq_test.parquet`** — para el
notebook 03:

```
chunk_uid · chunk_text · pregunta · correcta · incorrectas (lista de 3)
dificultad · question_type · question_focus · document_url
```

Se conservan `chunk_uid` y `document_url` para trazabilidad: cada pregunta puede
rastrearse hasta su fragmento y hasta la página del NIH de la que salió. La app
lo usa para mostrar la fuente al corregir.

---

# Notebook 03 — Fine-tuning

**Se corre en la GPU local, no en Colab.** El notebook detecta el entorno y
funciona igual en los dos, pero la medición dice que Colab no hace falta:

| Medido en la prueba de humo | |
|---|---|
| Memoria pico entrenando | **9.63 GB** de 16.3 |
| Tiempo por paso | 0.39 s |
| Tres épocas sobre 3,385 ejemplos | **33 minutos** |

Eso elimina la transferencia del adaptador, el riesgo de versiones distintas
entre entornos, y la dependencia de qué GPU toque en Colab.

## Modelo

**`Qwen/Qwen3-4B-Instruct-2507`**, verificado en el hub: arquitectura
`Qwen3ForCausalLM`, solo texto.

Una advertencia importante sobre esto. Toda la familia **Qwen3.5 es multimodal**
—`Qwen3_5ForConditionalGeneration`, con `vision_config` y tokens de imagen y
video—, incluido el `Qwen3.5-0.8B` del ejemplo del profesor. Para generar texto,
la torre de visión es peso muerto: ocupa memoria, complica decidir qué módulos
toca el LoRA, y `AutoModelForCausalLM` puede ni siquiera cargarlo. Además
`Qwen3.5-4B-Instruct` **no existe**: en Qwen3.5 el instruct es `Qwen3.5-4B` a
secas y el crudo es `-Base`.

Si por algún motivo hiciera falta algo más chico, la alternativa dentro de la
misma familia de texto es `Qwen/Qwen3-8B` hacia arriba, o modelos Qwen3 menores.

## Stack, sin Unsloth

```
transformers · peft · trl · accelerate · datasets · bitsandbytes
```

Es el stack estándar y el que se dictó en clase. Tres razones para preferirlo
sobre Unsloth: el adaptador queda 100% compatible al moverlo a la máquina local,
es lo que enseñó el profesor, y con 16 GB en un modelo de 4B no hace falta la
optimización de memoria que Unsloth aporta. El costo es entrenar quizá el doble
de lento, que para 4,000 ejemplos son unos 40 minutos en vez de 20.

## Pasos

1. **Prueba de humo primero.** Hecha: 10 pasos con 50 ejemplos validaron que el
   modelo acepta LoRA, que la pérdida baja (1.55 → 0.42), que el adaptador se
   guarda (144 MB) y que **recarga en un proceso nuevo**. También que
   `disable_adapter()` funciona, que es lo que hace viable el panel comparativo
   de la app con un solo modelo en memoria (8.31 GB).
2. Cargar `mcq_train.parquet` (subido a Colab o desde Drive).
3. Formatear al chat template de Qwen: `system` con la instrucción, `user` con el
   fragmento, `assistant` con el JSON esperado. Entrenar **solo sobre la
   respuesta** (`DataCollatorForCompletionOnlyLM`), no sobre el prompt.
4. Cargar el modelo en 4-bit con `BitsAndBytesConfig`, o en `bfloat16` si la GPU
   de Colab lo permite.
5. LoRA: `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, sobre los módulos de
   atención y MLP.
6. `SFTTrainer` con 2 o 3 épocas, monitoreando la pérdida de validación para
   detectar sobreajuste.
7. **Evaluación comparativa** sobre `mcq_test.parquet`, que es la parte que
   justifica el proyecto entero.
8. Guardar el adaptador junto a la aplicación. (Si se corriera en Colab, hay
   que copiarlo a Drive: el disco de Colab se borra al desconectar.)

## La evaluación

Se compara **el modelo base con few-shot** contra **el modelo afinado**, ambos
sobre chunks del split de test que ninguno vio:

| Métrica | Por qué importa |
|---|---|
| JSON válido al primer intento | si falla, la app tiene que reintentar; es donde el fine-tuning suele ganar más |
| Tokens de prompt | el base necesita 3-4 ejemplos en el prompt (~1,200 tokens); el afinado, ~150 |
| Segundos por pregunta | consecuencia de lo anterior |
| Tasa de aprobación de los filtros 2 a 7 | calidad estructural de la pregunta |
| Juez LLM sobre 100 casos | ¿respuesta única? ¿distractores plausibles? ¿respaldo en el chunk? |

No doy por sentado que el afinado gane en todas. Hay que medirlo: si le armás un
few-shot muy bueno al base, puede que también produzca JSON limpio. Las métricas
donde la diferencia salga más nítida son las que van al panel de la app.

## Salida

```
adapter/
├── adapter_config.json
├── adapter_model.safetensors      (60-200 MB según el rank)
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

Más un `metricas_evaluacion.json` con los resultados de la comparación, que la
app lee para el panel.

**Ejecutado.** El adaptador vive en `app/adapter/` —junto a la aplicación, no en
la raíz— y pesa 276 MB (r=32). El mejor checkpoint es `checkpoint-900`, con
`eval_loss` 0.3506 en la época 1.99.

Una nota operativa que costó cara: **ejecutar el notebook con
`nbconvert --execute` sin `--allow-errors`**. Si una celda falla, nbconvert
descarta todas las salidas, incluidas las de las celdas que sí corrieron. Se
perdieron 2h30 de resultados ya calculados por esto.

---

# Notebook 04 — Parámetros de generación

Separado del 03 a propósito: el 03 elige **cómo se entrena** el modelo, el 04
elige **cómo se muestrea de él**, con el adaptador ya congelado.

Barre `do_sample`, `temperature` y `repetition_penalty` sobre 20 fragmentos ×
3 repeticiones × 7 configuraciones = 420 generaciones (~48 minutos). `top_p` y
`top_k` quedan fijos en 0.9 y 50 porque recortan la misma cola de la
distribución que la temperatura.

Mide dos cosas enfrentadas: **formato** (JSON válido y estructura completa) y
**variedad** (preguntas distintas al regenerar tres veces el mismo fragmento).
Cuenta además las generaciones **truncadas** para validar `max_new_tokens`.

**Resultado:** las siete configuraciones dan 60/60 de formato, cero
truncamientos. La tensión formato-variedad no existe en este modelo.

Como el barrido no mide veracidad, se midió aparte con
`scripts/comparar_temperatura.py`: 0.7 y 1.0 sobre los mismos 32 fragmentos dan
**87.1% sin defectos las dos**. Se eligió 0.7. Detalle en
`INSIGHTS_Y_DECISIONES.md` §5.c.

---

# La aplicación

```
app/
├── app.py              UI de Streamlit: simulacro, corrección, revisión
├── modelo.py           carga, generación por lotes, auto-verificación
├── banco.py            consultas a chunks.parquet + filtro de utilidad
├── validacion.py       filtros de forma + prompts de verificación
├── panel.py            panel "bajo el capó" y juez externo gpt-4o
└── adapter/            el LoRA + metricas_evaluacion.json
```

`validacion.py` es código compartido con el notebook 02, no duplicado: los mismos
filtros que limpiaron el dataset de entrenamiento validan lo que el modelo genera
en vivo. Si una pregunta no pasa, se regenera.

## Las tres capas de validación

| Capa | Dónde | Qué ve | Qué NO ve |
|---|---|---|---|
| `es_util()` | `banco.py`, al cargar | fragmentos que solo remiten a otro sitio | todo lo demás |
| `revisar_forma()` | `validacion.py` | forma de la pregunta: opciones repetidas, longitudes, deícticos | el contenido |
| auto-verificación | `modelo.py` | si la correcta está respaldada, si un distractor es cierto | defectos de utilidad |

**La auto-verificación la ejecuta el modelo BASE, no el afinado.** Está medido:
ante el prompt de verificación el afinado no verifica, regenera. Detalle en
`INSIGHTS_Y_DECISIONES.md` §5.d. `disable_adapter()` apaga el LoRA sin cargar un
segundo modelo, así que no cuesta memoria.

## Flujo: simulacro de examen, no formulario

En vez de "elegí categoría y cantidad, esperá, recibí 50 preguntas":

1. El estudiante elige tema (por `question_type`, `category` o `question_focus`)
   y dificultad, y arranca.
2. Las preguntas aparecen **de a una**. Responde antes de ver la siguiente.
3. Al terminar: puntaje, y por cada error la explicación **con el enlace al
   documento del NIH** del que salió la pregunta.
4. Botón **"otro examen del mismo tema"**, que genera preguntas distintas.

Ese botón es lo que demuestra que hay generación real y no una tabla: un banco
pregenerado se repetiría. Ojo con el matiz, por si el jurado pregunta: demuestra
que hay un modelo generando, **no** que el fine-tuning haya sido mejor que el
base. Eso lo demuestra el panel.

## Latencia: generación progresiva

Generar 50 preguntas de a una son unos 3 minutos, lo cual arruina una demo. La
solución no es solo procesar por lotes, sino **cuándo mostrar**:

```
t=0s    se genera la pregunta 1        →  se muestra
t=3s    el estudiante lee y responde   |  se generan 2..10 de fondo
t=25s   responde, pasa a la 2          →  ya estaba lista
```

Un estudiante tarda entre 15 y 30 segundos por pregunta; en ese rato la GPU
genera varias. La latencia percibida es la de la primera pregunta —dos o tres
segundos— y de ahí en adelante nada se siente lento. En Streamlit se arma con un
hilo de fondo y una cola en `st.session_state`.

Complementos: lotes de 8 a 16 en una sola llamada a `generate`, y
`max_new_tokens` ajustado a unos 200 para que el modelo no divague.

## Panel "Bajo el capó"

Un `st.expander` cerrado por defecto. El estudiante no lo abre; se abre en la
sustentación. Adentro, un fragmento fijo y un botón que genera con los dos
modelos en dos columnas.

Y acá está el detalle que lo hace barato: **no hay que cargar dos modelos**. LoRA
es un parche sobre el mismo base, así que se prende y se apaga:

```python
with model.disable_adapter():
    salida_base = model.generate(...)      # Qwen crudo + few-shot
salida_afinada = model.generate(...)       # Qwen + adaptador
```

Memoria adicional: prácticamente cero. Y como compara **una** pregunta, tarda
segundos.

Contadores en vivo, acumulados durante toda la sesión: JSON válido al primer
intento, reintentos, tokens de prompt, segundos por pregunta. Los definitivos se
eligen según lo que haya salido en la evaluación del notebook 03.

---

# Riesgos y orden de trabajo

## Lo que puede salir mal

| Riesgo | Mitigación |
|---|---|
| ~~El adaptador de Colab no carga en local~~ | **descartado**: se entrena en local. Verificado igual que guarda y recarga |
| ~~Blackwell con bitsandbytes~~ | **descartado**: torch 2.11+cu128 soporta sm_120 y el modelo entra en bf16 (8.04 GB). No se cuantiza |
| El prompt de generación produce mucha basura | correr 200 chunks primero y mirar las tasas de descarte antes de escalar |
| El fine-tuning no gana en ninguna métrica | también es un resultado presentable, pero conviene saberlo temprano |
| Versiones distintas Colab/local | `pip freeze` de torch, transformers, peft, trl, bitsandbytes |

## Una objeción previsible: "esto es destilación"

Alguien va a preguntar si el proyecto no es simplemente destilar `gpt-4o-mini`
en un modelo chico. La respuesta es que sí, exactamente eso, y conviene decirlo
con su nombre —destilación de conocimiento mediante datos sintéticos— en vez de
esquivarla.

Lo defendible no es negarlo, sino lo que se gana con ello:

- El modelo resultante corre **local**, sin API ni conexión, con costo cero por
  consulta.
- Cubre los ~32,000 chunks que **nunca pasaron por la API**: ahí el maestro no
  llegó y el alumno sí.
- La evaluación mide **cuánto se acerca el alumno al maestro**, que es
  precisamente la pregunta interesante de una destilación.

Lo que no se puede afirmar es que el modelo afinado sea mejor que `gpt-4o-mini`
generando preguntas. No lo es, y no se midió contra él. Se midió contra el mismo
Qwen sin afinar, que es la comparación que responde "¿sirvió entrenar?".

## Orden ejecutado

1. ✅ **Prueba de humo de inferencia local.** Dio 23.4 tok/s secuencial y 199.8
   en lotes de 8. Confirmó que no hace falta cuantización.
2. ✅ **Notebook 01**, con los tres cambios.
3. ✅ **Notebook 02 hasta 2.4** sobre 201 fragmentos. Aquí se detectó que el
   modelo ignoraba la válvula de escape, y se corrigió el prompt.
4. ✅ **Notebook 02 completo**, 5,250 fragmentos, USD 0.90.
5. ✅ **Notebook 03**: barrido de 6 configuraciones, entrenamiento final de 2
   épocas, comparación base-vs-afinado y juez maestro-vs-alumno.
6. ✅ **Notebook 04**: barrido de parámetros de generación.
7. ✅ **Aplicación** y auditoría de un examen real con juez externo.

Se saltó el paso «prueba de humo del ciclo Colab → local» porque se decidió
entrenar en local: no hay transferencia que probar.

## Lo que el orden real enseñó

El paso 3 —mirar a mano 20 preguntas de un piloto pequeño antes de escalar— fue
el de mayor retorno de todo el proyecto. Ahí se detectó el fallo del prompt que,
sin corregir, habría contaminado las 5,250 generaciones.

El paso equivalente que **faltó** fue probar la auto-verificación contra entradas
que debía rechazar antes de darla por buena. Estuvo inerte durante toda la
construcción de la app, y solo se descubrió al desconfiar de un contador en cero.
