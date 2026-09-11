# Contexto del proyecto — para retomar desde cero

Documento de traspaso. Si se pierde el hilo de la conversación, esto reconstruye
todo lo decidido y por qué. El **qué hacer** está en
[PLAN_IMPLEMENTACION.md](PLAN_IMPLEMENTACION.md); acá está el **por qué**, lo que
se descartó, y lo que ya se verificó.

---

## 1. Qué es

Generador de preguntas de examen de opción múltiple sobre medicina, a partir del
corpus MedQuAD, usando un modelo de lenguaje afinado con fine-tuning y servido en
una aplicación local.

El estudiante elige un tema, la app genera un simulacro de examen, él responde, y
recibe puntaje con la fuente del NIH de cada pregunta.

## 2. Contexto académico y restricciones

- Proyecto del diplomado, curso de **ingeniería de IA generativa de texto**.
- **Independiente** del Proyecto Integrador anterior (asistente RAG médico), pero
  reutiliza su corpus y parte de su código de preparación de datos.
- **Fine-tuning es obligatorio** según el profesor. Few-shot prompting también es
  válido y puede combinarse — de hecho se usa como línea base para comparar.
- **Despliegue local únicamente**, autorizado por el profesor. No hace falta nube
  ni Hugging Face Spaces. Se ejecuta con `streamlit run app.py` durante la
  sustentación.
- **Colab está permitido** para entrenar, pero **no hace falta**: medido, el
  entrenamiento completo entra en la GPU local (9.63 GB de 16.3) y tarda 33
  minutos. El notebook 03 corre en ambos entornos.

## 3. Relación con el proyecto anterior

El proyecto anterior vive en
`D:\ANTIGRAVITY\proyectos\proyecto_integrador_diplomado_ia` y está desplegado
como repositorio público `medquad-rag-api`. Era un chatbot RAG médico:
Supabase + pgvector, FastAPI + Gradio, OpenAI, con guardrails y memoria.

De ahí se reutiliza:

- **`notebooks/01_data_preparation_and_cleaning.ipynb`** — los 11 pasos de
  limpieza de MedQuAD. Es lo más valioso que se hereda.
- **El código de chunking** del notebook 02 (`RecursiveCharacterTextSplitter`,
  estrategia B_512 con solape de 64).

Los notebooks del proyecto anterior están copiados en `notebooks/` de este
proyecto, como referencia. Los notebooks 03, 04 y 05 (RAG, evaluación, Gradio)
**no** se usan acá: son de la otra tarea.

Lo que **no** se reutiliza: la base vectorial, los embeddings, el reranking, los
guardrails. Esta tarea no es retrieval.

## 4. Estado actual — proyecto completo (2026-09-09)

**Completado:** notebooks 01 a 05 ejecutados con sus salidas, adaptador
entrenado, aplicación Streamlit funcionando y auditada de punta a punta.

**Pendiente:** nada bloquea la entrega. Queda como mejora de medición validar
`revisar_forma()` contra un juicio humano etiquetado a mano.

```
proyecto_ing_iagen/
├── venv/                    Python 3.12 + torch 2.11.0+cu128
├── .env                     OPENAI_API_KEY (en .gitignore)
├── data/
│   ├── medqa_clean.parquet      14,528 filas
│   ├── chunks.parquet           28,294 fragmentos + tipo_es/tema_es  ← BD DE LA APP
│   ├── muestra_api.parquet       5,250 fragmentos
│   ├── mcq_crudo.jsonl           5,250 respuestas crudas del modelo
│   ├── mcq_train.parquet         3,614 preguntas   (eran 3,385 antes de
│   ├── mcq_val.parquet             433 preguntas    recalibrar los filtros)
│   ├── mcq_test.parquet            436 preguntas
│   ├── temas_es.json             3,794 traducciones
│   ├── juez_calidad.json         veredictos sobre 150 preguntas
│   ├── barrido_hiperparametros.csv   las 6 configuraciones del notebook 03
│   ├── barrido_generacion.csv        las 7 configuraciones del notebook 04
│   ├── auditoria_examen.json         auditoría de un examen real + sonda
│   ├── comparacion_temperatura.json  0.7 contra 1.0, mismos fragmentos
│   └── comparacion_prompts_*         notebook 05: cinco variantes y control
├── notebooks/  01 a 05  (los cinco ejecutados, con salidas)
├── checkpoints/  checkpoint-900 (el mejor) y checkpoint-904
├── app/
│   ├── adapter/         el LoRA entrenado, 276 MB + metricas_evaluacion.json
│   ├── app.py           interfaz del simulacro
│   ├── modelo.py        carga, generación por lotes, auto-verificación
│   ├── banco.py         consultas a chunks.parquet + filtro de utilidad
│   ├── validacion.py    filtros de forma + prompts de verificación
│   └── panel.py         panel "bajo el capó" y juez externo
└── scripts/    generadores de notebooks, pruebas de humo, jueces, auditoría
```

### Resultados finales

| Métrica | Valor |
|---|---|
| Configuración ganadora | r=32, alpha=64, lr=2e-4, los 7 módulos |
| Mejor `eval_loss` | 0.3506 (época 1.99, `checkpoint-900`) |
| Estructura JSON, afinado vs base | 60/60 contra 57/60 |
| Tokens de prompt, afinado vs base | 135 contra 693 (**5.1x menos**) |
| Sin defectos: alumno vs maestro | 78.4% contra 79.3% (**empate**) |
| Auditoría de un examen real de 10 | 10/10 sin defectos tras arreglar el verificador |
| Temperatura 0.7 contra 1.0, mismos fragmentos | 87.1% contra 87.1%: sin diferencia detectable |
| Contenido: afinado contra el base con las reglas del maestro (notebook 05) | 76.7% contra 56.7%, p < 0.0001 |
| Contenido: afinado contra few-shot y contra un prompt mínimo | 76.7% contra 70.0% (p = 0.12) y 72.7% (p = 0.39): sin diferencia demostrable |
| Contra el maestro | el afinado no se distingue (p = 0.38); el few-shot queda por debajo (p = 0.014) |
| Estructura rota: afinado contra base | 0 de 150 contra 4, 7 y 25 según el prompt |

### Entorno verificado

| | |
|---|---|
| GPU | RTX 5070 Ti, 16,303 MiB, **sm_120** |
| torch | 2.11.0+cu128 — soporta sm_120 nativamente, **sin plan B** |
| transformers | **5.16.1** (la v5, no la 4.x) |
| Modelo | `Qwen/Qwen3-4B-Instruct-2507`: **8.04 GB** en bf16, 23.4 tok/s, **199.8 tok/s en lotes de 8 (8.5x)** |

El modelo base ya produce JSON válido **sin few-shot**: esa métrica puede no
discriminar en la comparación base contra afinado.

### El pipeline completo

```
47,441 filas crudas
  → 14,528  tras los 11 pasos de limpieza          (nb 01)
  → 47,771  fragmentos de 512 caracteres           (nb 02 §2.1)
  → 28,294  aptos tras el filtro sintáctico        (§2.2, 59%)
  →  5,250  muestreados para la API                (§2.3)
  →  4,908  preguntas generadas                    (§2.4, USD 0.79)
  →  4,269  aprobadas por los filtros de calidad   (§2.5, 87%)
  →  3,385 train / 432 val / 452 test              (§2.6, 0 fuga)
```

### Calidad medida

Juez `gpt-4o` sobre 150 preguntas al azar:

| | |
|---|---|
| Correcta respaldada por el fragmento | 96.0% |
| Algún distractor también verdadero | 5.3% |
| **Sin ningún defecto** | **92.7%** |

Formato, sobre las 4,908: 100% en español, 0.12% con palabras inglesas, ningún
voseo, 1 opción de 4,908 sobre 15 palabras.

Rechazo por filtro en §2.5: parecidas 8.1%, desparejas 2.1%, duplicadas 1.3%,
no autocontenidas 1.3%, negativas 0.6%, distractor en el fragmento 0.1%.

## 5. Hechos verificados contra el hub de Hugging Face

Verificados durante la conversación, no recordados. Importan porque contradicen
suposiciones razonables:

| Hecho | Detalle |
|---|---|
| `Qwen3.5` **sí existe** | 29 modelos en el hub; el 4B tiene ~7.2M descargas mensuales |
| Toda la familia Qwen3.5 es **multimodal** | `Qwen3_5ForConditionalGeneration`, con `vision_config`, `image_token_id`, `video_token_id`. Aplica a 0.8B, 2B, 4B y 9B |
| `Qwen/Qwen3.5-4B-Instruct` **no existe** | en Qwen3.5 el instruct es `Qwen3.5-4B` a secas; el crudo es `Qwen3.5-4B-Base` |
| Qwen3 sí es de solo texto | `Qwen3ForCausalLM`: `Qwen3-4B`, `Qwen3-4B-Instruct-2507`, `Qwen3-8B` |

**Consecuencia:** el modelo elegido es **`Qwen/Qwen3-4B-Instruct-2507`**. Para
generar texto, la torre de visión de Qwen3.5 es peso muerto: ocupa memoria,
complica decidir qué módulos toca el LoRA, y `AutoModelForCausalLM` puede ni
cargarlo.

Estructura de MedQuAD (`HoangHa/MedQuaD`), 13 columnas:
`document_id, document_source, document_url, category, umls_cui,
umls_semantic_types, umls_semantic_group, synonyms, question_id, question_focus,
question_type, question, answer`.

## 5.b Hallazgos de la ejecución que corrigen el plan

| Hallazgo | Consecuencia |
|---|---|
| **`category` es 94% NaN** tras la limpieza (13,661 de 14,528), y `Drug` —un tercio del dataset crudo— desaparece entera | El "Cambio 1" del notebook 01 resultó **inútil**. El selector de la app se construye con `question_type` (16 valores) y `question_focus` (5,125). La columna se conserva porque no molesta, pero no se depende de ella |
| **`question_type`: 39 valores en crudo, 16 tras limpiar** | Las diapositivas del proyecto anterior dicen "37 tipos" (la cifra del paper de MedQuAD). Lo que el sistema realmente usa son 16 |
| **`support groups` tiene 1 solo fragmento** en todo el corpus | Ningún muestreo proporcional lo conservaría; lo salva el mínimo por estrato. La app debería excluirlo del selector: no se puede armar un examen con un fragmento |
| El filtro "arranca cortado" descartaba el **56% del corpus** | Se resolvió recortando hasta la primera oración completa en vez de descartar: 24,424 fragmentos recuperados, y los aptos pasaron de 16,415 (34%) a 37,153 (78%) |

## 6. Decisiones tomadas y por qué

| Decisión | Razón |
|---|---|
| **La app genera en vivo, no sirve un banco pregenerado** | Es lo único que justifica el fine-tuning. Si sirviera preguntas ya generadas, bastaría un `SELECT` y el modelo sobraría |
| Solo 4,000–6,000 chunks pasan por la API | Alcanza para LoRA. Los otros ~32,000 los cubre el modelo afinado: eso *es* el aporte |
| **La respuesta correcta no es el chunk** | Es una afirmación corta extraída de él, máximo 15 palabras. Sin esto, las cuatro opciones serían párrafos de 400 caracteres y la pregunta sería inservible |
| Opciones de longitud pareja (ratio máx/mín ≤ 2.5) | Si la correcta es siempre la más larga, se acierta sin saber el tema. Es el sesgo más común en exámenes generados |
| Rechazar si un distractor aparece literal en el chunk | Si está en el texto, probablemente también es cierto → dos respuestas válidas |
| Chunking B_512 (no 1024) | ~512 caracteres es un hecho médico completo, la unidad correcta para *una* pregunta. Con más contexto el modelo elige arbitrariamente y las preguntas salen vagas |
| Split agrupado por `document_id` | Los chunks tienen 64 caracteres de solape. Sin agrupar, fragmentos casi idénticos caerían en train y test, y la evaluación mediría memorización |
| **Sin base vectorial** | El usuario elige de una lista y `question_type`, `category` y `question_focus` ya son columnas. Un filtro exacto con pandas basta. FAISS solo haría falta para tema libre en texto |
| **Sin Unsloth** | El adaptador queda 100% estándar al moverlo de Colab a local, es el stack que dictó el profesor, y con 16 GB en un 4B no hace falta su optimización de memoria |
| **Entrenar en local, no en Colab** | Medido en la prueba de humo: 9.63 GB de pico y 0.39 s por paso, o sea 33 min las 3 épocas. Elimina la transferencia del adaptador, el riesgo de versiones distintas y la dependencia de qué GPU toque en Colab. El notebook igual corre en Colab si hiciera falta |
| App como **simulacro**, no como formulario | "Elegí categoría y cantidad" no muestra nada. Un examen con corrección y fuentes es un producto |
| Generación progresiva (una pregunta a la vez) | Elimina la espera percibida: el estudiante tarda 15–30 s por pregunta y la GPU genera varias en ese rato |
| Panel "Bajo el capó" plegable | Es donde se ve la diferencia entre base y afinado. Cerrado por defecto para que la app siga siendo un producto |

## 7. Correcciones hechas durante el diseño

Se anotan para no volver a proponerlas:

- **"Distractores sembrados desde el corpus"** — propuesta inicial, descartada.
  Demasiada complejidad para el beneficio. La reemplaza el filtro de "distractor
  contenido en el chunk", que son tres líneas y resuelve el mismo riesgo.
- **"El botón de regenerar hace visible el fine-tuning"** — impreciso. Demuestra
  que hay **generación en vivo**, no que entrenar haya servido. Un modelo base con
  few-shot también daría preguntas distintas. Lo que demuestra el valor del
  fine-tuning es el panel comparativo.
- **"Qwen3.5 probablemente no existe"** — equivocado, sí existe. De ahí la regla
  de verificar identificadores contra el hub en lugar de confiar en la memoria del
  modelo.
- **Confusión entre latencia de entrenamiento e inferencia** — el entrenamiento
  no es problema (20–60 min). El cuello es generar 50 preguntas de a una en la
  app (~3 min), y se resuelve con lotes y generación progresiva.

## 8. Riesgos abiertos

| Riesgo | Plan B |
|---|---|
| ~~Blackwell + `bitsandbytes` en 4-bit~~ | **descartado**: torch 2.11+cu128 soporta sm_120 nativamente y el modelo entra en bf16 (8.04 GB). No se usa cuantización |
| ~~El adaptador de Colab no carga en local~~ | **descartado**: se entrena en local, no hay transferencia. Verificado igual que el adaptador guarda y recarga en un proceso nuevo |
| ~~El fine-tuning no gana al base con few-shot~~ | **resuelto**: gana en estructura (60/60 vs 57/60) y sobre todo en tokens de prompt (135 vs 693). Pierde en velocidad por pregunta (6.61 s vs 4.46 s). En contenido supera con claridad al base con las reglas del maestro (76.7% contra 56.7%) y empata con el few-shot y con un prompt mínimo (notebook 05) |
| ~~El prompt de generación produce basura~~ | **resuelto**: piloto de 201 y luego 5,250 fragmentos, con juez independiente |
| `revisar_forma()` nunca se validó contra un juicio humano | se sabe que dejó de rechazar de más (el incidente de las 220), no se sabe si rechaza lo suficiente. Está medido que es ciego a lo semántico: atrapó 0 de 7 defectos |
| El techo del 78-79% es el del maestro | se decidió **no** regenerar el dataset con gpt-4o ($15.05 medidos, medio día de trabajo). Se presenta como limitación con su costo de solución |
| El juez gpt-4o también se equivoca | marcó un error de traducción con una justificación que se contradice sola. El 78.4% lleva ese ruido dentro |

## 9. Objeción previsible en la sustentación

> "¿Esto no es simplemente destilar gpt-4o-mini en un modelo chico?"

Sí, exactamente eso, y se responde con el nombre técnico —destilación de
conocimiento mediante datos sintéticos— en lugar de esquivarla. Lo defendible:
corre local sin API, cuesta cero por consulta, cubre 32,000 chunks que nunca
pasaron por la API, y la evaluación mide cuánto se acerca el alumno al maestro.

Lo que **no** se puede afirmar: que el afinado sea mejor que gpt-4o-mini (76.7%
contra 80.0%, sin diferencia significativa), ni que escriba preguntas más
correctas que el Qwen base con few-shot (p = 0.12) o con un prompt mínimo con
esquema (p = 0.39). Lo que sí: supera con claridad al base que recibe por escrito
las reglas del maestro (76.7% contra 56.7%, p < 0.0001), alcanza al maestro
cuando el few-shot no, y nunca rompe el formato. Detalle en
`INSIGHTS_Y_DECISIONES.md` §5.f.

## 10. Notas de trabajo

- **Sin emojis**, en ningún entregable ni en el código.
- El usuario cuestiona las afirmaciones y pide fundamentos. Verificar antes de
  afirmar; decir "no lo medí" cuando corresponda.
- Preferencia por documentos `.md` en la raíz del proyecto (así se trabajó el
  proyecto anterior con `Contenido_Diapositivas.md` y `Guion_Diapositivas_6-9.md`).
- **Variante del español:** el usuario es peruano. Los textos que verá el usuario
  final —el prompt de generación y la interfaz— deben ir en **español neutro
  latinoamericano con tuteo** ("¿cuál es...?", "responde", "elige"), nunca en
  voseo rioplatense ("respondé", "elegí"). Conviene fijarlo explícitamente en el
  prompt de generación, o las preguntas pueden salir con registro inconsistente.

## 11. Estimación de esfuerzo

Los notebooks y la app los escribe Claude, así que el tiempo de programación no
es el factor limitante. Lo que manda son los tiempos de máquina y las decisiones
humanas.

| Etapa | Tiempo | Comprimible |
|---|---|---|
| Descargar Qwen3-4B (~8 GB) | 10–30 min | no, depende de la conexión |
| Prueba de humo de inferencia local | 15 min | no |
| Notebook 01 (escribir + ejecutar) | 20 min | escribir sí, ejecutar no |
| Notebook 02 — escribir | 20 min | sí |
| Notebook 02 — generar 5,000 preguntas con API | 30–40 min | no (12 hilos concurrentes) |
| **Revisar preguntas e iterar el prompt** | **1–2 h** | **no: es criterio humano** |
| Notebook 03 en Colab (setup + humo + entrenar + evaluar) | 1.5–2 h | no |
| App Streamlit (escribir + probar) | 1–2 h | escribir sí, probar no |

**Total: 6 a 8 horas si nada se rompe.** Cabe en un día de trabajo. Con un
tropiezo de entorno —Blackwell con `bitsandbytes` es el candidato— día y medio.

Dos condiciones para que entre en un día:

1. El usuario debe estar disponible para las idas y vueltas con Colab: subir el
   notebook, correrlo, descargar el adaptador.
2. La revisión del prompt de generación no conviene apurarla. Es el paso donde se
   define la calidad de todo el dataset.

Si hubiera que recortar: el panel comparativo se reduce a una tabla estática con
los números de la evaluación, y el modo práctica interactivo se reduce a mostrar
preguntas con sus respuestas. **El notebook 02 no se recorta.**
