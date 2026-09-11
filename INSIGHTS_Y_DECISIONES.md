# Insights y decisiones — material para la sustentación

Todo lo aprendido construyendo el generador de exámenes, con la evidencia que
respalda cada decisión. Documento vivo: se actualiza conforme avanza el proyecto.

Estado a la fecha: notebooks 01, 02 y 03 escritos; 01 y 02 ejecutados y el 03
validado de punta a punta. Pendiente la aplicación.

---

# 1. La idea que estructura todo el proyecto

Antes de escribir una línea de código hubo que responder una pregunta incómoda:

> Si generas las preguntas con la API de un modelo grande, ¿para qué entrenas
> uno pequeño? Guarda las preguntas en una tabla y sírvelas con un `SELECT`.

Es una objeción válida y hundiría el proyecto si no tuviera respuesta. La que se
adoptó define el diseño entero:

**La tabla de fragmentos completa es la base de datos de la aplicación. El
subconjunto que pasa por la API existe únicamente para entrenar.**

En números: hay **28,294 fragmentos** aptos, y solo **5,250** pasaron por la API.
Los otros 23,044 nunca vieron al modelo maestro. Cuando un estudiante pida
preguntas sobre esos temas, las genera el modelo afinado. **Eso es el aporte del
fine-tuning**, y es medible: cobertura de 28,294 fragmentos por 5,250 pagados.

De esta decisión se derivan las demás: la aplicación genera en vivo, nunca sirve
un banco pregenerado, y por eso necesita un modelo cargado en memoria.

---

# 2. Decisiones sobre los datos, con su evidencia

## 2.1 Reutilizar la limpieza del proyecto anterior

Los 11 pasos del notebook `01_data_preparation_and_cleaning.ipynb` del proyecto
de RAG ya estaban validados. Se heredan tal cual, con tres cambios menores.

**Resultado:** 47,441 filas crudas → 14,528 limpias.

## 2.2 Fragmentar el corpus completo, no solo el split de entrenamiento

El proyecto anterior fragmentaba únicamente `train`, porque `val` y `test` se
reservaban para evaluar el buscador: no se puede evaluar un retriever con
documentos que ya indexaste.

Aquí no hay retrieval que evaluar, y la aplicación debe poder generar preguntas
sobre **cualquier** tema. Se fragmenta todo.

| | Filas | Fragmentos |
|---|---|---|
| Proyecto anterior (solo `train`) | 11,630 | 38,127 |
| Este proyecto (corpus completo) | 14,528 | **47,771** |

Verificado: fragmentar solo `train` reproduce exactamente los 38,127 del proyecto
anterior, lo que confirma que la fragmentación en sí es idéntica.

## 2.3 Mantener el solape de 64, pero por otra razón

En el proyecto de RAG el solape servía para que un hecho partido por la frontera
apareciera entero en algún fragmento y fuera recuperable. Esa razón no aplica
aquí. Pero hay otra: **compensa el recorte** (ver 2.4).

Medido sobre 1,200 respuestas:

| Estrategia | Aptos | Texto conservado | Casi-duplicados |
|---|---|---|---|
| **solape 64** | 2,926 | **72.7%** | 7 |
| sin solape | 2,675 | 67.0% | 5 |
| sin solape, cortando en oraciones | 3,015 | 66.0% | 5 |

Sin solape, la oración que el troceado parte se pierde del todo: no está al final
del fragmento anterior (recortado) ni al principio del siguiente (recortado
también). Son **casi 6 puntos de contenido**.

El riesgo esperable del solape —preguntas casi idénticas generadas desde dos
fragmentos que comparten texto— **no se materializa**: 7 pares en casi 3,000. La
razón es elegante: el recorte elimina justamente la zona solapada, que es el
final de una oración partida. **El solape evita que el texto se pierda; el
recorte evita que se duplique.**

## 2.4 Recortar inicio y final hasta oración completa

El troceado por caracteres corta a mitad de frase. Dos hallazgos medidos:

- **59% de los fragmentos arrancaba cortado** (continuación de una oración).
- **78% terminaba colgando** (*"...downturned corners of the mouth, and a"*).

El primero se detectó pronto; el segundo se me pasó en la primera versión y lo
encontré revisando muestras a mano. Un final colgando es **peor** que un arranque
cortado, porque invita al modelo a completar de su propio conocimiento lo que el
fragmento no llegó a decir — exactamente lo que hay que evitar.

Solución: recortar ambos extremos en vez de descartar el fragmento. Se
recuperaron **24,424 fragmentos** que la primera versión tiraba: los aptos
pasaron de 16,415 (34%) a 37,153 (78%).

**Resultado final:** 100% de los fragmentos cierra en oración completa.

## 2.5 Descartar las tablas del Human Phenotype Ontology

**El hallazgo más importante del análisis de datos.**

MedQuAD contiene 2,240 respuestas (**15% del corpus limpio**) que no son texto
escrito por nadie: son volcados automáticos del Human Phenotype Ontology. De unas
200 palabras, el contenido médico real son cuatro términos sueltos con guiones:

> *Autosomal dominant inheritance - Autosomal recessive inheritance - Congenital
> onset - Menorrhagia -*

Todo lo demás explica **cómo leer la tabla**: qué es el HPO, que los datos vienen
de Orphanet, cómo interpretar una fracción.

**Por qué no sirven, demostrado empíricamente.** Se generó una pregunta desde una
tabla HPO y otra desde un fragmento normal, con el mismo modelo y el mismo prompt:

*Desde la tabla:* «¿Qué síntoma ocurre en el 90% de los pacientes con deficiencia
de dihidropteridina reductasa?» → correcta: *Impairimiento cognitivo*;
incorrectas: *Microcefalia*, *Calcificación cerebral*, *Herencia autosómica
recesiva*.

Cuatro fallos en una sola pregunta: hay **dos respuestas correctas** (la tabla
dice `Cognitive impairment 90%` **y** `Microcephaly 90%`), **los tres distractores
son ciertos** según el propio fragmento, *herencia autosómica recesiva* **no es un
síntoma**, y *"impairimiento cognitivo"* **no existe en español**.

*Desde el fragmento normal:* «¿Qué síntoma inicial puede presentarse en la
enfermedad de Huntington?» → correcta: *Irritabilidad*; incorrectas: *pérdida de
memoria a largo plazo*, *dolor articular crónico*, *hiperactividad motora*. Una
sola respuesta correcta, tres distractores plausibles y falsos.

**La razón de fondo:** una tabla es una lista de datos sin jerarquía, donde todos
los ítems son igual de ciertos. Para preguntar hace falta una **afirmación con
estructura** —"lo más frecuente es X"— que indique cuál es la respuesta y deje el
resto disponible como distractores falsos.

### El matiz que conviene tener preparado

En el proyecto de RAG se llegó a la conclusión **contraria**: allí se documentó
que el juez LLM penalizaba estas tablas y se anotó que era *"más estricto de lo
debido: son síntomas, aunque en formato de tabla"*.

No es contradicción, cambió la tarea:

| | RAG médico | Generador de exámenes |
|---|---|---|
| La pregunta | la trae el usuario | hay que **inventarla** |
| Una tabla HPO | responde "¿cuáles son los síntomas?" — sirve | no hay afirmación de la que extraer respuesta y distractores |

La regla general que se repitió durante todo el proyecto: **buscar tolera
fragmentos imperfectos; preguntar no.**

### Consecuencia en cobertura

Se pierden 1,333 temas respecto del corpus limpio. Verificado uno por uno: **los
1,333 contenían exclusivamente tablas del HPO** (1,283 de tipo `symptoms`), sin
una línea de prosa. No había pregunta posible ahí.

## 2.6 Descartar listas aplanadas

Variante del anterior que costó encontrar. El filtro inicial contaba líneas que
empiezan con viñeta, pero el corpus trae muchas tablas **en un solo renglón**:

> *Arrhythmia 7.5% Absent eyebrow - Bradycardia - Fair hair - Nail dystrophy -*

Cero líneas con viñeta, así que pasaban limpias. Se detectan por densidad de
separadores y porcentajes.

## 2.7 La respuesta correcta no es el fragmento

El error de diseño que habría hecho inservible todo el proyecto, y que surgió de
una pregunta directa: *"sabes que no se podría porque algunas respuestas son
larguísimas"*.

Si la respuesta correcta sale del fragmento tal cual, las cuatro opciones son
párrafos de 400 caracteres. Eso no es una pregunta de examen, es un ejercicio de
lectura comparada.

**La corrección:** el fragmento es la *fuente*, no la respuesta. El modelo extrae
una afirmación corta:

```
chunk (512 chars sobre cefaleas)
  ↓
pregunta:     "¿Cuál es el tipo más frecuente de dolor de cabeza?"
correcta:     "Cefalea tensional"
distractores: "Migraña" · "Cefalea en racimos" · "Cefalea sinusal"
```

Con dos restricciones en el prompt: máximo 15 palabras por opción, y longitudes
parejas entre las cuatro.

**Resultado medido:** de 4,908 preguntas, **1 sola** supera las 15 palabras, y el
ratio entre la opción más larga y la más corta es 1.61 de media.

La restricción de longitud pareja corrige el sesgo más común de los exámenes
generados automáticamente: si la correcta es siempre la más larga y detallada, se
acierta sin saber el tema.

## 2.8 Muestreo estratificado con techo y piso

`information` es el 27% del corpus y `support groups` tiene **un solo fragmento**
en todo MedQuAD. Un muestreo proporcional dejaría al modelo viendo sobre todo
preguntas de información general y perdería los tipos raros.

Se aplica un techo (1,200 por tipo) y un piso (60), y se toma **un fragmento por
respuesta** siempre que se pueda, para que el modelo no vea el mismo tema
repetido. Resultado: 99% de la muestra son respuestas distintas.

---

# 3. Decisiones sobre la generación

## 3.1 El modelo ignora la válvula de escape si no se le obliga

**El hallazgo más útil del piloto**, y la razón por la que valió la pena gastar 5
centavos antes de gastar 79.

El prompt v1 incluía: *"si el fragmento no permite una pregunta con respuesta
concreta, responde `{descartar: true}`"*. Sobre 201 fragmentos, el modelo usó esa
salida **0 veces**. Ante un fragmento sobre el público objetivo de un sitio web
generó igual una pregunta: *«¿Quiénes son el público objetivo del sitio web
GTR?»*, que no es medicina.

El modelo obedece la tarea principal y salta la instrucción condicional.

**La corrección:** obligarlo a decidir **antes** de escribir nada, con un campo
`apto` que va primero en el JSON, y describiendo con ejemplos qué **no** es
contenido médico.

| | v1 | v2 |
|---|---|---|
| JSON válido | 100% | 100% |
| **Descarta fragmentos no aptos** | **0 (0%)** | **12 (6.0%)** |
| Dificultad: fácil / media | 197 / 4 | 2 / 187 |
| Dos opciones muy parecidas | 11.9% | 9.0% |

**Lección general:** una instrucción condicional dentro de un prompt no es una
garantía. Si algo tiene que ocurrir siempre, hay que forzarlo estructuralmente.
Es el mismo patrón que en el proyecto anterior obligó a mover la abstención del
prompt al código.

## 3.2 Ese 6% era el dato que faltaba

La tasa de descarte del modelo **mide directamente cuántos fragmentos no aptos
deja pasar el filtro sintáctico**. Es la validación del filtro de §2.2.

Salió 6.5% sobre los 5,250. Revisando los 12 descartes del piloto a mano: **8
correctos** (sitios web, seguros médicos, ensayos clínicos, listas de podcasts) y
**3 falsos positivos** —contenido médico legítimo descartado, en un caso con un
motivo que el modelo inventó: descartó un texto sobre el tratamiento del prurigo
nodular alegando que "habla de un sitio web"—.

Perder 1.5% de material bueno para atrapar 4% de malo es favorable cuando hay
28,294 fragmentos y se necesitan 5,250.

## 3.3 El sesgo de dificultad

La v1 producía preguntas de dato literal casi siempre: **197 fáciles de 201**.
Pedir explícitamente *"prefiere preguntas que exijan relacionar dos datos"*
invirtió la distribución a 187 medias de 189.

**Limitación honesta:** ninguna versión genera preguntas "difíciles". El modelo no
usa esa categoría por más que se le ofrezca. Se documenta como límite conocido.

## 3.4 Las preguntas negativas salen mal

Detectado en la revisión con el juez. Las preguntas del tipo *"¿cuál factor NO
influye en el pronóstico?"* exigen que tres opciones sean verdaderas y una falsa,
al revés del diseño habitual. En la muestra evaluada, las dos que aparecieron eran
defectuosas.

Son el 1.7% del total, así que descartarlas cuesta poco y elimina un patrón de
riesgo completo. **Aviso metodológico:** la muestra son dos casos; la señal es
fuerte y el razonamiento se sostiene, pero se trata como precaución barata, no
como conclusión firme.

---

# 4. Decisiones técnicas

## 4.1 El modelo: Qwen3, no Qwen3.5

El ejemplo del curso usaba `Qwen/Qwen3.5-0.8B`. Verificado contra el hub de
Hugging Face:

| Hecho | Detalle |
|---|---|
| `Qwen3.5` existe | 29 modelos; el 4B tiene ~7.2M descargas mensuales |
| **Toda la familia Qwen3.5 es multimodal** | `Qwen3_5ForConditionalGeneration`, con `vision_config` y tokens de imagen y vídeo. Aplica a 0.8B, 2B, 4B y 9B |
| `Qwen3.5-4B-Instruct` **no existe** | el instruct es `Qwen3.5-4B` a secas; el crudo lleva `-Base` |
| Qwen3 sí es de solo texto | `Qwen3ForCausalLM` |

Para generar texto, la torre de visión es peso muerto: ocupa memoria, complica
decidir qué módulos toca el LoRA, y `AutoModelForCausalLM` puede ni cargarlo.

**Decisión: `Qwen/Qwen3-4B-Instruct-2507`.** La generación anterior, pero la
herramienta correcta para esta tarea.

### Revisado a fondo con el proyecto terminado

Con el entrenamiento hecho se volvió a examinar `Qwen/Qwen3.5-4B`, instanciando
su arquitectura sin descargar los pesos. Aparecieron diferencias que la primera
verificación no había visto y que pesan más que la torre de visión:

| Hecho | Detalle | Consecuencia |
|---|---|---|
| Arquitectura híbrida | de sus 32 capas de texto, **24 usan atención lineal** (regla delta con compuertas) y solo 8 atención completa | — |
| Proyecciones con otros nombres | en las capas lineales la atención es `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj` | la configuración ganadora (`q/k/v/o_proj` + MLP) solo tocaría las MLP en el 75% de las capas. **El barrido del notebook 03 no se puede reutilizar** |
| Sin kernels rápidos en esta máquina | `fla`, `causal_conv1d` y `triton` no están instalados; triton se distribuye oficialmente solo para Linux | `transformers` cae a la implementación en PyTorch puro en esas 24 capas. La penalización de velocidad **no se midió** |
| Más grande | **5.17 B de parámetros**: 4.84 B de lenguaje + 0.33 B de visión (6.4%), 9.32 GB en disco | contra ~4.0 B y 8.04 GB del modelo elegido |
| Otra clase de carga | se instancia con `AutoModelForImageTextToText` | todo el código del proyecto usa `AutoModelForCausalLM` |

`transformers` 5.16.1 sí lo soporta: el obstáculo no es la compatibilidad, es el
costo de adaptación.

### ¿Daría mejores resultados?

Probablemente poco en la cifra que importa. El alumno ya empata con su maestro
(78.4% contra 79.3%, §5.b), lo que indica que el techo está en los datos de
`gpt-4o-mini` y no en el modelo alumno: el 83% de «tema correcto» viene de las
preguntas con las que se entrenó, y cualquier alumno aprende a reproducirlo.

Donde un modelo más nuevo sí podría ayudar, **sin que esté medido**:
- traducción y fluidez en español (errores tipo *nipple* → «lactancia»);
- la auto-verificación, que la ejecuta el modelo base (§5.d): un base más capaz
  verificaría mejor.

**Decisión: no se cambia para la entrega.** Reabre el barrido, el entrenamiento,
todas las evaluaciones y el código de carga, por una ganancia incierta.

Si se quisiera probar, el diseño correcto es un piloto: una sola configuración,
150 preguntas generadas, la misma rúbrica del juez, y comparar con el 78.4%.
**El umbral debe fijarse antes:** con 150 preguntas por modelo, el intervalo al
95% de la diferencia entre dos tasas cercanas al 78% es de **±9.4 puntos**, así
que cualquier mejora menor a unos 10 puntos sería indistinguible del ruido.

## 4.2 Sin base vectorial

El usuario elige de una lista, y `question_type` y `question_focus` ya son
columnas del corpus. Un filtro exacto con pandas basta. Una búsqueda semántica
solo haría falta si se permitiera escribir un tema en texto libre.

## 4.3 Sin Unsloth

Tres razones: el adaptador queda 100% estándar al moverlo de Colab a la máquina
local, es el stack que se dictó en clase (`transformers` + `peft` + `trl`), y con
16 GB en un modelo de 4B no hace falta su optimización de memoria. El costo es
entrenar algo más lento.

## 4.4 Entrenar en local, no en Colab

El plan original era entrenar en Colab por miedo a que el modelo no entrara en la
GPU. Una prueba de humo de 10 pasos con 50 ejemplos midió lo que hacía falta
saber:

| | |
|---|---|
| Memoria pico entrenando | **9.63 GB** de 16.3 disponibles |
| Tiempo por paso | 0.39 s |
| Parámetros entrenables | 33M de 4,055M (**0.81%**) |
| La pérdida baja | 1.55 → 0.42 en 10 pasos |
| Tres épocas completas | **33 minutos** |

**El entrenamiento cabe en la máquina local.** Eso elimina tres riesgos de un
golpe: la transferencia del adaptador entre entornos, la incompatibilidad de
versiones entre Colab y local, y la dependencia de qué GPU toque en Colab.

También es mejor para la sustentación: todo el proyecto reproducible en una sola
máquina, sin pasos manuales de subir y descargar.

El notebook 03 igual detecta el entorno y corre en Colab sin cambios, así que la
opción sigue disponible como respaldo.

**El riesgo que esto evita, documentado por si vuelve a aparecer:**
`adapter_config.json` guarda `base_model_name_or_path`. Si se entrena contra un
identificador y se carga contra otro, el adaptador falla al aplicarse, o peor, se
aplica a medias sin avisar.

## 4.4.b Lo que la prueba de humo validó además

Cinco pasos, todos verificados antes de gastar media hora de entrenamiento:

1. el modelo acepta LoRA
2. el formateo al chat template funciona
3. el entrenamiento avanza (la pérdida baja)
4. el adaptador se guarda (144 MB)
5. **el adaptador recarga en un proceso nuevo y genera**

Y uno extra que importa para la aplicación: **`disable_adapter()` funciona**.
Permite servir el modelo base y el afinado con **8.31 GB** en total, sin cargar
dos veces 8 GB. Es lo que hace viable el panel comparativo de la sustentación.

## 4.4.c Se entrena solo sobre la respuesta

Detalle técnico que vale mencionar: en el formateo, los tokens del prompt se
marcan con `-100`, la etiqueta que PyTorch ignora al calcular la pérdida.

Sin eso, el modelo aprendería también a reproducir el fragmento **en inglés**,
que no es la tarea y gastaría capacidad del adaptador en algo inútil.

## 4.5 Los idiomas

El corpus está en inglés y los estudiantes leen en español.

```
chunks.parquet     fragmento EN INGLÉS (nunca lo ve el estudiante)
       ↓
modelo afinado     entrenado: fragmento inglés → pregunta español
       ↓
pantalla           todo EN ESPAÑOL
```

No se traducen los fragmentos: costaría más que generar las preguntas y rompería
la correspondencia con la que se entrena el modelo.

**Verificado sobre las 4,908 preguntas:** 100% en español (todas abren con `¿`),
0.12% con alguna palabra funcional inglesa, ningún caso de voseo.

Lo que sí requería solución era el **selector de la interfaz**: `question_type` y
`question_focus` están en inglés y sí los vería el estudiante. Se agregaron dos
columnas de presentación a `chunks.parquet`: `tipo_es` (16 valores, a mano) y
`tema_es` (3,794 temas, traducidos una vez con la API). La app filtra por la
columna inglesa y muestra la española.

## 4.6 Splits agrupados por documento

Los fragmentos tienen 64 caracteres de solape. Sin agrupar, dos fragmentos casi
idénticos del mismo documento caerían en entrenamiento y prueba, y la evaluación
mediría memorización.

**Verificado: 0 documentos compartidos** entre train-val, train-test y val-test.

---

# 5. El pipeline completo, con cifras

```
47,441 filas crudas
  → 14,528  tras los 11 pasos de limpieza              (31%)
  → 47,771  fragmentos de 512 caracteres con solape 64
  → 28,294  aptos tras el filtro sintáctico            (59%)
  →  5,250  muestreados para la API                    (estratificado)
  →  4,908  preguntas generadas                        (93.5%, USD 0.79)
  →  4,483  aprobadas por los filtros de calidad       (91%)
  →  3,614 train · 433 val · 436 test                  (0 fuga)
```

Las cifras de aprobación cambiaron a mitad del proyecto: los filtros originales
rechazaban 220 preguntas correctas y hubo que recalibrarlos. Está contado en §6.


## Calidad, medida con juez independiente

`gpt-4o` sobre 150 preguntas al azar, leyendo el fragmento fuente:

| | |
|---|---|
| Respuesta correcta respaldada por el fragmento | **96.0%** |
| Algún distractor también verdadero | 5.3% |
| Una sola respuesta defendible | 92.7% |
| **Sin ningún defecto** | **92.7%** |

Se usó `gpt-4o` y no `gpt-4o-mini` deliberadamente: evaluar es más difícil que
generar, y el mismo modelo se autoevaluaría con indulgencia.

## Rendimiento del modelo local

`Qwen3-4B-Instruct-2507` en la RTX 5070 Ti:

| | |
|---|---|
| Memoria en bfloat16 | **8.04 GB** (de 16 disponibles) |
| Generación secuencial | 23.4 tok/s |
| **Lotes de 8** | **199.8 tok/s — 8.5x** |

El batching es lo que hace viable la aplicación: 50 preguntas pasan de ~3 minutos
a ~40 segundos.

---

# 5.b El entrenamiento: lo que el barrido enseñó de verdad

## El barrido de hiperparámetros no eligió, midió

Seis configuraciones, 250 pasos cada una, evaluadas por pérdida de validación y
por calidad de generación sobre 24 casos:

| Configuración | Limpias | eval_loss | Entrenables | Pico VRAM |
|---|---|---|---|---|
| **5. rank alto r32** | 22 | **0.3626** | 66.1 M | 10.43 GB |
| 3. lr alto 3e-4 | 22 | 0.3639 | 33.0 M | 9.87 GB |
| 1. referencia | 22 | 0.3672 | 33.0 M | 9.87 GB |
| 2. lr bajo 1e-4 | 22 | 0.3760 | 33.0 M | 9.87 GB |
| 6. solo atención | 22 | 0.3834 | 11.8 M | 9.59 GB |
| 4. rank bajo r8 | 21 | 0.3727 | 16.5 M | 9.65 GB |

**El hallazgo no es quién ganó: es que la dispersión total es 0.0208.** Seis
configuraciones que van de 11.8 M a 66.1 M de parámetros entrenables —un factor
de 5.6— se separan en dos centésimas de pérdida, y las seis producen JSON válido
en 24 de 24 casos y escriben en español en 24 de 24.

La lectura honesta: **para esta tarea la configuración casi no importa; lo que
importó fue el dataset**. Presentar la ganadora como un hallazgo sería inflar el
resultado. El único efecto sistemático visible es que restringir el LoRA a las
capas de atención empeora de forma medible (0.3834 contra 0.3626): incluir las
capas MLP sí aporta, aunque poco.

## El entrenamiento final: sin sobreajuste, pero con memorización

| época | 0.22 | 0.44 | 0.66 | 0.89 | 1.11 | 1.33 | 1.55 | 1.77 | 1.99 |
|---|---|---|---|---|---|---|---|---|---|
| validación | .4018 | .3855 | .3667 | .3554 | .3618 | .3593 | .3528 | .3515 | **.3506** |

Dos lecturas que conviene dar juntas:

**No hubo sobreajuste.** La validación seguía bajando en el último paso. Con una
tercera época probablemente habría mejorado algo más; se fijó en 2 y la curva no
lo justificaba todavía.

**Sí hubo memorización.** La pérdida de entrenamiento cae a 0.21 contra 0.35 de
validación, y el salto ocurre exactamente al cambiar de época (0.3485 → 0.2215):
el modelo empieza a reconocer ejemplos que ya vio. Que la validación siga bajando
pese a eso es lo que hace aceptable la segunda época.

## El alumno alcanzó al maestro

`gpt-4o` juzgando 150 preguntas de cada uno, con la misma rúbrica:

| | maestro (gpt-4o-mini) | alumno (Qwen afinado) |
|---|---|---|
| Correcta respaldada | 97.2% | 97.1% |
| Sin distractor cierto | 97.2% | 95.7% |
| Una sola respuesta | 96.6% | 94.2% |
| Tema correcto | 83.4% | 82.7% |
| **Sin ningún defecto** | **79.3%** | **78.4%** |

Con n≈140 por lado, 0.9 puntos está dentro del ruido: la destilación cumplió su
objetivo, que nunca fue superar al maestro sino igualarlo sin depender de él.

> **Matiz que agregó el notebook 05 (§5.f):** el modelo base con few-shot, sin
> entrenar, **tampoco** se distingue del maestro con esta muestra (72.6% contra
> 78.2%, p = 0.22). El empate con el maestro no prueba por sí solo que el
> entrenamiento mejorara el contenido.

Tiene además una consecuencia práctica importante: **ningún ajuste de
entrenamiento va a mejorar el modelo, porque está en su techo**. El único
movimiento que queda es cambiar el maestro. Se midió lo que costaría: regenerar
los 5,250 fragmentos con `gpt-4o` son **USD 15.05** (contra USD 0.90 con
`gpt-4o-mini`), más reentrenar y rehacer todas las evaluaciones. Se decidió no
hacerlo.

**Y "tema correcto" al 83% es el techo de los dos**, no una falla del alumno. Es
un defecto heredado de la destilación.

---

# 5.c Los parámetros de generación: la tensión que no existía

El notebook 04 barre los parámetros de muestreo sobre el modelo ya congelado.
La hipótesis de partida era una tensión entre formato y variedad:

> `do_sample=False` da formato perfecto y variedad cero. Subir la temperatura
> da variedad pero rompe el JSON.

**La segunda mitad de esa frase es falsa.** Sobre 420 generaciones:

| Configuración | Formato | Variedad (de 3) | Truncadas |
|---|---|---|---|
| greedy | 60/60 | 1.00 | 0 |
| temp 0.3 | 60/60 | 1.95 | 0 |
| temp 0.7 | 60/60 | 2.30 | 0 |
| temp 1.0 | 60/60 | **2.70** | 0 |
| temp 0.7 + rep 1.1 | 60/60 | 2.45 | 0 |
| temp 0.7 + rep 1.2 | 60/60 | 2.60 | 0 |
| temp 0.7 sin top_k | 60/60 | 2.50 | 0 |

**Las siete dan 100% de formato válido, incluida temperatura 1.0.** Cero
truncamientos, lo que además valida `max_new_tokens=300`.

**Corrección posterior (notebook 05):** aquí se atribuyó esa robustez al
fine-tuning, pero el barrido corrió solo con el afinado, sin control. El control
con el base y su few-shot, sobre los mismos 20 fragmentos, da 57/60 a
temperatura 0.7 y 58/60 a 1.0, con variedad casi igual. **El base también aguanta
el muestreo**; lo propio del afinado es no romper nunca la estructura. Ver §5.f.

También confirma que greedy sería un error de producto: variedad 1.00 significa
que el botón «otro examen del mismo tema» devolvería siempre lo mismo.

**Lo que el barrido NO mide:** corrección factual. Se midió aparte, porque sin
ese dato quedarse en 0.7 habría sido precaución y no evidencia.

## La medición que faltaba: ¿temperatura 1.0 inventa más?

32 fragmentos de 4 temas, **los mismos para las dos temperaturas** —compararlas
sobre fragmentos distintos mediría la dificultad del texto, no el efecto del
parámetro— y `gpt-4o` juzgando las 62 preguntas resultantes:

| Temperatura | Formato ok | Sin defectos |
|---|---|---|
| 0.7 | 31/32 | **27/31 (87.1%)** |
| 1.0 | 31/32 | **27/31 (87.1%)** |

| Tipo de defecto | 0.7 | 1.0 |
|---|---|---|
| Sin respaldo en el fragmento | 1 | 2 |
| Distractor también cierto | 1 | 2 |
| Traducción | 3 | 1 |
| No sirve como pregunta | 0 | 0 |
| Tema equivocado | 0 | 0 |

**Idénticas en el total.** Los defectos de **invención** —sin respaldo y
distractor cierto— pasan de 2 a 4 al subir la temperatura, que es la dirección
que predice la teoría, pero con n=31 no es significativo.

**Decisión: se queda en 0.7.** No porque 1.0 sea peor —no se pudo demostrar que
lo sea— sino porque no se gana nada subiendo: la calidad es la misma y el único
indicio disponible apunta en la dirección equivocada. La variedad extra (2.70
contra 2.30) no compensa un riesgo no descartado.

`repetition_penalty` sí se movió de 1.05 a 1.1, porque 1.05 no estaba en la
grilla medida y 1.1 sí.

**Sobre `top_p` y `top_k`:** se fijan en 0.9 y 50 y no se barren. No es pereza —
los tres, junto con la temperatura, recortan la misma cola de la distribución, y
barrerlos juntos mediría tres veces el mismo efecto.

**Sobre `max_tokens`:** no existe en `transformers`. Es el nombre de la API de
OpenAI; el equivalente es `max_new_tokens` (tokens generados), y existe además
`max_length` (prompt + generación). Se usa el primero porque no depende del largo
del fragmento, que varía mucho.

---

# 5.d El fallo más instructivo: la auto-verificación no verificaba

Este es el hallazgo con más valor para la sustentación, porque muestra un modo de
fallo que no aparece en ningún tutorial.

La aplicación ofrecía «auto-verificación»: el modelo revisa su propia pregunta
antes de mostrarla. Estaba activada por defecto y duplicaba el tiempo de
generación.

**No rechazaba nada.** El contador daba 0 de 26, después 0 de 16, después 0 de 17.

Un cero no distingue *«no había nada que rechazar»* de *«está inerte»*, así que se
construyó una sonda: tres preguntas rotas a propósito —una respuesta inventada,
una pregunta de otra enfermedad, un fragmento de navegación— más un control sano.
**Aprobó las cuatro.**

La causa está en la salida cruda:

```
prompt de verificación  →  {"apto": true, "pregunta": "¿Cuál es el tratamiento
                            recomendado para el cáncer de mama...", "correcta": ...}
```

**El modelo afinado no verificaba: regeneraba.** El LoRA lo especializó tanto en
producir preguntas que cualquier entrada le produce una pregunta. Y como esa
respuesta no traía las claves `respaldo` / `distractor_cierto` / `tema_correcto`,
el lector no encontraba ningún `False` y lo contaba como aprobación.

## La solución salió gratis

`disable_adapter()` apaga el LoRA y devuelve el modelo base, que es un instruct de
propósito general y **no perdió la capacidad de seguir instrucciones**:

| Sonda | Afinado | Base |
|---|---|---|
| Respuesta correcta inventada | aprueba ❌ | **rechaza** ✓ |
| Pregunta de otra enfermedad | aprueba ❌ | **rechaza** ✓ |
| Fragmento de navegación | aprueba ❌ | aprueba ❌ |
| Control: pregunta sana | aprueba ✓ | aprueba ✓ |

No cuesta memoria: base y afinado son el mismo peso con el LoRA encendido o
apagado. El contador pasó a rechazar **2 de 18 (11%)**.

El caso que sigue pasando es el del fragmento de navegación, y es coherente: su
defecto es de **utilidad**, no de veracidad, y la rúbrica no pregunta eso.
`gpt-4o` tampoco lo detecta. Ese caso lo cubre el filtro de fragmentos.

## Las dos lecciones generalizables

**Un especialista pierde generalidad.** Afinar un modelo para una tarea degrada
su capacidad de hacer otras, incluso otras que parecen cercanas —«revisar una
pregunta» está a un paso de «escribir una pregunta»—. Si un sistema necesita que
el mismo modelo genere y evalúe, hay que verificar que sigue sabiendo evaluar.

**Un validador que nunca rechaza no es un validador.** Merece la misma
desconfianza que un test que siempre pasa. La forma de comprobarlo es darle
entradas que *debe* rechazar, no mirar su tasa de aprobación.

---

# 5.e El filtro de fragmentos, y una corrección de alcance

Un fragmento puede estar limpio y aun así no servir. Caso real, detectado usando
la propia aplicación:

> **¿Cuál es el tratamiento recomendado para el cáncer de mama en etapas I a IIIA?**
> ✓ Ver opciones para cáncer localizado o operable
> ✗ Ver opciones para cáncer recidivante localizado

El fragmento del CancerGov era una lista de referencias cruzadas
(`for treatment options... see X`), no contenido médico. Las opciones son
etiquetas de navegación.

Se añadió `es_util()` en `banco.py`: descarta fragmentos donde al menos un tercio
de las oraciones son navegación, o que traen dos o más señales de directorio
(teléfonos, URLs, códigos postales).

**Los umbrales se midieron, no se supusieron**, y tres firmas plausibles se
descartaron por falsos positivos:

| Firma probada | Por qué se descartó |
|---|---|
| `refer to` | captura *«researchers refer to this form as type 1»* |
| Tres o más signos `%` | captura *«50% chance to be affected»* |
| Líneas cortas | los chunks no conservan saltos de línea |

**Corrección de alcance, importante:** al proponer este filtro se afirmó que
quitaría «la clase de defecto más visible», citando que las tablas eran el 15%
del corpus. Ese 15% era del corpus **original**: las tablas ya se habían
eliminado en el notebook 01. Lo que este filtro quita son **79 fragmentos de
28,294 — el 0.28%**. Es correcto y preciso, pero mucho menos importante de lo
que se anunció.

---

# 5.f ¿Sirvió entrenar? La comparación que faltaba

El notebook 03 comparó base y afinado **solo en forma**, y el juez comparó al
afinado contra el **maestro**, nunca contra el base. La pregunta central —¿las
preguntas del afinado son mejores que las del base con few-shot?— no estaba
medida, y ningún documento lo advertía.

El notebook 05 la mide: los mismos 150 fragmentos del juez del notebook 03, la
misma rúbrica, decodificación greedy, y los tres modelos juzgados en la misma
corrida.

## Contenido: sin diferencia demostrable

| | maestro | base + few-shot | afinado |
|---|---|---|---|
| Correcta respaldada | 96.5% | 97.8% | 95.2% |
| Sin distractor cierto | 96.5% | 96.3% | 95.2% |
| Una sola respuesta | 95.8% | 96.3% | 94.5% |
| Tema correcto | 83.1% | 77.0% | 82.8% |
| **Sin ningún defecto** | **78.2%** | **72.6%** | **75.9%** |

Prueba pareada (McNemar exacto) sobre los 126 fragmentos con los tres
veredictos:

| Comparación | Solo acierta el primero | Solo acierta el segundo | p |
|---|---|---|---|
| afinado contra base | 18 | 15 | 0.73 |
| afinado contra maestro | 8 | 13 | 0.38 |
| base contra maestro | 12 | 20 | 0.22 |

Criterio por criterio, afinado contra base, tampoco hay nada significativo. La
diferencia mayor, «tema correcto» (107 contra 101 de 130), da p = 0.31; en
respaldo el base queda incluso por encima (127 contra 123, p = 0.29).

**Lectura:** con esta muestra no se puede afirmar que el fine-tuning mejorara la
corrección del contenido, ni que la empeorara. Y corrige una lectura de §5.b: el
base sin entrenar **tampoco** se distingue del maestro, así que el empate
alumno-maestro no era, por sí solo, evidencia de que el entrenamiento aportara
contenido.

## Forma: ahí sí hay diferencia

| Medición | Base con estructura rota | Afinado |
|---|---|---|
| Notebook 03, greedy | 3 de 60 | 0 de 60 |
| Notebook 05, greedy | 4 de 150 | 0 de 150 |
| Muestreo a 0.7 y 1.0 (notebooks 04 y 05) | 5 de 120 | 0 de 120 |
| **Total** | **12 de 330** | **0 de 330** |

Los fragmentos de las tres mediciones se solapan en parte, así que no son 330
casos independientes; pero el patrón es el mismo en las tres.

## La robustez al muestreo no era mérito del fine-tuning

§5.c la atribuyó al entrenamiento sin control. Con control, sobre los mismos 20
fragmentos y 3 repeticiones:

| Modelo | Temperatura | Estructura | Variedad |
|---|---|---|---|
| base + few-shot | 0.7 | 57/60 | 2.30 |
| afinado | 0.7 | 60/60 | 2.30 |
| base + few-shot | 1.0 | 58/60 | 2.65 |
| afinado | 1.0 | 60/60 | 2.70 |

El base también aguanta. La diferencia es la misma de siempre: dos o tres
estructuras rotas de cada 60.

## Ruido entre corridas

El mismo afinado sobre los mismos 150 fragmentos dio 78.4% en el notebook 03 y
75.9% en el 05; el maestro, 79.3% y 78.2%. El juez a temperatura 0 no es del todo
determinista, y los fallos de la API —24 de 450 llamadas en el notebook 05—
excluyen fragmentos distintos en cada corrida. **Variaciones de 2 o 3 puntos son
ruido, no cambios.**

## Qué compró el fine-tuning, en limpio

| Aspecto | ¿Mejoró? | Evidencia |
|---|---|---|
| Corrección del contenido | no demostrable | 75.9% contra 72.6%, p = 0.73 |
| Estructura válida | sí, de forma consistente | 0 contra 12 fallos en 330 generaciones |
| Tamaño del prompt | sí | 137 contra 695 tokens: 5.1 veces menos |
| Robustez al muestreo | no es mérito suyo | el base da 57-58 de 60 a 0.7 y 1.0 |
| Velocidad | no, empeora | 1.01 s contra 0.80 s por pregunta, en lotes |

**Cómo defenderlo:** el fine-tuning convirtió una tarea que necesitaba un prompt
de 695 tokens con ejemplos en una que se resuelve con 137 y que nunca rompe el
formato. No hizo al modelo más preciso. Presentarlo así resiste la pregunta de si
se comparó contra el base, que la versión anterior de este documento no
resistía.

---

# 6. Errores propios encontrados y corregidos

Se documentan porque muestran el método, y porque un jurado que pregunte "¿cómo
saben que esto está bien?" merece saber qué se revisó.

| Error | Cómo se detectó | Corrección |
|---|---|---|
| Recortaba el inicio pero no el final de los fragmentos | leyendo 12 fragmentos a mano | recorte en ambos extremos; 78% estaba afectado |
| El filtro de listas contaba líneas con viñeta, y el corpus las trae aplanadas en un renglón | inspeccionando los descartes | detección por densidad de separadores |
| La función de recorte tenía una guarda que solo contemplaba minúsculas | verificando arranques malformados | alinear las dos condiciones; de 33 casos a 5 |
| El código de reanudación contaba los registros fallidos como "ya hechos" | revisando los 11 errores de API | solo cuentan los que tienen respuesta; se recuperaron los 11 |
| Conservar `category` fue inútil: queda 94% vacía tras la limpieza | mirando la distribución antes de usarla | el selector usa `question_type` y `question_focus` |

## Errores de medición

Aparte, tres veces una métrica automática dio un resultado falso. Se anotan
porque la lección se repite: **una métrica que compara texto sin mirar el
contexto engaña**.

- **"53% de las preguntas no nombran la enfermedad".** Falso: el detector
  comparaba el tema en inglés contra la pregunta en español, así que marcaba como
  fallo *"¿Cómo se hereda la hipobetalipoproteinemia?"*. El valor real era **2.1%**.
- **El juez copiaba el placeholder del prompt.** Devolvía `{"unsupported": ["c"]}`
  porque `"c"` era el ejemplo de formato, y se contaba como afirmación sin
  respaldo.
- **La demostración de las tablas HPO agarró el fragmento equivocado**, uno que
  tenía prosa antes de la tabla, y por eso "demostraba" lo contrario.

---

## Errores de esta etapa (entrenamiento y aplicación)

| Error | Cómo se detectó |
|---|---|
| El adaptador se guardaba en `adapter/` y la app lo leía en `app/adapter/` | la aplicación no arrancaba |
| Los contadores del panel eran acumulativos entre exámenes | un «26 generadas» que no cerraba con 16+2 |
| El juez externo imprimía «sin problemas. sin problemas» | visible en pantalla |
| El notebook 03 se lanzó sin `--allow-errors`: al fallar una celda, nbconvert descartó **todas** las salidas | se perdieron 2h30 de resultados ya calculados |
| Se estimó el notebook 04 en 12 minutos y tardó 48 | se usó la velocidad por lotes (1.58 s) en vez de la individual (6.61 s), que ya estaba medida |
| Un comentario en `modelo.py` afirmaba que `repetition_penalty` por encima de 1.15 rompía el JSON | el barrido lo desmintió: 1.2 da 60/60 |
| El heredoc de Bash convirtió `\b` en un carácter de retroceso dentro del regex del filtro | el filtro descartaba 77 fragmentos pero **no** el que se quería atrapar |

El penúltimo y el último tienen una moraleja común: **un componente que produce
un número plausible no está verificado**. El filtro parecía funcionar porque
descartaba fragmentos; solo probarlo contra el caso concreto reveló que estaba
roto. Lo mismo con el verificador y su contador en cero.

---

# 7. Limitaciones conocidas

| Limitación | Alcance |
|---|---|
| **~1 de cada 4 o 5 preguntas tiene algún defecto** | 78.4% sin defectos en el notebook 03, 75.9% en el 05, con los mismos fragmentos |
| **El fine-tuning no mejoró la corrección de forma demostrable** | 75.9% contra 72.6% del base con few-shot, p = 0.73 (notebook 05) |
| «Tema correcto» al 83% | es el techo del maestro, no una falla del alumno |
| Ninguna pregunta de dificultad «difícil» | el modelo no usa la categoría |
| El juez `gpt-4o` también se equivoca | marcó un error de traducción con una justificación que se contradice sola |
| El 78.4% mide corrección factual, no utilidad pedagógica | una pregunta puede estar respaldada, ser del tema y aun así no servir |
| `revisar_forma()` nunca se validó contra un juicio humano | se sabe que dejó de rechazar de más; no se sabe si rechaza lo suficiente |
| El auto-verificador no detecta defectos de utilidad | su rúbrica pregunta por veracidad, no por si la pregunta sirve |
| El afinado es **más lento** por pregunta que el base | 6.61 s contra 4.46 s de a una, 1.01 s contra 0.80 s en lotes; probablemente porque el LoRA sin fusionar agrega cómputo en cada capa (no medido) |
| Cobertura: 3,794 temas de 5,127 | los perdidos eran tablas HPO |
| Preguntas mayormente de tipo `information` y `treatment` | consecuencia de filtrar las tablas HPO, casi todas `symptoms` |
| No se midió si temperatura 1.0 empeora la corrección | el barrido del notebook 04 mide formato y variedad, no veracidad |

---

# 8. Preguntas previsibles del jurado

**«¿Esto no es simplemente destilar gpt-4o-mini en un modelo chico?»**

Sí, exactamente eso, y conviene decirlo con su nombre: destilación de conocimiento
mediante datos sintéticos. Lo defendible es lo que se gana: el modelo corre local
sin API y con costo cero por consulta, cubre 23,044 fragmentos que nunca pasaron
por la API, y la evaluación mide cuánto se acerca el alumno al maestro. Lo que
**no** se puede afirmar es que el modelo afinado sea mejor que gpt-4o-mini: no lo
es y no se midió contra él.

**«¿Cómo saben que las preguntas son correctas?»**

Un juez independiente (`gpt-4o`) leyó el fragmento fuente y verificó respaldo de
la correcta, falsedad de los distractores, unicidad de la respuesta y
correspondencia con el tema. Sobre el dataset de entrenamiento: 92.7% sin
defectos. Sobre lo que **genera el modelo afinado**, que es lo que ve el
estudiante: **78.4%**, contra 79.3% del maestro que lo entrenó.

La diferencia entre esos dos números no es una contradicción: el 92.7% mide el
dataset ya filtrado, y el 78.4% mide generación nueva sobre fragmentos que el
modelo nunca vio.

**«Si una de cada cinco preguntas tiene un defecto, ¿la aplicación sirve?»**

Como asistente de estudio, sí; como generador autónomo de exámenes, no. La
aplicación está diseñada alrededor de esa limitación: **cada pregunta muestra
siempre su fuente del NIH con enlace al documento original**, que es la única
defensa real del estudiante. Y en la revisión hay un botón para pedir una
segunda opinión a `gpt-4o` sobre cualquier pregunta dudosa.

**«¿Sirvió de algo el fine-tuning?»**

Primero una distinción: «corre en local, sin API y sin costo» justifica usar un
modelo local, no afinarlo, porque el Qwen base también corre en local. La
comparación que decide es contra el base con few-shot, y está medida en el
notebook 05:

- **Contenido: no se pudo demostrar una mejora.** 75.9% sin defectos contra
  72.6% del base, sobre los mismos 150 fragmentos; la prueba pareada da
  p = 0.73, y ningún criterio por separado es significativo.
- **Forma: sí.** El afinado no rompió la estructura en 330 generaciones; el base,
  12 veces.
- **Prompt: 5.1 veces más corto** (137 tokens contra 695), sin ejemplos que
  mantener.
- **Velocidad: empeora** (1.01 s contra 0.80 s por pregunta, en lotes).

El fine-tuning convirtió una tarea que necesitaba un prompt largo con ejemplos en
una que se resuelve con uno corto y nunca rompe el formato. No hizo al modelo más
preciso. Decirlo así es más sólido que atribuirle lo que la medición no respalda.

**«¿Por qué no usaron un modelo más nuevo?»**

Se examinó `Qwen3.5-4B` y hay tres razones, de menor a mayor peso:

1. Es multimodal: 0.33 B de parámetros de visión que no se usarían.
2. **24 de sus 32 capas usan atención lineal**, con proyecciones de otro nombre,
   así que la configuración de LoRA elegida no se transfiere; y los kernels
   rápidos de esa atención no están disponibles en esta máquina Windows.
3. **El límite del proyecto está en los datos, no en el alumno.** El modelo
   afinado ya empata con el maestro que generó sus datos (78.4% contra 79.3%).
   Un alumno más nuevo no sube el techo que fija el maestro.

Detalle y diseño del piloto que lo zanjaría, en §4.1.

**«¿Por qué el corpus está en inglés y las preguntas en español?»**

MedQuAD es íntegramente en inglés. Traducir 28,294 fragmentos costaría más que
generar las preguntas. En vez de eso, el modelo aprende la conversión: recibe
inglés y produce español. Es una de las habilidades que el fine-tuning enseña.

**«¿Por qué descartaron el 41% de los fragmentos?»**

Porque no todo texto sirve para preguntar. El 15% del corpus son tablas
automáticas del Human Phenotype Ontology, que producen preguntas con varias
respuestas correctas. Se puede mostrar el ejemplo comparado de §2.5.

---

# 9. Lo que falta

- **Validar `revisar_forma()` contra un juicio humano.** Hoy solo se sabe que
  dejó de rechazar de más. Etiquetar a mano 50 preguntas y medir cuántas de las
  que aprueba deberían haberse rechazado.

Es una mejora de medición, no de producto. El proyecto está completo:
notebooks 01 a 05 ejecutados con sus salidas, adaptador entrenado y aplicación
funcionando y auditada.
