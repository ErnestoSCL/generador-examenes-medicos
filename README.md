# Generador de exámenes médicos en español

Simulacro de examen de opción múltiple generado en el momento por un modelo de
4.000 millones de parámetros que corre **en la máquina local, sin API y sin costo
por consulta**, a partir del corpus [MedQuAD](https://github.com/abachaa/MedQuAD)
de los Institutos Nacionales de Salud de Estados Unidos.

El corpus está íntegramente en inglés y las preguntas salen en español: la
conversión es una de las habilidades que el fine-tuning enseña.

---

## Qué hace

El estudiante elige un tema —hay 2.420 con material suficiente— y la aplicación
genera un examen. Cada pregunta muestra **siempre la fuente del NIH con enlace al
documento original**, que es la defensa real del estudiante contra el ~1 de cada
5 que sale con algún defecto.

<!-- Flujo -->

```
Usuario elige tema
        │
        ▼
  banco.py ─────────────  pandas sobre 28.215 fragmentos
        │                 (filtro es_util al cargar)
        │  16 fragmentos para 10 preguntas
        ▼
  modelo.py ────────────  Qwen3-4B + LoRA          ← MODELO AFINADO
        │  JSON en español
        ▼
  validacion.py ────────  reglas de forma
        │
        ▼
  modelo BASE ──────────  disable_adapter()        ← MISMO PESO, LoRA APAGADO
        │  auto-verificación
        ▼
  app.py ───────────────  examen + fuente NIH siempre visible
        │
        ▼
  panel.py ─────────────  juez gpt-4o, opcional y a pedido
```

**No es RAG.** No hay embeddings ni búsqueda por similitud: el usuario elige el
tema de una lista, así que un filtro exacto sobre columnas alcanza. Una base
vectorial haría falta solo si el tema se escribiera en texto libre.

---

## Cómo correrlo

Hace falta una GPU con 12 GB o más. Está probado en una RTX 5070 Ti de 16 GB.

```bash
python -m venv venv
venv/Scripts/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Los datos y los pesos no viajan en el repositorio; se reconstruyen ejecutando los
notebooks en orden:

| Notebook | Qué produce | Tiempo |
|---|---|---|
| `01_limpieza.ipynb` | `medqa_clean.parquet`, 14.528 filas | ~5 min |
| `02_dataset_mcq.ipynb` | el dataset de preguntas (USD 0,90 de API) | ~40 min |
| `03_finetuning.ipynb` | el adaptador LoRA en `app/adapter/` | ~2 h |
| `04_parametros_generacion.ipynb` | el barrido de muestreo | ~48 min |

Para el notebook 02 hace falta un `.env` en la raíz con `OPENAI_API_KEY=...`.

Después:

```bash
venv/Scripts/streamlit.exe run app/app.py
```

> **Nota sobre los notebooks.** Al ejecutarlos con `nbconvert`, usar siempre
> `--allow-errors`. Sin esa bandera, si una celda falla nbconvert descarta
> **todas** las salidas, incluidas las de las celdas que sí corrieron.

---

## Resultados

### El fine-tuning contra el modelo base

| | base + few-shot | afinado |
|---|---|---|
| JSON válido | 60/60 | 60/60 |
| Estructura completa | 57/60 | **60/60** |
| Tokens de prompt | 693 | **135** |
| Segundos por pregunta | **4,46** | 6,61 |

La ventaja real son los **tokens de prompt: 5,1 veces menos**. El afinado es más
lento por pregunta, porque el LoRA agrega cómputo en cada capa.

### El alumno contra su maestro

`gpt-4o` juzgando 150 preguntas de cada uno con la misma rúbrica:

| | maestro (gpt-4o-mini) | alumno (Qwen afinado) |
|---|---|---|
| Correcta respaldada | 97,2% | 97,1% |
| Sin distractor cierto | 97,2% | 95,7% |
| Tema correcto | 83,4% | 82,7% |
| **Sin ningún defecto** | **79,3%** | **78,4%** |

Con n≈140 por lado, 0,9 puntos está dentro del ruido: **el alumno alcanzó al
maestro**. La destilación cumplió su objetivo, que nunca fue superarlo sino
igualarlo sin depender de él.

Tiene una consecuencia práctica: ningún ajuste de entrenamiento va a mejorar el
modelo, porque está en su techo. El único movimiento que queda es cambiar el
maestro.

### Configuración elegida

Barrido de 6 configuraciones de LoRA (notebook 03). Ganó `r=32, alpha=64,
lr=2e-4` sobre los 7 módulos, con `eval_loss` 0,3506.

**Pero la dispersión total entre las seis fue 0,0208.** Configuraciones que van
de 11,8 M a 66,1 M de parámetros entrenables se separan en dos centésimas: para
esta tarea la configuración casi no importa, lo que importó fue el dataset.

### Parámetros de generación

```python
do_sample=True, temperature=0.7, top_p=0.9, top_k=50,
repetition_penalty=1.1, max_new_tokens=300
```

El barrido del notebook 04 (420 generaciones) mostró que **el formato no se rompe
en ningún punto**: las siete configuraciones dan 60/60 de JSON válido, incluida
temperatura 1.0, con cero truncamientos. Es mérito del fine-tuning.

Temperatura 1.0 da más variedad (2,70 preguntas distintas de 3, contra 2,30) sin
costo de formato, y se midió aparte si costaba corrección: **87,1% sin defectos
en las dos**. Se eligió 0,7 porque no se gana calidad subiendo y los defectos de
invención —sin respaldo, distractor cierto— pasan de 2 a 4, dentro del ruido pero
en la dirección equivocada.

---

## Limitaciones

| Limitación | Alcance |
|---|---|
| ~1 de cada 5 preguntas tiene algún defecto | 78,4% sin defectos sobre 139 |
| «Tema correcto» al 83% | es el techo del maestro, no una falla del alumno |
| El 78,4% mide corrección factual, no utilidad pedagógica | una pregunta puede estar respaldada y aun así no servir |
| Ninguna pregunta de dificultad «difícil» | el modelo no usa la categoría |
| El juez `gpt-4o` también se equivoca | ese ruido está dentro del 78,4% |
| Alta varianza entre exámenes | con 10 preguntas el resultado va de 7 a 10 sin que cambie nada |

**Como asistente de estudio con la fuente visible, funciona. Como generador
autónomo de exámenes que alguien tome sin revisar, no.** La aplicación está
diseñada alrededor de esa limitación.

---

## Un hallazgo que vale la pena leer

La auto-verificación —el modelo revisando su propia pregunta— **estuvo inerte
durante toda la construcción**. Aprobaba todo, incluidas preguntas rotas a
propósito.

La causa: ante el prompt de verificación, el modelo afinado no verificaba,
**regeneraba**. El LoRA lo especializó tanto que cualquier entrada le produce una
pregunta. La solución fue verificar con el modelo **base**, que no perdió la
capacidad de seguir instrucciones. Detalle en
[INSIGHTS_Y_DECISIONES.md](INSIGHTS_Y_DECISIONES.md) §5.d.

Dos lecciones generalizables: **un especialista pierde generalidad**, y **un
validador que nunca rechaza no es un validador** — merece la misma desconfianza
que un test que siempre pasa.

---

## Documentación

| Archivo | Para qué |
|---|---|
| [INSIGHTS_Y_DECISIONES.md](INSIGHTS_Y_DECISIONES.md) | decisiones con su evidencia, errores corregidos, preguntas previsibles del jurado |
| [PLAN_IMPLEMENTACION.md](PLAN_IMPLEMENTACION.md) | diseño de cada notebook y de la aplicación |
| [CONTEXTO_PROYECTO.md](CONTEXTO_PROYECTO.md) | estado, estructura de archivos y riesgos |

---

## Licencia y origen de los datos

MedQuAD proviene de sitios de los Institutos Nacionales de Salud de EE.UU. y su
contenido es de dominio público. Las preguntas generadas son material de estudio:
**no constituyen consejo médico**.
