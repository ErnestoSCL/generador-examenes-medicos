# Generador de exámenes médicos en español

Simulacro de examen de opción múltiple generado en el momento por un modelo de
4.000 millones de parámetros que corre **en la máquina local, sin API y sin costo
por consulta**, a partir del corpus [MedQuAD](https://github.com/abachaa/MedQuAD)
de los Institutos Nacionales de Salud de Estados Unidos.

El corpus está íntegramente en inglés y las preguntas salen en español: el modelo
hace la conversión al generar.

---

## Qué hace

El estudiante elige un tema —hay 2.420 con material suficiente— y la aplicación
genera un examen. Cada pregunta muestra **siempre la fuente del NIH con enlace al
documento original**, que es la defensa real del estudiante contra la cuarta o
quinta parte de preguntas que sale con algún defecto.

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
| `05_base_vs_afinado.ipynb` | base contra afinado en calidad (~USD 1 de API) | ~20 min |

Para los notebooks 02 y 05 hace falta un `.env` en la raíz con
`OPENAI_API_KEY=...`.

Después:

```bash
venv/Scripts/streamlit.exe run app/app.py
```

> **Nota sobre los notebooks.** Al ejecutarlos con `nbconvert`, usar siempre
> `--allow-errors`. Sin esa bandera, si una celda falla nbconvert descarta
> **todas** las salidas, incluidas las de las celdas que sí corrieron.

---

## Resultados

### ¿Sirvió el fine-tuning? Lo que compró y lo que no

La comparación que decide no es contra la API, sino contra el **mismo Qwen sin
afinar, con few-shot**, también en local. Sobre los mismos 150 fragmentos
(notebook 05):

| | base + few-shot | afinado |
|---|---|---|
| Estructura completa | 146/150 | **150/150** |
| Tokens de prompt | 695 | **137** |
| Segundos por pregunta (lotes de 8) | **0,80** | 1,01 |
| Sin ningún defecto, según el juez gpt-4o | 72,6% | 75,9% |

**Lo que compró, medido:** un prompt **5,1 veces más corto**, sin ejemplos que
elegir ni mantener, y una estructura que **no se rompe**: 0 fallos en 330
generaciones entre los notebooks 03 y 05, contra 12 del base.

**Lo que no compró:** preguntas más correctas de forma demostrable. La prueba
pareada fragmento a fragmento da **p = 0,73**, y ningún criterio por separado es
significativo. Tampoco velocidad: el afinado es más lento, probablemente porque
el LoRA sin fusionar agrega cómputo en cada capa (no medido).

### Los tres modelos ante el mismo juez

| | maestro (gpt-4o-mini) | base + few-shot | afinado |
|---|---|---|---|
| Correcta respaldada | 96,5% | 97,8% | 95,2% |
| Sin distractor cierto | 96,5% | 96,3% | 95,2% |
| Tema correcto | 83,1% | 77,0% | 82,8% |
| **Sin ningún defecto** | **78,2%** | **72,6%** | **75,9%** |

Con esta muestra, **ninguna diferencia entre los tres es estadísticamente
significativa** (p entre 0,22 y 0,73). El alumno no se distingue de su maestro
—la destilación cumplió su objetivo de igualarlo sin depender de él—, pero el
base sin entrenar tampoco.

En una corrida anterior sobre los mismos fragmentos, el afinado dio 78,4% y el
maestro 79,3%: variaciones de 2 o 3 puntos entre corridas son ruido.

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

El barrido del notebook 04 (420 generaciones) mostró que el afinado **no pierde
el formato en ningún punto**: las siete configuraciones dan 60/60 de JSON válido,
incluida temperatura 1.0, con cero truncamientos. El base con few-shot también
aguanta el muestreo (57 y 58 de 60 a temperatura 0,7 y 1,0), así que no es un
mérito exclusivo del fine-tuning: lo que el afinado agrega es no romper nunca la
estructura.

Temperatura 1.0 da más variedad (2,70 preguntas distintas de 3, contra 2,45) sin
costo de formato, y se midió aparte si costaba corrección: **87,1% sin defectos
en las dos**. Se eligió 0,7 porque no se gana calidad subiendo y los defectos de
invención —sin respaldo, distractor cierto— pasan de 2 a 4, dentro del ruido pero
en la dirección equivocada.

---

## Limitaciones

| Limitación | Alcance |
|---|---|
| Una de cada 4 o 5 preguntas tiene algún defecto | 76-78% sin defectos según la corrida |
| El fine-tuning no mejoró la corrección de forma demostrable | 75,9% contra 72,6% del base con few-shot, p = 0,73 |
| «Tema correcto» alrededor del 83% | es también el techo del maestro |
| El juez mide corrección factual, no utilidad pedagógica | una pregunta puede estar respaldada y aun así no servir |
| Ninguna pregunta de dificultad «difícil» | el modelo no usa la categoría |
| El juez `gpt-4o` también se equivoca | ese ruido está dentro de las tasas medidas |
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
