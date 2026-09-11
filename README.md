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

## Todo en un notebook, listo para Colab

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ErnestoSCL/generador-examenes-medicos/blob/main/notebooks/00_proyecto_completo_colab.ipynb)

[`notebooks/00_proyecto_completo_colab.ipynb`](notebooks/00_proyecto_completo_colab.ipynb)
reúne los cinco notebooks en un solo recorrido. **Se puede leer sin ejecutarlo**:
trae guardadas las salidas de las ejecuciones completas. Para ejecutarlo en Colab:

- Elige un entorno con GPU. La T4 gratuita alcanza para la prueba rápida: el
  notebook pasa a float16 si la GPU no tiene bfloat16. Para reproducir todo
  conviene una L4 o una A100.
- Con la configuración por defecto (`PRUEBA_RAPIDA = True`, `USAR_API = False`)
  no se gasta nada: el dataset se reconstruye con las respuestas originales de
  `gpt-4o-mini` guardadas en `data/mcq_crudo.jsonl`, y queda idéntico al que se
  usó para entrenar.

## Cómo correrlo en local

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
| `05_afinado_contra_base.ipynb` | el afinado contra el base con cuatro prompts (~USD 2 de API) | ~25 min |

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

### ¿Sirvió el fine-tuning?

La comparación que decide no es contra la API, sino contra el **mismo Qwen sin
afinar, en local, con el mejor prompt posible**. El notebook 05 lo compara con
cuatro prompts sobre los mismos 150 fragmentos, con un solo juez (gpt-4o):

| Fuente | Tokens de prompt | Estructura válida | Sin ningún defecto | Contra el afinado |
|---|---|---|---|---|
| maestro (gpt-4o-mini) | — | 150/150 | 80,0% | empate, p = 0,38 |
| **afinado** | **137** | **150/150** | **76,7%** | — |
| base B: instrucción + esquema | 187 | 143/150 | 72,7% | empate, p = 0,39 |
| base + few-shot | 695 | 146/150 | 70,0% | empate, p = 0,12 |
| base C: las reglas del maestro | 748 | 125/150 | 56,7% | **afinado mejor, p < 0,0001** |
| base A: el mismo prompt | 137 | 0/150 | 0% | **afinado mejor, p < 0,0001** |

«Sin ningún defecto» cuenta como fallo cada estructura rota.

**Lo que compró:**

- **Las reglas del maestro, incorporadas.** Dárselas al base por escrito da
  56,7% contra 76,7% del afinado (p < 0,0001): el modelo de 4B no logra
  aplicarlas leyéndolas en el prompt.
- **El formato.** Con el mismo prompt, el base inventa sus propias claves en las
  150 respuestas, y ninguna variante del base mantiene la estructura siempre
  (rompe 4, 7 y 25 de 150). El afinado no rompe ninguna.
- **Alcanzar al maestro.** El afinado no se distingue de gpt-4o-mini (p = 0,38);
  el base con few-shot queda por debajo (p = 0,014).
- **Un prompt corto:** 137 tokens contra 695 del few-shot y 748 de las reglas del
  maestro.

**Lo que no compró:**

- Frente a un prompt mínimo con el esquema (187 tokens), **empata en contenido**
  (p = 0,39). Ahí la ventaja del prompt es de 1,4 veces, no de 5.
- La diferencia directa con el few-shot no es significativa (p = 0,12), aunque el
  few-shot no alcance al maestro y el afinado sí.
- **Velocidad:** es el más lento, con 0,97 s por pregunta contra 0,76-0,85 s del
  base, probablemente porque el LoRA sin fusionar agrega cómputo en cada capa (no
  medido).

Entre corridas con los mismos fragmentos las cifras varían 2 o 3 puntos: el
afinado dio 78,4% en el notebook 03 y 76,7% en el 05.

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
aguanta el muestreo (58 de 60 a temperatura 0,7 y a 1,0), así que no es un mérito
exclusivo del fine-tuning: lo que el afinado agrega es no romper nunca la
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
| Frente a un buen prompt, el fine-tuning no mejora el contenido de forma demostrable | empata con el few-shot (p = 0,12) y con un prompt mínimo con esquema (p = 0,39) |
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
| [INSIGHTS_Y_DECISIONES.md](INSIGHTS_Y_DECISIONES.md) | decisiones con su evidencia, errores corregidos, guion de la exposición (§7.b) y preguntas previsibles del jurado |
| [PLAN_IMPLEMENTACION.md](PLAN_IMPLEMENTACION.md) | diseño de cada notebook y de la aplicación |
| [CONTEXTO_PROYECTO.md](CONTEXTO_PROYECTO.md) | estado, estructura de archivos y riesgos |

---

## Licencia y origen de los datos

MedQuAD proviene de sitios de los Institutos Nacionales de Salud de EE.UU. y su
contenido es de dominio público. Las preguntas generadas son material de estudio:
**no constituyen consejo médico**.
