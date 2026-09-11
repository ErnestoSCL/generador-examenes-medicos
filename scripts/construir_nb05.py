"""Arma el notebook 05: el afinado contra el modelo base con cuatro prompts.

La version anterior del 05 comparaba el afinado solo contra el base con few-shot.
Esta agrega tres variantes sin ejemplos, y genera y juzga todo en una sola
corrida para que el notebook se sostenga solo.
"""
import io
import json

MD, CODE = "markdown", "code"
celdas = []


def md(texto):
    celdas.append((MD, texto.strip() + "\n"))


def code(texto):
    celdas.append((CODE, texto.strip() + "\n"))


md(r"""
# 05 - ¿Sirvió el fine-tuning? El afinado contra el modelo base con cuatro prompts

El notebook 03 comparó el modelo base con el afinado **solo en forma**. La calidad
del contenido la midió un juez, pero contra el **maestro** (`gpt-4o-mini`), nunca
contra el base. Eso dejaba sin responder la pregunta central del proyecto:
**¿sirvió entrenar?**

«Corre en local, sin API y sin costo» no justifica el fine-tuning: justifica usar
un modelo local, y el Qwen base también corre en local. Lo que justifica el
fine-tuning es superar al base **con el mejor prompt que se le pueda dar**. Por eso
se lo compara con cuatro prompts distintos:

| Variante | Instrucción | Esquema JSON | Reglas del maestro | Ejemplos | Qué pone a prueba |
|---|---|---|---|---|---|
| **afinado** | corta | aprendido | no | no | — |
| base + few-shot | corta | implícito en los ejemplos | no | 3 | la línea de base clásica |
| base A | corta | no | no | no | lo que aprendió el fine-tuning: recibe **el mismo prompt** que el afinado |
| base B | corta | **sí** | no | no | el prompt mínimo con el que el base sabe qué claves usar: la prueba más dura para el argumento del prompt corto |
| base C | la del maestro | **sí** | **sí** | no | exactamente las instrucciones que recibió `gpt-4o-mini` al generar el dataset: lo que se usaría sin entrenar |

## Diseño

| Decisión | Por qué |
|---|---|
| Los **mismos 150 fragmentos** que juzgó el notebook 03 (`random_state=13` sobre el test) | comparable con sus cifras, y nunca vistos en el entrenamiento |
| La **misma rúbrica** y el mismo juez (`gpt-4o`, temperatura 0) | que la diferencia sea del modelo, no de la vara |
| Decodificación **greedy** | que la diferencia no sea azar de muestreo |
| El few-shot usa los **mismos 3 ejemplos** que el notebook 03 | comparable con lo medido antes |
| **Todo se genera y se juzga en esta corrida**, incluido el maestro | seis fuentes, un juez, los mismos fragmentos; el notebook no depende de resultados guardados por otro |
| **Prueba pareada** (McNemar) | todos responden sobre los mismos fragmentos: se compara fragmento a fragmento |

La sección B hace el control de formato bajo muestreo que le faltó al notebook 04.

*Reemplaza una versión anterior de este notebook, que solo comparaba el afinado
contra el base con few-shot.*
""")

code(r'''
import json, re, time, unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = Path("../data")
ADAPTADOR = Path("../app/adapter")
MODELO = "Qwen/Qwen3-4B-Instruct-2507"
SEMILLA = 42
LOTE = 8
MAX_NEW = 300
torch.manual_seed(SEMILLA)

# Identica a la del entrenamiento (notebook 03) y a la de la aplicacion.
INSTRUCCION = (
    "Eres un docente de medicina. A partir del FRAGMENTO escribe UNA pregunta "
    "de opcion multiple en espanol neutro. Responde SOLO con JSON."
)
# El prompt con el que gpt-4o-mini genero el dataset (notebook 02).
MAESTRO = json.load(open("../scripts/prompt_v2.json", encoding="utf-8"))["prompt"]

train_df = pd.read_parquet(DATA / "mcq_train.parquet")
test_df = pd.read_parquet(DATA / "mcq_test.parquet")

tok = AutoTokenizer.from_pretrained(MODELO)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"            # obligatorio para generar por lotes

base = AutoModelForCausalLM.from_pretrained(MODELO, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(base, ADAPTADOR)
model.eval()
print(f"train {len(train_df):,} | test {len(test_df):,}")
print(f"GPU: {torch.cuda.get_device_name(0)} | memoria usada {torch.cuda.memory_allocated()/1e9:.2f} GB")
print("un solo modelo en memoria: el base se obtiene apagando el LoRA con disable_adapter()")
''')

code(r'''
def parsear(t):
    if not t:
        return None
    t = t.strip()
    if t.startswith("```"):
        t = t.strip("`").removeprefix("json").strip()
    try:
        d = json.loads(t)
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def estructura_ok(d):
    return bool(d and d.get("pregunta") and d.get("correcta")
                and isinstance(d.get("incorrectas"), list) and len(d["incorrectas"]) == 3)


def sin_acentos(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s).lower())
                   if unicodedata.category(c) != "Mn")


# Variante B: la instruccion corta mas el esquema, sin reglas ni ejemplos.
ESQUEMA = (INSTRUCCION + " Usa exactamente esta forma:\n"
           '{"apto": true, "pregunta": "...", "correcta": "...", '
           '"incorrectas": ["...", "...", "..."], "dificultad": "facil|media|dificil"}')

# Los mismos 3 ejemplos que recibio el base en el notebook 03.
ejemplos = train_df.sample(3, random_state=SEMILLA)
FEWSHOT = []
for _, e in ejemplos.iterrows():
    FEWSHOT.append({"role": "user", "content": "FRAGMENTO:\n" + e["chunk_text"]})
    FEWSHOT.append({"role": "assistant", "content": json.dumps({
        "apto": True, "pregunta": e["pregunta"], "correcta": e["correcta"],
        "incorrectas": list(e["incorrectas"]), "dificultad": e["dificultad"]},
        ensure_ascii=False)})


def chat(sistema, fragmento, previos=()):
    msgs = [{"role": "system", "content": sistema}, *previos,
            {"role": "user", "content": "FRAGMENTO:\n" + fragmento}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


VARIANTES = {   # clave: (etiqueta, usa el modelo base, constructor del prompt)
    "afinado": ("afinado (LoRA)", False, lambda f: chat(INSTRUCCION, f)),
    "fewshot": ("base + few-shot", True, lambda f: chat(INSTRUCCION, f, FEWSHOT)),
    "A": ("base A: mismo prompt", True, lambda f: chat(INSTRUCCION, f)),
    "B": ("base B: + esquema", True, lambda f: chat(ESQUEMA, f)),
    "C": ("base C: prompt del maestro", True, lambda f: chat(MAESTRO, f)),
}


def generar(prompts, usar_base, **muestreo):
    """Genera por lotes. Devuelve textos, tokens de prompt reales y truncadas."""
    kw = dict(max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id)
    kw.update(muestreo or dict(do_sample=False))
    textos, tokens_prompt, truncadas = [], [], 0
    for i in range(0, len(prompts), LOTE):
        ids = tok(prompts[i:i + LOTE], return_tensors="pt", padding=True).to(model.device)
        n = ids.input_ids.shape[1]
        with torch.no_grad():
            if usar_base:
                with model.disable_adapter():
                    out = model.generate(**ids, **kw)
            else:
                out = model.generate(**ids, **kw)
        for fila in out:
            nuevos = fila[n:]
            if int((nuevos != tok.pad_token_id).sum()) >= MAX_NEW:
                truncadas += 1           # llego al tope sin cerrar
            textos.append(tok.decode(nuevos, skip_special_tokens=True))
        tokens_prompt += ids.attention_mask.sum(1).tolist()   # sin contar el relleno
    return textos, tokens_prompt, truncadas

print("ejemplos de few-shot:", ejemplos["question_focus"].tolist())
print("variantes:", [v[0] for v in VARIANTES.values()])
''')

md(r"""
## A. Generación

Las cinco variantes generan sobre los mismos 150 fragmentos, todas con greedy.
""")

code(r'''
casos = test_df.sample(150, random_state=13).reset_index(drop=True)
fragmentos = casos["chunk_text"].tolist()

salidas, dicts, filas_forma = {}, {}, []
for clave, (etiqueta, usar_base, construir) in VARIANTES.items():
    t0 = time.time()
    textos, toks, _ = generar([construir(f) for f in fragmentos], usar_base)
    seg = (time.time() - t0) / len(fragmentos)
    ds = [parsear(t) for t in textos]
    salidas[clave], dicts[clave] = textos, ds
    filas_forma.append({
        "variante": etiqueta,
        "tokens de prompt": round(sum(toks) / len(toks)),
        "JSON valido": f"{sum(d is not None for d in ds)}/150",
        "estructura completa": f"{sum(estructura_ok(d) for d in ds)}/150",
        "descarto el fragmento (apto=false)": sum(1 for d in ds if d and d.get("apto") is False),
        "segundos por pregunta (lotes de 8)": round(seg, 2)})
    print(f"  {etiqueta}: listo en {seg * len(fragmentos) / 60:.1f} min", flush=True)

forma = pd.DataFrame(filas_forma).set_index("variante")
print()
print(forma.to_string())

claves_A = Counter(k for d in dicts["A"] if d for k in d.keys())
print("\nclaves que usa la variante A, que no conoce el esquema:", claves_A.most_common(8))
''')

code(r'''
try:
    from dotenv import load_dotenv
    load_dotenv("../.env")
except Exception:
    pass
from openai import OpenAI

client = OpenAI()
MODELO_JUEZ = "gpt-4o"

# La MISMA rubrica del notebook 03, palabra por palabra.
JUEZ = """Eres un revisor de examenes de medicina. Recibes un FRAGMENTO, el TEMA
al que pertenece, y una PREGUNTA de opcion multiple construida a partir de el.

Evalua con severidad:
1. respaldo: la opcion correcta esta afirmada explicitamente en el fragmento?
2. distractor_verdadero: alguna incorrecta es TAMBIEN cierta segun el fragmento?
3. unica_respuesta: hay exactamente una respuesta defendible?
4. tema_correcto: la pregunta se refiere al TEMA indicado y no a otra enfermedad?

Responde SOLO con JSON:
{"respaldo": true/false, "distractor_verdadero": true/false,
 "unica_respuesta": true/false, "tema_correcto": true/false,
 "comentario": "una frase solo si algo falla"}"""

ESTRUCTURA_ROTA = {"_estructura_rota": True}   # no se juzga: cuenta como fallo


def juzgar(args):
    idx, quien, fila, d = args
    if not estructura_ok(d):
        return idx, quien, ESTRUCTURA_ROTA, None
    texto = (f"TEMA: {fila['question_focus']}\n\nFRAGMENTO:\n{fila['chunk_text']}\n\n"
             f"PREGUNTA: {d['pregunta']}\nCORRECTA: {d['correcta']}\n"
             + "\n".join(f"INCORRECTA: {x}" for x in d["incorrectas"]))
    motivo = None
    for intento in range(5):
        try:
            r = client.chat.completions.create(
                model=MODELO_JUEZ, temperature=0,
                messages=[{"role": "system", "content": JUEZ},
                          {"role": "user", "content": texto}])
            v = parsear(r.choices[0].message.content)
            if v is not None:
                return idx, quien, v, None
            motivo = "respuesta no parseable"
        except Exception as exc:
            motivo = type(exc).__name__
        time.sleep(2 ** intento)                  # 1, 2, 4, 8, 16 segundos
    return idx, quien, None, motivo              # tras 5 intentos: se excluye


QUIENES = ["maestro"] + list(VARIANTES)
tareas = []
for i, fila in casos.iterrows():
    tareas.append((i, "maestro", fila, {"pregunta": fila["pregunta"], "correcta": fila["correcta"],
                                        "incorrectas": list(fila["incorrectas"])}))
    for clave in VARIANTES:
        tareas.append((i, clave, fila, dicts[clave][i]))

a_juzgar = sum(1 for t in tareas if estructura_ok(t[3]))
print(f"{len(tareas)} preguntas, {a_juzgar} con estructura valida para el juez...", flush=True)
veredictos = {q: {} for q in QUIENES}
motivos = Counter()
with ThreadPoolExecutor(6) as pool:
    for fut in as_completed([pool.submit(juzgar, t) for t in tareas]):
        idx, quien, v, motivo = fut.result()
        veredictos[quien][idx] = v
        if motivo:
            motivos[f"{quien}: {motivo}"] += 1
print("fallos del juez tras 5 intentos (se excluyen):", dict(motivos) or "ninguno")
''')

code(r'''
def limpia(v):
    # La misma definicion de "sin ningun defecto" del notebook 03.
    return bool(v and not v.get("_estructura_rota") and v.get("respaldo")
                and not v.get("distractor_verdadero") and v.get("unica_respuesta")
                and v.get("tema_correcto"))


ETIQUETAS = {"maestro": "maestro (gpt-4o-mini)", **{k: v[0] for k, v in VARIANTES.items()}}
filas = []
for quien in QUIENES:
    todas = [v for v in veredictos[quien].values() if v is not None]
    juzgadas = [v for v in todas if not v.get("_estructura_rota")]
    n = max(len(juzgadas), 1)
    filas.append({
        "modelo": ETIQUETAS[quien],
        "estructura valida": f"{len(juzgadas)}/{len(todas)}",
        "correcta respaldada": f"{sum(bool(v.get('respaldo')) for v in juzgadas)/n*100:.1f}%",
        "sin distractor cierto": f"{sum(not v.get('distractor_verdadero', True) for v in juzgadas)/n*100:.1f}%",
        "una sola respuesta": f"{sum(bool(v.get('unica_respuesta')) for v in juzgadas)/n*100:.1f}%",
        "tema correcto": f"{sum(bool(v.get('tema_correcto')) for v in juzgadas)/n*100:.1f}%",
        "SIN DEFECTO (de las juzgadas)": f"{sum(limpia(v) for v in juzgadas)/n*100:.1f}%",
        "SIN DEFECTO (de los fragmentos)": f"{sum(limpia(v) for v in todas)/max(len(todas),1)*100:.1f}%",
    })

calidad = pd.DataFrame(filas).set_index("modelo")
print(calidad.T.to_string())
''')

md(r"""
### Cómo leer la tabla

Hay dos filas de «sin defecto», y la diferencia pesa mucho en las variantes que
rompen la estructura:

- **de las juzgadas** solo cuenta las preguntas con estructura válida. Es la cifra
  comparable con el notebook 03, pero favorece a quien rompe muchas: si una
  variante solo produce 10 preguntas válidas y las 10 salen bien, da 100%.
- **de los fragmentos** cuenta como fallo cada estructura rota. Es la tasa que vive
  el usuario: de cada fragmento, ¿sale una pregunta aprovechable? **Es la que hay
  que mirar para comparar variantes.**

### La prueba pareada

Como todos respondieron sobre **los mismos** fragmentos, lo que decide es en
cuántos acierta uno y falla el otro: los pares discordantes. La prueba de McNemar
pregunta si esos discordantes se reparten de forma tan desigual que no puede ser
azar. Aquí se compara siempre **el afinado contra cada una de las otras fuentes**,
contando la estructura rota como fallo.
""")

code(r'''
from scipy.stats import binomtest

validos = [i for i in range(len(casos))
           if all(veredictos[q].get(i) is not None for q in QUIENES)]
ok = {q: {i: limpia(veredictos[q][i]) for i in validos} for q in QUIENES}

filas_p = []
for otro in [q for q in QUIENES if q != "afinado"]:
    solo_af = sum(ok["afinado"][i] and not ok[otro][i] for i in validos)
    solo_otro = sum(ok[otro][i] and not ok["afinado"][i] for i in validos)
    p = binomtest(solo_af, solo_af + solo_otro, 0.5).pvalue if solo_af + solo_otro else 1.0
    filas_p.append({"afinado contra": ETIQUETAS[otro],
                    "solo acierta el afinado": solo_af, "solo acierta el otro": solo_otro,
                    "p (McNemar exacto)": round(p, 4)})

pareadas = pd.DataFrame(filas_p).set_index("afinado contra")
print(f"fragmentos con los seis veredictos: {len(validos)}\n")
print(pareadas.to_string())
print("\np < 0.05: la diferencia difícilmente es azar. p alto: no se distingue del ruido.")
''')

md(r"""
### Criterio por criterio

Si el total no muestra diferencia, puede que un criterio concreto sí la tenga y
quede diluido. Aquí solo entran los fragmentos donde **las dos** preguntas se
pudieron juzgar, así que mide la calidad del contenido sin mezclarla con la
estructura.
""")

code(r'''
real = lambda v: v is not None and not v.get("_estructura_rota")
BUENO = {"respaldo": True, "distractor_verdadero": False,
         "unica_respuesta": True, "tema_correcto": True}

filas_c = []
for otro in ("fewshot", "A", "B", "C"):
    pares = [(veredictos[otro].get(i), veredictos["afinado"].get(i)) for i in range(len(casos))]
    pares = [(o, a) for o, a in pares if real(o) and real(a)]
    fila = {"base": ETIQUETAS[otro], "pares juzgados": len(pares)}
    for criterio, bueno in BUENO.items():
        if bueno:
            cumple = lambda v: bool(v.get(criterio))
        else:
            cumple = lambda v: v.get(criterio) is False
        solo_af = sum(cumple(a) and not cumple(o) for o, a in pares)
        solo_o = sum(cumple(o) and not cumple(a) for o, a in pares)
        p = binomtest(solo_af, solo_af + solo_o, 0.5).pvalue if solo_af + solo_o else 1.0
        fila[criterio] = f"{solo_af} contra {solo_o} (p={p:.2f})"
    filas_c.append(fila)

criterios = pd.DataFrame(filas_c).set_index("base")
print("cada celda: fragmentos donde solo cumple el afinado contra donde solo cumple el base (p)\n")
print(criterios.T.to_string())
''')

code(r'''
forma.to_csv(DATA / "comparacion_prompts_forma.csv")
calidad.to_csv(DATA / "comparacion_prompts_calidad.csv")
pareadas.to_csv(DATA / "comparacion_prompts_pareadas.csv")

detalle = []
for i, fila in casos.iterrows():
    reg = {"chunk_uid": fila["chunk_uid"], "tema": fila["question_focus"],
           "fragmento": fila["chunk_text"],
           "maestro": {"pregunta": fila["pregunta"], "correcta": fila["correcta"],
                       "veredicto": veredictos["maestro"].get(i)}}
    for clave in VARIANTES:
        reg[clave] = {"salida": dicts[clave][i] or salidas[clave][i],
                      "veredicto": veredictos[clave].get(i)}
    detalle.append(reg)
json.dump(detalle, open(DATA / "comparacion_prompts_generaciones.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("guardado: comparacion_prompts_{forma,calidad,pareadas}.csv y _generaciones.json")
''')

md(r"""
## B. ¿La robustez del formato es mérito del fine-tuning?

El notebook 04 mostró que el modelo afinado da 60 de 60 JSON válidos en todas las
temperaturas, incluida 1.0, y se atribuyó eso al fine-tuning. **Pero ese barrido
corrió solo con el afinado**: no hubo control con el base.

Aquí se hace el control, con el base y su few-shot. Los **mismos 20 fragmentos**
del notebook 04 (`random_state=42`), 3 repeticiones y las mismas temperaturas,
con semilla fija para que sea reproducible.
""")

code(r'''
casos_b = test_df.sample(20, random_state=42)       # los mismos del notebook 04
frag_b = casos_b["chunk_text"].tolist()
REPS = 3
construir_fs = VARIANTES["fewshot"][2]

filas_b = []
for temp in (0.7, 1.0):
    torch.manual_seed(SEMILLA)
    prompts = [construir_fs(f) for f in frag_b] * REPS   # rep 1, rep 2, rep 3
    textos, _, truncadas = generar(prompts, usar_base=True, do_sample=True,
                                   temperature=temp, top_p=0.9, top_k=50)
    ds = [parsear(t) for t in textos]
    variedad = []
    for j in range(len(frag_b)):
        preguntas = {sin_acentos(ds[j + k * len(frag_b)]["pregunta"]).strip()
                     for k in range(REPS) if estructura_ok(ds[j + k * len(frag_b)])}
        variedad.append(len(preguntas))
    total = len(textos)
    filas_b.append({"modelo": "base + few-shot", "configuracion": f"temp {temp}",
                    "JSON valido": f"{sum(d is not None for d in ds)}/{total}",
                    "estructura": f"{sum(estructura_ok(d) for d in ds)}/{total}",
                    "variedad": round(sum(variedad) / len(variedad), 2),
                    "truncadas": truncadas})

# Los resultados del afinado, tal como los midio el notebook 04.
nb04 = pd.read_csv(DATA / "barrido_generacion.csv")
for conf in ("temp 0.7", "temp 1.0"):
    f = nb04[nb04["configuracion"] == conf].iloc[0]
    filas_b.append({"modelo": "afinado (notebook 04)", "configuracion": conf,
                    "JSON valido": f["JSON valido"], "estructura": f["estructura"],
                    "variedad": f["variedad"], "truncadas": f["truncadas"]})

control = pd.DataFrame(filas_b).sort_values(["configuracion", "modelo"]).reset_index(drop=True)
print(control.to_string(index=False))
control.to_csv(DATA / "comparacion_prompts_control_formato.csv", index=False)
''')

nb = {"cells": [], "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}}}
for i, (tipo, texto) in enumerate(celdas):
    c = {"cell_type": tipo, "id": f"c{i:02d}", "metadata": {},
         "source": texto.splitlines(keepends=True)}
    if tipo == CODE:
        compile(texto, f"celda{i}", "exec")
        c["outputs"], c["execution_count"] = [], None
    nb["cells"].append(c)

SALIDA = "notebooks/05_afinado_contra_base.ipynb"
json.dump(nb, io.open(SALIDA, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{SALIDA}: {len(celdas)} celdas, todas las de codigo compilan")
