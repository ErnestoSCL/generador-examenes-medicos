"""Arma el notebook 05: base con few-shot contra afinado, en CALIDAD.

Cierra el hueco central del proyecto. El notebook 03 comparo base y afinado
solo en forma; el juez comparo al afinado contra el maestro, nunca contra el
base. Este notebook responde "sirvio entrenar?" en contenido.
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
# 05 - Base con few-shot contra afinado: ¿las preguntas son mejores?

El notebook 03 comparó el modelo base con el afinado **solo en forma**: JSON
válido, estructura, tokens de prompt y velocidad. La calidad del contenido la
midió un juez, pero contra el **maestro** (`gpt-4o-mini`), nunca contra el base.

Eso dejaba sin responder, en contenido, la pregunta central del proyecto:
**¿sirvió entrenar?**

La distinción importa. «Corre en local, sin API y sin costo» no justifica el
fine-tuning: justifica usar un modelo local, y el Qwen base también corre en
local. Lo que justifica el fine-tuning es superar al **base con few-shot**, en
las mismas condiciones.

## Diseño

| Decisión | Por qué |
|---|---|
| Los **mismos 150 fragmentos** que juzgó el notebook 03 (`random_state=13` sobre el test) | comparable con el 78.4%, y nunca vistos en el entrenamiento |
| La **misma rúbrica** y el mismo juez (`gpt-4o`, temperatura 0) | que la diferencia sea del modelo, no de la vara |
| Decodificación **greedy**, como en el notebook 03 | que la diferencia no sea azar de muestreo |
| El base recibe los **mismos 3 ejemplos** de few-shot que en el notebook 03 | sin ellos no sabría qué formato devolver y la comparación sería trivial |
| El maestro se vuelve a juzgar **en la misma corrida** | tres modelos, un juez, los mismos fragmentos |
| **Prueba pareada** (McNemar) | los dos modelos responden sobre los mismos fragmentos: se compara fragmento a fragmento, que es más potente que comparar dos porcentajes sueltos |

La sección B verifica además una afirmación del notebook 04 que quedó sin
control: que la robustez del formato bajo muestreo es mérito del fine-tuning.
""")

code(r'''
import json, re, time, unicodedata
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

# Identica a la del entrenamiento (notebook 03) y a la de la aplicacion.
INSTRUCCION = (
    "Eres un docente de medicina. A partir del FRAGMENTO escribe UNA pregunta "
    "de opcion multiple en espanol neutro. Responde SOLO con JSON."
)

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


# Los mismos 3 ejemplos que recibio el base en el notebook 03.
ejemplos = train_df.sample(3, random_state=SEMILLA)


def prompt_afinado(fragmento):
    return tok.apply_chat_template(
        [{"role": "system", "content": INSTRUCCION},
         {"role": "user", "content": "FRAGMENTO:\n" + fragmento}],
        tokenize=False, add_generation_prompt=True)


def prompt_base(fragmento):
    msgs = [{"role": "system", "content": INSTRUCCION}]
    for _, e in ejemplos.iterrows():
        msgs.append({"role": "user", "content": "FRAGMENTO:\n" + e["chunk_text"]})
        msgs.append({"role": "assistant", "content": json.dumps({
            "apto": True, "pregunta": e["pregunta"], "correcta": e["correcta"],
            "incorrectas": list(e["incorrectas"]), "dificultad": e["dificultad"]},
            ensure_ascii=False)})
    msgs.append({"role": "user", "content": "FRAGMENTO:\n" + fragmento})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


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
''')

md(r"""
## A. Calidad del contenido

Se generan las 150 preguntas con cada modelo sobre los mismos fragmentos.
""")

code(r'''
casos = test_df.sample(150, random_state=13).reset_index(drop=True)
fragmentos = casos["chunk_text"].tolist()

t0 = time.time()
sal_base, tok_base, _ = generar([prompt_base(f) for f in fragmentos], usar_base=True)
seg_base = (time.time() - t0) / len(fragmentos)

t0 = time.time()
sal_af, tok_af, _ = generar([prompt_afinado(f) for f in fragmentos], usar_base=False)
seg_af = (time.time() - t0) / len(fragmentos)

d_base = [parsear(s) for s in sal_base]
d_af = [parsear(s) for s in sal_af]

forma = pd.DataFrame({
    "base + few-shot": {
        "JSON valido": f"{sum(d is not None for d in d_base)}/150",
        "estructura completa": f"{sum(estructura_ok(d) for d in d_base)}/150",
        "tokens de prompt": round(sum(tok_base) / len(tok_base)),
        "segundos por pregunta (lotes de 8)": round(seg_base, 2)},
    "afinado": {
        "JSON valido": f"{sum(d is not None for d in d_af)}/150",
        "estructura completa": f"{sum(estructura_ok(d) for d in d_af)}/150",
        "tokens de prompt": round(sum(tok_af) / len(tok_af)),
        "segundos por pregunta (lotes de 8)": round(seg_af, 2)},
})
print(forma.to_string())
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
        return idx, quien, ESTRUCTURA_ROTA
    texto = (f"TEMA: {fila['question_focus']}\n\nFRAGMENTO:\n{fila['chunk_text']}\n\n"
             f"PREGUNTA: {d['pregunta']}\nCORRECTA: {d['correcta']}\n"
             + "\n".join(f"INCORRECTA: {x}" for x in d["incorrectas"]))
    for intento in range(3):
        try:
            r = client.chat.completions.create(
                model=MODELO_JUEZ, temperature=0,
                messages=[{"role": "system", "content": JUEZ},
                          {"role": "user", "content": texto}])
            v = parsear(r.choices[0].message.content)
            if v is not None:
                return idx, quien, v
        except Exception:
            time.sleep(2 * (intento + 1))
    return idx, quien, None                        # fallo de la API: se excluye


tareas = []
for i, fila in casos.iterrows():
    tareas.append((i, "maestro", fila, {"pregunta": fila["pregunta"], "correcta": fila["correcta"],
                                        "incorrectas": list(fila["incorrectas"])}))
    tareas.append((i, "base", fila, d_base[i]))
    tareas.append((i, "afinado", fila, d_af[i]))

print(f"juzgando {len(tareas)} preguntas con {MODELO_JUEZ}...", flush=True)
veredictos = {"maestro": {}, "base": {}, "afinado": {}}
with ThreadPoolExecutor(8) as pool:
    for fut in as_completed([pool.submit(juzgar, t) for t in tareas]):
        idx, quien, v = fut.result()
        veredictos[quien][idx] = v

fallos_api = {q: sum(v is None for v in vs.values()) for q, vs in veredictos.items()}
print("fallos de la API (se excluyen):", fallos_api)
''')

code(r'''
def limpia(v):
    # La misma definicion de "sin ningun defecto" del notebook 03.
    return bool(v and not v.get("_estructura_rota") and v.get("respaldo")
                and not v.get("distractor_verdadero") and v.get("unica_respuesta")
                and v.get("tema_correcto"))


ETIQUETAS = {"maestro": "maestro (gpt-4o-mini)", "base": "base + few-shot",
             "afinado": "afinado (LoRA)"}
filas = []
for quien, etiqueta in ETIQUETAS.items():
    todas = [v for v in veredictos[quien].values() if v is not None]
    juzgadas = [v for v in todas if not v.get("_estructura_rota")]
    n = max(len(juzgadas), 1)
    filas.append({
        "modelo": etiqueta,
        "estructura valida": f"{len(juzgadas)}/{len(todas)}",
        "correcta respaldada": f"{sum(bool(v.get('respaldo')) for v in juzgadas)/n*100:.1f}%",
        "sin distractor cierto": f"{sum(not v.get('distractor_verdadero', True) for v in juzgadas)/n*100:.1f}%",
        "una sola respuesta": f"{sum(bool(v.get('unica_respuesta')) for v in juzgadas)/n*100:.1f}%",
        "tema correcto": f"{sum(bool(v.get('tema_correcto')) for v in juzgadas)/n*100:.1f}%",
        "SIN DEFECTO (de las juzgadas)": f"{sum(limpia(v) for v in juzgadas)/n*100:.1f}%",
        "SIN DEFECTO (de los 150 fragmentos)": f"{sum(limpia(v) for v in todas)/max(len(todas),1)*100:.1f}%",
    })

calidad = pd.DataFrame(filas).set_index("modelo")
print(calidad.T.to_string())
''')

md(r"""
### Cómo leer la tabla

Hay dos filas de «sin defecto» y la diferencia importa:

- **de las juzgadas** es la cifra comparable con el 78.4% del notebook 03, que
  solo juzgaba preguntas con estructura válida;
- **de los 150 fragmentos** cuenta como fallo una pregunta con la estructura
  rota. Es la tasa que vive el usuario: de cada fragmento, ¿sale una pregunta
  aprovechable?

### La prueba pareada

Como base y afinado respondieron sobre **los mismos** fragmentos, lo que decide
es en cuántos fragmentos acierta uno y falla el otro (los pares discordantes).
Los fragmentos donde los dos aciertan, o los dos fallan, no informan sobre la
diferencia. La prueba de McNemar pregunta si esos discordantes se reparten de
forma tan desigual que no puede ser azar.
""")

code(r'''
from scipy.stats import binomtest

validos = [i for i in range(len(casos))
           if all(veredictos[q].get(i) is not None for q in ETIQUETAS)]
ok = {q: {i: limpia(veredictos[q][i]) for i in validos} for q in ETIQUETAS}


def pareada(a, b):
    solo_a = sum(ok[a][i] and not ok[b][i] for i in validos)
    solo_b = sum(ok[b][i] and not ok[a][i] for i in validos)
    p = binomtest(solo_a, solo_a + solo_b, 0.5).pvalue if solo_a + solo_b else 1.0
    return {"comparacion": f"{a} contra {b}",
            f"solo {a} acierta": solo_a, f"solo {b} acierta": solo_b, "p (McNemar exacto)": round(p, 4)}


print(f"fragmentos con los tres veredictos: {len(validos)}\n")
for a, b in (("afinado", "base"), ("afinado", "maestro"), ("base", "maestro")):
    r = pareada(a, b)
    vals = list(r.values())
    print(f"{vals[0]:<22}  solo el primero: {vals[1]:>3}   solo el segundo: {vals[2]:>3}   p = {vals[3]}")
print("\np < 0.05: la diferencia difícilmente es azar. p alto: no se distingue del ruido.")
''')

code(r'''
calidad.to_csv(DATA / "base_vs_afinado_calidad.csv")

detalle = []
for i, fila in casos.iterrows():
    detalle.append({
        "chunk_uid": fila["chunk_uid"], "tema": fila["question_focus"],
        "fragmento": fila["chunk_text"],
        "maestro": {"pregunta": fila["pregunta"], "correcta": fila["correcta"],
                    "veredicto": veredictos["maestro"].get(i)},
        "base": {"salida": d_base[i] or sal_base[i], "veredicto": veredictos["base"].get(i)},
        "afinado": {"salida": d_af[i] or sal_af[i], "veredicto": veredictos["afinado"].get(i)},
    })
json.dump(detalle, open(DATA / "base_vs_afinado_generaciones.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("guardado: base_vs_afinado_calidad.csv y base_vs_afinado_generaciones.json (para revisar a mano)")
''')

md(r"""
## B. ¿La robustez del formato es mérito del fine-tuning?

El notebook 04 mostró que el modelo afinado da 60 de 60 JSON válidos en todas las
temperaturas, incluida 1.0, y se atribuyó eso al fine-tuning. **Pero ese barrido
corrió solo con el afinado**: no hubo control con el base. La atribución no
estaba demostrada.

Aquí se hace el control. Los **mismos 20 fragmentos** del notebook 04
(`random_state=42`), 3 repeticiones, las mismas temperaturas, ahora con el base
y su few-shot.
""")

code(r'''
casos_b = test_df.sample(20, random_state=42)       # los mismos del notebook 04
frag_b = casos_b["chunk_text"].tolist()
REPS = 3

filas_b = []
for temp in (0.7, 1.0):
    prompts = [prompt_base(f) for f in frag_b] * REPS   # rep 1, rep 2, rep 3
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
control.to_csv(DATA / "control_formato_base.csv", index=False)
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

json.dump(nb, io.open("notebooks/05_base_vs_afinado.ipynb", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"notebook 05 armado: {len(celdas)} celdas, todas las de codigo compilan")
