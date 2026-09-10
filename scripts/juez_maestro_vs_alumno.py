"""Compara la calidad del maestro (gpt-4o-mini) contra el alumno (Qwen afinado).

Es la pregunta central de una destilacion: cuanto se acerca el alumno al maestro.

El diseno permite una comparacion limpia: **los mismos fragmentos**. El split de
test ya trae la pregunta que genero el maestro para cada fragmento; aqui se
genera la del alumno sobre esos mismos fragmentos, y un tercer modelo (gpt-4o,
mas capaz que ambos) juzga las dos sin saber cual es cual.

Los tres primeros criterios son identicos a los del juez que se corrio sobre el
dataset de entrenamiento, para que las cifras sean comparables. El cuarto es
nuevo: salio de leer 20 preguntas a mano y encontrar dos que perdian el tema del
fragmento (una pregunta sobre equinococosis que hablaba de cisticercosis).
"""
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch
from dotenv import load_dotenv
from openai import OpenAI
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv(".env")
client = OpenAI()

MODELO = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTADOR = Path("adapter")
MODELO_JUEZ = "gpt-4o"
N = 150
LOTE = 8

INSTRUCCION = (
    "Eres un docente de medicina. A partir del FRAGMENTO escribe UNA pregunta "
    "de opcion multiple en espanol neutro. Responde SOLO con JSON."
)

JUEZ = """Eres un revisor de examenes de medicina. Recibes un FRAGMENTO, el TEMA
al que pertenece, y una PREGUNTA de opcion multiple construida a partir de el.

Evalua con severidad:

1. respaldo: la opcion marcada como correcta, esta afirmada explicitamente en el
   fragmento? (true/false)
2. distractor_verdadero: alguna de las tres opciones incorrectas es TAMBIEN
   cierta segun el fragmento, o defendible como respuesta? (true/false)
3. unica_respuesta: hay exactamente una respuesta defendible? (true/false)
4. tema_correcto: la pregunta se refiere al TEMA indicado, y no a otra
   enfermedad o asunto distinto? (true/false)

Responde SOLO con JSON:
{"respaldo": true/false, "distractor_verdadero": true/false,
 "unica_respuesta": true/false, "tema_correcto": true/false,
 "comentario": "una frase solo si algo falla"}"""


def parsear(t):
    if not t:
        return None
    t = t.strip()
    if t.startswith("```"):
        t = t.strip("`").removeprefix("json").strip()
    try:
        return json.loads(t)
    except Exception:
        return None


# ---------------------------------------------------- el alumno genera
test = pd.read_parquet("data/mcq_test.parquet")
casos = test.sample(min(N, len(test)), random_state=13).reset_index(drop=True)
print(f"comparando sobre {len(casos)} fragmentos del split de test\n")

print("cargando el alumno...", flush=True)
tok = AutoTokenizer.from_pretrained(ADAPTADOR)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(MODELO, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(base, ADAPTADOR)
model.eval()

textos = [tok.apply_chat_template(
    [{"role": "system", "content": INSTRUCCION},
     {"role": "user", "content": "FRAGMENTO:\n" + f["chunk_text"]}],
    tokenize=False, add_generation_prompt=True) for _, f in casos.iterrows()]

print(f"generando {len(textos)} preguntas...", flush=True)
t0 = time.time()
salidas = []
for i in range(0, len(textos), LOTE):
    ids = tok(textos[i:i + LOTE], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=300, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    for f in out:
        salidas.append(tok.decode(f[ids.input_ids.shape[1]:], skip_special_tokens=True))
print(f"generadas en {time.time()-t0:.0f}s\n")

del model, base
gc = __import__("gc")
gc.collect()
torch.cuda.empty_cache()

# ---------------------------------------------------- el juez evalua las dos
def evaluar(args):
    idx, quien, fila, d = args
    if not d or not d.get("pregunta"):
        return idx, quien, {"no_parseable": True}
    texto = (f"TEMA: {fila['question_focus']}\n\nFRAGMENTO:\n{fila['chunk_text']}\n\n"
             f"PREGUNTA: {d['pregunta']}\nCORRECTA: {d['correcta']}\n"
             + "\n".join(f"INCORRECTA: {x}" for x in d["incorrectas"]))
    for intento in range(3):
        try:
            r = client.chat.completions.create(
                model=MODELO_JUEZ, temperature=0,
                messages=[{"role": "system", "content": JUEZ},
                          {"role": "user", "content": texto}])
            return idx, quien, parsear(r.choices[0].message.content)
        except Exception:
            time.sleep(2 * (intento + 1))
    return idx, quien, None


tareas = []
for i, fila in casos.iterrows():
    maestro = {"pregunta": fila["pregunta"], "correcta": fila["correcta"],
               "incorrectas": list(fila["incorrectas"])}
    tareas.append((i, "maestro", fila, maestro))
    tareas.append((i, "alumno", fila, parsear(salidas[i])))

print(f"juzgando {len(tareas)} preguntas con {MODELO_JUEZ}...", flush=True)
veredictos = {"maestro": {}, "alumno": {}}
with ThreadPoolExecutor(8) as pool:
    for fut in as_completed([pool.submit(evaluar, t) for t in tareas]):
        idx, quien, v = fut.result()
        veredictos[quien][idx] = v

# ---------------------------------------------------- resultados
filas = []
for quien in ("maestro", "alumno"):
    vs = [v for v in veredictos[quien].values() if v and "no_parseable" not in v]
    n = len(vs)
    limpias = sum(1 for v in vs if v.get("respaldo") and not v.get("distractor_verdadero")
                  and v.get("unica_respuesta") and v.get("tema_correcto"))
    filas.append({
        "modelo": "maestro (gpt-4o-mini)" if quien == "maestro" else "alumno (Qwen afinado)",
        "evaluadas": n,
        "correcta respaldada": f"{sum(v.get('respaldo', False) for v in vs)/n*100:.1f}%",
        "sin distractor cierto": f"{sum(not v.get('distractor_verdadero', True) for v in vs)/n*100:.1f}%",
        "una sola respuesta": f"{sum(v.get('unica_respuesta', False) for v in vs)/n*100:.1f}%",
        "tema correcto": f"{sum(v.get('tema_correcto', False) for v in vs)/n*100:.1f}%",
        "SIN NINGUN DEFECTO": f"{limpias/n*100:.1f}%",
    })

tabla = pd.DataFrame(filas).set_index("modelo")
print("\n" + "=" * 90)
print(tabla.T.to_string())
tabla.to_csv("data/maestro_vs_alumno.csv")

# ejemplos donde el alumno falla y el maestro no
print("\n--- fragmentos donde el maestro acierta y el alumno falla ---")
def limpio(v):
    return bool(v) and "no_parseable" not in v and v.get("respaldo") and \
           not v.get("distractor_verdadero") and v.get("unica_respuesta") and v.get("tema_correcto")

mostrados = 0
for i in range(len(casos)):
    vm, va = veredictos["maestro"].get(i), veredictos["alumno"].get(i)
    if limpio(vm) and not limpio(va) and mostrados < 4:
        d = parsear(salidas[i])
        print(f"\n  TEMA: {casos.iloc[i]['question_focus']}")
        print(f"  maestro: {casos.iloc[i]['pregunta']}")
        if d and d.get("pregunta"):
            print(f"  alumno : {d['pregunta']}")
            print(f"           -> {d['correcta']}")
        print(f"  juez sobre el alumno: {va}")
        mostrados += 1

json.dump({"veredictos": {k: {str(i): v for i, v in d.items()} for k, d in veredictos.items()},
           "alumno_crudo": salidas},
          open("data/maestro_vs_alumno.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\nguardado en data/maestro_vs_alumno.json")
