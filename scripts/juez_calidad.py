"""Juez de calidad medica sobre una muestra de las preguntas generadas.

Los filtros automaticos verifican forma: longitud, formato, duplicados. Ninguno
verifica lo que importa: que la respuesta correcta este respaldada por el
fragmento y que los tres distractores sean falsos. Eso hay que leerlo, y a esta
escala hay que leerlo con un modelo.

Se usa gpt-4o (no mini) como juez: evaluar es mas dificil que generar, y usar el
mismo modelo que genero se autoevaluaria con indulgencia.
"""
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".env")
client = OpenAI()
N = 150
MODELO_JUEZ = "gpt-4o"

JUEZ = """Eres un revisor de examenes de medicina. Recibes un FRAGMENTO y una
PREGUNTA de opcion multiple construida a partir de el.

Evalua tres cosas, con severidad:

1. respaldo: la opcion marcada como correcta, esta afirmada explicitamente en el
   fragmento? (true/false)
2. distractor_verdadero: alguna de las tres opciones incorrectas es TAMBIEN
   cierta segun el fragmento, o defendible como respuesta? (true/false)
3. unica_respuesta: hay exactamente una respuesta defendible? (true/false)

Responde SOLO con JSON:
{"respaldo": true/false, "distractor_verdadero": true/false,
 "unica_respuesta": true/false, "comentario": "una frase solo si algo falla"}"""


def par(t):
    if not t:
        return None
    t = t.strip()
    if t.startswith("```"):
        t = t.strip("`").removeprefix("json").strip()
    try:
        return json.loads(t)
    except Exception:
        return None


regs = [json.loads(l) for l in open("data/mcq_crudo.jsonl", encoding="utf-8")]
frag = pd.read_parquet("data/muestra_api.parquet").set_index("chunk_uid")["chunk_text"]

items = [(r["chunk_uid"], d) for r, d in ((r, par(r["crudo"])) for r in regs)
         if d and d.get("pregunta")]
random.seed(13)
muestra = random.sample(items, N)
print(f"evaluando {N} preguntas de {len(items):,} con {MODELO_JUEZ}")


def evaluar(par_):
    uid, d = par_
    texto = (f"FRAGMENTO:\n{frag[uid]}\n\nPREGUNTA: {d['pregunta']}\n"
             f"CORRECTA: {d['correcta']}\n"
             + "\n".join(f"INCORRECTA: {x}" for x in d["incorrectas"]))
    try:
        r = client.chat.completions.create(
            model=MODELO_JUEZ, temperature=0,
            messages=[{"role": "system", "content": JUEZ},
                      {"role": "user", "content": texto}])
        return uid, d, par(r.choices[0].message.content)
    except Exception as exc:
        return uid, d, {"error": str(exc)[:80]}


res = []
with ThreadPoolExecutor(6) as pool:
    for fut in as_completed([pool.submit(evaluar, m) for m in muestra]):
        res.append(fut.result())

validos = [(u, d, v) for u, d, v in res if v and "error" not in v]
n = len(validos)
print(f"evaluadas: {n}\n")

resp = sum(v["respaldo"] for _, _, v in validos)
dist = sum(v["distractor_verdadero"] for _, _, v in validos)
unic = sum(v["unica_respuesta"] for _, _, v in validos)

print(f"  correcta respaldada por el fragmento : {resp}/{n}  ({resp/n*100:.1f}%)")
print(f"  ALGUN distractor tambien verdadero   : {dist}/{n}  ({dist/n*100:.1f}%)")
print(f"  una sola respuesta defendible        : {unic}/{n}  ({unic/n*100:.1f}%)")

limpias = sum(1 for _, _, v in validos
              if v["respaldo"] and not v["distractor_verdadero"] and v["unica_respuesta"])
print(f"\n  SIN NINGUN DEFECTO                   : {limpias}/{n}  ({limpias/n*100:.1f}%)")

print("\n--- ejemplos marcados como defectuosos ---")
malas = [(u, d, v) for u, d, v in validos
         if not (v["respaldo"] and not v["distractor_verdadero"] and v["unica_respuesta"])]
for u, d, v in malas[:6]:
    print(f"\n  P: {d['pregunta']}")
    print(f"     correcta: {d['correcta']}")
    for x in d["incorrectas"]:
        print(f"     x {x}")
    print(f"     juez: respaldo={v['respaldo']} distractor_cierto={v['distractor_verdadero']} "
          f"unica={v['unica_respuesta']}")
    print(f"     -> {v.get('comentario', '')[:150]}")

json.dump([{"uid": u, "pregunta": d["pregunta"], "veredicto": v} for u, d, v in validos],
          open("data/juez_calidad.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
