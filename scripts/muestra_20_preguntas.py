"""Genera 20 preguntas con el modelo afinado, para revisarlas a mano.

Los numeros agregados (92.7% sin defectos, 60/60 de estructura) no reemplazan
leer preguntas. Este script produce una muestra legible sobre fragmentos del
split de test, que el modelo no vio al entrenar.

Usa lotes de 8: medido, 8.5x mas rapido que generar de a una.
"""
import json
import re
import textwrap
import time
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELO = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTADOR = Path("adapter")
N = 20
LOTE = 8

INSTRUCCION = (
    "Eres un docente de medicina. A partir del FRAGMENTO escribe UNA pregunta "
    "de opcion multiple en espanol neutro. Responde SOLO con JSON."
)


def parsear(t):
    t = t.strip()
    if t.startswith("```"):
        t = t.strip("`").removeprefix("json").strip()
    try:
        return json.loads(t)
    except Exception:
        return None


test = pd.read_parquet("data/mcq_test.parquet")
# variedad de temas: un fragmento por tipo de pregunta hasta completar N
casos = (test.sample(frac=1, random_state=7)
             .drop_duplicates(subset="question_type")
             .head(N))
if len(casos) < N:
    resto = test.drop(index=casos.index).sample(N - len(casos), random_state=7)
    casos = pd.concat([casos, resto])

print(f"cargando modelo + adaptador...", flush=True)
tok = AutoTokenizer.from_pretrained(ADAPTADOR)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(MODELO, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(base, ADAPTADOR)
model.eval()

textos = []
for _, fila in casos.iterrows():
    msgs = [{"role": "system", "content": INSTRUCCION},
            {"role": "user", "content": "FRAGMENTO:\n" + fila["chunk_text"]}]
    textos.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

print(f"generando {len(textos)} preguntas en lotes de {LOTE}...", flush=True)
t0 = time.time()
salidas = []
for i in range(0, len(textos), LOTE):
    ids = tok(textos[i:i + LOTE], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=300, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    for f in out:
        salidas.append(tok.decode(f[ids.input_ids.shape[1]:], skip_special_tokens=True))
print(f"generadas en {time.time()-t0:.0f}s ({(time.time()-t0)/len(textos):.1f}s por pregunta)\n")

registros = []
for (_, fila), salida in zip(casos.iterrows(), salidas):
    d = parsear(salida)
    print("=" * 100)
    print(f"[{fila['question_type']}] {fila['question_focus']}")
    print(textwrap.fill(" ".join(fila["chunk_text"].split()), 96,
                        initial_indent="  fuente: ", subsequent_indent="          ")[:400])
    print()
    if not d or not d.get("pregunta"):
        print("  NO PARSEABLE:", salida[:200])
        continue
    print("  " + d["pregunta"] + f"   [{d.get('dificultad','?')}]")
    print("     (correcta)  " + str(d["correcta"]))
    for x in d.get("incorrectas", []):
        print("                 " + str(x))
    print()
    registros.append({"tipo": fila["question_type"], "tema": fila["question_focus"],
                      "url": fila["document_url"], **d})

json.dump(registros, open("data/muestra_20_preguntas.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("=" * 100)
print(f"{len(registros)}/{N} parseadas. Guardadas en data/muestra_20_preguntas.json")
