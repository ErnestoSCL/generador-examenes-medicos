"""Barrido de parametros de generacion sobre el modelo ya afinado.

Estos parametros no cambian el modelo, solo como se muestrea de el. Se eligen
DESPUES de fijar la configuracion de entrenamiento.

La tension que hay que resolver:

  do_sample=False (greedy)  formato perfecto, CERO variedad: el mismo fragmento
                            da siempre la misma pregunta
  do_sample=True            hace falta para el boton "otro examen del mismo
                            tema", pero puede romper el JSON

Por eso se miden dos cosas enfrentadas:
  - JSON valido y estructura completa  (formato)
  - preguntas distintas al regenerar 3 veces el mismo fragmento  (variedad)

La mejor configuracion es la que maximiza variedad SIN perder formato.
"""
import json
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELO = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTADOR = Path("adapter")
DATA = Path("data")
N_CASOS = 20
REPETICIONES = 3        # para medir variedad hay que regenerar el mismo fragmento

INSTRUCCION = (
    "Eres un docente de medicina. A partir del FRAGMENTO escribe UNA pregunta "
    "de opcion multiple en espanol neutro. Responde SOLO con JSON."
)

# top_p=0.9 y top_k=50 son los valores estandar y quedan fijos como referencia.
# Se varia lo que de verdad mueve la aguja: la temperatura y la penalizacion.
CONFIGS = [
    dict(nombre="greedy (referencia)", do_sample=False),
    dict(nombre="temp 0.3", do_sample=True, temperature=0.3, top_p=0.9, top_k=50),
    dict(nombre="temp 0.7", do_sample=True, temperature=0.7, top_p=0.9, top_k=50),
    dict(nombre="temp 1.0", do_sample=True, temperature=1.0, top_p=0.9, top_k=50),
    dict(nombre="temp 0.7 + rep 1.1", do_sample=True, temperature=0.7, top_p=0.9,
         top_k=50, repetition_penalty=1.1),
    dict(nombre="temp 0.7 + rep 1.2", do_sample=True, temperature=0.7, top_p=0.9,
         top_k=50, repetition_penalty=1.2),
    dict(nombre="temp 0.7 sin top_k", do_sample=True, temperature=0.7, top_p=0.9),
]

MAX_NEW = 300


def sin_acentos(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s).lower())
                   if unicodedata.category(c) != "Mn")


def parsear(t):
    t = t.strip()
    if t.startswith("```"):
        t = t.strip("`").removeprefix("json").strip()
    try:
        return json.loads(t)
    except Exception:
        return None


tok = AutoTokenizer.from_pretrained(ADAPTADOR)
base = AutoModelForCausalLM.from_pretrained(MODELO, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(base, ADAPTADOR)
model.eval()

test = pd.read_parquet(DATA / "mcq_test.parquet").sample(N_CASOS, random_state=42)
print(f"{N_CASOS} fragmentos x {REPETICIONES} repeticiones x {len(CONFIGS)} configuraciones")
print(f"= {N_CASOS * REPETICIONES * len(CONFIGS)} generaciones\n")

filas = []
for cfg in CONFIGS:
    nombre = cfg.pop("nombre")
    t0 = time.time()
    validos = completas = truncadas = 0
    variedad = []

    for _, fila in test.iterrows():
        msgs = [{"role": "system", "content": INSTRUCCION},
                {"role": "user", "content": "FRAGMENTO:\n" + fila["chunk_text"]}]
        texto = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok([texto], return_tensors="pt").to(model.device)

        preguntas = set()
        for _ in range(REPETICIONES):
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=MAX_NEW,
                                     pad_token_id=tok.eos_token_id, **cfg)
            nuevos = out.shape[1] - ids.input_ids.shape[1]
            if nuevos >= MAX_NEW:
                truncadas += 1
            d = parsear(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))
            if d is None:
                continue
            validos += 1
            if (d.get("pregunta") and d.get("correcta")
                    and isinstance(d.get("incorrectas"), list) and len(d["incorrectas"]) == 3):
                completas += 1
                preguntas.add(sin_acentos(d["pregunta"]).strip())

        # cuantas preguntas DISTINTAS salieron de este mismo fragmento
        variedad.append(len(preguntas))

    total = N_CASOS * REPETICIONES
    filas.append({
        "configuracion": nombre,
        "JSON valido": f"{validos}/{total}",
        "estructura": f"{completas}/{total}",
        "% formato ok": round(completas / total * 100, 1),
        "variedad": round(sum(variedad) / len(variedad), 2),   # de 3 posibles
        "truncadas": truncadas,
        "segundos": round((time.time() - t0) / total, 2),
    })
    print(f"  {nombre:<22} formato {completas}/{total}  "
          f"variedad {sum(variedad)/len(variedad):.2f}/3  "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)

tabla = pd.DataFrame(filas)
print("\n" + "=" * 88)
print(tabla.to_string(index=False))
tabla.to_csv(DATA / "barrido_generacion.csv", index=False)

print("\nvariedad = preguntas distintas de 3 generaciones del MISMO fragmento")
print("  1.00 = siempre la misma (sin variedad)")
print("  3.00 = las tres distintas")
print("\nLa mejor configuracion es la de mayor variedad SIN perder formato.")
