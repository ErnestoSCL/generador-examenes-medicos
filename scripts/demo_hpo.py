"""Demuestra por que una tabla del HPO no sirve para generar una pregunta.

Genera una pregunta desde un fragmento HPO y otra desde un fragmento normal,
con el mismo modelo y el mismo prompt, y las pone lado a lado.
"""
import json
import textwrap

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELO = "Qwen/Qwen3-4B-Instruct-2507"

INSTRUCCION = (
    "Eres un docente de medicina que redacta preguntas de examen.\n"
    "A partir del FRAGMENTO, escribe UNA pregunta de opcion multiple en espanol.\n"
    "- La respuesta correcta debe estar respaldada por el fragmento.\n"
    "- Las tres incorrectas deben ser plausibles pero falsas segun el fragmento.\n"
    "- Las cuatro opciones, maximo 15 palabras y de longitud similar.\n"
    "- La pregunta debe entenderse sola, sin referirse al texto.\n"
    'Responde SOLO con JSON: {"pregunta": "...", "correcta": "...", '
    '"incorrectas": ["...","...","..."], "dificultad": "facil|media|dificil"}'
)

# --- los dos fragmentos ----------------------------------------------------
clean = pd.read_parquet("../data/medqa_clean.parquet")
# Tiene que EMPEZAR con el boilerplate: hay respuestas que traen prosa antes de
# la tabla, y esas si sirven. El fragmento problematico es el que es solo tabla.
solo_tabla = clean[clean.answer.str.startswith("The Human Phenotype Ontology provides")]
fila_hpo = solo_tabla.iloc[3]
FRAG_HPO = fila_hpo.answer[:512]

ch = pd.read_parquet("../data/chunks.parquet")
fila_ok = ch[ch.question_focus.str.contains("Huntington", case=False, na=False)].iloc[0]
FRAG_OK = fila_ok.chunk_text

print("cargando el modelo...", flush=True)
tok = AutoTokenizer.from_pretrained(MODELO)
model = AutoModelForCausalLM.from_pretrained(MODELO, dtype=torch.bfloat16, device_map="cuda")
model.eval()


def generar(fragmento):
    msgs = [{"role": "system", "content": INSTRUCCION},
            {"role": "user", "content": "FRAGMENTO:\n" + fragmento}]
    texto = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok([texto], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=260, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


def mostrar(titulo, fragmento, salida):
    print("\n" + "=" * 100)
    print(titulo)
    print("=" * 100)
    print("FRAGMENTO QUE RECIBE EL MODELO:")
    print(textwrap.fill(" ".join(fragmento.split()), 96, initial_indent="  ", subsequent_indent="  "))
    print("\nPREGUNTA QUE GENERA:")
    crudo = salida.strip()
    if crudo.startswith("```"):
        crudo = crudo.strip("`").removeprefix("json").strip()
    try:
        d = json.loads(crudo)
        print("  P: " + d["pregunta"])
        print("  correcta   -> " + str(d["correcta"]))
        for inc in d["incorrectas"]:
            print("  incorrecta -> " + str(inc))
    except Exception:
        print(textwrap.fill(salida, 96, initial_indent="  ", subsequent_indent="  "))


mostrar("CASO A - fragmento con tabla del Human Phenotype Ontology", FRAG_HPO, generar(FRAG_HPO))
mostrar("CASO B - fragmento normal (Huntington)", FRAG_OK, generar(FRAG_OK))
