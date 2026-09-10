"""Segunda mitad de la prueba de humo: recargar el adaptador en un proceso nuevo.

Es la parte que valida el traspaso: si el adaptador guardado se aplica bien
sobre el modelo base descargado aparte, el ciclo entrenar-guardar-servir
funciona, venga el adaptador de esta maquina o de Colab.

Genera con el adaptador puesto y con el adaptador apagado, para comprobar de
paso que `disable_adapter()` funciona: es el mecanismo del panel comparativo de
la aplicacion, que necesita los dos modelos sin cargar dos veces 8 GB.
"""
import json
import time

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELO = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTADOR = "adapter_humo"

INSTRUCCION = (
    "Eres un docente de medicina. A partir del FRAGMENTO escribe UNA pregunta "
    "de opcion multiple en espanol. Responde SOLO con JSON."
)

test = pd.read_parquet("data/mcq_test.parquet")
fragmento = test.iloc[0]["chunk_text"]

print("cargando modelo base...", flush=True)
tok = AutoTokenizer.from_pretrained(ADAPTADOR)
base = AutoModelForCausalLM.from_pretrained(MODELO, dtype=torch.bfloat16, device_map="cuda")
print(f"  memoria: {torch.cuda.memory_allocated()/1e9:.2f} GB")

print("aplicando el adaptador...", flush=True)
model = PeftModel.from_pretrained(base, ADAPTADOR)
model.eval()
print(f"  memoria con adaptador: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"  adaptadores activos: {model.active_adapters}")


def generar(etiqueta):
    msgs = [{"role": "system", "content": INSTRUCCION},
            {"role": "user", "content": "FRAGMENTO:\n" + fragmento}]
    texto = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok([texto], return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=260, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    salida = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(f"\n--- {etiqueta} ({time.time()-t0:.1f}s) ---")
    crudo = salida
    if crudo.startswith("```"):
        crudo = crudo.strip("`").removeprefix("json").strip()
    try:
        d = json.loads(crudo)
        print("  JSON valido: SI")
        print("  P:", d.get("pregunta"))
        print("  V", d.get("correcta"))
        for x in d.get("incorrectas", []):
            print("  x", x)
    except Exception:
        print("  JSON valido: NO")
        print(" ", salida[:320])


generar("CON el adaptador (afinado)")

with model.disable_adapter():
    generar("SIN el adaptador (modelo base)")

print(f"\nmemoria pico total: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print("un solo modelo en memoria para los dos: el panel comparativo es viable")
