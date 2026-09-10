"""Prueba de humo: cargar Qwen3-4B en local, generar, y medir tokens/s.

Define dos cosas que el resto del proyecto necesita saber:
  - si el modelo entra en la GPU y con que dtype
  - cuantos tokens por segundo genera, que decide si hace falta batching
"""
import json, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELO = "Qwen/Qwen3-4B-Instruct-2507"
FRAGMENTO = (
    "Tension headache is the most common type of headache. It is caused by "
    "tight muscles in your shoulders, neck, scalp and jaw. Tension headaches "
    "are often related to stress, depression or anxiety. You are more likely "
    "to get one if you work too much, do not get enough sleep, miss meals or "
    "use alcohol."
)
INSTRUCCION = (
    "Eres un docente de medicina. A partir del FRAGMENTO escribe UNA pregunta "
    "de opcion multiple en espanol neutro. Las cuatro opciones deben ser cortas "
    "(maximo 15 palabras) y de longitud similar. Responde SOLO con JSON: "
    '{"pregunta": "...", "correcta": "...", "incorrectas": ["...","...","..."], '
    '"dificultad": "facil|media|dificil"}'
)

print("cargando", MODELO, flush=True)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODELO)
model = AutoModelForCausalLM.from_pretrained(
    MODELO, dtype=torch.bfloat16, device_map="cuda",
)
model.eval()
print("cargado en %.1f s" % (time.time() - t0), flush=True)
print("memoria GPU ocupada: %.2f GB" % (torch.cuda.memory_allocated() / 1e9), flush=True)

mensajes = [
    {"role": "system", "content": INSTRUCCION},
    {"role": "user", "content": "FRAGMENTO:\n" + FRAGMENTO},
]
entrada = tok.apply_chat_template(mensajes, tokenize=False, add_generation_prompt=True)
ids = tok([entrada], return_tensors="pt").to(model.device)
print("tokens de prompt:", ids.input_ids.shape[1], flush=True)

# generacion unica, para medir velocidad secuencial
t0 = time.time()
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=220, do_sample=True,
                         temperature=0.7, top_p=0.9,
                         pad_token_id=tok.eos_token_id)
dt = time.time() - t0
nuevos = out.shape[1] - ids.input_ids.shape[1]
texto = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)

print("\n--- salida ---")
print(texto)
print("\n--- medicion ---")
print("tokens generados: %d en %.1f s  =>  %.1f tok/s" % (nuevos, dt, nuevos / dt))

crudo = texto.strip()
if crudo.startswith("```"):
    crudo = crudo.strip("`").removeprefix("json").strip()
try:
    d = json.loads(crudo)
    print("JSON valido al primer intento: SI  |  claves:", list(d.keys()))
except Exception as e:
    print("JSON valido al primer intento: NO  (%s)" % e)

# generacion por lotes, para ver cuanto acelera
LOTE = 8
ids_lote = tok([entrada] * LOTE, return_tensors="pt", padding=True).to(model.device)
t0 = time.time()
with torch.no_grad():
    out_l = model.generate(**ids_lote, max_new_tokens=220, do_sample=True,
                           temperature=0.7, top_p=0.9,
                           pad_token_id=tok.eos_token_id)
dt_l = time.time() - t0
nuevos_l = (out_l.shape[1] - ids_lote.input_ids.shape[1]) * LOTE
print("lote de %d: %d tokens en %.1f s  =>  %.1f tok/s efectivos (%.1fx)"
      % (LOTE, nuevos_l, dt_l, nuevos_l / dt_l, (nuevos_l / dt_l) / (nuevos / dt)))
print("memoria GPU pico: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))
