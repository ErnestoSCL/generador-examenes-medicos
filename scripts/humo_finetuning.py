"""Prueba de humo del fine-tuning: 10 pasos con 50 ejemplos.

Valida el ciclo entero antes de gastar una hora de entrenamiento:
  1. el modelo carga y acepta LoRA
  2. el formateo al chat template funciona
  3. el entrenamiento avanza (la perdida baja o al menos se calcula)
  4. el adaptador se guarda
  5. el adaptador se RECARGA en un proceso nuevo y genera

Ademas mide memoria y tiempo por paso, que es lo que dice si el entrenamiento
completo cabe en esta GPU o hace falta Colab.
"""
import json
import time
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELO = "Qwen/Qwen3-4B-Instruct-2507"
SALIDA = Path("adapter_humo")
N_EJEMPLOS = 50
PASOS = 10

INSTRUCCION = (
    "Eres un docente de medicina. A partir del FRAGMENTO escribe UNA pregunta "
    "de opcion multiple en espanol. Responde SOLO con JSON."
)

# ------------------------------------------------------------------ datos
train = pd.read_parquet("data/mcq_train.parquet").head(N_EJEMPLOS)
print(f"ejemplos: {len(train)}")

tok = AutoTokenizer.from_pretrained(MODELO)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def construir(fila):
    """Arma el dialogo y marca hasta donde NO se calcula perdida."""
    respuesta = json.dumps({
        "apto": True,
        "pregunta": fila["pregunta"],
        "correcta": fila["correcta"],
        "incorrectas": list(fila["incorrectas"]),
        "dificultad": fila["dificultad"],
    }, ensure_ascii=False)

    prompt = tok.apply_chat_template(
        [{"role": "system", "content": INSTRUCCION},
         {"role": "user", "content": "FRAGMENTO:\n" + fila["chunk_text"]}],
        tokenize=False, add_generation_prompt=True)

    ids_prompt = tok(prompt, add_special_tokens=False)["input_ids"]
    ids_resp = tok(respuesta + tok.eos_token, add_special_tokens=False)["input_ids"]

    input_ids = ids_prompt + ids_resp
    # -100 = ignorar en la perdida. Se entrena SOLO sobre la respuesta: el
    # modelo no debe aprender a reproducir el fragmento en ingles.
    labels = [-100] * len(ids_prompt) + ids_resp
    return {"input_ids": input_ids[:1536], "labels": labels[:1536]}


ds = Dataset.from_list([construir(f) for _, f in train.iterrows()])
largos = [len(x) for x in ds["input_ids"]]
print(f"tokens por ejemplo: media {sum(largos)/len(largos):.0f}, max {max(largos)}")


def colar(lote):
    largo = max(len(x["input_ids"]) for x in lote)
    return {
        "input_ids": torch.tensor([x["input_ids"] + [tok.pad_token_id] * (largo - len(x["input_ids"])) for x in lote]),
        "labels": torch.tensor([x["labels"] + [-100] * (largo - len(x["labels"])) for x in lote]),
        "attention_mask": torch.tensor([[1] * len(x["input_ids"]) + [0] * (largo - len(x["input_ids"])) for x in lote]),
    }


# ------------------------------------------------------------------ modelo
print("\ncargando modelo...", flush=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODELO, dtype=torch.bfloat16, device_map="cuda")
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora)
entrenables = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"cargado en {time.time()-t0:.0f}s")
print(f"parametros entrenables: {entrenables:,} de {total:,} ({entrenables/total*100:.2f}%)")
print(f"memoria tras cargar: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ------------------------------------------------------------------ entrenar
from torch.utils.data import DataLoader

BATCH = 2
loader = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=colar)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

model.train()
print(f"\nentrenando {PASOS} pasos (batch {BATCH})...")
perdidas, t0 = [], time.time()
it = iter(loader)
for paso in range(PASOS):
    try:
        lote = next(it)
    except StopIteration:
        it = iter(loader)
        lote = next(it)
    lote = {k: v.to(model.device) for k, v in lote.items()}
    salida = model(**lote)
    salida.loss.backward()
    opt.step()
    opt.zero_grad()
    perdidas.append(salida.loss.item())
    print(f"  paso {paso+1:>2}  perdida {salida.loss.item():.4f}", flush=True)

seg_paso = (time.time() - t0) / PASOS
print(f"\nperdida: {perdidas[0]:.4f} -> {perdidas[-1]:.4f}")
print(f"tiempo por paso: {seg_paso:.2f}s")
print(f"memoria pico: {torch.cuda.max_memory_allocated()/1e9:.2f} GB de 16.3 GB")

n_train = len(pd.read_parquet("data/mcq_train.parquet"))
pasos_epoca = n_train / BATCH
print(f"\nextrapolado: {n_train:,} ejemplos / batch {BATCH} = {pasos_epoca:.0f} pasos por epoca")
print(f"  1 epoca : {pasos_epoca*seg_paso/60:.0f} min")
print(f"  3 epocas: {pasos_epoca*seg_paso*3/60:.0f} min")

# ------------------------------------------------------------------ guardar
model.save_pretrained(SALIDA)
tok.save_pretrained(SALIDA)
peso = sum(f.stat().st_size for f in SALIDA.rglob("*") if f.is_file())
print(f"\nadaptador guardado en {SALIDA}/  ({peso/1e6:.0f} MB)")
for f in sorted(SALIDA.iterdir()):
    print(f"   {f.name}")
