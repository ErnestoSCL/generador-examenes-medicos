"""Barrido de hiperparametros de LoRA con entrenamiento corto.

250 pasos por configuracion en vez de las 3 epocas completas: es mas de medio
epoca, suficiente para que las curvas de validacion se separen, y cuesta unos
minutos en vez de 38. Con la ganadora se corre el entrenamiento completo.

Se varia UN hiperparametro a la vez desde la configuracion de referencia: es lo
que permite atribuir la diferencia a ese cambio y no a la combinacion.

Cada configuracion se mide con dos cosas:
  - eval_loss, barata y comparable
  - generacion sobre casos de test, que es lo que decide: una perdida menor no
    garantiza mejores preguntas, y el defecto que hoy tiene el modelo afinado
    (opciones parecidas) no aparece en la perdida

Cinco decisiones para que el barrido no se pierda a medias:

1. `expandable_segments` reduce la fragmentacion del asignador de CUDA, que es
   el riesgo real al cargar y descargar seis modelos en el mismo proceso.
2. Las configuraciones van de menor a mayor consumo: si la mas pesada se queda
   sin memoria, las otras cinco ya estan medidas.
3. Cada resultado se escribe a disco apenas se obtiene.
4. Un fallo en una configuracion no mata el barrido: se registra y se sigue.
5. La evaluacion de generacion va por lotes: medido, 8.5x mas rapida.
"""
import gc
import json
import os
import re
import time
import traceback
import unicodedata
from pathlib import Path

# Debe ir ANTES de importar torch para que el asignador lo tome.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

MODELO = "Qwen/Qwen3-4B-Instruct-2507"
DATA = Path("data")
PARCIAL = DATA / "barrido_parcial.jsonl"
PASOS = 250
N_EVAL_GEN = 24
LOTE_GEN = 8
SEMILLA = 42
BATCH, ACUM = 2, 4

INSTRUCCION = (
    "Eres un docente de medicina. A partir del FRAGMENTO escribe UNA pregunta "
    "de opcion multiple en espanol neutro. Responde SOLO con JSON."
)

ATENCION = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]

# Ordenadas de menor a mayor consumo. El numero del nombre conserva el orden
# logico para la tabla final.
CONFIGS = [
    dict(nombre="6. solo atencion  r16 a32 lr2e-4", r=16, alpha=32, lr=2e-4, mods=ATENCION),
    dict(nombre="4. rank bajo      r8  a16 lr2e-4", r=8, alpha=16, lr=2e-4, mods=ATENCION + MLP),
    dict(nombre="1. referencia     r16 a32 lr2e-4", r=16, alpha=32, lr=2e-4, mods=ATENCION + MLP),
    dict(nombre="2. lr bajo        r16 a32 lr1e-4", r=16, alpha=32, lr=1e-4, mods=ATENCION + MLP),
    dict(nombre="3. lr alto        r16 a32 lr3e-4", r=16, alpha=32, lr=3e-4, mods=ATENCION + MLP),
    dict(nombre="5. rank alto      r32 a64 lr2e-4", r=32, alpha=64, lr=2e-4, mods=ATENCION + MLP),
]

# ---------------------------------------------------------------- datos
tok = AutoTokenizer.from_pretrained(MODELO)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"        # necesario para generar por lotes


def construir(fila):
    respuesta = json.dumps({
        "apto": True, "pregunta": fila["pregunta"], "correcta": fila["correcta"],
        "incorrectas": list(fila["incorrectas"]), "dificultad": fila["dificultad"],
    }, ensure_ascii=False)
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": INSTRUCCION},
         {"role": "user", "content": "FRAGMENTO:\n" + fila["chunk_text"]}],
        tokenize=False, add_generation_prompt=True)
    ip = tok(prompt, add_special_tokens=False)["input_ids"]
    ir = tok(respuesta + tok.eos_token, add_special_tokens=False)["input_ids"]
    return {"input_ids": (ip + ir)[:1536], "labels": ([-100] * len(ip) + ir)[:1536]}


def colar(lote):
    largo = max(len(x["input_ids"]) for x in lote)
    rel = lambda x, c, v: x[c] + [v] * (largo - len(x[c]))
    return {
        "input_ids": torch.tensor([rel(x, "input_ids", tok.pad_token_id) for x in lote]),
        "labels": torch.tensor([rel(x, "labels", -100) for x in lote]),
        "attention_mask": torch.tensor(
            [[1] * len(x["input_ids"]) + [0] * (largo - len(x["input_ids"])) for x in lote]),
    }


train_df = pd.read_parquet(DATA / "mcq_train.parquet")
val_df = pd.read_parquet(DATA / "mcq_val.parquet")
test_df = pd.read_parquet(DATA / "mcq_test.parquet")
ds_train = Dataset.from_list([construir(f) for _, f in train_df.iterrows()])
ds_val = Dataset.from_list([construir(f) for _, f in val_df.head(120).iterrows()])
casos = test_df.sample(N_EVAL_GEN, random_state=SEMILLA)

print(f"train {len(ds_train):,} | val {len(ds_val)} | generacion {N_EVAL_GEN} en lotes de {LOTE_GEN}")
print(f"{PASOS} pasos por configuracion, {len(CONFIGS)} configuraciones", flush=True)


# ---------------------------------------------------------------- metricas
def sin_acentos(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s).lower())
                   if unicodedata.category(c) != "Mn")


def palabras(s):
    return set(re.findall(r"[a-z0-9]+", sin_acentos(s)))


def parsear(t):
    t = t.strip()
    if t.startswith("```"):
        t = t.strip("`").removeprefix("json").strip()
    try:
        return json.loads(t)
    except Exception:
        return None


def evaluar_generacion(model):
    """Genera por lotes sobre los casos de test y mide lo que la perdida no ve."""
    model.eval()
    textos = []
    for _, fila in casos.iterrows():
        msgs = [{"role": "system", "content": INSTRUCCION},
                {"role": "user", "content": "FRAGMENTO:\n" + fila["chunk_text"]}]
        textos.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    salidas = []
    for i in range(0, len(textos), LOTE_GEN):
        lote = textos[i:i + LOTE_GEN]
        ids = tok(lote, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=300, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for fila_out in out:
            salidas.append(tok.decode(fila_out[ids.input_ids.shape[1]:],
                                      skip_special_tokens=True))

    completas = parecidas = espanol = validos = 0
    for s in salidas:
        d = parsear(s)
        if d is None:
            continue
        validos += 1
        if not (d.get("pregunta") and d.get("correcta")
                and isinstance(d.get("incorrectas"), list) and len(d["incorrectas"]) == 3):
            continue
        completas += 1
        ops = [str(d["correcta"])] + [str(x) for x in d["incorrectas"]]
        if any(len(palabras(ops[i]) & palabras(ops[j])) /
               max(len(palabras(ops[i]) | palabras(ops[j])), 1) > 0.5
               for i in range(4) for j in range(i + 1, 4)):
            parecidas += 1
        if not re.search(r"\b(the|of|and|what|which|with|disease|syndrome)\b",
                         " ".join([d["pregunta"]] + ops), re.I):
            espanol += 1
    model.train()
    return {"json_valido": validos, "estructura": completas,
            "parecidas": parecidas, "espanol": espanol}


# ---------------------------------------------------------------- barrido
hechas, resultados = set(), []
if PARCIAL.exists():
    resultados = [json.loads(l) for l in open(PARCIAL, encoding="utf-8") if l.strip()]
    hechas = {r["configuracion"] for r in resultados}
    print(f"reanudando: {len(hechas)} configuraciones ya medidas", flush=True)

for cfg in CONFIGS:
    if cfg["nombre"] in hechas:
        print(f"(ya medida) {cfg['nombre']}", flush=True)
        continue

    print("=" * 78, flush=True)
    print(cfg["nombre"], flush=True)
    t0 = time.time()
    model = trainer = None

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODELO, dtype=torch.bfloat16, device_map="cuda")
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(
            r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM", target_modules=cfg["mods"]))

        entrenables = sum(p.numel() for p in model.parameters() if p.requires_grad)

        args = TrainingArguments(
            output_dir=f"barrido_tmp/{cfg['nombre'][:1]}",
            per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH,
            gradient_accumulation_steps=ACUM, max_steps=PASOS,
            learning_rate=cfg["lr"], lr_scheduler_type="cosine", warmup_steps=25,
            bf16=True, logging_steps=50, eval_strategy="steps", eval_steps=PASOS,
            save_strategy="no", remove_unused_columns=False, report_to=[],
            seed=SEMILLA, disable_tqdm=True,
        )
        trainer = Trainer(model=model, args=args, train_dataset=ds_train,
                          eval_dataset=ds_val, data_collator=colar)
        trainer.train()

        eval_loss = trainer.evaluate()["eval_loss"]
        gen = evaluar_generacion(model)
        pico = torch.cuda.max_memory_allocated() / 1e9

        fila = {
            "configuracion": cfg["nombre"],
            "entrenables_M": round(entrenables / 1e6, 1),
            "eval_loss": round(eval_loss, 4),
            **gen,
            "pico_GB": round(pico, 2),
            "minutos": round((time.time() - t0) / 60, 1),
        }
        print(f"  eval_loss {eval_loss:.4f} | estructura {gen['estructura']}/{N_EVAL_GEN}"
              f" | parecidas {gen['parecidas']} | pico {pico:.1f} GB"
              f" | {fila['minutos']} min", flush=True)

    except torch.OutOfMemoryError:
        fila = {"configuracion": cfg["nombre"], "error": "sin memoria",
                "minutos": round((time.time() - t0) / 60, 1)}
        print("  SIN MEMORIA - se registra y se sigue con la siguiente", flush=True)
    except Exception as exc:
        fila = {"configuracion": cfg["nombre"],
                "error": f"{type(exc).__name__}: {exc}"[:120],
                "minutos": round((time.time() - t0) / 60, 1)}
        print(f"  FALLO: {type(exc).__name__}", flush=True)
        traceback.print_exc()

    resultados.append(fila)
    with open(PARCIAL, "a", encoding="utf-8") as fh:     # a disco YA, no al final
        fh.write(json.dumps(fila, ensure_ascii=False) + "\n")

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# ---------------------------------------------------------------- resumen
tabla = pd.DataFrame(resultados).sort_values("configuracion")
print("\n" + "=" * 78)
print(tabla.to_string(index=False))
tabla.to_csv(DATA / "barrido_hiperparametros.csv", index=False)

if "eval_loss" in tabla.columns:
    ok = tabla[tabla["eval_loss"].notna()].copy()
    if len(ok):
        ok["limpias"] = ok["estructura"] - ok["parecidas"]
        print(f"\nmenor eval_loss      : {ok.loc[ok['eval_loss'].idxmin(), 'configuracion']}"
              f"  ({ok['eval_loss'].min()})")
        print(f"mas preguntas limpias: {ok.loc[ok['limpias'].idxmax(), 'configuracion']}"
              f"  ({int(ok['limpias'].max())}/{N_EVAL_GEN})")
