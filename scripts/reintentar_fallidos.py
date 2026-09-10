"""Reintenta los fragmentos cuya llamada a la API fallo (429 y similares).

El JSONL guarda tambien los fallos, para saber que paso. Este script los
reemplaza por el resultado bueno, con menos concurrencia para no volver a
chocar con el limite de tasa.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".env")
client = OpenAI()
RUTA = Path("data/mcq_crudo.jsonl")

PROMPT = json.load(open("scripts/prompt_v2.json", encoding="utf-8"))["prompt"]

regs = [json.loads(l) for l in open(RUTA, encoding="utf-8")]
fallidos = [r for r in regs if r["error"]]
buenos = [r for r in regs if not r["error"]]
print(f"registros: {len(regs):,} | fallidos: {len(fallidos)}")
if not fallidos:
    raise SystemExit("nada que reintentar")

muestra = pd.read_parquet("data/muestra_api.parquet").set_index("chunk_uid")["chunk_text"]


def pedir(uid, reintentos=5):
    mensajes = [{"role": "system", "content": PROMPT},
                {"role": "user", "content": "FRAGMENTO:\n" + muestra[uid]}]
    for i in range(reintentos):
        try:
            t0 = time.time()
            r = client.chat.completions.create(
                model="gpt-4o-mini", messages=mensajes,
                temperature=0.7, max_tokens=320)
            return {"chunk_uid": uid, "crudo": r.choices[0].message.content,
                    "tok_in": r.usage.prompt_tokens, "tok_out": r.usage.completion_tokens,
                    "segundos": round(time.time() - t0, 2), "error": None}
        except Exception as exc:
            if i == reintentos - 1:
                return {"chunk_uid": uid, "crudo": None, "tok_in": 0, "tok_out": 0,
                        "segundos": 0, "error": f"{type(exc).__name__}: {exc}"}
            time.sleep(3 * (i + 1))          # espera creciente y generosa


nuevos = []
with ThreadPoolExecutor(3) as pool:          # 3 hilos, no 12: el fallo fue por tasa
    futs = {pool.submit(pedir, r["chunk_uid"]): r["chunk_uid"] for r in fallidos}
    for fut in as_completed(futs):
        nuevos.append(fut.result())

recuperados = sum(1 for n in nuevos if not n["error"])
print(f"recuperados: {recuperados}/{len(fallidos)}")

with open(RUTA, "w", encoding="utf-8") as fh:
    for r in buenos + nuevos:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"archivo reescrito: {len(buenos) + len(nuevos):,} registros")
