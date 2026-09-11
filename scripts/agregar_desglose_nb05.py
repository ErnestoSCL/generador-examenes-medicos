"""Agrega al notebook 05 el desglose de por que se rompe cada estructura.

Descarta dos objeciones con datos: que las roturas sean respuestas cortadas por
el limite de 300 tokens, y que las listas de cuatro incorrectas se pudieran
rescatar quitando la correcta repetida. La celda solo lee
`comparacion_prompts_generaciones.json`: se ejecuta aparte en un kernel limpio y
se inserta con sus salidas reales, antes de la seccion B.
"""
import io
import json
import os
import subprocess
import sys

NB = "notebooks/05_afinado_contra_base.ipynb"
TMP, TMP_EJEC = "_tmp_desglose.ipynb", "_tmp_desglose_ejec.ipynb"

CELDA = r'''
# Por que se rompe la estructura: clasifica cada salida rota. Lee lo guardado
# arriba: no regenera nada.
import json, unicodedata
from collections import Counter
from pathlib import Path

DATA = Path("../data")
detalle = json.load(open(DATA / "comparacion_prompts_generaciones.json", encoding="utf-8"))
norm = lambda s: " ".join("".join(c for c in unicodedata.normalize("NFD", str(s).lower())
                                  if unicodedata.category(c) != "Mn").strip(" .").split())


def estructura_ok(d):
    return bool(isinstance(d, dict) and d.get("pregunta") and d.get("correcta")
                and isinstance(d.get("incorrectas"), list) and len(d["incorrectas"]) == 3)


def motivo(s):
    if isinstance(s, dict):
        if s.get("apto") is False:
            return "descarto el fragmento (apto=false)"
        inc = s.get("incorrectas")
        if isinstance(inc, list) and len(inc) == 4:
            repetida = norm(s.get("correcta")) in [norm(x) for x in inc]
            return "4 incorrectas, " + ("una es la correcta repetida" if repetida
                                        else "ninguna es la correcta: no se puede rescatar")
        if isinstance(inc, list):
            return f"{len(inc)} incorrecta(s)"
        return f"otras claves: {sorted(s)}"
    txt = str(s).strip()
    return "JSON invalido, " + ("cerrado con }: no es un corte" if txt.endswith("}")
                                else "sin cerrar: posible corte por limite de tokens")


for clave, etiqueta in (("afinado", "afinado (LoRA)"), ("fewshot", "base + few-shot"),
                        ("A", "base A: mismo prompt"), ("B", "base B: + esquema"),
                        ("C", "base C: prompt del maestro")):
    rotas = Counter(motivo(x[clave]["salida"]) for x in detalle
                    if not estructura_ok(x[clave]["salida"]))
    print(f"{etiqueta}: {sum(rotas.values())} rotas de {len(detalle)}")
    for m, n in rotas.most_common():
        print(f"   {n:>3}  {m}")
'''.strip() + "\n"

MD = """### ¿Por qué se rompe la estructura?

Clasifica cada salida con la estructura rota. Descarta dos objeciones: que las
roturas sean respuestas cortadas por el límite de 300 tokens, y que las listas con
cuatro incorrectas se pudieran rescatar quitando la respuesta correcta repetida.
"""

nb = json.load(io.open(NB, encoding="utf-8"))
if any("¿Por qué se rompe la estructura?" in "".join(c["source"]) for c in nb["cells"]):
    sys.exit("la celda ya existe: no se toca")

compile(CELDA, "celda", "exec")
tmp = {"cells": [{"cell_type": "code", "id": "tmp0", "metadata": {}, "outputs": [],
                  "execution_count": None, "source": CELDA.splitlines(keepends=True)}],
       "metadata": nb["metadata"], "nbformat": 4, "nbformat_minor": 5}
json.dump(tmp, io.open(os.path.join("notebooks", TMP), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                    "--execute", "--output", TMP_EJEC, TMP],
                   cwd="notebooks", capture_output=True, text=True)
if r.returncode != 0:
    sys.exit("fallo la ejecucion de la celda:\n" + r.stderr[-1500:])
celda = json.load(io.open(os.path.join("notebooks", TMP_EJEC), encoding="utf-8"))["cells"][0]
if any(o["output_type"] == "error" for o in celda["outputs"]):
    sys.exit("la celda dio error")
for f in (TMP, TMP_EJEC):
    os.remove(os.path.join("notebooks", f))

pos = next(i for i, c in enumerate(nb["cells"]) if "".join(c["source"]).startswith("## B."))
celda["id"] = "c17"
nb["cells"][pos:pos] = [
    {"cell_type": "markdown", "id": "c16", "metadata": {}, "source": MD.splitlines(keepends=True)},
    celda,
]
json.dump(nb, io.open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

sys.stdout.reconfigure(encoding="utf-8")
print("salida real de la celda insertada:\n")
print("".join("".join(o.get("text", "")) for o in celda["outputs"]))
print(f"notebook 05: {len(nb['cells'])} celdas")
