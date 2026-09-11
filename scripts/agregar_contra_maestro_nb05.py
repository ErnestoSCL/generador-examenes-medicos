"""Agrega al notebook 05 la comparacion de cada fuente contra el maestro.

La celda solo lee `comparacion_prompts_generaciones.json`, asi que se ejecuta
aparte en un kernel limpio con nbconvert y se inserta con sus salidas reales,
despues de la celda que guarda ese archivo. No regenera ni vuelve a juzgar nada.
"""
import io
import json
import os
import subprocess
import sys

NB = "notebooks/05_afinado_contra_base.ipynb"
TMP, TMP_EJEC = "_tmp_maestro.ipynb", "_tmp_maestro_ejec.ipynb"

CELDA = r'''
# Cada fuente contra el maestro, fragmento a fragmento. Lee lo guardado arriba:
# no regenera nada.
import json
from pathlib import Path
from scipy.stats import binomtest

DATA = Path("../data")
detalle = json.load(open(DATA / "comparacion_prompts_generaciones.json", encoding="utf-8"))


def limpia(v):
    return bool(v and not v.get("_estructura_rota") and v.get("respaldo")
                and not v.get("distractor_verdadero") and v.get("unica_respuesta")
                and v.get("tema_correcto"))


ETIQ = {"afinado": "afinado (LoRA)", "fewshot": "base + few-shot",
        "A": "base A: mismo prompt", "B": "base B: + esquema",
        "C": "base C: prompt del maestro"}
print(f"{'contra el maestro':<28}{'sin defecto':>12}{'solo esta':>11}{'solo maestro':>14}{'p':>9}")
for clave, etiqueta in ETIQ.items():
    pares = [(x["maestro"]["veredicto"], x[clave]["veredicto"]) for x in detalle
             if x["maestro"]["veredicto"] is not None and x[clave]["veredicto"] is not None]
    solo_esta = sum(limpia(v) and not limpia(m) for m, v in pares)
    solo_m = sum(limpia(m) and not limpia(v) for m, v in pares)
    p = binomtest(solo_esta, solo_esta + solo_m, 0.5).pvalue if solo_esta + solo_m else 1.0
    tasa = sum(limpia(v) for m, v in pares) / len(pares) * 100
    print(f"{etiqueta:<28}{tasa:>11.1f}%{solo_esta:>11}{solo_m:>14}{p:>9.4f}")
print("\np < 0.05: se distingue del maestro. p alto: no se distingue del ruido.")
'''.strip() + "\n"

MD = """### Cada fuente contra el maestro

La prueba anterior compara todo contra el afinado. Esta responde otra pregunta:
**¿quién alcanza al maestro?** Si el afinado no se distingue del maestro y las
variantes del base sí, el empate con el maestro dice algo del entrenamiento; si
todas empatan, no.
"""

nb = json.load(io.open(NB, encoding="utf-8"))
if any("Cada fuente contra el maestro" in "".join(c["source"]) for c in nb["cells"]):
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

# Va despues de la celda que guarda el JSON y antes de la seccion B: asi, al
# re-ejecutar el notebook de arriba abajo, el archivo ya existe cuando se lee.
pos = next(i for i, c in enumerate(nb["cells"]) if "".join(c["source"]).startswith("## B."))
celda["id"] = "c15"
nb["cells"][pos:pos] = [
    {"cell_type": "markdown", "id": "c14", "metadata": {}, "source": MD.splitlines(keepends=True)},
    celda,
]
json.dump(nb, io.open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

sys.stdout.reconfigure(encoding="utf-8")
print("salida real de la celda insertada:\n")
print("".join("".join(o.get("text", "")) for o in celda["outputs"]))
print(f"notebook 05: {len(nb['cells'])} celdas")
