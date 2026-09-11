"""Agrega al notebook 05 la prueba por criterio (ejecutada de verdad) y las conclusiones.

La celda de la prueba por criterio solo lee `base_vs_afinado_generaciones.json`,
asi que se ejecuta aparte en un kernel limpio con nbconvert y se inserta con sus
salidas reales. No hace falta volver a generar ni a juzgar nada.
"""
import io
import json
import os
import subprocess
import sys

NB = "notebooks/05_base_vs_afinado.ipynb"
TMP, TMP_EJEC = "_tmp_criterios.ipynb", "_tmp_criterios_ejec.ipynb"

CRITERIOS = r'''
# Prueba pareada criterio por criterio, sobre los fragmentos donde los dos
# modelos tienen una pregunta juzgada. Lee lo guardado arriba: no regenera nada.
import json
from pathlib import Path
from scipy.stats import binomtest

DATA = Path("../data")
detalle = json.load(open(DATA / "base_vs_afinado_generaciones.json", encoding="utf-8"))
real = lambda v: v is not None and not v.get("_estructura_rota")
pares = [(x["base"]["veredicto"], x["afinado"]["veredicto"]) for x in detalle
         if real(x["base"]["veredicto"]) and real(x["afinado"]["veredicto"])]
print(f"fragmentos con las dos preguntas juzgadas: {len(pares)}\n")

BUENO = {"respaldo": True, "distractor_verdadero": False,
         "unica_respuesta": True, "tema_correcto": True}
print(f"{'criterio':<22}{'base':>6}{'afinado':>9}{'solo afinado':>14}{'solo base':>11}{'p':>8}")
for criterio, bueno in BUENO.items():
    if bueno:
        ok = lambda v: bool(v.get(criterio))
    else:
        ok = lambda v: v.get(criterio) is False
    n_base = sum(ok(b) for b, a in pares)
    n_af = sum(ok(a) for b, a in pares)
    solo_af = sum(ok(a) and not ok(b) for b, a in pares)
    solo_base = sum(ok(b) and not ok(a) for b, a in pares)
    p = binomtest(solo_af, solo_af + solo_base, 0.5).pvalue if solo_af + solo_base else 1.0
    print(f"{criterio:<22}{n_base:>6}{n_af:>9}{solo_af:>14}{solo_base:>11}{p:>8.3f}")
'''.strip() + "\n"

MD_CRITERIOS = """### La misma prueba, criterio por criterio

Si el total no muestra diferencia, puede que un criterio concreto sí la tenga y
quede diluido. Se comprueba sobre lo guardado arriba, sin regenerar nada.
"""

CONCLUSIONES = """## Conclusiones

### A. Contenido: no hay diferencia demostrable

| | maestro | base + few-shot | afinado |
|---|---|---|---|
| Sin defecto (de las juzgadas) | 78.2% | 72.6% | 75.9% |

- La prueba pareada afinado contra base da **p = 0.73**: 18 fragmentos donde
  solo acierta el afinado, 15 donde solo acierta el base. Criterio por criterio
  tampoco hay nada significativo. La diferencia mayor, «tema correcto», da
  p = 0.31, y en respaldo el base queda incluso por encima.
- **Con esta muestra no se puede afirmar que el fine-tuning mejorara la
  corrección del contenido**, ni que la empeorara.
- Consecuencia para el notebook 03: el base sin entrenar **tampoco** se distingue
  del maestro (p = 0.22). El empate alumno-maestro no prueba por sí solo que el
  entrenamiento aportara contenido.

### B. Formato: la diferencia está aquí, pero no donde se había dicho

- El base con few-shot aguanta el muestreo: 57/60 a temperatura 0.7 y 58/60 a
  1.0, con variedad casi igual. **La robustez del formato bajo muestreo no es
  mérito del fine-tuning**, como se había afirmado a partir del notebook 04.
- Lo que sí es propio del afinado: **no rompió la estructura ni una vez** en 330
  generaciones (notebooks 03, 04 y 05). El base la rompió 12 veces: 3 de 60,
  4 de 150 y 5 de 120.

### Qué compró el fine-tuning

| Aspecto | ¿Mejoró? | Evidencia |
|---|---|---|
| Corrección del contenido | no demostrable | 75.9% contra 72.6%, p = 0.73 |
| Estructura válida | sí, de forma consistente | 0 contra 12 fallos en 330 generaciones |
| Tamaño del prompt | sí | 137 contra 695 tokens: 5.1 veces menos |
| Robustez al muestreo | no es mérito suyo | el base da 57-58 de 60 a 0.7 y 1.0 |
| Velocidad | no, empeora | 1.01 s contra 0.80 s por pregunta, en lotes de 8 |

El fine-tuning convirtió una tarea que necesitaba un prompt de 695 tokens con
ejemplos en una que se resuelve con 137 y que nunca rompe el formato. **No hizo al
modelo más preciso.** Presentarlo así es más sólido que atribuirle mejoras que la
medición no respalda.

### Dos avisos sobre las cifras

- **24 de las 450 llamadas al juez fallaron** tras tres intentos (8 del maestro,
  11 del base y 5 del afinado) y se excluyeron. Por eso la fila «de los 150
  fragmentos» se calcula en realidad sobre 142, 139 y 145, y la prueba pareada
  sobre los 126 fragmentos con los tres veredictos.
- **Hay ruido entre corridas.** El mismo afinado sobre los mismos 150 fragmentos
  dio 78.4% en el notebook 03 y 75.9% aquí; el maestro, 79.3% y 78.2%. El juez a
  temperatura 0 no es del todo determinista, y los fallos de la API excluyen
  fragmentos distintos en cada corrida. Variaciones de 2 o 3 puntos son ruido,
  no cambios.
"""

nb = json.load(io.open(NB, encoding="utf-8"))
if any("## Conclusiones" in "".join(c["source"]) for c in nb["cells"]):
    sys.exit("el notebook 05 ya tiene conclusiones: no se toca")

compile(CRITERIOS, "criterios", "exec")
tmp = {"cells": [{"cell_type": "code", "id": "tmp0", "metadata": {}, "outputs": [],
                  "execution_count": None, "source": CRITERIOS.splitlines(keepends=True)}],
       "metadata": nb["metadata"], "nbformat": 4, "nbformat_minor": 5}
json.dump(tmp, io.open(os.path.join("notebooks", TMP), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                    "--execute", "--output", TMP_EJEC, TMP],
                   cwd="notebooks", capture_output=True, text=True)
if r.returncode != 0:
    sys.exit("fallo la ejecucion de la celda:\n" + r.stderr[-1500:])

celda = json.load(io.open(os.path.join("notebooks", TMP_EJEC), encoding="utf-8"))["cells"][0]
errores = [o for o in celda["outputs"] if o["output_type"] == "error"]
if errores:
    sys.exit("la celda dio error: %s" % errores[0]["evalue"])
for f in (TMP, TMP_EJEC):
    os.remove(os.path.join("notebooks", f))

n = len(nb["cells"])
celda["id"] = "c%02d" % (n + 1)
nb["cells"].append({"cell_type": "markdown", "id": "c%02d" % n, "metadata": {},
                    "source": MD_CRITERIOS.splitlines(keepends=True)})
nb["cells"].append(celda)
nb["cells"].append({"cell_type": "markdown", "id": "c%02d" % (n + 2), "metadata": {},
                    "source": CONCLUSIONES.splitlines(keepends=True)})
json.dump(nb, io.open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

sys.stdout.reconfigure(encoding="utf-8")
print("salida real de la celda insertada:\n")
print("".join("".join(o.get("text", "")) for o in celda["outputs"]))
print("notebook 05: %d celdas" % len(nb["cells"]))
