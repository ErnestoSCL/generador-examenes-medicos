"""Mide si subir la temperatura empeora la correccion factual.

El notebook 04 dejo un hueco: mide formato y variedad, no veracidad. Senala
temperatura 1.0 como ganadora (variedad 2.70 de 3 contra 2.30) sin perder
formato, pero subir la temperatura suele aumentar la invencion, y eso no estaba
medido. Sin este dato, quedarse en 0.7 es precaucion, no evidencia.

Las dos temperaturas se corren sobre **los mismos fragmentos**: comparar sobre
fragmentos distintos mediria la dificultad del texto, no el efecto del parametro.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "app")
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from concurrent.futures import ThreadPoolExecutor

import banco
import modelo

TEMAS = ["Asthma", "Breast Cancer", "Parkinson's Disease", "Diabetes"]
POR_TEMA = 8
TEMPERATURAS = [0.7, 1.0]
SALIDA = Path("data/comparacion_temperatura.json")

RUBRICA = """Eres un revisor severo de examenes de medicina. Recibes un FRAGMENTO
de un documento de los NIH y una PREGUNTA de opcion multiple construida a partir
de el.

Responde SOLO con JSON:
{"respaldo": true/false, "distractor_cierto": true/false,
 "tema_correcto": true/false, "util": true/false, "traduccion_ok": true/false,
 "defecto": "una frase, o 'ninguno'"}

respaldo         : la opcion correcta esta afirmada explicitamente en el fragmento?
distractor_cierto: alguna opcion incorrecta es TAMBIEN cierta segun el fragmento?
tema_correcto    : la pregunta habla del tema del fragmento?
util             : sirve como pregunta de examen? FALSE si las opciones son
                   etiquetas de navegacion o si no evalua conocimiento medico.
traduccion_ok    : el espanol es fiel al fragmento en ingles?"""

CRITERIOS = [("respaldo", True, "sin respaldo"),
             ("distractor_cierto", False, "distractor cierto"),
             ("tema_correcto", True, "tema equivocado"),
             ("util", True, "no sirve"),
             ("traduccion_ok", True, "traduccion")]


def juzgar(p):
    from openai import OpenAI
    contenido = ("FRAGMENTO:\n" + p["fragmento"] + "\n\nPREGUNTA: " + p["pregunta"] +
                 "\nCORRECTA: " + p["correcta"] + "\n" +
                 "\n".join("INCORRECTA: " + str(x) for x in p["incorrectas"]))
    try:
        r = OpenAI().chat.completions.create(
            model="gpt-4o", temperature=0,
            messages=[{"role": "system", "content": RUBRICA},
                      {"role": "user", "content": contenido}])
        t = r.choices[0].message.content.strip()
        if t.startswith("```"):
            t = t.strip("`").removeprefix("json").strip()
        return json.loads(t)
    except Exception as exc:
        return {"error": "%s: %s" % (type(exc).__name__, exc)}


# Los mismos fragmentos para las dos temperaturas, con semilla fija.
trozos = []
for tema in TEMAS:
    trozos += banco.fragmentos(question_focus=tema, n=POR_TEMA, semilla=11)
print("%d fragmentos de %d temas, identicos para ambas temperaturas\n" % (len(trozos), len(TEMAS)))

gen = modelo.Generador()
resultados = {}

for temp in TEMPERATURAS:
    modelo.GENERACION["temperature"] = temp
    # Sin auto-verificacion: se quiere medir el efecto de la temperatura sobre
    # lo que el modelo produce, no lo que sobrevive al filtro.
    preguntas = gen.generar(trozos, verificar=False, reintentos=0)
    print("temperatura %.1f -> %d preguntas generadas, %d pasaron la forma"
          % (temp, gen.contadores["generadas"], len(preguntas)), flush=True)

    with ThreadPoolExecutor(max_workers=8) as ex:
        veredictos = list(ex.map(juzgar, preguntas))

    filas = []
    for p, v in zip(preguntas, veredictos):
        if "error" in v:
            continue
        fallos = [msg for campo, esp, msg in CRITERIOS if v.get(campo) is not esp]
        filas.append({"pregunta": p["pregunta"], "tema": p["tema"], "fallos": fallos})
    resultados[str(temp)] = {
        "generadas": gen.contadores["generadas"],
        "paso_forma": len(preguntas),
        "juzgadas": len(filas),
        "sin_defectos": sum(1 for f in filas if not f["fallos"]),
        "filas": filas,
    }

print("\n" + "=" * 78)
print("%-14s %10s %10s %10s %12s" % ("temperatura", "generadas", "forma ok", "juzgadas", "sin defectos"))
print("=" * 78)
for temp in TEMPERATURAS:
    r = resultados[str(temp)]
    pct = r["sin_defectos"] / max(r["juzgadas"], 1) * 100
    print("%-14s %10d %10d %10d %7d (%.1f%%)"
          % (temp, r["generadas"], r["paso_forma"], r["juzgadas"], r["sin_defectos"], pct))

print("\ndefectos por tipo:")
tipos = sorted({m for _, _, m in CRITERIOS})
print("  %-20s %10s %10s" % ("tipo", "temp 0.7", "temp 1.0"))
for t in tipos:
    fila = ["%d" % sum(1 for f in resultados[str(x)]["filas"] if t in f["fallos"])
            for x in TEMPERATURAS]
    print("  %-20s %10s %10s" % (t, fila[0], fila[1]))

json.dump(resultados, open(SALIDA, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nguardado en %s" % SALIDA)
