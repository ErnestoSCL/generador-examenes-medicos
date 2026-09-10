"""Genera un examen como lo hace la aplicacion y lo audita pregunta por pregunta.

Sirve para responder con un numero, no con una impresion: cuantas de las N
preguntas que ve un estudiante tienen algun defecto, y de que tipo.

Anade dos criterios que el juez de la aplicacion no tiene:

  util          una pregunta puede estar respaldada por el fragmento y ser del
                tema y aun asi no servir. Las opciones "Ver opciones para cancer
                localizado" son un caso real: el juez de la app la APROBO.
  traduccion_ok el corpus esta en ingles y las preguntas salen en espanol. Caso
                real: "nipple" traducido como "lactancia".

Incluye ademas una **sonda del auto-verificador**: se le pasan preguntas con
defectos conocidos para comprobar si los rechaza. Hizo falta porque en dos
corridas seguidas el contador de rechazos por verificacion dio cero, y un
contador en cero no distingue "no habia nada que rechazar" de "esta inerte".
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "app")
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import pandas as pd

import banco
import modelo
from validacion import construir_verificacion, leer_verificacion

TEMA = sys.argv[1] if len(sys.argv) > 1 else "Breast Cancer"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
SALIDA = Path("data/auditoria_examen.json")

RUBRICA = """Eres un revisor severo de examenes de medicina. Recibes un FRAGMENTO
de un documento de los NIH y una PREGUNTA de opcion multiple construida a partir
de el.

Responde SOLO con JSON:
{"respaldo": true/false,
 "distractor_cierto": true/false,
 "tema_correcto": true/false,
 "util": true/false,
 "traduccion_ok": true/false,
 "defecto": "una frase describiendo el problema principal, o 'ninguno'"}

respaldo         : la opcion correcta esta afirmada explicitamente en el fragmento?
distractor_cierto: alguna opcion incorrecta es TAMBIEN cierta segun el fragmento?
tema_correcto    : la pregunta habla del tema del fragmento y no de otra enfermedad?
util             : sirve como pregunta de examen? Marca FALSE si las opciones son
                   etiquetas de navegacion ("ver opciones para..."), referencias
                   cruzadas, encabezados de tabla, o si no evalua conocimiento
                   medico sino trivialidades.
traduccion_ok    : el espanol es correcto y fiel al fragmento en ingles? Marca
                   FALSE si un termino se tradujo mal de forma que cambie el
                   sentido (por ejemplo "nipple" como "lactancia")."""

CRITERIOS = [
    ("respaldo", True, "la correcta no esta en el fragmento"),
    ("distractor_cierto", False, "un distractor tambien es cierto"),
    ("tema_correcto", True, "no corresponde al tema"),
    ("util", True, "no sirve como pregunta de examen"),
    ("traduccion_ok", True, "error de traduccion"),
]


def juzgar(p):
    from openai import OpenAI
    contenido = (
        "FRAGMENTO:\n" + p["fragmento"] + "\n\nPREGUNTA: " + p["pregunta"] +
        "\nCORRECTA: " + p["correcta"] + "\n" +
        "\n".join("INCORRECTA: " + str(x) for x in p["incorrectas"]))
    r = OpenAI().chat.completions.create(
        model="gpt-4o", temperature=0,
        messages=[{"role": "system", "content": RUBRICA},
                  {"role": "user", "content": contenido}])
    t = r.choices[0].message.content.strip()
    if t.startswith("```"):
        t = t.strip("`").removeprefix("json").strip()
    return json.loads(t)


def sondear_verificador(gen):
    """Le da al auto-verificador preguntas con defectos conocidos.

    Si aprueba las cuatro, el segundo nivel de validacion no esta protegiendo
    nada y conviene saberlo antes de presentarlo como una garantia.
    """
    crudo = pd.read_parquet("data/chunks.parquet")
    nav = crudo[crudo.chunk_text.str.contains(
        "operable stage IIIC breast cancer, see", regex=False)].iloc[0]
    asma = crudo[crudo.question_focus == "Asthma"].iloc[0]

    casos = [
        ("navegacion (caso real de la app)", nav.chunk_text, "Cancer de mama", {
            "pregunta": "Cual es el tratamiento recomendado para el cancer de mama en etapas I a IIIA?",
            "correcta": "Ver opciones para cancer localizado o operable",
            "incorrectas": ["Ver opciones para cancer recidivante localizado",
                            "Ver opciones para cancer avanzado o inflamatorio",
                            "Ver opciones para cancer metastasico generalizado"]}),
        ("correcta inventada", asma.chunk_text, "Asma", {
            "pregunta": "Cual es el tratamiento de primera linea para el asma?",
            "correcta": "La administracion diaria de penicilina inyectable",
            "incorrectas": ["Los broncodilatadores de accion corta",
                            "Los corticoides inhalados",
                            "Los antagonistas de leucotrienos"]}),
        ("tema equivocado", asma.chunk_text, "Asma", {
            "pregunta": "Cual es el agente causal de la tuberculosis pulmonar?",
            "correcta": "Mycobacterium tuberculosis",
            "incorrectas": ["Streptococcus pneumoniae",
                            "Haemophilus influenzae",
                            "Klebsiella pneumoniae"]}),
        ("control: pregunta sana", asma.chunk_text, "Asma", None),
    ]

    print("\n" + "=" * 90)
    print("SONDA DEL AUTO-VERIFICADOR")
    print("=" * 90)

    sana = gen.generar([dict(chunk_text=asma.chunk_text, tema_es="Asma",
                             document_url=asma.document_url,
                             document_source=asma.document_source)],
                       verificar=False)
    if sana:
        casos[-1] = (casos[-1][0], asma.chunk_text, "Asma", {
            "pregunta": sana[0]["pregunta"], "correcta": sana[0]["correcta"],
            "incorrectas": sana[0]["incorrectas"]})

    resultados = []
    for nombre, fragmento, tema, d in casos:
        if d is None:
            print("  %-34s  (no se pudo generar el control)" % nombre)
            continue
        prompt = construir_verificacion(gen.tok, d, fragmento, tema)
        # Mismo camino que usa la aplicacion: verifica el modelo base.
        with gen.model.disable_adapter():
            salidas, _ = gen._generar_lote([prompt], max_new_tokens=90, do_sample=False,
                                           temperature=None, top_p=None, top_k=None)
        fallos, err = leer_verificacion(salidas[0])
        veredicto = "RECHAZA: " + "; ".join(fallos) if fallos else "aprueba"
        if err:
            veredicto = "no respondio -> aprueba por defecto"
        esperado = "aprueba" if nombre.startswith("control") else "RECHAZA"
        acierta = veredicto.startswith(esperado)
        print("  %-34s  %-55s  %s" % (nombre, veredicto, "OK" if acierta else "FALLA"))
        resultados.append({"caso": nombre, "veredicto": veredicto, "acierta": acierta})
    return resultados


trozos = banco.fragmentos(question_focus=TEMA, n=int(N * 1.6), semilla=7)
print("tema: %s  |  %d fragmentos pedidos" % (TEMA, len(trozos)))

gen = modelo.Generador()
preguntas = gen.generar(trozos, verificar=True)[:N]
contadores = gen.resumen()
print("generadas y aprobadas por la app: %d" % len(preguntas))
print("contadores:", contadores)

# Se guarda ANTES de juzgar: si falla la API no se pierde la generacion.
SALIDA.parent.mkdir(exist_ok=True)
json.dump({"tema": TEMA, "contadores": contadores, "preguntas": preguntas},
          open(SALIDA, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

sonda = sondear_verificador(gen)

filas = []
print("\n" + "=" * 90)
print("AUDITORIA DE LAS %d PREGUNTAS" % len(preguntas))
for i, p in enumerate(preguntas, 1):
    try:
        v = juzgar(p)
    except Exception as exc:
        print("%d. no se pudo juzgar: %s" % (i, type(exc).__name__))
        continue
    fallos = [msg for campo, esperado, msg in CRITERIOS if v.get(campo) is not esperado]
    filas.append({"n": i, "fallos": fallos, "defecto": v.get("defecto", "")})
    print("=" * 90)
    print("%d. %s" % (i, p["pregunta"]))
    print("   V %s" % p["correcta"])
    for x in p["incorrectas"]:
        print("   x %s" % x)
    print("   fuente: %s" % p["fuente"])
    print("   >>> %s" % ("SIN DEFECTOS" if not fallos else "DEFECTOS: " + "; ".join(fallos)))
    if fallos:
        print("       juez: %s" % v.get("defecto", ""))

buenas = sum(1 for f in filas if not f["fallos"])
print("\n" + "=" * 90)
print("SIN DEFECTOS: %d de %d" % (buenas, len(filas)))
conteo = {}
for f in filas:
    for x in f["fallos"]:
        conteo[x] = conteo.get(x, 0) + 1
for k, v in sorted(conteo.items(), key=lambda kv: -kv[1]):
    print("  %-42s %d" % (k, v))

json.dump({"tema": TEMA, "contadores": contadores, "preguntas": preguntas,
           "auditoria": filas, "sonda_verificador": sonda},
          open(SALIDA, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\nguardado en %s" % SALIDA)
