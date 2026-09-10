"""Validacion de preguntas generadas.

Codigo COMPARTIDO con el notebook 02: los mismos filtros que limpiaron el set de
entrenamiento validan lo que el modelo genera en vivo. Si una pregunta no pasa,
la aplicacion la regenera en vez de mostrarla.

Hay dos niveles, y la diferencia importa:

  revisar_forma()      reglas sobre el texto. Baratas, instantaneas, y ciegas al
                       contenido: no ven si la respuesta correcta esta realmente
                       en el fragmento.

  autoverificar()      el modelo revisa su propia pregunta. Ve lo que las reglas
                       no ven, a cambio de una segunda pasada de generacion.

La medicion que justifica el segundo nivel: sobre 150 preguntas del modelo
afinado, los nueve filtros de forma no detectaron NINGUNO de los siete defectos
encontrados leyendo a mano. Todos eran semanticos: la correcta no estaba en el
fragmento, un distractor tambien era cierto, o la pregunta cambiaba el nombre de
la enfermedad (equinococosis por cisticercosis, ratones por ratas).
"""
import json
import re
import unicodedata

MAX_PALABRAS = 15
MAX_RATIO = 2.5
MIN_LARGO_PARA_RATIO = 40   # por debajo, todas las opciones caben en un renglon
SOLAPE_MAX = 0.5

DEICTICO = re.compile(
    r"\b(mencionad[oa]s?|citad[oa]s?|el (texto|fragmento|documento|pasaje)|"
    r"seg[uú]n el|el siguiente|indicad[oa]s?)\b", re.IGNORECASE)
NEGATIVA = re.compile(
    r"\bno\s+(influye|afecta|es|son|se|tiene|forma|corresponde|pertenece|"
    r"deber[ía]|puede|causa|produce)\b|\bexcepto\b", re.IGNORECASE)


def sin_acentos(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s).lower())
                   if unicodedata.category(c) != "Mn")


def palabras(s):
    return set(re.findall(r"[a-z0-9]+", sin_acentos(s)))


def parsear(texto):
    """Extrae el JSON de la salida del modelo, tolerando cercos de markdown."""
    if not texto:
        return None
    t = texto.strip()
    if t.startswith("```"):
        t = t.strip("`").removeprefix("json").strip()
    try:
        d = json.loads(t)
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def estructura_ok(d):
    return bool(d and d.get("pregunta") and d.get("correcta")
                and isinstance(d.get("incorrectas"), list)
                and len(d["incorrectas"]) == 3)


def revisar_forma(d, fragmento):
    """Devuelve la lista de reglas que la pregunta incumple. Vacia = pasa."""
    if not estructura_ok(d):
        return ["estructura"]

    fallos = []
    opciones = [str(d["correcta"])] + [str(x) for x in d["incorrectas"]]

    if len(set(sin_acentos(o).strip(" .") for o in opciones)) < 4:
        fallos.append("opciones identicas")

    # Antes de comparar se quita el prefijo comun a las cuatro opciones. La
    # estructura paralela ("Mutaciones en el gen X") es BUENA practica en
    # examenes, y sin este descuento el filtro rechazaba justo las preguntas
    # mejor construidas: cuatro genes distintos daban 0.8 de solape porque
    # compartian "mutaciones en el gen".
    comunes = set.intersection(*(palabras(o) for o in opciones))
    distintivas = [palabras(o) - comunes for o in opciones]

    for i in range(4):
        for j in range(i + 1, 4):
            a, b = distintivas[i], distintivas[j]
            if not a or not b:
                # una opcion sin nada propio ES el problema: solo el prefijo
                fallos.append("una opcion no aporta nada distinto")
                break
            if len(a & b) / len(a | b) > SOLAPE_MAX:
                fallos.append("dos opciones demasiado parecidas")
                break
        else:
            continue
        break

    if max(len(o.split()) for o in opciones) > MAX_PALABRAS:
        fallos.append("una opcion supera 15 palabras")

    largos = [len(o) for o in opciones]
    # El ratio solo importa cuando alguna opcion es larga de verdad. Con cuatro
    # opciones cortas ("Migrana" contra "Cefalea en racimos") el ratio se dispara
    # sin que haya desequilibrio real: 18/7 = 2.6 y las dos caben en un renglon.
    if max(largos) > MIN_LARGO_PARA_RATIO and max(largos) / max(min(largos), 1) > MAX_RATIO:
        # Si la correcta es siempre la mas larga se acierta sin saber el tema.
        fallos.append("longitudes desparejas")

    frag = sin_acentos(fragmento)
    for x in d["incorrectas"]:
        t = sin_acentos(x).strip(" .")
        if len(t) > 12 and t in frag:
            # Si esta literal en el texto, probablemente tambien sea cierta.
            fallos.append("un distractor aparece en el fragmento")
            break

    if DEICTICO.search(d["pregunta"]):
        fallos.append("se refiere a un texto que el estudiante no ve")

    if NEGATIVA.search(d["pregunta"]):
        # Exigen tres opciones verdaderas y una falsa: salen mal casi siempre.
        fallos.append("pregunta negativa")

    return fallos


# ─────────────────────────────────────────────────────── autoverificacion

PROMPT_VERIFICAR = """Eres un revisor de examenes de medicina. Recibes un
FRAGMENTO, el TEMA, y una PREGUNTA de opcion multiple.

Responde SOLO con JSON:
{"respaldo": true/false, "distractor_cierto": true/false,
 "tema_correcto": true/false}

respaldo         : la opcion correcta esta afirmada en el fragmento?
distractor_cierto: alguna incorrecta es TAMBIEN cierta segun el fragmento?
tema_correcto    : la pregunta habla del TEMA indicado y no de otra enfermedad?"""


def construir_verificacion(tokenizer, d, fragmento, tema):
    """Arma el dialogo para que el propio modelo revise su pregunta."""
    contenido = (f"TEMA: {tema}\n\nFRAGMENTO:\n{fragmento}\n\n"
                 f"PREGUNTA: {d['pregunta']}\nCORRECTA: {d['correcta']}\n"
                 + "\n".join(f"INCORRECTA: {x}" for x in d["incorrectas"]))
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": PROMPT_VERIFICAR},
         {"role": "user", "content": contenido}],
        tokenize=False, add_generation_prompt=True)


def leer_verificacion(salida):
    """Interpreta la respuesta del verificador. Ante la duda, aprueba.

    Un verificador que rechaza por no haber sabido responder dejaria la
    aplicacion sin preguntas. Se prefiere dejar pasar una dudosa -que ademas
    lleva su fuente visible- a bloquear el examen entero.
    """
    v = parsear(salida)
    if not v:
        return [], "el verificador no respondio"
    # Un JSON valido que no trae NINGUNA de las tres claves no es una
    # verificacion: es otra cosa. Sin esta guarda, una respuesta con la forma
    # {"apto": true, "pregunta": ...} pasaba el parseo, no encontraba ningun
    # False, y se contaba como aprobacion.
    if not any(c in v for c in ("respaldo", "distractor_cierto", "tema_correcto")):
        return [], "el verificador respondio otra cosa"
    fallos = []
    if v.get("respaldo") is False:
        fallos.append("la respuesta correcta no esta en el fragmento")
    if v.get("distractor_cierto") is True:
        fallos.append("un distractor tambien es correcto")
    if v.get("tema_correcto") is False:
        fallos.append("la pregunta no corresponde al tema")
    return fallos, None
