"""Panel "bajo el capo" y juez externo.

Dos cosas que no son parte del producto pero si de la sustentacion:

**El panel comparativo.** Genera con el modelo base y con el afinado sobre el
mismo fragmento y los pone lado a lado. Va en un desplegable cerrado: el
estudiante no lo abre, se abre frente al jurado. Lo barato del asunto es que
`disable_adapter()` apaga el LoRA sin cargar un segundo modelo, asi que los dos
caben en 8 GB.

**El juez externo.** Un boton para pedirle a `gpt-4o` que revise una pregunta
concreta. Es una **ayuda opcional**, no parte del flujo: la aplicacion funciona
entera sin conexion y con costo cero. Existe porque los filtros de forma son
ciegos al contenido y la auto-verificacion la hace el mismo modelo de 4B que
genero la pregunta, con lo que comparte sus puntos ciegos. Cuando una pregunta
parece dudosa, un modelo mas capaz da una segunda opinion por menos de un
centavo.
"""
import json
import os
from pathlib import Path

RUTA_METRICAS = Path(__file__).parent / "adapter" / "metricas_evaluacion.json"

PROMPT_JUEZ = """Eres un revisor de examenes de medicina. Recibes un FRAGMENTO,
el TEMA, y una PREGUNTA de opcion multiple construida a partir de el.

Evalua con severidad y responde SOLO con JSON:
{"respaldo": true/false, "distractor_cierto": true/false,
 "tema_correcto": true/false, "veredicto": "una frase explicando el problema,
 o 'sin problemas' si esta bien"}

respaldo         : la opcion correcta esta afirmada explicitamente en el fragmento?
distractor_cierto: alguna incorrecta es TAMBIEN cierta segun el fragmento?
tema_correcto    : la pregunta habla del TEMA indicado y no de otra enfermedad?"""


def hay_api():
    """El juez externo solo aparece si hay clave configurada."""
    if os.environ.get("OPENAI_API_KEY"):
        return True
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env")
        return bool(os.environ.get("OPENAI_API_KEY"))
    except Exception:
        return False


def juzgar(pregunta, modelo="gpt-4o"):
    """Pide una segunda opinion a un modelo mas capaz. Devuelve (dict, error)."""
    try:
        from openai import OpenAI
        contenido = (
            f"TEMA: {pregunta['tema']}\n\nFRAGMENTO:\n{pregunta['fragmento']}\n\n"
            f"PREGUNTA: {pregunta['pregunta']}\nCORRECTA: {pregunta['correcta']}\n"
            + "\n".join(f"INCORRECTA: {x}" for x in pregunta["incorrectas"]))
        r = OpenAI().chat.completions.create(
            model=modelo, temperature=0,
            messages=[{"role": "system", "content": PROMPT_JUEZ},
                      {"role": "user", "content": contenido}])
        texto = r.choices[0].message.content.strip()
        if texto.startswith("```"):
            texto = texto.strip("`").removeprefix("json").strip()
        return json.loads(texto), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"[:160]


def leer_metricas():
    if RUTA_METRICAS.exists():
        return json.load(open(RUTA_METRICAS, encoding="utf-8"))
    return None


def ejemplos_fewshot(train_df, n=3):
    """Los ejemplos que necesita el modelo base para saber que formato producir.

    Es la comparacion honesta: sin ellos el base no sabria que devolver, y la
    diferencia medida seria trivial. Cuantos tokens cuesta ese andamiaje es
    justamente una de las metricas del panel.
    """
    salida = []
    for _, e in train_df.sample(n, random_state=42).iterrows():
        salida.append({
            "chunk_text": e["chunk_text"],
            "json": json.dumps({
                "apto": True, "pregunta": e["pregunta"], "correcta": e["correcta"],
                "incorrectas": list(e["incorrectas"]), "dificultad": e["dificultad"],
            }, ensure_ascii=False),
        })
    return salida
