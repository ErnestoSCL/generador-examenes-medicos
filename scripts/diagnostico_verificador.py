"""Por que el auto-verificador aprueba todo.

La sonda mostro que aprueba preguntas con defectos evidentes. Hay dos hipotesis:

  A) el prompt no le llega bien y devuelve basura o JSON con todo en true
  B) el LoRA lo especializo tanto en GENERAR preguntas que perdio la capacidad
     de seguir cualquier otra instruccion

Si es B, la solucion es gratis: `disable_adapter()` devuelve el modelo base, que
es un instruct de proposito general y deberia verificar mejor. Se prueban los
dos sobre los mismos casos con defectos conocidos.
"""
import sys
from pathlib import Path

sys.path.insert(0, "app")
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

import modelo
from validacion import construir_verificacion, leer_verificacion

crudo = pd.read_parquet("data/chunks.parquet")
nav = crudo[crudo.chunk_text.str.contains(
    "operable stage IIIC breast cancer, see", regex=False)].iloc[0].chunk_text
asma = crudo[crudo.question_focus == "Asthma"].iloc[0].chunk_text

CASOS = [
    ("navegacion", nav, "Cancer de mama", {
        "pregunta": "Cual es el tratamiento recomendado para el cancer de mama en etapas I a IIIA?",
        "correcta": "Ver opciones para cancer localizado o operable",
        "incorrectas": ["Ver opciones para cancer recidivante localizado",
                        "Ver opciones para cancer avanzado o inflamatorio",
                        "Ver opciones para cancer metastasico generalizado"]}),
    ("correcta inventada", asma, "Asma", {
        "pregunta": "Cual es el tratamiento de primera linea para el asma?",
        "correcta": "La administracion diaria de penicilina inyectable",
        "incorrectas": ["Los broncodilatadores de accion corta",
                        "Los corticoides inhalados",
                        "Los antagonistas de leucotrienos"]}),
    ("tema equivocado", asma, "Asma", {
        "pregunta": "Cual es el agente causal de la tuberculosis pulmonar?",
        "correcta": "Mycobacterium tuberculosis",
        "incorrectas": ["Streptococcus pneumoniae", "Haemophilus influenzae",
                        "Klebsiella pneumoniae"]}),
]

gen = modelo.Generador()


def verificar(d, fragmento, tema):
    prompt = construir_verificacion(gen.tok, d, fragmento, tema)
    salidas, _ = gen._generar_lote([prompt], max_new_tokens=90, do_sample=False,
                                   temperature=None, top_p=None, top_k=None)
    return salidas[0]


for etiqueta, usar_base in (("AFINADO (el que usa la app)", False), ("BASE (sin LoRA)", True)):
    print("=" * 92)
    print(etiqueta)
    print("=" * 92)
    for nombre, fragmento, tema, d in CASOS:
        if usar_base:
            with gen.model.disable_adapter():
                salida = verificar(d, fragmento, tema)
        else:
            salida = verificar(d, fragmento, tema)
        fallos, err = leer_verificacion(salida)
        estado = "RECHAZA: " + "; ".join(fallos) if fallos else ("aprueba" if not err else err)
        print("  %-20s %s" % (nombre, estado))
        print("     salida cruda: %s" % salida.strip()[:220].replace("\n", " "))
    print()
