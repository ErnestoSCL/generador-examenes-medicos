"""Compara estrategias de troceado para la tarea de generar preguntas.

El overlap de 64 se heredo del proyecto de RAG, donde servia para que un hecho
partido por la frontera apareciera entero en algun fragmento. Aqui la pregunta
es otra: cuanto contenido se conserva, y cuantos casi-duplicados entran al set
de entrenamiento.
"""
import re
from difflib import SequenceMatcher

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

CIERRES = (".", "!", "?")
FIN_ORACION = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def termina_en_oracion(t):
    t = t.rstrip().rstrip("'\")]")
    return bool(t) and t[-1] in CIERRES


def arranca_cortado(t):
    t = t.lstrip()
    return (not t) or t[0].islower() or t[0] in "-–—,;:)"


def recortar(texto):
    t = texto.lstrip()
    if arranca_cortado(t):
        m = FIN_ORACION.search(t)
        t = t[m.end():] if m else ""
    t = t.rstrip()
    if t and not termina_en_oracion(t):
        ultimo = max(t.rfind(c) for c in CIERRES)
        t = t[:ultimo + 1] if ultimo > 0 else ""
    return t


ESTRATEGIAS = {
    "A. overlap 64 (actual)": dict(chunk_size=512, chunk_overlap=64),
    "B. sin overlap": dict(chunk_size=512, chunk_overlap=0),
    "C. sin overlap, corta en oraciones": dict(
        chunk_size=512, chunk_overlap=0,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]),
}

clean = pd.read_parquet("../data/medqa_clean.parquet")
muestra = clean.sample(1200, random_state=42)
chars_originales = muestra.answer.str.len().sum()

print(f"muestra: {len(muestra):,} respuestas, {chars_originales:,} caracteres\n")
print(f"{'estrategia':36} {'chunks':>8} {'aptos':>8} {'% texto':>9} {'cortados':>9} {'casi-dup':>9}")
print("-" * 84)

for nombre, kwargs in ESTRATEGIAS.items():
    sp = RecursiveCharacterTextSplitter(length_function=len, **kwargs)
    crudos, por_doc = [], {}
    for fila in muestra.itertuples(index=False):
        partes = sp.split_text(fila.answer)
        crudos.extend(partes)
        por_doc[fila.question_id] = partes

    # cuantos arrancaban o terminaban cortados ANTES de recortar
    cortados = sum(arranca_cortado(c) or not termina_en_oracion(c) for c in crudos)

    recortados = [recortar(c) for c in crudos]
    aptos = [c for c in recortados if len(c) >= 200]
    chars_aptos = sum(len(c) for c in aptos)

    # casi-duplicados: pares consecutivos del MISMO documento que comparten texto
    casi_dup = 0
    for partes in por_doc.values():
        rec = [recortar(c) for c in partes]
        rec = [c for c in rec if len(c) >= 200]
        for a, b in zip(rec, rec[1:]):
            if SequenceMatcher(None, a, b).ratio() > 0.5:
                casi_dup += 1

    print(f"{nombre:36} {len(crudos):>8,} {len(aptos):>8,} "
          f"{chars_aptos / chars_originales * 100:>8.1f}% "
          f"{cortados / len(crudos) * 100:>8.1f}% {casi_dup:>9,}")

print("\n% texto  = caracteres conservados en los fragmentos aptos, sobre el original")
print("cortados = fragmentos que arrancaban o terminaban a mitad de oracion")
print("casi-dup = pares consecutivos del mismo documento con >50% de texto comun")
