"""Acceso a la base de fragmentos.

No es RAG: no hay embeddings ni busqueda por similitud. El usuario elige el tema
de una lista, asi que un filtro exacto sobre columnas alcanza y sobra. Una base
vectorial solo haria falta si se permitiera escribir el tema en texto libre.

`chunks.parquet` trae los nombres en dos idiomas: las columnas en ingles
(`question_type`, `question_focus`) vienen del corpus y se usan para filtrar; las
columnas `tipo_es` y `tema_es` son las que ve el estudiante. Se tradujeron una
sola vez en el notebook 02 y quedaron guardadas: la aplicacion carga una tabla
lista, no traduce en tiempo de ejecucion.
"""
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd

RUTA = Path(__file__).parent.parent / "data" / "chunks.parquet"

# Con un solo fragmento no se puede armar un examen ni ofrecer variedad.
MIN_FRAGMENTOS_POR_TEMA = 3

# ─────────────────────────────────────────── fragmentos que no ensenan nada
#
# Un fragmento puede estar limpio y aun asi no servir: si lo unico que hace es
# mandar a otro documento, el modelo genera una pregunta cuyas opciones son
# etiquetas de navegacion. Caso real medido en la aplicacion:
#
#   "For treatment options for stage I... see Early, Localized, or Operable
#    Breast Cancer. For treatment options for stage IIIB... see Locally
#    Advanced or Inflammatory Breast Cancer."
#
#   -> "Cual es el tratamiento recomendado para el cancer de mama en etapas I
#       a IIIA?"  con la respuesta "Ver opciones para cancer localizado".
#
# Los patrones son deliberadamente estrechos. Un "refer to" suelto NO alcanza:
# "researchers refer to this form as type 1" es contenido legitimo, y filtrar
# por ahi descartaba fragmentos buenos. Se exige que la oracion entera sea
# navegacion, y que al menos un tercio de las oraciones lo sean.
NAVEGACION = re.compile(
    r"see (the )?(PDQ |NCI )?summar|"
    r"for (treatment options|more information)[^.]{0,200}\bsee\b|"
    r"click here|use the advanced search|"
    r"(this|these) (link|resource)s? (can |will )?(help|provide)|"
    r"following (online )?resources", re.I)

# Directorios: telefonos, direcciones postales, listas de URLs.
DIRECTORIO = re.compile(
    r"(www\.|https?://|1-\d{3}-\d{3}|\(\d{3}\)\s?\d{3}|\b[A-Z]{2}\s\d{5}\b)")

FRAC_NAVEGACION = 0.34
MIN_SENALES_DIRECTORIO = 2


def es_util(texto):
    """False si el fragmento solo remite a otro sitio en vez de ensenar."""
    if len(DIRECTORIO.findall(texto)) >= MIN_SENALES_DIRECTORIO:
        return False
    oraciones = [o for o in re.split(r"(?<=[.!?])\s+", texto) if o.strip()]
    if not oraciones:
        return False
    navegacion = sum(bool(NAVEGACION.search(o)) for o in oraciones)
    return navegacion / len(oraciones) < FRAC_NAVEGACION


def sin_acentos(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s).lower())
                   if unicodedata.category(c) != "Mn")


@lru_cache(maxsize=1)
def cargar():
    df = pd.read_parquet(RUTA)
    # Se descartan al cargar: la aplicacion no debe servirlos nunca, y los
    # conteos de la barra lateral reflejan lo que de verdad hay disponible.
    df = df[df["chunk_text"].map(es_util)].reset_index(drop=True)
    df["busqueda"] = (df["tema_es"].astype(str) + " " +
                      df["question_focus"].astype(str)).map(sin_acentos)
    return df


def tipos_disponibles():
    """Tipos de pregunta con su nombre en espanol, ordenados por cobertura."""
    df = cargar()
    conteo = df.groupby(["tipo_es", "question_type"]).size().reset_index(name="fragmentos")
    return conteo.sort_values("fragmentos", ascending=False)


def temas_disponibles(question_type=None, buscar=None, limite=300):
    """Temas que tienen material suficiente para un examen.

    `buscar` filtra por el nombre en espanol o en ingles, sin acentos: escribir
    "asma" encuentra los fragmentos de `Asthma`.
    """
    df = cargar()
    if question_type:
        df = df[df["question_type"] == question_type]

    conteo = (df.groupby(["tema_es", "question_focus"]).size()
                .reset_index(name="fragmentos"))
    conteo = conteo[conteo["fragmentos"] >= MIN_FRAGMENTOS_POR_TEMA]

    if buscar:
        clave = sin_acentos(buscar)
        mascara = (conteo["tema_es"].map(sin_acentos).str.contains(clave, regex=False) |
                   conteo["question_focus"].map(sin_acentos).str.contains(clave, regex=False))
        conteo = conteo[mascara]

    return conteo.sort_values("fragmentos", ascending=False).head(limite)


def fragmentos(question_focus=None, question_type=None, n=10, semilla=None):
    """Devuelve n fragmentos del tema pedido, sin repetir respuesta si se puede.

    Se toma como mucho un fragmento por respuesta original mientras haya
    suficientes: dos fragmentos de la misma respuesta darian preguntas sobre lo
    mismo. Solo si no alcanzan se completa con los que quedaron.
    """
    df = cargar()
    if question_focus:
        df = df[df["question_focus"] == question_focus]
    if question_type:
        df = df[df["question_type"] == question_type]
    if df.empty:
        return []

    barajado = df.sample(frac=1, random_state=semilla)
    variados = barajado.drop_duplicates(subset="question_id", keep="first")
    elegidos = variados.head(n)
    if len(elegidos) < n:
        resto = barajado.drop(index=elegidos.index)
        faltan = min(n - len(elegidos), len(resto))
        if faltan:
            elegidos = pd.concat([elegidos, resto.head(faltan)])

    columnas = ["chunk_uid", "chunk_text", "tema_es", "tipo_es",
                "question_focus", "question_type", "document_source", "document_url"]
    return elegidos[columnas].to_dict("records")


def resumen():
    df = cargar()
    return {
        "fragmentos": len(df),
        "temas": df["question_focus"].nunique(),
        "temas con material suficiente": len(temas_disponibles(limite=10**6)),
        "tipos de pregunta": df["question_type"].nunique(),
    }
