"""Une los notebooks 01 a 05 en uno solo, pensado para ejecutarse en Colab.

Copia las celdas y sus salidas tal como quedaron en las ejecuciones completas en
local, para que el notebook se pueda leer sin ejecutarlo. Encima agrega:

  - la configuracion (prueba rapida, uso de la API),
  - la preparacion del entorno: clona el repositorio y se ubica en notebooks/,
    en Colab o en un clon temporal si se ejecuta en local,
  - el cambio automatico a float16 en GPUs sin bfloat16 (la T4 de Colab),
  - la liberacion de la GPU entre partes.

Los cambios al codigo original son pocos y puntuales; cada uno se comprueba con
un assert para que el script falle en vez de producir un notebook a medias.
"""
import copy
import io
import json

ORIGEN = {
    1: ("01_limpieza", "Limpieza del corpus MedQuAD"),
    2: ("02_dataset_mcq", "Dataset sintético de preguntas"),
    3: ("03_finetuning", "Fine-tuning con LoRA"),
    4: ("04_parametros_generacion", "Parámetros de generación"),
    5: ("05_afinado_contra_base", "El afinado contra el modelo base"),
}
SALIDA = "notebooks/00_proyecto_completo_colab.ipynb"
COLAB = ("https://colab.research.google.com/github/ErnestoSCL/generador-examenes-medicos/"
         "blob/main/notebooks/00_proyecto_completo_colab.ipynb")

celdas = []


def md(texto):
    celdas.append({"cell_type": "markdown", "metadata": {}, "source": texto.strip() + "\n"})


def code(texto):
    celdas.append({"cell_type": "code", "metadata": {}, "outputs": [],
                   "execution_count": None, "source": texto.strip() + "\n"})


# ═══════════════════════════════════════════════════════ encabezado
md(f"""
# Generador de exámenes médicos — el proyecto completo

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})

Un modelo de 4.000 millones de parámetros, afinado con LoRA, que genera preguntas
de opción múltiple en español a partir del corpus MedQuAD de los Institutos
Nacionales de Salud de EE. UU. Este notebook reúne los cinco del proyecto en un
solo recorrido:

| Parte | Qué hace | Notebook original |
|---|---|---|
| 1 | Limpieza del corpus: de 47,441 a 14,528 filas | 01 |
| 2 | Fragmentación y dataset sintético generado por `gpt-4o-mini` | 02 |
| 3 | Barrido de 6 configuraciones de LoRA y entrenamiento final | 03 |
| 4 | Barrido de parámetros de generación | 04 |
| 5 | El afinado contra el modelo base con cuatro prompts | 05 |

## Se puede leer sin ejecutarlo

**Las salidas que ves están guardadas.** Provienen de la ejecución completa de
los notebooks 01 a 05 en una RTX 5070 Ti de 16 GB. El código es el mismo, con
tres añadidos: la preparación del entorno (abajo), el cambio automático a
float16 en GPUs sin bfloat16, y el modo de prueba rápida.

Algunas salidas mencionan rutas locales (`D:\\...`): son de esa ejecución. La
celda del entrenamiento final muestra que se reanudó desde un checkpoint porque
el entrenamiento ya había terminado antes (48 minutos). Si ejecutas el notebook,
todas las salidas se reemplazan por las de tu corrida.

## Cómo ejecutarlo en Colab

1. **Entorno de ejecución → Cambiar tipo de entorno → GPU.** La T4 gratuita
   alcanza para la prueba rápida; para reproducir todo conviene una L4 o A100.
2. **Opcional:** un secreto `OPENAI_API_KEY` (icono de la llave, a la
   izquierda) y `USAR_API = True` para volver a correr los jueces de `gpt-4o`.
3. Revisa la configuración y usa **Entorno de ejecución → Ejecutar todas**.

El repositorio es público: Colab lo clona sin credenciales.

| Configuración | Qué hace | Tiempo | API |
|---|---|---|---|
| `PRUEBA_RAPIDA = True` *(por defecto)* | recorta datos, pasos y casos: comprueba que todo corre | minutos | nada |
| `PRUEBA_RAPIDA = False` | reproduce el proyecto completo | ~4 h en la RTX 5070 Ti; más en Colab | nada |
| `USAR_API = True` | además vuelve a correr los jueces | — | ~USD 0.3 en prueba, ~USD 3 completo |

**Con `USAR_API = False` no se llama a la API.** El dataset se reconstruye con
las 5,250 respuestas originales de `gpt-4o-mini`, guardadas en
`data/mcq_crudo.jsonl`, así que es el mismo dataset con el que se entrenó. Las
celdas de los jueces se omiten y lo avisan.

La aplicación Streamlit no se ejecuta aquí. Al terminar la parte 3, el
adaptador queda comprimido en `/content/adapter.zip`: descomprimido en
`app/adapter/`, la aplicación lo usa en local.
""")

md("## Configuración")
code(r'''
PRUEBA_RAPIDA = True   # True: recorta todo para comprobar que corre. False: reproduce completo (horas).
USAR_API = False       # True: vuelve a correr los jueces de gpt-4o (necesita el secreto OPENAI_API_KEY).
''')

md("""
## Preparación del entorno

Clona el repositorio y se ubica en `notebooks/`, de modo que todas las rutas del
proyecto (`../data`, `../app/adapter`) funcionan igual que en local. En local
hace lo mismo sobre un clon temporal: así una prueba no pisa los datos ni el
adaptador del proyecto.
""")
code(r'''
import os, shutil, subprocess, sys, tempfile
from pathlib import Path

EN_COLAB = "google.colab" in sys.modules
REPO = "github.com/ErnestoSCL/generador-examenes-medicos.git"


def secreto(nombre):
    """Lee un secreto de Colab; None si no existe o no se autorizo."""
    try:
        from google.colab import userdata
        return userdata.get(nombre)
    except Exception:
        return None


if EN_COLAB:
    TRABAJO = Path("/content/generador-examenes-medicos")
    if not TRABAJO.exists():
        token = secreto("GITHUB_TOKEN")
        url = f"https://{token}@{REPO}" if token else f"https://{REPO}"
        r = subprocess.run(["git", "clone", "-q", "--depth", "1", url, str(TRABAJO)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("No se pudo clonar el repositorio. Si es privado, agrega el "
                               "secreto GITHUB_TOKEN y autoriza su uso en este notebook.")
    # Solo la pila de Hugging Face: torch, numpy y pandas quedan los de Colab,
    # para no obligar a reiniciar el entorno.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=5",
                    "peft", "accelerate", "datasets", "langchain-text-splitters",
                    "openai", "python-dotenv"], check=True)
    if secreto("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = secreto("OPENAI_API_KEY")
else:
    RAIZ = Path.cwd().parent
    from dotenv import load_dotenv
    load_dotenv(RAIZ / ".env")
    TRABAJO = Path(tempfile.gettempdir()) / "generador_examenes_trabajo"
    shutil.rmtree(TRABAJO, ignore_errors=True)
    subprocess.run(["git", "clone", "-q", str(RAIZ), str(TRABAJO)], check=True)

os.chdir(TRABAJO / "notebooks")
HAY_API = (USAR_API and bool(os.environ.get("OPENAI_API_KEY"))
           and os.environ.get("FORZAR_SIN_API") != "1")
os.environ["PRUEBA_RAPIDA"] = "1" if PRUEBA_RAPIDA else "0"
print("entorno:", "Colab" if EN_COLAB else "local", "| trabajando en:", TRABAJO)
print("modo:", "prueba rapida" if PRUEBA_RAPIDA else "completo")
print("API de OpenAI:", "si" if HAY_API else "no: el dataset sale de la cache y los jueces se omiten")
''')

code(r'''
import torch

assert torch.cuda.is_available(), ("Hace falta GPU. En Colab: Entorno de ejecucion > "
                                   "Cambiar tipo de entorno > GPU.")
# bfloat16 necesita una GPU Ampere o posterior (L4, A100, RTX 30xx en adelante).
# La T4 gratuita de Colab no lo tiene: ahi se trabaja en float16.
BF16 = (torch.cuda.get_device_capability(0)[0] >= 8
        and os.environ.get("FORZAR_FP16") != "1")
DTYPE = torch.bfloat16 if BF16 else torch.float16
props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name} | {props.total_memory/1e9:.0f} GB | "
      f"precision: {'bfloat16' if BF16 else 'float16'}")
''')

code(r'''
# Las celdas que llaman a la API empiezan con %%con_api: si no hay API, se
# omiten con un aviso en vez de fallar, y la ejecucion sigue.
from IPython.core.magic import register_cell_magic


@register_cell_magic
def con_api(linea, celda):
    if HAY_API:
        get_ipython().run_cell(celda).raise_error()
    else:
        print("[sin API] celda omitida: la salida guardada es la de la ejecucion original")
''')


# ═══════════════════════════════════════════════════════ transformaciones
def cambiar(src, viejo, nuevo, donde):
    assert src.count(viejo) == 1, f"{donde}: no se encontro una vez -> {viejo[:60]!r}"
    return src.replace(viejo, nuevo)


LIBERAR = r'''
# Libera la GPU antes de cargar el modelo de nuevo: la parte anterior lo dejo en memoria.
import gc
for _nombre in ("model", "trainer", "base"):
    globals().pop(_nombre, None)
gc.collect()
torch.cuda.empty_cache()
print(f"memoria GPU ocupada tras liberar: {torch.cuda.memory_allocated()/1e9:.2f} GB")
'''

NB03_CELDA_RUTAS = r'''
import gc, json, os, random, re, time, unicodedata
from pathlib import Path

# Rutas relativas a notebooks/: la preparacion del entorno ya se ubico ahi.
DATA = Path("../data")
SALIDA = Path("../app/adapter")
assert (DATA / "mcq_train.parquet").exists(), f"falta mcq_train.parquet en {DATA}: ejecuta la parte 2"
print("datos:", DATA.resolve())
'''

PILOTO_VIEJO = "generar(piloto, SALIDA)"
PILOTO_NUEVO = r'''if HAY_API:
    generar(piloto, SALIDA)
else:
    # Sin API: el piloto se toma de la generacion completa guardada, que uso el
    # mismo prompt sobre los mismos fragmentos. No se gasta nada.
    ids_piloto = set(piloto["chunk_uid"])
    with open(DATA / "mcq_crudo.jsonl", encoding="utf-8") as fe:
        guardadas = [l for l in fe if l.strip() and json.loads(l)["chunk_uid"] in ids_piloto]
    with open(SALIDA, "w", encoding="utf-8") as fs:
        fs.writelines(guardadas)
    print(f"piloto tomado de la cache: {len(guardadas)} respuestas, sin llamar a la API")'''

GENERACION_VIEJA = 'SALIDA_COMPLETA = DATA / "mcq_crudo.jsonl"\ngenerar(muestra, SALIDA_COMPLETA)'
GENERACION_NUEVA = r'''SALIDA_COMPLETA = DATA / "mcq_crudo.jsonl"

# La cache trae las 5,250 respuestas originales: si la muestra coincide, no se
# llama a la API y el dataset es el mismo con el que se entreno.
en_cache = set()
if SALIDA_COMPLETA.exists():
    for l in open(SALIDA_COMPLETA, encoding="utf-8"):
        if l.strip() and not json.loads(l).get("error"):
            en_cache.add(json.loads(l)["chunk_uid"])
faltan = len(set(muestra["chunk_uid"]) - en_cache)
print(f"respuestas en cache: {len(en_cache):,} | faltan por generar: {faltan:,}")
if faltan and not HAY_API:
    raise RuntimeError(f"Faltan {faltan} respuestas en la cache y la API esta desactivada. "
                       "Activa USAR_API para generarlas (cuesta dinero y cambia el dataset).")
generar(muestra, SALIDA_COMPLETA)'''

FP16_ANCLA = '''        task_type="CAUSAL_LM", target_modules=cfg["mods"]))
'''
FP16_NUEVO = '''        task_type="CAUSAL_LM", target_modules=cfg["mods"]))
    if not BF16:
        # En float16 (GPU T4) los pesos entrenables del LoRA van en float32: es
        # lo que exige el escalado de gradientes de la precision mixta.
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
'''

ZIP_VIEJO = '''if EN_COLAB:
    !cd /content && zip -qr adapter.zip adapter
    print("descarga adapter.zip y descomprimelo junto a app.py")'''
ZIP_NUEVO = '''if EN_COLAB:
    import shutil
    comprimido = shutil.make_archive("/content/adapter", "zip", SALIDA)
    print(f"adaptador comprimido en {comprimido}: descargalo desde el panel de archivos")
    print("y descomprimelo en app/adapter/ para usar la aplicacion en local")'''


def transformar(parte, i, src):
    donde = f"parte {parte}, celda {i}"
    # generales: la precision la decide la GPU disponible
    src = src.replace("dtype=torch.bfloat16", "dtype=DTYPE")
    src = src.replace("bf16=True,", "bf16=BF16, fp16=not BF16,")

    if parte == 1 and src.lstrip().startswith("# Dependencias requeridas"):
        return "# Las dependencias se instalan en la preparacion del entorno, al principio.\n"

    if parte == 2:
        if "client = OpenAI()" in src:
            src = cambiar(src, "client = OpenAI()",
                          "client = OpenAI() if HAY_API else None", donde)
        if src.rstrip().endswith(PILOTO_VIEJO) and "def pedir_una" in src:
            src = src.rstrip()[: -len(PILOTO_VIEJO)] + PILOTO_NUEVO + "\n"
        if GENERACION_VIEJA in src:
            src = cambiar(src, GENERACION_VIEJA, GENERACION_NUEVA, donde)

    if parte == 3:
        if "EN_COLAB = " in src and "SALIDA = Path" in src:
            return NB03_CELDA_RUTAS.strip() + "\n"
        if 'PRUEBA = os.environ.get("PRUEBA_RAPIDA") == "1"' in src:
            src = cambiar(src, 'PRUEBA = os.environ.get("PRUEBA_RAPIDA") == "1"',
                          "PRUEBA = PRUEBA_RAPIDA      # se fija en la configuracion", donde)
        if 'PARCIAL = DATA / "barrido_parcial.jsonl"' in src:
            src = cambiar(src, 'PARCIAL = DATA / "barrido_parcial.jsonl"',
                          'PARCIAL = DATA / ("barrido_parcial_prueba.jsonl" if PRUEBA '
                          'else "barrido_parcial.jsonl")', donde)
        if "def entrenar(" in src:
            src = cambiar(src, FP16_ANCLA, FP16_NUEVO, donde)
        if "EPOCAS = 1 if PRUEBA else 2" in src:
            # En prueba, 1 epoca sobre 40 ejemplos eran 5 pasos, todos de
            # calentamiento: el adaptador no aprendia el formato (0 de 16
            # estructuras). 4 epocas son 20 pasos, como el barrido.
            src = cambiar(src, "EPOCAS = 1 if PRUEBA else 2", "EPOCAS = 4 if PRUEBA else 2", donde)
        if 'CKPTS = SALIDA.parent.parent / "checkpoints"' in src:
            src = cambiar(src, 'CKPTS = SALIDA.parent.parent / "checkpoints"',
                          'CKPTS = SALIDA.parent.parent / ("checkpoints_prueba" if PRUEBA '
                          'else "checkpoints")', donde)
        if "OpenAI()" in src or "veredictos" in src:
            src = "%%con_api\n" + src
        if '"maestro_vs_alumno": maestro_alumno.to_dict(),' in src:
            src = cambiar(src, '"maestro_vs_alumno": maestro_alumno.to_dict(),',
                          '"maestro_vs_alumno": maestro_alumno.to_dict() if HAY_API else None,',
                          donde)
            src = cambiar(src, ZIP_VIEJO, ZIP_NUEVO, donde)

    if parte == 4 and "N_CASOS = 20" in src:
        src = cambiar(src, "N_CASOS = 20", "N_CASOS = 4 if PRUEBA_RAPIDA else 20", donde)

    if parte == 5:
        if "test_df.sample(150, random_state=13)" in src:
            src = cambiar(src, "test_df.sample(150, random_state=13)",
                          "test_df.sample(16 if PRUEBA_RAPIDA else 150, random_state=13)", donde)
            assert src.count('}/150"') == 2, donde
            src = src.replace('}/150"', '}/{len(fragmentos)}"')
        if "test_df.sample(20, random_state=42)" in src:
            src = cambiar(src, "test_df.sample(20, random_state=42)",
                          "test_df.sample(4 if PRUEBA_RAPIDA else 20, random_state=42)", donde)
        if ("OpenAI()" in src or "veredictos" in src
                or "comparacion_prompts_generaciones.json" in src):
            src = "%%con_api\n" + src
    return src


# ═══════════════════════════════════════════════════════ las cinco partes
conteo_api = 0
for parte, (nombre, titulo) in ORIGEN.items():
    nb = json.load(io.open(f"notebooks/{nombre}.ipynb", encoding="utf-8"))
    md(f"# Parte {parte} — {titulo}\n\n*Notebook original: `{nombre}.ipynb`.*")
    if parte in (4, 5):
        code(LIBERAR)
    for i, c in enumerate(nb["cells"]):
        c = copy.deepcopy(c)
        c["source"] = "".join(c["source"])
        c.pop("id", None)
        if c["cell_type"] == "code":
            c["source"] = transformar(parte, i, c["source"])
            conteo_api += c["source"].startswith("%%con_api")
            c["execution_count"] = None
            for o in c.get("outputs", []):
                if "execution_count" in o:
                    o["execution_count"] = None
        celdas.append(c)

# Comprobaciones de que cada cambio puntual se aplico donde debia.
fuente = "\n".join(c["source"] for c in celdas if c["cell_type"] == "code")
for esperado in ["dtype=DTYPE", "bf16=BF16, fp16=not BF16,", "if not BF16:",
                 "client = OpenAI() if HAY_API else None", "piloto tomado de la cache",
                 "faltan por generar", "PRUEBA = PRUEBA_RAPIDA", "barrido_parcial_prueba",
                 "checkpoints_prueba", "make_archive", "N_CASOS = 4 if PRUEBA_RAPIDA",
                 "EPOCAS = 4 if PRUEBA else 2",
                 "16 if PRUEBA_RAPIDA else 150", "4 if PRUEBA_RAPIDA else 20"]:
    assert esperado in fuente, f"no se aplico: {esperado}"
for prohibido in ["dtype=torch.bfloat16", "bf16=True", "!cd /content", "!pip",
                  'os.environ.get("PRUEBA_RAPIDA") == "1"']:
    assert prohibido not in fuente, f"quedo sin cambiar: {prohibido}"

for n, c in enumerate(celdas):
    c["id"] = f"c{n:03d}"
    c["source"] = c["source"].splitlines(keepends=True)

nb = {"cells": celdas, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}}}
json.dump(nb, io.open(SALIDA, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
con_salida = sum(1 for c in celdas if c["cell_type"] == "code" and c.get("outputs"))
print(f"{SALIDA}: {len(celdas)} celdas, {con_salida} con salida guardada, "
      f"{conteo_api} dependen de la API")
