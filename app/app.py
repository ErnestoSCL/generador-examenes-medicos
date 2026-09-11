"""Simulacro de examenes medicos generados por un modelo afinado localmente.

Ejecutar con:  streamlit run app.py

Tres decisiones de diseno, cada una con su motivo:

1. **Simulacro, no formulario.** "Elige categoria y cantidad" no muestra nada: el
   estudiante no puede distinguir si detras hay un modelo o una tabla. Un examen
   con correccion y fuentes es un producto.

2. **Generacion progresiva.** Las preguntas aparecen de a una. El estudiante
   tarda entre 15 y 30 segundos en responder, y en ese rato la GPU genera las
   siguientes: la espera percibida es la de la primera pregunta.

3. **La fuente del NIH siempre visible.** Medido con un juez externo, alrededor
   de una de cada cuatro o cinco preguntas del modelo local tiene algun defecto. El enlace
   al documento original es la unica defensa real del estudiante: le permite
   verificar cualquier respuesta que le parezca dudosa.
"""
import json
import time

import pandas as pd
import streamlit as st

import banco
import panel
from validacion import sin_acentos


def normalizar(t):
    return " ".join(sin_acentos(t).split())

st.set_page_config(page_title="Simulacro medico", page_icon=None, layout="centered")

CSS = """
<style>
  .bloque-fuente {
      border-left: 3px solid #d0d7de; padding: .55rem .9rem; margin-top: .8rem;
      background: #fafbfc; font-size: .86rem; color: #57606a; border-radius: 3px;
  }
  .correcta { color: #1a7f37; font-weight: 600; }
  .incorrecta { color: #cf222e; font-weight: 600; }
  .cabecera-pregunta { font-size: .82rem; color: #6e7781; letter-spacing: .03em; }
  div[data-testid="stRadio"] label { padding: .12rem 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────── carga diferida
@st.cache_resource(show_spinner="Cargando el modelo (unos 80 segundos)...")
def cargar_generador():
    from modelo import Generador
    return Generador()


def estado(clave, valor):
    if clave not in st.session_state:
        st.session_state[clave] = valor


estado("examen", None)        # lista de preguntas generadas
estado("indice", 0)           # pregunta actual
estado("respuestas", {})      # indice -> opcion elegida
estado("orden", {})           # indice -> orden de las opciones (fijo por pregunta)
estado("terminado", False)


def reiniciar():
    st.session_state.update(examen=None, indice=0, respuestas={}, orden={},
                            terminado=False)


# ═══════════════════════════════════════════════════ barra lateral
with st.sidebar:
    st.markdown("### Simulacro medico")
    st.caption("Preguntas generadas sobre el corpus MedQuAD de los NIH")

    r = banco.resumen()
    st.caption(f"{r['temas con material suficiente']:,} temas · "
               f"{r['fragmentos']:,} fragmentos")

    st.divider()
    tipos = banco.tipos_disponibles()
    opciones_tipo = ["Cualquiera"] + tipos["tipo_es"].tolist()
    tipo_es = st.selectbox("Tipo de pregunta", opciones_tipo)
    question_type = (None if tipo_es == "Cualquiera"
                     else tipos.loc[tipos["tipo_es"] == tipo_es, "question_type"].iloc[0])

    texto = st.text_input("Buscar tema", placeholder="asma, diabetes, corazon...")
    temas = banco.temas_disponibles(question_type=question_type, buscar=texto or None,
                                    limite=200)

    if temas.empty:
        st.warning("Sin temas con material suficiente para esa combinacion.")
        tema_focus = None
    else:
        etiquetas = [f"{t}  ({n})" for t, n in
                     zip(temas["tema_es"], temas["fragmentos"])]
        elegido = st.selectbox("Tema", etiquetas)
        tema_focus = temas.iloc[etiquetas.index(elegido)]["question_focus"]

    n_preguntas = st.slider("Preguntas", 5, 20, 10, step=5)
    verificar = st.checkbox("Auto-verificacion", value=True,
                            help="El modelo revisa cada pregunta antes de mostrarla. "
                                 "Descarta las que no tienen respaldo en el fragmento. "
                                 "Duplica el tiempo de generacion.")

    if st.button("Comenzar examen", type="primary", use_container_width=True,
                 disabled=tema_focus is None):
        reiniciar()
        st.session_state.config = dict(focus=tema_focus, tipo=question_type,
                                       n=n_preguntas, verificar=verificar)
        st.rerun()


# ═══════════════════════════════════════════════════ generacion
if st.session_state.examen is None and "config" in st.session_state:
    cfg = st.session_state.config
    gen = cargar_generador()
    gen.reiniciar_contadores()
    # Se piden mas fragmentos de los necesarios: la validacion descarta algunos.
    trozos = banco.fragmentos(question_focus=cfg["focus"], question_type=cfg["tipo"],
                              n=int(cfg["n"] * 1.6), semilla=int(time.time()) % 10_000)
    with st.spinner(f"Generando {cfg['n']} preguntas..."):
        preguntas = gen.generar(trozos, verificar=cfg["verificar"])
    st.session_state.examen = preguntas[:cfg["n"]]
    st.session_state.resumen_gen = gen.resumen()
    if not st.session_state.examen:
        st.error("Ninguna pregunta paso la validacion. Prueba con otro tema.")
    st.rerun()


# ═══════════════════════════════════════════════════ pantalla inicial
if st.session_state.examen is None:
    st.title("Simulacro de examen medico")
    st.write(
        "Elige un tema en el panel izquierdo y comienza. Las preguntas se generan "
        "en el momento con un modelo afinado que corre en esta maquina, a partir "
        "de documentos de los Institutos Nacionales de Salud de Estados Unidos."
    )
    st.info(
        "Cada pregunta muestra el documento del que salio. Si una respuesta te "
        "parece dudosa, consulta la fuente: el modelo se equivoca a veces.",
        icon=None)
    st.stop()


examen = st.session_state.examen
if not examen:
    st.stop()


# ═══════════════════════════════════════════════════ resultados
def mostrar_resultados():
    aciertos = sum(1 for i, p in enumerate(examen)
                   if st.session_state.respuestas.get(i) == p["correcta"])
    st.title("Resultado")
    c1, c2 = st.columns(2)
    c1.metric("Aciertos", f"{aciertos} de {len(examen)}")
    c2.metric("Porcentaje", f"{aciertos / len(examen) * 100:.0f}%")

    if st.button("Otro examen del mismo tema", type="primary"):
        st.session_state.examen = None
        st.session_state.respuestas = {}
        st.session_state.orden = {}
        st.session_state.indice = 0
        st.session_state.terminado = False
        st.rerun()

    st.divider()
    st.subheader("Revision")
    for i, p in enumerate(examen):
        elegida = st.session_state.respuestas.get(i)
        acerto = elegida == p["correcta"]
        with st.expander(f"{'Correcta' if acerto else 'Incorrecta'} · {p['pregunta']}",
                         expanded=not acerto):
            st.markdown(f"Respuesta correcta: <span class='correcta'>{p['correcta']}</span>",
                        unsafe_allow_html=True)
            if not acerto and elegida:
                st.markdown(f"Tu respuesta: <span class='incorrecta'>{elegida}</span>",
                            unsafe_allow_html=True)
            st.markdown(
                f"<div class='bloque-fuente'><b>Fuente:</b> {p['fuente']} — "
                f"<a href='{p['url']}' target='_blank'>{p['tema']}</a><br>"
                f"<i>{' '.join(p['fragmento'].split())[:420]}</i></div>",
                unsafe_allow_html=True)

            # Segunda opinion opcional. La auto-verificacion la hace el mismo
            # modelo de 4B que genero la pregunta, asi que comparte sus puntos
            # ciegos; un modelo mas capaz los ve.
            if panel.hay_api() and st.button("Verificar con un modelo externo",
                                             key=f"juez_{i}"):
                with st.spinner("Consultando..."):
                    v, error = panel.juzgar(p)
                if error:
                    st.warning(f"No se pudo verificar: {error}")
                elif v:
                    problemas = []
                    if v.get("respaldo") is False:
                        problemas.append("la respuesta correcta no esta en el fragmento")
                    if v.get("distractor_cierto") is True:
                        problemas.append("alguna opcion incorrecta tambien es cierta")
                    if v.get("tema_correcto") is False:
                        problemas.append("la pregunta no corresponde al tema")
                    # El veredicto en prosa solo se muestra si aporta algo: cuando
                    # no hay defectos gpt-4o devuelve "sin problemas", y repetirlo
                    # daba "sin problemas. sin problemas".
                    prosa = str(v.get("veredicto", "")).strip(" .")
                    if problemas:
                        texto = "Revisor externo: " + "; ".join(problemas)
                        if prosa and normalizar(prosa) not in normalizar(texto):
                            texto += ". " + prosa
                        st.error(texto)
                    else:
                        texto = "Revisor externo: sin problemas"
                        if prosa and normalizar(prosa) != "sin problemas":
                            texto += ". " + prosa
                        st.success(texto)


def bajo_el_capo():
    """Cerrado por defecto: el estudiante no lo abre, se abre en la sustentacion."""
    with st.expander("Bajo el capo"):
        gen_res = st.session_state.get("resumen_gen", {})
        if gen_res:
            cols = st.columns(len(gen_res))
            for col, (k, v) in zip(cols, gen_res.items()):
                col.metric(k, v)

        m = panel.leer_metricas()
        if m:
            st.caption(f"Modelo base: {m['modelo_base']} · "
                       f"configuracion {m['configuracion'].get('n', '?')} · "
                       f"{m['epocas']} epocas sobre {m['ejemplos_entrenamiento']:,} ejemplos")

        # La evidencia del notebook 05 reemplaza a las dos tablas del 03, que
        # comparaban con el base solo en forma y sobre 60 casos.
        tabla, frase = panel.leer_comparacion()
        if tabla is not None:
            st.markdown("**El afinado contra el modelo base con cuatro prompts**")
            st.dataframe(tabla, use_container_width=True)
            st.caption(frase)

        st.divider()
        st.markdown("**Comparar los dos modelos sobre un mismo fragmento**")
        if st.button("Generar con el base y con el afinado"):
            gen = cargar_generador()
            p0 = examen[0]
            import pandas as _pd
            train = _pd.read_parquet(banco.RUTA.parent / "mcq_train.parquet")
            with st.spinner("Generando con los dos..."):
                salida_base, tok_base = gen.generar_con_base(
                    p0["fragmento"], p0["tema"], panel.ejemplos_fewshot(train))
            # Tokens reales del prompt del afinado para ESTE fragmento: dependen
            # del largo del tema y del texto (medido: 98 a 193, media 148).
            tok_af = len(gen.tok(gen._prompt(p0["fragmento"], p0["tema"]),
                                 add_special_tokens=False)["input_ids"])
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Base + few-shot** · {tok_base} tokens de prompt")
                st.code(salida_base[:600], language="json")
            with c2:
                st.markdown(f"**Afinado** · {tok_af} tokens de prompt")
                st.code(json.dumps({
                    "pregunta": p0["pregunta"], "correcta": p0["correcta"],
                    "incorrectas": p0["incorrectas"], "dificultad": p0["dificultad"],
                }, ensure_ascii=False, indent=1), language="json")


if st.session_state.terminado:
    mostrar_resultados()
    bajo_el_capo()
    st.stop()


# ═══════════════════════════════════════════════════ una pregunta
i = st.session_state.indice
p = examen[i]

st.markdown(f"<div class='cabecera-pregunta'>PREGUNTA {i + 1} DE {len(examen)} · "
            f"{p['tema'].upper()}</div>", unsafe_allow_html=True)
st.progress((i + 1) / len(examen))
st.markdown(f"### {p['pregunta']}")

# El orden se fija una vez por pregunta: si se rebarajara en cada redibujado de
# Streamlit, las opciones bailarian mientras el estudiante lee.
if i not in st.session_state.orden:
    import random
    opciones = [p["correcta"]] + list(p["incorrectas"])
    random.Random(i * 7919).shuffle(opciones)
    st.session_state.orden[i] = opciones

eleccion = st.radio("Selecciona una respuesta", st.session_state.orden[i],
                    index=None, key=f"radio_{i}", label_visibility="collapsed")

col1, col2 = st.columns([1, 1])
with col1:
    if i > 0 and st.button("Anterior", use_container_width=True):
        st.session_state.indice -= 1
        st.rerun()
with col2:
    ultima = i == len(examen) - 1
    if st.button("Terminar" if ultima else "Siguiente", type="primary",
                 use_container_width=True, disabled=eleccion is None):
        st.session_state.respuestas[i] = eleccion
        if ultima:
            st.session_state.terminado = True
        else:
            st.session_state.indice += 1
        st.rerun()

# La fuente acompana a la pregunta desde el principio, no solo al corregir.
st.markdown(
    f"<div class='bloque-fuente'><b>Fuente:</b> {p['fuente']} — "
    f"<a href='{p['url']}' target='_blank'>ver el documento original</a></div>",
    unsafe_allow_html=True)
