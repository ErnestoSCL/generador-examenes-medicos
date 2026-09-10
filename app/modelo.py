"""Carga del modelo y generacion de preguntas.

Un solo modelo en memoria (~8 GB) sirve para todo: con el adaptador puesto es el
modelo afinado, y `disable_adapter()` lo apaga para obtener el modelo base. Eso
es lo que hace viable el panel comparativo sin cargar 16 GB.

La generacion va **por lotes**: medido, 199.8 tokens/s efectivos con lote de 8
contra 23.4 de a una, o sea 8.5x. Es lo que convierte un examen de 50 preguntas
de tres minutos a cuarenta segundos.
"""
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from validacion import (construir_verificacion, leer_verificacion, parsear,
                        revisar_forma)

MODELO_BASE = "Qwen/Qwen3-4B-Instruct-2507"
RUTA_ADAPTADOR = Path(__file__).parent / "adapter"

INSTRUCCION = (
    "Eres un docente de medicina. A partir del FRAGMENTO escribe UNA pregunta "
    "de opcion multiple en espanol neutro. Responde SOLO con JSON."
)

# Elegidos midiendo (ver notebook 03 y barrido_generacion): se necesita muestreo
# -no greedy- porque el boton "otro examen del mismo tema" exige que el mismo
# fragmento produzca preguntas distintas. Con greedy siempre saldria la misma.
GENERACION = dict(
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    # Medido en el notebook 04 sobre 420 generaciones: el formato NO se rompe en
    # ningun punto del barrido -las siete configuraciones dan 60/60 de JSON
    # valido, incluida temperatura 1.0-. Lo que cambia es la variedad, medida
    # como preguntas distintas al regenerar tres veces el mismo fragmento:
    #
    #   greedy   1.00/3     temp 0.7          2.30/3
    #   temp 0.3 1.95/3     temp 0.7 + rep 1.1 2.45/3
    #   temp 1.0 2.70/3     temp 0.7 + rep 1.2 2.60/3
    #
    # Se sube de 1.05 a 1.1 porque 1.05 no estaba en la grilla medida y 1.1 si.
    # No se va a temperatura 1.0, que da mas variedad, porque el barrido mide
    # formato y variedad pero NO correccion factual: subir la temperatura suele
    # aumentar la invencion, y eso aqui no esta medido.
    repetition_penalty=1.1,
    max_new_tokens=300,
)


class Generador:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(RUTA_ADAPTADOR)
        self.tok.padding_side = "left"          # obligatorio para lotes
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            MODELO_BASE, dtype=torch.bfloat16, device_map="cuda")
        self.model = PeftModel.from_pretrained(base, RUTA_ADAPTADOR)
        self.model.eval()
        self.reiniciar_contadores()

    def reiniciar_contadores(self):
        """El panel informa sobre UN examen, no sobre la vida del proceso.

        El generador se cachea con st.cache_resource, asi que sobrevive entre
        examenes y entre sesiones. Sin este reinicio los numeros del panel se
        acumulaban: un examen de 10 preguntas mostraba 26 generadas porque
        arrastraba las de un examen anterior.
        """
        self.contadores = {"generadas": 0, "rechazadas_forma": 0,
                           "rechazadas_verificacion": 0, "segundos": 0.0}

    # ------------------------------------------------------------ interno
    def _generar_lote(self, prompts, **kw):
        opciones = {**GENERACION, **kw}
        ids = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
        t0 = time.time()
        with torch.no_grad():
            salida = self.model.generate(**ids, pad_token_id=self.tok.pad_token_id,
                                         **opciones)
        self.contadores["segundos"] += time.time() - t0
        return [self.tok.decode(f[ids.input_ids.shape[1]:], skip_special_tokens=True)
                for f in salida], ids.input_ids.shape[1]

    def _prompt(self, fragmento, tema):
        # El tema va en el prompt porque el modelo confunde nombres parecidos
        # cuando no lo tiene delante: genero "cisticercosis" leyendo un texto de
        # equinococosis, y "ratas" donde el fragmento decia "ratones".
        return self.tok.apply_chat_template(
            [{"role": "system", "content": INSTRUCCION},
             {"role": "user", "content": f"FRAGMENTO (tema: {tema}):\n{fragmento}"}],
            tokenize=False, add_generation_prompt=True)

    # ------------------------------------------------------------ publico
    def generar(self, fragmentos, verificar=True, reintentos=1):
        """Genera una pregunta por fragmento, validada.

        `fragmentos` son filas con chunk_text, tema_es y document_url.
        Devuelve solo las que pasan; las rechazadas se reintentan una vez.
        """
        pendientes = list(fragmentos)
        aprobadas = []

        # Las cuentas se llevan en un diccionario LOCAL y recien al final se
        # publican en self.contadores. Antes se acumulaban directo sobre el
        # objeto, que Streamlit cachea con st.cache_resource y comparte entre
        # ejecuciones: si el usuario tocaba un control mientras se generaba,
        # Streamlit relanzaba el script pero la generacion anterior seguia viva
        # en su hilo, y las dos sumaban sobre el mismo contador. El panel
        # llego a informar 54 preguntas generadas para un examen de 10, cuando
        # el techo por diseno son 32 (16 pedidas + 16 reintentos).
        c = {"generadas": 0, "rechazadas_forma": 0,
             "rechazadas_verificacion": 0, "segundos": 0.0}

        for intento in range(reintentos + 1):
            if not pendientes:
                break
            prompts = [self._prompt(f["chunk_text"], f["tema_es"]) for f in pendientes]
            t0 = time.time()
            salidas, _ = self._generar_lote(prompts)
            c["segundos"] += time.time() - t0
            c["generadas"] += len(salidas)

            rechazadas = []
            candidatas = []
            for fragmento, salida in zip(pendientes, salidas):
                d = parsear(salida)
                fallos = revisar_forma(d, fragmento["chunk_text"])
                if fallos:
                    c["rechazadas_forma"] += 1
                    rechazadas.append(fragmento)
                else:
                    candidatas.append((fragmento, d))

            # Segundo nivel: el modelo revisa su propia pregunta. Las reglas de
            # forma son ciegas al contenido; esto ve si la correcta esta
            # respaldada y si algun distractor tambien es cierto.
            if verificar and candidatas:
                prompts_v = [construir_verificacion(self.tok, d, f["chunk_text"], f["tema_es"])
                             for f, d in candidatas]
                # Verifica el modelo BASE, no el afinado. Medido: ante el prompt
                # de verificacion el afinado no verifica, REGENERA -devuelve
                # {"apto": true, "pregunta": ...}-, y como esa respuesta no trae
                # las claves esperadas, se aprobaba todo. Sobre tres preguntas
                # con defectos deliberados el afinado aprobo las tres y el base
                # rechazo dos. Apagar el LoRA no cuesta memoria ni carga.
                with self.model.disable_adapter():
                    salidas_v, _ = self._generar_lote(prompts_v, max_new_tokens=90,
                                                      do_sample=False, temperature=None,
                                                      top_p=None, top_k=None)
                for (fragmento, d), sv in zip(candidatas, salidas_v):
                    fallos, _ = leer_verificacion(sv)
                    if fallos:
                        c["rechazadas_verificacion"] += 1
                        rechazadas.append(fragmento)
                    else:
                        aprobadas.append(self._empaquetar(fragmento, d))
            else:
                aprobadas.extend(self._empaquetar(f, d) for f, d in candidatas)

            pendientes = rechazadas

        self.contadores = c
        return aprobadas

    @staticmethod
    def _empaquetar(fragmento, d):
        return {
            "pregunta": d["pregunta"],
            "correcta": d["correcta"],
            "incorrectas": list(d["incorrectas"]),
            "dificultad": d.get("dificultad", "media"),
            # La fuente viaja SIEMPRE con la pregunta: es lo que permite al
            # estudiante verificar una respuesta que le parezca dudosa, y la
            # unica defensa real contra el ~15% de preguntas con algun defecto.
            "tema": fragmento["tema_es"],
            "url": fragmento["document_url"],
            "fuente": fragmento["document_source"],
            "fragmento": fragmento["chunk_text"],
        }

    def generar_con_base(self, fragmento, tema, ejemplos_fewshot):
        """Genera con el modelo SIN afinar, para el panel comparativo."""
        msgs = [{"role": "system", "content": INSTRUCCION}]
        for e in ejemplos_fewshot:
            msgs.append({"role": "user", "content": "FRAGMENTO:\n" + e["chunk_text"]})
            msgs.append({"role": "assistant", "content": e["json"]})
        msgs.append({"role": "user", "content": f"FRAGMENTO (tema: {tema}):\n{fragmento}"})
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        with self.model.disable_adapter():        # apaga el LoRA temporalmente
            salidas, tok_prompt = self._generar_lote([prompt])
        return salidas[0], tok_prompt

    def resumen(self):
        c = self.contadores
        total = max(c["generadas"], 1)
        return {
            "preguntas generadas": c["generadas"],
            "rechazadas por forma": f"{c['rechazadas_forma']} ({c['rechazadas_forma']/total*100:.0f}%)",
            "rechazadas por verificacion": f"{c['rechazadas_verificacion']} ({c['rechazadas_verificacion']/total*100:.0f}%)",
            "segundos por generacion": round(c["segundos"] / total, 2),
        }
