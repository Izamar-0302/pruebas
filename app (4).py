"""
AgroDetect - Aplicacion Streamlit
Sistema de deteccion de enfermedades y plagas foliares en hojas de cafe.

Carga el modelo entrenado (MobileNetV2) directamente desde este repositorio
y consume la API de Groq (mediante API key) para generar recomendaciones
de manejo agronomico a partir del diagnostico.
"""

import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============================================================
# CONFIGURACION GENERAL
# ============================================================

MODEL_DIR = "model"
MODEL_PATH_KERAS = os.path.join(MODEL_DIR, "agrodetect_mobilenetv2.keras")
MODEL_PATH_H5 = os.path.join(MODEL_DIR, "agrodetect_mobilenetv2.h5")
CONFIG_PATH = os.path.join(MODEL_DIR, "config_app.json")

# Valores de respaldo por si config_app.json no estuviera disponible
CLASES_DEFAULT = ["sana", "roya", "cercospora", "phoma", "arana_roja", "minador"]
IMG_SIZE_DEFAULT = (224, 224)
UMBRAL_DEFAULT = 0.60

# Que tan cerca deben estar la 1ra y 2da clase mas probables para sospechar
# que hay mas de una enfermedad/plaga presente en la misma hoja (ver README)
MARGEN_COINFECCION = 0.15

NOMBRES_LEGIBLES = {
    "sana": "Hoja sana",
    "roya": "Roya (Hemileia vastatrix)",
    "cercospora": "Cercospora",
    "phoma": "Phoma",
    "arana_roja": "Araña roja (red spider mite)",
    "minador": "Minador de la hoja (leaf miner)",
}

RECOMENDACIONES_RESPALDO = {
    "sana": "La hoja no muestra signos de enfermedad ni plaga. Continúa con el monitoreo periódico habitual.",
    "roya": "Se recomienda poda sanitaria de las hojas afectadas, mejorar la ventilación de la plantación y "
            "consultar con un técnico del IHCAFE sobre el uso de fungicidas a base de cobre.",
    "cercospora": "Retira las hojas con manchas visibles, evita el exceso de sombra y humedad, y considera un "
                  "programa de fertilización balanceada para fortalecer la planta.",
    "phoma": "Realiza poda sanitaria, evita heridas mecánicas en la planta y mejora el drenaje del suelo. "
             "Consulta a un técnico agrónomo si la infección se extiende.",
    "arana_roja": "Aumenta la humedad relativa alrededor de la planta (la araña roja prospera en ambientes secos) "
                  "y considera un control biológico o acaricida específico, según recomendación técnica.",
    "minador": "Elimina las hojas con galerías visibles y evita el uso excesivo de insecticidas de amplio "
               "espectro, que eliminan a los enemigos naturales del minador.",
}


# ============================================================
# CARGA DEL MODELO Y CONFIGURACION (con cache para no recargar
# en cada interaccion del usuario)
# ============================================================

@st.cache_resource(show_spinner="Cargando modelo de IA...")
def cargar_modelo():
    """Carga el modelo entrenado desde este repositorio (.keras, con .h5 como respaldo)."""
    if os.path.exists(MODEL_PATH_KERAS):
        return keras.models.load_model(MODEL_PATH_KERAS)
    if os.path.exists(MODEL_PATH_H5):
        return keras.models.load_model(MODEL_PATH_H5)
    raise FileNotFoundError(
        f"No se encontró el modelo en '{MODEL_PATH_KERAS}' ni en '{MODEL_PATH_H5}'. "
        "Verifica que la carpeta 'model/' esté presente en el repositorio."
    )


@st.cache_data(show_spinner=False)
def cargar_config():
    """Carga clases, tamaño de imagen y umbral de decisión desde config_app.json,
    generado automáticamente al final del notebook de entrenamiento."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        clases = config.get("clases", CLASES_DEFAULT)
        img_size = tuple(config.get("img_size", IMG_SIZE_DEFAULT))
        umbral = config.get("umbral_decision", UMBRAL_DEFAULT)
        return clases, img_size, umbral
    st.warning(
        "No se encontró 'model/config_app.json'; usando valores por defecto. "
        "Verifica que el archivo esté presente en el repositorio."
    )
    return CLASES_DEFAULT, IMG_SIZE_DEFAULT, UMBRAL_DEFAULT


# ============================================================
# PREPROCESAMIENTO Y PREDICCION
# ============================================================

def preprocesar_imagen(imagen_pil: Image.Image, img_size: tuple) -> np.ndarray:
    """Redimensiona y normaliza la imagen exactamente igual que en el entrenamiento
    (mismo IMG_SIZE y mismo preprocess_input de MobileNetV2, que escala a [-1, 1])."""
    imagen_rgb = imagen_pil.convert("RGB")
    x = tf.image.resize(np.array(imagen_rgb), img_size)
    x = preprocess_input(tf.expand_dims(x, axis=0))
    return x


def predecir(modelo, x: np.ndarray, clases: list) -> np.ndarray:
    """Devuelve el arreglo de probabilidades (softmax) por clase."""
    probs = modelo.predict(x, verbose=0)[0]
    return probs


def interpretar_prediccion(probs: np.ndarray, clases: list, umbral: float):
    """Aplica el umbral de decision y detecta posible coinfeccion (ver README:
    seccion de mejoras futuras). Devuelve un diccionario con el resultado."""
    orden = np.argsort(probs)[::-1]
    idx1, idx2 = orden[0], orden[1]
    clase1, clase2 = clases[idx1], clases[idx2]
    p1, p2 = float(probs[idx1]), float(probs[idx2])

    if p1 < umbral:
        return {
            "estado": "no_concluyente",
            "clase_principal": clase1,
            "confianza_principal": p1,
        }

    posible_coinfeccion = (p1 - p2) < MARGEN_COINFECCION and clase2 != "sana"
    return {
        "estado": "coinfeccion" if posible_coinfeccion else "concluyente",
        "clase_principal": clase1,
        "confianza_principal": p1,
        "clase_secundaria": clase2 if posible_coinfeccion else None,
        "confianza_secundaria": p2 if posible_coinfeccion else None,
    }


# ============================================================
# MODULO DE RECOMENDACIONES (API de Groq, mediante API key)
# ============================================================

def obtener_recomendaciones(clase_principal: str, clase_secundaria: str = None) -> str:
    """Consulta la API de Groq para generar recomendaciones de manejo agronomico
    a partir del diagnostico. La API key se lee de st.secrets, nunca del codigo
    fuente. Si no esta configurada o la llamada falla, se usa una recomendacion
    de respaldo para no dejar al usuario sin informacion."""
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))

    diagnostico = NOMBRES_LEGIBLES.get(clase_principal, clase_principal)
    if clase_secundaria:
        diagnostico += f" junto con posibles signos de {NOMBRES_LEGIBLES.get(clase_secundaria, clase_secundaria)}"

    if not api_key:
        st.info("No se configuró GROQ_API_KEY; mostrando una recomendación general de respaldo.")
        return RECOMENDACIONES_RESPALDO.get(clase_principal, "Consulta a un técnico agrónomo del IHCAFE.")

    try:
        from groq import Groq

        cliente = Groq(api_key=api_key)
        prompt = (
            "Eres un asistente agronómico especializado en caficultura hondureña. "
            f"Un productor de café en Comayagua, Honduras, tomó una foto de una hoja de cafeto y el "
            f"diagnóstico automático fue: {diagnostico}. "
            "Da una recomendación breve (máximo 4 líneas), práctica y en español sencillo, sobre el manejo "
            "agronómico adecuado. Si corresponde, sugiere consultar a un técnico del IHCAFE para confirmar "
            "el diagnóstico antes de aplicar cualquier tratamiento."
        )
        respuesta = cliente.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=250,
        )
        return respuesta.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"No se pudo consultar la API de Groq ({e}). Mostrando recomendación de respaldo.")
        return RECOMENDACIONES_RESPALDO.get(clase_principal, "Consulta a un técnico agrónomo del IHCAFE.")


# ============================================================
# INTERFAZ DE USUARIO (Streamlit)
# ============================================================

def main():
    st.set_page_config(page_title="AgroDetect", page_icon="🌿", layout="centered")

    st.title("🌿 AgroDetect")
    st.caption("Detección de enfermedades y plagas foliares en el cultivo de café")

    modelo = cargar_modelo()
    clases, img_size, umbral = cargar_config()

    with st.sidebar:
        st.header("Acerca de AgroDetect")
        st.write(
            "Sube o captura una fotografía de una hoja de café. El modelo (MobileNetV2, "
            "entrenado mediante *transfer learning*) clasifica la imagen entre 6 categorías: "
            "hoja sana, roya, cercospora, phoma, araña roja y minador de la hoja."
        )
        st.write(f"**Umbral de confianza mínimo:** {umbral:.0%}")
        st.write(
            "Este sistema es una herramienta de apoyo al diagnóstico y **no reemplaza** la "
            "confirmación de un técnico agrónomo del IHCAFE."
        )

    origen = st.radio("¿Cómo quieres proporcionar la imagen?", ["Subir archivo", "Usar cámara"], horizontal=True)

    archivo_imagen = None
    if origen == "Subir archivo":
        archivo_imagen = st.file_uploader("Selecciona una fotografía de la hoja", type=["jpg", "jpeg", "png"])
    else:
        archivo_imagen = st.camera_input("Captura una fotografía de la hoja")

    if archivo_imagen is None:
        st.info("Sube o captura una imagen para comenzar el diagnóstico.")
        return

    imagen_pil = Image.open(archivo_imagen)
    st.image(imagen_pil, caption="Imagen recibida", use_container_width=True)

    with st.spinner("Analizando la imagen..."):
        x = preprocesar_imagen(imagen_pil, img_size)
        probs = predecir(modelo, x, clases)
        resultado = interpretar_prediccion(probs, clases, umbral)

    st.divider()
    st.subheader("Resultado del diagnóstico")

    if resultado["estado"] == "no_concluyente":
        st.warning(
            f"⚠️ El resultado no es concluyente (confianza de {resultado['confianza_principal']:.0%}, "
            f"por debajo del umbral de {umbral:.0%}). Te recomendamos tomar otra fotografía con mejor "
            "enfoque/iluminación, o consultar directamente a un técnico del IHCAFE."
        )
    else:
        clase_principal = resultado["clase_principal"]
        st.success(
            f"**{NOMBRES_LEGIBLES.get(clase_principal, clase_principal)}** "
            f"(confianza: {resultado['confianza_principal']:.0%})"
        )

        if resultado["estado"] == "coinfeccion":
            clase_secundaria = resultado["clase_secundaria"]
            st.info(
                f"🔎 Posible coinfección: el modelo también detectó señales de "
                f"**{NOMBRES_LEGIBLES.get(clase_secundaria, clase_secundaria)}** "
                f"(confianza: {resultado['confianza_secundaria']:.0%}). "
                "Se recomienda inspección adicional presencial para confirmar."
            )

        with st.expander("Ver probabilidades por clase"):
            for idx in np.argsort(probs)[::-1]:
                st.write(f"{NOMBRES_LEGIBLES.get(clases[idx], clases[idx])}: {probs[idx]:.1%}")

        st.subheader("💬 Recomendación de manejo agronómico")
        with st.spinner("Generando recomendación..."):
            clase_secundaria = resultado.get("clase_secundaria")
            recomendacion = obtener_recomendaciones(clase_principal, clase_secundaria)
        st.write(recomendacion)

    st.caption(
        "Este diagnóstico es una herramienta de apoyo. Ante cualquier duda, consulta a un "
        "técnico agrónomo del IHCAFE antes de aplicar tratamientos."
    )


if __name__ == "__main__":
    main()
