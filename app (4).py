"""
AgroDetect - Aplicacion Streamlit
Sistema de deteccion de enfermedades y plagas foliares en hojas de cafe.

Carga el modelo entrenado (MobileNetV2) directamente desde este repositorio
y consume la API de Groq (mediante API key) para generar recomendaciones
de manejo agronomico a partir del diagnostico.
"""

import json
import os
import uuid
from datetime import datetime

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

GITHUB_REPO = "Izamar-0302/AgroDetect"  # usado para el enlace de feedback -> issue de GitHub

CLASES_DEFAULT = ["sana", "roya", "cercospora", "phoma", "arana_roja", "minador"]
IMG_SIZE_DEFAULT = (224, 224)
UMBRAL_DEFAULT = 0.60
MARGEN_COINFECCION = 0.15

NOMBRES_LEGIBLES = {
    "sana": "Hoja sana",
    "roya": "Roya (Hemileia vastatrix)",
    "cercospora": "Cercospora",
    "phoma": "Phoma",
    "arana_roja": "Araña roja (red spider mite)",
    "minador": "Minador de la hoja (leaf miner)",
}

ICONOS = {
    "sana": "✅", "roya": "🟠", "cercospora": "🟤",
    "phoma": "⚫", "arana_roja": "🕷️", "minador": "🍃",
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

GUIA_ENFERMEDADES = {
    "roya": {
        "sintomas": "Manchas amarillo-anaranjadas en el envés de la hoja, que luego se cubren de un polvo "
                    "naranja (esporas). Provoca defoliación si no se controla.",
        "manejo": "Poda sanitaria, variedades resistentes, fungicidas a base de cobre, buena ventilación.",
    },
    "cercospora": {
        "sintomas": "Manchas circulares de color marrón claro/grisáceo con borde definido, a veces con anillos "
                    "concéntricos ('ojo de gallo').",
        "manejo": "Reducir sombra excesiva y humedad, fertilización balanceada, retirar hojas afectadas.",
    },
    "phoma": {
        "sintomas": "Lesiones oscuras e irregulares, a menudo cerca de heridas o puntas de la hoja, que pueden "
                    "extenderse en condiciones de alta humedad y frío.",
        "manejo": "Evitar heridas mecánicas, mejorar drenaje, poda sanitaria.",
    },
    "arana_roja": {
        "sintomas": "Moteado o decoloración amarillenta difusa en la hoja, a veces con finas telarañas en el "
                    "envés. Prospera en clima seco.",
        "manejo": "Aumentar humedad ambiental, control biológico (depredadores naturales) o acaricidas.",
    },
    "minador": {
        "sintomas": "Galerías o túneles serpenteantes visibles dentro del tejido de la hoja, causados por larvas.",
        "manejo": "Eliminar hojas afectadas, evitar insecticidas de amplio espectro que maten enemigos naturales.",
    },
}

# ============================================================
# ESTILO VISUAL (paleta café/verde + modo oscuro)
# ============================================================

def inyectar_estilo(dark_mode: bool):
    if dark_mode:
        bg, text, card = "#160E0A", "#F3EFEA", "#231710"
    else:
        bg, text, card = "#FDFBF7", "#2C1A11", "#FFFFFF"

    verde, ambar = "#1E5631", "#D97706"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {bg};
            color: {text};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {card};
        }}
        h1, h2, h3, h4 {{
            color: {text} !important;
            font-family: 'Georgia', 'Times New Roman', serif;
        }}
        .agro-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.9rem 1.2rem;
            background-color: #2C1A11;
            border-bottom: 3px solid {ambar};
            border-radius: 8px;
            margin-bottom: 1.2rem;
        }}
        .agro-header-title {{
            font-family: 'Georgia', serif;
            font-weight: bold;
            font-size: 1.3rem;
            color: white;
        }}
        .agro-header-sub {{
            color: {ambar};
            font-size: 0.85rem;
        }}
        .agro-card {{
            background-color: {card};
            border: 1px solid {verde}33;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
        }}
        .agro-badge {{
            display: inline-block;
            background-color: {verde};
            color: white;
            padding: 0.15rem 0.6rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .agro-footer {{
            background-color: #2C1A11;
            color: #D1D5DB;
            border-top: 1px solid {ambar}55;
            padding: 1.2rem 1rem;
            text-align: center;
            font-size: 0.75rem;
            border-radius: 8px;
            margin-top: 2rem;
        }}
        .agro-footer a {{
            color: {ambar};
            text-decoration: none;
        }}
        .stButton > button {{
            background-color: {verde};
            color: white;
            border-radius: 8px;
            border: none;
        }}
        .stButton > button:hover {{
            background-color: #163f24;
            color: white;
        }}
        ::selection {{
            background-color: {verde};
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(dark_mode: bool):
    st.markdown(
        """
        <div class="agro-header">
            <div>
                <div class="agro-header-title">🌿 AgroDetect</div>
                <div class="agro-header-sub">Caficultura Honduras</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        f"""
        <div class="agro-footer">
            <p><strong>🌿 AgroDetect</strong> &nbsp;•&nbsp; Herramienta de diagnóstico de IA y recomendación
            agronómica IHCAFE &nbsp;•&nbsp; MobileNetV2 &amp; Groq</p>
            <p>
                <a href="https://www.ihcafe.hn" target="_blank">Sitio Oficial IHCAFE</a>
                &nbsp;•&nbsp;
                <a href="https://github.com/{GITHUB_REPO}" target="_blank">Repositorio en GitHub</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CARGA DEL MODELO Y CONFIGURACION
# ============================================================

@st.cache_resource(show_spinner="Cargando modelo de IA...")
def cargar_modelo():
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
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        clases = config.get("clases", CLASES_DEFAULT)
        img_size = tuple(config.get("img_size", IMG_SIZE_DEFAULT))
        umbral = config.get("umbral_decision", UMBRAL_DEFAULT)
        return clases, img_size, umbral
    st.warning("No se encontró 'model/config_app.json'; usando valores por defecto.")
    return CLASES_DEFAULT, IMG_SIZE_DEFAULT, UMBRAL_DEFAULT


# ============================================================
# PREPROCESAMIENTO, PREDICCION Y RECOMENDACIONES
# ============================================================

def preprocesar_imagen(imagen_pil: Image.Image, img_size: tuple) -> np.ndarray:
    imagen_rgb = imagen_pil.convert("RGB")
    x = tf.image.resize(np.array(imagen_rgb), img_size)
    x = preprocess_input(tf.expand_dims(x, axis=0))
    return x


def interpretar_prediccion(probs: np.ndarray, clases: list, umbral: float):
    orden = np.argsort(probs)[::-1]
    idx1, idx2 = orden[0], orden[1]
    clase1, clase2 = clases[idx1], clases[idx2]
    p1, p2 = float(probs[idx1]), float(probs[idx2])

    if p1 < umbral:
        return {"estado": "no_concluyente", "clase_principal": clase1, "confianza_principal": p1}

    posible_coinfeccion = (p1 - p2) < MARGEN_COINFECCION and clase2 != "sana"
    return {
        "estado": "coinfeccion" if posible_coinfeccion else "concluyente",
        "clase_principal": clase1,
        "confianza_principal": p1,
        "clase_secundaria": clase2 if posible_coinfeccion else None,
        "confianza_secundaria": p2 if posible_coinfeccion else None,
    }


def obtener_recomendaciones(clase_principal: str, clase_secundaria: str = None) -> str:
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    diagnostico = NOMBRES_LEGIBLES.get(clase_principal, clase_principal)
    if clase_secundaria:
        diagnostico += f" junto con posibles signos de {NOMBRES_LEGIBLES.get(clase_secundaria, clase_secundaria)}"

    if not api_key:
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
# PESTAÑA: ESCÁNER
# ============================================================

def tab_scanner(modelo, clases, img_size, umbral):
    st.subheader("📷 Escáner de hojas")
    st.write("Sube o captura una fotografía de una hoja de café para obtener un diagnóstico.")

    origen = st.radio("Origen de la imagen", ["Subir archivo", "Usar cámara"], horizontal=True, label_visibility="collapsed")
    archivo_imagen = (
        st.file_uploader("Selecciona una fotografía", type=["jpg", "jpeg", "png"])
        if origen == "Subir archivo"
        else st.camera_input("Captura una fotografía")
    )

    if archivo_imagen is None:
        st.info("Sube o captura una imagen para comenzar el diagnóstico.")
        return

    imagen_pil = Image.open(archivo_imagen)
    st.image(imagen_pil, caption="Imagen recibida", use_container_width=True)

    with st.spinner("Analizando la imagen..."):
        x = preprocesar_imagen(imagen_pil, img_size)
        probs = modelo.predict(x, verbose=0)[0]
        resultado = interpretar_prediccion(probs, clases, umbral)

    st.markdown("<div class='agro-card'>", unsafe_allow_html=True)

    if resultado["estado"] == "no_concluyente":
        st.warning(
            f"⚠️ Resultado no concluyente (confianza de {resultado['confianza_principal']:.0%}, "
            f"por debajo del umbral de {umbral:.0%}). Toma otra fotografía con mejor enfoque/iluminación, "
            "o consulta a un técnico del IHCAFE."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    clase_principal = resultado["clase_principal"]
    icono = ICONOS.get(clase_principal, "🌿")
    st.markdown(
        f"### {icono} {NOMBRES_LEGIBLES.get(clase_principal, clase_principal)} "
        f"<span class='agro-badge'>{resultado['confianza_principal']:.0%} confianza</span>",
        unsafe_allow_html=True,
    )

    clase_secundaria = resultado.get("clase_secundaria")
    if resultado["estado"] == "coinfeccion":
        st.info(
            f"🔎 Posible coinfección con **{NOMBRES_LEGIBLES.get(clase_secundaria, clase_secundaria)}** "
            f"({resultado['confianza_secundaria']:.0%}). Se recomienda inspección adicional presencial."
        )

    with st.expander("Ver probabilidades por clase"):
        for idx in np.argsort(probs)[::-1]:
            st.write(f"{NOMBRES_LEGIBLES.get(clases[idx], clases[idx])}: {probs[idx]:.1%}")

    st.markdown("**💬 Recomendación de manejo agronómico**")
    with st.spinner("Generando recomendación..."):
        recomendacion = obtener_recomendaciones(clase_principal, clase_secundaria)
    st.write(recomendacion)

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("💾 Guardar en historial"):
        nuevo = {
            "id": str(uuid.uuid4()),
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "clase": clase_principal,
            "confianza": resultado["confianza_principal"],
            "recomendacion": recomendacion,
        }
        st.session_state.diagnosticos.insert(0, nuevo)
        st.success("Diagnóstico guardado en el historial.")

    with st.expander("¿No estás de acuerdo con este diagnóstico?"):
        if st.button("📝 Reportar este resultado"):
            st.session_state.feedback_ref = clase_principal
            st.session_state.ir_a_feedback = True
            st.rerun()


# ============================================================
# PESTAÑA: HISTORIAL
# ============================================================

def tab_historial():
    st.subheader("🕓 Historial de diagnósticos")
    st.caption("El historial se guarda solo durante esta sesión del navegador.")

    if not st.session_state.diagnosticos:
        st.info("Aún no has guardado ningún diagnóstico. Ve a la pestaña Escáner para analizar una hoja.")
        return

    if st.button("🗑️ Borrar todo el historial"):
        st.session_state.diagnosticos = []
        st.rerun()

    for item in st.session_state.diagnosticos:
        icono = ICONOS.get(item["clase"], "🌿")
        st.markdown(
            f"""<div class="agro-card">
                <strong>{icono} {NOMBRES_LEGIBLES.get(item['clase'], item['clase'])}</strong>
                <span class="agro-badge">{item['confianza']:.0%}</span>
                <br><small>{item['fecha']}</small>
                <p>{item['recomendacion']}</p>
            </div>""",
            unsafe_allow_html=True,
        )


# ============================================================
# PESTAÑA: FEEDBACK (mediante GitHub Issues, sin backend propio)
# ============================================================

def tab_feedback():
    st.subheader("📝 Feedback")
    st.write(
        "¿El diagnóstico fue incorrecto o tienes una sugerencia? Repórtalo directamente como un "
        "*issue* en el repositorio de GitHub del proyecto — no necesitas cuenta especial, solo una "
        "cuenta de GitHub."
    )

    ref_default = st.session_state.get("feedback_ref", "")
    clase_reportada = st.selectbox(
        "Clase relacionada (opcional)",
        [""] + list(NOMBRES_LEGIBLES.keys()),
        format_func=lambda c: "Sin especificar" if c == "" else NOMBRES_LEGIBLES[c],
        index=(list(NOMBRES_LEGIBLES.keys()).index(ref_default) + 1) if ref_default in NOMBRES_LEGIBLES else 0,
    )
    comentario = st.text_area("Describe el problema o tu sugerencia", height=120)

    titulo = f"[Feedback] {NOMBRES_LEGIBLES.get(clase_reportada, 'General')}"
    cuerpo = comentario or "(sin descripción)"

    import urllib.parse
    url_issue = (
        f"https://github.com/{GITHUB_REPO}/issues/new?"
        f"title={urllib.parse.quote(titulo)}&body={urllib.parse.quote(cuerpo)}"
    )

    st.link_button("📮 Abrir reporte en GitHub", url_issue, use_container_width=True)
    st.caption("Se abrirá GitHub en una nueva pestaña con el reporte ya redactado para que lo envíes.")


# ============================================================
# PESTAÑA: GUÍA
# ============================================================

def tab_guia():
    st.subheader("📚 Guía de enfermedades y plagas")
    st.write("Referencia rápida de síntomas y manejo agronómico para cada clase detectada por AgroDetect.")

    for clase, info in GUIA_ENFERMEDADES.items():
        icono = ICONOS.get(clase, "🌿")
        with st.expander(f"{icono} {NOMBRES_LEGIBLES[clase]}"):
            st.markdown(f"**Síntomas:** {info['sintomas']}")
            st.markdown(f"**Manejo recomendado:** {info['manejo']}")

    st.info(
        "Esta guía es de carácter informativo. Ante una infección o infestación severa, consulta "
        "siempre a un técnico agrónomo del IHCAFE."
    )


# ============================================================
# APP PRINCIPAL
# ============================================================

def main():
    st.set_page_config(page_title="AgroDetect", page_icon="🌿", layout="centered")

    if "diagnosticos" not in st.session_state:
        st.session_state.diagnosticos = []
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
    if "ir_a_feedback" not in st.session_state:
        st.session_state.ir_a_feedback = False

    with st.sidebar:
        st.session_state.dark_mode = st.toggle("🌙 Modo oscuro", value=st.session_state.dark_mode)
        st.divider()
        st.header("Acerca de AgroDetect")
        st.write(
            "Modelo MobileNetV2 (transfer learning) que clasifica hojas de café entre 6 categorías: "
            "sana, roya, cercospora, phoma, araña roja y minador de la hoja."
        )
        st.write(
            "Herramienta de apoyo al diagnóstico — **no reemplaza** la confirmación de un técnico "
            "agrónomo del IHCAFE."
        )

    inyectar_estilo(st.session_state.dark_mode)
    render_header(st.session_state.dark_mode)

    modelo = cargar_modelo()
    clases, img_size, umbral = cargar_config()

    tabs_labels = ["📷 Escáner", "🕓 Historial", "📝 Feedback", "📚 Guía"]
    st.session_state.ir_a_feedback = False

    tab_escaner, tab_hist, tab_fb, tab_guia_ui = st.tabs(tabs_labels)

    with tab_escaner:
        tab_scanner(modelo, clases, img_size, umbral)
    with tab_hist:
        tab_historial()
    with tab_fb:
        tab_feedback()
    with tab_guia_ui:
        tab_guia()

    render_footer()


if __name__ == "__main__":
    main()
