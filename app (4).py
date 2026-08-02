import streamlit as st
import json
import os
import shutil
import tempfile
from datetime import datetime

import numpy as np
import requests
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ====================================================
# AGRODETECT v2.0 - CAFICULTURA HONDURAS (IHCAFE)
# SINGLE-USER ELEGANT APP.PY (STREAMLIT)
# ====================================================

st.set_page_config(
    page_title="AgroDetect v2.0 - Diagnóstico Foliar de Café",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------------------------------
# CONFIGURACIÓN GITHUB (para retroalimentación persistente)
# ----------------------------------------------------
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", os.environ.get("GITHUB_TOKEN"))
GITHUB_REPO = st.secrets.get("GITHUB_REPO", os.environ.get("GITHUB_REPO"))

# ----------------------------------------------------
# MODELO ENTRENADO (cargado desde este repositorio)
# ----------------------------------------------------
MODEL_DIR = "model"
MODEL_PATH_KERAS = os.path.join(MODEL_DIR, "agrodetect_mobilenetv2.keras.zip")
MODEL_PATH_H5 = os.path.join(MODEL_DIR, "model.weights.h5")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

CLASES_DEFAULT = ["sana", "roya", "cercospora", "phoma", "arana_roja", "minador"]
IMG_SIZE_DEFAULT = (224, 224)
UMBRAL_DEFAULT = 0.60
MARGEN_COINFECCION = 0.15


@st.cache_resource(show_spinner="Cargando modelo de IA...")
def cargar_modelo():
    # Keras 3 exige que el archivo tenga extension .keras para reconocerlo.
    # Como GitHub descomprime los .keras al subirlos por web, lo renombraste a .zip.
    # Lo copiamos a un archivo temporal con la extension correcta y cargamos desde ahi.
    if os.path.exists(MODEL_PATH_KERAS):
        temp_path = os.path.join(tempfile.gettempdir(), "agrodetect_temp.keras")
        shutil.copy(MODEL_PATH_KERAS, temp_path)
        return keras.models.load_model(temp_path)

    # Fallback: si GitHub descomprimio todo y no existe el .zip,
    # reconstruimos desde config.json + model.weights.h5
    if os.path.exists(CONFIG_PATH) and os.path.exists(MODEL_PATH_H5):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        model = keras.models.model_from_json(json.dumps(config))
        model.load_weights(MODEL_PATH_H5)
        return model

    raise FileNotFoundError("No se encontro el modelo en ningun formato compatible.")


@st.cache_data(show_spinner=False)
def cargar_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        clases = config.get("clases", CLASES_DEFAULT)
        img_size = tuple(config.get("img_size", IMG_SIZE_DEFAULT))
        umbral = config.get("umbral_decision", UMBRAL_DEFAULT)
        return clases, img_size, umbral
    st.warning("No se encontro 'model/config_app.json'; usando valores por defecto.")
    return CLASES_DEFAULT, IMG_SIZE_DEFAULT, UMBRAL_DEFAULT


def preprocesar_imagen(imagen_pil, img_size):
    imagen_rgb = imagen_pil.convert("RGB")
    x = tf.image.resize(np.array(imagen_rgb), img_size)
    x = preprocess_input(tf.expand_dims(x, axis=0))
    return x


def interpretar_prediccion(probs, clases, umbral):
    orden = np.argsort(probs)[::-1]
    idx1, idx2 = orden[0], orden[1]
    clase1, clase2 = clases[idx1], clases[idx2]
    p1, p2 = float(probs[idx1]), float(probs[idx2])

    if p1 < umbral:
        return {"estado": "no_concluyente", "clase": clase1, "confianza": p1}

    posible_coinfeccion = (p1 - p2) < MARGEN_COINFECCION and clase2 != "sana"
    return {
        "estado": "coinfeccion" if posible_coinfeccion else "concluyente",
        "clase": clase1,
        "confianza": p1,
        "clase_secundaria": clase2 if posible_coinfeccion else None,
        "confianza_secundaria": p2 if posible_coinfeccion else None,
    }


def obtener_recomendacion(clase, clase_secundaria=None):
    """Genera la recomendacion con la API de Groq (API key en st.secrets); si no esta
    configurada o falla la llamada, usa el texto curado de COFFEE_DISEASES como respaldo."""
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    respaldo = COFFEE_DISEASES[clase]["recommendation"]

    if not api_key:
        return respaldo

    diagnostico = COFFEE_DISEASES[clase]["name"]
    if clase_secundaria:
        diagnostico += f" junto con posibles signos de {COFFEE_DISEASES[clase_secundaria]['name']}"

    try:
        from groq import Groq
        cliente = Groq(api_key=api_key)
        prompt = (
            "Eres un asistente agronomico especializado en caficultura hondurena. "
            f"Un productor de cafe en Comayagua, Honduras, tomo una foto de una hoja de cafeto y el "
            f"diagnostico automatico fue: {diagnostico}. "
            "Da una recomendacion breve (maximo 4 lineas), practica y en espanol sencillo, sobre el manejo "
            "agronomico adecuado. Si corresponde, sugiere consultar a un tecnico del IHCAFE para confirmar "
            "el diagnostico antes de aplicar cualquier tratamiento."
        )
        respuesta = cliente.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=250,
        )
        return respuesta.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"No se pudo consultar la API de Groq ({e}). Mostrando recomendacion de respaldo.")
        return respaldo


# ============================================================
# GITHUB ISSUES API — Retroalimentacion persistente
# ============================================================
def enviar_feedback_github(comentario, diagnostico_relacionado=None):
    """Crea un Issue en GitHub con la retroalimentacion del usuario."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return False, "GITHUB_TOKEN o GITHUB_REPO no configurados en secrets."

    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    title = f"[Feedback] {comentario[:50]}{'...' if len(comentario) > 50 else ''}"
    body = (
        f"**Comentario:** {comentario}\n\n"
        f"**Diagnostico relacionado:** {diagnostico_relacionado or 'N/A'}\n\n"
        f"**Fecha:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n"
        f"_Enviado desde AgroDetect v2.0_"
    )

    payload = {
        "title": title,
        "body": body,
        "labels": ["feedback"]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 201:
            return True, "Feedback enviado a GitHub correctamente."
        else:
            return False, f"Error GitHub {response.status_code}: {response.json().get('message', 'Unknown')}"
    except Exception as e:
        return False, str(e)


def obtener_feedback_github():
    """Obtiene los Issues etiquetados como 'feedback' desde GitHub."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return []

    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    params = {
        "labels": "feedback",
        "state": "all",
        "per_page": 30
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        if response.status_code == 200:
            issues = response.json()
            feedback_list = []
            for issue in issues:
                # Extraer diagnostico del body
                body = issue.get("body", "")
                diag = "N/A"
                for line in body.split("\n"):
                    if "**Diagnostico relacionado:**" in line:
                        diag = line.split("**Diagnostico relacionado:**")[-1].strip()
                        break

                feedback_list.append({
                    "comentario": issue.get("title", "").replace("[Feedback] ", ""),
                    "diagnostico_relacionado": diag,
                    "fecha": datetime.strptime(issue["created_at"], "%Y-%m-%dT%H:%M:%SZ").strftime("%d/%m/%Y %H:%M"),
                    "url": issue.get("html_url", ""),
                    "estado": issue.get("state", "open")
                })
            return feedback_list
        return []
    except Exception:
        return []


# ----------------------------------------------------
# DATASET & DISEASE METADATA
# ----------------------------------------------------
COFFEE_DISEASES = {
    "roya": {
        "name": "Roya del Cafe",
        "scientificName": "Hemileia vastatrix",
        "category": "Enfermedad Fungica Grave",
        "color": "#D97706",
        "severity": "Alta",
        "symptoms": [
            "Pustulas de color naranja amarillento en el enves de la hoja.",
            "Manchas cloroticas translucidas en el haz.",
            "Defoliacion severa e incapacidad de llenado del grano."
        ],
        "recommendation": "Se observa una infeccion severa. Se recomienda poda sanitaria inmediata y aplicacion de fungicidas a base de oxicloruro de cobre. Incremente el drenaje en la seccion de su finca para reducir la humedad estancada."
    },
    "cercospora": {
        "name": "Cercospora / Mancha de Hierro",
        "scientificName": "Cercospora coffeicola",
        "category": "Enfermedad Fungica",
        "color": "#B45309",
        "severity": "Media - Alta",
        "symptoms": [
            "Manchas circulares marron rojizo con centro grisaceo y halo amarillo.",
            "Afecta principalmente hojas expuestas a deficiencia de Nitrogeno y exceso de sol."
        ],
        "recommendation": "Reforzar la fertilizacion nitrogenada foliar e implementar regulacion de sombra. Aplicar caldo bordeles u oxicloruro de cobre de manera preventiva."
    },
    "phoma": {
        "name": "Phoma / Quema o Derretimiento",
        "scientificName": "Phoma costarricensis",
        "category": "Enfermedad Fungica de Altura",
        "color": "#DC2626",
        "severity": "Alta en Zonas Altas (>1300 msnm)",
        "symptoms": [
            "Lesiones oscuras e irregulares en los bordes y apices de la hoja.",
            "Aspecto de quemadura o derretido foliar en brotes tiernos."
        ],
        "recommendation": "Establecer cortinas rompevientos, reducir el exceso de humedad relativa en la parcela y aplicar fungicidas especificos autorizados por IHCAFE."
    },
    "arana_roja": {
        "name": "Arana Roja del Cafeto",
        "scientificName": "Oligonychus coffeae",
        "category": "Plaga de Acaros",
        "color": "#EA580C",
        "severity": "Media",
        "symptoms": [
            "Bronceado o cambio a tono cafe rojizo en la cara superior de la hoja.",
            "Telaranas finas microscopicas en el enves en epocas secas."
        ],
        "recommendation": "Aplicar acaricidas especificos de origen botanico o azufre mojable. Evitar aplicaciones excesivas de insecticidas de amplio espectro."
    },
    "minador": {
        "name": "Minador de la Hoja",
        "scientificName": "Leucoptera coffeella",
        "category": "Plaga de Insecto (Microlepidoptero)",
        "color": "#7C2D12",
        "severity": "Media",
        "symptoms": [
            "Galerias o minas transparentes/marrones necroticas que secan la epidermis.",
            "Enrollado foliar prematuro."
        ],
        "recommendation": "Realizar control biologico favoreciendo avispitas parasitoides y aplicar tratamientos botanicos autorizados en parches de mayor infestacion."
    },
    "sana": {
        "name": "Hoja Sana",
        "scientificName": "Coffea arabica / canephora",
        "category": "Tejido Foliar Saludable",
        "color": "#1E5631",
        "severity": "Ninguna",
        "symptoms": [
            "Tejido verde brillante, epidermis continua sin pustulas ni perforaciones.",
            "Fisiologia foliar optima."
        ],
        "recommendation": "Tejido foliar sano y en optimas condiciones. Se sugiere mantener el plan de nutricion foliar y continuar los monitoreos fitosanitarios rutinarios."
    }
}

HISTORY_FILE = "diagnosis_history.json"


def guardar_historial():
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # en despliegues de solo lectura, el historial vive solo en la sesion


# ----------------------------------------------------
# SESSION STATE INITIALIZATION
# ----------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "claro"

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "diagnostico"

if "history" not in st.session_state:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                st.session_state.history = json.load(f)
        except Exception:
            st.session_state.history = []
    else:
        st.session_state.history = []

if "current_diag" not in st.session_state:
    st.session_state.current_diag = st.session_state.history[0] if st.session_state.history else None

modelo = cargar_modelo()
CLASES, IMG_SIZE, UMBRAL = cargar_config()

# ----------------------------------------------------
# STYLING (DARK / LIGHT LUXURY WARM THEME)
# ----------------------------------------------------
is_dark = st.session_state.theme == "oscuro"

bg_body = "#140D0A" if is_dark else "#FAF7F2"
card_bg = "#1F1510" if is_dark else "#FFFFFF"
text_main = "#F5EFE9" if is_dark else "#2C1A11"
text_sub = "#B8ACA2" if is_dark else "#6B5E55"
border_color = "#36271D" if is_dark else "#EAE3D9"
highlight_bg = "#271B14" if is_dark else "#F5F1EA"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  {{
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}

    .stApp {{
        background-color: {bg_body};
        color: {text_main};
    }}

    /* =====================================================
       FIX MODO CLARO: forzar colores en componentes nativos
       ===================================================== */

    /* Radio buttons / Tabs */
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] label p,
    [data-testid="stRadio"] label span,
    [data-testid="stRadio"] div[role="radiogroup"] label {{
        color: {text_main} !important;
    }}

    /* Inputs de texto */
    [data-testid="stTextInput"] > div > div > input,
    [data-testid="stTextArea"] > div > div > textarea {{
        background-color: {card_bg} !important;
        color: {text_main} !important;
        border: 1px solid {border_color} !important;
        border-radius: 12px !important;
    }}
    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder {{
        color: {text_sub} !important;
        opacity: 0.7;
    }}

    /* File uploader */
    [data-testid="stFileUploader"] section {{
        background-color: {card_bg} !important;
        border: 1px dashed {border_color} !important;
        border-radius: 16px !important;
    }}
    [data-testid="stFileUploader"] section > div > span,
    [data-testid="stFileUploader"] section > div > small {{
        color: {text_sub} !important;
    }}

    /* Botones nativos */
    .stButton > button {{
        border-radius: 9999px !important;
        font-weight: 700 !important;
        font-size: 12px !important;
        background-color: {card_bg} !important;
        color: {text_main} !important;
        border: 1px solid {border_color} !important;
    }}
    .stButton > button:hover {{
        border-color: #D97706 !important;
        color: #D97706 !important;
        background-color: {highlight_bg} !important;
    }}
    .stButton > button[kind="primary"] {{
        background-color: {"#2C1A11" if not is_dark else "#F5EFE9"} !important;
        color: {"#FAF7F2" if not is_dark else "#2C1A11"} !important;
        border: none !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: #D97706 !important;
        color: #FFFFFF !important;
    }}

    /* Textos nativos */
    h1, h2, h3, h4, h5, h6, p, label, .stCaption, 
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stHeading"] h1,
    [data-testid="stHeading"] h2,
    [data-testid="stHeading"] h3 {{
        color: {text_main} !important;
    }}

    /* Expanders */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {{
        color: {text_main} !important;
    }}

    /* Elegant Navbar */
    .nav-container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 24px;
        background-color: {bg_body};
        border-bottom: 1px solid {border_color};
        margin-bottom: 24px;
    }}
    .brand-title {{
        font-family: 'Playfair Display', serif;
        font-size: 24px;
        font-weight: 700;
        color: {text_main};
    }}
    .brand-version {{
        font-size: 11px;
        font-family: monospace;
        color: #D97706;
        margin-left: 6px;
    }}

    /* Main Card Frame */
    .agro-card-frame {{
        background-color: {card_bg};
        border: 1px solid {border_color};
        border-radius: 32px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.06);
        padding: 0px;
        overflow: hidden;
    }}

    .recommendation-box {{
        background-color: {highlight_bg};
        border: 1px solid {border_color};
        border-radius: 20px;
        padding: 20px;
        margin-top: 16px;
        margin-bottom: 20px;
    }}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# TOP NAVBAR
# ----------------------------------------------------
col_nav1, col_nav2, col_nav3 = st.columns([2, 3, 2])

with col_nav1:
    st.markdown(f'<div style="display:flex; align-items:baseline;"><span class="brand-title">AgroDetect</span><span class="brand-version">v2.0</span></div>', unsafe_allow_html=True)

with col_nav2:
    tab_choice = st.radio(
        "",
        ["DIAGNOSTICO", "HISTORIAL", "GUIAS TECNICAS", "GITHUB"],
        horizontal=True,
        label_visibility="collapsed"
    )

with col_nav3:
    c_btn1, c_btn2 = st.columns([2, 1])
    with c_btn1:
        if st.button("📄 EXPORTAR PDF", use_container_width=True):
            st.toast("Generando informe PDF impreso...", icon="📄")
    with c_btn2:
        theme_label = "🌙" if is_dark else "☀️"
        if st.button(theme_label, use_container_width=True):
            st.session_state.theme = "claro" if is_dark else "oscuro"
            st.rerun()

st.markdown("<hr style='margin-top:0; margin-bottom:24px; border-color:" + border_color + ";'>", unsafe_allow_html=True)

# ----------------------------------------------------
# 1. TAB: DIAGNOSTICO (MAIN ELEGANT SPLIT UI)
# ----------------------------------------------------
if tab_choice == "DIAGNOSTICO":

    col_left, col_right = st.columns([1, 1], gap="large")

    # LEFT PANEL: CAPTURA DE IMAGEN FOLIAR
    with col_left:
        st.markdown(f"""
        <h2 style="font-family: 'Playfair Display', serif; margin-bottom: 4px;">Captura de Imagen Foliar</h2>
        <p style="font-size: 13px; color: {text_sub}; margin-bottom: 24px;">
            Posicione la hoja de cafe bajo luz natural. El sistema detectara automaticamente signos de Roya, Cercospora o Plagas.
        </p>
        """, unsafe_allow_html=True)

        uploaded_img = st.file_uploader("Sube la imagen foliar (.jpg, .png):", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

        if uploaded_img is not None:
            image = Image.open(uploaded_img)
            st.image(image, use_container_width=True)
            if st.button("🚀 DIAGNOSTICAR AHORA", type="primary", use_container_width=True):
                with st.spinner("Analizando pigmentacion necrotica foliar..."):
                    x = preprocesar_imagen(image, IMG_SIZE)
                    probs = modelo.predict(x, verbose=0)[0]
                    resultado = interpretar_prediccion(probs, CLASES, UMBRAL)

                    if resultado["estado"] == "no_concluyente":
                        st.warning(
                            f"⚠️ Resultado no concluyente (confianza de {resultado['confianza']*100:.1f}%, "
                            f"por debajo del umbral de {UMBRAL*100:.0f}%). Toma otra fotografia con mejor "
                            "enfoque/iluminacion, o consulta a un tecnico del IHCAFE."
                        )
                    else:
                        recomendacion = obtener_recomendacion(resultado["clase"], resultado.get("clase_secundaria"))
                        if resultado["estado"] == "coinfeccion":
                            recomendacion = (
                                f"🔎 Posible coinfeccion con {COFFEE_DISEASES[resultado['clase_secundaria']]['name']} "
                                f"({resultado['confianza_secundaria']*100:.1f}%). Se recomienda inspeccion adicional. \n\n"
                                + recomendacion
                            )
                        new_item = {
                            "id": f"DIAG-{int(datetime.now().timestamp())}",
                            "primaryDisease": resultado["clase"],
                            "primaryConfidence": resultado["confianza"],
                            "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
                            "recommendation": recomendacion,
                        }
                        st.session_state.history.insert(0, new_item)
                        st.session_state.current_diag = new_item
                        guardar_historial()
                        st.success("¡Diagnostico completado con exito!")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:10px; font-weight:bold; letter-spacing:1px; color:#8C7D73;'>RETROALIMENTACION:</p>", unsafe_allow_html=True)
        q_comment = st.text_input("Comentario para mejora del modelo:", placeholder="Escriba su comentario para mejora del modelo...", label_visibility="collapsed")
        if st.button("ENVIAR RETROALIMENTACION"):
            if q_comment:
                diag_rel = st.session_state.current_diag["primaryDisease"] if st.session_state.current_diag else None
                success, msg = enviar_feedback_github(q_comment, diag_rel)
                if success:
                    st.toast("Comentario registrado en GitHub Issues.", icon="✅")
                else:
                    st.warning(f"No se pudo enviar a GitHub: {msg}. Intente mas tarde.")
            else:
                st.warning("Escribe un comentario antes de enviar.")

    # RIGHT PANEL: ULTIMO DIAGNOSTICO & RECOMENDACION
    with col_right:
        cd = st.session_state.current_diag

        if cd is None:
            st.markdown(f"""
            <div class="recommendation-box" style="text-align:center;">
                <p style="font-size:13px; color:{text_sub}; margin:0;">
                    Aun no se ha realizado ningun diagnostico. Sube una fotografia de una hoja de cafe
                    y presiona <strong>DIAGNOSTICAR AHORA</strong> para comenzar.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            d_info = COFFEE_DISEASES[cd["primaryDisease"]]
            conf_percent = f"{cd['primaryConfidence']*100:.1f}%"

            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <span style="font-size:10px; font-weight:bold; letter-spacing:1px; color:#8C7D73;">ULTIMO DIAGNOSTICO</span>
                <span style="font-size:11px; font-family:monospace; color:#9CA3AF;">{cd['timestamp']}</span>
            </div>
            """, unsafe_allow_html=True)

            col_title, col_perc = st.columns([2, 1])
            with col_title:
                st.markdown(f"""
                <h1 style="font-family: 'Playfair Display', serif; margin:0; font-size: 32px;">{d_info['name']}</h1>
                <p style="font-size: 12px; font-style: italic; color: #8C7D73; margin-top: 4px;">{d_info.get('scientificName', 'Coffea arabica')} • Detectado recientemente</p>
                """, unsafe_allow_html=True)
            with col_perc:
                st.markdown(f"""
                <div style="text-align: right;">
                    <span style="font-family: 'Playfair Display', serif; font-size: 36px; font-weight: bold;">{conf_percent}</span>
                    <p style="font-size: 10px; font-weight: bold; color: #9CA3AF; text-transform: uppercase;">CONFIANZA IA</p>
                </div>
                """, unsafe_allow_html=True)

            # Recommendation box
            st.markdown(f"""
            <div class="recommendation-box">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
                    <span style="background-color:#2C1A11; color:#FCD34D; border-radius:50%; width:20px; height:20px; display:inline-flex; justify-content:center; align-items:center; font-size:10px; font-weight:bold;">Q</span>
                    <span style="font-size:10px; font-weight:bold; letter-spacing:1px; color:#8C7D73;">RECOMENDACION GROQ AI / IHCAFE</span>
                </div>
                <p style="font-size:13px; font-style:italic; line-height:1.6; border-left: 2px solid #2C1A11; padding-left: 12px; margin:0; white-space: pre-line;">
                    "{cd['recommendation']}"
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Historial Reciente List
            st.markdown("""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:20px; margin-bottom:12px;">
                <span style="font-size:10px; font-weight:bold; letter-spacing:1px; color:#8C7D73;">HISTORIAL RECIENTE</span>
            </div>
            """, unsafe_allow_html=True)

            for item in st.session_state.history[:3]:
                item_d = COFFEE_DISEASES[item["primaryDisease"]]
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center; padding: 12px 16px; background-color:{card_bg}; border: 1px solid {border_color}; border-radius:14px; margin-bottom:8px;">
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span style="width:10px; height:10px; border-radius:50%; background-color:{item_d['color']}; display:inline-block;"></span>
                        <span style="font-size:13px; font-weight:600;">{item_d['name']}</span>
                    </div>
                    <span style="font-size:11px; font-family:monospace; color:#9CA3AF;">{item['timestamp']}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color:" + border_color + "; margin-top:24px;'>", unsafe_allow_html=True)
        st.caption("© 2026 AGRODETECT • SOPORTE IHCAFE")

# ----------------------------------------------------
# 2. TAB: HISTORIAL
# ----------------------------------------------------
elif tab_choice == "HISTORIAL":
    st.subheader("📜 Historial de Diagnosticos Foliares")
    if not st.session_state.history:
        st.info("Aun no hay diagnosticos registrados. Ve a la pestana DIAGNOSTICO para analizar una hoja.")
    for item in st.session_state.history:
        d_meta = COFFEE_DISEASES[item["primaryDisease"]]
        st.markdown(f"""
        <div style="padding:16px; background-color:{card_bg}; border:1px solid {border_color}; border-radius:16px; margin-bottom:12px;">
            <span style="background-color:{d_meta['color']}; color:white; font-size:10px; font-weight:bold; padding:4px 10px; border-radius:12px;">{d_meta['name']}</span>
            <span style="float:right; font-size:11px; color:gray;">{item['timestamp']}</span>
            <h4 style="margin-top:10px; margin-bottom:4px;">Certeza: {item['primaryConfidence']*100:.1f}%</h4>
            <p style="font-size:12px; color:{text_sub};">{item['recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------
# 3. TAB: GUIAS TECNICAS
# ----------------------------------------------------
elif tab_choice == "GUIAS TECNICAS":
    st.subheader("📚 Guia Tecnica de Plagas y Enfermedades IHCAFE")
    for k, info in COFFEE_DISEASES.items():
        with st.expander(f"🍂 {info['name']} ({info.get('scientificName', 'N/A')})"):
            st.write(f"**Categoria:** {info['category']}")
            st.write(f"**Gravedad:** {info['severity']}")
            st.write("**Sintomatologia:**")
            for s in info["symptoms"]:
                st.write(f"- {s}")
            st.info(f"**Manejo Agronomico:** {info['recommendation']}")

# ----------------------------------------------------
# 4. TAB: GITHUB
# ----------------------------------------------------
elif tab_choice == "GITHUB":
    st.subheader("💬 Registro de Control de Calidad GitHub")
    st.write("Cada observacion queda registrada como un Issue en GitHub con la etiqueta `feedback`.")

    if not GITHUB_TOKEN or not GITHUB_REPO:
        st.error("⚠️ No se han configurado los secrets `GITHUB_TOKEN` y `GITHUB_REPO` en Streamlit Cloud. La retroalimentacion no se guardara de forma persistente.")
    else:
        st.info(f"Conectado al repositorio: `{GITHUB_REPO}`")

    feedback_list = obtener_feedback_github()

    if not feedback_list:
        st.info("Aun no hay retroalimentacion registrada. Usa el campo de comentarios en la pestana DIAGNOSTICO.")
    else:
        for fb in feedback_list:
            estado_icon = "🟢" if fb['estado'] == 'open' else "🔴"
            st.markdown(f"""
            <div style="padding:14px; background-color:{card_bg}; border:1px solid {border_color}; border-radius:14px; margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:11px; color:#9CA3AF; font-family:monospace;">{fb['fecha']} {estado_icon}</span>
                    <a href="{fb['url']}" target="_blank" style="font-size:11px; color:#D97706;">Ver en GitHub →</a>
                </div>
                <p style="font-size:13px; margin:6px 0 0 0;"><strong>{fb['comentario']}</strong></p>
                <p style="font-size:11px; color:{text_sub}; margin:2px 0 0 0;">Diagnostico: {fb['diagnostico_relacionado']}</p>
            </div>
            """, unsafe_allow_html=True)
