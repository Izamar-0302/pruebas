import streamlit as st
import base64
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from io import BytesIO

import numpy as np
import requests
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm

# ====================================================
# AGRODETECT v2.0 - CAFICULTURA HONDURAS (IHCAFE)
# ====================================================

st.set_page_config(
    page_title="AgroDetect v2.0 - Diagnostico Foliar de Cafe",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ----------------------------------------------------
# HELPER: RENDER HTML SEGURO (FIX BUG DE INDENTACION)
# ----------------------------------------------------
def render_html(html_content: str):
    """
    Renderiza HTML con st.markdown de forma segura.

    Streamlit/Markdown interpreta cualquier linea indentada 4+ espacios
    como un bloque de codigo, lo cual hace que el HTML se muestre como
    texto plano (las etiquetas <div>, <span>, etc. aparecen literalmente
    en pantalla) en vez de renderizarse. Esta funcion quita la indentacion
    de cada linea antes de pasarla a st.markdown para evitar ese problema.
    """
    html_content = "\n".join(line.strip() for line in html_content.split("\n"))
    st.markdown(html_content, unsafe_allow_html=True)


def construir_recomendacion_html(rec_data, text_sub_color):
    """
    Construye el HTML del bloque de recomendacion (numerado, con titulos y texto)
    a partir de la lista estructurada, o de un texto plano de respaldo.
    Reutilizado tanto en el panel de Diagnostico como en el Historial.
    """
    if isinstance(rec_data, list) and len(rec_data) > 0:
        html = f"""
        <div class="rec-container">
            <div class="rec-header">
                <span class="rec-header-icon">💡</span>
                <span class="rec-header-label">ORIENTACION Y MANEJO PREVENTIVO</span>
            </div>
            <p style="font-size:12px; color:{text_sub_color}; margin-bottom:12px;">
                Aqui tienes una recomendacion tecnica detallada para manejar la situacion:
            </p>
        """
        for sec in rec_data:
            html += f"""
            <div class="rec-section">
                <div class="rec-num">{sec.get('num', '01')}</div>
                <div>
                    <div class="rec-title">{sec.get('titulo', '')}</div>
                    <p class="rec-text">{sec.get('texto', '')}</p>
                </div>
            </div>
            """
        html += "</div>"
        return html
    txt = rec_data if isinstance(rec_data, str) and rec_data else "Sin recomendacion disponible."
    return f"""
    <div class="rec-container">
        <p style="font-size:13px; line-height:1.6; margin:0; white-space: pre-line;">{txt}</p>
    </div>
    """


# ----------------------------------------------------
# CAPTURA DE FOTO CON WIDGET NATIVO DE STREAMLIT
# ----------------------------------------------------
def capturar_foto_camara(key="camara_agrodetect"):
    """
    Muestra el widget nativo de camara de Streamlit y devuelve una PIL.Image
    si el usuario capturo una foto, o None si aun no ha capturado nada.
    """
    foto = st.camera_input("Toma una foto de la hoja de café", key=key, label_visibility="collapsed")
    if foto is not None:
        return Image.open(foto)
    return None


# ----------------------------------------------------
# ALMACENAMIENTO DE IMAGENES DE DIAGNOSTICO
# ----------------------------------------------------
IMAGES_DIR = "diagnosis_images"
os.makedirs(IMAGES_DIR, exist_ok=True)


def guardar_imagen_diagnostico(imagen_pil, diag_id):
    """
    Guarda la imagen foliar asociada a un diagnostico en disco (JPEG)
    y devuelve la ruta relativa del archivo guardado.
    """
    try:
        ruta = os.path.join(IMAGES_DIR, f"{diag_id}.jpg")
        imagen_pil.convert("RGB").save(ruta, "JPEG", quality=88)
        return ruta
    except Exception:
        return None


# ----------------------------------------------------
# EXPORTAR DIAGNOSTICO(S) A PDF
# ----------------------------------------------------
def generar_pdf_diagnosticos(diagnosticos):
    """
    Genera un PDF con uno o varios diagnosticos (lista de items del historial).
    Devuelve un BytesIO listo para descargar con st.download_button.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=2*cm, bottomMargin=2*cm, leftMargin=2*cm, rightMargin=2*cm
    )
    styles = getSampleStyleSheet()
    titulo_style = ParagraphStyle("TituloReporte", parent=styles["Title"], fontSize=20, spaceAfter=6)
    subtitulo_style = ParagraphStyle("Subtitulo", parent=styles["Normal"], fontSize=10,
                                      textColor=colors.HexColor("#6B5E55"), spaceAfter=20)
    seccion_titulo_style = ParagraphStyle("SeccionTitulo", parent=styles["Heading2"], fontSize=14,
                                           textColor=colors.HexColor("#2C1A11"), spaceBefore=10, spaceAfter=6)
    campo_style = ParagraphStyle("Campo", parent=styles["Normal"], fontSize=11, spaceAfter=4)
    rec_titulo_style = ParagraphStyle("RecTitulo", parent=styles["Normal"], fontSize=11,
                                       fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=2)
    rec_texto_style = ParagraphStyle("RecTexto", parent=styles["Normal"], fontSize=10,
                                      textColor=colors.HexColor("#333333"), spaceAfter=6, leading=14)

    story = [
        Paragraph("AgroDetect - Informe de Diagnostico Foliar", titulo_style),
        Paragraph(f"Generado el {datetime.now().strftime('%d/%m/%Y %H:%M')} · Soporte IHCAFE", subtitulo_style),
    ]

    for i, diag in enumerate(diagnosticos):
        info = COFFEE_DISEASES.get(diag["primaryDisease"], COFFEE_DISEASES["sana"])
        conf = f"{diag['primaryConfidence']*100:.1f}%"

        story.append(Paragraph(f"Diagnostico #{i+1}: {info['name']}", seccion_titulo_style))

        # ---- Imagen de la hoja analizada (si existe) ----
        img_path = diag.get("image_path")
        if img_path and os.path.exists(img_path):
            try:
                pil_img = Image.open(img_path)
                img_w, img_h = pil_img.size
                max_w = 8 * cm
                max_h = 7 * cm
                ratio = min(max_w / img_w, max_h / img_h)
                rl_img = RLImage(img_path, width=img_w * ratio, height=img_h * ratio)
                rl_img.hAlign = "LEFT"
                story.append(rl_img)
                story.append(Spacer(1, 10))
            except Exception:
                pass

        story.append(Paragraph(f"<b>Nombre cientifico:</b> {info.get('scientificName', 'N/A')}", campo_style))
        story.append(Paragraph(f"<b>Categoria:</b> {info['category']}", campo_style))
        story.append(Paragraph(f"<b>Gravedad:</b> {info['severity']}", campo_style))
        story.append(Paragraph(f"<b>Confianza IA:</b> {conf}", campo_style))
        story.append(Paragraph(f"<b>Fecha:</b> {diag['timestamp']}", campo_style))

        if diag.get("coinfeccion"):
            story.append(Paragraph(f"<b>Nota:</b> {diag['coinfeccion']}", campo_style))

        story.append(Spacer(1, 8))

        rec_data = diag.get("recommendation", [])
        if isinstance(rec_data, list):
            for sec in rec_data:
                story.append(Paragraph(f"{sec.get('num', '')}. {sec.get('titulo', '')}", rec_titulo_style))
                story.append(Paragraph(sec.get('texto', ''), rec_texto_style))
        elif isinstance(rec_data, str) and rec_data:
            story.append(Paragraph(rec_data, rec_texto_style))

        if i < len(diagnosticos) - 1:
            story.append(PageBreak())

    doc.build(story)
    buffer.seek(0)
    return buffer


# ----------------------------------------------------
# CONFIGURACION GITHUB
# ----------------------------------------------------
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", os.environ.get("GITHUB_TOKEN"))
GITHUB_REPO = st.secrets.get("GITHUB_REPO", os.environ.get("GITHUB_REPO"))

# ----------------------------------------------------
# MODELO
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
    if os.path.exists(MODEL_PATH_KERAS):
        temp_path = os.path.join(tempfile.gettempdir(), "agrodetect_temp.keras")
        shutil.copy(MODEL_PATH_KERAS, temp_path)
        return keras.models.load_model(temp_path)
    if os.path.exists(CONFIG_PATH) and os.path.exists(MODEL_PATH_H5):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        model = keras.models.model_from_json(json.dumps(config))
        model.load_weights(MODEL_PATH_H5)
        return model
    raise FileNotFoundError("No se encontro el modelo.")


@st.cache_data(show_spinner=False)
def cargar_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        clases = config.get("clases", CLASES_DEFAULT)
        img_size = tuple(config.get("img_size", IMG_SIZE_DEFAULT))
        umbral = config.get("umbral_decision", UMBRAL_DEFAULT)
        return clases, img_size, umbral
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


def limpiar_respuesta_groq(texto):
    """Limpia bloques de codigo markdown y extrae JSON o texto plano."""
    # Quitar bloques ```json ... ``` o ```html ... ``` o ``` ... ```
    texto = re.sub(r'```(?:json|html)?\s*', '', texto, flags=re.IGNORECASE)
    texto = re.sub(r'\s*```', '', texto)
    texto = texto.strip()
    return texto


def parsear_recomendacion(texto):
    """Intenta parsear JSON, luego markdown ## 01, luego HTML. Si todo falla, devuelve None."""
    texto_limpio = limpiar_respuesta_groq(texto)

    # 1. Intentar JSON
    try:
        data = json.loads(texto_limpio)
        if isinstance(data, list) and len(data) >= 3:
            secciones = []
            for item in data:
                secciones.append({
                    "num": str(item.get("num", "01")).zfill(2),
                    "titulo": item.get("titulo", item.get("title", "")),
                    "texto": item.get("texto", item.get("text", item.get("contenido", "")))
                })
            return secciones
    except Exception:
        pass

    # 2. Intentar HTML con rec-section
    if '<div class="rec-section"' in texto_limpio or "<div class='rec-section'" in texto_limpio:
        secciones = []
        # Extraer bloques rec-section de forma robusta
        pattern = r'<div\s+class=["\']rec-section["\']\s*>(.*?)</div>\s*</div>\s*</div>'
        blocks = re.findall(pattern, texto_limpio, re.DOTALL)
        if not blocks:
            pattern2 = r'<div\s+class=["\']rec-section["\']\s*>(.*?)</div>\s*</div>'
            blocks = re.findall(pattern2, texto_limpio, re.DOTALL)
        for block in blocks:
            num_m = re.search(r'<div\s+class=["\']rec-num["\'][^>]*>(\d+)</div>', block)
            title_m = re.search(r'<div\s+class=["\']rec-title["\'][^>]*>(.*?)</div>', block)
            text_m = re.search(r'<p\s+class=["\']rec-text["\'][^>]*>(.*?)</p>', block)
            if num_m and title_m:
                secciones.append({
                    "num": num_m.group(1).zfill(2),
                    "titulo": re.sub(r'<[^>]+>', '', title_m.group(1)).strip(),
                    "texto": re.sub(r'<[^>]+>', '', text_m.group(1)).strip() if text_m else ""
                })
        if secciones:
            return secciones

    # 3. Intentar Markdown ## 01. Titulo
    secciones = []
    patron = r'(?:##\s*)?(\d{1,2})[\.\)]\s*(.+?)(?=\n(?:##\s*)?\d{1,2}[\.\)]|\Z)'
    matches = list(re.finditer(patron, texto_limpio, re.DOTALL))
    for m in matches:
        num = m.group(1).zfill(2)
        lines = m.group(2).strip().split('\n')
        titulo = lines[0].strip()
        texto_sec = '\n'.join(lines[1:]).strip()
        secciones.append({"num": num, "titulo": titulo, "texto": texto_sec})
    if len(secciones) >= 3:
        return secciones

    return None


def obtener_recomendacion(clase, clase_secundaria=None):
    """Genera recomendacion estructurada con Groq (JSON); si falla, usa respaldo."""
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    respaldo = RECOMENDACIONES_ESTRUCTURADAS.get(clase, RECOMENDACIONES_ESTRUCTURADAS["sana"])

    if not api_key:
        return respaldo

    nombre = COFFEE_DISEASES[clase]["name"]
    categoria = COFFEE_DISEASES[clase]["category"]

    extra = ""
    if clase_secundaria:
        extra = f" NOTA: tambien detecto signos de {COFFEE_DISEASES[clase_secundaria]['name']}. Incluye breve diferenciacion."

    prompt = (
        "Eres agronomo senior IHCAFE. Diagnostico: " + nombre + " (" + categoria + "). "
        "Responde UNICAMENTE con un array JSON valido de 5 objetos. "
        "NO uses markdown, NO uses HTML, NO uses bloques de codigo. Solo JSON puro.\n\n"
        "Formato exacto:\n"
        '[{"num":"01","titulo":"Diferenciacion a simple vista","texto":"3-5 oraciones..."},'
        '{"num":"02","titulo":"Manejo agronomico preventivo y correctivo","texto":"..."},'
        '{"num":"03","titulo":"Consulta a un tecnico IHCAFE","texto":"..."},'
        '{"num":"04","titulo":"Monitoreo y seguimiento","texto":"..."},'
        '{"num":"05","titulo":"Registro y trazabilidad","texto":"..."}]\n\n'
        "Cada 'texto' debe tener 3-5 oraciones tecnicas y detalladas."
        + extra
    )

    try:
        from groq import Groq
        cliente = Groq(api_key=api_key)
        resp = cliente.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1200,
        )
        texto = resp.choices[0].message.content.strip()
        parsed = parsear_recomendacion(texto)
        if parsed and len(parsed) >= 3:
            return parsed
        return respaldo
    except Exception as e:
        st.warning(f"Groq no disponible ({e}). Usando recomendacion curada.")
        return respaldo


# ============================================================
# ALMACENAMIENTO DE COMENTARIOS (backend: GitHub Issues, uso interno)
# ============================================================
def enviar_feedback_github(comentario, diagnostico_relacionado=None):
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return False, "El servicio de comentarios no esta configurado."
    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    title = f"[Feedback] {comentario[:50]}{'...' if len(comentario) > 50 else ''}"
    body = f"**Comentario:** {comentario}\n\n**Diagnostico:** {diagnostico_relacionado or 'N/A'}\n\n**Fecha:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n_Enviado desde AgroDetect v2.0_"
    try:
        r = requests.post(url, headers=headers, json={"title": title, "body": body, "labels": ["feedback"]}, timeout=15)
        return r.status_code == 201, "OK" if r.status_code == 201 else f"Error {r.status_code}"
    except Exception as e:
        return False, str(e)


def obtener_feedback_github():
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return []
    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues"
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    try:
        r = requests.get(url, headers=headers, params={"labels": "feedback", "state": "all", "per_page": 30}, timeout=15)
        if r.status_code != 200:
            return []
        items = []
        for issue in r.json():
            body = issue.get("body", "")
            diag = "N/A"
            for line in body.split("\n"):
                if "**Diagnostico:**" in line:
                    diag = line.split("**Diagnostico:**")[-1].strip()
                    break
            items.append({
                "comentario": issue.get("title", "").replace("[Feedback] ", ""),
                "diagnostico_relacionado": diag,
                "fecha": datetime.strptime(issue["created_at"], "%Y-%m-%dT%H:%M:%SZ").strftime("%d/%m/%Y %H:%M"),
                "estado": issue.get("state", "open")
            })
        return items
    except Exception:
        return []


# ----------------------------------------------------
# METADATA ENFERMEDADES
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
        "recommendation": "Se observa una infeccion severa. Se recomienda poda sanitaria inmediata y aplicacion de fungicidas a base de oxicloruro de cobre."
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
        "recommendation": "Reforzar la fertilizacion nitrogenada foliar e implementar regulacion de sombra."
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
        "recommendation": "Establecer cortinas rompevientos, reducir humedad relativa y aplicar fungicidas especificos autorizados por IHCAFE."
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
        "recommendation": "Aplicar acaricidas especificos de origen botanico o azufre mojable."
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
        "recommendation": "Realizar control biologico favoreciendo avispitas parasitoides."
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
        "recommendation": "Tejido foliar sano. Mantener plan de nutricion y monitoreos rutinarios."
    }
}

RECOMENDACIONES_ESTRUCTURADAS = {
    "roya": [
        {"num": "01", "titulo": "Diferenciacion a simple vista", "texto": "La Roya (Hemileia vastatrix) se identifica por pustulas anaranjadas en el enves de la hoja y manchas amarillentas en el haz. No confundir con Cercospora: la Roya tiene pustulas en relieve con esporas al tacto, mientras que Cercospora presenta manchas circulares planas con centro grisaceo. En estadios avanzados, la hoja se torna necrotica y cae prematuramente."},
        {"num": "02", "titulo": "Manejo agronomico preventivo y correctivo", "texto": "Aplicar fungicidas protectores a base de oxicloruro de cobre (2.5 g/L) o caldo bordeles (0.5%) cada 21-30 dias en epoca lluviosa. Realizar poda sanitaria eliminando ramas con >50% de incidencia. Mejorar drenaje de suelos para reducir humedad foliar >90%. En zonas severas, usar fungicidas sistemicos autorizados por IHCAFE alternando mecanismos de accion."},
        {"num": "03", "titulo": "Consulta a un tecnico IHCAFE", "texto": "Acuda al tecnico de IHCAFE si la incidencia supera el 10% de la parcela o si observa defoliacion masiva. Lleve muestras de hojas con pustulas frescas en bolsas de papel (no plastico) y registre fotografias de la distribucion en la planta. El tecnico podra confirmar si se trata de la raza IIv5 y ajustar el protocolo."},
        {"num": "04", "titulo": "Monitoreo y seguimiento", "texto": "Inspeccione semanalmente durante la transicion de lluvias a seca (mayo-julio), que es el pico de infeccion. Evalue 50 plantas al azar por hectarea. Indicadores de mejora: nuevo brote sin pustulas, reduccion de esporulacion. Indicadores de empeora: expansion a brotes terminales, aparicion en frutos."},
        {"num": "05", "titulo": "Registro y trazabilidad", "texto": "Lleve una bitacora con fecha de deteccion, porcentaje de incidencia, producto aplicado (nombre comercial e ingrediente activo), dosis, volumen de caldo por hectarea y condiciones climaticas. Esto permitira identificar resistencias, optimizar costos y cumplir con certificaciones BPA exigidas por compradores."}
    ],
    "cercospora": [
        {"num": "01", "titulo": "Diferenciacion a simple vista", "texto": "Cercospora coffeicola produce manchas circulares de 3-8 mm con centro grisaceo-necrotico y halo amarillo-anaranjado. A diferencia de la Roya, no hay pustulas en relieve ni esporas en el enves. Se confunde frecuentemente con manchas de nutricion (deficiencia de Mn), pero estas carecen del halo definido y son mas irregulares."},
        {"num": "02", "titulo": "Manejo agronomico preventivo y correctivo", "texto": "Aumentar fertilizacion nitrogenada (urea foliar al 2%) y potasica. Regular sombra al 40-50% para reducir estres hidrico. Aplicar caldo bordeles preventivo antes de lluvias intensas. En brotes activos, usar fungicidas especificos con mancozeb o clorotalonil. Evitar trabajos en campo con follaje mojado para no diseminar esporas."},
        {"num": "03", "titulo": "Consulta a un tecnico IHCAFE", "texto": "Consulte si las manchas aparecen en mas del 30% del follaje o si persisten tras dos aplicaciones fungicidas. El tecnico analizara niveles de N y K en suelo/hoja para descartar que sea un problema nutricional primario que debilite el tejido y facilite la infeccion."},
        {"num": "04", "titulo": "Monitoreo y seguimiento", "texto": "Monitoree quincenalmente en epocas secas y calurosas (febrero-abril), cuando el estres hidrico potencia la enfermedad. Revise hojas del tercio medio de la planta. Mejora: nuevas hojas sin manchas, recuperacion del color verde intenso. Empeora: coalescencia de manchas, secado de bordes foliares."},
        {"num": "05", "titulo": "Registro y trazabilidad", "texto": "Registre analisis foliar bianuales, niveles de sombra (% cobertura), tipo de sombra (Inga, Erythrina, malla), fecha de aplicaciones y condiciones climaticas previas. Documente si la parcela esta en ladera expuesta al sol (mayor riesgo). Estos datos son clave para ajustar el manejo integral del cultivo."}
    ],
    "phoma": [
        {"num": "01", "titulo": "Diferenciacion a simple vista", "texto": "Phoma costarricensis afecta principalmente zonas altas (>1300 msnm). Se manifiesta como lesiones oscuras, irregulares, en bordes y apices de hojas jovenes, con aspecto de quemadura o derretimiento. A diferencia de Roya o Cercospora, las lesiones son asimetricas, sin halo definido, y progresan rapidamente en brotes tiernos tras lluvias frias."},
        {"num": "02", "titulo": "Manejo agronomico preventivo y correctivo", "texto": "Establecer cortinas rompevientos (Cypress, Eucalyptus o bambu) para reducir viento frio-humedo. Poda de formacion para mejorar aireacion. Aplicar fungicidas protectores (oxicloruro de cobre + mancozeb) antes de frentes frios. Eliminar brotes afectados con corte limpio y desinfeccion de herramientas con alcohol al 70% entre plantas."},
        {"num": "03", "titulo": "Consulta a un tecnico IHCAFE", "texto": "Obligatorio en zonas altas si la mortalidad de brotes supera el 5%. El tecnico evaluara si se requiere cambio de variedad a una mas tolerante (ej. Lempira, Parainema) o si hay que reforzar el sistema de cortinas. Lleve registro de temperaturas minimas y horas de humedad relativa >85%."},
        {"num": "04", "titulo": "Monitoreo y seguimiento", "texto": "Vigile diariamente tras frentes frios o neblina persistente. La enfermedad progresa en 48-72 horas bajo esas condiciones. Indicadores de control: brotes nuevos con hojas intactas, ausencia de lesiones en el tercio superior. Alarma: presencia de picnidios negros en lesiones maduras."},
        {"num": "05", "titulo": "Registro y trazabilidad", "texto": "Documente altitud exacta (GPS), exposicion geografica, velocidad del viento dominante, tipo de cortina existente y densidad de siembra. Registre fechas de heladas o neblina prolongada. Esta informacion permite a IHCAFE mapear zonas de riesgo y recomendar manejo diferenciado por microclima."}
    ],
    "arana_roja": [
        {"num": "01", "titulo": "Diferenciacion a simple vista", "texto": "Oligonychus coffeae causa broncizado uniforme en el haz de la hoja (cambio a tono cafe-rojizo), a diferencia de la Roya que es irregular y por el enves. En el enves se observan puntos oscuros (acaros) y telaranas finas en condiciones de sequia. No confundir con deficiencia de hierro, que es amarillamiento interveinal uniforme sin broncizado."},
        {"num": "02", "titulo": "Manejo agronomico preventivo y correctivo", "texto": "Aplicar azufre mojable (2-3 g/L) o acaricidas botanicos a base de aceite de nim (3-5 mL/L) en el enves de las hojas. Incrementar humedad relativa con riego por aspersion (evita telaranas). Evitar insecticidas de amplio espectro (organofosforados, piretroides) que eliminan depredadores naturales como Stethorus y Amblyseius."},
        {"num": "03", "titulo": "Consulta a un tecnico IHCAFE", "texto": "Solicite asistencia si la poblacion supera 10 acaros por hoja o si el broncizado afecta mas del 20% del follaje. El tecnico realizara conteos con lupa de campo y determinara si se requiere acaricida quimico especifico (etoxazole, spiromesifen) respetando periodos de carencia para cosecha."},
        {"num": "04", "titulo": "Monitoreo y seguimiento", "texto": "Inspeccione semanalmente en epoca seca (noviembre-abril), cuando la poblacion se dispara. Use una lupa de 10x en el enves de 20 hojas por planta (muestreo en 4 puntos cardinales de la parcela). Control efectivo: <3 acaros/hoja y presencia de acaros depredadores. Alarma: telaranas abundantes, hojas secas permaneciendo en la planta."},
        {"num": "05", "titulo": "Registro y trazabilidad", "texto": "Registre conteos de acaros/plaga y acaros beneficios por fecha, producto aplicado, pH del agua de caldo (debe ser 6.0-7.0 para azufre), y estacion climatica. Documente si hay cultivos vecinos (maiz, frijol) que puedan hospedar la plaga. Esto ayuda a predecir brotes futuros y programar liberaciones de enemigos naturales."}
    ],
    "minador": [
        {"num": "01", "titulo": "Diferenciacion a simple vista", "texto": "Leucoptera coffeella crea galerias serpenteantes transparentes o marrones en el mesofilo de la hoja (minas), visibles a contraluz. El enrollado foliar prematuro es caracteristico. Diferenciar de danos mecanicos o de trips: las minas tienen trayectoria definida y frass (excrementos) en linea dentro de la galeria, visible con lupa."},
        {"num": "02", "titulo": "Manejo agronomico preventivo y correctivo", "texto": "Favorecer biodiversidad floral (boton de oro, cilantro) para atraer avispas parasitoides (Cirrospilus, Closterocerus). En umbrales criticos (>5% de hojas minadas), aplicar Bacillus thuringiensis kurstaki o extractos de nim. Recolectar y destruir hojas severamente minadas (no compostar). Mantener sombra moderada; el minador prospera en exceso de sol."},
        {"num": "03", "titulo": "Consulta a un tecnico IHCAFE", "texto": "Consulte si la incidencia supera el 15% de hojas afectadas o si hay brotes con enrollado masivo. El tecnico evaluara la presencia de parasitoides (hojas con agujeros de salida del parasitoide) y determinara si es viable una liberacion masiva de enemigos naturales en lugar de quimicos."},
        {"num": "04", "titulo": "Monitoreo y seguimiento", "texto": "Revise quincenalmente hojas del tercio medio-inferior. Busque huevos en el haz (puntos blancos) y minas frescas (verdes-translucidas vs. marrones viejas). Control exitoso: >30% de minas con agujeros de salida de parasitoides. Empeora: minas en hojas del brote terminal, reduccion de area foliar fotosintetica >20%."},
        {"num": "05", "titulo": "Registro y trazabilidad", "texto": "Lleve registro de % hojas minadas, % parasitismo estimado, presencia de enemigos naturales, aplicaciones biologicas realizadas y condiciones climaticas (temperatura, precipitacion). Este seguimiento permite a IHCAFE validar umbrales de accion especificos para su microregion y ajustar calendarios de liberacion de parasitoides."}
    ],
    "sana": [
        {"num": "01", "titulo": "Diferenciacion a simple vista", "texto": "Hoja sana presenta color verde intenso y uniforme, venacion bien definida, epidermis intacta sin pustulas, manchas, minas ni broncizado. El borde foliar es liso, sin necrosis. Asegurese de que no sea una hoja asintomatica en etapa muy temprana de infeccion: revise el enves con lupa para descartar presencia de acaros o esporas incipientes."},
        {"num": "02", "titulo": "Manejo agronomico preventivo y correctivo", "texto": "Mantener fertilizacion balanceada N-P-K segun analisis de suelo y foliar. Aplicar cal dolomitica si pH <5.5 para mejorar absorcion de Ca y Mg. Realizar poda de mantenimiento anual. Mantener sombra al 30-40% para proteger de estres termico sin generar exceso de humedad. Aplicar fungicidas preventivos antes de epocas de lluvia intensa si hay historial de roya en la zona."},
        {"num": "03", "titulo": "Consulta a un tecnico IHCAFE", "texto": "Programar visita tecnica semestral para analisis foliar completo (N, P, K, Ca, Mg, S, B, Zn, Mn, Fe) y evaluacion del sistema de sombra. El tecnico identificara deficiencias ocultas que predisponen al ataque futuro. Aproveche la visita para actualizar su calendario fitosanitario segun pronostico climatico."},
        {"num": "04", "titulo": "Monitoreo y seguimiento", "texto": "Realice inspecciones fitosanitarias quincenales durante la epoca de crecimiento activo (lluvias). Evalue 30 plantas distribuidas al azar. Parametros de salud: brotes con 2-3 pares de hojas sanas, ausencia de clorosis, firmeza foliar al tacto. Detecte a tiempo cualquier cambio: mancha, pustula o deformacion."},
        {"num": "05", "titulo": "Registro y trazabilidad", "texto": "Documente resultados de analisis de suelo/foliar, programa de fertilizacion, podas realizadas, control de malezas y monitoreos fitosanitarios. Mantenga historial fotografico de la evolucion del follaje. Esta documentacion es esencial para certificaciones de origen, trazabilidad de compradores y acceso a programas de IHCAFE de cafes de especialidad."}
    ]
}

HISTORY_FILE = "diagnosis_history.json"


def guardar_historial():
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ----------------------------------------------------
# SESSION STATE
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
# STYLING
# ----------------------------------------------------
is_dark = st.session_state.theme == "oscuro"
bg_body = "#140D0A" if is_dark else "#FAF7F2"
card_bg = "#1F1510" if is_dark else "#FFFFFF"
text_main = "#F5EFE9" if is_dark else "#2C1A11"
text_sub = "#B8ACA2" if is_dark else "#6B5E55"
border_color = "#36271D" if is_dark else "#EAE3D9"
highlight_bg = "#271B14" if is_dark else "#F5F1EA"
rec_box_bg = "#1A120D" if is_dark else "#F0EBE3"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
    .stApp {{ background-color: {bg_body}; color: {text_main}; }}

    [data-testid="stRadio"] label, [data-testid="stRadio"] label p, [data-testid="stRadio"] label span,
    [data-testid="stRadio"] div[role="radiogroup"] label {{ color: {text_main} !important; }}

    [data-testid="stTextInput"] > div > div > input, [data-testid="stTextArea"] > div > div > textarea {{
        background-color: {card_bg} !important; color: {text_main} !important;
        border: 1px solid {border_color} !important; border-radius: 12px !important;
    }}
    [data-testid="stTextInput"] input::placeholder, [data-testid="stTextArea"] textarea::placeholder {{
        color: {text_sub} !important; opacity: 0.7;
    }}

    [data-testid="stFileUploader"] section {{
        background-color: {card_bg} !important; border: 1px dashed {border_color} !important; border-radius: 16px !important;
    }}
    [data-testid="stFileUploader"] section > div > span, [data-testid="stFileUploader"] section > div > small {{
        color: {text_sub} !important;
    }}

    .stButton > button {{
        border-radius: 9999px !important; font-weight: 700 !important; font-size: 12px !important;
        background-color: {card_bg} !important; color: {text_main} !important; border: 1px solid {border_color} !important;
    }}
    .stButton > button:hover {{
        border-color: #D97706 !important; color: #D97706 !important; background-color: {highlight_bg} !important;
    }}
    .stButton > button[kind="primary"] {{
        background-color: {"#2C1A11" if not is_dark else "#F5EFE9"} !important;
        color: {"#FAF7F2" if not is_dark else "#2C1A11"} !important; border: none !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: #D97706 !important; color: #FFFFFF !important;
    }}

    h1, h2, h3, h4, h5, h6, p, label, .stCaption,
    [data-testid="stMarkdownContainer"] p, [data-testid="stHeading"] h1,
    [data-testid="stHeading"] h2, [data-testid="stHeading"] h3 {{ color: {text_main} !important; }}

    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p {{ color: {text_main} !important; }}

    .brand-title {{ font-family: 'Playfair Display', serif; font-size: 24px; font-weight: 700; color: {text_main}; }}
    .brand-version {{ font-size: 11px; font-family: monospace; color: #D97706; margin-left: 6px; }}
    .brand-bean {{
        display: inline-block; vertical-align: -3px; width: 17px; height: 17px; margin: 0 1px;
    }}

    .rec-section {{ display: flex; gap: 14px; align-items: flex-start; padding: 14px 0; border-bottom: 1px solid {border_color}; }}
    .rec-section:last-child {{ border-bottom: none; }}
    .rec-num {{
        min-width: 32px; height: 32px; border-radius: 10px; background-color: #2D5A3D;
        color: #FFFFFF; font-size: 12px; font-weight: 700;
        display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    }}
    .rec-title {{ font-size: 14px; font-weight: 700; color: {text_main}; margin-bottom: 4px; }}
    .rec-text {{ font-size: 13px; color: {text_sub}; line-height: 1.6; margin: 0; }}
    .rec-container {{
        background-color: {rec_box_bg}; border: 1px solid {border_color};
        border-radius: 20px; padding: 20px 24px; margin-top: 16px; margin-bottom: 20px;
    }}
    .rec-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }}
    .rec-header-icon {{
        background-color: #2C1A11; color: #FCD34D; border-radius: 50%;
        width: 20px; height: 20px; display: inline-flex; justify-content: center; align-items: center;
        font-size: 10px; font-weight: bold;
    }}
    .rec-header-label {{ font-size: 10px; font-weight: bold; letter-spacing: 1px; color: #8C7D73; }}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# ICONO: GRANO DE CAFE (reemplaza la "o" de AgroDetect)
# ----------------------------------------------------
BEAN_ICON = """<svg class="brand-bean" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <linearGradient id="beanBodyGrad" x1="10%" y1="5%" x2="90%" y2="100%">
            <stop offset="0%" stop-color="#B47C4E"/>
            <stop offset="45%" stop-color="#7A4A26"/>
            <stop offset="100%" stop-color="#3F2716"/>
        </linearGradient>
    </defs>
    <g transform="rotate(16 12 12)">
        <ellipse cx="12" cy="12" rx="6.1" ry="9.6" fill="url(#beanBodyGrad)"/>
        <path d="M12 3.6 C10.3 6.6 10 9.3 11.3 12 C10 14.7 10.3 17.4 12 20.4"
              stroke="#F2E2C8" stroke-width="1.5" fill="none" stroke-linecap="round" opacity="0.85"/>
        <ellipse cx="9.6" cy="6.6" rx="1.4" ry="2.3" fill="#F7ECDA" opacity="0.45"/>
    </g>
</svg>"""

# ----------------------------------------------------
# NAVBAR
# ----------------------------------------------------
col_nav1, col_nav2, col_nav3 = st.columns([2, 3, 2])
with col_nav1:
    render_html(f'<div style="display:flex; align-items:baseline;"><span class="brand-title">Agr{BEAN_ICON}Detect</span><span class="brand-version">v2.0</span></div>')
with col_nav2:
    tab_choice = st.radio("", ["DIAGNOSTICO", "GUIAS TECNICAS", "COMENTARIOS"], horizontal=True, label_visibility="collapsed")
with col_nav3:
    c1, c2 = st.columns([2, 1])
    with c1:
        cd_export = st.session_state.current_diag
        if cd_export is not None:
            pdf_buffer = generar_pdf_diagnosticos([cd_export])
            st.download_button(
                "📄 EXPORTAR PDF",
                data=pdf_buffer,
                file_name=f"diagnostico_{cd_export['id']}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            if st.button("📄 EXPORTAR PDF", use_container_width=True):
                st.warning("Aun no hay ningun diagnostico para exportar.")
    with c2:
        lbl = "🌙" if is_dark else "☀️"
        if st.button(lbl, use_container_width=True):
            st.session_state.theme = "claro" if is_dark else "oscuro"
            st.rerun()
render_html("<hr style='margin-top:0; margin-bottom:24px; border-color:" + border_color + ";'>")

# ----------------------------------------------------
# TAB: DIAGNOSTICO
# ----------------------------------------------------
if tab_choice == "DIAGNOSTICO":
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        render_html(f"""
        <h2 style="font-family: 'Playfair Display', serif; margin-bottom: 4px;">Captura de Imagen Foliar</h2>
        <p style="font-size: 13px; color: {text_sub}; margin-bottom: 24px;">
            Posicione la hoja de cafe bajo luz natural. El sistema detectara automaticamente signos de Roya, Cercospora o Plagas.
        </p>
        """)

        metodo_captura = st.radio(
            "Metodo de captura:",
            ["📁 Subir archivo", "📷 Usar camara"],
            horizontal=True,
            label_visibility="collapsed",
        )

        image = None
        if metodo_captura == "📁 Subir archivo":
            uploaded_img = st.file_uploader("Sube la imagen foliar (.jpg, .png):", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
            if uploaded_img is not None:
                image = Image.open(uploaded_img)
        else:
            image = capturar_foto_camara()

        if image is not None:
            st.image(image, use_container_width=True)
            if st.button("🚀 DIAGNOSTICAR AHORA", type="primary", use_container_width=True):
                with st.spinner("Analizando pigmentacion necrotica foliar..."):
                    x = preprocesar_imagen(image, IMG_SIZE)
                    probs = modelo.predict(x, verbose=0)[0]
                    resultado = interpretar_prediccion(probs, CLASES, UMBRAL)

                    if resultado["estado"] == "no_concluyente":
                        st.warning(
                            f"⚠️ Resultado no concluyente (confianza {resultado['confianza']*100:.1f}%). "
                            f"Toma otra foto con mejor enfoque/iluminacion o consulta a un tecnico IHCAFE."
                        )
                    else:
                        rec = obtener_recomendacion(resultado["clase"], resultado.get("clase_secundaria"))
                        coinf = None
                        if resultado["estado"] == "coinfeccion":
                            coinf = (
                                f"🔎 Posible coinfeccion con {COFFEE_DISEASES[resultado['clase_secundaria']]['name']} "
                                f"({resultado['confianza_secundaria']*100:.1f}%). Se recomienda inspeccion adicional."
                            )
                        diag_id = f"DIAG-{int(datetime.now().timestamp())}"
                        image_path = guardar_imagen_diagnostico(image, diag_id)
                        new_item = {
                            "id": diag_id,
                            "primaryDisease": resultado["clase"],
                            "primaryConfidence": resultado["confianza"],
                            "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
                            "recommendation": rec,
                            "coinfeccion": coinf,
                            "image_path": image_path,
                        }
                        st.session_state.history.insert(0, new_item)
                        st.session_state.current_diag = new_item
                        guardar_historial()
                        st.success("¡Diagnostico completado con exito!")
                        st.rerun()

        render_html("<br>")
        render_html("<p style='font-size:10px; font-weight:bold; letter-spacing:1px; color:#8C7D73;'>RETROALIMENTACION:</p>")
        q_comment = st.text_input("Comentario:", placeholder="Escriba su comentario para mejora del modelo...", label_visibility="collapsed")
        if st.button("ENVIAR RETROALIMENTACION"):
            if q_comment:
                diag_rel = st.session_state.current_diag["primaryDisease"] if st.session_state.current_diag else None
                ok, msg = enviar_feedback_github(q_comment, diag_rel)
                if ok:
                    st.toast("Comentario registrado.", icon="✅")
                else:
                    st.warning(f"No se pudo enviar el comentario: {msg}")
            else:
                st.warning("Escribe un comentario antes de enviar.")

    with col_right:
        cd = st.session_state.current_diag
        if cd is None:
            render_html(f"""
            <div class="rec-container" style="text-align:center;">
                <p style="font-size:13px; color:{text_sub}; margin:0;">
                    Aun no se ha realizado ningun diagnostico. Sube una fotografia y presiona <strong>DIAGNOSTICAR AHORA</strong>.
                </p>
            </div>
            """)
        else:
            d_info = COFFEE_DISEASES[cd["primaryDisease"]]
            conf_p = f"{cd['primaryConfidence']*100:.1f}%"

            render_html(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <span style="font-size:10px; font-weight:bold; letter-spacing:1px; color:#8C7D73;">ULTIMO DIAGNOSTICO</span>
                <span style="font-size:11px; font-family:monospace; color:#9CA3AF;">{cd['timestamp']}</span>
            </div>
            """)

            ct, cp = st.columns([2, 1])
            with ct:
                render_html(f"""
                <h1 style="font-family: 'Playfair Display', serif; margin:0; font-size: 32px;">{d_info['name']}</h1>
                <p style="font-size: 12px; font-style: italic; color: #8C7D73; margin-top: 4px;">{d_info.get('scientificName', 'Coffea arabica')} • Detectado recientemente</p>
                """)
            with cp:
                render_html(f"""
                <div style="text-align: right;">
                    <span style="font-family: 'Playfair Display', serif; font-size: 36px; font-weight: bold;">{conf_p}</span>
                    <p style="font-size: 10px; font-weight: bold; color: #9CA3AF; text-transform: uppercase;">CONFIANZA IA</p>
                </div>
                """)

            if cd.get("coinfeccion"):
                st.warning(cd["coinfeccion"])

            rec_data = cd.get("recommendation", [])
            render_html(construir_recomendacion_html(rec_data, text_sub))

            render_html("""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:20px; margin-bottom:12px;">
                <span style="font-size:10px; font-weight:bold; letter-spacing:1px; color:#8C7D73;">HISTORIAL RECIENTE</span>
            </div>
            """)

            for item in st.session_state.history[:3]:
                item_d = COFFEE_DISEASES[item["primaryDisease"]]
                render_html(f"""
                <div style="display:flex; justify-content:space-between; align-items:center; padding: 12px 16px; background-color:{card_bg}; border: 1px solid {border_color}; border-radius:14px; margin-bottom:8px;">
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span style="width:10px; height:10px; border-radius:50%; background-color:{item_d['color']}; display:inline-block;"></span>
                        <span style="font-size:13px; font-weight:600;">{item_d['name']}</span>
                    </div>
                    <span style="font-size:11px; font-family:monospace; color:#9CA3AF;">{item['timestamp']}</span>
                </div>
                """)

            render_html("<hr style='border-color:" + border_color + "; margin-top:24px;'>")
            st.caption("© 2026 AGRODETECT • SOPORTE IHCAFE")

# ----------------------------------------------------
# TAB: GUIAS TECNICAS
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
# TAB: COMENTARIOS
# ----------------------------------------------------
elif tab_choice == "COMENTARIOS":
    st.subheader("💬 Comentarios y Retroalimentacion")
    st.write("Aqui puedes ver los comentarios que se han dejado sobre los diagnosticos.")

    if not GITHUB_TOKEN or not GITHUB_REPO:
        st.error("⚠️ El servicio de comentarios no esta configurado.")

    feedback_list = obtener_feedback_github()
    if not feedback_list:
        st.info("Aun no hay comentarios registrados.")
    else:
        for fb in feedback_list:
            render_html(f"""
            <div style="padding:14px; background-color:{card_bg}; border:1px solid {border_color}; border-radius:14px; margin-bottom:10px;">
                <span style="font-size:11px; color:#9CA3AF; font-family:monospace;">{fb['fecha']}</span>
                <p style="font-size:13px; margin:6px 0 0 0;"><strong>{fb['comentario']}</strong></p>
                <p style="font-size:11px; color:{text_sub}; margin:2px 0 0 0;">Diagnostico: {fb['diagnostico_relacionado']}</p>
            </div>
            """)
