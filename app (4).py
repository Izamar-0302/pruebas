import streamlit as st
import json
import os
import re
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
    page_title="AgroDetect v2.0 - Diagnostico Foliar de Cafe",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------------------------------
# CONFIGURACION GITHUB (para retroalimentacion persistente)
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


def parsear_recomendacion_estructurada(texto):
    """Convierte texto con secciones ## 01. Titulo en lista de dicts.
    Tambien limpia bloques de codigo Markdown que Groq a veces envuelve."""

    # 1. Limpiar bloques de codigo Markdown que Groq a veces envuelve
    texto = re.sub(r'```html\s*', '', texto, flags=re.IGNORECASE)
    texto = re.sub(r'```\s*', '', texto)
    texto = texto.strip()

    # 2. Si el texto ya es HTML con rec-section, extraer datos del HTML
    if '<div class="rec-section"' in texto or "<div class='rec-section'" in texto:
        secciones = []
        # Extraer cada rec-section
        section_pattern = r'<div class=["\']rec-section["\'][^>]*>(.*?)</div>\s*</div>\s*</div>'
        # Patron mas flexible
        section_blocks = re.findall(r'<div class=["\']rec-section["\'][^>]*>(.*?)</div>\s*</div>\s*</div>', texto, re.DOTALL)
        if not section_blocks:
            # Intentar con pattern mas simple
            section_blocks = re.findall(r'<div class=["\']rec-section["\'][^>]*>(.*?)</div>\s*</div>', texto, re.DOTALL)

        for block in section_blocks:
            num_match = re.search(r'<div class=["\']rec-num["\'][^>]*>(\d+)</div>', block)
            title_match = re.search(r'<div class=["\']rec-title["\'][^>]*>(.*?)</div>', block)
            text_match = re.search(r'<p class=["\']rec-text["\'][^>]*>(.*?)</p>', block)

            if num_match and title_match:
                secciones.append({
                    "num": num_match.group(1).zfill(2),
                    "titulo": re.sub(r'<[^>]+>', '', title_match.group(1)).strip(),
                    "texto": re.sub(r'<[^>]+>', '', text_match.group(1)).strip() if text_match else ""
                })

        if secciones:
            return secciones

    # 3. Parsear formato Markdown ## 01. Titulo
    secciones = []
    patron = r'(?:##\s*)?(\d{1,2})[\.\)]\s*(.+?)(?=\n(?:##\s*)?\d{1,2}[\.\)]|\Z)'
    matches = list(re.finditer(patron, texto, re.DOTALL))

    if not matches:
        # Si no hay formato estructurado, devolver como una sola seccion
        return [{"num": "01", "titulo": "Recomendacion general", "texto": texto.strip()}]

    for m in matches:
        num = m.group(1).zfill(2)
        titulo = m.group(2).split('\n')[0].strip()
        texto_seccion = '\n'.join(m.group(2).split('\n')[1:]).strip()
        secciones.append({"num": num, "titulo": titulo, "texto": texto_seccion})

    return secciones


def obtener_recomendacion(clase, clase_secundaria=None):
    """Genera recomendacion estructurada con Groq; si falla, usa respaldo curado."""
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))

    # Respaldo estructurado por enfermedad (siempre disponible)
    respaldo = RECOMENDACIONES_ESTRUCTURADAS.get(clase, RECOMENDACIONES_ESTRUCTURADAS["sana"])

    if not api_key:
        return respaldo

    nombre_enfermedad = COFFEE_DISEASES[clase]["name"]
    categoria = COFFEE_DISEASES[clase]["category"]

    prompt_extra = ""
    if clase_secundaria:
        prompt_extra = (
            f"ADEMAS, el sistema detecto posibles signos de {COFFEE_DISEASES[clase_secundaria]['name']}. "
            f"Incluye una nota breve sobre como diferenciar o manejar ambas condiciones simultaneas."
        )

    prompt = (
        "Eres un agronomo senior especializado en caficultura hondurena (IHCAFE). "
        f"Un productor de Comayagua, Honduras, tiene un diagnostico de: {nombre_enfermedad} ({categoria}). "
        "Genera una recomendacion TECNICA, DETALLADA y ESTRUCTURADA en EXACTAMENTE 5 secciones numeradas del 01 al 05. "
        "Cada seccion debe tener un titulo claro y un parrafo explicativo de 3-5 oraciones. "
        "Usa este formato obligatorio (texto plano, NO uses bloques de codigo ni HTML):\n\n"
        "## 01. Diferenciacion a simple vista\n[Descripcion detallada de como identificar visualmente la enfermedad/plaga y diferenciarla de otras similares]\n\n"
        "## 02. Manejo agronomico preventivo y correctivo\n[Acciones concretas: poda, fertilizacion, drenaje, sombra, fungicidas/acaricidas autorizados, dosis si aplica]\n\n"
        "## 03. Consulta a un tecnico IHCAFE\n[Por que es importante confirmar con un tecnico, que muestras llevar, cuando acudir urgentemente]\n\n"
        "## 04. Monitoreo y seguimiento\n[Frecuencia de inspeccion, indicadores de mejora o empeora, ajustes segun estacion]\n\n"
        "## 05. Registro y trazabilidad\n[Como llevar bitacora de observaciones, tratamientos aplicados, fechas, resultados para futuras campanas]\n\n"
        f"{prompt_extra}\n\n"
        "Responde UNICAMENTE con las 5 secciones en el formato indicado. No agregues introduccion ni conclusion. NO uses markdown de codigo (```)."
    )

    try:
        from groq import Groq
        cliente = Groq(api_key=api_key)
        respuesta = cliente.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1200,
        )
        texto = respuesta.choices[0].message.content.strip()
        secciones = parsear_recomendacion_estructurada(texto)
        if len(secciones) >= 3:
            return secciones
        return respaldo
    except Exception as e:
        st.warning(f"No se pudo consultar la API de Groq ({e}). Mostrando recomendacion curada de respaldo.")
        return respaldo


# ============================================================
# GITHUB ISSUES API — Retroalimentacion persistente
# ============================================================
def enviar_feedback_github(comentario, diagnostico_relacionado=None):
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

# Recomendaciones estructuradas de respaldo (se usan si Groq falla)
RECOMENDACIONES_ESTRUCTURADAS = {
    "roya": [
        {"num": "01", "titulo": "Diferenciacion a simple vista", "texto": "La Roya (Hemileia vastatrix) se identifica por pustulas anaranjadas en el enves de la hoja y manchas amarillentas en el haz. No confundir con Cercospora: la Roya tiene pustulas en relieve con esporas al tacto, mientras que Cercospora presenta manchas circulares planas con centro grisaceo. En estadios avanzados, la hoja se torna necrotica y cae prematuramente."},
        {"num": "02", "titulo": "Manejo agronomico preventivo y correctivo", "texto": "Aplicar fungicidas protectores a base de oxicloruro de cobre (2.5 g/L) o caldo bordeles (0.5%) cada 21-30 dias en epoca lluviosa. Realizar poda sanitaria eliminando ramas con >50% de incidencia. Mejorar drenaje de suelos para reducir humedad foliar >90%. En zonas severas, usar fungicidas sistémicos autorizados por IHCAFE alternando mecanismos de accion."},
        {"num": "03", "titulo": "Consulta a un tecnico IHCAFE", "texto": "Acuda al tecnico de IHCAFE si la incidencia supera el 10% de la parcela o si observa defoliacion masiva. Lleve muestras de hojas con pustulas frescas en bolsas de papel (no plastico) y registre fotografias de la distribucion en la planta. El tecnico podra confirmar si se trata de la raza IIv5 (resistente a algunos fungicidas) y ajustar el protocolo."},
        {"num": "04", "titulo": "Monitoreo y seguimiento", "texto": "Inspeccione semanalmente durante la transicion de lluvias a seca (mayo-julio), que es el pico de infeccion. Evalue 50 plantas al azar por hectarea. Indicadores de mejora: nuevo brote sin pustulas, reduccion de esporulacion. Indicadores de empeora: expansion a brotes terminales, aparicion en frutos. Ajuste frecuencia de aplicacion segun pluviometria."},
        {"num": "05", "titulo": "Registro y trazabilidad", "texto": "Lleve una bitacora con fecha de deteccion, porcentaje de incidencia, producto aplicado (nombre comercial e ingrediente activo), dosis, volumen de caldo por hectarea y condiciones climaticas. Esto permitira identificar resistencias, optimizar costos y cumplir con certificaciones de buenas practicas agricolas (BPA) exigidas por compradores."}
    ],
    "cercospora": [
        {"num": "01", "titulo": "Diferenciacion a simple vista", "texto": "Cercospora coffeicola produce manchas circulares de 3-8 mm con centro grisaceo-necrotico y halo amarillo-anaranjado. A diferencia de la Roya, no hay pustulas en relieve ni esporas en el enves. Se confunde frecuentemente con manchas de nutricion (deficiencia de Mn), pero estas carecen del halo definido y son mas irregulares."},
        {"num": "02", "titulo": "Manejo agronomico preventivo y correctivo", "texto": "Aumentar fertilizacion nitrogenada (urea foliar al 2%) y potasica. Regular sombra al 40-50% para reducir estres hidrico. Aplicar caldo bordeles preventivo antes de lluvias intensas. En brotes activos, usar fungicidas especificos con mancozeb o clorotalonil. Evitar trabajos en campo con follaje mojado para no diseminar esporas."},
        {"num": "03", "titulo": "Consulta a un tecnico IHCAFE", "texto": "Consulte si las manchas aparecen en mas del 30% del follaje o si persisten tras dos aplicaciones fungicidas. El tecnico analizara niveles de N y K en suelo/hoja para descartar que sea un problema nutricional primario que debilite el tejido y facilite la infeccion."},
        {"num": "04", "titulo": "Monitoreo y seguimiento", "texto": "Monitoree quincenalmente en epocas secas y calurosas (febrero-abril), cuando el estres hidrico potencia la enfermedad. Revise hojas del tercio medio de la planta. Mejora: nuevas hojas sin manchas, recuperacion del color verde intenso. Empeora: coalescencia de manchas, secado de bordes foliares, caida de hojas jovenes."},
        {"num": "05", "titulo": "Registro y trazabilidad", "texto": "Registre analisis foliar bianuales, niveles de sombra (% cobertura), tipo de sombra (Inga, Erythrina, malla), fecha de aplicaciones y condiciones climaticas previas. Documente si la parcela esta en ladera expuesta al sol (mayor riesgo). Estos datos son clave para ajustar el manejo integral del cultivo."}
    ],
    "phoma": [
        {"num": "01", "titulo": "Diferenciacion a simple vista", "texto": "Phoma costarricensis afecta principalmente zonas altas (>1300 msnm). Se manifiesta como lesiones oscuras, irregulares, en bordes y apices de hojas jovenes, con aspecto de quemadura o derretimiento. A diferencia de Roya o Cercospora, las lesiones son asimetricas, sin halo definido, y progresan rapidamente en brotes tiernos tras lluvias frias."},
        {"num": "02", "titulo": "Manejo agronomico preventivo y correctivo", "texto": "Establecer cortinas rompevientos (Cypress, Eucalyptus o bambu) para reducir viento frio-humedo. Poda de formacion para mejorar aireacion. Aplicar fungicidas protectores (oxicloruro de cobre + mancozeb) antes de frentes frios. Eliminar brotes afectados con corte limpio y desinfeccion de herramientas con alcohol al 70% entre plantas."},
        {"num": "03", "titulo": "Consulta a un tecnico IHCAFE", "texto": "Obligatorio en zonas altas si la mortalidad de brotes supera el 5%. El tecnico evaluara si se requiere cambio de variedad a una mas tolerante (ej. Lempira, Parainema) o si hay que reforzar el sistema de cortinas. Lleve registro de temperaturas minimas y horas de humedad relativa >85%."},
        {"num": "04", "titulo": "Monitoreo y seguimiento", "texto": "Vigile diariamente tras frentes frios o neblina persistente. La enfermedad progresa en 48-72 horas bajo esas condiciones. Indicadores de control: brotes nuevos con hojas intactas, ausencia de lesiones en el tercio superior. Alarma: presencia de picnidios negros (estructuras reproductivas del hongo) en lesiones maduras."},
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
rec_box_bg = "#1A120D" if is_dark else "#F0EBE3"

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

    /* Recommendation Section Cards */
    .rec-section {{
        display: flex;
        gap: 14px;
        align-items: flex-start;
        padding: 14px 0;
        border-bottom: 1px solid {border_color};
    }}
    .rec-section:last-child {{
        border-bottom: none;
    }}
    .rec-num {{
        min-width: 32px;
        height: 32px;
        border-radius: 10px;
        background-color: #2D5A3D;
        color: #FFFFFF;
        font-size: 12px;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }}
    .rec-title {{
        font-size: 14px;
        font-weight: 700;
        color: {text_main};
        margin-bottom: 4px;
    }}
    .rec-text {{
        font-size: 13px;
        color: {text_sub};
        line-height: 1.6;
        margin: 0;
    }}
    .rec-container {{
        background-color: {rec_box_bg};
        border: 1px solid {border_color};
        border-radius: 20px;
        padding: 20px 24px;
        margin-top: 16px;
        margin-bottom: 20px;
    }}
    .rec-header {{
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }}
    .rec-header-icon {{
        background-color: #2C1A11;
        color: #FCD34D;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        display: inline-flex;
        justify-content: center;
        align-items: center;
        font-size: 10px;
        font-weight: bold;
    }}
    .rec-header-label {{
        font-size: 10px;
        font-weight: bold;
        letter-spacing: 1px;
        color: #8C7D73;
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
                            coinfeccion_nota = (
                                f"🔎 Posible coinfeccion con {COFFEE_DISEASES[resultado['clase_secundaria']]['name']} "
                                f"({resultado['confianza_secundaria']*100:.1f}%). Se recomienda inspeccion adicional."
                            )
                        else:
                            coinfeccion_nota = None

                        new_item = {
                            "id": f"DIAG-{int(datetime.now().timestamp())}",
                            "primaryDisease": resultado["clase"],
                            "primaryConfidence": resultado["confianza"],
                            "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
                            "recommendation": recomendacion,
                            "coinfeccion": coinfeccion_nota,
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
            <div class="rec-container" style="text-align:center;">
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

            # Coinfeccion alerta
            if cd.get("coinfeccion"):
                st.warning(cd["coinfeccion"])

            # Recommendation box estructurado
            rec_secciones = cd.get("recommendation", [])
            if isinstance(rec_secciones, list) and len(rec_secciones) > 0:
                html_rec = f"""
                <div class="rec-container">
                    <div class="rec-header">
                        <span class="rec-header-icon">💡</span>
                        <span class="rec-header-label">ORIENTACION Y MANEJO PREVENTIVO</span>
                    </div>
                    <p style="font-size:12px; color:{text_sub}; margin-bottom:12px;">
                        Aqui tienes una recomendacion tecnica detallada para manejar la situacion:
                    </p>
                """
                for sec in rec_secciones:
                    html_rec += f"""
                    <div class="rec-section">
                        <div class="rec-num">{sec['num']}</div>
                        <div>
                            <div class="rec-title">{sec['titulo']}</div>
                            <p class="rec-text">{sec['texto']}</p>
                        </div>
                    </div>
                    """
                html_rec += "</div>"
                st.markdown(html_rec, unsafe_allow_html=True)
            else:
                # Fallback texto plano
                st.markdown(f"""
                <div class="rec-container">
                    <div class="rec-header">
                        <span class="rec-header-icon">💡</span>
                        <span class="rec-header-label">RECOMENDACION GROQ AI / IHCAFE</span>
                    </div>
                    <p style="font-size:13px; font-style:italic; line-height:1.6; border-left: 2px solid #2C1A11; padding-left: 12px; margin:0; white-space: pre-line;">
                        "{rec_secciones if isinstance(rec_secciones, str) else 'Sin recomendacion disponible.'}"
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
