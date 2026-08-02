import streamlit as st
import json
import os
import base64
import time
from datetime import datetime
from PIL import Image
import io

# ==========================================
# AGRODETECT - CAFICULTURA HONDURAS (IHCAFE)
# APP.PY STREAMLIT IMPLEMENTATION
# ==========================================

st.set_page_config(
    page_title="AgroDetect - Diagnóstico Foliar de Café",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------
# CONSTANTS & COFFEE DISEASE DATASET (IHCAFE)
# ------------------------------------------
COFFEE_DISEASES = {
    "roya": {
        "name": "Roya del Café",
        "scientificName": "Hemileia vastatrix",
        "category": "Enfermedad Fúngica Grave",
        "color": "#D97706",
        "severity": "Alta",
        "symptoms": [
            "Pústulas de color naranja amarrillento en el envés de la hoja.",
            "Manchas cloróticas translúcidas en el haz.",
            "Defoliación severa e incapacidad de llenado del grano."
        ],
        "recommendations": "Aplicar fungicida sistémico a base de Triazoles o Estrobilurinas según ciclo de lluvias. Monitorear sombra y realizar podas de aireación. Consultar boletín alerta IHCAFE."
    },
    "cercospora": {
        "name": "Cercospora / Mancha de Hierro",
        "scientificName": "Cercospora coffeicola",
        "category": "Enfermedad Fúngica",
        "color": "#B45309",
        "severity": "Media - Alta",
        "symptoms": [
            "Manchas circulares marrón rojizo con centro grisáceo y halo amarillo.",
            "Afecta principalmente hojas expuestas a deficiencia de Nitrógeno y exceso de sol."
        ],
        "recommendations": "Reforzar fertilización nitrogenada e implementar manejo de sombra. Aplicar cobre u oxicloruro preventivo."
    },
    "phoma": {
        "name": "Phoma / Quema o Derretimiento",
        "scientificName": "Phoma costarricensis",
        "category": "Enfermedad Fúngica de Altura",
        "color": "#DC2626",
        "severity": "Alta en Zonas Altas (>1300 msnm)",
        "symptoms": [
            "Lesiones oscuras e irregulares en los bordes y ápices de la hoja.",
            "Aspecto de quemadura o derretido foliar en brotes tiernos."
        ],
        "recommendations": "Establecer cortinas rompevientos, reducir humedad relativa alta en la parcela y aplicar fungicidas con ingrediente activo Cyprodinil + Fludioxonil."
    },
    "arana_roja": {
        "name": "Araña Roja del Cafeto",
        "scientificName": "Oligonychus coffeae",
        "category": "Plaga de Ácaros",
        "color": "#EA580C",
        "severity": "Media",
        "symptoms": [
            "Bronceado o cambio a tono café rojizo en la cara superior de la hoja.",
            "Telarañas finas microscópicas en el envés en épocas secas."
        ],
        "recommendations": "Aplicar acaricidas específicos o azufre mojable. Evitar aplicaciones excesivas de insecticidas que eliminen depredadores naturales."
    },
    "minador": {
        "name": "Minador de la Hoja",
        "scientificName": "Leucoptera coffeella",
        "category": "Plaga de Insecto (Microlepidóptero)",
        "color": "#7C2D12",
        "severity": "Media",
        "symptoms": [
            "Galerías o minas transparentes/marrones necróticas que secan la epidermis.",
            "Enrollado foliar prematuro."
        ],
        "recommendations": "Realizar control biológico con parasitoides de larvas y aplicar insecticidas específicos de bajo impacto ambiental aprobados por IHCAFE."
    },
    "sana": {
        "name": "Hoja Sana (Sin Patología)",
        "scientificName": "Coffea arabica / canephora",
        "category": "Tejido Foliar Saludable",
        "color": "#1E5631",
        "severity": "Ninguna",
        "symptoms": [
            "Tejido verde brillante, epidermis continua sin pústulas ni perforaciones.",
            "Fisiología foliar óptima."
        ],
        "recommendations": "Mantener plan de nutrición foliar y radicular balanceado. Continuar monitoreos fitosanitarios quincenales."
    }
}

FEEDBACK_FILE = "github_feedback.json"
HISTORY_FILE = "diagnosis_history.json"

# ------------------------------------------
# INITIALIZE SESSION STATE
# ------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "claro"

if "history" not in st.session_state:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                st.session_state.history = json.load(f)
        except Exception:
            st.session_state.history = []
    else:
        st.session_state.history = []

if "feedback_list" not in st.session_state:
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                st.session_state.feedback_list = json.load(f)
        except Exception:
            st.session_state.feedback_list = []
    else:
        st.session_state.feedback_list = []

# ------------------------------------------
# THEME CSS STYLING (DARK / LIGHT MODES)
# ------------------------------------------
theme_mode = st.session_state.theme

if theme_mode == "oscuro":
    bg_color = "#140D0A"
    card_bg = "#211612"
    text_color = "#F5EFE9"
    subtext_color = "#A39B93"
    border_color = "#3A2B23"
    accent_green = "#2D7A46"
else:
    bg_color = "#FDFBF7"
    card_bg = "#FFFFFF"
    text_color = "#2C1A11"
    subtext_color = "#6B7280"
    border_color = "#E5E0D8"
    accent_green = "#1E5631"

st.markdown(f"""
<style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .css-1d3e158, .css-[#211612] {{
        background-color: {card_bg};
    }}
    .card-box {{
        background-color: {card_bg};
        border: 1px solid {border_color};
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }}
    .badge-disease {{
        padding: 4px 12px;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        font-size: 12px;
        text-transform: uppercase;
        display: inline-block;
    }}
    .stButton>button {{
        border-radius: 12px;
        font-weight: bold;
    }}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# SIDEBAR NAVIGATION & THEME TOGGLE
# ------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?w=400&auto=format&fit=crop&q=80", use_container_width=True)
    st.title("🌿 AgroDetect Café")
    st.caption("Diagnóstico IA Fitosanitario IHCAFE")
    
    st.divider()

    # Mode Selector
    mode_toggle = st.radio(
        "🎨 Tema de Interfaz:",
        ["Modo Claro ☀️", "Modo Oscuro 🌙"],
        index=0 if st.session_state.theme == "claro" else 1
    )
    if "Modo Oscuro" in mode_toggle and st.session_state.theme != "oscuro":
        st.session_state.theme = "oscuro"
        st.rerun()
    elif "Modo Claro" in mode_toggle and st.session_state.theme != "claro":
        st.session_state.theme = "claro"
        st.rerun()

    st.divider()

    menu = st.radio(
        "Navegación Principal:",
        ["🔍 Diagnóstico Foliar", "📜 Historial de Evaluaciones", "💬 Feedback GitHub", "📚 Guía Técnica IHCAFE"]
    )

    st.info("💡 **Caficultura Honduras**\nSoporte técnico para roya, cercospora, phoma, araña roja y minador de la hoja.")

# ------------------------------------------
# MOCK AI ANALYSIS FUNCTION
# ------------------------------------------
def analyze_coffee_leaf(image_bytes):
    # Simulated neural network analysis (MobileNetV2 / Gemini Vision backend)
    time.sleep(1.2)  # simulate processing
    
    # Simple deterministic hash on bytes for reproducible mock demo
    val = sum(image_bytes[:100]) % 100
    
    if val < 30:
        primary = "roya"
        confidence = 0.942
        probs = {"roya": 0.942, "cercospora": 0.038, "phoma": 0.012, "arana_roja": 0.005, "minador": 0.002, "sana": 0.001}
    elif val < 50:
        primary = "cercospora"
        confidence = 0.895
        probs = {"cercospora": 0.895, "roya": 0.065, "phoma": 0.020, "arana_roja": 0.010, "minador": 0.006, "sana": 0.004}
    elif val < 70:
        primary = "phoma"
        confidence = 0.912
        probs = {"phoma": 0.912, "roya": 0.045, "cercospora": 0.028, "arana_roja": 0.008, "minador": 0.005, "sana": 0.002}
    elif val < 85:
        primary = "arana_roja"
        confidence = 0.878
        probs = {"arana_roja": 0.878, "minador": 0.072, "roya": 0.025, "cercospora": 0.015, "phoma": 0.006, "sana": 0.004}
    else:
        primary = "sana"
        confidence = 0.965
        probs = {"sana": 0.965, "roya": 0.012, "cercospora": 0.010, "phoma": 0.008, "arana_roja": 0.003, "minador": 0.002}
        
    return primary, confidence, probs

# ------------------------------------------
# 1. TAB: DIAGNÓSTICO FOLIAR
# ------------------------------------------
if menu == "🔍 Diagnóstico Foliar":
    st.header("🌿 Escáner Foliar de Café con Inteligencia Artificial")
    st.write("Sube o toma una fotografía de la hoja de cafeto afectada para identificar patologías foliares en segundos.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Muestra Foliar")
        uploaded_file = st.file_uploader("Selecciona una fotografía de la hoja (.jpg, .png):", type=["jpg", "png", "jpeg"])
        camera_photo = st.camera_input("O toma una foto directamente con la cámara:")

        active_img = uploaded_file if uploaded_file is not None else camera_photo

        if active_img is not None:
            image = Image.open(active_img)
            st.image(image, caption="Muestra Foliar Cargada", use_container_width=True)

    with col2:
        st.subheader("2. Resultado del Análisis AI")
        
        if active_img is not None:
            if st.button("🚀 Diagnosticar Hoja", type="primary", use_container_width=True):
                with st.spinner("Analizando pigmentos foliares y necrosis con red neuronal..."):
                    img_bytes = active_img.getvalue()
                    primary_id, confidence, probs = analyze_coffee_leaf(img_bytes)
                    disease_info = COFFEE_DISEASES[primary_id]

                    st.session_state.current_eval = {
                        "id": f"DIAG-{int(time.time())}",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "primaryDisease": primary_id,
                        "primaryConfidence": confidence,
                        "probabilities": probs,
                        "name": disease_info["name"],
                        "color": disease_info["color"],
                        "recommendation": disease_info["recommendations"]
                    }

        if "current_eval" in st.session_state:
            res = st.session_state.current_eval
            d_info = COFFEE_DISEASES[res["primaryDisease"]]

            st.markdown(f"""
            <div class="card-box">
                <span class="badge-disease" style="background-color: {d_info['color']};">{d_info['name']}</span>
                <h3 style="margin-top: 10px;">Certeza Diagnóstica: {res['primaryConfidence']*100:.1f}%</h3>
                <p><b>Categoría:</b> {d_info['category']}</p>
                <p><b>Gravedad Fitosanitaria:</b> {d_info['severity']}</p>
                <hr>
                <p><b>📋 Recomendación Agronómica IHCAFE:</b><br>{d_info['recommendations']}</p>
            </div>
            """, unsafe_allow_html=True)

            c_save, c_pdf = st.columns([1, 1])
            with c_save:
                if st.button("💾 Guardar en Historial", use_container_width=True):
                    st.session_state.history.insert(0, res)
                    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                        json.dump(st.session_state.history, f, ensure_ascii=False, indent=2)
                    st.success("¡Diagnóstico guardado exitosamente en el historial!")

            with c_pdf:
                # PDF content generator
                pdf_summary = f"""
===================================================
AGRODETECT HONDURAS - INFORME FITOSANITARIO IHCAFE
===================================================
Código de Informe: {res['id']}
Fecha de Evaluación: {res['timestamp']}

DIAGNÓSTICO PRINCIPAL:
- Patología: {d_info['name']} ({d_info.get('scientificName', 'N/A')})
- Confianza del Modelo AI: {res['primaryConfidence']*100:.1f}%
- Gravedad Estimada: {d_info['severity']}

SÍNTOMAS VISUALES:
{chr(10).join(['- ' + s for s in d_info['symptoms']])}

RECOMENDACIÓN AGRONÓMICA INSTITUCIONAL IHCAFE:
{d_info['recommendations']}
===================================================
"""
                st.download_button(
                    label="📄 Exportar PDF Informe",
                    data=pdf_summary,
                    file_name=f"Informe_{res['id']}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

# ------------------------------------------
# 2. TAB: HISTORIAL DE EVALUACIONES
# ------------------------------------------
elif menu == "📜 Historial de Evaluaciones":
    st.header("📜 Historial Cronológico de Diagnósticos")
    st.write("Registro de evaluaciones foliares escaneadas.")

    if len(st.session_state.history) == 0:
        st.info("Aún no tienes evaluaciones registradas. Ve a la pestaña 'Diagnóstico Foliar' para realizar tu primer escaneo.")
    else:
        if st.button("🗑️ Limpiar Historial Completo"):
            st.session_state.history = []
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.rerun()

        for item in st.session_state.history:
            d_meta = COFFEE_DISEASES.get(item["primaryDisease"], COFFEE_DISEASES["sana"])
            
            st.markdown(f"""
            <div class="card-box">
                <div style="display: flex; justify-space-between; align-items: center;">
                    <span class="badge-disease" style="background-color: {d_meta['color']};">{d_meta['name']}</span>
                    <span style="font-size: 12px; color: gray;">{item['timestamp']}</span>
                </div>
                <h4 style="margin-top: 10px;">ID: {item['id']} — Certeza: {item['primaryConfidence']*100:.1f}%</h4>
                <p style="font-size: 13px;"><b>Recomendación:</b> {item['recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------
# 3. TAB: FEEDBACK GITHUB
# ------------------------------------------
elif menu == "💬 Feedback GitHub":
    st.header("💬 Registro de Retroalimentación GitHub (`github_feedback.json`)")
    st.write("Escribe tus observaciones agronómicas. Cada comentario genera una entrada en formato estructurado para control de calidad.")

    col_form, col_log = st.columns([1, 1])

    with col_form:
        st.subheader("Nuevo Comentario")
        author = st.text_input("Tu Nombre / Seudónimo:", "Ing. Agrónomo")
        role = st.selectbox("Rol:", ["Productor de Café", "Técnico IHCAFE", "Ing. Agrónomo", "Investigador", "Otro"])
        rating = st.slider("Calificación de Precisión:", 1, 5, 5)
        disease_ref = st.selectbox("Enfermedad Asociada:", ["(General)", "Roya del Café", "Cercospora", "Phoma", "Araña Roja", "Minador", "Hoja Sana"])
        comment = st.text_area("Observación o Sugerencia Técnica:")

        if st.button("✉️ Registrar Comentario (Git Commit)", type="primary"):
            if comment.strip():
                commit_hash = f"a7f{int(time.time())%1000000:06d}"
                entry = {
                    "id": len(st.session_state.feedback_list) + 1,
                    "commitHash": commit_hash,
                    "author": author,
                    "role": role,
                    "rating": rating,
                    "diseaseRef": disease_ref,
                    "comment": comment,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.feedback_list.insert(0, entry)
                with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
                    json.dump(st.session_state.feedback_list, f, ensure_ascii=False, indent=2)
                st.success(f"¡Comentario guardado! Commit generado: `{commit_hash}`")
            else:
                st.error("Por favor escribe una observación antes de enviar.")

    with col_log:
        st.subheader("Historial Registrado GitHub")
        for fb in st.session_state.feedback_list:
            st.markdown(f"""
            <div class="card-box">
                <p><b>Commit:</b> <code>{fb['commitHash']}</code> | <b>{fb['author']}</b> ({fb['role']})</p>
                <p><b>Calificación:</b> {'⭐'*fb['rating']} | <b>Ref:</b> {fb['diseaseRef']}</p>
                <p style="font-style: italic;">"{fb['comment']}"</p>
                <span style="font-size: 11px; color: gray;">Registrado el {fb['timestamp']}</span>
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------
# 4. TAB: GUÍA TÉCNICA IHCAFE
# ------------------------------------------
elif menu == "📚 Guía Técnica IHCAFE":
    st.header("📚 Catálogo Fitosanitario de Plagas y Enfermedades en Café")
    st.write("Manual de referencia para la identificación foliar en fincas de Honduras.")

    selected_d = st.selectbox("Selecciona una enfermedad para ver sus características:", list(COFFEE_DISEASES.keys()), format_func=lambda k: COFFEE_DISEASES[k]["name"])
    info = COFFEE_DISEASES[selected_d]

    st.markdown(f"""
    <div class="card-box" style="border-left: 6px solid {info['color']};">
        <h2>{info['name']}</h2>
        <p><i>Nombre Científico: {info.get('scientificName', 'N/A')}</i></p>
        <p><b>Categoría:</b> {info['category']} | <b>Gravedad:</b> {info['severity']}</p>
        <hr>
        <h4>Sintomatología Característica:</h4>
        <ul>
            {''.join([f'<li>{s}</li>' for s in info['symptoms']])}
        </ul>
        <hr>
        <h4>Plan de Manejo Integrado IHCAFE:</h4>
        <p>{info['recommendations']}</p>
    </div>
    """, unsafe_allow_html=True)
