import streamlit as st
import pandas as pd
from pathlib import Path

# =========================================
# CONFIGURACIÓN BÁSICA
# =========================================
st.set_page_config(
    page_title="Scoring Centrales – Monterrey",
    layout="wide"
)

# Colores estilo Monterrey
PRIMARY_BG = "#0B1F38"    # azul oscuro
SECONDARY_BG = "#091325"  # sidebar
ACCENT = "#6CA0DC"        # celeste
TEXT = "#FFFFFF"          # blanco
GOLD = "#c49308"          # dorado ponderaciones

# Rutas
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "scoring_final_centrales_v4.csv"
LOGO_PATH = BASE_DIR / "assets" / "monterrey_logo.png"

# =========================================
# ESTILOS GLOBALES (CSS)
# =========================================
st.markdown(
    f"""
    <style>
        /* Fondo principal de la app */
        .stApp {{
            background-color: {PRIMARY_BG};
        }}

        .block-container {{
            padding-top: 0rem !important;
        }}

        /* Header (barra donde está Deploy) */
        header[data-testid="stHeader"] {{
            background-color: transparent;
        }}
        header[data-testid="stHeader"] > div {{
            background-color: {PRIMARY_BG};
            box-shadow: none;
        }}

        /* Iconos y texto del header (Deploy, menú, etc.) siempre blancos */
        header[data-testid="stHeader"] * {{
            color: #FFFFFF !important;
        }}
        header[data-testid="stHeader"] svg,
        header[data-testid="stHeader"] path {{
            fill: #FFFFFF !important;
        }}

        /* Sidebar */
        [data-testid="stSidebar"] > div:first-child {{
            background-color: {SECONDARY_BG};
        }}

        /* Títulos y textos generales */
        h1, h2, h3, h4, h5, h6, p, label {{
            color: {TEXT} !important;
        }}

        /* Botón de reset en el sidebar */
        .stSidebar .stButton > button {{
            width: 100%;
            border-radius: 999px;
            background-color: {ACCENT};
            color: #FFFFFF !important;
            border: none;
            font-weight: 600;
            padding: 0.4rem 0.75rem;
        }}
        .stSidebar .stButton > button:hover {{
            background-color: #4E82C0;
            color: #FFFFFF !important;
        }}

        /* === SLIDERS: solo tocamos el texto, barra y thumb dependen del primaryColor del tema === */
        div[data-baseweb="slider"] span {{
            color: {GOLD} !important;
            font-weight: 700 !important;
        }}

        /* ====== TABLA TOP 10 ====== */
        table.mty-table {{
            border-collapse: separate;
            border-spacing: 0;
            width: 100%;
            background-color: #ffffff;
            border-radius: 12px;
            overflow: hidden;
            font-size: 0.90rem;
        }}

        /* Encabezado de la tabla */
        table.mty-table thead {{
            background-color: {PRIMARY_BG};
        }}
        table.mty-table thead th {{
            color: #ffffff !important;
            padding: 0.55rem 0.75rem;
        }}

        /* Alinear encabezado:
           - primera columna (Jugador) a la izquierda
           - resto centrado
        */
        table.mty-table thead th:first-child {{
            text-align: left !important;
            padding-left: 12px !important;
        }}
        table.mty-table thead th:not(:first-child) {{
            text-align: center !important;
        }}

        /* Celdas del cuerpo */
        table.mty-table tbody td {{
            color: #1f2933 !important;
            padding: 0.55rem 0.75rem;
        }}

        /* Alinear body:
           - Jugador a la izquierda
           - resto centrado
        */
        table.mty-table tbody td:first-child {{
            text-align: left !important;
            padding-left: 12px !important;
        }}
        table.mty-table tbody td:not(:first-child) {{
            text-align: center !important;
        }}

        /* Filas alternadas */
        table.mty-table tbody tr:nth-child(even) {{
            background-color: #f4f6fb;
        }}

        /* Hover fila */
        table.mty-table tbody tr:hover {{
            background-color: #e8f0ff;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<style>
/* ================= SLIDERS DORADOS MONTERREY ================= */

/* 1) Ocultar el fondo gris dinámico original */
div.stSlider > div[data-baseweb="slider"] > div[data-testid="stTickBar"] > div {
    background: transparent !important;
}

/* 2) Barra activa dorada (el “track” donde se mueve el círculo) */
div.stSlider > div[data-baseweb="slider"] > div > div {
    background: #c49308 !important;
    border-radius: 8px !important;
    height: 6px !important;
}

/* 3) Thumb / círculo dorado */
div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"] {
    background-color: #c49308 !important;
    box-shadow: 0 0 0 0.2rem rgba(196,147,8,0.3) !important;
    border: 2px solid #ffffff !important;
}

/* 4) Número rojo encima del círculo → dorado */
div.stSlider > div[data-baseweb="slider"] > div > div > div > div {
    color: #c49308 !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)


# =========================================
# CARGA DE DATOS
# =========================================
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

df = load_data(DATA_PATH)

# =========================================
# HEADER CON LOGO + TÍTULO
# =========================================
col_logo, col_title = st.columns([1, 4])

with col_logo:
    st.image(str(LOGO_PATH), width=90)

with col_title:
    st.markdown(
        f"""
        <h1 style="margin-bottom:0;">Scoring Centrales – Monterrey</h1>
        <p style="color:{ACCENT}; font-size:0.95rem; margin-top:0.25rem;">
            Ajuste interactivo de ponderaciones por categoría
        </p>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# =========================================
# SIDEBAR – PONDERACIONES
# =========================================
BASE_W_ACC  = 0.35
BASE_W_CTRL = 0.25
BASE_W_PROG = 0.25
BASE_W_OFE  = 0.15

st.sidebar.title("⚖️ Ajustar ponderaciones")

# Inicializar session_state
defaults = {
    "w_acc": BASE_W_ACC,
    "w_ctrl": BASE_W_CTRL,
    "w_prog": BASE_W_PROG,
    "w_ofe": BASE_W_OFE,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def reset_pesos():
    st.session_state.w_acc  = BASE_W_ACC
    st.session_state.w_ctrl = BASE_W_CTRL
    st.session_state.w_prog = BASE_W_PROG
    st.session_state.w_ofe  = BASE_W_OFE

st.sidebar.button("🔁 Resetear ponderaciones", on_click=reset_pesos)

w_acc = st.sidebar.slider("Acción defensiva", 0.0, 1.0, key="w_acc", step=0.01)
w_ctrl = st.sidebar.slider("Control defensivo", 0.0, 1.0, key="w_ctrl", step=0.01)
w_prog = st.sidebar.slider("Progresión / Distribución", 0.0, 1.0, key="w_prog", step=0.01)
w_ofe = st.sidebar.slider("Impacto ofensivo", 0.0, 1.0, key="w_ofe", step=0.01)

total = w_acc + w_ctrl + w_prog + w_ofe
if total == 0:
    total = 1.0

w_acc /= total
w_ctrl /= total
w_prog /= total
w_ofe /= total

st.sidebar.markdown(
    f"**Normalizado:** Acción {w_acc:.2f} · Control {w_ctrl:.2f} · "
    f"Prog {w_prog:.2f} · Ofensivo {w_ofe:.2f}"
)

# =========================================
# RECÁLCULO DEL SCORING CUSTOM
# =========================================
df["score_custom"] = (
    df["score_accion_defensiva"] * w_acc +
    df["score_control_defensivo"] * w_ctrl +
    df["score_progresion_distribucion"] * w_prog +
    df["score_impacto_ofensivo"] * w_ofe
)

# =========================================
# TABLA TOP 10
# =========================================
st.subheader("🏆 Top 10 según tu ponderación")

cols_show = [
    "player_name", "minutes",
    "score_custom",
    "score_accion_defensiva",
    "score_control_defensivo",
    "score_progresion_distribucion",
    "score_impacto_ofensivo",
]

df_display = (
    df.sort_values("score_custom", ascending=False)[cols_show]
      .head(10)
      .reset_index(drop=True)
)

df_display = df_display.rename(columns={
    "player_name": "Jugador",
    "minutes": "Minutos",
    "score_custom": "Performance Score",
    "score_accion_defensiva": "Acción defensiva",
    "score_control_defensivo": "Control defensivo",
    "score_progresion_distribucion": "Progresión distribución",
    "score_impacto_ofensivo": "Impacto ofensivo",
})

# Formatear números con dos decimales (excepto Minutos que queda entero)
float_cols = [
    "Performance Score",
    "Acción defensiva",
    "Control defensivo",
    "Progresión distribución",
    "Impacto ofensivo",
]
for c in float_cols:
    df_display[c] = df_display[c].map(lambda x: f"{x:.2f}")

# Convertir a HTML con clase personalizada
html_table = df_display.to_html(index=False, classes="mty-table")
st.markdown(html_table, unsafe_allow_html=True)
