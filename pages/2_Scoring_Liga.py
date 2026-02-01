# pages/2_Scoring_Liga.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from position_scoring_defensor_central import run_cb_scoring
from position_scoring_delantero import run_delantero_scoring
from position_scoring_extremos import run_extremo_scoring
from position_scoring_interior import run_interior_scoring
from position_scoring_lateral import run_position_scoring
from position_scoring_volante import run_volante_scoring

# =========================================
# CONFIG BÁSICA + ESTILO (igual a tu app.py)
# =========================================
st.set_page_config(page_title="Scoring Liga – Monterrey", layout="wide")

PRIMARY_BG = "#0B1F38"
SECONDARY_BG = "#091325"
ACCENT = "#6CA0DC"
TEXT = "#FFFFFF"
GOLD = "#c49308"

BASE_DIR = Path(__file__).resolve().parents[1]  # repo root (asumiendo pages/ dentro)
LOGO_PATH = BASE_DIR / "assets" / "monterrey_logo.png"

# ✅ Cambiá esto a tu “base per90” real
# Ej: outputs/all_players_complete_2025_2026.csv
PER90_PATH = BASE_DIR / "outputs" / "all_players_complete_2025_2026.csv"

st.markdown(
    f"""
    <style>
        .stApp {{ background-color: {PRIMARY_BG}; }}
        .block-container {{ padding-top: 0rem !important; }}

        header[data-testid="stHeader"] {{ background-color: transparent; }}
        header[data-testid="stHeader"] > div {{
            background-color: {PRIMARY_BG};
            box-shadow: none;
        }}
        header[data-testid="stHeader"] * {{ color: #FFFFFF !important; }}
        header[data-testid="stHeader"] svg,
        header[data-testid="stHeader"] path {{ fill: #FFFFFF !important; }}

        [data-testid="stSidebar"] > div:first-child {{
            background-color: {SECONDARY_BG};
        }}

        h1, h2, h3, h4, h5, h6, p, label {{
            color: {TEXT} !important;
        }}

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

        div[data-baseweb="slider"] span {{
            color: {GOLD} !important;
            font-weight: 700 !important;
        }}

        table.mty-table {{
            border-collapse: separate;
            border-spacing: 0;
            width: 100%;
            background-color: #ffffff;
            border-radius: 12px;
            overflow: hidden;
            font-size: 0.90rem;
        }}
        table.mty-table thead {{ background-color: {PRIMARY_BG}; }}
        table.mty-table thead th {{
            color: #ffffff !important;
            padding: 0.55rem 0.75rem;
        }}
        table.mty-table thead th:first-child {{
            text-align: left !important;
            padding-left: 12px !important;
        }}
        table.mty-table thead th:not(:first-child) {{
            text-align: center !important;
        }}

        table.mty-table tbody td {{
            color: #1f2933 !important;
            padding: 0.55rem 0.75rem;
        }}
        table.mty-table tbody td:first-child {{
            text-align: left !important;
            padding-left: 12px !important;
        }}
        table.mty-table tbody td:not(:first-child) {{
            text-align: center !important;
        }}

        table.mty-table tbody tr:nth-child(even) {{ background-color: #f4f6fb; }}
        table.mty-table tbody tr:hover {{ background-color: #e8f0ff; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    div.stSlider > div[data-baseweb="slider"] > div[data-testid="stTickBar"] > div {
        background: transparent !important;
    }
    div.stSlider > div[data-baseweb="slider"] > div > div {
        background: #c49308 !important;
        border-radius: 8px !important;
        height: 6px !important;
    }
    div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"] {
        background-color: #c49308 !important;
        box-shadow: 0 0 0 0.2rem rgba(196,147,8,0.3) !important;
        border: 2px solid #ffffff !important;
    }
    div.stSlider > div[data-baseweb="slider"] > div > div > div > div {
        color: #c49308 !important;
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================
# DATA LOAD
# =========================================
@st.cache_data
def load_per90(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, encoding="latin1")


# =========================================
# SCORING WRAPPER (dinámico por posición)
# - aplica filtro edad ANTES de score (para que percentiles sean del grupo filtrado)
# - usa tmp csv para reutilizar run_*_scoring sin reescribirlos
# =========================================
@st.cache_data(show_spinner=False)
def compute_scoring(
    per90_path: str,
    position_key: str,
    min_minutes: int,
    min_matches: int,
    age_min: int | None,
    age_max: int | None,
) -> pd.DataFrame:
    df = load_per90(Path(per90_path))

    # Filtro edad si existe columna
    if age_min is not None and age_max is not None:
        if "age" in df.columns:
            df["age"] = pd.to_numeric(df["age"], errors="coerce")
            df = df[(df["age"] >= age_min) & (df["age"] <= age_max)].copy()

    # Guardar a temp para consumir por los run_* (que esperan Path)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    df.to_csv(tmp_path, index=False, encoding="utf-8")

    out_tmp = (Path("outputs") / "_tmp_scoring.csv")  # se crea en cwd (ok en local/streamlit)
    out_tmp.parent.mkdir(parents=True, exist_ok=True)

    # Ejecutar scoring según posición
    if position_key == "Zaguero":
        out = run_cb_scoring(
            per90_csv=tmp_path,
            out_csv=out_tmp,
            position_group="Zaguero",
            min_minutes=min_minutes,
            min_matches=min_matches,
            flag_q=0.75,
        )

    elif position_key == "Lateral":
        out = run_position_scoring(
            per90_csv=tmp_path,
            out_csv=out_tmp,
            position_group="Lateral",
            min_minutes=min_minutes,
            min_matches=min_matches,
            flag_q=0.75,
            def_exec_w=0.60,
            def_obv_w=0.40,
        )

    elif position_key == "Volante":
        out = run_volante_scoring(
            per90_csv=tmp_path,
            out_csv=out_tmp,
            position_group="Volante",
            min_minutes=min_minutes,
            min_matches=min_matches,
            flag_q=0.75,
        )

    elif position_key == "Interior/Mediapunta":
        out = run_interior_scoring(
            per90_csv=tmp_path,
            out_csv=out_tmp,
            position_group="Interior/Mediapunta",
            min_minutes=min_minutes,
            min_matches=min_matches,
            flag_q=0.75,
        )

    elif position_key == "Extremo":
        out = run_extremo_scoring(
            per90_csv=tmp_path,
            out_csv=out_tmp,
            position_group="Extremo",
            min_minutes=min_minutes,
            min_matches=min_matches,
            flag_q=0.75,
        )

    elif position_key == "Delantero":
        out = run_delantero_scoring(
            per90_csv=tmp_path,
            out_csv=out_tmp,
            position_group="Delantero",
            min_minutes=min_minutes,
            min_matches=min_matches,
            flag_q=0.75,
        )

    else:
        raise ValueError(f"Posición no soportada: {position_key}")

    return out


# =========================================
# HEADER
# =========================================
c1, c2 = st.columns([1, 5])
with c1:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=90)
with c2:
    st.markdown(
        f"""
        <h1 style="margin-bottom:0;">Scoring Liga – Ranking</h1>
        <p style="color:{ACCENT}; font-size:0.95rem; margin-top:0.25rem;">
            Top 10 por posición (filtros: minutos, edad, posición)
        </p>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# =========================================
# SIDEBAR FILTERS
# =========================================
st.sidebar.title("Filtros")

positions = ["Zaguero", "Lateral", "Volante", "Interior/Mediapunta", "Extremo", "Delantero"]
pos = st.sidebar.selectbox("Posición", positions, index=0)

min_minutes = st.sidebar.slider("Minutos mínimos", 0, 3000, 450, step=50)
min_matches = 3  # fijo (si querés slider, lo agregamos)

# rango edad si existe
per90_preview = load_per90(PER90_PATH) if PER90_PATH.exists() else pd.DataFrame()
age_min, age_max = None, None
if not per90_preview.empty and "age" in per90_preview.columns:
    age_col = pd.to_numeric(per90_preview["age"], errors="coerce")
    a0 = int(age_col.dropna().min()) if age_col.notna().any() else 16
    a1 = int(age_col.dropna().max()) if age_col.notna().any() else 40
    age_range = st.sidebar.slider("Edad", a0, a1, (a0, a1), step=1)
    age_min, age_max = age_range[0], age_range[1]
else:
    st.sidebar.caption("Edad: no disponible (no existe columna 'age' en el CSV base).")

# =========================================
# MAIN: RUN + TOP 10
# =========================================
if not PER90_PATH.exists():
    st.error(f"No encuentro el archivo base per90 en: {PER90_PATH}")
    st.stop()

with st.spinner("Calculando scoring..."):
    scores = compute_scoring(
        per90_path=str(PER90_PATH),
        position_key=pos,
        min_minutes=int(min_minutes),
        min_matches=int(min_matches),
        age_min=age_min,
        age_max=age_max,
    )

if scores.empty:
    st.warning("No hay jugadores que cumplan los filtros.")
    st.stop()

# normalizar nombres esperados (según cada script)
rename_map = {
    "Score_Overall": "Score",
    "player_name": "Jugador",
    "team_name": "Equipo",
    "minutes": "Minutos",
    "matches": "PJ",
    "Flags": "Perfil",
}
scores_disp = scores.rename(columns=rename_map).copy()

cols_show = [c for c in ["Jugador", "Equipo", "Minutos", "PJ", "Score", "Perfil"] if c in scores_disp.columns]
top10 = (
    scores_disp.sort_values("Score", ascending=False)[cols_show]
    .head(10)
    .reset_index(drop=True)
)

# formateo
if "Minutos" in top10.columns:
    top10["Minutos"] = pd.to_numeric(top10["Minutos"], errors="coerce").fillna(0).astype(int)
if "PJ" in top10.columns:
    top10["PJ"] = pd.to_numeric(top10["PJ"], errors="coerce").fillna(0).astype(int)
if "Score" in top10.columns:
    top10["Score"] = pd.to_numeric(top10["Score"], errors="coerce").map(lambda x: f"{x:.2f}")

st.subheader(f"🏆 Top 10 – {pos}")
st.markdown(top10.to_html(index=False, classes="mty-table"), unsafe_allow_html=True)

# opcional: tabla completa
with st.expander("Ver ranking completo"):
    st.dataframe(scores_disp.sort_values("Score", ascending=False), use_container_width=True)
