# pages/2_Scoring_Liga.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.loaders import load_per90
from utils.filters import sidebar_filters

from position_scoring_defensor_central import run_cb_scoring
from position_scoring_delantero import run_delantero_scoring
from position_scoring_extremos import run_extremo_scoring
from position_scoring_interior import run_interior_scoring
from position_scoring_lateral import run_position_scoring
from position_scoring_volante import run_volante_scoring


# =========================================
# CONFIG + ESTILO
# =========================================
st.set_page_config(page_title="Scoring Liga – Monterrey", layout="wide")

PRIMARY_BG = "#0B1F38"
SECONDARY_BG = "#091325"
ACCENT = "#6CA0DC"
TEXT = "#FFFFFF"
GOLD = "#c49308"

BASE_DIR = Path(__file__).resolve().parents[1]
LOGO_PATH = BASE_DIR / "assets" / "monterrey_logo.png"

PER90_PATH = BASE_DIR / "outputs" / "all_players_complete_2025_2026.csv"
if not PER90_PATH.exists():
    st.error(f"No encuentro el archivo base per90 en: {PER90_PATH}")
    st.stop()

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
# SCORING WRAPPER
# =========================================
@st.cache_data(show_spinner=False)
def compute_scoring(
    per90_path: str,
    position_key: str,
    min_minutes: int,
    min_matches: int,
    selected_teams: list[str],
) -> pd.DataFrame:
    df = load_per90(Path(per90_path))

    # filtro equipos (antes del scoring)
    if selected_teams:
        if "teams" in df.columns:
            df = df[df["teams"].astype(str).isin([str(t) for t in selected_teams])].copy()

    # temp csv para reutilizar scripts run_*
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    df.to_csv(tmp_path, index=False, encoding="utf-8")

    out_tmp = (Path("outputs") / "_tmp_scoring.csv")
    out_tmp.parent.mkdir(parents=True, exist_ok=True)

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

def pick_category_score_cols(df: pd.DataFrame) -> list[str]:
    """
    Devuelve columnas de score por categoría (excluye overall/total).
    Busca patrones comunes en tus scripts:
    - 'score_' prefijo
    - o 'Score_' prefijo
    y excluye overall/total.
    """
    candidates = []
    for c in df.columns:
        cl = c.lower()
        if cl.startswith("score_") or cl.startswith("score"):
            # excluir overall/total
            if any(k in cl for k in ["overall", "total"]):
                continue
            # evitar columnas que no sean numéricas (por si acaso)
            candidates.append(c)

    # ordenar para que quede estable
    return sorted(set(candidates))


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
            Top 10 por posición (filtros: minutos, equipos, posición)
        </p>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# =========================================
# FILTERS
# =========================================
df_base = load_per90(PER90_PATH)

filters = sidebar_filters(
    df_base,
    show_position=True,
    show_minutes=True,
    show_team=True,
)

pos = filters["position"]
min_minutes = int(filters["min_minutes"])
selected_teams = filters.get("teams", [])
min_matches = 3

# =========================================
# MAIN
# =========================================
with st.spinner("Calculando scoring..."):
    scores = compute_scoring(
        per90_path=str(PER90_PATH),
        position_key=pos,
        min_minutes=min_minutes,
        min_matches=min_matches,
        selected_teams=selected_teams,
    )

if scores is None or scores.empty:
    st.warning("No hay jugadores que cumplan los filtros.")
    st.stop()

# --- renombres base (sin crear duplicados)
rename_map = {
    "player_name": "Jugador",
    "team_name": "Equipo",
    "teams": "Equipo",
    "minutes": "Minutos",
    "matches": "PJ",
    "Flags": "Perfil",
}
scores_disp = scores.rename(columns=rename_map).copy()

# --- asegurar score overall en una columna única
# (algunos scripts devuelven Score_Overall, otros score_total, etc.)
score_overall_candidates = [c for c in scores_disp.columns if c.lower() in ["score_overall", "score_total", "overall_score", "score"]]
# prioridad: Score_Overall si existe
if "Score_Overall" in scores.columns:
    scores_disp["Score"] = pd.to_numeric(scores["Score_Overall"], errors="coerce")
elif "score_overall" in scores.columns:
    scores_disp["Score"] = pd.to_numeric(scores["score_overall"], errors="coerce")
elif "score_total" in scores.columns:
    scores_disp["Score"] = pd.to_numeric(scores["score_total"], errors="coerce")
elif "Score" in scores_disp.columns:
    # si ya existe Score, lo fuerza a numérico (y si estaba duplicado lo evitamos porque lo creamos nosotros arriba)
    scores_disp["Score"] = pd.to_numeric(scores_disp["Score"], errors="coerce")
else:
    st.error("No encuentro columna de score overall (Score_Overall / score_overall / score_total / Score).")
    st.stop()

# --- detectar scores por categoría (usar df original 'scores' para encontrarlos mejor)
cat_cols_raw = pick_category_score_cols(scores)

# construir df con columnas de categoría ya normalizadas
cat_df = scores[cat_cols_raw].copy() if cat_cols_raw else pd.DataFrame(index=scores.index)

# nombres lindos
pretty_map = {}
for c in cat_cols_raw:
    name = c.replace("score_", "").replace("Score_", "")
    name = name.replace("_", " ").strip().title()
    pretty_map[c] = name

cat_df = cat_df.rename(columns=pretty_map)

# unir al display (por índice)
scores_disp = pd.concat([scores_disp, cat_df], axis=1)

# --- columnas a mostrar
cat_cols_pretty = list(cat_df.columns)
base_cols = [c for c in ["Jugador", "Equipo", "Minutos", "PJ", "Score", "Perfil"] if c in scores_disp.columns]
cols_show = base_cols + cat_cols_pretty

# --- Top 10
top10 = (
    scores_disp.sort_values("Score", ascending=False)[cols_show]
    .head(10)
    .reset_index(drop=True)
)

# --- formateo
if "Minutos" in top10.columns:
    top10["Minutos"] = pd.to_numeric(top10["Minutos"], errors="coerce").fillna(0).astype(int)
if "PJ" in top10.columns:
    top10["PJ"] = pd.to_numeric(top10["PJ"], errors="coerce").fillna(0).astype(int)
if "Score" in top10.columns:
    top10["Score"] = pd.to_numeric(top10["Score"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2f}")

for c in cat_cols_pretty:
    top10[c] = pd.to_numeric(top10[c], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2f}")

team_label = ""
if selected_teams:
    team_label = f" ({', '.join(selected_teams[:2])}{'...' if len(selected_teams) > 2 else ''})"

st.subheader(f"🏆 Top 10 – {pos}{team_label}")
st.markdown(top10.to_html(index=False, classes="mty-table"), unsafe_allow_html=True)

with st.expander("Ver ranking completo"):
    st.dataframe(scores_disp.sort_values("Score", ascending=False), use_container_width=True)
