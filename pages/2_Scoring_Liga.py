# pages/2_Scoring_Liga.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go

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

# =========================================
# VIZ: Mapa de Perfiles por Cuadrantes (2+2)
# =========================================
st.markdown("---")
st.subheader("🧭 Perfiles – Mapa por cuadrantes (2+2)")

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Fallback paleta (por si no están definidos arriba)
PRIMARY_BG = globals().get("PRIMARY_BG", "#0B1F38")
ACCENT     = globals().get("ACCENT", "#6CA0DC")
GOLD       = globals().get("GOLD", "#807C42")
TEXT       = globals().get("TEXT", "#E8EEF6")

# --- Helper robusto para matching de nombres
def _norm_player(x: str) -> str:
    if x is None:
        return ""
    return (
        str(x)
        .strip()
        .replace("\u00A0", " ")
        .replace("  ", " ")
        .lower()
    )

# --- 1) Elegir 4 categorías (defensivas y ofensivas)
available_cats = [c for c in cat_cols_pretty if c in scores_disp.columns]
if len(available_cats) < 4:
    st.warning(f"Necesito 4 categorías para este mapa, pero encontré {len(available_cats)}: {available_cats}.")
else:
    # defaults: intenta mapear por nombre si existen
    def _pick_default(name_contains: str, fallback_idx: int) -> str:
        for c in available_cats:
            if name_contains.lower() in str(c).lower():
                return c
        return available_cats[fallback_idx]

    default_def1 = _pick_default("accion", 0)
    default_def2 = _pick_default("control", 1)
    default_off1 = _pick_default("progre", 2)
    default_off2 = _pick_default("impacto", 3)

    with st.expander("Configurar ejes (2+2)", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            def1 = st.selectbox("Eje Y – Categoría 1 (Defensivo):", options=available_cats, index=available_cats.index(default_def1))
            def2 = st.selectbox("Eje Y – Categoría 2 (Defensivo):", options=available_cats, index=available_cats.index(default_def2))
            w_def1 = st.slider("Peso Defensivo 1", 0.0, 1.0, 0.50, 0.05)
            w_def2 = 1.0 - w_def1
            st.caption(f"Peso Def2 = {w_def2:.2f}")
        with c2:
            off1 = st.selectbox("Eje X – Categoría 1 (Ofensivo):", options=available_cats, index=available_cats.index(default_off1))
            off2 = st.selectbox("Eje X – Categoría 2 (Ofensivo):", options=available_cats, index=available_cats.index(default_off2))
            w_off1 = st.slider("Peso Ofensivo 1", 0.0, 1.0, 0.50, 0.05)
            w_off2 = 1.0 - w_off1
            st.caption(f"Peso Off2 = {w_off2:.2f}")

    # --- 2) Preparar DF
    plot_df = scores_disp.copy()
    base_need = ["Jugador", "Equipo", "Perfil", "Score", def1, def2, off1, off2]
    for col in base_need:
        if col not in plot_df.columns:
            st.error(f"Falta la columna '{col}' en scores_disp.")
            st.stop()

    for col in ["Score", def1, def2, off1, off2]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    plot_df = plot_df[base_need].dropna(subset=[def1, def2, off1, off2]).copy()
    if plot_df.empty:
        st.warning("No hay jugadores con valores válidos para esas 4 categorías.")
        st.stop()

    # --- 3) Construir ejes compuestos (escala original)
    plot_df["X_off"] = w_off1 * plot_df[off1] + w_off2 * plot_df[off2]
    plot_df["Y_def"] = w_def1 * plot_df[def1] + w_def2 * plot_df[def2]

    # --- 4) Líneas promedio para cuadrantes
    x_mean = float(plot_df["X_off"].mean())
    y_mean = float(plot_df["Y_def"].mean())

    # --- 5) Selector highlight
    plot_df["player_key"] = plot_df["Jugador"].astype(str).map(_norm_player)

    players_display = plot_df["Jugador"].astype(str).tolist()
    key_by_display = dict(zip(players_display, plot_df["player_key"].tolist()))

    selected_display = st.selectbox("Highlight jugador:", options=["(ninguno)"] + players_display, index=0)
    selected_key = "" if selected_display == "(ninguno)" else key_by_display.get(selected_display, _norm_player(selected_display))
    plot_df["highlight"] = (plot_df["player_key"] == selected_key)

    # --- 6) Rango de ejes (si tus scores son 0-100, queda perfecto)
    # Si no, auto-range con padding
    def _axis_range(s: pd.Series):
        lo = float(np.nanpercentile(s, 2))
        hi = float(np.nanpercentile(s, 98))
        pad = (hi - lo) * 0.08 if hi > lo else 1.0
        return [lo - pad, hi + pad]

    xr = _axis_range(plot_df["X_off"])
    yr = _axis_range(plot_df["Y_def"])

    # --- 7) Figura
    fig = go.Figure()

    # puntos por Perfil (si hay demasiados perfiles, podemos pasar a Dominante)
    # acá mantenemos Perfil porque en cuadrantes suele funcionar
    for prof, ddf in plot_df.groupby("Perfil", dropna=False):
        fig.add_trace(go.Scattergl(
            x=ddf["X_off"],
            y=ddf["Y_def"],
            mode="markers",
            name=str(prof),
            marker=dict(size=9, opacity=0.78, line=dict(width=0)),
            customdata=np.stack([
                ddf["Jugador"].astype(str),
                ddf["Equipo"].astype(str),
                ddf["Score"].astype(float),
                ddf[off1].astype(float),
                ddf[off2].astype(float),
                ddf[def1].astype(float),
                ddf[def2].astype(float),
            ], axis=1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                "Score: %{customdata[2]:.2f}<br>"
                f"{off1}: "+"%{customdata[3]:.2f}<br>"
                f"{off2}: "+"%{customdata[4]:.2f}<br>"
                f"{def1}: "+"%{customdata[5]:.2f}<br>"
                f"{def2}: "+"%{customdata[6]:.2f}<extra></extra>"
            ),
        ))

    # líneas punteadas en promedios
    fig.add_shape(type="line", x0=x_mean, y0=yr[0], x1=x_mean, y1=yr[1],
                  line=dict(color="rgba(232,238,246,0.35)", width=2, dash="dot"))
    fig.add_shape(type="line", x0=xr[0], y0=y_mean, x1=xr[1], y1=y_mean,
                  line=dict(color="rgba(232,238,246,0.35)", width=2, dash="dot"))

    # labels de cuadrantes
    fig.add_annotation(x=xr[0], y=yr[1], xanchor="left", yanchor="top",
                       text="<b>Defensivo + Bajo Ofensivo</b>",
                       showarrow=False, font=dict(color=TEXT, size=12),
                       bgcolor="rgba(11,31,56,0.55)", borderpad=6)
    fig.add_annotation(x=xr[1], y=yr[1], xanchor="right", yanchor="top",
                       text="<b>Defensivo + Alto Ofensivo</b>",
                       showarrow=False, font=dict(color=TEXT, size=12),
                       bgcolor="rgba(11,31,56,0.55)", borderpad=6)
    fig.add_annotation(x=xr[0], y=yr[0], xanchor="left", yanchor="bottom",
                       text="<b>Bajo Defensivo + Bajo Ofensivo</b>",
                       showarrow=False, font=dict(color=TEXT, size=12),
                       bgcolor="rgba(11,31,56,0.55)", borderpad=6)
    fig.add_annotation(x=xr[1], y=yr[0], xanchor="right", yanchor="bottom",
                       text="<b>Bajo Defensivo + Alto Ofensivo</b>",
                       showarrow=False, font=dict(color=TEXT, size=12),
                       bgcolor="rgba(11,31,56,0.55)", borderpad=6)

    # highlight
    h = plot_df[plot_df["highlight"]]
    if not h.empty:
        fig.add_trace(go.Scattergl(
            x=h["X_off"], y=h["Y_def"],
            mode="markers+text",
            text=h["Jugador"],
            textposition="top center",
            textfont=dict(color=TEXT, size=12),
            marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(width=4, color=GOLD)),
            hoverinfo="skip",
            showlegend=False
        ))

    fig.update_layout(
        height=650,
        paper_bgcolor=PRIMARY_BG,
        plot_bgcolor=PRIMARY_BG,
        margin=dict(l=18, r=18, t=10, b=10),
        font=dict(color=TEXT, family="Inter, Segoe UI, Arial"),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0.0,
            bgcolor="rgba(0,0,0,0)",
            title_text=""
        ),
        xaxis_title=f"{off1} / {off2}  (Eje ofensivo)",
        yaxis_title=f"{def1} / {def2}  (Eje defensivo)",
    )

    fig.update_xaxes(range=xr, showgrid=False, zeroline=False, ticks="outside", tickfont=dict(color="rgba(232,238,246,0.65)"))
    fig.update_yaxes(range=yr, showgrid=False, zeroline=False, ticks="outside", tickfont=dict(color="rgba(232,238,246,0.65)"))

    st.plotly_chart(fig, use_container_width=True)
