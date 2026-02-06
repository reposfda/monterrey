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

from position_scoring_golero import run_goalkeeper_scoring
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

    if position_key == "Golero":
        return run_goalkeeper_scoring(
            per90_csv=tmp_path,
            out_csv=out_tmp,
            position_group="Golero",
            min_minutes=min_minutes,
            min_matches=min_matches,
            flag_q=0.75,
        )

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
cat_w = filters.get("cat_weights", {})

# =========================================
# MAIN
# =========================================
with st.spinner("Calculando scoring..."):
    try:
        scores = compute_scoring(
            per90_path=str(PER90_PATH),
            position_key=pos,
            min_minutes=min_minutes,
            min_matches=min_matches,
            selected_teams=selected_teams,
        )
    except ValueError:
        st.warning(f"No hay jugadores de {pos} que cumplan los filtros actuales.")
        st.info("Probá bajar Minutos mínimos, cambiar Equipo o elegir otra Posición.")
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

# =========================================
# APPLY CAT WEIGHTS (what-if) -> Score_Ajustado
# =========================================
# cat_w viene de sidebar_filters: keys tipo "Score_Progresion" etc.
# Nosotros tenemos en pantalla columnas "pretty" (ej: "Progresion", "Impacto Ofensivo"...)
# Armamos un mapping para poder aplicar los pesos.
def _to_pretty_key(score_key: str) -> str:
    # "Score_AccionDefensiva" -> "Acciondefensiva" (luego title del pretty_map puede variar)
    k = score_key.replace("Score_", "").replace("score_", "").replace("_", " ").strip().title()
    return k

# 1) construir pares (pretty_col, weight) que existan en scores_disp
pairs = []
for k, w in (cat_w or {}).items():
    pretty_k = _to_pretty_key(k)
    if pretty_k in scores_disp.columns:
        pairs.append((pretty_k, float(w)))

# 2) si tenemos 4 pesos válidos, calculamos Score_Ajustado (0-100)
if len(pairs) >= 2:
    # normalizar por si acaso
    s = sum(w for _, w in pairs)
    if s > 0:
        pairs = [(c, w / s) for c, w in pairs]

    # weighted sum
    score_adj = 0
    for c, w in pairs:
        score_adj = score_adj + (pd.to_numeric(scores_disp[c], errors="coerce") * w)

    scores_disp["Score_Ajustado"] = score_adj
else:
    scores_disp["Score_Ajustado"] = pd.to_numeric(scores_disp["Score"], errors="coerce")


# agregar Score_Ajustado al display
cols_show = base_cols + (["Score_Ajustado"] if "Score_Ajustado" in scores_disp.columns else []) + cat_cols_pretty

top10 = (
    scores_disp.sort_values("Score_Ajustado", ascending=False)[cols_show]
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
if "Score_Ajustado" in top10.columns:
    top10["Score_Ajustado"] = pd.to_numeric(top10["Score_Ajustado"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2f}")


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
# VIZ: Boxplot por categoría + highlight jugador
# =========================================
st.markdown("---")
st.subheader("Distribución por categoría")

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Fallback paleta
PRIMARY_BG = globals().get("PRIMARY_BG", "#0B1F38")
ACCENT     = globals().get("ACCENT", "#6CA0DC")
GOLD       = globals().get("GOLD", "#807C42")
TEXT       = globals().get("TEXT", "#E8EEF6")

# --- Base DF (dedupe columnas + índice seguro)
plot_df = scores_disp.copy()
plot_df = plot_df.reset_index(drop=True)
plot_df.index = pd.RangeIndex(len(plot_df))

# Normalizar nombres columnas (por si hay NBSP)
plot_df.columns = (
    pd.Index(plot_df.columns)
      .astype(str)
      .str.replace("\u00A0", " ", regex=False)
      .str.strip()
)

# --- Categorías a graficar: usa tu config por posición si existe; si no, intenta tomar 4 Score_*
# 1) si ya tenés cat_cols_pretty (4 categorías por posición), usalo:
cats = [c for c in cat_cols_pretty if c in plot_df.columns]

# Si tenés más de 4 (ej. por bug de scoring), preferí las que usa el role_config actual:
# si en tu página ya estás armando `cat_cols_pretty` para esa posición, normalmente ya viene bien.
# En caso de exceso, recortamos a 4 por lo que esté primero.
if len(cats) > 4:
    cats = cats[:4]

if len(cats) < 2:
    st.warning("No encontré categorías válidas para graficar en este DF.")
else:
    # --- Convertir a numérico
    for c in cats:
        plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")

    # Tirar NaNs en todas las cats
    plot_df = plot_df.dropna(subset=cats).copy()
    if plot_df.empty:
        st.warning("No hay filas con valores válidos en las categorías seleccionadas.")
    else:
        # --- Selector highlight robusto
        if "player_id" in plot_df.columns:
            plot_df["hl_id"] = plot_df["player_id"].astype(str)
        else:
            plot_df["hl_id"] = (plot_df["Jugador"].astype(str) + "||" + plot_df["Equipo"].astype(str))

        plot_df["hl_label"] = plot_df["Jugador"].astype(str) + " — " + plot_df["Equipo"].astype(str)

        uniq = plot_df.drop_duplicates(subset=["hl_id"], keep="first")[["hl_id", "hl_label"]].copy()
        label_map = dict(zip(uniq["hl_id"], uniq["hl_label"]))

        selected_id = st.selectbox(
            "Highlight jugador:",
            options=["(ninguno)"] + uniq["hl_id"].tolist(),
            format_func=lambda x: x if x == "(ninguno)" else label_map.get(x, x),
            index=0,
            key=f"box_hl_{pos}",
        )

        # --- Reestructurar a formato long
        long = plot_df.melt(
            id_vars=["Jugador", "Equipo", "hl_id"],
            value_vars=cats,
            var_name="Categoria",
            value_name="Valor"
        )

        # --- Figura
        fig = go.Figure()

        # Boxplot por categoría (whiskers = IQR por defecto)
        # points='all' muestra puntos, jitter para ver densidad
        for cat in cats:
            d = long.loc[long["Categoria"] == cat, "Valor"].astype(float).to_numpy()

            fig.add_trace(go.Box(
                y=d,
                name=cat,
                boxpoints="all",
                jitter=0.35,
                pointpos=0,
                marker=dict(size=5, opacity=0.22, color="rgba(232,238,246,0.55)"),
                line=dict(color="rgba(108,160,220,0.55)", width=1),   # líneas finas
                fillcolor="rgba(108,160,220,0.08)",                   # caja más liviana
                whiskerwidth=0.35,
                width=0.30,                                           # caja más angosta
                showlegend=False,
                # median line visible por defecto (no hace falta leyenda)
            ))

        # Highlight overlay (un punto por categoría + línea horizontal)
        if selected_id != "(ninguno)":
            hlong = long[long["hl_id"].astype(str) == str(selected_id)].copy()

        # Highlight overlay (solo scores, sin nombre)
        if selected_id != "(ninguno)":
            hlong = long[long["hl_id"].astype(str) == str(selected_id)].copy()

            if not hlong.empty:
                fig.add_trace(go.Scatter(
                    x=hlong["Categoria"],
                    y=hlong["Valor"],
                    mode="markers+text",
                    text=[f"{v:.1f}" if pd.notna(v) else "" for v in hlong["Valor"]],
                    textposition="top center",
                    textfont=dict(color=TEXT, size=12),
                    marker=dict(
                        size=10,
                        color=GOLD,
                        opacity=0.95,
                        line=dict(width=0),
                        symbol="circle"
                    ),
                    hovertemplate="<b>%{x}</b><br>Valor: %{y:.1f}<extra></extra>",
                    showlegend=False
                ))

        # Layout
        fig.update_layout(
            height=560,
            paper_bgcolor=PRIMARY_BG,
            plot_bgcolor=PRIMARY_BG,
            margin=dict(l=18, r=18, t=10, b=10),
            font=dict(color=TEXT, family="Inter, Segoe UI, Arial"),
            xaxis=dict(
                title="",
                tickfont=dict(color="rgba(232,238,246,0.75)"),
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                title="",
                tickfont=dict(color="rgba(232,238,246,0.75)"),
                gridcolor="rgba(232,238,246,0.10)",
                zeroline=False
            ),
        )

        fig.update_yaxes(range=[0, 100])

        st.plotly_chart(fig, use_container_width=True)
        st.caption("Boxplot: línea = mediana · caja = IQR (P25–P75) · whiskers = 1.5×IQR · puntos = jugadores")

