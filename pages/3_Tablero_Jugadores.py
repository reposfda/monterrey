# pages/3_Tablero_Jugadores.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from utils.scoring_wrappers import compute_scoring_from_df

# =========================================================
# STREAMLIT CONFIG (DEBE SER LO PRIMERO)
# =========================================================
st.set_page_config(page_title="Tablero Jugadores – Monterrey", layout="wide")

# =========================================================
# IMPORTS PROYECTO
# =========================================================
from utils.loaders import load_per90
from utils.filters import sidebar_filters
from utils.scoring import compute_scoring
from utils.radar_chart import plot_radar
from utils.lollipop_chart import plot_lollipop_mty
from utils.metrics_labels import METRICS_ES
from utils.role_config import (
    get_macro_config,
    get_detail_categories,
    get_detail_metric_list,
)

# =========================================================
# COLORES (IGUAL QUE PÁGINA 2)
# =========================================================
PRIMARY_BG = "#0B1F38"
SECONDARY_BG = "#091325"
ACCENT = "#6CA0DC"
TEXT = "#FFFFFF"
GOLD = "#c49308"

# =========================================================
# CSS (IGUAL QUE PÁGINA 2) -> fondo + sidebar + tablas + sliders
# =========================================================
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

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1]
LOGO_PATH = BASE_DIR / "assets" / "monterrey_logo.png"
PER90_PATH = BASE_DIR / "outputs" / "all_players_complete_2025_2026.csv"

if not PER90_PATH.exists():
    st.error(f"No encuentro el archivo base per90 en: {PER90_PATH}")
    st.stop()

# =========================================================
# HELPERS
# =========================================================
def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def render_mty_table(df: pd.DataFrame, *, index: bool = False) -> None:
    st.markdown(df.to_html(index=index, classes="mty-table"), unsafe_allow_html=True)


def pct_rank_0_100(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    m = x.notna()
    out = pd.Series(index=x.index, dtype="float64")
    out.loc[m] = x.loc[m].rank(pct=True) * 100
    out.loc[~m] = pd.NA
    return out


def make_key(df: pd.DataFrame, player_col: str, team_col: str) -> pd.Series:
    return df[player_col].astype(str).str.strip() + "||" + df[team_col].astype(str).str.strip()


def safe_int(x):
    try:
        if pd.isna(x):
            return ""
        return int(float(x))
    except Exception:
        return ""


def safe_float2(x):
    try:
        if pd.isna(x):
            return ""
        return float(x)
    except Exception:
        return ""

def build_lollipop_inputs(
    *,
    df_base: pd.DataFrame,
    scores: pd.DataFrame,
    pos: str,
    min_minutes: int,
    selected_teams: list,
    compare_mode: str,
    player_key: tuple[str, str],
    second_key: tuple[str, str] | None,
    second_name: str | None,
    BASE_DIR: Path,
    col_player: str,
    col_team: str,
) -> tuple[list[str], list[float], list[float] | None, str, str, str]:
    """
    Devuelve todo lo necesario para plot_lollipop_mty:
    final_labels, vals_player, ref_vals, title_left, title_right, reference_kind
    """
    base_player_col = find_col(df_base, ["player_name", "player", "Jugador"])
    base_team_col = find_col(df_base, ["teams", "team_name", "Equipo"])
    base_minutes_col = find_col(df_base, ["minutes", "Minutos"])

    if base_player_col is None or base_team_col is None:
        raise ValueError("df_base no tiene columnas de jugador/equipo necesarias para el detalle (lollipop).")

    df_cohort = df_base.copy()

    # filtro equipos
    if selected_teams and "teams" in df_cohort.columns:
        df_cohort = df_cohort[df_cohort["teams"].astype(str).isin([str(t) for t in selected_teams])].copy()

    # filtro minutos
    if base_minutes_col:
        df_cohort = df_cohort[pd.to_numeric(df_cohort[base_minutes_col], errors="coerce") >= min_minutes].copy()

    # restringir a cohorte de scoring
    key_scores = make_key(scores, col_player, col_team).unique()
    key_series = make_key(df_cohort, base_player_col, base_team_col)
    df_cohort = df_cohort[key_series.isin(key_scores)].copy()

    # index jugador principal
    key_series = make_key(df_cohort, base_player_col, base_team_col)
    target_key = f"{player_key[0].strip()}||{player_key[1].strip()}"
    idx_list = key_series[key_series == target_key].index
    player_idx = idx_list[0] if len(idx_list) else None
    if player_idx is None:
        raise ValueError("El jugador no está en la cohorte base para el detalle (revisá llaves player/team).")

    detail_opts = get_detail_categories(pos)
    if not detail_opts:
        raise ValueError("No hay categorías detalladas configuradas para esta posición en role_config.py.")

    # concatenar categorías
    metric_lists: list[tuple[str, float, bool]] = []
    metric_labels: list[str] = []

    cat_alias = {}
    if pos.lower() == "lateral":
        cat_alias = {"Defensivo (Exec)": "Defensivo", "Defensivo (OBV)": "Defensivo"}

    for cat_name in detail_opts:
        ml = get_detail_metric_list(pos, cat_name, base_dir=BASE_DIR)  # [(metric, w, invert), ...]
        if not ml:
            continue

        cat_out = cat_alias.get(cat_name, cat_name)

        for metric, w, inv in ml:
            metric_lists.append((metric, w, inv))
            label_es = METRICS_ES.get(metric, metric)
            metric_labels.append(f"{cat_out}: {label_es}")

    if not metric_lists:
        raise ValueError("No pude leer listas detalladas desde role_config.")

    # construir labels + valores jugador + promedio (para percentil)
    final_labels: list[str] = []
    vals_player: list[float] = []
    vals_avg: list[float] = []

    for (metric, w, inv), lab in zip(metric_lists, metric_labels):
        if metric not in df_cohort.columns:
            continue

        s0 = pd.to_numeric(df_cohort[metric], errors="coerce")
        if s0.dropna().empty:
            continue

        p = pct_rank_0_100(s0)
        v = p.loc[player_idx]
        if pd.isna(v):
            continue

        final_labels.append(lab)
        vals_player.append(float(v))
        vals_avg.append(float(pd.to_numeric(p, errors="coerce").mean()))

    if len(final_labels) < 3:
        raise ValueError("Muy pocas métricas disponibles para armar el detalle (revisá nombres de columnas).")

    # referencia
    ref_vals: list[float] | None = None
    title_right = ""
    reference_kind = "player"

    if compare_mode == "Promedio":
        ref_vals = vals_avg
        title_right = "Promedio"
        reference_kind = "avg"

    elif compare_mode == "Otro jugador" and second_key is not None:
        target_key2 = f"{second_key[0].strip()}||{second_key[1].strip()}"
        idx2_list = key_series[key_series == target_key2].index
        second_idx = idx2_list[0] if len(idx2_list) else None

        if second_idx is not None:
            map_other: dict[str, float] = {}
            for (metric, w, inv), lab in zip(metric_lists, metric_labels):
                if metric not in df_cohort.columns:
                    continue
                s0 = pd.to_numeric(df_cohort[metric], errors="coerce")
                if s0.dropna().empty:
                    continue
                p2 = pct_rank_0_100(s0)
                v2 = p2.loc[second_idx]
                if pd.isna(v2):
                    continue
                map_other[lab] = float(v2)

            aligned = [map_other.get(lab, np.nan) for lab in final_labels]
            ref_vals = aligned
            title_right = second_name or ""
            reference_kind = "player"

    title_left = f"{player_key[0]}\n{player_key[1]}"
    return final_labels, vals_player, ref_vals, title_left, title_right, reference_kind


def estimate_lollipop_fig_h(n_rows: int, n_cats: int, *, row_h: float = 0.28, min_h: float = 3.1) -> float:
    """
    Replica el cálculo interno del lollipop (aprox) para alinear verticalmente.
    """
    return max(min_h, row_h * (n_rows + 0.8 * n_cats) + 1.1)


def compute_radar_vertical_spacers(fig_h_lolli: float, fig_h_radar: float = 4.2) -> tuple[int, int]:
    """
    Devuelve (top_px, bottom_px) para centrar radar respecto al lollipop.
    Convertimos 'fig inches' a px con un factor fijo (aprox).
    """
    # 1 inch ~ 96px en pantalla (aprox). Ajustá si querés.
    px_per_in = 25

    lolli_px = fig_h_lolli * px_per_in
    radar_px = fig_h_radar * px_per_in

    extra = max(0, lolli_px - radar_px)
    top = int(extra * 0.50)
    bottom = int(extra * 0.50)
    return top, bottom

# =========================================================
# HEADER
# =========================================================
c1, c2 = st.columns([1, 6])
with c1:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=90)
with c2:
    st.markdown("## Tablero Jugadores")
    st.caption("Ficha individual con radar de categorías y métricas detalladas")

st.markdown("---")

# =========================================================
# LOAD BASE + FILTERS
# =========================================================
df_base = load_per90(PER90_PATH)

filters = sidebar_filters(df_base, show_position=True, show_minutes=True, show_team=True)
pos = filters["position"]
min_minutes = int(filters["min_minutes"])
selected_teams = filters.get("teams", [])
min_matches = 3

# =========================================================
# SCORING (cohorte filtrada)
# =========================================================
with st.spinner("Calculando scoring…"):
    scores = compute_scoring_from_df(
        df_base=df_base,
        position_key=pos,
        min_minutes=min_minutes,
        min_matches=min_matches,
        selected_teams=selected_teams,
    )

if scores is None or scores.empty:
    st.warning("No hay jugadores que cumplan los filtros.")
    st.stop()

# columnas clave en scores
col_player = find_col(scores, ["player_name", "player", "Jugador"])
col_team = find_col(scores, ["teams", "team_name", "Equipo"])
col_minutes = find_col(scores, ["minutes", "Minutos"])
col_matches = find_col(scores, ["matches", "PJ"])
col_profile = find_col(scores, ["Flags", "Perfil"])
col_score_overall = find_col(scores, ["Score_Overall", "score_overall", "score_total", "Score"])

if col_player is None or col_team is None:
    st.error("No encuentro columnas necesarias en scores: player_name/player y teams/team_name.")
    st.stop()

# selector de jugador
labels_players = scores[col_player].astype(str) + " — " + scores[col_team].astype(str)
player_choice = st.selectbox(
    "Jugador",
    options=list(range(len(scores))),
    format_func=lambda i: labels_players.iloc[i],
)
player_row = scores.iloc[player_choice]
player_key = (str(player_row[col_player]), str(player_row[col_team]))

# =========================================================
# FICHA (TABLA ESTILO mty-table)
# =========================================================
ficha = {
    "Jugador": player_row[col_player] if col_player else "",
    "Equipo": player_row[col_team] if col_team else "",
    "Minutos": safe_int(player_row[col_minutes]) if col_minutes else "",
    "PJ": safe_int(player_row[col_matches]) if col_matches else "",
    "Score": safe_float2(player_row[col_score_overall]) if col_score_overall else "",
    "Perfil": player_row[col_profile] if col_profile else "",
}
ficha_df = pd.DataFrame([ficha])
if "Score" in ficha_df.columns:
    ficha_df["Score"] = pd.to_numeric(ficha_df["Score"], errors="coerce").map(
        lambda x: "" if pd.isna(x) else f"{x:.2f}"
    )

render_mty_table(ficha_df)
st.markdown("---")

# =========================================================
# COMPARACIÓN (DEFAULT = SOLO JUGADOR)
# =========================================================
compare_mode = st.radio(
    "Comparar contra",
    ["Solo jugador", "Promedio", "Otro jugador"],
    horizontal=True,
    index=0,
)

second_row = None
second_key = None
second_name = None
if compare_mode == "Otro jugador":
    other_choice = st.selectbox(
        "Jugador a comparar",
        options=list(range(len(scores))),
        index=0,
        format_func=lambda i: labels_players.iloc[i],
        key="other_player_select",
    )
    second_row = scores.iloc[other_choice]
    second_key = (str(second_row[col_player]), str(second_row[col_team]))
    second_name = str(second_row[col_player])

# =========================================================
# RADAR + LOLLIPOP (alineados)
# =========================================================
st.subheader("Perfil del jugador")

# 1) Precalcular inputs del lollipop ANTES de dibujar columnas
try:
    final_labels, vals_player, ref_vals, title_left, title_right, reference_kind = build_lollipop_inputs(
        df_base=df_base,
        scores=scores,
        pos=pos,
        min_minutes=min_minutes,
        selected_teams=selected_teams,
        compare_mode=compare_mode,
        player_key=player_key,
        second_key=second_key,
        second_name=second_name,
        BASE_DIR=BASE_DIR,
        col_player=col_player,
        col_team=col_team,
    )
except Exception as e:
    st.warning(str(e))
    final_labels, vals_player, ref_vals, title_left, title_right, reference_kind = [], [], None, "", "", "player"

# 2) Estimar altura del lollipop y calcular espaciadores del radar
n_rows = len(final_labels)
n_cats = len({lab.split(":")[0].strip() for lab in final_labels if ":" in lab}) if final_labels else 0

fig_h_radar = 4.2
fig_h_lolli = estimate_lollipop_fig_h(n_rows=n_rows, n_cats=max(1, n_cats), row_h=0.28, min_h=3.1) if n_rows else 6.0
top_px, bottom_px = compute_radar_vertical_spacers(fig_h_lolli=fig_h_lolli, fig_h_radar=fig_h_radar)

col_radar, col_lolli = st.columns([1.0, 1.35], gap="large")

# -------------------------
# IZQUIERDA: RADAR MACRO
# -------------------------
with col_radar:
    st.markdown("#### Radar – Categorías del rol")

    # ✅ espaciador dinámico arriba para centrar
    if top_px > 0:
        st.markdown(f"<div style='height: {top_px}px;'></div>", unsafe_allow_html=True)

    macro = get_macro_config(pos)
    if not macro:
        st.info("No hay mapeo de categorías para esta posición en role_config.py.")
    else:
        macro_cols = [c for c, _ in macro]
        macro_labels = [lab for _, lab in macro]

        player_vals = [
            float(player_row[c]) if (c in scores.columns and pd.notna(player_row[c])) else float("nan")
            for c in macro_cols
        ]

        ref_vals_macro = None
        head_right_macro = ""

        if compare_mode == "Promedio":
            ref_vals_macro = [
                float(pd.to_numeric(scores[c], errors="coerce").mean()) if c in scores.columns else float("nan")
                for c in macro_cols
            ]
            head_right_macro = "Promedio"
        elif compare_mode == "Otro jugador" and second_row is not None:
            ref_vals_macro = [
                float(second_row[c]) if (c in scores.columns and pd.notna(second_row[c])) else float("nan")
                for c in macro_cols
            ]
            head_right_macro = (second_key[0] if second_key is not None else (second_name or "")).strip()

        low = [0] * len(macro_labels)
        high = [100] * len(macro_labels)

        head_left = f"{player_key[0]} | {player_key[1]}"
        fig = plot_radar(
            metrics=macro_labels,
            values=player_vals,
            reference=ref_vals_macro,
            low=low,
            high=high,
            head_left=head_left,
            head_right=head_right_macro,
            figsize=(fig_h_radar, fig_h_radar),
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=220, transparent=True, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        buf.seek(0)
        st.image(buf, use_container_width=True)

    # ✅ espaciador dinámico abajo
    if bottom_px > 0:
        st.markdown(f"<div style='height: {bottom_px}px;'></div>", unsafe_allow_html=True)

# -------------------------
# DERECHA: LOLLIPOP (siempre visible, sin modo/toggle)
# -------------------------
with col_lolli:
    st.markdown("#### Métricas detalladas")

    if n_rows >= 3:
        fig_lolli = plot_lollipop_mty(
            labels=final_labels,
            values=vals_player,
            reference=ref_vals,
            xlim=(0.0, 100.0),
            title_left=title_left,
            title_right=title_right,
            value_fmt="{:.0f}",
            mode="percentil",
            reference_kind=reference_kind,
            show_value_annotations=True,
            fig_w=7.0,
            row_h=0.28,
            min_h=3.1,
            font_family="monospace",
        )
        st.pyplot(fig_lolli, use_container_width=True)
    else:
        st.info("No hay suficientes métricas para mostrar el detalle.")


st.markdown("---")
