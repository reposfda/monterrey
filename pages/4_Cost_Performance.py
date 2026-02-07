# pages/4_Cost_Performance.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go

#from ui.ui_tables import render_mty_table

from utils.predicion_model import fit_market_curve, suggest_salary_range_for_player, performance_required_ranges
from utils.plot_prediction_model import plot_market_curve_interactive, render_performance_salary_scale_table, render_mty_table


# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(page_title="Cost ↔ Performance – Monterrey", layout="wide")


# =========================================================
# ESTILO
# =========================================================
PRIMARY_BG = "#0B1F38"
SECONDARY_BG = "#091325"
ACCENT = "#6CA0DC"   # azul MTY
TEXT = "#FFFFFF"
GOLD = "#c49308"

BASE_DIR = Path(__file__).resolve().parents[1]
LOGO_PATH = BASE_DIR / "assets" / "monterrey_logo.png"

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

        /* Tablas estilo MTY (render_mty_table usa class mty-table) */
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
        table.mty-table tbody td {{
            color: #1f2933 !important;
            padding: 0.55rem 0.75rem;
        }}
        table.mty-table tbody tr:nth-child(even) {{ background-color: #f4f6fb; }}
        table.mty-table tbody tr:hover {{ background-color: #e8f0ff; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DATA PATHS (DINÁMICO POR POSICIÓN)
# =========================================================
POSITION_FILES = {
    "Golero": "data/scores/score_cost/arq_score_cost.csv",
    "Zaguero": "data/scores/score_cost/def_score_cost.csv",
    "Lateral": "data/scores/score_cost/lat_score_cost.csv",
    "Volante": "data/scores/score_cost/vol_score_cost.csv",
    "Interior/Mediapunta": "data/scores/score_cost/int_score_cost.csv",
    "Extremo": "data/scores/score_cost/ext_score_cost.csv",
    "Delantero": "data/scores/score_cost/del_score_cost.csv",
}


# =========================================================
# LOADERS (CACHE)
# =========================================================
@st.cache_data(show_spinner=False)
def load_score_cost_csv(rel_path: str) -> pd.DataFrame:
    path = BASE_DIR / rel_path
    if not path.exists():
        raise FileNotFoundError(f"No encuentro el CSV en: {path}")
    df = pd.read_csv(path)
    return df

# columnas esperadas
COL_TEAM = "team_name"
COL_PLAYER = "player_name"
COL_PERF = "Overall_Score_Final"
COL_COST = "Cost_Share"

# =========================================================
# UI
# =========================================================
# Header con logo
c1, c2 = st.columns([1, 7])
with c1:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=90)
with c2:
    st.markdown("# Cost ↔ Performance")
    st.markdown(
        "Metodología para conectar **rendimiento en cancha** con **% del gasto total del club** "
        "por posición, usando la relación observada en el mercado de la Liga MX."
    )

st.divider()

# Sidebar: seleccionar posición
st.sidebar.markdown("## Configuración")
pos = st.sidebar.selectbox("Posición / Rol", list(POSITION_FILES.keys()), index=0)

rel_path = POSITION_FILES[pos]
try:
    df = load_score_cost_csv(rel_path)
except Exception as e:
    st.error(str(e))
    st.stop()

# Validación mínima
missing = [c for c in [COL_TEAM, COL_PLAYER, COL_PERF, COL_COST] if c not in df.columns]
if missing:
    st.error(f"El CSV seleccionado no tiene las columnas esperadas: {missing}")
    st.stop()

# Selección de jugador para mostrar “precio justo”
players = df[COL_PLAYER].dropna().astype(str).sort_values().unique().tolist()
default_player = "Luis Alberto Cárdenas López" if "Luis Alberto Cárdenas López" in players else (players[0] if players else "")
player_selected = st.sidebar.selectbox("Jugador (para estimación individual)", players, index=players.index(default_player) if default_player in players else 0)

# Parámetros de escala
st.sidebar.markdown("## Escala rendimiento ↔ salario")
start_cs = st.sidebar.number_input("Cost share inicial", min_value=0.0, max_value=0.50, value=0.00, step=0.01, format="%.2f")
end_cs = st.sidebar.number_input("Cost share final", min_value=0.01, max_value=0.50, value=0.10, step=0.01, format="%.2f")
step_cs = st.sidebar.number_input("Paso", min_value=0.005, max_value=0.10, value=0.01, step=0.005, format="%.3f")

# Fit modelo (una vez)
model, residual_std = fit_market_curve(df, COL_PERF, COL_COST)

# Resultados para jugador seleccionado
# Resultados para jugador seleccionado
# Resultados para jugador seleccionado
res = suggest_salary_range_for_player(df, player_selected, COL_PERF, COL_COST, COL_PLAYER)

# ===== TÍTULO: jugador seleccionado (blanco y más grande) =====
team_selected = (
    df.loc[df[COL_PLAYER] == res["player_name"], COL_TEAM].iloc[0]
    if not df.loc[df[COL_PLAYER] == res["player_name"]].empty
    else "—"
)

st.markdown(
    f"""
    <div style="margin-top: 0.25rem; margin-bottom: 0.75rem;">
        <div style="color: {TEXT}; font-size: 1.6rem; font-weight: 800; line-height: 1.2;">
            {res['player_name']}
        </div>
        <div style="color: rgba(255,255,255,0.75); font-size: 0.95rem; margin-top: 0.15rem;">
            Posición: {pos} · Equipo: {team_selected}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Helpers para mostrar cost_share como %
def fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:.1f}%"

current_cs = float(res["current_cost_share"]) if res["current_cost_share"] is not None else np.nan
fair_cs = float(res["fair_cost_share"]) if res["fair_cost_share"] is not None else np.nan

# ===== Eficiencia contractual: etiqueta + símbolo INLINE =====
# Banda neutral: dentro de ±5% del justo
NEUTRAL_BAND = 0.05  # 5%
label = "—"
symbol = ""
symbol_color = "rgba(255,255,255,0.75)"

# Diferencia relativa: (actual - justo) / justo
rel_diff = np.nan
if np.isfinite(current_cs) and np.isfinite(fair_cs) and fair_cs > 0:
    rel_diff = (current_cs - fair_cs) / fair_cs  # ej: -0.28 = 28% menos

    if rel_diff <= -0.30:
        label = "Muy Buena"
        symbol = "▲"
        symbol_color = "#22c55e"
    elif rel_diff < -NEUTRAL_BAND:
        label = "Buena"
        symbol = "▲"
        symbol_color = "#22c55e"
    elif abs(rel_diff) <= NEUTRAL_BAND:
        label = "Neutral"
        symbol = "="
        symbol_color = "rgba(255,255,255,0.75)"
    elif rel_diff < 0.30:
        label = "Mala"
        symbol = "▼"
        symbol_color = "#ef4444"
    else:
        label = "Muy Mala"
        symbol = "▼"
        symbol_color = "#ef4444"

# Frase explicativa (en vez de 0.72x)
eff_sentence = ""
if np.isfinite(rel_diff):
    pct_points = abs(rel_diff) * 100
    if rel_diff < -NEUTRAL_BAND:
        eff_sentence = (
            f"El jugador **{res['player_name']}** está cobrando un **{pct_points:.0f}% menos** "
            f"de lo que el mercado pagaría por su rendimiento."
        )
    elif rel_diff > NEUTRAL_BAND:
        eff_sentence = (
            f"El jugador **{res['player_name']}** está cobrando un **{pct_points:.0f}% más** "
            f"de lo que debería cobrar según su performance dentro del campo comparado con otros "
            f"jugadores de su misma posición."
        )
    else:
        eff_sentence = (
            f"El jugador **{res['player_name']}** está cobrando un valor **muy cercano** "
            f"al que el mercado pagaría por su rendimiento."
        )

# ===== KPIs (5 columnas) =====
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Performance Score", f"{res['perf']:.2f}")
k2.metric("Cost share actual", fmt_pct(current_cs))
k3.metric("Cost share 'justo' mercado", fmt_pct(fair_cs))
k4.metric("Techo máximo razonable", fmt_pct(res["max_reasonable_cost_share"]))

# KPI “Eficiencia contractual” (alineado)
k5.metric("Eficiencia contractual", "", help="Comparación entre el cost share actual y el cost share 'justo' del mercado (por rendimiento).")

with k5:
    st.markdown(
        f"""
        <div style="margin-top: -6px; display: flex; align-items: center; gap: 10px;">
            <div style="color: rgba(255,255,255,0.92); font-size: 18px; font-weight: 800;">
                {label}
            </div>
            <div style="color: {symbol_color}; font-size: 28px; font-weight: 900; line-height: 1;">
                {symbol}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Rango negociación en %
st.markdown(
    f"**Rango negociación (±0.5 STD mercado):** "
    f"`{fmt_pct(res['negotiation_low'])} – {fmt_pct(res['negotiation_high'])}`"
)

# Frase explicativa de eficiencia contractual (nuevo renglón)
if eff_sentence:
    st.markdown(eff_sentence)

st.divider()


# Gráfico
fig = plot_market_curve_interactive(
    df=df,
    model=model,
    residual_std=residual_std,
    perf_col=COL_PERF,
    cost_col=COL_COST,
    team_col=COL_TEAM,
    player_col=COL_PLAYER,
    highlight_team="Monterrey",
    title=f"Curva de mercado (posición: {pos}) – Rendimiento vs % del gasto total",
    selected_player=player_selected,   
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# Tabla de escala
df_ranges = performance_required_ranges(
    model=model,
    start_cs=float(start_cs),
    end_cs=float(end_cs),
    step=float(step_cs),
    clip_min=0.0,
    clip_max=100.0,
)

st.markdown("## Escala: cuánto rendimiento justifica cuánto salario")
st.markdown(
    "Para cada rango de **cost share**, se muestra el rango de **Performance Score** que el mercado suele exigir "
    "para que ese nivel de asignación presupuestaria tenga sentido."
)

styled_table = render_performance_salary_scale_table(
    df_ranges=df_ranges,
    player_perf=res["perf"],
    accent_color="#0B1F38",
    table_width="62%",
)

st.dataframe(styled_table, hide_index=True, width="stretch")


st.divider()

# =========================================================
# Top 10 mejores contratos (por diferencia vs precio justo)
# =========================================================
with st.expander("Top 10 mejores contratos en esta posición", expanded=False):
    # Base limpia
    d = df.dropna(subset=[COL_TEAM, COL_PLAYER, COL_PERF, COL_COST]).copy()

    # Precio justo para cada jugador (predicción del modelo)
    d["fair_cost_share"] = model.predict(d[[COL_PERF]].values)

    # Evitar divisiones por cero / negativos raros
    d["fair_cost_share"] = pd.to_numeric(d["fair_cost_share"], errors="coerce")
    d[COL_COST] = pd.to_numeric(d[COL_COST], errors="coerce")

    # Eficiencia contractual (%): cuánto menos cobra vs justo
    # positivo = buen contrato (cobra menos), negativo = sobrepago
    d["efficiency_pct"] = np.where(
        d["fair_cost_share"] > 0,
        (d["fair_cost_share"] - d[COL_COST]) / d["fair_cost_share"],
        np.nan,
    )

    # Ordenar por mejor contrato (más % por debajo del justo)
    d = d.sort_values("efficiency_pct", ascending=False)

    # Formateos para mostrar (en %)
    def fmt_pct(x):
        return f"{x * 100:.1f}%" if pd.notna(x) else "—"

    # Columnas a mostrar
    cols = [COL_TEAM, COL_PLAYER, COL_PERF, COL_COST, "fair_cost_share", "efficiency_pct"]

    # Si existe player_id, lo dejamos afuera (no lo pediste)
    dshow = d[cols].copy()

    dshow = dshow.rename(
        columns={
            COL_TEAM: "Equipo",
            COL_PLAYER: "Jugador",
            COL_PERF: "Performance Score",
            COL_COST: "Cost Share",
            "fair_cost_share": "Precio Justo",
            "efficiency_pct": "Eficiencia contractual",
        }
    )

    # Aplicar formatos de presentación
    dshow["Cost Share"] = dshow["Cost Share"].apply(fmt_pct)
    dshow["Precio Justo"] = dshow["Precio Justo"].apply(fmt_pct)

    # Eficiencia en formato "X% menos / X% más"
    def fmt_eff(x):
        if pd.isna(x):
            return "—"
        if x >= 0:
            return f"{x * 100:.0f}% menos"
        return f"{abs(x) * 100:.0f}% más"

    dshow["Eficiencia contractual"] = dshow["Eficiencia contractual"].apply(fmt_eff)

    # Mostrar top 10
    render_mty_table(dshow.head(10), index=False)

