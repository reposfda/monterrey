# pages/4_Evaluacion_de_Contratos.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    BASE_DIR,
    LOGO_PATH,
    Colors,
    apply_global_styles,
)


from utils.predicion_model import (
    fit_market_curve,
    suggest_salary_range_for_player,
    performance_required_ranges,
)
from utils.plot_prediction_model import (
    plot_market_curve_interactive,
    render_performance_salary_scale_table,
    render_mty_table,
)

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Evaluación de Contratos – Monterrey",
    layout="wide",
    page_icon="💰",
)

# ✅ Estilos globales (config.py)
apply_global_styles()

# =============================================================================
# DATA PATHS (DINÁMICO POR POSICIÓN)
# =============================================================================
POSITION_FILES = {
    "Golero": "data/scores/score_cost/arq_score_cost.csv",
    "Zaguero": "data/scores/score_cost/def_score_cost.csv",
    "Lateral": "data/scores/score_cost/lat_score_cost.csv",
    "Volante": "data/scores/score_cost/vol_score_cost.csv",
    "Interior/Mediapunta": "data/scores/score_cost/int_score_cost.csv",
    "Extremo": "data/scores/score_cost/ext_score_cost.csv",
    "Delantero": "data/scores/score_cost/del_score_cost.csv",
}

# columnas esperadas
COL_TEAM = "team_name"
COL_PLAYER = "player_name"
COL_PERF = "Overall_Score_Final"
COL_COST = "Cost_Share"

# =============================================================================
# LOADERS (CACHE)
# =============================================================================
@st.cache_data(show_spinner=False)
def load_score_cost_csv(rel_path: str) -> pd.DataFrame:
    path = BASE_DIR / rel_path
    if not path.exists():
        raise FileNotFoundError(f"No encuentro el CSV en: {path}")
    return pd.read_csv(path)


def fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:.1f}%"


# =============================================================================
# HEADER
# =============================================================================
col_logo, col_title = st.columns([1, 6])

with col_logo:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=90)

with col_title:
    st.markdown("## Evaluación de Contratos")
    st.caption("Relación entre performance en cancha y asignación presupuestaria (% del gasto total).")

st.markdown(
    """
Esta sección conecta el **rendimiento en cancha** de un jugador con el **% del gasto total** que el club destina a ese futbolista
y lo compara con la situación de otros jugadores de la misma posición de la Liga MX.

El modelo utiliza como input el promedio ponderado de la performance de los jugadores en las últimas dos temporadas,
asignándole mayor importancia a la más reciente.
"""
)

st.markdown("---")

# =============================================================================
# SIDEBAR (igual estilo que Tablero Jugadores)
# =============================================================================
st.sidebar.markdown("## Filtros")

pos = st.sidebar.selectbox(
    "Posición",
    list(POSITION_FILES.keys()),
    index=0,
)

st.sidebar.markdown("---")

st.sidebar.markdown("## Escala rendimiento ↔ salario")

start_cs = st.sidebar.slider(
    "Cost share inicial",
    min_value=0.00,
    max_value=0.50,
    value=0.00,
    step=0.01,
    format="%.2f",
)

end_cs = st.sidebar.slider(
    "Cost share final",
    min_value=0.01,
    max_value=0.50,
    value=0.10,
    step=0.01,
    format="%.2f",
)

step_cs = st.sidebar.slider(
    "Paso",
    min_value=0.005,
    max_value=0.10,
    value=0.01,
    step=0.005,
    format="%.3f",
)

st.sidebar.markdown("---")

# =============================================================================
# LOAD DATA
# =============================================================================
rel_path = POSITION_FILES[pos]
try:
    df = load_score_cost_csv(rel_path)
except Exception as e:
    st.error(str(e))
    st.stop()

missing = [c for c in [COL_TEAM, COL_PLAYER, COL_PERF, COL_COST] if c not in df.columns]
if missing:
    st.error(f"El CSV seleccionado no tiene las columnas esperadas: {missing}")
    st.stop()

# Cast numéricos (robusto)
df[COL_PERF] = pd.to_numeric(df[COL_PERF], errors="coerce")
df[COL_COST] = pd.to_numeric(df[COL_COST], errors="coerce")

# =============================================================================
# SELECTOR DE JUGADOR (EN EL CUERPO, NO EN SIDEBAR) ✅
# =============================================================================
players = df[COL_PLAYER].dropna().astype(str).sort_values().unique().tolist()
default_player = "Luis Alberto Cárdenas López" if "Luis Alberto Cárdenas López" in players else (players[0] if players else "")

labels_players = df[COL_PLAYER].astype(str) + " — " + df[COL_TEAM].astype(str)

# elegimos índice para mantener formato tipo Tablero Jugadores
player_choice = st.selectbox(
    "Jugador (para estimación individual)",
    options=list(range(len(df))),
    format_func=lambda i: labels_players.iloc[i],
    index=(
        int(df.index[df[COL_PLAYER].astype(str) == default_player][0])
        if default_player and (df[COL_PLAYER].astype(str) == default_player).any()
        else 0
    ),
)

player_selected = str(df.iloc[player_choice][COL_PLAYER])

# =============================================================================
# MODEL FIT
# =============================================================================
model, residual_std = fit_market_curve(df, COL_PERF, COL_COST)

# Resultados para jugador seleccionado
res = suggest_salary_range_for_player(df, player_selected, COL_PERF, COL_COST, COL_PLAYER)

team_selected = (
    df.loc[df[COL_PLAYER].astype(str) == str(res["player_name"]), COL_TEAM].iloc[0]
    if not df.loc[df[COL_PLAYER].astype(str) == str(res["player_name"])].empty
    else "—"
)

# =============================================================================
# TÍTULO JUGADOR
# =============================================================================
st.markdown(
    f"""
    <div style="margin-top: 0.25rem; margin-bottom: 0.75rem;">
        <div style="color: {Colors.TEXT}; font-size: 1.6rem; font-weight: 800; line-height: 1.2;">
            {res['player_name']}
        </div>
        <div style="color: rgba(255,255,255,0.75); font-size: 0.95rem; margin-top: 0.15rem;">
            Posición: {pos} · Equipo: {team_selected}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

current_cs = float(res["current_cost_share"]) if res["current_cost_share"] is not None else np.nan
fair_cs = float(res["fair_cost_share"]) if res["fair_cost_share"] is not None else np.nan

# =============================================================================
# EFICIENCIA CONTRACTUAL (label + símbolo inline + frase)
# =============================================================================
NEUTRAL_BAND = 0.05  # 5%

label = "—"
symbol = ""
symbol_color = "rgba(255,255,255,0.75)"

rel_diff = np.nan
if np.isfinite(current_cs) and np.isfinite(fair_cs) and fair_cs > 0:
    rel_diff = (current_cs - fair_cs) / fair_cs  # -0.28 => 28% menos

    if rel_diff <= -0.30:
        label, symbol, symbol_color = "Muy Buena", "▲", "#22c55e"
    elif rel_diff < -NEUTRAL_BAND:
        label, symbol, symbol_color = "Buena", "▲", "#22c55e"
    elif abs(rel_diff) <= NEUTRAL_BAND:
        label, symbol, symbol_color = "Neutral", "=", "rgba(255,255,255,0.75)"
    elif rel_diff < 0.30:
        label, symbol, symbol_color = "Mala", "▼", "#ef4444"
    else:
        label, symbol, symbol_color = "Muy Mala", "▼", "#ef4444"

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

# =============================================================================
# KPIs
# =============================================================================
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Performance Score", f"{res['perf']:.2f}")
k2.metric("Cost share actual", fmt_pct(current_cs))
k3.metric("Cost share 'justo' mercado", fmt_pct(fair_cs))
k4.metric("Techo máximo razonable", fmt_pct(res["max_reasonable_cost_share"]))

k5.metric(
    "Eficiencia contractual",
    "",
    help="Comparación entre el cost share actual y el cost share 'justo' del mercado (por rendimiento).",
)
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
        unsafe_allow_html=True,
    )

st.markdown(
    f"**Rango negociación (±0.5 STD mercado):** "
    f"`{fmt_pct(res['negotiation_low'])} – {fmt_pct(res['negotiation_high'])}`"
)

if eff_sentence:
    st.markdown(eff_sentence)

st.markdown("---")

# =============================================================================
# GRÁFICO – Curva de mercado
# =============================================================================
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

st.markdown("---")

# =============================================================================
# TABLA DE ESCALA
# =============================================================================
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
    accent_color=Colors.PRIMARY_BG,
    table_width="62%",
)

# ✅ Evita el error de tu compañero (width="stretch" no es compatible en algunas versiones)
st.dataframe(styled_table, hide_index=True, use_container_width=True)

st.markdown("---")

# =============================================================================
# Top 10 mejores contratos (por diferencia vs precio justo)
# =============================================================================
with st.expander("Top 10 mejores contratos en esta posición", expanded=False):
    d = df.dropna(subset=[COL_TEAM, COL_PLAYER, COL_PERF, COL_COST]).copy()

    # Precio justo (predicción del modelo)
    d["fair_cost_share"] = model.predict(d[[COL_PERF]].values)
    d["fair_cost_share"] = pd.to_numeric(d["fair_cost_share"], errors="coerce")
    d[COL_COST] = pd.to_numeric(d[COL_COST], errors="coerce")

    # efficiency_pct: positivo = cobra menos que lo justo (buen contrato)
    d["efficiency_pct"] = np.where(
        d["fair_cost_share"] > 0,
        (d["fair_cost_share"] - d[COL_COST]) / d["fair_cost_share"],
        np.nan,
    )

    d = d.sort_values("efficiency_pct", ascending=False)

    def fmt_pct_local(x):
        return f"{x * 100:.1f}%" if pd.notna(x) else "—"

    def fmt_eff(x):
        if pd.isna(x):
            return "—"
        if x >= 0:
            return f"{x * 100:.0f}% menos"
        return f"{abs(x) * 100:.0f}% más"

    dshow = d[[COL_TEAM, COL_PLAYER, COL_PERF, COL_COST, "fair_cost_share", "efficiency_pct"]].copy()
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

    dshow["Cost Share"] = dshow["Cost Share"].apply(fmt_pct_local)
    dshow["Precio Justo"] = dshow["Precio Justo"].apply(fmt_pct_local)
    dshow["Eficiencia contractual"] = dshow["Eficiencia contractual"].apply(fmt_eff)

    render_mty_table(dshow.head(10), index=False)
