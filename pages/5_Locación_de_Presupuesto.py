# pages/5_Locacion_de_Presupuesto.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import (
    BASE_DIR,
    LOGO_PATH,
    Colors,
    apply_global_styles,
)

from utils.plot_budget_efficiency import plot_talent_cost_utilization_bar


# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Locación de Presupuesto – Monterrey",
    layout="wide",
    page_icon="📊",
)

# ✅ Estilos globales (config.py) -> corrige el padding-top y unifica estética
apply_global_styles()


# =============================================================================
# LOADERS
# =============================================================================
@st.cache_data(show_spinner=False)
def load_talent_cost_utilization_csv() -> pd.DataFrame:
    path = BASE_DIR / "data" / "economica" / "talent_cost_utilization.csv"
    if not path.exists():
        raise FileNotFoundError(f"No encuentro el CSV en: {path}")
    return pd.read_csv(path)


def compute_talent_cost_utilization_per_team(
    df: pd.DataFrame,
    team_col: str = "team_name",
    util_col: str = "talent_cost_utilization_per_player",
) -> pd.DataFrame:
    """
    Devuelve un DF por equipo con:
    - talent_cost_utilization_per_team (%)
    - agrega fila 'Promedio Liga'
    """
    d = df.dropna(subset=[team_col, util_col]).copy()
    d[util_col] = pd.to_numeric(d[util_col], errors="coerce")
    d = d.dropna(subset=[util_col])

    per_team = (
        d.groupby(team_col)[util_col]
        .sum()
        .reset_index()
        .rename(columns={util_col: "talent_cost_utilization_per_team"})
        .sort_values("talent_cost_utilization_per_team", ascending=False)
    )

    # pasar a porcentaje
    per_team["talent_cost_utilization_per_team"] = per_team["talent_cost_utilization_per_team"] * 100.0

    # promedio liga
    league_avg = float(per_team["talent_cost_utilization_per_team"].mean())
    per_team = pd.concat(
        [
            per_team,
            pd.DataFrame(
                {
                    team_col: ["Promedio Liga"],
                    "talent_cost_utilization_per_team": [league_avg],
                }
            ),
        ],
        ignore_index=True,
    )

    # sort final
    per_team = per_team.sort_values("talent_cost_utilization_per_team", ascending=False).reset_index(drop=True)
    return per_team


# =============================================================================
# HEADER
# =============================================================================
col_logo, col_title = st.columns([1, 6])

with col_logo:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=90)

with col_title:
    st.markdown("## Locación de Presupuesto")
    st.caption("Costo de utilización del plantel por club (% del gasto total).")

st.markdown(
    """
Esta sección mide el **costo de utilización** del plantel por club: cuánto “utiliza” cada equipo su inversión en talento,
sumando el indicador a nivel jugador y llevándolo a nivel **equipo** (en %).

*Lectura rápida:* barras más altas ⇒ mayor utilización; **Promedio Liga** es el benchmark de referencia.
"""
)

st.markdown("---")


# =============================================================================
# DATA
# =============================================================================
try:
    talent_cost_ut = load_talent_cost_utilization_csv()
except Exception as e:
    st.error(str(e))
    st.stop()

df_team = compute_talent_cost_utilization_per_team(talent_cost_ut)


# =============================================================================
# PLOT
# =============================================================================
fig = plot_talent_cost_utilization_bar(df_team)
st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# TABLA (OPCIONAL)
# =============================================================================
with st.expander("Ver tabla (equipo)", expanded=False):
    dshow = df_team.copy()
    dshow = dshow.rename(columns={"team_name": "Equipo", "talent_cost_utilization_per_team": "% Utilización"})
    dshow["% Utilización"] = dshow["% Utilización"].map(lambda x: f"{x:.1f}%")
    st.dataframe(dshow, hide_index=True, use_container_width=True)
