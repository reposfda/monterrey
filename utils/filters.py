# utils/filters.py
from __future__ import annotations

import pandas as pd
import streamlit as st


def sidebar_filters(
    df_base: pd.DataFrame,
    *,
    show_position: bool = True,
    show_minutes: bool = True,
    show_team: bool = True,
) -> dict:
    st.sidebar.title("Filtros")
    out = {}

    if show_position:
        positions = ["Zaguero", "Lateral", "Volante", "Interior/Mediapunta", "Extremo", "Delantero"]
        out["position"] = st.sidebar.selectbox("Posición", positions, index=0)

    if show_minutes:
        out["min_minutes"] = st.sidebar.slider("Minutos mínimos", 0, 3000, 450, step=50)

    if show_team:
        if "teams" not in df_base.columns:
            st.sidebar.caption("Equipo: no disponible (no existe columna 'teams').")
            out["teams"] = []
        else:
            teams = sorted(df_base["teams"].dropna().astype(str).unique().tolist())
            out["teams"] = st.sidebar.multiselect(
                "Equipo",
                options=teams,
                default=[],  # vacío = todos
            )

    return out
