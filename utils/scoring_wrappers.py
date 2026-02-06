# utils/scoring_wrappers.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import streamlit as st

from position_scoring_golero import run_goalkeeper_scoring
from position_scoring_defensor_central import run_cb_scoring
from position_scoring_delantero import run_delantero_scoring
from position_scoring_extremos import run_extremo_scoring
from position_scoring_interior import run_interior_scoring
from position_scoring_volante import score_volante_df
from position_scoring_lateral import score_lateral_df


@st.cache_data(show_spinner=False)
def compute_scoring_from_df(
    df_base: pd.DataFrame,
    position_key: str,
    min_minutes: int,
    min_matches: int,
    selected_teams: list[str] | None = None,
):
    """
    Wrapper único de scoring.
    Consume SIEMPRE DataFrame (per90) y devuelve scores por posición.

    No lee ni escribe archivos.
    """

    df = df_base.copy()

    # -----------------------------
    # Filtro por equipo
    # -----------------------------
    if selected_teams:
        if "teams" in df.columns:
            df = df[df["teams"].astype(str).isin(map(str, selected_teams))].copy()
        elif "team_name" in df.columns:
            df = df[df["team_name"].astype(str).isin(map(str, selected_teams))].copy()

    # -----------------------------
    # Dispatch por posición
    # -----------------------------
    if position_key == "Golero":
        return run_goalkeeper_scoring(
            df=df,
            position_group="Golero",
            min_minutes=min_minutes,
            min_matches=min_matches,
            flag_q=0.75,
        )

    if position_key == "Volante":
        return score_volante_df(
            per90_df=df,
            position_group="Volante",
            min_minutes=min_minutes,
            min_matches=min_matches,
            flag_q=0.75,
            verbose=False,
        )

    if position_key == "Lateral":
        return score_lateral_df(
            per90_df=df,
            position_group="Lateral",
            min_minutes=min_minutes,
            min_matches=min_matches,
            flag_q=0.75,
            def_exec_w=0.60,
            def_obv_w=0.40,
            verbose=False,
        )

    if position_key == "Interior/Mediapunta":
        return run_interior_scoring(
            df=df,
            position_group="Interior/Mediapunta",
            min_minutes=min_minutes,
            min_matches=min_matches,
            flag_q=0.75,
        )

    if position_key == "Extremo":
        return run_extremo_scoring(
            df=df,
            position_group="Extremo",
            min_minutes=min_minutes,
            min_matches=min_matches,
            flag_q=0.75,
        )

    if position_key == "Delantero":
        return run_delantero_scoring(
            df=df,
            position_group="Delantero",
            min_minutes=min_minutes,
            min_matches=min_matches,
            flag_q=0.75,
        )

    if position_key == "Zaguero":
        return run_cb_scoring(
            df=df,
            position_group="Zaguero",
            min_minutes=min_minutes,
            min_matches=min_matches,
            flag_q=0.75,
        )

    raise ValueError(f"Posición no soportada: {position_key}")
