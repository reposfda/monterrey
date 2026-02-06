# utils/scoring_wrappers.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import streamlit as st

from position_scoring_golero import run_goalkeeper_scoring
from position_scoring_defensor_central import run_cb_scoring
from position_scoring_lateral import run_lateral_scoring
from position_scoring_volante import run_volante_scoring
from position_scoring_interior import run_interior_scoring
from position_scoring_extremos import run_extremo_scoring
from position_scoring_delantero import run_delantero_scoring


def _filter_teams(df: pd.DataFrame, selected_teams: list[str] | None) -> pd.DataFrame:
    if not selected_teams:
        return df

    # prioridad: columna cruda del per90 (antes de renombres)
    if "teams" in df.columns:
        return df[df["teams"].astype(str).isin([str(x) for x in selected_teams])].copy()

    # fallback por si algún per90 viene distinto
    if "team_name" in df.columns:
        return df[df["team_name"].astype(str).isin([str(x) for x in selected_teams])].copy()

    # si no hay columna, no filtramos
    return df


@st.cache_data(show_spinner=False)
def compute_scoring_from_df(
    df_base: pd.DataFrame,
    position_key: str,
    min_minutes: int,
    min_matches: int,
    selected_teams: list[str] | None = None,
) -> pd.DataFrame:
    """
    Wrapper único de scoring.
    - Consume SIEMPRE DataFrame (per90).
    - Aplica filtro de equipos ANTES del scoring (para que todos los módulos lo respeten).
    - Llama a 1 entrypoint por posición usando SIEMPRE df=...
    """
    df = df_base.copy()

    # 1) filtro equipos (antes del scoring)
    df = _filter_teams(df, selected_teams)

    if df.empty:
        raise ValueError("No hay filas tras el filtro de equipos.")

    # 2) dispatch consistente
    if position_key == "Golero":
        return run_goalkeeper_scoring(df=df, position_group="Golero", min_minutes=min_minutes, min_matches=min_matches)

    if position_key == "Zaguero":
        return run_cb_scoring(df=df, position_group="Zaguero", min_minutes=min_minutes, min_matches=min_matches)

    if position_key == "Lateral":
        return run_lateral_scoring(df=df, position_group="Lateral", min_minutes=min_minutes, min_matches=min_matches)

    if position_key == "Volante":
        return run_volante_scoring(df=df, position_group="Volante", min_minutes=min_minutes, min_matches=min_matches)

    if position_key == "Interior/Mediapunta":
        return run_interior_scoring(df=df, position_group="Interior/Mediapunta", min_minutes=min_minutes, min_matches=min_matches)

    if position_key == "Extremo":
        return run_extremo_scoring(df=df, position_group="Extremo", min_minutes=min_minutes, min_matches=min_matches)

    if position_key == "Delantero":
        return run_delantero_scoring(df=df, position_group="Delantero", min_minutes=min_minutes, min_matches=min_matches)

    raise ValueError(f"position_key no reconocido: {position_key}")
