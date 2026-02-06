# utils/scoring_wrappers.py
# -*- coding: utf-8 -*-
"""
Wrapper único de scoring que usa el nuevo módulo scoring/.

Consume SIEMPRE DataFrames (no CSVs) y llama a las funciones
legacy del módulo scoring/ para mantener compatibilidad.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

# Imports corregidos - TODAS las funciones necesarias
from scoring.goalkeeper import run_goalkeeper_scoring
from scoring.defenders import run_cb_scoring, score_lateral_df  # ← Agregado score_lateral_df
from scoring.midfielders import run_interior_scoring, run_volante_scoring  # ← Agregado run_volante_scoring
from scoring.forwards import run_extremo_scoring, run_delantero_scoring


def _filter_teams(df: pd.DataFrame, selected_teams: list[str] | None) -> pd.DataFrame:
    """
    Filtra DataFrame por equipos seleccionados.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame base con datos de jugadores
    selected_teams : list[str] | None
        Lista de equipos a filtrar, o None para no filtrar
    
    Returns
    -------
    pd.DataFrame
        DataFrame filtrado (o sin cambios si no hay filtro)
    """
    if not selected_teams:
        return df
    
    # Prioridad: columna cruda del per90 (antes de renombres)
    if "teams" in df.columns:
        return df[df["teams"].astype(str).isin([str(x) for x in selected_teams])].copy()
    
    # Fallback por si algún per90 viene distinto
    if "team_name" in df.columns:
        return df[df["team_name"].astype(str).isin([str(x) for x in selected_teams])].copy()
    
    # Si no hay columna, no filtramos
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
    Wrapper único de scoring para todas las posiciones.
    
    - Consume SIEMPRE DataFrame (per90)
    - Aplica filtro de equipos ANTES del scoring (para que todos los módulos lo respeten)
    - Llama a 1 entrypoint por posición usando SIEMPRE df=...
    
    Parameters
    ----------
    df_base : pd.DataFrame
        DataFrame con datos per90 de todos los jugadores
    position_key : str
        Clave de posición: 'Golero', 'Zaguero', 'Lateral', 'Volante', 
        'Interior/Mediapunta', 'Extremo', 'Delantero'
    min_minutes : int
        Minutos mínimos jugados
    min_matches : int
        Partidos mínimos jugados
    selected_teams : list[str] | None, optional
        Lista de equipos a filtrar, o None para todos
    
    Returns
    -------
    pd.DataFrame
        DataFrame con scores calculados por posición
    
    Raises
    ------
    ValueError
        Si no hay jugadores tras filtros o si position_key no es válida
    
    Examples
    --------
    >>> df = pd.read_csv("outputs/all_players_complete.csv")
    >>> result = compute_scoring_from_df(
    ...     df_base=df,
    ...     position_key="Delantero",
    ...     min_minutes=450,
    ...     min_matches=3,
    ...     selected_teams=["Monterrey", "Tigres"]
    ... )
    """
    df = df_base.copy()
    
    # 1) Filtro equipos (antes del scoring)
    df = _filter_teams(df, selected_teams)
    
    if df.empty:
        raise ValueError("No hay filas tras el filtro de equipos.")
    
    # 2) Dispatch consistente por posición
    if position_key == "Golero":
        return run_goalkeeper_scoring(
            df=df,
            position_group="Golero",
            min_minutes=min_minutes,
            min_matches=min_matches,
        )
    
    if position_key == "Zaguero":
        return run_cb_scoring(
            df=df,
            position_group="Zaguero",
            min_minutes=min_minutes,
            min_matches=min_matches,
        )
    
    if position_key == "Lateral":
        # Usar score_lateral_df (existe en scoring/defenders.py)
        return score_lateral_df(
            per90_df=df,
            position_group="Lateral",
            min_minutes=min_minutes,
            min_matches=min_matches,
            verbose=False,  # ← No imprimir en Streamlit
        )
    
    if position_key == "Volante":
        # Ahora está importado correctamente
        return run_volante_scoring(
            df=df,
            position_group="Volante",
            min_minutes=min_minutes,
            min_matches=min_matches,
        )
    
    if position_key == "Interior/Mediapunta":
        return run_interior_scoring(
            df=df,
            position_group="Interior/Mediapunta",
            min_minutes=min_minutes,
            min_matches=min_matches,
        )
    
    if position_key == "Extremo":
        return run_extremo_scoring(
            df=df,
            position_group="Extremo",
            min_minutes=min_minutes,
            min_matches=min_matches,
        )
    
    if position_key == "Delantero":
        return run_delantero_scoring(
            df=df,
            position_group="Delantero",
            min_minutes=min_minutes,
            min_matches=min_matches,
        )
    
    # Si llegamos acá, position_key no es válida
    raise ValueError(
        f"position_key no reconocido: '{position_key}'. "
        f"Valores válidos: 'Golero', 'Zaguero', 'Lateral', 'Volante', "
        f"'Interior/Mediapunta', 'Extremo', 'Delantero'"
    )