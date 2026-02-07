# utils/scoring_wrappers.py
# -*- coding: utf-8 -*-
"""
Wrapper único de scoring para todas las posiciones.

Agrega columna "Flags" con perfiles descriptivos del jugador.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from scoring.goalkeeper import run_goalkeeper_scoring
from scoring.defenders import run_cb_scoring, score_lateral_df
from scoring.midfielders import run_interior_scoring, run_volante_scoring
from scoring.forwards import run_extremo_scoring, run_delantero_scoring


# =============================================================================
# MAPEO DE CATEGORÍAS A ETIQUETAS DESCRIPTIVAS POR POSICIÓN
# =============================================================================

PROFILE_LABELS = {
    "Golero": {
        "Score_Effectiveness": "Shot Stopper",
        "Score_Area_Domination": "Dominante",
        "Score_Foot_Play": "Con Pies",
        "Score_Outside_Box": "Sweeper",
    },
    "Zaguero": {
        "Score_AccionDefensiva": "Acción Def",
        "Score_ControlDefensivo": "Control Def",
        "Score_Progresion": "Progresión",
        "Score_ImpactoOfensivo": "Ofensivo",
    },
    "Lateral": {
        "Score_Profundidad": "Profundos",
        "Score_Calidad": "Técnicos",
        "Score_Presion": "Presionantes",
        "Score_Defensivo": "Protectores",
    },
    "Volante": {
        "Score_Posesion": "Posesión",
        "Score_Progresion": "Progresión",
        "Score_Territoriales": "Territoriales",
        "Score_Contencion": "Contención",
    },
    "Interior/Mediapunta": {
        "Score_BoxToBox": "Box to Box",
        "Score_Desequilibrio": "Desequilibrantes",
        "Score_Organizacion": "Organizadores",
        "Score_ContencionPresion": "Contención/Presión",
    },
    "Extremo": {
        "Score_CompromisoDef": "Compromiso Def",
        "Score_Desequilibrio": "Desequilibrio",
        "Score_Finalizacion": "Finalización",
        "Score_ZonaInfluencia": "Zona Influencia",
    },
    "Delantero": {
        "Score_Finalizacion": "Killer",
        "Score_Presionante": "Presionante",
        "Score_Conector": "Falso 9",
        "Score_Disruptivo": "Disruptivo",
    },
}


def _filter_teams(df: pd.DataFrame, selected_teams: list[str] | None) -> pd.DataFrame:
    """Filtra DataFrame por equipos seleccionados."""
    if not selected_teams:
        return df
    
    if "teams" in df.columns:
        return df[df["teams"].astype(str).isin([str(x) for x in selected_teams])].copy()
    
    if "team_name" in df.columns:
        return df[df["team_name"].astype(str).isin([str(x) for x in selected_teams])].copy()
    
    return df


def _generate_profile_flags(df: pd.DataFrame, position_key: str) -> pd.DataFrame:
    """
    Genera columna 'Flags' con perfiles descriptivos basados en flags activos.
    
    Solo incluye etiquetas donde flag == 1.
    """
    labels_map = PROFILE_LABELS.get(position_key, {})
    
    if not labels_map:
        df["Flags"] = "Sin perfil"
        return df
    
    def get_profile_tags(row):
        """Genera tags solo para flags activos (valor == 1)."""
        tags = []
        
        for score_cat, label in labels_map.items():
            flag_col = f"flag_{score_cat}"
            
            # Verificar que existe la columna
            if flag_col not in row.index:
                continue
            
            # Obtener valor y verificar que sea == 1
            flag_value = row[flag_col]
            
            # Solo agregar si el flag está activo
            if pd.notna(flag_value) and int(flag_value) == 1:
                tags.append(label)
        
        return " | ".join(tags) if tags else "Balanceado"
    
    df["Flags"] = df.apply(get_profile_tags, axis=1)
    return df


@st.cache_data(show_spinner=False)
def compute_scoring_from_df(
    df_base: pd.DataFrame,
    position_key: str,
    min_minutes: int,
    min_matches: int,
    selected_teams: list[str] | None = None,
) -> pd.DataFrame:
    """Wrapper único de scoring para todas las posiciones."""
    df = df_base.copy()
    
    # 1) Filtro equipos
    df = _filter_teams(df, selected_teams)
    
    if df.empty:
        raise ValueError("No hay filas tras el filtro de equipos.")
    
    # 2) Dispatch por posición
    if position_key == "Golero":
        result = run_goalkeeper_scoring(
            df=df,
            position_group="Golero",
            min_minutes=min_minutes,
            min_matches=min_matches,
        )
    
    elif position_key == "Zaguero":
        result = run_cb_scoring(
            df=df,
            position_group="Zaguero",
            min_minutes=min_minutes,
            min_matches=min_matches,
        )
    
    elif position_key == "Lateral":
        result = score_lateral_df(
            per90_df=df,
            position_group="Lateral",
            min_minutes=min_minutes,
            min_matches=min_matches,
            verbose=False,
        )
    
    elif position_key == "Volante":
        result = run_volante_scoring(
            df=df,
            position_group="Volante",
            min_minutes=min_minutes,
            min_matches=min_matches,
        )
    
    elif position_key == "Interior/Mediapunta":
        result = run_interior_scoring(
            df=df,
            position_group="Interior/Mediapunta",
            min_minutes=min_minutes,
            min_matches=min_matches,
        )
    
    elif position_key == "Extremo":
        result = run_extremo_scoring(
            df=df,
            position_group="Extremo",
            min_minutes=min_minutes,
            min_matches=min_matches,
        )
    
    elif position_key == "Delantero":
        result = run_delantero_scoring(
            df=df,
            position_group="Delantero",
            min_minutes=min_minutes,
            min_matches=min_matches,
        )
    
    else:
        raise ValueError(f"position_key no reconocido: {position_key}")
    
    # 3) Generar columna Flags
    result = _generate_profile_flags(result, position_key)
    
    return result