# utils/role_config.py
# -*- coding: utf-8 -*-
"""
Configuración de roles y métricas por posición.

ACTUALIZADO para usar el nuevo módulo scoring/ en lugar de parsear archivos.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
from pathlib import Path


# =============================================================================
# MÉTRICAS DETALLADAS POR POSICIÓN (importadas desde scoring/)
# =============================================================================

# Estas son las listas de métricas que antes se parseaban de los archivos
# Ahora las definimos directamente aquí para que role_config sea independiente

DETAIL_METRICS_BY_POSITION = {
    "Golero": {
        "Efectividad": [
            ("gk_goals_prevented_per90", 0.50, False),
            ("gk_save_pct", 0.25, False),
            ("gk_errors_leading_to_shot_per90", 0.10, True),
            ("gk_errors_leading_to_goal_per90", 0.15, True),
        ],
        "Dominio de Área": [
            ("gk_claims_per90", 0.50, False),
            ("gk_shots_open_play_in_box_against_per90", 0.50, True),
        ],
        "Juego de Pies": [
            ("gk_pass_obv_per90", 0.40, False),
            ("gk_long_ball_pct", 0.20, False),
            ("gk_pressured_passes_def_third_per90", 0.20, False),
            ("gk_pressured_passes_def_third_completion_pct", 0.20, False),
        ],
        "Fuera del Área": [
            ("gk_actions_outside_box_per90", 0.50, False),
            ("gk_aggressive_distance_avg", 0.50, False),
        ],
    },
    
    "Zaguero": {
        "Acción Defensiva": [
            ("duel_success_rate", 0.25, False),
            ("tackle_success_pct", 0.25, False),
            ("ball_recovery_success_pct", 0.15, False),
            ("interception_success_rate", 0.10, False),
            ("clearances_total_per90", 0.10, False),
            ("blocks_total_per90", 0.05, False),
            ("defensive_actions_lost_per90", 0.10, True),
        ],
        "Control Defensivo": [
            ("pressure_per90", 0.15, False),
            ("counterpress_per90", 0.15, False),
            ("obv_into_per90", 0.25, True),
            ("obv_from_per90", 0.25, True),
            ("shots_from_area_per90", 0.10, True),
            ("xg_from_area_per90", 0.10, True),
        ],
        "Progresión": [
            ("pass_completion_rate", 0.15, False),
            ("pass_into_final_third_per90", 0.20, False),
            ("pass_switch_per90", 0.20, False),
            ("carry_into_final_third_per90", 0.10, False),
            ("pass_through_ball_per90", 0.15, False),
            ("obv_total_net_type_pass_per90", 0.20, False),
        ],
        "Impacto Ofensivo": [
            ("shot_statsbomb_xg_play_pattern_regular_play_per90", 0.15, False),
            ("shot_statsbomb_xg_play_pattern_from_corner_per90", 0.075, False),
            ("shot_statsbomb_xg_play_pattern_from_free_kick_per90", 0.075, False),
            ("obv_total_net_play_pattern_regular_play_per90", 0.15, False),
            ("obv_total_net_play_pattern_from_free_kick_per90", 0.075, False),
            ("obv_total_net_play_pattern_from_corner_per90", 0.075, False),
        ],
    },
    
    "Lateral": {
        "Defensivo (Exec)": [
            ("duel_success_rate", 0.25, False),
            ("tackle_success_pct", 0.25, False),
            ("ball_recovery_success_pct", 0.15, False),
            ("interception_success_rate", 0.10, False),
            ("clearances_total_per90", 0.10, False),
            ("blocks_total_per90", 0.05, False),
            ("defensive_actions_lost_per90", 0.10, True),
        ],
        "Defensivo (OBV)": [
            ("obv_total_net_type_duel_per90", 0.20, False),
            ("obv_total_net_duel_type_tackle_per90", 0.25, False),
            ("obv_total_net_type_interception_per90", 0.20, False),
            ("obv_total_net_type_ball_recovery_per90", 0.20, False),
            ("obv_total_net_type_clearance_per90", 0.15, False),
        ],
        "Presión": [
            ("pressure_per90", 0.35, False),
            ("n_events_third_attacking_pressure_per90", 0.35, False),
            ("ball_recovery_offensive_per90", 0.15, False),
            ("counterpress_per90", 0.15, False),
        ],
        "Profundidad": [
            ("pass_into_final_third_per90", 0.15, False),
            ("carry_into_final_third_per90", 0.25, False),
            ("n_events_third_attacking_pass_per90", 0.20, False),
            ("n_events_third_attacking_pass_cross_openplay_per90", 0.20, False),
            ("xa_per90", 0.20, False),
        ],
        "Calidad": [
            ("obv_total_net_type_pass_per90", 0.45, False),
            ("obv_total_net_type_dribble_per90", 0.25, False),
            ("obv_total_net_type_carry_per90", 0.20, False),
            ("total_turnovers_per90", 0.10, True),
        ],
    },
    
    "Volante": {
        "Posesión": [
            ("complete_passes_per90", 0.20, False),
            ("completed_passes_under_pressure_per90", 0.30, False),
            ("total_turnovers_per90", 0.20, True),
            ("obv_total_net_type_pass_per90", 0.30, False),
        ],
        "Progresión": [
            ("pass_into_final_third_per90", 0.20, False),
            ("carry_into_final_third_per90", 0.20, False),
            ("obv_total_net_type_carry_per90", 0.20, False),
            ("pass_switch_per90", 0.20, False),
            ("pass_through_ball_per90", 0.20, False),
        ],
        "Territoriales": [
            ("n_events_third_defensive_pressure_per90", 0.12, False),
            ("n_events_third_middle_pressure_per90", 0.18, False),
            ("counterpress_per90", 0.05, False),
            ("n_events_third_defensive_ball_recovery_per90", 0.15, False),
            ("n_events_third_middle_ball_recovery_per90", 0.20, False),
            ("obv_total_net_type_interception_per90", 0.15, False),
            ("obv_total_net_type_ball_recovery_per90", 0.15, False),
        ],
        "Contención": [
            ("duel_tackle_per90", 0.22, False),
            ("obv_total_net_duel_type_tackle_per90", 0.23, False),
            ("interception_success_rate", 0.17, False),
            ("obv_total_net_type_interception_per90", 0.23, False),
            ("n_events_third_defensive_interception_per90", 0.15, False),
        ],
    },
    
    "Interior/Mediapunta": {
        "Box to Box": [
            ("n_events_third_defensive_ball_recovery_per90", 0.11, False),
            ("n_events_third_middle_ball_recovery_per90", 0.14, False),
            ("n_events_third_attacking_ball_recovery_per90", 0.11, False),
            ("n_events_third_defensive_duel_per90", 0.08, False),
            ("n_events_third_middle_duel_per90", 0.14, False),
            ("n_events_third_attacking_duel_per90", 0.08, False),
            ("carry_into_final_third_per90", 0.10, False),
            ("touches_in_opp_box_per90", 0.10, False),
            ("shot_touch_pct", 0.05, False),
            ("total_touches_per90", 0.09, False),
        ],
        "Desequilibrio": [
            ("obv_total_net_type_dribble_per90", 0.30, False),
            ("obv_total_net_type_carry_per90", 0.25, False),
            ("carry_into_final_third_per90", 0.15, False),
            ("pass_into_final_third_per90", 0.10, False),
            ("obv_total_net_type_shot_per90", 0.10, False),
            ("shot_statsbomb_xg_per90", 0.10, False),
        ],
        "Organización": [
            ("obv_total_net_type_pass_per90", 0.30, False),
            ("complete_passes_per90", 0.20, False),
            ("pass_shot_assist_per90", 0.12, False),
            ("obv_total_net_third_attacking_pass_per90", 0.13, False),
            ("obv_total_net_play_pattern_regular_play_per90", 0.10, False),
            ("total_turnovers_per90", 0.15, True),
        ],
        "Contención/Presión": [
            ("n_events_third_middle_pressure_per90", 0.18, False),
            ("n_events_third_attacking_pressure_per90", 0.12, False),
            ("counterpress_per90", 0.10, False),
            ("n_events_third_middle_ball_recovery_per90", 0.12, False),
            ("n_events_third_attacking_ball_recovery_per90", 0.13, False),
            ("obv_total_net_duel_type_tackle_per90", 0.10, False),
            ("duel_tackle_per90", 0.10, False),
            ("obv_total_net_type_interception_per90", 0.08, False),
            ("obv_total_net_third_middle_interception_per90", 0.07, False),
        ],
    },
    
    "Extremo": {
        "Compromiso defensivo": [
            ("n_events_third_attacking_pressure_per90", 0.20, False),
            ("counterpress_per90", 0.20, False),
            ("n_events_third_attacking_ball_recovery_per90", 0.15, False),
            ("obv_total_net_type_ball_recovery_per90", 0.20, False),
            ("obv_total_net_type_interception_per90", 0.10, False),
            ("pressure_per90", 0.15, False),
        ],
        "Desequilibrio": [
            ("obv_total_net_type_dribble_per90", 0.18, False),
            ("obv_total_net_type_carry_per90", 0.18, False),
            ("carry_into_final_third_per90", 0.10, False),
            ("pass_into_final_third_per90", 0.08, False),
            ("pass_shot_assist_per90", 0.15, False),
            ("pass_goal_assist_per90", 0.05, False),
            ("xa_per90", 0.12, False),
            ("obv_total_net_type_pass_per90", 0.14, False),
        ],
        "Finalización": [
            ("shot_statsbomb_xg_per90", 0.35, False),
            ("obv_total_net_type_shot_per90", 0.25, False),
            ("xg_per_shot", 0.20, False),
            ("touches_in_opp_box_per90", 0.20, False),
        ],
        "Zona de influencia": [
            ("obv_from_ext_per90", 0.35, False),
            ("obv_from_int_per90", 0.35, False),
            ("obv_total_net_type_pass_per90", 0.30, False),
        ],
    },
    
    "Delantero": {
        "Finalización": [
            ("xg_per_shot", 0.20, False),
            ("shot_statsbomb_xg_per90", 0.18, False),
            ("obv_total_net_type_shot_per90", 0.15, False),
            ("goals_per90", 0.05, False),
            ("touches_in_opp_box_per90", 0.15, False),
            ("touches_in_opp_box_pct", 0.10, False),
            ("obv_total_net_play_pattern_regular_play_per90", 0.10, False),
            ("total_shots_per90", 0.04, False),
            ("shot_touch_pct", 0.03, False),
        ],
        "Presionante": [
            ("pressure_per90", 0.30, False),
            ("n_events_third_attacking_pressure_per90", 0.20, False),
            ("counterpress_per90", 0.15, False),
            ("ball_recovery_offensive_per90", 0.15, False),
            ("n_events_third_attacking_ball_recovery_per90", 0.10, False),
            ("obv_total_net_type_interception_per90", 0.05, False),
            ("obv_total_net_type_ball_recovery_per90", 0.05, False),
        ],
        "Conector": [
            ("complete_passes_per90", 0.25, False),
            ("pass_completion_rate", 0.15, False),
            ("pass_into_final_third_per90", 0.15, False),
            ("obv_total_net_type_pass_per90", 0.25, False),
            ("pass_shot_assist_per90", 0.15, False),
            ("total_turnovers_per90", 0.05, True),
        ],
        "Disruptivo": [
            ("obv_total_net_type_dribble_per90", 0.40, False),
            ("obv_total_net_type_carry_per90", 0.35, False),
            ("carry_into_final_third_per90", 0.15, False),
            ("pass_into_final_third_per90", 0.10, False),
        ],
    },
}


# =============================================================================
# CONFIGURACIÓN DE RADAR MACRO (sin cambios)
# =============================================================================

ROLE_MACRO: Dict[str, List[Tuple[str, str]]] = {
    "Golero": [
        ("Score_Effectiveness", "Efectividad"),
        ("Score_Area_Domination", "Dominio de Área"),
        ("Score_Foot_Play", "Juego de Pies"),
        ("Score_Outside_Box", "Fuera del Área"),
    ],
    "Zaguero": [
        ("Score_AccionDefensiva", "Acción Defensiva"),
        ("Score_ControlDefensivo", "Control Defensivo"),
        ("Score_Progresion", "Progresión"),
        ("Score_ImpactoOfensivo", "Impacto Ofensivo"),
    ],
    "Lateral": [
        ("Score_Profundidad", "Profundidad"),
        ("Score_Calidad", "Calidad"),
        ("Score_Presion", "Presión"),
        ("Score_Defensivo", "Defensivo"),
    ],
    "Volante": [
        ("Score_Posesion", "Posesión"),
        ("Score_Progresion", "Progresión"),
        ("Score_Territoriales", "Territoriales"),
        ("Score_Contencion", "Contención"),
    ],
    "Interior/Mediapunta": [
        ("Score_BoxToBox", "Box to Box"),
        ("Score_Desequilibrio", "Desequilibrio"),
        ("Score_Organizacion", "Organización"),
        ("Score_ContencionPresion", "Contención/Presión"),
    ],
    "Extremo": [
        ("Score_CompromisoDef", "Compromiso defensivo"),
        ("Score_Desequilibrio", "Desequilibrio"),
        ("Score_Finalizacion", "Finalización"),
        ("Score_ZonaInfluencia", "Zona de influencia"),
    ],
    "Delantero": [
        ("Score_Finalizacion", "Finalización"),
        ("Score_Presionante", "Presionante"),
        ("Score_Conector", "Conector"),
        ("Score_Disruptivo", "Disruptivo"),
    ],
}


# =============================================================================
# FUNCIONES PÚBLICAS (actualizadas para usar diccionario en lugar de parsear)
# =============================================================================

def get_macro_config(position_key: str) -> List[Tuple[str, str]]:
    """
    Devuelve lista de (col_score, label) para el radar macro.
    
    Parameters
    ----------
    position_key : str
        Nombre de la posición
    
    Returns
    -------
    List[Tuple[str, str]]
        Lista de tuplas (columna_score, etiqueta_display)
    """
    return ROLE_MACRO.get(position_key, [])


def get_detail_categories(position_key: str) -> List[str]:
    """
    Devuelve labels disponibles para el lollipop detallado.
    
    Parameters
    ----------
    position_key : str
        Nombre de la posición
    
    Returns
    -------
    List[str]
        Lista de nombres de categorías detalladas
    """
    metrics = DETAIL_METRICS_BY_POSITION.get(position_key)
    if not metrics:
        return []
    return list(metrics.keys())


def get_detail_metric_list(
    position_key: str, 
    detail_label: str, 
    base_dir: Path = None  # Ya no se usa pero mantenemos firma
) -> List[Tuple[str, float, bool]] | None:
    """
    Devuelve la lista de métricas para una categoría detallada.
    
    Parameters
    ----------
    position_key : str
        Nombre de la posición
    detail_label : str
        Nombre de la categoría detallada
    base_dir : Path, optional
        No se usa (mantenido por compatibilidad)
    
    Returns
    -------
    List[Tuple[str, float, bool]] | None
        Lista de tuplas (metric_name, weight, inverted) o None si no existe
    """
    metrics = DETAIL_METRICS_BY_POSITION.get(position_key)
    if not metrics:
        return None
    
    return metrics.get(detail_label)


# =============================================================================
# PESOS DE CATEGORÍAS (sin cambios)
# =============================================================================

CATEGORY_WEIGHTS_BY_POSITION: Dict[str, Dict[str, float]] = {
    "Golero": {
        "Score_Effectiveness": 0.50,
        "Score_Area_Domination": 0.20,
        "Score_Foot_Play": 0.15,
        "Score_Outside_Box": 0.15,
    },
    "Zaguero": {
        "Score_AccionDefensiva": 0.25,
        "Score_ControlDefensivo": 0.45,
        "Score_Progresion": 0.20,
        "Score_ImpactoOfensivo": 0.10,
    },
    "Lateral": {
        "Score_Profundidad": 0.30,
        "Score_Calidad": 0.30,
        "Score_Presion": 0.20,
        "Score_Defensivo": 0.20,
    },
    "Volante": {
        "Score_Posesion": 0.25,
        "Score_Progresion": 0.30,
        "Score_Territoriales": 0.25,
        "Score_Contencion": 0.20,
    },
    "Interior/Mediapunta": {
        "Score_BoxToBox": 0.25,
        "Score_Desequilibrio": 0.30,
        "Score_Organizacion": 0.25,
        "Score_ContencionPresion": 0.20,
    },
    "Extremo": {
        "Score_CompromisoDef": 0.20,
        "Score_Desequilibrio": 0.35,
        "Score_Finalizacion": 0.30,
        "Score_ZonaInfluencia": 0.15,
    },
    "Delantero": {
        "Score_Finalizacion": 0.40,
        "Score_Presionante": 0.10,
        "Score_Conector": 0.25,
        "Score_Disruptivo": 0.25,
    },
}


def get_category_weights(position_key: str) -> Dict[str, float]:
    """
    Devuelve los pesos de categorías para una posición.
    
    Parameters
    ----------
    position_key : str
        Nombre de la posición
    
    Returns
    -------
    Dict[str, float]
        Diccionario de {categoria: peso}
    """
    return CATEGORY_WEIGHTS_BY_POSITION.get(position_key, {})


def strip_score_prefix(cat: str) -> str:
    """
    Quita el prefijo 'Score_' de nombres de categorías.
    
    Parameters
    ----------
    cat : str
        Nombre de categoría (ej: 'Score_Progresion')
    
    Returns
    -------
    str
        Nombre sin prefijo (ej: 'Progresion')
    
    Examples
    --------
    >>> strip_score_prefix('Score_Progresion')
    'Progresion'
    >>> strip_score_prefix('Score_BoxToBox')
    'BoxToBox'
    """
    return cat.replace("Score_", "", 1)