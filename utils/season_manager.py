# utils/season_manager.py
# -*- coding: utf-8 -*-
"""
Gestión centralizada de temporadas.

Este módulo maneja:
- Detección automática de temporadas disponibles
- Selector de temporada en sidebar (singleton)
- Paths dinámicos según temporada seleccionada
"""
from __future__ import annotations

from pathlib import Path
import re
import streamlit as st

# Importar paths base (sin importar PER90_CSV para evitar circular)
from config import DATA_DIR

# Directorio donde están los archivos per90
PER90_DIR = DATA_DIR / "per90"


# =============================================================================
# DETECCIÓN DE TEMPORADAS
# =============================================================================

def get_available_seasons() -> list[str]:
    """
    Detecta temporadas disponibles basándose en archivos existentes.
    
    Busca archivos con patrón: all_players_complete_YYYY_YYYY.csv
    
    Returns:
        Lista de temporadas ordenadas (más reciente primero)
        Ejemplo: ["2025_2026", "2024_2025"]
    """
    pattern = re.compile(r"all_players_complete_(\d{4}_\d{4})\.csv")
    seasons = []
    
    if PER90_DIR.exists():
        for f in PER90_DIR.glob("all_players_complete_*.csv"):
            match = pattern.match(f.name)
            if match:
                seasons.append(match.group(1))
    
    # Ordenar descendente (más reciente primero)
    seasons.sort(reverse=True)
    
    return seasons


def format_season_label(season_id: str) -> str:
    """
    Formatea ID de temporada para mostrar al usuario.
    
    Args:
        season_id: "2025_2026"
    
    Returns:
        "2025/2026"
    """
    if "_" in season_id:
        parts = season_id.split("_")
        return f"{parts[0]}/{parts[1]}"
    return season_id


# =============================================================================
# SELECTOR DE TEMPORADA (SINGLETON EN SESSION STATE)
# =============================================================================

def get_selected_season() -> str:
    """
    Retorna la temporada actualmente seleccionada.
    
    Si no hay ninguna seleccionada, usa la más reciente disponible.
    
    Returns:
        ID de temporada (ej: "2025_2026")
    """
    available = get_available_seasons()
    
    if not available:
        # Fallback si no hay archivos
        return "2025_2026"
    
    # Inicializar si no existe
    if "selected_season" not in st.session_state:
        st.session_state.selected_season = available[0]
    
    # Validar que la temporada guardada aún existe
    if st.session_state.selected_season not in available:
        st.session_state.selected_season = available[0]
    
    return st.session_state.selected_season


def season_selector() -> str:
    """
    Renderiza el selector de temporada en el sidebar.
    
    IMPORTANTE: Llamar solo una vez por página, idealmente antes 
    de otros filtros del sidebar.
    
    Returns:
        ID de temporada seleccionada (ej: "2025_2026")
    """
    available = get_available_seasons()
    
    if not available:
        st.sidebar.warning("⚠️ No se encontraron archivos de temporada")
        return "2025_2026"
    
    # Crear opciones con labels formateados
    options = available
    
    # Índice actual
    current = get_selected_season()
    current_idx = options.index(current) if current in options else 0
    
    # Selector
    st.sidebar.markdown("### 📅 Temporada")
    selected = st.sidebar.selectbox(
        "Seleccionar temporada",
        options=options,
        index=current_idx,
        format_func=format_season_label,
        key="season_selector_widget",
        label_visibility="collapsed"
    )
    
    # Actualizar session state
    st.session_state.selected_season = selected
    
    st.sidebar.markdown("---")
    
    return selected


# =============================================================================
# PATHS DINÁMICOS
# =============================================================================

def get_per90_path(season_id: str | None = None) -> Path:
    """
    Retorna el path al CSV de datos per90 para una temporada.
    
    Args:
        season_id: ID de temporada. Si None, usa la seleccionada.
    
    Returns:
        Path al archivo CSV
    """
    if season_id is None:
        season_id = get_selected_season()
    
    return PER90_DIR / f"all_players_complete_{season_id}.csv"


def get_minutes_path(season_id: str | None = None) -> Path:
    """
    Retorna el path al CSV de minutos por partido.
    
    Args:
        season_id: ID de temporada. Si None, usa la seleccionada.
    
    Returns:
        Path al archivo CSV
    """
    if season_id is None:
        season_id = get_selected_season()
    
    return OUTPUTS_DIR / f"player_minutes_by_match_{season_id}.csv"


def get_scores_path(position: str, season_id: str | None = None) -> Path:
    """
    Retorna el path al CSV de scores de una posición.
    
    Args:
        position: Nombre de posición (ej: "delantero", "extremo")
        season_id: ID de temporada. Si None, usa la seleccionada.
    
    Returns:
        Path al archivo CSV
    """
    if season_id is None:
        season_id = get_selected_season()
    
    return DATA_DIR / "scores" / f"{position.lower()}_scores_{season_id}.csv"


# =============================================================================
# COMPATIBILIDAD CON CÓDIGO EXISTENTE
# =============================================================================

def get_current_per90_csv() -> Path:
    """
    Función de compatibilidad que retorna PER90_CSV dinámico.
    
    Usar en lugar de importar PER90_CSV directamente de config.
    
    Returns:
        Path al CSV de la temporada seleccionada
    """
    return get_per90_path()