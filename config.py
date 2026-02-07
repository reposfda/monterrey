# config.py
# -*- coding: utf-8 -*-
"""
Configuración centralizada del proyecto.

Incluye:
- Paths del proyecto
- Colores corporativos
- Estilos CSS globales
- Constantes de configuración
- Parámetros de procesamiento
"""

from __future__ import annotations
from pathlib import Path
import streamlit as st


# =============================================================================
# PATHS DEL PROYECTO
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

# Directorios principales
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
ASSETS_DIR = BASE_DIR / "assets"
UTILS_DIR = BASE_DIR / "utils"

# Directorios para multi-temporada (futuro)
SEASONS_DATA_DIR = DATA_DIR / "seasons"
SEASONS_OUTPUT_DIR = OUTPUTS_DIR / "seasons"

# Assets
LOGO_PATH = ASSETS_DIR / "monterrey_logo.png"

# Archivos de datos (actual - temporada única)
# NOTA: Estos paths cambiarán cuando implementemos multi-temporada
EVENTS_CSV = DATA_DIR / "events_2025_2026.csv"
# PER90_CSV = OUTPUTS_DIR / "all_players_complete_2025_2026.csv"


# =============================================================================
# COLORES CORPORATIVOS MONTERREY
# =============================================================================

class Colors:
    """Paleta de colores corporativos del Club de Fútbol Monterrey"""
    
    # Colores principales
    PRIMARY_BG = "#0B1F38"      # Azul oscuro (fondo principal)
    SECONDARY_BG = "#091325"    # Azul más oscuro (sidebar)
    ACCENT = "#6CA0DC"          # Azul claro (acentos)
    TEXT = "#FFFFFF"            # Blanco (texto)
    GOLD = "#c49308"            # Dorado (highlights, sliders)
    
    # Colores para tablas
    TABLE_BG = "#ffffff"        # Fondo de tabla
    TABLE_TEXT = "#1f2933"      # Texto de tabla
    TABLE_HOVER = "#e8f0ff"     # Hover en filas
    TABLE_EVEN = "#f4f6fb"      # Filas pares
    
    # Colores para gradientes (scores)
    GRADIENT_LOW = "#fee5d9"    # Bajo
    GRADIENT_MID = "#fcae91"    # Medio
    GRADIENT_HIGH = "#fb6a4a"   # Alto


# =============================================================================
# ESTILOS CSS GLOBALES
# =============================================================================

def get_global_css() -> str:
    """
    Genera el CSS global para la aplicación Streamlit.
    
    Incluye estilos para:
    - Background y layout general
    - Header
    - Sidebar
    - Tipografía
    - Sliders personalizados
    - Tablas corporativas
    
    Returns:
        str: CSS completo como string HTML
    """
    return f"""
    <style>
        /* ============================================
           LAYOUT GENERAL
           ============================================ */
        .stApp {{
            background-color: {Colors.PRIMARY_BG};
        }}
        
        .block-container {{
            padding-top: 1.5rem !important;
        }}

        /* ============================================
           HEADER
           ============================================ */
        header[data-testid="stHeader"] {{
            background-color: transparent;
        }}
        
        header[data-testid="stHeader"] > div {{
            background-color: {Colors.PRIMARY_BG};
            box-shadow: none;
        }}
        
        header[data-testid="stHeader"] * {{
            color: {Colors.TEXT} !important;
        }}
        
        header[data-testid="stHeader"] svg,
        header[data-testid="stHeader"] path {{
            fill: {Colors.TEXT} !important;
        }}

        /* ============================================
           SIDEBAR
           ============================================ */
        [data-testid="stSidebar"] > div:first-child {{
            background-color: {Colors.SECONDARY_BG};
        }}

        /* ============================================
           TIPOGRAFÍA
           ============================================ */
        h1, h2, h3, h4, h5, h6, p, label {{
            color: {Colors.TEXT} !important;
        }}
        
        a {{
            color: {Colors.ACCENT} !important;
        }}

        /* ============================================
           SLIDERS PERSONALIZADOS
           ============================================ */
        
        /* Labels de sliders */
        div[data-baseweb="slider"] span {{
            color: {Colors.GOLD} !important;
            font-weight: 700 !important;
        }}
        
        /* Fondo del track (transparente) */
        div.stSlider > div[data-baseweb="slider"] > div[data-testid="stTickBar"] > div {{
            background: transparent !important;
        }}
        
        /* Track del slider */
        div.stSlider > div[data-baseweb="slider"] > div > div {{
            background: {Colors.GOLD} !important;
            border-radius: 8px !important;
            height: 6px !important;
        }}
        
        /* Thumb (bolita) del slider */
        div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"] {{
            background-color: {Colors.GOLD} !important;
            box-shadow: 0 0 0 0.2rem rgba(196,147,8,0.3) !important;
            border: 2px solid {Colors.TEXT} !important;
        }}
        
        /* Valor del slider */
        div.stSlider > div[data-baseweb="slider"] > div > div > div > div {{
            color: {Colors.GOLD} !important;
            font-weight: 700 !important;
        }}

        /* ============================================
           TABLAS CORPORATIVAS
           ============================================ */
        table.mty-table {{
            border-collapse: separate;
            border-spacing: 0;
            width: 100%;
            background-color: {Colors.TABLE_BG};
            border-radius: 12px;
            overflow: hidden;
            font-size: 0.90rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        /* Header de tabla */
        table.mty-table thead {{
            background-color: {Colors.PRIMARY_BG};
        }}
        
        table.mty-table thead th {{
            color: {Colors.TEXT} !important;
            padding: 0.55rem 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 0.5px;
        }}
        
        /* Primera columna (alineada a izquierda) */
        table.mty-table thead th:first-child {{
            text-align: left !important;
            padding-left: 12px !important;
        }}
        
        /* Resto de columnas (centradas) */
        table.mty-table thead th:not(:first-child) {{
            text-align: center !important;
        }}

        /* Celdas del body */
        table.mty-table tbody td {{
            color: {Colors.TABLE_TEXT} !important;
            padding: 0.55rem 0.75rem;
        }}
        
        table.mty-table tbody td:first-child {{
            text-align: left !important;
            padding-left: 12px !important;
            font-weight: 500;
        }}
        
        table.mty-table tbody td:not(:first-child) {{
            text-align: center !important;
        }}

        /* Filas alternadas */
        table.mty-table tbody tr:nth-child(even) {{
            background-color: {Colors.TABLE_EVEN};
        }}
        
        /* Hover en filas */
        table.mty-table tbody tr:hover {{
            background-color: {Colors.TABLE_HOVER};
            transition: background-color 0.2s ease;
        }}
        
        /* ============================================
           BOTONES
           ============================================ */
        .stButton > button {{
            background-color: {Colors.GOLD};
            color: {Colors.PRIMARY_BG};
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }}
        
        .stButton > button:hover {{
            background-color: #dba909;
            box-shadow: 0 4px 12px rgba(196,147,8,0.3);
        }}
        
        /* ============================================
           SELECT BOXES Y MULTI-SELECT
           ============================================ */
        div[data-baseweb="select"] > div {{
            background-color: {Colors.SECONDARY_BG};
            border-color: {Colors.ACCENT};
        }}
        
        /* ============================================
           EXPANDERS
           ============================================ */
        .streamlit-expanderHeader {{
            background-color: {Colors.SECONDARY_BG};
            border-radius: 6px;
        }}
        
        /* ============================================
           MÉTRICAS (st.metric)
           ============================================ */
        [data-testid="stMetricValue"] {{
            color: {Colors.GOLD} !important;
            font-size: 1.5rem !important;
            font-weight: 700 !important;
        }}
        
        [data-testid="stMetricLabel"] {{
            color: {Colors.TEXT} !important;
        }}
    </style>
    """


def apply_global_styles():
    """
    Aplica los estilos globales a la página actual de Streamlit.
    
    Uso:
        from config import apply_global_styles
        
        apply_global_styles()
    
    IMPORTANTE: Debe llamarse DESPUÉS de st.set_page_config()
    """
    st.markdown(get_global_css(), unsafe_allow_html=True)


# =============================================================================
# CONSTANTES DE CONFIGURACIÓN
# =============================================================================

class Defaults:
    """Valores por defecto para filtros y configuraciones de la app"""
    
    # Filtros de jugadores
    MIN_MINUTES = 450           # Minutos mínimos para incluir jugador
    MIN_MATCHES = 5             # Partidos mínimos
    FLAG_QUANTILE = 0.75        # Percentil para flags (top 25%)
    
    # Paginación y límites
    TOP_N_RANKING = 10          # Top N jugadores en ranking
    MAX_COMPARISON = 5          # Máximo de jugadores para comparar
    
    # Configuración de gráficos
    RADAR_HEIGHT = 600          # Altura de radar charts
    LOLLIPOP_HEIGHT = 400       # Altura base de lollipop charts


class Processing:
    """Configuraciones para el procesamiento de eventos"""
    
    # Límites de tiempo
    ASSUME_END_CAP = 120        # Minuto máximo asumido (con tiempo extra)
    
    # Flags de análisis
    ENABLE_THIRDS_ANALYSIS = True       # Análisis por tercios de cancha
    ENABLE_TURNOVER_ANALYSIS = True     # Análisis de turnovers
    ENABLE_CROSS_ATTACKING = True       # Análisis de centros
    
    # Configuración de turnovers
    TURNOVER_OPEN_PLAY_ONLY = True      # Solo turnovers en juego abierto
    TURNOVER_EXCLUDE_RESTARTS = True    # Excluir saques/tiros libres


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_directories():
    """
    Crea los directorios necesarios si no existen.
    
    Útil para ejecutar antes de procesar datos o guardar outputs.
    """
    directories = [
        DATA_DIR,
        OUTPUTS_DIR,
        ASSETS_DIR,
        SEASONS_DATA_DIR,
        SEASONS_OUTPUT_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_season_data_dir(season_id: str) -> Path:
    """
    Retorna el directorio de datos de una temporada específica.
    
    Args:
        season_id: ID de temporada (ej: '2025_2026')
    
    Returns:
        Path al directorio de datos de la temporada
    
    Example:
        >>> data_dir = get_season_data_dir('2025_2026')
        >>> events_file = data_dir / 'events.csv'
    """
    return SEASONS_DATA_DIR / season_id


def get_season_output_dir(season_id: str) -> Path:
    """
    Retorna el directorio de outputs de una temporada específica.
    Crea el directorio si no existe.
    
    Args:
        season_id: ID de temporada (ej: '2025_2026')
    
    Returns:
        Path al directorio de outputs de la temporada
    """
    path = SEASONS_OUTPUT_DIR / season_id
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# INFORMACIÓN DEL PROYECTO
# =============================================================================

class ProjectInfo:
    """Información general del proyecto"""
    
    NAME = "Monterrey Scoring App"
    VERSION = "1.0.0"
    DESCRIPTION = "Sistema de análisis y scoring de jugadores de fútbol"
    CLUB = "Club de Fútbol Monterrey"
    
    # URLs útiles
    CLUB_WEBSITE = "https://www.rayados.com"
    DOCS_URL = None  # TODO: agregar cuando exista documentación


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    """
    Ejemplos de uso del módulo config
    """
    print("=" * 70)
    print("CONFIGURACIÓN DEL PROYECTO")
    print("=" * 70)
    
    print(f"\n📁 Directorios:")
    print(f"  Base: {BASE_DIR}")
    print(f"  Data: {DATA_DIR}")
    print(f"  Outputs: {OUTPUTS_DIR}")
    print(f"  Assets: {ASSETS_DIR}")
    
    print(f"\n🎨 Colores:")
    print(f"  Primary: {Colors.PRIMARY_BG}")
    print(f"  Accent: {Colors.ACCENT}")
    print(f"  Gold: {Colors.GOLD}")
    
    print(f"\n⚙️  Defaults:")
    print(f"  Min minutes: {Defaults.MIN_MINUTES}")
    print(f"  Min matches: {Defaults.MIN_MATCHES}")
    print(f"  Flag quantile: {Defaults.FLAG_QUANTILE}")
    
    print(f"\n📊 Processing:")
    print(f"  End cap: {Processing.ASSUME_END_CAP} min")
    print(f"  Thirds analysis: {Processing.ENABLE_THIRDS_ANALYSIS}")
    print(f"  Turnover analysis: {Processing.ENABLE_TURNOVER_ANALYSIS}")
    
    print(f"\n✅ Directorios verificados")
    ensure_directories()
    
    print("=" * 70)