# app.py
# -*- coding: utf-8 -*-
"""
Página principal de la aplicación Monterrey Scoring App.
Landing page con navegación a las secciones principales.
"""
from __future__ import annotations
import streamlit as st

# Importar desde config
from config import LOGO_PATH, ProjectInfo, apply_global_styles

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================
st.set_page_config(
    page_title=f"{ProjectInfo.CLUB} – Scoring App",
    layout="wide",
    page_icon="⚽"
)

# Aplicar estilos globales
apply_global_styles()

# =============================================================================
# CONTENIDO PRINCIPAL
# =============================================================================

# Header con logo y título
col_logo, col_title = st.columns([1, 5])

with col_logo:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=90)

with col_title:
    st.markdown(
        f"""
        # {ProjectInfo.NAME}
        
        Diseñada para asistir a las máximas autoridades del CFM en la evaluación estratégica de renovar, renegociar o no extender los contratos de los futbolistas profesionales.

        ### Navegación
        
        Navega a través de las páginas de la columna izquierda:        
               
        Usá el sidebar para acceder a las diferentes secciones:
        
        - **Scoring Liga (Ranking):** Filtros de posición, minutos y equipos. Top 10 jugadores.
        - **Tablero Jugadores:** Comparación detallada entre jugadores con radares y métricas.
        
        """,
    )

# Footer con información
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"**Versión:** {ProjectInfo.VERSION}")

with col2:
    st.markdown(f"**Club:** {ProjectInfo.CLUB}")

with col3:
    if ProjectInfo.CLUB_WEBSITE:
        st.markdown(f"[🌐 Sitio Web]({ProjectInfo.CLUB_WEBSITE})")