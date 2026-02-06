# app.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Monterrey – Scoring App", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "monterrey_logo.png"

PRIMARY_BG = "#0B1F38"
SECONDARY_BG = "#091325"
ACCENT = "#6CA0DC"
TEXT = "#FFFFFF"

st.markdown(
    f"""
    <style>
        .stApp {{ background-color: {PRIMARY_BG}; }}
        header[data-testid="stHeader"] {{ background-color: transparent; }}
        [data-testid="stSidebar"] > div:first-child {{ background-color: {SECONDARY_BG}; }}
        h1, h2, h3, h4, h5, h6, p, label {{ color: {TEXT} !important; }}
        a {{ color: {ACCENT} !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

c1, c2 = st.columns([1, 5])
with c1:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=90)

with c2:
    st.markdown(
        """
        # Scoring App – Monterrey
        Diseñada para asistir a las máximas autoridades del CFM en la evaluación estratégica de renovar, renegociar o no extender los contratos de los futbolistas profesionales.

        Navega a través de las páginas de la columna izquierda:

        **Scoring Liga (Ranking)**: filtros por posición, minutos, edad y top 10.
        **Tablero Jugadores**: Evaluación individual de cada jugador y referencias ante la liga.
        """,
    )

st.info("")
