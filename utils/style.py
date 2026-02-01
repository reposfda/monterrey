# utils/style.py
from __future__ import annotations
import streamlit as st

PRIMARY_BG = "#0B1F38"
SECONDARY_BG = "#091325"
ACCENT = "#6CA0DC"
TEXT = "#FFFFFF"
GOLD = "#c49308"

def apply_mty_style():
    # ... tu CSS actual ...
    st.markdown(
        f"""
        <style>
        /* --- TABLAS MTY (HTML df.to_html) --- */
        div.stMarkdown table.mty-table,
        div.stMarkdown table.dataframe.mty-table,
        table.mty-table,
        table.dataframe.mty-table {{
            border-collapse: separate !important;
            border-spacing: 0 !important;
            width: 100% !important;
            background-color: #ffffff !important;
            border-radius: 12px !important;
            overflow: hidden !important;
            font-size: 0.90rem !important;
        }}

        div.stMarkdown table.mty-table thead,
        div.stMarkdown table.dataframe.mty-table thead,
        table.mty-table thead,
        table.dataframe.mty-table thead {{
            background-color: {PRIMARY_BG} !important;
        }}

        div.stMarkdown table.mty-table thead th,
        div.stMarkdown table.dataframe.mty-table thead th,
        table.mty-table thead th,
        table.dataframe.mty-table thead th {{
            color: #ffffff !important;
            padding: 0.55rem 0.75rem !important;
            border: none !important;
            text-align: center !important;
        }}

        div.stMarkdown table.mty-table thead th:first-child,
        div.stMarkdown table.dataframe.mty-table thead th:first-child,
        table.mty-table thead th:first-child,
        table.dataframe.mty-table thead th:first-child {{
            text-align: left !important;
            padding-left: 12px !important;
        }}

        div.stMarkdown table.mty-table tbody td,
        div.stMarkdown table.dataframe.mty-table tbody td,
        table.mty-table tbody td,
        table.dataframe.mty-table tbody td {{
            color: #1f2933 !important;
            padding: 0.55rem 0.75rem !important;
            border: none !important;
            text-align: center !important;
            background-color: transparent !important;
        }}

        div.stMarkdown table.mty-table tbody td:first-child,
        div.stMarkdown table.dataframe.mty-table tbody td:first-child,
        table.mty-table tbody td:first-child,
        table.dataframe.mty-table tbody td:first-child {{
            text-align: left !important;
            padding-left: 12px !important;
        }}

        div.stMarkdown table.mty-table tbody tr:nth-child(even),
        div.stMarkdown table.dataframe.mty-table tbody tr:nth-child(even),
        table.mty-table tbody tr:nth-child(even),
        table.dataframe.mty-table tbody tr:nth-child(even) {{
            background-color: #f4f6fb !important;
        }}

        div.stMarkdown table.mty-table tbody tr:hover,
        div.stMarkdown table.dataframe.mty-table tbody tr:hover,
        table.mty-table tbody tr:hover,
        table.dataframe.mty-table tbody tr:hover {{
            background-color: #e8f0ff !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
