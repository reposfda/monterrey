# utils/filters.py
from __future__ import annotations

import pandas as pd
import streamlit as st
from utils.role_config import get_category_weights, strip_score_prefix


def sidebar_filters(
    df_base: pd.DataFrame,
    *,
    show_position: bool = True,
    show_minutes: bool = True,
    show_team: bool = True,
    show_cat_weights: bool = True,
) -> dict:
    st.sidebar.title("Filtros")
    out = {}

    # -------------------------
    # Posición
    # -------------------------
    pos = None
    if show_position:
        positions = ["Zaguero", "Lateral", "Volante", "Interior/Mediapunta", "Extremo", "Delantero"]
        out["position"] = st.sidebar.selectbox("Posición", positions, index=0)
        pos = out["position"]

    # -------------------------
    # Minutos
    # -------------------------
    if show_minutes:
        out["min_minutes"] = st.sidebar.slider("Minutos mínimos", 0, 3000, 450, step=50)

    # -------------------------
    # Equipo
    # -------------------------
    if show_team:
        if "teams" not in df_base.columns:
            st.sidebar.caption("Equipo: no disponible (no existe columna 'teams').")
            out["teams"] = []
        else:
            teams = sorted(df_base["teams"].dropna().astype(str).unique().tolist())
            out["teams"] = st.sidebar.multiselect("Equipo", options=teams, default=[])

    # -------------------------
    # Pesos por categoría (4 sliders + Apply)
    # -------------------------
    if show_cat_weights and pos is not None:
        base_w = get_category_weights(pos)

        if base_w:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Pesos por categoría")

            key_active = f"cat_w_active_{pos}"   # pesos en uso
            key_draft  = f"cat_w_draft_{pos}"    # pesos que el user está editando

            # init
            if key_active not in st.session_state:
                st.session_state[key_active] = base_w.copy()
            if key_draft not in st.session_state:
                st.session_state[key_draft] = st.session_state[key_active].copy()

            cats = list(base_w.keys())

            # sliders "libres" (no normalizan)
            draft = st.session_state[key_draft].copy()
            for cat in cats:
                draft[cat] = st.sidebar.slider(
                    strip_score_prefix(cat),
                    0.0, 1.0,
                    float(draft.get(cat, base_w[cat])),
                    0.05,
                    key=f"{key_draft}_{cat}",
                )

            st.session_state[key_draft] = draft

            # feedback suma
            s = float(sum(draft.values()))
            st.sidebar.caption(f"Suma actual: {s:.2f}")

            cbtn1, cbtn2 = st.sidebar.columns(2)
            with cbtn1:
                apply = st.button("Aplicar", key=f"apply_{pos}", use_container_width=True)
            with cbtn2:
                reset = st.button("Reset", key=f"reset_{pos}", use_container_width=True)

            # reset a modelo base
            if reset:
                st.session_state[key_active] = base_w.copy()
                st.session_state[key_draft] = base_w.copy()
                st.sidebar.success("Pesos reseteados al modelo base.")

            # aplicar (solo si suma ~1)
            if apply:
                if abs(s - 1.0) <= 0.02:
                    st.session_state[key_active] = draft.copy()
                    st.sidebar.success("Pesos aplicados.")
                else:
                    delta = 1.0 - s
                    if delta > 0:
                        st.sidebar.warning(f"Te falta {delta:.2f} para llegar a 1.00.")
                    else:
                        st.sidebar.warning(f"Te pasaste {-delta:.2f} (bajá algún peso).")

            # devolver SIEMPRE los pesos activos (los ya aplicados)
            out["cat_weights"] = st.session_state[key_active].copy()


    return out
