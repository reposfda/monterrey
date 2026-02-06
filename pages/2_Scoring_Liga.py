# pages/2_Scoring_Liga.py
# -*- coding: utf-8 -*-
"""
Página de Scoring Liga - Ranking de jugadores por posición.

Permite filtrar por:
- Posición
- Minutos jugados
- Partidos jugados
- Equipos

Muestra:
- Top 10 jugadores
- Ranking completo
- Gráficos de distribución por categoría
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ✅ Importar desde config centralizado
from config import (
    PER90_CSV,
    LOGO_PATH,
    Colors,
    Defaults,
    apply_global_styles
)

from utils.loaders import load_per90
from utils.filters import sidebar_filters
from utils.metrics_labels import COL_LABELS_ES
from utils.role_config import get_macro_config
from utils.scoring_wrappers import compute_scoring_from_df

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Scoring Liga – Monterrey",
    layout="wide",
    page_icon="📊"
)

# ✅ Aplicar estilos globales (reemplaza TODO el CSS duplicado)
apply_global_styles()

# Verificar que existe el archivo base
if not PER90_CSV.exists():
    st.error(f"❌ No se encontró el archivo base per90 en: {PER90_CSV}")
    st.info(
        "Por favor, ejecutá primero:\n\n"
        "```bash\n"
        "python calculate_main_csv.py\n"
        "```"
    )
    st.stop()

# =============================================================================
# HEADER
# =============================================================================
col_logo, col_title = st.columns([1, 5])

with col_logo:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=90)

with col_title:
    st.markdown(
        f"""
        <h1 style="margin-bottom:0;">Scoring Liga – Ranking</h1>
        <p style="color:{Colors.ACCENT}; font-size:0.95rem; margin-top:0.25rem;">
            Top 10 por posición (filtros: minutos, equipos, posición)
        </p>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# =============================================================================
# FILTERS
# =============================================================================
df_base = load_per90(PER90_CSV)

filters = sidebar_filters(
    df_base,
    show_position=True,
    show_minutes=True,
    show_team=True,
)

pos = filters["position"]
min_minutes = int(filters["min_minutes"])
selected_teams = filters.get("teams", [])
min_matches = Defaults.MIN_MATCHES  # ✅ Ahora viene de config
cat_w = filters.get("cat_weights", {})

# =============================================================================
# MAIN - CÁLCULO DE SCORING
# =============================================================================
with st.spinner("Calculando scoring..."):
    try:
        scores = compute_scoring_from_df(
            df_base=df_base,
            position_key=pos,
            min_minutes=min_minutes,
            min_matches=min_matches,
            selected_teams=selected_teams,
        )
    except ValueError:
        st.warning(f"No hay jugadores de {pos} que cumplan los filtros actuales.")
        st.info("Probá bajar Minutos mínimos, cambiar Equipo o elegir otra Posición.")
        st.stop()

# =============================================================================
# PREPARAR DATOS PARA DISPLAY
# =============================================================================

# Renombres base (sin crear duplicados)
rename_map = {
    "player_name": "Jugador",
    "team_name": "Equipo",
    "teams": "Equipo",
    "minutes": "Minutos",
    "matches": "PJ",
    "Flags": "Perfil",
}
scores_disp = scores.rename(columns=rename_map).copy()

# Asegurar score overall en una columna única
score_overall_candidates = [
    c for c in scores_disp.columns 
    if c.lower() in ["score_overall", "score_total", "overall_score", "score"]
]

if "Score_Overall" in scores.columns:
    scores_disp["Score"] = pd.to_numeric(scores["Score_Overall"], errors="coerce")
elif "score_overall" in scores.columns:
    scores_disp["Score"] = pd.to_numeric(scores["score_overall"], errors="coerce")
elif "score_total" in scores.columns:
    scores_disp["Score"] = pd.to_numeric(scores["score_total"], errors="coerce")
elif "Score" in scores_disp.columns:
    scores_disp["Score"] = pd.to_numeric(scores_disp["Score"], errors="coerce")
else:
    st.error("No encuentro columna de score overall (Score_Overall / score_overall / score_total / Score).")
    st.stop()

# Detectar scores por categoría
macro = get_macro_config(pos)  # [(Score_x, Label), ...]
cat_cols_raw = [c for c, _ in macro if c in scores.columns]

# Construir df con columnas de categoría ya normalizadas
cat_df = scores[cat_cols_raw].copy() if cat_cols_raw else pd.DataFrame(index=scores.index)

# Nombres lindos
pretty_map = {c: label for c, label in macro if c in cat_cols_raw}
cat_df = cat_df.rename(columns=pretty_map)

# Unir al display (por índice)
scores_disp = pd.concat([scores_disp, cat_df], axis=1)

# Columnas a mostrar
cat_cols_pretty = list(cat_df.columns)
base_cols = [
    c for c in ["Jugador", "Equipo", "Minutos", "PJ", "Score", "Perfil"] 
    if c in scores_disp.columns
]
cols_show = base_cols + cat_cols_pretty

# =============================================================================
# APLICAR PESOS DE CATEGORÍAS (WHAT-IF) -> Score_Ajustado
# =============================================================================

def _to_pretty_key(score_key: str) -> str:
    """Convierte Score_AccionDefensiva -> Accion Defensiva"""
    k = score_key.replace("Score_", "").replace("score_", "").replace("_", " ").strip().title()
    return k

# Construir pares (pretty_col, weight) que existan en scores_disp
pairs = []
for k, w in (cat_w or {}).items():
    pretty_k = _to_pretty_key(k)
    if pretty_k in scores_disp.columns:
        pairs.append((pretty_k, float(w)))

# Si tenemos 2+ pesos válidos, calculamos Score_Ajustado
if len(pairs) >= 2:
    # Normalizar pesos
    s = sum(w for _, w in pairs)
    if s > 0:
        pairs = [(c, w / s) for c, w in pairs]
    
    # Weighted sum
    score_adj = 0
    for c, w in pairs:
        score_adj = score_adj + (pd.to_numeric(scores_disp[c], errors="coerce") * w)
    
    scores_disp["Score_Ajustado"] = score_adj
else:
    scores_disp["Score_Ajustado"] = pd.to_numeric(scores_disp["Score"], errors="coerce")

# Agregar Score_Ajustado al display
cols_show = base_cols + (
    ["Score_Ajustado"] if "Score_Ajustado" in scores_disp.columns else []
) + cat_cols_pretty

# =============================================================================
# TOP 10
# =============================================================================

top10 = (
    scores_disp.sort_values("Score_Ajustado", ascending=False)[cols_show]
    .head(Defaults.TOP_N_RANKING)  # ✅ Ahora viene de config
    .reset_index(drop=True)
)

# Formateo
if "Minutos" in top10.columns:
    top10["Minutos"] = pd.to_numeric(top10["Minutos"], errors="coerce").fillna(0).astype(int)
if "PJ" in top10.columns:
    top10["PJ"] = pd.to_numeric(top10["PJ"], errors="coerce").fillna(0).astype(int)
if "Score" in top10.columns:
    top10["Score"] = pd.to_numeric(top10["Score"], errors="coerce").map(
        lambda x: "" if pd.isna(x) else f"{x:.2f}"
    )
if "Score_Ajustado" in top10.columns:
    top10["Score_Ajustado"] = pd.to_numeric(top10["Score_Ajustado"], errors="coerce").map(
        lambda x: "" if pd.isna(x) else f"{x:.2f}"
    )

for c in cat_cols_pretty:
    top10[c] = pd.to_numeric(top10[c], errors="coerce").map(
        lambda x: "" if pd.isna(x) else f"{x:.2f}"
    )

# Label de equipos
team_label = ""
if selected_teams:
    team_label = f" ({', '.join(selected_teams[:2])}{'...' if len(selected_teams) > 2 else ''})"

st.subheader(f"🏆 Top {Defaults.TOP_N_RANKING} – {pos}{team_label}")
st.markdown(top10.to_html(index=False, classes="mty-table"), unsafe_allow_html=True)

# =============================================================================
# RANKING COMPLETO
# =============================================================================

with st.expander("Ver ranking completo"):
    # Elegir SOLO columnas display
    cols_full = [c for c in cols_show if c in scores_disp.columns]
    df_disp = scores_disp[cols_full].copy()
    
    # Ordenar
    sort_col = "Score_Ajustado" if "Score_Ajustado" in df_disp.columns else "Score"
    if sort_col in df_disp.columns:
        df_disp = df_disp.sort_values(sort_col, ascending=False)
    
    # Renombrar SOLO columnas base (NO scores por rol, ya están "pretty")
    base_labels = COL_LABELS_ES.copy()
    base_labels.pop("Score_Overall", None)  # Evitar conflictos
    df_disp = df_disp.rename(columns=base_labels)
    
    st.dataframe(df_disp, use_container_width=True)

# =============================================================================
# VISUALIZACIÓN: BOXPLOT POR CATEGORÍA + HIGHLIGHT JUGADOR
# =============================================================================

st.markdown("---")
st.subheader("Distribución por categoría")

# Base DF (dedupe columnas + índice seguro)
plot_df = scores_disp.copy()
plot_df = plot_df.reset_index(drop=True)
plot_df.index = pd.RangeIndex(len(plot_df))

# Normalizar nombres columnas (por si hay NBSP)
plot_df.columns = (
    pd.Index(plot_df.columns)
      .astype(str)
      .str.replace("\u00A0", " ", regex=False)
      .str.strip()
)

# Categorías a graficar
cats = [c for c in cat_cols_pretty if c in plot_df.columns]

# Si tenés más de 4, recortar a 4
if len(cats) > 4:
    cats = cats[:4]

if len(cats) < 2:
    st.warning("No encontré categorías válidas para graficar en este DF.")
else:
    # Convertir a numérico
    for c in cats:
        plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")
    
    # Tirar NaNs
    plot_df = plot_df.dropna(subset=cats).copy()
    
    if plot_df.empty:
        st.warning("No hay filas con valores válidos en las categorías seleccionadas.")
    else:
        # Selector highlight robusto
        if "player_id" in plot_df.columns:
            plot_df["hl_id"] = plot_df["player_id"].astype(str)
        else:
            plot_df["hl_id"] = (
                plot_df["Jugador"].astype(str) + "||" + plot_df["Equipo"].astype(str)
            )
        
        plot_df["hl_label"] = (
            plot_df["Jugador"].astype(str) + " — " + plot_df["Equipo"].astype(str)
        )
        
        uniq = plot_df.drop_duplicates(subset=["hl_id"], keep="first")[
            ["hl_id", "hl_label"]
        ].copy()
        label_map = dict(zip(uniq["hl_id"], uniq["hl_label"]))
        
        selected_id = st.selectbox(
            "Highlight jugador:",
            options=["(ninguno)"] + uniq["hl_id"].tolist(),
            format_func=lambda x: x if x == "(ninguno)" else label_map.get(x, x),
            index=0,
            key=f"box_hl_{pos}",
        )
        
        # Reestructurar a formato long
        long = plot_df.melt(
            id_vars=["Jugador", "Equipo", "hl_id"],
            value_vars=cats,
            var_name="Categoria",
            value_name="Valor"
        )
        
        # =================================================================
        # FIGURA
        # =================================================================
        
        fig = go.Figure()
        
        # Boxplot por categoría
        for cat in cats:
            d = long.loc[long["Categoria"] == cat, "Valor"].astype(float).to_numpy()
            
            fig.add_trace(go.Box(
                y=d,
                name=cat,
                boxpoints="all",
                jitter=0.35,
                pointpos=0,
                marker=dict(size=5, opacity=0.22, color="rgba(232,238,246,0.55)"),
                line=dict(color="rgba(108,160,220,0.55)", width=1),
                fillcolor="rgba(108,160,220,0.08)",
                whiskerwidth=0.35,
                width=0.30,
                showlegend=False,
            ))
        
        # Highlight overlay
        if selected_id != "(ninguno)":
            hlong = long[long["hl_id"].astype(str) == str(selected_id)].copy()
            
            if not hlong.empty:
                fig.add_trace(go.Scatter(
                    x=hlong["Categoria"],
                    y=hlong["Valor"],
                    mode="markers+text",
                    text=[f"{v:.1f}" if pd.notna(v) else "" for v in hlong["Valor"]],
                    textposition="top center",
                    textfont=dict(color=Colors.TEXT, size=12),  # ✅ Usa Colors
                    marker=dict(
                        size=10,
                        color=Colors.GOLD,  # ✅ Usa Colors
                        opacity=0.95,
                        line=dict(width=0),
                        symbol="circle"
                    ),
                    hovertemplate="<b>%{x}</b><br>Valor: %{y:.1f}<extra></extra>",
                    showlegend=False
                ))
        
        # Layout
        fig.update_layout(
            height=560,
            paper_bgcolor=Colors.PRIMARY_BG,  # ✅ Usa Colors
            plot_bgcolor=Colors.PRIMARY_BG,   # ✅ Usa Colors
            margin=dict(l=18, r=18, t=10, b=10),
            font=dict(color=Colors.TEXT, family="Inter, Segoe UI, Arial"),  # ✅ Usa Colors
            xaxis=dict(
                title="",
                tickfont=dict(color="rgba(232,238,246,0.75)"),
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                title="",
                tickfont=dict(color="rgba(232,238,246,0.75)"),
                gridcolor="rgba(232,238,246,0.10)",
                zeroline=False
            ),
        )
        
        fig.update_yaxes(range=[0, 100])
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Boxplot: línea = mediana · caja = IQR (P25–P75) · "
            "whiskers = 1.5×IQR · puntos = jugadores"
        )