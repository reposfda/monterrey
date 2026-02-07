import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

import plotly.graph_objects as go

# columnas esperadas
COL_TEAM = "team_name"
COL_PLAYER = "player_name"
COL_PERF = "Overall_Score_Final"
COL_COST = "Cost_Share"

# =========================================================
# ESTILO
# =========================================================
PRIMARY_BG = "#0B1F38"
SECONDARY_BG = "#091325"
ACCENT = "#6CA0DC"   # azul MTY
TEXT = "#FFFFFF"
GOLD = "#c49308"



# =========================================================
# PLOT INTERACTIVO
# =========================================================
def plot_market_curve_interactive(
    df: pd.DataFrame,
    model: LinearRegression,
    residual_std: float,
    perf_col: str = COL_PERF,
    cost_col: str = COL_COST,
    team_col: str = COL_TEAM,
    player_col: str = COL_PLAYER,
    highlight_team: str = "Monterrey",
    title: str = "Curva de Mercado: Rendimiento vs % del gasto total",
    selected_player: str | None = None,
):
    """
    Interactivo (Plotly):
    - Resto de equipos: gris
    - Monterrey: azul (ACCENT)
    - Jugador seleccionado: se resalta SIEMPRE (encima de todo):
        * Rojo si está por encima de la curva (sobrepago)
        * Verde si está por debajo de la curva (buen negocio)
      y se dibuja su nombre en el mismo color.
    """
    d = df.dropna(subset=[perf_col, cost_col]).copy()

    # Split Monterrey vs resto (para nube base)
    is_mty = d[team_col].astype(str).str.strip().str.lower() == highlight_team.lower()
    d_mty = d[is_mty].copy()
    d_oth = d[~is_mty].copy()

    # Buscar jugador seleccionado en df ORIGINAL (no en d) para detectar si existe aunque tenga NaNs
    selected_row = None
    if selected_player is not None and player_col in df.columns:
        sel_mask_all = df[player_col].astype(str).str.strip() == str(selected_player).strip()
        if sel_mask_all.any():
            candidate = df.loc[sel_mask_all].iloc[0]
            # solo lo usamos si tiene perf y cost no nulos (si no, no se puede plotear)
            if pd.notna(candidate.get(perf_col, np.nan)) and pd.notna(candidate.get(cost_col, np.nan)):
                selected_row = candidate

    # Si existe selected_row, lo sacamos de las nubes para evitar duplicación
    if selected_row is not None:
        sel_name = str(selected_row[player_col]).strip()
        d_mty = d_mty.loc[d_mty[player_col].astype(str).str.strip() != sel_name]
        d_oth = d_oth.loc[d_oth[player_col].astype(str).str.strip() != sel_name]

    # Rango para la curva
    x_min = float(d[perf_col].min())
    x_max = float(d[perf_col].max())
    perf_range = np.linspace(x_min, x_max, 250)

    pred_curve = model.intercept_ + model.coef_[0] * perf_range
    band_low = pred_curve - residual_std
    band_high = pred_curve + residual_std

    fig = go.Figure()

    # Banda de variabilidad (gris)
    fig.add_trace(
        go.Scatter(
            x=perf_range,
            y=band_high,
            mode="lines",
            line=dict(color="rgba(180,180,180,0.0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=perf_range,
            y=band_low,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(180,180,180,0.25)",
            line=dict(color="rgba(180,180,180,0.0)"),
            hoverinfo="skip",
            name="Variabilidad del mercado (±1 STD)",
        )
    )

    # Línea del mercado
    fig.add_trace(
        go.Scatter(
            x=perf_range,
            y=pred_curve,
            mode="lines",
            line=dict(color="black", width=3),
            name="Curva de mercado",
            hoverinfo="skip",
        )
    )

    # Resto de equipos (gris)
    fig.add_trace(
        go.Scatter(
            x=d_oth[perf_col],
            y=d_oth[cost_col],
            mode="markers",
            name="Liga MX (otros clubes)",
            marker=dict(size=9, color="rgba(200,200,200,0.70)", line=dict(width=0)),
            customdata=np.stack([d_oth[player_col].astype(str), d_oth[team_col].astype(str)], axis=1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Equipo: %{customdata[1]}<br>"
                "Performance: %{x:.2f}<br>"
                "Cost share: %{y:.3f}<extra></extra>"
            ),
        )
    )

    # Monterrey (azul)
    fig.add_trace(
        go.Scatter(
            x=d_mty[perf_col],
            y=d_mty[cost_col],
            mode="markers",
            name="Monterrey",
            marker=dict(size=11, color=ACCENT, line=dict(width=1, color="rgba(255,255,255,0.55)")),
            customdata=np.stack([d_mty[player_col].astype(str), d_mty[team_col].astype(str)], axis=1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Equipo: %{customdata[1]}<br>"
                "Performance: %{x:.2f}<br>"
                "Cost share: %{y:.3f}<extra></extra>"
            ),
        )
    )

    # Jugador seleccionado (encima de TODO)
    if selected_row is not None:
        sel_perf = float(selected_row[perf_col])
        sel_cost = float(selected_row[cost_col])

        # Predicción (precio justo) para ese rendimiento
        sel_fair = float(model.predict(np.array([[sel_perf]]))[0])

        # Color según desvío
        # (arriba de la curva = sobrepago = rojo; abajo = buen negocio = verde)
        sel_color = "#ef4444" if sel_cost > sel_fair else "#22c55e"

        sel_team = str(selected_row[team_col]) if team_col in selected_row else ""
        sel_name = str(selected_row[player_col])

        # Un pequeño offset vertical del texto para que no tape el punto
        text_pos = "top center"

        fig.add_trace(
            go.Scatter(
                x=[sel_perf],
                y=[sel_cost],
                mode="markers+text",
                name="Jugador seleccionado",
                marker=dict(
                    size=18,                 # MÁS GRANDE para que se note
                    color=sel_color,
                    symbol="circle",
                    line=dict(width=3, color="rgba(0,0,0,0.55)"),
                ),
                text=[sel_name],
                textposition=text_pos,
                textfont=dict(color=sel_color, size=14),
                customdata=np.array([[sel_name, sel_team, sel_fair]]),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Equipo: %{customdata[1]}<br>"
                    "Performance: %{x:.2f}<br>"
                    "Cost share: %{y:.3f}<br>"
                    "Cost share justo (modelo): %{customdata[2]:.3f}<extra></extra>"
                ),
                showlegend=True,
            )
        )

    fig.update_layout(
        title=title,
        paper_bgcolor=PRIMARY_BG,
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color=TEXT),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=10, r=10, t=60, b=10),
        height=520,
    )
    fig.update_xaxes(title_text="Performance Score", gridcolor="rgba(255,255,255,0.10)", zeroline=False)
    fig.update_yaxes(title_text="% del gasto total del club (Cost Share)", gridcolor="rgba(255,255,255,0.10)", zeroline=False)

    return fig

def render_performance_salary_scale_table(
    df_ranges: pd.DataFrame,
    player_perf: float,
    accent_color: str = "#0B1F38",  # azul oscuro MTY (fila resaltada)
    table_width: str = "62%",       # ancho más finito
):
    """
    Devuelve un pandas Styler para renderizar una tabla con:
    - Degradé rojo → verde según performance (más bajo = rojo, más alto = verde)
    - Resalta en azul oscuro la fila donde cae el jugador seleccionado
    - Encabezados centrados
    - Tabla más angosta (no ocupa todo el ancho)

    IMPORTANTE: implementada para ser robusta a pandas/streamlit donde apply(subset=...)
    pasa a la función SOLO las columnas del subset.
    """

    d = df_ranges.copy()

    # Sanidad: asegurar numéricos
    d["performance_min"] = pd.to_numeric(d.get("performance_min"), errors="coerce")
    d["performance_max"] = pd.to_numeric(d.get("performance_max"), errors="coerce")

    # Midpoint para gradiente
    perf_mid = (d["performance_min"] + d["performance_max"]) / 2

    # Normalización 0..1
    if perf_mid.notna().any():
        pmin = float(perf_mid.min())
        pmax = float(perf_mid.max())
        denom = (pmax - pmin) if (pmax - pmin) != 0 else 1.0
        perf_norm = ((perf_mid - pmin) / denom).fillna(0.0).clip(0.0, 1.0).to_numpy()
    else:
        perf_norm = np.zeros(len(d), dtype=float)

    # Fila donde cae el jugador
    is_player_row = (
        (player_perf >= d["performance_min"]) &
        (player_perf <= d["performance_max"])
    ).fillna(False).to_numpy()

    # DF a mostrar
    d_disp = d[["cost_share_range_label", "performance_range_label"]].copy()
    d_disp = d_disp.rename(
        columns={
            "cost_share_range_label": "Rango Salarial (% del gasto total)",
            "performance_range_label": "Rango Performance requerido",
        }
    )

    visible_cols = list(d_disp.columns)

    def row_style(row: pd.Series):
        i = int(row.name)  # índice de fila
        if i < 0 or i >= len(is_player_row):
            # fallback por seguridad
            return ["text-align: center;"] * len(visible_cols)

        # Resaltar fila del jugador
        if bool(is_player_row[i]):
            return [
                f"background-color: {accent_color}; color: #ffffff; font-weight: 800; text-align: center;"
            ] * len(visible_cols)

        # Gradiente rojo -> verde
        t = float(perf_norm[i])
        # rojo rgb(239,68,68) -> verde rgb(34,197,94)
        r = int(239 * (1 - t) + 34 * t)
        g = int(68 * (1 - t) + 197 * t)
        b = int(68 * (1 - t) + 94 * t)

        bg = f"rgb({r},{g},{b})"
        return [
            f"background-color: {bg}; color: #0f172a; text-align: center; font-weight: 600;"
        ] * len(visible_cols)

    styler = (
        d_disp.style
        .apply(row_style, axis=1)
        .set_properties(**{
            "text-align": "center",
            "font-size": "0.92rem",
            "padding": "0.55rem 0.75rem",
        })
        .set_table_styles([
            # Headers centrados
            {"selector": "th", "props": [
                ("text-align", "center"),
                ("font-weight", "800"),
                ("font-size", "0.85rem"),
                ("background-color", "#0B1F38"),
                ("color", "white"),
                ("padding", "0.65rem 0.75rem"),
            ]},
            # Tabla más finita y centrada
            {"selector": "table", "props": [
                ("margin-left", "auto"),
                ("margin-right", "auto"),
                ("width", table_width),
                ("border-radius", "12px"),
                ("overflow", "hidden"),
                ("border-collapse", "separate"),
                ("border-spacing", "0"),
            ]},
            {"selector": "td", "props": [("border", "none")]},
        ])
    )

    return styler

def render_mty_table(df, index: bool = False):
    """
    Renderiza una tabla HTML con estilo Monterrey.
    - Headers centrados
    - Celdas centradas (podés cambiar a left si querés)
    """
    import streamlit as st

    d = df.copy()

    # Convertir a HTML
    html = d.to_html(index=index, escape=False)

    # Inyectar class para aplicar CSS
    html = html.replace('<table border="1" class="dataframe">', '<table class="mty-table">')

    # CSS específico (centrar headers)
    st.markdown(
        """
        <style>
        table.mty-table thead th {
            text-align: center !important;
        }
        table.mty-table tbody td {
            text-align: center !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(html, unsafe_allow_html=True)

