import pandas as pd
import numpy as np

from pathlib import Path

import plotly.graph_objects as go

# =========================================================
# ESTILO (mismo look & feel)
# =========================================================
PRIMARY_BG = "#0B1F38"
SECONDARY_BG = "#091325"
ACCENT = "#6CA0DC"
TEXT = "#FFFFFF"
GOLD = "#c49308"

BASE_DIR = Path(__file__).resolve().parents[1]
LOGO_PATH = BASE_DIR / "assets" / "monterrey_logo.png"

def plot_talent_cost_utilization_bar(
    df_team: pd.DataFrame,
    team_col: str = "team_name",
    val_col: str = "talent_cost_utilization_per_team",
    title: str = "COSTO DE UTILIZACIÓN – CONTEXTO LIGA MX",
):
    """
    Plotly bar chart estilo "imagen":
    - Fondo oscuro
    - Degradé azul (alto) → rojo (bajo)
    - Promedio Liga en gris con borde negro
    - Labels arriba en %
    - Grid punteada
    """

    d = df_team.copy()
    d[team_col] = d[team_col].astype(str)
    d[val_col] = pd.to_numeric(d[val_col], errors="coerce")
    d = d.dropna(subset=[val_col])

    d = d.sort_values(val_col, ascending=False).reset_index(drop=True)

    vals = d[val_col].to_numpy()
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
    t = (vals - vmin) / denom  # 0..1

    # 0=rojo, 0.5=claro, 1=azul
    def interp_color(tt: float) -> str:
        if tt <= 0.5:
            a = tt / 0.5
            c1 = np.array([0xB4, 0x04, 0x26]) / 255.0  # red
            c2 = np.array([0xF7, 0xF7, 0xF7]) / 255.0  # light
        else:
            a = (tt - 0.5) / 0.5
            c1 = np.array([0xF7, 0xF7, 0xF7]) / 255.0  # light
            c2 = np.array([0x08, 0x30, 0x6B]) / 255.0  # blue
        c = (1 - a) * c1 + a * c2
        return f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"

    bar_colors = [interp_color(float(tt)) for tt in t]

    # Promedio Liga en gris + borde negro
    avg_mask = d[team_col].str.strip().str.lower().eq("promedio liga")
    if avg_mask.any():
        avg_idx = int(np.where(avg_mask.to_numpy())[0][0])
        bar_colors[avg_idx] = "rgba(220,220,220,1)"
        line_colors = ["rgba(0,0,0,0)"] * len(d)
        line_widths = [0] * len(d)
        line_colors[avg_idx] = "black"
        line_widths[avg_idx] = 2
    else:
        line_colors = ["rgba(0,0,0,0)"] * len(d)
        line_widths = [0] * len(d)

    text_labels = [f"{v:.1f}%" for v in d[val_col]]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=d[team_col],
            y=d[val_col],
            marker=dict(
                color=bar_colors,
                line=dict(color=line_colors, width=line_widths),
            ),
            text=text_labels,
            textposition="outside",
            textfont=dict(color="#9fb7ff", size=14),
            hovertemplate="<b>%{x}</b><br>Utilización: %{y:.1f}%<extra></extra>",
        )
    )

    ymax = float(d[val_col].max())

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=34, color=TEXT, family="Arial Black"),
        ),
        paper_bgcolor=PRIMARY_BG,
        plot_bgcolor=PRIMARY_BG,
        margin=dict(l=20, r=20, t=90, b=60),
        height=640,
        xaxis=dict(
            title="",
            tickangle=0,
            tickfont=dict(color="#b9c6d6", size=12),
            showgrid=False,
        ),
        yaxis=dict(
            title=dict(
                text="% Utilización",
                font=dict(color=TEXT, size=16),
            ),
            tickfont=dict(color="#8aa3c5", size=16),
            showgrid=True,
            gridcolor="rgba(255,255,255,0.25)",
            gridwidth=1,
            griddash="dash",
            zeroline=False,
            range=[0, ymax + 8],
        ),
        showlegend=False,
    )

    return fig
